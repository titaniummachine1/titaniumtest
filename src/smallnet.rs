//! `ti train smallnet`: refit the futility SmallNet (engine `smallnet.rs`, TSN1) to the CURRENT big net (D287/D289: the
//! trainer lives in the hub, Rust only; training/smallnet_train.py is closed for good).
//!
//! The SmallNet is a pessimistic LOWER BOUND on the big net's eval of a search child: the futility step prunes a move without
//! paying the big net when it is pruned even at this bound, so a bound that sits above the true eval on more than ~tau of the
//! calls prunes wrongly, and one that sits far below prunes too little. It goes stale whenever the big net is retrained or
//! output-calibrated. This command rebuilds the data from the big net itself and refits:
//!   1. positions: evenly spread rows of a packed `.tdat` (train / val), a random D4 symmetry per row;
//!   2. samples: every legal child (kept with probability 1/child_every by position hash): child stones (own/opp/blockers as
//!      seen by the child's side to move), parent cp (big net on the parent), hand eval of the child, TARGET = big net on the
//!      child, all in cp, children evaluated with the hub's own bit-exact TrainableNet forward (no engine run needed);
//!   3. fit: epochs of MSE (eval mode) then epochs of pinball at `tau` (lower bound), Adam, cosine lr, threads;
//!   4. report on held-out rows before (the --init net) and after: RMSE, P(bound > truth) [design tau], slack;
//!   5. export TSN1 and verify it against the ENGINE's own `SmallNet::parse` + `bound` (max |diff| must be <= 1 cp).
//!
//!   ti train smallnet --net nnue_main.tnn --train data/nnue/packed/train_v9 --val data/nnue/packed/val_v9_clean \
//!       --out sn_new.tsn --init data/nnue/fut_small.tsn [--val-rows 60000] [--child-every 24] \
//!   (D1130: always EVERY row of --train; --rows N below that only with --experiment NAME, output under experiments/;
//!    the run refuses when the samples would not fit in RAM and names the --child-every that fits)
//!       [--patience 4] [--tau 0.02] [--lr 0.002] [--threads 7] [--seed 1]
//!   (D898: trains the pinball bound until --patience epochs bring no val gain; no epoch counts exist, LAWS C15)
use std::process::ExitCode;
use std::time::Instant;

use crate::data::Rec;
use crate::mapper;
use crate::train::TrainableNet;
use crate::trainer::{load_val, Tdat};
use titanium::board::MoveList;

const W: usize = 32;
const H: usize = 16;
const NB: usize = 8;
const D2: usize = 2 * W;
const NIN: usize = 147;

const O_FT: usize = 0;
const O_FTB: usize = O_FT + NIN * W;
const O_L1W: usize = O_FTB + W;
const O_L1B: usize = O_L1W + NB * H * D2;
const O_OW: usize = O_L1B + NB * H;
const O_OB: usize = O_OW + NB * H;
const O_SK: usize = O_OB + NB; // skip[0][NB] (parent), skip[1][NB] (hand)
const NP: usize = O_SK + 2 * NB;

#[derive(Clone, Copy)]
struct Sample {
    own: u64,
    opp: u64,
    blk: u64,
    parent: i16,
    hand: i16,
    target: i16,
}

#[derive(Clone)]
pub struct Net {
    p: Vec<f32>,
}

struct Fwd {
    x: [f32; D2],
    hpre: [f32; H],
    h: [f32; H],
    y: f32,
    bk: usize,
}

impl Net {
    fn random(seed: u64) -> Net {
        let mut s = seed | 1;
        let mut rnd = move || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 11) as f64 / (1u64 << 53) as f64) as f32 * 2.0 - 1.0
        };
        let mut p = vec![0f32; NP];
        for v in &mut p[O_FT..O_FTB] {
            *v = rnd() * 0.15;
        }
        for v in &mut p[O_L1W..O_L1B] {
            *v = rnd() * (3.0 / (D2 as f32)).sqrt();
        }
        for v in &mut p[O_L1B..O_OW] {
            *v = 0.5;
        }
        for v in &mut p[O_OW..O_OB] {
            *v = rnd() * 0.3;
        }
        for k in 0..NB {
            p[O_SK + k] = 0.8; // parent eval skip
            p[O_SK + NB + k] = 0.0;
        }
        Net { p }
    }

    pub fn parse(d: &[u8]) -> Result<Net, String> {
        if d.len() < 12 || &d[..4] != b"TSN1" {
            return Err("not a TSN1 file".into());
        }
        let u = |i: usize| u32::from_le_bytes(d[4 + 4 * i..8 + 4 * i].try_into().unwrap()) as usize;
        if (u(0), u(1)) != (W, H) {
            return Err(format!("small net is {}x{}, the hub trains {W}x{H}", u(0), u(1)));
        }
        if d.len() != 12 + 4 * NP {
            return Err(format!("size {} != {}", d.len(), 12 + 4 * NP));
        }
        let p = d[12..].chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
        Ok(Net { p })
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut o = Vec::with_capacity(12 + 4 * NP);
        o.extend_from_slice(b"TSN1");
        o.extend_from_slice(&(W as u32).to_le_bytes());
        o.extend_from_slice(&(H as u32).to_le_bytes());
        for v in &self.p {
            o.extend_from_slice(&v.to_le_bytes());
        }
        o
    }

    fn forward(&self, s: &Sample) -> Fwd {
        let p = &self.p;
        let mut a = [0f32; D2];
        a[..W].copy_from_slice(&p[O_FTB..O_FTB + W]);
        a[W..].copy_from_slice(&p[O_FTB..O_FTB + W]);
        let mut add = |bb: u64, us_off: usize, them_off: usize| {
            let mut bb = bb;
            while bb != 0 {
                let sq = bb.trailing_zeros() as usize;
                bb &= bb - 1;
                let u = &p[O_FT + (us_off + sq) * W..O_FT + (us_off + sq + 1) * W];
                let t = &p[O_FT + (them_off + sq) * W..O_FT + (them_off + sq + 1) * W];
                for j in 0..W {
                    a[j] += u[j];
                    a[W + j] += t[j];
                }
            }
        };
        add(s.own, 0, 49);
        add(s.opp, 49, 0);
        add(s.blk, 98, 98);
        let mut x = [0f32; D2];
        for j in 0..D2 {
            x[j] = a[j].clamp(0.0, 1.0);
        }
        let empties = 49 - (s.own | s.opp | s.blk).count_ones() as usize;
        let bk = (empties * NB / 50).min(NB - 1);
        let mut hpre = [0f32; H];
        let mut h = [0f32; H];
        let mut y = p[O_OB + bk] + p[O_SK + bk] * (s.parent as f32 / 400.0) + p[O_SK + NB + bk] * (s.hand as f32 / 400.0);
        for i in 0..H {
            let w = &p[O_L1W + (bk * H + i) * D2..O_L1W + (bk * H + i + 1) * D2];
            let mut z = p[O_L1B + bk * H + i];
            for j in 0..D2 {
                z += w[j] * x[j];
            }
            hpre[i] = z;
            h[i] = z.clamp(0.0, 1.0);
            y += p[O_OW + bk * H + i] * h[i];
        }
        Fwd { x, hpre, h, y, bk }
    }

    /// Accumulate d loss / d params for one sample given dL/dy.
    fn backward(&self, s: &Sample, f: &Fwd, dy: f32, g: &mut [f32]) {
        let p = &self.p;
        let bk = f.bk;
        g[O_OB + bk] += dy;
        g[O_SK + bk] += dy * (s.parent as f32 / 400.0);
        g[O_SK + NB + bk] += dy * (s.hand as f32 / 400.0);
        let mut dx = [0f32; D2];
        for i in 0..H {
            g[O_OW + bk * H + i] += dy * f.h[i];
            if f.hpre[i] > 0.0 && f.hpre[i] < 1.0 {
                let dz = dy * p[O_OW + bk * H + i];
                g[O_L1B + bk * H + i] += dz;
                let w = &p[O_L1W + (bk * H + i) * D2..O_L1W + (bk * H + i + 1) * D2];
                let gw = &mut g[O_L1W + (bk * H + i) * D2..O_L1W + (bk * H + i + 1) * D2];
                for j in 0..D2 {
                    gw[j] += dz * f.x[j];
                    dx[j] += dz * w[j];
                }
            }
        }
        // through the accumulator clamp: the pre-clamp value is inside (0,1) iff x is strictly inside, except the exact
        // 0/1 saturation which carries no gradient (x == 0.0 or 1.0)
        let mut da = [0f32; D2];
        for j in 0..D2 {
            if f.x[j] > 0.0 && f.x[j] < 1.0 {
                da[j] = dx[j];
            }
        }
        for j in 0..W {
            g[O_FTB + j] += da[j] + da[W + j];
        }
        let mut sc = |bb: u64, us_off: usize, them_off: usize| {
            let mut bb = bb;
            while bb != 0 {
                let sq = bb.trailing_zeros() as usize;
                bb &= bb - 1;
                for j in 0..W {
                    g[O_FT + (us_off + sq) * W + j] += da[j];
                    g[O_FT + (them_off + sq) * W + j] += da[W + j];
                }
            }
        };
        sc(s.own, 0, 49);
        sc(s.opp, 49, 0);
        sc(s.blk, 98, 98);
    }
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
}

/// Children of every row, evaluated with the big net. `every`: keep a child iff its hash % every == 0.
fn build_samples(rows: &[Rec], big: &TrainableNet, every: u64, threads: usize, seed: u64) -> Vec<Sample> {
    let part = (rows.len() + threads - 1) / threads.max(1);
    let parts: Vec<Vec<Sample>> = std::thread::scope(|sc| {
        let hs: Vec<_> = rows
            .chunks(part.max(1))
            .enumerate()
            .map(|(ti, chunk)| {
                sc.spawn(move || {
                    let mut rng = Rng(seed ^ ((ti as u64 + 1) * 0x9E37_79B9_7F4A_7C15) | 1);
                    let mut out = Vec::new();
                    for r in chunk {
                        let mode = (rng.next() % mapper::MODES as u64) as usize;
                        let b = mapper::board_of(r, mode);
                        if b.game_over() {
                            continue;
                        }
                        let parent = (400.0 * big.forward_y(&b)).round().clamp(-30000.0, 30000.0) as i16;
                        let mut list = MoveList::new();
                        b.legal_moves_dedup_into(&mut list);
                        for i in 0..list.len() {
                            let c = b.make(list.move_at(i));
                            if c.game_over() || c.hash % every != 0 {
                                continue;
                            }
                            let ct = c.turn as usize;
                            let hand = titanium::search::evaluate(&c, None, false).clamp(-30000, 30000) as i16;
                            let target = (400.0 * big.forward_y(&c)).round().clamp(-30000.0, 30000.0) as i16;
                            out.push(Sample { own: c.occ[ct], opp: c.occ[1 - ct], blk: c.blockers, parent, hand, target });
                        }
                    }
                    out
                })
            })
            .collect();
        hs.into_iter().map(|h| h.join().unwrap()).collect()
    });
    parts.into_iter().flatten().collect()
}

#[derive(Default, Clone, Copy)]
struct Report {
    n: usize,
    rmse_cp: f64,
    viol: f64,
    slack_mean: f64,
    viol_end: f64,
}

fn report(net: &Net, val: &[Sample]) -> Report {
    let (mut se, mut v, mut sl, mut ve, mut ne) = (0f64, 0usize, 0f64, 0usize, 0usize);
    for s in val {
        let y = 400.0 * net.forward(s).y as f64;
        let t = s.target as f64;
        se += (y - t) * (y - t);
        sl += t - y;
        if y > t {
            v += 1;
        }
        if (49 - (s.own | s.opp | s.blk).count_ones()) <= 10 {
            ne += 1;
            if y > t {
                ve += 1;
            }
        }
    }
    let n = val.len().max(1) as f64;
    Report { n: val.len(), rmse_cp: (se / n).sqrt(), viol: v as f64 / n, slack_mean: sl / n, viol_end: ve as f64 / ne.max(1) as f64 }
}

fn print_report(tag: &str, r: &Report) {
    println!(
        "[smallnet] {tag}: n {} | RMSE {:.1} cp | P(bound>truth) {:.2}% (endgame <=10 empties {:.2}%) | mean slack (truth-bound) {:.1} cp",
        r.n,
        r.rmse_cp,
        100.0 * r.viol,
        100.0 * r.viol_end,
        r.slack_mean
    );
}

/// Val objective of the phase: MSE (tau None) or pinball loss at tau, in (cp/400) units, lower is better.
fn val_objective(net: &Net, val: &[Sample], tau: Option<f32>) -> f64 {
    let mut l = 0f64;
    for s in val {
        let d = (s.target as f32 / 400.0 - net.forward(s).y) as f64;
        l += match tau {
            None => d * d,
            Some(q) => {
                let q = q as f64;
                if d > 0.0 { q * d } else { (q - 1.0) * d }
            }
        };
    }
    l / val.len().max(1) as f64
}

/// FNV-1a over every sample field (equal checksums = the same samples; a resume refuses other samples).
fn samples_checksum(set: &[Sample]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for s in set {
        for x in [s.own, s.opp, s.blk, (s.parent as u16 as u64) | (s.hand as u16 as u64) << 16 | (s.target as u16 as u64) << 32] {
            h ^= x;
            h = h.wrapping_mul(0x100_0000_01b3);
        }
    }
    h
}

/// LAWS A14 (user D1305-D1307): where a run keeps its checkpoint + best snapshot, and the folder its STOP file sits in.
pub struct Ck {
    /// full resumable state (atomic, every epoch and on STOP)
    pub state: String,
    /// the best weights so far as an engine blob (atomic, the moment they improve)
    pub best: String,
    pub run_dir: std::path::PathBuf,
    pub data_sum: u64,
    pub resume: bool,
}

/// How a fit ended.
#[derive(PartialEq, Debug)]
pub enum FitEnd {
    Saturated,
    /// stopped by a stop file: the state is saved, `--resume` continues
    Stopped(String),
}

/// LAWS C15 + LAWS D5 (user D812, D1553, D1554): Schedule-Free AdamW (Defazio et al. 2024, evaluated on averaged
/// weights x, no LR decay schedule) stopping ONLY when `patience` (4) consecutive epochs of data bring no validation
/// loss improvement >= 1e-5 (the 5th decimal place). The best epoch's weights are kept.
/// LAWS A14: every epoch writes a resumable checkpoint, every new best writes the best blob, a stop file (run dir STOP or
/// STOP_ALL) is checked before every batch: the exact state (inside the epoch) is saved and the fit returns `Stopped`.
/// The epoch's sample order is a pure function of (seed, epoch), so a resume replays it and skips the batches done.
#[allow(clippy::too_many_arguments)]
fn fit(net: &mut Net, train: &[Sample], val: &[Sample], patience: usize, tau: Option<f32>, lr0: f32, threads: usize, seed: u64, tag: &str, ck: &Ck) -> Result<FitEnd, String> {
    let batch = 16384usize;
    let warmup = 200usize;
    let mut zz = net.p.clone();
    let mut xx = net.p.clone();
    let mut vv = vec![0f32; NP];
    let mut muon_m = vec![0f32; O_L1B - O_L1W];
    let mut sum_lr2 = 0f64;
    let mut step = 0usize;
    let mut grads: Vec<Vec<f32>> = (0..threads).map(|_| vec![0f32; NP]).collect();
    let (b1, b2, eps, mu_mom) = (0.9f32, 0.999f32, 1e-8f32, 0.95f32);
    let val0 = val_objective(net, val, tau);
    let mut best = (val0, xx.clone(), 0usize);
    let mut best_saved = val0;
    println!("[smallnet] {tag} start (Muon+SF-AdamW, patience {patience}, min gain >= 1e-5, NO lr decay): val objective {:.6}", best.0);
    if !ck.resume {
        // a best snapshot exists from the first second on (the init), however short the run (LAWS A14)
        crate::stopctl::write_atomic(&ck.best, &net.to_bytes())?;
    }
    let mut bad = 0usize;
    let (mut ep0, mut chunk0, mut loss0) = (0usize, 0usize, 0f64);
    if ck.resume {
        let d = std::fs::read(&ck.state).map_err(|e| format!("--resume: {}: {e}", ck.state))?;
        let mut r = crate::stopctl::Rd::new(&d);
        if r.str()? != "SNCK2" || r.str()? != tag {
            return Err(format!("{}: not a Muon+SF smallnet {tag} checkpoint", ck.state));
        }
        let sum = r.u64()?;
        if sum != ck.data_sum {
            return Err(format!("{}: checkpoint samples {sum:016x} != this run {:016x} (other data / flags)", ck.state, ck.data_sum));
        }
        ep0 = r.u64()? as usize;
        chunk0 = r.u64()? as usize;
        step = r.u64()? as usize;
        sum_lr2 = r.f64()?;
        bad = r.u64()? as usize;
        loss0 = r.f64()?;
        best.0 = r.f64()?;
        best_saved = r.f64()?;
        best.2 = r.u64()? as usize;
        best.1 = r.f32s()?;
        zz = r.f32s()?;
        xx = r.f32s()?;
        vv = r.f32s()?;
        muon_m = r.f32s()?;
        for k in 0..NP {
            net.p[k] = (1.0 - b1) * zz[k] + b1 * xx[k];
        }
        println!("[smallnet] {tag} RESUMED (Muon+SF-AdamW) from {}: epoch {} batch {chunk0}, best epoch {} = {:.6}", ck.state, ep0 + 1, best.2, best.0);
    }
    #[allow(clippy::too_many_arguments)]
    let save_state = |ep: usize, chunk: usize, step: usize, sum_lr2: f64, bad: usize, loss: f64, best: &(f64, Vec<f32>, usize), best_saved: f64, zz: &[f32], xx: &[f32], vv: &[f32], muon_m: &[f32]| -> Result<(), String> {
        let mut w = crate::stopctl::Wr::default();
        w.str("SNCK2").str(tag).u64(ck.data_sum).u64(ep as u64).u64(chunk as u64).u64(step as u64).f64(sum_lr2).u64(bad as u64).f64(loss).f64(best.0).f64(best_saved).u64(best.2 as u64).f32s(&best.1).f32s(zz).f32s(xx).f32s(vv).f32s(muon_m);
        crate::stopctl::write_atomic(&ck.state, &w.0)
    };
    for ep in ep0.. {
        let t0 = Instant::now();
        for k in 0..NP {
            net.p[k] = (1.0 - b1) * zz[k] + b1 * xx[k];
        }
        let mut idx: Vec<u32> = (0..train.len() as u32).collect();
        let mut rng = Rng((seed ^ (ep as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15)) | 1);
        for i in (1..idx.len()).rev() {
            let j = (rng.next() % (i as u64 + 1)) as usize;
            idx.swap(i, j);
        }
        let mut loss_sum = if ep == ep0 { loss0 } else { 0.0 };
        let first = if ep == ep0 { chunk0 } else { 0 };
        for (ci, chunk) in idx.chunks(batch).enumerate().skip(first) {
            if chunk.len() < batch / 2 {
                break;
            }
            if let Some(f) = crate::stopctl::requested(Some(&ck.run_dir)) {
                save_state(ep, ci, step, sum_lr2, bad, loss_sum, &best, best_saved, &zz, &xx, &vv, &muon_m)?;
                println!("[smallnet] {tag} STOPPED by {f} at epoch {} batch {ci}: state saved to {}, best so far (epoch {}) in {}; continue with the same command + --resume", ep + 1, ck.state, best.2, ck.best);
                net.p = best.1.clone();
                return Ok(FitEnd::Stopped(f));
            }
            step += 1;
            let lr = if step <= warmup { lr0 * (step as f32 / warmup as f32) } else { lr0 };
            let per = (chunk.len() + threads - 1) / threads;
            let netr = &*net;
            let ls: Vec<f64> = std::thread::scope(|sc| {
                let hs: Vec<_> = grads
                    .iter_mut()
                    .zip(chunk.chunks(per))
                    .map(|(g, part)| {
                        sc.spawn(move || {
                            g.iter_mut().for_each(|x| *x = 0.0);
                            let mut l = 0f64;
                            for &si in part {
                                let s = &train[si as usize];
                                let f = netr.forward(s);
                                let t = s.target as f32 / 400.0;
                                let d = t - f.y;
                                let dy = match tau {
                                    None => {
                                        l += (d * d) as f64;
                                        -2.0 * d
                                    }
                                    Some(q) => {
                                        l += (if d > 0.0 { q * d } else { (q - 1.0) * d }) as f64;
                                        if d > 0.0 {
                                            -q
                                        } else {
                                            1.0 - q
                                        }
                                    }
                                };
                                netr.backward(s, &f, dy / part.len() as f32, g);
                            }
                            l
                        })
                    })
                    .collect();
                hs.into_iter().map(|h| h.join().unwrap()).collect()
            });
            loss_sum += ls.iter().sum::<f64>();
            let used = chunk.chunks(per).count();
            let bc2 = 1.0 - b2.powi((step.min(i32::MAX as usize)) as i32);
            sum_lr2 += (lr as f64) * (lr as f64);
            let c = ((lr as f64) * (lr as f64) / sum_lr2) as f32;
            // Muon (Nesterov + NS5) on the NB phase-bucket H x D2 hidden matrices
            let blk = H * D2;
            let scale = ((H as f32) / (D2 as f32)).max(1.0).sqrt();
            let mut u = vec![0f32; blk];
            for bk in 0..NB {
                let lo = bk * blk;
                let off = O_L1W + lo;
                let mut any = false;
                for i in 0..blk {
                    let mut gk = 0f32;
                    for g in grads.iter().take(used) {
                        gk += g[off + i];
                    }
                    gk /= used as f32;
                    any |= gk != 0.0 || muon_m[lo + i] != 0.0;
                    let mm = mu_mom * muon_m[lo + i] + gk;
                    muon_m[lo + i] = mm;
                    u[i] = gk + mu_mom * mm;
                }
                if any {
                    let o = crate::train::newton_schulz5(&u, H, D2);
                    for i in 0..blk {
                        zz[off + i] -= lr * scale * o[i];
                    }
                }
            }
            for k in 0..NP {
                if (O_L1W..O_L1B).contains(&k) {
                    xx[k] += c * (zz[k] - xx[k]);
                    net.p[k] = (1.0 - b1) * zz[k] + b1 * xx[k];
                    continue;
                }
                let mut gk = 0f32;
                for g in grads.iter().take(used) {
                    gk += g[k];
                }
                gk /= used as f32;
                if gk != 0.0 {
                    vv[k] = b2 * vv[k] + (1.0 - b2) * gk * gk;
                    zz[k] -= lr * gk / ((vv[k] / bc2).sqrt() + eps);
                }
                xx[k] += c * (zz[k] - xx[k]);
                net.p[k] = (1.0 - b1) * zz[k] + b1 * xx[k];
            }
        }
        // LAWS D5 (D1554): evaluate and save the averaged iterate x
        net.p.copy_from_slice(&xx);
        let r = report(net, val);
        let obj = val_objective(net, val, tau);
        if obj < best_saved {
            best_saved = obj;
            best.1.clone_from(&xx);
            // the best weights as an engine blob, the moment they improve (LAWS A14)
            crate::stopctl::write_atomic(&ck.best, &net.to_bytes())?;
        }
        let gain = obj <= best.0 - 1e-5;
        println!(
            "[smallnet] {tag} epoch {} lr {:.2e} loss {:.5} | val objective {:.6}{} | val RMSE {:.1} cp, P(bound>truth) {:.2}% (end {:.2}%), slack {:.1} cp | {:.0}s",
            ep + 1,
            lr0,
            loss_sum / (train.len() as f64),
            obj,
            if gain { " (best >=1e-5)" } else { "" },
            r.rmse_cp,
            100.0 * r.viol,
            100.0 * r.viol_end,
            r.slack_mean,
            t0.elapsed().as_secs_f64()
        );
        if gain {
            best.0 = obj;
            best.2 = ep + 1;
            bad = 0;
        } else {
            bad += 1;
        }
        save_state(ep + 1, 0, step, sum_lr2, bad, 0.0, &best, best_saved, &zz, &xx, &vv, &muon_m)?;
        if bad >= patience {
            println!("[smallnet] {tag} SATURATED (LAWS C15, D1553/D1554): {patience} epochs without a val gain >= 1e-5, best epoch {} = {:.6} (best_saved {best_saved:.6})", best.2, best.0);
            break;
        }
    }
    net.p = best.1;
    Ok(FitEnd::Saturated)
}

/// Max |hub forward - engine SmallNet::bound| in cp over the samples.
fn engine_parity(bytes: &[u8], net: &Net, samples: &[Sample]) -> Result<f64, String> {
    let eng = titanium::smallnet::SmallNet::parse(bytes)?;
    let mut worst = 0f64;
    for s in samples {
        let mine = (400.0 * net.forward(s).y).round().clamp(-30000.0, 30000.0) as f64;
        let theirs = eng.bound(s.own, s.opp, s.blk, s.parent as i32, s.hand as i32) as f64;
        worst = worst.max((mine - theirs).abs());
    }
    Ok(worst)
}

pub fn run_smallnet(args: &[String]) -> ExitCode {
    match run(args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("smallnet: {e}");
            ExitCode::FAILURE
        }
    }
}

fn run(args: &[String]) -> Result<(), String> {
    let flag = |n: &str| args.iter().position(|a| a == n).and_then(|i| args.get(i + 1)).cloned();
    let need = |n: &str| flag(n).ok_or_else(|| format!("{n} is required"));
    let num = |n: &str, d: f64| flag(n).and_then(|x| x.parse::<f64>().ok()).unwrap_or(d);
    let net_path = need("--net")?;
    let train_stem = need("--train")?;
    let val_stem = need("--val")?;
    let out = need("--out")?;
    // D1130 (user: "remove any way to disobey me accidentally"): the default is EVERY row of --train; fewer rows only with
    // --experiment NAME (output under experiments/), so a subset can never be trained by accident
    let experiment = flag("--experiment");
    let rows_req = flag("--rows").map(|x| x.parse::<usize>().map_err(|e| format!("--rows {x}: {e}"))).transpose()?;
    let val_rows = num("--val-rows", 60_000.0) as usize;
    let every = num("--child-every", 3.0) as u64;
    // LAWS C15 (D868/D1553): no epoch limit of any size; standard runs always stop only after 4 epochs without a val gain >= 1e-5
    if args.iter().any(|a| a == "--epochs-mse" || a == "--epochs-pin" || a == "--max-epochs") {
        return Err("--epochs-mse / --epochs-pin / --max-epochs are forbidden (LAWS C15: train to saturation)".into());
    }
    if experiment.is_none() && args.iter().any(|a| a == "--patience") {
        return Err("--patience on standard runs is forbidden (LAWS C15: always 4 epochs without val gain >= 1e-5); use --experiment NAME".into());
    }
    let patience = (num("--patience", 4.0) as usize).max(1);
    let tau = num("--tau", 0.02) as f32;
    let lr = num("--lr", 0.002) as f32;
    let threads = num("--threads", 4.0) as usize;
    let seed = num("--seed", 1.0) as u64;
    if std::path::Path::new(&out).exists() {
        return Err(format!("{out} exists (refusing to overwrite)"));
    }
    // LAWS A14: checkpoint + best snapshot next to --out; STOP in that folder (or C:/gitProjects/STOP_ALL) stops safely
    let resume = args.iter().any(|a| a == "--resume");
    let (ck_state, ck_best) = (format!("{out}.ckpt"), format!("{out}.best.tsn"));
    let run_dir = std::path::Path::new(&out).parent().map(|d| if d.as_os_str().is_empty() { std::path::PathBuf::from(".") } else { d.to_path_buf() }).unwrap_or_else(|| ".".into());
    if resume && !std::path::Path::new(&ck_state).exists() {
        return Err(format!("--resume: no checkpoint {ck_state}"));
    }
    if !resume && std::path::Path::new(&ck_state).exists() {
        return Err(format!("{ck_state} exists: continue it with --resume, or delete it to start over"));
    }
    if let Some(f) = crate::stopctl::requested(Some(&run_dir)) {
        return Err(format!("stop file {f} exists: remove it to train"));
    }

    let mut tr = Tdat::open(&train_stem)?;
    let rows = crate::trainer::whole_rows_or_experiment("smallnet", &tr, rows_req, &experiment, &out, every, std::mem::size_of::<Sample>())?;
    let bytes = std::fs::read(&net_path).map_err(|e| format!("{net_path}: {e}"))?;
    let big = TrainableNet::from_tnn_bytes(&bytes)?;
    println!("[smallnet] big net {} (width {}), tau {tau}, {rows} source rows, child 1/{every}", net_path, big.width);

    let t0 = Instant::now();
    let mut va = Tdat::open(&val_stem)?;
    let tr_rows = load_val(&mut tr, rows)?;
    let va_rows = load_val(&mut va, val_rows)?;
    let train = build_samples(&tr_rows, &big, every, threads, seed);
    let val = build_samples(&va_rows, &big, (every / 2).max(1), threads, seed ^ 0x5555);
    println!("[smallnet] samples: train {} ({:.0} MB), val {} in {:.0}s", train.len(), (train.len() * std::mem::size_of::<Sample>()) as f64 / 1e6, val.len(), t0.elapsed().as_secs_f64());

    // LAWS C14: never from random weights; warm start from the shipped (or a previous) SmallNet, straight into the
    // lower-bound objective (an MSE phase would first pull the bound to the mean and throw the init's tail away)
    let Some(p) = flag("--init") else {
        return Err("--init SMALLNET.tsn is required (LAWS C14: no random init)".into());
    };
    let mut net = Net::parse(&std::fs::read(&p).map_err(|e| format!("{p}: {e}"))?)?;
    print_report(&format!("BEFORE (init {p})"), &report(&net, &val));
    let ck = Ck { state: ck_state.clone(), best: ck_best.clone(), run_dir, data_sum: samples_checksum(&train) ^ samples_checksum(&val).rotate_left(1), resume };
    if let FitEnd::Stopped(f) = fit(&mut net, &train, &val, patience, Some(tau), lr * 0.5, threads, seed ^ 0xAAAA, "pinball", &ck)? {
        print_report("BEST SO FAR (stopped)", &report(&net, &val));
        println!("[smallnet] stopped by {f}: NOT saturated; best so far = {ck_best}, state = {ck_state}; run the same command + --resume to continue");
        return Ok(());
    }
    print_report("AFTER", &report(&net, &val));

    let blob = net.to_bytes();
    let par = engine_parity(&blob, &net, &val[..val.len().min(5000)])?;
    println!("[smallnet] engine parity (SmallNet::parse + bound vs hub forward, 5000 val samples): max |diff| = {par:.1} cp");
    if par > 1.0 {
        return Err(format!("engine parity failed: {par} cp"));
    }
    std::fs::write(&out, &blob).map_err(|e| format!("{out}: {e}"))?;
    println!("[smallnet] wrote {out} ({} bytes)", blob.len());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hub_forward_matches_engine_bound_on_random_nets() {
        let net = Net::random(7);
        let mut s = 0x1234_5678_9ABC_DEF1u64;
        let mut next = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        };
        let full = (1u64 << 49) - 1;
        let samples: Vec<Sample> = (0..300)
            .map(|_| {
                let a = next() & next() & full;
                let b = next() & next() & full & !a;
                let c = next() & next() & next() & full & !a & !b;
                Sample { own: a, opp: b, blk: c, parent: (next() % 600) as i16 - 300, hand: (next() % 600) as i16 - 300, target: 0 }
            })
            .collect();
        let bytes = net.to_bytes();
        let worst = engine_parity(&bytes, &net, &samples).unwrap();
        assert!(worst <= 1.0, "hub forward vs engine bound: {worst} cp");
    }

    #[test]
    fn roundtrip_bytes() {
        let n = Net::random(3);
        assert_eq!(Net::parse(&n.to_bytes()).unwrap().p, n.p);
    }

    #[test]
    fn gradient_matches_finite_difference() {
        let mut net = Net::random(11);
        let s = Sample { own: 0b1011_0110, opp: 0b100_0101_0000, blk: 1 << 40, parent: 80, hand: -40, target: 30 };
        let f = net.forward(&s);
        let mut g = vec![0f32; NP];
        net.backward(&s, &f, 1.0, &mut g);
        for &k in &[O_FT + 3 * W + 2, O_FTB + 5, O_L1W + 17, O_L1B + 2, O_OW + 1, O_OB, O_SK, O_SK + NB] {
            let e = 1e-3f32;
            let y0 = net.forward(&s).y;
            net.p[k] += e;
            let y1 = net.forward(&s).y;
            net.p[k] -= e;
            let fd = (y1 - y0) / e;
            assert!((fd - g[k]).abs() < 5e-2 * (1.0 + fd.abs()), "param {k}: fd {fd} vs bp {}", g[k]);
        }
    }
}

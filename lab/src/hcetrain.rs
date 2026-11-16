//! `lab hce-train` (D977): train the hand-crafted eval EXACTLY like the NNUE: same dataset, same target, same loss, same
//! validation, stop only at saturation (LAWS C15), initialised from the closed-form fit to the champion (LAWS C14).
//!
//!   lab hce-train --data train_v10.tdat --val val_v9_clean.tdat [--set 4] [--threads 7] [--bs 16384] [--lr 2.0]
//!                 [--max-rows N (smoke tests only)] [--out logs/hce/run]
//!
//! Model: linear in the chosen hcecmp term set (default 4 = ours + gataxx, 844 weights, cp units): y = w.x / 400 (sigmoid units).
//! Target (hub targets.rs, champion recipe D330/D427): proven rows (|cp| >= 98,000) = exact 1 / 0; otherwise
//!   (1 - w) * sigmoid(clamp(cp, +-1500) / 400) + w * outcome, w = 0.55 (<= 6 empties), 0.55 (7..12), 0.25 (>= 13).
//! Loss: wsum * |sigmoid(y) - target|^2.5 (hub train.rs). Validation: target_of = 0.75 * sigmoid + 0.25 * outcome on val rows,
//! the same metric printed for the bundled NNUE on the same rows. Saturation: stop when 4 epochs in a row do not improve the
//! best validation loss by >= 1e-7 (lr halves after each non-improving epoch).
use crate::hcecmp::{block, feats, masks, nbk, set_nbk};
use crate::hcefit::{dot, phase, Ridge, CLAMP};
use crate::util::{arg, ring1, Rec, REC};
use std::fs::File;
use std::io::{BufReader, Read, Write};
use std::process::ExitCode;
use titanium::nnue::Nnue;
use titanium::{Board, MoveList};

const MATE: f32 = 100_000.0;

fn sig(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}
fn sig_cp(cp: f32) -> f32 {
    sig(cp.clamp(-1500.0, 1500.0) / 400.0)
}

/// training target (target_train_rw with result weight 0.55/0.55/0.25) or validation target (target_of)
fn target(r: &Rec, train: bool) -> f32 {
    if r.cp >= MATE - 2000.0 {
        return 1.0;
    }
    if r.cp <= -(MATE - 2000.0) {
        return 0.0;
    }
    if !train {
        return 0.75 * sig_cp(r.cp) + 0.25 * r.outcome;
    }
    let empties = 49 - (r.occ0 | r.occ1).count_ones();
    let w = if empties <= 12 { 0.55 } else { 0.25 };
    (1.0 - w) * sig_cp(r.cp) + w * r.outcome
}

fn board(r: &Rec) -> Board {
    let (own, opp) = if r.stm == 0 { (r.occ0, r.occ1) } else { (r.occ1, r.occ0) };
    Board::from_occ(own, opp, 0, 0)
}

fn read_rows(path: &str, max: usize) -> Vec<Rec> {
    let f = File::open(path).expect("tdat");
    let n = (f.metadata().map(|m| m.len() as usize).unwrap_or(0) / REC).min(max);
    let mut rd = BufReader::with_capacity(1 << 22, f);
    let mut buf = vec![0u8; REC];
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        if rd.read_exact(&mut buf).is_err() {
            break;
        }
        out.push(Rec::from_bytes(&buf));
    }
    out
}

pub fn run(args: &[String]) -> ExitCode {
    let (Some(data), Some(valp)) = (arg(args, "--data"), arg(args, "--val")) else {
        eprintln!("usage: lab hce-train --data X.tdat --val V.tdat [--set 4] [--threads 7] [--bs 16384] [--lr 2.0] [--max-rows N] [--out PREFIX]");
        return ExitCode::from(2);
    };
    let g = |k: &str, d: f64| arg(args, k).and_then(|v| v.parse::<f64>().ok()).unwrap_or(d);
    let set = g("--set", 4.0) as usize;
    let threads = g("--threads", 7.0) as usize;
    let bs = g("--bs", 16384.0) as usize;
    let mut lr = g("--lr", 2.0) as f32;
    let max_rows = g("--max-rows", 1e12) as usize;
    let out = arg(args, "--out").unwrap_or_else(|| "C:/gitProjects/titanium-ataxx/logs/hce/d977_hcetrain".into());
    set_nbk(g("--buckets", 16.0) as usize);
    let d = nbk() * block(set);
    let ring = ring1();
    let m = masks();
    let net = Nnue::bundled();
    let t0 = std::time::Instant::now();
    let mut log = String::new();
    let mut say = |s: String, log: &mut String| {
        println!("{s}");
        log.push_str(&s);
        log.push('\n');
        let _ = std::fs::write(format!("{out}.log"), log.as_bytes());
    };

    let train = read_rows(&data, max_rows);
    let val = read_rows(&valp, 600_000);
    say(format!("hce-train: set {set}, {} buckets ({d} weights), {} train rows, {} val rows, {threads} threads, bs {bs} ({:.0}s)", nbk(), train.len(), val.len(), t0.elapsed().as_secs_f64()), &mut log);

    // init (C14): closed-form ridge fit to the champion eval on the first 20k train positions + their children
    let mut ridge = Ridge::new(d);
    let mut ml = MoveList::new();
    let mut x = Vec::new();
    for r in train.iter().step_by((train.len() / 20_000).max(1)).take(20_000) {
        let b = board(r);
        b.legal_moves_dedup_into(&mut ml);
        feats(set, &b, &ring, &m, &mut x);
        ridge.add(&x, (net.eval(&b, 0) as f64).clamp(-CLAMP, CLAMP));
        for mv in ml.iter() {
            let c = b.make(mv);
            feats(set, &c, &ring, &m, &mut x);
            ridge.add(&x, (net.eval(&c, 0) as f64).clamp(-CLAMP, CLAMP));
        }
    }
    let mut w: Vec<f32> = ridge.solve(&vec![true; d], 1.0).iter().map(|&v| v as f32).collect();

    // validation loss (|p - t|^2.5, wsum-weighted) and MSE, for any scorer giving cp
    let vloss = |f: &(dyn Fn(&Board) -> f32 + Sync)| -> (f64, f64) {
        let chunk = val.len().div_ceil(threads);
        let parts: Vec<(f64, f64, f64)> = std::thread::scope(|sc| {
            val.chunks(chunk)
                .map(|vs| {
                    sc.spawn(move || {
                        let (mut l, mut q, mut ws) = (0f64, 0f64, 0f64);
                        for r in vs {
                            let p = sig(f(&board(r)) / 400.0);
                            let t = target(r, false);
                            let wt = r.wsum.max(0.0) as f64;
                            l += wt * ((p - t).abs() as f64).powf(2.5);
                            q += wt * ((p - t) as f64).powi(2);
                            ws += wt;
                        }
                        (l, q, ws)
                    })
                })
                .collect::<Vec<_>>()
                .into_iter()
                .map(|h| h.join().unwrap())
                .collect()
        });
        let ws: f64 = parts.iter().map(|p| p.2).sum();
        (parts.iter().map(|p| p.0).sum::<f64>() / ws, parts.iter().map(|p| p.1).sum::<f64>() / ws)
    };
    let (nn_l, nn_q) = vloss(&|b: &Board| net.eval(b, 0) as f32);
    let hce = |w: &[f32]| {
        let w = w.to_vec();
        move |b: &Board| {
            let mut x = Vec::new();
            feats(set, b, &ring1(), &masks(), &mut x);
            x.iter().map(|&(i, v)| w[i as usize] * v).sum::<f32>()
        }
    };
    let (i_l, i_q) = vloss(&hce(&w));
    say(format!("reference: champion NNUE val loss {nn_l:.6} (mse {nn_q:.6}); HCE init (fit to the NNUE) {i_l:.6} (mse {i_q:.6}) ({:.0}s)", t0.elapsed().as_secs_f64()), &mut log);

    // AdamW-free plain Adam on the linear weights
    let (mut mo, mut ve) = (vec![0f32; d], vec![0f32; d]);
    let (b1, b2, eps) = (0.9f32, 0.999f32, 1e-8f32);
    let mut step = 0i32;
    let mut best = (i_l, w.clone(), 0usize);
    let mut bad = 0usize;
    let mut order: Vec<usize> = (0..train.len()).collect();
    let mut rng = 0x9E37_79B9_7F4A_7C15u64;
    for epoch in 1.. {
        for i in (1..order.len()).rev() {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            order.swap(i, (rng % (i as u64 + 1)) as usize);
        }
        let (mut tl, mut tw) = (0f64, 0f64);
        for batch in order.chunks(bs) {
            let chunk = batch.len().div_ceil(threads);
            let wr = &w;
            let trr = &train;
            let parts: Vec<(Vec<f32>, f64, f64)> = std::thread::scope(|sc| {
                batch
                    .chunks(chunk)
                    .map(|idx| {
                        sc.spawn(move || {
                            let ring = ring1();
                            let m = masks();
                            let mut gr = vec![0f32; wr.len()];
                            let mut x = Vec::new();
                            let (mut l, mut ws) = (0f64, 0f64);
                            for &k in idx {
                                let r = &trr[k];
                                let wt = r.wsum.max(0.0);
                                if wt == 0.0 {
                                    continue;
                                }
                                feats(set, &board(r), &ring, &m, &mut x);
                                let y = x.iter().map(|&(i, v)| wr[i as usize] * v).sum::<f32>() / 400.0;
                                let p = sig(y);
                                let t = target(r, true);
                                let diff = p - t;
                                let ad = diff.abs();
                                l += (wt * ad.powf(2.5)) as f64;
                                ws += wt as f64;
                                let dy = wt * 2.5 * diff.signum() * ad.powf(1.5) * p * (1.0 - p) / 400.0;
                                for &(i, v) in &x {
                                    gr[i as usize] += dy * v;
                                }
                            }
                            (gr, l, ws)
                        })
                    })
                    .collect::<Vec<_>>()
                    .into_iter()
                    .map(|h| h.join().unwrap())
                    .collect()
            });
            let ws: f64 = parts.iter().map(|p| p.2).sum::<f64>().max(1e-9);
            tl += parts.iter().map(|p| p.1).sum::<f64>();
            tw += ws;
            step += 1;
            let (c1, c2) = (1.0 - b1.powi(step), 1.0 - b2.powi(step));
            for i in 0..d {
                let gi = parts.iter().map(|p| p.0[i]).sum::<f32>() / ws as f32;
                mo[i] = b1 * mo[i] + (1.0 - b1) * gi;
                ve[i] = b2 * ve[i] + (1.0 - b2) * gi * gi;
                w[i] -= lr * (mo[i] / c1) / ((ve[i] / c2).sqrt() + eps);
            }
        }
        let (vl, vq) = vloss(&hce(&w));
        let improved = vl < best.0 - 1e-7;
        if improved {
            best = (vl, w.clone(), epoch);
            bad = 0;
        } else {
            bad += 1;
            lr *= 0.5;
        }
        say(format!(
            "epoch {epoch}: train loss {:.6} | val loss {vl:.6} (mse {vq:.6}) | best {:.6} @ {} | lr {lr:.3} | NNUE {nn_l:.6} | {:.0}s",
            tl / tw.max(1e-9), best.0, best.2, t0.elapsed().as_secs_f64()
        ), &mut log);
        if bad >= 4 {
            say(format!("SATURATED: 4 epochs without improvement; best epoch {} val loss {:.6} (init {i_l:.6}, NNUE {nn_l:.6})", best.2, best.0), &mut log);
            break;
        }
    }
    // sibling best-move agreement with the NNUE, init vs trained (2,000 val positions)
    let agree = |wv: &[f32]| -> f64 {
        let (mut ag, mut n) = (0f64, 0f64);
        let mut ml = MoveList::new();
        let mut x = Vec::new();
        for r in val.iter().step_by((val.len() / 2000).max(1)).take(2000) {
            let b = board(r);
            b.legal_moves_dedup_into(&mut ml);
            if ml.len() < 2 {
                continue;
            }
            let (mut bn, mut bh) = ((f32::MIN, 0usize), (f32::MIN, 0usize));
            for (i, mv) in ml.iter().enumerate() {
                let c = b.make(mv);
                let yn = -(net.eval(&c, 0) as f32);
                feats(set, &c, &ring, &m, &mut x);
                let yh = -dot(&wv.iter().map(|&v| v as f64).collect::<Vec<_>>(), &x) as f32;
                if yn > bn.0 {
                    bn = (yn, i);
                }
                if yh > bh.0 {
                    bh = (yh, i);
                }
            }
            ag += (bn.1 == bh.1) as u8 as f64;
            n += 1.0;
        }
        100.0 * ag / n
    };
    let init_w: Vec<f32> = ridge.solve(&vec![true; d], 1.0).iter().map(|&v| v as f32).collect();
    say(format!("sibling best-move agreement with the NNUE: init (fit to NNUE) {:.2}%, trained like the NNUE {:.2}%", agree(&init_w), agree(&best.1)), &mut log);
    let js = format!("{{\"set\":{set},\"dims\":{d},\"best_epoch\":{},\"val_loss\":{},\"weights\":{:?}}}", best.2, best.0, best.1);
    let _ = File::create(format!("{out}_weights.json")).and_then(|mut f| f.write_all(js.as_bytes()));
    let _ = phase(0);
    ExitCode::SUCCESS
}

//! `lab loss-elo-corr` & `lab spsa-blend` (D1270-D1272):
//! 1. Measure & optimize how per-bucket training targets (`w_res[0..16]`) and loss weights (`w_loss[0..16]`)
//!    correlate with actual `eval-gate 5k` Elo across our benchmarked net panel.
//! 2. Precompute raw `.tdat` + chain TD(lambda) + global D4 pooled WDL + backward proof distance once in RAM
//!    on preserved pre-`proof-wdl` data (`train_v14h_a.tdat` + `train_v14.chains`), then run SPSA over:
//!      - `w_res[0..16]` (16 NNUE buckets matching `(empties * 16 / 50).min(15)`)
//!      - `gamma_proof` (replacing the guessed `0.9^n` decay)
//!      - `pool_trust` (shrinking single-game proof override when multi-game `N_t > 1` WDL exists)
//!      - `proof_direct` (decoupling near-proof pull from `w_res`)
//!      - `td_alpha` (mixing `TD(lambda)` search backup from later chain plies)
//!      - `proof_blend` (softening exact proofs toward `cp5k`)
//!    by refitting per-bucket heads on a frozen trunk (~seconds/step) and playing `eval-gate` matches (`$0` Modal).

use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::Path;
use std::process::ExitCode;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use titanium::nnfeat::both as feat_both;
use titanium::nnue::Nnue;
use titanium::Board;

use crate::evalgate::{play_match, Ev};
use crate::util::{arg, REC};

const NB: usize = 16;

#[derive(Clone, Copy)]
struct FullRec {
    occ0: u64,
    occ1: u64,
    outcome: f32,
    cp: f32,
    wsum: f32,
    cp5k: f32,
    stm: u8,
    proof_kind: u8,
}

impl FullRec {
    fn from_bytes(b: &[u8]) -> Self {
        Self {
            occ0: u64::from_le_bytes(b[0..8].try_into().unwrap()),
            occ1: u64::from_le_bytes(b[8..16].try_into().unwrap()),
            outcome: f32::from_le_bytes(b[16..20].try_into().unwrap()),
            cp: f32::from_le_bytes(b[20..24].try_into().unwrap()),
            wsum: f32::from_le_bytes(b[24..28].try_into().unwrap()),
            cp5k: f32::from_le_bytes(b[32..36].try_into().unwrap()),
            stm: b[40],
            proof_kind: b[42],
        }
    }
}

#[inline(always)]
fn sig(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

#[inline(always)]
fn sig_cp(cp: f32) -> f64 {
    sig((cp.clamp(-1500.0, 1500.0) / 400.0) as f64)
}

#[inline(always)]
fn bucket_of(empties: usize) -> usize {
    (empties * NB / 50).min(NB - 1)
}

#[inline(always)]
fn board_of(occ0: u64, occ1: u64, stm: u8) -> Board {
    let (own, opp) = if stm == 0 { (occ0, occ1) } else { (occ1, occ0) };
    Board::from_occ(own, opp, 0, 0)
}

#[inline(always)]
fn proof_value(r: &FullRec) -> Option<f64> {
    if r.cp >= 98_000.0 {
        Some(1.0)
    } else if r.cp <= -98_000.0 {
        Some(0.0)
    } else if matches!(r.proof_kind, 1 | 2 | 5) && r.cp == 0.0 {
        Some(0.5)
    } else {
        None
    }
}

fn d4_table() -> [[u8; 49]; 8] {
    let mut tab = [[0u8; 49]; 8];
    for k in 0..8usize {
        let flip = (k & 4) != 0;
        let rot = k & 3;
        for sq in 0..49usize {
            let (mut r, mut c) = (sq / 7, sq % 7);
            if flip {
                c = 6 - c;
            }
            for _ in 0..rot {
                let nr = c;
                let nc = 6 - r;
                r = nr;
                c = nc;
            }
            tab[k][sq] = (r * 7 + c) as u8;
        }
    }
    tab
}

#[inline]
fn xform_bb(bb: u64, map: &[u8; 49]) -> u64 {
    let mut out = 0u64;
    let mut x = bb;
    while x != 0 {
        let sq = x.trailing_zeros() as usize;
        x &= x - 1;
        out |= 1u64 << map[sq];
    }
    out
}

#[inline]
fn pool_key(r: &FullRec, tab: &[[u8; 49]; 8]) -> (u64, u64) {
    let (m, o) = if r.stm == 0 { (r.occ0, r.occ1) } else { (r.occ1, r.occ0) };
    let mut best = (u64::MAX, u64::MAX);
    for k in 0..8 {
        let key = (xform_bb(m, &tab[k]), xform_bb(o, &tab[k]));
        if key < best {
            best = key;
        }
    }
    best
}

fn read_chains(path: &str) -> Result<Vec<(u32, u16)>, String> {
    let b = fs::read(path).map_err(|e| format!("{path}: {e}"))?;
    if b.len() < 24 || &b[0..4] != b"TCHN" {
        return Err(format!("{path}: not a TCHN file"));
    }
    let n = u64::from_le_bytes(b[16..24].try_into().unwrap()) as usize;
    if b.len() != 24 + n * 8 {
        return Err(format!("{path}: {} bytes, expected {}", b.len(), 24 + n * 8));
    }
    Ok(b[24..]
        .chunks_exact(8)
        .map(|c| {
            (
                u32::from_le_bytes(c[0..4].try_into().unwrap()),
                u16::from_le_bytes(c[4..6].try_into().unwrap()),
            )
        })
        .collect())
}

fn read_strided_recs(path: &str, max_rows: usize) -> Vec<FullRec> {
    let mut f = File::open(path).expect("open tdat");
    let len = f.seek(SeekFrom::End(0)).expect("seek");
    let total = (len as usize) / REC;
    let n = max_rows.min(total).max(1);
    let stride = (total / n).max(1);
    let mut out = Vec::with_capacity(n);
    let mut buf = [0u8; REC];
    for k in 0..total {
        let idx = k * stride;
        if idx >= total || out.len() >= n {
            break;
        }
        f.seek(SeekFrom::Start((idx * REC) as u64)).expect("seek rec");
        if f.read_exact(&mut buf).is_err() {
            break;
        }
        let r = FullRec::from_bytes(&buf);
        if r.occ0 & r.occ1 == 0 {
            out.push(r);
        }
    }
    out
}

#[derive(Clone)]
struct RawTnn {
    magic: [u8; 4],
    num_feats: usize,
    width: usize,
    l1: usize,
    l2: usize,
    buckets: usize,
    groups: u32,
    pairwise: bool,
    ft: Vec<f32>,
    ftb: Vec<f32>,
    psqt: Vec<f32>,
    l1_w: Vec<f32>,
    l1_b: Vec<f32>,
    l2_w: Vec<f32>,
    l2_b: Vec<f32>,
    out_w: Vec<f32>,
    out_b: Vec<f32>,
    s57w: Vec<f32>,
    dis_w: Vec<f32>,
    dis_b: Vec<f32>,
}

impl RawTnn {
    fn load(path: &str) -> Result<Self, String> {
        let d = fs::read(path).map_err(|e| format!("{path}: {e}"))?;
        let magic: [u8; 4] = d[..4].try_into().map_err(|_| "short header")?;
        let v2 = &magic == b"TNN2";
        let dis = &magic == b"TNN3";
        if d.len() < 32 || (!v2 && !dis && &magic != b"TNN1") {
            return Err("not a TNN1/TNN2/TNN3 file".into());
        }
        let u = |i: usize| u32::from_le_bytes(d[4 + 4 * i..8 + 4 * i].try_into().unwrap()) as usize;
        let (num_feats, width, l1, l2, buckets, groups) = (u(0), u(1), u(2), u(3), u(4), u(5) as u32);
        let pairwise = (v2 || dis) && u(6) != 0;
        let din = if pairwise { width } else { 2 * width };
        let out_in = if v2 { l1 + l2 } else { l2 };
        let mut p = if v2 || dis { 32 } else { 28 };
        let mut take = |n: usize| -> Vec<f32> {
            let v = d[p..p + 4 * n].chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
            p += 4 * n;
            v
        };
        let out = RawTnn {
            magic,
            num_feats,
            width,
            l1,
            l2,
            buckets,
            groups,
            pairwise,
            ft: take(num_feats * width),
            ftb: take(width),
            psqt: take(num_feats * buckets),
            l1_w: take(buckets * l1 * din),
            l1_b: take(buckets * l1),
            l2_w: take(buckets * l2 * l1),
            l2_b: take(buckets * l2),
            out_w: take(buckets * out_in),
            out_b: take(buckets),
            s57w: if v2 { take(buckets) } else { Vec::new() },
            dis_w: if dis { take(buckets * (l1 + l2)) } else { Vec::new() },
            dis_b: if dis { take(buckets) } else { Vec::new() },
        };
        if p != d.len() {
            return Err(format!("trailing/missing bytes ({p} vs {})", d.len()));
        }
        Ok(out)
    }

    fn save(&self, path: &str) -> Result<(), String> {
        let mut out = Vec::with_capacity(self.ft.len() * 4 + 65536);
        out.extend_from_slice(&self.magic);
        for &val in &[self.num_feats as u32, self.width as u32, self.l1 as u32, self.l2 as u32, self.buckets as u32, self.groups] {
            out.extend_from_slice(&val.to_le_bytes());
        }
        if &self.magic == b"TNN2" || &self.magic == b"TNN3" {
            out.extend_from_slice(&(if self.pairwise { 1u32 } else { 0u32 }).to_le_bytes());
        }
        let mut push = |s: &[f32]| {
            for &x in s {
                out.extend_from_slice(&x.to_le_bytes());
            }
        };
        push(&self.ft);
        push(&self.ftb);
        push(&self.psqt);
        push(&self.l1_w);
        push(&self.l1_b);
        push(&self.l2_w);
        push(&self.l2_b);
        push(&self.out_w);
        push(&self.out_b);
        if &self.magic == b"TNN2" {
            push(&self.s57w);
        }
        if &self.magic == b"TNN3" {
            push(&self.dis_w);
            push(&self.dis_b);
        }
        if let Some(par) = Path::new(path).parent() {
            let _ = fs::create_dir_all(par);
        }
        fs::write(path, out).map_err(|e| format!("{path}: {e}"))
    }

    fn forward_h2(&self, b: &Board, acc: &mut [f32], x: &mut [f32]) -> (usize, f32, [f32; 32], f32) {
        let w = self.width;
        let nb = self.buckets;
        let l1 = self.l1;
        let l2 = self.l2;
        let din = x.len();
        let bk = bucket_of(b.empty().count_ones() as usize);
        let (us, nu, them, nt) = feat_both(b, self.groups);
        let mut ps = [0f32; 2];
        for (half, (fidx, n)) in [(&us, nu), (&them, nt)].into_iter().enumerate() {
            let off = half * w;
            acc[off..off + w].copy_from_slice(&self.ftb);
            for &f in &fidx[..n] {
                let fi = f as usize;
                let row = &self.ft[fi * w..(fi + 1) * w];
                for j in 0..w {
                    acc[off + j] += row[j];
                }
                ps[half] += self.psqt[fi * nb + bk];
            }
            for j in 0..w {
                acc[off + j] = acc[off + j].clamp(0.0, 1.0);
            }
        }
        if self.pairwise {
            let h = w / 2;
            for half in 0..2 {
                let a = &acc[half * w..(half + 1) * w];
                for j in 0..h {
                    x[half * h + j] = a[j] * a[h + j];
                }
            }
        } else {
            x.copy_from_slice(acc);
        }
        let mut h1 = [0f32; 64];
        for j in 0..l1 {
            let r_idx = bk * l1 + j;
            let row = &self.l1_w[r_idx * din..(r_idx + 1) * din];
            let mut s = self.l1_b[r_idx];
            for i in 0..din {
                s += row[i] * x[i];
            }
            h1[j] = s.clamp(0.0, 1.0);
        }
        let mut h2 = [0f32; 32];
        for k in 0..l2 {
            let r_idx = bk * l2 + k;
            let row = &self.l2_w[r_idx * l1..(r_idx + 1) * l1];
            let mut s = self.l2_b[r_idx];
            for j in 0..l1 {
                s += row[j] * h1[j];
            }
            h2[k] = s.clamp(0.0, 1.0);
        }
        let psqt_half = 0.5 * (ps[0] - ps[1]);
        let row = &self.out_w[bk * l2..(bk + 1) * l2];
        let mut mlp = self.out_b[bk];
        for k in 0..l2 {
            mlp += row[k] * h2[k];
        }
        (bk, psqt_half, h2, mlp + psqt_half)
    }
}

fn solve_linear(a: &mut Vec<Vec<f64>>, b: &mut [f64]) -> Vec<f64> {
    let n = b.len();
    for i in 0..n {
        let mut piv = i;
        let mut best = a[i][i].abs();
        for r in (i + 1)..n {
            if a[r][i].abs() > best {
                best = a[r][i].abs();
                piv = r;
            }
        }
        a.swap(i, piv);
        b.swap(i, piv);
        let diag = a[i][i];
        if diag.abs() < 1e-12 {
            continue;
        }
        for r in (i + 1)..n {
            let f = a[r][i] / diag;
            if f == 0.0 {
                continue;
            }
            b[r] -= f * b[i];
            for c in i..n {
                a[r][c] -= f * a[i][c];
            }
        }
    }
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut s = b[i];
        for c in (i + 1)..n {
            s -= a[i][c] * x[c];
        }
        x[i] = if a[i][i].abs() > 1e-12 { s / a[i][i] } else { 0.0 };
    }
    x
}

fn pearson(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len() as f64;
    if n < 2.0 {
        return 0.0;
    }
    let mx = x.iter().sum::<f64>() / n;
    let my = y.iter().sum::<f64>() / n;
    let (mut num, mut vx, mut vy) = (0.0, 0.0, 0.0);
    for (&a, &b) in x.iter().zip(y.iter()) {
        let dx = a - mx;
        let dy = b - my;
        num += dx * dy;
        vx += dx * dx;
        vy += dy * dy;
    }
    if vx <= 1e-18 || vy <= 1e-18 {
        0.0
    } else {
        num / (vx * vy).sqrt()
    }
}

fn ranks(v: &[f64]) -> Vec<f64> {
    let mut idx: Vec<usize> = (0..v.len()).collect();
    idx.sort_by(|&a, &b| v[a].partial_cmp(&v[b]).unwrap_or(std::cmp::Ordering::Equal));
    let mut r = vec![0.0; v.len()];
    let mut i = 0;
    while i < idx.len() {
        let mut j = i + 1;
        while j < idx.len() && (v[idx[j]] - v[idx[i]]).abs() < 1e-12 {
            j += 1;
        }
        let rk = 0.5 * ((i + j - 1) as f64) + 1.0;
        for k in i..j {
            r[idx[k]] = rk;
        }
        i = j;
    }
    r
}

fn spearman(x: &[f64], y: &[f64]) -> f64 {
    pearson(&ranks(x), &ranks(y))
}

fn mean(v: &[f64]) -> f64 {
    if v.is_empty() {
        0.0
    } else {
        v.iter().sum::<f64>() / v.len() as f64
    }
}

/// D1270 Stage 1: analyze correlation between per-bucket loss / concordance and real `eval-gate 5k` Elo.
pub fn run_corr(args: &[String]) -> ExitCode {
    let t0 = Instant::now();
    let val_path = arg(args, "--val").unwrap_or_else(|| "C:/gitProjects/titanium-ataxx/data/nnue/packed/val_v13tail.tdat".into());
    let n_val: usize = arg(args, "--n-val").map_or(300_000, |v| v.parse().unwrap());
    let threads: usize = arg(args, "--threads").map_or(7, |v| v.parse().unwrap());

    // Verified `eval-gate 5k` net panel: (label, path, consensus_elo, direct_vs_champ_elo, is_nnue)
    let specs: [(&str, &str, f64, Option<f64>, bool); 8] = [
        ("nn_v14muon", "C:/gitProjects/titanium-ataxx/runs/nn_v14muon/nn_v14muon/best.tnn", 24.5, Some(19.1), true),
        ("WS_SAFECAL", "C:/gitProjects/titanium-ataxx/nets/ablate/ws_v13anneal_safecal.tnn", 11.3, None, true),
        ("WS_SAFE", "C:/gitProjects/titanium-ataxx/nets/ablate/ws_v13anneal_safe.tnn", 10.0, None, true),
        ("nn_v13anneal", "C:/gitProjects/titanium-ataxx/runs/nn_v13anneal/nn_v13anneal/best.tnn", 9.1, Some(9.1), true),
        ("Champion", "C:/gitProjects/mainpromo-ataxx/data/nnue/nnue_main.tnn", 0.0, Some(0.0), true),
        ("nn_v13sat", "C:/gitProjects/titanium-ataxx/runs/nn_v13sat/nn_v13sat/best.tnn", -0.9, Some(-0.9), true),
        ("nn_v11bsat", "C:/gitProjects/titanium-ataxx/runs/nn_v11bsat/nn_v11bsat/best.tnn", -11.3, Some(-11.3), true),
        ("levatt_ep4", "C:/gitProjects/titanium-ataxx/logs/hce/d1203_levatt_attn_levatt.bin", -13.9, Some(-13.9), false),
    ];

    struct Model {
        name: &'static str,
        elo_all: f64,
        elo_dir: Option<f64>,
        is_nnue: bool,
        ev: Ev,
    }
    let mut models = Vec::new();
    for (name, path, ea, ed, is_nnue) in specs {
        if Path::new(path).exists() {
            models.push(Model { name, elo_all: ea, elo_dir: ed, is_nnue, ev: Ev::load(path) });
        } else {
            println!("[skip] missing net: {name} ({path})");
        }
    }
    if models.len() < 4 {
        eprintln!("loss-elo-corr: need at least 4 nets on disk");
        return ExitCode::FAILURE;
    }

    let recs = read_strided_recs(&val_path, n_val);
    let m = models.len();
    let nr = recs.len();
    println!("loss-elo-corr: loaded {m} nets, {nr} strided rows from {val_path}...");

    struct ValItem {
        bk: u8,
        sig_cp: f32,
        outcome: f32,
        proven: i8,
    }
    let items: Vec<ValItem> = recs
        .iter()
        .map(|r| {
            let empties = (49 - (r.occ0 | r.occ1).count_ones()) as usize;
            let proven = match proof_value(r) {
                Some(v) if v >= 0.99 => 1,
                Some(v) if v <= 0.01 => 0,
                _ => -1,
            };
            ValItem {
                bk: bucket_of(empties) as u8,
                sig_cp: sig_cp(r.cp) as f32,
                outcome: r.outcome.clamp(0.0, 1.0),
                proven,
            }
        })
        .collect();

    let mut preds: Vec<Vec<f32>> = vec![vec![0.0f32; nr]; m];
    let chunk = ((nr + threads - 1) / threads).max(1);
    std::thread::scope(|s| {
        for (out_slice, md) in preds.iter_mut().zip(models.iter()) {
            let parts: Vec<&mut [f32]> = out_slice.chunks_mut(chunk).collect();
            for (ci, p_chunk) in parts.into_iter().enumerate() {
                let start = ci * chunk;
                let end = (start + p_chunk.len()).min(nr);
                let sub_recs = &recs[start..end];
                let ev = &md.ev;
                s.spawn(move || {
                    for (j, r) in sub_recs.iter().enumerate() {
                        let b = board_of(r.occ0, r.occ1, r.stm);
                        p_chunk[j] = (1.0 / (1.0 + (-ev.eval_f(&b) as f64 / 400.0).exp())) as f32;
                    }
                });
            }
        }
    });
    println!("precomputed {} evaluations in {:.1}s.", m * nr, t0.elapsed().as_secs_f64());

    let mut bk_idx: [Vec<usize>; NB] = std::array::from_fn(|_| Vec::new());
    for (i, it) in items.iter().enumerate() {
        bk_idx[it.bk as usize].push(i);
    }

    // Per-model per-bucket baseline rw=[0.55,0.55,0.25] pow2.5 loss + concordance
    let mut conc_bk = vec![vec![0.0f64; NB]; m];
    let conc_pairs_max = 2000usize;
    for bk in 0..NB {
        let idxs = &bk_idx[bk];
        if idxs.len() < 20 {
            continue;
        }
        let base_w = if bk <= 3 { 0.55 } else { 0.25 };
        let mut pairs = Vec::with_capacity(conc_pairs_max);
        let step = (idxs.len() / conc_pairs_max.min(idxs.len())).max(1);
        for k in (0..idxs.len().saturating_sub(1)).step_by(step) {
            let a = idxs[k];
            let b = idxs[(k * 7 + 13) % idxs.len()];
            if a == b {
                continue;
            }
            let ta = if items[a].proven >= 0 {
                items[a].proven as f64
            } else {
                (1.0 - base_w) * items[a].sig_cp as f64 + base_w * items[a].outcome as f64
            };
            let tb = if items[b].proven >= 0 {
                items[b].proven as f64
            } else {
                (1.0 - base_w) * items[b].sig_cp as f64 + base_w * items[b].outcome as f64
            };
            if (ta - tb).abs() > 0.15 {
                pairs.push((a, b, ta - tb));
            }
            if pairs.len() >= conc_pairs_max {
                break;
            }
        }
        for mi in 0..m {
            let mut ok = 0usize;
            for &(a, b, dt) in &pairs {
                let dp = (preds[mi][a] - preds[mi][b]) as f64;
                if dp * dt > 0.0 {
                    ok += 1;
                }
            }
            conc_bk[mi][bk] = if pairs.is_empty() { 0.5 } else { ok as f64 / pairs.len() as f64 };
        }
    }

    let loss_grid = |bk: usize, w_res: f64| -> Vec<f64> {
        let mut out = vec![0.0; m];
        let idxs = &bk_idx[bk];
        if idxs.is_empty() {
            return out;
        }
        for &i in idxs {
            let it = &items[i];
            let t = if it.proven >= 0 {
                it.proven as f64
            } else {
                (1.0 - w_res) * it.sig_cp as f64 + w_res * it.outcome as f64
            };
            for mi in 0..m {
                out[mi] += (preds[mi][i] as f64 - t).abs().powf(2.5);
            }
        }
        let inv = 1.0 / idxs.len() as f64;
        for v in &mut out {
            *v *= inv;
        }
        out
    };

    let elo_all: Vec<f64> = models.iter().map(|x| x.elo_all).collect();
    let dir_idx: Vec<usize> = models.iter().enumerate().filter_map(|(i, x)| x.elo_dir.map(|_| i)).collect();
    let elo_dir: Vec<f64> = dir_idx.iter().map(|&i| models[i].elo_dir.unwrap()).collect();
    let nnue_idx: Vec<usize> = models.iter().enumerate().filter_map(|(i, x)| if x.is_nnue { Some(i) } else { None }).collect();
    let elo_nnue: Vec<f64> = nnue_idx.iter().map(|&i| models[i].elo_all).collect();

    let eval_corrs = |losses: &[f64]| -> (f64, f64, f64, f64) {
        let neg: Vec<f64> = losses.iter().map(|&v| -v).collect();
        let neg_dir: Vec<f64> = dir_idx.iter().map(|&i| -losses[i]).collect();
        let neg_nnue: Vec<f64> = nnue_idx.iter().map(|&i| -losses[i]).collect();
        (
            pearson(&neg, &elo_all),
            spearman(&neg, &elo_all),
            pearson(&neg_dir, &elo_dir),
            pearson(&neg_nnue, &elo_nnue),
        )
    };

    let rw_base = rw_base_init();
    let wl_base = [1.0f64; NB];
    let combine = |wr: &[f64; NB], wl: &[f64; NB]| -> Vec<f64> {
        let mut tot = vec![0.0; m];
        let mut wsum = 0.0;
        for bk in 0..NB {
            let n = bk_idx[bk].len() as f64;
            if n == 0.0 {
                continue;
            }
            let l = loss_grid(bk, wr[bk]);
            let weight = n * wl[bk];
            wsum += weight;
            for mi in 0..m {
                tot[mi] += weight * l[mi];
            }
        }
        for v in &mut tot {
            *v /= wsum.max(1e-12);
        }
        tot
    };

    let base_losses = combine(&rw_base, &wl_base);
    println!("\n=== Net Panel Baseline (`rw=[0.55,0.55,0.25]` pow2.5, unweighted buckets) ===");
    for (mi, md) in models.iter().enumerate() {
        let mc: f64 = (0..15).map(|b| conc_bk[mi][b]).sum::<f64>() / 15.0;
        println!(
            "  {:>12}: Elo {:+5.1} (dir {:>5}) | rw_loss {:.6} | conc {:.4}",
            md.name,
            md.elo_all,
            md.elo_dir.map_or("  n/a".into(), |v| format!("{v:+5.1}")),
            base_losses[mi],
            mc
        );
    }
    let (r_all0, rho_all0, r_dir0, r_nnue0) = eval_corrs(&base_losses);
    let mean_concs: Vec<f64> = (0..m).map(|mi| (0..15).map(|b| conc_bk[mi][b]).sum::<f64>() / 15.0).collect();
    println!(
        "  CORR(-rw_loss, Elo): all r={r_all0:+.3} rho={rho_all0:+.3} | direct6 r={r_dir0:+.3} | nnue7 r={r_nnue0:+.3}",
    );
    println!(
        "  CORR(concordance, Elo): all r={:+.3} | direct6 r={:+.3}",
        pearson(&mean_concs, &elo_all),
        pearson(&dir_idx.iter().map(|&i| mean_concs[i]).collect::<Vec<_>>(), &elo_dir)
    );

    println!("\n=== Per-Bucket Diagnostics (`bk 0..15`: rows, rw_loss of Champ/v13sat/v13an/v14mu/lev, best single-bucket w, conc r) ===");
    let find = |n: &str| models.iter().position(|x| x.name == n);
    let (ic, isat, ian, imu, ilv) = (find("Champion"), find("nn_v13sat"), find("nn_v13anneal"), find("nn_v14muon"), find("levatt_ep4"));
    let w_cands = [0.0, 0.15, 0.30, 0.45, 0.60, 0.75, 0.90, 1.0];
    let mut cache_bk_loss = vec![vec![vec![0.0; m]; w_cands.len()]; NB];
    for bk in 0..NB {
        for (wi, &w) in w_cands.iter().enumerate() {
            cache_bk_loss[bk][wi] = loss_grid(bk, w);
        }
        let n = bk_idx[bk].len();
        if n < 20 {
            println!("  bk {:>2} (n={:>5}): skipped (too few rows)", bk, n);
            continue;
        }
        let mut best_w = 0.25;
        let mut best_r = -9.0;
        for (wi, &w) in w_cands.iter().enumerate() {
            let neg: Vec<f64> = cache_bk_loss[bk][wi].iter().map(|&v| -v).collect();
            let r = pearson(&neg, &elo_all);
            if r > best_r {
                best_r = r;
                best_w = w;
            }
        }
        let base_bk_loss = loss_grid(bk, if bk <= 3 { 0.55 } else { 0.25 });
        let gl = |opt: Option<usize>| opt.map_or(f64::NAN, |mi| base_bk_loss[mi]);
        let r_conc = pearson(&(0..m).map(|mi| conc_bk[mi][bk]).collect::<Vec<_>>(), &elo_all);
        println!(
            "  bk {:>2} (n={:>5}): ch={:.4} sat={:.4} v13={:.4} v14={:.4} lev={:.4} | best_w={:.2} (r={:+.3}) | conc_r={:+.3}",
            bk,
            n,
            gl(ic),
            gl(isat),
            gl(ian),
            gl(imu),
            gl(ilv),
            best_w,
            best_r,
            r_conc
        );
    }

    // Coordinate ascent optimizing `(w_res[0..16], w_loss[0..16])` for max correlation + smoothness
    let mut wr = rw_base;
    let mut wl = wl_base;
    let eval_obj = |wr: &[f64; NB], wl: &[f64; NB]| -> (f64, f64, f64, f64, f64) {
        let mut tot = vec![0.0; m];
        let mut wsum = 0.0;
        for bk in 0..NB {
            let n = bk_idx[bk].len() as f64;
            if n == 0.0 {
                continue;
            }
            // interpolate from w_cands grid for speed
            let pos = (wr[bk] / 0.15).min((w_cands.len() - 1) as f64);
            let lo = (pos.floor() as usize).min(w_cands.len() - 2);
            let frac = (wr[bk] - w_cands[lo]) / (w_cands[lo + 1] - w_cands[lo]);
            let frac = frac.clamp(0.0, 1.0);
            let weight = n * wl[bk];
            wsum += weight;
            for mi in 0..m {
                let lv = (1.0 - frac) * cache_bk_loss[bk][lo][mi] + frac * cache_bk_loss[bk][lo + 1][mi];
                tot[mi] += weight * lv;
            }
        }
        for v in &mut tot {
            *v /= wsum.max(1e-12);
        }
        let (ra, rho, rd, rn) = eval_corrs(&tot);
        let pen_wr: f64 = (0..NB - 1).map(|b| (wr[b + 1] - wr[b]).powi(2)).sum();
        let pen_wl: f64 = (0..NB - 1).map(|b| (wl[b + 1].ln() - wl[b].ln()).powi(2)).sum();
        let obj = 0.45 * ra + 0.35 * rd + 0.20 * rn - 0.015 * pen_wr - 0.005 * pen_wl;
        (obj, ra, rho, rd, rn)
    };

    for _pass in 0..5 {
        for bk in 0..15 {
            let mut best_val = wr[bk];
            let mut best_obj = eval_obj(&wr, &wl).0;
            for &cand in &[0.0, 0.10, 0.20, 0.30, 0.40, 0.55, 0.70, 0.85, 0.95] {
                wr[bk] = cand;
                let o = eval_obj(&wr, &wl).0;
                if o > best_obj + 1e-6 {
                    best_obj = o;
                    best_val = cand;
                }
            }
            wr[bk] = best_val;
        }
        for bk in 0..15 {
            let mut best_val = wl[bk];
            let mut best_obj = eval_obj(&wr, &wl).0;
            for &cand in &[0.15, 0.35, 0.65, 1.0, 1.5, 2.5, 3.5] {
                wl[bk] = cand;
                let o = eval_obj(&wr, &wl).0;
                if o > best_obj + 1e-6 {
                    best_obj = o;
                    best_val = cand;
                }
            }
            wl[bk] = best_val;
        }
    }
    let wl_mean = mean(&wl[..15]);
    for v in &mut wl {
        *v /= wl_mean.max(1e-9);
    }
    let (_, ra1, rho1, rd1, rn1) = eval_obj(&wr, &wl);
    println!("\n=== Optimized Smoothed 16-Bucket Loss/Blend vs Elo Correlation ===");
    println!("  Before: all r={r_all0:+.3} rho={rho_all0:+.3} | direct6 r={r_dir0:+.3} | nnue7 r={r_nnue0:+.3}");
    println!("  After : all r={ra1:+.3} rho={rho1:+.3} | direct6 r={rd1:+.3} | nnue7 r={rn1:+.3}");
    println!(
        "  opt w_res [bk 0..15]: {}",
        wr.iter().map(|v| format!("{v:.2}")).collect::<Vec<_>>().join(",")
    );
    println!(
        "  opt w_loss[bk 0..15]: {}",
        wl.iter().map(|v| format!("{v:.2}")).collect::<Vec<_>>().join(",")
    );
    ExitCode::SUCCESS
}

fn rw_base_init() -> [f64; NB] {
    let mut a = [0.25; NB];
    for i in 0..4 {
        a[i] = 0.55;
    }
    a
}

#[derive(Clone)]
struct TrainSample {
    bk: u8,
    psqt_half: f32,
    h2: [f32; 32],
    sig_cp: f32,
    sig_cp5k: f32,
    td_search: f32,
    wdl_pool: f32,
    pool_n: f32,
    proven: i8,
    proof_v: f32,
    proof_dist: u16,
}

#[derive(Clone)]
struct Theta {
    w_res: [f64; NB],
    gamma_proof: f64,
    pool_trust: f64,
    proof_direct: f64,
    td_alpha: f64,
    proof_blend: f64,
}

impl Theta {
    fn default_init() -> Self {
        Self {
            w_res: rw_base_init(),
            gamma_proof: 0.90,
            pool_trust: 0.0,
            proof_direct: 0.0,
            td_alpha: 0.0,
            proof_blend: 0.50,
        }
    }
    fn spsa_start() -> Self {
        Self {
            w_res: [
                0.50, 0.55, 0.60, 0.55, 0.35, 0.35, 0.35, 0.35,
                0.35, 0.30, 0.25, 0.20, 0.20, 0.20, 0.20, 0.25,
            ],
            gamma_proof: 0.75,
            pool_trust: 0.50,
            proof_direct: 0.35,
            td_alpha: 0.25,
            proof_blend: 0.60,
        }
    }
    fn to_vec(&self) -> [f64; 21] {
        let mut v = [0.0; 21];
        v[..NB].copy_from_slice(&self.w_res);
        v[16] = self.gamma_proof;
        v[17] = self.pool_trust;
        v[18] = self.proof_direct;
        v[19] = self.td_alpha;
        v[20] = self.proof_blend;
        v
    }
    fn from_vec(v: &[f64; 21]) -> Self {
        let mut w_res = [0.0; NB];
        for i in 0..NB {
            w_res[i] = v[i].clamp(0.0, 0.95);
        }
        Self {
            w_res,
            gamma_proof: v[16].clamp(0.0, 0.98),
            pool_trust: v[17].clamp(0.0, 2.0),
            proof_direct: v[18].clamp(0.0, 1.0),
            td_alpha: v[19].clamp(0.0, 0.85),
            proof_blend: v[20].clamp(0.20, 1.0),
        }
    }
}

#[inline(always)]
fn compute_target(s: &TrainSample, th: &Theta) -> f64 {
    if s.proven >= 0 {
        return (th.proof_blend * s.proven as f64 + (1.0 - th.proof_blend) * s.sig_cp5k as f64).clamp(0.0, 1.0);
    }
    let bk = s.bk as usize;
    let v_search = (1.0 - th.td_alpha) * s.sig_cp as f64 + th.td_alpha * s.td_search as f64;
    let pw = if s.proof_dist > 0 {
        th.gamma_proof.powi(s.proof_dist as i32) / (1.0 + th.pool_trust * (s.pool_n as f64 - 1.0).max(0.0))
    } else {
        0.0
    };
    let pw_dir = th.proof_direct * pw;
    let pw_wdl = (1.0 - th.proof_direct) * pw;
    let v_wdl_m = pw_wdl * s.proof_v as f64 + (1.0 - pw_wdl) * s.wdl_pool as f64;
    let v_nonproof = (1.0 - th.w_res[bk]) * v_search + th.w_res[bk] * v_wdl_m;
    (pw_dir * s.proof_v as f64 + (1.0 - pw_dir) * v_nonproof).clamp(0.0, 1.0)
}

fn fit_head(raw_orig: &RawTnn, samples: &[TrainSample], bk_idx: &[Vec<usize>; NB], th: &Theta) -> RawTnn {
    let mut raw = raw_orig.clone();
    let l2 = raw.l2;
    let pdim = l2 + 1;
    let mut new_out_w = raw.out_w.clone();
    let mut new_out_b = raw.out_b.clone();

    let mut per_bk: Vec<(Vec<f64>, f64)> = vec![(Vec::new(), 0.0); NB];
    std::thread::scope(|s| {
        for (bk, slot) in per_bk.iter_mut().enumerate() {
            let idxs = &bk_idx[bk];
            let orig_w = &raw_orig.out_w[bk * l2..(bk + 1) * l2];
            let orig_b = raw_orig.out_b[bk];
            s.spawn(move || {
                let mut w: Vec<f64> = orig_w.iter().map(|&v| v as f64).collect();
                w.push(orig_b as f64);
                if idxs.len() < 40 {
                    *slot = (w[..l2].to_vec(), w[l2]);
                    return;
                }
                let anchor = w.clone();
                let reg = 25.0;
                for _iter in 0..5 {
                    let mut a = vec![vec![0.0f64; pdim]; pdim];
                    let mut rhs = vec![0.0f64; pdim];
                    for p in 0..pdim {
                        a[p][p] = reg;
                        rhs[p] = reg * (anchor[p] - w[p]);
                    }
                    for &si in idxs {
                        let sm = &samples[si];
                        let t = compute_target(sm, th);
                        let mut y = sm.psqt_half as f64 + w[l2];
                        for k in 0..l2 {
                            y += w[k] * sm.h2[k] as f64;
                        }
                        let p = sig(y);
                        let g = p * (1.0 - p);
                        let e = t - p;
                        // IRLS weights for 0.5 * |p - t|^2.5 minimization
                        let wt = e.abs().max(1e-3).powf(0.5) * 1.25;
                        let wg2 = wt * g * g;
                        let wge = wt * g * e;
                        for r in 0..l2 {
                            let xr = sm.h2[r] as f64;
                            if xr == 0.0 {
                                continue;
                            }
                            rhs[r] += wge * xr;
                            for c in 0..l2 {
                                a[r][c] += wg2 * xr * sm.h2[c] as f64;
                            }
                            a[r][l2] += wg2 * xr;
                            a[l2][r] += wg2 * xr;
                        }
                        rhs[l2] += wge;
                        a[l2][l2] += wg2;
                    }
                    let dw = solve_linear(&mut a, &mut rhs);
                    for p in 0..pdim {
                        w[p] = (w[p] + dw[p]).clamp(-1.95, 1.95);
                    }
                }
                *slot = (w[..l2].to_vec(), w[l2]);
            });
        }
    });

    for bk in 0..NB {
        if per_bk[bk].0.is_empty() {
            continue;
        }
        for k in 0..l2 {
            new_out_w[bk * l2 + k] = per_bk[bk].0[k] as f32;
        }
        new_out_b[bk] = per_bk[bk].1 as f32;
    }
    raw.out_w = new_out_w;
    raw.out_b = new_out_b;
    raw
}

/// D1271 Stage 2: Jointly SPSA-tune 16-bucket blend weights + proof decay + TD(lambda) + pool trust.
pub fn run_spsa(args: &[String]) -> ExitCode {
    let t0 = Instant::now();
    let net_path = arg(args, "--net").unwrap_or_else(|| "C:/gitProjects/titanium-ataxx/runs/nn_v14muon/nn_v14muon/best.tnn".into());
    let train_raw = arg(args, "--train-raw").unwrap_or_else(|| "C:/gitProjects/titanium-ataxx/data/nnue/packed/train_v14h_a.tdat".into());
    let chains_path = arg(args, "--chains").unwrap_or_else(|| "C:/gitProjects/titanium-ataxx/data/nnue/packed/train_v14.chains".into());
    let book_path = arg(args, "--book").unwrap_or_else(|| "C:/gitProjects/titanium-ataxx/data/nnue/book50k_std5k.txt".into());
    let n_train: usize = arg(args, "--n-train").map_or(400_000, |v| v.parse().unwrap());
    let iters: usize = arg(args, "--iters").map_or(30, |v| v.parse().unwrap());
    let pairs: usize = arg(args, "--pairs").map_or(160, |v| v.parse().unwrap());
    let nodes: u64 = arg(args, "--nodes").map_or(2000, |v| v.parse().unwrap());
    let val_pairs: usize = arg(args, "--val-pairs").map_or(400, |v| v.parse().unwrap());
    let val_nodes: u64 = arg(args, "--val-nodes").map_or(5000, |v| v.parse().unwrap());
    let threads: usize = arg(args, "--threads").map_or(7, |v| v.parse().unwrap());
    let seed: u64 = arg(args, "--seed").map_or(42, |v| v.parse().unwrap());
    let out_json = arg(args, "--out-json").unwrap_or_else(|| "C:/gitProjects/titanium-ataxx/logs/arch/d1272_spsa_blend_best.json".into());
    let out_net = arg(args, "--out-net").unwrap_or_else(|| "C:/gitProjects/titanium-ataxx/nets/ablate/v14muon_spsa_head.tnn".into());

    let raw_orig = RawTnn::load(&net_path).expect("load base raw net");
    assert!(&raw_orig.magic != b"TNN2" && raw_orig.l2 == 32 && raw_orig.buckets == NB, "spsa-blend expects TNN1/TNN3 with l2=32, buckets=16");

    let chains = read_chains(&chains_path).expect("read chains");
    let raw_len = fs::metadata(&train_raw).expect("stat train_raw").len() as usize;
    let head_rows = raw_len / REC;

    // Select evenly strided whole chains whose start < head_rows
    let valid_chains: Vec<(usize, usize)> = chains
        .into_iter()
        .filter_map(|(st, ln)| {
            let s = st as usize;
            if s >= head_rows {
                return None;
            }
            let e = (s + ln as usize).min(head_rows);
            if e > s { Some((s, e)) } else { None }
        })
        .collect();
    let est_chains = ((valid_chains.len() as f64) * (n_train as f64) / (head_rows as f64)).ceil() as usize;
    let c_stride = (valid_chains.len() / est_chains.max(1)).max(1);

    let mut f = BufReader::with_capacity(1 << 20, File::open(&train_raw).expect("open train_raw"));
    let d4 = d4_table();
    struct PendingRow {
        occ0: u64,
        occ1: u64,
        stm: u8,
        sig_cp: f32,
        sig_cp5k: f32,
        td_search: f32,
        own_wdl: f32,
        proven: i8,
        proof_v: f32,
        proof_dist: u16,
        key: Option<(u64, u64)>,
    }
    let mut pending = Vec::with_capacity(n_train + 5000);
    let mut key_accum: HashMap<(u64, u64), (f64, f64)> = HashMap::new();
    let mut buf = vec![0u8; 4096 * REC];

    for ci in (0..valid_chains.len()).step_by(c_stride) {
        if pending.len() >= n_train {
            break;
        }
        let (cs, ce) = valid_chains[ci];
        let clen = ce - cs;
        if clen > 4096 {
            continue;
        }
        f.seek(SeekFrom::Start((cs * REC) as u64)).expect("seek chain");
        f.read_exact(&mut buf[..clen * REC]).expect("read chain");
        let grows: Vec<FullRec> = buf[..clen * REC].chunks_exact(REC).map(FullRec::from_bytes).collect();

        // Backward nearest proof per chain
        let mut p_dist = vec![0u16; clen];
        let mut p_val = vec![0.5f32; clen];
        let mut last: Option<(usize, f64)> = None;
        for t in (0..clen).rev() {
            if grows[t].occ0 & grows[t].occ1 != 0 {
                continue;
            }
            if let Some(pv) = proof_value(&grows[t]) {
                last = Some((t, pv));
                continue;
            }
            if let Some((tp, pv)) = last {
                let d = tp - t;
                let v = if (pv - 0.5).abs() < 1e-3 {
                    0.5
                } else if d % 2 == 1 {
                    1.0 - pv
                } else {
                    pv
                };
                p_dist[t] = d.min(u16::MAX as usize) as u16;
                p_val[t] = v as f32;
            }
        }

        // Forward TD(lambda=0.7, H=32) search return from later plies (future_mix, gamedata.rs:281)
        let lambda = 0.7f64;
        for t in 0..clen {
            let r = &grows[t];
            if r.occ0 & r.occ1 != 0 {
                continue;
            }
            let pv_opt = proof_value(r);
            let sc = sig_cp(r.cp) as f32;
            let sc5k = sig_cp(r.cp5k) as f32;
            let mut sum = 0.0f64;
            let mut coef = 1.0f64;
            for n in 1..=32.min(clen - 1 - t) {
                let rn = &grows[t + n];
                if rn.occ0 & rn.occ1 != 0 {
                    break;
                }
                let (vm, is_p) = match proof_value(rn) {
                    Some(v) => (v, true),
                    None => (sig_cp(rn.cp), false),
                };
                let v = if n % 2 == 1 { 1.0 - vm } else { vm };
                if is_p {
                    sum += coef * v;
                    coef = 0.0;
                    break;
                }
                sum += coef * (1.0 - lambda) * v;
                coef *= lambda;
            }
            let td_s = if coef < 0.9999 { (sum / (1.0 - coef)) as f32 } else { sc };
            let proven = match pv_opt {
                Some(v) if v >= 0.99 => 1,
                Some(v) if v <= 0.01 => 0,
                _ => -1,
            };
            let key = if proven < 0 {
                let k = pool_key(r, &d4);
                key_accum.entry(k).or_insert((0.0, 0.0));
                Some(k)
            } else {
                None
            };
            pending.push(PendingRow {
                occ0: r.occ0,
                occ1: r.occ1,
                stm: r.stm,
                sig_cp: sc,
                sig_cp5k: sc5k,
                td_search: td_s,
                own_wdl: if r.outcome.is_finite() { r.outcome.clamp(0.0, 1.0) } else { 0.5 },
                proven,
                proof_v: p_val[t],
                proof_dist: p_dist[t],
                key,
            });
        }
    }
    println!(
        "spsa-blend: sampled {} rows ({} unique non-proven keys); streaming {} to compute exact global pooled WDL...",
        pending.len(),
        key_accum.len(),
        train_raw
    );

    // Stream train_raw once to populate exact pooled multi-game WDL on sampled keys
    let mut f_all = BufReader::with_capacity(4 << 20, File::open(&train_raw).expect("open stream"));
    let mut sbuf = vec![0u8; 65536 * REC];
    let mut read_total = 0usize;
    while read_total < head_rows {
        let want = (head_rows - read_total).min(65536);
        let bytes = want * REC;
        f_all.read_exact(&mut sbuf[..bytes]).expect("stream read");
        for c in sbuf[..bytes].chunks_exact(REC) {
            let r = FullRec::from_bytes(c);
            if r.occ0 & r.occ1 != 0 || proof_value(&r).is_some() {
                continue;
            }
            let k = pool_key(&r, &d4);
            if let Some(entry) = key_accum.get_mut(&k) {
                let ne = if r.wsum > 1.001 && r.wsum.is_finite() { (r.wsum * r.wsum) as f64 } else { 1.0 };
                let oc = if r.outcome.is_finite() { r.outcome.clamp(0.0, 1.0) as f64 } else { 0.5 };
                entry.0 += oc * ne;
                entry.1 += ne;
            }
        }
        read_total += want;
    }

    // Parallel frozen-trunk feature extraction (`h2`, `psqt_half`) + parity guard
    let np = pending.len();
    let mut samples: Vec<Option<TrainSample>> = vec![None; np];
    let chunk = (np + threads - 1) / threads;
    let raw_ref = &raw_orig;
    let keys_ref = &key_accum;
    let next_chk = AtomicUsize::new(0);
    let net_check = Nnue::load(&net_path).expect("load Nnue parity ref");
    let parity_ok = std::sync::Mutex::new(0.0f32);
    std::thread::scope(|s| {
        let out_chunks: Vec<&mut [Option<TrainSample>]> = samples.chunks_mut(chunk.max(1)).collect();
        for (ci, slot) in out_chunks.into_iter().enumerate() {
            let start = ci * chunk;
            let end = (start + slot.len()).min(np);
            let sub = &pending[start..end];
            let next_ref = &next_chk;
            let nc_ref = &net_check;
            let po_ref = &parity_ok;
            s.spawn(move || {
                let w = raw_ref.width;
                let din = if raw_ref.pairwise { w } else { 2 * w };
                let mut acc = vec![0f32; 2 * w];
                let mut x = vec![0f32; din];
                let mut local_max = 0.0f32;
                for (j, pr) in sub.iter().enumerate() {
                    let b = board_of(pr.occ0, pr.occ1, pr.stm);
                    let (bk, psqt_half, h2, y_f32) = raw_ref.forward_h2(&b, &mut acc, &mut x);
                    if start + j < 200 {
                        let y_ref = nc_ref.forward_f32(&b, 0.0);
                        local_max = local_max.max((y_ref - y_f32).abs());
                    }
                    let (wdl_pool, pool_n) = match pr.key.and_then(|k| keys_ref.get(&k)) {
                        Some(&(sw, sn)) if sn > 0.0 => ((sw / sn) as f32, sn as f32),
                        _ => (pr.own_wdl, 1.0),
                    };
                    slot[j] = Some(TrainSample {
                        bk: bk as u8,
                        psqt_half,
                        h2,
                        sig_cp: pr.sig_cp,
                        sig_cp5k: pr.sig_cp5k,
                        td_search: pr.td_search,
                        wdl_pool,
                        pool_n,
                        proven: pr.proven,
                        proof_v: pr.proof_v,
                        proof_dist: pr.proof_dist,
                    });
                }
                let mut g = po_ref.lock().unwrap();
                *g = g.max(local_max);
                next_ref.fetch_add(1, Ordering::Relaxed);
            });
        }
    });
    let samples: Vec<TrainSample> = samples.into_iter().map(|o| o.unwrap()).collect();
    let max_par = *parity_ok.lock().unwrap();
    println!(
        "spsa-blend: precomputed {} samples in {:.1}s (frozen trunk parity max |diff| = {:.2e}).",
        samples.len(),
        t0.elapsed().as_secs_f64(),
        max_par
    );
    if max_par > 1e-2 {
        eprintln!("spsa-blend: parity mismatch ({max_par}) between RawTnn and Nnue::forward_f32");
        return ExitCode::FAILURE;
    }

    let mut bk_idx: [Vec<usize>; NB] = std::array::from_fn(|_| Vec::new());
    for (i, s) in samples.iter().enumerate() {
        bk_idx[s.bk as usize].push(i);
    }

    let book_text = fs::read_to_string(&book_path).expect("read book");
    let all_books: Vec<Board> = book_text.lines().filter_map(|l| Board::from_str(l.trim())).collect();
    if all_books.len() < 10_000 + val_pairs {
        eprintln!("spsa-blend: book too short ({})", all_books.len());
        return ExitCode::FAILURE;
    }

    let tmp_dir = std::env::temp_dir().join(format!("ti_spsa_{}", std::process::id()));
    fs::create_dir_all(&tmp_dir).expect("mkdir tmp");
    let p_plus = tmp_dir.join("plus.tnn").to_string_lossy().into_owned();
    let p_minus = tmp_dir.join("minus.tnn").to_string_lossy().into_owned();
    let p_base = tmp_dir.join("base.tnn").to_string_lossy().into_owned();
    let p_chk = tmp_dir.join("chk.tnn").to_string_lossy().into_owned();

    // D1275 `--match-a W0..W15,gamma,trust,dir,td,pbl [--match-b ...] [--book0 N]`: no SPSA, one A-vs-B match of two refit heads
    if let Some(sa) = arg(args, "--match-a") {
        let parse = |s: &str| -> Theta {
            let v: Vec<f64> = s.split(',').map(|x| x.trim().parse::<f64>().expect("match theta number")).collect();
            let a: [f64; 21] = v.as_slice().try_into().expect("--match-a/--match-b need 21 comma-separated numbers");
            Theta::from_vec(&a)
        };
        let th_a = parse(&sa);
        let th_b = arg(args, "--match-b").map_or_else(Theta::default_init, |s| parse(&s));
        let b0: usize = arg(args, "--book0").map_or(10_000, |v| v.parse().unwrap());
        assert!(b0 + val_pairs <= all_books.len(), "book too short for --book0 {b0} + {val_pairs}");
        let (ra, rb) = (fit_head(&raw_orig, &samples, &bk_idx, &th_a), fit_head(&raw_orig, &samples, &bk_idx, &th_b));
        ra.save(&p_plus).expect("save A");
        rb.save(&p_minus).expect("save B");
        let (ea, eb) = (Ev::Net(Nnue::load(&p_plus).expect("load A")), Ev::Net(Nnue::load(&p_minus).expect("load B")));
        let (sc, pn) = play_match(&ea, &eb, &all_books[b0..b0 + val_pairs], 0, val_nodes, threads);
        let elo = -400.0 * (1.0 / sc.clamp(1e-6, 1.0 - 1e-6) - 1.0).log10();
        println!("[match] A={sa}\n[match] B={}", arg(args, "--match-b").unwrap_or_else(|| "BASE_HEAD".into()));
        println!("[match] A vs B @book0={b0} {val_pairs} pairs {val_nodes} nodes: {:.2}% | Elo {:+.1} | penta {:?}", 100.0 * sc, elo, pn);
        if args.iter().any(|a| a == "--vs-orig") {
            let eo = Ev::Net(Nnue::load(&net_path).expect("load orig"));
            let (sc2, pn2) = play_match(&ea, &eo, &all_books[b0..b0 + val_pairs], 0, val_nodes, threads);
            let elo2 = -400.0 * (1.0 / sc2.clamp(1e-6, 1.0 - 1e-6) - 1.0).log10();
            println!("[match] A vs ORIG_NET: {:.2}% | Elo {:+.1} | penta {:?}", 100.0 * sc2, elo2, pn2);
        }
        let _ = fs::remove_dir_all(&tmp_dir);
        return ExitCode::SUCCESS;
    }
    let raw_base = fit_head(&raw_orig, &samples, &bk_idx, &Theta::default_init());
    raw_base.save(&p_base).expect("save base head");
    let ev_base = Ev::Net(Nnue::load(&p_base).expect("load base head"));
    let to_elo = |s: f64| -400.0 * (1.0 / s.clamp(1e-6, 1.0 - 1e-6) - 1.0).log10();

    let mut th = Theta::spsa_start();
    let mut th_vec = th.to_vec();
    let mut sum_vec = [0.0f64; 21];
    let mut sum_n = 0usize;
    let avg_start = iters / 3;
    let mut scale = [0.15f64; 21];
    scale[16] = 0.12; // gamma_proof
    scale[17] = 0.30; // pool_trust
    scale[18] = 0.20; // proof_direct
    scale[19] = 0.15; // td_alpha
    scale[20] = 0.15; // proof_blend

    let mut rng = seed ^ 0x9E37_79B9_7F4A_7C15;
    fn next_bit(rng: &mut u64) -> f64 {
        *rng ^= *rng << 13;
        *rng ^= *rng >> 7;
        *rng ^= *rng << 17;
        if (*rng & 1) == 0 { 1.0 } else { -1.0 }
    }

    let mut history_lines: Vec<String> = Vec::new();
    // LAWS A14 (user D1307): the full SPSA state goes to `{out_json}.ckpt` after every iteration; STOP in the out folder
    // or C:/gitProjects/STOP_ALL is checked before every iteration and inside every match (a stopped iteration is
    // discarded, nothing of it is applied); the same command + --resume continues at that iteration.
    let ck_path = format!("{out_json}.ckpt");
    let run_dir = std::path::Path::new(&out_json).parent().map(|p| p.to_path_buf()).unwrap_or_else(|| ".".into());
    let _ = crate::stopctl::RUN_DIR.set(run_dir.clone());
    let mut k0 = 0usize;
    if args.iter().any(|a| a == "--resume") {
        let dd = match fs::read(&ck_path) {
            Ok(x) => x,
            Err(e) => {
                eprintln!("spsa-blend --resume: {ck_path}: {e}");
                return ExitCode::from(2);
            }
        };
        let mut r = crate::stopctl::Rd::new(&dd);
        let ok = (|| -> Result<(), String> {
            if r.str()? != "SPCK1" || r.u64()? != iters as u64 || r.u64()? != seed {
                return Err("not a spsa-blend checkpoint of this setup (--iters / --seed)".into());
            }
            k0 = r.u64()? as usize;
            rng = r.u64()?;
            sum_n = r.u64()? as usize;
            for v in th_vec.iter_mut() {
                *v = r.f64()?;
            }
            for v in sum_vec.iter_mut() {
                *v = r.f64()?;
            }
            let nh = r.u64()? as usize;
            for _ in 0..nh {
                history_lines.push(r.str()?);
            }
            Ok(())
        })();
        if let Err(e) = ok {
            eprintln!("spsa-blend --resume: {ck_path}: {e}");
            return ExitCode::from(2);
        }
        th = Theta::from_vec(&th_vec);
        th_vec = th.to_vec();
        println!("spsa-blend: RESUMED from {ck_path} at iteration {} of {iters}", k0 + 1);
    } else if std::path::Path::new(&ck_path).exists() {
        eprintln!("spsa-blend: {ck_path} exists: continue it with --resume, or delete it to start over");
        return ExitCode::from(2);
    }
    let save_ck = |k: usize, rng: u64, th_vec: &[f64; 21], sum_vec: &[f64; 21], sum_n: usize, hist: &[String]| -> Result<(), String> {
        let mut x = crate::stopctl::Wr::default();
        x.str("SPCK1").u64(iters as u64).u64(seed).u64(k as u64).u64(rng).u64(sum_n as u64);
        for v in th_vec.iter() {
            x.f64(*v);
        }
        for v in sum_vec.iter() {
            x.f64(*v);
        }
        x.u64(hist.len() as u64);
        for h in hist {
            x.str(h);
        }
        crate::stopctl::write_atomic(&ck_path, &x.0)
    };
    for k in k0..iters {
        if let Some(f) = crate::stopctl::requested(Some(&run_dir)) {
            if let Err(e) = save_ck(k, rng, &th_vec, &sum_vec, sum_n, &history_lines) {
                eprintln!("spsa-blend: checkpoint: {e}");
                return ExitCode::from(1);
            }
            println!("spsa-blend: STOPPED by {f} before iteration {}: state {ck_path}; the same command + --resume continues", k + 1);
            return ExitCode::SUCCESS;
        }
        let rng_before = rng;
        let it_t0 = Instant::now();
        let ck = 1.0 / ((k + 1) as f64).powf(0.101);
        let ak = 8.0 / ((k + 5) as f64).powf(0.602);
        let mut delta = [0.0f64; 21];
        let mut v_p = th_vec;
        let mut v_m = th_vec;
        for i in 0..21 {
            delta[i] = next_bit(&mut rng);
            v_p[i] += ck * scale[i] * delta[i];
            v_m[i] -= ck * scale[i] * delta[i];
        }
        let th_p = Theta::from_vec(&v_p);
        let th_m = Theta::from_vec(&v_m);
        let raw_p = fit_head(&raw_orig, &samples, &bk_idx, &th_p);
        let raw_m = fit_head(&raw_orig, &samples, &bk_idx, &th_m);
        raw_p.save(&p_plus).expect("save plus");
        raw_m.save(&p_minus).expect("save minus");

        let ev_p = Ev::Net(Nnue::load(&p_plus).expect("load plus"));
        let ev_m = Ev::Net(Nnue::load(&p_minus).expect("load minus"));
        let b0 = 1200 + (k * pairs) % 8000;
        let (score_p, penta) = play_match(&ev_p, &ev_m, &all_books[b0..b0 + pairs], 0, nodes, threads);
        if let Some(f) = crate::stopctl::requested(Some(&run_dir)) {
            // the match was cut short: discard this iteration (nothing applied), resume replays it from the same rng
            if let Err(e) = save_ck(k, rng_before, &th_vec, &sum_vec, sum_n, &history_lines) {
                eprintln!("spsa-blend: checkpoint: {e}");
                return ExitCode::from(1);
            }
            println!("spsa-blend: STOPPED by {f} during iteration {} (discarded): state {ck_path}; the same command + --resume replays it", k + 1);
            return ExitCode::SUCCESS;
        }
        let g_sig = ((score_p - 0.5) * 2.0).clamp(-0.20, 0.20);

        // Compute raw gradient update, then smooth only the w_res[0..16] update (never diffuse theta directly)
        let mut upd = [0.0f64; 21];
        for i in 0..21 {
            upd[i] = ak * scale[i] * g_sig / (ck * delta[i]);
        }
        let mut upd_sm = upd;
        for b in 0..NB {
            let left = if b > 0 { upd[b - 1] } else { upd[b] };
            let right = if b + 1 < NB { upd[b + 1] } else { upd[b] };
            upd_sm[b] = 0.20 * left + 0.60 * upd[b] + 0.20 * right;
        }
        for i in 0..21 {
            th_vec[i] += upd_sm[i];
        }
        th = Theta::from_vec(&th_vec);
        th_vec = th.to_vec();
        if k >= avg_start {
            for i in 0..21 {
                sum_vec[i] += th_vec[i];
            }
            sum_n += 1;
        }
        let msg = format!(
            "[spsa {:>2}/{iters}] {:.1}% penta {:?} ({:.1}s) | gamma={:.3} trust={:.3} dir={:.3} td={:.3} pbl={:.3} | w_res=[{}]",
            k + 1,
            100.0 * score_p,
            penta,
            it_t0.elapsed().as_secs_f64(),
            th.gamma_proof,
            th.pool_trust,
            th.proof_direct,
            th.td_alpha,
            th.proof_blend,
            th.w_res.iter().map(|v| format!("{v:.2}")).collect::<Vec<_>>().join(",")
        );
        println!("{msg}");
        history_lines.push(msg);
        if let Err(e) = save_ck(k + 1, rng, &th_vec, &sum_vec, sum_n, &history_lines) {
            eprintln!("spsa-blend: checkpoint: {e}");
            return ExitCode::from(1);
        }

        if (k + 1) % 10 == 0 {
            let chk_vec: [f64; 21] = if sum_n > 0 {
                let mut avg = [0.0; 21];
                for i in 0..21 {
                    avg[i] = sum_vec[i] / sum_n as f64;
                }
                avg
            } else {
                th_vec
            };
            let th_chk = Theta::from_vec(&chk_vec);
            let raw_chk = fit_head(&raw_orig, &samples, &bk_idx, &th_chk);
            raw_chk.save(&p_chk).expect("save chk head");
            let ev_chk = Ev::Net(Nnue::load(&p_chk).expect("load chk head"));
            let chk_n = pairs.min(800);
            let (sc_chk, pn_chk) = play_match(&ev_chk, &ev_base, &all_books[9200..9200 + chk_n], 0, nodes, threads);
            let chk_msg = format!(
                "[chk iter {}] CURRENT vs BASE_HEAD: {:.1}% | Elo {:+.1} | penta {:?}",
                k + 1,
                100.0 * sc_chk,
                to_elo(sc_chk),
                pn_chk
            );
            println!("{chk_msg}");
            history_lines.push(chk_msg);
            if let Some(f) = crate::stopctl::requested(Some(&run_dir)) {
                println!("spsa-blend: STOPPED by {f} after iteration {} (its check match may be partial): state {ck_path}; the same command + --resume continues", k + 1);
                return ExitCode::SUCCESS;
            }
        }
    }

    let final_vec: [f64; 21] = if sum_n > 0 {
        let mut avg = [0.0; 21];
        for i in 0..21 {
            avg[i] = sum_vec[i] / sum_n as f64;
        }
        avg
    } else {
        th_vec
    };
    let th_best = Theta::from_vec(&final_vec);
    let raw_best = fit_head(&raw_orig, &samples, &bk_idx, &th_best);
    raw_best.save(&out_net).expect("save best head");

    println!("\n=== Final Validation ({val_pairs} unseen pairs @book0=10000, {val_nodes} nodes/move) ===");
    println!(
        "  Polyak best Theta: gamma={:.4} pool_trust={:.4} proof_direct={:.4} td_alpha={:.4} proof_blend={:.4}",
        th_best.gamma_proof, th_best.pool_trust, th_best.proof_direct, th_best.td_alpha, th_best.proof_blend
    );
    println!(
        "  w_res[0..15]: {}",
        th_best.w_res.iter().map(|v| format!("{v:.4}")).collect::<Vec<_>>().join(",")
    );

    let ev_best = Ev::Net(Nnue::load(&out_net).expect("load best head"));
    let ev_orig = Ev::Net(Nnue::load(&net_path).expect("load orig net"));
    let val_slice = &all_books[10_000..10_000 + val_pairs];
    let (sc_b, pn_b) = play_match(&ev_best, &ev_base, val_slice, 0, val_nodes, threads);
    let (sc_o, pn_o) = play_match(&ev_best, &ev_orig, val_slice, 0, val_nodes, threads);
    let res_line1 = format!("  SPSA_HEAD vs BASE_HEAD: {:.2}% | Elo {:+.1} | penta {:?}", 100.0 * sc_b, to_elo(sc_b), pn_b);
    let res_line2 = format!("  SPSA_HEAD vs ORIG_NET : {:.2}% | Elo {:+.1} | penta {:?}", 100.0 * sc_o, to_elo(sc_o), pn_o);
    println!("{res_line1}\n{res_line2}");

    let json = format!(
        "{{\n  \"w_res\": [{}],\n  \"gamma_proof\": {:.6},\n  \"pool_trust\": {:.6},\n  \"proof_direct\": {:.6},\n  \"td_alpha\": {:.6},\n  \"proof_blend\": {:.6},\n  \"val_vs_base_score\": {:.6},\n  \"val_vs_base_elo\": {:.2},\n  \"val_vs_orig_score\": {:.6},\n  \"val_vs_orig_elo\": {:.2},\n  \"out_net\": \"{}\"\n}}\n",
        th_best.w_res.iter().map(|v| format!("{v:.6}")).collect::<Vec<_>>().join(", "),
        th_best.gamma_proof,
        th_best.pool_trust,
        th_best.proof_direct,
        th_best.td_alpha,
        th_best.proof_blend,
        sc_b,
        to_elo(sc_b),
        sc_o,
        to_elo(sc_o),
        out_net.replace('\\', "/"),
    );
    if let Some(par) = Path::new(&out_json).parent() {
        let _ = fs::create_dir_all(par);
    }
    fs::write(&out_json, json).expect("write out_json");
    let _ = fs::write(format!("{out_json}.log"), history_lines.join("\n") + "\n");
    let _ = fs::remove_dir_all(&tmp_dir);
    ExitCode::SUCCESS
}

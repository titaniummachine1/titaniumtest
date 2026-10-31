//! `ti train shrink --net in.tnn --width W2 --data <stem> --out out.tnn [--rows 20000]`: general Net2Net-style downscale.
//! Keeps the W2 feature-transformer channels that matter most and drops the rest (never trains from scratch, D130).
//! Importance of channel c = E[a_c^2] over calibration rows (activation after the clamp, both perspectives) times the squared
//! L1 weight leaving it (both perspective halves, all buckets): roughly the variance that channel feeds into the next layer.
//! The same channel set is kept in both perspective halves, so the shrunk net stays a valid TNN of width W2.
//! The pruned net is NOT as good as the original: it is the START of a distillation run. The val loss before and after is
//! printed so the cost of the cut itself is visible. The disagreement head sits on h1||h2 (unchanged) but must be refitted
//! (`ti train export`) after distillation.
use std::process::ExitCode;

use crate::mapper;
use crate::train::TrainableNet;
use crate::trainer::{evaluate, load_val, Tdat};

pub fn shrink(net: &TrainableNet, w2: usize, cal: &[crate::data::Rec]) -> Result<(TrainableNet, Vec<usize>), String> {
    let w = net.width;
    if net.pairwise {
        return Err("shrink: pairwise nets are not supported".into());
    }
    if w2 == 0 || w2 >= w {
        return Err(format!("shrink: --width must be in 1..{w}"));
    }
    // E[a^2] per channel over calibration rows, both perspectives pooled
    let mut ea2 = vec![0f64; w];
    let mut n = 0f64;
    let mut a = vec![0f32; w];
    for r in cal {
        let s = mapper::sample(r, 0, net.buckets, net.groups >> 8);
        for idx in [&s.us, &s.them] {
            a.copy_from_slice(&net.ftb);
            for &f in idx.iter() {
                for (v, x) in a.iter_mut().zip(&net.ft[f as usize * w..(f as usize + 1) * w]) {
                    *v += *x;
                }
            }
            for (c, v) in a.iter().enumerate() {
                let t = v.clamp(0.0, 1.0) as f64;
                ea2[c] += t * t;
            }
            n += 1.0;
        }
    }
    let din = 2 * w;
    let rows = net.buckets * net.l1;
    let mut score = vec![0f64; w];
    for c in 0..w {
        let mut wsq = 0f64;
        for r in 0..rows {
            let row = &net.l1_w[r * din..(r + 1) * din];
            wsq += (row[c] as f64).powi(2) + (row[w + c] as f64).powi(2);
        }
        score[c] = ea2[c] / n.max(1.0) * wsq;
    }
    let mut order: Vec<usize> = (0..w).collect();
    order.sort_by(|&x, &y| score[y].partial_cmp(&score[x]).unwrap());
    let mut keep: Vec<usize> = order[..w2].to_vec();
    keep.sort_unstable();

    let mut out = net.clone();
    out.width = w2;
    out.ft = Vec::with_capacity(net.num_feats * w2);
    for f in 0..net.num_feats {
        for &c in &keep {
            out.ft.push(net.ft[f * w + c]);
        }
    }
    out.ftb = keep.iter().map(|&c| net.ftb[c]).collect();
    out.l1_w = Vec::with_capacity(rows * 2 * w2);
    for r in 0..rows {
        let row = &net.l1_w[r * din..(r + 1) * din];
        out.l1_w.extend(keep.iter().map(|&c| row[c]));
        out.l1_w.extend(keep.iter().map(|&c| row[w + c]));
    }
    Ok((out, keep))
}

pub fn run_shrink(args: &[String]) -> ExitCode {
    match run_inner(args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("shrink: {e}");
            ExitCode::FAILURE
        }
    }
}

fn run_inner(args: &[String]) -> Result<(), String> {
    let f = |n: &str| crate::flag_str(args, n);
    let (net_p, data, out_p) = (f("--net").ok_or("--net required")?, f("--data").ok_or("--data <stem> required")?, f("--out").ok_or("--out required")?);
    let w2 = crate::flag_u64(args, "--width").ok_or("--width required")? as usize;
    let rows = crate::flag_u64(args, "--rows").unwrap_or(20000) as usize;
    let threads = crate::flag_u64(args, "--threads").unwrap_or(6) as usize;
    let net = TrainableNet::from_tnn_bytes(&std::fs::read(&net_p).map_err(|e| format!("{net_p}: {e}"))?)?;
    let cal = load_val(&mut Tdat::open(&data)?, rows)?;
    if crate::has_flag(args, "--refit") {
        let t = std::time::Instant::now();
        let (refit, _plain, o, p, r) = shrink_refit(&net, w2, &cal, threads)?;
        println!("[shrink --refit] W {} -> {} on {} held-out rows: original {o:.6}  same channels pruned {p:.6}  pruned + L1 refit {r:.6}  ({:.0} s)", net.width, w2, cal.len() / 2, t.elapsed().as_secs_f64());
        println!("[shrink --refit] refit wins back {:.1}% of the pruning loss", 100.0 * (p - r) / (p - o).max(1e-12));
        std::fs::write(&out_p, refit.to_tnn_bytes()).map_err(|e| e.to_string())?;
        println!("[shrink --refit] wrote {out_p}");
        return Ok(());
    }
    let (small, keep) = shrink(&net, w2, &cal)?;
    let before = evaluate(&net, &cal, threads);
    let after = evaluate(&small, &cal, threads);
    println!("[shrink] W {} -> {}, kept {} channels of {}", net.width, w2, keep.len(), net.width);
    println!("[shrink] val mse on {} rows: original {:.6}  pruned {:.6}  (distillation must win back {:.6})", cal.len(), before.mse, after.mse, after.mse - before.mse);
    std::fs::write(&out_p, small.to_tnn_bytes()).map_err(|e| e.to_string())?;
    println!("[shrink] wrote {out_p}");
    Ok(())
}

// ------------------------------------------------------------------------------------------------------------------
// `ti train rank` / `ti train shrink --refit`: "image-style" downscale. Channels are chosen by pivoted Cholesky on the
// downstream-weighted activation covariance (greedy: next channel = the one whose activation is least explained by the ones
// already kept), and the L1 layer is then REFITTED by ridge least squares on real positions so the kept channels absorb what
// the dropped ones contributed (nothing is simply thrown away).
// ------------------------------------------------------------------------------------------------------------------

struct Acts {
    w: usize,
    /// per row: [us (w) | them (w)] activations after the clamp
    a: Vec<f32>,
    bucket: Vec<usize>,
}

fn collect(net: &TrainableNet, rows: &[crate::data::Rec]) -> Acts {
    let w = net.width;
    let mut a = Vec::with_capacity(rows.len() * 2 * w);
    let mut bucket = Vec::with_capacity(rows.len());
    let mut t = vec![0f32; w];
    for r in rows {
        let s = mapper::sample(r, 0, net.buckets, net.groups >> 8);
        for idx in [&s.us, &s.them] {
            t.copy_from_slice(&net.ftb);
            for &f in idx.iter() {
                for (v, x) in t.iter_mut().zip(&net.ft[f as usize * w..(f as usize + 1) * w]) {
                    *v += *x;
                }
            }
            a.extend(t.iter().map(|v| v.clamp(0.0, 1.0)));
        }
        bucket.push(s.bucket);
    }
    Acts { w, a, bucket }
}

/// downstream-scaled covariance of the channel activations (both perspectives pooled) and the per-channel scale
fn scaled_cov(net: &TrainableNet, ac: &Acts) -> Vec<Vec<f64>> {
    let w = ac.w;
    let din = 2 * w;
    let rows = net.buckets * net.l1;
    let n = ac.bucket.len() * 2;
    let mut mean = vec![0f64; w];
    for r in 0..ac.bucket.len() {
        for h in 0..2 {
            for c in 0..w {
                mean[c] += ac.a[(r * 2 + h) * w + c] as f64;
            }
        }
    }
    for m in mean.iter_mut() {
        *m /= n as f64;
    }
    let mut cov = vec![vec![0f64; w]; w];
    let mut d = vec![0f64; w];
    for r in 0..ac.bucket.len() {
        for h in 0..2 {
            let row = &ac.a[(r * 2 + h) * w..(r * 2 + h + 1) * w];
            for c in 0..w {
                d[c] = row[c] as f64 - mean[c];
            }
            for p in 0..w {
                if d[p] != 0.0 {
                    for q in p..w {
                        cov[p][q] += d[p] * d[q];
                    }
                }
            }
        }
    }
    let mut sc = vec![0f64; w];
    for c in 0..w {
        let mut wsq = 0f64;
        for r in 0..rows {
            let rw = &net.l1_w[r * din..(r + 1) * din];
            wsq += (rw[c] as f64).powi(2) + (rw[w + c] as f64).powi(2);
        }
        sc[c] = wsq.sqrt();
    }
    for p in 0..w {
        for q in p..w {
            let v = cov[p][q] / n as f64 * sc[p] * sc[q];
            cov[p][q] = v;
            cov[q][p] = v;
        }
    }
    cov
}

/// pivoted Cholesky: returns (pivot order, residual trace after each pivot) ; trace[0] = total
fn pivots(cov: &[Vec<f64>], k: usize) -> (Vec<usize>, Vec<f64>) {
    let w = cov.len();
    let mut diag: Vec<f64> = (0..w).map(|i| cov[i][i]).collect();
    let mut l: Vec<Vec<f64>> = Vec::new(); // l[j][i] = column j of the Cholesky factor
    let mut order = Vec::new();
    let mut used = vec![false; w];
    let mut trace = vec![diag.iter().sum::<f64>()];
    for _ in 0..k.min(w) {
        let (mut p, mut best) = (usize::MAX, -1.0);
        for i in 0..w {
            if !used[i] && diag[i] > best {
                best = diag[i];
                p = i;
            }
        }
        if p == usize::MAX || best <= 1e-18 {
            break;
        }
        used[p] = true;
        order.push(p);
        let mut col = vec![0f64; w];
        for i in 0..w {
            let mut s = cov[i][p];
            for lj in &l {
                s -= lj[i] * lj[p];
            }
            col[i] = s / best.sqrt();
        }
        for i in 0..w {
            diag[i] = (diag[i] - col[i] * col[i]).max(0.0);
        }
        l.push(col);
        trace.push(diag.iter().enumerate().filter(|(i, _)| !used[*i]).map(|(_, v)| *v).sum());
    }
    (order, trace)
}

pub fn run_rank(args: &[String]) -> ExitCode {
    let f = |n: &str| crate::flag_str(args, n);
    let (Some(net_p), Some(data)) = (f("--net"), f("--data")) else {
        eprintln!("usage: ti train rank --net n.tnn --data <stem> [--rows 40000]");
        return ExitCode::FAILURE;
    };
    let rows = crate::flag_u64(args, "--rows").unwrap_or(40000) as usize;
    let net = match std::fs::read(&net_p).map_err(|e| e.to_string()).and_then(|b| TrainableNet::from_tnn_bytes(&b)) {
        Ok(n) => n,
        Err(e) => {
            eprintln!("rank: {e}");
            return ExitCode::FAILURE;
        }
    };
    let cal = match Tdat::open(&data).and_then(|mut t| load_val(&mut t, rows)) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("rank: {e}");
            return ExitCode::FAILURE;
        }
    };
    let ac = collect(&net, &cal);
    let cov = scaled_cov(&net, &ac);
    let (order, trace) = pivots(&cov, net.width);
    let total = trace[0];
    println!("net {net_p}: W={}  ({} rows, {} activation vectors)", net.width, cal.len(), cal.len() * 2);
    println!("  downstream-weighted variance NOT explained by the first k channels (pivoted Cholesky):");
    let mut ks: Vec<usize> = vec![16, 32, 48, 64, 96, 128, 160, 192, 256, 320, 384, 448];
    ks.retain(|&k| k < order.len());
    ks.push(order.len());
    for k in ks {
        println!("    k={k:>4}  unexplained {:>8.4}%", 100.0 * trace[k] / total.max(1e-18));
    }
    let dead = (0..net.width).filter(|&c| cov[c][c] < 1e-6 * total / net.width as f64).count();
    println!("  nearly dead channels (variance x downstream weight < 1e-6 of the average): {dead} of {}", net.width);
    ExitCode::SUCCESS
}

/// Gaussian elimination with partial pivoting for (A + ridge) X = B, A is n x n, B is n x m.
fn solve_multi(mut a: Vec<Vec<f64>>, mut b: Vec<Vec<f64>>, ridge: f64) -> Vec<Vec<f64>> {
    let n = a.len();
    for i in 0..n {
        a[i][i] += ridge;
    }
    for c in 0..n {
        let p = (c..n).max_by(|&x, &y| a[x][c].abs().partial_cmp(&a[y][c].abs()).unwrap()).unwrap();
        a.swap(c, p);
        b.swap(c, p);
        let d = a[c][c];
        for k in c..n {
            a[c][k] /= d;
        }
        for k in 0..b[c].len() {
            b[c][k] /= d;
        }
        for r in 0..n {
            if r != c {
                let f = a[r][c];
                if f != 0.0 {
                    for k in c..n {
                        let t = a[c][k];
                        a[r][k] -= f * t;
                    }
                    for k in 0..b[c].len() {
                        let t = b[c][k];
                        b[r][k] -= f * t;
                    }
                }
            }
        }
    }
    b
}

/// `ti train shrink --refit`: pivoted-Cholesky channel choice + ridge refit of every bucket's L1 on real positions.
/// Rows are split even/odd: fit on even, report on odd (held out). Returns (refitted net, plain-pruned net with the SAME channels).
pub fn shrink_refit(net: &TrainableNet, w2: usize, rows: &[crate::data::Rec], threads: usize) -> Result<(TrainableNet, TrainableNet, f64, f64, f64), String> {
    let w = net.width;
    let fit_rows: Vec<crate::data::Rec> = rows.iter().cloned().step_by(2).collect();
    let test_rows: Vec<crate::data::Rec> = rows.iter().skip(1).cloned().step_by(2).collect();
    let ac = collect(net, &fit_rows);
    let cov = scaled_cov(net, &ac);
    let (order, _) = pivots(&cov, w2);
    if order.len() < w2 {
        return Err(format!("only {} usable channels", order.len()));
    }
    let mut keep: Vec<usize> = order[..w2].to_vec();
    keep.sort_unstable();
    // plain pruned net with the same channels (same construction as `shrink`, but with this channel set)
    let build = |l1_w: Vec<f32>, l1_b: Vec<f32>| {
        let mut o = net.clone();
        o.width = w2;
        o.ft = Vec::with_capacity(net.num_feats * w2);
        for f in 0..net.num_feats {
            for &c in &keep {
                o.ft.push(net.ft[f * w + c]);
            }
        }
        o.ftb = keep.iter().map(|&c| net.ftb[c]).collect();
        o.l1_w = l1_w;
        o.l1_b = l1_b;
        o
    };
    let din = 2 * w;
    let rows_l1 = net.buckets * net.l1;
    let mut plain_w = Vec::with_capacity(rows_l1 * 2 * w2);
    for r in 0..rows_l1 {
        let row = &net.l1_w[r * din..(r + 1) * din];
        plain_w.extend(keep.iter().map(|&c| row[c]));
        plain_w.extend(keep.iter().map(|&c| row[w + c]));
    }
    let plain = build(plain_w, net.l1_b.clone());

    // ridge refit per bucket
    let n = 2 * w2 + 1;
    let m = net.l1;
    let mut g = vec![vec![vec![0f64; n]; n]; net.buckets];
    let mut h = vec![vec![vec![0f64; m]; n]; net.buckets];
    let mut x = vec![0f64; n];
    for (ri, &b) in ac.bucket.iter().enumerate() {
        let us = &ac.a[(ri * 2) * w..(ri * 2 + 1) * w];
        let them = &ac.a[(ri * 2 + 1) * w..(ri * 2 + 2) * w];
        for (j, &c) in keep.iter().enumerate() {
            x[j] = us[c] as f64;
            x[w2 + j] = them[c] as f64;
        }
        x[2 * w2] = 1.0;
        // original pre-activations of this bucket's l1 units are the regression targets
        let mut z = vec![0f64; m];
        for (j, zj) in z.iter_mut().enumerate() {
            let r = b * net.l1 + j;
            let rw = &net.l1_w[r * din..(r + 1) * din];
            let mut s = net.l1_b[r] as f64;
            for c in 0..w {
                s += rw[c] as f64 * us[c] as f64 + rw[w + c] as f64 * them[c] as f64;
            }
            *zj = s;
        }
        for p in 0..n {
            let xp = x[p];
            if xp != 0.0 {
                for q in p..n {
                    g[b][p][q] += xp * x[q];
                }
                for j in 0..m {
                    h[b][p][j] += xp * z[j];
                }
            }
        }
    }
    let mut l1_w = vec![0f32; rows_l1 * 2 * w2];
    let mut l1_b = net.l1_b.clone();
    for b in 0..net.buckets {
        for p in 0..n {
            for q in 0..p {
                g[b][p][q] = g[b][q][p];
            }
        }
        let tr: f64 = (0..n).map(|i| g[b][i][i]).sum::<f64>() / n as f64;
        let sol = solve_multi(std::mem::take(&mut g[b]), std::mem::take(&mut h[b]), 1e-4 * tr.max(1e-9));
        for j in 0..m {
            let r = b * net.l1 + j;
            for p in 0..2 * w2 {
                l1_w[r * 2 * w2 + p] = sol[p][j] as f32;
            }
            l1_b[r] = sol[2 * w2][j] as f32;
        }
    }
    let refit = build(l1_w, l1_b);
    let o = evaluate(net, &test_rows, threads).mse;
    let p = evaluate(&plain, &test_rows, threads).mse;
    let r = evaluate(&refit, &test_rows, threads).mse;
    Ok((refit, plain, o, p, r))
}

// ------------------------------------------------------------------------------------------------------------------
// `ti train upscale`: Net2Net-wider (function preserving). New channel c' copies source channel c' % w; every copy of a source
// channel gets l1 weight / copies, so the l1 pre-activations are unchanged. Tiny noise on the copied ft columns breaks the
// symmetry (identical copies would receive identical gradients forever). Step-0 parity is measured, not assumed (D130).
// ------------------------------------------------------------------------------------------------------------------

pub fn upscale(net: &TrainableNet, w2: usize, noise: f32) -> Result<TrainableNet, String> {
    let w = net.width;
    if net.pairwise {
        return Err("upscale: pairwise nets are not supported".into());
    }
    if w2 <= w {
        return Err(format!("upscale: --width must exceed {w}"));
    }
    let src = |c: usize| c % w;
    let copies = |c: usize| (w2 - c + w - 1) / w; // number of new channels mapping to source c
    let mut seed = 0x9E3779B97F4A7C15u64;
    let mut rnd = move || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        ((seed >> 11) as f64 / (1u64 << 53) as f64) as f32 * 2.0 - 1.0
    };
    let mut out = net.clone();
    out.width = w2;
    out.ft = Vec::with_capacity(net.num_feats * w2);
    for f in 0..net.num_feats {
        for c in 0..w2 {
            let v = net.ft[f * w + src(c)];
            out.ft.push(if c >= w { v + noise * rnd() } else { v });
        }
    }
    out.ftb = (0..w2).map(|c| net.ftb[src(c)]).collect();
    let (din, din2) = (2 * w, 2 * w2);
    out.l1_w = Vec::with_capacity(net.buckets * net.l1 * din2);
    for r in 0..net.buckets * net.l1 {
        let row = &net.l1_w[r * din..(r + 1) * din];
        for half in 0..2 {
            out.l1_w.extend((0..w2).map(|c| row[half * w + src(c)] / copies(src(c)) as f32));
        }
    }
    Ok(out)
}

pub fn run_upscale(args: &[String]) -> ExitCode {
    let r = (|| -> Result<(), String> {
        let f = |n: &str| crate::flag_str(args, n);
        let (net_p, data, out_p) = (f("--net").ok_or("--net required")?, f("--data").ok_or("--data <stem> required")?, f("--out").ok_or("--out required")?);
        let w2 = crate::flag_u64(args, "--width").ok_or("--width required")? as usize;
        let rows = crate::flag_u64(args, "--rows").unwrap_or(20000) as usize;
        let threads = crate::flag_u64(args, "--threads").unwrap_or(6) as usize;
        let noise = crate::flag_f64(args, "--noise").unwrap_or(1e-4) as f32;
        let net = TrainableNet::from_tnn_bytes(&std::fs::read(&net_p).map_err(|e| format!("{net_p}: {e}"))?)?;
        let big = upscale(&net, w2, noise)?;
        let cal = load_val(&mut Tdat::open(&data)?, rows)?;
        let (a, b) = (evaluate(&net, &cal, threads), evaluate(&big, &cal, threads));
        println!("[upscale] W {} -> {} (noise {noise}); val mse on {} rows: original {:.8}  upscaled {:.8}  delta {:+.2e}", net.width, w2, cal.len(), a.mse, b.mse, b.mse - a.mse);
        std::fs::write(&out_p, big.to_tnn_bytes()).map_err(|e| e.to_string())?;
        println!("[upscale] wrote {out_p}");
        Ok(())
    })();
    match r {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("upscale: {e}");
            ExitCode::FAILURE
        }
    }
}

/// Split every phase bucket in two (B -> 2B) by copying the parent's weights. The engine and trainer compute
/// `bucket = empties * B / 50`, so bucket b2 of the new net nests exactly inside bucket b2 / 2 of the old one:
/// step-0 parity is exact and the extra buckets only help once calibrated / fine-tuned.
pub fn split_buckets(net: &TrainableNet) -> Result<TrainableNet, String> {
    let nb = net.buckets;
    let nb2 = nb * 2;
    let mut n = net.clone();
    n.buckets = nb2;
    let dup = |v: &Vec<f32>| -> Vec<f32> {
        if v.is_empty() {
            return Vec::new();
        }
        let chunk = v.len() / nb;
        let mut o = Vec::with_capacity(chunk * nb2);
        for b2 in 0..nb2 {
            o.extend_from_slice(&v[(b2 / 2) * chunk..(b2 / 2 + 1) * chunk]);
        }
        o
    };
    let nf = net.psqt.len() / nb;
    let mut psqt = Vec::with_capacity(nf * nb2);
    for fi in 0..nf {
        for b2 in 0..nb2 {
            psqt.push(net.psqt[fi * nb + b2 / 2]);
        }
    }
    n.psqt = psqt;
    n.l1_w = dup(&net.l1_w);
    n.l1_b = dup(&net.l1_b);
    n.l2_w = dup(&net.l2_w);
    n.l2_b = dup(&net.l2_b);
    n.out_w = dup(&net.out_w);
    n.out_b = dup(&net.out_b);
    n.s57w = dup(&net.s57w);
    n.dis_w = dup(&net.dis_w);
    n.dis_b = dup(&net.dis_b);
    Ok(n)
}

/// `ti train split --net a.tnn --data <stem> --out b.tnn [--rows N] [--threads T]`: B -> 2B buckets with val-mse parity check.
pub fn run_split(args: &[String]) -> ExitCode {
    let r = (|| -> Result<(), String> {
        let f = |n: &str| crate::flag_str(args, n);
        let (net_p, data, out_p) = (f("--net").ok_or("--net required")?, f("--data").ok_or("--data <stem> required")?, f("--out").ok_or("--out required")?);
        let rows = crate::flag_u64(args, "--rows").unwrap_or(20000) as usize;
        let threads = crate::flag_u64(args, "--threads").unwrap_or(6) as usize;
        let net = TrainableNet::from_tnn_bytes(&std::fs::read(&net_p).map_err(|e| format!("{net_p}: {e}"))?)?;
        let big = split_buckets(&net)?;
        let cal = load_val(&mut Tdat::open(&data)?, rows)?;
        let (a, b) = (evaluate(&net, &cal, threads), evaluate(&big, &cal, threads));
        println!("[split] buckets {} -> {}; val mse on {} rows: original {:.8}  split {:.8}  delta {:+.2e}", net.buckets, big.buckets, cal.len(), a.mse, b.mse, b.mse - a.mse);
        std::fs::write(&out_p, big.to_tnn_bytes()).map_err(|e| e.to_string())?;
        println!("[split] wrote {out_p}");
        Ok(())
    })();
    match r {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("split: {e}");
            ExitCode::FAILURE
        }
    }
}

/// Re-map the phase buckets of a net onto bucket map `id` (same bucket count): new bucket b' copies the weights of the old bucket
/// that holds the median empties count of b'. Parity is approximate unless the new buckets nest in the old ones, so the
/// val-mse change is printed; train / calibrate afterwards.
pub fn remap_buckets(net: &TrainableNet, id: u32) -> Result<TrainableNet, String> {
    let nb = net.buckets;
    let old = |e: usize| crate::mapper::bucket_of_id(e, nb, net.groups >> 8);
    let mut src = vec![0usize; nb];
    for b2 in 0..nb {
        let es: Vec<usize> = (1..=45).filter(|&e| crate::mapper::bucket_of_id(e, nb, id) == b2).collect();
        if es.is_empty() {
            return Err(format!("new bucket {b2} covers no reachable empties count (1..=45)"));
        }
        src[b2] = old(es[es.len() / 2]);
    }
    let mut n = net.clone();
    n.groups = (net.groups & 0xFF) | (id << 8);
    let pick = |v: &Vec<f32>| -> Vec<f32> {
        if v.is_empty() {
            return Vec::new();
        }
        let chunk = v.len() / nb;
        let mut o = Vec::with_capacity(v.len());
        for b2 in 0..nb {
            o.extend_from_slice(&v[src[b2] * chunk..(src[b2] + 1) * chunk]);
        }
        o
    };
    let nf = net.psqt.len() / nb;
    let mut psqt = Vec::with_capacity(net.psqt.len());
    for fi in 0..nf {
        for b2 in 0..nb {
            psqt.push(net.psqt[fi * nb + src[b2]]);
        }
    }
    n.psqt = psqt;
    n.l1_w = pick(&net.l1_w);
    n.l1_b = pick(&net.l1_b);
    n.l2_w = pick(&net.l2_w);
    n.l2_b = pick(&net.l2_b);
    n.out_w = pick(&net.out_w);
    n.out_b = pick(&net.out_b);
    n.s57w = pick(&net.s57w);
    n.dis_w = pick(&net.dis_w);
    n.dis_b = pick(&net.dis_b);
    Ok(n)
}

/// `ti train remap --net a.tnn --data <stem> --out b.tnn --map 1 [--rows N] [--threads T]`
pub fn run_remap(args: &[String]) -> ExitCode {
    let r = (|| -> Result<(), String> {
        let f = |n: &str| crate::flag_str(args, n);
        let (net_p, data, out_p) = (f("--net").ok_or("--net required")?, f("--data").ok_or("--data <stem> required")?, f("--out").ok_or("--out required")?);
        let id = crate::flag_u64(args, "--map").ok_or("--map <id> required")? as u32;
        let rows = crate::flag_u64(args, "--rows").unwrap_or(20000) as usize;
        let threads = crate::flag_u64(args, "--threads").unwrap_or(6) as usize;
        let net = TrainableNet::from_tnn_bytes(&std::fs::read(&net_p).map_err(|e| format!("{net_p}: {e}"))?)?;
        let big = remap_buckets(&net, id)?;
        let cal = load_val(&mut Tdat::open(&data)?, rows)?;
        let (a, b) = (evaluate(&net, &cal, threads), evaluate(&big, &cal, threads));
        println!("[remap] map {} -> {}; val mse on {} rows: original {:.8}  remapped {:.8}  delta {:+.2e}", net.groups >> 8, id, cal.len(), a.mse, b.mse, b.mse - a.mse);
        std::fs::write(&out_p, big.to_tnn_bytes()).map_err(|e| e.to_string())?;
        println!("[remap] wrote {out_p}");
        Ok(())
    })();
    match r {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("remap: {e}");
            ExitCode::FAILURE
        }
    }
}

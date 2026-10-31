//! H7: `ti train export` — turn a trained `TrainableNet` into a deployable TNN3:
//!  1. refit the zero-cost Disagreement Head by closed-form ridge regression per phase bucket
//!     (port of training/train_nnue_disagreement_head.py: target min(2000, |static_cp - search_cp|) / 400 on h1 || h2),
//!  2. report the quantisation range (the engine rejects ft/ftb weights that overflow i16 at QA 1024),
//!  3. write the file and verify it with the ENGINE's own loader and forward (f32 and quantised) + dis head.
use std::process::ExitCode;

use crate::mapper;
use crate::targets::MATE;
use crate::train::TrainableNet;
use crate::trainer::Tdat;
use titanium::nnue::Nnue;

const BLOCK: usize = 1024;

/// Solve (A + l2 I) x = v by Gaussian elimination with partial pivoting (f64, n <= 64).
pub(crate) fn solve(a: &[Vec<f64>], v: &[f64], l2: f64) -> Vec<f64> {
    let n = v.len();
    let mut m: Vec<Vec<f64>> = (0..n)
        .map(|i| {
            let mut r = a[i].clone();
            r[i] += l2;
            r.push(v[i]);
            r
        })
        .collect();
    for c in 0..n {
        let p = (c..n).max_by(|&x, &y| m[x][c].abs().partial_cmp(&m[y][c].abs()).unwrap()).unwrap();
        m.swap(c, p);
        let d = m[c][c];
        for k in c..=n {
            m[c][k] /= d;
        }
        for r in 0..n {
            if r != c {
                let f = m[r][c];
                if f != 0.0 {
                    for k in c..=n {
                        let t = m[c][k];
                        m[r][k] -= f * t;
                    }
                }
            }
        }
    }
    (0..n).map(|i| m[i][n]).collect()
}

/// Refit `dis_w` / `dis_b` on `rows` evenly spread training rows. Proven rows are excluded: their search value is a
/// +-(100000-ply) proof score, so |static - search| is a constant clip, not "disagreement".
pub fn refit_dis(net: &mut TrainableNet, train: &mut Tdat, rows: usize, l2: f64) -> Result<(), String> {
    let (l1, l2n, nb) = (net.l1, net.l2, net.buckets);
    let dim = l1 + l2n;
    let nblk = (train.rows / BLOCK).max(1);
    let blocks = (rows / BLOCK).max(1).min(nblk);
    let mut a: Vec<Vec<Vec<f64>>> = vec![vec![vec![0f64; dim + 1]; dim + 1]; nb];
    let mut v: Vec<Vec<f64>> = vec![vec![0f64; dim + 1]; nb];
    let (mut cnt, mut sy2) = (vec![0usize; nb], vec![0f64; nb]);
    let mut buf = Vec::with_capacity(BLOCK);
    let mut used = 0usize;
    for i in 0..blocks {
        buf.clear();
        train.read_block((i * nblk / blocks) * BLOCK, BLOCK, &mut buf)?;
        for r in &buf {
            if r.cp.abs() >= (MATE - 2000.0) {
                continue;
            }
            let f = net.forward_full(&mapper::board_of(r, 0));
            let delta = ((400.0 * f.y - r.cp).abs().min(2000.0) / 400.0) as f64;
            let mut x = Vec::with_capacity(dim + 1);
            x.extend(f.h1.iter().map(|&t| t as f64));
            x.extend(f.h2.iter().map(|&t| t as f64));
            x.push(1.0);
            let b = f.bucket;
            for p in 0..=dim {
                if x[p] != 0.0 {
                    for q in 0..=dim {
                        a[b][p][q] += x[p] * x[q];
                    }
                    v[b][p] += x[p] * delta;
                }
            }
            cnt[b] += 1;
            sy2[b] += delta * delta;
            used += 1;
        }
    }
    net.dis_w = vec![0f32; nb * dim];
    net.dis_b = vec![0f32; nb];
    net.has_dis = true;
    for b in 0..nb {
        if cnt[b] < 50 {
            net.dis_b[b] = 0.5;
            println!("  bucket {b}: only {} samples, baseline 0.5", cnt[b]);
            continue;
        }
        let w = solve(&a[b], &v[b], l2);
        for k in 0..dim {
            net.dis_w[b * dim + k] = w[k] as f32;
        }
        net.dis_b[b] = w[dim] as f32;
        // training RMSE from the normal equations: (w'Aw - 2 w'v + sum y^2) / n
        let mut waw = 0f64;
        for p in 0..=dim {
            for q in 0..=dim {
                waw += w[p] * a[b][p][q] * w[q];
            }
        }
        let wv: f64 = (0..=dim).map(|p| w[p] * v[b][p]).sum();
        let rmse = ((waw - 2.0 * wv + sy2[b]) / cnt[b] as f64).max(0.0).sqrt() * 400.0;
        println!("  bucket {b}: N={} bias {:.4} approx RMSE {:.1} cp", cnt[b], net.dis_b[b], rmse);
    }
    println!("  dis-head fit used {used} rows (proven excluded)");
    Ok(())
}


/// Texel-style output calibration: per phase bucket fit `y' = s*y + c` so that `sigmoid(y')` best matches the training
/// target (squared error in probability space), then fold it EXACTLY into out_w / out_b / psqt (y is linear in those).
/// No trunk change, no games. The dis head must be refit afterwards (it predicts |400 y - cp5k|).
pub fn calibrate(net: &mut TrainableNet, train: &mut Tdat, rows: usize) -> Result<(), String> {
    let nb = net.buckets;
    let out_in = net.out_in();
    let nblk = (train.rows / BLOCK).max(1);
    let blocks = (rows / BLOCK).max(1).min(nblk);
    let mut per: Vec<Vec<(f64, f64)>> = vec![Vec::new(); nb];
    let mut buf = Vec::with_capacity(BLOCK);
    for i in 0..blocks {
        buf.clear();
        train.read_block((i * nblk / blocks) * BLOCK, BLOCK, &mut buf)?;
        for r in &buf {
            let f = net.forward_full(&mapper::board_of(r, 0));
            per[f.bucket].push((f.y as f64, crate::targets::target_of(r).target as f64));
        }
    }
    let sig = |z: f64| 1.0 / (1.0 + (-z).exp());
    let mse = |d: &Vec<(f64, f64)>, s: f64, c: f64| d.iter().map(|&(y, t)| (sig(s * y + c) - t).powi(2)).sum::<f64>() / d.len() as f64;
    for b in 0..nb {
        let d = &per[b];
        if d.len() < 500 {
            println!("  cal bucket {b}: only {} rows, left alone", d.len());
            continue;
        }
        let (mut s, mut c) = (1f64, 0f64);
        let before = mse(d, 1.0, 0.0);
        for _ in 0..30 {
            let (mut a00, mut a01, mut a11, mut g0, mut g1) = (0f64, 0f64, 0f64, 0f64, 0f64);
            for &(y, t) in d {
                let p = sig(s * y + c);
                let (r, j) = (p - t, p * (1.0 - p));
                let (j0, j1) = (j * y, j);
                a00 += j0 * j0; a01 += j0 * j1; a11 += j1 * j1; g0 += j0 * r; g1 += j1 * r;
            }
            let lm = 1e-6 * (a00 + a11) + 1e-12;
            let (m00, m11) = (a00 + lm, a11 + lm);
            let det = m00 * m11 - a01 * a01;
            if det.abs() < 1e-18 { break; }
            let (ds, dc) = ((m11 * g0 - a01 * g1) / det, (m00 * g1 - a01 * g0) / det);
            let (ns, nc) = (s - ds, c - dc);
            if mse(d, ns, nc) > mse(d, s, c) { break; }
            s = ns; c = nc;
            if ds.abs() < 1e-7 && dc.abs() < 1e-7 { break; }
        }
        let after = mse(d, s, c);
        println!("  cal bucket {b}: N={} scale {:.4} shift {:+.4} mse {:.5} -> {:.5}", d.len(), s, c, before, after);
        let (sf, cf) = (s as f32, c as f32);
        for w in &mut net.out_w[b * out_in..(b + 1) * out_in] { *w *= sf; }
        net.out_b[b] = sf * net.out_b[b] + cf;
        let nf = net.psqt.len() / nb;
        for fi in 0..nf { net.psqt[fi * nb + b] *= sf; }
    }
    Ok(())
}

pub fn run_export(args: &[String]) -> ExitCode {
    match run_export_inner(args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("export: {e}");
            ExitCode::FAILURE
        }
    }
}

fn run_export_inner(args: &[String]) -> Result<(), String> {
    let f = |n: &str| crate::flag_str(args, n);
    let (net_path, data, out) = (f("--net").ok_or("--net <trained.tnn> required")?, f("--data").ok_or("--data <stem> required")?, f("--out").ok_or("--out <final.tnn> required")?);
    let rows = crate::flag_u64(args, "--rows").unwrap_or(200_000) as usize;
    let l2 = crate::flag_f64(args, "--l2").unwrap_or(1.0);
    let mut net = TrainableNet::from_tnn_bytes(&std::fs::read(&net_path).map_err(|e| format!("{net_path}: {e}"))?)?;
    let mut train = Tdat::open(&data)?;
    if args.iter().any(|a| a == "--calibrate") {
        let cr = crate::flag_u64(args, "--cal-rows").unwrap_or(400_000) as usize;
        println!("[export] Texel calibration of {net_path} on ~{cr} rows (per-bucket scale+shift of y, folded into out/psqt)");
        calibrate(&mut net, &mut train, cr)?;
    }
    println!("[export] refitting disagreement head of {net_path} on ~{rows} rows, ridge l2 {l2}");
    refit_dis(&mut net, &mut train, rows, l2)?;

    let m = |v: &Vec<f32>| v.iter().fold(0f32, |a, &x| a.max(x.abs()));
    let (mft, mftb) = (m(&net.ft), m(&net.ftb));
    println!("[export] max|w| ft {:.3} ftb {:.3} psqt {:.3} l1 {:.3} l2 {:.3} out {:.3} dis {:.3}", mft, mftb, m(&net.psqt), m(&net.l1_w), m(&net.l2_w), m(&net.out_w), m(&net.dis_w));
    // engine: i16 weights at QA 1024 => |w| < 32767/1024 = 31.99; the trainer clips at 1.98, so this is a sanity check
    if mft >= 31.9 || mftb >= 31.9 {
        return Err(format!("ft/ftb overflow i16 at QA 1024 (max {mft:.2}/{mftb:.2}); the engine would reject this net"));
    }
    std::fs::write(&out, net.to_tnn_bytes()).map_err(|e| e.to_string())?;
    println!("[export] wrote {out} ({} B) sha256 {}", std::fs::metadata(&out).map(|x| x.len()).unwrap_or(0), crate::hash::file_label(&out));

    // verify with the engine's own loader
    let eng = Nnue::load(&out).map_err(|e| format!("ENGINE REJECTED THE NET: {e}"))?;
    let mut t2 = Tdat::open(&data)?;
    let mut rows_v = Vec::new();
    let nblk = t2.rows / BLOCK;
    for i in 0..8 {
        t2.read_block((i * nblk / 8) * BLOCK + 17, 256, &mut rows_v)?;
    }
    let (mut wf, mut wq, mut wd) = (0f32, 0f32, 0i32);
    let dim = net.l1 + net.l2;
    for r in &rows_v {
        let b = mapper::board_of(r, 0);
        let fw = net.forward_full(&b);
        wf = wf.max((fw.y - eng.forward_f32(&b, 0.0)).abs());
        let (cp, dis) = eng.eval_with_dis(&b, 0);
        wq = wq.max((400.0 * fw.y - cp as f32).abs());
        let row = &net.dis_w[fw.bucket * dim..(fw.bucket + 1) * dim];
        let d: f32 = row.iter().zip(fw.h1.iter().chain(fw.h2.iter())).map(|(a, b)| a * b).sum::<f32>() + net.dis_b[fw.bucket];
        let want = ((400.0 * d.max(0.0)).round() as i32).clamp(0, 30000);
        wd = wd.max((want - dis).abs());
    }
    println!("[export] ENGINE LOADS IT. {} rows: hub vs engine f32 max {:.1e}; quantised eval vs f32 max {:.1} cp; dis head hub vs engine max {} cp", rows_v.len(), wf, wq, wd);
    if wf > 1e-3 {
        return Err("hub forward != engine f32 forward on the exported file".into());
    }
    Ok(())
}

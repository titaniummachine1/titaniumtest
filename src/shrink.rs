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
        let s = mapper::sample(r, 0, net.buckets);
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
    let (small, keep) = shrink(&net, w2, &cal)?;
    let before = evaluate(&net, &cal, threads);
    let after = evaluate(&small, &cal, threads);
    println!("[shrink] W {} -> {}, kept {} channels of {}", net.width, w2, keep.len(), net.width);
    println!("[shrink] val mse on {} rows: original {:.6}  pruned {:.6}  (distillation must win back {:.6})", cal.len(), before.mse, after.mse, after.mse - before.mse);
    std::fs::write(&out_p, small.to_tnn_bytes()).map_err(|e| e.to_string())?;
    println!("[shrink] wrote {out_p}");
    Ok(())
}

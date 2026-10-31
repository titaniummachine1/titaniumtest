//! `ti data check-leak --train <stem> --val <stem>`: is the validation set honest?
//! Reports how many val positions also occur in train, exactly and up to a D4 rotation/reflection (the trainer augments
//! with D4, so a rotated copy of a train position is a leaked val position). Also the D4-duplicate rate inside train
//! (informational: how much the 8x augmentation is repeating positions the data already contains).
//! RAM: one sorted u64 per train row for the exact key and one for the D4-canonical key (~0.5 GB for v9).
use std::process::ExitCode;

use crate::mapper::transform_bb;
use crate::trainer::Tdat;

const BLOCK: usize = 1024;

fn mix(o0: u64, o1: u64, stm: u8) -> u64 {
    let mut h = o0.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ o1.rotate_left(29).wrapping_mul(0xC2B2_AE3D_27D4_EB4F) ^ (stm as u64 + 1).wrapping_mul(0x1656_67B1_9E37_79F9);
    h ^= h >> 32;
    h = h.wrapping_mul(0xD6E8_FEB8_6659_FD93);
    h ^ (h >> 32)
}

/// (exact key, D4-canonical key = min over the 8 orientations)
fn keys(o0: u64, o1: u64, stm: u8) -> (u64, u64) {
    let exact = mix(o0, o1, stm);
    let mut canon = exact;
    for m in 1..8 {
        canon = canon.min(mix(transform_bb(o0, m), transform_bb(o1, m), stm));
    }
    (exact, canon)
}

fn all_keys(stem: &str, threads: usize) -> Result<(Vec<u64>, Vec<u64>), String> {
    let rows = Tdat::open(stem)?.rows;
    let nblk = (rows + BLOCK - 1) / BLOCK;
    let per = (nblk + threads - 1) / threads;
    let parts: Vec<Result<(Vec<u64>, Vec<u64>), String>> = std::thread::scope(|s| {
        (0..threads)
            .map(|t| {
                s.spawn(move || -> Result<(Vec<u64>, Vec<u64>), String> {
                    let mut f = Tdat::open(stem)?;
                    let (mut ex, mut ca) = (Vec::new(), Vec::new());
                    let mut buf = Vec::with_capacity(BLOCK);
                    for b in (t * per)..((t + 1) * per).min(nblk) {
                        buf.clear();
                        let start = b * BLOCK;
                        f.read_block(start, BLOCK.min(rows - start), &mut buf)?;
                        for r in &buf {
                            let (e, c) = keys(r.occ0, r.occ1, r.stm);
                            ex.push(e);
                            ca.push(c);
                        }
                    }
                    Ok((ex, ca))
                })
            })
            .collect::<Vec<_>>()
            .into_iter()
            .map(|h| h.join().unwrap())
            .collect()
    });
    let (mut ex, mut ca) = (Vec::with_capacity(rows), Vec::with_capacity(rows));
    for p in parts {
        let (e, c) = p?;
        ex.extend(e);
        ca.extend(c);
    }
    Ok((ex, ca))
}

fn dup_count(sorted: &[u64]) -> usize {
    sorted.windows(2).filter(|w| w[0] == w[1]).count()
}

pub fn run_check(args: &[String]) -> ExitCode {
    let f = |n: &str| crate::flag_str(args, n);
    let (Some(tr), Some(va)) = (f("--train"), f("--val")) else {
        eprintln!("usage: ti data check-leak --train <stem> --val <stem> [--threads 6]");
        return ExitCode::FAILURE;
    };
    let threads = crate::flag_u64(args, "--threads").unwrap_or(6) as usize;
    let t0 = std::time::Instant::now();
    let (mut tr_ex, mut tr_ca) = match all_keys(&tr, threads) {
        Ok(k) => k,
        Err(e) => {
            eprintln!("check-leak: {e}");
            return ExitCode::FAILURE;
        }
    };
    let (va_ex, va_ca) = match all_keys(&va, threads) {
        Ok(k) => k,
        Err(e) => {
            eprintln!("check-leak: {e}");
            return ExitCode::FAILURE;
        }
    };
    println!("[leak] keys computed: train {} val {} in {:.0} s", tr_ex.len(), va_ex.len(), t0.elapsed().as_secs_f64());
    tr_ex.sort_unstable();
    tr_ca.sort_unstable();
    let n = va_ex.len() as f64;
    let in_ex = va_ex.iter().filter(|k| tr_ex.binary_search(k).is_ok()).count();
    let in_ca = va_ca.iter().filter(|k| tr_ca.binary_search(k).is_ok()).count();
    let tdup_ex = dup_count(&tr_ex);
    let tdup_ca = dup_count(&tr_ca);
    let mut vs = va_ca.clone();
    vs.sort_unstable();
    let vdup_ca = dup_count(&vs);
    println!("val rows with an EXACT copy in train:         {in_ex} ({:.3}%)", 100.0 * in_ex as f64 / n);
    println!("val rows with a D4-equivalent copy in train:  {in_ca} ({:.3}%)   <- the leak that hides overfit", 100.0 * in_ca as f64 / n);
    println!("train rows repeating an earlier train position: exact {tdup_ex} ({:.2}%), up to D4 {tdup_ca} ({:.2}%)", 100.0 * tdup_ex as f64 / tr_ex.len() as f64, 100.0 * tdup_ca as f64 / tr_ex.len() as f64);
    println!("val rows repeating an earlier val position (up to D4): {vdup_ca} ({:.2}%)", 100.0 * vdup_ca as f64 / n);
    // where does the leak live? by stones on board (early positions repeat naturally: symmetric openings)
    let mut vf = Tdat::open(&va).expect("val");
    let mut vrows = Vec::with_capacity(vf.rows);
    for s in (0..vf.rows).step_by(BLOCK) {
        vf.read_block(s, BLOCK, &mut vrows).expect("read val");
    }
    let mut by: [(usize, usize); 6] = [(0, 0); 6]; // stones <12, <20, <28, <36, <44, >=44 : (rows, leaked)
    let clean_path = crate::flag_str(args, "--write-clean");
    let mut clean: Vec<u8> = Vec::new();
    let mut kept = 0usize;
    for (r, k) in vrows.iter().zip(va_ca.iter()) {
        let st = (r.occ0 | r.occ1).count_ones() as usize;
        let b = (st / 8).clamp(1, 6) - 1;
        let leaked = tr_ca.binary_search(k).is_ok();
        by[b].0 += 1;
        by[b].1 += leaked as usize;
        if !leaked {
            kept += 1;
            if clean_path.is_some() {
                clean.extend_from_slice(&r.to_bytes());
            }
        }
    }
    for (i, (rows, lk)) in by.iter().enumerate() {
        println!("  stones {:>2}-{:>2}: {:>7} val rows, leaked {:>6} ({:.1}%)", (i + 1) * 8, (i + 1) * 8 + 7, rows, lk, 100.0 * *lk as f64 / (*rows).max(1) as f64);
    }
    if let Some(p) = clean_path {
        let p = if p.ends_with(".tdat") { p } else { format!("{p}.tdat") };
        std::fs::write(&p, &clean).expect("write clean val");
        println!("[leak] wrote {p}: {kept} rows kept of {} (leaked rows dropped)", vrows.len());
    }
    println!("verdict: {}", if in_ca as f64 / n > 0.02 { "LEAK > 2%: val loss under-reports overfitting, build a position-disjoint val" } else { "val is position-disjoint enough (<= 2%)" });
    ExitCode::SUCCESS
}

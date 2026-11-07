//! `ti policy2` (D523): compact policy trainer for the FULL KataAtaxx set (~21.7M rows) + our gate-game labels on the
//! 16 GB PC. A sample stores only the board (own/opp bitboards) and its labels (the played move, or Kata's 200-visit
//! counts per square); moves, scalar features and 3x3 pattern rows are rebuilt from the board every epoch (~60 B per
//! sample instead of ~550 B). Head = what the engine runs on the non-pairwise champion (dim 0): slot bias + move scalars
//! + 3x3 pattern tables (k3 per square = TPOL2, k3c class-shared = TPOL3). Kata rows: stage 0 groups = clone destinations
//! and jump ORIGINS, stage 1 groups = destinations of the picked stone; soft loss -sum_g y_g ln(sum_{m in g} q_m).
//! Epochs run until the validation loss stops improving (early stop), the best epoch is exported.
//!
//!   ti policy2 --net NET.tnn --games a.jsonl,.. --kata DIR[,DIR] --pat k3c --init POL.bin --out-bin OUT.bin
//!              [--max-epochs 30] [--lr 0.003] [--pat-lr 0.01] [--kata-w 1] [--games-w 1] [--patience 1]
use crate::games;
use crate::policy::{arg, feats, k3code, npz_files, patidx, ring1, NONE, NSCAL, P3, SLOTS};
use std::fs;
use std::io::{BufRead, BufReader};
use std::process::ExitCode;
use titanium::{Board, Move, MoveList};

#[derive(Clone, Copy)]
struct S {
    own: u64,
    opp: u64,
    /// hard: played move from/to; soft: unused
    from: u8,
    to: u8,
    /// 0 = hard (game move), 1 = Kata stage 0, 2 = Kata stage 1 (picked stone `sel`)
    kind: u8,
    sel: u8,
    kn: u8,
    val: bool,
    ko: u32,
}

struct Scratch {
    ml: MoveList,
    mv: Vec<[u8; 5]>,
    pv: Vec<[u32; 2]>,
    kg: Vec<i8>,
    ky: Vec<f32>,
    keys: Vec<u8>,
    z: Vec<f32>,
    dz: Vec<f32>,
    qg: Vec<f32>,
}

/// Rebuild moves / features / pattern rows / labels of a sample. Returns the hard label index (hard samples) or
/// Some(usize::MAX) for a soft sample with >= 2 groups; None = skip.
fn expand(s: &S, kpool: &[(u8, u16)], ring: &[u64; 49], k3c: bool, use_pat: bool, x: &mut Scratch) -> Option<usize> {
    let b = Board::from_occ(s.own, s.opp, 0, 0);
    b.legal_moves_dedup_into(&mut x.ml);
    if x.ml.len() < 2 {
        return None;
    }
    x.mv.clear();
    x.pv.clear();
    x.kg.clear();
    x.ky.clear();
    x.keys.clear();
    let mut label = None;
    for (i, m) in x.ml.iter().enumerate() {
        x.mv.push(feats(&b, m, ring));
        if use_pat {
            x.pv.push(patidx(&b, m, k3c));
        }
        if s.kind == 0 {
            if m.to == s.to && (m.is_clone() || m.from == s.from) && m.is_clone() == (Move { from: s.from, to: s.to }).is_clone() {
                label = Some(i);
            }
        } else {
            let key = if s.kind == 2 {
                if !m.is_clone() && m.from == s.sel { Some(m.to) } else { None }
            } else if m.is_clone() {
                Some(m.to)
            } else {
                Some(m.from)
            };
            match key {
                None => x.kg.push(-1),
                Some(k) => {
                    let gi = match x.keys.iter().position(|&q| q == k) {
                        Some(p) => p,
                        None => {
                            x.keys.push(k);
                            let v = kpool[s.ko as usize..s.ko as usize + s.kn as usize].iter().find(|e| e.0 == k).map(|e| e.1).unwrap_or(0);
                            x.ky.push(v as f32);
                            x.keys.len() - 1
                        }
                    };
                    x.kg.push(gi as i8);
                }
            }
        }
    }
    if s.kind == 0 {
        return label;
    }
    let tot: f32 = x.ky.iter().sum();
    if x.keys.len() < 2 || tot <= 0.0 {
        return None;
    }
    for y in x.ky.iter_mut() {
        *y /= tot;
    }
    Some(usize::MAX)
}

/// D523: per-thread gradient buffers (dense; the k3c tables are 177k floats, cheap per thread)
struct Grad {
    gb: Vec<f32>,
    gg: Vec<f32>,
    gtd: Vec<f32>,
    gto: Vec<f32>,
    touched: Vec<u32>,
    touched_o: Vec<u32>,
    sc: Scratch,
    z: Vec<f32>,
}

fn new_scratch() -> Scratch {
    Scratch { ml: MoveList::new(), mv: vec![], pv: vec![], kg: vec![], ky: vec![], keys: vec![], z: vec![], dz: vec![], qg: vec![] }
}

#[allow(clippy::too_many_arguments)]
fn grad_one(s: &S, kpool: &[(u8, u16)], ring: &[u64; 49], k3c: bool, use_pat: bool, prm: (&[f32], &[f32], &[f32], &[f32]), w: (f32, f32), inv_n: f32, gr: &mut Grad) {
    let (bias, g, td, tor) = prm;
    let Some(lab) = expand(s, kpool, ring, k3c, use_pat, &mut gr.sc) else { return };
    let x = &mut gr.sc;
    let z = &mut gr.z;
    z.clear();
    for (i, m) in x.mv.iter().enumerate() {
        let mut v = bias[m[0] as usize] + g[m[1] as usize] + g[m[2] as usize];
        if m[3] != 255 {
            v += g[m[3] as usize];
        }
        if !td.is_empty() {
            let [d, o] = x.pv[i];
            v += td[d as usize];
            if o != NONE {
                v += tor[o as usize];
            }
        }
        z.push(v);
    }
    x.dz.clear();
    if s.kind == 0 {
        let mx = z.iter().cloned().fold(f32::MIN, f32::max);
        let den: f32 = z.iter().map(|v| (v - mx).exp()).sum();
        for (i, v) in z.iter().enumerate() {
            x.dz.push(w.0 * ((v - mx).exp() / den - (i == lab) as u32 as f32));
        }
    } else {
        let mx = z.iter().zip(&x.kg).filter(|(_, k)| **k >= 0).map(|(v, _)| *v).fold(f32::MIN, f32::max);
        x.qg.clear();
        x.qg.resize(x.ky.len(), 0.0);
        let mut den = 0f32;
        for (v, k) in z.iter().zip(&x.kg) {
            if *k >= 0 {
                let e = (v - mx).exp();
                x.qg[*k as usize] += e;
                den += e;
            }
        }
        for (v, k) in z.iter().zip(&x.kg) {
            if *k < 0 {
                x.dz.push(0.0);
            } else {
                let q = (v - mx).exp() / den;
                let gq = (x.qg[*k as usize] / den).max(1e-9);
                x.dz.push(w.1 * (q - x.ky[*k as usize] * q / gq));
            }
        }
    }
    for (i, m) in x.mv.iter().enumerate() {
        let d = x.dz[i] * inv_n;
        if d == 0.0 {
            continue;
        }
        gr.gb[m[0] as usize] += d;
        gr.gg[m[1] as usize] += d;
        gr.gg[m[2] as usize] += d;
        if m[3] != 255 {
            gr.gg[m[3] as usize] += d;
        }
        if use_pat {
            let [pd, po] = x.pv[i];
            gr.touched.push(pd);
            gr.gtd[pd as usize] += d;
            if po != NONE {
                gr.touched_o.push(po);
                gr.gto[po as usize] += d;
            }
        }
    }
}

pub fn run(args: &[String]) -> ExitCode {
    let Some(net_path) = arg(args, "--net") else {
        eprintln!("usage: ti policy2 --net NET.tnn --games a.jsonl[,..] [--kata DIR[,DIR]] --pat k3c|k3 [--init POL.bin] --out-bin OUT.bin [--max-epochs 30] [--lr 0.003] [--pat-lr 0.01] [--kata-w 1] [--games-w 1] [--patience 1] [--max-samples N]");
        return ExitCode::from(2);
    };
    let _ = net_path; // dim 0: the head reads no accumulator (the net only fixes which head the file belongs to)
    let files: Vec<String> = arg(args, "--games").unwrap_or_default().split(',').filter(|s| !s.is_empty()).map(String::from).collect();
    let kata_dirs: Vec<String> = arg(args, "--kata").unwrap_or_default().split(',').filter(|s| !s.is_empty()).map(String::from).collect();
    let pat_kind = arg(args, "--pat").unwrap_or_else(|| "k3c".into());
    let use_pat = pat_kind == "k3" || pat_kind == "k3c";
    let k3c = pat_kind == "k3c";
    let (nd_slots, no_slots) = if k3c { (18usize, 9usize) } else { (SLOTS, 49usize) };
    let max_epochs: usize = arg(args, "--max-epochs").and_then(|v| v.parse().ok()).unwrap_or(30);
    let lr0: f32 = arg(args, "--lr").and_then(|v| v.parse().ok()).unwrap_or(0.003);
    let pat_lr: f32 = arg(args, "--pat-lr").and_then(|v| v.parse().ok()).unwrap_or(0.01);
    let kata_w: f32 = arg(args, "--kata-w").and_then(|v| v.parse().ok()).unwrap_or(1.0);
    let games_w: f32 = arg(args, "--games-w").and_then(|v| v.parse().ok()).unwrap_or(1.0);
    let patience: usize = arg(args, "--patience").and_then(|v| v.parse().ok()).unwrap_or(1);
    let max_samples: usize = arg(args, "--max-samples").and_then(|v| v.parse().ok()).unwrap_or(usize::MAX);
    let threads: usize = arg(args, "--threads").and_then(|v| v.parse().ok()).unwrap_or(8).max(1);
    let ring = ring1();

    let mut ss: Vec<S> = Vec::new();
    let mut kpool: Vec<(u8, u16)> = Vec::new();
    // games: every played move is a hard label; val = game pair % 10 == 0
    let mut ml = MoveList::new();
    for f in &files {
        let Ok(fh) = fs::File::open(f) else {
            eprintln!("{f}: cannot open");
            continue;
        };
        let mut got = 0usize;
        for line in BufReader::new(fh).lines().map_while(Result::ok) {
            let Ok(g) = games::parse_game(&line) else { continue };
            let Ok(boards) = games::replay(&g) else { continue };
            let val = g.pair % 10 == 0;
            for (r, b) in g.moves.iter().zip(&boards) {
                if r.mv == "FORFEIT" {
                    break;
                }
                let pm = match b.parse_uai_move(&r.mv) {
                    Some(m) if !m.is_pass() => m,
                    _ => continue,
                };
                b.legal_moves_dedup_into(&mut ml);
                if ml.len() < 2 {
                    continue;
                }
                ss.push(S { own: b.me(), opp: b.opp(), from: pm.from, to: pm.to, kind: 0, sel: 255, kn: 0, val, ko: 0 });
                got += 1;
            }
        }
        eprintln!("{f}: {got} positions");
    }
    let n_games = ss.len();
    // Kata tdata (all npz under the dirs); val = every 10th file
    let mut kf: Vec<std::path::PathBuf> = Vec::new();
    for d in &kata_dirs {
        npz_files(std::path::Path::new(d), &mut kf);
    }
    kf.sort();
    let (mut bb, mut pb) = (Vec::new(), Vec::new());
    let (mut nk0, mut nk1, mut bad) = (0usize, 0usize, 0usize);
    for (fi, fp) in kf.iter().enumerate() {
        if ss.len() >= max_samples {
            break;
        }
        let val = fi % 10 == 0;
        let path = fp.to_string_lossy().to_string();
        let z = match crate::data::Npz::open(&path) {
            Ok(z) => z,
            Err(e) => {
                eprintln!("{path}: {e}");
                bad += 1;
                continue;
            }
        };
        let (Ok(mut cb), Ok(mut cp)) = (z.col("binaryInputNCHWPacked"), z.col("policyTargetsNCMove")) else {
            bad += 1;
            continue;
        };
        if cb.width * cb.elem != 154 || cp.width * cp.elem != 200 {
            bad += 1;
            continue;
        }
        let n = cb.rows;
        if cb.read(n, &mut bb).is_err() || cp.read(n, &mut pb).is_err() {
            bad += 1;
            continue;
        }
        for r in 0..n {
            let row = &bb[r * 154..(r + 1) * 154];
            let plane = |c: usize| -> u64 {
                let mut x = 0u64;
                for i in 0..49 {
                    if row[c * 7 + i / 8] >> (7 - i % 8) & 1 == 1 {
                        x |= 1u64 << i;
                    }
                }
                x
            };
            let (own, opp, sel) = (plane(1), plane(2), plane(4));
            let ko = kpool.len() as u32;
            for i in 0..49 {
                let v = i16::from_le_bytes([pb[r * 200 + 2 * i], pb[r * 200 + 2 * i + 1]]);
                if v > 0 {
                    kpool.push((i as u8, v as u16));
                }
            }
            let kn = kpool.len() as u32 - ko;
            if kn == 0 {
                continue;
            }
            let st1 = sel != 0;
            ss.push(S { own, opp, from: 0, to: 0, kind: if st1 { 2 } else { 1 }, sel: if st1 { sel.trailing_zeros() as u8 } else { 255 }, kn: kn as u8, val, ko });
            if st1 {
                nk1 += 1;
            } else {
                nk0 += 1;
            }
        }
        if fi % 50 == 49 {
            eprintln!("kata: {}/{} files, {} rows", fi + 1, kf.len(), nk0 + nk1);
        }
    }
    let nval = ss.iter().filter(|s| s.val).count();
    println!(
        "samples {} = games {} + kata {} (stage0 {} stage1 {}, {} bad files of {}); val {}; kpool {} entries; ~{} MB",
        ss.len(), n_games, nk0 + nk1, nk0, nk1, bad, kf.len(), nval, kpool.len(),
        (ss.len() * std::mem::size_of::<S>() + kpool.len() * 4) / 1_000_000
    );

    // parameters (dim 0 head)
    let mut bias = vec![0f32; SLOTS];
    let mut g = vec![0f32; NSCAL];
    let mut td = vec![0f32; if use_pat { nd_slots * P3 } else { 0 }];
    let mut tor = vec![0f32; if use_pat { no_slots * P3 } else { 0 }];
    if let Some(ip) = arg(args, "--init") {
        let Ok(d) = fs::read(&ip) else {
            eprintln!("{ip}: cannot read");
            return ExitCode::from(1);
        };
        let idim = u32::from_le_bytes(d[8..12].try_into().unwrap()) as usize;
        let (v2, v3) = (&d[0..5] == b"TPOL2", &d[0..5] == b"TPOL3");
        let n1 = SLOTS + NSCAL + SLOTS * idim;
        let ntab = if v3 { 27 * P3 } else if v2 { (SLOTS + 49) * P3 } else { 0 };
        if !(&d[0..5] == b"TPOL1" || v2 || v3) || d.len() != 12 + 4 * (n1 + ntab) || ((v2 || v3) && v3 != k3c) {
            eprintln!("--init {ip}: bad file or table kind does not match --pat");
            return ExitCode::from(2);
        }
        let f: Vec<f32> = d[12..].chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
        bias.copy_from_slice(&f[..SLOTS]);
        g.copy_from_slice(&f[SLOTS..SLOTS + NSCAL]);
        if (v2 || v3) && use_pat {
            td.copy_from_slice(&f[n1..n1 + nd_slots * P3]);
            tor.copy_from_slice(&f[n1 + nd_slots * P3..]);
        }
        println!("init from {ip} (file dim {idim}{}, tables {})", if idim > 0 { ", U dropped as in the engine" } else { "" }, if v2 || v3 { "loaded" } else { "zero" });
    }

    let logits = |x: &Scratch, bias: &[f32], g: &[f32], td: &[f32], tor: &[f32], out: &mut Vec<f32>| {
        out.clear();
        for (i, m) in x.mv.iter().enumerate() {
            let mut z = bias[m[0] as usize] + g[m[1] as usize] + g[m[2] as usize];
            if m[3] != 255 {
                z += g[m[3] as usize];
            }
            if !td.is_empty() {
                let [d, o] = x.pv[i];
                z += td[d as usize];
                if o != NONE {
                    z += tor[o as usize];
                }
            }
            out.push(z);
        }
    };
    let mut sc = Scratch { ml: MoveList::new(), mv: vec![], pv: vec![], kg: vec![], ky: vec![], keys: vec![], z: vec![], dz: vec![], qg: vec![] };
    // validation: (game CE, top1, top3, top5, n) and (kata soft CE, top1 most-visited group, n)
    let evaluate = |bias: &[f32], g: &[f32], td: &[f32], tor: &[f32], sc: &mut Scratch| -> ([f64; 5], [f64; 3]) {
        let (mut gm, mut km) = ([0f64; 5], [0f64; 3]);
        let mut z = std::mem::take(&mut sc.z);
        for s in ss.iter().filter(|s| s.val) {
            let Some(lab) = expand(s, &kpool, &ring, k3c, use_pat, sc) else { continue };
            logits(sc, bias, g, td, tor, &mut z);
            if s.kind == 0 {
                let mx = z.iter().cloned().fold(f32::MIN, f32::max);
                let den: f32 = z.iter().map(|v| (v - mx).exp()).sum();
                gm[0] += (den.ln() + mx - z[lab]) as f64;
                let rank = z.iter().filter(|v| **v > z[lab]).count();
                gm[1] += (rank == 0) as u32 as f64;
                gm[2] += (rank < 3) as u32 as f64;
                gm[3] += (rank < 5) as u32 as f64;
                gm[4] += 1.0;
            } else {
                let mx = z.iter().zip(&sc.kg).filter(|(_, k)| **k >= 0).map(|(v, _)| *v).fold(f32::MIN, f32::max);
                let mut qg = vec![0f32; sc.ky.len()];
                let mut den = 0f32;
                for (v, k) in z.iter().zip(&sc.kg) {
                    if *k >= 0 {
                        let e = (v - mx).exp();
                        qg[*k as usize] += e;
                        den += e;
                    }
                }
                for (y, q) in sc.ky.iter().zip(&qg) {
                    if *y > 0.0 {
                        km[0] -= (*y as f64) * ((q / den).max(1e-9) as f64).ln();
                    }
                }
                let am = |v: &[f32]| (0..v.len()).max_by(|&a, &b| v[a].partial_cmp(&v[b]).unwrap()).unwrap();
                km[1] += (am(&qg) == am(&sc.ky)) as u32 as f64;
                km[2] += 1.0;
            }
        }
        sc.z = z;
        for i in 0..4 {
            gm[i] /= gm[4].max(1.0);
        }
        km[0] /= km[2].max(1.0);
        km[1] /= km[2].max(1.0);
        (gm, km)
    };
    let show = |tag: &str, r: &([f64; 5], [f64; 3])| {
        println!(
            "{tag}: games CE {:.4} top1 {:.4} top3 {:.4} top5 {:.4} ({:.0}) | kata soft CE {:.4} top1 {:.4} ({:.0})",
            r.0[0], r.0[1], r.0[2], r.0[3], r.0[4], r.1[0], r.1[1], r.1[2]
        );
    };
    let r0 = evaluate(&bias, &g, &td, &tor, &mut sc);
    show("val init", &r0);
    let score = |r: &([f64; 5], [f64; 3])| -> f64 { (if r.0[4] > 0.0 { r.0[0] } else { 0.0 }) + (if r.1[2] > 0.0 { r.1[0] } else { 0.0 }) };
    let mut best = (score(&r0), bias.clone(), g.clone(), td.clone(), tor.clone(), 0usize);

    // Adam (dense on bias/g, lazy sparse on the tables)
    let mut tr: Vec<u32> = (0..ss.len() as u32).filter(|&i| !ss[i as usize].val).collect();
    let (b1, b2, eps) = (0.9f32, 0.999f32, 1e-8f32);
    let (mut mb, mut vb, mut mg, mut vg) = (vec![0f32; SLOTS], vec![0f32; SLOTS], vec![0f32; NSCAL], vec![0f32; NSCAL]);
    let (mut mtd, mut vtd, mut mto, mut vto) = (vec![0f32; td.len()], vec![0f32; td.len()], vec![0f32; tor.len()], vec![0f32; tor.len()]);
    let (mut gtd, mut gto) = (vec![0f32; td.len()], vec![0f32; tor.len()]);
    let (mut touched, mut touched_o): (Vec<u32>, Vec<u32>) = (Vec::new(), Vec::new());
    let mut t = 0i32;
    let bs = 256usize;
    let mut bad_epochs = 0usize;
    let _z: Vec<f32> = Vec::new();
    for ep in 0..max_epochs {
        let lr = lr0 * (0.85f32).powi(ep as i32);
        let mut x: u64 = 0x9E37_79B9_7F4A_7C15 ^ (ep as u64 + 1).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        for i in (1..tr.len()).rev() {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            tr.swap(i, (x % (i as u64 + 1)) as usize);
        }
        let t0 = std::time::Instant::now();
        let (mut gb, mut gg) = (vec![0f32; SLOTS], vec![0f32; NSCAL]);
        // D526: PERSISTENT workers (one spawn per epoch, two barriers per batch) instead of 8 spawns per batch.
        // Workers read the parameters only between `start` and `done`; the main thread updates them only outside.
        let grs: Vec<std::sync::Mutex<Grad>> = (0..threads)
            .map(|_| std::sync::Mutex::new(Grad { gb: vec![0f32; SLOTS], gg: vec![0f32; NSCAL], gtd: vec![0f32; td.len()], gto: vec![0f32; tor.len()], touched: vec![], touched_o: vec![], sc: new_scratch(), z: vec![] }))
            .collect();
        let bt = bs * threads;
        let nb = tr.len().div_ceil(bt);
        let (start_b, done_b) = (std::sync::Barrier::new(threads + 1), std::sync::Barrier::new(threads + 1));
        let cur = std::sync::atomic::AtomicUsize::new(0);
        struct Ptr(*mut f32, usize);
        unsafe impl Send for Ptr {}
        unsafe impl Sync for Ptr {}
        let pp = [Ptr(bias.as_mut_ptr(), bias.len()), Ptr(g.as_mut_ptr(), g.len()), Ptr(td.as_mut_ptr(), td.len()), Ptr(tor.as_mut_ptr(), tor.len())];
        let trr = &tr[..];
        std::thread::scope(|scp| {
            for ti in 0..threads {
                let (grs, start_b, done_b, cur, pp) = (&grs, &start_b, &done_b, &cur, &pp);
                let (kp, rg, ssr) = (&kpool[..], &ring, &ss[..]);
                scp.spawn(move || loop {
                    start_b.wait();
                    let b = cur.load(std::sync::atomic::Ordering::Acquire);
                    if b == usize::MAX {
                        break;
                    }
                    let chunk = &trr[b * bt..((b + 1) * bt).min(trr.len())];
                    let inv_n = 1.0 / chunk.len() as f32;
                    let per = chunk.len().div_ceil(threads);
                    let part = &chunk[(ti * per).min(chunk.len())..((ti + 1) * per).min(chunk.len())];
                    // SAFETY: between the barriers nobody writes the parameters
                    let prm = unsafe {
                        (
                            std::slice::from_raw_parts(pp[0].0 as *const f32, pp[0].1),
                            std::slice::from_raw_parts(pp[1].0 as *const f32, pp[1].1),
                            std::slice::from_raw_parts(pp[2].0 as *const f32, pp[2].1),
                            std::slice::from_raw_parts(pp[3].0 as *const f32, pp[3].1),
                        )
                    };
                    {
                        let mut gr = grs[ti].lock().unwrap();
                        for &si in part {
                            grad_one(&ssr[si as usize], kp, rg, k3c, use_pat, prm, (games_w, kata_w), inv_n, &mut gr);
                        }
                    }
                    done_b.wait();
                });
            }
            for bi in 0..nb {
                gb.iter_mut().for_each(|v| *v = 0.0);
                gg.iter_mut().for_each(|v| *v = 0.0);
                cur.store(bi, std::sync::atomic::Ordering::Release);
                start_b.wait();
                done_b.wait();
                for grm in grs.iter() {
                    let mut gr = grm.lock().unwrap();
                    let gr = &mut *gr;
                    for (a, b) in gb.iter_mut().zip(gr.gb.iter_mut()) {
                        *a += *b;
                        *b = 0.0;
                    }
                    for (a, b) in gg.iter_mut().zip(gr.gg.iter_mut()) {
                        *a += *b;
                        *b = 0.0;
                    }
                    for &k in &gr.touched {
                        let k = k as usize;
                        gtd[k] += gr.gtd[k];
                        gr.gtd[k] = 0.0;
                    }
                    for &k in &gr.touched_o {
                        let k = k as usize;
                        gto[k] += gr.gto[k];
                        gr.gto[k] = 0.0;
                    }
                    touched.append(&mut gr.touched);
                    touched_o.append(&mut gr.touched_o);
                }
                t += 1;
                let c1 = 1.0 - b1.powi(t);
                let c2 = 1.0 - b2.powi(t);
                // SAFETY: workers are parked at `start_b`; the main thread is the only accessor now
                let (pbias, pg, ptd, ptor) = unsafe {
                    (
                        std::slice::from_raw_parts_mut(pp[0].0, pp[0].1),
                        std::slice::from_raw_parts_mut(pp[1].0, pp[1].1),
                        std::slice::from_raw_parts_mut(pp[2].0, pp[2].1),
                        std::slice::from_raw_parts_mut(pp[3].0, pp[3].1),
                    )
                };
                for (p, gr, m, v) in [(pbias, &gb, &mut mb, &mut vb), (pg, &gg, &mut mg, &mut vg)] {
                    for i in 0..p.len() {
                        m[i] = b1 * m[i] + (1.0 - b1) * gr[i];
                        v[i] = b2 * v[i] + (1.0 - b2) * gr[i] * gr[i];
                        p[i] -= lr * (m[i] / c1) / ((v[i] / c2).sqrt() + eps);
                    }
                }
                if use_pat {
                    for (tab, gr, m, v, list) in [(ptd, &mut gtd, &mut mtd, &mut vtd, &mut touched), (ptor, &mut gto, &mut mto, &mut vto, &mut touched_o)] {
                        list.sort_unstable();
                        list.dedup();
                        for &k in list.iter() {
                            let k = k as usize;
                            m[k] = b1 * m[k] + (1.0 - b1) * gr[k];
                            v[k] = b2 * v[k] + (1.0 - b2) * gr[k] * gr[k];
                            tab[k] -= pat_lr * (lr / lr0) * (m[k] / c1) / ((v[k] / c2).sqrt() + eps);
                            gr[k] = 0.0;
                        }
                        list.clear();
                    }
                }
                if bi % 2000 == 1999 {
                    eprint!(".");
                }
            }
            cur.store(usize::MAX, std::sync::atomic::Ordering::Release);
            start_b.wait();
        });
        let r = evaluate(&bias, &g, &td, &tor, &mut sc);
        show(&format!("epoch {} lr {:.5} ({:.0} s)", ep + 1, lr, t0.elapsed().as_secs_f64()), &r);
        let sc_now = score(&r);
        if sc_now < best.0 - 0.0005 {
            best = (sc_now, bias.clone(), g.clone(), td.clone(), tor.clone(), ep + 1);
            bad_epochs = 0;
        } else {
            bad_epochs += 1;
            if bad_epochs > patience {
                println!("early stop: no val improvement for {} epochs (best epoch {})", bad_epochs, best.5);
                break;
            }
        }
    }
    println!("best epoch {} (val score {:.4} = games CE + kata CE)", best.5, best.0);
    if let Some(p) = arg(args, "--out-bin") {
        let mut o = if k3c { b"TPOL3\0\0\0".to_vec() } else if use_pat { b"TPOL2\0\0\0".to_vec() } else { b"TPOL1\0\0\0".to_vec() };
        o.extend_from_slice(&0u32.to_le_bytes());
        for v in best.1.iter().chain(best.2.iter()).chain(best.3.iter()).chain(best.4.iter()) {
            o.extend_from_slice(&v.to_le_bytes());
        }
        match fs::write(&p, &o) {
            Ok(()) => println!("wrote {p} ({} B, best epoch {})", o.len(), best.5),
            Err(e) => {
                eprintln!("{p}: {e}");
                return ExitCode::from(1);
            }
        }
    }
    let _ = k3code; // shared definition lives in policy.rs
    ExitCode::SUCCESS
}

//! `ti policy`: step P0 of the engine repo's docs/research/policy/POLICY_PLAN.md. Fits a move-ordering policy head on the frozen
//! activations of a net, with the moves already played in gate games (the engine's own 5k-node best move) as free labels.
//!
//! logit(m) = bias[slot] + sum(scalar move features) + U[slot] . h     (slot = to-square x clone/jump, 98 slots)
//! Softmax over the legal moves (clone moves deduplicated per destination), cross entropy on the played move.
//! No random init (LAWS C14): bias starts at ln((chosen+1)/(offered+1)) per slot, everything else at zero.
//! Reports top-1 / top-3 / CE on a held-out split (by game pair) against uniform and greedy-capture baselines.

use crate::games;
use crate::train::TrainableNet;
use std::fs;
use std::io::{BufRead, BufReader};
use std::process::ExitCode;
use titanium::{Board, Move, MoveList};

pub(crate) const SLOTS: usize = 98;
pub(crate) const NSCAL: usize = 9 * 2 + 9 * 2 + 9; // flips[clone], to-holes[clone], from-holes (jump only)

struct Sample {
    moves: Vec<[u8; 5]>, // slot, flip idx, to-hole idx, from-hole idx (255 none), unused
    pat: Vec<[u32; 2]>,  // D486 pattern rows: dest table index, origin table index (u32::MAX none); empty without --pat
    label: usize, // usize::MAX for a soft (Kata) sample
    h: usize, // offset into the feature pool
    /// D517 Kata soft targets: kg[i] = group of move i (-1 = outside the support), ky[g] = target mass of group g
    kg: Vec<i8>,
    ky: Vec<f32>,
}

/// pattern rows of move m (same rule as the game loader)
pub(crate) fn patidx(b: &Board, m: Move, k3c: bool) -> [u32; 2] {
    let cls = |q: u8| -> u32 {
        let f = |x: u8| if x == 0 { 0 } else if x == 6 { 2 } else { 1 };
        f(q / 7) * 3 + f(q % 7)
    };
    let slot = if k3c { cls(m.to) * 2 + (!m.is_clone()) as u32 } else { m.to as u32 + 49 * (!m.is_clone()) as u32 };
    let d = slot * P3 as u32 + k3code(b.me(), b.opp(), m.to);
    let osl = if k3c { cls(m.from) } else { m.from as u32 };
    let o = if m.is_clone() { NONE } else { osl * P3 as u32 + k3code(b.me(), b.opp(), m.from) };
    [d, o]
}

pub(crate) fn npz_files(dir: &std::path::Path, out: &mut Vec<std::path::PathBuf>) {
    if let Ok(rd) = fs::read_dir(dir) {
        for e in rd.flatten() {
            let p = e.path();
            if p.is_dir() {
                npz_files(&p, out);
            } else if p.extension().map(|x| x == "npz").unwrap_or(false) {
                out.push(p);
            }
        }
    }
}

pub(crate) const P3: usize = 6561; // 3^8 codes of the 8 cells around a square
pub(crate) const NONE: u32 = u32::MAX;

/// D486: base-3 code of the 8 neighbours of `sq` from the mover's view (0 empty or off-board, 1 mover, 2 opponent),
/// fixed order NW N NE W E SW S SE (row-major 7x7 squares).
pub(crate) fn k3code(me: u64, opp: u64, sq: u8) -> u32 {
    let (r, c) = ((sq / 7) as i32, (sq % 7) as i32);
    let mut code = 0u32;
    for (dr, dc) in [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)] {
        let (rr, cc) = (r + dr, c + dc);
        let d = if (0..7).contains(&rr) && (0..7).contains(&cc) {
            let bit = 1u64 << (rr * 7 + cc);
            if me & bit != 0 {
                1
            } else if opp & bit != 0 {
                2
            } else {
                0
            }
        } else {
            0
        };
        code = code * 3 + d;
    }
    code
}

pub(crate) fn ring1() -> [u64; 49] {
    let mut r = [0u64; 49];
    for i in 0..49u8 {
        for j in 0..49u8 {
            if i != j && (Move { from: i, to: j }).is_clone() {
                r[i as usize] |= 1u64 << j;
            }
        }
    }
    r
}

pub(crate) fn arg(args: &[String], key: &str) -> Option<String> {
    args.iter().position(|a| a == key).and_then(|i| args.get(i + 1)).cloned()
}

pub(crate) fn feats(b: &Board, m: Move, ring: &[u64; 49]) -> [u8; 5] {
    let clone = m.is_clone();
    let c = clone as usize;
    let empty = b.empty();
    let flips = (b.captures(m.to) as usize).min(8);
    let tholes = ((ring[m.to as usize] & empty).count_ones() as usize).min(8);
    let slot = m.to as usize + 49 * (!clone) as usize;
    let fh = if clone { 255 } else { (36 + (ring[m.from as usize] & empty).count_ones().min(8)) as u8 };
    [slot as u8, (c * 9 + flips) as u8, (18 + c * 9 + tholes) as u8, fh, 0]
}

pub fn run_policy(args: &[String]) -> ExitCode {
    let net_path = match arg(args, "--net") {
        Some(p) => p,
        None => {
            eprintln!("usage: ti policy --net NET.tnn --games a.jsonl[,b.jsonl..] [--feat none|h|x|hx] [--max-samples 200000] [--epochs 4] [--lr 0.01]");
            return ExitCode::from(2);
        }
    };
    let files: Vec<String> = arg(args, "--games").unwrap_or_default().split(',').filter(|s| !s.is_empty()).map(String::from).collect();
    let feat = arg(args, "--feat").unwrap_or_else(|| "h".into());
    let max_samples: usize = arg(args, "--max-samples").and_then(|v| v.parse().ok()).unwrap_or(200_000);
    let epochs: usize = arg(args, "--epochs").and_then(|v| v.parse().ok()).unwrap_or(4);
    let lr0: f32 = arg(args, "--lr").and_then(|v| v.parse().ok()).unwrap_or(0.01);
    if files.is_empty() {
        eprintln!("--games needed");
        return ExitCode::from(2);
    }
    let net = match fs::read(&net_path).map_err(|e| e.to_string()).and_then(|d| TrainableNet::from_tnn_bytes(&d)) {
        Ok(n) => n,
        Err(e) => {
            eprintln!("net: {e}");
            return ExitCode::from(1);
        }
    };
    let ring = ring1();
    let use_h = feat.contains('h');
    let use_x = feat.contains('x');
    let dim = (if use_h { net.l1 + net.l2 } else { 0 }) + (if use_x { net.din() } else { 0 });
    let per_file = (max_samples / files.len()).max(1);
    // D502: k3 = per-square tables [98][6561] + [49][6561]; k3c = tables shared by the 9 edge classes of a square
    // (row in {0, 1..5, 6} x col in {0, 1..5, 6}): [9 x 2][6561] + [9][6561], ~350 KB as i16 (cache-resident)
    let pat_kind = arg(args, "--pat").unwrap_or_default();
    let use_pat = pat_kind == "k3" || pat_kind == "k3c";
    let k3c = pat_kind == "k3c";
    let (nd_slots, no_slots) = if k3c { (18usize, 9usize) } else { (SLOTS, 49usize) };
    let pat_lr: f32 = arg(args, "--pat-lr").and_then(|v| v.parse().ok()).unwrap_or(lr0);
    let pat_l2: f32 = arg(args, "--pat-l2").and_then(|v| v.parse().ok()).unwrap_or(0.0);
    let init = arg(args, "--init");

    let mut samples: Vec<Sample> = Vec::new();
    let mut pool: Vec<f32> = Vec::new();
    let mut val_mask: Vec<bool> = Vec::new();
    let mut ml = MoveList::new();
    for f in &files {
        let rd = match fs::File::open(f) {
            Ok(x) => BufReader::new(x),
            Err(e) => {
                eprintln!("{f}: {e}");
                continue;
            }
        };
        let mut got = 0usize;
        for line in rd.lines().map_while(Result::ok) {
            if got >= per_file {
                break;
            }
            let g = match games::parse_game(&line) {
                Ok(g) => g,
                Err(_) => continue,
            };
            let boards = match games::replay(&g) {
                Ok(b) => b,
                Err(_) => continue,
            };
            let is_val = (g.pair % 10) == 0;
            for (r, b) in g.moves.iter().zip(&boards) {
                if r.mv == "FORFEIT" {
                    break;
                }
                let pm = match b.parse_uai_move(r.label_mv()) {
                    Some(m) if !m.is_pass() => m,
                    _ => continue,
                };
                b.legal_moves_dedup_into(&mut ml);
                if ml.len() < 2 {
                    continue;
                }
                let mut label = None;
                let mut pv = Vec::new();
                let mut mv = Vec::with_capacity(ml.len());
                for (i, m) in ml.iter().enumerate() {
                    if m.to == pm.to && m.is_clone() == pm.is_clone() && (pm.is_clone() || m.from == pm.from) {
                        label = Some(i);
                    }
                    mv.push(feats(b, m, &ring));
                    if use_pat {
                        let cls = |q: u8| -> u32 {
                            let f = |x: u8| if x == 0 { 0 } else if x == 6 { 2 } else { 1 };
                            f(q / 7) * 3 + f(q % 7)
                        };
                        let slot = if k3c { cls(m.to) * 2 + (!m.is_clone()) as u32 } else { m.to as u32 + 49 * (!m.is_clone()) as u32 };
                        let d = slot * P3 as u32 + k3code(b.me(), b.opp(), m.to);
                        let osl = if k3c { cls(m.from) } else { m.from as u32 };
                        let o = if m.is_clone() { NONE } else { osl * P3 as u32 + k3code(b.me(), b.opp(), m.from) };
                        pv.push([d, o]);
                    }
                }
                let Some(label) = label else { continue };
                let off = pool.len();
                if dim > 0 {
                    let fw = net.forward_full(b);
                    if use_h {
                        pool.extend_from_slice(&fw.h1);
                        pool.extend_from_slice(&fw.h2);
                    }
                    if use_x {
                        pool.extend_from_slice(&fw.xin);
                    }
                }
                samples.push(Sample { moves: mv, pat: pv, label, h: off, kg: Vec::new(), ky: Vec::new() });
                val_mask.push(is_val);
                got += 1;
            }
        }
        eprintln!("{f}: {got} samples");
    }
    // D517: KataAtaxx self-play tdata (200-visit search policy). Rows: binaryInputNCHWPacked 22 x 7 bytes (ch1 = side to
    // move, ch2 = opponent, ch4 = the stone picked for a jump = stage 1), policyTargetsNCMove[0] = visits per square (+pass).
    // Stage 0: groups = clone destinations and jump ORIGINS (all legal moves); stage 1: groups = destinations of jumps from
    // the picked stone (support = those jumps only). Soft loss -sum_g y_g ln(sum_{m in g} q_m), q = softmax over the support.
    let kata_dirs: Vec<String> = arg(args, "--kata").unwrap_or_default().split(',').filter(|s| !s.is_empty()).map(String::from).collect();
    let kata_max: usize = arg(args, "--kata-max").and_then(|v| v.parse().ok()).unwrap_or(usize::MAX);
    let kata_w: f32 = arg(args, "--kata-w").and_then(|v| v.parse().ok()).unwrap_or(1.0);
    let (mut nk0, mut nk1) = (0usize, 0usize);
    if !kata_dirs.is_empty() {
        if dim != 0 {
            eprintln!("--kata needs --feat none (dim 0)");
            return ExitCode::from(2);
        }
        let mut kf: Vec<std::path::PathBuf> = Vec::new();
        for d in &kata_dirs {
            npz_files(std::path::Path::new(d), &mut kf);
        }
        kf.sort();
        let (mut bb, mut pb) = (Vec::new(), Vec::new());
        'files: for (fi, fp) in kf.iter().enumerate() {
            let is_val = fi % 10 == 0;
            let path = fp.to_string_lossy().to_string();
            let z = match crate::data::Npz::open(&path) {
                Ok(z) => z,
                Err(e) => {
                    eprintln!("{path}: {e}");
                    continue;
                }
            };
            let (mut cb, mut cp) = match (z.col("binaryInputNCHWPacked"), z.col("policyTargetsNCMove")) {
                (Ok(a), Ok(b)) => (a, b),
                _ => {
                    eprintln!("{path}: missing columns");
                    continue;
                }
            };
            if cb.width * cb.elem != 154 || cp.width * cp.elem != 200 {
                eprintln!("{path}: unexpected layout ({} / {} bytes per row)", cb.width * cb.elem, cp.width * cp.elem);
                continue;
            }
            let n = cb.rows;
            if cb.read(n, &mut bb).is_err() || cp.read(n, &mut pb).is_err() {
                eprintln!("{path}: read error");
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
                let vis = |i: usize| -> f32 { i16::from_le_bytes([pb[r * 200 + 2 * i], pb[r * 200 + 2 * i + 1]]).max(0) as f32 };
                let b = Board::from_occ(own, opp, 0, 0);
                b.legal_moves_dedup_into(&mut ml);
                if ml.len() < 2 {
                    continue;
                }
                let stage1 = sel != 0;
                let ssq = sel.trailing_zeros() as u8;
                let (mut kg, mut keys, mut ky): (Vec<i8>, Vec<u8>, Vec<f32>) = (Vec::with_capacity(ml.len()), Vec::new(), Vec::new());
                for m in ml.iter() {
                    let key = if stage1 {
                        if !m.is_clone() && m.from == ssq { Some(m.to) } else { None }
                    } else if m.is_clone() {
                        Some(m.to)
                    } else {
                        Some(m.from)
                    };
                    match key {
                        None => kg.push(-1),
                        Some(k) => {
                            let gi = match keys.iter().position(|&x| x == k) {
                                Some(p) => p,
                                None => {
                                    keys.push(k);
                                    ky.push(vis(k as usize));
                                    keys.len() - 1
                                }
                            };
                            kg.push(gi as i8);
                        }
                    }
                }
                let tot: f32 = ky.iter().sum();
                if keys.len() < 2 || tot <= 0.0 {
                    continue;
                }
                for y in ky.iter_mut() {
                    *y /= tot;
                }
                let mv: Vec<[u8; 5]> = ml.iter().map(|m| feats(&b, m, &ring)).collect();
                let pv: Vec<[u32; 2]> = if use_pat { ml.iter().map(|m| patidx(&b, m, k3c)).collect() } else { Vec::new() };
                samples.push(Sample { moves: mv, pat: pv, label: usize::MAX, h: 0, kg, ky });
                val_mask.push(is_val);
                if stage1 {
                    nk1 += 1;
                } else {
                    nk0 += 1;
                }
                if nk0 + nk1 >= kata_max {
                    break 'files;
                }
            }
        }
        println!("kata: {} files, {} stage-0 + {} stage-1 samples (val = every 10th file)", kf.len(), nk0, nk1);
    }
    let n_val = val_mask.iter().filter(|v| **v).count();
    println!("samples {} (train {} val {}), feat '{}' dim {}", samples.len(), samples.len() - n_val, n_val, feat, dim);
    if n_val == 0 || n_val == samples.len() {
        eprintln!("need both train and val samples");
        return ExitCode::from(1);
    }

    // parameters
    let mut bias = vec![0f32; SLOTS];
    let mut g = vec![0f32; NSCAL];
    let mut u = vec![0f32; SLOTS * dim];
    {
        let (mut chosen, mut offered) = ([0f64; SLOTS], [0f64; SLOTS]);
        for (s, v) in samples.iter().zip(&val_mask) {
            if *v || !s.ky.is_empty() {
                continue;
            }
            for (i, m) in s.moves.iter().enumerate() {
                offered[m[0] as usize] += 1.0;
                if i == s.label {
                    chosen[m[0] as usize] += 1.0;
                }
            }
        }
        for k in 0..SLOTS {
            bias[k] = (((chosen[k] + 1.0) / (offered[k] + 1.0)) as f32).ln();
        }
    }
    let mut init_tabs: Option<Vec<f32>> = None;
    // D486: warm start from a trained TPOL1 head (LAWS C14); pattern tables start at 0 => step-0 parity with it
    if let Some(ip) = &init {
        let d = match fs::read(ip) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("{ip}: {e}");
                return ExitCode::from(1);
            }
        };
        let idim = u32::from_le_bytes(d[8..12].try_into().unwrap()) as usize;
        let v3 = &d[0..5] == b"TPOL3";
        let v2 = &d[0..5] == b"TPOL2" || v3;
        let n1 = SLOTS + NSCAL + SLOTS * idim;
        let ntab = if v3 { 27 * P3 } else { (SLOTS + 49) * P3 };
        let ok_len = d.len() == 12 + 4 * (n1 + if v2 { ntab } else { 0 });
        if v2 && (v3 != k3c) {
            eprintln!("--init {ip}: table kind does not match --pat (TPOL3 needs --pat k3c, TPOL2 --pat k3)");
            return ExitCode::from(2);
        }
        // D490: dim 0 (--feat none) may start from a dim>0 file: U is DROPPED, exactly as the engine runs a head on a
        // non-pairwise net (policy_xin is empty there)
        if (&d[0..5] != b"TPOL1" && !v2) || !ok_len || (idim != dim && dim != 0) {
            eprintln!("--init {ip}: not a TPOL1/TPOL2 head of dim {dim} (file dim {idim}, {} B)", d.len());
            return ExitCode::from(2);
        }
        let f: Vec<f32> = d[12..].chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
        bias.copy_from_slice(&f[..SLOTS]);
        g.copy_from_slice(&f[SLOTS..SLOTS + NSCAL]);
        if dim == idim {
            u.copy_from_slice(&f[SLOTS + NSCAL..n1]);
        }
        init_tabs = if v2 { Some(f[n1..].to_vec()) } else { None };
        println!("init from {ip} (file dim {idim}, used dim {dim}{}{})", if dim != idim { ", U DROPPED as in the engine" } else { "" }, if v2 { ", TPOL2 tables" } else { "" });
    }
    let mut td = vec![0f32; if use_pat { nd_slots * P3 } else { 0 }];
    let mut tor = vec![0f32; if use_pat { no_slots * P3 } else { 0 }];
    if let (true, Some(t)) = (use_pat, &init_tabs) {
        td.copy_from_slice(&t[..nd_slots * P3]);
        tor.copy_from_slice(&t[nd_slots * P3..]);
    }
    let logits = |s: &Sample, bias: &[f32], g: &[f32], u: &[f32], td: &[f32], tor: &[f32], out: &mut Vec<f32>| {
        out.clear();
        let h = &pool[s.h..s.h + dim];
        for (mi, m) in s.moves.iter().enumerate() {
            let slot = m[0] as usize;
            let mut z = bias[slot] + g[m[1] as usize] + g[m[2] as usize];
            if m[3] != 255 {
                z += g[m[3] as usize];
            }
            if dim > 0 {
                z += u[slot * dim..(slot + 1) * dim].iter().zip(h).map(|(a, b)| a * b).sum::<f32>();
            }
            if !td.is_empty() {
                let [d, o] = s.pat[mi];
                z += td[d as usize];
                if o != NONE {
                    z += tor[o as usize];
                }
            }
            out.push(z);
        }
    };
    let eval = |which: bool, bias: &[f32], g: &[f32], u: &[f32], td: &[f32], tor: &[f32]| -> (f64, f64, f64, f64, f64) {
        let (mut ce, mut t1, mut t3, mut t5, mut n) = (0f64, 0f64, 0f64, 0f64, 0f64);
        let mut z = Vec::new();
        for (s, v) in samples.iter().zip(&val_mask) {
            if *v != which || !s.ky.is_empty() {
                continue;
            }
            logits(s, bias, g, u, td, tor, &mut z);
            let mx = z.iter().cloned().fold(f32::MIN, f32::max);
            let den: f32 = z.iter().map(|x| (x - mx).exp()).sum();
            ce += (den.ln() + mx - z[s.label]) as f64;
            let rank = z.iter().filter(|x| **x > z[s.label]).count();
            t1 += (rank == 0) as u32 as f64;
            t3 += (rank < 3) as u32 as f64;
            t5 += (rank < 5) as u32 as f64;
            n += 1.0;
        }
        (ce / n, t1 / n, t3 / n, t5 / n, n)
    };
    // soft (Kata) metrics: CE to the visit distribution over groups, top-1 = our best group is Kata's most visited
    let qgroups = |s: &Sample, z: &[f32], qg: &mut Vec<f32>| {
        let mx = z.iter().zip(&s.kg).filter(|(_, g)| **g >= 0).map(|(x, _)| *x).fold(f32::MIN, f32::max);
        qg.clear();
        qg.resize(s.ky.len(), 0.0);
        let mut den = 0f32;
        for (x, g) in z.iter().zip(&s.kg) {
            if *g >= 0 {
                let e = (x - mx).exp();
                qg[*g as usize] += e;
                den += e;
            }
        }
        for q in qg.iter_mut() {
            *q /= den;
        }
    };
    let eval_soft = |which: bool, bias: &[f32], g: &[f32], u: &[f32], td: &[f32], tor: &[f32]| -> (f64, f64, f64) {
        let (mut ce, mut t1, mut n) = (0f64, 0f64, 0f64);
        let (mut z, mut qg) = (Vec::new(), Vec::new());
        for (s, v) in samples.iter().zip(&val_mask) {
            if *v != which || s.ky.is_empty() {
                continue;
            }
            logits(s, bias, g, u, td, tor, &mut z);
            qgroups(s, &z, &mut qg);
            for (y, q) in s.ky.iter().zip(&qg) {
                if *y > 0.0 {
                    ce -= (*y as f64) * (q.max(1e-9) as f64).ln();
                }
            }
            let am = |v: &[f32]| (0..v.len()).max_by(|&a, &b| v[a].partial_cmp(&v[b]).unwrap()).unwrap();
            t1 += (am(&qg) == am(&s.ky)) as u32 as f64;
            n += 1.0;
        }
        (ce / n.max(1.0), t1 / n.max(1.0), n)
    };

    // baselines on val
    {
        let (mut uce, mut ut1, mut ut3, mut gt1, mut n) = (0f64, 0f64, 0f64, 0f64, 0f64);
        for (s, v) in samples.iter().zip(&val_mask) {
            if !*v || !s.ky.is_empty() {
                continue;
            }
            let k = s.moves.len() as f64;
            uce += k.ln();
            ut1 += 1.0 / k;
            ut3 += (3.0f64).min(k) / k;
            // greedy capture: slot-free flips index = m[1] % 9
            let fl = |m: &[u8; 5]| (m[1] % 9) as i32;
            let best = s.moves.iter().map(fl).max().unwrap();
            let tied = s.moves.iter().filter(|m| fl(m) == best).count() as f64;
            if fl(&s.moves[s.label]) == best {
                gt1 += 1.0 / tied;
            }
            n += 1.0;
        }
        println!("val baselines: uniform CE {:.4} top1 {:.4} top3 {:.4} | greedy-capture top1 {:.4}", uce / n, ut1 / n, ut3 / n, gt1 / n);
    }
    let (ce, t1, t3, t5, _) = eval(true, &bias, &g, &u, &td, &tor);
    println!("val  init ({}): CE {:.4} top1 {:.4} top3 {:.4} top5 {:.4}", init.as_deref().unwrap_or("log-freq slot bias only"), ce, t1, t3, t5);
    if nk0 + nk1 > 0 {
        let (kce, kt1, kn) = eval_soft(true, &bias, &g, &u, &td, &tor);
        println!("kata val init: soft CE {:.4} top1 (most-visited group) {:.4} ({} samples)", kce, kt1, kn);
    }

    // Adam
    let mut tr: Vec<usize> = (0..samples.len()).filter(|i| !val_mask[*i]).collect();
    // D517: deterministic shuffle (games and Kata rows must mix inside a batch)
    {
        let mut x: u64 = 0x9E37_79B9_7F4A_7C15;
        for i in (1..tr.len()).rev() {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            tr.swap(i, (x % (i as u64 + 1)) as usize);
        }
    }
    let mut dz: Vec<f32> = Vec::new();
    let mut qg: Vec<f32> = Vec::new();
    let (b1, b2, eps) = (0.9f32, 0.999f32, 1e-8f32);
    let mut ms = (vec![0f32; SLOTS], vec![0f32; NSCAL], vec![0f32; u.len()]);
    let (mut mtd, mut vtd, mut mto, mut vto) = (vec![0f32; td.len()], vec![0f32; td.len()], vec![0f32; tor.len()], vec![0f32; tor.len()]);
    let mut gtd = vec![0f32; td.len()];
    let mut gto = vec![0f32; tor.len()];
    let mut touched: Vec<u32> = Vec::new();
    let mut touched_o: Vec<u32> = Vec::new();
    let mut vs = ms.clone();
    let mut t = 0i32;
    let mut z = Vec::new();
    let bs = 256;
    for ep in 0..epochs {
        let lr = lr0 * (1.0 - ep as f32 / epochs as f32).max(0.1);
        let mut gb = vec![0f32; SLOTS];
        let mut gg = vec![0f32; NSCAL];
        let mut gu = vec![0f32; u.len()];
        for (bi, chunk) in tr.chunks(bs).enumerate() {
            for x in gb.iter_mut().chain(gg.iter_mut()).chain(gu.iter_mut()) {
                *x = 0.0;
            }
            for &si in chunk {
                let s = &samples[si];
                logits(s, &bias, &g, &u, &td, &tor, &mut z);
                dz.clear();
                if s.ky.is_empty() {
                    let mx = z.iter().cloned().fold(f32::MIN, f32::max);
                    let mut den = 0f32;
                    for x in z.iter_mut() {
                        *x = (*x - mx).exp();
                        den += *x;
                    }
                    for (i, x) in z.iter().enumerate() {
                        dz.push(x / den - (i == s.label) as u32 as f32);
                    }
                } else {
                    // soft 2-stage loss: dL/dz_i = q_i - y_g q_i / Q_g on the support, 0 outside
                    qgroups(s, &z, &mut qg);
                    let mx = z.iter().zip(&s.kg).filter(|(_, g)| **g >= 0).map(|(x, _)| *x).fold(f32::MIN, f32::max);
                    let den: f32 = z.iter().zip(&s.kg).filter(|(_, g)| **g >= 0).map(|(x, _)| (x - mx).exp()).sum();
                    for (x, gi) in z.iter().zip(&s.kg) {
                        if *gi < 0 {
                            dz.push(0.0);
                        } else {
                            let q = (x - mx).exp() / den;
                            let gq = qg[*gi as usize].max(1e-9);
                            dz.push(kata_w * (q - s.ky[*gi as usize] * q / gq));
                        }
                    }
                }
                let h = &pool[s.h..s.h + dim];
                for (i, m) in s.moves.iter().enumerate() {
                    if dz[i] == 0.0 {
                        continue;
                    }
                    let d = dz[i] / chunk.len() as f32;
                    let slot = m[0] as usize;
                    gb[slot] += d;
                    gg[m[1] as usize] += d;
                    gg[m[2] as usize] += d;
                    if m[3] != 255 {
                        gg[m[3] as usize] += d;
                    }
                    if dim > 0 {
                        for (a, x) in gu[slot * dim..(slot + 1) * dim].iter_mut().zip(h) {
                            *a += d * x;
                        }
                    }
                    if use_pat {
                        let [pd, po] = s.pat[i];
                        touched.push(pd);
                        gtd[pd as usize] += d;
                        if po != NONE {
                            touched_o.push(po);
                            gto[po as usize] += d;
                        }
                    }
                }
            }
            t += 1;
            let c1 = 1.0 - b1.powi(t);
            let c2 = 1.0 - b2.powi(t);
            let step = |p: &mut [f32], gr: &[f32], m: &mut [f32], v: &mut [f32]| {
                for i in 0..p.len() {
                    m[i] = b1 * m[i] + (1.0 - b1) * gr[i];
                    v[i] = b2 * v[i] + (1.0 - b2) * gr[i] * gr[i];
                    p[i] -= lr * (m[i] / c1) / ((v[i] / c2).sqrt() + eps);
                }
            };
            step(&mut bias, &gb, &mut ms.0, &mut vs.0);
            step(&mut g, &gg, &mut ms.1, &mut vs.1);
            if dim > 0 {
                step(&mut u, &gu, &mut ms.2, &mut vs.2);
            }
            // pattern tables: lazy (sparse) Adam on the touched rows only, then clear their gradients
            if use_pat {
                for (tab, gr, m, v, list) in [(&mut td, &mut gtd, &mut mtd, &mut vtd, &mut touched), (&mut tor, &mut gto, &mut mto, &mut vto, &mut touched_o)] {
                    list.sort_unstable();
                    list.dedup();
                    for &k in list.iter() {
                        let k = k as usize;
                        let gk = gr[k] + pat_l2 * tab[k];
                        m[k] = b1 * m[k] + (1.0 - b1) * gk;
                        v[k] = b2 * v[k] + (1.0 - b2) * gk * gk;
                        tab[k] -= pat_lr * (lr / lr0) * (m[k] / c1) / ((v[k] / c2).sqrt() + eps);
                        gr[k] = 0.0;
                    }
                    list.clear();
                }
            }
            if bi % 200 == 199 {
                eprint!(".");
            }
        }
        let (tce, tt1, _, _, _) = eval(false, &bias, &g, &u, &td, &tor);
        let (vce, vt1, vt3, vt5, _) = eval(true, &bias, &g, &u, &td, &tor);
        println!("epoch {} lr {:.4}: train CE {:.4} top1 {:.4} | val CE {:.4} top1 {:.4} top3 {:.4} top5 {:.4}", ep + 1, lr, tce, tt1, vce, vt1, vt3, vt5);
        if nk0 + nk1 > 0 {
            let (kce, kt1, _) = eval_soft(true, &bias, &g, &u, &td, &tor);
            println!("         kata val: soft CE {:.4} top1 {:.4}", kce, kt1);
        }
    }
    if let Some(p) = arg(args, "--out-bin") {
        if feat != "x" && dim != 0 {
            eprintln!("--out-bin needs --feat x or none (x = the pairwise accumulator product; none = dim 0, what the engine runs on non-pairwise nets, D490)");
            return ExitCode::from(2);
        }
        // TPOL1 = bias, g, u; TPOL2 (D486) = the same + dest table [98][6561] + origin table [49][6561]
        // TPOL3 (D502) = TPOL1 + class-shared dest [18][6561] + origin [9][6561]
        let mut o = if k3c { b"TPOL3\0\0\0".to_vec() } else if use_pat { b"TPOL2\0\0\0".to_vec() } else { b"TPOL1\0\0\0".to_vec() };
        o.extend_from_slice(&(dim as u32).to_le_bytes());
        for v in bias.iter().chain(g.iter()).chain(u.iter()).chain(td.iter()).chain(tor.iter()) {
            o.extend_from_slice(&v.to_le_bytes());
        }
        match fs::write(&p, &o) {
            Ok(()) => println!("wrote {p} ({} B)", o.len()),
            Err(e) => {
                eprintln!("{p}: {e}");
                return ExitCode::from(1);
            }
        }
    }
    ExitCode::SUCCESS
}

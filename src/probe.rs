//! `ti train probe --net n.tnn --data <stem> [--rows 60000]`: what does the net look at, and what does it crave?
//!  A. WHERE THE WEIGHT SITS: first-layer weight mass per input group (G1 square state, G2 anchor, G3 stone ring, G5 exchange,
//!     G6 globals), weighted by how often each feature is active, plus a 7x7 map of the square-state (G1) mass.
//!  B. WHAT IT CRAVES: the residual (target - net prediction) regressed on features the net does NOT receive (mobility, stuck
//!     flags, best capture, empty-region structure, territory). Held-out R^2 per phase bucket: if a set explains a real share of
//!     the residual, the net is missing that information; ~0 means the information is already inside or not linearly useful.
use std::process::ExitCode;

use crate::mapper;
use crate::targets::target_of;
use crate::train::TrainableNet;
use crate::trainer::{load_val, Tdat};
use titanium::board::Board;
use titanium::nnfeat::{G2_OFF, G3_OFF, G5_OFF, G6_OFF, NUM_FEATS, SQ_STATES};

const NC: usize = 26;
const NAMES: [&str; NC] = [
    "mob_us", "mob_them", "stuck_us", "stuck_them", "cap_us", "cap_them", "regions", "largest_region", "odd_regions", "terr_us",
    "terr_them", "contested", "empties", "stone_diff", "perm_us", "perm_them", "safe_us", "safe_them", "exposed_us", "exposed_them",
    "pst_us", "pst_them", "corner_us", "corner_them", "edge_us", "edge_them",
];

const FULL: u64 = (1u64 << 49) - 1;
const NOT_C0: u64 = {
    let mut m = 0u64;
    let mut s = 0;
    while s < 49 {
        if s % 7 != 0 {
            m |= 1 << s;
        }
        s += 1;
    }
    m
};
const NOT_C6: u64 = {
    let mut m = 0u64;
    let mut s = 0;
    while s < 49 {
        if s % 7 != 6 {
            m |= 1 << s;
        }
        s += 1;
    }
    m
};

/// king-step dilation on the 7x7 grid (symmetric, so independent of row/column bit order)
fn dil(x: u64) -> u64 {
    (x | ((x << 1) & NOT_C0) | ((x >> 1) & NOT_C6) | (x << 7) | (x >> 7) | ((x << 8) & NOT_C0) | ((x << 6) & NOT_C6) | ((x >> 6) & NOT_C0) | ((x >> 8) & NOT_C6)) & FULL
}

/// square-value table: Chebyshev distance from the centre (0..3), the 4 corners, the outer ring
const fn sq_masks() -> ([u64; 4], u64, u64) {
    let (mut ring, mut corner, mut edge) = ([0u64; 4], 0u64, 0u64);
    let mut s = 0;
    while s < 49 {
        let (r, c) = ((s / 7) as i32, (s % 7) as i32);
        let (dr, dc) = ((r - 3).abs(), (c - 3).abs());
        let d = if dr > dc { dr } else { dc };
        ring[d as usize] |= 1 << s;
        if dr == 3 && dc == 3 {
            corner |= 1 << s;
        }
        if d == 3 {
            edge |= 1 << s;
        }
        s += 1;
    }
    (ring, corner, edge)
}
const SQM: ([u64; 4], u64, u64) = sq_masks();
fn pst(x: u64) -> f64 {
    (0..4).map(|d| d as f64 * (x & SQM.0[d]).count_ones() as f64).sum()
}

fn mobility_and_capture(b: &Board) -> (f64, f64) {
    let ml = b.legal_moves();
    let (mut tgt, mut best) = (0u64, 0i32);
    let before = b.piece_cnt[b.turn as usize] as i32;
    for m in ml.iter() {
        tgt |= 1u64 << m.to;
        let c = b.make(m);
        let delta = c.piece_cnt[b.turn as usize] as i32 - before;
        best = best.max(delta - if m.is_clone() { 1 } else { 0 });
    }
    (tgt.count_ones() as f64, best as f64)
}

fn cand_features(b: &Board) -> [f64; NC] {
    let (mu, cu) = mobility_and_capture(b);
    let flipped = Board::from_occ(b.occ[0], b.occ[1], 0, b.turn ^ 1);
    let (mt, ct) = mobility_and_capture(&flipped);
    let us = b.occ[b.turn as usize];
    let them = b.occ[(b.turn ^ 1) as usize];
    let empty = !(us | them) & FULL;
    // empty regions (king-connected components)
    let (mut regions, mut largest, mut odd) = (0.0, 0.0f64, 0.0);
    let mut left = empty;
    while left != 0 {
        let mut comp = left & left.wrapping_neg();
        loop {
            let nx = dil(comp) & empty;
            if nx == comp {
                break;
            }
            comp = nx;
        }
        let n = comp.count_ones() as f64;
        regions += 1.0;
        largest = largest.max(n);
        if (n as u32) % 2 == 1 {
            odd += 1.0;
        }
        left &= !comp;
    }
    // territory: empties within distance 2 of only one side's stones
    let reach = |s: u64| dil(dil(s)) & empty;
    let (ru, rt) = (reach(us), reach(them));
    // stone safety: permanent = no empty neighbour (can never be flipped); exposed = a neighbouring empty is a landing square of
    // the enemy (can be flipped next move); safe = neither permanent nor exposed-now but still flippable later is NOT counted here
    let near_empty = dil(empty);
    let (perm_us, perm_them) = (us & !near_empty, them & !near_empty);
    let (exp_us, exp_them) = (us & dil(rt), them & dil(ru));
    let (safe_us, safe_them) = (us & !exp_us, them & !exp_them);
    [
        mu,
        mt,
        (mu == 0.0) as u8 as f64,
        (mt == 0.0) as u8 as f64,
        cu,
        ct,
        regions,
        largest,
        odd,
        (ru & !rt).count_ones() as f64,
        (rt & !ru).count_ones() as f64,
        (ru & rt).count_ones() as f64,
        empty.count_ones() as f64,
        us.count_ones() as f64 - them.count_ones() as f64,
        perm_us.count_ones() as f64,
        perm_them.count_ones() as f64,
        safe_us.count_ones() as f64,
        safe_them.count_ones() as f64,
        exp_us.count_ones() as f64,
        exp_them.count_ones() as f64,
        pst(us),
        pst(them),
        (us & SQM.1).count_ones() as f64,
        (them & SQM.1).count_ones() as f64,
        (us & SQM.2).count_ones() as f64,
        (them & SQM.2).count_ones() as f64,
    ]
}

/// ridge fit on `tr`, R^2 on `te` of the residual using feature columns `cols`
fn r2(cols: &[usize], x: &[[f64; NC]], y: &[f64], tr: &[usize], te: &[usize]) -> f64 {
    if tr.len() < 50 || te.len() < 50 || cols.is_empty() {
        return f64::NAN;
    }
    let k = cols.len();
    let (mut mu, mut sd) = (vec![0.0; k], vec![0.0; k]);
    for (j, &c) in cols.iter().enumerate() {
        mu[j] = tr.iter().map(|&i| x[i][c]).sum::<f64>() / tr.len() as f64;
        sd[j] = (tr.iter().map(|&i| (x[i][c] - mu[j]).powi(2)).sum::<f64>() / tr.len() as f64).sqrt().max(1e-9);
    }
    let ym = tr.iter().map(|&i| y[i]).sum::<f64>() / tr.len() as f64;
    let z = |i: usize, j: usize| (x[i][cols[j]] - mu[j]) / sd[j];
    let mut a = vec![vec![0.0; k]; k];
    let mut v = vec![0.0; k];
    for &i in tr {
        for p in 0..k {
            v[p] += z(i, p) * (y[i] - ym);
            for q in 0..k {
                a[p][q] += z(i, p) * z(i, q);
            }
        }
    }
    let w = crate::export::solve(&a, &v, 1.0);
    let (mut sse, mut sst) = (0.0, 0.0);
    for &i in te {
        let pred = ym + (0..k).map(|j| w[j] * z(i, j)).sum::<f64>();
        sse += (y[i] - pred).powi(2);
        sst += (y[i] - ym).powi(2);
    }
    1.0 - sse / sst.max(1e-18)
}

pub fn run_probe(args: &[String]) -> ExitCode {
    let f = |n: &str| crate::flag_str(args, n);
    let (Some(net_p), Some(data)) = (f("--net"), f("--data")) else {
        eprintln!("usage: ti train probe --net <n.tnn> --data <stem> [--rows 60000]");
        return ExitCode::FAILURE;
    };
    let rows = crate::flag_u64(args, "--rows").unwrap_or(60000) as usize;
    let net = match std::fs::read(&net_p).map_err(|e| e.to_string()).and_then(|b| TrainableNet::from_tnn_bytes(&b)) {
        Ok(n) => n,
        Err(e) => {
            eprintln!("probe: {e}");
            return ExitCode::FAILURE;
        }
    };
    let v = match Tdat::open(&data).and_then(|mut t| load_val(&mut t, rows)) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("probe: {e}");
            return ExitCode::FAILURE;
        }
    };
    println!("net {net_p}: W={} buckets={}, {} rows of {data}\n", net.width, net.buckets, v.len());

    // ---- A. weight mass per input group, weighted by activation frequency
    let w = net.width;
    let mut freq = vec![0f64; NUM_FEATS];
    for r in &v {
        let s = mapper::sample(r, 0, net.buckets, net.groups >> 8);
        for &f in s.us.iter().chain(s.them.iter()) {
            freq[f as usize] += 1.0;
        }
    }
    let norm2: Vec<f64> = (0..NUM_FEATS).map(|f| net.ft[f * w..(f + 1) * w].iter().map(|&x| (x as f64).powi(2)).sum()).collect();
    let groups = [("G1 square state", 0, G2_OFF), ("G2 anchor", G2_OFF, G3_OFF), ("G3 stone ring", G3_OFF, G5_OFF), ("G5 2-ply exchange", G5_OFF, G6_OFF), ("G6 globals", G6_OFF, NUM_FEATS)];
    let total: f64 = (0..NUM_FEATS).map(|f| freq[f] * norm2[f]).sum();
    println!("A. WHERE THE FIRST-LAYER WEIGHT SITS (importance = activity x |w|^2)");
    println!("   group               features   active/pos   mean |w|    importance share");
    for (name, lo, hi) in groups {
        let act: f64 = (lo..hi).map(|f| freq[f]).sum::<f64>() / (2.0 * v.len() as f64);
        let imp: f64 = (lo..hi).map(|f| freq[f] * norm2[f]).sum();
        let mean_w = (lo..hi).map(|f| norm2[f].sqrt()).sum::<f64>() / (hi - lo) as f64;
        println!("   {name:<20}{:>8}   {act:>9.1}   {mean_w:>8.3}    {:>6.1}%", hi - lo, 100.0 * imp / total.max(1e-18));
    }
    let mut sq = [0f64; 49];
    for s in 0..49 {
        for st in 0..SQ_STATES {
            let f = s * SQ_STATES + st;
            sq[s] += freq[f] * norm2[f];
        }
    }
    let sqt: f64 = sq.iter().sum();
    println!("\n   G1 square map (share of G1 importance per square, %; rows of the 7x7 board):");
    for r in 0..7 {
        let line: Vec<String> = (0..7).map(|c| format!("{:>5.1}", 100.0 * sq[r * 7 + c] / sqt.max(1e-18))).collect();
        println!("      {}", line.join(" "));
    }

    // ---- B. residual vs features the net does not receive
    let mut x: Vec<[f64; NC]> = Vec::with_capacity(v.len());
    let mut y: Vec<f64> = Vec::with_capacity(v.len());
    let mut bk: Vec<usize> = Vec::with_capacity(v.len());
    for r in &v {
        let b = mapper::board_of(r, 0);
        let fw = net.forward_full(&b);
        let p = 1.0 / (1.0 + (-fw.y as f64).exp());
        y.push(target_of(r).target as f64 - p);
        bk.push(fw.bucket);
        x.push(cand_features(&b));
    }
    let sets: [(&str, Vec<usize>); 6] = [
        ("mobility + stuck", vec![0, 1, 2, 3]),
        ("best capture (tactics)", vec![4, 5]),
        ("regions + territory", vec![6, 7, 8, 9, 10, 11]),
        ("stone safety (permanent/safe/exposed)", vec![14, 15, 16, 17, 18, 19]),
        ("ALL candidates", (0..NC).collect()),
        ("reference: empties+stone diff (net has G6)", vec![12, 13]),
    ];
    println!("\nB. WHAT THE NET CRAVES: held-out R^2 of the residual (target - prediction) from features it does NOT receive");
    print!("   {:<44}{:>8}", "feature set", "all");
    for b in 0..net.buckets {
        print!("{:>8}", format!("bkt{b}"));
    }
    println!();
    for (name, cols) in &sets {
        let idx: Vec<usize> = (0..v.len()).collect();
        let (tr, te): (Vec<usize>, Vec<usize>) = (idx.iter().cloned().filter(|i| i % 2 == 0).collect(), idx.iter().cloned().filter(|i| i % 2 == 1).collect());
        print!("   {name:<44}{:>7.2}%", 100.0 * r2(cols, &x, &y, &tr, &te));
        for b in 0..net.buckets {
            let tb: Vec<usize> = tr.iter().cloned().filter(|&i| bk[i] == b).collect();
            let eb: Vec<usize> = te.iter().cloned().filter(|&i| bk[i] == b).collect();
            print!("{:>7.2}%", 100.0 * r2(cols, &x, &y, &tb, &eb));
        }
        println!();
    }
    println!("\n   single-feature correlation with the residual (all rows | endgame bucket 0):");
    for c in 0..NC {
        let corr = |ids: &[usize]| -> f64 {
            let n = ids.len() as f64;
            if n < 50.0 {
                return f64::NAN;
            }
            let (mx, my) = (ids.iter().map(|&i| x[i][c]).sum::<f64>() / n, ids.iter().map(|&i| y[i]).sum::<f64>() / n);
            let (mut sxy, mut sxx, mut syy) = (0.0, 0.0, 0.0);
            for &i in ids {
                sxy += (x[i][c] - mx) * (y[i] - my);
                sxx += (x[i][c] - mx).powi(2);
                syy += (y[i] - my).powi(2);
            }
            sxy / (sxx.sqrt() * syy.sqrt()).max(1e-18)
        };
        let all: Vec<usize> = (0..v.len()).collect();
        let b0: Vec<usize> = all.iter().cloned().filter(|&i| bk[i] == 0).collect();
        println!("      {:<16}{:>+8.3} | {:>+8.3}", NAMES[c], corr(&all), corr(&b0));
    }
    ExitCode::SUCCESS
}

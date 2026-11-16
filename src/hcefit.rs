//! `ti hce-fit` (D948): what does the champion NNUE look at? Closed-form ridge fits (exactly saturated, no epochs) of
//! interpretable hand-crafted terms to the bundled champion's static eval, on random train positions and ALL their children.
//!
//!   ti hce-fit --tdat X.tdat [--train 20000] [--val 2000] [--seed 7] [--out logs/hce/run.txt]
//!
//! A) POSITION model: eval(stm) ~ per phase: bias, material, stones per square class (own / enemy, D4-symmetric 10 classes),
//!    hand terms (mobility, landings, private landings, endangered, multi-capture exposure, best capture now / threatened,
//!    interior, edge, corner, frontier, 3+/5+ threats, holes, no-move flags, parity, exact 2-ply exchange both sides,
//!    components). Reported: R2, MAE, sibling best-move agreement vs the NNUE, drop-one ablation of every group.
//! B) MOVE model: delta = value of the move for the mover = -eval(child) - eval(parent), ~ per phase: clone / jump, captures
//!    (one-hot 0..8 per kind), destination class for clones, origin + destination class for jumps, opponent best reply
//!    (exact 2-ply state, one-hot). Printed as 7x7 tables in cp = "piece-square tables for captures and jumps".
use crate::data::{Rec, REC};
use crate::policy::{arg, ring1};
use crate::policy2::{box1, planes_max, victim_planes};
use crate::seescreen::{est, node_ctx};
use std::fs::File;
use std::io::{Read, Seek, SeekFrom, Write};
use std::process::ExitCode;
use titanium::nnue::Nnue;
use titanium::{Board, MoveList};

const NPH: usize = 4;
const CLAMP: f64 = 2000.0;

fn phase(empty: u64) -> usize {
    match empty.count_ones() {
        0..=12 => 0,
        13..=24 => 1,
        25..=36 => 2,
        _ => 3,
    }
}
const PH_NAME: [&str; NPH] = ["empties 0-12", "empties 13-24", "empties 25-36", "empties 37-49"];

fn class(sq: usize) -> usize {
    let (r, c) = (sq / 7, sq % 7);
    let (a, b) = (r.min(6 - r), c.min(6 - c));
    let (a, b) = if a <= b { (a, b) } else { (b, a) };
    // pairs (a<=b) of 0..3: 0:(0,0) 1:(0,1) 2:(0,2) 3:(0,3) 4:(1,1) 5:(1,2) 6:(1,3) 7:(2,2) 8:(2,3) 9:(3,3)
    [[0, 1, 2, 3], [1, 4, 5, 6], [2, 5, 7, 8], [3, 6, 8, 9]][a][b]
}
const NCL: usize = 10;

// ---------------------------------------------------------------- position model
const HAND: [&str; 19] = [
    "clone_targets", "landings", "contested_landings", "endangered", "multicap", "my_best_capture", "opp_best_capture",
    "move_count", "frontier", "threat3", "threat5", "holes_private", "holes_shared", "me_no_moves", "opp_no_moves",
    "parity", "see2_me", "see2_opp", "components",
];
const NH: usize = HAND.len();
// per phase block: bias, material, own class counts, enemy class counts, DYNAMIC PST (user D949: stone value by square
// class x its 3x3 neighbourhood: own stone x own neighbours 0..8, own stone x enemy neighbours, enemy stone x enemy
// neighbours, enemy stone x my neighbours), hand terms
const DYN: usize = 4 * NCL * 9;
const PB: usize = 2 + 2 * NCL + DYN + NH;
const DYN_NAME: [&str; 4] = ["OWN stone x OWN neighbours", "OWN stone x ENEMY neighbours", "ENEMY stone x ENEMY neighbours", "ENEMY stone x OWN neighbours"];

fn edge_mask() -> u64 {
    let mut m = 0u64;
    for sq in 0..49 {
        let (r, c) = (sq / 7, sq % 7);
        if r == 0 || r == 6 || c == 0 || c == 6 {
            m |= 1u64 << sq;
        }
    }
    m
}

#[inline]
fn cnt_at(p: &[u64; 4], sq: usize) -> usize {
    (((p[0] >> sq) & 1) | (((p[1] >> sq) & 1) << 1) | (((p[2] >> sq) & 1) << 2) | (((p[3] >> sq) & 1) << 3)) as usize
}
const PD: usize = NPH * PB;

#[inline]
fn box2(x: u64) -> u64 {
    box1(box1(x))
}

fn components(mut s: u64) -> u32 {
    let mut n = 0;
    while s != 0 {
        let mut c = s & s.wrapping_neg();
        loop {
            let g = box1(c) & s;
            if g == c {
                break;
            }
            c = g;
        }
        s &= !c;
        n += 1;
    }
    n
}

fn ge(p: &[u64; 4], k: u32) -> u64 {
    match k {
        3 => (p[0] & p[1]) | p[2] | p[3],
        5 => (p[2] & (p[0] | p[1])) | p[3],
        _ => unreachable!(),
    }
}

/// best 2-ply exchange for the side to move of `b`: max over its moves of (my gain - the opponent's best reply gain)
fn see2(b: &Board, ring: &[u64; 49], ml: &mut MoveList) -> (f32, f32) {
    b.legal_moves_dedup_into(ml);
    if ml.len() == 0 {
        return (-4.0, 0.0);
    }
    let n = node_ctx(b);
    let mut best = i32::MIN;
    for m in ml.iter() {
        let my = m.is_clone() as i32 + 2 * (b.opp() & ring[m.to as usize]).count_ones() as i32;
        let s = est(&n, ring, m.from, m.to, m.is_clone(), 0);
        let og = if s == 0 { -3 } else { s as i32 - 1 };
        best = best.max(my - og);
    }
    (best as f32, ml.len() as f32)
}

/// side terms of `s` against `o` (empty `e`)
fn side_terms(s: u64, o: u64, e: u64) -> [f32; 15] {
    let lo = box2(o) & e; // where the enemy can land
    let ls = box2(s) & e;
    let ps = victim_planes(s); // count of s stones around every square
    let pe = victim_planes(e);
    let any_e = pe[0] | pe[1] | pe[2] | pe[3];
    let holes = e & !any_e;
    let sum_n: u32 = (0..4).map(|k| (lo & ps[k]).count_ones() << k).sum();
    let touched = (lo & (ps[0] | ps[1] | ps[2] | ps[3])).count_ones();
    let edge = edge_mask();
    let corner = 1u64 | (1 << 6) | (1 << 42) | (1 << 48);
    [
        (box1(s) & e).count_ones() as f32,
        ls.count_ones() as f32,
        (ls & !lo).count_ones() as f32,
        (s & box1(lo)).count_ones() as f32,
        (sum_n - touched) as f32,
        0.0,
        0.0,
        (s & !box1(e)).count_ones() as f32,
        (s & edge).count_ones() as f32,
        (s & corner).count_ones() as f32,
        (s & box1(e)).count_ones() as f32,
        (lo & ge(&ps, 3)).count_ones() as f32,
        (lo & ge(&ps, 5)).count_ones() as f32,
        (holes & box2(s) & !box2(o)).count_ones() as f32,
        components(s) as f32,
    ]
}

fn best_capture(s: u64, o: u64, e: u64) -> f32 {
    let po = victim_planes(o);
    let cl = box1(s) & e;
    let ju = box2(s) & e;
    if ju == 0 {
        return -1.0;
    }
    let c = if cl != 0 { 1 + 2 * planes_max(cl, &po) } else { 0 };
    let j = 2 * planes_max(ju, &po);
    c.max(j) as f32
}

/// sparse position features (index, value)
fn pos_feats(b: &Board, ring: &[u64; 49], ml: &mut MoveList, out: &mut Vec<(u16, f32)>) {
    out.clear();
    let (me, op, e) = (b.me(), b.opp(), b.empty());
    let base = phase(e) * PB;
    out.push((base as u16, 1.0));
    out.push(((base + 1) as u16, me.count_ones() as f32 - op.count_ones() as f32));
    let mut cls = [0f32; 2 * NCL];
    for sq in 0..49 {
        if (me >> sq) & 1 == 1 {
            cls[class(sq)] += 1.0;
        } else if (op >> sq) & 1 == 1 {
            cls[NCL + class(sq)] += 1.0;
        }
    }
    for (i, v) in cls.iter().enumerate() {
        if *v != 0.0 {
            out.push(((base + 2 + i) as u16, *v));
        }
    }
    // dynamic PST: per stone, class x neighbour count (accumulated, then pushed once per index)
    let (pm, po) = (victim_planes(me), victim_planes(op));
    let mut dyn_c = [0f32; DYN];
    for sq in 0..49 {
        let cl = class(sq);
        if (me >> sq) & 1 == 1 {
            dyn_c[cl * 9 + cnt_at(&pm, sq)] += 1.0;
            dyn_c[NCL * 9 + cl * 9 + cnt_at(&po, sq)] += 1.0;
        } else if (op >> sq) & 1 == 1 {
            dyn_c[2 * NCL * 9 + cl * 9 + cnt_at(&po, sq)] += 1.0;
            dyn_c[3 * NCL * 9 + cl * 9 + cnt_at(&pm, sq)] += 1.0;
        }
    }
    let db = base + 2 + 2 * NCL;
    for (i, v) in dyn_c.iter().enumerate() {
        if *v != 0.0 {
            out.push(((db + i) as u16, *v));
        }
    }
    let a = side_terms(me, op, e);
    let z = side_terms(op, me, e);
    let mut h = [0f32; NH];
    // differences me - opp for the side terms (index map into HAND)
    let map = [0usize, 1, 99, 3, 4, 99, 99, 99, 99, 99, 8, 9, 10, 11, 18];
    for (k, &hi) in map.iter().enumerate() {
        if hi != 99 {
            h[hi] = a[k] - z[k];
        }
    }
    let holes_all = {
        let pe = victim_planes(e);
        e & !(pe[0] | pe[1] | pe[2] | pe[3])
    };
    h[12] = (holes_all & box2(me) & box2(op)).count_ones() as f32;
    h[2] = (box2(me) & box2(op) & e).count_ones() as f32;
    h[5] = best_capture(me, op, e);
    h[6] = best_capture(op, me, e);
    h[13] = (box2(me) & e == 0) as u8 as f32;
    h[14] = (box2(op) & e == 0) as u8 as f32;
    h[15] = if e.count_ones() % 2 == 1 { 1.0 } else { -1.0 };
    let (s_me, n_me) = see2(b, ring, ml);
    h[16] = s_me;
    let (s_op, n_op) = see2(&Board::from_occ(op, me, 0, 0), ring, ml);
    h[17] = s_op;
    h[7] = n_me - n_op;
    let hb = base + 2 + 2 * NCL + DYN;
    for (i, v) in h.iter().enumerate() {
        if *v != 0.0 {
            out.push(((hb + i) as u16, *v));
        }
    }
}

// ---------------------------------------------------------------- move model
// per phase block: clone, jump, clone caps 0..8, jump caps 0..8, clone dest class, jump from class, jump dest class,
// opponent best reply state 0..18
const MB: usize = 2 + 9 + 9 + 3 * NCL + 19;
const MD: usize = NPH * MB;

fn move_feats(ph: usize, clone: bool, caps: usize, from: u8, to: u8, opp_state: usize, out: &mut Vec<(u16, f32)>) {
    out.clear();
    let b = ph * MB;
    out.push(((b + if clone { 0 } else { 1 }) as u16, 1.0));
    out.push(((b + 2 + if clone { 0 } else { 9 } + caps.min(8)) as u16, 1.0));
    if clone {
        out.push(((b + 20 + class(to as usize)) as u16, 1.0));
    } else {
        out.push(((b + 20 + NCL + class(from as usize)) as u16, 1.0));
        out.push(((b + 20 + 2 * NCL + class(to as usize)) as u16, 1.0));
    }
    out.push(((b + 20 + 3 * NCL + opp_state.min(18)) as u16, 1.0));
}

// ---------------------------------------------------------------- ridge
struct Ridge {
    d: usize,
    xtx: Vec<f64>,
    xty: Vec<f64>,
    n: u64,
}

impl Ridge {
    fn new(d: usize) -> Ridge {
        Ridge { d, xtx: vec![0.0; d * d], xty: vec![0.0; d], n: 0 }
    }
    fn add(&mut self, x: &[(u16, f32)], y: f64) {
        for &(i, vi) in x {
            let i = i as usize;
            self.xty[i] += vi as f64 * y;
            for &(j, vj) in x {
                self.xtx[i * self.d + j as usize] += vi as f64 * vj as f64;
            }
        }
        self.n += 1;
    }
    /// solve on the index subset `keep` (others weight 0)
    fn solve(&self, keep: &[bool], lam: f64) -> Vec<f64> {
        let idx: Vec<usize> = (0..self.d).filter(|&i| keep[i] && self.xtx[i * self.d + i] > 0.0).collect();
        let k = idx.len();
        let mut a = vec![0f64; k * k];
        let mut r = vec![0f64; k];
        for (p, &i) in idx.iter().enumerate() {
            r[p] = self.xty[i];
            for (q, &j) in idx.iter().enumerate() {
                a[p * k + q] = self.xtx[i * self.d + j];
            }
            a[p * k + p] += lam * (1.0 + self.xtx[i * self.d + i] * 1e-6);
        }
        // Cholesky
        for j in 0..k {
            let mut s = a[j * k + j];
            for t in 0..j {
                s -= a[j * k + t] * a[j * k + t];
            }
            let dj = s.max(1e-12).sqrt();
            a[j * k + j] = dj;
            for i in j + 1..k {
                let mut s = a[i * k + j];
                for t in 0..j {
                    s -= a[i * k + t] * a[j * k + t];
                }
                a[i * k + j] = s / dj;
            }
        }
        let mut y = r.clone();
        for i in 0..k {
            for t in 0..i {
                y[i] -= a[i * k + t] * y[t];
            }
            y[i] /= a[i * k + i];
        }
        for i in (0..k).rev() {
            for t in i + 1..k {
                y[i] -= a[t * k + i] * y[t];
            }
            y[i] /= a[i * k + i];
        }
        let mut w = vec![0f64; self.d];
        for (p, &i) in idx.iter().enumerate() {
            w[i] = y[p];
        }
        w
    }
}

fn dot(w: &[f64], x: &[(u16, f32)]) -> f64 {
    x.iter().map(|&(i, v)| w[i as usize] * v as f64).sum()
}

struct ValPos {
    y: f64,
    x: Vec<(u16, f32)>,
    kids: Vec<(f64, Vec<(u16, f32)>, f64, Vec<(u16, f32)>)>, // (nnue child eval, child pos feats, move delta, move feats)
}

pub fn run(args: &[String]) -> ExitCode {
    let Some(path) = arg(args, "--tdat") else {
        eprintln!("usage: ti hce-fit --tdat X.tdat [--train 20000] [--val 2000] [--seed 7] [--out FILE]");
        return ExitCode::from(2);
    };
    let ntr: usize = arg(args, "--train").and_then(|v| v.parse().ok()).unwrap_or(20_000);
    let nva: usize = arg(args, "--val").and_then(|v| v.parse().ok()).unwrap_or(2_000);
    let mut seed: u64 = arg(args, "--seed").and_then(|v| v.parse::<u64>().ok()).unwrap_or(7) | 1;
    let out_path = arg(args, "--out");
    let mut f = match File::open(&path) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("{path}: {e}");
            return ExitCode::from(1);
        }
    };
    let total = f.metadata().map(|m| m.len() as usize / REC).unwrap_or(0);
    let net = Nnue::bundled();
    let ring = ring1();
    let mut ml = MoveList::new();
    let mut ml2 = MoveList::new();
    let mut xs = Vec::new();
    let mut xm = Vec::new();
    let mut pos_r = Ridge::new(PD);
    let mut mov_r = Ridge::new(MD);
    let mut val: Vec<ValPos> = Vec::with_capacity(nva);
    let mut buf = [0u8; REC];
    let (mut done, mut nkids) = (0usize, 0u64);
    let t0 = std::time::Instant::now();
    while done < ntr + nva {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        if f.seek(SeekFrom::Start((seed % total as u64) * REC as u64)).is_err() || f.read_exact(&mut buf).is_err() {
            continue;
        }
        let r = Rec::from_bytes(&buf);
        let (own, opp) = if r.stm == 0 { (r.occ0, r.occ1) } else { (r.occ1, r.occ0) };
        let b = Board::from_occ(own, opp, 0, 0);
        b.legal_moves_dedup_into(&mut ml);
        if ml.len() < 2 {
            continue;
        }
        let is_val = done >= ntr;
        let yp = (net.eval(&b, 0) as f64).clamp(-CLAMP, CLAMP);
        pos_feats(&b, &ring, &mut ml2, &mut xs);
        let mut vp = if is_val { Some(ValPos { y: yp, x: xs.clone(), kids: Vec::with_capacity(ml.len()) }) } else { None };
        if !is_val {
            pos_r.add(&xs, yp);
        }
        let ph = phase(b.empty());
        let n = node_ctx(&b);
        let moves: Vec<_> = ml.iter().collect();
        for m in moves {
            let c = b.make(m);
            let yc = (net.eval(&c, 0) as f64).clamp(-CLAMP, CLAMP);
            let delta = (-yc - yp).clamp(-CLAMP, CLAMP);
            let caps = (b.opp() & ring[m.to as usize]).count_ones() as usize;
            let os = est(&n, &ring, m.from, m.to, m.is_clone(), 0);
            move_feats(ph, m.is_clone(), caps, m.from, m.to, os, &mut xm);
            pos_feats(&c, &ring, &mut ml2, &mut xs);
            if let Some(v) = vp.as_mut() {
                v.kids.push((yc, xs.clone(), delta, xm.clone()));
            } else {
                pos_r.add(&xs, yc);
                mov_r.add(&xm, delta);
            }
            nkids += 1;
        }
        if let Some(v) = vp {
            val.push(v);
        }
        done += 1;
    }
    let mut rep = String::new();
    let mut say = |s: String| {
        println!("{s}");
        rep.push_str(&s);
        rep.push('\n');
    };
    say(format!(
        "hce-fit: {ntr} train + {nva} val parents, {nkids} children, bundled champion net, {} positions in the position fit, {:.1}s",
        pos_r.n,
        t0.elapsed().as_secs_f64()
    ));

    // ---------------- position model, groups and ablations
    let grp = |i: usize| -> usize {
        let k = i % PB;
        match k {
            0 => 0,
            1 => 1,
            2..=21 => 2,
            _ if k < 2 + 2 * NCL + DYN => 3,
            _ => 4 + (k - 2 - 2 * NCL - DYN),
        }
    }; // 0 bias, 1 material, 2 static pst, 3 dynamic pst, 4.. hand terms
    let ngrp = 4 + NH;
    let gname = |g: usize| -> String {
        match g {
            0 => "bias".into(),
            1 => "material".into(),
            2 => "static PST".into(),
            3 => "dynamic PST (3x3)".into(),
            _ => HAND[g - 4].into(),
        }
    };
    let lam = 1.0;
    let eval_pos = |w: &[f64]| -> (f64, f64, f64, f64) {
        let (mut se, mut st, mut sa, mut n) = (0f64, 0f64, 0f64, 0f64);
        let mut ys = Vec::new();
        let (mut agree, mut regret, mut np) = (0f64, 0f64, 0f64);
        for v in &val {
            let p = dot(w, &v.x);
            ys.push((v.y, p));
            let mut best_nn = (f64::MIN, 0usize);
            let mut best_h = (f64::MIN, 0usize);
            for (i, k) in v.kids.iter().enumerate() {
                let ph = dot(w, &k.1);
                ys.push((k.0, ph));
                if -k.0 > best_nn.0 {
                    best_nn = (-k.0, i);
                }
                if -ph > best_h.0 {
                    best_h = (-ph, i);
                }
            }
            agree += (best_nn.1 == best_h.1) as u8 as f64;
            regret += best_nn.0 - (-v.kids[best_h.1].0);
            np += 1.0;
        }
        let mean = ys.iter().map(|q| q.0).sum::<f64>() / ys.len() as f64;
        for (y, p) in &ys {
            se += (y - p) * (y - p);
            st += (y - mean) * (y - mean);
            sa += (y - p).abs();
            n += 1.0;
        }
        (1.0 - se / st, sa / n, 100.0 * agree / np, regret / np)
    };
    say(String::new());
    say("A) POSITION MODEL (target = champion static eval, cp, clamp +-2000; val parents + all children)".into());
    say(format!("{:<34} {:>7} {:>8} {:>9} {:>10}", "model", "R2", "MAE cp", "best-move%", "regret cp"));
    let only = |gs: &[usize]| -> Vec<bool> { (0..PD).map(|i| gs.contains(&grp(i))).collect() };
    let mut row = |name: &str, keep: Vec<bool>| -> (f64, f64) {
        let w = pos_r.solve(&keep, lam);
        let (r2, mae, ag, rg) = eval_pos(&w);
        say(format!("{:<34} {:>7.4} {:>8.1} {:>9.2} {:>10.1}", name, r2, mae, ag, rg));
        (r2, ag)
    };
    row("bias + material", only(&[0, 1]));
    row("bias + material + static PST", only(&[0, 1, 2]));
    row("bias + material + dynamic PST", only(&[0, 1, 3]));
    row("bias + material + both PST", only(&[0, 1, 2, 3]));
    row("bias + material + hand terms", (0..PD).map(|i| grp(i) != 2 && grp(i) != 3).collect());
    row("ALL minus dynamic PST", (0..PD).map(|i| grp(i) != 3).collect());
    let (r2_all, ag_all) = row("ALL", vec![true; PD]);
    say(String::new());
    say("drop-one ablation from ALL (positive = the champion's eval needs this term):".into());
    let mut abl = Vec::new();
    for g in 1..ngrp {
        let keep: Vec<bool> = (0..PD).map(|i| grp(i) != g).collect();
        let w = pos_r.solve(&keep, lam);
        let (r2, _, ag, _) = eval_pos(&w);
        abl.push((g, r2_all - r2, ag_all - ag));
    }
    abl.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    say(format!("{:<20} {:>9} {:>12}", "term", "dR2", "d best-move%"));
    for (g, dr, da) in &abl {
        say(format!("{:<20} {:>9.4} {:>12.2}", gname(*g), dr, da));
    }
    let w_all = pos_r.solve(&vec![true; PD], lam);
    say(String::new());
    say("ALL-model weights per phase (cp per unit; hand terms are me-minus-opponent unless the name says me/opp):".into());
    for ph in 0..NPH {
        let b = ph * PB;
        let mut s = format!("  {:<14} material {:>6.1} |", PH_NAME[ph], w_all[b + 1]);
        for (i, n) in HAND.iter().enumerate() {
            s.push_str(&format!(" {n} {:.1}", w_all[b + 2 + 2 * NCL + DYN + i]));
        }
        say(s);
    }
    let grid = |w: &[f64], off: usize, title: &str, say: &mut dyn FnMut(String)| {
        say(format!("  {title}"));
        for r in 0..7 {
            let mut s = String::from("   ");
            for c in 0..7 {
                s.push_str(&format!("{:>7.1}", w[off + class(r * 7 + c)]));
            }
            say(s);
        }
    };
    say(String::new());
    say("STONE PST (ALL model, cp for one stone on the square, on top of material):".into());
    for ph in 0..NPH {
        let b = ph * PB;
        grid(&w_all, b + 2, &format!("{} OWN stone", PH_NAME[ph]), &mut say);
        grid(&w_all, b + 2 + NCL, &format!("{} ENEMY stone", PH_NAME[ph]), &mut say);
    }

    say(String::new());
    say("DYNAMIC PST (ALL model, cp per stone; rows = square class, cols = neighbours of that kind 0..8 in its 3x3):".into());
    say("  classes: 0 corner, 1/2/3 edge (1 next to corner .. 3 edge centre), 4 (1,1), 5 (1,2), 6 (1,3), 7 (2,2), 8 (2,3), 9 centre".into());
    for ph in 0..NPH {
        for t in 0..4 {
            say(format!("  {} {}", PH_NAME[ph], DYN_NAME[t]));
            for cl in 0..NCL {
                let off = ph * PB + 2 + 2 * NCL + t * NCL * 9 + cl * 9;
                let cells: Vec<String> = (0..9)
                    .map(|k| if pos_r.xtx[(off + k) * PD + off + k] > 0.0 { format!("{:>7.1}", w_all[off + k]) } else { "      .".into() })
                    .collect();
                say(format!("   cls {cl} {}", cells.join("")));
            }
        }
    }

    // ---------------- move model
    say(String::new());
    say("B) MOVE MODEL (target = value of the move for the mover, -eval(child) - eval(parent), cp)".into());
    let mg = |i: usize| -> usize {
        let k = i % MB;
        match k {
            0 | 1 => 0,
            2..=19 => 1,
            20..=49 => 2,
            _ => 3,
        }
    }; // 0 kind, 1 captures, 2 squares, 3 opponent reply
    let mgn = ["clone/jump", "captures", "square tables", "opponent best reply (2-ply)"];
    let eval_mov = |w: &[f64]| -> (f64, f64) {
        let (mut se, mut st, mut ag, mut np) = (0f64, 0f64, 0f64, 0f64);
        let all: Vec<f64> = val.iter().flat_map(|v| v.kids.iter().map(|k| k.2)).collect();
        let mean = all.iter().sum::<f64>() / all.len() as f64;
        for v in &val {
            let (mut bn, mut bh) = ((f64::MIN, 0), (f64::MIN, 0));
            for (i, k) in v.kids.iter().enumerate() {
                let p = dot(w, &k.3);
                se += (k.2 - p) * (k.2 - p);
                st += (k.2 - mean) * (k.2 - mean);
                if k.2 > bn.0 {
                    bn = (k.2, i);
                }
                if p > bh.0 {
                    bh = (p, i);
                }
            }
            ag += (bn.1 == bh.1) as u8 as f64;
            np += 1.0;
        }
        (1.0 - se / st, 100.0 * ag / np)
    };
    let full_m = mov_r.solve(&vec![true; MD], lam);
    let (mr2, mag) = eval_mov(&full_m);
    say(format!("  ALL move terms: R2 {:.4}, best-move agreement {:.2}%", mr2, mag));
    for g in 0..4 {
        let keep: Vec<bool> = (0..MD).map(|i| mg(i) != g).collect();
        let (r2, ag) = eval_mov(&mov_r.solve(&keep, lam));
        say(format!("  without {:<28} R2 {:.4} (d {:+.4}), best-move {:.2}% (d {:+.2})", mgn[g], r2, r2 - mr2, ag, ag - mag));
    }
    let only_m: Vec<bool> = (0..MD).map(|i| mg(i) == 0 || mg(i) == 1).collect();
    let (r2c, agc) = eval_mov(&mov_r.solve(&only_m, lam));
    say(format!("  captures + kind only (greedy capture count): R2 {:.4}, best-move {:.2}%", r2c, agc));
    for ph in 0..NPH {
        let b = ph * MB;
        say(String::new());
        say(format!("  {}: clone {:.1}, jump {:.1} cp; capture value by number of stones converted:", PH_NAME[ph], full_m[b], full_m[b + 1]));
        let caps = |o: usize| (0..9).map(|k| format!("{:.0}", full_m[b + 2 + o + k])).collect::<Vec<_>>().join(" ");
        say(format!("    clone caps 0..8: {}", caps(0)));
        say(format!("    jump  caps 0..8: {}", caps(9)));
        say(format!(
            "    opponent best reply state 0(no move),1..18 (1+gain): {}",
            (0..19).map(|k| format!("{:.0}", full_m[b + 20 + 3 * NCL + k])).collect::<Vec<_>>().join(" ")
        ));
        grid(&full_m, b + 20, &format!("{} CLONE destination PST", PH_NAME[ph]), &mut say);
        grid(&full_m, b + 20 + NCL, &format!("{} JUMP origin PST (square left empty)", PH_NAME[ph]), &mut say);
        grid(&full_m, b + 20 + 2 * NCL, &format!("{} JUMP destination PST", PH_NAME[ph]), &mut say);
    }
    if let Some(p) = out_path {
        if let Some(dir) = std::path::Path::new(&p).parent() {
            let _ = std::fs::create_dir_all(dir);
        }
        match File::create(&p).and_then(|mut f| f.write_all(rep.as_bytes())) {
            Ok(_) => println!("report written: {p}"),
            Err(e) => eprintln!("{p}: {e}"),
        }
    }
    ExitCode::SUCCESS
}

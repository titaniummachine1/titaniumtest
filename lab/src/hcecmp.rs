//! `lab hce-compare` (D974): how much of the champion's eval do OTHER engines' hand-crafted term sets capture?
//! Each set's terms are refitted closed-form to the champion eval (so this compares the TERMS, not their weights), per 4
//! phases, on train_v10 positions + all children; scored on held-out positions by R2 and sibling best-move agreement.
//!
//!   lab hce-compare --tdat X.tdat [--train 20000] [--val 2000] [--seed 7] [--out FILE]
//!
//! Sets (all me minus opponent, side to move):
//!   gataxx  : material, safe (no empty neighbour), edge, reach (no neighbour square the enemy can land on), cohesion (own-own
//!             adjacent pairs)                                  [gataxx FeatureEvaluator]
//!   funes   : material, corner + edge PSQT regions, corner-pocket counts (0..4 per corner, one-hot), centre-region counts,
//!             holes (empty next to us, enemy within 2) graded by own neighbours 0..8 (one-hot), weak / safe singles
//!                                                               [Funes src/eval.cpp]
//!   autaxx  : material, raw 49-square PST, holes by own neighbours (one-hot), surrounded stones      [autaxx tryhard eval]
//!   ours    : material, frontier, holes private / shared, no-move flags, square-class x weakness table, soft weaknesses
//!                                                               [lab hce-fast, D963]
//!   ours+gataxx : both
use crate::hcefit::{box2, class, dot, phase, Ridge, CLAMP, NPH};
use crate::util::{arg, box1, ring1, victim_planes, Rec, REC};
use std::fs::File;
use std::io::{Read, Seek, SeekFrom, Write};
use std::process::ExitCode;
use titanium::nnue::Nnue;
use titanium::{Board, MoveList};

const SETS: [&str; 5] = ["gataxx", "funes", "autaxx", "ours", "ours+gataxx"];
// per-phase block sizes
pub(crate) const G_N: usize = 6; // bias, material, safe, edge, reach, cohesion
const F_N: usize = 2 + 2 + 5 + 8 + 9 + 2; // bias, material, corner, edge, pocket 0..4, centre 0..7, holes 0..8, weak, safe
const A_N: usize = 2 + 49 + 9 + 1; // bias, material, pst, holes 0..8, surrounded
pub(crate) const O_N: usize = 7 + 2 * 10 * 9 + 18; // as hce-fast

pub(crate) fn block(set: usize) -> usize {
    match set {
        0 => G_N,
        1 => F_N,
        2 => A_N,
        3 => O_N,
        _ => O_N + G_N,
    }
}

fn edge_mask() -> u64 {
    (0..49u64).filter(|&s| { let (r, c) = (s / 7, s % 7); r == 0 || r == 6 || c == 0 || c == 6 }).fold(0, |m, s| m | 1 << s)
}
const CORNERS: u64 = 1 | (1 << 6) | (1 << 42) | (1 << 48);

fn sq_mask(f: impl Fn(u64, u64) -> bool) -> u64 {
    (0..49u64).filter(|&s| f(s / 7, s % 7)).fold(0, |m, s| m | 1 << s)
}

/// own-own adjacent pairs (each unordered pair once)
fn pairs(s: u64, ring: &[u64; 49]) -> f32 {
    let mut n = 0u32;
    let mut x = s;
    while x != 0 {
        let q = x.trailing_zeros() as usize;
        x &= x - 1;
        n += (ring[q] & s).count_ones();
    }
    n as f32 / 2.0
}

fn cnt(p: &[u64; 4], sq: usize) -> usize {
    (((p[0] >> sq) & 1) | (((p[1] >> sq) & 1) << 1) | (((p[2] >> sq) & 1) << 2) | (((p[3] >> sq) & 1) << 3)) as usize
}

fn gataxx_terms(me: u64, op: u64, e: u64, ring: &[u64; 49], edge: u64) -> [f32; 4] {
    let ez = box1(e);
    let (rm, ro) = (box2(me) & e, box2(op) & e);
    let safe = (me & !ez).count_ones() as f32 - (op & !ez).count_ones() as f32;
    let ed = (me & edge).count_ones() as f32 - (op & edge).count_ones() as f32;
    let reach = (me & !box1(ro)).count_ones() as f32 - (op & !box1(rm)).count_ones() as f32;
    let coh = pairs(me, ring) - pairs(op, ring);
    [safe, ed, reach, coh]
}

pub(crate) struct Masks {
    edge: u64,
    pockets: [u64; 4],
    centre: [u64; 4],
}

pub(crate) fn masks() -> Masks {
    // corner 2x2 pockets and the 4 central cross arms (Funes layout, folded to our indexing)
    let pk = |r0: u64, c0: u64| sq_mask(|r, c| (r == r0 || r == r0 + 1) && (c == c0 || c == c0 + 1));
    let cen = |k: u64| sq_mask(|r, c| match k { 0 => c == 3 && r <= 3, 1 => r == 3 && c >= 3, 2 => c == 3 && r >= 3, _ => r == 3 && c <= 3 });
    Masks { edge: edge_mask(), pockets: [pk(0, 0), pk(0, 5), pk(5, 0), pk(5, 5)], centre: [cen(0), cen(1), cen(2), cen(3)] }
}

fn push(out: &mut Vec<(u16, f32)>, i: usize, v: f32) {
    if v != 0.0 {
        out.push((i as u16, v));
    }
}

pub(crate) fn feats(set: usize, b: &Board, ring: &[u64; 49], m: &Masks, out: &mut Vec<(u16, f32)>) {
    out.clear();
    let (me, op, e) = (b.me(), b.opp(), b.empty());
    let ph = phase(e);
    let base = ph * block(set);
    let mat = me.count_ones() as f32 - op.count_ones() as f32;
    match set {
        0 | 4 => {
            let off = if set == 4 { base + O_N } else { base };
            push(out, off, 1.0);
            push(out, off + 1, mat);
            for (k, v) in gataxx_terms(me, op, e, ring, m.edge).iter().enumerate() {
                push(out, off + 2 + k, *v);
            }
            if set == 4 {
                ours(b, ring, base, out);
            }
        }
        1 => {
            push(out, base, 1.0);
            push(out, base + 1, mat);
            push(out, base + 2, (me & CORNERS).count_ones() as f32 - (op & CORNERS).count_ones() as f32);
            let ed = m.edge & !CORNERS;
            push(out, base + 3, (me & ed).count_ones() as f32 - (op & ed).count_ones() as f32);
            let mut acc = [0f32; F_N];
            for (s, sg) in [(me, 1.0f32), (op, -1.0)] {
                let o = if sg > 0.0 { op } else { me };
                for k in 0..4 {
                    acc[4 + (s & m.pockets[k]).count_ones() as usize] += sg;
                    acc[9 + ((s & m.centre[k]).count_ones() as usize).min(7)] += sg;
                }
                // holes: empty squares next to us that the enemy reaches within 2, graded by our neighbours
                let hb = box1(s) & box2(o) & e;
                let pl = victim_planes(s);
                let mut x = hb;
                while x != 0 {
                    let q = x.trailing_zeros() as usize;
                    x &= x - 1;
                    acc[17 + cnt(&pl, q)] += sg;
                }
                let singles = box1(s) & !s;
                acc[26] += sg * (singles & e).count_ones() as f32;
                acc[27] += sg * (singles & s).count_ones() as f32;
            }
            for (i, v) in acc.iter().enumerate().skip(4) {
                push(out, base + i, *v);
            }
        }
        2 => {
            push(out, base, 1.0);
            push(out, base + 1, mat);
            let mut acc = [0f32; A_N];
            for (s, sg) in [(me, 1.0f32), (op, -1.0)] {
                let o = if sg > 0.0 { op } else { me };
                let mut x = s;
                while x != 0 {
                    let q = x.trailing_zeros() as usize;
                    x &= x - 1;
                    acc[2 + q] += sg;
                }
                let hb = box1(s) & e & box2(o);
                let pl = victim_planes(s);
                let mut x = hb;
                while x != 0 {
                    let q = x.trailing_zeros() as usize;
                    x &= x - 1;
                    acc[51 + cnt(&pl, q)] += sg;
                }
                acc[60] += sg * (s & !box1(e)).count_ones() as f32;
            }
            for (i, v) in acc.iter().enumerate().skip(2) {
                push(out, base + i, *v);
            }
        }
        _ => ours(b, ring, base, out),
    }
}

/// the hce-fast term set (same definitions as lab hce-fast)
fn ours(b: &Board, ring: &[u64; 49], base: usize, out: &mut Vec<(u16, f32)>) {
    let (me, op, e) = (b.me(), b.opp(), b.empty());
    let (bm, bo) = (box2(me), box2(op));
    let (lo, lm) = (bo & e, bm & e);
    let pe = victim_planes(e);
    let holes = e & !(pe[0] | pe[1] | pe[2] | pe[3]);
    let be = box1(e);
    let near_e = box2(e);
    let (pl_o, pl_m, ps_o, ps_m) = (victim_planes(lo), victim_planes(lm), victim_planes(op & near_e), victim_planes(me & near_e));
    push(out, base, 1.0);
    push(out, base + 1, me.count_ones() as f32 - op.count_ones() as f32);
    push(out, base + 2, (me & be).count_ones() as f32 - (op & be).count_ones() as f32);
    push(out, base + 3, (holes & bm & !bo).count_ones() as f32 - (holes & bo & !bm).count_ones() as f32);
    push(out, base + 4, (holes & bm & bo).count_ones() as f32);
    push(out, base + 5, (lm == 0) as u8 as f32);
    push(out, base + 6, (lo == 0) as u8 as f32);
    let mut acc = [0f32; O_N];
    let mut x = me;
    while x != 0 {
        let q = x.trailing_zeros() as usize;
        x &= x - 1;
        acc[7 + class(q) * 9 + cnt(&pl_o, q)] += 1.0;
        acc[7 + 180 + cnt(&ps_o, q)] += 1.0;
    }
    let mut x = op;
    while x != 0 {
        let q = x.trailing_zeros() as usize;
        x &= x - 1;
        acc[7 + 90 + class(q) * 9 + cnt(&pl_m, q)] += 1.0;
        acc[7 + 189 + cnt(&ps_m, q)] += 1.0;
    }
    let _ = ring;
    for (i, v) in acc.iter().enumerate().skip(7) {
        push(out, base + i, *v);
    }
}

pub fn run(args: &[String]) -> ExitCode {
    let Some(path) = arg(args, "--tdat") else {
        eprintln!("usage: lab hce-compare --tdat X.tdat [--train 20000] [--val 2000] [--seed 7] [--out FILE]");
        return ExitCode::from(2);
    };
    let ntr: usize = arg(args, "--train").and_then(|v| v.parse().ok()).unwrap_or(20_000);
    let nva: usize = arg(args, "--val").and_then(|v| v.parse().ok()).unwrap_or(2_000);
    let mut seed: u64 = arg(args, "--seed").and_then(|v| v.parse::<u64>().ok()).unwrap_or(7) | 1;
    let net = Nnue::bundled();
    let ring = ring1();
    let m = masks();
    let mut f = File::open(&path).expect("tdat");
    let total = f.metadata().map(|x| x.len() as usize / REC).unwrap_or(0) as u64;
    let mut buf = [0u8; REC];
    let mut ml = MoveList::new();
    let mut ridges: Vec<Ridge> = (0..SETS.len()).map(|s| Ridge::new(NPH * block(s))).collect();
    let mut val: Vec<(Vec<Board>, Vec<f64>)> = Vec::new();
    let mut x = Vec::new();
    let t0 = std::time::Instant::now();
    let mut done = 0;
    while done < ntr + nva {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        if f.seek(SeekFrom::Start((seed % total) * REC as u64)).is_err() || f.read_exact(&mut buf).is_err() {
            continue;
        }
        let rc = Rec::from_bytes(&buf);
        let (own, opp) = if rc.stm == 0 { (rc.occ0, rc.occ1) } else { (rc.occ1, rc.occ0) };
        let b = Board::from_occ(own, opp, 0, 0);
        b.legal_moves_dedup_into(&mut ml);
        if ml.len() < 2 {
            continue;
        }
        let kids: Vec<Board> = ml.iter().map(|mv| b.make(mv)).collect();
        let ky: Vec<f64> = kids.iter().map(|c| (net.eval(c, 0) as f64).clamp(-CLAMP, CLAMP)).collect();
        if done < ntr {
            let yp = (net.eval(&b, 0) as f64).clamp(-CLAMP, CLAMP);
            for s in 0..SETS.len() {
                feats(s, &b, &ring, &m, &mut x);
                ridges[s].add(&x, yp);
                for (c, y) in kids.iter().zip(&ky) {
                    feats(s, c, &ring, &m, &mut x);
                    ridges[s].add(&x, *y);
                }
            }
        } else {
            val.push((kids, ky));
        }
        done += 1;
    }
    let mut rep = format!("hce-compare: {ntr} train + {nva} val parents (+ children), champion eval target, {:.0}s\n", t0.elapsed().as_secs_f64());
    rep.push_str(&format!("{:<12} {:>6} {:>8} {:>11} {:>10}\n", "term set", "dims", "R2", "best-move%", "regret cp"));
    for s in 0..SETS.len() {
        let d = NPH * block(s);
        let w = ridges[s].solve(&vec![true; d], 1.0);
        let (mut se, mut st, mut n, mut ag, mut rg) = (0f64, 0f64, 0f64, 0f64, 0f64);
        let mean: f64 = val.iter().flat_map(|v| v.1.iter()).sum::<f64>() / val.iter().map(|v| v.1.len()).sum::<usize>() as f64;
        for (kids, ky) in &val {
            let (mut bn, mut bh) = ((f64::MIN, 0usize), (f64::MIN, 0usize));
            for (i, (c, y)) in kids.iter().zip(ky).enumerate() {
                feats(s, c, &ring, &m, &mut x);
                let h = dot(&w, &x);
                se += (y - h).powi(2);
                st += (y - mean).powi(2);
                n += 1.0;
                if -y > bn.0 {
                    bn = (-y, i);
                }
                if -h > bh.0 {
                    bh = (-h, i);
                }
            }
            ag += (bn.1 == bh.1) as u8 as f64;
            rg += bn.0 - (-ky[bh.1]);
        }
        let _ = n;
        rep.push_str(&format!(
            "{:<12} {:>6} {:>8.4} {:>11.2} {:>10.1}\n",
            SETS[s], d, 1.0 - se / st, 100.0 * ag / val.len() as f64, rg / val.len() as f64
        ));
        if s == 0 {
            rep.push_str("  gataxx terms refitted to our net, per phase (cp per unit; late..open): ");
            for (k, nm) in ["material", "safe", "edge", "reach", "cohesion"].iter().enumerate() {
                let v: Vec<String> = (0..NPH).map(|ph| format!("{:.1}", w[ph * G_N + 1 + k])).collect();
                rep.push_str(&format!("{nm} {} | ", v.join("/")));
            }
            rep.push('\n');
        }
    }
    print!("{rep}");
    if let Some(p) = arg(args, "--out") {
        let _ = File::create(&p).and_then(|mut f| f.write_all(rep.as_bytes()));
        println!("report written: {p}");
    }
    ExitCode::SUCCESS
}

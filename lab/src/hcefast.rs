//! `lab hce-fast` (D963): the fastest useful hand-crafted eval built from the D948-D962 findings, fitted closed-form to
//! the champion and timed against it.
//!
//!   lab hce-fast --tdat X.tdat [--train 20000] [--val 2000] [--seed 7] [--out FILE]
//!
//! Terms per phase (4 by empties): bias, material, frontier, private / shared holes, no-move flags, and per stone a
//! square x weakness table (weaknesses = empty squares of its 3x3 the enemy can land on) + a soft-weakness table
//! (enemy neighbours that can jump out; approximation: enemy stones with an empty within distance 2).
//! Evaluation = integer table lookups over the stones + ~20 bitboard ops (victim planes), no search, no movegen.
use crate::hcefit::{box2, class, dot, phase, Ridge, CLAMP, NPH, PH_NAME};
use crate::util::{arg, box1, ring1, victim_planes, Rec, REC};
use std::fs::File;
use std::io::{Read, Seek, SeekFrom, Write};
use std::process::ExitCode;
use titanium::nnue::Nnue;
use titanium::{Board, MoveList};

const NCL: usize = 10;
// per phase: 0 bias, 1 material, 2 frontier, 3 holes private, 4 holes shared, 5 me no moves, 6 opp no moves,
// 7.. own class x weakness (90), enemy class x weakness (90), own soft 0..8 (9), enemy soft 0..8 (9)
const B_OWN: usize = 7;
const B_EN: usize = B_OWN + NCL * 9;
const B_OS: usize = B_EN + NCL * 9;
const B_ES: usize = B_OS + 9;
const PBF: usize = B_ES + 9;
const DF: usize = NPH * PBF;

#[inline(always)]
fn cnt(p: &[u64; 4], sq: usize) -> usize {
    (((p[0] >> sq) & 1) | (((p[1] >> sq) & 1) << 1) | (((p[2] >> sq) & 1) << 2) | (((p[3] >> sq) & 1) << 3)) as usize
}

struct Ctx {
    me: u64,
    op: u64,
    e: u64,
    ph: usize,
    pl_o: [u64; 4],
    pl_m: [u64; 4],
    ps_o: [u64; 4],
    ps_m: [u64; 4],
    front: i32,
    hp: i32,
    hs: i32,
    nm_me: bool,
    nm_op: bool,
}

#[inline(always)]
fn ctx(b: &Board) -> Ctx {
    let (me, op, e) = (b.me(), b.opp(), b.empty());
    let (bm, bo) = (box2(me), box2(op));
    let (lo, lm) = (bo & e, bm & e);
    let near_e = box2(e);
    let pe = victim_planes(e);
    let holes = e & !(pe[0] | pe[1] | pe[2] | pe[3]);
    let be = box1(e);
    Ctx {
        me,
        op,
        e,
        ph: phase(e),
        pl_o: victim_planes(lo),
        pl_m: victim_planes(lm),
        ps_o: victim_planes(op & near_e),
        ps_m: victim_planes(me & near_e),
        front: (me & be).count_ones() as i32 - (op & be).count_ones() as i32,
        hp: (holes & bm & !bo).count_ones() as i32 - (holes & bo & !bm).count_ones() as i32,
        hs: (holes & bm & bo).count_ones() as i32,
        nm_me: lm == 0,
        nm_op: lo == 0,
    }
}

fn feats(b: &Board, out: &mut Vec<(u16, f32)>) {
    out.clear();
    let c = ctx(b);
    let base = c.ph * PBF;
    let push = |out: &mut Vec<(u16, f32)>, i: usize, v: f32| {
        if v != 0.0 {
            out.push(((base + i) as u16, v));
        }
    };
    push(out, 0, 1.0);
    push(out, 1, c.me.count_ones() as f32 - c.op.count_ones() as f32);
    push(out, 2, c.front as f32);
    push(out, 3, c.hp as f32);
    push(out, 4, c.hs as f32);
    push(out, 5, c.nm_me as u8 as f32);
    push(out, 6, c.nm_op as u8 as f32);
    let mut acc = [0f32; PBF];
    let mut s = c.me;
    while s != 0 {
        let sq = s.trailing_zeros() as usize;
        s &= s - 1;
        acc[B_OWN + class(sq) * 9 + cnt(&c.pl_o, sq)] += 1.0;
        acc[B_OS + cnt(&c.ps_o, sq)] += 1.0;
    }
    let mut s = c.op;
    while s != 0 {
        let sq = s.trailing_zeros() as usize;
        s &= s - 1;
        acc[B_EN + class(sq) * 9 + cnt(&c.pl_m, sq)] += 1.0;
        acc[B_ES + cnt(&c.ps_m, sq)] += 1.0;
    }
    for (i, v) in acc.iter().enumerate().skip(B_OWN) {
        push(out, i, *v);
    }
}

/// integer tables: per phase scalars + per (square, weakness) own / enemy + soft
struct Tab {
    sc: [[i32; 7]; NPH],
    own: Vec<[[i32; 9]; 49]>,
    en: Vec<[[i32; 9]; 49]>,
    os: [[i32; 9]; NPH],
    es: [[i32; 9]; NPH],
}

fn tables(w: &[f64]) -> Tab {
    let q = |x: f64| x.round() as i32;
    let mut t = Tab { sc: [[0; 7]; NPH], own: vec![[[0; 9]; 49]; NPH], en: vec![[[0; 9]; 49]; NPH], os: [[0; 9]; NPH], es: [[0; 9]; NPH] };
    for ph in 0..NPH {
        let b = ph * PBF;
        for i in 0..7 {
            t.sc[ph][i] = q(w[b + i]);
        }
        for sq in 0..49 {
            for k in 0..9 {
                t.own[ph][sq][k] = q(w[b + B_OWN + class(sq) * 9 + k]);
                t.en[ph][sq][k] = q(w[b + B_EN + class(sq) * 9 + k]);
            }
        }
        for k in 0..9 {
            t.os[ph][k] = q(w[b + B_OS + k]);
            t.es[ph][k] = q(w[b + B_ES + k]);
        }
    }
    t
}

#[inline(always)]
fn fast_eval(b: &Board, t: &Tab) -> i32 {
    let c = ctx(b);
    let s = &t.sc[c.ph];
    let mut v = s[0]
        + s[1] * (c.me.count_ones() as i32 - c.op.count_ones() as i32)
        + s[2] * c.front
        + s[3] * c.hp
        + s[4] * c.hs
        + if c.nm_me { s[5] } else { 0 }
        + if c.nm_op { s[6] } else { 0 };
    let (own, en, os, es) = (&t.own[c.ph], &t.en[c.ph], &t.os[c.ph], &t.es[c.ph]);
    let mut m = c.me;
    while m != 0 {
        let sq = m.trailing_zeros() as usize;
        m &= m - 1;
        v += own[sq][cnt(&c.pl_o, sq)] + os[cnt(&c.ps_o, sq)];
    }
    let mut m = c.op;
    while m != 0 {
        let sq = m.trailing_zeros() as usize;
        m &= m - 1;
        v += en[sq][cnt(&c.pl_m, sq)] + es[cnt(&c.ps_m, sq)];
    }
    let _ = c.e;
    v
}

pub fn run(args: &[String]) -> ExitCode {
    let Some(path) = arg(args, "--tdat") else {
        eprintln!("usage: lab hce-fast --tdat X.tdat [--train 20000] [--val 2000] [--seed 7] [--out FILE]");
        return ExitCode::from(2);
    };
    let ntr: usize = arg(args, "--train").and_then(|v| v.parse().ok()).unwrap_or(20_000);
    let nva: usize = arg(args, "--val").and_then(|v| v.parse().ok()).unwrap_or(2_000);
    let mut seed: u64 = arg(args, "--seed").and_then(|v| v.parse::<u64>().ok()).unwrap_or(7) | 1;
    let out_path = arg(args, "--out");
    let net = Nnue::bundled();
    let _ring = ring1();
    let mut f = File::open(&path).expect("tdat");
    let total = f.metadata().map(|m| m.len() as usize / REC).unwrap_or(0) as u64;
    let mut buf = [0u8; REC];
    let mut ml = MoveList::new();
    let mut x = Vec::new();
    let mut r = Ridge::new(DF);
    // validation: (parent board, children boards, nnue child evals)
    let mut val: Vec<(Board, Vec<Board>, Vec<f64>)> = Vec::new();
    let mut all_kids: Vec<Board> = Vec::new();
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
        let kids: Vec<Board> = ml.iter().map(|m| b.make(m)).collect();
        let ky: Vec<f64> = kids.iter().map(|c| (net.eval(c, 0) as f64).clamp(-CLAMP, CLAMP)).collect();
        if done < ntr {
            feats(&b, &mut x);
            r.add(&x, (net.eval(&b, 0) as f64).clamp(-CLAMP, CLAMP));
            for (c, y) in kids.iter().zip(&ky) {
                feats(c, &mut x);
                r.add(&x, *y);
            }
        } else {
            all_kids.extend(kids.iter().cloned());
            val.push((b, kids, ky));
        }
        done += 1;
    }
    let w = r.solve(&vec![true; DF], 1.0);
    let t = tables(&w);
    let mut rep = String::new();
    let mut say = |s: String| {
        println!("{s}");
        rep.push_str(&s);
        rep.push('\n');
    };
    say(format!("hce-fast: {ntr} train parents + children fitted closed-form, {nva} val parents ({:.0}s)", t0.elapsed().as_secs_f64()));
    // accuracy + parity of the integer eval with the float fit
    let (mut agree, mut regret, mut se, mut st, mut n, mut maxpar) = (0f64, 0f64, 0f64, 0f64, 0f64, 0f64);
    let mean_y: f64 = val.iter().flat_map(|v| v.2.iter()).sum::<f64>() / val.iter().map(|v| v.2.len()).sum::<usize>() as f64;
    for (_, kids, ky) in &val {
        let (mut bn, mut bh) = ((f64::MIN, 0usize), (f64::MIN, 0usize));
        for (i, (c, y)) in kids.iter().zip(ky).enumerate() {
            let h = fast_eval(c, &t) as f64;
            feats(c, &mut x);
            maxpar = f64::max(maxpar, (h - dot(&w, &x)).abs());
            se += (y - h).powi(2);
            st += (y - mean_y).powi(2);
            n += 1.0;
            if -y > bn.0 {
                bn = (-y, i);
            }
            if -h > bh.0 {
                bh = (-h, i);
            }
        }
        agree += (bn.1 == bh.1) as u8 as f64;
        regret += bn.0 - (-ky[bh.1]);
    }
    say(format!(
        "  integer HCE vs champion on {} val children: R2 {:.4}, sibling best-move agreement {:.2}%, regret {:.1} cp/move; integer vs float fit max diff {:.1} cp",
        n as u64,
        1.0 - se / st,
        100.0 * agree / val.len() as f64,
        regret / val.len() as f64,
        maxpar
    ));
    // speed: same boards, 5 rounds each, best round
    let time = |fun: &dyn Fn(&Board) -> i32| -> f64 {
        let mut best = f64::MAX;
        for _ in 0..5 {
            let t1 = std::time::Instant::now();
            let mut sink = 0i64;
            for b in &all_kids {
                sink = sink.wrapping_add(fun(std::hint::black_box(b)) as i64);
            }
            std::hint::black_box(sink);
            best = best.min(t1.elapsed().as_nanos() as f64 / all_kids.len() as f64);
        }
        best
    };
    let ns_hce = time(&|b| fast_eval(b, &t));
    let ns_nn = time(&|b| net.eval(b, 0));
    say(format!(
        "  speed on {} positions (single thread, best of 5): integer HCE {:.1} ns/eval, champion NNUE {:.1} ns/eval (full forward, no incremental accumulator) = {:.1}x",
        all_kids.len(),
        ns_hce,
        ns_nn,
        ns_nn / ns_hce
    ));
    say("  fitted tables (cp), own stone value by square class x weakness count, per phase:".into());
    for ph in 0..NPH {
        say(format!("   {} material {:.1} frontier {:.1} holes private {:.1} shared {:.1}", PH_NAME[ph], w[ph * PBF + 1], w[ph * PBF + 2], w[ph * PBF + 3], w[ph * PBF + 4]));
        for cl in 0..NCL {
            let cells: Vec<String> = (0..9).map(|k| format!("{:>6.0}", w[ph * PBF + B_OWN + cl * 9 + k])).collect();
            say(format!("     cls {cl} {}", cells.join("")));
        }
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

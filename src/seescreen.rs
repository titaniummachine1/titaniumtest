//! `ti see-screen` (D944): how well do CHEAP approximations of the 2-ply exchange (my capture gain, the opponent's best
//! reply gain) match the exact value, and what do they cost per move? Research tool, read-only, no engine change.
//!
//!   ti see-screen --tdat X.tdat [--rows 50000] [--seed 7] [--bench]
//!
//! Ground truth = brute force: play the move, enumerate the opponent's legal moves, best gain = clone + 2 * victims.
//! State index as the TPOL5 head uses it: 0 = opponent has no move, else 1 + best gain.
//! Candidates (what the net would be fed instead of the exact state):
//!   exact   victim planes + ring of the destination + every flipped stone + jump origin (policy2 / engine path)
//!   noflip  planes + ring of the destination only, landings from the opponent stones left after the flips
//!   parent  as noflip but landings from the parent position (no box1 per move)
//!   levels  per node: the top two count levels of the landing squares; per move 3-4 mask tests, no planes at all
//!   node    one constant per node (the opponent's best reply if we passed): no per-move work
use crate::data::{Rec, REC};
use crate::policy::{arg, ring1};
use crate::policy2::{box1, opp_state, planes_add, planes_max, planes_sub, victim_planes};
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::process::ExitCode;
use titanium::{Board, MoveList};

const NAMES: [&str; 6] = ["exact", "noflip", "parent", "levels", "node", "levels3"];
const NS: usize = 19;
static WIN: std::sync::OnceLock<[u64; 49]> = std::sync::OnceLock::new();

fn eq_level(p: &[u64; 4], k: u32) -> u64 {
    let mut m = !0u64;
    for i in 0..4 {
        m &= if (k >> i) & 1 == 1 { p[i] } else { !p[i] };
    }
    m
}

/// top count level of a landing set, the squares at that level and at one below
#[derive(Clone, Copy)]
struct Lv {
    k: u32,
    a: u64,
    b: u64,
}

fn lv(l: u64, p: &[u64; 4]) -> Lv {
    if l == 0 {
        return Lv { k: 0, a: 0, b: 0 };
    }
    let k = planes_max(l, p);
    Lv { k, a: l & eq_level(p, k), b: if k > 0 { l & eq_level(p, k - 1) } else { 0 } }
}

/// new best victim count after +1 on `r` (the new stone) with `tbit` no longer a landing square
#[inline(always)]
fn after(v: &Lv, tbit: u64, r: u64) -> u32 {
    if v.a & r != 0 {
        v.k + 1
    } else if (v.a & !tbit) != 0 || (v.b & r) != 0 {
        v.k
    } else {
        v.k.saturating_sub(1)
    }
}


/// top three count levels of a landing set: k and the squares at k, k-1, k-2
#[derive(Clone, Copy)]
struct Lv3 {
    k: u32,
    s: [u64; 3],
}

fn lv3(l: u64, p: &[u64; 4]) -> Lv3 {
    if l == 0 {
        return Lv3 { k: 0, s: [0; 3] };
    }
    let k = planes_max(l, p);
    let g = |j: u32| if k >= j { l & eq_level(p, k - j) } else { 0 };
    Lv3 { k, s: [g(0), g(1), g(2)] }
}

fn win_table(ring: &[u64; 49]) -> [u64; 49] {
    let mut w = [0u64; 49];
    for t in 0..49 {
        w[t] = box1(ring[t] | (1u64 << t));
    }
    w
}

/// best victim count after the move: squares in the 5x5 window around the destination get their exact bonus
/// (popcount of the new victims in their 3x3), squares outside keep their level
#[inline(always)]
fn after3(v: &Lv3, ring: &[u64; 49], win: u64, newv: u64, tbit: u64) -> u32 {
    let mut best = 0u32;
    for j in 0..3u32 {
        if v.k < j {
            break;
        }
        let lvl = v.k - j;
        let sq = v.s[j as usize] & !tbit;
        if sq & !win != 0 {
            best = best.max(lvl);
        }
        let mut inw = sq & win;
        while inw != 0 {
            let u = inw.trailing_zeros() as usize;
            inw &= inw - 1;
            best = best.max(lvl + (ring[u] & newv).count_ones());
        }
    }
    best
}
pub(crate) struct Node {
    me: u64,
    opp: u64,
    empty: u64,
    pm: [u64; 4],
    b1: u64,
    b2: u64,
    l1: u64,
    l2: u64,
    v1: Lv,
    v2: Lv,
    node_state: usize,
    w1: Lv3,
    w2: Lv3,
}

pub(crate) fn node_ctx(b: &Board) -> Node {
    let (me, opp, empty) = (b.me(), b.opp(), b.empty());
    let pm = victim_planes(me);
    let b1 = box1(opp);
    let b2 = box1(b1);
    let (l1, l2) = (b1 & empty, b2 & empty);
    Node { me, opp, empty, pm, b1, b2, l1, l2, v1: lv(l1, &pm), v2: lv(l2, &pm), node_state: opp_state(l1, l2, &pm), w1: lv3(l1, &pm), w2: lv3(l2, &pm) }
}

pub(crate) fn est(n: &Node, ring: &[u64; 49], from: u8, to: u8, clone: bool, which: usize) -> usize {
    let tu = to as usize;
    let tbit = 1u64 << tu;
    match which {
        0 => {
            let flips = n.opp & ring[tu];
            let b1 = box1(n.opp & !flips);
            let b2 = box1(b1);
            let mut p = n.pm;
            planes_add(&mut p, ring[tu]);
            let mut fm = flips;
            while fm != 0 {
                let u = fm.trailing_zeros() as usize;
                fm &= fm - 1;
                planes_add(&mut p, ring[u]);
            }
            let mut ea = n.empty & !tbit;
            if !clone {
                planes_sub(&mut p, ring[from as usize]);
                ea |= 1u64 << from;
            }
            opp_state(b1 & ea, b2 & ea, &p)
        }
        1 | 2 => {
            let (b1, b2) = if which == 1 {
                let b1 = box1(n.opp & !(n.opp & ring[tu]));
                (b1, box1(b1))
            } else {
                (n.b1, n.b2)
            };
            let mut p = n.pm;
            planes_add(&mut p, ring[tu]);
            let ea = n.empty & !tbit;
            opp_state(b1 & ea, b2 & ea, &p)
        }
        3 => {
            if n.l2 & !tbit == 0 {
                return 0;
            }
            let r = ring[tu];
            let cl = if n.l1 & !tbit != 0 { 1 + 2 * after(&n.v1, tbit, r) } else { 0 };
            let jp = 2 * after(&n.v2, tbit, r);
            1 + cl.max(jp) as usize
        }
        5 => {
            if n.l2 & !tbit == 0 {
                return 0;
            }
            let win = WIN.get_or_init(|| win_table(ring))[tu];
            let newv = tbit | (n.opp & ring[tu]);
            let cl = if n.l1 & !tbit != 0 { 1 + 2 * after3(&n.w1, ring, win, newv, tbit) } else { 0 };
            let jp = 2 * after3(&n.w2, ring, win, newv, tbit);
            1 + cl.max(jp) as usize
        }        _ => n.node_state,
    }
}

/// brute force 2-ply reference: 0 = opponent has no move, else 1 + best clone/jump gain
fn brute(b: &Board, m: titanium::Move, ring: &[u64; 49], ml: &mut MoveList) -> usize {
    let b2 = b.make(m);
    b2.legal_moves_dedup_into(ml);
    if ml.len() == 0 {
        return 0;
    }
    let ours = b2.opp();
    let mut best = 0u32;
    for m2 in ml.iter() {
        let g = (ours & ring[m2.to as usize]).count_ones() * 2 + m2.is_clone() as u32;
        best = best.max(g);
    }
    1 + best as usize
}

fn opp_gain(s: usize) -> i32 {
    if s == 0 { -3 } else { s as i32 - 1 }
}

fn coarse(s: usize) -> usize {
    match s {
        0 => 0,
        1..=3 => 1,
        4..=6 => 2,
        7..=9 => 3,
        _ => 4,
    }
}

pub fn run(args: &[String]) -> ExitCode {
    let Some(path) = arg(args, "--tdat") else {
        eprintln!("usage: ti see-screen --tdat X.tdat [--rows 50000] [--seed 7] [--bench]");
        return ExitCode::from(2);
    };
    let rows: usize = arg(args, "--rows").and_then(|v| v.parse().ok()).unwrap_or(50_000);
    let mut seed: u64 = arg(args, "--seed").and_then(|v| v.parse::<u64>().ok()).unwrap_or(7) | 1;
    let bench = args.iter().any(|a| a == "--bench");
    let mut f = match File::open(&path) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("{path}: {e}");
            return ExitCode::from(1);
        }
    };
    let total = f.metadata().map(|m| m.len() as usize / REC).unwrap_or(0);
    if total == 0 {
        eprintln!("{path}: empty");
        return ExitCode::from(1);
    }
    let ring = ring1();
    let mut boards = Vec::with_capacity(rows);
    let mut buf = [0u8; REC];
    while boards.len() < rows {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        let i = (seed % total as u64) * REC as u64;
        if f.seek(SeekFrom::Start(i)).is_err() || f.read_exact(&mut buf).is_err() {
            continue;
        }
        let r = Rec::from_bytes(&buf);
        let (own, opp) = if r.stm == 0 { (r.occ0, r.occ1) } else { (r.occ1, r.occ0) };
        let b = Board::from_occ(own, opp, 0, 0);
        if b.empty() == 0 {
            continue;
        }
        boards.push(b);
    }

    let mut ml = MoveList::new();
    let mut ml2 = MoveList::new();
    let mut joint = vec![[[0u64; NS]; NS]; 6];
    let mut matches = [0u64; 6];
    let mut coarse_matches = [0u64; 6];
    let mut abs_err = [0f64; 6];
    let mut top_dis = [0u64; 6];
    let mut regret = [0f64; 6];
    let (mut nmoves, mut npos, mut brute_bad) = (0u64, 0u64, 0u64);

    for b in &boards {
        b.legal_moves_dedup_into(&mut ml);
        if ml.len() < 2 {
            continue;
        }
        npos += 1;
        let n = node_ctx(b);
        let mut ex = Vec::with_capacity(ml.len());
        let mut my = Vec::with_capacity(ml.len());
        let mut ap = vec![Vec::with_capacity(ml.len()); 6];
        for m in ml.iter() {
            let clone = m.is_clone();
            let flips = (n.opp & ring[m.to as usize]).count_ones() as i32;
            my.push(clone as i32 + 2 * flips);
            let r = brute(b, m, &ring, &mut ml2);
            ex.push(r);
            for w in 0..6 {
                let e = est(&n, &ring, m.from, m.to, clone, w).min(NS - 1);
                ap[w].push(e);
                joint[w][r.min(NS - 1)][e] += 1;
                matches[w] += (e == r) as u64;
                coarse_matches[w] += (coarse(e) == coarse(r)) as u64;
                abs_err[w] += (e as f64 - r as f64).abs();
            }
            if ap[0].last() != Some(&r) {
                brute_bad += 1;
            }
            nmoves += 1;
        }
        let net = |i: usize, s: usize| my[i] - opp_gain(s);
        let best = (0..ex.len()).map(|i| net(i, ex[i])).max().unwrap();
        for w in 0..6 {
            let mut bi = 0;
            for i in 1..ex.len() {
                if net(i, ap[w][i]) > net(bi, ap[w][bi]) {
                    bi = i;
                }
            }
            let rg = best - net(bi, ex[bi]);
            top_dis[w] += (rg > 0) as u64;
            regret[w] += rg as f64;
        }
    }

    println!("see-screen: {npos} positions, {nmoves} moves from {path} (seed {seed}); exact(planes) vs brute-force 2-ply differ on {brute_bad} moves");
    println!("{:<8} {:>8} {:>8} {:>8} {:>10} {:>9} {:>9}", "approx", "match%", "coarse%", "MAE", "H(ex|ap)b", "top1 miss%", "regret");
    for w in 0..6 {
        // conditional entropy of the exact state given the approximation, in bits
        let mut h = 0f64;
        for e in 0..NS {
            let col: u64 = (0..NS).map(|r| joint[w][r][e]).sum();
            if col == 0 {
                continue;
            }
            for r in 0..NS {
                let c = joint[w][r][e];
                if c > 0 {
                    h -= c as f64 * (c as f64 / col as f64).log2();
                }
            }
        }
        println!(
            "{:<8} {:>8.2} {:>8.2} {:>8.3} {:>10.3} {:>9.2} {:>9.3}",
            NAMES[w],
            100.0 * matches[w] as f64 / nmoves as f64,
            100.0 * coarse_matches[w] as f64 / nmoves as f64,
            abs_err[w] / nmoves as f64,
            h / nmoves as f64,
            100.0 * top_dis[w] as f64 / npos as f64,
            regret[w] / npos as f64
        );
    }

    if bench {
        println!("\nns per move (single thread, node setup included, black_box; run on an idle PC):");
        let mut sink = 0usize;
        for w in 0..6 {
            let t = std::time::Instant::now();
            let mut cnt = 0u64;
            for b in &boards {
                b.legal_moves_dedup_into(&mut ml);
                if ml.len() < 2 {
                    continue;
                }
                let n = node_ctx(b);
                for m in ml.iter() {
                    sink = sink.wrapping_add(est(&n, &ring, m.from, m.to, m.is_clone(), w));
                    cnt += 1;
                }
            }
            let ns = t.elapsed().as_nanos() as f64 / cnt as f64;
            println!("  {:<8} {:>7.2} ns/move", NAMES[w], ns);
        }
        // movegen alone, so it can be subtracted
        let t = std::time::Instant::now();
        let mut cnt = 0u64;
        for b in &boards {
            b.legal_moves_dedup_into(&mut ml);
            if ml.len() < 2 {
                continue;
            }
            let n = node_ctx(b);
            sink = sink.wrapping_add(n.node_state);
            cnt += ml.len() as u64;
        }
        println!("  (movegen + node setup only: {:.2} ns/move) sink {}", t.elapsed().as_nanos() as f64 / cnt as f64, sink & 1);
    }
    ExitCode::SUCCESS
}

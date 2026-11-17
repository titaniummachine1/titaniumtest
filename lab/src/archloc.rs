//! `lab arch-locality` (D1110, user: "try many architectures, check how fast they are and how incremental they can be:
//! attacks are very local, a move changes up to ~10 stones, still not the entire board"). For every legal child of real
//! positions and for many CANDIDATE input families (absolute colours = one accumulator kept per colour, as an engine
//! with an accumulator stack would), counts the active feature rows of the child (= full-refresh adds) and the rows that
//! differ parent -> child (removed + added = incremental adds/subs), with the distribution of the diff. Counting only:
//! no timing (valid while the PC is busy), no claim about information content (that needs a residual / training run).
//! Usage: lab arch-locality --val V.tdat [--n 3000]

use std::process::ExitCode;

use titanium::nnfeat::{features, ALL_GROUPS, MAX_ACTIVE};
use titanium::{Board, MoveList};

use crate::util::arg;

const FULL: u64 = (1u64 << 49) - 1;

fn rc(sq: usize) -> (i32, i32) {
    ((sq / 7) as i32, (sq % 7) as i32)
}

/// squares at Chebyshev distance 1..=d from sq
fn ring(sq: usize, lo: i32, hi: i32) -> u64 {
    let (r, c) = rc(sq);
    let mut m = 0u64;
    for q in 0..49 {
        let (r2, c2) = rc(q);
        let d = (r - r2).abs().max((c - c2).abs());
        if d >= lo && d <= hi {
            m |= 1 << q;
        }
    }
    m
}

struct Geo {
    n1: [u64; 49],
    n2: [u64; 49],
    n12: [u64; 49],
}

impl Geo {
    fn new() -> Geo {
        let mut g = Geo { n1: [0; 49], n2: [0; 49], n12: [0; 49] };
        for s in 0..49 {
            g.n1[s] = ring(s, 1, 1);
            g.n2[s] = ring(s, 2, 2);
            g.n12[s] = ring(s, 1, 2);
        }
        g
    }
    fn dil(&self, m: u64, t: &[u64; 49]) -> u64 {
        let mut o = 0;
        let mut x = m;
        while x != 0 {
            let s = x.trailing_zeros() as usize;
            x &= x - 1;
            o |= t[s];
        }
        o
    }
}

fn bits(m: u64) -> impl Iterator<Item = usize> {
    let mut x = m;
    std::iter::from_fn(move || {
        if x == 0 {
            None
        } else {
            let s = x.trailing_zeros() as usize;
            x &= x - 1;
            Some(s)
        }
    })
}

/// 0 empty, 1 black, 2 white, 3 blocker / off board
fn cell(b: &Board, r: i32, c: i32) -> u32 {
    if !(0..7).contains(&r) || !(0..7).contains(&c) {
        return 3;
    }
    let s = (r * 7 + c) as u64;
    if b.occ[0] >> s & 1 != 0 {
        1
    } else if b.occ[1] >> s & 1 != 0 {
        2
    } else if b.blockers >> s & 1 != 0 {
        3
    } else {
        0
    }
}

fn win(b: &Board, r0: i32, c0: i32, h: i32) -> u32 {
    let mut code = 0u32;
    for r in r0..r0 + h {
        for c in c0..c0 + h {
            code = code * 3 + cell(b, r, c).min(2);
        }
    }
    code
}

type Fam = (&'static str, Box<dyn Fn(&Board, &Geo) -> Vec<u32>>);

fn families() -> Vec<Fam> {
    let mut v: Vec<Fam> = Vec::new();
    v.push(("stone-square (2x49)", Box::new(|b, _| {
        let mut o: Vec<u32> = bits(b.occ[0]).map(|s| (s * 2) as u32).chain(bits(b.occ[1]).map(|s| (s * 2 + 1) as u32)).collect();
        o.sort_unstable();
        o
    })));
    v.push(("2x2 windows, all 36", Box::new(|b, _| {
        let mut o = Vec::with_capacity(36);
        for r in 0..6 {
            for c in 0..6 {
                o.push(((r * 6 + c) * 81) as u32 + win(b, r, c, 2));
            }
        }
        o
    })));
    v.push(("2x2 windows, non-empty", Box::new(|b, _| {
        let mut o = Vec::new();
        for r in 0..6 {
            for c in 0..6 {
                let k = win(b, r, c, 2);
                if k != 0 {
                    o.push(((r * 6 + c) * 81) as u32 + k);
                }
            }
        }
        o
    })));
    v.push(("3x3 windows (25), non-empty", Box::new(|b, _| {
        let mut o = Vec::new();
        for r in 0..5 {
            for c in 0..5 {
                let k = win(b, r, c, 3);
                if k != 0 {
                    o.push(((r * 5 + c) * 19683) as u32 + k);
                }
            }
        }
        o
    })));
    v.push(("3x3 around each stone", Box::new(|b, _| {
        let mut o = Vec::new();
        for col in 0..2usize {
            for s in bits(b.occ[col]) {
                let (r, c) = rc(s);
                let mut k = 0u32;
                for dr in -1..=1 {
                    for dc in -1..=1 {
                        if dr != 0 || dc != 0 {
                            k = k * 4 + cell(b, r + dr, c + dc);
                        }
                    }
                }
                o.push(((s * 2 + col) as u32) * 65536 + k);
            }
        }
        o.sort_unstable();
        o
    })));
    v.push(("stone ring counts (=G3, abs colour)", Box::new(|b, g| {
        let mut o = Vec::new();
        for col in 0..2usize {
            for s in bits(b.occ[col]) {
                let own = (g.n1[s] & b.occ[col]).count_ones();
                let opp = (g.n1[s] & b.occ[1 - col]).count_ones();
                o.push(((s * 2 + col) as u32) * 81 + own * 9 + opp);
            }
        }
        o.sort_unstable();
        o
    })));
    for (name, d) in [("stone pairs, distance 1", 1usize), ("stone pairs, distance 2", 2), ("stone pairs, distance <=2", 3), ("stone pairs, distance 3", 4), ("stone pairs, ALL distances", 5)] {
        v.push((name, Box::new(move |b, g| {
            let occ = b.occ[0] | b.occ[1];
            let mut o = Vec::new();
            for a in bits(occ) {
                let ca = (b.occ[1] >> a & 1) as u32;
                let nb = match d {
                    1 => g.n1[a],
                    2 => g.n2[a],
                    3 => g.n12[a],
                    4 => ring(a, 3, 3),
                    _ => FULL & !(1 << a),
                };
                for q in bits(nb & occ & !((1u64 << (a + 1)) - 1)) {
                    let cq = (b.occ[1] >> q & 1) as u32;
                    o.push(((a * 49 + q) as u32) * 4 + ca * 2 + cq);
                }
            }
            o.sort_unstable();
            o
        })));
    }
    v.push(("threatened stones (capturable next move)", Box::new(|b, g| {
        let empty = FULL & !(b.occ[0] | b.occ[1] | b.blockers);
        let mut o = Vec::new();
        for col in 0..2usize {
            let land = g.dil(b.occ[1 - col], &g.n12) & empty;
            let thr = b.occ[col] & g.dil(land, &g.n1);
            o.extend(bits(thr).map(|s| (s * 2 + col) as u32));
        }
        o.sort_unstable();
        o
    })));
    v.push(("threats: landing sq x colour x clone/jump x gain", Box::new(|b, g| {
        let empty = FULL & !(b.occ[0] | b.occ[1] | b.blockers);
        let mut o = Vec::new();
        for col in 0..2usize {
            let cl = g.dil(b.occ[col], &g.n1) & empty;
            let jp = g.dil(b.occ[col], &g.n2) & empty & !cl;
            for (kind, m) in [(0u32, cl), (1, jp)] {
                for e in bits(m) {
                    let gain = (g.n1[e] & b.occ[1 - col]).count_ones();
                    if gain > 0 {
                        o.push((((e * 2 + col) as u32 * 2 + kind) * 5) + gain.min(4));
                    }
                }
            }
        }
        o.sort_unstable();
        o
    })));
    v.push(("ENGINE today: all groups G1-G6 (side 0)", Box::new(|b, _| {
        let mut out = [0u32; MAX_ACTIVE];
        let n = features(b, 0, ALL_GROUPS, &mut out);
        let mut o = out[..n].to_vec();
        o.sort_unstable();
        o
    })));
    v
}

fn sym_diff(a: &[u32], b: &[u32]) -> usize {
    let (mut i, mut j, mut d) = (0, 0, 0);
    while i < a.len() && j < b.len() {
        if a[i] == b[j] {
            i += 1;
            j += 1;
        } else if a[i] < b[j] {
            d += 1;
            i += 1;
        } else {
            d += 1;
            j += 1;
        }
    }
    d + (a.len() - i) + (b.len() - j)
}

fn board(r: &crate::util::Rec) -> Board {
    let (own, opp) = if r.stm == 0 { (r.occ0, r.occ1) } else { (r.occ1, r.occ0) };
    Board::from_occ(own, opp, 0, 0)
}

pub fn run(args: &[String]) -> ExitCode {
    let val = arg(args, "--val").expect("--val V.tdat");
    let n: usize = arg(args, "--n").map_or(3000, |v| v.parse().unwrap());
    let recs = crate::hcetrain::read_rows(&val, 600_000);
    let step = (recs.len() / n).max(1);
    let g = Geo::new();
    let fams = families();
    let mut ml = MoveList::new();
    let nf = fams.len();
    let mut act = vec![0u64; nf];
    let mut hist: Vec<Vec<u64>> = vec![vec![0u64; 4096]; nf];
    let (mut kids, mut pos, mut chg) = (0u64, 0usize, 0u64);
    let mut chg_hist = [0u64; 50];
    for r in recs.iter().step_by(step).take(n) {
        let b = board(r);
        b.legal_moves_dedup_into(&mut ml);
        if ml.is_empty() {
            continue;
        }
        pos += 1;
        let par: Vec<Vec<u32>> = fams.iter().map(|(_, f)| f(&b, &g)).collect();
        for m in ml.iter() {
            let c = b.make(m);
            let changed = ((b.occ[0] ^ c.occ[0]) | (b.occ[1] ^ c.occ[1])).count_ones() as usize;
            chg += changed as u64;
            chg_hist[changed.min(49)] += 1;
            for (k, (_, f)) in fams.iter().enumerate() {
                let cr = f(&c, &g);
                let d = sym_diff(&par[k], &cr);
                act[k] += cr.len() as u64;
                hist[k][d.min(4095)] += 1;
            }
            kids += 1;
        }
    }
    let q = |h: &Vec<u64>, p: f64| -> usize {
        let target = (p * kids as f64).ceil() as u64;
        let mut s = 0;
        for (i, &x) in h.iter().enumerate() {
            s += x;
            if s >= target {
                return i;
            }
        }
        h.len() - 1
    };
    let maxc = chg_hist.iter().rposition(|&x| x > 0).unwrap_or(0);
    println!("arch-locality: {pos} val positions ({val}), {kids} children; squares that change per move: mean {:.2}, p50 {}, p90 {}, p99 {}, max {maxc}", chg as f64 / kids as f64,
        q(&chg_hist.to_vec(), 0.5), q(&chg_hist.to_vec(), 0.9), q(&chg_hist.to_vec(), 0.99));
    println!("{:<48} {:>8} {:>9} {:>6} {:>5} {:>5} {:>5} {:>5}", "family (one accumulator, absolute colours)", "active", "diff", "ratio", "p50", "p90", "p99", "max");
    for (k, (name, _)) in fams.iter().enumerate() {
        let dsum: u64 = hist[k].iter().enumerate().map(|(i, &x)| i as u64 * x).sum();
        let mx = hist[k].iter().rposition(|&x| x > 0).unwrap_or(0);
        println!("{:<48} {:>8.1} {:>9.1} {:>6.3} {:>5} {:>5} {:>5} {:>5}", name, act[k] as f64 / kids as f64, dsum as f64 / kids as f64,
            dsum as f64 / act[k].max(1) as f64, q(&hist[k], 0.5), q(&hist[k], 0.9), q(&hist[k], 0.99), mx);
    }
    println!("diff = rows removed + added parent -> child (an incremental update's row ops); ratio < 1 = incremental beats a refresh on average. Counting only, no timing, no information content.");
    ExitCode::SUCCESS
}

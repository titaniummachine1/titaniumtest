//! `lab group-diff` (D1072, user: "a move changes much of the board, but that does not mean nothing can be updated
//! incrementally: isolate the parts worth updating incrementally"). For every legal child of real positions, per feature
//! group of the engine's big net (crate nnfeat: G1 square state, G2 anchor, G3 stone ring, G5 2-ply exchange, G6 global)
//! and per accumulator (absolute colour, as the engine would keep it): rows active in the child (= full-refresh adds) vs
//! rows that differ parent -> child (removed + added = incremental adds/subs). G6 counts as ONE row (the engine
//! pre-sums its 3 rows into one table entry). Prints per group: active, diff ops, ratio, share of children where the
//! incremental update is cheaper; and totals for: full refresh (today), everything incremental, a fixed split (groups
//! with ratio < 1 incremental, the rest re-added), and the adaptive per-node per-group minimum.
//! Usage: lab group-diff --val V.tdat [--n 20000]

use std::process::ExitCode;

use titanium::nnfeat::{features, G1, G2, G3, G5, G6, MAX_ACTIVE};
use titanium::{Board, MoveList};

use crate::util::arg;

const GROUPS: [(u32, &str); 5] = [(G1, "G1 square state"), (G2, "G2 anchor"), (G3, "G3 stone ring"), (G5, "G5 2-ply exchange"), (G6, "G6 global")];

fn board(r: &crate::util::Rec) -> Board {
    let (own, opp) = if r.stm == 0 { (r.occ0, r.occ1) } else { (r.occ1, r.occ0) };
    Board::from_occ(own, opp, 0, 0)
}

fn rows(b: &Board, side: usize, g: u32) -> Vec<u32> {
    let mut out = [0u32; MAX_ACTIVE];
    let n = features(b, side, g, &mut out);
    let mut v = out[..n].to_vec();
    if g == G6 {
        // one pre-summed row in the engine: identify it by the 3-row tuple
        v.sort_unstable();
        let key = v.iter().fold(0u32, |h, &x| h.wrapping_mul(1_000_003).wrapping_add(x));
        return vec![key];
    }
    v.sort_unstable();
    v
}

/// |a \ b| + |b \ a| for sorted lists
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

pub fn run(args: &[String]) -> ExitCode {
    let val = arg(args, "--val").expect("--val V.tdat");
    let n: usize = arg(args, "--n").map_or(20000, |v| v.parse().unwrap());
    let recs = crate::hcetrain::read_rows(&val, 600_000);
    let step = (recs.len() / n).max(1);
    let mut ml = MoveList::new();
    let mut act = [0u64; 5];
    let mut dif = [0u64; 5];
    let mut cheaper = [0u64; 5];
    let (mut kids, mut adaptive) = (0u64, 0u64);
    let mut per_kid: Vec<[u32; 10]> = Vec::new(); // (active, diff) per group, summed over the 2 accumulators
    let mut pos = 0usize;
    for r in recs.iter().step_by(step).take(n) {
        let b = board(r);
        b.legal_moves_dedup_into(&mut ml);
        if ml.is_empty() {
            continue;
        }
        pos += 1;
        let parent: Vec<Vec<Vec<u32>>> = (0..2).map(|s| GROUPS.iter().map(|&(g, _)| rows(&b, s, g)).collect()).collect();
        for m in ml.iter() {
            let c = b.make(m);
            let mut rec = [0u32; 10];
            for side in 0..2 {
                for (k, &(g, _)) in GROUPS.iter().enumerate() {
                    let cr = rows(&c, side, g);
                    let d = sym_diff(&parent[side][k], &cr);
                    act[k] += cr.len() as u64;
                    dif[k] += d as u64;
                    cheaper[k] += (d < cr.len()) as u64;
                    adaptive += d.min(cr.len()) as u64;
                    rec[2 * k] += cr.len() as u32;
                    rec[2 * k + 1] += d as u32;
                }
            }
            per_kid.push(rec);
            kids += 1;
        }
    }
    let accs = (2 * kids) as f64;
    println!("group-diff: {pos} val positions ({val}), {kids} children, per accumulator (2 per child), net features of the engine crate nnfeat");
    println!("{:<20} {:>9} {:>9} {:>7} {:>12}", "group", "active", "diff ops", "ratio", "incr cheaper");
    let mut incr_groups = 0u32;
    for (k, &(_, name)) in GROUPS.iter().enumerate() {
        let ratio = dif[k] as f64 / act[k].max(1) as f64;
        if ratio < 1.0 {
            incr_groups |= 1 << k;
        }
        println!("{:<20} {:>9.2} {:>9.2} {:>7.3} {:>11.1}%", name, act[k] as f64 / accs, dif[k] as f64 / accs, ratio, 100.0 * cheaper[k] as f64 / accs);
    }
    let full: u64 = act.iter().sum();
    let all_inc: u64 = dif.iter().sum();
    let split: u64 = (0..5).map(|k| if incr_groups >> k & 1 != 0 { dif[k] } else { act[k] }).sum();
    let names: Vec<&str> = (0..5).filter(|k| incr_groups >> k & 1 != 0).map(|k| GROUPS[k].1).collect();
    println!("row ops per accumulator: full refresh (today) {:.2} | everything incremental {:.2} ({:+.1}%) | fixed split [{}] incremental, rest re-added {:.2} ({:+.1}%) | adaptive per node+group {:.2} ({:+.1}%)",
        full as f64 / accs,
        all_inc as f64 / accs, 100.0 * (all_inc as f64 / full as f64 - 1.0),
        names.join(", "), split as f64 / accs, 100.0 * (split as f64 / full as f64 - 1.0),
        adaptive as f64 / accs, 100.0 * (adaptive as f64 / full as f64 - 1.0));
    // distribution of the fixed split's saving per child (how often it is a loss)
    let mut worse = 0u64;
    for r in &per_kid {
        let f: u32 = (0..5).map(|k| r[2 * k]).sum();
        let s: u32 = (0..5).map(|k| if incr_groups >> k & 1 != 0 { r[2 * k + 1] } else { r[2 * k] }).sum();
        worse += (s > f) as u64;
    }
    println!("fixed split costs MORE than a full refresh on {:.1}% of children", 100.0 * worse as f64 / kids.max(1) as f64);
    println!("NOT counted: feature extraction (still needed for the changed groups), the per-node accumulator copy, children whose parent has no accumulator (eval-cache hits), memory traffic per row");
    ExitCode::SUCCESS
}

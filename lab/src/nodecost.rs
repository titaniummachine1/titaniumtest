//! `lab node-cost` (D1051, user: "look at what eats the most performance"): where the time of a main-engine node goes.
//! (1) Search on real val positions at a fixed depth with the bundled policy head ON/OFF and the futility SmallNet ON/OFF:
//!     nodes (= what the component saves) and ns per node (= what it costs). (2) Micro costs of each evaluator on the same
//!     positions: big NNUE full forward (the engine is incremental, so this is an upper bound), policy logit per legal move,
//!     SmallNet bound per child, our fast HCE per eval. Single thread; run on an idle PC.
//! Usage: lab node-cost --val V.tdat [--n 300] [--depth 8] [--hce WEIGHTS.json]

use std::process::ExitCode;
use std::time::Instant;

use titanium::nnue::Nnue;
use titanium::{Board, MoveList, SearchLimits, Searcher};

use crate::util::arg;

fn board(r: &crate::util::Rec) -> Board {
    let (own, opp) = if r.stm == 0 { (r.occ0, r.occ1) } else { (r.occ1, r.occ0) };
    Board::from_occ(own, opp, 0, 0)
}

pub fn run(args: &[String]) -> ExitCode {
    let val = arg(args, "--val").expect("--val V.tdat");
    let n: usize = arg(args, "--n").map_or(300, |v| v.parse().unwrap());
    let depth: u8 = arg(args, "--depth").map_or(8, |v| v.parse().unwrap());
    let rows = crate::hcetrain::read_rows(&val, 600_000);
    let step = (rows.len() / n).max(1);
    let boards: Vec<Board> = rows.iter().step_by(step).take(n).map(board).filter(|b| b.has_moves(b.turn)).collect();
    println!("node-cost: {} val positions, fixed depth {depth}, Botzone search, TT 2^22, single thread", boards.len());

    // (1) search with components on/off
    let pol = titanium::policy::global();
    let mut base_nodes = 0u64;
    for (pol_on, sn_on, name) in [(true, true, "main (policy ON, SmallNet ON)"), (false, true, "policy OFF"), (true, false, "SmallNet OFF"), (false, false, "both OFF")] {
        titanium::search::set_fut_sn(sn_on);
        let (mut nodes, mut ns) = (0u64, 0u128);
        for b in &boards {
            let mut s = Searcher::with_tt_bits(22);
            if !pol_on {
                s.set_policy(None, 0.0, 99);
            } else if let Some(p) = pol {
                let _ = p; // bundled policy stays as configured
            }
            let t = Instant::now();
            let r = s.search_bz(b, &SearchLimits { time: None, max_nodes: None, max_depth: depth });
            ns += t.elapsed().as_nanos();
            nodes += r.nodes;
        }
        if base_nodes == 0 {
            base_nodes = nodes;
        }
        println!(
            "  {name:32} nodes {nodes:>12} ({:.3}x main) | {:.0} ns/node | {:.0} knps | total {:.1} s",
            nodes as f64 / base_nodes as f64,
            ns as f64 / nodes as f64,
            nodes as f64 / (ns as f64 / 1e9) / 1000.0,
            ns as f64 / 1e9
        );
    }
    titanium::search::set_fut_sn(true);

    // (2) micro costs
    let net = Nnue::bundled();
    let mut ml = MoveList::new();
    let reps = 5;
    let t = Instant::now();
    let mut acc = 0i64;
    for _ in 0..reps {
        for b in &boards {
            acc += net.eval(b, 0) as i64;
        }
    }
    let nn_ns = t.elapsed().as_nanos() as f64 / (reps * boards.len()) as f64;
    let (mut moves, mut pol_ns) = (0usize, 0f64);
    if let Some(p) = pol {
        let mut kc = titanium::policy::KCache::new();
        let t = Instant::now();
        for _ in 0..reps {
            for b in &boards {
                b.legal_moves_dedup_into(&mut ml);
                for m in ml.iter() {
                    acc += p.logit_k(b, m, &mut kc) as i64;
                    moves += 1;
                }
            }
        }
        pol_ns = t.elapsed().as_nanos() as f64 / moves as f64;
    }
    let sn = titanium::smallnet::SmallNet::builtin();
    let mut kids = 0usize;
    let t = Instant::now();
    for _ in 0..reps {
        for b in &boards {
            b.legal_moves_dedup_into(&mut ml);
            for m in ml.iter() {
                let c = b.make(m);
                acc += sn.bound(c.me(), c.opp(), 0, 0, 0) as i64;
                kids += 1;
            }
        }
    }
    let sn_ns = t.elapsed().as_nanos() as f64 / kids as f64;
    let mk = {
        let t = Instant::now();
        let mut k = 0usize;
        for _ in 0..reps {
            for b in &boards {
                b.legal_moves_dedup_into(&mut ml);
                for m in ml.iter() {
                    let c = b.make(m);
                    acc += c.me().count_ones() as i64;
                    k += 1;
                }
            }
        }
        t.elapsed().as_nanos() as f64 / k as f64
    };
    let mut line = format!(
        "  micro (per call): big NNUE full forward {nn_ns:.0} ns | policy logit {pol_ns:.1} ns/move (x {:.1} moves = {:.0} ns/node) | SmallNet bound (full, not the engine's parent-acc path) {sn_ns:.0} ns/child incl. make; make alone {mk:.0} ns",
        moves as f64 / (reps * boards.len()) as f64,
        pol_ns * moves as f64 / (reps * boards.len()) as f64
    );
    if let Some(hp) = arg(args, "--hce") {
        let h = crate::hceeval::HceEval::load(&hp);
        let t = Instant::now();
        let mut s = 0f32;
        for _ in 0..reps {
            for b in &boards {
                s += h.eval(b.me(), b.opp());
            }
        }
        line.push_str(&format!(" | fast HCE {:.0} ns/eval", t.elapsed().as_nanos() as f64 / (reps * boards.len()) as f64));
        acc += s as i64;
    }
    println!("{line}");
    println!("  (checksum {acc})");
    ExitCode::SUCCESS
}

/// `lab incr-stats --val V.tdat [--n 20000]` (D1056, user: "make the HCE incremental: a move only touches one part"):
/// per legal move of val positions, how many squares change state and how many of the 80 pair placements
/// (two 2x2 blocks each) contain a changed square = the work of an incremental pair-table update.
pub fn incr_stats(args: &[String]) -> ExitCode {
    let val = arg(args, "--val").expect("--val V.tdat");
    let n: usize = arg(args, "--n").map_or(20000, |v| v.parse().unwrap());
    let rows = crate::hcetrain::read_rows(&val, 600_000);
    let step = (rows.len() / n).max(1);
    let p = crate::pairs::Pairs::new();
    let (fa, fb) = p.first_second();
    let blk = |s: u8| -> u64 { let s = s as u64; (1 << s) | (1 << (s + 1)) | (1 << (s + 7)) | (1 << (s + 8)) };
    let masks: Vec<u64> = (0..crate::pairs::PLACEMENTS).map(|q| blk(fa[q]) | blk(fb[q])).collect();
    let mut ml = MoveList::new();
    let (mut moves, mut sq, mut pl, mut hist) = (0u64, 0u64, 0u64, [0u64; 81]);
    for r in rows.iter().step_by(step).take(n) {
        let b = board(r);
        b.legal_moves_dedup_into(&mut ml);
        for m in ml.iter() {
            let c = b.make(m);
            let changed = (b.me() ^ c.opp()) | (b.opp() ^ c.me());
            let k = masks.iter().filter(|&&mk| mk & changed != 0).count();
            moves += 1;
            sq += changed.count_ones() as u64;
            pl += k as u64;
            hist[k] += 1;
        }
    }
    let pct = |lo: usize, hi: usize| 100.0 * hist[lo..=hi].iter().sum::<u64>() as f64 / moves as f64;
    println!(
        "incr-stats: {moves} moves of {n} val positions: changed squares {:.2} per move; pair placements touched {:.1} of {} per move ({:.0}%); share of moves touching <=20 placements {:.1}%, 21-40 {:.1}%, >40 {:.1}%",
        sq as f64 / moves as f64, pl as f64 / moves as f64, crate::pairs::PLACEMENTS, 100.0 * pl as f64 / (moves * crate::pairs::PLACEMENTS as u64) as f64,
        pct(0, 20), pct(21, 40), pct(41, 80)
    );
    ExitCode::SUCCESS
}

/// `lab weak-cost --val V.tdat [--n 20000]` (D1062, user: "can the weaknesses be made cheaper to derive?"): cost of the
/// per-square weakness information as BITBOARD planes (4 bit-planes = a 0..15 count per square, for every square at once:
/// box2 reach + victim planes, the inputs of the dynamic PST and the soft weaknesses) vs the per-stone table extraction.
pub fn weak_cost(args: &[String]) -> ExitCode {
    use crate::hcefit::box2;
    use crate::util::victim_planes;
    let val = arg(args, "--val").expect("--val V.tdat");
    let n: usize = arg(args, "--n").map_or(20000, |v| v.parse().unwrap());
    let rows = crate::hcetrain::read_rows(&val, 600_000);
    let step = (rows.len() / n).max(1);
    let boards: Vec<Board> = rows.iter().step_by(step).take(n).map(board).collect();
    let reps = 20;
    let t = Instant::now();
    let mut acc = 0u64;
    for _ in 0..reps {
        for b in std::hint::black_box(&boards) {
            let (me, op, e) = (b.me(), b.opp(), b.empty());
            let (bm, bo) = (box2(me), box2(op));
            let (lo, lm) = (bo & e, bm & e);
            let near_e = box2(e);
            let p = [victim_planes(lo), victim_planes(lm), victim_planes(op & near_e), victim_planes(me & near_e)];
            for q in std::hint::black_box(&p) {
                acc = acc.wrapping_add(q[0].count_ones() as u64 + q[1].count_ones() as u64 * 2 + q[2].count_ones() as u64 * 4 + q[3].count_ones() as u64 * 8);
            }
        }
    }
    let planes = t.elapsed().as_nanos() as f64 / (reps * boards.len()) as f64;
    let ring = crate::util::ring1();
    let m = crate::hcecmp::masks();
    let mut out = Vec::new();
    let t = Instant::now();
    for _ in 0..reps {
        for b in &boards {
            crate::hcecmp::feats(7, b, &ring, &m, &mut out);
            acc ^= out.len() as u64;
        }
    }
    let table = t.elapsed().as_nanos() as f64 / (reps * boards.len()) as f64;
    println!("weak-cost: {} positions: weakness bit-planes for ALL squares (4 planes x 4 inputs) {planes:.1} ns; per-stone dynamic-PST feature extraction (lab code, allocating) {table:.1} ns (checksum {acc})", boards.len());
    ExitCode::SUCCESS
}

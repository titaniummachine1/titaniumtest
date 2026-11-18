//! `lab feat-bench [--n 200000] [--rounds 5]` (D1157, user: "the ~590 ns of feature gathering, did you try optimising
//! it"): time the champion's real feature extraction (titanium 0120fd12 `nnfeat::both_into`, both perspectives in one
//! joint pass) per input group on real positions, single thread, next to the full eval. Groups 0 = the shared masks +
//! anchors only (computed whatever the group set); each group alone = masks + that group; ALL = what the engine runs.
//! Median of the rounds. Run it ONLY on a quiet PC (nothing else training or generating).
use crate::hcetrain::read_rows;
use crate::util::arg;
use std::process::ExitCode;
use titanium::nnfeat::{both_into, ALL_GROUPS, G1, G2, G3, G5, G6, MAX_ACTIVE};
use titanium::nnue::Nnue;
use titanium::Board;

pub fn run(args: &[String]) -> ExitCode {
    let n = arg(args, "--n").and_then(|v| v.parse().ok()).unwrap_or(200_000usize);
    let rounds = arg(args, "--rounds").and_then(|v| v.parse().ok()).unwrap_or(5usize);
    let rows = read_rows("C:/gitProjects/titanium-ataxx/data/nnue/packed/val_v9_clean.tdat", n);
    let boards: Vec<Board> = rows.iter().map(|r| { let (me, op) = crate::extval::own_opp(r); Board::from_occ(me, op, 0, 0) }).collect();
    let mut us = [0u32; MAX_ACTIVE];
    let mut them = [0u32; MAX_ACTIVE];
    let sets: [(&str, u32); 7] = [("masks+anchors only", 0), ("G1 square state", G1), ("G2 anchor", G2), ("G3 stone ring", G3), ("G5 2-ply exchange", G5), ("G6 global", G6), ("ALL (engine)", ALL_GROUPS)];
    println!("feat-bench: {} positions (val_v9_clean), single thread, both perspectives per call, median of {rounds} rounds (titanium 0120fd12)", boards.len());
    let med = |mut v: Vec<f64>| { v.sort_by(|a, b| a.partial_cmp(b).unwrap()); v[v.len() / 2] };
    let mut sink = 0usize;
    for (name, g) in sets {
        let mut ts = Vec::new();
        for _ in 0..rounds {
            let t0 = std::time::Instant::now();
            for b in &boards {
                let (a, c) = both_into(b, g, &mut us, &mut them);
                sink = sink.wrapping_add(a + c + us[0] as usize);
            }
            ts.push(t0.elapsed().as_nanos() as f64 / boards.len() as f64);
        }
        let all = ts.iter().map(|t| format!("{t:.0}")).collect::<Vec<_>>().join(",");
        println!("{name:<22} {:>7.0} ns/position  (rounds: {all})", med(ts));
    }
    let net = Nnue::bundled();
    let mut ts = Vec::new();
    for _ in 0..rounds {
        let t0 = std::time::Instant::now();
        for b in &boards {
            sink = sink.wrapping_add(net.eval(b, 0) as usize);
        }
        ts.push(t0.elapsed().as_nanos() as f64 / boards.len() as f64);
    }
    println!("{:<22} {:>7.0} ns/position  (full champion eval incl. extraction)", "full eval", med(ts));
    std::hint::black_box(sink);
    ExitCode::SUCCESS
}

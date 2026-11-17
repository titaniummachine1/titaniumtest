//! `lab net-scale --a A.tnn --b B.tnn --val V.tdat [--n 20000]` (D1114): is net A's cp on the same SCALE as net B's?
//! The search's pruning margins are fixed cp numbers tuned on the champion; a net that is more (or less) confident
//! makes them misfire. On val positions and all their children: least-squares slope of A on B (through 0), mean |cp|,
//! correlation, by empties band, on the clipped range |cp| <= 1500.

use std::process::ExitCode;

use titanium::nnue::Nnue;
use titanium::{Board, MoveList};

use crate::util::arg;

fn board(r: &crate::util::Rec) -> Board {
    let (own, opp) = if r.stm == 0 { (r.occ0, r.occ1) } else { (r.occ1, r.occ0) };
    Board::from_occ(own, opp, 0, 0)
}

pub fn run(args: &[String]) -> ExitCode {
    let a = Nnue::load(&arg(args, "--a").expect("--a")).expect("load a");
    let b = Nnue::load(&arg(args, "--b").expect("--b")).expect("load b");
    let val = arg(args, "--val").expect("--val");
    let n: usize = arg(args, "--n").map_or(20000, |v| v.parse().unwrap());
    let recs = crate::hcetrain::read_rows(&val, 600_000);
    let step = (recs.len() / n).max(1);
    // bands: 0-6, 7-12, 13-24, 25-36, 37-49 empties; [sum ab, sum bb, sum aa, |a|, |b|, count, dis_a, dis_b]
    let mut acc = [[0f64; 8]; 5];
    let band = |e: u32| -> usize { match e { 0..=6 => 0, 7..=12 => 1, 13..=24 => 2, 25..=36 => 3, _ => 4 } };
    let mut ml = MoveList::new();
    for r in recs.iter().step_by(step).take(n) {
        let p = board(r);
        p.legal_moves_dedup_into(&mut ml);
        let mut boards = vec![p];
        boards.extend(ml.iter().map(|m| p.make(m)));
        for x in &boards {
            let (ea, da) = a.eval_with_dis(x, 0);
            let (eb, db) = b.eval_with_dis(x, 0);
            if ea.abs() > 1500 || eb.abs() > 1500 {
                continue;
            }
            let (fa, fb) = (ea as f64, eb as f64);
            let k = band(x.empty().count_ones());
            let s = &mut acc[k];
            s[0] += fa * fb;
            s[1] += fb * fb;
            s[2] += fa * fa;
            s[3] += fa.abs();
            s[4] += fb.abs();
            s[5] += 1.0;
            s[6] += da as f64;
            s[7] += db as f64;
        }
    }
    let names = ["empties 0-6", "empties 7-12", "empties 13-24", "empties 25-36", "empties 37-49"];
    println!("net-scale: A vs B on {n} val positions + children ({val}), |cp| <= 1500 on both");
    let mut t = [0f64; 8];
    for (k, s) in acc.iter().enumerate() {
        for i in 0..8 {
            t[i] += s[i];
        }
        let c = s[5].max(1.0);
        println!("{:<14} n {:>8} | slope A/B {:.3} | mean|cp| A {:.0} B {:.0} | corr {:.3} | mean dis A {:.0} B {:.0}", names[k], s[5], s[0] / s[1], s[3] / c, s[4] / c, s[0] / (s[1] * s[2]).sqrt(), s[6] / c, s[7] / c);
    }
    let c = t[5].max(1.0);
    println!("{:<14} n {:>8} | slope A/B {:.3} | mean|cp| A {:.0} B {:.0} | corr {:.3} | mean dis A {:.0} B {:.0}", "ALL", t[5], t[0] / t[1], t[3] / c, t[4] / c, t[0] / (t[1] * t[2]).sqrt(), t[6] / c, t[7] / c);
    ExitCode::SUCCESS
}

//! `lab bound-cmp` (D1051, user: "maybe the HCE can replace the SmallNet, or an HCE-ML hybrid under half a
//! microsecond"): how tight a LOWER BOUND on the big net's child eval each cheap predictor gives, on the SmallNet's own
//! job. Children of val positions; truth = champion NNUE eval of the child (child's view). Predictors (child's view):
//! SmallNet bound (engine inputs: parent NNUE cp + child hand eval), our fast HCE, the hand eval, and HYBRID = per bucket
//! least squares of truth on [HCE, -parent cp, hand, 1] (a learned linear combination, ~HCE cost). Every predictor gets
//! a per-bucket shift so 2% of the FIT half are violated (bound above the truth), then on the HELD-OUT half: violation
//! rate and mean slack (truth - bound; smaller = tighter = more children pruned without a big-net eval).
//! Usage: lab bound-cmp --val V.tdat --hce W.json [--n 20000]

use std::process::ExitCode;
use std::time::Instant;

use titanium::nnue::Nnue;
use titanium::{Board, MoveList};

use crate::util::arg;

const NBK: usize = 8;
const TAU: f64 = 0.02;

fn solve<const N: usize, const M: usize>(a: &mut [[f64; M]; N]) -> [f64; N] {
    for c in 0..N {
        let p = (c..N).max_by(|&i, &j| a[i][c].abs().partial_cmp(&a[j][c].abs()).unwrap()).unwrap();
        a.swap(c, p);
        let d = if a[c][c].abs() < 1e-12 { 1e-12 } else { a[c][c] };
        for k in c..M {
            a[c][k] /= d;
        }
        for r in 0..N {
            if r != c {
                let f = a[r][c];
                for k in c..M {
                    a[r][k] -= f * a[c][k];
                }
            }
        }
    }
    std::array::from_fn(|i| a[i][N])
}

pub fn run(args: &[String]) -> ExitCode {
    let val = arg(args, "--val").expect("--val V.tdat");
    let hp = arg(args, "--hce").expect("--hce WEIGHTS.json");
    let n: usize = arg(args, "--n").map_or(20000, |v| v.parse().unwrap());
    let rows = crate::hcetrain::read_rows(&val, 600_000);
    let step = (rows.len() / n).max(1);
    let net = Nnue::bundled();
    let sn = titanium::smallnet::SmallNet::builtin();
    let hce = crate::hceeval::HceEval::load(&hp);
    // sample: (bucket, truth, sn, hce, -parent, hand, fit-half?)
    let mut s: Vec<(usize, f64, [f64; 4], bool)> = Vec::new();
    let mut ml = MoveList::new();
    let (mut t_sn, mut t_hce, mut t_hand, mut cnt) = (0u128, 0u128, 0u128, 0u64);
    for (pi, r) in rows.iter().step_by(step).take(n).enumerate() {
        let (own, opp) = if r.stm == 0 { (r.occ0, r.occ1) } else { (r.occ1, r.occ0) };
        let b = Board::from_occ(own, opp, 0, 0);
        if !b.has_moves(b.turn) {
            continue;
        }
        let parent = net.eval(&b, 0);
        b.legal_moves_dedup_into(&mut ml);
        for m in ml.iter() {
            let c = b.make(m);
            let truth = net.eval(&c, 0) as f64;
            let t0 = Instant::now();
            let hand = titanium::search::evaluate(&c, None, false);
            let t1 = Instant::now();
            let lo = sn.bound(c.me(), c.opp(), c.blockers, parent, hand);
            let t2 = Instant::now();
            let h = hce.eval(c.me(), c.opp()) as f64;
            let t3 = Instant::now();
            t_hand += (t1 - t0).as_nanos();
            t_sn += (t2 - t1).as_nanos();
            t_hce += (t3 - t2).as_nanos();
            cnt += 1;
            let bk = (c.empty().count_ones() as usize * NBK / 50).min(NBK - 1);
            s.push((bk, truth, [lo as f64, h, -(parent as f64), hand as f64], pi % 2 == 0));
        }
    }
    println!(
        "bound-cmp: {} children of {} val positions (fit = even positions, scored on odd); per-call ns (incl. timer): hand eval {:.0}, SmallNet bound (full acc, the engine reuses the parent acc) {:.0}, fast HCE {:.0}",
        s.len(), n, t_hand as f64 / cnt as f64, t_sn as f64 / cnt as f64, t_hce as f64 / cnt as f64
    );
    // hybrid fit per bucket on the fit half: truth ~ a*hce + b*(-parent) + c*hand + d
    let mut coef = [[0f64; 4]; NBK];
    for (bk, co) in coef.iter_mut().enumerate() {
        let mut a = [[0f64; 5]; 4];
        for (b_, t, x, fit) in &s {
            if *b_ != bk || !fit {
                continue;
            }
            let f = [x[1], x[2], x[3], 1.0];
            for i in 0..4 {
                for j in 0..4 {
                    a[i][j] += f[i] * f[j];
                }
                a[i][4] += f[i] * t;
            }
        }
        for (i, row) in a.iter_mut().enumerate() {
            row[i] += 1e-3;
        }
        *co = solve::<4, 5>(&mut a);
    }
    // SN + HCE: does the HCE add information ON TOP of the SmallNet? truth ~ a*sn + b*hce + c*(-parent) + d*hand + e
    let mut coef5 = [[0f64; 5]; NBK];
    for (bk, co) in coef5.iter_mut().enumerate() {
        let mut a = [[0f64; 6]; 5];
        for (b_, t, x, fit) in &s {
            if *b_ != bk || !fit {
                continue;
            }
            let f = [x[0], x[1], x[2], x[3], 1.0];
            for i in 0..5 {
                for j in 0..5 {
                    a[i][j] += f[i] * f[j];
                }
                a[i][5] += f[i] * t;
            }
        }
        for (i, row) in a.iter_mut().enumerate() {
            row[i] += 1e-3;
        }
        *co = solve::<5, 6>(&mut a);
    }
    let pred = |k: usize, bk: usize, x: &[f64; 4]| -> f64 {
        match k {
            0 => x[0],
            1 => x[1],
            2 => x[3],
            3 => coef[bk][0] * x[1] + coef[bk][1] * x[2] + coef[bk][2] * x[3] + coef[bk][3],
            _ => coef5[bk][0] * x[0] + coef5[bk][1] * x[1] + coef5[bk][2] * x[2] + coef5[bk][3] * x[3] + coef5[bk][4],
        }
    };
    let names = ["SmallNet (main FA1FE34A)", "fast HCE (D991)", "hand eval", "HYBRID lin(HCE, parent, hand)", "SN + HCE lin(SN, HCE, parent, hand)"];
    for (k, name) in names.iter().enumerate() {
        // per-bucket shift: TAU quantile of (truth - pred) on the fit half
        let mut shift = [0f64; NBK];
        for (bk, sh) in shift.iter_mut().enumerate() {
            let mut r: Vec<f64> = s.iter().filter(|e| e.0 == bk && e.3).map(|e| e.1 - pred(k, bk, &e.2)).collect();
            if r.is_empty() {
                continue;
            }
            r.sort_by(|a, b| a.partial_cmp(b).unwrap());
            *sh = r[((r.len() as f64 * TAU) as usize).min(r.len() - 1)];
        }
        let (mut viol, mut slack, mut m) = (0u64, 0f64, 0u64);
        let mut per = [(0f64, 0u64); NBK];
        for (bk, t, x, fit) in &s {
            if *fit {
                continue;
            }
            let lb = pred(k, *bk, x) + shift[*bk];
            let d = t - lb;
            viol += (d < 0.0) as u64;
            slack += d;
            per[*bk].0 += d;
            per[*bk].1 += 1;
            m += 1;
        }
        let pb: Vec<String> = per.iter().map(|(a, c)| if *c > 0 { format!("{:.0}", a / *c as f64) } else { "-".into() }).collect();
        println!("  {name:32} held-out violation {:.2}% | mean slack {:.1} cp | by empties bucket (0 = full board) [{}]", 100.0 * viol as f64 / m as f64, slack / m as f64, pb.join(" "));
    }
    ExitCode::SUCCESS
}

//! `lab sibling-corr` (D1279/D1280, user idea: "train like a policy: eval all moves, loss on the move ordering"):
//! step 1 of the test plan. For N val positions every child is searched ONCE by a fixed teacher (plain alpha-beta,
//! `--teacher` net, `--teacher-nodes` per child); each net of the Elo panel then ranks the same children by its own 1-ply
//! eval. Metrics per net: top-1 agreement with the teacher's best child, mean regret (cp), pairwise concordance. They are
//! correlated with the net's measured `eval-gate 5k` Elo and compared with the validation-loss correlation
//! (r = +0.566, D1273). The teacher child values are written to a TSV (reusable as listwise training targets, step 2).
//! Caveat printed in the result: the teacher is the champion by default, so champion-family nets are favoured.

use std::fs::File;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::Path;
use std::process::ExitCode;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use titanium::Board;

use crate::evalgate::{ordered, search_value, terminal, Ev};
use crate::util::{arg, Rec, REC};

const CLAMP: f32 = 1500.0;

fn pearson(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len() as f64;
    let (mx, my) = (x.iter().sum::<f64>() / n, y.iter().sum::<f64>() / n);
    let (mut sxy, mut sxx, mut syy) = (0.0, 0.0, 0.0);
    for i in 0..x.len() {
        sxy += (x[i] - mx) * (y[i] - my);
        sxx += (x[i] - mx).powi(2);
        syy += (y[i] - my).powi(2);
    }
    if sxx == 0.0 || syy == 0.0 {
        0.0
    } else {
        sxy / (sxx * syy).sqrt()
    }
}

fn ranks(x: &[f64]) -> Vec<f64> {
    let mut idx: Vec<usize> = (0..x.len()).collect();
    idx.sort_by(|&a, &b| x[a].partial_cmp(&x[b]).unwrap());
    let mut r = vec![0.0; x.len()];
    for (k, &i) in idx.iter().enumerate() {
        r[i] = k as f64;
    }
    r
}

fn spearman(x: &[f64], y: &[f64]) -> f64 {
    pearson(&ranks(x), &ranks(y))
}

fn read_positions(path: &str, want: usize, lo: u32, hi: u32) -> Vec<Board> {
    let mut f = File::open(path).expect("open val tdat");
    let len = f.seek(SeekFrom::End(0)).unwrap() as usize;
    let total = len / REC;
    let stride = (total / (want * 4)).max(1);
    let mut out = Vec::with_capacity(want);
    let mut buf = [0u8; REC];
    let mut idx = stride / 2;
    while idx < total && out.len() < want {
        f.seek(SeekFrom::Start((idx * REC) as u64)).unwrap();
        if f.read_exact(&mut buf).is_err() {
            break;
        }
        let r = Rec::from_bytes(&buf);
        idx += stride;
        if r.occ0 & r.occ1 != 0 {
            continue;
        }
        let empties = 49 - (r.occ0 | r.occ1).count_ones();
        if empties < lo || empties > hi {
            continue;
        }
        let (own, opp) = if r.stm == 0 { (r.occ0, r.occ1) } else { (r.occ1, r.occ0) };
        let b = Board::from_occ(own, opp, 0, 0);
        if b.bz_over() || ordered(&b).len() < 2 {
            continue;
        }
        out.push(b);
    }
    out
}

fn child_value(ev: &Ev, c: &Board) -> f32 {
    // value of the child from the PARENT's point of view
    let v = if c.bz_over() { terminal(c) as f32 } else { ev.eval_f(c) };
    -(v.clamp(-CLAMP, CLAMP))
}

pub fn run(args: &[String]) -> ExitCode {
    let t0 = Instant::now();
    let val = arg(args, "--val").unwrap_or_else(|| "C:/gitProjects/titanium-ataxx/data/nnue/packed/val_v13tail.tdat".into());
    let n_pos: usize = arg(args, "--n-pos").map_or(3000, |v| v.parse().unwrap());
    let tnodes: u64 = arg(args, "--teacher-nodes").map_or(3000, |v| v.parse().unwrap());
    let threads: usize = arg(args, "--threads").map_or(7, |v| v.parse().unwrap());
    let lo: u32 = arg(args, "--min-empties").map_or(6, |v| v.parse().unwrap());
    let hi: u32 = arg(args, "--max-empties").map_or(44, |v| v.parse().unwrap());
    let teacher_path = arg(args, "--teacher").unwrap_or_else(|| "C:/gitProjects/mainpromo-ataxx/data/nnue/nnue_main.tnn".into());
    let out_tsv = arg(args, "--out").unwrap_or_else(|| "C:/gitProjects/titanium-ataxx/logs/arch/d1280_sibling_teacher.tsv".into());

    // (label, path, consensus eval-gate 5k Elo vs champion, is_nnue) = the loss-elo-corr panel (D1273)
    let specs: [(&str, &str, f64); 8] = [
        ("nn_v14muon", "C:/gitProjects/titanium-ataxx/runs/nn_v14muon/nn_v14muon/best.tnn", 24.5),
        ("WS_SAFECAL", "C:/gitProjects/titanium-ataxx/nets/ablate/ws_v13anneal_safecal.tnn", 11.3),
        ("WS_SAFE", "C:/gitProjects/titanium-ataxx/nets/ablate/ws_v13anneal_safe.tnn", 10.0),
        ("nn_v13anneal", "C:/gitProjects/titanium-ataxx/runs/nn_v13anneal/nn_v13anneal/best.tnn", 9.1),
        ("Champion", "C:/gitProjects/mainpromo-ataxx/data/nnue/nnue_main.tnn", 0.0),
        ("nn_v13sat", "C:/gitProjects/titanium-ataxx/runs/nn_v13sat/nn_v13sat/best.tnn", -0.9),
        ("nn_v11bsat", "C:/gitProjects/titanium-ataxx/runs/nn_v11bsat/nn_v11bsat/best.tnn", -11.3),
        ("levatt_ep4", "C:/gitProjects/titanium-ataxx/logs/hce/d1203_levatt_attn_levatt.bin", -13.9),
    ];
    let mut names = Vec::new();
    let mut elos = Vec::new();
    let mut evs = Vec::new();
    for (name, path, e) in specs {
        if Path::new(path).exists() {
            names.push(name);
            elos.push(e);
            evs.push(Ev::load(path));
        } else {
            println!("[skip] missing net: {name} ({path})");
        }
    }
    if evs.len() < 4 {
        eprintln!("sibling-corr: need at least 4 nets on disk");
        return ExitCode::FAILURE;
    }
    let teacher = Ev::load(&teacher_path);
    let pos = read_positions(&val, n_pos, lo, hi);
    println!("sibling-corr: {} nets, {} positions ({}..{} empties) from {val}; teacher {teacher_path} @ {tnodes} nodes/child", evs.len(), pos.len(), lo, hi);

    // 1. teacher value of every child (parent's point of view), parallel over positions
    let next = AtomicUsize::new(0);
    let mut tvals: Vec<Vec<f32>> = vec![Vec::new(); pos.len()];
    {
        let slots: Vec<std::sync::Mutex<&mut Vec<f32>>> = tvals.iter_mut().map(std::sync::Mutex::new).collect();
        std::thread::scope(|s| {
            for _ in 0..threads {
                s.spawn(|| loop {
                    // LAWS A14: STOP in the output folder or C:/gitProjects/STOP_ALL ends the run (nothing is half-written)
                    if crate::stopctl::requested(std::path::Path::new(&out_tsv).parent()).is_some() {
                        break;
                    }
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    if i >= pos.len() {
                        break;
                    }
                    let b = &pos[i];
                    let vals: Vec<f32> = ordered(b)
                        .into_iter()
                        .map(|m| {
                            let c = b.make(m);
                            -(search_value(&teacher, &c, tnodes) as f32).clamp(-CLAMP, CLAMP)
                        })
                        .collect();
                    **slots[i].lock().unwrap() = vals;
                });
            }
        });
    }
    if let Some(f) = crate::stopctl::requested(std::path::Path::new(&out_tsv).parent()) {
        println!("sibling-corr: STOPPED by {f} during the teacher pass (no result written; rerun to compute)");
        return ExitCode::SUCCESS;
    }
    println!("teacher child values done in {:.0}s", t0.elapsed().as_secs_f64());
    if let Ok(mut f) = File::create(&out_tsv) {
        for (b, v) in pos.iter().zip(tvals.iter()) {
            let s: Vec<String> = v.iter().map(|x| format!("{x:.0}")).collect();
            let _ = writeln!(f, "{}\t{}\t{}\t{}", b.me(), b.opp(), b.turn, s.join(" "));
        }
        println!("teacher values written to {out_tsv}");
    }

    // 2. per-net metrics on the non-trivial positions (teacher spread >= 50 cp)
    struct M {
        top1: f64,
        top3: f64,
        regret: f64,
        conc: f64,
        by: [(f64, f64, usize); 4], // (top1 sum, regret sum, n) per empties group
    }
    let group = |e: u32| -> usize {
        if e <= 14 {
            0
        } else if e <= 24 {
            1
        } else if e <= 34 {
            2
        } else {
            3
        }
    };
    let mut res: Vec<M> = Vec::new();
    let mut used = 0usize;
    for ev in &evs {
        let (mut t1, mut t3, mut rg, mut cc, mut cn) = (0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64);
        let mut by = [(0.0, 0.0, 0usize); 4];
        used = 0;
        for (b, t) in pos.iter().zip(tvals.iter()) {
            let tmax = t.iter().cloned().fold(f32::MIN, f32::max);
            let tmin = t.iter().cloned().fold(f32::MAX, f32::min);
            if tmax - tmin < 50.0 {
                continue;
            }
            used += 1;
            let kids: Vec<Board> = ordered(b).into_iter().map(|m| b.make(m)).collect();
            let n: Vec<f32> = kids.iter().map(|c| child_value(ev, c)).collect();
            let mut order: Vec<usize> = (0..n.len()).collect();
            order.sort_by(|&a, &c| n[c].partial_cmp(&n[a]).unwrap());
            let pick = order[0];
            let g = group(49 - b.me().count_ones() - b.opp().count_ones());
            let reg = (tmax - t[pick]) as f64;
            let hit1 = (t[pick] >= tmax - 1.0) as u8 as f64;
            t1 += hit1;
            t3 += order.iter().take(3).any(|&i| t[i] >= tmax - 1.0) as u8 as f64;
            rg += reg;
            by[g].0 += hit1;
            by[g].1 += reg;
            by[g].2 += 1;
            let (mut ok, mut tot) = (0usize, 0usize);
            for i in 0..n.len() {
                for j in (i + 1)..n.len() {
                    if (t[i] - t[j]).abs() > 20.0 {
                        tot += 1;
                        if (n[i] - n[j]) * (t[i] - t[j]) > 0.0 {
                            ok += 1;
                        }
                    }
                }
            }
            if tot > 0 {
                cc += ok as f64 / tot as f64;
                cn += 1.0;
            }
        }
        let u = used as f64;
        res.push(M { top1: t1 / u, top3: t3 / u, regret: rg / u, conc: cc / cn.max(1.0), by });
    }

    println!("\n=== sibling ranking vs 5k eval-gate Elo ({used} non-trivial positions; top1 = pick within 1 cp of teacher best) ===");
    println!("{:>14} {:>7} {:>7} {:>8} {:>8} {:>7} | top1 by empties <=14 / 15-24 / 25-34 / 35+", "net", "Elo", "top1%", "top3%", "regret", "conc%");
    let mut ord: Vec<usize> = (0..evs.len()).collect();
    ord.sort_by(|&a, &b| elos[b].partial_cmp(&elos[a]).unwrap());
    for &i in &ord {
        let m = &res[i];
        let g: Vec<String> = m.by.iter().map(|x| format!("{:.1}", 100.0 * x.0 / x.2.max(1) as f64)).collect();
        println!(
            "{:>14} {:>+7.1} {:>7.2} {:>8.2} {:>8.2} {:>7.2} | {}",
            names[i],
            elos[i],
            100.0 * m.top1,
            100.0 * m.top3,
            m.regret,
            100.0 * m.conc,
            g.join(" / ")
        );
    }
    let col = |f: &dyn Fn(&M) -> f64| -> Vec<f64> { res.iter().map(f).collect() };
    let top1 = col(&|m| m.top1);
    let top3 = col(&|m| m.top3);
    let negreg = col(&|m| -m.regret);
    let conc = col(&|m| m.conc);
    println!("\nCORR with Elo (all {} nets):  pearson r / spearman rho", evs.len());
    for (nm, v) in [("top1", &top1), ("top3", &top3), ("-regret", &negreg), ("concordance", &conc)] {
        println!("  {:>12}: r={:+.3} rho={:+.3}", nm, pearson(v, &elos), spearman(v, &elos));
    }
    println!("  reference  : val-loss  r=+0.566 rho=+0.833 (D1273, same panel)");
    // champion-family bias check: same correlations without the champion itself
    let keep: Vec<usize> = (0..evs.len()).filter(|&i| names[i] != "Champion").collect();
    let sel = |v: &Vec<f64>| -> Vec<f64> { keep.iter().map(|&i| v[i]).collect() };
    let el2: Vec<f64> = keep.iter().map(|&i| elos[i]).collect();
    println!("CORR without Champion (teacher-family bias check):");
    for (nm, v) in [("top1", &top1), ("-regret", &negreg), ("concordance", &conc)] {
        println!("  {:>12}: r={:+.3} rho={:+.3}", nm, pearson(&sel(v), &el2), spearman(&sel(v), &el2));
    }
    println!("NOTE: teacher = plain alpha-beta {tnodes} nodes/child with {teacher_path}; labels of the training data are champion 5k searches too, so the champion family is favoured. {:.0}s total", t0.elapsed().as_secs_f64());
    ExitCode::SUCCESS
}

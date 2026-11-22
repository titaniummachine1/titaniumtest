//! D1568 (user idea): fit the HAND-CRAFTED move-ordering heuristic to deep-search labels. Every self-play ply in a
//! .tgam/.tbst pair carries the engine's best move at the dataset standard (15,000 nodes). Per legal move, cheap features
//! (no eval, no policy head): captures, clone, jump-origin connections, own / empty neighbours of the destination, the
//! opponent's best recapture next move, and the net exchange. Model = linear score; fit = listwise softmax cross-entropy
//! (target = the engine best move), Adam on minibatches, stop by LAWS C15 (4 epochs without a val CE gain >= 1e-5).
//! Report (held-out games, DAG id % 10 == 0): top1 / top3 of today's hand weights, of the fitted weights, and of the
//! bundled policy head on the same positions.
//!
//!   ti ordfit --games a.tgam[,b.tgam|DIR] [--out weights.txt] [--lr 0.01] [--batch 4096]
use std::process::ExitCode;
use titanium::board::{reach_union, RING1};
use titanium::{Board, Move, MoveList};

const NF: usize = 9;
const NAMES: [&str; NF] = ["captures", "clone", "jump_origin_conn", "dest_own_nbrs", "dest_empty_nbrs_after", "opp_best_recapture", "net_exchange", "policy_logit", "hce_child_cp_x1000"];
/// D1578: --with-hce adds feature 9 = the engine's cheap HCE (search::evaluate, no tuple) of the child, mover's view, in 1000 cp
static WITH_HCE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
/// D1576: --with-policy adds the bundled policy logit as feature 8 (0 otherwise, so its weight stays 0)
static WITH_POL: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
/// today's engine ordering (search.rs: captures x 1000 + clone 500 - jump (500 + 250 per origin connection)) in units of 1000
const TODAY: [f32; NF] = [1.0, 1.0, -0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
/// the engine's real mix today: hand terms + policy at 100,000 units per logit = 100 in units of 1000
const ENGINE: [f32; NF] = [1.0, 1.0, -0.25, 0.0, 0.0, 0.0, 0.0, 100.0, 0.0];

struct Pos {
    b: Board,
    best: Move,
    val: bool,
}

fn feats(b: &Board, m: Move) -> [f32; NF] {
    let (me, opp, empty) = (b.me(), b.opp(), b.empty());
    let to = m.to as usize;
    let tb = 1u64 << to;
    let flips = opp & RING1[to];
    let caps = flips.count_ones() as f32;
    let clone = m.is_clone();
    let from_hole = if clone { 0 } else { 1u64 << m.from };
    let conn = if clone { 0.0 } else { (me & RING1[m.from as usize]).count_ones() as f32 };
    let own_n = (me & RING1[to] & !from_hole).count_ones() as f32;
    let me2 = (me | tb | flips) & !from_hole;
    let opp2 = opp & !flips;
    let emp2 = (empty & !tb) | from_hole;
    let empty_n = (emp2 & RING1[to]).count_ones() as f32;
    let mut t = reach_union(opp2) & emp2;
    let mut reply = 0u32;
    while t != 0 {
        let e = t.trailing_zeros() as usize;
        t &= t - 1;
        reply = reply.max((me2 & RING1[e]).count_ones());
    }
    let reply = reply as f32;
    let pl = if WITH_POL.load(std::sync::atomic::Ordering::Relaxed) { titanium::policy::bundled().map(|p| p.logit(b, m)).unwrap_or(0.0) } else { 0.0 };
    let hce = if WITH_HCE.load(std::sync::atomic::Ordering::Relaxed) { -(titanium::search::evaluate(&b.make(m), None, false) as f32) / 1000.0 } else { 0.0 };
    [caps, if clone { 1.0 } else { 0.0 }, conn, own_n, empty_n, reply, caps - reply, pl, hce]
}

fn same(a: Move, b: Move) -> bool {
    a.to == b.to && (a.is_clone() && b.is_clone() || a.from == b.from)
}

/// listwise CE of one position; returns (loss, (moves scored above the best move, other moves tied with it)), adds the
/// gradient when `g` is Some. Ties matter: integer-like hand scores tie often; credit is the expected share (tie_credit).
fn pos_eval(p: &Pos, w: &[f32; NF], ml: &mut MoveList, fv: &mut Vec<[f32; NF]>, g: Option<&mut [f64; NF]>) -> Option<(f64, (usize, usize))> {
    p.b.legal_moves_dedup_into(ml);
    fv.clear();
    let mut bi = usize::MAX;
    for (i, m) in ml.iter().enumerate() {
        if same(m, p.best) {
            bi = i;
        }
        fv.push(feats(&p.b, m));
    }
    if bi == usize::MAX || fv.len() < 2 {
        return None;
    }
    let sc: Vec<f32> = fv.iter().map(|f| f.iter().zip(w).map(|(a, b)| a * b).sum()).collect();
    let mx = sc.iter().cloned().fold(f32::MIN, f32::max);
    let z: f64 = sc.iter().map(|&s| ((s - mx) as f64).exp()).sum();
    let loss = -((sc[bi] - mx) as f64 - z.ln());
    let rank = (sc.iter().filter(|&&s| s > sc[bi]).count(), sc.iter().filter(|&&s| s == sc[bi]).count() - 1);
    if let Some(g) = g {
        for (i, f) in fv.iter().enumerate() {
            let pr = ((sc[i] - mx) as f64).exp() / z - if i == bi { 1.0 } else { 0.0 };
            for k in 0..NF {
                g[k] += pr * f[k] as f64;
            }
        }
    }
    Some((loss, rank))
}

/// expected top-k credit of the best move when `gt` moves score above it and `eq` others tie with it (random tie order)
fn tie_credit(gt: usize, eq: usize, k: usize) -> f64 {
    if gt >= k {
        0.0
    } else {
        ((k - gt).min(eq + 1)) as f64 / (eq + 1) as f64
    }
}

fn val_stats(ps: &[Pos], w: &[f32; NF]) -> (f64, f64, f64, usize) {
    let (mut ml, mut fv) = (MoveList::new(), Vec::new());
    let (mut l, mut t1, mut t3, mut n) = (0.0, 0f64, 0f64, 0usize);
    for p in ps.iter().filter(|p| p.val) {
        if let Some((loss, (gt, eq))) = pos_eval(p, w, &mut ml, &mut fv, None) {
            l += loss;
            t1 += tie_credit(gt, eq, 1);
            t3 += tie_credit(gt, eq, 3);
            n += 1;
        }
    }
    let nf = n.max(1) as f64;
    (l / nf, t1 / nf, t3 / nf, n)
}

fn policy_stats(ps: &[Pos]) -> Option<(f64, f64, usize)> {
    let pol = titanium::policy::bundled()?;
    let mut ml = MoveList::new();
    let (mut t1, mut t3, mut n) = (0f64, 0f64, 0usize);
    for p in ps.iter().filter(|p| p.val) {
        p.b.legal_moves_dedup_into(&mut ml);
        let mut sc = Vec::with_capacity(ml.len());
        let mut bs = None;
        for m in ml.iter() {
            let s = pol.logit(&p.b, m);
            if same(m, p.best) {
                bs = Some(s);
            }
            sc.push(s);
        }
        let Some(bs) = bs else { continue };
        let (gt, eq) = (sc.iter().filter(|&&s| s > bs).count(), sc.iter().filter(|&&s| s == bs).count() - 1);
        t1 += tie_credit(gt, eq, 1);
        t3 += tie_credit(gt, eq, 3);
        n += 1;
    }
    let nf = n.max(1) as f64;
    Some((t1 / nf, t3 / nf, n))
}

fn load(spec: &str) -> Result<Vec<Pos>, String> {
    let mut files = Vec::new();
    for s in spec.split(',').filter(|s| !s.is_empty()) {
        let p = std::path::Path::new(s);
        if p.is_dir() {
            let mut v: Vec<String> = std::fs::read_dir(p)
                .map_err(|e| format!("{s}: {e}"))?
                .filter_map(|e| e.ok().map(|e| e.path().to_string_lossy().to_string()))
                .filter(|f| f.ends_with(".tgam"))
                .collect();
            v.sort();
            files.extend(v);
        } else {
            files.push(s.to_string());
        }
    }
    let mut ps = Vec::new();
    for f in &files {
        let mut rd = crate::gamedata::GameReader::open(f).map_err(|e| format!("{f}: {e}"))?;
        let mut tb = crate::gamedata::TbstReader::open(f).map_err(|e| format!("{f} (.tbst): {e}"))?;
        let mut got = 0usize;
        for g in rd.by_ref() {
            let g = g.map_err(|e| format!("{f}: {e}"))?;
            let (gid, labels) = tb.next_game(g.plies.len()).map_err(|e| format!("{f}: {e}"))?;
            let val = gid % 10 == 0;
            let mut b = g.start();
            for (pi, ply) in g.plies.iter().enumerate() {
                let code = ply.0 & 0x3ff;
                if code == titanium::ttcodec::PASS_CODE {
                    b = b.make_uai(Move::PASS);
                    continue;
                }
                let Some(played) = titanium::ttcodec::decode_move(&b, code) else { break };
                if let Some(bm) = labels.get(pi).and_then(|&(lc, _)| titanium::ttcodec::decode_move(&b, lc & 0x3ff)).filter(|m| !m.is_pass()) {
                    ps.push(Pos { b, best: bm, val });
                    got += 1;
                }
                b = b.make(played);
            }
        }
        eprintln!("[ordfit] {f}: {got} labelled positions");
    }
    Ok(ps)
}

pub fn run(args: &[String]) -> ExitCode {
    let f = |n: &str| crate::flag_str(args, n);
    let Some(spec) = f("--games") else {
        eprintln!("usage: ti ordfit --games a.tgam[,b.tgam|DIR] [--with-policy] [--with-hce] [--out weights.txt] [--lr 0.01] [--batch 4096]");
        return ExitCode::from(2);
    };
    let lr = crate::flag_f64(args, "--lr").unwrap_or(0.01);
    let with_pol = crate::has_flag(args, "--with-policy");
    WITH_POL.store(with_pol, std::sync::atomic::Ordering::Relaxed);
    let with_hce = crate::has_flag(args, "--with-hce");
    WITH_HCE.store(with_hce, std::sync::atomic::Ordering::Relaxed);
    let bs = crate::flag_u64(args, "--batch").unwrap_or(4096) as usize;
    let ps = match load(&spec) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("ordfit: {e}");
            return ExitCode::from(2);
        }
    };
    let ntr = ps.iter().filter(|p| !p.val).count();
    println!("[ordfit] {} positions ({} train, {} held-out games' positions), features {:?}", ps.len(), ntr, ps.len() - ntr, NAMES);
    let (l0, a1, a3, n) = val_stats(&ps, &TODAY);
    println!("[ordfit] TODAY's hand weights {TODAY:?}: val CE {l0:.5} top1 {:.2}% top3 {:.2}% (n {n})", 100.0 * a1, 100.0 * a3);
    if let Some((p1, p3, pn)) = policy_stats(&ps) {
        println!("[ordfit] POLICY HEAD (bundled): top1 {:.2}% top3 {:.2}% (n {pn})", 100.0 * p1, 100.0 * p3);
    }
    if with_hce {
        let mut hw = [0f32; NF];
        hw[8] = 1.0;
        let (lh, h1, h3, _) = val_stats(&ps, &hw);
        println!("[ordfit] HCE ALONE (child evaluate, no tuple): val CE {lh:.5} top1 {:.2}% top3 {:.2}%", 100.0 * h1, 100.0 * h3);
    }
    if with_pol {
        let (le, e1, e3, _) = val_stats(&ps, &ENGINE);
        println!("[ordfit] ENGINE mix today (hand + 100 x policy logit): val CE {le:.5} top1 {:.2}% top3 {:.2}%", 100.0 * e1, 100.0 * e3);
    }
    let train: Vec<usize> = (0..ps.len()).filter(|&i| !ps[i].val).collect();
    // with the policy: start from the policy alone (weight 1 per logit) and let the fit mix the hand terms in
    let mut w = if with_pol { [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0] } else { TODAY };
    if with_hce && !with_pol {
        w = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0];
    }
    let (mut m1, mut m2) = ([0f64; NF], [0f64; NF]);
    let (b1, b2, eps) = (0.9f64, 0.999f64, 1e-8f64);
    let mut step = 0i32;
    let mut best = (if with_pol { val_stats(&ps, &w).0 } else { l0 }, w);
    let mut flat = 0usize;
    let mut rng = 0x9E3779B97F4A7C15u64;
    let (mut ml, mut fv) = (MoveList::new(), Vec::new());
    for ep in 1.. {
        let mut order = train.clone();
        for i in (1..order.len()).rev() {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            order.swap(i, (rng % (i as u64 + 1)) as usize);
        }
        let mut tl = 0.0;
        for chunk in order.chunks(bs.max(1)) {
            let mut g = [0f64; NF];
            let mut nb = 0usize;
            for &i in chunk {
                if let Some((l, _)) = pos_eval(&ps[i], &w, &mut ml, &mut fv, Some(&mut g)) {
                    tl += l;
                    nb += 1;
                }
            }
            if nb == 0 {
                continue;
            }
            step += 1;
            for k in 0..NF {
                let gk = g[k] / nb as f64;
                m1[k] = b1 * m1[k] + (1.0 - b1) * gk;
                m2[k] = b2 * m2[k] + (1.0 - b2) * gk * gk;
                let mh = m1[k] / (1.0 - b1.powi(step));
                let vh = m2[k] / (1.0 - b2.powi(step));
                w[k] -= (lr * mh / (vh.sqrt() + eps)) as f32;
            }
        }
        let (vl, v1, v3, _) = val_stats(&ps, &w);
        println!("[ordfit] epoch {ep}: train CE {:.5} | val CE {vl:.5} top1 {:.2}% top3 {:.2}% | w {:?}", tl / train.len().max(1) as f64, 100.0 * v1, 100.0 * v3, w.map(|x| (x * 1000.0).round() / 1000.0));
        if vl < best.0 - 1e-5 {
            best = (vl, w);
            flat = 0;
        } else {
            flat += 1;
            if flat >= 4 {
                println!("[ordfit] SATURATED (LAWS C15): 4 epochs without a val CE gain >= 1e-5");
                break;
            }
        }
    }
    let (vl, v1, v3, n) = val_stats(&ps, &best.1);
    println!("[ordfit] BEST fitted weights: val CE {vl:.5} top1 {:.2}% top3 {:.2}% (n {n})", 100.0 * v1, 100.0 * v3);
    for (k, name) in NAMES.iter().enumerate() {
        println!("[ordfit]   {name:24} {:+.4} (today {:+.4})", best.1[k], TODAY[k]);
    }
    if let Some(out) = f("--out") {
        let txt: String = NAMES.iter().zip(best.1.iter()).map(|(n, v)| format!("{n} {v}\n")).collect();
        if let Err(e) = std::fs::write(&out, txt) {
            eprintln!("ordfit: {out}: {e}");
        }
    }
    ExitCode::SUCCESS
}

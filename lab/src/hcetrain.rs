//! `lab hce-train` (D977): train the hand-crafted eval EXACTLY like the NNUE: same dataset, same target, same loss, same
//! validation, stop only at saturation (LAWS C15), initialised from the closed-form fit to the champion (LAWS C14).
//!
//!   lab hce-train --data train_v10.tdat --val val_v9_clean.tdat [--set 4] [--threads 7] [--bs 16384] [--lr 2.0]
//!                 [--max-rows N (smoke tests only)] [--out logs/hce/run]
//!
//! Model: linear in the chosen hcecmp term set (default 4 = ours + gataxx, 844 weights, cp units): y = w.x / 400 (sigmoid units).
//! Target (hub targets.rs, champion recipe D330/D427): proven rows (|cp| >= 98,000) = exact 1 / 0; otherwise
//!   (1 - w) * sigmoid(clamp(cp, +-1500) / 400) + w * outcome, w = 0.55 (<= 6 empties), 0.55 (7..12), 0.25 (>= 13).
//! Loss: wsum * |sigmoid(y) - target|^2.5 (hub train.rs). Validation: target_of = 0.75 * sigmoid + 0.25 * outcome on val rows,
//! the same metric printed for the bundled NNUE on the same rows. Saturation: stop when 4 epochs in a row do not improve the
//! best validation loss by >= 1e-7 (lr halves after each non-improving epoch).
use crate::hcecmp::{block, feats, masks, nbk, set_nbk};
use crate::hcefit::{phase, Ridge, CLAMP};
use crate::pairs::Pairs;
use crate::spairs::SPairs;
/// D1072: --spairs BANDS = stone pairs by distance band, appended after the pair / mobility block (offset stored here)
static SP: std::sync::OnceLock<(SPairs, u32)> = std::sync::OnceLock::new();
use crate::zattrain::{d4maps, sym};
use crate::util::{arg, ring1, Rec, REC};
use std::fs::File;
use std::io::{BufReader, Read, Write};
use std::process::ExitCode;
use titanium::nnue::Nnue;
use titanium::{Board, MoveList};

const MATE: f32 = 100_000.0;

pub(crate) fn sig(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}
fn sig_cp(cp: f32) -> f32 {
    sig(cp.clamp(-1500.0, 1500.0) / 400.0)
}

/// training target (target_train_rw with result weight 0.55/0.55/0.25) or validation target (target_of)
pub(crate) fn target(r: &Rec, train: bool) -> f32 {
    if r.cp >= MATE - 2000.0 {
        return 1.0;
    }
    if r.cp <= -(MATE - 2000.0) {
        return 0.0;
    }
    if !train {
        return 0.75 * sig_cp(r.cp) + 0.25 * r.outcome;
    }
    let empties = 49 - (r.occ0 | r.occ1).count_ones();
    let w = if empties <= 12 { 0.55 } else { 0.25 };
    (1.0 - w) * sig_cp(r.cp) + w * r.outcome
}

/// all features: the bucketed HCE term set + (with --pairs) gataxx's pair-pattern + mobility tables at offset `off` (D991)
#[allow(clippy::too_many_arguments)]
fn fx(set: usize, pr: Option<&Pairs>, off: u32, b: &Board, ring: &[u64; 49], m: &crate::hcecmp::Masks, tmp: &mut Vec<(u16, f32)>, out: &mut Vec<(u32, f32)>) {
    feats(set, b, ring, m, tmp);
    out.clear();
    out.extend(tmp.iter().map(|&(i, v)| (i as u32, v)));
    if let Some(p) = pr {
        p.feats(b.me(), b.opp(), off, out);
    }
    if let Some((sp, o)) = SP.get() {
        sp.feats(b.me(), b.opp(), *o, out);
    }
}

/// D1113 (user: "use the plain alpha-beta to see directly how the architecture changes strength"): a finished
/// `hce-train --on-nnue 1` model (`*_weights.json`) as an evaluator = the frozen bundled champion + its learned linear
/// correction, exactly as scored in training (same feature layout, pairs keep/mob, stone-pair bands, per-bucket scale).
/// One model per process (bucket count and the stone-pair table are process-global).
pub struct Residual {
    set: usize,
    pairs: Option<Pairs>,
    off: u32,
    sc_off: Option<u32>,
    /// own stone-pair table (not the process-global SP), so models with and without bands can play each other
    sp: Option<(SPairs, u32)>,
    w: Vec<f32>,
    ring: [u64; 49],
    m: crate::hcecmp::Masks,
    net: Nnue,
}

impl Residual {
    pub fn load(path: &str) -> Residual {
        let s = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("{path}: {e}"));
        let field = |k: &str| -> String {
            let t = s.split(&format!("\"{k}\":")).nth(1).unwrap_or_else(|| panic!("{path}: no field {k}"));
            t.split([',', '}']).next().unwrap().trim().trim_matches('"').to_string()
        };
        assert_eq!(field("on_nnue"), "true", "{path}: not an --on-nnue residual model");
        let set: usize = field("set").parse().unwrap();
        set_nbk(field("buckets").parse().unwrap());
        let d_base: usize = field("base_dims").parse().unwrap();
        assert_eq!(d_base, nbk() * block(set), "{path}: base_dims");
        let mut pairs = if field("pairs") == "true" { Some(Pairs::new()) } else { None };
        if let Some(p) = pairs.as_mut() {
            if let Some(t) = s.split("\"pair_keep\":[").nth(1).and_then(|t| t.split(']').next()) {
                if !t.trim().is_empty() {
                    let mut k = [false; crate::pairs::PLACEMENTS];
                    for v in t.split(',') {
                        k[v.trim().parse::<usize>().unwrap()] = true;
                    }
                    p.keep = k;
                }
            }
            if s.contains("\"mob\":false") {
                p.mob = false;
            }
        }
        let sp_off = d_base + pairs.as_ref().map_or(0, |p| p.dims());
        // models trained before D1072 have no spairs field
        let bands = if s.contains("\"spairs\":") { field("spairs") } else { String::new() };
        let sp = if bands.is_empty() || bands == "null" { None } else { Some((SPairs::new(&bands), sp_off as u32)) };
        let sp_dims = sp.as_ref().map_or(0, |s| s.0.dims());
        let body = s.split("\"weights\":[").nth(1).and_then(|t| t.split(']').next()).expect("weights array");
        let w: Vec<f32> = body.split(',').map(|v| v.trim().parse::<f32>().unwrap()).collect();
        let core = sp_off + sp_dims;
        let sc_off = if w.len() == core + nbk() {
            Some(core as u32)
        } else {
            assert_eq!(w.len(), core, "{path}: {} weights, layout needs {core} (+{} with --nnue-scale)", w.len(), nbk());
            None
        };
        Residual { set, pairs, off: d_base as u32, sc_off, sp, w, ring: ring1(), m: masks(), net: Nnue::bundled() }
    }

    /// cp from the side to move (same scale as `Nnue::eval`)
    pub fn eval(&self, b: &Board) -> i32 {
        self.eval_f(b).round().clamp(-30000.0, 30000.0) as i32
    }

    /// unrounded cp, exactly the training scorer
    pub fn eval_f(&self, b: &Board) -> f32 {
        let (mut t, mut x) = (Vec::new(), Vec::new());
        feats(self.set, b, &self.ring, &self.m, &mut t);
        x.extend(t.iter().map(|&(i, v)| (i as u32, v)));
        if let Some(p) = self.pairs.as_ref() {
            p.feats(b.me(), b.opp(), self.off, &mut x);
        }
        if let Some((sp, o)) = &self.sp {
            sp.feats(b.me(), b.opp(), *o, &mut x);
        }
        let base = self.net.eval(b, 0) as f32;
        let mut v = base + x.iter().map(|&(i, v)| self.w[i as usize] * v).sum::<f32>();
        if let Some(o) = self.sc_off {
            v += self.w[o as usize + crate::hcecmp::bk(b.empty())] * base / 400.0;
        }
        v
    }
}

/// `lab residual-check --model M_weights.json --data X.tdat [--holdout 0.1]` (D1113): parity of `Residual::eval` with the
/// training run: recomputes the held-out val loss (same split, same target, same loss) that hce-train logged as best.
pub fn residual_check(args: &[String]) -> ExitCode {
    let mp = arg(args, "--model").expect("--model M_weights.json");
    let data = arg(args, "--data").expect("--data X.tdat");
    let h: f64 = arg(args, "--holdout").map_or(0.1, |v| v.parse().unwrap());
    let r = Residual::load(&mp);
    let mut all = read_rows(&data, usize::MAX);
    let k = ((all.len() as f64) * (1.0 - h)) as usize;
    let val = all.split_off(k);
    drop(all);
    let (mut l, mut ln, mut ws) = (0f64, 0f64, 0f64);
    for row in &val {
        let b = board(row);
        let t = target(row, false);
        let wt = row.wsum.max(0.0) as f64;
        // the training scorer is f32 (not rounded to whole cp): use the unrounded value for parity
        let p = sig(r.eval_f(&b) / 400.0);
        let pn = sig(r.net.eval(&b, 0) as f32 / 400.0);
        l += wt * ((p - t).abs() as f64).powf(2.5);
        ln += wt * ((pn - t).abs() as f64).powf(2.5);
        ws += wt;
    }
    println!("residual-check: {mp}: {} held-out rows: val loss {:.6} (champion alone {:.6})", val.len(), l / ws, ln / ws);
    ExitCode::SUCCESS
}

fn board(r: &Rec) -> Board {
    let (own, opp) = if r.stm == 0 { (r.occ0, r.occ1) } else { (r.occ1, r.occ0) };
    Board::from_occ(own, opp, 0, 0)
}

pub(crate) fn read_rows(path: &str, max: usize) -> Vec<Rec> {
    let f = File::open(path).expect("tdat");
    let n = (f.metadata().map(|m| m.len() as usize).unwrap_or(0) / REC).min(max);
    let mut rd = BufReader::with_capacity(1 << 22, f);
    let mut buf = vec![0u8; REC];
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        if rd.read_exact(&mut buf).is_err() {
            break;
        }
        out.push(Rec::from_bytes(&buf));
    }
    out
}

pub fn run(args: &[String]) -> ExitCode {
    let (Some(data), Some(valp)) = (arg(args, "--data"), arg(args, "--val")) else {
        eprintln!("usage: lab hce-train --data X.tdat --val V.tdat [--set 4] [--threads 7] [--bs 16384] [--lr 2.0] [--max-rows N] [--out PREFIX]");
        return ExitCode::from(2);
    };
    let g = |k: &str, d: f64| arg(args, k).and_then(|v| v.parse::<f64>().ok()).unwrap_or(d);
    let set = g("--set", 4.0) as usize;
    let threads = g("--threads", 7.0) as usize;
    let bs = g("--bs", 16384.0) as usize;
    let mut lr = g("--lr", 2.0) as f32;
    let max_rows = g("--max-rows", 1e12) as usize;
    let out = arg(args, "--out").unwrap_or_else(|| "C:/gitProjects/titanium-ataxx/logs/hce/d977_hcetrain".into());
    set_nbk(g("--buckets", 16.0) as usize);
    let d_base = nbk() * block(set);
    // --pairs 1 (D991): gataxx's pair-pattern residual + mobility table, appended after the bucketed HCE, init 0
    let mut pairs = if g("--pairs", 0.0) as usize == 1 { Some(Pairs::new()) } else { None };
    // D1032: --pair-keep "i,j,.." = use only these placements (whole D4 orbits, from lab hce-ablate); --no-mob 1 = drop mobility
    if let Some(p) = pairs.as_mut() {
        if let Some(list) = arg(args, "--pair-keep") {
            let mut k = [false; crate::pairs::PLACEMENTS];
            for s in list.split(',').filter(|s| !s.trim().is_empty()) {
                k[s.trim().parse::<usize>().expect("--pair-keep: placement ids")] = true;
            }
            for q in 0..crate::pairs::PLACEMENTS {
                assert_eq!(k[q], k[p.orbit[q] as usize], "--pair-keep must hold whole D4 orbits (placement {q})");
            }
            p.keep = k;
        }
        if g("--no-mob", 0.0) as usize == 1 {
            p.mob = false;
        }
    }
    let pr = pairs.as_ref();
    let on_nnue0 = g("--on-nnue", 0.0) as usize == 1;
    // D1053: --nnue-scale 1 (with --on-nnue) adds one weight per bucket on the champion cp / 400 (a learned scale), so the
    // calibration baseline (set 5 = bias + scale) can be separated from real information in the other groups
    let nn_scale = on_nnue0 && g("--nnue-scale", 0.0) as usize == 1;
    let sp_off = d_base + pr.map_or(0, |p| p.dims());
    let sp_dims = match arg(args, "--spairs") {
        Some(bands) => {
            let s = SPairs::new(&bands);
            let n = s.dims();
            let _ = SP.set((s, sp_off as u32));
            n
        }
        None => 0,
    };
    let sc_off = (sp_off + sp_dims) as u32;
    let d = sc_off as usize + if nn_scale { nbk() } else { 0 };
    let off = d_base as u32;
    // lr multiplier for the pair / mobility block (Adam step size in cp)
    let plr = g("--plr", 1.0) as f32;
    // D1034 (user: HCE features into the NNUE): --on-nnue 1 = the frozen champion NNUE cp is a fixed base and the HCE
    // features learn only a linear correction on top (lower bound of a joint hybrid). Init = zero correction = exactly
    // the champion (step-0 parity, C14). Not a standalone HCE: the weights JSON says "on_nnue":true.
    let on_nnue = g("--on-nnue", 0.0) as usize == 1;
    // --val-target 1: train on the validation target (0.75 sigmoid + 0.25 outcome everywhere) instead of the NNUE recipe
    // (0.55 outcome at <= 12 empties): an information test then measures only what the features add, not the target gap.
    let val_tgt = g("--val-target", 0.0) as usize == 1;
    let ring = ring1();
    let m = masks();
    let net = Nnue::bundled();
    let t0 = std::time::Instant::now();
    let mut log = String::new();
    let mut say = |s: String, log: &mut String| {
        println!("{s}");
        log.push_str(&s);
        log.push('\n');
        let _ = std::fs::write(format!("{out}.log"), log.as_bytes());
    };

    let (train, val) = if g("--val-split", 0.0) as usize == 1 {
        // D1034 held-out information test: the champion NNUE was trained on train_v10, so its errors there are not its
        // errors on new positions; fit on the even rows of the val file, score on the odd rows (--data is ignored).
        let all = read_rows(&valp, 600_000);
        let (mut a, mut b) = (Vec::new(), Vec::new());
        for (i, r) in all.into_iter().enumerate() {
            if i % 2 == 0 { a.push(r) } else { b.push(r) }
        }
        (a, b)
    } else if g("--holdout", 0.0) > 0.0 {
        // D1053: train and score on the SAME distribution: --data rows are in game order (games-tdat chains), so the last
        // fraction is a set of whole games the fit never sees (--val is ignored). Needed because train rows labelled by
        // today's main and val_v9's older labels differ systematically (the correction learned that gap, D1053).
        let mut all = read_rows(&data, max_rows);
        let k = ((all.len() as f64) * (1.0 - g("--holdout", 0.0))) as usize;
        let v = all.split_off(k);
        (all, v)
    } else {
        (read_rows(&data, max_rows), read_rows(&valp, 600_000))
    };
    if let Some((s, o)) = SP.get() {
        say(format!("hce-train: stone pairs by distance (D1072), bands {} = {} D4-tied classes at offset {o}", s.bands, s.nclass), &mut log);
    }
    say(format!("hce-train: set {set}, {} buckets ({d_base} HCE weights + {} pair/mobility weights{}, plr {plr}), {} train rows, {} val rows, {threads} threads, bs {bs} ({:.0}s)", nbk(), d - d_base, pr.map_or(String::new(), |p| format!(" = {} pair classes + {}", p.nclass, crate::pairs::MOB)), train.len(), val.len(), t0.elapsed().as_secs_f64()), &mut log);

    // feature cost per eval, single thread (speed only, not the engine's incremental cost)
    {
        let (mut t, mut x) = (Vec::new(), Vec::new());
        let bs_: Vec<Board> = val.iter().take(100_000).map(board).collect();
        let t1 = std::time::Instant::now();
        let mut acc = 0usize;
        for b in &bs_ {
            feats(set, b, &ring, &m, &mut t);
            acc += t.len();
        }
        let base_ns = t1.elapsed().as_nanos() as f64 / bs_.len() as f64;
        let mut pair_ns = 0.0;
        if let Some(p) = pr {
            let t2 = std::time::Instant::now();
            for b in &bs_ {
                x.clear();
                p.feats(b.me(), b.opp(), off, &mut x);
                acc += x.len();
            }
            pair_ns = t2.elapsed().as_nanos() as f64 / bs_.len() as f64;
        }
        say(format!("feature extraction (lab code, unoptimised): HCE terms {base_ns:.0} ns/eval, pairs+mobility {pair_ns:.0} ns/eval ({acc} features)"), &mut log);
    }
    // init (C14): closed-form ridge fit to the champion eval on the first 20k train positions + their children.
    // --init FILE loads a finished run's weights instead (fresh-lr restart, D988): same dims required.
    let init_path = arg(args, "--init");
    let init_w: Vec<f32>;
    let mut w: Vec<f32>;
    if let Some(p) = &init_path {
        let s = std::fs::read_to_string(p).unwrap_or_else(|e| panic!("{p}: {e}"));
        let body = s
            .split("\"weights\":[")
            .nth(1)
            .and_then(|t| t.split(']').next())
            .unwrap_or_else(|| panic!("{p}: no \"weights\" array"));
        w = body.split(',').map(|v| v.trim().parse::<f32>().unwrap_or_else(|e| panic!("{p}: {e}"))).collect();
        if w.len() == d_base && d > d_base {
            w.resize(d, 0.0); // graft: trained HCE + zero residual = step-0 parity (C14)
        }
        assert_eq!(w.len(), d, "{p}: {} weights, this model needs {d}", w.len());
        init_w = w.clone();
    } else if on_nnue {
        w = vec![0f32; d];
        init_w = w.clone();
    } else {
        let mut ridge = Ridge::new(d_base);
        let mut ml = MoveList::new();
        let mut x = Vec::new();
        for r in train.iter().step_by((train.len() / 20_000).max(1)).take(20_000) {
            let b = board(r);
            b.legal_moves_dedup_into(&mut ml);
            feats(set, &b, &ring, &m, &mut x);
            ridge.add(&x, (net.eval(&b, 0) as f64).clamp(-CLAMP, CLAMP));
            for mv in ml.iter() {
                let c = b.make(mv);
                feats(set, &c, &ring, &m, &mut x);
                ridge.add(&x, (net.eval(&c, 0) as f64).clamp(-CLAMP, CLAMP));
            }
        }
        let mut iw: Vec<f32> = ridge.solve(&vec![true; d_base], 1.0).iter().map(|&v| v as f32).collect();
        iw.resize(d, 0.0);
        init_w = iw;
        w = init_w.clone();
    }

    // validation loss (|p - t|^2.5, wsum-weighted) and MSE, for any scorer giving cp
    let vloss = |f: &(dyn Fn(&Board) -> f32 + Sync)| -> (f64, f64) {
        let chunk = val.len().div_ceil(threads);
        let parts: Vec<(f64, f64, f64)> = std::thread::scope(|sc| {
            val.chunks(chunk)
                .map(|vs| {
                    sc.spawn(move || {
                        let (mut l, mut q, mut ws) = (0f64, 0f64, 0f64);
                        for r in vs {
                            let p = sig(f(&board(r)) / 400.0);
                            let t = target(r, false);
                            let wt = r.wsum.max(0.0) as f64;
                            l += wt * ((p - t).abs() as f64).powf(2.5);
                            q += wt * ((p - t) as f64).powi(2);
                            ws += wt;
                        }
                        (l, q, ws)
                    })
                })
                .collect::<Vec<_>>()
                .into_iter()
                .map(|h| h.join().unwrap())
                .collect()
        });
        let ws: f64 = parts.iter().map(|p| p.2).sum();
        (parts.iter().map(|p| p.0).sum::<f64>() / ws, parts.iter().map(|p| p.1).sum::<f64>() / ws)
    };
    let (nn_l, nn_q) = vloss(&|b: &Board| net.eval(b, 0) as f32);
    let netr = &net;
    let hce = |w: &[f32]| {
        let w = w.to_vec();
        move |b: &Board| {
            let (mut t, mut x) = (Vec::new(), Vec::new());
            fx(set, pr, off, b, &ring1(), &masks(), &mut t, &mut x);
            let base = if on_nnue { netr.eval(b, 0) as f32 } else { 0.0 };
            if nn_scale {
                x.push((sc_off + crate::hcecmp::bk(b.empty()) as u32, base / 400.0));
            }
            base + x.iter().map(|&(i, v)| w[i as usize] * v).sum::<f32>()
        }
    };
    let (i_l, i_q) = vloss(&hce(&w));
    let init_label = match &init_path {
        Some(p) => format!("loaded {p}"),
        None => "fit to the NNUE".to_string(),
    };
    say(format!("reference: champion NNUE val loss {nn_l:.6} (mse {nn_q:.6}); HCE init ({init_label}) {i_l:.6} (mse {i_q:.6}) ({:.0}s)", t0.elapsed().as_secs_f64()), &mut log);

    // D4 check (D1003): the init model's eval of 20k val positions under all 8 board symmetries
    let maps = d4maps();
    {
        let f = hce(&w);
        let (mut mx, mut sum, mut n, mut neq) = (0f32, 0f64, 0usize, 0usize);
        for r in val.iter().step_by((val.len() / 20_000).max(1)).take(20_000) {
            let b = board(r);
            let e0 = f(&b);
            for mt in &maps[1..] {
                let e = f(&Board::from_occ(sym(b.me(), mt), sym(b.opp(), mt), 0, 0));
                let dd = (e - e0).abs();
                mx = mx.max(dd);
                sum += dd as f64;
                n += 1;
                neq += (dd > 1e-3) as usize;
            }
        }
        say(format!("D4 check: init eval under the 7 other symmetries, {n} pairs: max |diff| {mx:.4} cp, mean {:.6} cp, {neq} pairs differ (0 = model is D4-invariant, augmentation cannot change it)", sum / n as f64), &mut log);
    }

    // AdamW-free plain Adam on the linear weights
    let (mut mo, mut ve) = (vec![0f32; d], vec![0f32; d]);
    let (b1, b2, eps) = (0.9f32, 0.999f32, 1e-8f32);
    let mut step = 0i32;
    let mut best = (i_l, w.clone(), 0usize);
    let mut bad = 0usize;
    let mut order: Vec<usize> = (0..train.len()).collect();
    let mut rng = 0x9E37_79B9_7F4A_7C15u64;
    for epoch in 1.. {
        for i in (1..order.len()).rev() {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            order.swap(i, (rng % (i as u64 + 1)) as usize);
        }
        let (mut tl, mut tw) = (0f64, 0f64);
        for batch in order.chunks(bs) {
            let chunk = batch.len().div_ceil(threads);
            let wr = &w;
            let trr = &train;
            let mp = &maps;
            let ep = epoch;
            let parts: Vec<(Vec<f32>, f64, f64)> = std::thread::scope(|sc| {
                batch
                    .chunks(chunk)
                    .map(|idx| {
                        sc.spawn(move || {
                            let ring = ring1();
                            let m = masks();
                            let mut gr = vec![0f32; wr.len()];
                            let (mut t, mut x) = (Vec::new(), Vec::new());
                            let (mut l, mut ws) = (0f64, 0f64);
                            for &k in idx {
                                let r = &trr[k];
                                let wt = r.wsum.max(0.0);
                                if wt == 0.0 {
                                    continue;
                                }
                                // online D4 augmentation, ALWAYS on (user D1001/D1003): random symmetry per row per epoch
                                let mut hs = (k as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ (ep as u64).wrapping_mul(0xD1B5_4A32_D192_ED03);
                                hs ^= hs >> 29;
                                let sm = &mp[(hs.wrapping_mul(0xBF58_476D_1CE4_E5B9) >> 61) as usize];
                                let b0 = board(r);
                                let bsym = Board::from_occ(sym(b0.me(), sm), sym(b0.opp(), sm), 0, 0);
                                fx(set, pr, off, &bsym, &ring, &m, &mut t, &mut x);
                                // base on the UNROTATED board: the champion NNUE is not D4-invariant (smoke 2026-11-17:
                                // max 858 cp, mean 40 cp across symmetries) and val scores it unrotated; the HCE part is D4-tied.
                                let base = if on_nnue { netr.eval(&b0, 0) as f32 } else { 0.0 };
                                if nn_scale {
                                    x.push((sc_off + crate::hcecmp::bk(b0.empty()) as u32, base / 400.0));
                                }
                                let y = (base + x.iter().map(|&(i, v)| wr[i as usize] * v).sum::<f32>()) / 400.0;
                                let p = sig(y);
                                let t = target(r, !val_tgt);
                                let diff = p - t;
                                let ad = diff.abs();
                                l += (wt * ad.powf(2.5)) as f64;
                                ws += wt as f64;
                                let dy = wt * 2.5 * diff.signum() * ad.powf(1.5) * p * (1.0 - p) / 400.0;
                                for &(i, v) in &x {
                                    gr[i as usize] += dy * v;
                                }
                            }
                            (gr, l, ws)
                        })
                    })
                    .collect::<Vec<_>>()
                    .into_iter()
                    .map(|h| h.join().unwrap())
                    .collect()
            });
            let ws: f64 = parts.iter().map(|p| p.2).sum::<f64>().max(1e-9);
            tl += parts.iter().map(|p| p.1).sum::<f64>();
            tw += ws;
            step += 1;
            let (c1, c2) = (1.0 - b1.powi(step), 1.0 - b2.powi(step));
            for i in 0..d {
                let gi = parts.iter().map(|p| p.0[i]).sum::<f32>() / ws as f32;
                mo[i] = b1 * mo[i] + (1.0 - b1) * gi;
                ve[i] = b2 * ve[i] + (1.0 - b2) * gi * gi;
                let li = if i < d_base { lr } else { lr * plr };
                w[i] -= li * (mo[i] / c1) / ((ve[i] / c2).sqrt() + eps);
            }
        }
        let (vl, vq) = vloss(&hce(&w));
        let improved = vl < best.0 - 1e-7;
        if improved {
            best = (vl, w.clone(), epoch);
            bad = 0;
        } else {
            bad += 1;
            lr *= 0.5;
        }
        say(format!(
            "epoch {epoch}: train loss {:.6} | val loss {vl:.6} (mse {vq:.6}) | best {:.6} @ {} | lr {lr:.3} | NNUE {nn_l:.6} | {:.0}s",
            tl / tw.max(1e-9), best.0, best.2, t0.elapsed().as_secs_f64()
        ), &mut log);
        if bad >= 4 {
            say(format!("SATURATED: 4 epochs without improvement; best epoch {} val loss {:.6} (init {i_l:.6}, NNUE {nn_l:.6})", best.2, best.0), &mut log);
            break;
        }
    }
    // sibling best-move agreement with the NNUE, init vs trained (2,000 val positions)
    let agree = |wv: &[f32]| -> f64 {
        let (mut ag, mut n) = (0f64, 0f64);
        let mut ml = MoveList::new();
        let (mut t, mut x) = (Vec::new(), Vec::new());
        for r in val.iter().step_by((val.len() / 2000).max(1)).take(2000) {
            let b = board(r);
            b.legal_moves_dedup_into(&mut ml);
            if ml.len() < 2 {
                continue;
            }
            let (mut bn, mut bh) = ((f32::MIN, 0usize), (f32::MIN, 0usize));
            for (i, mv) in ml.iter().enumerate() {
                let c = b.make(mv);
                let yn = -(net.eval(&c, 0) as f32);
                fx(set, pr, off, &c, &ring, &m, &mut t, &mut x);
                if nn_scale {
                    x.push((sc_off + crate::hcecmp::bk(c.empty()) as u32, -yn / 400.0));
                }
                let yh = if on_nnue { yn } else { 0.0 } - x.iter().map(|&(i, v)| wv[i as usize] * v).sum::<f32>();
                if yn > bn.0 {
                    bn = (yn, i);
                }
                if yh > bh.0 {
                    bh = (yh, i);
                }
            }
            ag += (bn.1 == bh.1) as u8 as f64;
            n += 1.0;
        }
        100.0 * ag / n
    };
    say(format!("sibling best-move agreement with the NNUE: init ({init_label}) {:.2}%, trained like the NNUE {:.2}%", agree(&init_w), agree(&best.1)), &mut log);
    let js = format!(
        "{{\"set\":{set},\"buckets\":{},\"base_dims\":{d_base},\"pairs\":{},\"pair_keep\":{:?},\"mob\":{},\"spairs\":\"{}\",\"spairs_dims\":{sp_dims},\"on_nnue\":{on_nnue},\"val_target\":{val_tgt},\"dims\":{d},\"best_epoch\":{},\"val_loss\":{},\"weights\":{:?}}}",
        nbk(),
        pr.is_some(),
        pr.map_or(Vec::new(), |p| (0..crate::pairs::PLACEMENTS).filter(|&q| p.keep[q]).collect::<Vec<_>>()),
        pr.is_none_or(|p| p.mob),
        SP.get().map_or(String::new(), |s| s.0.bands.clone()),
        best.2,
        best.0,
        best.1
    );
    let _ = File::create(format!("{out}_weights.json")).and_then(|mut f| f.write_all(js.as_bytes()));
    let _ = phase(0);
    ExitCode::SUCCESS
}

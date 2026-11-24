//! `ti analyze` and `ti harvest`: read the `<out>.games.jsonl` a gate writes (one game per line, every move with
//! the mover's own thought) and (a) explain each game: turning point, when the loser saw it, where the engines
//! disagreed; (b) turn every searched position into a HOLDOUT dataset row (never merged automatically).
//!
//! Every game is replayed from its start FEN; an illegal or unparsable move aborts with the game id, so a
//! corrupted log can never silently feed analysis or data.

use std::fs;
use std::io::Write;
use std::process::ExitCode;
use titanium::Board;

/// One completed/adopted depth snapshot from `"it"` (`D1668`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IterRec {
    pub depth: u32,
    pub score: Option<i32>,
    pub mate: Option<i32>,
    pub nodes: u64,
    pub ms: u64,
    pub rfh: u8,
    pub mirage: bool,
    pub pv0: String,
    pub pv1: String,
}

/// One move as stored in the JSONL.
#[derive(Debug, Clone, PartialEq)]
pub struct Rec {
    pub ply: u32,
    pub mover: String,
    /// the move PLAYED (the game line; replay follows this)
    pub mv: String,
    /// `ti gen` logs (D790): the engine's own first choice at this position before novelty changed the move played.
    /// Policy / best-move targets must use this (`label_mv()`), never `mv`.
    pub best_mv: Option<String>,
    /// the position's score. When the log has `engine_best_score` (ti gen) it is THAT (the value of the engine's own
    /// best move), not the score of the move novelty made it play.
    pub score: Option<i32>,
    pub mate: Option<i32>,
    pub depth: Option<u32>,
    pub nodes: Option<u64>,
    pub ms: Option<u64>,
    pub pv: String,
    pub iters: Vec<IterRec>,
}

#[derive(Debug, Clone)]
pub struct Game {
    pub pair: u64,
    pub book: u64,
    pub start: String,
    pub cand_seat: String,
    pub cand_points: f64,
    pub moves: Vec<Rec>,
}

// ---- minimal JSON field readers for the exact format game_json() writes ----

pub(crate) fn field<'a>(obj: &'a str, key: &str) -> Option<&'a str> {
    let pat = format!("\"{key}\":");
    let i = obj.find(&pat)? + pat.len();
    let rest = &obj[i..];
    if let Some(stripped) = rest.strip_prefix('"') {
        let mut end = 0;
        let bytes = stripped.as_bytes();
        while end < bytes.len() {
            if bytes[end] == b'\\' {
                end += 2;
                continue;
            }
            if bytes[end] == b'"' {
                break;
            }
            end += 1;
        }
        Some(&stripped[..end.min(stripped.len())])
    } else {
        let end = rest.find(|c| c == ',' || c == '}' || c == ']').unwrap_or(rest.len());
        Some(rest[..end].trim())
    }
}

pub(crate) fn num<T: std::str::FromStr>(obj: &str, key: &str) -> Option<T> {
    field(obj, key).filter(|v| *v != "null").and_then(|v| v.parse().ok())
}

fn parse_iters(raw: &str) -> Vec<IterRec> {
    raw.split(';')
        .filter(|s| !s.is_empty())
        .filter_map(|entry| {
            let mut f = entry.splitn(6, ':');
            let depth: u32 = f.next()?.parse().ok()?;
            let sc = f.next()?;
            let (score, mate) = if let Some(m) = sc.strip_prefix('m') {
                (None, m.parse().ok())
            } else if sc.is_empty() {
                (None, None)
            } else {
                (sc.parse().ok(), None)
            };
            let nodes: u64 = f.next()?.parse().ok()?;
            let ms: u64 = f.next()?.parse().ok()?;
            let rfh_raw = f.next()?;
            let (rfh_num, mirage) = match rfh_raw.strip_suffix('!') {
                Some(n) => (n, true),
                None => (rfh_raw, false),
            };
            let rfh: u8 = rfh_num.parse().unwrap_or(0);
            let pvs = f.next().unwrap_or("");
            let mut toks = pvs.split_whitespace();
            let pv0 = toks.next().unwrap_or("").to_string();
            let pv1 = toks.next().unwrap_or("").to_string();
            Some(IterRec { depth, score, mate, nodes, ms, rfh, mirage, pv0, pv1 })
        })
        .collect()
}

pub fn parse_game(line: &str) -> Result<Game, String> {
    let mstart = line.find("\"moves\":[").ok_or("no moves array")? + "\"moves\":[".len();
    let head = &line[..mstart];
    let body = &line[mstart..line.rfind(']').ok_or("unterminated moves")?];
    let mut moves = Vec::new();
    for obj in body.split("},{") {
        if obj.trim().is_empty() {
            continue;
        }
        moves.push(Rec {
            ply: num(obj, "ply").ok_or("move without ply")?,
            mover: field(obj, "mover").unwrap_or("").to_string(),
            mv: field(obj, "move").ok_or("move without move")?.to_string(),
            best_mv: field(obj, "engine_best").map(|s| s.to_string()),
            score: num::<i32>(obj, "engine_best_score").or_else(|| num(obj, "score")),
            mate: num(obj, "mate"),
            depth: num(obj, "depth"),
            nodes: num(obj, "nodes"),
            ms: num(obj, "ms"),
            pv: field(obj, "pv").unwrap_or("").to_string(),
            iters: field(obj, "it").map(parse_iters).unwrap_or_default(),
        });
    }
    Ok(Game {
        pair: num(head, "pair").ok_or("no pair")?,
        book: num(head, "book").ok_or("no book")?,
        start: field(head, "start").ok_or("no start")?.to_string(),
        cand_seat: field(head, "candidate_seat").ok_or("no seat")?.to_string(),
        cand_points: num(head, "candidate_points").ok_or("no points")?,
        moves,
    })
}

impl Rec {
    /// The best-move label of this ply: the engine's own first choice when the log has it, else the move played.
    pub fn label_mv(&self) -> &str {
        self.best_mv.as_deref().unwrap_or(&self.mv)
    }
}

/// Replay a game; returns the position BEFORE every move (same order as `moves`).
pub fn replay(g: &Game) -> Result<Vec<Board>, String> {
    let mut b = Board::from_ataxx_fen(&g.start).ok_or_else(|| format!("pair {} book {}: bad start fen", g.pair, g.book))?;
    let mut before = Vec::with_capacity(g.moves.len());
    for r in &g.moves {
        if r.mv == "FORFEIT" {
            before.push(b);
            break;
        }
        let m = b
            .parse_uai_move(&r.mv)
            .ok_or_else(|| format!("pair {} book {} ply {}: illegal move {}", g.pair, g.book, r.ply, r.mv))?;
        before.push(b);
        b = b.make_uai(m);
    }
    Ok(before)
}

/// Score in pawns-ish cp from the mover's view, mates mapped far outside the cp range for comparisons.
fn view(r: &Rec) -> Option<i32> {
    match (r.score, r.mate) {
        (_, Some(m)) if m > 0 => Some(20_000 - m),
        (_, Some(m)) => Some(-20_000 - m),
        (s, None) => s,
    }
}

pub struct Explained {
    /// ply, loser's eval before, loser's eval after (loser's view)
    pub turning_point: Option<(u32, i32, i32)>,
    /// first ply where the eventual loser's own eval was <= -200 cp
    pub loser_saw_it: Option<u32>,
    /// first ply where the loser itself reported being mated (the loss became provable)
    pub proven_lost: Option<u32>,
    /// ply, candidate view of the position by the mover, by the next mover (negated)
    pub disagreement: Option<(u32, i32, i32)>,
    /// largest off-PV eval surprise: (ply, expected pv[1], actual opp reply, mover cp drop at ply+2)
    pub off_pv_refutation: Option<(u32, String, String, i32)>,
}

/// Compare two UAI move strings modulo destination-only clone equivalence.
pub(crate) fn pv_mv_matches(a: &str, b: &str) -> bool {
    let ca: String = a.chars().filter(|&c| c != '-').collect();
    let cb: String = b.chars().filter(|&c| c != '-').collect();
    if ca.is_empty() || cb.is_empty() {
        return false;
    }
    if ca == cb {
        return true;
    }
    let is_adj = |s4: &[u8], s2: &[u8]| -> bool {
        if s4.len() != 4 || s2.len() != 2 || &s4[2..4] != s2 {
            return false;
        }
        (s4[0] as i32 - s4[2] as i32).abs() <= 1 && (s4[1] as i32 - s4[3] as i32).abs() <= 1
    };
    is_adj(ca.as_bytes(), cb.as_bytes()) || is_adj(cb.as_bytes(), ca.as_bytes())
}

pub fn explain(g: &Game) -> Explained {
    let cand_won = g.cand_points > 0.5;
    let loser = if (g.cand_points - 0.5).abs() < 1e-9 { None } else if cand_won { Some("baseline") } else { Some("candidate") };
    let mut tp = None;
    let mut saw = None;
    let mut proven = None;
    if let Some(l) = loser {
        let mine: Vec<&Rec> = g.moves.iter().filter(|r| r.mover == l).collect();
        proven = mine.iter().find(|r| r.mate.map_or(false, |m| m < 0)).map(|r| r.ply);
        // turning point = biggest drop between the loser's own NON-mate evals (a mate score only says the loss
        // became provable; the mistake happened earlier)
        let own: Vec<(u32, i32)> = mine.iter().filter(|r| r.mate.is_none()).filter_map(|r| r.score.map(|v| (r.ply, v))).collect();
        let mut best_drop = 0;
        for w in own.windows(2) {
            let drop = w[0].1 - w[1].1;
            if drop > best_drop {
                best_drop = drop;
                tp = Some((w[1].0, w[0].1, w[1].1));
            }
        }
        saw = own.iter().find(|(_, v)| *v <= -200).map(|(p, _)| *p);
    }
    // engines disagree: mover says v at ply p, the next mover says w at p+1 -> same position family, -w vs v
    let mut dis = None;
    let mut best_gap = 0;
    for w in g.moves.windows(2) {
        if w[0].mover == "-" || w[1].mover == "-" || w[0].mover == w[1].mover {
            continue;
        }
        if let (Some(a), Some(b)) = (view(&w[0]), view(&w[1])) {
            let gap = (a + b).abs();
            if gap > best_gap && a.abs() < 20_000 && b.abs() < 20_000 {
                best_gap = gap;
                dis = Some((w[0].ply, a, -b));
            }
        }
    }
    let mut off_pv_refutation = None;
    let mut best_ref_drop = 0i32;
    for i in 0..g.moves.len().saturating_sub(2) {
        let (r0, r1, r2) = (&g.moves[i], &g.moves[i + 1], &g.moves[i + 2]);
        if r0.mover == "-" || r1.mover == "-" || r0.mover == r1.mover || r0.mover != r2.mover {
            continue;
        }
        let Some(pv1) = r0.pv.split_whitespace().nth(1) else { continue };
        if pv_mv_matches(pv1, &r1.mv) {
            continue;
        }
        if let (Some(v0), Some(v2)) = (view(r0), view(r2)) {
            let drop = (v0 - v2).min(5000);
            if drop > best_ref_drop {
                best_ref_drop = drop;
                off_pv_refutation = Some((r1.ply, pv1.to_string(), r1.mv.clone(), drop));
            }
        }
    }
    Explained { turning_point: tp, loser_saw_it: saw, proven_lost: proven, disagreement: dis, off_pv_refutation }
}

pub(crate) fn read_games(paths: &[String]) -> Result<Vec<Game>, String> {
    let mut out = Vec::new();
    for p in paths {
        let text = fs::read_to_string(p).map_err(|e| format!("{p}: {e}"))?;
        for (n, line) in text.lines().enumerate().filter(|(_, l)| !l.trim().is_empty()) {
            let g = parse_game(line).map_err(|e| format!("{p}:{}: {e}", n + 1))?;
            replay(&g).map_err(|e| format!("{p}:{}: {e}", n + 1))?;
            out.push(g);
        }
    }
    Ok(out)
}

// ---- `ti insight` and the end-of-gate report: what the engines' own per-move evals say about how the games went ----

fn mean_z(xs: &[f64]) -> (f64, f64) {
    let n = xs.len() as f64;
    if xs.len() < 2 {
        return (xs.first().copied().unwrap_or(0.0), 0.0);
    }
    let mu = xs.iter().sum::<f64>() / n;
    let var = xs.iter().map(|x| (x - mu) * (x - mu)).sum::<f64>() / (n - 1.0);
    (mu, if var > 0.0 { mu / (var / n).sqrt() } else { 0.0 })
}

/// Information-only report (NEVER part of a verdict, docs/GATE_RULE.md): per side speed and depth, value lost per move
/// (mover's score + the opponent's next score, mates and |score| > 1500 skipped), the mid-game tilt of each game in the
/// candidate's view (mean of both engines' scores over plies 10..end-10) and how well it agrees with the results.
/// Needs >= 50 complete pairs with scored moves, else None.
pub fn insight(games: &[Game]) -> Option<String> {
    use std::collections::HashMap;
    #[derive(Default)]
    struct Side {
        nodes: u64,
        ms: u64,
        depths: Vec<u32>,
        moves: u64,
        blunders: u64,
        swing: f64,
    }
    let mut side: HashMap<&str, Side> = HashMap::new();
    // per pair: (result points, tilt cp, candidate swing mean, baseline swing mean) per game
    let mut pairs: HashMap<(u64, u64), Vec<(f64, f64, f64, f64)>> = HashMap::new();
    for g in games {
        let ok = |r: &Rec| r.mate.is_none() && r.score.map_or(false, |s| s.abs() <= 1500);
        let scored: Vec<&Rec> = g.moves.iter().filter(|r| ok(r)).collect();
        for r in &g.moves {
            let e = side.entry(if r.mover == "candidate" { "candidate" } else { "baseline" }).or_default();
            if let (Some(n), Some(ms)) = (r.nodes, r.ms) {
                e.nodes += n;
                e.ms += ms;
            }
            if let Some(d) = r.depth {
                e.depths.push(d);
            }
        }
        let mut sw = [(0.0f64, 0u32); 2]; // 0 = candidate, 1 = baseline
        for w in scored.windows(2) {
            // consecutive records only (a skipped unscored move breaks the chain)
            if w[1].ply != w[0].ply + 1 {
                continue;
            }
            let swing = (w[0].score.unwrap() + w[1].score.unwrap()) as f64;
            let i = if w[0].mover == "candidate" { 0 } else { 1 };
            sw[i].0 += swing;
            sw[i].1 += 1;
            let e = side.get_mut(if i == 0 { "candidate" } else { "baseline" }).unwrap();
            e.moves += 1;
            e.swing += swing;
            if swing > 150.0 {
                e.blunders += 1;
            }
        }
        if scored.len() < 30 || sw[0].1 == 0 || sw[1].1 == 0 {
            continue;
        }
        let cv: Vec<f64> = scored
            .iter()
            .map(|r| {
                let s = r.score.unwrap() as f64;
                if r.mover == "candidate" {
                    s
                } else {
                    -s
                }
            })
            .collect();
        let (lo, hi) = (10usize, (cv.len() - 10).max(11));
        let tilt = cv[lo..hi].iter().sum::<f64>() / (hi - lo) as f64;
        pairs
            .entry((g.pair, g.book))
            .or_default()
            .push((g.cand_points, tilt, sw[0].0 / sw[0].1 as f64, sw[1].0 / sw[1].1 as f64));
    }
    let full: Vec<&Vec<(f64, f64, f64, f64)>> = pairs.values().filter(|v| v.len() == 2).collect();
    if full.len() < 50 {
        return None;
    }
    let np = full.len() as f64;
    let act: Vec<f64> = full.iter().map(|v| (v[0].0 + v[1].0) / 2.0).collect();
    let tilt: Vec<f64> = full.iter().map(|v| (v[0].1 + v[1].1) / 2.0).collect();
    let lost: Vec<f64> = full.iter().map(|v| (v[0].3 - v[0].2 + v[1].3 - v[1].2) / 2.0).collect();
    let exp: f64 = full
        .iter()
        .map(|v| v.iter().map(|g| 1.0 / (1.0 + (-g.1 / 400.0).exp())).sum::<f64>() / 2.0)
        .sum::<f64>()
        / np;
    let (mt, zt) = mean_z(&tilt);
    let (ml, zl) = mean_z(&lost);
    let mut agree = 0u32;
    let mut decided = 0u32;
    for (a, t) in act.iter().zip(&tilt) {
        if *a != 0.5 && *t != 0.0 {
            decided += 1;
            if (a - 0.5) * t > 0.0 {
                agree += 1;
            }
        }
    }
    let mut out = String::from("-- eval insight (info only, not in the verdict) --\n");
    for name in ["candidate", "baseline"] {
        if let Some(e) = side.get(name) {
            let mut d = e.depths.clone();
            d.sort_unstable();
            out += &format!(
                "  {name:<9} {:>6.0} knps | median depth {} | value lost per move {:+.1} cp | blunders (>150 cp) {:.1}% of {} moves\n",
                e.nodes as f64 / e.ms.max(1) as f64,
                d.get(d.len() / 2).copied().unwrap_or(0),
                e.swing / e.moves.max(1) as f64,
                100.0 * e.blunders as f64 / e.moves.max(1) as f64,
                e.moves
            );
        }
    }
    out += &format!(
        "  actual score {:.4} | eval-expected score {:.4} ({} pairs)\n  mid-game tilt toward the candidate {:+.1} cp, paired z {:+.2} ({})\n  value lost per move, baseline minus candidate {:+.2} cp, paired z {:+.2} (+ = the candidate loses less)\n  result and tilt agree in sign on {}/{} decided pairs ({:.0}%)",
        act.iter().sum::<f64>() / np,
        exp,
        full.len(),
        mt,
        zt,
        if zt > 1.96 {
            "tilted to the CANDIDATE"
        } else if zt < -1.96 {
            "tilted to the BASELINE"
        } else {
            "no clear tilt"
        },
        ml,
        zl,
        agree,
        decided,
        100.0 * agree as f64 / decided.max(1) as f64
    );
    out += &insight_phases(games);
    let (kc, kb, wdl_blk) = insight_wdl_and_variance(games);
    if !wdl_blk.is_empty() {
        out.push('\n');
        out.push_str(&wdl_blk);
    }
    let pv_blk = insight_pv_dictation(games, kc, kb);
    if !pv_blk.is_empty() {
        out.push('\n');
        out.push_str(&pv_blk);
    }
    let it_blk = insight_iters(games);
    if !it_blk.is_empty() {
        out.push('\n');
        out.push_str(&it_blk);
    }
    Some(out)
}

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x.clamp(-30.0, 30.0)).exp())
}

/// Fit 1D logistic slope k in [0.0005, 0.020] (p = sigma(k * cp)) on non-mate moves vs game outcome, returning (k, Brier).
fn fit_wdl_scale(games: &[Game], target_mover: &str) -> (f64, f64) {
    let mut xs = Vec::new();
    let mut ys = Vec::new();
    for g in games {
        for r in &g.moves {
            if r.mover != target_mover || r.mate.is_some() {
                continue;
            }
            let Some(cp) = r.score else { continue };
            let y = if target_mover == "candidate" { g.cand_points } else { 1.0 - g.cand_points };
            xs.push(cp as f64);
            ys.push(y);
        }
    }
    if xs.len() < 20 {
        return (0.0025, 0.25);
    }
    let mut k = 0.0025f64;
    let reg = 1e-4 * xs.len() as f64;
    for _ in 0..12 {
        let mut g1 = reg * (k - 0.0025);
        let mut h = reg;
        for (&c, &y) in xs.iter().zip(&ys) {
            let p = sigmoid(k * c);
            g1 += (p - y) * c;
            h += p * (1.0 - p) * c * c;
        }
        if h <= 1e-12 {
            break;
        }
        k = (k - (g1 / h).clamp(-0.003, 0.003)).clamp(0.0005, 0.020);
    }
    let brier = xs
        .iter()
        .zip(&ys)
        .map(|(&c, &y)| {
            let d = sigmoid(k * c) - y;
            d * d
        })
        .sum::<f64>()
        / xs.len() as f64;
    (k, brier)
}

fn rec_wdl(r: &Rec, k: f64) -> Option<f64> {
    if let Some(m) = r.mate {
        return Some(if m > 0 { 1.0 } else { 0.0 });
    }
    r.score.map(|c| sigmoid(k * c as f64))
}

fn ols_slope(xs: &[f64], ys: &[f64]) -> f64 {
    if xs.len() < 4 || xs.len() != ys.len() {
        return 0.0;
    }
    let n = xs.len() as f64;
    let mx = xs.iter().sum::<f64>() / n;
    let my = ys.iter().sum::<f64>() / n;
    let mut num = 0.0;
    let mut den = 0.0;
    for (&x, &y) in xs.iter().zip(ys) {
        let dx = x - mx;
        num += dx * (y - my);
        den += dx * dx;
    }
    if den > 1e-12 { (num / den).clamp(0.0, 4.0) } else { 0.0 }
}

/// Calibrated WDL mutual grading (WPL%), empirical Brier trust, cross-ply mate mirages,
/// opening sharpness partition, and info-only cross-fitted soft-eval composite Elo.
fn insight_wdl_and_variance(games: &[Game]) -> (f64, f64, String) {
    use std::collections::HashMap;
    let (kc, bc) = fit_wdl_scale(games, "candidate");
    let (kb, bb) = fit_wdl_scale(games, "baseline");
    let trust_raw_c = 1.0 / bc.max(0.005);
    let trust_raw_b = 1.0 / bb.max(0.005);
    let trust_norm = 0.5 * (trust_raw_c + trust_raw_b);
    let (tc, tb) = (trust_raw_c / trust_norm, trust_raw_b / trust_norm);

    #[derive(Default, Clone, Copy)]
    struct SideW {
        wpl: f64,
        moves: u32,
        big: u32,
        mate_opp: u32,
        mate_mir: u32,
    }
    let mut agg = [SideW::default(); 2];
    // per pair: Vec of (cand_points, base_wpl_m - cand_wpl_m, base_wpl_sum - cand_wpl_sum, abs_tilt)
    let mut pairs: HashMap<(u64, u64), Vec<(f64, f64, f64, f64)>> = HashMap::new();
    for g in games {
        let mut sw = [SideW::default(); 2];
        for i in 0..g.moves.len().saturating_sub(1) {
            let (r0, r1) = (&g.moves[i], &g.moves[i + 1]);
            if r0.mover == "-" || r1.mover == "-" || r0.mover == r1.mover || r1.ply != r0.ply + 1 || r0.mate.is_some() {
                continue;
            }
            let who = if r0.mover == "candidate" { 0 } else { 1 };
            let (k0, k1) = if who == 0 { (kc, kb) } else { (kb, kc) };
            let (Some(w0), Some(w1)) = (rec_wdl(r0, k0), rec_wdl(r1, k1)) else { continue };
            let loss = w0 - (1.0 - w1);
            sw[who].wpl += loss;
            sw[who].moves += 1;
            agg[who].wpl += loss;
            agg[who].moves += 1;
            if loss >= 0.15 {
                agg[who].big += 1;
            }
        }
        for i in 0..g.moves.len().saturating_sub(2) {
            let (r0, r2) = (&g.moves[i], &g.moves[i + 2]);
            if r0.mover == "-" || r0.mover != r2.mover || r2.ply != r0.ply + 2 {
                continue;
            }
            if let Some(m0) = r0.mate.filter(|&m| m > 0) {
                let who = if r0.mover == "candidate" { 0 } else { 1 };
                agg[who].mate_opp += 1;
                let vanished = r2.mate.map_or(true, |m2| m2 <= 0 || (m0 >= 2 && m2 >= m0));
                if vanished {
                    agg[who].mate_mir += 1;
                }
            }
        }
        if sw[0].moves > 0 && sw[1].moves > 0 {
            let diff_m = sw[1].wpl / sw[1].moves as f64 - sw[0].wpl / sw[0].moves as f64;
            let edge_g = sw[1].wpl - sw[0].wpl;
            let abs_tilt = game_tilt(g).map(|t| t.abs()).unwrap_or(0.0);
            pairs.entry((g.pair, g.book)).or_default().push((g.cand_points, diff_m, edge_g, abs_tilt));
        }
    }

    let full: Vec<&Vec<(f64, f64, f64, f64)>> = pairs.values().filter(|v| v.len() == 2).collect();
    if full.len() < 50 {
        return (kc, kb, String::new());
    }
    let y_pair: Vec<f64> = full.iter().map(|v| 0.5 * (v[0].0 + v[1].0)).collect();
    let wpl_m_diff: Vec<f64> = full.iter().map(|v| 50.0 * (v[0].1 + v[1].1)).collect(); // in %/move
    let wpl_g_diff: Vec<f64> = full.iter().map(|v| 0.5 * (v[0].2 + v[1].2)).collect();
    let (mw, zw) = mean_z(&wpl_m_diff);

    // 2-fold cross-fitted outcome + calibrated WPL edge composite (info-only)
    let mut even_x = Vec::new();
    let mut even_y = Vec::new();
    let mut odd_x = Vec::new();
    let mut odd_y = Vec::new();
    for (idx, (&y, &x)) in y_pair.iter().zip(&wpl_g_diff).enumerate() {
        if idx % 2 == 0 {
            even_x.push(x);
            even_y.push(y - 0.5);
        } else {
            odd_x.push(x);
            odd_y.push(y - 0.5);
        }
    }
    let beta_even = ols_slope(&even_x, &even_y);
    let beta_odd = ols_slope(&odd_x, &odd_y);
    let soft: Vec<f64> = y_pair
        .iter()
        .zip(&wpl_g_diff)
        .enumerate()
        .map(|(idx, (&y, &x))| {
            let b = if idx % 2 == 0 { beta_odd } else { beta_even };
            let w_hat = (0.5 + b * x).clamp(0.0, 1.0);
            0.5 * y + 0.5 * w_hat
        })
        .collect();
    let np = full.len() as f64;
    let mu_raw = y_pair.iter().sum::<f64>() / np;
    let se_raw = (y_pair.iter().map(|&y| (y - mu_raw).powi(2)).sum::<f64>() / (np * (np - 1.0))).sqrt().max(1e-6);
    let mu_soft = soft.iter().sum::<f64>() / np;
    let se_soft = (soft.iter().map(|&y| (y - mu_soft).powi(2)).sum::<f64>() / (np * (np - 1.0))).sqrt().max(1e-6);
    let elo_of = |s: f64| -400.0 * (1.0 / s.clamp(1e-6, 1.0 - 1e-6) - 1.0).log10();
    let ci_of = |mu: f64, se: f64| 0.5 * (elo_of(mu + 1.95996 * se) - elo_of(mu - 1.95996 * se));

    // Book sharpness breakdown (balanced |tilt| < 100 vs skewed >= 100)
    let mut bal_y = Vec::new();
    let mut skew_dec = 0u32;
    let mut skew_n = 0u32;
    for (v, &y) in full.iter().zip(&y_pair) {
        let tilt = 0.5 * (v[0].3 + v[1].3);
        if tilt < 100.0 {
            bal_y.push(y);
        } else {
            skew_n += 1;
            if (y - 0.5).abs() > 1e-9 {
                skew_dec += 1;
            }
        }
    }
    let (bal_s, bal_z) = if bal_y.len() >= 2 {
        let shifted: Vec<f64> = bal_y.iter().map(|&y| y - 0.5).collect();
        let (dm, z) = mean_z(&shifted);
        (elo_of(0.5 + dm), z)
    } else {
        (0.0, 0.0)
    };

    let c_wpl = 100.0 * agg[0].wpl / agg[0].moves.max(1) as f64;
    let b_wpl = 100.0 * agg[1].wpl / agg[1].moves.max(1) as f64;
    let c_big = 100.0 * agg[0].big as f64 / agg[0].moves.max(1) as f64;
    let b_big = 100.0 * agg[1].big as f64 / agg[1].moves.max(1) as f64;
    let mut out = String::from("  calibrated WDL & variance analysis (info only; raw paired games decide SPRT):\n");
    out += &format!(
        "    candidate k={kc:.5} Brier={bc:.4} (trust {tc:.2}) | WPL {c_wpl:+.3}%/m | blnd>=15% {c_big:.1}% | cross-ply mate mirages {}/{}\n",
        agg[0].mate_mir, agg[0].mate_opp
    );
    out += &format!(
        "    baseline  k={kb:.5} Brier={bb:.4} (trust {tb:.2}) | WPL {b_wpl:+.3}%/m | blnd>=15% {b_big:.1}% | cross-ply mate mirages {}/{}\n",
        agg[1].mate_mir, agg[1].mate_opp
    );
    out += &format!(
        "    calibrated WPL diff (base - cand) {mw:+.3}%/m (z {zw:+.2}) | soft-WDL composite Elo {:+.1} +- {:.1} (raw {:+.1} +- {:.1}, xfold beta {:.2}/{:.2})\n",
        elo_of(mu_soft), ci_of(mu_soft, se_soft), elo_of(mu_raw), ci_of(mu_raw, se_raw), beta_even, beta_odd
    );
    out += &format!(
        "    book partition: balanced(|tilt|<100) {} pairs (Elo {:+.1}, z {:+.2}) | skewed {} pairs ({:.0}% decisive)",
        bal_y.len(), bal_s, bal_z, skew_n, 100.0 * skew_dec as f64 / skew_n.max(1) as f64
    );
    (kc, kb, out)
}

/// PV Dictation & Adversarial Surprise Matrix (`pv_A(t)[1..]` vs actual game replies & calibrated WDL deltas).
fn insight_pv_dictation(games: &[Game], kc: f64, kb: f64) -> String {
    use std::collections::HashMap;
    #[derive(Default, Clone, Copy)]
    struct SidePv {
        preds: u32,
        opp_hits: u32,
        span_sum: u64,
        pv2_ok: u32,
        pv2_tot: u32,
        true_dict: u32,
        forced_conc: u32,
        adv_refut: u32,
        refut_dw: f64,
    }
    let mut agg = [SidePv::default(); 2];
    let mut pairs: HashMap<(u64, u64), Vec<[SidePv; 2]>> = HashMap::new();
    for g in games {
        let mut s = [SidePv::default(); 2];
        for i in 0..g.moves.len().saturating_sub(1) {
            let (r0, r1) = (&g.moves[i], &g.moves[i + 1]);
            if r0.mover == "-" || r1.mover == "-" || r0.mover == r1.mover || r1.ply != r0.ply + 1 {
                continue;
            }
            let pv_toks: Vec<&str> = r0.pv.split_whitespace().collect();
            if pv_toks.len() < 2 {
                continue;
            }
            let who = if r0.mover == "candidate" { 0 } else { 1 };
            let k_self = if who == 0 { kc } else { kb };
            let opp_hit = pv_mv_matches(pv_toks[1], &r1.mv);
            let mut span = 0u64;
            for step in 1..pv_toks.len() {
                if i + step >= g.moves.len() || g.moves[i + step].ply != r0.ply + step as u32 {
                    break;
                }
                if !pv_mv_matches(pv_toks[step], &g.moves[i + step].mv) {
                    break;
                }
                span += 1;
            }
            for tgt in [&mut s[who], &mut agg[who]] {
                tgt.preds += 1;
                if opp_hit {
                    tgt.opp_hits += 1;
                }
                tgt.span_sum += span;
            }
            if let Some(r2) = g.moves.get(i + 2).filter(|r2| r2.ply == r0.ply + 2 && r2.mover == r0.mover) {
                if opp_hit && pv_toks.len() >= 3 {
                    let ok2 = pv_mv_matches(pv_toks[2], &r2.mv);
                    for tgt in [&mut s[who], &mut agg[who]] {
                        tgt.pv2_tot += 1;
                        if ok2 {
                            tgt.pv2_ok += 1;
                        }
                    }
                }
                if let (Some(w0), Some(w2)) = (rec_wdl(r0, k_self), rec_wdl(r2, k_self)) {
                    let dw = w2 - w0;
                    for tgt in [&mut s[who], &mut agg[who]] {
                        if opp_hit && dw >= -0.03 {
                            tgt.true_dict += 1;
                        } else if !opp_hit && dw > 0.03 {
                            tgt.forced_conc += 1;
                        } else if !opp_hit && dw < -0.03 {
                            tgt.adv_refut += 1;
                            tgt.refut_dw += dw;
                        }
                    }
                }
            }
        }
        if s[0].preds > 0 && s[1].preds > 0 {
            pairs.entry((g.pair, g.book)).or_default().push(s);
        }
    }
    let full: Vec<&Vec<[SidePv; 2]>> = pairs.values().filter(|v| v.len() == 2).collect();
    if full.len() < 20 {
        return String::new();
    }
    let mut hit_diff = Vec::with_capacity(full.len());
    let mut span_diff = Vec::with_capacity(full.len());
    for v in &full {
        let mut hd = 0.0;
        let mut sd = 0.0;
        for g in v.iter() {
            hd += 100.0 * (g[0].opp_hits as f64 / g[0].preds as f64 - g[1].opp_hits as f64 / g[1].preds as f64);
            sd += g[0].span_sum as f64 / g[0].preds as f64 - g[1].span_sum as f64 / g[1].preds as f64;
        }
        hit_diff.push(0.5 * hd);
        span_diff.push(0.5 * sd);
    }
    let (mhd, zhd) = mean_z(&hit_diff);
    let (msd, zsd) = mean_z(&span_diff);
    let mut out = String::from("  PV dictation & adversarial surprise (info only):\n");
    for (idx, name) in [(0usize, "candidate"), (1usize, "baseline ")] {
        let s = agg[idx];
        let p = s.preds.max(1) as f64;
        let mr = if s.adv_refut > 0 { 100.0 * s.refut_dw / s.adv_refut as f64 } else { 0.0 };
        out += &format!(
            "    {name} opp_pv1 {:5.1}% | dict_plies {:.2} | self_pv2 {:5.1}% | TrueDict {} | Concession {} | Refutation {} ({:+.1}% avg dw) | {} preds\n",
            100.0 * s.opp_hits as f64 / p,
            s.span_sum as f64 / p,
            100.0 * s.pv2_ok as f64 / s.pv2_tot.max(1) as f64,
            s.true_dict,
            s.forced_conc,
            s.adv_refut,
            mr,
            s.preds
        );
    }
    out += &format!(
        "    paired dictation diff (cand - base): opp_pv1 {mhd:+.2}% (z {zhd:+.2}) | dict_plies {msd:+.3} (z {zsd:+.2}) | {} pairs",
        full.len()
    );
    out
}

/// Per-depth tree efficiency (geoN / EBF) & iterative-deepening search stability from `"it"` telemetry.
fn insight_iters(games: &[Game]) -> String {
    use std::collections::{HashMap, HashSet};
    #[derive(Default, Clone, Copy)]
    struct SideIt {
        moves: u32,
        bmc: u64,
        pvc: u64,
        swg: f64,
        rfh: u64,
        mir: u64,
        ttfm_d: f64,
        ttfm_npct: f64,
        ttfm_n: u32,
    }
    const MAXD: usize = 32;
    let mut agg = [SideIt::default(); 2];
    let mut dlns = [[(0.0f64, 0u64); 2]; MAXD];
    let mut pair_counts: HashMap<(u64, u64), u32> = HashMap::new();
    let mut pairs_with_it: HashSet<(u64, u64)> = HashSet::new();
    for g in games {
        let mut has_it = false;
        for r in &g.moves {
            if (r.mover != "candidate" && r.mover != "baseline") || r.iters.is_empty() {
                continue;
            }
            has_it = true;
            let who = if r.mover == "candidate" { 0 } else { 1 };
            let st = &mut agg[who];
            st.moves += 1;
            for it in &r.iters {
                st.rfh += u64::from(it.rfh);
                if it.mirage {
                    st.mir += 1;
                }
                let d = it.depth as usize;
                if d > 0 && d < MAXD && it.nodes > 0 {
                    dlns[d][who].0 += (it.nodes as f64).ln();
                    dlns[d][who].1 += 1;
                }
            }
            if r.iters.len() >= 2 {
                for w in r.iters.windows(2) {
                    if !w[0].pv0.is_empty() && !w[1].pv0.is_empty() && !pv_mv_matches(&w[0].pv0, &w[1].pv0) {
                        st.bmc += 1;
                    }
                    if !w[0].pv1.is_empty() && !w[1].pv1.is_empty() && !pv_mv_matches(&w[0].pv1, &w[1].pv1) {
                        st.pvc += 1;
                    }
                    if let (Some(a), Some(b)) = (w[0].score, w[1].score) {
                        st.swg += (b - a).abs() as f64;
                    }
                    if w[0].mate.is_some() && w[1].mate.is_none() && !w[1].mirage {
                        st.mir += 1;
                    }
                }
            }
            let last = r.iters.last().unwrap();
            if !last.pv0.is_empty() {
                if let Some(lock) = (0..r.iters.len()).find(|&j| r.iters[j..].iter().all(|x| pv_mv_matches(&x.pv0, &last.pv0))) {
                    st.ttfm_d += r.iters[lock].depth as f64;
                    st.ttfm_npct += if last.nodes > 0 { 100.0 * r.iters[lock].nodes as f64 / last.nodes as f64 } else { 100.0 };
                    st.ttfm_n += 1;
                }
            }
        }
        if has_it {
            *pair_counts.entry((g.pair, g.book)).or_default() += 1;
        }
    }
    for (k, n) in pair_counts {
        if n == 2 {
            pairs_with_it.insert(k);
        }
    }
    if pairs_with_it.len() < 20 || agg[0].moves == 0 || agg[1].moves == 0 {
        return String::new();
    }
    let mut out = String::from("  per-depth search telemetry & EBF (from completed ID iterations):\n");
    for (idx, name) in [(0usize, "candidate"), (1usize, "baseline ")] {
        let s = agg[idx];
        let m = s.moves.max(1) as f64;
        let tn = s.ttfm_n.max(1) as f64;
        out += &format!(
            "    {name} bmc {:.2}/m | pvc {:.2}/m | swg {:.1} cp/m | rfh {:.2}/m | mir {} | ttfm d={:.1} ({:.0}% nodes)\n",
            s.bmc as f64 / m,
            s.pvc as f64 / m,
            s.swg / m,
            s.rfh as f64 / m,
            s.mir,
            s.ttfm_d / tn,
            s.ttfm_npct / tn
        );
    }
    let mut prev_geo: [Option<f64>; 2] = [None, None];
    for d in 2..MAXD {
        if dlns[d][0].1 < 30 || dlns[d][1].1 < 30 {
            prev_geo = [None, None];
            continue;
        }
        let gc = (dlns[d][0].0 / dlns[d][0].1 as f64).exp();
        let gb = (dlns[d][1].0 / dlns[d][1].1 as f64).exp();
        let ebf_c = prev_geo[0].map(|p| format!("{:.2}", gc / p)).unwrap_or_else(|| "   -".into());
        let ebf_b = prev_geo[1].map(|p| format!("{:.2}", gb / p)).unwrap_or_else(|| "   -".into());
        out += &format!(
            "    d={d:<2} geoN cand {:>7.0} (EBF {ebf_c:>4}) | base {:>7.0} (EBF {ebf_b:>4}) | n={}/{}\n",
            gc, gb, dlns[d][0].1, dlns[d][1].1
        );
        prev_geo = [Some(gc), Some(gb)];
    }
    out.trim_end().to_string()
}

/// Where in the game the candidate gained (user 2026-11-02: "was the win in the endgame or across the board?"). Plies are
/// bucketed 0-19 / 20-39 / 40-59 / 60-79 / 80+. Per bucket and per complete pair (both seats averaged): value lost per move of
/// the baseline minus the candidate (+ = the candidate loses less), the cp that adds up to per game, and the candidate-view eval
/// level (mean of both engines' scores, + = the candidate stands better), each with its paired z.
fn insight_phases(games: &[Game]) -> String {
    use std::collections::HashMap;
    const EDGES: [usize; 6] = [0, 20, 40, 60, 80, 1000];
    const NB: usize = 5;
    let bucket = |ply: usize| (0..NB).find(|&i| ply < EDGES[i + 1]).unwrap_or(NB - 1);
    // per game: per bucket (cand swing sum, cand n, base swing sum, base n, eval-level sum, eval n)
    type G = [(f64, u32, f64, u32, f64, u32); NB];
    let mut pairs: HashMap<(u64, u64), Vec<G>> = HashMap::new();
    for g in games {
        let ok = |r: &Rec| r.mate.is_none() && r.score.map_or(false, |s| s.abs() <= 1500);
        let scored: Vec<&Rec> = g.moves.iter().filter(|r| ok(r)).collect();
        if scored.len() < 30 {
            continue;
        }
        let mut a: G = [(0.0, 0, 0.0, 0, 0.0, 0); NB];
        for r in &scored {
            let s = r.score.unwrap() as f64;
            let e = &mut a[bucket(r.ply as usize)];
            e.4 += if r.mover == "candidate" { s } else { -s };
            e.5 += 1;
        }
        for w in scored.windows(2) {
            if w[1].ply != w[0].ply + 1 {
                continue;
            }
            let swing = (w[0].score.unwrap() + w[1].score.unwrap()) as f64;
            let e = &mut a[bucket(w[0].ply as usize)];
            if w[0].mover == "candidate" {
                e.0 += swing;
                e.1 += 1;
            } else {
                e.2 += swing;
                e.3 += 1;
            }
        }
        pairs.entry((g.pair, g.book)).or_default().push(a);
    }
    let full: Vec<&Vec<G>> = pairs.values().filter(|v| v.len() == 2).collect();
    if full.len() < 50 {
        return String::new();
    }
    let mut out = String::from("\n  by game phase (ply buckets; per-move diff = baseline minus candidate value lost, + = candidate better):\n");
    for b in 0..NB {
        let mut diff = Vec::new();
        let mut saved = Vec::new();
        let mut lvl = Vec::new();
        for v in &full {
            let (mut d, mut s, mut l, mut nd, mut nl) = (0.0, 0.0, 0.0, 0.0, 0.0);
            for g in v.iter() {
                let e = g[b];
                if e.1 > 0 && e.3 > 0 {
                    d += e.2 / e.3 as f64 - e.0 / e.1 as f64;
                    // cp per game this bucket adds up to: per-move diff x the moves one side makes here
                    s += (e.2 / e.3 as f64 - e.0 / e.1 as f64) * (e.1 + e.3) as f64 / 2.0;
                    nd += 1.0;
                }
                if e.5 > 0 {
                    l += e.4 / e.5 as f64;
                    nl += 1.0;
                }
            }
            if nd > 0.0 {
                diff.push(d / nd);
                saved.push(s / nd);
            }
            if nl > 0.0 {
                lvl.push(l / nl);
            }
        }
        let (md, zd) = mean_z(&diff);
        let (ms, _) = mean_z(&saved);
        let (ml, zl) = mean_z(&lvl);
        let name = if b + 1 == NB { format!("{}+", EDGES[b]) } else { format!("{}-{}", EDGES[b], EDGES[b + 1] - 1) };
        out += &format!(
            "    ply {name:<6} value lost diff {md:+6.2} cp/move (z {zd:+5.2}) = {ms:+7.1} cp per game | eval level for the candidate {ml:+7.1} cp (z {zl:+5.2}) | {} pairs\n",
            diff.len()
        );
    }
    out.trim_end().to_string()
}

/// D578 (user: 'use the analysis tools to find where each improvement gains, keep it only there; that is the point of
/// the 16 buckets for the NNUE and the search'): the same paired value-lost diff as insight_phases, but bucketed by the
/// NNUE bucket of the position the move was played in (empties * 16 / 50, 14 = start, 0 = full board), so a per-bucket
/// search knob can be kept exactly in the buckets where the candidate loses less value.
/// Per NNUE bucket (empties*16/50), per complete pair (both seats averaged): baseline minus candidate value lost per move
/// (cp, + = the candidate loses less), plus the number of moves seen. Shared by `ti insight --by-bucket` and the per-stage
/// SPSA fitness (D1326), so both read the games the same way.
pub fn bucket_value_lost_diffs(games: &[Game]) -> Vec<(Vec<f64>, u32)> {
    use std::collections::HashMap;
    const NB: usize = 16;
    type G = [(f64, u32, f64, u32); NB];
    let mut pairs: HashMap<(u64, u64), Vec<G>> = HashMap::new();
    for g in games {
        let Ok(boards) = replay(g) else { continue };
        let ok = |r: &Rec| r.mate.is_none() && r.score.map_or(false, |s| s.abs() <= 1500);
        let mut a: G = [(0.0, 0, 0.0, 0); NB];
        for i in 0..g.moves.len().saturating_sub(1) {
            let (r0, r1) = (&g.moves[i], &g.moves[i + 1]);
            if !ok(r0) || !ok(r1) || r1.ply != r0.ply + 1 || i >= boards.len() {
                continue;
            }
            let bk = ((boards[i].empty().count_ones() as usize) * NB / 50).min(NB - 1);
            let swing = (r0.score.unwrap() + r1.score.unwrap()) as f64;
            let e = &mut a[bk];
            if r0.mover == "candidate" {
                e.0 += swing;
                e.1 += 1;
            } else {
                e.2 += swing;
                e.3 += 1;
            }
        }
        pairs.entry((g.pair, g.book)).or_default().push(a);
    }
    let full: Vec<&Vec<G>> = pairs.values().filter(|v| v.len() == 2).collect();
    let mut res = Vec::with_capacity(NB);
    for b in 0..NB {
        let (mut diff, mut moves) = (Vec::new(), 0u32);
        for v in &full {
            let (mut d, mut nd) = (0.0, 0.0);
            for g in v.iter() {
                let e = g[b];
                if e.1 > 0 && e.3 > 0 {
                    d += e.2 / e.3 as f64 - e.0 / e.1 as f64;
                    nd += 1.0;
                    moves += e.1 + e.3;
                }
            }
            if nd > 0.0 {
                diff.push(d / nd);
            }
        }
        res.push((diff, moves));
    }
    res
}

pub(crate) fn mean_z_pub(v: &[f64]) -> (f64, f64) {
    mean_z(v)
}

pub fn bucket_wpl_diffs(games: &[Game], kc: f64, kb: f64) -> Vec<(Vec<f64>, Vec<f64>, u32)> {
    use std::collections::HashMap;
    const NB: usize = 16;
    type G = [(f64, u32, f64, u32); NB];
    let mut pairs: HashMap<(u64, u64), Vec<G>> = HashMap::new();
    for g in games {
        let Ok(boards) = replay(g) else { continue };
        let mut a: G = [(0.0, 0, 0.0, 0); NB];
        for i in 0..g.moves.len().saturating_sub(1) {
            let (r0, r1) = (&g.moves[i], &g.moves[i + 1]);
            if r0.mate.is_some() || r0.mover == "-" || r1.mover == "-" || r0.mover == r1.mover || r1.ply != r0.ply + 1 || i >= boards.len() {
                continue;
            }
            let who = if r0.mover == "candidate" { 0 } else { 1 };
            let (k0, k1) = if who == 0 { (kc, kb) } else { (kb, kc) };
            let (Some(w0), Some(w1)) = (rec_wdl(r0, k0), rec_wdl(r1, k1)) else { continue };
            let loss = w0 - (1.0 - w1);
            let bk = ((boards[i].empty().count_ones() as usize) * NB / 50).min(NB - 1);
            let e = &mut a[bk];
            if who == 0 {
                e.0 += loss;
                e.1 += 1;
            } else {
                e.2 += loss;
                e.3 += 1;
            }
        }
        pairs.entry((g.pair, g.book)).or_default().push(a);
    }
    let full: Vec<&Vec<G>> = pairs.values().filter(|v| v.len() == 2).collect();
    let mut res = Vec::with_capacity(NB);
    for b in 0..NB {
        let (mut pct_diff, mut game_wpl_diff, mut moves) = (Vec::new(), Vec::new(), 0u32);
        for v in &full {
            let (mut d, mut gdiff, mut nd) = (0.0, 0.0, 0.0);
            for g in v.iter() {
                let e = g[b];
                if e.1 > 0 && e.3 > 0 {
                    let dm = e.2 / e.3 as f64 - e.0 / e.1 as f64;
                    d += 100.0 * dm;
                    gdiff += dm * (e.1 + e.3) as f64 * 0.5;
                    nd += 1.0;
                    moves += e.1 + e.3;
                }
            }
            if nd > 0.0 {
                pct_diff.push(d / nd);
                game_wpl_diff.push(gdiff / nd);
            }
        }
        res.push((pct_diff, game_wpl_diff, moves));
    }
    res
}

fn insight_nnue_buckets(games: &[Game]) -> String {
    const NB: usize = 16;
    let per = bucket_value_lost_diffs(games);
    let (kc, _) = fit_wdl_scale(games, "candidate");
    let (kb, _) = fit_wdl_scale(games, "baseline");
    let wpl_per = bucket_wpl_diffs(games, kc, kb);
    let mut npairs = std::collections::HashSet::new();
    let mut seen: std::collections::HashMap<(u64, u64), u32> = std::collections::HashMap::new();
    for g in games {
        if replay(g).is_ok() {
            *seen.entry((g.pair, g.book)).or_default() += 1;
        }
    }
    for (k, n) in seen {
        if n == 2 {
            npairs.insert(k);
        }
    }
    if npairs.len() < 50 {
        return String::new();
    }
    let elo_scale = 1600.0 / std::f64::consts::LN_10;
    let mut out = String::from(
        "\n  by NNUE bucket (empties*16/50; per-move diff = baseline minus candidate lost, + = candidate better):\n",
    );
    for b in (0..NB).rev() {
        let (diff, moves) = (&per[b].0, per[b].1);
        let (w_pct, w_game, _) = (&wpl_per[b].0, &wpl_per[b].1, wpl_per[b].2);
        if diff.len() < 10 {
            continue;
        }
        let (md, zd) = mean_z(diff);
        let (mw, zw) = mean_z(w_pct);
        let (mg, _) = mean_z(w_game);
        let (lo, hi) = ((b * 50 + 15) / 16, ((b + 1) * 50 + 15) / 16 - 1);
        out += &format!(
            "    b{b:<2} empties {lo:2}-{hi:<2} val {md:+7.2} cp/m (z {zd:+5.2}) | WPL {mw:+6.3}%/m (z {zw:+5.2}, ~{:+5.1} Elo) | {} pairs, {moves} moves\n",
            elo_scale * mg,
            diff.len()
        );
    }
    out.trim_end().to_string()
}

pub fn run_insight(args: &[String]) -> ExitCode {
    let paths: Vec<String> = args.iter().filter(|a| !a.starts_with("--")).cloned().collect();
    if paths.is_empty() {
        eprintln!("usage: ti insight <run.games.jsonl> [...]");
        return ExitCode::FAILURE;
    }
    let mut games = Vec::new();
    for p in &paths {
        match fs::read_to_string(p) {
            Ok(t) => games.extend(t.lines().filter(|l| !l.trim().is_empty()).filter_map(|l| parse_game(l).ok())),
            Err(e) => {
                eprintln!("insight: {p}: {e}");
                return ExitCode::FAILURE;
            }
        }
    }
    match insight(&games) {
        Some(r) => {
            println!("{r}");
            if args.iter().any(|a| a == "--by-bucket") {
                println!("{}", insight_nnue_buckets(&games));
            }
        }
        None => println!("insight: fewer than 50 complete pairs with scored moves ({} games read)", games.len()),
    }
    ExitCode::SUCCESS
}

pub fn run_analyze(args: &[String]) -> ExitCode {
    let paths: Vec<String> = args.iter().filter(|a| !a.starts_with("--")).cloned().collect();
    if paths.is_empty() {
        eprintln!("usage: ti analyze <run.games.jsonl> [...]");
        return ExitCode::FAILURE;
    }
    let games = match read_games(&paths) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("analyze: {e}");
            return ExitCode::FAILURE;
        }
    };
    let has_pv = games.iter().any(|g| g.moves.iter().any(|r| r.pv.split_whitespace().nth(1).is_some()));
    if has_pv {
        println!("pair book seat result | turning point (ply: loser's own eval before -> after) | loser saw it (<= -200) | proven lost (loser reports mate) | biggest disagreement (ply: mover vs next mover, mover's view) | biggest off-PV surprise (ply: exp->act drop)");
    } else {
        println!("pair book seat result | turning point (ply: loser's own eval before -> after) | loser saw it (<= -200) | proven lost (loser reports mate) | biggest disagreement (ply: mover vs next mover, mover's view)");
    }
    for g in &games {
        let e = explain(g);
        let res = if g.cand_points > 0.5 { "W" } else if g.cand_points < 0.5 { "L" } else { "D" };
        let tp = e.turning_point.map(|(p, a, b)| format!("{p}: {a:+} -> {b:+}")).unwrap_or_else(|| "-".into());
        let saw = e.loser_saw_it.map(|p| p.to_string()).unwrap_or_else(|| "-".into());
        let dis = e.disagreement.map(|(p, a, b)| format!("{p}: {a:+} vs {b:+}")).unwrap_or_else(|| "-".into());
        let proven = e.proven_lost.map(|p| p.to_string()).unwrap_or_else(|| "-".into());
        if has_pv {
            let opr = e.off_pv_refutation.map(|(p, exp, act, d)| format!("{p}: {exp}->{act} ({d:+})")).unwrap_or_else(|| "-".into());
            println!("{:>4} {:>5} {} {} | {tp} | {saw} | {proven} | {dis} | {opr}", g.pair, g.book, g.cand_seat, res);
        } else {
            println!("{:>4} {:>5} {} {} | {tp} | {saw} | {proven} | {dis}", g.pair, g.book, g.cand_seat, res);
        }
    }
    println!("{} games replayed and checked", games.len());
    ExitCode::SUCCESS
}

/// Per-book-line decisiveness census for `ti bookprune` (unit-testable core).
/// A pair is informative iff its total candidate score != 1.0: WW/LL/WD/LD move
/// the pentanomial; 1-1 splits (WL, plus rare DD) carry no engine signal and are
/// usually the seats deciding, not the engines.
/// Mid-game tilt of one game, candidate view (same formula as `insight`):
/// mean of signed scores over plies 10..end-10, |score|<=1500, no mates.
/// Sign = which engine the engine thinks is winning; magnitude = how one-sided
/// the position is. High |tilt| on a book line = the opening is lopsided, so
/// pairs will split by seat regardless of the two engines' relative strength.
pub fn game_tilt(g: &Game) -> Option<f64> {
    let scored: Vec<&Rec> = g.moves.iter().filter(|r| r.mate.is_none() && r.score.map_or(false, |s| s.abs() <= 1500)).collect();
    if scored.len() < 30 {
        return None;
    }
    let cv: Vec<f64> = scored
        .iter()
        .map(|r| {
            let s = r.score.unwrap() as f64;
            if r.mover == "candidate" {
                s
            } else {
                -s
            }
        })
        .collect();
    let (lo, hi) = (10usize, (cv.len() - 10).max(11));
    Some(cv[lo..hi].iter().sum::<f64>() / (hi - lo) as f64)
}

pub fn book_decisiveness(pairs: &[(f64, f64)]) -> (u32, u32) {
    let mut decisive = 0u32;
    for &(a, b) in pairs {
        if (a + b - 1.0).abs() > 1e-9 {
            decisive += 1;
        }
    }
    (pairs.len() as u32, decisive)
}

/// Drop rule for one measured book line: drop only with enough pairs AND a
/// split rate above the cap. Unmeasured lines are always kept (never drop
/// what you haven't measured).
pub fn keep_book_line(pairs: u64, decisive: u32, min_pairs: u64, max_split: f64) -> bool {
    if (pairs as u64) < min_pairs {
        return true;
    }
    1.0 - decisive as f64 / pairs as f64 <= max_split
}

pub fn run_bookprune(args: &[String]) -> ExitCode {
    let Some(book_path) = crate::flag_str(args, "--book") else {
        eprintln!("usage: ti bookprune --book <50k file> --out <pruned> [--min-pairs N] [--max-split F] [--force] <run.games.jsonl> [...]");
        return ExitCode::FAILURE;
    };
    let Some(out_path) = crate::flag_str(args, "--out") else {
        eprintln!("usage: ti bookprune --book <50k file> --out <pruned> [--min-pairs N] [--max-split F] [--force] <run.games.jsonl> [...]");
        return ExitCode::FAILURE;
    };
    let min_pairs: u64 = crate::flag_str(args, "--min-pairs").and_then(|s| s.parse().ok()).unwrap_or(4);
    let max_split: f64 = crate::flag_str(args, "--max-split").and_then(|s| s.parse().ok()).unwrap_or(0.75);
    // Max mean |mid-game tilt| a kept line may have (cp). Default 400 = the eval's
    // own win-prob scale: beyond that both engines already see a won side, so the
    // pair is decided by the seat, not by the engine delta. 0 disables the tilt rule.
    let max_tilt: f64 = crate::flag_str(args, "--max-tilt").and_then(|s| s.parse().ok()).unwrap_or(400.0);
    let report = crate::flag_str(args, "--report");
    let force = args.iter().any(|a| a == "--force");
    if !(0.0..=1.0).contains(&max_split) {
        eprintln!("bookprune: --max-split {max_split} is not in [0, 1]");
        return ExitCode::FAILURE;
    }
    if !force && std::path::Path::new(&out_path).exists() {
        eprintln!("bookprune: {out_path} exists -> refusing to overwrite (pass --force)");
        return ExitCode::FAILURE;
    }
    let book_text = match fs::read_to_string(&book_path) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("bookprune: cannot read book {book_path}: {e}");
            return ExitCode::FAILURE;
        }
    };
    let book_lines: Vec<String> = book_text.lines().map(str::trim).filter(|l| !l.is_empty()).map(String::from).collect();
    if book_lines.is_empty() {
        eprintln!("bookprune: book {book_path} is empty");
        return ExitCode::FAILURE;
    }
    let mut paths: Vec<String> = Vec::new();
    {
        let mut i = 0;
        while i < args.len() {
            let a = &args[i];
            if ["--book", "--out", "--min-pairs", "--max-split", "--max-tilt", "--report"].contains(&a.as_str()) {
                i += 2;
            } else if a.starts_with("--") {
                i += 1;
            } else {
                paths.push(a.clone());
                i += 1;
            }
        }
    }
    if paths.is_empty() {
        eprintln!("usage: ti bookprune --book <50k file> --out <pruned> [--min-pairs N] [--max-split F] [--force] <run.games.jsonl> [...]");
        return ExitCode::FAILURE;
    }
    // book index -> (pairs, decisive). Pair ids restart per log, so group per file.
    // book index -> (pairs, decisive, tilt_abs_sum, tilt_n)
    let mut per_line: std::collections::HashMap<u64, (u32, u32, f64, u32)> = std::collections::HashMap::new();
    let mut files = 0usize;
    let mut games = 0usize;
    let mut skipped = 0usize;
    for p in &paths {
        let text = match fs::read_to_string(p) {
            Ok(t) => t,
            Err(e) => {
                eprintln!("bookprune: cannot read {p}: {e}");
                return ExitCode::FAILURE;
            }
        };
        files += 1;
        let mut by_pair: std::collections::HashMap<u64, Vec<Game>> = std::collections::HashMap::new();
        for (n, line) in text.lines().enumerate().filter(|(_, l)| !l.trim().is_empty()) {
            let g = match parse_game(line) {
                Ok(g) => g,
                Err(e) => {
                    eprintln!("bookprune: {p}:{}: {e}", n + 1);
                    return ExitCode::FAILURE;
                }
            };
            games += 1;
            // The log must have used this book: the recorded start must equal the book line.
            let bi = g.book as usize;
            if bi >= book_lines.len() {
                eprintln!("bookprune: {p}:{}: book index {} outside {}-line book (wrong --book?)", n + 1, bi, book_lines.len());
                return ExitCode::FAILURE;
            }
            match Board::from_ataxx_fen(&g.start) {
                Some(b) if b.to_string() == book_lines[bi] => {}
                _ => {
                    eprintln!("bookprune: {p}:{}: start position is not book line {} (wrong --book?)", n + 1, bi);
                    return ExitCode::FAILURE;
                }
            }
            by_pair.entry(g.pair).or_default().push(g);
        }
        for (_, gs) in by_pair {
            if gs.len() != 2 || gs[0].book != gs[1].book {
                skipped += 1;
                continue;
            }
            let (pairs, decisive) = book_decisiveness(&[(gs[0].cand_points, gs[1].cand_points)]);
            let e = per_line.entry(gs[0].book).or_insert((0, 0, 0.0, 0));
            e.0 += pairs;
            e.1 += decisive;
            for g in &gs {
                if let Some(t) = game_tilt(g) {
                    e.2 += t.abs();
                    e.3 += 1;
                }
            }
        }
    }
    // Decide: keep unmeasured lines; drop measured lines that split too often.
    let mut kept: Vec<&String> = Vec::with_capacity(book_lines.len());
    let mut dropped: Vec<u64> = Vec::new();
    let mut hist = [0u64; 10];
    for (i, line) in book_lines.iter().enumerate() {
        let idx = i as u64;
        match per_line.get(&idx) {
            None => kept.push(line),
            Some(&(p, d, tsum, tn)) => {
                let split = 1.0 - d as f64 / p as f64;
                hist[(split * 10.0).clamp(0.0, 9.999) as usize] += 1;
                let tilt = if tn > 0 { tsum / tn as f64 } else { 0.0 };
                let lopsided = max_tilt > 0.0 && tn > 0 && tilt > max_tilt;
                if keep_book_line(p as u64, d, min_pairs, max_split) && !lopsided {
                    kept.push(line);
                } else {
                    dropped.push(idx);
                }
            }
        }
    }
    let mut out = match fs::File::create(&out_path) {
        Ok(f) => std::io::BufWriter::new(f),
        Err(e) => {
            eprintln!("bookprune: cannot create {out_path}: {e}");
            return ExitCode::FAILURE;
        }
    };
    for l in &kept {
        if writeln!(out, "{l}").is_err() {
            eprintln!("bookprune: failed writing {out_path}");
            return ExitCode::FAILURE;
        }
    }
    let meas_pairs: u64 = per_line.values().map(|&(p, ..)| p as u64).sum();
    let meas_dec: u64 = per_line.values().map(|&(_, d, ..)| d as u64).sum();
    let drop_set: std::collections::HashSet<u64> = dropped.iter().copied().collect();
    let mut kept_pairs = 0u64;
    let mut kept_dec = 0u64;
    for (&idx, &(p, d, ..)) in &per_line {
        if !drop_set.contains(&idx) {
            kept_pairs += p as u64;
            kept_dec += d as u64;
        }
    }
    println!("# ti bookprune: {files} logs, {games} games, {} book lines ({} measured, {} dropped, {} kept)", book_lines.len(), per_line.len(), dropped.len(), kept.len());
    println!("# split-rate histogram over measured lines (1-1 share):");
    for (i, &c) in hist.iter().enumerate() {
        println!("#   0.{i}-0.{}: {c} lines", i + 1);
    }
    if meas_pairs > 0 {
        println!("# decisive-pair share: all measured {:.3} -> kept {:.3} ({} / {} pairs)", meas_dec as f64 / meas_pairs as f64, if kept_pairs > 0 { kept_dec as f64 / kept_pairs as f64 } else { 0.0 }, kept_dec, kept_pairs);
    }
    if skipped > 0 {
        println!("# skipped {skipped} incomplete/mismatched pairs");
    }
    if !dropped.is_empty() {
        println!("# dropped book lines (first 40): {:?}", &dropped[..dropped.len().min(40)]);
    }
    if let Some(rp) = report {
        let mut r = match fs::File::create(&rp) {
            Ok(f) => std::io::BufWriter::new(f),
            Err(e) => {
                eprintln!("bookprune: cannot create report {rp}: {e}");
                return ExitCode::FAILURE;
            }
        };
        let _ = writeln!(r, "book_line,pairs,decisive,split_rate,mean_abs_tilt,kept");
        let drop_set: std::collections::HashSet<u64> = dropped.iter().copied().collect();
        let mut idxs: Vec<u64> = per_line.keys().copied().collect();
        idxs.sort_unstable();
        for idx in idxs {
            let &(p, d, tsum, tn) = &per_line[&idx];
            let tilt = if tn > 0 { tsum / tn as f64 } else { 0.0 };
            let _ = writeln!(r, "{idx},{p},{d},{:.4},{:.1},{}", 1.0 - d as f64 / p as f64, tilt, !drop_set.contains(&idx));
        }
        println!("# per-line census -> {rp}");
    }
    println!("# wrote {} lines -> {out_path}", kept.len());
    ExitCode::SUCCESS
}

/// Holdout rows: `board side | score_cp | mate | result_stm | depth | nodes | mover | pair:book:ply`,
/// and/or `--tdat-out <stem>` for non-blocked `.tdat` records (`cp` = mate/score, `outcome` = `stm_pts`).
pub fn run_harvest(args: &[String]) -> ExitCode {
    let mut out_path = None;
    let mut tdat_out = None;
    let mut emin = 0u32;
    let mut emax = 49u32;
    let mut wsum = 1.0f32;
    let mut force = false;
    let mut paths = Vec::new();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--out" => {
                out_path = args.get(i + 1).cloned();
                i += 2;
            }
            "--tdat-out" => {
                tdat_out = args.get(i + 1).cloned();
                i += 2;
            }
            "--empties-min" => {
                emin = args.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(0);
                i += 2;
            }
            "--empties-max" => {
                emax = args.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(49);
                i += 2;
            }
            "--wsum" => {
                wsum = args.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(1.0);
                i += 2;
            }
            "--force" => {
                force = true;
                i += 1;
            }
            _ => {
                paths.push(args[i].clone());
                i += 1;
            }
        }
    }
    if (out_path.is_none() && tdat_out.is_none()) || paths.is_empty() || emin > emax || emax > 49 {
        eprintln!("usage: ti harvest <run.games.jsonl> [...] [--out <holdout.txt>] [--tdat-out <stem>] [--empties-min 0] [--empties-max 49] [--wsum 1.0] [--force]");
        return ExitCode::FAILURE;
    }
    if let Some(p) = &out_path {
        if !force && fs::metadata(p).is_ok() {
            eprintln!("harvest: {p} exists -> refusing to overwrite (pass --force)");
            return ExitCode::FAILURE;
        }
    }
    if let Some(stem) = &tdat_out {
        let p = format!("{stem}.tdat");
        if !force && fs::metadata(&p).is_ok() {
            eprintln!("harvest: {p} exists -> refusing to overwrite (pass --force)");
            return ExitCode::FAILURE;
        }
    }
    let games = match read_games(&paths) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("harvest: {e}");
            return ExitCode::FAILURE;
        }
    };
    let mut txt_f = out_path.as_ref().map(|p| {
        if let Some(parent) = std::path::Path::new(p).parent() {
            if !parent.as_os_str().is_empty() {
                let _ = fs::create_dir_all(parent);
            }
        }
        let mut f = std::io::BufWriter::new(fs::File::create(p).expect("create harvest file"));
        let _ = writeln!(f, "# ti harvest from {} | board side | score_cp | mate | result_stm | depth | nodes | mover | pair:book:ply", paths.join(","));
        f
    });
    let mut tdat_recs: Vec<crate::data::Rec> = Vec::new();
    let (mut rows, mut mates, mut skipped_blocked) = (0u64, 0u64, 0u64);
    for g in &games {
        let before = replay(g).expect("checked above");
        let cand_x = g.cand_seat == "x";
        // candidate points -> x points
        let x_pts = if cand_x { g.cand_points } else { 1.0 - g.cand_points };
        for (r, b) in g.moves.iter().zip(before.iter()) {
            if r.mover == "-" || r.mv == "FORFEIT" || (r.score.is_none() && r.mate.is_none()) {
                continue;
            }
            let ne = b.empty().count_ones();
            if ne < emin || ne > emax {
                continue;
            }
            let stm_pts = if b.turn == 0 { x_pts } else { 1.0 - x_pts };
            if let Some(f) = txt_f.as_mut() {
                let _ = writeln!(
                    f,
                    "{} | {} | {} | {} | {} | {} | {} | {}:{}:{}",
                    b.to_string(),
                    r.score.map(|s| s.to_string()).unwrap_or_default(),
                    r.mate.map(|m| format!("M{m:+}")).unwrap_or_default(),
                    stm_pts,
                    r.depth.map(|d| d.to_string()).unwrap_or_default(),
                    r.nodes.map(|n| n.to_string()).unwrap_or_default(),
                    r.mover,
                    g.pair,
                    g.book,
                    r.ply
                );
            }
            if tdat_out.is_some() {
                if b.blockers != 0 {
                    skipped_blocked += 1;
                } else {
                    let cp = if let Some(m) = r.mate {
                        let ply = (m.abs() * 2 - 1).clamp(1, 1999) as f32;
                        if m > 0 { 100_000.0 - ply } else { -100_000.0 + ply }
                    } else {
                        r.score.unwrap() as f32
                    };
                    tdat_recs.push(crate::data::Rec {
                        occ0: b.occ[0],
                        occ1: b.occ[1],
                        outcome: stm_pts as f32,
                        cp,
                        wsum,
                        p3: 0.0,
                        cp5k: cp,
                        zx: 0,
                        stm: b.turn,
                        teacher_kind: 4,
                        proof_kind: if r.mate.is_some() { 4 } else { 0 },
                        relabel_src: 4,
                        ti_clash: 0,
                    });
                }
            }
            rows += 1;
            mates += r.mate.is_some() as u64;
        }
    }
    if let Some(mut f) = txt_f {
        let _ = f.flush();
    }
    if let Some(stem) = &tdat_out {
        let src = format!("harvest({}) [empties {emin}..={emax}]", paths.join(","));
        if let Err(e) = crate::data::write_tdat_and_manifest(stem, &src, &tdat_recs) {
            eprintln!("harvest: tdat write: {e}");
            return ExitCode::FAILURE;
        }
        println!("harvest: {} games -> {} tdat rows ({skipped_blocked} blocked skipped) -> {stem}.tdat", games.len(), tdat_recs.len());
    }
    if let Some(out_path) = out_path {
        println!("harvest: {} games -> {rows} rows ({mates} proven mates) -> {out_path} (HOLDOUT, not merged)", games.len());
    }
    ExitCode::SUCCESS
}

/// Pool multiple independent gate `.games.jsonl` files into exact paired Pentanomial Elo / CI95 / LLR / LOS.
pub fn run_pool(args: &[String]) -> ExitCode {
    let elo0 = crate::flag_str(args, "--elo0").and_then(|s| s.parse::<f64>().ok()).unwrap_or(0.0);
    let elo1 = crate::flag_str(args, "--elo1").and_then(|s| s.parse::<f64>().ok()).unwrap_or(10.0);
    let mut paths = Vec::new();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--elo0" | "--elo1" => i += 2,
            s if s.starts_with("--") => i += 1,
            p => {
                paths.push(p.to_string());
                i += 1;
            }
        }
    }
    if paths.is_empty() {
        eprintln!("usage: ti pool <run1.games.jsonl> <run2.games.jsonl> [...] [--elo0 0] [--elo1 10]");
        return ExitCode::FAILURE;
    }
    let mut tally = crate::sprt::SprtTally::default();
    let mut files_used = 0usize;
    let (mut cn, mut cm, mut bn, mut bm) = (0u64, 0u64, 0u64, 0u64);
    for p in &paths {
        let games = match read_games(std::slice::from_ref(p)) {
            Ok(g) => g,
            Err(e) => {
                eprintln!("pool: {p}: {e}");
                return ExitCode::FAILURE;
            }
        };
        files_used += 1;
        let mut by_pair: std::collections::BTreeMap<u64, Vec<f64>> = std::collections::BTreeMap::new();
        for g in &games {
            by_pair.entry(g.pair).or_default().push(g.cand_points);
            for r in &g.moves {
                if let (Some(n), Some(m)) = (r.nodes, r.ms) {
                    if r.mover == "candidate" {
                        cn += n;
                        cm += m;
                    } else if r.mover == "baseline" {
                        bn += n;
                        bm += m;
                    }
                }
            }
        }
        for pts in by_pair.values().filter(|v| v.len() == 2) {
            tally.record_pair((pts[0] * 2.0).round() as u8, (pts[1] * 2.0).round() as u8);
        }
    }
    let st = tally.compute(elo0, elo1, 0.05, 0.05);
    let c_nps = if cm > 0 { cn * 1000 / cm } else { 0 };
    let b_nps = if bm > 0 { bn * 1000 / bm } else { 0 };
    let pct = if b_nps > 0 {
        format!("{:+.1}%", 100.0 * (c_nps as f64 - b_nps as f64) / b_nps as f64)
    } else {
        "n/a".into()
    };
    let verdict = match st.verdict {
        crate::sprt::SprtVerdict::Pass => "PASS",
        crate::sprt::SprtVerdict::Fail => "H0",
        crate::sprt::SprtVerdict::Continue => "Continue",
    };
    println!(
         "=== Pooled Pentanomial SPRT ({files_used} files) ===\n\
         Games : {} (+{} -{} ={}, {:.1}%)\n\
         Penta : LL={} LD={} DD+WL={} WD={} WW={}\n\
         Elo   : {:+.1} +- {:.1} (95% CI), LOS {:.2}%\n\
         LLR   : {:+.2} [{:.2}, {:.2}] (elo0={elo0:.1}, elo1={elo1:.1}) -> {verdict}\n\
         Split : {:.0}% of pairs split 1-1 by seat\n\
         Speed : {pct} ({c_nps} vs {b_nps} nps)",
        st.games,
        tally.wins,
        tally.losses,
        tally.draws,
        100.0 * st.score_rate,
        tally.penta[0],
        tally.penta[1],
        tally.penta[2],
        tally.penta[3],
        tally.penta[4],
        st.elo,
        st.elo_ci95,
        100.0 * st.los,
        st.llr,
        st.lower_bound,
        st.upper_bound,
        100.0 * st.seat_split,
    );
    println!("{}", tally.report_block(elo0, elo1, 0.05, 0.05));
    ExitCode::SUCCESS
}

#[cfg(test)]
mod tests {
    use super::*;

    fn line(moves: &str) -> String {
        format!("{{\"pair\":3,\"book\":7,\"start\":\"x5o/7/7/7/7/7/o5x x 0 1\",\"candidate_seat\":\"x\",\"candidate_points\":0,\"final_x\":10,\"final_o\":39,\"moves\":[{moves}]}}")
    }

    fn mv(ply: u32, side: &str, mover: &str, m: &str, score: &str, mate: &str) -> String {
        format!("{{\"ply\":{ply},\"side\":\"{side}\",\"mover\":\"{mover}\",\"move\":\"{m}\",\"score\":{score},\"mate\":{mate},\"depth\":5,\"nodes\":2000,\"ms\":3,\"pv\":\"{m}\"}}")
    }

    #[test]
    fn parse_replay_and_explain() {
        let l = line(&[
            mv(0, "x", "candidate", "b6", "30", "null"),
            mv(1, "o", "baseline", "f6", "-20", "null"),
            mv(2, "x", "candidate", "c5", "-400", "null"),
            mv(3, "o", "baseline", "e5", "null", "3"),
        ].join(","));
        let g = parse_game(&l).unwrap();
        assert_eq!((g.pair, g.book, g.moves.len()), (3, 7, 4));
        assert_eq!(g.moves[3].mate, Some(3));
        assert_eq!(replay(&g).unwrap().len(), 4);
        let e = explain(&g);
        assert_eq!(e.turning_point, Some((2, 30, -400)));
        assert_eq!(e.loser_saw_it, Some(2));
    }

    #[test]
    fn gen_log_label_is_the_engines_own_choice() {
        // ti gen (D790): `move` is what was played, `engine_best` / `engine_best_score` the engine's own first choice
        let line = r#"{"pair":0,"book":0,"start":"x5o/7/7/7/7/7/o5x x 0 1","candidate_seat":"x","candidate_points":1,"final_x":30,"final_o":19,"moves":[{"ply":0,"side":"x","mover":"candidate","move":"g1f2","engine_best":"a7b6","engine_best_score":41,"score":17,"mate":null,"depth":8,"nodes":5000,"ms":30}]}"#;
        let g = parse_game(line).unwrap();
        let r = &g.moves[0];
        assert_eq!(r.mv, "g1f2");
        assert_eq!(r.label_mv(), "a7b6");
        assert_eq!(r.score, Some(41));
        // a log without the fields (gate games, older ti gen runs) keeps the old meaning
        let old = line.replace(r#""engine_best":"a7b6","engine_best_score":41,"#, "");
        let r = &parse_game(&old).unwrap().moves[0];
        assert_eq!(r.label_mv(), "g1f2");
        assert_eq!(r.score, Some(17));
    }

    #[test]
    fn illegal_move_is_refused() {
        let g = parse_game(&line(&mv(0, "x", "candidate", "d4", "0", "null"))).unwrap();
        assert!(replay(&g).unwrap_err().contains("illegal move d4"));
    }

    #[test]
    fn book_decisiveness_counts_non_split_pairs() {
        assert_eq!(book_decisiveness(&[(1.0, 1.0), (0.0, 0.0)]), (2, 2));
        assert_eq!(book_decisiveness(&[(1.0, 0.0), (0.0, 1.0)]), (2, 0));
        assert_eq!(book_decisiveness(&[(1.0, 0.5), (0.5, 0.5)]), (2, 1));
        assert_eq!(book_decisiveness(&[]), (0, 0));
    }

    #[test]
    fn book_drop_rule_keeps_unmeasured_and_decisive() {
        assert!(keep_book_line(0, 0, 4, 0.75));
        assert!(keep_book_line(3, 0, 4, 0.75));
        assert!(keep_book_line(4, 4, 4, 0.75));
        assert!(keep_book_line(4, 1, 4, 0.75));
        assert!(!keep_book_line(4, 0, 4, 0.75));
        assert!(keep_book_line(8, 2, 4, 0.75));
        assert!(!keep_book_line(8, 1, 4, 0.75));
    }

    #[test]
    fn parses_iters_pv_matching_and_telemetry() {
        let mut g1_moves = Vec::new();
        let mut g2_moves = Vec::new();
        // Build 25 paired games (>= 20 threshold for PV & iter blocks) with realistic PV + "it" fields.
        for p in 0..32u32 {
            let ply = p * 2;
            let mv_x = if p % 2 == 0 { "b6" } else { "b6" };
            let mv_o = if p % 2 == 0 { "f6" } else { "f6" };
            let pv_x = if p % 3 == 0 { "b6 e6 c5" } else { "b6 f6 c5" };
            let pv_o = "f6 b6";
            let sc_x = if p % 3 == 0 { 30 } else { -120 };
            g1_moves.push(format!(
                "{{\"ply\":{ply},\"side\":\"x\",\"mover\":\"candidate\",\"move\":\"{mv_x}\",\"score\":{sc_x},\"mate\":null,\"depth\":5,\"nodes\":2000,\"ms\":3,\"pv\":\"{pv_x}\",\"it\":\"2:10:120:0:0:b6 f6;3:25:400:1:1:{mv_x} f6;4:{sc_x}:1000:2:0:{mv_x} f6;5:{sc_x}:2000:3:0:{pv_x}\"}}"
            ));
            g1_moves.push(format!(
                "{{\"ply\":{},\"side\":\"o\",\"mover\":\"baseline\",\"move\":\"{mv_o}\",\"score\":20,\"mate\":null,\"depth\":5,\"nodes\":2500,\"ms\":4,\"pv\":\"{pv_o}\",\"it\":\"2:5:150:0:0:f6 b6;3:15:550:1:0:f6 b6;4:20:1300:2:0:f6 b6;5:20:2500:4:0:{pv_o}\"}}",
                ply + 1
            ));
            g2_moves.push(format!(
                "{{\"ply\":{ply},\"side\":\"x\",\"mover\":\"baseline\",\"move\":\"{mv_x}\",\"score\":10,\"mate\":null,\"depth\":5,\"nodes\":2500,\"ms\":4,\"pv\":\"{pv_x}\",\"it\":\"2:5:150:0:0:b6 f6;3:15:550:1:0:b6 f6;4:10:1300:2:0:b6 f6;5:10:2500:4:0:{pv_x}\"}}"
            ));
            g2_moves.push(format!(
                "{{\"ply\":{},\"side\":\"o\",\"mover\":\"candidate\",\"move\":\"{mv_o}\",\"score\":-10,\"mate\":null,\"depth\":5,\"nodes\":2000,\"ms\":3,\"pv\":\"{pv_o}\",\"it\":\"2:0:120:0:0:f6 b6;3:-5:400:1:0:f6 b6;4:-10:1000:2:0:f6 b6;5:-10:2000:3:0:{pv_o}\"}}",
                ply + 1
            ));
        }
        let mut games = Vec::new();
        for pair in 0..55u64 {
            let l1 = format!(
                "{{\"pair\":{pair},\"book\":{pair},\"start\":\"x5o/7/7/7/7/7/o5x x 0 1\",\"candidate_seat\":\"x\",\"candidate_points\":1,\"final_x\":30,\"final_o\":19,\"moves\":[{}]}}",
                g1_moves.join(",")
            );
            let l2 = format!(
                "{{\"pair\":{pair},\"book\":{pair},\"start\":\"x5o/7/7/7/7/7/o5x x 0 1\",\"candidate_seat\":\"o\",\"candidate_points\":0,\"final_x\":30,\"final_o\":19,\"moves\":[{}]}}",
                g2_moves.join(",")
            );
            games.push(parse_game(&l1).unwrap());
            games.push(parse_game(&l2).unwrap());
        }
        assert!(pv_mv_matches("a7b6", "b6"));
        assert!(!pv_mv_matches("a7c5", "b6"));
        assert_eq!(games[0].moves[0].iters.len(), 4);
        assert_eq!(games[0].moves[0].iters[1].rfh, 1);

        let e = explain(&games[0]);
        assert!(e.off_pv_refutation.is_some());

        let rep = insight(&games).expect("insight report");
        assert!(rep.contains("calibrated WDL & variance analysis"));
        assert!(rep.contains("PV dictation & adversarial surprise"));
        assert!(rep.contains("per-depth search telemetry & EBF"));
        assert!(rep.contains("d=3  geoN cand"));
    }
}

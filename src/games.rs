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

/// One move as stored in the JSONL.
#[derive(Debug, Clone, PartialEq)]
pub struct Rec {
    pub ply: u32,
    pub mover: String,
    pub mv: String,
    pub score: Option<i32>,
    pub mate: Option<i32>,
    pub depth: Option<u32>,
    pub nodes: Option<u64>,
    pub ms: Option<u64>,
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
            score: num(obj, "score"),
            mate: num(obj, "mate"),
            depth: num(obj, "depth"),
            nodes: num(obj, "nodes"),
            ms: num(obj, "ms"),
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
    Explained { turning_point: tp, loser_saw_it: saw, proven_lost: proven, disagreement: dis }
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
    Some(out)
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
fn insight_nnue_buckets(games: &[Game]) -> String {
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
    if full.len() < 50 {
        return String::new();
    }
    let mut out = String::from("\n  by NNUE bucket (empties*16/50; per-move diff = baseline minus candidate value lost, + = candidate better):\n");
    for b in (0..NB).rev() {
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
        if diff.len() < 10 {
            continue;
        }
        let (md, zd) = mean_z(&diff);
        let (lo, hi) = ((b * 50 + 15) / 16, ((b + 1) * 50 + 15) / 16 - 1);
        out += &format!("    b{b:<2} empties {lo:2}-{hi:<2} value lost diff {md:+7.2} cp/move (z {zd:+5.2}) | {} pairs, {moves} moves\n", diff.len());
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
    println!("pair book seat result | turning point (ply: loser's own eval before -> after) | loser saw it (<= -200) | proven lost (loser reports mate) | biggest disagreement (ply: mover vs next mover, mover's view)");
    for g in &games {
        let e = explain(g);
        let res = if g.cand_points > 0.5 { "W" } else if g.cand_points < 0.5 { "L" } else { "D" };
        let tp = e.turning_point.map(|(p, a, b)| format!("{p}: {a:+} -> {b:+}")).unwrap_or_else(|| "-".into());
        let saw = e.loser_saw_it.map(|p| p.to_string()).unwrap_or_else(|| "-".into());
        let dis = e.disagreement.map(|(p, a, b)| format!("{p}: {a:+} vs {b:+}")).unwrap_or_else(|| "-".into());
        let proven = e.proven_lost.map(|p| p.to_string()).unwrap_or_else(|| "-".into());
        println!("{:>4} {:>5} {} {} | {tp} | {saw} | {proven} | {dis}", g.pair, g.book, g.cand_seat, res);
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
}

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
        Some(r) => println!("{r}"),
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

/// Holdout rows: `board side | score_cp | mate | result_stm | depth | nodes | mover | pair:book:ply`.
/// result_stm = 1 / 0.5 / 0 for the side to move of that position. score is empty when a mate is proven
/// (then `mate` = moves to mate, + = side to move mates): exact truth, not an approximation.
pub fn run_harvest(args: &[String]) -> ExitCode {
    let mut out_path = None;
    let mut paths = Vec::new();
    let mut i = 0;
    while i < args.len() {
        if args[i] == "--out" {
            out_path = args.get(i + 1).cloned();
            i += 2;
        } else {
            paths.push(args[i].clone());
            i += 1;
        }
    }
    let Some(out_path) = out_path else {
        eprintln!("usage: ti harvest <run.games.jsonl> [...] --out <holdout.txt>  (holdout only; merging is a separate decision)");
        return ExitCode::FAILURE;
    };
    if fs::metadata(&out_path).is_ok() {
        eprintln!("harvest: {out_path} exists -> refusing to overwrite");
        return ExitCode::FAILURE;
    }
    let games = match read_games(&paths) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("harvest: {e}");
            return ExitCode::FAILURE;
        }
    };
    let mut f = std::io::BufWriter::new(fs::File::create(&out_path).expect("create harvest file"));
    let _ = writeln!(f, "# ti harvest from {} | board side | score_cp | mate | result_stm | depth | nodes | mover | pair:book:ply", paths.join(","));
    let (mut rows, mut mates) = (0u64, 0u64);
    for g in &games {
        let before = replay(g).expect("checked above");
        let cand_x = g.cand_seat == "x";
        // candidate points -> x points
        let x_pts = if cand_x { g.cand_points } else { 1.0 - g.cand_points };
        for (r, b) in g.moves.iter().zip(before.iter()) {
            if r.mover == "-" || r.mv == "FORFEIT" || (r.score.is_none() && r.mate.is_none()) {
                continue;
            }
            let stm_pts = if b.turn == 0 { x_pts } else { 1.0 - x_pts };
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
            rows += 1;
            mates += r.mate.is_some() as u64;
        }
    }
    println!("harvest: {} games -> {rows} rows ({mates} proven mates) -> {out_path} (HOLDOUT, not merged)", games.len());
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
}

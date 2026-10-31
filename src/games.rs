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

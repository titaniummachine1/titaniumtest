//! `ti botzone`: Botzone match logs (`/match/<id>?lite=true` JSON) -> a deduplicated position file for the 5k relabel.
//!
//! Only POSITIONS are taken from the games (any player's moves, any bot). Opponent engine scores are never read: the
//! label comes from `ti relabel` (Titanium, fixed nodes, current main net), so Titanium never learns another engine's
//! search values as short-search labels (user 2026-11-01). The game result is kept as a side column (win/loss/draw
//! counts and the final stone margin, from the side to move), the Stockfish-style anchor of D132.
//!
//! Positions are canonicalised under the 8 dihedral symmetries (colours are never swapped, the side to move is part of
//! the key), so a position reached in several games or orientations is one row with accumulated W/L/D.
//!
//!   ti botzone <match.json|dir|URL> [...] --out DIR [--bots sotax,daybreak,hce] [--min-ply 3] [--no-stats]
//!
//! A Botzone match LINK (`https://www.botzone.org.cn/match/<id>`) is fetched with curl.exe (`?lite=true`) and analysed;
//! nothing has to be downloaded by hand. Each match prints both engines' per-move report (depth, nodes, engine ms,
//! eval, Botzone ms, verdict) and the summary statistics (`--no-stats` silences it).
//!
//! Writes DIR/positions.bin (25 B per row: occ0 u64, occ1 u64, blockers u64, turn u8; the `ti relabel --bin` format)
//! and DIR/meta.tsv (row, n, w, l, d, avg_margin, first_ply, movers, matches). Botzone rules: player 0 = black =
//! turn 0 starts on (0,0),(6,6); square index = y*7 + x; no passes; the log's final scores are stone counts.

use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode};

fn flag(args: &[String], name: &str) -> Option<String> {
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).cloned()
}

/// Image of square (x, y) under symmetry k of the dihedral group D4.
fn d4(k: usize, x: i32, y: i32) -> (i32, i32) {
    match k {
        0 => (x, y),
        1 => (6 - x, y),
        2 => (x, 6 - y),
        3 => (6 - x, 6 - y),
        4 => (y, x),
        5 => (6 - y, x),
        6 => (y, 6 - x),
        _ => (6 - y, 6 - x),
    }
}

fn map_bb(k: usize, mut bb: u64) -> u64 {
    let mut out = 0u64;
    while bb != 0 {
        let s = bb.trailing_zeros() as i32;
        bb &= bb - 1;
        let (x, y) = d4(k, s % 7, s / 7);
        out |= 1u64 << (y * 7 + x);
    }
    out
}

/// Lexicographically smallest image of (occ0, occ1, blockers) over the 8 symmetries.
fn canon(o0: u64, o1: u64, bl: u64) -> (u64, u64, u64) {
    (0..8).map(|k| (map_bb(k, o0), map_bb(k, o1), map_bb(k, bl))).min().unwrap()
}

struct Pos {
    ply: u32,
    occ0: u64,
    occ1: u64,
    turn: u8,
    mover: String,
}

struct Match {
    id: String,
    names: [String; 2],
    scores: [i64; 2],
    positions: Vec<Pos>,
}

fn grid_bbs(grid: &[[u8; 7]; 7]) -> (u64, u64) {
    let (mut a, mut b) = (0u64, 0u64);
    for x in 0..7 {
        for y in 0..7 {
            match grid[x][y] {
                1 => a |= 1u64 << (y * 7 + x),
                2 => b |= 1u64 << (y * 7 + x),
                _ => {}
            }
        }
    }
    (a, b)
}

fn read_match(path: &Path) -> Result<Match, String> {
    let id = path.file_stem().and_then(|s| s.to_str()).unwrap_or("?").to_string();
    let text = fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let v: Value = serde_json::from_str(&text).map_err(|e| format!("{}: bad json: {e}", path.display()))?;
    let name = |i: usize| v["players"][i]["name"].as_str().unwrap_or("?").to_string();
    let logs = v["logs"].as_array().ok_or_else(|| format!("{id}: no logs"))?;
    let fin = &logs.last().ok_or_else(|| format!("{id}: empty logs"))?["output"]["content"];
    let (Some(s0), Some(s1)) = (fin["0"].as_i64(), fin["1"].as_i64()) else {
        return Err(format!("{id}: no final scores (unfinished match?)"));
    };
    let names = [name(0), name(1)];

    let mut grid = [[0u8; 7]; 7];
    grid[0][0] = 1;
    grid[6][6] = 1;
    grid[6][0] = 2;
    grid[0][6] = 2;
    let mut positions = Vec::new();
    let mut ply = 0u32;
    for e in logs {
        let key = if e.get("0").is_some() { "0" } else if e.get("1").is_some() { "1" } else { continue };
        let turn: u8 = if key == "0" { 0 } else { 1 };
        let side = &e[key];
        let r = &side["response"];
        if side["verdict"].as_str() != Some("OK") || !r.is_object() {
            break;
        }
        let g = |k: &str| r[k].as_i64().unwrap_or(-1);
        let (x0, y0, x1, y1) = (g("x0"), g("y0"), g("x1"), g("y1"));
        if x0 < 0 || x1 < 0 {
            break;
        }
        if turn != (ply % 2) as u8 {
            return Err(format!("{id}: side to move {turn} at ply {} breaks alternation", ply + 1));
        }
        if [x0, y0, x1, y1].iter().any(|&c| c > 6) {
            return Err(format!("{id}: move off the board at ply {}", ply + 1));
        }
        let (x0, y0, x1, y1) = (x0 as usize, y0 as usize, x1 as usize, y1 as usize);
        let color = turn + 1;
        let dist = (x1 as i32 - x0 as i32).abs().max((y1 as i32 - y0 as i32).abs());
        if grid[x0][y0] != color || grid[x1][y1] != 0 || !(1..=2).contains(&dist) {
            return Err(format!("{id}: illegal move {x0}{y0}->{x1}{y1} at ply {}", ply + 1));
        }
        ply += 1;
        let (o0, o1) = grid_bbs(&grid);
        positions.push(Pos { ply, occ0: o0, occ1: o1, turn, mover: names[turn as usize].clone() });
        if dist == 2 {
            grid[x0][y0] = 0;
        }
        grid[x1][y1] = color;
        for dx in -1i32..=1 {
            for dy in -1i32..=1 {
                let (nx, ny) = (x1 as i32 + dx, y1 as i32 + dy);
                if (0..7).contains(&nx) && (0..7).contains(&ny) && grid[nx as usize][ny as usize] == 3 - color {
                    grid[nx as usize][ny as usize] = color;
                }
            }
        }
    }
    Ok(Match { id, names, scores: [s0, s1], positions })
}

#[derive(Default)]
struct Agg {
    w: u32,
    l: u32,
    d: u32,
    margin: i64,
    first_ply: u32,
    movers: BTreeSet<String>,
    matches: BTreeSet<String>,
}

fn is_url(s: &str) -> bool {
    s.starts_with("http://") || s.starts_with("https://")
}

fn url_match_id(url: &str) -> String {
    let no_q = url.split('?').next().unwrap_or(url);
    no_q.trim_end_matches('/').rsplit('/').next().filter(|s| !s.is_empty()).unwrap_or("match").to_string()
}

/// Download a Botzone match page (`?lite=true` JSON) with curl.exe into the temp cache; returns the local file.
fn fetch_url(url: &str) -> Result<PathBuf, String> {
    let id = url_match_id(url);
    let dir = std::env::temp_dir().join("ti_botzone");
    fs::create_dir_all(&dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    let path = dir.join(format!("{id}.json"));
    let sep = if url.contains('?') { '&' } else { '?' };
    let full = format!("{url}{sep}lite=true");
    let out = Command::new("curl.exe")
        .args(["-s", "-H", "X-Requested-With: XMLHttpRequest", "-o"])
        .arg(&path)
        .arg(&full)
        .output()
        .map_err(|e| format!("curl.exe not runnable: {e}"))?;
    if !out.status.success() {
        return Err(format!("{id}: curl failed ({}): {}", out.status, String::from_utf8_lossy(&out.stderr).trim()));
    }
    if fs::metadata(&path).map(|m| m.len()).unwrap_or(0) == 0 {
        return Err(format!("{id}: empty download from {url}"));
    }
    Ok(path)
}

fn collect_files(inputs: &[String]) -> Result<Vec<PathBuf>, String> {
    let mut files = Vec::new();
    for i in inputs {
        if is_url(i) {
            files.push(fetch_url(i)?);
            continue;
        }
        let p = Path::new(i);
        if p.is_dir() {
            let mut v: Vec<_> = fs::read_dir(p)
                .map(|r| r.filter_map(|e| e.ok()).map(|e| e.path()).filter(|p| p.extension().map_or(false, |x| x == "json")).collect())
                .unwrap_or_default();
            v.sort();
            files.extend(v);
        } else {
            files.push(p.to_path_buf());
        }
    }
    Ok(files)
}

/// One side's debug line: Titanium prints `d18 n73629 t3600 s99993 <net>/<bz>` (cp), other bots print
/// `nodes=... depth=... winrate=... scoring=... time=...` (winrate / scoring units).
#[derive(Default, Clone)]
struct MoveStat {
    ply: u32,
    side: usize,
    depth: Option<i64>,
    nodes: Option<i64>,
    eng_ms: Option<i64>,
    eval: Option<f64>,
    eval_kind: &'static str,
    bz_ms: Option<i64>,
    verdict: String,
    mv: String,
}

fn parse_debug(dbg: &str) -> (Option<i64>, Option<i64>, Option<i64>, Option<f64>, &'static str) {
    // D546: PyJataxx prints JSON: h = value in [-1, 1] (|h| > 1 = solved), maxLevel = tree depth, expansions = nodes
    if dbg.trim_start().starts_with('{') {
        if let Ok(j) = serde_json::from_str::<Value>(dbg) {
            return (j["maxLevel"].as_i64(), j["expansions"].as_i64(), None, j["h"].as_f64(), "h");
        }
    }
    let (mut depth, mut nodes, mut ms, mut eval) = (None, None, None, None);
    let mut kind = "";
    // seconds when the literal has a decimal point (`time=3.60`), milliseconds otherwise (UCI `time 209`)
    fn ms_of(v: &str) -> Option<i64> {
        if v.contains('.') {
            v.parse::<f64>().ok().map(|x| (x * 1000.0).round() as i64)
        } else {
            v.parse::<i64>().ok()
        }
    }
    let toks: Vec<&str> = dbg.split_whitespace().collect();
    let mut i = 0;
    while i < toks.len() {
        let t = toks[i];
        if let Some((k, v)) = t.split_once('=') {
            match k {
                "depth" => depth = v.parse().ok(),
                "nodes" => nodes = v.parse().ok(),
                "time" => ms = ms_of(v),
                "winrate" => {
                    eval = v.parse().ok();
                    kind = "wr";
                }
                "scoring" => {
                    if eval.is_none() {
                        eval = v.parse().ok();
                        kind = "sc";
                    }
                }
                _ => {}
            }
            i += 1;
            continue;
        }
        match t {
            "depth" => depth = toks.get(i + 1).and_then(|v| v.parse().ok()),
            "nodes" => nodes = toks.get(i + 1).and_then(|v| v.parse().ok()),
            "time" => ms = toks.get(i + 1).and_then(|v| ms_of(v)),
            // D938 cheetax: "depth 20 seldepth 30nodes 8908800 evaluation -41 eval score 56". `evaluation` is its SEARCH
            // score from the mover's view (-19952 = proven loss); `eval score` is another number and must not override it.
            "evaluation" => {
                if let Some(x) = toks.get(i + 1).and_then(|x| x.parse::<f64>().ok()) {
                    eval = Some(x);
                    kind = "ev";
                }
            }
            _ if t.ends_with("nodes") && t.len() > 5 && t[..t.len() - 5].chars().all(|c| c.is_ascii_digit()) => {
                nodes = toks.get(i + 1).and_then(|v| v.parse().ok());
            }
            "score" if kind == "ev" => {}
            "score" => {
                let (kind_tok, val) = match toks.get(i + 1).copied() {
                    Some(v @ ("cp" | "mate")) => (Some(v), toks.get(i + 2).copied()),
                    other => (None, other),
                };
                if let Some(x) = val.and_then(|x| x.parse::<f64>().ok()) {
                    eval = Some(x);
                    kind = if kind_tok == Some("mate") { "mate" } else { "cp" };
                }
            }
            _ => {
                // compact form: d15 n1988957 t9297 s62. Only assign when the remainder parses, so words that merely
                // start with d/n/t/s (seldepth, nps, ...) never erase an already parsed field.
                let mut it = t.chars();
                let (head, rest) = (it.next(), it.as_str());
                match head {
                    Some('d') => {
                        if let Ok(x) = rest.parse() {
                            depth = Some(x);
                        }
                    }
                    Some('n') => {
                        if let Ok(x) = rest.parse() {
                            nodes = Some(x);
                        }
                    }
                    Some('t') => {
                        if let Ok(x) = rest.parse() {
                            ms = Some(x);
                        }
                    }
                    Some('s') => {
                        if let Ok(x) = rest.parse::<f64>() {
                            eval = Some(x);
                            kind = "cp";
                        }
                    }
                    _ => {}
                }
            }
        }
        i += 1;
    }
    (depth, nodes, ms, eval, kind)
}

fn move_stats(m: &Match, path: &Path) -> Vec<MoveStat> {
    let Ok(text) = fs::read_to_string(path) else { return Vec::new() };
    let Ok(v) = serde_json::from_str::<Value>(&text) else { return Vec::new() };
    let Some(logs) = v["logs"].as_array() else { return Vec::new() };
    let mut out = Vec::new();
    let mut ply = 0u32;
    for e in logs {
        for side in 0..2usize {
            let key = side.to_string();
            let Some(s) = e.get(&key).filter(|s| s.is_object()) else { continue };
            if s.get("verdict").is_none() {
                continue;
            }
            ply += 1;
            let (depth, nodes, eng_ms, eval, eval_kind) = parse_debug(s["debug"].as_str().unwrap_or(""));
            let r = &s["response"];
            let mv = if r.is_object() {
                format!("{}{}->{}{}", r["x0"], r["y0"], r["x1"], r["y1"])
            } else {
                "-".to_string()
            };
            out.push(MoveStat {
                ply,
                side,
                depth,
                nodes,
                eng_ms,
                eval,
                eval_kind,
                bz_ms: s["time"].as_i64(),
                verdict: s["verdict"].as_str().unwrap_or("?").to_string(),
                mv,
            });
        }
    }
    let _ = m;
    out
}

fn median_i64(v: &[i64]) -> Option<i64> {
    if v.is_empty() {
        return None;
    }
    let mut s = v.to_vec();
    s.sort_unstable();
    Some(s[s.len() / 2])
}

fn median_f64(v: &[f64]) -> Option<f64> {
    if v.is_empty() {
        return None;
    }
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    Some(s[s.len() / 2])
}

fn print_stats(m: &Match, path: &Path) {
    let stats = move_stats(m, path);
    println!("\n== {} ==", m.id);
    println!("players: {} vs {}", m.names[0], m.names[1]);
    println!("final score (games won): {} - {}", m.scores[0], m.scores[1]);
    if stats.is_empty() {
        println!("(no per-move debug lines)");
        return;
    }
    println!("ply side depth nodes eng_ms eval bz_ms verdict move");
    for s in &stats {
        println!(
            "{:>3} {} {:>5} {:>9} {:>6} {:>7} {:>5} {:>7} {}",
            s.ply,
            s.side,
            s.depth.map(|x| x.to_string()).unwrap_or_else(|| "-".into()),
            s.nodes.map(|x| x.to_string()).unwrap_or_else(|| "-".into()),
            s.eng_ms.map(|x| x.to_string()).unwrap_or_else(|| "-".into()),
            s.eval
                .map(|x| if s.eval_kind == "cp" { format!("{x:.0}") } else { format!("{x:.3}") })
                .unwrap_or_else(|| "-".into()),
            s.bz_ms.map(|x| x.to_string()).unwrap_or_else(|| "-".into()),
            s.verdict,
            s.mv
        );
    }
    for side in 0..2usize {
        let rows: Vec<&MoveStat> = stats.iter().filter(|s| s.side == side).collect();
        if rows.is_empty() {
            continue;
        }
        let depth = median_i64(&rows.iter().filter_map(|s| s.depth).collect::<Vec<_>>());
        let nodes = median_i64(&rows.iter().filter_map(|s| s.nodes).collect::<Vec<_>>());
        let eng = median_i64(&rows.iter().filter_map(|s| s.eng_ms).collect::<Vec<_>>());
        let bz = median_i64(&rows.iter().filter_map(|s| s.bz_ms).collect::<Vec<_>>());
        let bz_max = rows.iter().filter_map(|s| s.bz_ms).max();
        let evals: Vec<f64> = rows.iter().filter_map(|s| s.eval).collect();
        let kind = rows.iter().find(|s| s.eval_kind != "").map(|s| s.eval_kind).unwrap_or("?");
        let nps: Vec<f64> = rows
            .iter()
            .filter_map(|s| match (s.nodes, s.eng_ms) {
                (Some(n), Some(t)) if t > 0 => Some(n as f64 * 1000.0 / t as f64),
                _ => None,
            })
            .collect();
        println!(
            "{}: {} moves | median depth {} | median nodes {} | median nps {} | median engine ms {} | median botzone ms {} | max botzone ms {} | eval[{}] median {} range [{}, {}]",
            m.names[side],
            rows.len(),
            depth.map(|x| x.to_string()).unwrap_or_else(|| "-".into()),
            nodes.map(|x| x.to_string()).unwrap_or_else(|| "-".into()),
            median_f64(&nps).map(|x| format!("{x:.0}")).unwrap_or_else(|| "-".into()),
            eng.map(|x| x.to_string()).unwrap_or_else(|| "-".into()),
            bz.map(|x| x.to_string()).unwrap_or_else(|| "-".into()),
            bz_max.map(|x| x.to_string()).unwrap_or_else(|| "-".into()),
            kind,
            median_f64(&evals).map(|x| format!("{x:.3}")).unwrap_or_else(|| "-".into()),
            if evals.is_empty() { "-".to_string() } else { format!("{}", evals.iter().cloned().fold(f64::INFINITY, f64::min)) },
            if evals.is_empty() { "-".to_string() } else { format!("{}", evals.iter().cloned().fold(f64::NEG_INFINITY, f64::max)) }
        );
    }
    let verdicts: BTreeSet<&str> = stats.iter().map(|s| s.verdict.as_str()).collect();
    println!("verdicts: {:?}", verdicts);
}

/// D546: one Botzone match as a gate-format games.jsonl line (ti insight / ti analyze read it): the side whose name
/// contains `cand` is the "candidate", the other the "baseline". Scores: Titanium `s` = cp from the mover's view;
/// a value in [-1, 1] (PyJataxx `h`, winrate-like) is mapped to cp = 400 ln((1+h)/(1-h)), |h| >= 1 = solved (+-19990).
fn game_jsonl(path: &Path, cand: &str) -> Result<String, String> {
    let m = read_match(path)?;
    let cs = m.names.iter().position(|n| n.to_lowercase().contains(&cand.to_lowercase())).ok_or_else(|| format!("{}: no player matches '{cand}'", m.id))?;
    let text = fs::read_to_string(path).map_err(|e| e.to_string())?;
    let v: Value = serde_json::from_str(&text).map_err(|e| e.to_string())?;
    let logs = v["logs"].as_array().ok_or("no logs")?;
    let mut grid = [[0u8; 7]; 7];
    grid[0][0] = 1;
    grid[6][6] = 1;
    grid[6][0] = 2;
    grid[0][6] = 2;
    let (o0, o1) = grid_bbs(&grid);
    let start = titanium::Board::from_occ(o0, o1, 0, 0);
    let mut b = start;
    let mut moves = Vec::new();
    let mut ply = 0usize;
    for e in logs {
        let key = if e.get("0").is_some() { "0" } else if e.get("1").is_some() { "1" } else { continue };
        let side = &e[key];
        let r = &side["response"];
        if side["verdict"].as_str() != Some("OK") || !r.is_object() {
            break;
        }
        let g = |k: &str| r[k].as_i64().unwrap_or(-1);
        let (x0, y0, x1, y1) = (g("x0"), g("y0"), g("x1"), g("y1"));
        if x0 < 0 || x1 < 0 {
            break;
        }
        let (from, to) = ((y0 * 7 + x0) as u8, (y1 * 7 + x1) as u8);
        let dist = (x1 - x0).abs().max((y1 - y0).abs());
        let uai = if dist == 1 { titanium::Board::sq_name(to) } else { format!("{}{}", titanium::Board::sq_name(from), titanium::Board::sq_name(to)) };
        let mv = b.parse_uai_move(&uai).ok_or_else(|| format!("{}: illegal {uai} at ply {}", m.id, ply + 1))?;
        let (depth, nodes, ms, eval, kind) = parse_debug(side["debug"].as_str().unwrap_or(""));
        let (score, mate) = match (eval, kind) {
            (Some(h), "h") if h.abs() < 0.9999 => (Some((400.0 * ((1.0 + h) / (1.0 - h)).ln()).round() as i64), None),
            (Some(h), "h") => (Some(if h > 0.0 { 19990 } else { -19990 }), None),
            (Some(x), "cp") => (Some(x.round() as i64), None),
            // cheetax search score: +-19950.. = proven win / loss (D938)
            (Some(x), "ev") if x.abs() >= 19000.0 => (Some(if x > 0.0 { 19990 } else { -19990 }), None),
            (Some(x), "ev") => (Some(x.round() as i64), None),
            _ => (None, None::<i64>),
        };
        let sidec = if key == "0" { "x" } else { "o" };
        let mover = if key == cs.to_string() { "candidate" } else { "baseline" };
        let js = |o: Option<i64>| o.map(|x| x.to_string()).unwrap_or_else(|| "null".into());
        moves.push(format!(
            "{{\"ply\":{ply},\"side\":\"{sidec}\",\"mover\":\"{mover}\",\"move\":\"{uai}\",\"score\":{},\"mate\":{},\"depth\":{},\"nodes\":{},\"ms\":{},\"pv\":\"\"}}",
            js(score), js(mate), js(depth), js(nodes), js(ms.or(side["time"].as_i64()))
        ));
        b = b.make_uai(mv);
        ply += 1;
    }
    let (fx, fo) = b.counts();
    let pts = match m.scores[cs].cmp(&m.scores[1 - cs]) {
        std::cmp::Ordering::Greater => "1",
        std::cmp::Ordering::Less => "0",
        _ => "0.5",
    };
    Ok(format!(
        "{{\"pair\":0,\"book\":0,\"start\":\"{}\",\"candidate_seat\":\"{}\",\"candidate_points\":{pts},\"final_x\":{fx},\"final_o\":{fo},\"match\":\"{}\",\"moves\":[{}]}}",
        start.to_ataxx_fen(),
        if cs == 0 { "x" } else { "o" },
        m.id,
        moves.join(",")
    ))
}

pub fn run_botzone(args: &[String]) -> ExitCode {
    match run(args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("botzone: {e}");
            ExitCode::FAILURE
        }
    }
}

fn run(args: &[String]) -> Result<(), String> {
    let out = flag(args, "--out").ok_or("--out DIR is required")?;
    let min_ply: u32 = flag(args, "--min-ply").and_then(|x| x.parse().ok()).unwrap_or(3);
    let bots: Vec<String> = flag(args, "--bots").map(|b| b.split(',').map(|s| s.trim().to_lowercase()).collect()).unwrap_or_default();
    let mut inputs = Vec::new();
    let mut i = 0;
    while i < args.len() {
        if args[i] == "--no-stats" {
            i += 1;
        } else if args[i].starts_with("--") {
            i += 2;
        } else {
            inputs.push(args[i].clone());
            i += 1;
        }
    }
    if inputs.is_empty() {
        return Err("usage: ti botzone <match.json|dir|URL> [...] --out DIR [--bots a,b] [--min-ply 3] [--no-stats]".into());
    }
    let (bin_p, meta_p) = (format!("{out}/positions.bin"), format!("{out}/meta.tsv"));
    if Path::new(&bin_p).exists() || Path::new(&meta_p).exists() {
        return Err(format!("{out} already holds positions.bin/meta.tsv: refusing to overwrite"));
    }
    let show_stats = !args.iter().any(|a| a == "--no-stats");
    if let Some(gj) = flag(args, "--games-jsonl") {
        let cand = flag(args, "--cand").unwrap_or_else(|| "titanium".into());
        let mut lines = Vec::new();
        for f in collect_files(&inputs)? {
            match game_jsonl(&f, &cand) {
                Ok(l) => lines.push(l),
                Err(e) => eprintln!("botzone: {e}"),
            }
        }
        fs::write(&gj, lines.join("\n") + "\n").map_err(|e| e.to_string())?;
        println!("wrote {gj}: {} games (candidate = '{cand}'); run ti analyze / ti insight on it", lines.len());
    }
    let mut agg: BTreeMap<(u64, u64, u64, u8), Agg> = BTreeMap::new();
    let (mut n_match, mut n_skip, mut n_bad, mut raw) = (0usize, 0usize, 0usize, 0usize);
    let mut by_mover: BTreeMap<String, usize> = BTreeMap::new();
    for f in collect_files(&inputs)? {
        let m = match read_match(&f) {
            Ok(m) => m,
            Err(e) => {
                eprintln!("botzone: warning: {e} (match skipped)");
                n_bad += 1;
                continue;
            }
        };
        if !bots.is_empty() && !m.names.iter().any(|n| bots.iter().any(|b| n.to_lowercase().contains(b))) {
            n_skip += 1;
            continue;
        }
        n_match += 1;
        if show_stats {
            print_stats(&m, &f);
        }
        let black_pts = match m.scores[0].cmp(&m.scores[1]) {
            std::cmp::Ordering::Greater => 1.0,
            std::cmp::Ordering::Less => 0.0,
            _ => 0.5,
        };
        for p in &m.positions {
            if p.ply < min_ply {
                continue;
            }
            raw += 1;
            *by_mover.entry(p.mover.clone()).or_default() += 1;
            let (c0, c1, cb) = canon(p.occ0, p.occ1, 0);
            let a = agg.entry((c0, c1, cb, p.turn)).or_default();
            let stm_pts: f64 = if p.turn == 0 { black_pts } else { 1.0 - black_pts };
            if stm_pts > 0.75 {
                a.w += 1;
            } else if stm_pts < 0.25 {
                a.l += 1;
            } else {
                a.d += 1;
            }
            a.margin += m.scores[p.turn as usize] - m.scores[1 - p.turn as usize];
            a.first_ply = if a.first_ply == 0 { p.ply } else { a.first_ply.min(p.ply) };
            a.movers.insert(p.mover.clone());
            a.matches.insert(m.id.clone());
        }
    }
    fs::create_dir_all(&out).map_err(|e| e.to_string())?;
    let mut bin = std::io::BufWriter::new(fs::File::create(&bin_p).map_err(|e| e.to_string())?);
    let mut meta = std::io::BufWriter::new(fs::File::create(&meta_p).map_err(|e| e.to_string())?);
    let _ = writeln!(meta, "row\tn\tw\tl\td\tavg_margin\tfirst_ply\tmovers\tmatches");
    for (row, ((o0, o1, bl, turn), a)) in agg.iter().enumerate() {
        let mut rec = [0u8; 25];
        rec[0..8].copy_from_slice(&o0.to_le_bytes());
        rec[8..16].copy_from_slice(&o1.to_le_bytes());
        rec[16..24].copy_from_slice(&bl.to_le_bytes());
        rec[24] = *turn;
        bin.write_all(&rec).map_err(|e| e.to_string())?;
        let n = a.w + a.l + a.d;
        let _ = writeln!(
            meta,
            "{row}\t{n}\t{}\t{}\t{}\t{:.1}\t{}\t{}\t{}",
            a.w,
            a.l,
            a.d,
            a.margin as f64 / n.max(1) as f64,
            a.first_ply,
            a.movers.iter().cloned().collect::<Vec<_>>().join("|"),
            a.matches.iter().cloned().collect::<Vec<_>>().join("|")
        );
    }
    println!(
        "botzone: {n_match} matches read ({n_skip} filtered out, {n_bad} unreadable) -> {raw} positions -> {} unique under D4 ({} merged duplicates) -> {bin_p}",
        agg.len(),
        raw - agg.len()
    );
    for (m, c) in &by_mover {
        println!("  moves by {m}: {c}");
    }
    println!("next: ti relabel --bin {bin_p} --out <dir>/relabel --nodes 5000 --net <main.tnn>   (Titanium labels only; opponent scores are not used)");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_titanium_compact_debug() {
        let (d, n, t, e, k) = parse_debug("d15 n1988957 t9297 s62 bundled-main/22");
        assert_eq!((d, n, t, e, k), (Some(15), Some(1988957), Some(9297), Some(62.0), "cp"));
    }

    #[test]
    fn parses_uci_style_debug() {
        let (d, n, t, e, k) = parse_debug("info depth 5 seldepth 5 score -38 time 209 nodes 1343 nps 6395 pv f7 e3 e6");
        assert_eq!((d, n, t, e, k), (Some(5), Some(1343), Some(209), Some(-38.0), "cp"));
        let (_, _, t, e, k) = parse_debug("nodes=1 depth=7 winrate=0.572 scoring=2.949 time=3.60");
        assert_eq!((t, e, k), (Some(3600), Some(0.572), "wr"));
    }

    #[test]
    fn parses_cheetax_debug_search_score_not_eval_score() {
        // D938: `evaluation` = search score (mover's view); `eval score` must not override it; "30nodes" glued token
        let (d, n, _, e, k) = parse_debug("depth 20 seldepth 30nodes 8908800 evaluation -41 eval score 56");
        assert_eq!((d, n, e, k), (Some(20), Some(8908800), Some(-41.0), "ev"));
        let (_, _, _, e, _) = parse_debug("depth 99 seldepth 44nodes 117567 evaluation 19964 eval score -524");
        assert_eq!(e, Some(19964.0));
    }

    #[test]
    fn canon_is_symmetric_invariant() {
        let (o0, o1) = (0b1000001u64 | (1 << 24), 1u64 << 42 | 1 << 6);
        let c = canon(o0, o1, 0);
        for k in 0..8 {
            assert_eq!(canon(map_bb(k, o0), map_bb(k, o1), 0), c, "symmetry {k}");
        }
    }

    #[test]
    fn start_position_is_fixed_by_the_group_up_to_colour() {
        // the four corners map to corners; stone counts survive every symmetry
        for k in 0..8 {
            assert_eq!(map_bb(k, (1 << 0) | (1 << 48)).count_ones(), 2);
        }
    }
}

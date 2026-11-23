//! D1618 LTC labelling: Titanium at a fixed time per position, stored as a SEPARATE label (never mixed into cp).
//!
//! The prover (zataxx-bz) answers a different question: "is this position a forced win, and in how many plies".
//! Titanium cannot match it for proving (74% of its proofs at 1 s, and none of the ones zataxx misses), but a
//! 1-second Titanium search is ~700k nodes - 45x deeper than the 5k/15k labels the dataset carries - so for the
//! positions WITHOUT a proof it is the strongest label we can produce, and it is in the units the net trains on.
//!
//! Output (STEM.tsv, one line per position): the game, ply, both bitboards, the side to move, stones, Titanium's
//! score, depth, nodes, ms, its best move and the move code. Nothing else is written: the merge into a .tdat
//! (cp = LTC, cp5k = STC, relabel_src = LTC) is a separate step, so a bad run cannot touch a dataset.
//!
//!   ti ltc --tgam DIR|a.tgam[,..] --out STEM [--ti EXE] [--ms 1000] [--threads 1] [--min-stones 40] [--limit-games N]

use std::collections::HashSet;
use std::io::Write;
use std::process::Command;
use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};
use std::sync::Mutex;
use titanium::ttcodec::move_code;
use titanium::{Board, Move};

/// The 49-char board string `titanium-cli bestmove` expects: rows top->bottom, `. x o #`, then the side to move.
fn board_str(b: &Board) -> String {
    let mut s = String::with_capacity(64);
    for y in 0..7 {
        for x in 0..7 {
            let sq = y * 7 + x;
            let bit = 1u64 << sq;
            let c = if b.occ[0] & bit != 0 {
                'x'
            } else if b.occ[1] & bit != 0 {
                'o'
            } else {
                '.'
            };
            s.push(c);
        }
        if y > 0 {
            s.push(' ');
        }
    }
    s.push(if b.turn == 0 { ' ' } else { ' ' });
    s.push(if b.turn == 0 { 'b' } else { 'w' });
    s
}

fn uai(m: Move) -> String {
    const F: [char; 7] = ['a', 'b', 'c', 'd', 'e', 'f', 'g'];
    if m.is_pass() {
        return "0000".into();
    }
    let (f0, t0) = ((m.from % 7) as usize, (m.to % 7) as usize);
    let (r0, r1) = (m.from / 7, m.to / 7);
    format!("{}{}{}{}", F[f0], r0 + 1, F[t0], r1 + 1)
}

fn key(b: &Board) -> u64 {
    crate::gamedata::pos_key(b.me(), b.opp(), 0)
}

pub fn run(args: &[String]) -> Result<(), String> {
    let f = |k: &str| args.iter().position(|a| a == k).and_then(|i| args.get(i + 1)).cloned();
    let n = |k: &str, d: u64| -> u64 { f(k).and_then(|v| v.parse().ok()).unwrap_or(d) };
    let spec = f("--tgam").ok_or(
        "usage: ti ltc --tgam DIR|a.tgam[,..] --out STEM [--ti EXE] [--ms 1000] [--threads 1] [--min-stones 40] [--limit-games N]",
    )?;
    let out = f("--out").ok_or("ltc: --out STEM is required")?;
    let ti = f("--ti").unwrap_or_else(|| "C:/gitProjects/mainpromo-ataxx/target/x86_64-pc-windows-gnu/release/titanium-cli.exe".into());
    let ms = n("--ms", 1000).max(20);
    let threads = n("--threads", 1).max(1) as usize;
    let min_stones = n("--min-stones", 40) as u32;
    let limit = n("--limit-games", u64::MAX) as usize;

    // every endgame position of every game, first occurrence only (same candidate rule as ti proof-lines)
    let mut cand: Vec<(usize, usize, Board)> = Vec::new(); // (game, ply, board)
    let mut seen: HashSet<u64> = HashSet::new();
    let mut games = 0usize;
    'files: for t in crate::proofline::list_tgams(&spec)? {
        for g in crate::gamedata::GameReader::open(&t)?.flatten() {
            if games >= limit {
                break 'files;
            }
            let sb = g.start();
            if sb.occ != Board::start().occ || sb.turn != Board::start().turn || sb.blockers != 0 {
                continue;
            }
            let Ok((plies, _)) = g.replay() else { continue };
            games += 1;
            for (i, p) in plies.iter().enumerate() {
                let b = p.board;
                if (b.occ[0] | b.occ[1]).count_ones() < min_stones || b.bz_over() || i >= 401 || !seen.insert(key(&b)) {
                    continue;
                }
                cand.push((games, i, b));
            }
        }
    }
    eprintln!("ltc: {games} games, {} positions (>= {min_stones} stones), titanium {ti}, {ms} ms, {threads} threads -> {out}.tsv", cand.len());

    let file = std::fs::File::create(format!("{out}.tsv")).map_err(|e| format!("{out}.tsv: {e}"))?;
    let w = Mutex::new(std::io::BufWriter::new(file));
    writeln!(
        w.lock().unwrap(),
        "game\tply\tocc0\tocc1\tturn\tstones\tscore\tdepth\tnodes\tms\tmove\tcode"
    )
    .map_err(|e| e.to_string())?;

    let next = AtomicUsize::new(0);
    let done = AtomicUsize::new(0);
    let t0 = std::time::Instant::now();
    let errs: Mutex<Vec<String>> = Mutex::new(Vec::new());
    std::thread::scope(|s| {
        for _ in 0..threads {
            s.spawn(|| loop {
                let i = next.fetch_add(1, Relaxed);
                let Some((gi, ply, b)) = cand.get(i) else { return };
                let mut cmd = Command::new(&ti);
                cmd.arg("bestmove").arg(board_str(b)).arg("--time").arg(ms.to_string());
                let o = match cmd.output() {
                    Ok(o) => o,
                    Err(e) => {
                        errs.lock().unwrap().push(format!("{ti}: {e}"));
                        return;
                    }
                };
                let text = String::from_utf8_lossy(&o.stdout);
                let Some(line) = text.lines().find(|l| l.starts_with("bestmove ")) else {
                    errs.lock().unwrap().push(format!("no bestmove for game {gi} ply {ply}: {}", text.trim()));
                    return;
                };
                // bestmove <mv>  score <s>  depth <d>  nodes <n>  <ms> ms
                let t: Vec<&str> = line.split_whitespace().collect();
                let get = |k: &str| t.iter().position(|&x| x == k).and_then(|j| t.get(j + 1)).and_then(|v| v.parse::<i64>().ok());
                let mv = t.get(1).copied().unwrap_or("0000");
                let code = if mv == "pass" { 0 } else { b.parse_uai_move(mv).map(move_code).unwrap_or(0) };
                let stones = (b.occ[0] | b.occ[1]).count_ones();
                let msv = t.iter().position(|&x| x == "ms").and_then(|j| j.checked_sub(1)).and_then(|j| t.get(j)).and_then(|v| v.parse::<i64>().ok()).unwrap_or(0);
                let rec = format!(
                    "{gi}\t{ply}\t{:016x}\t{:016x}\t{}\t{stones}\t{}\t{}\t{}\t{msv}\t{mv}\t{code}\n",
                    b.me(),
                    b.opp(),
                    b.turn,
                    get("score").unwrap_or(0),
                    get("depth").unwrap_or(0),
                    get("nodes").unwrap_or(0)
                );
                let mut w = w.lock().unwrap();
                if let Err(e) = w.write_all(rec.as_bytes()) {
                    drop(w);
                    errs.lock().unwrap().push(format!("write: {e}"));
                    return;
                }
                drop(w);
                let d = done.fetch_add(1, Relaxed) + 1;
                if d % 500 == 0 {
                    eprintln!("ltc: {d}/{} positions, {:.0}s", cand.len(), t0.elapsed().as_secs_f64());
                }
            });
        }
    });
    let _ = w.lock().unwrap().flush();
    let e = errs.into_inner().unwrap();
    if !e.is_empty() {
        return Err(format!("ltc: {} errors: {}", e.len(), e.join(" | ")));
    }
    eprintln!("ltc: done {} positions in {:.0}s", done.load(Relaxed), t0.elapsed().as_secs_f64());
    Ok(())
}

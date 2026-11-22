//! D1590-D1596 PROOF LINES, Botzone-exact (user orders): prove endgame positions of real games with the Botzone-rules
//! Zataxx (zataxx-bz) and PLAY OUT each proof's winning line as a game, exactly as if the engine had played it.
//!
//! * Every position is sent as `position startpos moves <the whole game so far>`: Zataxx then knows the real history,
//!   so the judge's repetition ban and the ply-401 end are exact (D1593/D1596). Games that do not start at the standard
//!   start position are skipped (counted): their history is unknown.
//! * Proven = |score| >= 31,000 (Zataxx mate bound) on `--confirm` consecutive depths (ZAX_STOP_ON_PROOF).
//! * Play-out: the proven move is played, the next position is proven again (same game, same history + that move),
//!   its proven sign must be the opposite (the other side to move), and so on until the game really ends. At the end
//!   the Botzone final count must give the result the proof claimed; a line that disagrees anywhere is DISCARDED and
//!   counted (never written). A line whose proof stops before the end keeps the positions proven so far.
//! * No principal variation is stored anywhere (LAWS D15). Each line is written as a game record (.tgam + .tbst, the
//!   `ti gen` format): per ply the played move = the engine's best move, the proven score (Titanium mate units,
//!   mover's view), result = the proven result. `ti data games-tdat` turns them into rows like any game.
//!
//!   ti proof-lines --tgam DIR|a.tgam[,..] --out STEM [--zataxx EXE] [--threads 4] [--slice-ms 100] [--max-ms 1000] [--confirm 2]
//!                  [--min-stones 40] [--limit-games N]
use std::collections::HashMap;
use std::io::{BufRead, BufReader, Write};
use std::process::{Child, ChildStdin, ChildStdout, Command, ExitCode, Stdio};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering::Relaxed};
use std::sync::Mutex;
use titanium::ttcodec::{enc_score, move_code};
use titanium::{Board, Move, MoveList};

const ZX_PROVEN: i32 = 31_000;
const ZX_MATE: i32 = 32_000;
const TI_MATE: i32 = 100_000;
const BZ_GAME_PLIES: usize = 401;
const DEFAULT_ZATAXX: &str = "C:/gitProjects/zataxx-bz/target/x86_64-pc-windows-gnu/release/zataxx.exe";
/// game record source tag for proof lines (gen = 0)
const SRC_PROOF_LINE: u16 = 202;

fn uai(m: Move) -> String {
    if m.is_clone() {
        Board::sq_name(m.to)
    } else {
        format!("{}{}", Board::sq_name(m.from), Board::sq_name(m.to))
    }
}

/// Botzone judge repetition history, kept by THIS tool independently of Zataxx (judge rule, as titanium search_bz /
/// collect_bz_bans_cold): `snaps[k]` = occupancy (side 0, side 1) right after the k-th move is PLACED, before captures
/// (k = 0: the start position); `base[k]` = index of the last clone at or before k (positions never repeat across a
/// clone). If the latest snapshot occurred earlier since the last clone (at distance >= 4), the snapshot that followed
/// it then is banned now (playing it = the judge's CIRCULAR_MOVE forfeit).
#[derive(Clone)]
struct Rep {
    snaps: Vec<(u64, u64)>,
    base: Vec<usize>,
}

impl Rep {
    fn new(start: &Board) -> Rep {
        Rep { snaps: vec![(start.occ[0], start.occ[1])], base: vec![0] }
    }
    fn snap(b: &Board, m: Move) -> (u64, u64) {
        let mut occ = b.occ;
        let me = b.turn as usize;
        if !m.is_clone() {
            occ[me] &= !(1u64 << m.from);
        }
        occ[me] |= 1u64 << m.to;
        (occ[0], occ[1])
    }
    fn banned(&self, b: &Board, m: Move) -> bool {
        let cur = self.snaps.len() - 1;
        let rb = self.base[cur];
        if cur < rb + 4 {
            return false;
        }
        let last = self.snaps[cur];
        let s = Rep::snap(b, m);
        (rb..cur - 3).any(|i| self.snaps[i] == last && self.snaps[i + 1] == s)
    }
    fn push(&mut self, b: &Board, m: Move) {
        let s = Rep::snap(b, m);
        let k = self.snaps.len();
        let nb = if m.is_clone() { k } else { self.base[k - 1] };
        self.snaps.push(s);
        self.base.push(nb);
    }
}

/// side-to-move-relative position key (the dataset convention: occ0 = side to move)
fn key(b: &Board) -> u64 {
    crate::gamedata::pos_key(b.me(), b.opp(), 0)
}

struct Ans {
    score: i32,
    proven: bool,
    best: String,
    depth: u32,
    /// the winning line of the last completed depth (Zataxx walks it from its TT, uncapped since D1592); used at once to
    /// play the line out, never stored (LAWS D15)
    line: Vec<String>,
}

/// D206 calibrated early bail (zataxx-bz ZAX_ABORT, 0 of 625 proofs lost): a search not heading for a proof stops
const DEFAULT_ABORT: &str = "1000000:300,2500000:700";

struct Zx {
    child: Child,
    inp: ChildStdin,
    out: BufReader<ChildStdout>,
    confirm: u32,
}

impl Zx {
    fn spawn(exe: &str, confirm: u32, abort: &str) -> Result<Zx, String> {
        let mut child = Command::new(exe)
            .env("ZAX_STOP_ON_PROOF", "1")
            .env("ZAX_PROOF_CONFIRM", confirm.to_string())
            .env("ZAX_ABORT", abort)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .map_err(|e| format!("spawn {exe}: {e}"))?;
        let inp = child.stdin.take().unwrap();
        let out = BufReader::new(child.stdout.take().unwrap());
        let mut z = Zx { child, inp, out, confirm };
        z.send("uai\nisready\n")?;
        z.wait_for("readyok")?;
        Ok(z)
    }
    fn send(&mut self, s: &str) -> Result<(), String> {
        self.inp.write_all(s.as_bytes()).and_then(|_| self.inp.flush()).map_err(|e| format!("zataxx stdin: {e}"))
    }
    fn wait_for(&mut self, tok: &str) -> Result<(), String> {
        let mut line = String::new();
        loop {
            line.clear();
            if self.out.read_line(&mut line).map_err(|e| e.to_string())? == 0 {
                return Err("zataxx exited".into());
            }
            if line.trim() == tok {
                return Ok(());
            }
        }
    }
    /// Prove the position reached by `moves` from the standard start (the whole game: Botzone-exact history).
    fn prove(&mut self, moves: &[String], ms: u64, fresh: bool) -> Result<Ans, String> {
        let mv = if moves.is_empty() { String::new() } else { format!(" moves {}", moves.join(" ")) };
        let ng = if fresh { "uainewgame\n" } else { "" };
        self.send(&format!("{ng}position startpos{mv}\nisready\n"))?;
        self.wait_for("readyok")?;
        self.send(&format!("go movetime {}\n", ms.max(20)))?;
        let (mut a, mut run, mut line) = (Ans { score: 0, proven: false, best: String::new(), depth: 0, line: Vec::new() }, 0u32, String::new());
        loop {
            line.clear();
            if self.out.read_line(&mut line).map_err(|e| e.to_string())? == 0 {
                return Err("zataxx exited".into());
            }
            let l = line.trim();
            if let Some(rest) = l.strip_prefix("bestmove") {
                a.best = rest.trim().to_string();
                break;
            }
            if l.contains("D1596 WARNING") {
                return Err(format!("zataxx says the history is unknown: {l}"));
            }
            if !l.starts_with("info depth") {
                continue;
            }
            let t: Vec<&str> = l.split_whitespace().collect();
            let get = |k: &str| {
                t.iter()
                    .position(|&x| x == k)
                    .and_then(|i| t.get(if t.get(i + 1) == Some(&"cp") { i + 2 } else { i + 1 }))
                    .and_then(|v| v.parse::<i64>().ok())
            };
            let (Some(d), Some(s)) = (get("depth"), get("score")) else { continue };
            a.depth = d as u32;
            a.score = s as i32;
            a.line = t.iter().position(|&x| x == "pv").map(|i| t[i + 1..].iter().map(|x| x.to_string()).collect()).unwrap_or_default();
            run = if a.score.abs() >= ZX_PROVEN { run + 1 } else { 0 };
        }
        a.proven = run >= self.confirm;
        Ok(a)
    }
}

impl Drop for Zx {
    fn drop(&mut self) {
        let _ = self.send("quit\n");
        let _ = self.child.wait();
    }
}

#[derive(Default)]
struct Counts {
    games: AtomicU64,
    not_startpos: AtomicU64,
    bad_replay: AtomicU64,
    queried: AtomicU64,
    proven_roots: AtomicU64,
    lines_written: AtomicU64,
    lines_complete: AtomicU64,
    lines_cut: AtomicU64,
    discarded_end_mismatch: AtomicU64,
    discarded_sign_flip: AtomicU64,
    discarded_bad_move: AtomicU64,
    discarded_ban: AtomicU64,
    /// searches made inside a line (the walked line ended before the game did)
    researches: AtomicU64,
    /// complete lines whose length differs from the proven mate distance (kept: the final count decides)
    dist_mismatch: AtomicU64,
    /// candidates that used the whole per-position budget without a proof
    dropped_budget: AtomicU64,
    /// positions walked that an earlier line had already written (kept in the line, removed at conversion)
    lines_joined: AtomicU64,
    positions_written: AtomicU64,
    covered_skips: AtomicU64,
    conflicts: AtomicU64,
}

fn list_tgams(spec: &str) -> Result<Vec<String>, String> {
    let mut v = Vec::new();
    for p in spec.split(',').filter(|s| !s.is_empty()) {
        let path = std::path::Path::new(p);
        if path.is_dir() {
            let mut d: Vec<String> = std::fs::read_dir(path)
                .map_err(|e| format!("{p}: {e}"))?
                .filter_map(|e| e.ok().map(|e| e.path()))
                .filter(|q| q.extension().map(|x| x == "tgam").unwrap_or(false))
                .map(|q| q.to_string_lossy().replace('\\', "/"))
                .collect();
            d.sort();
            v.extend(d);
        } else {
            v.push(p.to_string());
        }
    }
    if v.is_empty() {
        return Err(format!("no .tgam in {spec}"));
    }
    Ok(v)
}

/// result for the side to move at a FINISHED Botzone game (bz_over or 401 moves played): +1 / 0 / -1
fn final_sign(b: &Board, game_ply: usize) -> i32 {
    let (x, o) = if b.bz_over() { b.bz_counts() } else { (b.occ[0].count_ones(), b.occ[1].count_ones()) };
    let _ = game_ply;
    let d = if b.turn == 0 { x as i32 - o as i32 } else { o as i32 - x as i32 };
    d.signum()
}

pub fn run(args: &[String]) -> ExitCode {
    match run_inner(args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("proof-lines: {e}");
            ExitCode::FAILURE
        }
    }
}

/// D1599 chance that a position proves, from a score: D204 calibration (Titanium cp: > 5000 100%, 1500-5000 92%,
/// 500-1500 56%, < 500 ~28%); Titanium-proven labels are certain.
fn chance_ti(abs_cp: i32) -> f64 {
    if abs_cp >= 98_000 {
        1.0
    } else if abs_cp >= 5000 {
        0.99
    } else if abs_cp >= 1500 {
        0.92
    } else if abs_cp >= 500 {
        0.56
    } else {
        0.28
    }
}

/// the same in Zataxx units (D995 scale: Titanium cp = 0.573 x Zataxx)
fn chance_zx(abs_zx: i32) -> f64 {
    chance_ti((abs_zx as f64 * 0.573) as i32)
}

/// one game: its moves from the start, the board before every ply, and the judge's repetition history
struct Ctx {
    moves: Vec<String>,
    boards: Vec<Board>,
    rep: Rep,
}

/// queue entry: priority = chance of a proof per slice, discounted by the slices it already used (max-heap)
#[derive(PartialEq, Eq, PartialOrd, Ord)]
struct QItem {
    prio: u64,
    cand: u32,
    slices: u16,
}

fn qprio(chance: f64, slices: u16) -> u64 {
    (chance * 0.7f64.powi(slices as i32) * 1e12) as u64
}

/// Play out a proven position: walk the proven search's own winning line (no search per ply); every position gets
/// the proven result for its side to move and the mate distance counted down; where the line ends before the game
/// does, the position is proven again (same game) and walked on. Checked: legal move (Titanium), the judge's
/// repetition ban (this tool's own history), the final Botzone count. Writes one game record per line.
#[allow(clippy::too_many_arguments)]
fn play_out(
    z: &mut Zx,
    ctx: &Ctx,
    i: usize,
    a: Ans,
    ms: u64,
    gi: usize,
    c: &Counts,
    covered: &Mutex<HashMap<u64, i8>>,
    w: &Mutex<crate::gen::BinOut>,
) -> Result<(), String> {
    let mut hist: Vec<String> = ctx.moves[..i].to_vec();
    let mut rep = Rep { snaps: ctx.rep.snaps[..=i].to_vec(), base: ctx.rep.base[..=i].to_vec() };
    let (mut cur, mut expect, mut ans) = (ctx.boards[i], a.score.signum(), a);
    let mut line: Vec<(Board, Move, i32, u32)> = Vec::new();
    let mut verdict: Result<bool, &AtomicU64> = Ok(false); // Ok(complete?) / Err(discard counter)
    let mut dist = (ZX_MATE - ans.score.abs()).max(0);
    let mut pv: std::collections::VecDeque<String> = if ans.line.is_empty() { vec![ans.best.clone()].into() } else { ans.line.clone().into() };
    loop {
        // D1599: a position an earlier line already wrote is walked THROUGH (not stopped at: from there each proof
        // search picks its own winning moves, so the rest of this line finds new positions; measured 1,219 vs 952
        // distinct positions on 20 games). Repeated rows are removed at conversion (ti data games-tdat --dedup).
        if !line.is_empty() && covered.lock().unwrap().contains_key(&key(&cur)) {
            c.lines_joined.fetch_add(1, Relaxed);
        }
        let tok = match pv.pop_front() {
            Some(t) => t,
            None => {
                c.researches.fetch_add(1, Relaxed);
                ans = z.prove(&hist, ms, false)?;
                if !ans.proven {
                    break; // proof stops here: keep the positions proven so far
                }
                if ans.score.signum() != expect {
                    verdict = Err(&c.discarded_sign_flip);
                    break;
                }
                dist = (ZX_MATE - ans.score.abs()).max(0);
                pv = if ans.line.is_empty() { vec![ans.best.clone()].into() } else { ans.line.clone().into() };
                continue;
            }
        };
        let mut legal = MoveList::new();
        cur.legal_moves_dedup_into(&mut legal);
        let Some(m) = cur.parse_uai_move(&tok) else {
            verdict = Err(&c.discarded_bad_move);
            break;
        };
        let is_legal = legal.iter().any(|l| l.to == m.to && (l.is_clone() && m.is_clone() || l.from == m.from));
        if !is_legal {
            verdict = Err(&c.discarded_bad_move);
            break;
        }
        // the judge's repetition ban, checked by this tool on the real game history
        if rep.banned(&cur, m) {
            verdict = Err(&c.discarded_ban);
            break;
        }
        rep.push(&cur, m);
        line.push((cur, m, expect * (TI_MATE - dist.max(1)), ans.depth));
        hist.push(uai(m));
        cur = cur.make(m);
        expect = -expect;
        dist -= 1;
        if cur.bz_over() || hist.len() >= BZ_GAME_PLIES {
            verdict = if final_sign(&cur, hist.len()) == expect { Ok(true) } else { Err(&c.discarded_end_mismatch) };
            if dist != 0 {
                c.dist_mismatch.fetch_add(1, Relaxed);
            }
            break;
        }
    }
    let complete = match verdict {
        Err(cnt) => {
            cnt.fetch_add(1, Relaxed);
            return Ok(());
        }
        Ok(done) => done,
    };
    if line.is_empty() {
        return Ok(());
    }
    {
        let mut cv = covered.lock().unwrap();
        for (pb, _, sc, _) in &line {
            let sg = sc.signum() as i8;
            match cv.get(&key(pb)) {
                Some(&old) if old != sg => {
                    c.conflicts.fetch_add(1, Relaxed);
                }
                Some(_) => {}
                None => {
                    cv.insert(key(pb), sg);
                }
            }
        }
    }
    let root = line[0].0;
    // result from side 0's view: the root's side to move wins if its proven score is > 0
    let root_wins = line[0].2 > 0;
    let side0_wins = root_wins == (root.turn == 0);
    let fin = if complete { cur.bz_counts() } else { (255, 255) };
    let plies_bin: Vec<(u16, i16)> = line.iter().map(|(_, m, sc, d)| (move_code(*m) | ((*d as u16).min(63) << 10), enc_score(*sc))).collect();
    let best: Vec<(u16, i16)> = line.iter().map(|(_, m, sc, _)| (move_code(*m), enc_score(*sc))).collect();
    let pg = crate::gamedata::PackedGame {
        occ: root.occ,
        blockers: root.blockers,
        turn: root.turn,
        cand_seat: 2,
        forfeit: false,
        final_cnt: [fin.0.min(254) as u8, fin.1.min(254) as u8],
        result: if side0_wins { 2 } else { 0 },
        src: SRC_PROOF_LINE,
        book: gi as u32,
        pair: i as u32,
        plies: plies_bin,
    };
    w.lock().unwrap().append(&pg, u32::MAX, &best).map_err(|e| e.to_string())?;
    c.lines_written.fetch_add(1, Relaxed);
    c.positions_written.fetch_add(line.len() as u64, Relaxed);
    if complete {
        c.lines_complete.fetch_add(1, Relaxed);
    } else {
        c.lines_cut.fetch_add(1, Relaxed);
    }
    Ok(())
}

fn run_inner(args: &[String]) -> Result<(), String> {
    let f = |n: &str| crate::flag_str(args, n);
    let (Some(spec), Some(out)) = (f("--tgam"), f("--out")) else {
        return Err("usage: ti proof-lines --tgam DIR|a.tgam[,..] --out STEM [--zataxx EXE] [--threads 4] [--slice-ms 100] [--max-ms 1000] [--confirm 2] [--min-stones 40] [--limit-games N] [--abort N:S,..]".into());
    };
    let zexe = f("--zataxx").unwrap_or_else(|| DEFAULT_ZATAXX.to_string());
    let threads = crate::flag_u64(args, "--threads").unwrap_or(4).max(1) as usize;
    let slice_ms = crate::flag_u64(args, "--slice-ms").unwrap_or(100).max(20);
    let max_ms = crate::flag_u64(args, "--max-ms").unwrap_or(1000).max(slice_ms);
    let confirm = crate::flag_u64(args, "--confirm").unwrap_or(2).max(1) as u32;
    let min_stones = crate::flag_u64(args, "--min-stones").unwrap_or(40) as u32;
    let abort = f("--abort").unwrap_or_else(|| DEFAULT_ABORT.to_string());
    let limit = crate::flag_u64(args, "--limit-games").unwrap_or(u64::MAX) as usize;
    let c = Counts::default();
    // 1. games -> contexts; every endgame position (first occurrence) -> a candidate with its prior chance of a proof
    let start = Board::start();
    let mut ctxs: Vec<Ctx> = Vec::new();
    let mut cands: Vec<(u32, u32)> = Vec::new(); // (ctx, ply)
    let mut heap = std::collections::BinaryHeap::new();
    let mut seen = std::collections::HashSet::new();
    'files: for t in list_tgams(&spec)? {
        for g in crate::gamedata::GameReader::open(&t)?.flatten() {
            if c.games.load(Relaxed) as usize >= limit {
                break 'files;
            }
            c.games.fetch_add(1, Relaxed);
            let sb = g.start();
            if sb.occ != start.occ || sb.turn != start.turn || sb.blockers != 0 {
                c.not_startpos.fetch_add(1, Relaxed);
                continue;
            }
            let Ok((plies, _)) = g.replay() else {
                c.bad_replay.fetch_add(1, Relaxed);
                continue;
            };
            let mut rep = Rep::new(&sb);
            for p in &plies {
                rep.push(&p.board, p.mv);
            }
            let ci = ctxs.len() as u32;
            for (i, p) in plies.iter().enumerate() {
                let b = p.board;
                if (b.occ[0] | b.occ[1]).count_ones() < min_stones || b.bz_over() || i >= BZ_GAME_PLIES || !seen.insert(key(&b)) {
                    continue;
                }
                let chance = p.score.map(|s| chance_ti(s.abs())).unwrap_or(0.28);
                heap.push(QItem { prio: qprio(chance, 0), cand: cands.len() as u32, slices: 0 });
                cands.push((ci, i as u32));
            }
            ctxs.push(Ctx { moves: plies.iter().map(|p| uai(p.mv)).collect(), boards: plies.iter().map(|p| p.board).collect(), rep });
        }
    }
    let total_cands = cands.len();
    eprintln!(
        "proof-lines: {} games, {total_cands} candidate positions (>= {min_stones} stones), zataxx {zexe}, {threads} threads, slice {slice_ms} ms, at most {max_ms} ms per position, confirm {confirm} -> {out}.tgam/.tbst",
        c.games.load(Relaxed)
    );
    let heap = Mutex::new(heap);
    let active = AtomicUsize::new(0);
    let w = Mutex::new(crate::gen::BinOut::create(&out).map_err(|e| format!("{out}: {e}"))?);
    // position -> proven sign for its side to move (first proof wins; a later opposite proof is a conflict)
    let covered: Mutex<HashMap<u64, i8>> = Mutex::new(HashMap::new());
    let t0 = std::time::Instant::now();
    let stop_dir = std::path::PathBuf::from(format!("{out}.run"));
    let _ = std::fs::create_dir_all(&stop_dir);
    let errs: Mutex<Vec<String>> = Mutex::new(Vec::new());
    let done_n = AtomicUsize::new(0);
    std::thread::scope(|s| {
        for _ in 0..threads {
            s.spawn(|| {
                let mut z = match Zx::spawn(&zexe, confirm, &abort) {
                    Ok(z) => z,
                    Err(e) => {
                        errs.lock().unwrap().push(e);
                        return;
                    }
                };
                loop {
                    if crate::stopctl::requested(Some(stop_dir.as_path())).is_some() || !errs.lock().unwrap().is_empty() {
                        return;
                    }
                    // take the best candidate; an empty queue ends the worker only when no other worker can push back
                    let item = {
                        let mut h = heap.lock().unwrap();
                        let it = h.pop();
                        if it.is_some() {
                            active.fetch_add(1, Relaxed);
                        }
                        it
                    };
                    let Some(it) = item else {
                        if active.load(Relaxed) == 0 {
                            return;
                        }
                        std::thread::sleep(std::time::Duration::from_millis(5));
                        continue;
                    };
                    let (ci, i) = cands[it.cand as usize];
                    let ctx = &ctxs[ci as usize];
                    let b = ctx.boards[i as usize];
                    if covered.lock().unwrap().contains_key(&key(&b)) {
                        c.covered_skips.fetch_add(1, Relaxed);
                        done_n.fetch_add(1, Relaxed);
                        active.fetch_sub(1, Relaxed);
                        continue;
                    }
                    c.queried.fetch_add(1, Relaxed);
                    let r = z.prove(&ctx.moves[..i as usize], slice_ms, false).and_then(|a| {
                        if a.proven {
                            c.proven_roots.fetch_add(1, Relaxed);
                            done_n.fetch_add(1, Relaxed);
                            play_out(&mut z, ctx, i as usize, a, max_ms, ci as usize, &c, &covered, &w)
                        } else {
                            // not proven in this slice: re-rank from the score Zataxx reached, or drop at the budget
                            let sl = it.slices + 1;
                            if sl as u64 * slice_ms < max_ms {
                                heap.lock().unwrap().push(QItem { prio: qprio(chance_zx(a.score.abs()), sl), cand: it.cand, slices: sl });
                            } else {
                                c.dropped_budget.fetch_add(1, Relaxed);
                                done_n.fetch_add(1, Relaxed);
                            }
                            Ok(())
                        }
                    });
                    active.fetch_sub(1, Relaxed);
                    if let Err(e) = r {
                        errs.lock().unwrap().push(e);
                        return;
                    }
                    let q = c.queried.load(Relaxed);
                    if q % 500 == 0 {
                        eprintln!(
                            "proof-lines: {}/{total_cands} candidates done, {q} slices, {} proven, {} lines ({} to the end), {} positions, discarded {}+{}+{}+{}, conflicts {}, {:.0}s",
                            done_n.load(Relaxed),
                            c.proven_roots.load(Relaxed),
                            c.lines_written.load(Relaxed),
                            c.lines_complete.load(Relaxed),
                            c.positions_written.load(Relaxed),
                            c.discarded_end_mismatch.load(Relaxed),
                            c.discarded_sign_flip.load(Relaxed),
                            c.discarded_bad_move.load(Relaxed),
                            c.discarded_ban.load(Relaxed),
                            c.conflicts.load(Relaxed),
                            t0.elapsed().as_secs_f64()
                        );
                    }
                }
            });
        }
    });
    let e = errs.into_inner().unwrap();
    let rep = format!(
        "proof-lines report ({:.0}s)\ngames {} (not from the start position, skipped: {}, replay errors: {})\ncandidate positions {}, slices searched {}, skipped as already on a proof line {}, dropped at the per-position budget {}\nproven roots {}\nlines written {} (to the real end {}, proof stopped earlier {}), positions written {} (of them also on an earlier line {}: removed by games-tdat --dedup)\nDISCARDED: final count disagrees with the proof {}, proven sign flipped mid-line {}, illegal / unparsable move {}, repetition-banned move {}\nconflicting proofs for the same position {}\nsearches inside lines {}, complete lines whose length != proven distance {}\nerrors: {}\n",
        t0.elapsed().as_secs_f64(),
        c.games.load(Relaxed),
        c.not_startpos.load(Relaxed),
        c.bad_replay.load(Relaxed),
        total_cands,
        c.queried.load(Relaxed),
        c.covered_skips.load(Relaxed),
        c.dropped_budget.load(Relaxed),
        c.proven_roots.load(Relaxed),
        c.lines_written.load(Relaxed),
        c.lines_complete.load(Relaxed),
        c.lines_cut.load(Relaxed),
        c.positions_written.load(Relaxed),
        c.lines_joined.load(Relaxed),
        c.discarded_end_mismatch.load(Relaxed),
        c.discarded_sign_flip.load(Relaxed),
        c.discarded_bad_move.load(Relaxed),
        c.discarded_ban.load(Relaxed),
        c.conflicts.load(Relaxed),
        c.researches.load(Relaxed),
        c.dist_mismatch.load(Relaxed),
        if e.is_empty() { "none".to_string() } else { e.join(" | ") }
    );
    eprint!("{rep}");
    std::fs::write(format!("{out}.report.txt"), &rep).map_err(|e| e.to_string())?;
    if !e.is_empty() {
        return Err("stopped on errors (see report)".into());
    }
    Ok(())
}

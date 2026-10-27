//! TitaniumTest — Distributed & Multi-Threaded SPRT Gate + Teacher Relabeling System (Fishtest / Fishnet style).
//!
//! Subcommands:
//!   titaniumtest sprt --candidate <CMD|self> --baseline <CMD> --book <FILE>
//!                     [--time-ms 100 | --nodes 5000] [--elo0 0.0 --elo1 5.0]
//!                     [--max-games 200] [--threads 4] [--out FILE]
//!
//!   titaniumtest coordinator [--bind 0.0.0.0:8080]
//!                            [--relabel-bin FILE --relabel-out DIR --relabel-nodes 5000]
//!                            [--sprt-candidate CMD --sprt-baseline CMD --sprt-book FILE --sprt-time-ms 100]
//!
//!   titaniumtest worker --server http://127.0.0.1:8080 [--threads 4] [--worker-id NAME]

mod sprt;
mod train;

use sprt::{SprtTally, SprtVerdict};
use std::collections::HashMap;
use std::fs;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::Path;
use std::process::{Child, ChildStdin, ChildStdout, Command, ExitCode, Stdio};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use titanium::nnue::Nnue;
use titanium::{Board, Move, SearchLimits, Searcher};

const RELABEL_CHUNK_SIZE: usize = 5000;
const LEASE_TTL_SECS: u64 = 300;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let sub = args.first().map(String::as_str).unwrap_or("help");
    match sub {
        "sprt" => run_sprt_cli(&args[1..]),
        "train" => run_train_subcmd(&args[1..]),
        "coordinator" => run_coordinator(&args[1..]),
        "worker" => run_worker(&args[1..]),
        _ => {
            print_help();
            ExitCode::SUCCESS
        }
    }
}

fn print_help() {
    println!(
        "titaniumtest — Fishtest/Fishnet/Bullet-style SPRT, Relabeling & NNUE Trainer for Titanium Ataxx\n\
         \n\
         SUBCOMMANDS:\n\
         \x20 sprt         Run a multi-threaded Pentanomial SPRT gate locally\n\
         \x20 train        Bullet-style native Rust sparse-feature NNUE trainer (+ optional --auto-sprt)\n\
         \x20 coordinator  Start the HTTP job coordinator + live web dashboard (:8080)\n\
         \x20 worker       Connect to a coordinator and execute SPRT / relabel jobs\n\
         \n\
         EXAMPLES:\n\
         \x20 titaniumtest sprt --candidate self --baseline \"./old_titanium serve --network bundled\" \\\n\
         \x20                   --book data/nnue/book100.txt --time-ms 100 --elo0 0 --elo1 10 --hyperthread\n\
         \x20 titaniumtest train --init bundled --bin dis_positions.bin --labels-dir out_dir --out cand.tnn --auto-sprt\n\
         \x20 titaniumtest coordinator --bind 0.0.0.0:8080 --relabel-bin dis_positions.bin --relabel-out out_dir\n\
         \x20 titaniumtest worker --server http://127.0.0.1:8080 --hyperthread"
    );
}

fn run_train_subcmd(args: &[String]) -> ExitCode {
    let init_spec = flag_str(args, "--init").unwrap_or_else(|| "bundled".to_string());
    let Some(bin_path) = flag_str(args, "--bin") else {
        eprintln!("train: --bin <dis_positions.bin> is required");
        return ExitCode::FAILURE;
    };
    let Some(labels_dir) = flag_str(args, "--labels-dir") else {
        eprintln!("train: --labels-dir <out_dir> is required");
        return ExitCode::FAILURE;
    };
    let out_path = flag_str(args, "--out").unwrap_or_else(|| "nnue_candidate.tnn".to_string());
    let epochs = flag_u64(args, "--epochs").unwrap_or(3).max(1) as usize;
    let batch_size = flag_u64(args, "--batch-size").unwrap_or(1024).max(1) as usize;
    let lr = flag_f64(args, "--lr").unwrap_or(0.0002) as f32;
    let wd = flag_f64(args, "--wd").unwrap_or(0.0001) as f32;
    let max_chunks = flag_u64(args, "--max-chunks").map(|v| v as usize);
    let threads = resolve_threads(args);

    let code = train::run_train_cli(
        &init_spec,
        &bin_path,
        &labels_dir,
        &out_path,
        epochs,
        batch_size,
        lr,
        wd,
        threads,
        max_chunks,
    );
    if code != ExitCode::SUCCESS {
        return code;
    }
    if has_flag(args, "--auto-sprt") {
        let book = flag_str(args, "--book")
            .unwrap_or_else(|| "C:\\gitProjects\\titanium-ataxx\\data\\nnue\\book100.txt".to_string());
        let sprt_args = vec![
            "--candidate".to_string(),
            out_path,
            "--baseline".to_string(),
            "self".to_string(),
            "--book".to_string(),
            book,
            "--threads".to_string(),
            threads.to_string(),
            "--time-ms".to_string(),
            flag_u64(args, "--time-ms").unwrap_or(50).to_string(),
            "--max-games".to_string(),
            flag_u64(args, "--max-games").unwrap_or(100).to_string(),
            "--elo0".to_string(),
            "0.0".to_string(),
            "--elo1".to_string(),
            "5.0".to_string(),
        ];
        return run_sprt_cli(&sprt_args);
    }
    ExitCode::SUCCESS
}

fn flag_str(args: &[String], name: &str) -> Option<String> {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1))
        .cloned()
}

fn flag_u64(args: &[String], name: &str) -> Option<u64> {
    flag_str(args, name).and_then(|v| v.parse().ok())
}

fn flag_f64(args: &[String], name: &str) -> Option<f64> {
    flag_str(args, name).and_then(|v| v.parse().ok())
}

fn has_flag(args: &[String], name: &str) -> bool {
    args.iter().any(|a| a == name)
}

/// Automatically hill-climbs thread count from physical cores (`logical / 2`) up to
/// `logical` hyperthreads, stopping as soon as throughput hits diminishing returns (< 2% gain)
/// or regresses due to SMT / L3 cache contention.
fn autotune_threads() -> usize {
    let logical = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
        .max(1);
    if logical <= 2 {
        return logical;
    }
    let phys = (logical / 2).max(1);
    let step = if logical - phys >= 4 { 2 } else { 1 };
    let mut candidates: Vec<usize> = (phys..=logical).step_by(step).collect();
    if *candidates.last().unwrap_or(&0) != logical {
        candidates.push(logical);
    }

    println!(
        "[Autotune] Probing hyperthread scaling across {:?} threads (1 thread/search, parallel queues)...",
        candidates
    );
    let nn = Nnue::bundled_static();
    let probe_board = Board::start();
    let mut best_threads = phys;
    let mut best_nps = 0.0f64;

    for &t in &candidates {
        let stop = AtomicBool::new(false);
        let total_nodes = std::sync::atomic::AtomicU64::new(0);
        let t0 = Instant::now();
        std::thread::scope(|sc| {
            for _ in 0..t {
                let stop = &stop;
                let total_nodes = &total_nodes;
                sc.spawn(move || {
                    let mut searcher = Searcher::with_tt_bits_unconfigured(18);
                    searcher.set_nnue(Some(nn));
                    let limits = SearchLimits {
                        time: None,
                        max_nodes: Some(2500),
                        max_depth: u8::MAX,
                    };
                    while !stop.load(Ordering::Relaxed) {
                        searcher.reset_for_new_game();
                        let r = searcher.search(&probe_board, &limits);
                        total_nodes.fetch_add(r.nodes, Ordering::Relaxed);
                    }
                });
            }
            std::thread::sleep(Duration::from_millis(280));
            stop.store(true, Ordering::Relaxed);
        });
        let elapsed = t0.elapsed().as_secs_f64().max(1e-6);
        let nodes = total_nodes.load(Ordering::Relaxed);
        let mnps = (nodes as f64 / elapsed) / 1_000_000.0;
        let gain_pct = if best_nps > 0.0 {
            (mnps - best_nps) * 100.0 / best_nps
        } else {
            100.0
        };
        println!("  -> {t:>2} threads: {mnps:>6.2} Mnps ({gain_pct:+.1}%)");

        if best_nps == 0.0 {
            best_nps = mnps;
            best_threads = t;
        } else if gain_pct >= 2.0 {
            best_nps = mnps;
            best_threads = t;
        } else {
            println!(
                "[Autotune] Diminishing returns / contention at {t} threads ({gain_pct:+.1}%); locking best={best_threads} threads ({best_nps:.2} Mnps)."
            );
            return best_threads;
        }
    }
    println!(
        "[Autotune] Locked optimal thread count: {best_threads} threads ({best_nps:.2} Mnps)."
    );
    best_threads
}

/// Resolves worker concurrency:
/// - Explicit `--threads N` overrides everything.
/// - `--hyperthread` / `--autotune` runs the hill-climbing throughput probe into the SMT range.
/// - Otherwise defaults safely to physical cores (`(logical / 2).max(1)`) so hyperthreading is never
///   used unless explicitly requested.
fn resolve_threads(args: &[String]) -> usize {
    if let Some(t) = flag_u64(args, "--threads") {
        return t.max(1) as usize;
    }
    if has_flag(args, "--hyperthread") || has_flag(args, "--autotune") {
        return autotune_threads();
    }
    let logical = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
        .max(1);
    (logical / 2).max(1)
}

// ============================================================================
// UAI External Engine Process Driver
// ============================================================================

struct UaiEngine {
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
}

impl UaiEngine {
    fn spawn(cmd_line: &str) -> Result<Self, String> {
        let parts: Vec<&str> = cmd_line.split_whitespace().collect();
        assert!(!parts.is_empty(), "UaiEngine::spawn: empty command");
        let mut command = Command::new(parts[0]);
        for arg in &parts[1..] {
            command.arg(arg);
        }
        let mut child = command
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .map_err(|e| format!("spawn '{cmd_line}' failed: {e}"))?;
        let stdin = child.stdin.take().ok_or("missing stdin")?;
        let stdout = BufReader::new(child.stdout.take().ok_or("missing stdout")?);
        let mut eng = UaiEngine {
            child,
            stdin,
            stdout,
        };
        eng.send("uai")?;
        eng.wait_for("uaiok")?;
        Ok(eng)
    }

    fn send(&mut self, line: &str) -> Result<(), String> {
        writeln!(self.stdin, "{line}").map_err(|e| format!("write '{line}': {e}"))?;
        self.stdin
            .flush()
            .map_err(|e| format!("flush '{line}': {e}"))
    }

    fn wait_for(&mut self, prefix: &str) -> Result<String, String> {
        let mut buf = String::new();
        loop {
            buf.clear();
            let n = self
                .stdout
                .read_line(&mut buf)
                .map_err(|e| format!("read_line waiting for {prefix}: {e}"))?;
            if n == 0 {
                return Err(format!("EOF waiting for '{prefix}'"));
            }
            let trimmed = buf.trim();
            if trimmed.starts_with(prefix) {
                return Ok(trimmed.to_string());
            }
        }
    }

    fn new_game(&mut self) -> Result<(), String> {
        self.send("uainewgame")?;
        self.send("isready")?;
        self.wait_for("readyok")?;
        Ok(())
    }

    fn search_move(
        &mut self,
        b: &Board,
        time_ms: Option<u64>,
        nodes: Option<u64>,
    ) -> Result<Option<Move>, String> {
        self.send(&format!("position fen {}", b.to_ataxx_fen()))?;
        let go_cmd = match (time_ms, nodes) {
            (Some(ms), _) => format!("go movetime {ms}"),
            (None, Some(n)) => format!("go nodes {n}"),
            (None, None) => "go movetime 100".to_string(),
        };
        self.send(&go_cmd)?;
        let line = self.wait_for("bestmove ")?;
        let tok = line.split_whitespace().nth(1).unwrap_or("pass");
        Ok(parse_uai_move(b, tok))
    }
}

impl Drop for UaiEngine {
    fn drop(&mut self) {
        let _ = writeln!(self.stdin, "quit");
        let _ = self.stdin.flush();
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn parse_uai_move(b: &Board, text: &str) -> Option<Move> {
    let token = text.trim();
    if token == "pass" || token == "0000" || token == "none" {
        return None;
    }
    let clean: String = token.chars().filter(|&c| c != '-').collect();
    if clean.len() == 2 {
        let to = Board::parse_sq(&clean)?;
        let legal = b.legal_moves();
        let found = legal.iter().find(|cand| cand.is_clone() && cand.to == to);
        return found;
    }
    if clean.len() == 4 {
        let from = Board::parse_sq(&clean[0..2])?;
        let to = Board::parse_sq(&clean[2..4])?;
        let m = Move { from, to };
        let legal = b.legal_moves();
        let found = legal
            .iter()
            .find(|cand| cand.to == m.to && (cand.is_clone() == m.is_clone() && (cand.is_clone() || cand.from == m.from)));
        return found;
    }
    None
}

enum PlayerSlot {
    InProcess(Searcher<'static>),
    External(UaiEngine),
}

impl PlayerSlot {
    fn create(spec: &str) -> Result<Self, String> {
        if spec == "self" {
            Ok(PlayerSlot::InProcess(Searcher::with_main_network()))
        } else if spec.ends_with(".tnn") || spec.starts_with("tnn:") {
            let path = spec.strip_prefix("tnn:").unwrap_or(spec);
            let loaded = Nnue::load(path).map_err(|e| format!("load tnn '{path}': {e}"))?;
            let leaked: &'static Nnue = Box::leak(Box::new(loaded));
            let mut s = Searcher::new_unconfigured();
            s.set_nnue(Some(leaked));
            Ok(PlayerSlot::InProcess(s))
        } else {
            Ok(PlayerSlot::External(UaiEngine::spawn(spec)?))
        }
    }

    fn reset(&mut self) -> Result<(), String> {
        match self {
            PlayerSlot::InProcess(s) => {
                s.reset_for_new_game();
                Ok(())
            }
            PlayerSlot::External(e) => e.new_game(),
        }
    }

    fn best_move(
        &mut self,
        b: &Board,
        time_ms: Option<u64>,
        nodes: Option<u64>,
    ) -> Result<Option<Move>, String> {
        match self {
            PlayerSlot::InProcess(s) => {
                let limits = SearchLimits {
                    time: time_ms.map(Duration::from_millis),
                    max_nodes: nodes,
                    max_depth: u8::MAX,
                };
                Ok(s.search(b, &limits).best)
            }
            PlayerSlot::External(e) => e.search_move(b, time_ms, nodes),
        }
    }
}

/// Play one single game from `start_board`. Returns candidate points * 2 (`2` = win, `1` = draw, `0` = loss).
fn play_single_game(
    start_board: &Board,
    candidate_is_black: bool,
    cand: &mut PlayerSlot,
    base: &mut PlayerSlot,
    time_ms: Option<u64>,
    nodes: Option<u64>,
) -> Result<(u8, u32, u32, u16), String> {
    cand.reset()?;
    base.reset()?;
    let mut b = *start_board;
    let mut forfeit_winner: Option<u8> = None;

    while !b.game_over() && b.ply < 300 {
        let side = b.turn;
        if !b.has_moves(side) {
            b = b.make_pass();
            continue;
        }
        let cand_turn = (side == 0) == candidate_is_black;
        let mv = if cand_turn {
            cand.best_move(&b, time_ms, nodes)?
        } else {
            base.best_move(&b, time_ms, nodes)?
        };
        match mv {
            Some(m) => {
                b = b.make(m);
            }
            None => {
                forfeit_winner = Some(1 - side);
                break;
            }
        }
    }

    let winner = forfeit_winner.unwrap_or_else(|| {
        if b.ply >= 300 && !b.game_over() {
            2
        } else {
            b.winner()
        }
    });
    let (bl, wh) = b.counts();
    let pts_x2 = if winner == 2 {
        1
    } else if (winner == 0 && candidate_is_black) || (winner == 1 && !candidate_is_black) {
        2
    } else {
        0
    };
    Ok((pts_x2, bl, wh, b.ply as u16))
}

// ============================================================================
// 1. Multi-Threaded Local SPRT Gate Command
// ============================================================================

fn run_sprt_cli(args: &[String]) -> ExitCode {
    let cand_spec = flag_str(args, "--candidate").unwrap_or_else(|| "self".to_string());
    let Some(base_spec) = flag_str(args, "--baseline") else {
        eprintln!("sprt: --baseline <CMD|self> is required");
        return ExitCode::FAILURE;
    };
    let book_path = flag_str(args, "--book");
    let time_ms = flag_u64(args, "--time-ms");
    let nodes = flag_u64(args, "--nodes");
    let time_ms = if time_ms.is_none() && nodes.is_none() {
        Some(100)
    } else {
        time_ms
    };
    let elo0 = flag_f64(args, "--elo0").unwrap_or(0.0);
    let elo1 = flag_f64(args, "--elo1").unwrap_or(10.0);
    let max_games = flag_u64(args, "--max-games").unwrap_or(200) as usize;
    let threads = resolve_threads(args);
    let out_path = flag_str(args, "--out");

    let book_boards: Vec<Board> = match &book_path {
        Some(p) => {
            let text = fs::read_to_string(p).unwrap_or_else(|e| panic!("sprt: cannot read book {p}: {e}"));
            let bs: Vec<Board> = text
                .lines()
                .map(str::trim)
                .filter(|l| !l.is_empty())
                .map(|l| Board::from_str(l).unwrap_or_else(|| panic!("sprt: bad book line: {l}")))
                .collect();
            assert!(!bs.is_empty(), "sprt: book is empty");
            bs
        }
        None => vec![Board::start()],
    };

    let max_pairs = (max_games + 1) / 2;
    println!(
        "=== TitaniumTest SPRT Gate ===\n\
         Candidate : {cand_spec}\n\
         Baseline  : {base_spec}\n\
         Limits    : time_ms={time_ms:?}, nodes={nodes:?}, threads={threads}\n\
         SPRT      : elo0={elo0:.1}, elo1={elo1:.1} (alpha=0.05, beta=0.05, bounds [-2.94, +2.94])\n\
         Book      : {} positions (max {max_pairs} pairs = {} games)\n",
        book_boards.len(),
        max_pairs * 2
    );

    let next_pair = AtomicUsize::new(0);
    let stop_flag = AtomicBool::new(false);
    let tally = Mutex::new(SprtTally::default());
    let t0 = Instant::now();

    std::thread::scope(|sc| {
        for tid in 0..threads {
            let cand_spec = &cand_spec;
            let base_spec = &base_spec;
            let book_boards = &book_boards;
            let next_pair = &next_pair;
            let stop_flag = &stop_flag;
            let tally = &tally;
            let out_path = &out_path;
            sc.spawn(move || {
                let mut cand = PlayerSlot::create(cand_spec)
                    .unwrap_or_else(|e| panic!("worker {tid}: candidate init failed: {e}"));
                let mut base = PlayerSlot::create(base_spec)
                    .unwrap_or_else(|e| panic!("worker {tid}: baseline init failed: {e}"));
                loop {
                    if stop_flag.load(Ordering::Relaxed) {
                        break;
                    }
                    let pair_idx = next_pair.fetch_add(1, Ordering::Relaxed);
                    if pair_idx >= max_pairs {
                        break;
                    }
                    let board = &book_boards[pair_idx % book_boards.len()];
                    let (g1, b1, w1, p1) = play_single_game(board, true, &mut cand, &mut base, time_ms, nodes)
                        .unwrap_or_else(|e| panic!("worker {tid} pair {pair_idx} game 1 failed: {e}"));
                    if stop_flag.load(Ordering::Relaxed) {
                        break;
                    }
                    let (g2, b2, w2, p2) = play_single_game(board, false, &mut cand, &mut base, time_ms, nodes)
                        .unwrap_or_else(|e| panic!("worker {tid} pair {pair_idx} game 2 failed: {e}"));

                    let mut guard = tally.lock().unwrap();
                    guard.record_pair(g1, g2);
                    let st = guard.compute(elo0, elo1, 0.05, 0.05);
                    let line = format!(
                        "[Pair {:>3} | {:>3}g | {:>5.1}s] +{:>2} -{:>2} ={:>2} ({:>5.1}%) | Elo {:>+6.1} +- {:>5.1} | LLR {:>+6.2} [{:.2}, {:.2}] | Penta {:?} | LOS {:>5.1}% (g1 {b1}-{w1}/{p1}p, g2 {b2}-{w2}/{p2}p)",
                        st.pairs,
                        st.games,
                        t0.elapsed().as_secs_f64(),
                        guard.wins,
                        guard.losses,
                        guard.draws,
                        st.score_rate * 100.0,
                        st.elo,
                        st.elo_ci95,
                        st.llr,
                        st.lower_bound,
                        st.upper_bound,
                        guard.penta,
                        st.los * 100.0
                    );
                    println!("{line}");
                    if let Some(p) = out_path {
                        if let Ok(mut f) = fs::OpenOptions::new().create(true).append(true).open(p) {
                            let _ = writeln!(f, "{line}");
                        }
                    }
                    if st.verdict != SprtVerdict::Continue {
                        println!("\n>>> SPRT TERMINATED: {:?} <<<", st.verdict);
                        stop_flag.store(true, Ordering::Relaxed);
                        break;
                    }
                }
            });
        }
    });

    let final_tally = tally.lock().unwrap();
    let st = final_tally.compute(elo0, elo1, 0.05, 0.05);
    println!(
        "\n=== Final SPRT Summary ({:.1}s) ===\n\
         Games : {} (+{} -{} ={}, {:.1}%)\n\
         Penta : LL={} LD={} DD+WL={} WD={} WW={}\n\
         Elo   : {:+.1} +- {:.1} (95% CI), LOS {:.1}%\n\
         LLR   : {:+.2} [{:.2}, {:.2}] -> {:?}",
        t0.elapsed().as_secs_f64(),
        st.games,
        final_tally.wins,
        final_tally.losses,
        final_tally.draws,
        st.score_rate * 100.0,
        final_tally.penta[0],
        final_tally.penta[1],
        final_tally.penta[2],
        final_tally.penta[3],
        final_tally.penta[4],
        st.elo,
        st.elo_ci95,
        st.los * 100.0,
        st.llr,
        st.lower_bound,
        st.upper_bound,
        st.verdict
    );
    ExitCode::SUCCESS
}

// ============================================================================
// 2. HTTP Job Coordinator & Live Dashboard (SPRT Preemption + Relabel Queue)
// ============================================================================

struct ActiveSprtJob {
    candidate_spec: String,
    baseline_spec: String,
    book_lines: Vec<String>,
    time_ms: u64,
    nodes: u64,
    max_pairs: usize,
    elo0: f64,
    elo1: f64,
    next_pair: usize,
    leased_pairs: HashMap<usize, (String, u64)>,
    tally: SprtTally,
    finished: bool,
}

struct CoordinatorState {
    sprt_job: Option<ActiveSprtJob>,
    relabel_bin_path: Option<String>,
    relabel_out_dir: Option<String>,
    relabel_nodes: u64,
    total_positions: usize,
    total_chunks: usize,
    completed_chunks: Vec<bool>,
    leased_chunks: HashMap<usize, (String, u64)>, // chunk_id -> (worker_id, timestamp_secs)
    workers_seen: HashMap<String, u64>,          // worker_id -> last_seen_secs
    started_at: Instant,
    session_chunks_done: usize,
}

fn now_epoch_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn run_coordinator(args: &[String]) -> ExitCode {
    let bind_addr = flag_str(args, "--bind").unwrap_or_else(|| "0.0.0.0:8080".to_string());
    let relabel_bin = flag_str(args, "--relabel-bin");
    let relabel_out = flag_str(args, "--relabel-out");
    let relabel_nodes = flag_u64(args, "--relabel-nodes").unwrap_or(5000);

    let sprt_cand = flag_str(args, "--candidate");
    let sprt_base = flag_str(args, "--baseline");
    let sprt_book = flag_str(args, "--book");
    let sprt_time = flag_u64(args, "--time-ms").unwrap_or(100);
    let sprt_nodes = flag_u64(args, "--nodes").unwrap_or(0);
    let sprt_max_games = flag_u64(args, "--max-games").unwrap_or(400) as usize;
    let sprt_elo0 = flag_f64(args, "--elo0").unwrap_or(0.0);
    let sprt_elo1 = flag_f64(args, "--elo1").unwrap_or(5.0);

    let sprt_job = match (sprt_cand, sprt_base) {
        (Some(cand), Some(base)) => {
            let book_lines = match sprt_book {
                Some(p) => fs::read_to_string(&p)
                    .unwrap_or_else(|e| panic!("coordinator: failed to read book {p}: {e}"))
                    .lines()
                    .map(|l| l.trim().to_string())
                    .filter(|l| !l.is_empty())
                    .collect(),
                None => vec![Board::start().to_string()],
            };
            Some(ActiveSprtJob {
                candidate_spec: cand,
                baseline_spec: base,
                book_lines,
                time_ms: sprt_time,
                nodes: sprt_nodes,
                max_pairs: (sprt_max_games + 1) / 2,
                elo0: sprt_elo0,
                elo1: sprt_elo1,
                next_pair: 0,
                leased_pairs: HashMap::new(),
                tally: SprtTally::default(),
                finished: false,
            })
        }
        _ => None,
    };

    let (total_positions, total_chunks, completed_chunks) = match (&relabel_bin, &relabel_out) {
        (Some(bin_p), Some(out_d)) => {
            fs::create_dir_all(out_d).expect("coordinator: create relabel-out dir");
            let meta = fs::metadata(bin_p).unwrap_or_else(|e| panic!("coordinator: {bin_p}: {e}"));
            let n_pos = (meta.len() as usize) / 25;
            let n_chunks = (n_pos + RELABEL_CHUNK_SIZE - 1) / RELABEL_CHUNK_SIZE;
            let mut done = vec![false; n_chunks];
            for cid in 0..n_chunks {
                let p_c = format!("{out_d}/c_{cid}.txt");
                let p_chunk = format!("{out_d}/chunk_{cid:05}.txt");
                if Path::new(&p_c).exists() || Path::new(&p_chunk).exists() {
                    done[cid] = true;
                }
            }
            (n_pos, n_chunks, done)
        }
        _ => (0, 0, Vec::new()),
    };

    let done_init = completed_chunks.iter().filter(|&&b| b).count();
    println!(
        "=== TitaniumTest Coordinator ===\n\
         Listening : http://{bind_addr}\n\
         SPRT Job  : {}\n\
         Relabel   : {done_init}/{total_chunks} chunks done ({total_positions} positions, {relabel_nodes} nodes/pos)",
        if sprt_job.is_some() { "ACTIVE (preempts relabel)" } else { "NONE" }
    );

    let state = Arc::new(Mutex::new(CoordinatorState {
        sprt_job,
        relabel_bin_path: relabel_bin,
        relabel_out_dir: relabel_out,
        relabel_nodes,
        total_positions,
        total_chunks,
        completed_chunks,
        leased_chunks: HashMap::new(),
        workers_seen: HashMap::new(),
        started_at: Instant::now(),
        session_chunks_done: 0,
    }));

    let listener = TcpListener::bind(&bind_addr)
        .unwrap_or_else(|e| panic!("coordinator: failed to bind {bind_addr}: {e}"));

    for stream in listener.incoming().flatten() {
        let st = Arc::clone(&state);
        std::thread::spawn(move || {
            handle_http_client(stream, st);
        });
    }
    ExitCode::SUCCESS
}

fn handle_http_client(mut stream: TcpStream, state: Arc<Mutex<CoordinatorState>>) {
    let _ = stream.set_read_timeout(Some(Duration::from_secs(30)));
    let mut reader = BufReader::new(match stream.try_clone() {
        Ok(s) => s,
        Err(_) => return,
    });
    let mut req_line = String::new();
    if reader.read_line(&mut req_line).unwrap_or(0) == 0 {
        return;
    }
    let parts: Vec<&str> = req_line.split_whitespace().collect();
    if parts.len() < 2 {
        return;
    }
    let (method, path) = (parts[0], parts[1]);

    let mut content_len = 0usize;
    loop {
        let mut h = String::new();
        if reader.read_line(&mut h).unwrap_or(0) == 0 || h == "\r\n" || h == "\n" {
            break;
        }
        let lower = h.to_ascii_lowercase();
        if let Some(rest) = lower.strip_prefix("content-length:") {
            content_len = rest.trim().parse().unwrap_or(0);
        }
    }

    let mut body = vec![0u8; content_len];
    if content_len > 0 && reader.read_exact(&mut body).is_err() {
        return;
    }

    match (method, path) {
        ("GET", "/") => {
            let st = state.lock().unwrap();
            let done = st.completed_chunks.iter().filter(|&&b| b).count();
            let pct = if st.total_chunks > 0 {
                done as f64 * 100.0 / st.total_chunks as f64
            } else {
                0.0
            };
            let sprt_html = if let Some(job) = &st.sprt_job {
                let s = job.tally.compute(job.elo0, job.elo1, 0.05, 0.05);
                format!(
                    "<div class='card'><h2>Active SPRT Gate ({})</h2>\
                     <p>Candidate: <code>{}</code> vs Baseline: <code>{}</code></p>\
                     <p>Games: <b>{} / {}</b> (+{} -{} ={}, {:.1}%)</p>\
                     <p>Elo: <b>{:+.1} +- {:.1}</b> | LOS: <b>{:.1}%</b> | Penta: <code>{:?}</code></p>\
                     <p>LLR: <b>{:+.2}</b> [{:.2}, {:.2}] &rarr; <b>{:?}</b></p></div>",
                    if job.finished { "FINISHED" } else { "RUNNING - HIGH PRIORITY" },
                    job.candidate_spec,
                    job.baseline_spec,
                    s.games,
                    job.max_pairs * 2,
                    job.tally.wins,
                    job.tally.losses,
                    job.tally.draws,
                    s.score_rate * 100.0,
                    s.elo,
                    s.elo_ci95,
                    s.los * 100.0,
                    job.tally.penta,
                    s.llr,
                    s.lower_bound,
                    s.upper_bound,
                    s.verdict
                )
            } else {
                "<div class='card'><h2>SPRT Gate</h2><p>No active SPRT gate.</p></div>".to_string()
            };
            let html = format!(
                "<!doctype html><html><head><meta charset='utf-8'><meta http-equiv='refresh' content='5'>\
                 <title>TitaniumTest Coordinator</title>\
                 <style>body{{background:#0d1117;color:#e6edf3;font-family:monospace;padding:24px;}}\
                 .card{{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:16px;margin-bottom:16px;}}\
                 h1{{color:#58a6ff;margin-top:0;}}</style></head><body>\
                 <h1>TitaniumTest Compute Network</h1>\
                 {sprt_html}\
                 <div class='card'><h2>Relabel Queue</h2>\
                 <p>Progress: <b>{done} / {}</b> chunks ({pct:.2}%)</p>\
                 <p>In-flight leases: <b>{}</b> | Active workers: <b>{}</b> | Session completed: <b>{}</b></p>\
                 <p>Uptime: {:.0}s</p></div></body></html>",
                st.total_chunks,
                st.leased_chunks.len(),
                st.workers_seen.len(),
                st.session_chunks_done,
                st.started_at.elapsed().as_secs_f64()
            );
            send_http(&mut stream, 200, "text/html; charset=utf-8", html.as_bytes());
        }
        ("POST", "/api/lease") => {
            let worker_id = String::from_utf8_lossy(&body).trim().to_string();
            let now = now_epoch_secs();
            let mut st = state.lock().unwrap();
            st.workers_seen.insert(worker_id.clone(), now);

            // Priority 1: Active SPRT Gate Job (preempts background relabeling)
            if let Some(job) = &mut st.sprt_job {
                job.leased_pairs
                    .retain(|_, (_, ts)| now.saturating_sub(*ts) < LEASE_TTL_SECS);
                if !job.finished && job.next_pair < job.max_pairs {
                    let pid = job.next_pair;
                    job.next_pair += 1;
                    job.leased_pairs.insert(pid, (worker_id, now));
                    let board_line = job.book_lines[pid % job.book_lines.len()].clone();
                    let payload = format!(
                        "SPRT {pid} {} {}\n{}\n{}\n{}\n",
                        job.time_ms, job.nodes, job.candidate_spec, job.baseline_spec, board_line
                    );
                    send_http(&mut stream, 200, "text/plain", payload.as_bytes());
                    return;
                }
            }

            // Priority 2: Background Relabel Queue
            st.leased_chunks
                .retain(|_, (_, ts)| now.saturating_sub(*ts) < LEASE_TTL_SECS);

            let mut chosen: Option<usize> = None;
            for cid in 0..st.total_chunks {
                if !st.completed_chunks[cid] && !st.leased_chunks.contains_key(&cid) {
                    chosen = Some(cid);
                    break;
                }
            }
            let Some(cid) = chosen else {
                send_http(&mut stream, 200, "text/plain", b"IDLE");
                return;
            };
            let Some(bin_path) = st.relabel_bin_path.clone() else {
                send_http(&mut stream, 200, "text/plain", b"IDLE");
                return;
            };
            let lo = cid * RELABEL_CHUNK_SIZE;
            let hi = ((cid + 1) * RELABEL_CHUNK_SIZE).min(st.total_positions);
            let count = hi.saturating_sub(lo);
            let nodes = st.relabel_nodes;
            st.leased_chunks.insert(cid, (worker_id, now));
            drop(st);

            let mut f = fs::File::open(&bin_path).expect("open relabel bin");
            use std::io::{Seek, SeekFrom};
            f.seek(SeekFrom::Start((lo * 25) as u64)).expect("seek");
            let mut slice = vec![0u8; count * 25];
            f.read_exact(&mut slice).expect("read_exact");

            let header = format!("RELABEL {cid} {count} {nodes}\n");
            let mut payload = Vec::with_capacity(header.len() + slice.len());
            payload.extend_from_slice(header.as_bytes());
            payload.extend_from_slice(&slice);
            send_http(&mut stream, 200, "application/octet-stream", &payload);
        }
        ("POST", "/api/submit_sprt") => {
            let text = String::from_utf8_lossy(&body);
            let parts: Vec<&str> = text.split_whitespace().collect();
            if parts.len() < 4 || parts[0] != "PAIR" {
                send_http(&mut stream, 400, "text/plain", b"ERR bad sprt report");
                return;
            }
            let pid: usize = parts[1].parse().unwrap_or(usize::MAX);
            let g1: u8 = parts[2].parse().unwrap_or(1);
            let g2: u8 = parts[3].parse().unwrap_or(1);
            let mut st = state.lock().unwrap();
            if let Some(job) = &mut st.sprt_job {
                job.leased_pairs.remove(&pid);
                if !job.finished {
                    job.tally.record_pair(g1, g2);
                    let status = job.tally.compute(job.elo0, job.elo1, 0.05, 0.05);
                    println!(
                        "[SPRT Pair {:>3}] +{} -{} ={} ({:.1}%) | Elo {:+.1} +- {:.1} | LLR {:+.2} [{:.2}, {:.2}] | Penta {:?} -> {:?}",
                        status.pairs,
                        job.tally.wins,
                        job.tally.losses,
                        job.tally.draws,
                        status.score_rate * 100.0,
                        status.elo,
                        status.elo_ci95,
                        status.llr,
                        status.lower_bound,
                        status.upper_bound,
                        job.tally.penta,
                        status.verdict
                    );
                    if status.verdict != SprtVerdict::Continue || (status.pairs as usize) >= job.max_pairs {
                        job.finished = true;
                        println!(">>> COORDINATOR SPRT FINISHED: {:?} (resuming relabel queue) <<<", status.verdict);
                    }
                }
            }
            send_http(&mut stream, 200, "text/plain", b"OK");
        }
        ("POST", "/api/submit_relabel") => {
            let text = String::from_utf8_lossy(&body);
            let mut lines = text.lines();
            let Some(first) = lines.next() else {
                send_http(&mut stream, 400, "text/plain", b"ERR missing header");
                return;
            };
            let hdr: Vec<&str> = first.split_whitespace().collect();
            if hdr.len() < 3 || hdr[0] != "CHUNK" {
                send_http(&mut stream, 400, "text/plain", b"ERR bad header");
                return;
            }
            let cid: usize = hdr[1].parse().unwrap_or(usize::MAX);
            let expected_rows: usize = hdr[2].parse().unwrap_or(0);
            let rows: Vec<&str> = lines.filter(|l| !l.trim().is_empty()).collect();
            if rows.len() != expected_rows {
                send_http(&mut stream, 400, "text/plain", b"ERR row count mismatch");
                return;
            }
            let mut st = state.lock().unwrap();
            if cid >= st.total_chunks {
                send_http(&mut stream, 400, "text/plain", b"ERR bad cid");
                return;
            }
            if let Some(out_d) = &st.relabel_out_dir {
                let tmp = format!("{out_d}/c_{cid}.tmp");
                let final_p = format!("{out_d}/c_{cid}.txt");
                let mut out_str = String::with_capacity(rows.len() * 16);
                for r in &rows {
                    out_str.push_str(r);
                    out_str.push('\n');
                }
                fs::write(&tmp, out_str).expect("write tmp");
                fs::rename(&tmp, &final_p).expect("rename final");
            }
            st.completed_chunks[cid] = true;
            st.leased_chunks.remove(&cid);
            st.session_chunks_done += 1;
            send_http(&mut stream, 200, "text/plain", b"OK");
        }
        _ => {
            send_http(&mut stream, 404, "text/plain", b"Not Found");
        }
    }
}

fn send_http(stream: &mut TcpStream, status: u16, content_type: &str, body: &[u8]) {
    let status_text = if status == 200 { "OK" } else { "Error" };
    let hdr = format!(
        "HTTP/1.1 {status} {status_text}\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        body.len()
    );
    let _ = stream.write_all(hdr.as_bytes());
    let _ = stream.write_all(body);
    let _ = stream.flush();
}

// ============================================================================
// 3. Stateless Compute Worker (Fishnet-style)
// ============================================================================

fn parse_host_port(server_url: &str) -> String {
    let stripped = server_url
        .strip_prefix("http://")
        .or_else(|| server_url.strip_prefix("https://"))
        .unwrap_or(server_url);
    stripped.trim_end_matches('/').to_string()
}

fn http_post(host_port: &str, path: &str, body: &[u8]) -> Result<Vec<u8>, String> {
    let mut stream = TcpStream::connect(host_port).map_err(|e| format!("connect {host_port}: {e}"))?;
    let _ = stream.set_read_timeout(Some(Duration::from_secs(60)));
    let req = format!(
        "POST {path} HTTP/1.1\r\nHost: {host_port}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        body.len()
    );
    stream.write_all(req.as_bytes()).map_err(|e| e.to_string())?;
    stream.write_all(body).map_err(|e| e.to_string())?;
    stream.flush().map_err(|e| e.to_string())?;

    let mut resp = Vec::new();
    stream.read_to_end(&mut resp).map_err(|e| e.to_string())?;
    let sep = resp
        .windows(4)
        .position(|w| w == b"\r\n\r\n")
        .ok_or("missing HTTP header separator")?;
    Ok(resp[sep + 4..].to_vec())
}

fn execute_sprt_pair_payload(
    worker_id: &str,
    host_port: &str,
    hdr: &[&str],
    body_after_nl: &[u8],
) {
    let pid: usize = hdr[1].parse().expect("pid");
    let time_ms_val: u64 = hdr[2].parse().expect("time_ms");
    let nodes_val: u64 = hdr[3].parse().expect("nodes");
    let time_ms = if time_ms_val > 0 { Some(time_ms_val) } else { None };
    let nodes = if nodes_val > 0 { Some(nodes_val) } else { None };
    let rest = String::from_utf8_lossy(body_after_nl);
    let lines: Vec<&str> = rest.lines().collect();
    if lines.len() >= 3 {
        let cand_spec = lines[0].trim();
        let base_spec = lines[1].trim();
        let board_line = lines[2].trim();
        if let Some(board) = Board::from_str(board_line) {
            let t0 = Instant::now();
            let mut cand = PlayerSlot::create(cand_spec).expect("worker: cand init");
            let mut base = PlayerSlot::create(base_spec).expect("worker: base init");
            let (g1, _, _, _) = play_single_game(&board, true, &mut cand, &mut base, time_ms, nodes)
                .expect("worker: sprt game 1");
            let (g2, _, _, _) = play_single_game(&board, false, &mut cand, &mut base, time_ms, nodes)
                .expect("worker: sprt game 2");
            let submit = format!("PAIR {pid} {g1} {g2}\n");
            match http_post(host_port, "/api/submit_sprt", submit.as_bytes()) {
                Ok(r) if r == b"OK" => {
                    println!(
                        "[{worker_id}] completed SPRT pair {pid} (g1={g1}, g2={g2}) in {:.1}s",
                        t0.elapsed().as_secs_f64()
                    );
                }
                Ok(r) => eprintln!("[{worker_id}] submit SPRT {pid} rejected: {}", String::from_utf8_lossy(&r)),
                Err(e) => eprintln!("[{worker_id}] submit SPRT {pid} error: {e}"),
            }
        }
    }
}

fn run_worker(args: &[String]) -> ExitCode {
    let server = flag_str(args, "--server").unwrap_or_else(|| "http://127.0.0.1:8080".to_string());
    let threads = resolve_threads(args);
    let worker_id = flag_str(args, "--worker-id").unwrap_or_else(|| format!("worker-{}", std::process::id()));
    let host_port = parse_host_port(&server);

    println!("=== TitaniumTest Worker ({worker_id}) -> {host_port} ({threads} parallel threads, 1 thread/search) ===");
    let nn = Nnue::bundled_static();

    loop {
        let resp = match http_post(&host_port, "/api/lease", worker_id.as_bytes()) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("[{worker_id}] lease error: {e}; retrying in 3s...");
                std::thread::sleep(Duration::from_secs(3));
                continue;
            }
        };
        if resp == b"IDLE" {
            std::thread::sleep(Duration::from_secs(2));
            continue;
        }
        let Some(nl_pos) = resp.iter().position(|&b| b == b'\n') else {
            continue;
        };
        let hdr_str = String::from_utf8_lossy(&resp[..nl_pos]);
        let hdr: Vec<&str> = hdr_str.split_whitespace().collect();
        if hdr.len() >= 4 && hdr[0] == "SPRT" {
            // Drain SPRT pairs across all `threads` concurrently until the SPRT job finishes!
            let first_resp = &resp;
            let host_port_ref = host_port.as_str();
            let worker_id_ref = worker_id.as_str();
            std::thread::scope(|sc| {
                sc.spawn(move || {
                    let h_str = String::from_utf8_lossy(&first_resp[..nl_pos]);
                    let h_parts: Vec<&str> = h_str.split_whitespace().collect();
                    execute_sprt_pair_payload(worker_id_ref, host_port_ref, &h_parts, &first_resp[nl_pos + 1..]);
                });
                for tid in 0..threads {
                    sc.spawn(move || {
                        let sub_id = format!("{worker_id_ref}-t{tid}");
                        loop {
                            let Ok(r) = http_post(host_port_ref, "/api/lease", sub_id.as_bytes()) else {
                                break;
                            };
                            let Some(nl) = r.iter().position(|&b| b == b'\n') else {
                                break;
                            };
                            let hs = String::from_utf8_lossy(&r[..nl]);
                            let hp: Vec<&str> = hs.split_whitespace().collect();
                            if hp.len() >= 4 && hp[0] == "SPRT" {
                                execute_sprt_pair_payload(&sub_id, host_port_ref, &hp, &r[nl + 1..]);
                            } else {
                                break;
                            }
                        }
                    });
                }
            });
            continue;
        }
        if hdr.len() >= 4 && hdr[0] == "RELABEL" {
            let cid: usize = hdr[1].parse().expect("cid");
            let count: usize = hdr[2].parse().expect("count");
            let nodes: u64 = hdr[3].parse().expect("nodes");
            let data = &resp[nl_pos + 1..];
            assert_eq!(data.len(), count * 25, "worker: relabel payload length mismatch");

            let t0 = Instant::now();
            let results: Vec<Mutex<(i32, u32, u64)>> = (0..count).map(|_| Mutex::new((0, 0, 0))).collect();
            let next_idx = AtomicUsize::new(0);

            std::thread::scope(|sc| {
                for _ in 0..threads {
                    let results = &results;
                    let next_idx = &next_idx;
                    sc.spawn(move || {
                        let mut searcher = Searcher::with_tt_bits_unconfigured(18);
                        searcher.set_nnue(Some(nn));
                        let limits = SearchLimits {
                            time: None,
                            max_nodes: Some(nodes),
                            max_depth: u8::MAX,
                        };
                        loop {
                            let i = next_idx.fetch_add(1, Ordering::Relaxed);
                            if i >= count {
                                break;
                            }
                            let rec = &data[i * 25..i * 25 + 25];
                            let u = |k: usize| u64::from_le_bytes(rec[k..k + 8].try_into().unwrap());
                            let b = Board::from_occ(u(0), u(8), u(16), rec[24]);
                            searcher.reset_for_new_game();
                            let r = searcher.search(&b, &limits);
                            *results[i].lock().unwrap() = (r.score, r.depth, r.nodes);
                        }
                    });
                }
            });

            let mut submit = format!("CHUNK {cid} {count}\n");
            for slot in &results {
                let (sc, d, n) = *slot.lock().unwrap();
                submit.push_str(&format!("{sc} {d} {n}\n"));
            }
            match http_post(&host_port, "/api/submit_relabel", submit.as_bytes()) {
                Ok(r) if r == b"OK" => {
                    println!(
                        "[{worker_id}] completed chunk {cid:05} ({count} positions) in {:.1}s",
                        t0.elapsed().as_secs_f64()
                    );
                }
                Ok(r) => eprintln!("[{worker_id}] submit chunk {cid} rejected: {}", String::from_utf8_lossy(&r)),
                Err(e) => eprintln!("[{worker_id}] submit chunk {cid} error: {e}"),
            }
        }
    }
}

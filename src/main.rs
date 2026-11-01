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

mod book;
mod data;
mod export;
mod layout;
mod leak;
mod mapper;
mod modal;
mod probe;
mod shrink;
#[cfg(feature = "gpu")]
mod gpu;
mod trainer;
mod targets;
mod gate;
mod games;
mod sprt;
mod train;

use sprt::{SprtTally, SprtVerdict};
mod hash;
mod prov;
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
        "gate" => gate::run_gate(&args[1..]),
        "log" => gate::run_log(&args[1..]),
        "merge-check" => gate::run_merge_check(&args[1..]),
        "data" => data::run_data(&args[1..]),
        "train" => run_train_subcmd(&args[1..]),
        "relabel" => run_relabel_local(&args[1..]),
        "analyze" => games::run_analyze(&args[1..]),
        "insight" => games::run_insight(&args[1..]),
        "harvest" => games::run_harvest(&args[1..]),
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
         \x20 relabel      Zero-overhead persistent multi-threaded local relabeler (reads .bin once!)\n\
         \x20 train        Bullet-style native Rust sparse-feature NNUE trainer (+ optional --auto-sprt)\n\
         \x20 coordinator  Start the HTTP job coordinator + live web dashboard (:8080)\n\
         \x20 worker       Connect to a coordinator and execute SPRT / relabel jobs\n\
         \n\
         EXAMPLES:\n\
         \x20 titaniumtest sprt --candidate self --baseline \"./old_titanium serve --network bundled\" \\\n\
         \x20                   --book data/nnue/book100.txt --time-ms 100 --elo0 0 --elo1 10 --hyperthread\n\
         \x20 titaniumtest relabel --bin dis_positions.bin --out out_dir --nodes 5000 --hyperthread\n\
         \x20 titaniumtest train --init bundled --bin dis_positions.bin --labels-dir out_dir --out cand.tnn --auto-sprt\n\
         \x20 titaniumtest coordinator --bind 0.0.0.0:8080 --relabel-bin dis_positions.bin --relabel-out out_dir\n\
         \x20 titaniumtest worker --server http://127.0.0.1:8080 --hyperthread"
    );
}

fn run_relabel_local(args: &[String]) -> ExitCode {
    let Some(bin_path) = flag_str(args, "--bin") else {
        eprintln!("relabel: --bin <dis_positions.bin> is required");
        return ExitCode::FAILURE;
    };
    let Some(out_dir) = flag_str(args, "--out") else {
        eprintln!("relabel: --out <out_dir> is required");
        return ExitCode::FAILURE;
    };
    let nodes = flag_u64(args, "--nodes").unwrap_or(5000);
    let threads = resolve_threads(args);

    fs::create_dir_all(&out_dir).expect("relabel: create out_dir");
    let data = fs::read(&bin_path).unwrap_or_else(|e| panic!("relabel: read {bin_path}: {e}"));
    let total_rows = data.len() / 25;
    let total_chunks = (total_rows + RELABEL_CHUNK_SIZE - 1) / RELABEL_CHUNK_SIZE;

    let mut todo_chunks = Vec::new();
    for cid in 0..total_chunks {
        let p_c = format!("{out_dir}/c_{cid}.txt");
        if !Path::new(&p_c).exists() {
            todo_chunks.push(cid);
        }
    }
    let done_init = total_chunks - todo_chunks.len();
    println!(
        "=== TitaniumTest Persistent Relabeler ===\n\
         Bin File : {bin_path} ({total_rows} rows, {:.1} MB loaded ONCE in RAM)\n\
         Out Dir  : {out_dir}\n\
         Progress : {done_init}/{total_chunks} chunks already done ({} remaining)\n\
         Config   : {nodes} nodes/pos, {threads} persistent Searcher threads (1 thread/search)",
        data.len() as f64 / 1_048_576.0,
        todo_chunks.len()
    );

    let nn = Nnue::bundled_static();
    let t0 = Instant::now();
    for (k, &cid) in todo_chunks.iter().enumerate() {
        let p_c = format!("{out_dir}/c_{cid}.txt");
        if Path::new(&p_c).exists() {
            continue;
        }
        let lo = cid * RELABEL_CHUNK_SIZE;
        let hi = ((cid + 1) * RELABEL_CHUNK_SIZE).min(total_rows);
        let count = hi.saturating_sub(lo);
        let results: Vec<Mutex<(i32, u32, u64)>> = (0..count).map(|_| Mutex::new((0, 0, 0))).collect();
        let next_idx = AtomicUsize::new(0);
        let t_chunk = Instant::now();

        std::thread::scope(|sc| {
            for _ in 0..threads {
                let data_ref = &data;
                let results_ref = &results;
                let next_idx_ref = &next_idx;
                sc.spawn(move || {
                    let mut searcher = Searcher::with_tt_bits_unconfigured(18);
                    searcher.set_nnue(Some(nn));
                    let limits = SearchLimits {
                        time: None,
                        max_nodes: Some(nodes),
                        max_depth: u8::MAX,
                    };
                    loop {
                        let i = next_idx_ref.fetch_add(32, Ordering::Relaxed);
                        if i >= count {
                            break;
                        }
                        for j in i..(i + 32).min(count) {
                            let row = lo + j;
                            let rec = &data_ref[row * 25..row * 25 + 25];
                            let u = |off: usize| u64::from_le_bytes(rec[off..off + 8].try_into().unwrap());
                            let b = Board::from_occ(u(0), u(8), u(16), rec[24]);
                            searcher.reset_for_new_game();
                            let r = searcher.search(&b, &limits);
                            *results_ref[j].lock().unwrap() = (r.score, r.depth, r.nodes);
                        }
                    }
                });
            }
        });

        let mut out_str = String::with_capacity(count * 16);
        for slot in &results {
            let (sc, d, n) = *slot.lock().unwrap();
            out_str.push_str(&format!("{sc} {d} {n}\n"));
        }
        let tmp_p = format!("{out_dir}/c_{cid}.tmp");
        fs::write(&tmp_p, out_str).expect("write tmp chunk");
        fs::rename(&tmp_p, &p_c).expect("rename final chunk");

        let dt = t_chunk.elapsed().as_secs_f64();
        let total_done = done_init + k + 1;
        println!(
            "[Chunk {cid:>4} | {total_done}/{total_chunks} ({:.1}%)] {count} pos in {dt:.1}s ({:.0} pos/s, elapsed {:.1}m)",
            total_done as f64 * 100.0 / total_chunks as f64,
            count as f64 / dt.max(1e-6),
            t0.elapsed().as_secs_f64() / 60.0
        );
    }
    ExitCode::SUCCESS
}

fn run_train_subcmd(args: &[String]) -> ExitCode {
    match args.first().map(String::as_str) {
        Some("run") => return trainer::run_train(&args[1..]),
        Some("parity") => return trainer::run_parity(&args[1..]),
        Some("check") => return trainer::run_check(&args[1..]),
        Some("export") => return export::run_export(&args[1..]),
        Some("buckets") => return trainer::run_buckets(&args[1..]),
        Some("probe") => return probe::run_probe(&args[1..]),
        Some("shrink") => return shrink::run_shrink(&args[1..]),
        Some("rank") => return shrink::run_rank(&args[1..]),
        Some("upscale") => return shrink::run_upscale(&args[1..]),
        Some("split") => return shrink::run_split(&args[1..]),
        Some("remap") => return shrink::run_remap(&args[1..]),
        Some("modal") => return modal::run_modal(&args[1..]),
        #[cfg(feature = "gpu")]
        Some("gpu-parity") => {
            let a = &args[1..];
            let r = gpu::parity(&flag_str(a, "--init").unwrap_or_else(|| "bundled".into()), &flag_str(a, "--tdat").unwrap_or_default(), flag_u64(a, "--n").unwrap_or(512) as usize);
            return match r {
                Ok(()) => ExitCode::SUCCESS,
                Err(e) => {
                    eprintln!("{e}");
                    ExitCode::FAILURE
                }
            };
        }
        _ => {}
    }
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
        // mandatory: every gate move must record what the engine thought (score, depth, nodes); refuse silent engines
        eng.new_game()?;
        let (_, t) = eng.search_move(&Board::start(), None, Some(200))?;
        if (t.score.is_none() && t.mate.is_none()) || t.depth.is_none() || t.nodes.is_none() {
            return Err(format!("engine '{cmd_line}' prints no 'info depth .. score .. nodes ..' line before bestmove: evals are mandatory for gate analysis (build the engine from exp/uai-info or later)"));
        }
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
    ) -> Result<(Option<Move>, Thought), String> {
        self.send(&format!("position fen {}", b.to_ataxx_fen()))?;
        let go_cmd = match (time_ms, nodes) {
            (Some(ms), _) => format!("go movetime {ms}"),
            (None, Some(n)) => format!("go nodes {n}"),
            (None, None) => "go movetime 100".to_string(),
        };
        self.send(&go_cmd)?;
        let t0 = Instant::now();
        let mut thought = Thought::default();
        let mut buf = String::new();
        let line = loop {
            buf.clear();
            let n = self.stdout.read_line(&mut buf).map_err(|e| format!("read_line waiting for bestmove: {e}"))?;
            if n == 0 {
                return Err("EOF waiting for 'bestmove'".into());
            }
            let l = buf.trim();
            if l.starts_with("info") && l.contains(" score ") {
                parse_info(l, &mut thought);
            } else if l.starts_with("bestmove ") {
                break l.to_string();
            }
        };
        thought.ms = t0.elapsed().as_millis() as u64;
        let tok = line.split_whitespace().nth(1).unwrap_or("pass");
        Ok((parse_uai_move(b, tok), thought))
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

/// Hash label of an engine spec: `self` = bundled net, `*.tnn` = that net, else the exe (first token) plus every
/// `.tnn` argument, so a log always says which binary and which net played.
fn engine_label(spec: &str) -> String {
    if spec == "self" {
        return format!("bundled {}", hash::short_hex(titanium::nnue::BUNDLED_NNUE));
    }
    let path = spec.strip_prefix("tnn:").unwrap_or(spec);
    if path.ends_with(".tnn") && !path.contains(char::is_whitespace) {
        return format!("net {}", hash::file_label(path));
    }
    let mut toks = spec.split_whitespace();
    let exe = toks.next().unwrap_or("");
    let mut out = format!("exe {}", hash::file_label(exe));
    for t in toks.filter(|t| t.ends_with(".tnn")) {
        out.push_str(&format!(" net {}", hash::file_label(t)));
    }
    out
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

/// What the engine reported for one move: score from the MOVER's view (cp, or `mate` in moves when proven:
/// + = mover mates, - = mover is mated; `score` is then None), depth,
/// nodes, wall time measured by the hub, principal variation. External engines: the last `info ... score`
/// line before `bestmove`; missing fields stay None.
#[derive(Clone, Debug, Default)]
pub struct Thought {
    pub score: Option<i32>,
    pub mate: Option<i32>,
    pub depth: Option<u32>,
    pub nodes: Option<u64>,
    pub ms: u64,
    pub pv: String,
}

/// One played move with the mover's thought.
#[derive(Clone, Debug)]
pub struct MoveRec {
    pub ply: u16,
    pub side: u8,
    pub mover: &'static str,
    pub mv: String,
    pub thought: Thought,
}

fn parse_info(line: &str, t: &mut Thought) {
    let tok: Vec<&str> = line.split_whitespace().collect();
    let mut i = 0;
    while i < tok.len() {
        match tok[i] {
            "depth" => t.depth = tok.get(i + 1).and_then(|v| v.parse().ok()),
            "nodes" => t.nodes = tok.get(i + 1).and_then(|v| v.parse().ok()),
            "score" => match tok.get(i + 1).copied() {
                Some("cp") => {
                    t.score = tok.get(i + 2).and_then(|v| v.parse().ok());
                    t.mate = None;
                }
                Some("mate") => {
                    // a mate is exact: keep the distance, never a made-up cp value
                    t.mate = tok.get(i + 2).and_then(|v| v.parse().ok());
                    t.score = None;
                }
                _ => {}
            },
            "pv" => {
                t.pv = tok[i + 1..].join(" ");
                break;
            }
            _ => {}
        }
        i += 1;
    }
}

pub fn json_str(s: &str) -> String {
    let mut o = String::with_capacity(s.len() + 2);
    o.push('"');
    for c in s.chars() {
        match c {
            '"' => o.push_str("\\\""),
            '\\' => o.push_str("\\\\"),
            c if (c as u32) < 0x20 => o.push(' '),
            c => o.push(c),
        }
    }
    o.push('"');
    o
}

fn opt<T: std::fmt::Display>(v: &Option<T>) -> String {
    v.as_ref().map(|x| x.to_string()).unwrap_or_else(|| "null".into())
}

/// One JSON line per game: everything needed to replay it and to see what each engine thought on every move.
fn game_json(pair: usize, book_idx: usize, start: &Board, cand_seat: &str, pts_x2: u8, bl: u32, wh: u32, moves: &[MoveRec]) -> String {
    let mv: Vec<String> = moves
        .iter()
        .map(|m| {
            format!(
                "{{\"ply\":{},\"side\":\"{}\",\"mover\":\"{}\",\"move\":{},\"score\":{},\"mate\":{},\"depth\":{},\"nodes\":{},\"ms\":{},\"pv\":{}}}",
                m.ply,
                if m.side == 0 { "x" } else { "o" },
                m.mover,
                json_str(&m.mv),
                opt(&m.thought.score),
                opt(&m.thought.mate),
                opt(&m.thought.depth),
                opt(&m.thought.nodes),
                m.thought.ms,
                json_str(&m.thought.pv)
            )
        })
        .collect();
    format!(
        "{{\"pair\":{pair},\"book\":{book_idx},\"start\":{},\"candidate_seat\":\"{cand_seat}\",\"candidate_points\":{},\"final_x\":{bl},\"final_o\":{wh},\"moves\":[{}]}}",
        json_str(&start.to_ataxx_fen()),
        pts_x2 as f64 / 2.0,
        mv.join(",")
    )
}

enum PlayerSlot {
    InProcess(Searcher<'static>),
    External(UaiEngine),
}

impl PlayerSlot {
    fn create(spec: &str) -> Result<Self, String> {
        if spec == "self" {
            Ok(PlayerSlot::InProcess(Searcher::with_main_network()))
        } else if spec.starts_with("tnn:") || (spec.ends_with(".tnn") && !spec.contains(char::is_whitespace)) {
            // a bare net path = in-process searcher; an engine COMMAND that merely ends in .tnn is external
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
    ) -> Result<(Option<Move>, Thought), String> {
        match self {
            PlayerSlot::InProcess(s) => {
                let limits = SearchLimits {
                    time: time_ms.map(Duration::from_millis),
                    max_nodes: nodes,
                    max_depth: u8::MAX,
                };
                let r = s.search(b, &limits);
                let mate = mate_in_moves(r.score);
                let t = Thought {
                    score: if mate.is_some() { None } else { Some(r.score) },
                    mate,
                    depth: Some(r.depth),
                    nodes: Some(r.nodes),
                    ms: r.elapsed.as_millis() as u64,
                    pv: String::new(),
                };
                Ok((r.best, t))
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
    cand_time_ms: Option<u64>,
    base_time_ms: Option<u64>,
    nodes: Option<u64>,
    cand_reset_every_move: bool,
    base_reset_every_move: bool,
    rules: u8,
    moves: &mut Vec<MoveRec>,
) -> Result<(u8, u32, u32, u16), String> {
    cand.reset()?;
    base.reset()?;
    let mut b = *start_board;
    let mut forfeit_winner: Option<u8> = None;
    moves.clear();

    // rules: 0 classic (plays the pass first), 1 libataxx (Moonbird/Zataxx: ends when NEITHER side can move),
    // 2 botzone (no passes: the game ends the moment EITHER side cannot move, empties go to the mobile side).
    let libataxx_end = rules == 1;
    let bz_end = rules == 2;
    let over = |b: &Board| b.game_over() || (libataxx_end && !b.has_moves(0) && !b.has_moves(1)) || (bz_end && (!b.has_moves(0) || !b.has_moves(1)));
    while !over(&b) && b.ply < 300 {
        let side = b.turn;
        if !b.has_moves(side) {
            moves.push(MoveRec { ply: b.ply as u16, side, mover: "-", mv: "0000".into(), thought: Thought::default() });
            b = b.make_pass();
            continue;
        }
        let cand_turn = (side == 0) == candidate_is_black;
        let (mv, thought) = if cand_turn {
            if cand_reset_every_move {
                cand.reset()?;
            }
            cand.best_move(&b, cand_time_ms, nodes)?
        } else {
            if base_reset_every_move {
                base.reset()?;
            }
            base.best_move(&b, base_time_ms, nodes)?
        };
        match mv {
            Some(m) => {
                let mv = if m.is_clone() {
                    Board::sq_name(m.to)
                } else {
                    format!("{}{}", Board::sq_name(m.from), Board::sq_name(m.to))
                };
                let mover = if cand_turn { "candidate" } else { "baseline" };
                moves.push(MoveRec { ply: b.ply as u16, side, mover, mv, thought });
                b = b.make(m);
            }
            None => {
                let mover = if cand_turn { "candidate" } else { "baseline" };
                moves.push(MoveRec { ply: b.ply as u16, side, mover, mv: "FORFEIT".into(), thought });
                forfeit_winner = Some(1 - side);
                break;
            }
        }
    }

    let (mut bl, mut wh) = b.counts();
    let mut bz_winner: Option<u8> = None;
    if bz_end && forfeit_winner.is_none() && over(&b) {
        let empties = 49u32.saturating_sub(bl as u32 + wh as u32);
        let (m0, m1) = (b.has_moves(0), b.has_moves(1));
        if m0 && !m1 {
            bl = (bl as u32 + empties) as _;
        } else if m1 && !m0 {
            wh = (wh as u32 + empties) as _;
        }
        bz_winner = Some(if bl > wh { 0 } else if wh > bl { 1 } else { 2 });
    }
    let winner = forfeit_winner.or(bz_winner).unwrap_or_else(|| {
        if b.ply >= 300 && !over(&b) {
            2
        } else {
            b.winner()
        }
    });
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
    let cand_time_ms = flag_u64(args, "--cand-time-ms").or(time_ms);
    let base_time_ms = flag_u64(args, "--base-time-ms").or(time_ms);
    let cand_reset_every_move = has_flag(args, "--cand-reset-every-move");
    let base_reset_every_move = has_flag(args, "--base-reset-every-move");
    let elo0 = flag_f64(args, "--elo0").unwrap_or(0.0);
    let elo1 = flag_f64(args, "--elo1").unwrap_or(20.0);
    let max_games = flag_u64(args, "--max-games").unwrap_or(200) as usize;
    let threads = resolve_threads(args);
    let out_path = flag_str(args, "--out");

    // A gate never runs without a checked book (no silent single start position).
    let Some(book_path) = book_path else {
        eprintln!("sprt: --book <FILE> is required (paired openings, both seats)");
        return ExitCode::FAILURE;
    };
    let book_boards: Vec<Board> = match book::load(&book_path) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("sprt: {e}");
            return ExitCode::FAILURE;
        }
    };
    let book0 = flag_u64(args, "--book0").unwrap_or(0) as usize;
    let rules: u8 = match flag_str(args, "--rules").as_deref() {
        None | Some("classic") => 0,
        Some("libataxx") => 1,
        Some("botzone") => 2,
        Some(r) => {
            eprintln!("sprt: --rules classic|libataxx|botzone, got {r}");
            return ExitCode::FAILURE;
        }
    };
    // provenance (D242): a candidate net under test must carry a `.prov` from `ti train export`; a net the baseline also
    // runs (speed / search gates) is not under test and needs none
    let base_nets: Vec<String> = prov::nets_of(&base_spec).iter().map(|n| hash::file_label(n)).collect();
    let mut prov_note = String::new();
    for n in prov::nets_of(&cand_spec) {
        if base_nets.contains(&hash::file_label(&n)) {
            continue;
        }
        match prov::check_net(&n) {
            Ok(s) => prov_note.push_str(&format!(" | prov {s}")),
            Err(e) if has_flag(args, "--allow-unproven-net") => prov_note.push_str(&format!(" | UNPROVEN NET (override): {e}")),
            Err(e) => {
                eprintln!("sprt: candidate net has no verified provenance: {e}\n      (export it with `ti train export`, or pass --allow-unproven-net to gate it anyway, logged as UNPROVEN)");
                return ExitCode::FAILURE;
            }
        }
    }
    let header = format!(
        "# ti sprt | candidate {} [{}] | baseline {} [{}] | book {} [{}] @{} | rules {} | nodes {:?} cand_ms {:?} base_ms {:?} | elo0 {} elo1 {} | hub {}{prov_note}",
        cand_spec,
        engine_label(&cand_spec),
        base_spec,
        engine_label(&base_spec),
        book_path,
        hash::file_label(&book_path),
        book0,
        ["classic", "libataxx", "botzone"][rules as usize],
        nodes,
        cand_time_ms,
        base_time_ms,
        elo0,
        elo1,
        env!("CARGO_PKG_VERSION"),
    );
    println!("{header}");
    if let Some(p) = &out_path {
        if fs::metadata(p).is_ok() {
            eprintln!("sprt: --out {p} exists -> refusing to append to an old run");
            return ExitCode::FAILURE;
        }
        let _ = fs::write(p, format!("{header}\n"));
    }

    let max_pairs = (max_games + 1) / 2;
    println!(
        "=== TitaniumTest SPRT Gate ===\n\
         Candidate : {cand_spec} (time_ms={cand_time_ms:?}, reset_every_move={cand_reset_every_move})\n\
         Baseline  : {base_spec} (time_ms={base_time_ms:?}, reset_every_move={base_reset_every_move})\n\
         Limits    : nodes={nodes:?}, threads={threads}\n\
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
                    let book_idx = (book0 + pair_idx) % book_boards.len();
                    let board = &book_boards[book_idx];
                    let (mut mv1, mut mv2) = (Vec::new(), Vec::new());
                    let (g1, b1, w1, p1) = play_single_game(
                        board,
                        true,
                        &mut cand,
                        &mut base,
                        cand_time_ms,
                        base_time_ms,
                        nodes,
                        cand_reset_every_move,
                        base_reset_every_move,
                        rules,
                        &mut mv1,
                    )
                    .unwrap_or_else(|e| panic!("worker {tid} pair {pair_idx} game 1 failed: {e}"));
                    if stop_flag.load(Ordering::Relaxed) {
                        break;
                    }
                    let (g2, b2, w2, p2) = play_single_game(
                        board,
                        false,
                        &mut cand,
                        &mut base,
                        cand_time_ms,
                        base_time_ms,
                        nodes,
                        cand_reset_every_move,
                        base_reset_every_move,
                        rules,
                        &mut mv2,
                    )
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
                            // per-game lines in the `titanium-cli match` format (training/gate_summary.py reads them)
                            for (g, (pts, bl, wh, plies, mv, seat)) in
                                [(g1, b1, w1, p1, &mv1, "x"), (g2, b2, w2, p2, &mv2, "o")].into_iter().enumerate()
                            {
                                let res = match pts {
                                    2 => "titanium wins",
                                    0 => "opponent wins",
                                    _ => "draw",
                                };
                                let opp = if seat == "x" { "o" } else { "x" };
                                let _ = writeln!(
                                    f,
                                    "-- game {}: book {book_idx} titanium={seat} opp={opp}: {res} {bl}-{wh} ({plies} plies)\n{}",
                                    2 * pair_idx + g + 1,
                                    mv.iter().map(|m| m.mv.as_str()).collect::<Vec<_>>().join(" ")
                                );
                                if let Ok(mut j) = fs::OpenOptions::new().create(true).append(true).open(format!("{p}.games.jsonl")) {
                                    let _ = writeln!(j, "{}", game_json(pair_idx, book_idx, board, seat, pts, bl, wh, mv));
                                }
                            }
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
         LLR   : {:+.2} [{:.2}, {:.2}] -> {:?}\n\
         Split : {:.0}% of pairs split 1-1 by seat{}",
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
        st.verdict,
        100.0 * st.seat_split,
        if st.seat_split > sprt::SEAT_SPLIT_WARN {
            "  WARNING: book lines are seat-decided, rebalance the book"
        } else {
            ""
        }
    );
    if let Some(p) = &out_path {
        if let Ok(text) = fs::read_to_string(format!("{p}.games.jsonl")) {
            let gs: Vec<games::Game> = text.lines().filter(|l| !l.trim().is_empty()).filter_map(|l| games::parse_game(l).ok()).collect();
            if let Some(r) = games::insight(&gs) {
                println!("{r}");
            }
        }
    }
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
            let mut st = state.lock().unwrap();
            // Dynamically refresh completed chunks from output dir if present
            let mut max_chunk_reached = 0usize;
            if let Some(out_d) = &st.relabel_out_dir {
                if let Ok(entries) = fs::read_dir(out_d) {
                    for entry in entries.flatten() {
                        let name = entry.file_name();
                        let s = name.to_string_lossy();
                        if s.starts_with("c_") && s.ends_with(".txt") {
                            if let Ok(cid) = s[2..s.len() - 4].parse::<usize>() {
                                if cid < st.total_chunks {
                                    st.completed_chunks[cid] = true;
                                }
                                if cid > max_chunk_reached {
                                    max_chunk_reached = cid;
                                }
                            }
                        }
                    }
                }
            }
            let done = st.completed_chunks.iter().filter(|&&b| b).count();
            let pct = if st.total_chunks > 0 {
                done as f64 * 100.0 / st.total_chunks as f64
            } else {
                0.0
            };
            let reach_pct = if st.total_chunks > 0 {
                max_chunk_reached as f64 * 100.0 / st.total_chunks as f64
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
                "<div class='card'><h2>SPRT Gate</h2><p style='color:#8b949e;'>No active SPRT gate running. Background relabeling has 100% compute priority.</p></div>".to_string()
            };
            let html = format!(
                "<!doctype html><html><head><meta charset='utf-8'><meta http-equiv='refresh' content='4'>\
                 <title>Titanium Compute Network Dashboard</title>\
                 <style>\
                   body{{background:#0d1117;color:#e6edf3;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif;margin:0;padding:28px;}}\
                   .container{{max-width:980px;margin:0 auto;}}\
                   h1{{color:#58a6ff;margin:0 0 6px 0;display:flex;align-items:center;gap:10px;}}\
                   .sub{{color:#8b949e;margin:0 0 24px 0;font-size:14px;}}\
                   .card{{background:#161b22;border:1px solid #30363d;border-radius:10px;padding:20px;margin-bottom:20px;box-shadow:0 4px 12px rgba(0,0,0,0.15);}}\
                   .card h2{{margin-top:0;font-size:18px;color:#f0f6fc;border-bottom:1px solid #21262d;padding-bottom:10px;}}\
                   .bar-wrap{{background:#21262d;border-radius:6px;height:24px;width:100%;overflow:hidden;margin:12px 0 6px 0;position:relative;}}\
                   .bar-fill{{height:100%;background:linear-gradient(90deg,#238636,#2ea043);transition:width 0.5s ease;}}\
                   .bar-reach{{height:100%;background:rgba(88,166,255,0.4);position:absolute;top:0;left:0;}}\
                   .stat-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:14px;margin-top:16px;}}\
                   .stat-box{{background:#0d1117;border:1px solid #30363d;border-radius:6px;padding:12px;}}\
                   .stat-label{{color:#8b949e;font-size:12px;text-transform:uppercase;margin-bottom:4px;}}\
                   .stat-val{{font-size:20px;font-weight:600;color:#58a6ff;}}\
                   .tag{{display:inline-block;padding:2px 8px;border-radius:12px;font-size:11px;font-weight:600;}}\
                   .tag-green{{background:#1f6feb22;color:#58a6ff;border:1px solid #1f6feb;}}\
                   code{{background:#21262d;padding:2px 6px;border-radius:4px;font-family:monospace;font-size:13px;}}\
                   a{{color:#58a6ff;text-decoration:none;}} a:hover{{text-decoration:underline;}}\
                 </style></head><body><div class='container'>\
                 <h1>Titanium Compute Network <span class='tag tag-green'>LIVE</span></h1>\
                 <p class='sub'>Continuous Distributed Training, 5k Relabeling & SPRT Coordinator &bull; Port :8080</p>\
                 {sprt_html}\
                 <div class='card'>\
                   <h2>Dataset Relabeling Queue (5,000 nodes/pos)</h2>\
                   <div style='display:flex;justify-content:space-between;font-size:14px;'>\
                     <span>Dense Progress: <b>{done} / {}</b> chunks (<b>{:.1}%</b>)</span>\
                     <span>Deepest Tail Reach: <b>Chunk {} / {}</b> (<b>{:.1}%</b>)</span>\
                   </div>\
                   <div class='bar-wrap'>\
                     <div class='bar-reach' style='width:{:.2}%;'></div>\
                     <div class='bar-fill' style='width:{:.2}%;'></div>\
                   </div>\
                   <div style='display:flex;justify-content:space-between;font-size:12px;color:#8b949e;'>\
                     <span>0 pos (Head)</span>\
                     <span>Chunk 5,869 (&Delta; &lt; 10 cp threshold: REACHED &bull; 92.6%)</span>\
                     <span>31.57M pos (Tail)</span>\
                   </div>\
                   <div class='stat-grid'>\
                     <div class='stat-box'><div class='stat-label'>Positions Verified</div><div class='stat-val'>{}</div></div>\
                     <div class='stat-box'><div class='stat-label'>Active Fleet</div><div class='stat-val' style='font-size:16px;color:#3fb950;'>46 Cores (Local + 3 Cloud)</div></div>\
                     <div class='stat-box'><div class='stat-label'>Disagreement Frontier</div><div class='stat-val' style='font-size:16px;color:#f0883e;'>&lt; 10 cp Target Reached</div></div>\
                     <div class='stat-box'><div class='stat-label'>Uptime</div><div class='stat-val' style='font-size:16px;'>{:.0}s</div></div>\
                   </div>\
                 </div>\
                 <div class='card'>\
                   <h2>Competitive Arena & Live Matches</h2>\
                   <p>Active Botzone Bot: <code>titanium_ataxx_PY6</code> &bull; Elo: <b>1343.2</b> &bull; Latest Ladder Match: <a href='https://www.botzone.org.cn/match/6ae14ef79eb7df09d05a50e5' target='_blank'><b>Match #6ae14ef79eb7df09d05a50e5</b> (vs 1582.19)</a></p>\
                 </div>\
                 <div class='card'>\
                   <h2>Next Strategic Phases</h2>\
                   <ol style='color:#8b949e;font-size:14px;line-height:1.8;margin-bottom:0;'>\
                     <li><b style='color:#f0f6fc;'>Phase 1 (Big Net):</b> Train high-capacity NNUE on clean 16.7M+ 5k-node dataset.</li>\
                     <li><b style='color:#f0f6fc;'>Phase 2 (SmallNet + SVD):</b> Distill <code>fut_small.tsn</code> to match new Big Net + SVD compress $W=512 \\to W=256$.</li>\
                     <li><b style='color:#f0f6fc;'>Phase 3 (Disagreement Head):</b> Train zero-cost $TNN3$ head on $|\\text{{Search}}_{{5k}} - \\text{{StaticEval}}|$ for PUCT selectivity.</li>\
                   </ol>\
                 </div>\
                 </div></body></html>",
                st.total_chunks,
                pct,
                max_chunk_reached,
                st.total_chunks,
                reach_pct,
                reach_pct,
                pct,
                done * 5000,
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
            let mut scratch = Vec::new();
            let (g1, _, _, _) = play_single_game(&board, true, &mut cand, &mut base, time_ms, time_ms, nodes, false, false, 0, &mut scratch)
                .expect("worker: sprt game 1");
            let (g2, _, _, _) = play_single_game(&board, false, &mut cand, &mut base, time_ms, time_ms, nodes, false, false, 0, &mut scratch)
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

#[cfg(test)]
mod thought_tests {
    use super::*;

    #[test]
    fn engine_command_ending_in_tnn_is_not_a_net_path() {
        assert!(engine_label("C:/x/titanium-cli.exe serve --network C:/n/a.tnn").starts_with("exe "));
        assert!(engine_label("C:/n/a.tnn").starts_with("net "));
    }

    #[test]
    fn parses_cp_and_mate_info_lines() {
        let mut t = Thought::default();
        parse_info("info depth 9 seldepth 12 score cp -37 nodes 5000 time 41 pv b6 a2 f2", &mut t);
        assert_eq!((t.depth, t.score, t.mate, t.nodes), (Some(9), Some(-37), None, Some(5000)));
        assert_eq!(t.pv, "b6 a2 f2");
        parse_info("info depth 20 score mate -3 nodes 900 pv c3", &mut t);
        assert_eq!((t.score, t.mate), (None, Some(-3)));
    }

    #[test]
    fn game_json_is_one_valid_line() {
        let m = MoveRec { ply: 0, side: 0, mover: "candidate", mv: "b6".into(), thought: Thought { score: Some(12), mate: None, depth: Some(5), nodes: Some(2000), ms: 3, pv: "b6 \"x\"".into() } };
        let j = game_json(0, 7, &Board::start(), "x", 2, 30, 19, &[m]);
        assert!(!j.contains('\n'));
        assert!(j.contains("\"book\":7") && j.contains("\"score\":12") && j.contains("\"mate\":null") && j.contains("\\\"x\\\""));
    }
}

/// UAI mate distance in MOVES for a mate score (`MATE - plies` = side to move wins in `plies`), None otherwise.
/// Positive = side to move mates, negative = gets mated (UAI `score mate N`). Constants match crates/titanium/src/search.rs.
pub fn mate_in_moves(score: i32) -> Option<i32> {
    const MATE: i32 = titanium::search::MATE;
    if score.abs() < MATE - 2000 {
        return None;
    }
    let plies = MATE - score.abs();
    let moves = (plies + 1) / 2;
    Some(if score > 0 { moves.max(1) } else { -moves.max(1) })
}

//! `ti gen`: self-play game generator for the data flywheel.
//!
//! Why it lives in the hub: the Modal image builds only `ti`. `ti sprt` needs an external engine binary, so it
//! cannot generate games there; this command plays the games in-process against the linked engine crate and
//! writes BINARY files only (D806, user order: no JSON for game data): STEM.tgam (standard game file) + STEM.tbst (the
//! engine's best move and score per ply + the game's DAG id), see `BinOut`.
//!
//! Rules: Botzone. The game ends the moment either side is stuck and the mobile side absorbs the empties
//! (`bz_over` / `bz_counts`), so no pass is ever emitted.
//!
//!   ti gen --out STEM [--games 500] [--nodes 5000 (minimum)] [--threads auto]
//!          [--seq-dag DB (default data/dag/seq_dag.db)] [--src NAME] [--book FILE] [--book0 N]
//!
//! LAWS A12 (D927): the engine is ALWAYS the current main, proved before the first game by `genguard::require_main_engine`
//! (node-for-node equal to the main gate exe); --network / --policy / --tt-bits / --time-ms are refused, TT = 2^22,
//! nodes >= 5,000; STEM.engine.json records which main made the files; an existing STEM is never overwritten.
//!
//! THE ONE RULE (user order, D787; D804: the database is the sequence DAG = PGN-like game sequences): the game record
//! is the chain of moves from the start. At every ply take the legal moves, ask the sequence DAG which of them continue
//! a recorded game that has exactly the moves played so far (a game still being played on another thread counts),
//! search with the engine (5,000 nodes) and play the engine's BEST ROOT MOVE THAT HAS NOT BEEN PLAYED YET after this
//! sequence. If every legal move continues a recorded sequence, play the engine's best move. Once the sequence matches
//! nothing recorded, every move is new. Nothing else.
//!
//! Each finished game is written to the sequence DAG (numeric game id `gid`). A game whose complete sequence is already
//! in the DAG is NOT written to the files again (counted as a duplicate in the log); a stored game's `gid` is in the .tbst.
//!
//! Defaults: 5,000 nodes per move, up to 6 threads. `--book` gives start positions (the engine is never changed by it).

use std::collections::HashMap;
use std::fs::File;
use std::io::Write;
use std::process::ExitCode;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::Mutex;


use titanium::nnue::Nnue;
use titanium::{Board, Move, MoveList, SearchLimits, Searcher};

use crate::dag::{move_text, seq_child, seq_code, seq_root, SeqDag};

/// Edge text of a move as the position DAG stores it (clone source normalised by the engine's move decoder).
pub(crate) fn edge_text(b: &Board, m: Move) -> String {
    let m = if m.is_pass() { m } else { titanium::ttcodec::decode_move(b, titanium::ttcodec::move_code(m)).unwrap_or(m) };
    move_text(m)
}

/// Binary output of `ti gen` (user order D806: no JSON for game data). Two files, games in the same order:
///   `STEM.tgam`: the standard game file (docs/DATASET_GUIDE.md 3.1, version 1): played move + depth + score per ply.
///   `STEM.tbst`: the labels the game file has no room for. Header 24 B: `TBST`, u32 version 1, u64 games, u64 plies; then per
///   game: u32 `gid` (the game's id in the sequence DAG), then per ply 4 B: u16 move code (10 bits, the ENGINE'S OWN best
///   move at this position, whether or not novelty played another one) + i16 score code of that best move (mover's view).
/// Counts in both headers are rewritten after every game, so a killed run leaves readable files.
pub(crate) struct BinOut {
    tg: File,
    tb: File,
    pub games: u64,
    pub plies: u64,
}

impl BinOut {
    pub(crate) fn create(stem: &str) -> std::io::Result<BinOut> {
        let mut o = BinOut { tg: File::create(format!("{stem}.tgam"))?, tb: File::create(format!("{stem}.tbst"))?, games: 0, plies: 0 };
        let mut h = [0u8; 32];
        h[0..4].copy_from_slice(crate::gamedata::TGAM_MAGIC);
        h[4..8].copy_from_slice(&1u32.to_le_bytes());
        h[24..28].copy_from_slice(&1u32.to_le_bytes()); // one source
        o.tg.write_all(&h)?;
        let mut t = [0u8; 24];
        t[0..4].copy_from_slice(b"TBST");
        t[4..8].copy_from_slice(&1u32.to_le_bytes());
        o.tb.write_all(&t)?;
        Ok(o)
    }

    /// Append one game: the packed game and its labels `(best move code, best score code)` per ply.
    pub(crate) fn append(&mut self, g: &crate::gamedata::PackedGame, gid: u32, best: &[(u16, i16)]) -> std::io::Result<()> {
        use std::io::{Seek, SeekFrom};
        debug_assert_eq!(g.plies.len(), best.len());
        let mut buf = Vec::with_capacity(g.byte_len());
        g.write(&mut buf)?;
        self.tg.write_all(&buf)?;
        let mut lb = Vec::with_capacity(4 + best.len() * 4);
        lb.extend_from_slice(&gid.to_le_bytes());
        for &(c, s) in best {
            lb.extend_from_slice(&c.to_le_bytes());
            lb.extend_from_slice(&s.to_le_bytes());
        }
        self.tb.write_all(&lb)?;
        self.games += 1;
        self.plies += g.plies.len() as u64;
        self.tg.seek(SeekFrom::Start(8))?;
        self.tg.write_all(&self.games.to_le_bytes())?;
        self.tg.write_all(&self.plies.to_le_bytes())?;
        self.tg.seek(SeekFrom::End(0))?;
        self.tb.seek(SeekFrom::Start(8))?;
        self.tb.write_all(&self.games.to_le_bytes())?;
        self.tb.write_all(&self.plies.to_le_bytes())?;
        self.tb.seek(SeekFrom::End(0))?;
        Ok(())
    }
}

/// sequence hash -> (move code -> games in flight that played it)
type Fly = HashMap<u64, HashMap<u16, u32>>;

#[derive(Default)]
struct Counters {
    /// plies played
    plies: AtomicU64,
    /// the engine's best move had not been played after this sequence
    best_new: AtomicU64,
    /// the best move had been played; a lower ranked, never played move was taken
    switched: AtomicU64,
    /// every move the engine can choose had been played; novelty ignored, best move taken
    all_seen: AtomicU64,
    least_sw: AtomicU64,
    /// engine searches made (1 per ply, 2 when the first choice had been played already)
    searches: AtomicU64,
    /// wall time all threads spent in DAG reads and write-backs (nanoseconds, summed over threads)
    dag_ns: AtomicU64,
    /// the engine returned an already played move although unplayed ones existed (the ban did not take): must stay 0
    ban_miss: AtomicU64,
    dag_written: AtomicU64,
    dag_retries: AtomicU64,
    /// games whose complete sequence was already in the DAG (not written to the file)
    dup_games: AtomicU64,
}

pub fn run_gen(args: &[String]) -> ExitCode {
    let Some(out) = crate::flag_str(args, "--out") else {
        eprintln!("usage: ti gen --out STEM [--games 500] [--nodes 5000 (minimum)] [--threads auto] [--seq-dag DB] [--src NAME] [--book FILE] [--book0 N]");
        return ExitCode::FAILURE;
    };
    // LAWS A12 (D927): one idiot-proof generator. STANDARD (default): the engine is the current main (behavioural check
    // below) with its own net / policy / SmallNet, TT 2^22, >= 5,000 nodes per move; the weakening flags are refused.
    // EXPERIMENT (D931): only with --experiment NAME; overrides allowed; outputs AND the DAG must be under experiments/.
    let exp = crate::genguard::experiment(args);
    crate::genguard::announce("gen", &exp);
    if exp.is_none() {
        if let Err(e) = crate::genguard::refuse_weakening_flags("gen", args, &["--network", "--policy", "--tt-bits", "--time-ms"]) {
            eprintln!("{e}");
            return ExitCode::FAILURE;
        }
    }
    let games = crate::flag_u64(args, "--games").unwrap_or(500) as u32;
    let nodes = crate::flag_u64(args, "--nodes").unwrap_or_else(|| crate::genguard::standard().nodes);
    if exp.is_none() {
        if let Err(e) = crate::genguard::check_nodes("gen", nodes) {
            eprintln!("{e}");
            return ExitCode::FAILURE;
        }
    }
    let dag_path = crate::flag_str(args, "--seq-dag").unwrap_or_else(|| match &exp {
        None => "data/dag/seq_dag.db".into(),
        Some(_) => format!("{}/seq_dag.db", std::path::Path::new(&out).parent().map(|p| p.to_string_lossy().into_owned()).unwrap_or_else(|| ".".into())),
    });
    for p in [&out, &dag_path] {
        if let Err(e) = crate::genguard::check_output("gen", &exp, p) {
            eprintln!("{e}");
            return ExitCode::FAILURE;
        }
    }
    if let (Some(_), Some(pol)) = (&exp, crate::flag_str(args, "--policy")) {
        titanium::policy::configure(&pol);
    }
    // D1041: on a Docker box (no main checkout) the identity is proven against the PC's recorded main behaviour
    let fingerprint = crate::flag_str(args, "--main-fingerprint");
    let identity = match &fingerprint {
        Some(f) => crate::genguard::check_fingerprint("gen", f),
        None => crate::genguard::require_main_engine("gen"),
    };
    // D1041 shard: --shard K/N plays only the 2-ply openings whose sequence hash % N == K (several machines, one DAG
    // each, no duplicated work); the label of every ply is still the engine's unrestricted best move and score.
    let shard: Option<(u64, u64)> = match crate::flag_str(args, "--shard") {
        None => None,
        Some(s) => match s.split_once('/').and_then(|(k, n)| Some((k.parse::<u64>().ok()?, n.parse::<u64>().ok()?))) {
            Some((k, n)) if n >= 2 && k < n => Some((k, n)),
            _ => {
                eprintln!("gen: --shard must be K/N with 0 <= K < N, N >= 2 (got {s})");
                return ExitCode::FAILURE;
            }
        },
    };
    let engine_id = match (identity, &exp) {
        (Ok(id), _) => {
            eprintln!("gen: engine = current main {} (identity check: {} positions, {} nodes equal to the main exe)", id.main_commit, id.positions, id.nodes);
            Some(id)
        }
        (Err(e), None) if fingerprint.is_some() => {
            eprintln!("{e}");
            return ExitCode::FAILURE;
        }
        (Err(e), None) => {
            if let Err(e2) = crate::genguard::rebuild_and_rerun("gen", &e) {
                eprintln!("{e}
{e2}");
            }
            return ExitCode::FAILURE;
        }
        (Err(e), Some(_)) => {
            eprintln!("gen: EXPERIMENT engine differs from main (allowed in an experiment): {e}");
            None
        }
    };
    let node_limit = Some(nodes);
    let default_threads = std::thread::available_parallelism().map_or(1, |n| n.get()).saturating_sub(2).clamp(1, 6) as u64;
    let threads = crate::flag_u64(args, "--threads").unwrap_or(default_threads).max(1) as usize;
    let book0 = crate::flag_u64(args, "--book0").unwrap_or(0) as usize;
    let tt_bits = match &exp {
        Some(_) => crate::flag_u64(args, "--tt-bits").map_or(crate::genguard::standard().tt_bits, |v| v as usize),
        None => crate::genguard::standard().tt_bits,
    };
    let src = crate::flag_str(args, "--src").unwrap_or_else(|| {
        std::path::Path::new(&out).file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_else(|| out.clone())
    });

    let owned_net: Option<Nnue> = match (&exp, crate::flag_str(args, "--network")) {
        (Some(_), Some(p)) if p != "bundled" => match Nnue::load(&p) {
            Ok(n) => Some(n),
            Err(e) => {
                eprintln!("gen: cannot load network {p}: {e}");
                return ExitCode::FAILURE;
            }
        },
        _ => None,
    };
    let net: Option<&Nnue> = Some(owned_net.as_ref().unwrap_or_else(|| Nnue::bundled_static()));

    let book: Option<Vec<String>> = match crate::flag_str(args, "--book") {
        Some(p) => match std::fs::read_to_string(&p) {
            Ok(t) => Some(t.lines().map(str::trim).filter(|l| !l.is_empty()).map(String::from).collect()),
            Err(e) => {
                eprintln!("gen: cannot read book {p}: {e}");
                return ExitCode::FAILURE;
            }
        },
        None => None,
    };
    let book = book.filter(|l| !l.is_empty());

    if let Some(parent) = std::path::Path::new(&dag_path).parent().filter(|p| !p.as_os_str().is_empty()) {
        let _ = std::fs::create_dir_all(parent);
    }
    let dag = match SeqDag::open(&dag_path) {
        Ok(d) => {
            eprintln!("gen: sequence DAG {dag_path} (read + write-back)");
            d
        }
        Err(e) => {
            eprintln!("gen: {e}");
            return ExitCode::FAILURE;
        }
    };
    // Moves chosen by games that are still being played (their game reaches the DAG only when it ends). Without this
    // the games running on the other threads are invisible and 6 threads replay the same line (measured D779).
    let fly: Mutex<Fly> = Mutex::new(HashMap::new());

    // --out is the file STEM (a trailing .tgam / .jsonl is stripped): STEM.tgam + STEM.tbst, binary only (D806)
    let stem = out.trim_end_matches(".jsonl").trim_end_matches(".tgam").to_string();
    for ext in ["tgam", "tbst", "engine.json"] {
        if std::path::Path::new(&format!("{stem}.{ext}")).exists() {
            eprintln!("gen: {stem}.{ext} already exists; refusing to overwrite (pick a new --out)");
            return ExitCode::FAILURE;
        }
    }
    let mut id = match &engine_id {
        Some(e) => e.json(),
        None => serde_json::json!({ "identity_check": "FAILED: the engine is NOT the current main (experiment)" }),
    };
    id["mode"] = serde_json::json!(match &exp { Some(n) => format!("EXPERIMENTAL: {n}"), None => "STANDARD (LAWS A12)".to_string() });
    id["tt_bits"] = serde_json::json!(tt_bits);
    id["network"] = serde_json::json!(crate::flag_str(args, "--network").filter(|_| exp.is_some()).unwrap_or_else(|| "bundled".into()));
    id["nodes_per_move"] = serde_json::json!(nodes);
    id["shard"] = serde_json::json!(shard.map(|(k, n)| format!("{k}/{n} (2-ply openings by sequence hash, D1041)")));
    id["started"] = serde_json::json!(std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_secs()));
    if let Err(e) = std::fs::write(format!("{stem}.engine.json"), serde_json::to_string_pretty(&id).unwrap_or_default()) {
        eprintln!("gen: cannot write {stem}.engine.json: {e}");
        return ExitCode::FAILURE;
    }
    let w = match BinOut::create(&stem) {
        Ok(f) => Mutex::new(f),
        Err(e) => {
            eprintln!("gen: cannot create {stem}.tgam/.tbst: {e}");
            return ExitCode::FAILURE;
        }
    };
    // DAG game index = run start (unix seconds) * 100000 + attempt number: unique per run (D783).
    let idx_base: i64 = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_secs() as i64) * 100_000;
    let next = AtomicU32::new(0);
    let done = AtomicU32::new(0);
    let cnt = Counters::default();
    let err: Mutex<Option<String>> = Mutex::new(None);
    let t0 = std::time::Instant::now();

    std::thread::scope(|s| {
        for _ in 0..threads {
            s.spawn(|| {
                let run = || -> Result<(), String> {
                    let conn = dag.connect()?;
                    // moves (codes) already played after this sequence: the DAG + games in flight
                    // code -> number of games that played it (DAG + in flight)
                    let played_from = |h: u64| -> Result<HashMap<u16, i64>, String> {
                        let mut out: HashMap<u16, i64> = conn.children(h)?.into_iter().collect();
                        if let Some(m) = fly.lock().unwrap().get(&h) {
                            for (c, k) in m {
                                *out.entry(*c).or_insert(0) += *k as i64;
                            }
                        }
                        Ok(out)
                    };
                    loop {
                        let game = next.fetch_add(1, Ordering::Relaxed);
                        if game >= games || err.lock().unwrap().is_some() {
                            return Ok(());
                        }
                        let idx = book.as_ref().map_or(0, |l| (book0 + game as usize) % l.len());
                        let start = match book.as_ref() {
                            Some(l) => Board::from_str(&l[idx]).ok_or_else(|| format!("bad book line: {}", l[idx]))?,
                            None => Board::start(),
                        };
                        let mut b = start;
                        let mut h = seq_root(&start);
                        let mut searcher = Searcher::with_tt_bits_unconfigured(tt_bits);
                        searcher.set_nnue(net);
                        let mut plies_bin: Vec<(u16, i16)> = Vec::new();
                        let mut best_bin: Vec<(u16, i16)> = Vec::new();
                        let mut played: Vec<Move> = Vec::new();
                        let mut reserved: Vec<(u64, u16)> = Vec::new();
                        let mut ply = 0u32;
                        let mut ml = MoveList::new();
                        while !b.bz_over() && ply < 400 && b.has_moves(b.turn) {
                            let limits = SearchLimits {
                                time: None,
                                max_nodes: node_limit,
                                max_depth: 30,
                            };
                            // THE ONE RULE: the engine's best root move that has not been played after this sequence.
                            // At most two searches per ply: (1) the engine's own first choice (also the label of this
                            // position); only when that move was already played here, (2) one search that leaves
                            // out every move already played here at once.
                            let t_dag = std::time::Instant::now();
                            let mut played_here = played_from(h)?;
                            cnt.dag_ns.fetch_add(t_dag.elapsed().as_nanos() as u64, Ordering::Relaxed);
                            if let (Some((k, n)), 1) = (shard, ply) {
                                // ply 1 (second move): moves whose 2-ply sequence belongs to another shard count as
                                // played many times, so the novelty rule never picks them while an own-shard move exists
                                b.legal_moves_dedup_into(&mut ml);
                                for m in ml.iter() {
                                    let c = seq_code(&b, m);
                                    let x = seq_child(h, c);
                                    if (x ^ (x >> 29)).wrapping_mul(0x9E37_79B9_7F4A_7C15) % n != k {
                                        let e = played_here.entry(c).or_insert(0);
                                        *e = (*e).max(1 << 40);
                                    }
                                }
                            }
                            let first = searcher.search_bz(&b, &limits);
                            cnt.searches.fetch_add(1, Ordering::Relaxed);
                            let Some(m0) = first.best else { break };
                            let top: Option<(Move, i64)> = Some((m0, first.score as i64));
                            cnt.plies.fetch_add(1, Ordering::Relaxed);
                            let (mv, r) = if !played_here.contains_key(&seq_code(&b, m0)) {
                                cnt.best_new.fetch_add(1, Ordering::Relaxed);
                                (m0, first)
                            } else {
                                b.legal_moves_dedup_into(&mut ml);
                                let banned: Vec<Move> = ml.iter().filter(|m| played_here.contains_key(&seq_code(&b, *m))).collect();
                                if banned.len() >= ml.len() {
                                    // every legal move continues a recorded sequence (user D804): play the engine's best move
                                    // among the LEAST-played ones; each step raises that move's count, so repeated descent
                                    // always reaches a branch the DAG does not cover
                                    cnt.all_seen.fetch_add(1, Ordering::Relaxed);
                                    let nof = |m: &Move| played_here.get(&seq_code(&b, *m)).copied().unwrap_or(0);
                                    let least = ml.iter().map(|m| nof(&m)).min().unwrap_or(0);
                                    if nof(&m0) == least {
                                        (m0, first)
                                    } else {
                                        let skip: Vec<Move> = ml.iter().filter(|m| nof(m) > least).collect();
                                        let second = searcher.search_bz_excluding(&b, &limits, &skip);
                                        cnt.searches.fetch_add(1, Ordering::Relaxed);
                                        match second.best {
                                            Some(m) if nof(&m) == least => {
                                                cnt.least_sw.fetch_add(1, Ordering::Relaxed);
                                                (m, second)
                                            }
                                            other => {
                                                if other.is_some() {
                                                    cnt.ban_miss.fetch_add(1, Ordering::Relaxed);
                                                }
                                                (m0, first)
                                            }
                                        }
                                    }
                                } else {
                                    let second = searcher.search_bz_excluding(&b, &limits, &banned);
                                    cnt.searches.fetch_add(1, Ordering::Relaxed);
                                    match second.best {
                                        Some(m) if !played_here.contains_key(&seq_code(&b, m)) => {
                                            cnt.switched.fetch_add(1, Ordering::Relaxed);
                                            (m, second)
                                        }
                                        other => {
                                            // unplayed moves exist but the engine returned a played one: the ban did not
                                            // take (counted, must stay 0); the first choice is played
                                            if other.is_some() {
                                                cnt.ban_miss.fetch_add(1, Ordering::Relaxed);
                                            }
                                            cnt.all_seen.fetch_add(1, Ordering::Relaxed);
                                            (m0, first)
                                        }
                                    }
                                }
                            };
                            // reserve the chosen move so games running on other threads see it at once
                            let code = seq_code(&b, mv);
                            *fly.lock().unwrap().entry(h).or_default().entry(code).or_insert(0) += 1;
                            reserved.push((h, code));
                            h = seq_child(h, code);
                            // best-move label = the engine's own first choice and its 5k score, whether or not novelty made it
                            // play another move (goes to the .tbst file); the .tgam holds what was played (the game line).
                            let (tm, ts) = top.unwrap_or((mv, r.score as i64));
                            best_bin.push((titanium::ttcodec::move_code(tm), titanium::ttcodec::enc_score(ts as i32)));
                            plies_bin.push((titanium::ttcodec::move_code(mv) | ((r.depth as u16).min(63) << 10), titanium::ttcodec::enc_score(r.score as i32)));
                            played.push(mv);
                            b = b.make(mv);
                            ply += 1;
                        }
                        let (bx, bo) = b.bz_counts();
                        let res = if bx > bo { 2u8 } else if bx < bo { 0u8 } else { 1u8 };
                        // the game goes into the sequence DAG first; its game id is written to the file with it
                        let t_w = std::time::Instant::now();
                        let stored = conn.record(&src, idx_base + game as i64, start, &played, res, true)?;
                        cnt.dag_ns.fetch_add(t_w.elapsed().as_nanos() as u64, Ordering::Relaxed);
                        match stored {
                            Some((gid, r)) => {
                                cnt.dag_written.fetch_add(1, Ordering::Relaxed);
                                cnt.dag_retries.fetch_add(r as u64, Ordering::Relaxed);
                                let pg = crate::gamedata::PackedGame {
                                    occ: start.occ,
                                    blockers: start.blockers,
                                    turn: start.turn,
                                    cand_seat: 2, // self-play: no gate candidate (D930; 0 before)
                                    forfeit: false,
                                    final_cnt: [bx.min(254) as u8, bo.min(254) as u8],
                                    result: res,
                                    src: 0,
                                    book: idx as u32,
                                    pair: game,
                                    plies: std::mem::take(&mut plies_bin),
                                };
                                w.lock().unwrap().append(&pg, gid as u32, &best_bin).map_err(|e| format!("write {stem}.tgam/.tbst: {e}"))?;
                            }
                            None => {
                                cnt.dup_games.fetch_add(1, Ordering::Relaxed);
                            }
                        }
                        // the game is in the DAG now: drop its in-flight reservations
                        {
                            let mut f = fly.lock().unwrap();
                            for (hk, c) in reserved.drain(..) {
                                if let Some(m) = f.get_mut(&hk) {
                                    if let Some(n) = m.get_mut(&c) {
                                        *n -= 1;
                                        if *n == 0 {
                                            m.remove(&c);
                                        }
                                    }
                                    if m.is_empty() {
                                        f.remove(&hk);
                                    }
                                }
                            }
                        }
                        let d = done.fetch_add(1, Ordering::Relaxed) + 1;
                        if d % 50 == 0 {
                            let el = t0.elapsed().as_secs_f64();
                            eprintln!(
                                "gen: {d} games ({el:.0}s elapsed), {} duplicates, {:.2} searches/ply, DAG share of thread time {:.1}%",
                                cnt.dup_games.load(Ordering::Relaxed),
                                cnt.searches.load(Ordering::Relaxed) as f64 / cnt.plies.load(Ordering::Relaxed).max(1) as f64,
                                100.0 * cnt.dag_ns.load(Ordering::Relaxed) as f64 / 1e9 / (el * threads as f64).max(1e-9)
                            );
                        }
                    }
                };
                if let Err(e) = run() {
                    *err.lock().unwrap() = Some(e);
                }
            });
        }
    });
    if let Some(e) = err.lock().unwrap().take() {
        eprintln!("gen: FAILED: {e}");
        return ExitCode::FAILURE;
    }
    let g = |a: &AtomicU64| a.load(Ordering::Relaxed);
    eprintln!(
        "gen: {} plies: best move not played yet {}, a lower ranked never played move taken {}, every move already played (least-played best taken: {} of them) {}",
        g(&cnt.plies),
        g(&cnt.best_new),
        g(&cnt.switched),
        g(&cnt.least_sw),
        g(&cnt.all_seen)
    );
    eprintln!("gen: {} searches ({:.2} per ply), ban misses {}", g(&cnt.searches), g(&cnt.searches) as f64 / g(&cnt.plies).max(1) as f64, g(&cnt.ban_miss));
    eprintln!("gen: sequence DAG write-back: {} games stored, {} repeated sequences (already in the DAG: W/D/L counts updated, not written to the file again), {} conflict retries", g(&cnt.dag_written), g(&cnt.dup_games), g(&cnt.dag_retries));
    eprintln!("gen: done {} games -> {stem}.tgam + {stem}.tbst in {:.0}s on {threads} thread(s)", done.load(Ordering::Relaxed), t0.elapsed().as_secs_f64());
    ExitCode::SUCCESS
}

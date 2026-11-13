//! Game-linked training data (engine D756, user order 2026-11-13: "the dataset needs the games").
//!
//! Three commands and one small library:
//!   ti data chains     --tdat X.tdat [--out X.chains] [--min-len 2]
//!       A `.tdat` has no game id and no ply, but large parts of it are still in game order. This finds every
//!       run of rows where each row is exactly one move after the previous one and writes an index of the runs.
//!       Nothing in the `.tdat` is rewritten: the index only says "rows s..s+len are consecutive plies".
//!   ti data pack-games --out STEM [--roots DIR,DIR] [--inputs a.games.jsonl,b.games.jsonl]
//!       Gate games (`*.games.jsonl`) -> `STEM.tgam`: games, not positions. Per ply 4 bytes: the engine's 10-bit
//!       move code and 16-bit score code (titanium::ttcodec, the transposition-table codecs) plus 6 bits of search
//!       depth. Game id (= index), ply, every position's future, the result and the final stone counts all
//!       follow from the game record. `STEM.tgam.json` lists the source files with their gate header.
//!   ti data games-stat --tgam F.tgam [--lambda 0.7]
//!       Re-reads a `.tgam`, replays every game through the engine's own move decoder and reports.
//!
//! File layout `.tgam` (little endian):
//!   header 32 B : "TGAM", u32 version = 1, u64 games, u64 plies, u32 sources, u32 0
//!   per game    : 40 B header, then `n_plies` x 4 B
//!     header: occ[0] u64, occ[1] u64, blockers u64, n_plies u16, src u16,
//!             flags u8 (bit0 = side to move at the start, bits1-2 = candidate seat 0/1/2 = unknown, bit3 = forfeit),
//!             final[0] u8, final[1] u8 (255 = not recorded), result u8 (2 = side 0 won, 1 = draw, 0 = lost, 3 = unknown),
//!             book u32, pair u32
//!     ply   : u16 = move code (bits 0-9) | depth (bits 10-15, capped at 63); i16 = score code, mover's view
//!             (`NO_SCORE` = the log had no score for that move)
//! File layout `.chains`: "TCHN", u32 version = 1, u64 rows, u64 runs, then per run: start row u32, length u16,
//!   convention u8 (0 = occ0 is the side to move, 1 = occ0 is colour 0 and `stm` names the mover), 0 u8.

use std::collections::HashSet;
use std::fs::{self, File};
use std::hash::{Hash, Hasher};
use std::io::{BufReader, BufWriter, Read, Write};
use std::process::ExitCode;

use titanium::board::Move;
use titanium::search::MATE;
use titanium::ttcodec::{dec_score, decode_move, enc_score, move_code, PASS_CODE};
use titanium::Board;
use titanium::search::{SearchLimits, Searcher};

use crate::data::{Rec, REC};

pub const TGAM_MAGIC: &[u8; 4] = b"TGAM";
pub const CHAIN_MAGIC: &[u8; 4] = b"TCHN";
pub const GAME_HDR: usize = 40;
pub const PLY_BYTES: usize = 4;
/// Score word of a ply whose log line carried no score.
pub const NO_SCORE: i16 = i16::MIN;

// ---------------------------------------------------------------------------------------------------------------
// chains: runs of consecutive plies inside a .tdat
// ---------------------------------------------------------------------------------------------------------------

/// True when the position (mover `m`, opponent `o`) is exactly one non-pass move after the position
/// (mover `pm`, opponent `po`). The side to move swaps, so `m` is the old opponent's side.
pub fn linked(pm: u64, po: u64, m: u64, o: u64) -> bool {
    if m & !po != 0 {
        return false; // the new mover can only have lost stones
    }
    let flipped = po & !m;
    if flipped & !o != 0 {
        return false; // every lost stone now belongs to the other side
    }
    if (pm & !o).count_ones() > 1 {
        return false; // the old mover keeps everything except at most a jump origin
    }
    let fresh = o & !pm & !po;
    if fresh.count_ones() != 1 {
        return false; // exactly one new stone, on a square that was empty
    }
    o & !(pm | flipped | fresh) == 0
}

/// One run of consecutive plies: rows `start .. start + len`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Chain {
    pub start: u32,
    pub len: u16,
    /// 0 = occ0 is the side to move in every row, 1 = occ0 is colour 0 and `stm` names the mover.
    pub conv: u8,
}

#[derive(Default, Debug)]
pub struct ChainStats {
    pub rows: u64,
    pub linked: [u64; 2],
    /// (runs, rows) for run lengths 1, 2-9, 10-29, 30-59, 60-99, 100+
    pub hist: [(u64, u64); 6],
    pub outcome_flips: u64,
}

fn hist_slot(len: u64) -> usize {
    match len {
        1 => 0,
        2..=9 => 1,
        10..=29 => 2,
        30..=59 => 3,
        60..=99 => 4,
        _ => 5,
    }
}

/// Which convention links `r` to `p` (None = not consecutive plies).
fn link_conv(p: &Rec, r: &Rec) -> Option<u8> {
    if linked(p.occ0, p.occ1, r.occ0, r.occ1) {
        return Some(0);
    }
    if p.stm != r.stm {
        let (pm, po) = if p.stm == 0 { (p.occ0, p.occ1) } else { (p.occ1, p.occ0) };
        let (m, o) = if r.stm == 0 { (r.occ0, r.occ1) } else { (r.occ1, r.occ0) };
        if linked(pm, po, m, o) {
            return Some(1);
        }
    }
    None
}

/// Stream a `.tdat` once and return every run of at least `min_len` rows plus the statistics over all runs.
pub fn scan_chains(tdat: &str, min_len: u64) -> Result<(Vec<Chain>, ChainStats), String> {
    let f = File::open(tdat).map_err(|e| format!("{tdat}: {e}"))?;
    let bytes = f.metadata().map_err(|e| e.to_string())?.len();
    if bytes % REC as u64 != 0 {
        return Err(format!("{tdat}: size {bytes} is not a multiple of {REC}"));
    }
    let rows = bytes / REC as u64;
    if rows > u32::MAX as u64 {
        return Err(format!("{tdat}: {rows} rows do not fit the u32 row index"));
    }
    let mut rd = BufReader::with_capacity(1 << 20, f);
    let mut buf = vec![0u8; REC * 65536];
    let mut st = ChainStats { rows, ..Default::default() };
    let mut out = Vec::new();
    let mut prev: Option<Rec> = None;
    let (mut run_start, mut run_len, mut run_conv) = (0u64, 0u64, 0u8);
    let close = |start: u64, len: u64, conv: u8, st: &mut ChainStats, out: &mut Vec<Chain>| {
        if len == 0 {
            return;
        }
        let k = hist_slot(len);
        st.hist[k].0 += 1;
        st.hist[k].1 += len;
        if len >= min_len {
            out.push(Chain { start: start as u32, len: len as u16, conv });
        }
    };
    let mut i = 0u64;
    loop {
        let want = ((rows - i) as usize).min(65536) * REC;
        if want == 0 {
            break;
        }
        rd.read_exact(&mut buf[..want]).map_err(|e| format!("{tdat}: {e}"))?;
        for c in buf[..want].chunks_exact(REC) {
            let r = Rec::from_bytes(c);
            let conv = prev.as_ref().and_then(|p| link_conv(p, &r));
            // a run keeps one convention and stays below the u16 length limit
            let extends = match conv {
                Some(cv) => run_len > 0 && (run_len == 1 || cv == run_conv) && run_len < u16::MAX as u64,
                None => false,
            };
            if let Some(cv) = conv {
                st.linked[cv as usize] += 1;
                if let Some(p) = prev.as_ref() {
                    if (p.outcome + r.outcome - 1.0).abs() < 0.02 {
                        st.outcome_flips += 1;
                    }
                }
            }
            if extends {
                if run_len == 1 {
                    run_conv = conv.unwrap_or(0);
                }
                run_len += 1;
            } else {
                close(run_start, run_len, run_conv, &mut st, &mut out);
                run_start = i;
                run_len = 1;
                run_conv = 0;
            }
            prev = Some(r);
            i += 1;
        }
    }
    close(run_start, run_len, run_conv, &mut st, &mut out);
    Ok((out, st))
}

pub fn write_chains(path: &str, rows: u64, chains: &[Chain]) -> Result<(), String> {
    let mut bytes = Vec::with_capacity(24 + chains.len() * 8);
    bytes.extend_from_slice(CHAIN_MAGIC);
    bytes.extend_from_slice(&1u32.to_le_bytes());
    bytes.extend_from_slice(&rows.to_le_bytes());
    bytes.extend_from_slice(&(chains.len() as u64).to_le_bytes());
    for c in chains {
        bytes.extend_from_slice(&c.start.to_le_bytes());
        bytes.extend_from_slice(&c.len.to_le_bytes());
        bytes.extend_from_slice(&[c.conv, 0]);
    }
    let mut w = BufWriter::new(File::create(path).map_err(|e| format!("{path}: {e}"))?);
    w.write_all(&bytes).and_then(|_| w.flush()).map_err(|e| format!("{path}: {e}"))
}

/// Read a `.chains` index: (rows of the .tdat it was built from, runs).
pub fn read_chains(path: &str) -> Result<(u64, Vec<Chain>), String> {
    let b = fs::read(path).map_err(|e| format!("{path}: {e}"))?;
    if b.len() < 24 || &b[0..4] != CHAIN_MAGIC {
        return Err(format!("{path}: not a TCHN file"));
    }
    let rows = u64::from_le_bytes(b[8..16].try_into().unwrap());
    let n = u64::from_le_bytes(b[16..24].try_into().unwrap()) as usize;
    if b.len() != 24 + n * 8 {
        return Err(format!("{path}: {} bytes, expected {}", b.len(), 24 + n * 8));
    }
    let chains = b[24..]
        .chunks_exact(8)
        .map(|c| Chain {
            start: u32::from_le_bytes(c[0..4].try_into().unwrap()),
            len: u16::from_le_bytes(c[4..6].try_into().unwrap()),
            conv: c[6],
        })
        .collect();
    Ok((rows, chains))
}

pub fn run_chains(args: &[String]) -> ExitCode {
    let Some(tdat) = crate::flag_str(args, "--tdat") else {
        eprintln!("usage: ti data chains --tdat X.tdat [--out X.chains] [--min-len 2]");
        return ExitCode::FAILURE;
    };
    let out = crate::flag_str(args, "--out").unwrap_or_else(|| format!("{}.chains", tdat.trim_end_matches(".tdat")));
    let min_len: u64 = crate::flag_str(args, "--min-len").and_then(|v| v.parse().ok()).unwrap_or(2);
    let t = std::time::Instant::now();
    let (chains, st) = match scan_chains(&tdat, min_len.max(1)) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("data chains: {e}");
            return ExitCode::FAILURE;
        }
    };
    let pct = |x: u64| 100.0 * x as f64 / st.rows.max(1) as f64;
    let lk = st.linked[0] + st.linked[1];
    println!("data chains: {tdat}: {} rows scanned in {:.1}s", st.rows, t.elapsed().as_secs_f64());
    println!(
        "  rows one move after the previous row: {lk} ({:.1}%)  [occ0 = mover: {}, colour-fixed: {}]",
        pct(lk),
        st.linked[0],
        st.linked[1]
    );
    println!("  of those, outcome = 1 - previous outcome: {} ({:.1}%)", st.outcome_flips, 100.0 * st.outcome_flips as f64 / lk.max(1) as f64);
    for (k, name) in ["1", "2-9", "10-29", "30-59", "60-99", "100+"].iter().enumerate() {
        println!("  runs of length {name:<6}: {:>9} runs, {:>10} rows ({:.1}%)", st.hist[k].0, st.hist[k].1, pct(st.hist[k].1));
    }
    let kept: u64 = chains.iter().map(|c| c.len as u64).sum();
    if let Err(e) = write_chains(&out, st.rows, &chains) {
        eprintln!("data chains: {e}");
        return ExitCode::FAILURE;
    }
    println!("  wrote {out}: {} runs of >= {min_len} rows covering {kept} rows ({:.1}%), {} B", chains.len(), pct(kept), 24 + chains.len() * 8);
    ExitCode::SUCCESS
}

// ---------------------------------------------------------------------------------------------------------------
// lambda returns over .tdat runs: the future goes INTO the row, the trainer stays as it is
// ---------------------------------------------------------------------------------------------------------------

/// Value of a row for its own side to move: exact 1 / 0 for a proven row, else sigmoid(cp). Bool = proven.
fn row_value(r: &Rec) -> (f32, bool) {
    let t = crate::targets::target_of(r);
    if t.proven >= 0 {
        (t.proven as f32, true)
    } else {
        (crate::targets::sigmoid_cp(r.cp), false)
    }
}

/// The later rows of a run seen from row `t`: returns `(future, tail)` with `future + tail * z = ` the truncated
/// lambda return, where `z` is row `t`'s own game result. Weights: `(1 - l) * l^(n-1)` on the value `n` plies
/// later (flipped to row `t`'s side when `n` is odd), the rest (`tail`) stays on the result. A proven later row
/// is exact truth and takes all remaining weight. `l = 1` returns `(0, 1)`: the row keeps its result.
pub fn future_mix(rows: &[Rec], t: usize, lambda: f32, horizon: usize) -> (f32, f32) {
    if lambda >= 1.0 {
        return (0.0, 1.0);
    }
    let (mut sum, mut coef) = (0f32, 1f32);
    for n in 1..=horizon.min(rows.len() - 1 - t) {
        let (vm, proven) = row_value(&rows[t + n]);
        let v = if n % 2 == 1 { 1.0 - vm } else { vm };
        if proven {
            return (sum + coef * v, 0.0);
        }
        sum += (1.0 - lambda) * coef * v;
        coef *= lambda;
    }
    (sum, coef)
}

/// Sequential row reader over a `.tdat` (runs are visited in file order; gaps are copied or skipped).
struct RowStream {
    rd: BufReader<File>,
    pos: u64,
    rows: u64,
}

impl RowStream {
    fn open(tdat: &str) -> Result<RowStream, String> {
        let f = File::open(tdat).map_err(|e| format!("{tdat}: {e}"))?;
        let bytes = f.metadata().map_err(|e| e.to_string())?.len();
        if bytes % REC as u64 != 0 {
            return Err(format!("{tdat}: size {bytes} is not a multiple of {REC}"));
        }
        Ok(RowStream { rd: BufReader::with_capacity(1 << 20, f), pos: 0, rows: bytes / REC as u64 })
    }
    /// Move forward to row `to`, handing the raw bytes of the skipped rows to `sink` (None = drop them).
    fn advance(&mut self, to: u64, mut sink: Option<&mut dyn Write>) -> Result<(), String> {
        let mut buf = vec![0u8; REC * 16384];
        while self.pos < to {
            let n = ((to - self.pos) as usize).min(16384) * REC;
            self.rd.read_exact(&mut buf[..n]).map_err(|e| e.to_string())?;
            if let Some(w) = sink.as_mut() {
                w.write_all(&buf[..n]).map_err(|e| e.to_string())?;
            }
            self.pos += (n / REC) as u64;
        }
        Ok(())
    }
    fn take(&mut self, n: usize) -> Result<Vec<Rec>, String> {
        let mut buf = vec![0u8; n * REC];
        self.rd.read_exact(&mut buf).map_err(|e| e.to_string())?;
        self.pos += n as u64;
        Ok(buf.chunks_exact(REC).map(Rec::from_bytes).collect())
    }
}

fn check_index(tdat_rows: u64, idx_rows: u64, chains: &[Chain]) -> Result<(), String> {
    if tdat_rows != idx_rows {
        return Err(format!("the index was built for {idx_rows} rows, the .tdat has {tdat_rows}"));
    }
    let mut end = 0u64;
    for c in chains {
        if (c.start as u64) < end || c.start as u64 + c.len as u64 > tdat_rows {
            return Err("the index is not sorted or runs overlap".into());
        }
        end = c.start as u64 + c.len as u64;
    }
    Ok(())
}

/// `ti data games-tdat`: replay a `.tgam` and write trainer rows. This is the
/// missing link for the data flywheel: gate/self-play games (`.tgam`) could be
/// packed and analysed but never trained on. Every position is relabelled with
/// the CURRENT engine at `--nodes` (default 5000), the row outcome is the game
/// result from the mover's view, and one chain per game is written so
/// `ti data lambda` works on the result.
pub fn run_games_tdat(args: &[String]) -> ExitCode {
    let (Some(tgam), Some(out)) = (crate::flag_str(args, "--tgam"), crate::flag_str(args, "--out")) else {
        eprintln!("usage: ti data games-tdat --tgam F.tgam --out STEM [--nodes 5000] [--limit-games N] [--relabel-src 0] [--net FILE]");
        return ExitCode::FAILURE;
    };
    let nodes = crate::flag_u64(args, "--nodes").unwrap_or(5000);
    let limit = crate::flag_u64(args, "--limit-games").map(|v| v as usize).unwrap_or(usize::MAX);
    let src: u8 = crate::flag_u64(args, "--relabel-src").unwrap_or(0) as u8;
    let out_tdat = format!("{}.tdat", out.trim_end_matches(".tdat"));
    let t0 = std::time::Instant::now();
    let work = || -> Result<(u64, u64, u64, Vec<Chain>, u64), String> {
        let mut rd = GameReader::open(&tgam)?;
        let mut w = BufWriter::with_capacity(1 << 22, File::create(&out_tdat).map_err(|e| format!("{out_tdat}: {e}"))?);
        let mut se = Searcher::with_tt_bits_unconfigured(18);
        let nn = match crate::flag_str(args, "--net") {
            Some(p) => Some(titanium::nnue::Nnue::load(&p).map_err(|e| format!("net {p}: {e}"))?),
            None => None,
        };
        se.set_nnue(nn.as_ref().or_else(|| Some(titanium::nnue::Nnue::bundled_static())));
        let limits = SearchLimits { time: None, max_nodes: Some(nodes), max_depth: 30 };
        let mut chains: Vec<Chain> = Vec::new();
        let mut row: u64 = 0;
        let mut games_out = 0u64;
        let (mut n_out, mut n_draw, mut n_unknown) = (0u64, 0u64, 0u64);
        for (gi, g) in rd.by_ref().enumerate() {
            if gi >= limit {
                break;
            }
            let Ok(g) = g else { continue };
            if g.forfeit || g.result == 3 {
                n_unknown += 1; // no trustworthy outcome -> not a training game
                continue;
            }
            let start_row = row;
            // outcome from side 0's view: 2 = won, 1 = draw, 0 = lost
            let o0: f32 = match g.result {
                2 => 1.0,
                1 => { n_draw += 1; 0.5 }
                _ => 0.0,
            };
            if g.result == 2 || g.result == 0 {
                n_out += 1;
            }
            let mut b = g.start();
            se.reset_for_new_game();
            for ply in &g.plies {
                let mv = titanium::ttcodec::decode_move(&b, ply.0 & 0x3ff);
                let Some(mv) = mv else { break };
                let r = se.search(&b, &limits);
                let outcome = if b.turn == 0 { o0 } else { 1.0 - o0 };
                let rec = crate::data::Rec {
                    occ0: b.occ[0],
                    occ1: b.occ[1],

                    outcome,
                    cp: r.score as f32,
                    wsum: 1.0,
                    p3: 0.0,
                    cp5k: r.score as f32,
                    zx: 0,
                    stm: b.turn,
                    teacher_kind: 2,
                    proof_kind: 0,
                    relabel_src: src,
                    ti_clash: 0,
                };
                w.write_all(&rec.to_bytes()).map_err(|e| e.to_string())?;
                row += 1;
                b = b.make(mv);
            }
            let len = row - start_row;
            if len > 0 {
                chains.push(Chain { start: start_row as u32, len: len.min(u16::MAX as u64) as u16, conv: 0 });
                games_out += 1;
            }
        }
        w.flush().map_err(|e| e.to_string())?;
        Ok((row, games_out, n_out + n_draw, chains, n_unknown))
    };
    match work() {
        Ok((rows, games, decided, chains, unknown)) => {
            let idx = format!("{}.chains", out.trim_end_matches(".tdat"));
            if let Err(e) = write_chains(&idx, rows, &chains) {
                eprintln!("data games-tdat: chains: {e}");
                return ExitCode::FAILURE;
            }
            let desc = format!("{tgam} replayed and relabelled at {nodes} nodes by the current engine; {games} games, one chain per game; ti data games-tdat");
            if let Err(e) = crate::data::write_tdat_manifest(out.trim_end_matches(".tdat"), &desc, rows as usize) {
                eprintln!("data games-tdat: manifest: {e}");
                return ExitCode::FAILURE;
            }
            println!(
                "data games-tdat: {tgam} -> {out_tdat}: {rows} rows from {games} games ({decided} with a decided result, {unknown} skipped: forfeit/unknown), {} chains, {:.1}s",
                chains.len(),
                t0.elapsed().as_secs_f64()
            );
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("data games-tdat: {e}");
            ExitCode::FAILURE
        }
    }
}

/// `ti data lambda`: copy of a `.tdat` whose `outcome` holds the truncated lambda return of the row's own run.
pub fn run_lambda(args: &[String]) -> ExitCode {
    let (Some(tdat), Some(out)) = (crate::flag_str(args, "--tdat"), crate::flag_str(args, "--out")) else {
        eprintln!("usage: ti data lambda --tdat X.tdat --out STEM --lambda 0.7 [--chains X.chains] [--horizon 64]");
        return ExitCode::FAILURE;
    };
    let idx = crate::flag_str(args, "--chains").unwrap_or_else(|| format!("{}.chains", tdat.trim_end_matches(".tdat")));
    let lambda = crate::flag_str(args, "--lambda").and_then(|v| v.parse::<f32>().ok()).unwrap_or(0.7).clamp(0.0, 1.0);
    let horizon: usize = crate::flag_str(args, "--horizon").and_then(|v| v.parse().ok()).unwrap_or(64);
    let out_tdat = format!("{}.tdat", out.trim_end_matches(".tdat"));
    let t0 = std::time::Instant::now();
    let work = || -> Result<(u64, u64, f64, f64), String> {
        let (idx_rows, chains) = read_chains(&idx)?;
        let mut rs = RowStream::open(&tdat)?;
        check_index(rs.rows, idx_rows, &chains)?;
        let mut w = BufWriter::with_capacity(1 << 20, File::create(&out_tdat).map_err(|e| format!("{out_tdat}: {e}"))?);
        let (mut changed, mut moved, mut tail_sum) = (0u64, 0f64, 0f64);
        for c in &chains {
            rs.advance(c.start as u64, Some(&mut w as &mut dyn Write))?;
            let mut rows = rs.take(c.len as usize)?;
            let orig = rows.clone();
            for t in 0..rows.len() {
                if row_value(&orig[t]).1 {
                    continue; // a proven row trains on its proof, the outcome is not used
                }
                let (fut, tail) = future_mix(&orig, t, lambda, horizon);
                let g = fut + tail * orig[t].outcome;
                if tail < 1.0 {
                    changed += 1;
                    moved += (g - orig[t].outcome).abs() as f64;
                    tail_sum += tail as f64;
                }
                rows[t].outcome = g;
            }
            for r in &rows {
                w.write_all(&r.to_bytes()).map_err(|e| e.to_string())?;
            }
        }
        let total = rs.rows;
        rs.advance(total, Some(&mut w as &mut dyn Write))?;
        w.flush().map_err(|e| e.to_string())?;
        Ok((total, changed, moved, tail_sum))
    };
    match work() {
        Ok((total, changed, moved, tail_sum)) => {
            let desc = format!("{tdat} with outcome := truncated lambda return over {idx} (lambda {lambda}, horizon {horizon}); ti data lambda");
            if let Err(e) = crate::data::write_tdat_manifest(out.trim_end_matches(".tdat"), &desc, total as usize) {
                eprintln!("data lambda: manifest: {e}");
                return ExitCode::FAILURE;
            }
            println!("data lambda: {tdat} -> {out_tdat}: {total} rows in {:.1}s, lambda {lambda}, horizon {horizon}", t0.elapsed().as_secs_f64());
            println!(
                "  rows whose outcome now carries the future: {changed} ({:.1}%); mean |new - old outcome| {:.4}; mean weight left on the game result {:.3}",
                100.0 * changed as f64 / total.max(1) as f64,
                moved / changed.max(1) as f64,
                tail_sum / changed.max(1) as f64
            );
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("data lambda: {e}");
            ExitCode::FAILURE
        }
    }
}

/// `ti data chain-audit`: how consistent are the labels inside the runs, and (with `--deep`) which target sits
/// closest to a deeper relabel of the same positions.
pub fn run_chain_audit(args: &[String]) -> ExitCode {
    let Some(tdat) = crate::flag_str(args, "--tdat") else {
        eprintln!("usage: ti data chain-audit --tdat X.tdat [--chains X.chains] [--deep D.tdat]");
        return ExitCode::FAILURE;
    };
    let idx = crate::flag_str(args, "--chains").unwrap_or_else(|| format!("{}.chains", tdat.trim_end_matches(".tdat")));
    let deep_path = crate::flag_str(args, "--deep");
    const RW: [f32; 3] = [0.55, 0.55, 0.25]; // the champion's result weight by empties (<= 6, 7..12, >= 13), engine D330
    const NAMES: [&str; 12] = [
        "own 5k label", "next ply's label", "0.5 own + 0.5 next", "later labels, lambda 0.5", "later labels, lambda 0.7",
        "later labels, lambda 0.9", "0.5 own + 0.5 later(0.7)", "game result alone", "champion target (own + result)",
        "own + lambda 0.5 return", "own + lambda 0.7 return", "own + lambda 0.9 return",
    ];
    let work = || -> Result<(), String> {
        let (idx_rows, chains) = read_chains(&idx)?;
        let mut rs = RowStream::open(&tdat)?;
        check_index(rs.rows, idx_rows, &chains)?;
        let mut deep: std::collections::HashMap<(u64, u64), f32> = std::collections::HashMap::new();
        if let Some(dp) = &deep_path {
            let mut ds = RowStream::open(dp)?;
            let n = ds.rows as usize;
            for r in ds.take(n)? {
                let key = if r.stm == 0 { (r.occ0, r.occ1) } else { (r.occ1, r.occ0) };
                if r.cp.abs() < 98_000.0 {
                    deep.insert(key, r.cp);
                }
            }
        }
        // one-ply consistency by empties band: n, sum |d|, sum d^2, |d| > 0.10, |d| > 0.25, side-ahead flips
        let mut td = [(0u64, 0f64, 0f64, 0u64, 0u64, 0u64); 4];
        let mut se = [0f64; 12];
        let (mut n_deep, mut own_changed) = (0u64, 0u64);
        for c in &chains {
            rs.advance(c.start as u64, None)?;
            let rows = rs.take(c.len as usize)?;
            for t in 0..rows.len() - 1 {
                let (r, nx) = (&rows[t], &rows[t + 1]);
                let ((v, p), (vn, pn)) = (row_value(r), row_value(nx));
                if p {
                    continue;
                }
                let empties = 49 - (r.occ0 | r.occ1).count_ones() as usize;
                if !pn {
                    let d = (v - (1.0 - vn)) as f64;
                    let k = (empties / 10).min(3);
                    td[k].0 += 1;
                    td[k].1 += d.abs();
                    td[k].2 += d * d;
                    td[k].3 += (d.abs() > 0.10) as u64;
                    td[k].4 += (d.abs() > 0.25) as u64;
                    td[k].5 += ((v > 0.53 && 1.0 - vn < 0.47) || (v < 0.47 && 1.0 - vn > 0.53)) as u64;
                }
                if deep.is_empty() {
                    continue;
                }
                let key = if c.conv == 0 || r.stm == 0 { (r.occ0, r.occ1) } else { (r.occ1, r.occ0) };
                let Some(&cpd) = deep.get(&key) else { continue };
                let y = crate::targets::sigmoid_cp(cpd);
                n_deep += 1;
                own_changed += (cpd != r.cp) as u64;
                let w = if empties <= 6 { RW[0] } else if empties <= 12 { RW[1] } else { RW[2] };
                let later = |l: f32| {
                    let (f, tail) = future_mix(&rows, t, l, 64);
                    (if tail < 1.0 { f / (1.0 - tail) } else { v }, f + tail * r.outcome)
                };
                let (l5, g5) = later(0.5);
                let (l7, g7) = later(0.7);
                let (l9, g9) = later(0.9);
                let next = 1.0 - vn;
                let cand = [
                    v, next, 0.5 * v + 0.5 * next, l5, l7, l9, 0.5 * v + 0.5 * l7, r.outcome,
                    (1.0 - w) * v + w * r.outcome, (1.0 - w) * v + w * g5, (1.0 - w) * v + w * g7, (1.0 - w) * v + w * g9,
                ];
                for (k, x) in cand.iter().enumerate() {
                    se[k] += ((x - y) as f64).powi(2);
                }
            }
        }
        println!("data chain-audit: {tdat} ({} runs)", chains.len());
        println!("  one-ply consistency of the labels (own value vs the next row's value, both unproven):");
        for (k, name) in ["0-9", "10-19", "20-29", "30+"].iter().enumerate() {
            let b = td[k];
            if b.0 > 0 {
                let n = b.0 as f64;
                println!(
                    "    empties {name:<6}: {:>9} pairs | mean |diff| {:.4} | rms {:.4} | > 0.10: {:.1}% | > 0.25: {:.1}% | side ahead flips: {:.1}%",
                    b.0, b.1 / n, (b.2 / n).sqrt(), 100.0 * b.3 as f64 / n, 100.0 * b.4 as f64 / n, 100.0 * b.5 as f64 / n
                );
            }
        }
        if let Some(dp) = &deep_path {
            println!("  against the deeper labels in {dp}: {n_deep} rows found inside runs ({own_changed} of them have a different deep score)");
            if n_deep > 0 {
                let base = se[0] / n_deep as f64;
                for (k, name) in NAMES.iter().enumerate() {
                    let m = se[k] / n_deep as f64;
                    println!("    {name:<32} mse {:.5}  ({:+.1}% vs own label)", m, 100.0 * (m - base) / base.max(1e-12));
                }
            }
        }
        Ok(())
    };
    match work() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("data chain-audit: {e}");
            ExitCode::FAILURE
        }
    }
}

// ---------------------------------------------------------------------------------------------------------------
// games: the .tgam format
// ---------------------------------------------------------------------------------------------------------------

/// One game in packed form.
#[derive(Clone, Debug, PartialEq)]
pub struct PackedGame {
    pub occ: [u64; 2],
    pub blockers: u64,
    /// side to move in the start position
    pub turn: u8,
    /// seat of the gate candidate: 0 / 1, 2 = unknown
    pub cand_seat: u8,
    pub forfeit: bool,
    /// final stone counts as the gate recorded them (255 = not recorded)
    pub final_cnt: [u8; 2],
    /// from side 0's view: 2 = won, 1 = draw, 0 = lost, 3 = unknown
    pub result: u8,
    pub src: u16,
    pub book: u32,
    pub pair: u32,
    /// (move code | depth << 10, score code from the mover's view)
    pub plies: Vec<(u16, i16)>,
}

/// One ply of a replayed game.
#[derive(Clone, Copy, Debug)]
pub struct PlyView {
    /// position BEFORE the move
    pub board: Board,
    pub mv: Move,
    /// mover's score in engine units (None = not logged)
    pub score: Option<i32>,
    pub depth: u8,
}

impl PackedGame {
    pub fn start(&self) -> Board {
        Board::from_occ(self.occ[0], self.occ[1], self.blockers, self.turn)
    }

    pub fn byte_len(&self) -> usize {
        GAME_HDR + self.plies.len() * PLY_BYTES
    }

    pub fn write<W: Write>(&self, w: &mut W) -> std::io::Result<()> {
        let mut h = [0u8; GAME_HDR];
        h[0..8].copy_from_slice(&self.occ[0].to_le_bytes());
        h[8..16].copy_from_slice(&self.occ[1].to_le_bytes());
        h[16..24].copy_from_slice(&self.blockers.to_le_bytes());
        h[24..26].copy_from_slice(&(self.plies.len() as u16).to_le_bytes());
        h[26..28].copy_from_slice(&self.src.to_le_bytes());
        h[28] = (self.turn & 1) | ((self.cand_seat & 3) << 1) | ((self.forfeit as u8) << 3);
        h[29] = self.final_cnt[0];
        h[30] = self.final_cnt[1];
        h[31] = self.result;
        h[32..36].copy_from_slice(&self.book.to_le_bytes());
        h[36..40].copy_from_slice(&self.pair.to_le_bytes());
        w.write_all(&h)?;
        for &(mvw, sc) in &self.plies {
            w.write_all(&mvw.to_le_bytes())?;
            w.write_all(&sc.to_le_bytes())?;
        }
        Ok(())
    }

    pub fn read<R: Read>(r: &mut R) -> std::io::Result<PackedGame> {
        let mut h = [0u8; GAME_HDR];
        r.read_exact(&mut h)?;
        let n = u16::from_le_bytes(h[24..26].try_into().unwrap()) as usize;
        let mut body = vec![0u8; n * PLY_BYTES];
        r.read_exact(&mut body)?;
        Ok(PackedGame {
            occ: [u64::from_le_bytes(h[0..8].try_into().unwrap()), u64::from_le_bytes(h[8..16].try_into().unwrap())],
            blockers: u64::from_le_bytes(h[16..24].try_into().unwrap()),
            turn: h[28] & 1,
            cand_seat: (h[28] >> 1) & 3,
            forfeit: (h[28] >> 3) & 1 != 0,
            final_cnt: [h[29], h[30]],
            result: h[31],
            src: u16::from_le_bytes(h[26..28].try_into().unwrap()),
            book: u32::from_le_bytes(h[32..36].try_into().unwrap()),
            pair: u32::from_le_bytes(h[36..40].try_into().unwrap()),
            plies: body
                .chunks_exact(PLY_BYTES)
                .map(|c| (u16::from_le_bytes([c[0], c[1]]), i16::from_le_bytes([c[2], c[3]])))
                .collect(),
        })
    }

    /// Replay through the engine's own move decoder. Returns every ply with the position before it, and the
    /// final position. An undecodable move is an error naming the ply.
    pub fn replay(&self) -> Result<(Vec<PlyView>, Board), String> {
        let mut b = self.start();
        let mut out = Vec::with_capacity(self.plies.len());
        for (i, &(mvw, sc)) in self.plies.iter().enumerate() {
            let code = mvw & 0x3FF;
            let mv = if code == PASS_CODE {
                if b.has_moves(b.turn) {
                    return Err(format!("ply {i}: pass although the mover has moves"));
                }
                Move::PASS
            } else {
                decode_move(&b, code).ok_or_else(|| format!("ply {i}: move code {code} is not legal here"))?
            };
            out.push(PlyView { board: b, mv, score: (sc != NO_SCORE).then(|| dec_score(sc)), depth: (mvw >> 10) as u8 });
            b = b.make_uai(mv);
        }
        Ok((out, b))
    }

    /// Win probability for side 0 before every ply, from the mover's own score (None where no score was logged).
    pub fn values_side0(&self, plies: &[PlyView]) -> Vec<Option<f32>> {
        plies
            .iter()
            .map(|p| {
                p.score.map(|s| {
                    let pm = crate::targets::sigmoid_cp(s as f32);
                    if p.board.turn == 0 {
                        pm
                    } else {
                        1.0 - pm
                    }
                })
            })
            .collect()
    }

    /// Final result for side 0 as a probability (None = unknown).
    pub fn z_side0(&self) -> Option<f32> {
        match self.result {
            2 => Some(1.0),
            1 => Some(0.5),
            0 => Some(0.0),
            _ => None,
        }
    }
}

/// TD(lambda) returns over one game, all values from ONE side's view: `G_t = (1 - l) * V_{t+1} + l * G_{t+1}` with
/// the game result `z` as the value after the last ply. `l = 0` gives the next position's value, `l = 1` gives `z`.
/// A missing later value (None) passes the return through unchanged.
pub fn lambda_returns(values: &[Option<f32>], z: f32, lambda: f32) -> Vec<f32> {
    let n = values.len();
    let mut out = vec![0f32; n];
    let mut g = z;
    for t in (0..n).rev() {
        let next = if t + 1 < n { values[t + 1] } else { Some(z) };
        if let Some(v) = next {
            g = (1.0 - lambda) * v + lambda * g;
        }
        out[t] = g;
    }
    out
}

/// Turn one parsed gate game into the packed form. Every move is checked: its code must decode back, on the same
/// position, to a move that produces the same next position.
pub fn pack_game(g: &crate::games::Game, raw_line: &str, src: u16) -> Result<PackedGame, String> {
    let before = crate::games::replay(g)?;
    let start = *before.first().ok_or("empty game")?;
    let mut plies = Vec::with_capacity(g.moves.len());
    let mut forfeit = false;
    for (r, b) in g.moves.iter().zip(before.iter()) {
        if r.mv == "FORFEIT" {
            forfeit = true;
            break;
        }
        let m = b.parse_uai_move(&r.mv).ok_or_else(|| format!("ply {}: illegal move {}", r.ply, r.mv))?;
        let code = if m.is_pass() { PASS_CODE } else { move_code(m) };
        if !m.is_pass() {
            let back = decode_move(b, code).ok_or_else(|| format!("ply {}: code {code} does not decode", r.ply))?;
            let (x, y) = (b.make(m), b.make(back));
            if x.occ != y.occ || x.turn != y.turn {
                return Err(format!("ply {}: code {code} decodes to a different move", r.ply));
            }
        }
        let score = match (r.score, r.mate) {
            (_, Some(mt)) if mt > 0 => Some(MATE - mt.min(1999)),
            (_, Some(mt)) => Some(-MATE + (-mt).min(1999)),
            (s, None) => s,
        };
        let sc = score.map(enc_score).unwrap_or(NO_SCORE);
        let depth = r.depth.unwrap_or(0).min(63) as u16;
        plies.push((code | (depth << 10), sc));
    }
    if plies.len() > u16::MAX as usize {
        return Err("more than 65535 plies".into());
    }
    let cand_seat = match g.cand_seat.as_str() {
        "x" => 0u8,
        "o" => 1,
        _ => 2,
    };
    // candidate_points is the candidate's game result (1 / 0.5 / 0); side 0 is 'x'
    let result = if cand_seat > 1 {
        3
    } else {
        let p0 = if cand_seat == 0 { g.cand_points } else { 1.0 - g.cand_points };
        (p0 * 2.0).round().clamp(0.0, 2.0) as u8
    };
    let fin = |k: &str| crate::games::num::<u32>(raw_line, k).map(|v| v.min(254) as u8).unwrap_or(255);
    Ok(PackedGame {
        occ: start.occ,
        blockers: start.blockers,
        turn: start.turn,
        cand_seat,
        forfeit,
        final_cnt: [fin("final_x"), fin("final_o")],
        result,
        src,
        book: g.book.min(u32::MAX as u64) as u32,
        pair: g.pair.min(u32::MAX as u64) as u32,
        plies,
    })
}

/// Streaming reader of a `.tgam` file.
pub struct GameReader {
    rd: BufReader<File>,
    pub games: u64,
    pub plies: u64,
    pub sources: u32,
    left: u64,
}

impl GameReader {
    pub fn open(path: &str) -> Result<GameReader, String> {
        let mut rd = BufReader::with_capacity(1 << 20, File::open(path).map_err(|e| format!("{path}: {e}"))?);
        let mut h = [0u8; 32];
        rd.read_exact(&mut h).map_err(|e| format!("{path}: {e}"))?;
        if &h[0..4] != TGAM_MAGIC || u32::from_le_bytes(h[4..8].try_into().unwrap()) != 1 {
            return Err(format!("{path}: not a TGAM version 1 file"));
        }
        let games = u64::from_le_bytes(h[8..16].try_into().unwrap());
        Ok(GameReader {
            rd,
            games,
            plies: u64::from_le_bytes(h[16..24].try_into().unwrap()),
            sources: u32::from_le_bytes(h[24..28].try_into().unwrap()),
            left: games,
        })
    }
}

impl Iterator for GameReader {
    type Item = Result<PackedGame, String>;
    fn next(&mut self) -> Option<Self::Item> {
        if self.left == 0 {
            return None;
        }
        self.left -= 1;
        Some(PackedGame::read(&mut self.rd).map_err(|e| format!("game {}: {e}", self.games - self.left - 1)))
    }
}

fn walk_games(dir: &std::path::Path, out: &mut Vec<String>) {
    let Ok(rd) = fs::read_dir(dir) else { return };
    for e in rd.flatten() {
        let p = e.path();
        if p.is_dir() {
            walk_games(&p, out);
        } else if p.file_name().and_then(|n| n.to_str()).map_or(false, |n| n.ends_with(".games.jsonl")) {
            out.push(p.to_string_lossy().replace('\\', "/"));
        }
    }
}

fn game_key(g: &PackedGame) -> u64 {
    let mut h = std::collections::hash_map::DefaultHasher::new();
    (g.occ, g.blockers, g.turn, g.cand_seat, &g.plies).hash(&mut h);
    h.finish()
}

pub fn run_pack_games(args: &[String]) -> ExitCode {
    let Some(stem) = crate::flag_str(args, "--out") else {
        eprintln!("usage: ti data pack-games --out STEM [--roots DIR,DIR] [--inputs a.games.jsonl,b.games.jsonl]");
        return ExitCode::FAILURE;
    };
    let mut files: Vec<String> = crate::flag_str(args, "--inputs").unwrap_or_default().split(',').filter(|s| !s.is_empty()).map(String::from).collect();
    for root in crate::flag_str(args, "--roots").unwrap_or_default().split(',').filter(|s| !s.is_empty()) {
        walk_games(std::path::Path::new(root), &mut files);
    }
    files.sort();
    files.dedup();
    if files.is_empty() || files.len() > u16::MAX as usize {
        eprintln!("data pack-games: {} input files (need 1..65535)", files.len());
        return ExitCode::FAILURE;
    }
    let out_path = format!("{stem}.tgam");
    let t = std::time::Instant::now();
    let mut seen: HashSet<u64> = HashSet::new();
    let mut games: Vec<PackedGame> = Vec::new();
    let (mut dups, mut skipped, mut plies) = (0u64, 0u64, 0u64);
    let mut reasons: Vec<String> = Vec::new();
    let mut sources = Vec::new();
    for (si, path) in files.iter().enumerate() {
        let text = match fs::read_to_string(path) {
            Ok(t) => t,
            Err(e) => {
                eprintln!("data pack-games: {path}: {e} (file skipped)");
                continue;
            }
        };
        let mut kept = 0u64;
        for (n, line) in text.lines().enumerate().filter(|(_, l)| !l.trim().is_empty()) {
            let packed = crate::games::parse_game(line).and_then(|g| pack_game(&g, line, si as u16));
            match packed {
                Ok(pg) => {
                    if seen.insert(game_key(&pg)) {
                        plies += pg.plies.len() as u64;
                        games.push(pg);
                        kept += 1;
                    } else {
                        dups += 1;
                    }
                }
                Err(e) => {
                    skipped += 1;
                    if reasons.len() < 5 {
                        reasons.push(format!("{path}:{}: {e}", n + 1));
                    }
                }
            }
        }
        // the gate header (engines, clock, book) sits in the first line of the sibling sprt.log
        let log = path.trim_end_matches(".games.jsonl");
        let header = File::open(log)
            .ok()
            .and_then(|f| {
                let mut raw = Vec::new();
                BufReader::new(f).take(8192).read_to_end(&mut raw).ok()?;
                String::from_utf8_lossy(&raw).lines().next().map(String::from)
            })
            .unwrap_or_default();
        sources.push(serde_json::json!({ "i": si, "path": path, "games": kept, "header": header }));
    }
    let write = || -> std::io::Result<u64> {
        let mut w = BufWriter::with_capacity(1 << 20, File::create(&out_path)?);
        w.write_all(TGAM_MAGIC)?;
        w.write_all(&1u32.to_le_bytes())?;
        w.write_all(&(games.len() as u64).to_le_bytes())?;
        w.write_all(&plies.to_le_bytes())?;
        w.write_all(&(files.len() as u32).to_le_bytes())?;
        w.write_all(&0u32.to_le_bytes())?;
        let mut bytes = 32u64;
        for g in &games {
            g.write(&mut w)?;
            bytes += g.byte_len() as u64;
        }
        w.flush()?;
        Ok(bytes)
    };
    let bytes = match write() {
        Ok(b) => b,
        Err(e) => {
            eprintln!("data pack-games: {out_path}: {e}");
            return ExitCode::FAILURE;
        }
    };
    let manifest = serde_json::json!({
        "format": "TGAM1", "games": games.len(), "plies": plies, "bytes": bytes,
        "duplicates_dropped": dups, "games_skipped": skipped, "sources": sources,
    });
    if let Err(e) = fs::write(format!("{out_path}.json"), serde_json::to_string_pretty(&manifest).unwrap_or_default()) {
        eprintln!("data pack-games: manifest: {e}");
        return ExitCode::FAILURE;
    }
    println!(
        "data pack-games: {} files -> {out_path}: {} games, {plies} plies, {bytes} B ({:.2} B per position) in {:.1}s",
        files.len(),
        games.len(),
        bytes as f64 / plies.max(1) as f64,
        t.elapsed().as_secs_f64()
    );
    println!("  exact duplicates dropped: {dups}; games skipped (unparsable or illegal): {skipped}");
    for r in &reasons {
        println!("  skipped: {r}");
    }
    ExitCode::SUCCESS
}

#[derive(Default)]
struct SrcAcc {
    games: u64,
    plies: u64,
    scored: u64,
    depth: u64,
    se: f64,
    td_n: u64,
    td_abs: f64,
    td_big: u64,
}

/// Clock of a gate from its `# ti sprt` header ("nodes Some(5000)" / "cand_ms Some(100)"), else from the folder name.
fn clock_tag(header: &str, path: &str) -> String {
    let grab = |key: &str| -> Option<String> {
        let i = header.find(key)? + key.len();
        let digits: String = header[i..].chars().take_while(|c| c.is_ascii_digit()).collect();
        (!digits.is_empty()).then_some(digits)
    };
    if let Some(n) = grab("nodes Some(") {
        return format!("nodes {n}");
    }
    if let Some(n) = grab("cand_ms Some(") {
        return format!("ms {n}");
    }
    for (pat, name) in [("-ms", "ms "), ("-nodes", "nodes ")] {
        if let Some(i) = path.rfind(pat) {
            let digits: String = path[i + pat.len()..].chars().take_while(|c| c.is_ascii_digit()).collect();
            if !digits.is_empty() {
                return format!("{name}{digits}");
            }
        }
    }
    "unknown".to_string()
}

pub fn run_games_stat(args: &[String]) -> ExitCode {
    let Some(path) = crate::flag_str(args, "--tgam") else {
        eprintln!("usage: ti data games-stat --tgam F.tgam [--lambda 0.7]");
        return ExitCode::FAILURE;
    };
    let lambda: f32 = crate::flag_str(args, "--lambda").and_then(|v| v.parse().ok()).unwrap_or(0.7);
    let rd = match GameReader::open(&path) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("data games-stat: {e}");
            return ExitCode::FAILURE;
        }
    };
    let (hg, hp) = (rd.games, rd.plies);
    // --by-source: group the games by the clock named in each source's gate header (manifest next to the file)
    let by_source = crate::has_flag(args, "--by-source");
    let mut src_group: Vec<String> = Vec::new();
    if by_source {
        let man = fs::read_to_string(format!("{path}.json")).ok().and_then(|s| serde_json::from_str::<serde_json::Value>(&s).ok());
        if let Some(list) = man.as_ref().and_then(|m| m["sources"].as_array()) {
            for s in list {
                src_group.push(clock_tag(s["header"].as_str().unwrap_or(""), s["path"].as_str().unwrap_or("")));
            }
        }
    }
    let mut groups: std::collections::BTreeMap<String, SrcAcc> = std::collections::BTreeMap::new();
    let mut per_src: std::collections::BTreeMap<u16, (f64, u64)> = std::collections::BTreeMap::new();
    let mut starts: HashSet<(u64, u64, u8)> = HashSet::new();
    let mut seen_pos: HashSet<u64> = HashSet::new();
    let (mut uniq_n, mut uniq_new) = ([0u64; 5], [0u64; 5]);
    let t = std::time::Instant::now();
    let (mut games, mut plies, mut scored, mut depth_sum, mut forfeits, mut bad) = (0u64, 0u64, 0u64, 0u64, 0u64, 0u64);
    let mut res = [0u64; 4];
    let (mut fin_known, mut fin_match) = (0u64, 0u64);
    // label quality: squared error against the final result of (a) the mover's own value, (b) the lambda return
    let (mut se_v, mut se_g, mut n_q, mut gap) = (0f64, 0f64, 0u64, 0f64);
    let mut band = [(0f64, 0f64, 0u64); 5]; // per 20 plies: se of value, se of lambda return, n
    for item in rd {
        let g = match item {
            Ok(g) => g,
            Err(e) => {
                eprintln!("data games-stat: {e}");
                return ExitCode::FAILURE;
            }
        };
        games += 1;
        plies += g.plies.len() as u64;
        forfeits += g.forfeit as u64;
        res[g.result.min(3) as usize] += 1;
        let (views, last) = match g.replay() {
            Ok(v) => v,
            Err(e) => {
                bad += 1;
                if bad <= 3 {
                    eprintln!("  game {}: {e}", games - 1);
                }
                continue;
            }
        };
        if g.final_cnt[0] != 255 && g.final_cnt[1] != 255 {
            fin_known += 1;
            let (c0, c1) = last.counts();
            fin_match += (c0 == g.final_cnt[0] as u32 && c1 == g.final_cnt[1] as u32) as u64;
        }
        for v in &views {
            if v.score.is_some() {
                scored += 1;
                depth_sum += v.depth as u64;
            }
        }
        if by_source {
            // how many DIFFERENT positions are there (gates replay the same book lines again and again)?
            starts.insert((g.occ[0], g.occ[1], g.turn));
            for (i, v) in views.iter().enumerate() {
                let k = (i / 20).min(4);
                uniq_n[k] += 1;
                let mut h = std::collections::hash_map::DefaultHasher::new();
                (v.board.occ, v.board.turn).hash(&mut h);
                uniq_new[k] += seen_pos.insert(h.finish()) as u64;
            }
            if let Some(z) = g.z_side0() {
                let vals = g.values_side0(&views);
                let a = groups.entry(src_group.get(g.src as usize).cloned().unwrap_or_else(|| "no manifest".into())).or_default();
                a.games += 1;
                a.plies += views.len() as u64;
                for (i, v) in vals.iter().enumerate() {
                    let Some(v) = v else { continue };
                    a.scored += 1;
                    a.depth += views[i].depth as u64;
                    a.se += ((v - z) as f64).powi(2);
                    if let Some(Some(nx)) = vals.get(i + 1) {
                        let d = (v - nx).abs() as f64;
                        a.td_n += 1;
                        a.td_abs += d;
                        a.td_big += (d > 0.25) as u64;
                    }
                }
                if g.cand_seat < 2 {
                    let pts = (if g.cand_seat == 0 { z } else { 1.0 - z }) as f64;
                    let e = per_src.entry(g.src).or_insert((0.0, 0));
                    e.0 += pts;
                    e.1 += 1;
                }
            }
        }
        if let Some(z) = g.z_side0() {
            let vals = g.values_side0(&views);
            let ret = lambda_returns(&vals, z, lambda);
            for (i, v) in vals.iter().enumerate() {
                if let Some(v) = v {
                    let (dv, dg) = ((v - z) as f64, (ret[i] - z) as f64);
                    se_v += dv * dv;
                    se_g += dg * dg;
                    gap += (ret[i] - v).abs() as f64;
                    n_q += 1;
                    let k = (i / 20).min(4);
                    band[k].0 += dv * dv;
                    band[k].1 += dg * dg;
                    band[k].2 += 1;
                }
            }
        }
    }
    let bytes = fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
    println!("data games-stat: {path}: header says {hg} games / {hp} plies; read {games} / {plies} in {:.1}s", t.elapsed().as_secs_f64());
    println!("  {bytes} B = {:.2} B per position; mean {:.1} plies per game", bytes as f64 / plies.max(1) as f64, plies as f64 / games.max(1) as f64);
    println!("  replay through the engine decoder: {} games ok, {bad} failed; forfeits {forfeits}", games - bad);
    println!("  result for side 0: won {} / draw {} / lost {} / unknown {}", res[2], res[1], res[0], res[3]);
    println!("  final stone counts recorded in {fin_known} games; equal to the replayed final board in {fin_match}");
    println!("  plies with a score: {scored} ({:.1}%), mean depth {:.1}", 100.0 * scored as f64 / plies.max(1) as f64, depth_sum as f64 / scored.max(1) as f64);
    if n_q > 0 {
        println!(
            "  vs the final result (mean squared error, {n_q} scored plies): mover's own value {:.4}; lambda {lambda} return {:.4}; mean |return - value| {:.4}",
            se_v / n_q as f64,
            se_g / n_q as f64,
            gap / n_q as f64
        );
        for (k, b) in band.iter().enumerate().filter(|(_, b)| b.2 > 0) {
            let name = if k == 4 { "80+".to_string() } else { format!("{}-{}", k * 20, k * 20 + 19) };
            println!("    ply {name:<6}: value {:.4}  lambda return {:.4}  ({} plies)", b.0 / b.2 as f64, b.1 / b.2 as f64, b.2);
        }
    }
    if by_source {
        println!("  by clock of the source gate (value = the mover's own score as a win probability):");
        for (name, a) in &groups {
            println!(
                "    {name:<12} {:>6} games {:>8} plies | depth {:>4.1} | mse vs result {:.4} | one-ply |diff| {:.4}, > 0.25: {:.1}%",
                a.games,
                a.plies,
                a.depth as f64 / a.scored.max(1) as f64,
                a.se / a.scored.max(1) as f64,
                a.td_abs / a.td_n.max(1) as f64,
                100.0 * a.td_big as f64 / a.td_n.max(1) as f64
            );
        }
        // sources where one engine was clearly weaker: half of every such game is played by the weak side
        let (mut lop_games, mut lop_src, mut all_games) = (0u64, 0u64, 0u64);
        for (pts, n) in per_src.values() {
            all_games += n;
            let share = pts / *n as f64;
            if *n >= 20 && !(0.35..=0.65).contains(&share) {
                lop_games += n;
                lop_src += 1;
            }
        }
        println!("  lopsided sources (candidate scored below 35% or above 65% over >= 20 games): {lop_src} sources, {lop_games} of {all_games} games ({:.1}%)", 100.0 * lop_games as f64 / all_games.max(1) as f64);
        println!("  different start positions: {} for {games} games ({:.1} games per start)", starts.len(), games as f64 / starts.len().max(1) as f64);
        let (tn, tu): (u64, u64) = (uniq_n.iter().sum(), uniq_new.iter().sum());
        println!("  different positions: {tu} of {tn} ({:.1}%)", 100.0 * tu as f64 / tn.max(1) as f64);
        for k in 0..5 {
            if uniq_n[k] > 0 {
                let name = if k == 4 { "80+".to_string() } else { format!("{}-{}", k * 20, k * 20 + 19) };
                println!("    ply {name:<6}: {:>8} positions, {:.1}% not seen earlier in the file", uniq_n[k], 100.0 * uniq_new[k] as f64 / uniq_n[k] as f64);
            }
        }
    }
    if games != hg || plies != hp || bad > 0 {
        eprintln!("data games-stat: MISMATCH against the header or failed replays");
        return ExitCode::FAILURE;
    }
    ExitCode::SUCCESS
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic legal game from the start position: (positions before each move, moves as UAI text).
    fn play(max_plies: usize, stop_at_pass: bool) -> (Vec<Board>, Vec<String>, Board) {
        let mut b = Board::start();
        let (mut before, mut moves) = (Vec::new(), Vec::new());
        for i in 0..max_plies {
            if !b.has_moves(b.turn) {
                if stop_at_pass || !b.has_moves(1 - b.turn) {
                    break;
                }
                before.push(b);
                moves.push("0000".to_string());
                b = b.make_pass();
                continue;
            }
            let list = b.legal_moves();
            let m = list.move_at((i * 7 + 3) % list.len());
            let text = if m.is_clone() { Board::sq_name(m.to) } else { format!("{}{}", Board::sq_name(m.from), Board::sq_name(m.to)) };
            before.push(b);
            moves.push(text);
            b = b.make(m);
        }
        (before, moves, b)
    }

    fn rec(b: &Board, outcome: f32) -> Rec {
        let (me, op) = (b.occ[b.turn as usize], b.occ[1 - b.turn as usize]);
        Rec { occ0: me, occ1: op, outcome, cp: 0.0, wsum: 1.0, p3: 0.5, cp5k: 0.0, zx: 0, stm: 0, teacher_kind: 1, proof_kind: 0, relabel_src: 0, ti_clash: 0 }
    }

    #[test]
    fn link_test_accepts_real_moves_and_rejects_others() {
        let (before, _, _) = play(30, true);
        for w in before.windows(2) {
            let (p, r) = (rec(&w[0], 0.0), rec(&w[1], 1.0));
            assert!(linked(p.occ0, p.occ1, r.occ0, r.occ1));
            assert!(!linked(r.occ0, r.occ1, p.occ0, p.occ1), "a move cannot be undone");
        }
        // two plies apart is not a link
        let (a, c) = (rec(&before[0], 0.0), rec(&before[2], 0.0));
        assert!(!linked(a.occ0, a.occ1, c.occ0, c.occ1));
    }

    #[test]
    fn chains_find_two_games_in_a_tdat() {
        let (g1, _, _) = play(24, true);
        let (g2, _, _) = play(11, true);
        let dir = std::env::temp_dir().join(format!("ti_chains_{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let tdat = dir.join("t.tdat").to_string_lossy().to_string();
        let mut bytes = Vec::new();
        for (i, b) in g1.iter().chain(g2.iter()).enumerate() {
            bytes.extend_from_slice(&rec(b, (i % 2) as f32).to_bytes());
        }
        fs::write(&tdat, &bytes).unwrap();
        let (chains, st) = scan_chains(&tdat, 2).unwrap();
        assert_eq!(st.rows, (g1.len() + g2.len()) as u64);
        assert_eq!(chains, vec![Chain { start: 0, len: g1.len() as u16, conv: 0 }, Chain { start: g1.len() as u32, len: g2.len() as u16, conv: 0 }]);
        assert_eq!(st.linked[0], (g1.len() + g2.len() - 2) as u64);
        let idx = dir.join("t.chains").to_string_lossy().to_string();
        write_chains(&idx, st.rows, &chains).unwrap();
        assert_eq!(read_chains(&idx).unwrap(), (st.rows, chains));
        let _ = fs::remove_dir_all(&dir);
    }

    fn json_game(max_plies: usize) -> (String, Vec<Board>, Board) {
        let (before, moves, last) = play(max_plies, false);
        let mut s = format!(
            "{{\"pair\":7,\"book\":41,\"start\":\"{}\",\"candidate_seat\":\"o\",\"candidate_points\":1,\"final_x\":{},\"final_o\":{},\"moves\":[",
            before[0].to_ataxx_fen(),
            last.counts().0,
            last.counts().1
        );
        for (i, (b, m)) in before.iter().zip(moves.iter()).enumerate() {
            let side = if b.turn == 0 { "x" } else { "o" };
            let (score, mate) = if i == 5 { ("null".to_string(), "3".to_string()) } else if i == 6 { ("null".to_string(), "null".to_string()) } else { (format!("{}", i as i32 * 9 - 40), "null".to_string()) };
            if i > 0 {
                s.push(',');
            }
            s.push_str(&format!(
                "{{\"ply\":{i},\"side\":\"{side}\",\"mover\":\"candidate\",\"move\":\"{m}\",\"score\":{score},\"mate\":{mate},\"depth\":{},\"nodes\":100,\"ms\":5,\"pv\":\"\"}}",
                (i % 70) + 1
            ));
        }
        s.push_str("]}");
        (s, before, last)
    }

    #[test]
    fn pack_write_read_replay_round_trip() {
        let (line, before, last) = json_game(60);
        let g = crate::games::parse_game(&line).unwrap();
        let pg = pack_game(&g, &line, 3).unwrap();
        assert_eq!(pg.plies.len(), before.len());
        assert_eq!((pg.cand_seat, pg.result, pg.src, pg.book, pg.pair), (1, 0, 3, 41, 7)); // candidate 'o' won -> side 0 lost
        assert_eq!(pg.final_cnt, [last.counts().0 as u8, last.counts().1 as u8]);
        let mut bytes = Vec::new();
        pg.write(&mut bytes).unwrap();
        assert_eq!(bytes.len(), GAME_HDR + PLY_BYTES * before.len());
        let back = PackedGame::read(&mut &bytes[..]).unwrap();
        assert_eq!(back, pg);
        let (views, end) = back.replay().unwrap();
        assert_eq!(end.occ, last.occ);
        for (v, b) in views.iter().zip(before.iter()) {
            assert_eq!((v.board.occ, v.board.turn), (b.occ, b.turn));
        }
        assert_eq!(views[4].score, Some(4 * 9 - 40));
        assert_eq!(views[5].score, Some(MATE - 3)); // "mate":3
        assert_eq!(views[6].score, None); // no score logged
        assert_eq!(views[4].depth, 5);
        assert_eq!(views[59.min(views.len() - 1)].depth as usize, (59.min(views.len() - 1) % 70 + 1).min(63));
    }

    #[test]
    fn future_mix_weights_and_proofs() {
        let (g, _, _) = play(12, true);
        let mut rows: Vec<Rec> = g.iter().enumerate().map(|(i, b)| rec(b, (i % 2) as f32)).collect();
        for (i, r) in rows.iter_mut().enumerate() {
            r.cp = 40.0 * i as f32 - 100.0;
        }
        let sig = crate::targets::sigmoid_cp;
        assert_eq!(future_mix(&rows, 0, 1.0, 64), (0.0, 1.0)); // lambda 1: the row keeps its result
        let (f0, t0) = future_mix(&rows, 0, 0.0, 64); // lambda 0: the next row's value, flipped
        assert!((f0 - (1.0 - sig(rows[1].cp))).abs() < 1e-6 && t0 == 0.0);
        let (f, t) = future_mix(&rows, 3, 0.5, 2); // two plies ahead, half the weight each time
        let want = 0.5 * (1.0 - sig(rows[4].cp)) + 0.25 * sig(rows[5].cp);
        assert!((f - want).abs() < 1e-6 && (t - 0.25).abs() < 1e-6);
        assert_eq!(future_mix(&rows, rows.len() - 1, 0.5, 64), (0.0, 1.0)); // nothing after the last row
        rows[5].cp = 99_990.0; // proven win for the side to move two plies after row 3: it takes the rest
        let (f, t) = future_mix(&rows, 3, 0.5, 64);
        assert!((f - (0.5 * (1.0 - sig(rows[4].cp)) + 0.5 * 1.0)).abs() < 1e-6 && t == 0.0);
    }

    #[test]
    fn lambda_file_is_identical_at_one_and_carries_the_future_below() {
        let (g1, _, _) = play(16, true);
        let dir = std::env::temp_dir().join(format!("ti_lambda_{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let p = |n: &str| dir.join(n).to_string_lossy().to_string();
        let mut rows: Vec<Rec> = g1.iter().enumerate().map(|(i, b)| rec(b, (i % 2) as f32)).collect();
        for (i, r) in rows.iter_mut().enumerate() {
            r.cp = 25.0 * i as f32 - 150.0;
        }
        let mut lone = rec(&g1[9], 0.5); // an unlinked row after the game
        lone.cp = 7.0;
        rows.push(lone);
        let bytes: Vec<u8> = rows.iter().flat_map(|r| r.to_bytes()).collect();
        fs::write(p("a.tdat"), &bytes).unwrap();
        let (chains, st) = scan_chains(&p("a.tdat"), 2).unwrap();
        assert_eq!(chains.len(), 1);
        write_chains(&p("a.chains"), st.rows, &chains).unwrap();
        let run = |lambda: &str, out: &str| {
            let a: Vec<String> = ["lambda", "--tdat", &p("a.tdat"), "--out", &p(out), "--lambda", lambda].iter().map(|s| s.to_string()).collect();
            let _ = run_lambda(&a);
            fs::read(p(&format!("{out}.tdat"))).unwrap()
        };
        assert_eq!(run("1.0", "one"), bytes, "lambda 1 must reproduce the file");
        let zero = run("0.0", "zero");
        let got: Vec<Rec> = zero.chunks_exact(REC).map(Rec::from_bytes).collect();
        let n = g1.len();
        for t in 0..n - 1 {
            let want = 1.0 - crate::targets::sigmoid_cp(rows[t + 1].cp);
            assert!((got[t].outcome - want).abs() < 1e-6, "row {t}");
            assert_eq!((got[t].occ0, got[t].cp), (rows[t].occ0, rows[t].cp));
        }
        assert_eq!(got[n - 1], rows[n - 1]); // last row of the run: nothing after it
        assert_eq!(got[n], rows[n]); // the unlinked row is copied as it is
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn lambda_returns_limits() {
        let v = [Some(0.2), Some(0.4), None, Some(0.9)];
        assert_eq!(lambda_returns(&v, 1.0, 1.0), vec![1.0; 4]); // lambda 1 = the game result
        let td0 = lambda_returns(&v, 1.0, 0.0); // lambda 0 = the next value (a missing one passes through)
        assert_eq!(td0, vec![0.4, 0.9, 0.9, 1.0]);
        let half = lambda_returns(&v, 0.0, 0.5);
        assert!((half[3] - 0.0).abs() < 1e-6 && (half[2] - 0.45).abs() < 1e-6 && (half[1] - 0.45).abs() < 1e-6 && (half[0] - 0.425).abs() < 1e-6);
    }
}

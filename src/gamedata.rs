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
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Mutex;

use titanium::board::Move;
use titanium::search::SearchLimits;
use titanium::search::MATE;
use titanium::ttcodec::{dec_score, decode_move, enc_score, move_code, PASS_CODE};
use titanium::Board;

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

/// Key of a position for de-duplication: (mover stones, opponent stones), whatever the row convention (old rows keep the mover
/// in occ0 with stm 0; rows from game files keep colour 0 in occ0 and name the mover in stm). Blocked squares are not in rows.
pub fn pos_key(occ0: u64, occ1: u64, stm: u8) -> u64 {
    let (m, o) = if stm == 0 { (occ0, occ1) } else { (occ1, occ0) };
    crate::dag::splitmix(crate::dag::splitmix(m) ^ o.rotate_left(17))
}

/// Reader of the `.tbst` label file `ti gen` writes next to a `.tgam` (layout: see `gen::BinOut`).
pub struct TbstReader {
    rd: BufReader<File>,
    pub games: u64,
    pub plies: u64,
}

impl TbstReader {
    pub fn open(tgam: &str) -> Result<TbstReader, String> {
        let p = format!("{}.tbst", tgam.trim_end_matches(".tgam"));
        let mut rd = BufReader::with_capacity(1 << 20, File::open(&p).map_err(|e| format!("{p}: {e}"))?);
        let mut h = [0u8; 24];
        rd.read_exact(&mut h).map_err(|e| format!("{p}: {e}"))?;
        if &h[0..4] != b"TBST" || u32::from_le_bytes(h[4..8].try_into().unwrap()) != 1 {
            return Err(format!("{p}: not a TBST version 1 file"));
        }
        Ok(TbstReader { rd, games: u64::from_le_bytes(h[8..16].try_into().unwrap()), plies: u64::from_le_bytes(h[16..24].try_into().unwrap()) })
    }

    /// Next game: its DAG id and the (best move code, best score code) of each of its `n` plies.
    pub fn next_game(&mut self, n: usize) -> Result<(u32, Vec<(u16, i16)>), String> {
        let mut b = vec![0u8; 4 + n * 4];
        self.rd.read_exact(&mut b).map_err(|e| format!("tbst: {e}"))?;
        let gid = u32::from_le_bytes(b[0..4].try_into().unwrap());
        Ok((gid, b[4..].chunks_exact(4).map(|c| (u16::from_le_bytes([c[0], c[1]]), i16::from_le_bytes([c[2], c[3]]))).collect()))
    }
}

/// `ti data append --base STEM --add STEM --out STEM`: the rows of `--add` appended to a COPY of `--base` (both stay
/// untouched). Copies STEM.tdat / .kata / .aux, appends the new rows, zero-pads .kata (16 B) and .aux (12 B) for them
/// (no Kata policy target, no aux data: exactly as the 2026-11-14 append did), merges .chains (new runs shifted by the
/// base row count) and writes the manifest. The row count of every file is checked afterwards.
pub fn run_append(args: &[String]) -> ExitCode {
    let (Some(base), Some(add), Some(out)) = (crate::flag_str(args, "--base"), crate::flag_str(args, "--add"), crate::flag_str(args, "--out")) else {
        eprintln!("usage: ti data append --base STEM --add STEM --out STEM   (STEM = path without .tdat)");
        return ExitCode::FAILURE;
    };
    let strip = |s: &str| s.trim_end_matches(".tdat").to_string();
    let (base, add, out) = (strip(&base), strip(&add), strip(&out));
    // D1146: --before-tail N inserts the new rows BEFORE the last N base rows (the held-out validation games of the lab
    // standard stay the last N rows, so every validation number stays comparable); 0 = append at the end (old behaviour)
    let tail = crate::flag_u64(args, "--before-tail").unwrap_or(0);
    if tail > 0 {
        return match insert_before_tail(&base, &add, &out, tail) {
            Ok((nb, na)) => {
                println!("data append: {out}.tdat = {} base rows + {na} new rows + the last {tail} base rows (held-out tail kept last) = {} rows (base and add files untouched)", nb - tail, nb + na);
                ExitCode::SUCCESS
            }
            Err(e) => {
                eprintln!("data append: {e}");
                ExitCode::FAILURE
            }
        };
    }
    let r = || -> Result<(u64, u64), String> {
        if out == base || out == add {
            return Err("--out must be a new stem".into());
        }
        if std::path::Path::new(&format!("{out}.tdat")).exists() {
            return Err(format!("{out}.tdat already exists; refusing to overwrite"));
        }
        let len = |p: &str| fs::metadata(p).map(|m| m.len()).map_err(|e| format!("{p}: {e}"));
        let (nb, na) = (len(&format!("{base}.tdat"))? / REC as u64, len(&format!("{add}.tdat"))? / REC as u64);
        for (ext, rec) in [("tdat", REC as u64), ("kata", 16), ("aux", 12)] {
            let (src, dst) = (format!("{base}.{ext}"), format!("{out}.{ext}"));
            if !std::path::Path::new(&src).exists() {
                if ext == "tdat" {
                    return Err(format!("{src} missing"));
                }
                continue;
            }
            let have = len(&src)?;
            if have != nb * rec {
                // sidecars shorter than the rows: pad with zeros up to the base row count first (keeps row i <-> record i)
                eprintln!("append: {src} holds {} records for {nb} rows; zero-padding the copy", have / rec);
            }
            fs::copy(&src, &dst).map_err(|e| format!("copy {src} -> {dst}: {e}"))?;
            let mut f = fs::OpenOptions::new().append(true).open(&dst).map_err(|e| format!("{dst}: {e}"))?;
            if ext == "tdat" {
                let mut a = BufReader::with_capacity(1 << 22, File::open(format!("{add}.tdat")).map_err(|e| e.to_string())?);
                std::io::copy(&mut a, &mut f).map_err(|e| format!("{dst}: {e}"))?;
            } else {
                let want = (nb + na) * rec;
                let mut left = want.saturating_sub(have);
                let zeros = vec![0u8; 1 << 20];
                while left > 0 {
                    let n = left.min(zeros.len() as u64) as usize;
                    f.write_all(&zeros[..n]).map_err(|e| format!("{dst}: {e}"))?;
                    left -= n as u64;
                }
            }
            f.flush().map_err(|e| e.to_string())?;
            let want = if ext == "tdat" { (nb + na) * rec } else { (nb + na) * rec };
            if len(&dst)? != want {
                return Err(format!("{dst}: {} bytes, expected {want}", len(&dst)?));
            }
        }
        // chains: base runs, then the added runs shifted by the base row count
        let mut chains: Vec<Chain> = Vec::new();
        if std::path::Path::new(&format!("{base}.chains")).exists() {
            chains.extend(read_chains(&format!("{base}.chains"))?.1);
        }
        if std::path::Path::new(&format!("{add}.chains")).exists() {
            for c in read_chains(&format!("{add}.chains"))?.1 {
                chains.push(Chain { start: (c.start as u64 + nb) as u32, len: c.len, conv: c.conv });
            }
        }
        write_chains(&format!("{out}.chains"), nb + na, &chains)?;
        let manifest = format!(
            "{{\"src\":{},\"rows\":{},\"rec\":{REC},\"kata_rec\":16,\"aux_rec\":12}}",
            crate::json_str(&format!("{base}.tdat ({nb} rows, unchanged) + {add}.tdat ({na} rows appended by ti data append; kata/aux zero-padded for them)")),
            nb + na
        );
        fs::write(format!("{out}.manifest.json"), manifest).map_err(|e| e.to_string())?;
        Ok((nb, na))
    };
    match r() {
        Ok((nb, na)) => {
            println!("data append: {out}.tdat = {nb} base rows + {na} new rows = {} rows (base and add files untouched)", nb + na);
            // D1450 (LAWS D1): fingerprint of the result, compared with the PC file when the append runs on Modal
            match fnv_file(&format!("{out}.tdat")) {
                Ok((h, n)) => println!("data append: {out}.tdat {n} bytes, fnv1a64 {h:016X}"),
                Err(e) => eprintln!("data append: fingerprint failed: {e}"),
            }
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("data append: {e}");
            ExitCode::FAILURE
        }
    }
}

/// D1146: `--before-tail N` form of `ti data append`: base rows [0, nb - N), then the added rows, then the last N base
/// rows. .kata / .aux get zero records for the added rows at the same place; chains are shifted, a base chain that
/// crosses the cut is split in two (its rows are unchanged, only the insertion point separates them). Every output size
/// is checked; nothing is written over an existing file.
fn insert_before_tail(base: &str, add: &str, out: &str, tail: u64) -> Result<(u64, u64), String> {
    if out == base || out == add {
        return Err("--out must be a new stem".into());
    }
    if std::path::Path::new(&format!("{out}.tdat")).exists() {
        return Err(format!("{out}.tdat already exists; refusing to overwrite"));
    }
    let len = |p: &str| fs::metadata(p).map(|m| m.len()).map_err(|e| format!("{p}: {e}"));
    let (nb, na) = (len(&format!("{base}.tdat"))? / REC as u64, len(&format!("{add}.tdat"))? / REC as u64);
    if tail >= nb {
        return Err(format!("--before-tail {tail} >= base rows {nb}"));
    }
    let cut = nb - tail;
    for (ext, rec) in [("tdat", REC as u64), ("kata", 16u64), ("aux", 12u64)] {
        let (src, dst) = (format!("{base}.{ext}"), format!("{out}.{ext}"));
        if !std::path::Path::new(&src).exists() {
            if ext == "tdat" {
                return Err(format!("{src} missing"));
            }
            continue;
        }
        if len(&src)? != nb * rec {
            return Err(format!("{src}: {} records for {nb} rows (refusing to insert into a short sidecar)", len(&src)? / rec));
        }
        let mut r = BufReader::with_capacity(1 << 22, File::open(&src).map_err(|e| format!("{src}: {e}"))?);
        let mut w = std::io::BufWriter::with_capacity(1 << 22, File::create(&dst).map_err(|e| format!("{dst}: {e}"))?);
        std::io::copy(&mut (&mut r).take(cut * rec), &mut w).map_err(|e| format!("{dst}: {e}"))?;
        if ext == "tdat" {
            let mut a = BufReader::with_capacity(1 << 22, File::open(format!("{add}.tdat")).map_err(|e| e.to_string())?);
            std::io::copy(&mut a, &mut w).map_err(|e| format!("{dst}: {e}"))?;
        } else {
            let zeros = vec![0u8; 1 << 20];
            let mut left = na * rec;
            while left > 0 {
                let n = left.min(zeros.len() as u64) as usize;
                w.write_all(&zeros[..n]).map_err(|e| format!("{dst}: {e}"))?;
                left -= n as u64;
            }
        }
        std::io::copy(&mut r, &mut w).map_err(|e| format!("{dst}: {e}"))?;
        w.flush().map_err(|e| e.to_string())?;
        drop(w);
        if len(&dst)? != (nb + na) * rec {
            return Err(format!("{dst}: {} bytes, expected {}", len(&dst)?, (nb + na) * rec));
        }
    }
    let mut chains: Vec<Chain> = Vec::new();
    if std::path::Path::new(&format!("{base}.chains")).exists() {
        for c in read_chains(&format!("{base}.chains"))?.1 {
            let (s, e) = (c.start as u64, c.start as u64 + c.len as u64);
            if e <= cut {
                chains.push(c);
            } else if s >= cut {
                chains.push(Chain { start: (s + na) as u32, len: c.len, conv: c.conv });
            } else {
                chains.push(Chain { start: c.start, len: (cut - s) as u16, conv: c.conv });
                chains.push(Chain { start: (cut + na) as u32, len: (e - cut) as u16, conv: c.conv });
            }
        }
    }
    if std::path::Path::new(&format!("{add}.chains")).exists() {
        for c in read_chains(&format!("{add}.chains"))?.1 {
            chains.push(Chain { start: (c.start as u64 + cut) as u32, len: c.len, conv: c.conv });
        }
    }
    chains.sort_by_key(|c| c.start);
    write_chains(&format!("{out}.chains"), nb + na, &chains)?;
    let manifest = format!(
        "{{\"src\":{},\"rows\":{},\"rec\":{REC},\"kata_rec\":16,\"aux_rec\":12,\"holdout_tail_rows\":{tail}}}",
        crate::json_str(&format!(
            "{base}.tdat rows [0, {cut}) + {add}.tdat ({na} rows inserted by ti data append --before-tail {tail}; kata/aux zero-padded for them) + {base}.tdat rows [{cut}, {nb}) = the held-out tail, kept last"
        )),
        nb + na
    );
    fs::write(format!("{out}.manifest.json"), manifest).map_err(|e| e.to_string())?;
    Ok((nb, na))
}

/// `ti data games-tdat`: replay a `.tgam` and write trainer rows. This is the
/// missing link for the data flywheel: gate/self-play games (`.tgam`) could be
/// packed and analysed but never trained on. Each row's cp is the engine's best-move
/// score logged while the game was played (D925; no second search, LAWS A12), the row
/// outcome is the game result from the mover's view, and one chain per game is written
/// so `ti data lambda` works on the result.
pub fn run_games_tdat(args: &[String]) -> ExitCode {
    let tgam = crate::flag_str(args, "--tgam").unwrap_or_default();
    let (true, Some(out)) = (!tgam.is_empty() || crate::flag_str(args, "--tgam-roots").is_some(), crate::flag_str(args, "--out")) else {
        eprintln!("usage: ti data games-tdat (--tgam F.tgam[,G.tgam] | --tgam-roots DIR,DIR) --out STEM [--min-cand-elo -50] [--limit-games N] [--relabel-src 0]");
        return ExitCode::FAILURE;
    };
    // LAWS A12 (D927): rows come only from the scores the generating engine logged (best-move score, D925). The old
    // re-search path (--scores search, --nodes, --net) labelled with a small-TT engine nobody checked: deleted.
    if let Err(e) = crate::genguard::refuse_weakening_flags("games-tdat", args, &["--scores", "--nodes", "--net"]) {
        eprintln!("{e}");
        return ExitCode::FAILURE;
    }
    let limit = crate::flag_u64(args, "--limit-games").map(|v| v as usize).unwrap_or(usize::MAX);
    // 200 = in-game eval, no separate relabel ran (see DATASET_GUIDE.md: 0/1/2 are
    // real relabel passes, so reusing one of them would mislabel the provenance).
    let src: u8 = crate::flag_u64(args, "--relabel-src").unwrap_or(200) as u8;
    // Labels come from the game itself by default (both engines evaluated every
    // position while playing); `--scores search` re-labels with a fresh search.
    let use_game_scores = true;
    let out_tdat = format!("{}.tdat", out.trim_end_matches(".tdat"));
    let t0 = std::time::Instant::now();
    // D809: several game files (comma list); cp is always the score of the ENGINE'S BEST move (D925, see below);
    // `--dedup-against X.tdat` / `--dedup` = write each position once (against X.tdat and against what this run wrote);
    // a skipped position ends the chain exactly like a pass does.
    let mut tgams: Vec<String> = tgam.split(',').filter(|s| !s.is_empty()).map(String::from).collect();
    // --tgam-roots DIR,DIR: every *.tgam below these folders (gate logs and generated batches), sorted
    if let Some(roots) = crate::flag_str(args, "--tgam-roots") {
        let mut stack: Vec<std::path::PathBuf> = roots.split(',').filter(|s| !s.is_empty()).map(std::path::PathBuf::from).collect();
        let mut found = Vec::new();
        while let Some(d) = stack.pop() {
            let Ok(rd) = fs::read_dir(&d) else { continue };
            for e in rd.flatten() {
                let p = e.path();
                if p.is_dir() {
                    stack.push(p);
                } else if p.extension().and_then(|x| x.to_str()) == Some("tgam") {
                    found.push(p.to_string_lossy().to_string());
                }
            }
        }
        found.sort();
        tgams.extend(found);
    }
    // D931: rows for the STANDARD dataset never come from experiment games, and an experiment writes only under experiments/
    let exp = crate::genguard::experiment(args);
    crate::genguard::announce("games-tdat", &exp);
    if let Err(e) = crate::genguard::check_inputs("games-tdat", &exp, tgams.iter().map(String::as_str))
        .and_then(|_| crate::genguard::check_output("games-tdat", &exp, &out))
    {
        eprintln!("{e}");
        return ExitCode::FAILURE;
    }
    let min_cand_elo = crate::flag_str(args, "--min-cand-elo").and_then(|v| v.parse::<f64>().ok());
    // D925 (user 2026-11-16): the value of a position IS the score of its best move. Every row's cp is the engine's
    // best-move score, never the score of the move that was played (a novelty move is worse by design). The played
    // move only decides the next position and the game result (outcome / lambda / WDL follow the played line).
    // Self-play files therefore REQUIRE their .tbst; gate games play the engine's own best move, so for them the
    // played score is the best score and a missing .tbst is fine. `--best-scores` was the opt-in; it is now the only
    // behaviour (the old flag is accepted and does nothing; there is no way to ask for played-move scores).
    if args.iter().any(|a| a == "--best-scores") {
        eprintln!("data games-tdat: note: --best-scores is the only behaviour now (D925); the flag is not needed");
    }
    let against = crate::flag_str(args, "--dedup-against");
    let want_dedup = against.is_some() || args.iter().any(|a| a == "--dedup");
    let work = || -> Result<(u64, u64, u64, Vec<Chain>, u64, u64, u64, u64), String> {
        let mut skipped_gates: Vec<String> = Vec::new();
        let mut seen: HashSet<u64> = HashSet::new();
        let mut base_rows = 0u64;
        if let Some(p) = &against {
            let mut rd = BufReader::with_capacity(1 << 22, File::open(p).map_err(|e| format!("{p}: {e}"))?);
            let mut buf = vec![0u8; REC * 65536];
            loop {
                let mut got = 0;
                while got < buf.len() {
                    let n = rd.read(&mut buf[got..]).map_err(|e| format!("{p}: {e}"))?;
                    if n == 0 {
                        break;
                    }
                    got += n;
                }
                for c in buf[..got - got % REC].chunks_exact(REC) {
                    let r = Rec::from_bytes(c);
                    seen.insert(pos_key(r.occ0, r.occ1, r.stm));
                    base_rows += 1;
                }
                if got < buf.len() {
                    break;
                }
            }
            eprintln!("games-tdat: {} distinct positions loaded from {p} ({base_rows} rows)", seen.len());
        }
        let (mut dups, mut n_best) = (0u64, 0u64);
        let mut w = BufWriter::with_capacity(1 << 22, File::create(&out_tdat).map_err(|e| format!("{out_tdat}: {e}"))?);
        let mut chains: Vec<Chain> = Vec::new();
        let mut row: u64 = 0;
        let mut games_out = 0u64;
        let (mut n_out, mut n_draw, mut n_unknown) = (0u64, 0u64, 0u64);
        for tpath in &tgams {
        // D832 (user): a GATE file whose candidate scored more than 50 Elo below its baseline gives no rows (the files stay).
        // Only gate files (<run>/sprt.log.tgam, cand_seat 0/1 per game); generated batches are self-play and never skipped.
        if let Some(min_elo) = min_cand_elo {
            if tpath.ends_with(".log.tgam") {
                let (mut pts, mut n) = (0f64, 0u64);
                for g in GameReader::open(tpath)?.flatten() {
                    if g.cand_seat <= 1 && g.result <= 2 {
                        let p0 = g.result as f64 / 2.0;
                        pts += if g.cand_seat == 0 { p0 } else { 1.0 - p0 };
                        n += 1;
                    }
                }
                let s = (pts / n.max(1) as f64).clamp(1e-3, 1.0 - 1e-3);
                let elo = -400.0 * (1.0 / s - 1.0).log10();
                if n > 0 && elo < min_elo {
                    skipped_gates.push(format!("{tpath} ({n} games, candidate {elo:+.0} Elo)"));
                    continue;
                }
            }
        }
        let mut rd = GameReader::open(tpath)?;
        let tbst_path = format!("{}.tbst", tpath.trim_end_matches(".tgam"));
        let mut tb = if use_game_scores && std::path::Path::new(&tbst_path).exists() { Some(TbstReader::open(tpath)?) } else { None };
        for (gi, g) in rd.by_ref().enumerate() {
            if gi >= limit {
                break;
            }
            let g = match g {
                Ok(g) => g,
                Err(e) if tb.is_some() => return Err(e),
                Err(_) => continue,
            };
            let labels: Vec<(u16, i16)> = match tb.as_mut() {
                Some(t) => t.next_game(g.plies.len())?.1,
                None => Vec::new(),
            };
            // D930: EVERY game file needs its .tbst (ti gen, ti gate's import and pack-games all write one). The seat field
            // cannot tell self-play from gate games (ti gen wrote cand_seat 0 before D930), so there is no exception.
            if use_game_scores && tb.is_none() {
                return Err(format!("{tpath}: no {tbst_path}: the best-move labels are missing, refusing (D925/D930; re-pack the source with ti data pack-games)"));
            }
            if g.forfeit || g.result == 3 {
                n_unknown += 1; // no trustworthy outcome -> not a training game
                continue;
            }
            let mut start_row = row;
            let mut wrote_any = false;
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
            for (pi, ply) in g.plies.iter().enumerate() {
                let code = ply.0 & 0x3ff;
                // A pass, or a ply without a logged score, gives no row. The run is closed there, so inside
                // one chain consecutive rows are always exactly one ply apart (`ti data lambda` relies on it).
                // The game itself goes on: before 2026-11-13 it was cut at its first pass.
                // the row needs the best-move label from the .tbst (gate files: best = played, written by the import)
                let tbst_label = labels.get(pi).filter(|&&(lc, ls)| lc & 0x3ff != 0 && ls != NO_SCORE).map(|&(_, ls)| ls);
                let best_label: Option<i16> = tbst_label;
                let no_row = code == PASS_CODE || (use_game_scores && best_label.is_none());
                if no_row {
                    if row > start_row {
                        chains.push(Chain { start: start_row as u32, len: (row - start_row).min(u16::MAX as u64) as u16, conv: 0 });
                        wrote_any = true;
                    }
                    start_row = row;
                    if code == PASS_CODE {
                        b = b.make_uai(Move::PASS);
                        continue;
                    }
                }
                let mv = titanium::ttcodec::decode_move(&b, code);
                let Some(mv) = mv else { break };
                if no_row {
                    b = b.make(mv);
                    continue;
                }
                // Default: use the eval BOTH engines already produced while playing
                // (the score recorded for this ply). No second search: the game is
                // already a labelled run. `--scores search` re-labels instead.
                if want_dedup && !seen.insert(pos_key(b.occ[0], b.occ[1], b.turn)) {
                    dups += 1;
                    if row > start_row {
                        chains.push(Chain { start: start_row as u32, len: (row - start_row).min(u16::MAX as u64) as u16, conv: 0 });
                        wrote_any = true;
                    }
                    start_row = row;
                    b = b.make(mv);
                    continue;
                }
                n_best += 1;
                let cp: i32 = dec_score(best_label.expect("no_row covers a missing label"));
                let outcome = if b.turn == 0 { o0 } else { 1.0 - o0 };
                let rec = crate::data::Rec {
                    occ0: b.occ[0],
                    occ1: b.occ[1],

                    outcome,
                    cp: cp as f32,
                    wsum: 1.0,
                    p3: 0.0,
                    cp5k: cp as f32,
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
                wrote_any = true;
            }
            if wrote_any {
                games_out += 1;
            }
        }
        }
        w.flush().map_err(|e| e.to_string())?;
        if min_cand_elo.is_some() {
            println!("data games-tdat: {} of {} input files skipped (gate candidate below --min-cand-elo, D832; files kept):", skipped_gates.len(), tgams.len());
            for s in &skipped_gates {
                println!("  skipped {s}");
            }
        }
        Ok((row, games_out, n_out + n_draw, chains, n_unknown, dups, n_best, base_rows))
    };
    match work() {
        Ok((rows, games, decided, chains, unknown, dups, n_best, _base_rows)) => {
            if want_dedup || use_game_scores {
                println!("data games-tdat: positions skipped as duplicates {dups}; rows whose cp is the engine's best-move score {n_best} of {rows}");
            }
            let idx = format!("{}.chains", out.trim_end_matches(".tdat"));
            if let Err(e) = write_chains(&idx, rows, &chains) {
                eprintln!("data games-tdat: chains: {e}");
                return ExitCode::FAILURE;
            }
            let labels = "cp = the engine's BEST-move score for the position (self-play: from the .tbst; gate games: the played move = the engine's best; no second search; a ply without that label gives no row); the played move only leads to the next row and the outcome (D925)".to_string();
            let desc = format!("{tgam} replayed; {labels}; outcome = game result from the mover's view; a pass or an unscored ply gives no row and ends the chain; {games} games; ti data games-tdat");
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

/// Proof value of a row for its own side to move: 1 / 0 for a proven win / loss (the trainer's threshold), 0.5 for an
/// exact proven draw (hunt proof kinds 1, 2, 5 with score 0), else None.
fn proof_value(r: &Rec) -> Option<f32> {
    let t = crate::targets::target_of(r);
    if t.proven >= 0 {
        Some(t.proven as f32)
    } else if matches!(r.proof_kind, 1 | 2 | 5) && r.cp == 0.0 {
        Some(0.5)
    } else {
        None
    }
}

/// Position key for pooling: side to move's stones, opponent's stones, minimum over the 8 board symmetries.
fn pool_key(r: &Rec) -> (u64, u64) {
    let (m, o) = if r.stm == 0 { (r.occ0, r.occ1) } else { (r.occ1, r.occ0) };
    (0..8).map(|k| (crate::mapper::transform_bb(m, k), crate::mapper::transform_bb(o, k))).min().unwrap()
}

/// `ti data proof-wdl` (D1246, user design): every non-proven row's outcome becomes `w * proof + (1 - w) * wdl` with
/// `w = decay^n`, `n` = plies to the first proven row later in its game run (the proof seen from the row's side: flipped
/// when `n` is odd), and `wdl` = the pooled result of all non-proven training rows with the same position (D4 + side to
/// move; a row counts `wsum^2` games when wsum > 1, as in the proof queue), or the row's own result when the position
/// occurs once. Rows with no later proof keep `w = 0` (only the pooling applies). Proven rows are copied unchanged. The
/// last `--preserve-tail` rows (held-out validation) are copied unchanged, never feed the pool and never lend a proof.
pub fn run_proof_wdl(args: &[String]) -> ExitCode {
    let (Some(tdat), Some(out)) = (crate::flag_str(args, "--tdat"), crate::flag_str(args, "--out")) else {
        eprintln!("usage: ti data proof-wdl --tdat X.tdat --out STEM --chains X.chains --decay 0.9 --preserve-tail N");
        return ExitCode::FAILURE;
    };
    let tdat = if tdat.ends_with(".tdat") { tdat } else { format!("{tdat}.tdat") };
    let idx = crate::flag_str(args, "--chains").unwrap_or_else(|| format!("{}.chains", tdat.trim_end_matches(".tdat")));
    let decay = crate::flag_str(args, "--decay").and_then(|v| v.parse::<f32>().ok()).unwrap_or(0.9).clamp(0.0, 1.0);
    let Some(tail) = crate::flag_str(args, "--preserve-tail").and_then(|v| v.parse::<u64>().ok()) else {
        eprintln!("data proof-wdl: --preserve-tail N is required (the held-out validation tail is never rewritten)");
        return ExitCode::FAILURE;
    };
    let out_tdat = format!("{}.tdat", out.trim_end_matches(".tdat"));
    if std::path::Path::new(&out_tdat) == std::path::Path::new(&tdat) {
        eprintln!("data proof-wdl: --out must differ from --tdat (LAWS A10)");
        return ExitCode::FAILURE;
    }
    let t0 = std::time::Instant::now();
    let work = || -> Result<(String, usize), String> {
        let (idx_rows, chains) = read_chains(&idx)?;
        let mut rs = RowStream::open(&tdat)?;
        check_index(rs.rows, idx_rows, &chains)?;
        let total = rs.rows as usize;
        let head = total.saturating_sub(tail as usize);
        // pass 1: proof value per row, pooling keys of the non-proven head rows
        let mut proof = vec![-1i8; total]; // -1 none, 0 loss, 1 win, 2 draw (all for the row's side to move)
        let mut keys: Vec<(u64, u64, u32)> = Vec::with_capacity(head);
        let mut res = vec![(0f32, 0f32); head]; // (outcome, effective game count)
        let mut done = 0usize;
        while done < total {
            let n = (total - done).min(1 << 16);
            for (i, r) in rs.take(n)?.iter().enumerate() {
                let row = done + i;
                if row >= head || r.occ0 & r.occ1 != 0 {
                    continue;
                }
                match proof_value(r) {
                    Some(v) => proof[row] = if v == 0.5 { 2 } else { v as i8 },
                    None => {
                        let (km, ko) = pool_key(r);
                        keys.push((km, ko, row as u32));
                        let ne = if r.wsum > 1.001 && r.wsum.is_finite() { r.wsum * r.wsum } else { 1.0 };
                        res[row] = (if r.outcome.is_finite() { r.outcome.clamp(0.0, 1.0) } else { 0.5 }, ne);
                    }
                }
            }
            done += n;
        }
        // pooled result per position (NaN = the position occurs once: the row keeps its own result)
        keys.sort_unstable();
        let mut pooled = vec![f32::NAN; head];
        let (mut groups_multi, mut rows_pooled) = (0u64, 0u64);
        let mut i = 0;
        while i < keys.len() {
            let mut j = i;
            let (mut w, mut nn) = (0f64, 0f64);
            while j < keys.len() && keys[j].0 == keys[i].0 && keys[j].1 == keys[i].1 {
                let (o, ne) = res[keys[j].2 as usize];
                w += (o * ne) as f64;
                nn += ne as f64;
                j += 1;
            }
            if j - i > 1 {
                groups_multi += 1;
                rows_pooled += (j - i) as u64;
                let p = (w / nn.max(1e-9)) as f32;
                for k in &keys[i..j] {
                    pooled[k.2 as usize] = p;
                }
            }
            i = j;
        }
        drop(keys);
        // proof back-propagation along the runs: (proof value for the row's side, weight)
        let mut back = vec![(0f32, 0f32); head];
        let mut by_n = [0u64; 5]; // n 1..5, 6..10, 11..20, 21..40, > 40
        // sanity of the side flip: the row's OWN game result vs the later proof seen from the row's side, per n bucket
        // (a game that reached a proven position mostly ends that way, so agree must dominate if the flip is right)
        let mut agree = [[0u64; 2]; 5];
        for c in &chains {
            let (s, e) = (c.start as usize, (c.start as usize + c.len as usize).min(head));
            let mut last: Option<(usize, i8)> = None;
            for t in (s..e).rev() {
                if proof[t] >= 0 {
                    last = Some((t, proof[t]));
                    continue;
                }
                if let Some((tp, pv)) = last {
                    let n = tp - t;
                    let v = match pv {
                        2 => 0.5,
                        _ => {
                            let v = pv as f32;
                            if n % 2 == 1 { 1.0 - v } else { v }
                        }
                    };
                    back[t] = (v, decay.powi(n as i32));
                    let k = match n { 1..=5 => 0, 6..=10 => 1, 11..=20 => 2, 21..=40 => 3, _ => 4 };
                    by_n[k] += 1;
                    let own = res[t].0;
                    if v != 0.5 && own != 0.5 {
                        agree[k][((own > 0.5) != (v > 0.5)) as usize] += 1;
                    }
                }
            }
        }
        // pass 2: write
        let mut rs = RowStream::open(&tdat)?;
        let mut w = BufWriter::with_capacity(1 << 20, File::create(&out_tdat).map_err(|e| format!("{out_tdat}: {e}"))?);
        let (mut changed, mut moved, mut flipped) = (0u64, 0f64, 0u64);
        done = 0;
        while done < total {
            let n = (total - done).min(1 << 16);
            let mut rows = rs.take(n)?;
            for (i, r) in rows.iter_mut().enumerate() {
                let row = done + i;
                if row < head && proof[row] < 0 && r.occ0 & r.occ1 == 0 {
                    let wdl = if pooled[row].is_nan() { r.outcome } else { pooled[row] };
                    let (pv, pw) = back[row];
                    let g = pw * pv + (1.0 - pw) * wdl;
                    if (g - r.outcome).abs() > 1e-6 {
                        changed += 1;
                        moved += (g - r.outcome).abs() as f64;
                        flipped += ((g - 0.5) * (r.outcome - 0.5) < 0.0) as u64;
                    }
                    r.outcome = g;
                }
                w.write_all(&r.to_bytes()).map_err(|e| e.to_string())?;
            }
            done += n;
        }
        w.flush().map_err(|e| e.to_string())?;
        let proven_head = proof[..head].iter().filter(|&&p| p >= 0).count();
        Ok((format!(
            "data proof-wdl: {tdat} -> {out_tdat}: {total} rows ({head} head, {tail} tail copied unchanged) in {:.1}s, decay {decay}\n\
             \x20 proven head rows {proven_head}; positions occurring > 1 time {groups_multi} ({rows_pooled} rows pooled)\n\
             \x20 rows with a later proof in their run, by plies to it: 1-5 {} | 6-10 {} | 11-20 {} | 21-40 {} | > 40 {}\n\
             \x20 own game result agrees / disagrees with that proof: 1-5 {}/{} | 6-10 {}/{} | 11-20 {}/{} | 21-40 {}/{} | > 40 {}/{}\n\
             \x20 rows whose outcome changed {changed} ({:.2}%), mean |change| {:.4}, side of 0.5 flipped {flipped}",
            t0.elapsed().as_secs_f64(),
            by_n[0], by_n[1], by_n[2], by_n[3], by_n[4],
            agree[0][0], agree[0][1], agree[1][0], agree[1][1], agree[2][0], agree[2][1], agree[3][0], agree[3][1], agree[4][0], agree[4][1],
            100.0 * changed as f64 / head.max(1) as f64,
            moved / changed.max(1) as f64
        ), total))
    };
    match work() {
        Ok((s, total)) => {
            let desc = format!("{tdat} with outcome := decay^n * proof + (1 - decay^n) * pooled WDL (decay {decay}, tail {tail} unchanged); ti data proof-wdl (D1246)");
            if let Err(e) = crate::data::write_tdat_manifest(out.trim_end_matches(".tdat"), &desc, total) {
                eprintln!("data proof-wdl: manifest: {e}");
            }
            println!("{s}");
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("data proof-wdl: {e}");
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
        } else if p.file_name().and_then(|n| n.to_str()).map_or(false, |n| n.ends_with(".games.jsonl") || n.ends_with(".tgam")) {
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
    // D926 (user order 2026-11-16 "fix this immediately"): pack-games used to read JSON only, so the binary .tgam/.tbst
    // that `ti gen` writes (D806) failed with "stream did not contain valid UTF-8" and gave an EMPTY output. Now every
    // game source is read directly, no conversion step:
    //   *.tgam (+ sibling .tbst)   binary games from ti gen or a gate; the .tbst best-move labels travel with each game
    //   *.games.jsonl              old gate logs (JSON); the engine played its own best move, so label = played move
    // Output: STEM.tgam + STEM.tbst (same layout as ti gen) + STEM.tgam.json. A self-play .tgam without its .tbst is
    // refused, gate or not (D930: the seat field cannot tell them apart; its novelty moves would lose their values, D925). Binary games are never dropped as duplicates
    // (D809: repeated games keep their own outcome); identical JSON records (the same gate log read twice) still are.
    // When a gate run has both <log>.games.jsonl and <log>.tgam among the inputs, only the .tgam is read.
    let Some(stem) = crate::flag_str(args, "--out") else {
        eprintln!("usage: ti data pack-games --out STEM [--roots DIR,DIR] [--inputs a.tgam,b.games.jsonl,...]");
        return ExitCode::FAILURE;
    };
    let stem = stem.trim_end_matches(".tgam").to_string();
    let mut files: Vec<String> =
        crate::flag_str(args, "--inputs").unwrap_or_default().split(',').filter(|s| !s.is_empty()).map(|s| s.replace('\\', "/")).collect();
    for root in crate::flag_str(args, "--roots").unwrap_or_default().split(',').filter(|s| !s.is_empty()) {
        walk_games(std::path::Path::new(root), &mut files);
    }
    files.sort();
    files.dedup();
    let tgam_set: HashSet<String> = files.iter().filter(|f| f.ends_with(".tgam")).map(|f| f.trim_end_matches(".tgam").to_string()).collect();
    let mut shadowed = 0u64;
    files.retain(|f| {
        let keep = !(f.ends_with(".games.jsonl") && tgam_set.contains(f.trim_end_matches(".games.jsonl")));
        if !keep {
            shadowed += 1;
        }
        keep
    });
    if files.is_empty() || files.len() > u16::MAX as usize {
        eprintln!("data pack-games: {} input files (need 1..65535)", files.len());
        return ExitCode::FAILURE;
    }
    let exp = crate::genguard::experiment(args);
    crate::genguard::announce("pack-games", &exp);
    if let Err(e) = crate::genguard::check_inputs("pack-games", &exp, files.iter().map(String::as_str))
        .and_then(|_| crate::genguard::check_output("pack-games", &exp, &stem))
    {
        eprintln!("{e}");
        return ExitCode::FAILURE;
    }
    let out_path = format!("{stem}.tgam");
    for p in [&out_path, &format!("{stem}.tbst")] {
        if std::path::Path::new(p).exists() {
            eprintln!("data pack-games: {p} already exists; refusing to overwrite (pick a new --out)");
            return ExitCode::FAILURE;
        }
        if files.iter().any(|f| f.trim_end_matches(".tgam") == stem) {
            eprintln!("data pack-games: --out {stem} is one of the inputs");
            return ExitCode::FAILURE;
        }
    }
    let t = std::time::Instant::now();
    let mut out = match crate::gen::BinOut::create(&stem) {
        Ok(o) => o,
        Err(e) => {
            eprintln!("data pack-games: {out_path}: {e}");
            return ExitCode::FAILURE;
        }
    };
    let mut seen_json: HashSet<u64> = HashSet::new();
    let mut seen_bin: HashSet<u64> = HashSet::new();
    let (mut dups_json, mut repeats_bin, mut skipped) = (0u64, 0u64, 0u64);
    let mut reasons: Vec<String> = Vec::new();
    let mut sources = Vec::new();
    let res = (|| -> Result<(), String> {
        for (si, path) in files.iter().enumerate() {
            let mut kept = 0u64;
            if path.ends_with(".tgam") {
                let mut rd = GameReader::open(path)?;
                let tbp = format!("{}.tbst", path.trim_end_matches(".tgam"));
                let mut tb = if std::path::Path::new(&tbp).exists() { Some(TbstReader::open(path)?) } else { None };
                for g in rd.by_ref() {
                    let mut g = g.map_err(|e| format!("{path}: {e}"))?;
                    let (gid, labels) = match tb.as_mut() {
                        Some(tr) => tr.next_game(g.plies.len()).map_err(|e| format!("{tbp}: {e}"))?,
                        None => return Err(format!("{path}: no {tbp}: best-move labels missing (D925/D930); refusing")),
                    };
                    if !seen_bin.insert(game_key(&g)) {
                        repeats_bin += 1;
                    }
                    g.src = si as u16;
                    out.append(&g, gid, &labels).map_err(|e| format!("{out_path}: {e}"))?;
                    kept += 1;
                }
                sources.push(serde_json::json!({ "i": si, "path": path, "games": kept, "labels": "tbst" }));
                continue;
            }
            let text = match fs::read_to_string(path) {
                Ok(t) => t,
                Err(e) => return Err(format!("{path}: {e}")),
            };
            for (n, line) in text.lines().enumerate().filter(|(_, l)| !l.trim().is_empty()) {
                match crate::games::parse_game(line).and_then(|g| pack_game(&g, line, si as u16)) {
                    Ok(pg) => {
                        if seen_json.insert(game_key(&pg)) {
                            let labels: Vec<(u16, i16)> = pg.plies.iter().map(|p| (p.0 & 0x3ff, p.1)).collect();
                            out.append(&pg, 0, &labels).map_err(|e| format!("{out_path}: {e}"))?;
                            kept += 1;
                        } else {
                            dups_json += 1;
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
            sources.push(serde_json::json!({ "i": si, "path": path, "games": kept, "labels": "played (gate games)", "header": header }));
        }
        Ok(())
    })();
    if let Err(e) = res {
        // never leave a half-written pair that looks valid
        let _ = fs::remove_file(&out_path);
        let _ = fs::remove_file(format!("{stem}.tbst"));
        eprintln!("data pack-games: {e} (nothing written)");
        return ExitCode::FAILURE;
    }
    let (games, plies) = (out.games, out.plies);
    drop(out);
    let bytes = fs::metadata(&out_path).map(|m| m.len()).unwrap_or(0);
    let manifest = serde_json::json!({
        "format": "TGAM1+TBST1", "games": games, "plies": plies, "bytes": bytes,
        "json_duplicates_dropped": dups_json, "binary_repeated_games_kept": repeats_bin,
        "jsonl_shadowed_by_tgam": shadowed, "games_skipped": skipped, "sources": sources,
    });
    if let Err(e) = fs::write(format!("{out_path}.json"), serde_json::to_string_pretty(&manifest).unwrap_or_default()) {
        eprintln!("data pack-games: manifest: {e}");
        return ExitCode::FAILURE;
    }
    println!(
        "data pack-games: {} files -> {out_path} + {stem}.tbst: {games} games, {plies} plies, {bytes} B in {:.1}s",
        files.len(),
        t.elapsed().as_secs_f64()
    );
    println!(
        "  binary repeated games kept (D809): {repeats_bin}; identical JSON records dropped: {dups_json}; JSON logs shadowed by their .tgam: {shadowed}; games skipped (unparsable or illegal): {skipped}"
    );
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

// ---------------------------------------------------------------------------------------------------------------
// bestmove: reconstruct the search-best move per row, with early stop on dataset agreement.
//   ti data bestmove --tdat X.tdat --dag DB [--out STEM] [--rows N] [--seed S] [--all]
//       [--threads 4] [--tt-bits 18] [--network bundled] [--tol-base 3] [--gap-conf 30] [--gap-dec 50]
//       [--first-check 1000] [--fresh-rows 101830]
//   Per row: forced single legal move costs 0 nodes. Otherwise chunked search (default 1k steps to a 5k
//   budget); at each checkpoint the root best must (a) match a dataset-attested DAG edge out of the
//   position and (b) sit within tol(N) = tol-base + (5000-N)/1000 cp of the row label before the second
//   best is even measured (exclusion search). Gap >= gap-dec stops decisively, gap >= gap-conf stops
//   confidently. Anything else runs to the full budget. Smoke mode (default) samples --rows stratified
//   by empties x proven (+ fresh tail), always completes to 5k to measure early-vs-full agreement, and
//   reports full-budget % / flip % / node extrapolation. --all writes STEM.best (u16 LE per row:
//   move code, PASS_CODE for pass, 0xFFFF unknown) plus a manifest; early stop is then real.
// ---------------------------------------------------------------------------------------------------------------

pub const BEST_UNKNOWN: u16 = 0xFFFF;

/// In-RAM index of every dataset position, bucketed by stone count.
/// A legal child with k stones is looked up ONLY in bucket[k]; clones (+1) and jumps (+0)
/// never cross buckets, so each attestation check is one O(1) hash probe.
struct DatasetIndex {
    buckets: Vec<std::collections::HashMap<(u64, u64, u8), u32>>,
}

fn build_dataset_index(tdat: &str) -> Result<DatasetIndex, String> {
    let f = File::open(tdat).map_err(|e| format!("{tdat}: {e}"))?;
    let total = f.metadata().map_err(|e| e.to_string())?.len() / REC as u64;
    let mut idx = DatasetIndex { buckets: (0..50).map(|_| std::collections::HashMap::new()).collect() };
    let mut rd = BufReader::with_capacity(1 << 20, f);
    let mut buf = vec![0u8; REC * 16384];
    let mut left = total;
    let mut done = 0u64;
    while left > 0 {
        let n = (left as usize).min(16384);
        rd.read_exact(&mut buf[..n * REC]).map_err(|e| format!("{tdat}: short read at row {done}: {e}"))?;
        for c in buf[..n * REC].chunks_exact(REC) {
            let r = Rec::from_bytes(c);
            let stones = (r.occ0 | r.occ1).count_ones() as usize;
            if stones < 50 {
                let e = idx.buckets[stones].entry((r.occ0, r.occ1, r.stm)).or_insert(0);
                *e = e.saturating_add(1);
            }
            done += 1;
        }
        left -= n as u64;
        if done % 5_000_000 < 16384 {
            eprintln!("data bestmove: index {done}/{total} rows");
        }
    }
    let uniq: usize = idx.buckets.iter().map(|b| b.len()).sum();
    eprintln!("data bestmove: index built: {total} rows -> {uniq} unique positions in 50 buckets");
    Ok(idx)
}

impl DatasetIndex {
    /// True when the child position occurs anywhere in the dataset (any game, any time).
    fn attested(&self, occ0: u64, occ1: u64, stm: u8, stones: usize) -> bool {
        if stones >= 50 {
            return false;
        }
        self.buckets[stones].contains_key(&(occ0, occ1, stm))
    }
    /// How many dataset rows hold the position (0 = unattested). Saturates.
    fn count(&self, occ0: u64, occ1: u64, stm: u8, stones: usize) -> u32 {
        if stones >= 50 {
            return 0;
        }
        self.buckets[stones].get(&(occ0, occ1, stm)).copied().unwrap_or(0)
    }
}

// ---------------------------------------------------------------------------------------------------------------
// stitch: unroll dataset-supported game lines from the root(s), classify F1-F4, report. No DB writes.
//   ti data stitch --tdat X.tdat [--max-lines N] [--max-nodes M] [--max-depth D] [--threads T]
//   Phase 1 (this command): DFS from the start position over ATTESTED children only (piece-count buckets
//   k/k+1), most-travelled first. Every completed terminal line is hashed (F3), checked for mid-line
//   ambiguity (F2: >1 attested child off the opening 8 plies) and counted. Dead ends (no attested child)
//   and caps stop lines. Report-only: lines completed/unique, main-bucket vs purgatory-bucket, dead
//   ends, node visits, branching histogram, coverage vs the index. Commit (phase 2) is separate.
// ---------------------------------------------------------------------------------------------------------------

struct StLine {
    path: Vec<u16>,
    result: u8,
    amb_mid: u32,
    amb_open: u32,
    forced: u32,
}

#[derive(Default)]
struct StStats {
    nodes: u64,
    dead_ends: u64,
    capped: u64,
    branch_hist: [u64; 8],
}

fn enc_stitch_move(m: Move) -> u16 {
    if m.is_pass() {
        return PASS_CODE;
    }
    move_code(m)
}

fn stitch_finish(path: &[u16], amb_path: &[bool], b: &Board, out_lines: &mut Vec<StLine>) {
    let (bx, bo) = b.bz_counts();
    let res = if bx > bo { 2 } else if bx == bo { 1 } else { 0 };
    let amb_open = amb_path.iter().take(8).filter(|&&x| x).count() as u32;
    let amb_mid = amb_path.iter().skip(8).filter(|&&x| x).count() as u32;
    let forced = path.len() as u32 - amb_open - amb_mid;
    out_lines.push(StLine { path: path.to_vec(), result: res, amb_mid, amb_open, forced });
}

/// Legal moves whose child occurs anywhere in the dataset (either row convention),
/// most-travelled first. Empty = dead end from here.
fn attested_children(b: &Board, idx: &DatasetIndex) -> Vec<(Move, u32)> {
    let legal = b.legal_moves();
    let mut kids: Vec<(Move, u32)> = Vec::new();
    for i in 0..legal.len() {
        let m = legal.move_at(i);
        let c = b.make(m);
        let stones = (c.occ[0] | c.occ[1]).count_ones() as usize;
        if stones >= 50 {
            continue;
        }
        // Two storage forms (D769): mover-relative rows (the teacher_kind-1 body) keep the mover's stones in
        // occ0 and stm 0 ALWAYS; colour-absolute rows keep side 0 in occ0 and the real stm. Probing the first
        // form with stm = c.turn missed every child of a body row. When the child's mover is side 0 both forms
        // are the same key, so it is counted once.
        let (a0, a1) = if c.turn == 0 { (c.occ[0], c.occ[1]) } else { (c.occ[1], c.occ[0]) };
        let n = idx.count(a0, a1, 0, stones) + if c.turn == 0 { 0 } else { idx.count(c.occ[0], c.occ[1], c.turn, stones) };
        if n > 0 {
            kids.push((m, n));
        }
    }
    kids.sort_by(|x, y| y.1.cmp(&x.1));
    kids
}

fn stitch_subtree(
    root: Board,
    root_moves: Vec<Move>,
    idx: &DatasetIndex,
    max_lines: u64,
    max_nodes: u64,
    max_depth: usize,
    lines_done: &AtomicUsize,
    nodes_done: &AtomicU64,
    out_lines: &mut Vec<StLine>,
    out_visited: &mut std::collections::HashSet<(u64, u64, u8)>,
    st: &mut StStats,
) {
    struct Frame {
        board: Board,
        children: Vec<(Move, u32)>,
        next: usize,
    }
    macro_rules! backtrack {
        ($stack:expr, $path:expr, $amb_path:expr, $depth:expr, $b:expr) => {{
            let mut advanced = false;
            while let Some(mut fr) = $stack.pop() {
                $path.pop();
                $amb_path.pop();
                $depth -= 1;
                if fr.next < fr.children.len() {
                    let (m, _) = fr.children[fr.next];
                    fr.next += 1;
                    let nb = fr.board.make(m);
                    let multi = fr.children.len() > 1;
                    $stack.push(fr);
                    $b = nb;
                    $path.push(enc_stitch_move(m));
                    $amb_path.push(multi);
                    $depth += 1;
                    advanced = true;
                    break;
                }
            }
            advanced
        }};
    }
    for &first in &root_moves {
        if lines_done.load(Ordering::Relaxed) as u64 >= max_lines {
            break;
        }
        let mut b = root.make(first);
        let mut path: Vec<u16> = vec![enc_stitch_move(first)];
        let mut amb_path: Vec<bool> = vec![root_moves.len() > 1];
        let mut stack: Vec<Frame> = Vec::new();
        let mut depth = 1usize;
        loop {
            if nodes_done.fetch_add(1, Ordering::Relaxed) >= max_nodes {
                st.capped += 1;
                break;
            }
            st.nodes += 1;
            out_visited.insert((b.occ[0], b.occ[1], b.turn));
            if b.bz_over() || depth >= max_depth {
                stitch_finish(&path, &amb_path, &b, out_lines);
                lines_done.fetch_add(1, Ordering::Relaxed);
                if !backtrack!(stack, path, amb_path, depth, b) {
                    break;
                }
                continue;
            }
            if !b.has_moves(b.turn) {
                stitch_finish(&path, &amb_path, &b, out_lines);
                lines_done.fetch_add(1, Ordering::Relaxed);
                st.dead_ends += 1;
                if !backtrack!(stack, path, amb_path, depth, b) {
                    break;
                }
                continue;
            }
            // Expand: legal moves with dataset-attested children, most-travelled first.
            let mut kids = attested_children(&b, idx);
            let nb = kids.len().min(7);
            st.branch_hist[nb] += 1;
            if kids.is_empty() {
                st.dead_ends += 1;
                if !backtrack!(stack, path, amb_path, depth, b) {
                    break;
                }
                continue;
            }
            let multi = kids.len() > 1;
            let (m0, _) = kids[0];
            stack.push(Frame { board: b, children: kids, next: 1 });
            b = b.make(m0);
            path.push(enc_stitch_move(m0));
            amb_path.push(multi);
            depth += 1;
            if lines_done.load(Ordering::Relaxed) as u64 >= max_lines {
                st.capped += 1;
                break;
            }
        }
    }
}

pub fn run_stitch(args: &[String]) -> ExitCode {
    let Some(tdat) = crate::flag_str(args, "--tdat") else {
        eprintln!("usage: ti data stitch --tdat X.tdat [--max-lines N] [--max-nodes M] [--max-depth D] [--threads T] [--diag-roots N] [--diag-conv N] [--diag-show N] [--diag-run START]");
        return ExitCode::FAILURE;
    };
    let diag_roots = crate::flag_u64(args, "--diag-roots").unwrap_or(0) as usize;
    let diag_conv = crate::flag_u64(args, "--diag-conv").unwrap_or(0) as usize;
    let diag_show = crate::flag_u64(args, "--diag-show").unwrap_or(0) as usize;
    if crate::flag_u64(args, "--diag-run").is_some() {
        // Print every row of one chain run as grids: is a run a real game or not?
        let start = crate::flag_u64(args, "--diag-run").unwrap();
        let chain_path = format!("{}.chains", tdat.trim_end_matches(".tdat"));
        let (_, chains) = match read_chains(&chain_path) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("data stitch: {e}");
                return ExitCode::FAILURE;
            }
        };
        let ch = match chains.iter().find(|c| c.start as u64 == start) {
            Some(c) => c,
            None => {
                eprintln!("data stitch: no run starts at row {start}");
                return ExitCode::FAILURE;
            }
        };
        let mut rs5 = match RowStream::open(&tdat) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("data stitch: {e}");
                return ExitCode::FAILURE;
            }
        };
        if rs5.advance(ch.start as u64, None).is_err() {
            eprintln!("data stitch: seek failed");
            return ExitCode::FAILURE;
        }
        let rows = match rs5.take(ch.len as usize) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("data stitch: {e}");
                return ExitCode::FAILURE;
            }
        };
        println!("data stitch diag-run: run@{} len={} conv={}", ch.start, ch.len, ch.conv);
        for (k, r) in rows.iter().enumerate() {
            let mut s = format!("row+{k} stm={} stones={}\n", r.stm, (r.occ0 | r.occ1).count_ones());
            for rr in 0..7 {
                for cc in 0..7 {
                    let i = (rr * 7 + cc) as u64;
                    s.push(if r.occ0 >> i & 1 == 1 { 'x' } else if r.occ1 >> i & 1 == 1 { 'o' } else { '.' });
                }
                s.push('\n');
            }
            println!("{s}");
        }
        return ExitCode::SUCCESS;
    }
    if diag_show > 0 {
        // Adjacency truth: sample N random adjacent file-row pairs and test the link
        // predicate both conventions. ~100% linked = game-ordered file; ~0% = shuffled
        // (chains stale, lambda targets meaningless).
        let adj_n = diag_show;
        let total = std::fs::metadata(&tdat).map(|m| m.len() / REC as u64).unwrap_or(0);
        let mut rs4 = match RowStream::open(&tdat) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("data stitch: {e}");
                return ExitCode::FAILURE;
            }
        };
        let mut rng: u64 = 0x12345678 | 1;
        let mut counts = [0u64; 3];
        let mut empt: [u64; 3] = [0, 0, 0];
        for _ in 0..adj_n {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            let at = (rng % total.saturating_sub(1).max(1)) as u64;
            if rs4.advance(at, None).is_err() {
                break;
            }
            let rows = match rs4.take(2) {
                Ok(r) => r,
                Err(_) => break,
            };
            if rows.len() < 2 {
                break;
            }
            let (a, b) = (rows[0], rows[1]);
            let e = 49 - (a.occ0 | a.occ1).count_ones();
            let bucket = if e < 10 { 0 } else if e < 25 { 1 } else { 2 };
            match crate::gamedata::link_conv(&a, &b) {
                Some(0) => {
                    counts[0] += 1;
                    empt[bucket] += 1;
                }
                Some(_) => {
                    counts[1] += 1;
                    empt[bucket] += 1;
                }
                None => {
                    counts[2] += 1;
                }
            }
        }
        let tot = (counts[0] + counts[1] + counts[2]).max(1) as f64;
        println!("data stitch diag-adjacency: {adj_n} random adjacent pairs: conv0={} conv1={} unlinked={} ({:.1}% linked)", counts[0], counts[1], counts[2], 100.0 * (counts[0] + counts[1]) as f64 / tot);
        return ExitCode::SUCCESS;
    }
    let max_lines = crate::flag_u64(args, "--max-lines").unwrap_or(100_000);
    let max_nodes = crate::flag_u64(args, "--max-nodes").unwrap_or(20_000_000);
    let max_depth = crate::flag_u64(args, "--max-depth").unwrap_or(400) as usize;
    let threads = crate::flag_u64(args, "--threads").unwrap_or(8).max(1) as usize;
    let t0 = std::time::Instant::now();
    let idx = match build_dataset_index(&tdat) {
        Ok(i) => i,
        Err(e) => {
            eprintln!("data stitch: {e}");
            return ExitCode::FAILURE;
        }
    };
    eprintln!("data stitch: index ready in {:.0}s", t0.elapsed().as_secs_f64());
    // Roots: every chain start is a real game beginning (or post-pass resume) WITH its row
    // convention flag, so each resolves to an exact engine board. Dedupe identical roots.
    let chain_path = format!("{}.chains", tdat.trim_end_matches(".tdat"));
    let (idx_rows, chains) = match read_chains(&chain_path) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("data stitch: {e}");
            return ExitCode::FAILURE;
        }
    };
    let mut rs = match RowStream::open(&tdat) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("data stitch: {e}");
            return ExitCode::FAILURE;
        }
    };
    let mut seen_root = std::collections::HashSet::new();
    let mut units: Vec<(Board, Vec<Move>)> = Vec::new();
    let mut root_dead = 0u64;
    for ch in &chains {
        if rs.advance(ch.start as u64, None).is_err() {
            eprintln!("data stitch: chain seek failed at row {}", ch.start);
            return ExitCode::FAILURE;
        }
        let rows = match rs.take(1) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("data stitch: {e}");
                return ExitCode::FAILURE;
            }
        };
        if rows.is_empty() {
            break;
        }
        let r = rows[0];
        let (s0, s1) = if ch.conv == 0 {
            if r.stm == 0 {
                (r.occ0, r.occ1)
            } else {
                (r.occ1, r.occ0)
            }
        } else {
            (r.occ0, r.occ1)
        };
        if !seen_root.insert((s0, s1, r.stm)) {
            continue;
        }
        let b = Board::from_occ(s0, s1, 0, r.stm);
        let kids = attested_children(&b, &idx);
        if kids.is_empty() {
            root_dead += 1;
            continue;
        }
        units.push((b, kids.into_iter().map(|(m, _)| m).collect()));
    }
    println!("data stitch: {} chains -> {} unique roots with attested children ({} roots dead, {} duplicate)", chains.len(), units.len(), root_dead, seen_root.len().saturating_sub(units.len() + root_dead as usize));
    if diag_conv > 0 {
        // Settle the row convention once and for all: for sampled chain-adjacent transitions
        // (parent row -> TRUE next row in the same run), score 4 reconstruction hypotheses by
        // whether engine movegen from the reconstructed parent reproduces the true child.
        // 6 hypotheses, body vs tail separately. A hypothesis scores when engine movegen
        // from its reconstructed parent reproduces the TRUE next row of the same chain run.
        // H0 black-absolute, H1 white-absolute, H2 conv-flag, H3 order-insensitive,
        // H4 turn-flipped raw, H5 turn-flipped swapped (uniform colour-flip of the file).
        let mut tested = [0u64; 2];
        let mut hits = [[0u64; 6]; 2];
        let tail_start = std::fs::metadata(&tdat).map(|m| m.len() / REC as u64).unwrap_or(u64::MAX).saturating_sub(101_830);
        let mut rs2 = match RowStream::open(&tdat) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("data stitch: {e}");
                return ExitCode::FAILURE;
            }
        };
        let mut done = 0usize;
        for ch in &chains {
            if done >= diag_conv {
                break;
            }
            if (ch.len as usize) < 3 {
                continue;
            }
            if rs2.advance(ch.start as u64, None).is_err() {
                break;
            }
            let rows = match rs2.take(ch.len as usize) {
                Ok(r) => r,
                Err(_) => break,
            };
            let i = rows.len() / 2;
            if i + 1 >= rows.len() {
                continue;
            }
            let (p, n) = (rows[i], rows[i + 1]);
            let region = if (ch.start as u64 + i as u64) >= tail_start { 1 } else { 0 };
            let b_black = Board::from_occ(p.occ0, p.occ1, 0, p.stm);
            let b_white = Board::from_occ(p.occ1, p.occ0, 0, p.stm);
            let (bm0, bm1) = if ch.conv == 0 {
                if p.stm == 0 { (p.occ0, p.occ1) } else { (p.occ1, p.occ0) }
            } else {
                (p.occ0, p.occ1)
            };
            let b_flag = Board::from_occ(bm0, bm1, 0, p.stm);
            let b_flip = Board::from_occ(p.occ0, p.occ1, 0, 1 - p.stm);
            tested[region] += 1;
            done += 1;
            let mut hb = [false; 6];
            // H0 black-absolute, H1 white-absolute, H2 conv-flag: exact child match.
            for (h, b) in [&b_black, &b_white, &b_flag].into_iter().enumerate() {
                let swap = h == 1;
                let legal = b.legal_moves();
                for k in 0..legal.len() {
                    let c = b.make(legal.move_at(k));
                    let (e0, e1) = if swap { (c.occ[1], c.occ[0]) } else { (c.occ[0], c.occ[1]) };
                    if e0 == n.occ0 && e1 == n.occ1 && c.turn == n.stm {
                        hb[h] = true;
                        break;
                    }
                }
            }
            // H3 order-insensitive under the black board.
            {
                let legal = b_black.legal_moves();
                for k in 0..legal.len() {
                    let c = b_black.make(legal.move_at(k));
                    if (c.occ[0] == n.occ0 && c.occ[1] == n.occ1 || c.occ[0] == n.occ1 && c.occ[1] == n.occ0) && c.turn == n.stm {
                        hb[3] = true;
                        break;
                    }
                }
            }
            // H4/H5: the file's stm means the opposite colour.
            {
                let legal = b_flip.legal_moves();
                for k in 0..legal.len() {
                    let c = b_flip.make(legal.move_at(k));
                    if c.turn != 1 - n.stm {
                        continue;
                    }
                    if c.occ[0] == n.occ0 && c.occ[1] == n.occ1 {
                        hb[4] = true;
                    }
                    if c.occ[1] == n.occ0 && c.occ[0] == n.occ1 {
                        hb[5] = true;
                    }
                    if hb[4] && hb[5] {
                        break;
                    }
                }
            }
            for h in 0..6 {
                if hb[h] {
                    hits[region][h] += 1;
                }
            }
        }
        println!("data stitch diag-conv: body tested={} black={} white={} convflag={} order-insens={} flip-raw={} flip-swapped={}", tested[0], hits[0][0], hits[0][1], hits[0][2], hits[0][3], hits[0][4], hits[0][5]);
        println!("data stitch diag-conv: tail tested={} black={} white={} convflag={} order-insens={} flip-raw={} flip-swapped={}", tested[1], hits[1][0], hits[1][1], hits[1][2], hits[1][3], hits[1][4], hits[1][5]);
        return ExitCode::SUCCESS;
    }
    if diag_roots > 0 {
        // Diagnose WHY roots are dead: for the first N dead roots, test every key spelling
        // against the whole index and report which (if any) hits.
        let mut shown = 0usize;
        let mut hit_pat = [0u64; 4];
        let mut checked = 0u64;
        'outer: for ch in &chains {
            if shown >= diag_roots {
                break;
            }
            if rs.advance(ch.start as u64, None).is_err() {
                break;
            }
            let rows = match rs.take(1) {
                Ok(r) => r,
                _ => break,
            };
            if rows.is_empty() {
                break;
            }
            let r = rows[0];
            let (s0, s1) = if ch.conv == 0 {
                if r.stm == 0 { (r.occ0, r.occ1) } else { (r.occ1, r.occ0) }
            } else {
                (r.occ0, r.occ1)
            };
            let b = Board::from_occ(s0, s1, 0, r.stm);
            let legal = b.legal_moves();
            let mut any_att = false;
            for i in 0..legal.len() {
                let m = legal.move_at(i);
                let c = b.make(m);
                let stones = (c.occ[0] | c.occ[1]).count_ones() as usize;
                if stones >= 50 {
                    continue;
                }
                let (a0, a1) = if c.turn == 0 { (c.occ[0], c.occ[1]) } else { (c.occ[1], c.occ[0]) };
                // mover-relative rows are stored with stm 0 (D769)
                if idx.count(a0, a1, 0, stones) + idx.count(c.occ[0], c.occ[1], c.turn, stones) > 0 {
                    any_att = true;
                    break;
                }
            }
            if any_att {
                continue;
            }
            // Dead root: try all 4 key spellings of every legal child across ALL buckets.
            checked += 1;
            let mut pat = [false; 4];
            for i in 0..legal.len() {
                let m = legal.move_at(i);
                let c = b.make(m);
                for bb in 0..50 {
                    if idx.buckets[bb].contains_key(&(c.occ[0], c.occ[1], c.turn)) { pat[0] = true; }
                    if idx.buckets[bb].contains_key(&(c.occ[1], c.occ[0], c.turn)) { pat[1] = true; }
                    if idx.buckets[bb].contains_key(&(c.occ[0], c.occ[1], 1 - c.turn)) { pat[2] = true; }
                    if idx.buckets[bb].contains_key(&(c.occ[1], c.occ[0], 1 - c.turn)) { pat[3] = true; }
                }
                for k in 0..4 {
                    if pat[k] {
                        hit_pat[k] += 1;
                    }
                }
                if shown < 3 {
                    eprintln!("diag root stm={} conv={} stones={} legal={} pat={:?}", r.stm, ch.conv, (r.occ0 | r.occ1).count_ones(), legal.len(), pat);
                    shown += 1;
                }
                if shown >= 3 && checked > 50 {
                    // keep counting patterns cheaply without printing
                }
                if checked >= 200 {
                    break 'outer;
                }
            }
        }
        println!("data stitch diag: {checked} dead roots probed; key-spelling hits: raw-turn={} swapped-turn={} raw-flipped={} swapped-flipped={}", hit_pat[0], hit_pat[1], hit_pat[2], hit_pat[3]);
        return ExitCode::SUCCESS;
    }
    if units.is_empty() {
        eprintln!("data stitch: no live roots; nothing to unroll");
        return ExitCode::FAILURE;
    }
    let next_unit = AtomicUsize::new(0);
    let lines_done = AtomicUsize::new(0);
    let nodes_done = AtomicU64::new(0);
    let merged_lines = Mutex::new(Vec::<StLine>::new());
    let merged_visited = Mutex::new(std::collections::HashSet::new());
    let merged_stats = Mutex::new(StStats::default());
    let t1 = std::time::Instant::now();
    std::thread::scope(|s| {
        for _ in 0..threads {
            s.spawn(|| {
                let mut local_lines: Vec<StLine> = Vec::new();
                let mut local_visited: std::collections::HashSet<(u64, u64, u8)> = std::collections::HashSet::new();
                let mut local_st = StStats::default();
                loop {
                    let r = next_unit.fetch_add(1, Ordering::Relaxed);
                    if r >= units.len() {
                        break;
                    }
                    let (rt, rm) = &units[r];
                    stitch_subtree(
                        *rt,
                        rm.clone(),
                        &idx,
                        max_lines,
                        max_nodes,
                        max_depth,
                        &lines_done,
                        &nodes_done,
                        &mut local_lines,
                        &mut local_visited,
                        &mut local_st,
                    );
                    if lines_done.load(Ordering::Relaxed) as u64 >= max_lines {
                        break;
                    }
                }
                merged_lines.lock().unwrap().extend(local_lines);
                let mut mv = merged_visited.lock().unwrap();
                for k in local_visited {
                    mv.insert(k);
                }
                let mut g = merged_stats.lock().unwrap();
                g.nodes += local_st.nodes;
                g.dead_ends += local_st.dead_ends;
                g.capped += local_st.capped;
                for i in 0..8 {
                    g.branch_hist[i] += local_st.branch_hist[i];
                }
            });
        }
    });
    let dt = t1.elapsed().as_secs_f64().max(0.001);
    let lines = merged_lines.lock().unwrap();
    let visited = merged_visited.lock().unwrap();
    let g = merged_stats.lock().unwrap();
    // F3: exact line dedup by path hash.
    let mut seen = std::collections::HashSet::<u64>::new();
    let mut uniq_main = 0u64;
    let mut uniq_purg = 0u64;
    let mut uniq_eval = 0u64;
    for l in lines.iter() {
        let mut h: u64 = 0xcbf29ce484222325;
        for &c in &l.path {
            h ^= c as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
        h ^= l.result as u64;
        h = h.wrapping_mul(0x100000001b3);
        if !seen.insert(h) {
            continue;
        }
        if l.amb_mid == 0 && l.amb_open == 0 {
            uniq_main += 1;
        } else if l.amb_mid == 0 {
            uniq_eval += 1;
        } else {
            uniq_purg += 1;
        }
    }
    println!("data stitch: {} lines completed in {:.1}s ({:.0} lines/s, {:.0} nodes/s)", lines.len(), dt, lines.len() as f64 / dt, g.nodes as f64 / dt);
    println!("  unique lines: {} (main-clean {} | opening-only {} | purgatory {})", seen.len(), uniq_main, uniq_eval, uniq_purg);
    println!("  dead ends: {} | capped stops: {} | nodes visited: {} | positions covered: {}", g.dead_ends, g.capped, g.nodes, visited.len());
    println!("  branching histogram (attested children per node, capped at 7): {:?}", g.branch_hist);
    ExitCode::SUCCESS
}

struct BmCfg {
    checks: Vec<u64>,
    budget: u64,
    label_bound: i64,
    gap_conf: i64,
    gap_dec: i64,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum BmKind {
    Forced,
    Terminal,
    Decisive,
    Confident,
    Full,
    NoResult,
}

struct BmOutcome {
    kind: BmKind,
    early: Option<Move>,
    full: Option<Move>,
    nodes: u64,
    had_options: bool,
    label_ok: bool,
    gap_small: bool,
    /// label-|eval| bin at first checkpoint (0..6) or 255; gap bin at first exclusion (0..4) or 255.
    d0bin: u8,
    g0bin: u8,
}

fn diff_bin(d: i64) -> u8 {
    let a = d.abs() as u64;
    if a < 3 { 0 } else if a < 7 { 1 } else if a < 15 { 2 } else if a < 30 { 3 } else if a < 60 { 4 } else if a < 150 { 5 } else { 6 }
}

fn gap_bin(g: i64) -> u8 {
    let a = g as u64;
    if a < 10 { 0 } else if a < 20 { 1 } else if a < 30 { 2 } else if a < 50 { 3 } else { 4 }
}

#[derive(Default)]
struct BmStats {
    n: u64,
    forced: u64,
    terminal: u64,
    no_options: u64,
    no_result: u64,
    label_ok: u64,
    gap_small: u64,
    decisive: u64,
    confident: u64,
    full: u64,
    agree: u64,
    nodes: u64,
    hdiff: [u64; 7],
    hgap: [u64; 5],
}

/// Stratum 0..7: empties bucket (0-9, 10-19, 20-29, 30+) x proven; 8 = fresh tail (forced sample).
fn bm_stratum(rec: &Rec) -> u8 {
    let empties = 49 - (rec.occ0 | rec.occ1).count_ones();
    let b = if empties < 10 { 0 } else if empties < 20 { 1 } else if empties < 30 { 2 } else { 3 };
    let p = if rec.cp.abs() >= 98000.0 { 1 } else { 0 };
    b * 2 + p
}

fn bm_process_row(
    rec: &Rec,
    searcher: &mut titanium::search::Searcher,
    idx: &DatasetIndex,
    cfg: &BmCfg,
    finish: bool,
) -> BmOutcome {
    let b = Board::from_occ(rec.occ0, rec.occ1, 0, rec.stm);
    let legal = b.legal_moves();
    if legal.len() == 0 {
        return BmOutcome { kind: BmKind::Terminal, early: None, full: None, nodes: 0, had_options: false, label_ok: false, gap_small: false, d0bin: 255, g0bin: 255 };
    }
    if legal.len() == 1 {
        let m = legal.move_at(0);
        return BmOutcome { kind: BmKind::Forced, early: Some(m), full: Some(m), nodes: 0, had_options: false, label_ok: false, gap_small: false, d0bin: 255, g0bin: 255 };
    }
    // Dataset-attested: the child position occurs anywhere in the 29M rows (either row convention:
    // mover-relative occ0 or colour-absolute occ0). One O(1) probe in exactly one stone-count bucket.
    let attested = |m: Move| {
        let c = b.make(m);
        let stones = (c.occ[0] | c.occ[1]).count_ones() as usize;
        if stones >= 50 {
            return false;
        }
        let (a0, a1) = if c.turn == 0 { (c.occ[0], c.occ[1]) } else { (c.occ[1], c.occ[0]) };
        // mover-relative rows are stored with stm 0 (D769); probing them with stm = c.turn never matched
        idx.attested(a0, a1, 0, stones) || idx.attested(c.occ[0], c.occ[1], c.turn, stones)
    };
    let mut had_options = false;
    for i in 0..legal.len() {
        if attested(legal.move_at(i)) {
            had_options = true;
            break;
        }
    }
    // Fast path: nothing in the dataset attests any child, so no gate can ever fire.
    // One straight budget search, no chunking, no exclusion.
    if !had_options {
        let limits = SearchLimits { time: None, max_nodes: Some(cfg.budget), max_depth: 30 };
        let r = searcher.search_bz(&b, &limits);
        let nodes = r.nodes as u64;
        return match r.best {
            Some(m) => BmOutcome { kind: BmKind::Full, early: None, full: Some(m), nodes, had_options: false, label_ok: false, gap_small: false, d0bin: 255, g0bin: 255 },
            None => BmOutcome { kind: BmKind::NoResult, early: None, full: None, nodes, had_options: false, label_ok: false, gap_small: false, d0bin: 255, g0bin: 255 },
        };
    }
    let label = rec.cp.round() as i64;
    let mut nodes = 0u64;
    let mut early: Option<(BmKind, Move)> = None;
    let mut full_mv: Option<Move> = None;
    let mut saw_label = false;
    let mut saw_small = false;
    let mut prev: Option<(Move, i32)> = None;
    let mut d0: u8 = 255;
    let mut g0: u8 = 255;
    for (ci, &cp) in cfg.checks.iter().enumerate() {
        let last = ci + 1 == cfg.checks.len();
        let limits = SearchLimits { time: None, max_nodes: Some(cp), max_depth: 30 };
        let r = searcher.search_bz(&b, &limits);
        nodes += r.nodes as u64;
        let Some(m1) = r.best else {
            if last {
                return BmOutcome { kind: BmKind::NoResult, early: early.map(|(_, m)| m), full: full_mv, nodes, had_options, label_ok: saw_label, gap_small: saw_small, d0bin: d0, g0bin: g0 };
            }
            continue;
        };
        full_mv = Some(m1);
        if ci == 0 && had_options {
            d0 = diff_bin(r.score as i64 - label);
        }
        // Stability: the same root move at two consecutive checkpoints. Transient 1k-node
        // blips never fire; earliest possible fire is the second checkpoint.
        let stable = match prev {
            Some((pm, _)) => pm.from == m1.from && pm.to == m1.to,
            None => false,
        };
        prev = Some((m1, r.score));
        if !last && early.is_none() && had_options && stable && attested(m1) && (r.score as i64 - label).abs() <= cfg.label_bound {
            saw_label = true;
            let re = searcher.search_bz_excluding(&b, &limits, &[m1]);
            nodes += re.nodes as u64;
            let gap = match re.best {
                Some(_) => r.score - re.score,
                None => i32::MAX / 2, // best stands alone: maximally decisive
            };
            if g0 == 255 {
                g0 = gap_bin(gap as i64);
            }
            if gap as i64 >= cfg.gap_dec {
                early = Some((BmKind::Decisive, m1));
            } else if gap as i64 >= cfg.gap_conf {
                early = Some((BmKind::Confident, m1));
            } else {
                saw_small = true;
            }
            if !finish {
                if let Some((k, m)) = early {
                    return BmOutcome { kind: k, early: Some(m), full: Some(m), nodes, had_options, label_ok: saw_label, gap_small: saw_small, d0bin: d0, g0bin: g0 };
                }
            }
        }
        if last {
            break;
        }
    }
    match early {
        Some((BmKind::Decisive, m)) => BmOutcome { kind: BmKind::Decisive, early: Some(m), full: full_mv, nodes, had_options, label_ok: saw_label, gap_small: saw_small, d0bin: d0, g0bin: g0 },
        Some((BmKind::Confident, m)) => BmOutcome { kind: BmKind::Confident, early: Some(m), full: full_mv, nodes, had_options, label_ok: saw_label, gap_small: saw_small, d0bin: d0, g0bin: g0 },
        _ => BmOutcome {
            kind: if full_mv.is_some() { BmKind::Full } else { BmKind::NoResult },
            early: None,
            full: full_mv,
            nodes,
            had_options,
            label_ok: saw_label,
            gap_small: saw_small,
            d0bin: d0,
            g0bin: g0,
        },
    }
}

fn bm_merge(dst: &mut BmStats, s: &BmStats, strat: &mut [[u64; 5]; 9], ls: &[[u64; 5]; 9]) {
    dst.n += s.n;
    dst.forced += s.forced;
    dst.terminal += s.terminal;
    dst.no_options += s.no_options;
    dst.no_result += s.no_result;
    dst.label_ok += s.label_ok;
    dst.gap_small += s.gap_small;
    dst.decisive += s.decisive;
    dst.confident += s.confident;
    dst.full += s.full;
    dst.agree += s.agree;
    dst.nodes += s.nodes;
    for i in 0..7 {
        dst.hdiff[i] += s.hdiff[i];
    }
    for i in 0..5 {
        dst.hgap[i] += s.hgap[i];
    }
    for i in 0..9 {
        for j in 0..5 {
            strat[i][j] += ls[i][j];
        }
    }
}

pub fn run_bestmove(args: &[String]) -> ExitCode {
    let Some(tdat) = crate::flag_str(args, "--tdat") else {
        eprintln!("usage: ti data bestmove --tdat X.tdat [--out STEM] [--rows N] [--seed S] [--all] [--threads 4] [--tt-bits 18] [--network bundled] [--label-bound 150] [--gap-conf 30] [--gap-dec 50] [--first-check 1000] [--fresh-rows 101830]");
        return ExitCode::FAILURE;
    };
    let out = crate::flag_str(args, "--out").unwrap_or_else(|| format!("{}.best", tdat.trim_end_matches(".tdat")));
    let all = crate::has_flag(args, "--all");
    let want_rows = crate::flag_u64(args, "--rows").unwrap_or(10_000) as usize;
    let seed = crate::flag_u64(args, "--seed").unwrap_or(7) | 1;
    let threads = crate::flag_u64(args, "--threads").unwrap_or(4).max(1) as usize;
    let tt_bits = crate::flag_u64(args, "--tt-bits").unwrap_or(18) as usize;
    let network = crate::flag_str(args, "--network").unwrap_or_else(|| "bundled".into());
    let label_bound = crate::flag_u64(args, "--label-bound").unwrap_or(150) as i64;
    let gap_conf = crate::flag_u64(args, "--gap-conf").unwrap_or(30) as i64;
    let gap_dec = crate::flag_u64(args, "--gap-dec").unwrap_or(50) as i64;
    let first = crate::flag_u64(args, "--first-check").unwrap_or(1000);
    let fresh_rows = crate::flag_u64(args, "--fresh-rows").unwrap_or(101_830);
    let budget = 5000u64;
    let mut checks = vec![first, 2000, 3000, 4000];
    checks.retain(|&c| c < budget);
    checks.push(budget);
    let flat = crate::has_flag(args, "--flat");
    if flat {
        // No chunking, no exclusion: one straight budget search per row. The move recorded IS
        // the budget-best, so early-vs-full disagreement is zero by construction. Early stop
        // only pays when fires dominate; at single-digit fire rates chunking costs ~3x.
        checks = vec![budget];
    }
    let cfg = BmCfg { checks, budget, label_bound, gap_conf, gap_dec };

    let owned_net: Option<titanium::nnue::Nnue> = if network == "bundled" || network == "none" {
        None
    } else {
        match titanium::nnue::Nnue::load(&network) {
            Ok(n) => Some(n),
            Err(e) => {
                eprintln!("data bestmove: cannot load network {network}: {e}");
                return ExitCode::FAILURE;
            }
        }
    };
    let net_wrapped: Option<&titanium::nnue::Nnue> = match owned_net.as_ref() {
        Some(n) => Some(n),
        None if network == "none" => None,
        None => Some(titanium::nnue::Nnue::bundled_static()),
    };

    // Work items: (row index, row, stratum). Smoke samples stratified; --all takes every row.
    #[derive(Clone, Copy)]
    struct Item {
        idx: u64,
        rec: Rec,
        stratum: u8,
    }
    let corpus_rows = match File::open(&tdat).and_then(|f| f.metadata()) {
        Ok(m) => m.len() / REC as u64,
        Err(e) => {
            eprintln!("data bestmove: {e}");
            return ExitCode::FAILURE;
        }
    };
    if corpus_rows == 0 {
        eprintln!("data bestmove: {tdat}: 0 rows");
        return ExitCode::FAILURE;
    }
    // Smoke sample up front (stratified reservoir, one streaming pass); --all streams blocks below.
    let mut sampled: Option<Vec<Item>> = if all {
        None
    } else {
        // Stratified reservoir sample: 8 body strata + fresh tail, one streaming pass.
        let f = match File::open(&tdat) {
            Ok(f) => f,
            Err(e) => {
                eprintln!("data bestmove: {tdat}: {e}");
                return ExitCode::FAILURE;
            }
        };
        let total = corpus_rows;
        let tail_start = total.saturating_sub(fresh_rows);
        let mut caps = [0usize; 9];
        let body_each = (want_rows * 92 / 100) / 8;
        for i in 0..8 {
            caps[i] = body_each;
        }
        caps[8] = want_rows.saturating_sub(body_each * 8);
        let mut res: [Vec<Item>; 9] = Default::default();
        let mut seen = [0u64; 9];
        let mut rng = seed;
        let mut next_u64 = || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            rng
        };
        let mut rd = BufReader::with_capacity(1 << 20, f);
        let mut buf = vec![0u8; REC * 16384];
        let mut idx = 0u64;
        let mut left = total;
        while left > 0 {
            let n = (left as usize).min(16384);
            if rd.read_exact(&mut buf[..n * REC]).is_err() {
                eprintln!("data bestmove: short read at row {idx}");
                return ExitCode::FAILURE;
            }
            for c in buf[..n * REC].chunks_exact(REC) {
                let r = Rec::from_bytes(c);
                let s = if idx >= tail_start { 8 } else { bm_stratum(&r) as usize };
                let it = Item { idx, rec: r, stratum: s as u8 };
                if res[s].len() < caps[s] {
                    res[s].push(it);
                } else if caps[s] > 0 {
                    let j = (next_u64() % (seen[s] + 1)) as usize;
                    if j < caps[s] {
                        res[s][j] = it;
                    }
                }
                seen[s] += 1;
                idx += 1;
            }
            left -= n as u64;
        }
        Some(res.into_iter().flatten().collect())
    };
    if !all && sampled.as_ref().map_or(true, |v| v.is_empty()) {
        eprintln!("data bestmove: no work items");
        return ExitCode::FAILURE;
    }

    // Piece-count index over the whole corpus: the attestation source for every gate.
    // Built once (~minutes, ~2-3 GB), shared read-only, never stored.
    let t0 = std::time::Instant::now();
    let idx = match build_dataset_index(&tdat) {
        Ok(i) => i,
        Err(e) => {
            eprintln!("data bestmove: {e}");
            return ExitCode::FAILURE;
        }
    };
    eprintln!("data bestmove: index ready in {:.0}s", t0.elapsed().as_secs_f64());
    let next = AtomicUsize::new(0);
    let merged_stats = Mutex::new(BmStats::default());
    let merged_strat = Mutex::new([[0u64; 5]; 9]);
    let codes: Mutex<Vec<(u64, u16)>> = Mutex::new(Vec::new());
    let finish = !all; // smoke completes to 5k for agreement; --all stops at first verdict
    let mut sidecar: Option<BufWriter<File>> = None;
    let mut sidecar_n = 0u64;
    if all {
        let path = format!("{}.best", out.trim_end_matches(".best"));
        match File::create(&path) {
            Ok(f) => sidecar = Some(BufWriter::with_capacity(1 << 20, f)),
            Err(e) => {
                eprintln!("data bestmove: cannot create {path}: {e}");
                return ExitCode::FAILURE;
            }
        }
    }
    let mut rs_main: Option<RowStream> = None;
    let mut block_start = 0u64;
    const BLOCK: u64 = 65536;
    loop {
        // One block per iteration (--all, streamed) or the single sample (smoke).
        let items: Vec<Item> = if all {
            if rs_main.is_none() {
                match RowStream::open(&tdat) {
                    Ok(r) => rs_main = Some(r),
                    Err(e) => {
                        eprintln!("data bestmove: {e}");
                        return ExitCode::FAILURE;
                    }
                }
            }
            let rd = match rs_main.as_mut() {
                Some(r) => r,
                None => {
                    eprintln!("data bestmove: reader missing");
                    return ExitCode::FAILURE;
                }
            };
            let n = (corpus_rows - block_start).min(BLOCK) as usize;
            if n == 0 {
                Vec::new()
            } else {
                match rd.take(n) {
                    Ok(rows) => {
                        let mut v = Vec::with_capacity(n);
                        for (k, r) in rows.into_iter().enumerate() {
                            v.push(Item { idx: block_start + k as u64, rec: r, stratum: bm_stratum(&r) });
                        }
                        block_start += n as u64;
                        v
                    }
                    Err(e) => {
                        eprintln!("data bestmove: {e}");
                        return ExitCode::FAILURE;
                    }
                }
            }
        } else {
            match sampled.take() {
                Some(v) => v,
                None => Vec::new(),
            }
        };
        if items.is_empty() {
            break;
        }
        next.store(0, Ordering::Relaxed);
    std::thread::scope(|s| {
        for _ in 0..threads {
            s.spawn(|| {
                let mut searcher = titanium::search::Searcher::with_tt_bits_unconfigured(tt_bits);
                searcher.set_nnue(net_wrapped);
                let mut ls = BmStats::default();
                let mut lstrat = [[0u64; 5]; 9];
                let mut lcodes: Vec<(u64, u16)> = Vec::new();
                loop {
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    if i >= items.len() {
                        break;
                    }
                    let it = &items[i];
                    let o = bm_process_row(&it.rec, &mut searcher, &idx, &cfg, finish);
                    ls.n += 1;
                    ls.nodes += o.nodes;
                    let s = it.stratum as usize;
                    lstrat[s][0] += 1;
                    lstrat[s][4] += o.nodes;
                    let same = |a: Option<Move>, b: Option<Move>| match (a, b) {
                        (Some(x), Some(y)) => x.from == y.from && x.to == y.to,
                        _ => false,
                    };
                    let mv_code = |m: Option<Move>| -> u16 {
                        match m {
                            None => BEST_UNKNOWN,
                            Some(m) if m.is_pass() => PASS_CODE,
                            Some(m) => move_code(m),
                        }
                    };
                    match o.kind {
                        BmKind::Forced => {
                            ls.forced += 1;
                            lcodes.push((it.idx, mv_code(o.full)));
                        }
                        BmKind::Terminal => {
                            ls.terminal += 1;
                            lcodes.push((it.idx, BEST_UNKNOWN));
                        }
                        BmKind::Decisive => {
                            ls.decisive += 1;
                            lstrat[s][2] += 1;
                            if same(o.early, o.full) {
                                ls.agree += 1;
                                lstrat[s][3] += 1;
                            }
                            lcodes.push((it.idx, mv_code(o.full)));
                        }
                        BmKind::Confident => {
                            ls.confident += 1;
                            lstrat[s][2] += 1;
                            if same(o.early, o.full) {
                                ls.agree += 1;
                                lstrat[s][3] += 1;
                            }
                            lcodes.push((it.idx, mv_code(o.full)));
                        }
                        BmKind::Full => {
                            ls.full += 1;
                            lstrat[s][1] += 1;
                            lcodes.push((it.idx, mv_code(o.full)));
                        }
                        BmKind::NoResult => {
                            ls.no_result += 1;
                            lstrat[s][1] += 1;
                            lcodes.push((it.idx, BEST_UNKNOWN));
                        }
                    }
                    // searched rows with zero dataset options can never fire the gate
                    if !matches!(o.kind, BmKind::Forced | BmKind::Terminal) && !o.had_options {
                        ls.no_options += 1;
                    }
                    if o.label_ok {
                        ls.label_ok += 1;
                    }
                    if o.d0bin < 7 {
                        ls.hdiff[o.d0bin as usize] += 1;
                    }
                    if o.g0bin < 5 {
                        ls.hgap[o.g0bin as usize] += 1;
                    }
                    if o.gap_small {
                        ls.gap_small += 1;
                    }
                }
                bm_merge(&mut merged_stats.lock().unwrap(), &ls, &mut merged_strat.lock().unwrap(), &lstrat);
                codes.lock().unwrap().extend(lcodes);
            });
        }
    });
        if all {
            let mut v = codes.lock().unwrap();
            v.sort_by_key(|&(i, _)| i);
            let w = match sidecar.as_mut() {
                Some(w) => w,
                None => {
                    eprintln!("data bestmove: sidecar missing");
                    return ExitCode::FAILURE;
                }
            };
            for &(_, c) in v.iter() {
                if w.write_all(&c.to_le_bytes()).is_err() {
                    eprintln!("data bestmove: sidecar write failed");
                    return ExitCode::FAILURE;
                }
            }
            sidecar_n += v.len() as u64;
            v.clear();
        }
    }
    let st = merged_stats.lock().unwrap();
    let sstrat = merged_strat.lock().unwrap();
    let searched = st.n - st.forced - st.terminal;
    let fired = st.decisive + st.confident;
    let burned = st.full + st.no_result;
    let full_pct = if searched > 0 { 100.0 * burned as f64 / searched as f64 } else { 0.0 };
    let flips = if fired > 0 { 100.0 * (fired - st.agree) as f64 / fired as f64 } else { 0.0 };
    println!("data bestmove: {} rows ({} forced 0-node, {} terminal skipped, {} with no dataset options) over {} threads", st.n, st.forced, st.terminal, st.no_options, threads);
    println!("  funnel: searched={} had-options={} label-ok={} gap-small={}", searched, searched - st.no_options, st.label_ok, st.gap_small);
    println!("  |e1-label|@1k bins [0-3,3-7,7-15,15-30,30-60,60-150,150+]: {:?}", st.hdiff);
    println!("  gap bins [0-10,10-20,20-30,30-50,50+]: {:?}", st.hgap);
    println!("  decisive {} | confident {} | full-budget {} ({:.1}% of searched, incl. {} no-result)", st.decisive, st.confident, burned, full_pct, st.no_result);
    println!("  early-vs-full agreement: {}/{} ({:.2}% flips)", st.agree, fired, flips);
    println!("  avg nodes/row incl. forced: {:.0}; excl. forced/terminal: {:.0}", st.nodes as f64 / st.n.max(1) as f64, st.nodes as f64 / searched.max(1) as f64);
    println!("  stratum (n, burned, fired, agree, nodes):");
    for i in 0..9 {
        println!("    s{}: n={} burned={} fired={} agree={} nodes={}", i, sstrat[i][0], sstrat[i][1], sstrat[i][2], sstrat[i][3], sstrat[i][4]);
    }
    let avg_all = st.nodes as f64 / st.n.max(1) as f64;
    let total_f = corpus_rows as f64;
    println!("  extrapolated full corpus: {:.1}h single-thread, {:.1}h on {} threads", total_f * avg_all / 3.6e9, total_f * avg_all / 3.6e9 / threads as f64, threads);
    let pass = full_pct < 10.0 && flips < 1.0;
    println!("  smoke {}: full-budget {:.1}% (target <10%), flips {:.2}% (target <1%)", if pass { "PASS" } else { "FAIL" }, full_pct, flips);

    if all {
        if sidecar_n != corpus_rows {
            eprintln!("data bestmove: sidecar holds {sidecar_n} codes, expected {corpus_rows}; refusing manifest");
            return ExitCode::FAILURE;
        }
        let mut ok = true;
        if let Some(w) = sidecar.as_mut() {
            if w.flush().is_err() {
                eprintln!("data bestmove: flush failed");
                ok = false;
            }
        }
        if ok {
            let manifest = format!("{{\"src\":\"{tdat} via ti data bestmove\",\"rows\":{sidecar_n},\"rec\":2}}");
            let mpath = format!("{}.manifest.json", out.trim_end_matches(".best"));
            if std::fs::write(&mpath, manifest).is_err() {
                eprintln!("data bestmove: manifest write failed");
                ok = false;
            }
        }
        if ok {
            println!("data bestmove: wrote {}.best ({} u16 codes)", out.trim_end_matches(".best"), sidecar_n);
        } else {
            return ExitCode::FAILURE;
        }
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

/// D1177: FNV-1a 64 over a file's bytes (portable, no table; used to prove two copies of a dataset are byte-identical,
/// e.g. the PC file and the one built on Modal)
fn fnv_file(p: &str) -> Result<(u64, u64), String> {
    let mut r = BufReader::with_capacity(1 << 22, File::open(p).map_err(|e| format!("{p}: {e}"))?);
    let (mut h, mut n) = (0xcbf2_9ce4_8422_2325u64, 0u64);
    let mut buf = vec![0u8; 1 << 22];
    loop {
        let k = r.read(&mut buf).map_err(|e| format!("{p}: {e}"))?;
        if k == 0 {
            break;
        }
        for &b in &buf[..k] {
            h = (h ^ b as u64).wrapping_mul(0x0000_0100_0000_01b3);
        }
        n += k as u64;
    }
    Ok((h, n))
}

// D1257 (user): a new dataset version goes to Modal as a DIFF against the version already there, never in full.
// .tdif (40-byte header) = "TDIF" | u32 version 1 | u64 rows | u64 base fnv1a64 | u64 new fnv1a64 | u64 changed rows, then per changed row:
// u32 row index | u16 mask of the 4-byte words (12 per 48-byte record) that differ | those words, in order.
const TDIF_MAGIC: &[u8; 4] = b"TDIF";
const WORDS: usize = REC / 4;

fn fnv_step(mut h: u64, bytes: &[u8]) -> u64 {
    for &b in bytes {
        h = (h ^ b as u64).wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// `ti data diff --base A.tdat --new B.tdat --out D.tdif`: the rows of B that differ from A (same row count required).
pub fn run_diff(args: &[String]) -> ExitCode {
    let (Some(base), Some(new), Some(out)) = (crate::flag_str(args, "--base"), crate::flag_str(args, "--new"), crate::flag_str(args, "--out")) else {
        eprintln!("usage: ti data diff --base A.tdat --new B.tdat --out D.tdif");
        return ExitCode::FAILURE;
    };
    let r = || -> Result<String, String> {
        let len = |p: &str| fs::metadata(p).map(|m| m.len()).map_err(|e| format!("{p}: {e}"));
        let (la, lb) = (len(&base)?, len(&new)?);
        if la != lb || la % REC as u64 != 0 {
            return Err(format!("row counts differ or not whole rows: {base} {la} B, {new} {lb} B"));
        }
        let rows = la / REC as u64;
        let (mut ra, mut rb) = (BufReader::with_capacity(1 << 22, File::open(&base).map_err(|e| e.to_string())?), BufReader::with_capacity(1 << 22, File::open(&new).map_err(|e| e.to_string())?));
        let mut body: Vec<u8> = Vec::new();
        let (mut ha, mut hb, mut changed, mut words) = (0xcbf2_9ce4_8422_2325u64, 0xcbf2_9ce4_8422_2325u64, 0u64, 0u64);
        let (mut a, mut b) = (vec![0u8; REC * 65536], vec![0u8; REC * 65536]);
        let mut row = 0u64;
        while row < rows {
            let n = ((rows - row) as usize).min(65536);
            ra.read_exact(&mut a[..n * REC]).map_err(|e| e.to_string())?;
            rb.read_exact(&mut b[..n * REC]).map_err(|e| e.to_string())?;
            ha = fnv_step(ha, &a[..n * REC]);
            hb = fnv_step(hb, &b[..n * REC]);
            for i in 0..n {
                let (x, y) = (&a[i * REC..(i + 1) * REC], &b[i * REC..(i + 1) * REC]);
                if x == y {
                    continue;
                }
                let mut mask = 0u16;
                for w in 0..WORDS {
                    if x[w * 4..w * 4 + 4] != y[w * 4..w * 4 + 4] {
                        mask |= 1 << w;
                    }
                }
                body.extend_from_slice(&((row + i as u64) as u32).to_le_bytes());
                body.extend_from_slice(&mask.to_le_bytes());
                for w in 0..WORDS {
                    if mask & (1 << w) != 0 {
                        body.extend_from_slice(&y[w * 4..w * 4 + 4]);
                        words += 1;
                    }
                }
                changed += 1;
            }
            row += n as u64;
        }
        let mut f = BufWriter::new(File::create(&out).map_err(|e| format!("{out}: {e}"))?);
        f.write_all(TDIF_MAGIC).map_err(|e| e.to_string())?;
        for v in [1u64.to_le_bytes()[..4].to_vec(), rows.to_le_bytes().to_vec(), ha.to_le_bytes().to_vec(), hb.to_le_bytes().to_vec(), changed.to_le_bytes().to_vec()] {
            f.write_all(&v).map_err(|e| e.to_string())?;
        }
        f.write_all(&body).map_err(|e| e.to_string())?;
        f.flush().map_err(|e| e.to_string())?;
        Ok(format!(
            "data diff: {base} [fnv {ha:016X}] -> {new} [fnv {hb:016X}]: {rows} rows, {changed} changed ({:.2}%), {words} words, diff {out} = {} B ({:.1}% of the full file)",
            100.0 * changed as f64 / rows.max(1) as f64,
            body.len() + 40,
            100.0 * (body.len() + 40) as f64 / la.max(1) as f64
        ))
    };
    match r() {
        Ok(s) => {
            println!("{s}");
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("data diff: {e}");
            ExitCode::FAILURE
        }
    }
}

/// `ti data patch --base A.tdat --diff D.tdif --out B.tdat`: rebuilds B from A + the diff. Refuses a base whose
/// fingerprint is not the one the diff was made from; writes B.tdat.tmp and renames it only if its fingerprint equals
/// the one recorded in the diff (so a wrong or half result never takes the name). Never overwrites an existing B.
pub fn run_patch(args: &[String]) -> ExitCode {
    let (Some(base), Some(diff), Some(out)) = (crate::flag_str(args, "--base"), crate::flag_str(args, "--diff"), crate::flag_str(args, "--out")) else {
        eprintln!("usage: ti data patch --base A.tdat --diff D.tdif --out B.tdat");
        return ExitCode::FAILURE;
    };
    let r = || -> Result<String, String> {
        if std::path::Path::new(&out).exists() {
            return Err(format!("{out} already exists; refusing to overwrite"));
        }
        let d = fs::read(&diff).map_err(|e| format!("{diff}: {e}"))?;
        if d.len() < 40 || &d[..4] != TDIF_MAGIC {
            return Err(format!("{diff}: not a TDIF file"));
        }
        let u64_at = |o: usize| u64::from_le_bytes(d[o..o + 8].try_into().unwrap());
        let (rows, hbase, hnew, changed) = (u64_at(8), u64_at(16), u64_at(24), u64_at(32));
        let la = fs::metadata(&base).map(|m| m.len()).map_err(|e| format!("{base}: {e}"))?;
        if la != rows * REC as u64 {
            return Err(format!("{base}: {la} B, the diff expects {rows} rows"));
        }
        let tmp = format!("{out}.tmp");
        let mut ra = BufReader::with_capacity(1 << 22, File::open(&base).map_err(|e| e.to_string())?);
        let mut w = BufWriter::with_capacity(1 << 22, File::create(&tmp).map_err(|e| format!("{tmp}: {e}"))?);
        let (mut hb, mut ho) = (0xcbf2_9ce4_8422_2325u64, 0xcbf2_9ce4_8422_2325u64);
        let mut p = 40usize;
        let mut next = |p: &mut usize| -> Option<(u64, u16, usize)> {
            if *p + 6 > d.len() {
                return None;
            }
            let idx = u32::from_le_bytes(d[*p..*p + 4].try_into().unwrap()) as u64;
            let mask = u16::from_le_bytes(d[*p + 4..*p + 6].try_into().unwrap());
            let at = *p + 6;
            *p = at + 4 * mask.count_ones() as usize;
            Some((idx, mask, at))
        };
        let mut pend = next(&mut p);
        let (mut applied, mut buf, mut row) = (0u64, vec![0u8; REC * 65536], 0u64);
        while row < rows {
            let n = ((rows - row) as usize).min(65536);
            ra.read_exact(&mut buf[..n * REC]).map_err(|e| e.to_string())?;
            hb = fnv_step(hb, &buf[..n * REC]);
            while let Some((idx, mask, at)) = pend {
                if idx >= row + n as u64 {
                    break;
                }
                let o = (idx - row) as usize * REC;
                let mut k = at;
                for wi in 0..WORDS {
                    if mask & (1 << wi) != 0 {
                        buf[o + wi * 4..o + wi * 4 + 4].copy_from_slice(&d[k..k + 4]);
                        k += 4;
                    }
                }
                applied += 1;
                pend = next(&mut p);
            }
            ho = fnv_step(ho, &buf[..n * REC]);
            w.write_all(&buf[..n * REC]).map_err(|e| e.to_string())?;
            row += n as u64;
        }
        w.flush().map_err(|e| e.to_string())?;
        drop(w);
        let fail = |m: String| {
            let _ = fs::remove_file(&tmp);
            Err(m)
        };
        if hb != hbase {
            return fail(format!("{base} fnv {hb:016X} is not the diff's base {hbase:016X}; nothing written"));
        }
        if applied != changed || ho != hnew {
            return fail(format!("result fnv {ho:016X} (applied {applied} of {changed}) != expected {hnew:016X}; nothing written"));
        }
        fs::rename(&tmp, &out).map_err(|e| e.to_string())?;
        Ok(format!("data patch: {base} [fnv {hb:016X}] + {diff} ({changed} rows) -> {out} [fnv {ho:016X}] = verified"))
    };
    match r() {
        Ok(s) => {
            println!("{s}");
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("data patch: {e}");
            ExitCode::FAILURE
        }
    }
}

/// `ti data fp --tdat X.tdat [--rows A..B]`: FNV-1a 64 fingerprint of a file (or of rows [A, B) via a temp-free stream)
pub fn run_fp(args: &[String]) -> ExitCode {
    let Some(p) = crate::flag_str(args, "--tdat") else {
        eprintln!("usage: ti data fp --tdat X.tdat");
        return ExitCode::FAILURE;
    };
    match fnv_file(&p) {
        Ok((h, n)) => {
            println!("data fp: {p} {n} bytes ({} rows) fnv1a64 {h:016X}", n / REC as u64);
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("data fp: {e}");
            ExitCode::FAILURE
        }
    }
}

/// D1177: `ti data split --base STEM --tail N --head-out A --tail-out B`: rows [0, nb-N) -> A, the last N rows -> B
/// (.tdat + .manifest.json only: what the trainer reads). The base is untouched; outputs are never overwritten.
pub fn run_split(args: &[String]) -> ExitCode {
    let (Some(base), Some(tail), Some(ho), Some(to)) = (
        crate::flag_str(args, "--base"),
        crate::flag_u64(args, "--tail"),
        crate::flag_str(args, "--head-out"),
        crate::flag_str(args, "--tail-out"),
    ) else {
        eprintln!("usage: ti data split --base STEM --tail N --head-out STEM --tail-out STEM");
        return ExitCode::FAILURE;
    };
    let strip = |s: &str| s.trim_end_matches(".tdat").to_string();
    let (base, ho, to) = (strip(&base), strip(&ho), strip(&to));
    let r = || -> Result<(), String> {
        for o in [&ho, &to] {
            if *o == base || std::path::Path::new(&format!("{o}.tdat")).exists() {
                return Err(format!("{o}.tdat exists or equals the base; refusing to overwrite"));
            }
        }
        let src = format!("{base}.tdat");
        let nb = fs::metadata(&src).map_err(|e| format!("{src}: {e}"))?.len() / REC as u64;
        if tail == 0 || tail >= nb {
            return Err(format!("--tail {tail} must be in 1..{nb}"));
        }
        let cut = nb - tail;
        let mut rd = BufReader::with_capacity(1 << 22, File::open(&src).map_err(|e| format!("{src}: {e}"))?);
        for (o, rows, what) in [(&ho, cut, format!("rows [0, {cut})")), (&to, tail, format!("rows [{cut}, {nb}) = the held-out tail"))] {
            let dst = format!("{o}.tdat");
            let mut w = BufWriter::with_capacity(1 << 22, File::create(&dst).map_err(|e| format!("{dst}: {e}"))?);
            let k = std::io::copy(&mut (&mut rd).take(rows * REC as u64), &mut w).map_err(|e| format!("{dst}: {e}"))?;
            w.flush().map_err(|e| e.to_string())?;
            if k != rows * REC as u64 {
                return Err(format!("{dst}: wrote {k} bytes, expected {}", rows * REC as u64));
            }
            let m = format!("{{\"src\":{},\"rows\":{rows},\"rec\":{REC}}}", crate::json_str(&format!("{base}.tdat {what} (ti data split)")));
            fs::write(format!("{o}.manifest.json"), m).map_err(|e| e.to_string())?;
            let (h, n) = fnv_file(&dst)?;
            println!("data split: {dst} = {base}.tdat {what}: {rows} rows, {n} bytes, fnv1a64 {h:016X}");
        }
        Ok(())
    };
    match r() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("data split: {e}");
            ExitCode::FAILURE
        }
    }
}

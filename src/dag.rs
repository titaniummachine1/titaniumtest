//! Game DAG store on **Turso** (the in-process Rust database, crate `turso`, NATIVE storage format).
//! This is NOT SQLite: stock sqlite3/rusqlite report "file is not a database" (verified 2026-11-14).
//! All access goes through `ti dag`; never open the file with any other tool.
//!
//! What it is: every position ever played, keyed by `(o0, o1, blk, stm)` = (side-0 stones, side-1 stones,
//! blockers, side to move), with a visit count; every (position, move) edge; one row per game.
//! What it is for: (1) opening novelty in `ti gen` (an unseen child = a new line), (2) the catalogue of which
//! games are in the dataset, (3) later: per-position move statistics for policy targets.
//!
//! Parallel access: the database runs in Turso's MVCC journal mode; every thread owns one connection of ONE
//! shared `Dag` and writes a whole game in one `BEGIN CONCURRENT` transaction (retry on conflict). Measured on
//! this PC (D762): 4 writer threads = 16,000 upserts/s, 163,000 point lookups/s, 2.3 M rows/s scan.
//! Limit: this is in-process concurrency. Two PROCESSES must not write the same file at once.
//!
//! Columns `w/d/l` count results from the view of the side to move in that row (verified by `ti dag verify`).
//!
//!   ti dag init   --db NEW.db [--from OLD.db]      create (or copy + convert) a DAG database
//!   ti dag stat   --db X.db                        row counts + the n == w+d+l identity
//!   ti dag verify --db X.db [--sample 200000]      replay edges through the engine, prove the w/d/l view
//!   ti dag add    --db X.db --games F.tgam [--threads 4]   import packed games (idempotent per file + index)
//!   ti dag bench  --db X.db [--threads 4]          parallel point-lookup rate

use std::future::Future;
use std::process::ExitCode;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;
use std::task::{Context, Poll};
use std::time::Instant;

use titanium::{Board, Move};
use turso::{Builder, Connection, Database, Error, Value};

/// Turso futures never need a runtime: they return `Pending` after doing their own I/O. Poll until ready.
pub fn block_on<F: Future>(f: F) -> F::Output {
    let mut cx = Context::from_waker(std::task::Waker::noop());
    let mut f = std::pin::pin!(f);
    loop {
        if let Poll::Ready(v) = f.as_mut().poll(&mut cx) {
            return v;
        }
        std::hint::spin_loop();
    }
}

fn retryable(e: &Error) -> bool {
    matches!(e, Error::Busy(_) | Error::BusySnapshot(_)) || matches!(e, Error::Error(m) if m.contains("conflict"))
}

const SCHEMA: [&str; 3] = [
    "CREATE TABLE IF NOT EXISTS games(src TEXT, idx INTEGER, header TEXT, book INTEGER, titanium_is TEXT, result TEXT, plies INTEGER, PRIMARY KEY(src, idx))",
    "CREATE TABLE IF NOT EXISTS nodes(o0 INTEGER, o1 INTEGER, blk INTEGER, stm INTEGER, n INTEGER, w INTEGER, d INTEGER, l INTEGER, min_ply INTEGER, PRIMARY KEY(o0, o1, blk, stm))",
    "CREATE TABLE IF NOT EXISTS edges(o0 INTEGER, o1 INTEGER, blk INTEGER, stm INTEGER, mv TEXT, c0 INTEGER, c1 INTEGER, cblk INTEGER, cstm INTEGER, n INTEGER, w INTEGER, d INTEGER, l INTEGER, PRIMARY KEY(o0, o1, blk, stm, mv))",
];

/// "a7".."g1" for a 0..48 square (row 0 = rank 7): the text the DAG and the gate logs use.
pub fn sq_name(sq: u8) -> String {
    format!("{}{}", (b'a' + sq % 7) as char, 7 - sq / 7)
}

/// Edge text of a move: "pass" or from-square + to-square ("a1a2"), clones included.
pub fn move_text(m: Move) -> String {
    if m.is_pass() {
        "pass".into()
    } else {
        format!("{}{}", sq_name(m.from), sq_name(m.to))
    }
}

pub fn parse_move_text(t: &str) -> Option<Move> {
    if t == "pass" {
        return Some(Move::PASS);
    }
    let b = t.as_bytes();
    if b.len() != 4 {
        return None;
    }
    let sq = |c: u8, r: u8| -> Option<u8> {
        ((b'a'..=b'g').contains(&c) && (b'1'..=b'7').contains(&r)).then(|| (7 - (r - b'0')) * 7 + (c - b'a'))
    };
    Some(Move { from: sq(b[0], b[1])?, to: sq(b[2], b[3])? })
}

/// One finished game to write back.
pub struct GameRec<'a> {
    pub src: &'a str,
    pub idx: i64,
    pub header: &'a str,
    pub book: i64,
    /// "x", "o" or "both" (self-play)
    pub titanium_is: &'a str,
    pub start: Board,
    pub moves: &'a [Move],
    /// from side 0's view: 2 = won, 1 = draw, 0 = lost
    pub result_side0: u8,
}

/// (w, d, l) increment for a row whose side to move is `stm`.
fn wdl(stm: u8, result_side0: u8) -> (i64, i64, i64) {
    match result_side0 {
        1 => (0, 1, 0),
        r => {
            if (r == 2) == (stm == 0) {
                (1, 0, 0)
            } else {
                (0, 0, 1)
            }
        }
    }
}

fn key(b: &Board) -> [Value; 4] {
    [Value::Integer(b.occ[0] as i64), Value::Integer(b.occ[1] as i64), Value::Integer(b.blockers as i64), Value::Integer(b.turn as i64)]
}

/// Shared handle: clone it (or share by reference) across threads; every thread calls `connect()`.
#[derive(Clone)]
pub struct Dag {
    db: Database,
    pub read_only: bool,
}

pub struct DagConn {
    conn: Connection,
}

/// One recorded edge out of a position: the move text, the child key, and visits.
pub struct EdgeOpt {
    pub mv: String,
    pub c0: u64,
    pub c1: u64,
    pub n: u64,
}

impl Dag {
    /// Read-write open: switches the file to MVCC and creates missing tables. A new path makes an empty DAG.
    pub fn open(path: &str) -> Result<Dag, String> {
        let db = block_on(Builder::new_local(path).build()).map_err(|e| format!("dag open {path}: {e}"))?;
        let conn = db.connect().map_err(|e| format!("dag connect {path}: {e}"))?;
        block_on(conn.pragma_update("journal_mode", "'mvcc'")).map_err(|e| format!("dag mvcc {path}: {e}"))?;
        for s in SCHEMA {
            block_on(conn.execute(s, ())).map_err(|e| format!("dag schema {path}: {e}"))?;
        }
        Ok(Dag { db, read_only: false })
    }

    /// Read-only open: lookups only, the file is never changed.
    pub fn open_read(path: &str) -> Result<Dag, String> {
        if !std::path::Path::new(path).exists() {
            return Err(format!("dag open {path}: no such file"));
        }
        let db = block_on(Builder::new_local(path).read_only(true).build()).map_err(|e| format!("dag open {path}: {e}"))?;
        Ok(Dag { db, read_only: true })
    }

    pub fn connect(&self) -> Result<DagConn, String> {
        let conn = self.db.connect().map_err(|e| format!("dag connect: {e}"))?;
        let _ = conn.busy_timeout(std::time::Duration::from_secs(30));
        Ok(DagConn { conn })
    }
}

impl DagConn {
    /// Outgoing dataset-attested moves from a position: every recorded edge with its child key + visits.
    /// Used by `ti data bestmove` to gate early stop on moves real games actually played here.
    pub fn edge_options(&self, b: &Board) -> Result<Vec<EdgeOpt>, String> {
        block_on(async {
            let mut st = self.conn.prepare_cached("SELECT mv, c0, c1, n FROM edges WHERE o0=?1 AND o1=?2 AND blk=?3 AND stm=?4").await?;
            let mut rows = st.query(key(b).to_vec()).await?;
            let mut out = Vec::new();
            while let Some(r) = rows.next().await? {
                out.push(EdgeOpt {
                    mv: r.get::<String>(0)?,
                    c0: r.get::<i64>(1)? as u64,
                    c1: r.get::<i64>(2)? as u64,
                    n: r.get::<i64>(3)? as u64,
                });
            }
            Ok::<Vec<EdgeOpt>, Error>(out)
        })
        .map_err(|e| format!("dag edge_options: {e}"))
    }
    pub fn visits(&self, b: &Board) -> Result<u64, String> {
        block_on(async {
            let mut st = self.conn.prepare_cached("SELECT n FROM nodes WHERE o0=?1 AND o1=?2 AND blk=?3 AND stm=?4").await?;
            let mut rows = st.query(key(b).to_vec()).await?;
            let mut n = 0u64;
            while let Some(r) = rows.next().await? {
                n = r.get::<i64>(0)? as u64;
            }
            Ok::<u64, Error>(n)
        })
        .map_err(|e| format!("dag visits: {e}"))
    }

    fn one_i64(&self, sql: &str) -> Result<i64, String> {
        block_on(async {
            let mut rows = self.conn.query(sql, ()).await?;
            let mut v = 0i64;
            while let Some(r) = rows.next().await? {
                v = r.get::<Option<i64>>(0)?.unwrap_or(0);
            }
            Ok::<i64, Error>(v)
        })
        .map_err(|e| format!("dag query `{sql}`: {e}"))
    }

    /// Write one game in ONE transaction: the game row, every position (n+1, result, earliest ply) and every
    /// edge. `Ok(None)` = a game with this (src, idx) is already stored, nothing written. `Ok(Some(r))` =
    /// written after `r` conflict retries.
    pub fn record_game(&self, g: &GameRec) -> Result<Option<u32>, String> {
        // Replay first, outside the transaction.
        let mut b = g.start;
        let mut nodes: Vec<Vec<Value>> = Vec::with_capacity(g.moves.len() + 1);
        let mut edges: Vec<Vec<Value>> = Vec::with_capacity(g.moves.len());
        let push_node = |nodes: &mut Vec<Vec<Value>>, b: &Board, ply: usize| {
            let (w, d, l) = wdl(b.turn, g.result_side0);
            let mut v = key(b).to_vec();
            v.extend([Value::Integer(w), Value::Integer(d), Value::Integer(l), Value::Integer(ply as i64)]);
            nodes.push(v);
        };
        push_node(&mut nodes, &b, 0);
        for (i, &m) in g.moves.iter().enumerate() {
            // A clone can start from any neighbouring stone with the same result. Store ONE spelling: the
            // source square the engine's move decoder picks (lowest neighbouring own stone).
            let m = if m.is_pass() { m } else { titanium::ttcodec::decode_move(&b, titanium::ttcodec::move_code(m)).unwrap_or(m) };
            let child = b.make_uai(m);
            let (w, d, l) = wdl(b.turn, g.result_side0);
            let mut v = key(&b).to_vec();
            v.push(Value::Text(move_text(m)));
            v.extend(key(&child));
            v.extend([Value::Integer(w), Value::Integer(d), Value::Integer(l)]);
            edges.push(v);
            b = child;
            push_node(&mut nodes, &b, i + 1);
        }
        let result = match g.result_side0 {
            2 => "x",
            1 => "draw",
            _ => "o",
        };
        let mut retries = 0u32;
        loop {
            let r: Result<bool, Error> = block_on(async {
                self.conn.execute("BEGIN CONCURRENT", ()).await?;
                let mut q = self.conn.prepare_cached("SELECT COUNT(*) FROM games WHERE src=?1 AND idx=?2").await?;
                let mut rows = q.query(vec![Value::Text(g.src.to_string()), Value::Integer(g.idx)]).await?;
                let mut have = 0i64;
                while let Some(r) = rows.next().await? {
                    have = r.get::<i64>(0)?;
                }
                drop(rows);
                if have > 0 {
                    self.conn.execute("ROLLBACK", ()).await?;
                    return Ok(false);
                }
                self.conn
                    .execute(
                        "INSERT INTO games(src,idx,header,book,titanium_is,result,plies) VALUES (?1,?2,?3,?4,?5,?6,?7)",
                        vec![
                            Value::Text(g.src.to_string()),
                            Value::Integer(g.idx),
                            Value::Text(g.header.to_string()),
                            Value::Integer(g.book),
                            Value::Text(g.titanium_is.to_string()),
                            Value::Text(result.to_string()),
                            Value::Integer(g.moves.len() as i64),
                        ],
                    )
                    .await?;
                let mut ins = self
                    .conn
                    .prepare_cached(
                        "INSERT INTO nodes (o0,o1,blk,stm,n,w,d,l,min_ply) VALUES (?1,?2,?3,?4,1,?5,?6,?7,?8) \
                         ON CONFLICT(o0,o1,blk,stm) DO UPDATE SET n=n+1, w=w+excluded.w, d=d+excluded.d, l=l+excluded.l, \
                         min_ply=min(min_ply,excluded.min_ply)",
                    )
                    .await?;
                for v in &nodes {
                    ins.execute(v.clone()).await?;
                }
                let mut ins = self
                    .conn
                    .prepare_cached(
                        "INSERT INTO edges (o0,o1,blk,stm,mv,c0,c1,cblk,cstm,n,w,d,l) VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,1,?10,?11,?12) \
                         ON CONFLICT(o0,o1,blk,stm,mv) DO UPDATE SET n=n+1, w=w+excluded.w, d=d+excluded.d, l=l+excluded.l",
                    )
                    .await?;
                for v in &edges {
                    ins.execute(v.clone()).await?;
                }
                self.conn.execute("COMMIT", ()).await?;
                Ok(true)
            });
            match r {
                Ok(true) => return Ok(Some(retries)),
                Ok(false) => return Ok(None),
                Err(e) if retryable(&e) && retries < 100_000 => {
                    let _ = block_on(self.conn.execute("ROLLBACK", ()));
                    retries += 1;
                    std::thread::yield_now();
                }
                Err(e) => {
                    let _ = block_on(self.conn.execute("ROLLBACK", ()));
                    return Err(format!("dag record_game {} #{}: {e}", g.src, g.idx));
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// Sequence DAG (D804, user design): the PGN-like database of GAME SEQUENCES, a NEW file (the position DAG above is kept
// untouched as the archive). A game is the chain of its moves from the start position; games that share their first
// moves share that part of the chain. Table `seq_lines` is the chain as a prefix tree: the key (parent, mv) says "from
// the sequence with hash `parent`, the move `mv` was played", the child sequence is seq_child(parent, mv). It carries
// n / w / d / l (from the view of the player who made the move) = the WDL distribution of every move at every point of
// every sequence. `seq_games` holds one row per distinct game (numeric `gid`, start fen, result, the moves as u16 codes);
// training rows point to a game with (gid, ply). Identical sequences are stored once (`seq_keys`).
// ---------------------------------------------------------------------------------------------------------------------

const SEQ_SCHEMA: [&str; 3] = [
    "CREATE TABLE IF NOT EXISTS seq_games(gid INTEGER PRIMARY KEY, src TEXT, idx INTEGER, start TEXT, result INTEGER, plies INTEGER, moves BLOB)",
    "CREATE TABLE IF NOT EXISTS seq_keys(gkey INTEGER PRIMARY KEY, gid INTEGER)",
    "CREATE TABLE IF NOT EXISTS seq_lines(parent INTEGER, mv INTEGER, n INTEGER, w INTEGER, d INTEGER, l INTEGER, PRIMARY KEY(parent, mv))",
];

pub fn splitmix(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

/// Hash of the EMPTY sequence for a start position (different start positions never share sequences).
pub fn seq_root(b: &Board) -> u64 {
    splitmix(splitmix(splitmix(b.occ[0]) ^ b.occ[1]) ^ splitmix(b.blockers ^ ((b.turn as u64) << 56)))
}

/// 16-bit code of a move in position `b`: from * 64 + to with the clone source normalised by the engine's move decoder
/// (the same clone always gets the same code); 0xFFFF = pass.
pub fn seq_code(b: &Board, m: Move) -> u16 {
    if m.is_pass() {
        return 0xFFFF;
    }
    let m = titanium::ttcodec::decode_move(b, titanium::ttcodec::move_code(m)).unwrap_or(m);
    (m.from as u16) * 64 + m.to as u16
}

/// Hash of the sequence `h` extended by the move with code `code`.
pub fn seq_child(h: u64, code: u16) -> u64 {
    splitmix(h ^ splitmix(code as u64 | (1 << 20)))
}

/// Inverse of `seq_code` for the engine's move maker (clone source = the decoder's spelling).
pub fn seq_move(code: u16) -> Move {
    if code == 0xFFFF {
        Move::PASS
    } else {
        Move { from: (code / 64) as u8, to: (code % 64) as u8 }
    }
}

fn seq_retry(e: &Error) -> bool {
    retryable(e) || matches!(e, Error::Error(m) if m.contains("UNIQUE") || m.contains("constraint"))
}

#[derive(Clone)]
pub struct SeqDag {
    db: Database,
    pub read_only: bool,
    next_gid: std::sync::Arc<std::sync::atomic::AtomicI64>,
}

pub struct SeqConn {
    conn: Connection,
    next_gid: std::sync::Arc<std::sync::atomic::AtomicI64>,
}

impl SeqDag {
    pub fn open(path: &str) -> Result<SeqDag, String> {
        let db = block_on(Builder::new_local(path).build()).map_err(|e| format!("seq dag open {path}: {e}"))?;
        let conn = db.connect().map_err(|e| format!("seq dag connect {path}: {e}"))?;
        block_on(conn.pragma_update("journal_mode", "'mvcc'")).map_err(|e| format!("seq dag mvcc {path}: {e}"))?;
        for s in SEQ_SCHEMA {
            block_on(conn.execute(s, ())).map_err(|e| format!("seq dag schema {path}: {e}"))?;
        }
        let max = SeqConn { conn, next_gid: Default::default() }.one_i64("SELECT MAX(gid) FROM seq_games")?;
        Ok(SeqDag { db, read_only: false, next_gid: std::sync::Arc::new(std::sync::atomic::AtomicI64::new(max + 1)) })
    }

    pub fn open_read(path: &str) -> Result<SeqDag, String> {
        if !std::path::Path::new(path).exists() {
            return Err(format!("seq dag open {path}: no such file"));
        }
        let db = block_on(Builder::new_local(path).read_only(true).build()).map_err(|e| format!("seq dag open {path}: {e}"))?;
        Ok(SeqDag { db, read_only: true, next_gid: Default::default() })
    }

    pub fn connect(&self) -> Result<SeqConn, String> {
        let conn = self.db.connect().map_err(|e| format!("seq dag connect: {e}"))?;
        let _ = conn.busy_timeout(std::time::Duration::from_secs(30));
        Ok(SeqConn { conn, next_gid: self.next_gid.clone() })
    }
}

impl SeqConn {
    fn one_i64(&self, sql: &str) -> Result<i64, String> {
        block_on(async {
            let mut rows = self.conn.query(sql, ()).await?;
            let mut v = 0i64;
            while let Some(r) = rows.next().await? {
                v = r.get::<Option<i64>>(0)?.unwrap_or(0);
            }
            Ok::<i64, Error>(v)
        })
        .map_err(|e| format!("seq dag query `{sql}`: {e}"))
    }

    /// Moves (codes) recorded from the sequence with hash `parent`, with the number of games that played each.
    pub fn children(&self, parent: u64) -> Result<Vec<(u16, i64)>, String> {
        block_on(async {
            let mut st = self.conn.prepare_cached("SELECT mv, n FROM seq_lines WHERE parent=?1").await?;
            let mut rows = st.query(vec![Value::Integer(parent as i64)]).await?;
            let mut out = Vec::new();
            while let Some(r) = rows.next().await? {
                out.push((r.get::<i64>(0)? as u16, r.get::<i64>(1)?));
            }
            Ok::<Vec<(u16, i64)>, Error>(out)
        })
        .map_err(|e| format!("seq dag children: {e}"))
    }

    /// Store one finished game: the game row, its place in the sequence tree and the W/D/L of every move on it, in ONE
    /// transaction. `Ok(None)` = this exact sequence is already stored (no new game row; its W/D/L counts ARE updated). `Ok(Some((gid, retries)))` =
    /// written under game id `gid`. `result_side0`: 2 = side 0 won, 1 = draw, 0 = lost.
    /// `count_repeats`: true for live generation (a repeated sequence is a new observation of the same line);
    /// false for importing old files (the replayed copies of batches 5-7 must not inflate the counts).
    pub fn record(&self, src: &str, idx: i64, start: Board, moves: &[Move], result_side0: u8, count_repeats: bool) -> Result<Option<(i64, u32)>, String> {
        let mut b = start;
        let mut h = seq_root(&start);
        let mut codes: Vec<u16> = Vec::with_capacity(moves.len());
        let mut lines: Vec<Vec<Value>> = Vec::with_capacity(moves.len());
        for &m in moves {
            let code = seq_code(&b, m);
            let (w, d, l) = wdl(b.turn, result_side0);
            lines.push(vec![Value::Integer(h as i64), Value::Integer(code as i64), Value::Integer(w), Value::Integer(d), Value::Integer(l)]);
            codes.push(code);
            h = seq_child(h, code);
            b = b.make_uai(seq_move(code));
        }
        let gkey = h as i64;
        let blob: Vec<u8> = codes.iter().flat_map(|c| c.to_le_bytes()).collect();
        let fen = start.to_ataxx_fen();
        let mut retries = 0u32;
        loop {
            let r: Result<Option<i64>, Error> = block_on(async {
                self.conn.execute("BEGIN CONCURRENT", ()).await?;
                let mut q = self.conn.prepare_cached("SELECT COUNT(*) FROM seq_keys WHERE gkey=?1").await?;
                let mut rows = q.query(vec![Value::Integer(gkey)]).await?;
                let mut have = 0i64;
                while let Some(r) = rows.next().await? {
                    have = r.get::<i64>(0)?;
                }
                drop(rows);
                // the sequence already exists: no new game row, but its W/D/L counts are still updated (user D804)
                let is_new = have == 0;
                if !is_new && !count_repeats {
                    self.conn.execute("ROLLBACK", ()).await?;
                    return Ok(None);
                }
                let gid = if is_new {
                    let gid = self.next_gid.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    self.conn
                        .execute(
                            "INSERT INTO seq_games(gid,src,idx,start,result,plies,moves) VALUES (?1,?2,?3,?4,?5,?6,?7)",
                            vec![
                                Value::Integer(gid),
                                Value::Text(src.to_string()),
                                Value::Integer(idx),
                                Value::Text(fen.clone()),
                                Value::Integer(result_side0 as i64),
                                Value::Integer(codes.len() as i64),
                                Value::Blob(blob.clone()),
                            ],
                        )
                        .await?;
                    self.conn.execute("INSERT INTO seq_keys(gkey,gid) VALUES (?1,?2)", vec![Value::Integer(gkey), Value::Integer(gid)]).await?;
                    gid
                } else {
                    let mut q = self.conn.prepare_cached("SELECT gid FROM seq_keys WHERE gkey=?1").await?;
                    let mut rows = q.query(vec![Value::Integer(gkey)]).await?;
                    let mut g = -1i64;
                    while let Some(r) = rows.next().await? {
                        g = r.get::<i64>(0)?;
                    }
                    g
                };
                let mut ins = self
                    .conn
                    .prepare_cached(
                        "INSERT INTO seq_lines(parent,mv,n,w,d,l) VALUES (?1,?2,1,?3,?4,?5) \
                         ON CONFLICT(parent,mv) DO UPDATE SET n=n+1, w=w+excluded.w, d=d+excluded.d, l=l+excluded.l",
                    )
                    .await?;
                for v in &lines {
                    ins.execute(v.clone()).await?;
                }
                self.conn.execute("COMMIT", ()).await?;
                Ok(Some(if is_new { gid } else { -gid - 1 }))
            });
            match r {
                // a negative marker means "sequence already stored, counts updated": gid = -marker-1
                Ok(Some(g)) if g < 0 => return Ok(None),
                Ok(Some(gid)) => return Ok(Some((gid, retries))),
                Ok(None) => return Ok(None),
                Err(e) if seq_retry(&e) && retries < 100_000 => {
                    let _ = block_on(self.conn.execute("ROLLBACK", ()));
                    retries += 1;
                    std::thread::yield_now();
                }
                Err(e) => {
                    let _ = block_on(self.conn.execute("ROLLBACK", ()));
                    return Err(format!("seq dag record {src} #{idx}: {e}"));
                }
            }
        }
    }
}

fn need_db(args: &[String]) -> Result<String, ExitCode> {
    crate::flag_str(args, "--db").ok_or_else(|| {
        eprintln!("usage: ti dag init|stat|verify|add|bench --db FILE ...  (see the top of src/dag.rs)");
        ExitCode::FAILURE
    })
}

fn fail(e: String) -> ExitCode {
    eprintln!("dag: {e}");
    ExitCode::FAILURE
}

pub fn run_dag(args: &[String]) -> ExitCode {
    let sub = args.first().map(String::as_str).unwrap_or("");
    let rest = if args.is_empty() { args } else { &args[1..] };
    let db = match need_db(rest) {
        Ok(p) => p,
        Err(c) => return c,
    };
    let r = match sub {
        "init" => run_init(&db, rest),
        "stat" => run_stat(&db),
        "verify" => run_verify(&db, rest),
        "add" => run_add(&db, rest),
        "bench" => run_bench(&db, rest),
        "probe" => run_probe(&db, rest),
        "seq-import" => run_seq_import(&db, rest),
        "seq-stat" => run_seq_stat(&db),
        "seq-verify" => run_seq_verify(&db, rest),
        "jsonl-to-bin" => run_jsonl_to_bin(&db, rest),
        "tbst-check" => run_tbst_check(&db, rest),
        "gate-import" => run_gate_import(&db, rest),
        _ => Err(format!("unknown subcommand `{sub}` (init | stat | verify | add | bench | probe | seq-import | seq-stat)")),
    };
    match r {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => fail(e),
    }
}

fn run_init(db: &str, args: &[String]) -> Result<(), String> {
    if std::path::Path::new(db).exists() {
        return Err(format!("{db} already exists; refusing to overwrite"));
    }
    if let Some(from) = crate::flag_str(args, "--from") {
        let n = std::fs::copy(&from, db).map_err(|e| format!("copy {from} -> {db}: {e}"))?;
        println!("copied {from} -> {db} ({n} bytes); the source file is untouched");
    }
    Dag::open(db)?;
    println!("ready: {db} (Turso, MVCC journal, tables games / nodes / edges)");
    run_stat(db)
}

fn run_stat(db: &str) -> Result<(), String> {
    let c = Dag::open_read(db)?.connect()?;
    let games = c.one_i64("SELECT COUNT(*) FROM games")?;
    let plies = c.one_i64("SELECT SUM(plies) FROM games")?;
    let nodes = c.one_i64("SELECT COUNT(*) FROM nodes")?;
    let visits = c.one_i64("SELECT SUM(n) FROM nodes")?;
    let edges = c.one_i64("SELECT COUNT(*) FROM edges")?;
    let etrav = c.one_i64("SELECT SUM(n) FROM edges")?;
    let bad = c.one_i64("SELECT COUNT(*) FROM nodes WHERE n != w + d + l")?;
    let starts = c.one_i64("SELECT COUNT(*) FROM nodes WHERE min_ply = 0")?;
    println!("games {games}  plies {plies}");
    println!("nodes {nodes}  visits {visits}  (expected plies + games = {})  start positions {starts}", plies + games);
    println!("edges {edges}  traversals {etrav}  (expected plies = {plies})");
    println!("rows with n != w+d+l: {bad}");
    if visits != plies + games || etrav != plies || bad != 0 {
        return Err("identity check FAILED".into());
    }
    println!("identities OK");
    Ok(())
}

fn row_u64(r: &turso::Row, i: usize) -> Result<u64, Error> {
    Ok(r.get::<i64>(i)? as u64)
}

fn run_verify(db: &str, args: &[String]) -> Result<(), String> {
    let sample = crate::flag_u64(args, "--sample").unwrap_or(200_000);
    let c = Dag::open_read(db)?.connect()?;
    // 1. Edges: parent + move text must give the stored child, through the engine's own move maker.
    let (mut seen, mut ok, mut bad_text, mut bad_child) = (0u64, 0u64, 0u64, 0u64);
    block_on(async {
        let mut rows = c.conn.query("SELECT o0,o1,blk,stm,mv,c0,c1,cblk,cstm FROM edges", ()).await?;
        while let Some(r) = rows.next().await? {
            seen += 1;
            if seen % 7 != 0 || ok + bad_text + bad_child >= sample {
                continue;
            }
            let b = Board::from_occ(row_u64(&r, 0)?, row_u64(&r, 1)?, row_u64(&r, 2)?, row_u64(&r, 3)? as u8);
            let Some(m) = parse_move_text(&r.get::<String>(4)?) else {
                bad_text += 1;
                continue;
            };
            let ch = b.make_uai(m);
            if ch.occ == [row_u64(&r, 5)?, row_u64(&r, 6)?] && ch.blockers == row_u64(&r, 7)? && ch.turn as u64 == row_u64(&r, 8)? {
                ok += 1;
            } else {
                bad_child += 1;
            }
        }
        Ok::<(), Error>(())
    })
    .map_err(|e| format!("verify edges: {e}"))?;
    println!("edges replayed {}: child matches {ok}, wrong child {bad_child}, unreadable move {bad_text}", ok + bad_child + bad_text);
    // 2. Whose wins is `w`? On a full board the winner is known from the stones alone.
    // t[stm][x_ahead] = (rows, w, l)
    let mut t = [[(0u64, 0u64, 0u64); 2]; 2];
    block_on(async {
        let mut rows = c.conn.query("SELECT o0,o1,blk,stm,w,l FROM nodes", ()).await?;
        while let Some(r) = rows.next().await? {
            let (o0, o1, blk) = (row_u64(&r, 0)?, row_u64(&r, 1)?, row_u64(&r, 2)?);
            if (o0 | o1 | blk).count_ones() != 49 || o0.count_ones() == o1.count_ones() {
                continue;
            }
            let e = &mut t[row_u64(&r, 3)? as usize & 1][(o0.count_ones() > o1.count_ones()) as usize];
            e.0 += 1;
            e.1 += row_u64(&r, 4)?;
            e.2 += row_u64(&r, 5)?;
        }
        Ok::<(), Error>(())
    })
    .map_err(|e| format!("verify nodes: {e}"))?;
    println!("full-board positions (winner known from the stones):  rows / w / l");
    for stm in 0..2 {
        for xa in [1usize, 0] {
            let e = t[stm][xa];
            println!("  side {stm} to move, side {} ahead: {} / {} / {}", if xa == 1 { 0 } else { 1 }, e.0, e.1, e.2);
        }
    }
    // mover view: w only where the mover is ahead; side-0 view: w only where side 0 is ahead
    let mover_bad = t[0][1].2 + t[0][0].1 + t[1][0].2 + t[1][1].1;
    let side0_bad = t[0][1].2 + t[1][1].2 + t[0][0].1 + t[1][0].1;
    let total: u64 = t.iter().flatten().map(|e| e.1 + e.2).sum();
    println!("results that contradict 'w = side to move won': {mover_bad} of {total}");
    println!("results that contradict 'w = side 0 won':       {side0_bad} of {total}");
    if bad_child + bad_text != 0 {
        return Err("edge replay FAILED".into());
    }
    Ok(())
}

/// `ti dag seq-stat --db SEQ.db`: counts of the sequence DAG and its identities.
fn run_seq_stat(db: &str) -> Result<(), String> {
    let c = SeqDag::open_read(db)?.connect()?;
    let games = c.one_i64("SELECT COUNT(*) FROM seq_games")?;
    let plies = c.one_i64("SELECT SUM(plies) FROM seq_games")?;
    let keys = c.one_i64("SELECT COUNT(*) FROM seq_keys")?;
    let lines = c.one_i64("SELECT COUNT(*) FROM seq_lines")?;
    let trav = c.one_i64("SELECT SUM(n) FROM seq_lines")?;
    let bad = c.one_i64("SELECT COUNT(*) FROM seq_lines WHERE n != w + d + l")?;
    let (x, dr, o) = (
        c.one_i64("SELECT COUNT(*) FROM seq_games WHERE result=2")?,
        c.one_i64("SELECT COUNT(*) FROM seq_games WHERE result=1")?,
        c.one_i64("SELECT COUNT(*) FROM seq_games WHERE result=0")?,
    );
    println!("games {games}  plies {plies}  (x won {x}, draw {dr}, o won {o})");
    println!("sequence points (parent, move) {lines}  traversals {trav} (= plies {plies} + repeated sequences' plies {})  keys {keys} (expected games = {games})", trav - plies);
    println!("rows with n != w+d+l: {bad}");
    if trav < plies || keys != games || bad != 0 {
        return Err("identity check FAILED".into());
    }
    println!("identities OK");
    Ok(())
}

/// `ti dag seq-verify --db SEQ.db [--apply]`: recompute what `seq_lines` must hold from `seq_games` alone (every distinct
/// game counted once: n / w / d / l of each (parent, move), from the mover's view) and compare it with the stored table.
/// Without `--apply` it only reads and reports. With `--apply` it corrects ONLY the rows whose numbers differ (UPDATE, or
/// INSERT for a missing row); it never deletes a row and never touches `seq_games` / `seq_keys` (so every gid stays valid).
fn run_seq_verify(db: &str, args: &[String]) -> Result<(), String> {
    let apply = args.iter().any(|a| a == "--apply");
    let dag = if apply { SeqDag::open(db)? } else { SeqDag::open_read(db)? };
    let c = dag.connect()?;
    let mut exp: std::collections::HashMap<(u64, u16), [i64; 4]> = std::collections::HashMap::new();
    let mut games = 0u64;
    block_on(async {
        let mut rows = c.conn.query("SELECT start, result, moves FROM seq_games", ()).await?;
        while let Some(r) = rows.next().await? {
            let fen = r.get::<String>(0)?;
            let res = r.get::<i64>(1)? as u8;
            let blob = r.get::<Vec<u8>>(2)?;
            let Some(mut b) = Board::from_ataxx_fen(&fen) else { continue };
            let mut h = seq_root(&b);
            for ch in blob.chunks_exact(2) {
                let code = u16::from_le_bytes([ch[0], ch[1]]);
                let (w, d, l) = wdl(b.turn, res);
                let e = exp.entry((h, code)).or_insert([0; 4]);
                e[0] += 1;
                e[1] += w;
                e[2] += d;
                e[3] += l;
                h = seq_child(h, code);
                b = b.make_uai(seq_move(code));
            }
            games += 1;
        }
        Ok::<(), Error>(())
    })
    .map_err(|e| format!("seq-verify games: {e}"))?;
    println!("games replayed {games}, expected sequence points {}", exp.len());
    let (mut same, mut diff, mut extra) = (0u64, 0u64, 0u64);
    let (mut over, mut shown) = (0i64, 0);
    let mut fixes: Vec<(u64, u16, [i64; 4])> = Vec::new();
    block_on(async {
        let mut rows = c.conn.query("SELECT parent, mv, n, w, d, l FROM seq_lines", ()).await?;
        while let Some(r) = rows.next().await? {
            let key = (r.get::<i64>(0)? as u64, r.get::<i64>(1)? as u16);
            let have = [r.get::<i64>(2)?, r.get::<i64>(3)?, r.get::<i64>(4)?, r.get::<i64>(5)?];
            match exp.remove(&key) {
                Some(e) if e == have => same += 1,
                Some(e) => {
                    diff += 1;
                    over += have[0] - e[0];
                    if shown < 10 {
                        println!("  differs: parent {:016x} mv {:04x}: stored n/w/d/l {:?}, from the games {:?}", key.0, key.1, have, e);
                        shown += 1;
                    }
                    fixes.push((key.0, key.1, e));
                }
                None => extra += 1,
            }
        }
        Ok::<(), Error>(())
    })
    .map_err(|e| format!("seq-verify lines: {e}"))?;
    let missing = exp.len();
    println!("rows equal {same}, rows with other numbers {diff} (stored minus expected traversals: {over}), rows with no game behind them {extra}, game rows missing in the table {missing}");
    if !apply {
        println!("read-only: nothing changed (use --apply to correct the differing rows)");
        return Ok(());
    }
    block_on(async {
        for (p, m, e) in &fixes {
            c.conn
                .execute(
                    "UPDATE seq_lines SET n=?3, w=?4, d=?5, l=?6 WHERE parent=?1 AND mv=?2",
                    vec![Value::Integer(*p as i64), Value::Integer(*m as i64), Value::Integer(e[0]), Value::Integer(e[1]), Value::Integer(e[2]), Value::Integer(e[3])],
                )
                .await?;
        }
        for ((p, m), e) in &exp {
            c.conn
                .execute(
                    "INSERT INTO seq_lines(parent,mv,n,w,d,l) VALUES (?1,?2,?3,?4,?5,?6)",
                    vec![Value::Integer(*p as i64), Value::Integer(*m as i64), Value::Integer(e[0]), Value::Integer(e[1]), Value::Integer(e[2]), Value::Integer(e[3])],
                )
                .await?;
        }
        Ok::<(), Error>(())
    })
    .map_err(|e| format!("seq-verify apply: {e}"))?;
    println!("applied: {} rows corrected, {} rows inserted; nothing deleted", fixes.len(), missing);
    Ok(())
}

/// `ti dag tbst-check --tgam X.tgam [--db SEQ.db]`: read X.tgam and X.tbst together and check the labels: every best-move
/// label decodes to a LEGAL move at its position, how often it equals the played move, the score ranges, and (with
/// `--db`) that every game id exists in the sequence DAG and belongs to the same game.
fn run_tbst_check(_db: &str, args: &[String]) -> Result<(), String> {
    use std::io::Read;
    let tgam = crate::flag_str(args, "--tgam").ok_or("usage: ti dag tbst-check --tgam X.tgam [--db SEQ.db]")?;
    let tbst = format!("{}.tbst", tgam.trim_end_matches(".tgam"));
    let mut tb = std::io::BufReader::with_capacity(1 << 20, std::fs::File::open(&tbst).map_err(|e| format!("{tbst}: {e}"))?);
    let mut h = [0u8; 24];
    tb.read_exact(&mut h).map_err(|e| format!("{tbst}: {e}"))?;
    if &h[0..4] != b"TBST" || u32::from_le_bytes(h[4..8].try_into().unwrap()) != 1 {
        return Err(format!("{tbst}: not a TBST version 1 file"));
    }
    let (tg_games, tg_plies) = (u64::from_le_bytes(h[8..16].try_into().unwrap()), u64::from_le_bytes(h[16..24].try_into().unwrap()));
    let rd = crate::gamedata::GameReader::open(&tgam)?;
    if rd.games != tg_games || rd.plies != tg_plies {
        return Err(format!("headers disagree: tgam {}/{} plies, tbst {tg_games}/{tg_plies}", rd.games, rd.plies));
    }
    let dag = match crate::flag_str(args, "--db") {
        Some(p) => Some(SeqDag::open_read(&p)?.connect()?),
        None => None,
    };
    let (mut games, mut plies, mut labelled, mut same, mut illegal, mut unlabelled_games) = (0u64, 0u64, 0u64, 0u64, 0u64, 0u64);
    let (mut gid_ok, mut gid_bad) = (0u64, 0u64);
    let (mut sum_best, mut sum_gap) = (0f64, 0f64);
    let (mut no_score, mut mate) = (0u64, 0u64);
    for g in rd {
        let g = g.map_err(|e| format!("tgam: {e}"))?;
        let mut gb = [0u8; 4];
        tb.read_exact(&mut gb).map_err(|e| format!("tbst gid: {e}"))?;
        let gid = u32::from_le_bytes(gb) as i64;
        let (views, _) = g.replay()?;
        let mut any_unl = false;
        let mut h = seq_root(&g.start());
        for (i, v) in views.iter().enumerate() {
            let mut pb = [0u8; 4];
            tb.read_exact(&mut pb).map_err(|e| format!("tbst ply: {e}"))?;
            let (code, sc) = (u16::from_le_bytes([pb[0], pb[1]]) & 0x3FF, i16::from_le_bytes([pb[2], pb[3]]));
            plies += 1;
            if dag.is_some() {
                h = seq_child(h, seq_code(&v.board, v.mv));
            }
            if code == 0 {
                any_unl = true;
                continue;
            }
            labelled += 1;
            match titanium::ttcodec::decode_move(&v.board, code) {
                None => illegal += 1,
                Some(bm) => {
                    if v.board.make_uai(bm).occ == v.board.make_uai(v.mv).occ {
                        same += 1;
                    }
                }
            }
            if sc == crate::gamedata::NO_SCORE {
                no_score += 1;
            } else {
                let s = titanium::ttcodec::dec_score(sc);
                if s.abs() >= 90_000 {
                    mate += 1;
                }
                sum_best += s as f64;
                if let Some(ps) = v.score {
                    sum_gap += (s - ps) as f64;
                }
            }
            let _ = i;
        }
        if any_unl {
            unlabelled_games += 1;
        }
        if let Some(c) = &dag {
            let ok = c.one_i64(&format!("SELECT COUNT(*) FROM seq_keys WHERE gkey={} AND gid={gid}", h as i64))?;
            if ok == 1 {
                gid_ok += 1;
            } else {
                gid_bad += 1;
            }
        }
        games += 1;
    }
    println!("games {games}, plies {plies}; plies with a best-move label {labelled} ({:.1}%), games with unlabelled plies {unlabelled_games}", 100.0 * labelled as f64 / plies.max(1) as f64);
    println!("best-move labels that are NOT legal at their position: {illegal}; best move equals the played move in {:.2}% of labelled plies", 100.0 * same as f64 / labelled.max(1) as f64);
    println!("best-move score: none {no_score}, proven win/loss {mate}, mean {:.1} cp; mean (best score - played score) {:.2} cp", sum_best / labelled.max(1) as f64, sum_gap / labelled.max(1) as f64);
    if dag.is_some() {
        println!("game ids matching their sequence in the DAG: {gid_ok} ok, {gid_bad} wrong");
    }
    if illegal != 0 || gid_bad != 0 {
        return Err("label check FAILED".into());
    }
    println!("label check OK");
    Ok(())
}

/// D830 (user: gate games join the dataset like generated games, no relabelling): every gate log `<x>.games.jsonl` under
/// `roots` that has no `<x>.dag_done` marker yet is (1) written as binary `<x>.tgam` + `<x>.tbst` (played move, depth and the
/// mover's own score; the "best move" label is the played move, a gate engine plays its own best; gid = the game's DAG id,
/// 0 when the sequence was already stored), (2) recorded in the sequence DAG (W/D/L counted also for repeated sequences,
/// user D804), (3) marked done so it is never counted twice. The JSON log is only read. Returns (files, games, new sequences).
pub fn import_gate_logs(db: &str, roots: &[String]) -> Result<(u64, u64, u64), String> {
    import_gate_logs_ex(db, roots, false)
}

/// `redo`: also process logs that already have a `.dag_done` marker (the first D830 run recorded only the games whose start
/// was in FEN form; `redo` records the OTHERS, so no game is counted twice).
pub fn import_gate_logs_ex(db: &str, roots: &[String], redo: bool) -> Result<(u64, u64, u64), String> {
    let mut files: Vec<std::path::PathBuf> = Vec::new();
    let mut stack: Vec<std::path::PathBuf> = roots.iter().map(std::path::PathBuf::from).collect();
    while let Some(d) = stack.pop() {
        let Ok(rd) = std::fs::read_dir(&d) else { continue };
        for e in rd.flatten() {
            let p = e.path();
            if p.is_dir() {
                stack.push(p);
            } else if p.to_string_lossy().ends_with(".games.jsonl") {
                files.push(p);
            }
        }
    }
    files.sort();
    let todo: Vec<_> = files
        .into_iter()
        .filter(|p| redo || !std::path::Path::new(&format!("{}.dag_done", p.to_string_lossy().trim_end_matches(".games.jsonl"))).exists())
        .collect();
    if todo.is_empty() {
        return Ok((0, 0, 0));
    }
    let c = SeqDag::open(db)?.connect()?;
    let (mut nf, mut ng, mut nnew) = (0u64, 0u64, 0u64);
    for p in &todo {
        let path = p.to_string_lossy().to_string();
        let stem = path.trim_end_matches(".games.jsonl").to_string();
        let src = p.parent().and_then(|x| x.file_name()).map(|n| n.to_string_lossy().into_owned()).unwrap_or_else(|| stem.clone());
        let text = std::fs::read_to_string(&path).map_err(|e| format!("{path}: {e}"))?;
        let mut bin = crate::gen::BinOut::create(&stem).map_err(|e| format!("{stem}.tgam: {e}"))?;
        let (mut g_here, mut new_here, mut skipped) = (0u64, 0u64, 0u64);
        for (i, line) in text.lines().enumerate().filter(|(_, l)| !l.trim().is_empty()) {
            let Ok(pg) = crate::games::parse_game(line).and_then(|g| crate::gamedata::pack_game(&g, line, 0)) else {
                skipped += 1;
                continue;
            };
            let mut gid = 0u32;
            // the sequence comes from the packed game itself (gate logs keep the start in book form, not FEN, so the
            // self-play line parser cannot read them); a forfeit ends the recorded sequence; unknown results are not recorded
            let was_fen_readable = parse_game_line(line).is_some();
            let already_recorded = redo && was_fen_readable;
            if pg.result <= 2 && !already_recorded {
                if let Ok((views, _)) = pg.replay() {
                    let moves: Vec<Move> = views.iter().map(|v| v.mv).collect();
                    if let Some((id, _)) = c.record(&src, i as i64, pg.start(), &moves, pg.result, true)? {
                        gid = id as u32;
                        new_here += 1;
                    }
                }
            }
            let best: Vec<(u16, i16)> = pg.plies.iter().map(|&(mvw, sc)| (mvw & 0x3FF, sc)).collect();
            bin.append(&pg, gid, &best).map_err(|e| format!("{stem}.tgam: {e}"))?;
            g_here += 1;
        }
        std::fs::write(format!("{stem}.dag_done"), format!("games {g_here}\nnew_sequences {new_here}\nunreadable {skipped}\n"))
            .map_err(|e| format!("{stem}.dag_done: {e}"))?;
        nf += 1;
        ng += g_here;
        nnew += new_here;
    }
    Ok((nf, ng, nnew))
}

fn run_gate_import(db: &str, args: &[String]) -> Result<(), String> {
    let roots: Vec<String> = crate::flag_str(args, "--roots")
        .ok_or("usage: ti dag gate-import --db SEQ.db --roots DIR,DIR")?
        .split(',')
        .filter(|s| !s.is_empty())
        .map(String::from)
        .collect();
    let t0 = Instant::now();
    let (f, g, n) = import_gate_logs_ex(db, &roots, args.iter().any(|a| a == "--redo"))?;
    println!("gate-import: {f} gate logs -> binary .tgam/.tbst + sequence DAG: {g} games, {n} new sequences ({:.0}s)", t0.elapsed().as_secs_f64());
    Ok(())
}

/// `ti dag jsonl-to-bin --db SEQ.db --out STEM --in a.jsonl,b.jsonl [--src N]`: one-time conversion of the OLD `ti gen` JSON
/// game files (D806: no JSON for game data) into the binary pair STEM.tgam + STEM.tbst (see `gen::BinOut`). Each distinct
/// game (same sequence = same DAG id) is written once, in file order; the game's id comes from the DAG (read only).
/// Lines written before D787 have no best-move label (move code 0, score "none"). The JSON files are only read.
fn run_jsonl_to_bin(db: &str, args: &[String]) -> Result<(), String> {
    use titanium::ttcodec::{enc_score, move_code};
    let out = crate::flag_str(args, "--out").ok_or("usage: ti dag jsonl-to-bin --db SEQ.db --out STEM --in a.jsonl,b.jsonl")?;
    let inputs: Vec<String> = crate::flag_str(args, "--in").unwrap_or_default().split(',').filter(|s| !s.is_empty()).map(String::from).collect();
    if inputs.is_empty() {
        return Err("no --in files".into());
    }
    let c = SeqDag::open_read(db)?.connect()?;
    let mut bin = crate::gen::BinOut::create(&out).map_err(|e| format!("{out}: {e}"))?;
    let mut seen: std::collections::HashSet<i64> = std::collections::HashSet::new();
    let (mut lines, mut dups, mut nogid, mut bad, mut nolabel) = (0u64, 0u64, 0u64, 0u64, 0u64);
    // value of `"key":` up to the next , } or ] (strings keep their quotes stripped)
    fn field<'a>(s: &'a str, key: &str) -> Option<&'a str> {
        let i = s.find(key)? + key.len();
        let t = &s[i..];
        let e = t.find(|c| c == ',' || c == '}' || c == ']').unwrap_or(t.len());
        Some(t[..e].trim_matches('"'))
    }
    for path in &inputs {
        let text = std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?;
        for line in text.lines().filter(|l| !l.trim().is_empty()) {
            lines += 1;
            let Some((start, moves, res)) = parse_game_line(line) else {
                bad += 1;
                continue;
            };
            // the game's DAG id from its sequence hash
            let mut b = start;
            let mut h = seq_root(&start);
            for &m in &moves {
                let code = seq_code(&b, m);
                h = seq_child(h, code);
                b = b.make_uai(seq_move(code));
            }
            let exists = c.one_i64(&format!("SELECT COUNT(*) FROM seq_keys WHERE gkey={}", h as i64))?;
            if exists == 0 {
                nogid += 1;
                continue;
            }
            let gid = c.one_i64(&format!("SELECT gid FROM seq_keys WHERE gkey={}", h as i64))?;
            if !seen.insert(gid) {
                dups += 1;
                continue;
            }
            let Some(mi) = line.find("\"moves\":[") else {
                bad += 1;
                continue;
            };
            let segs: Vec<&str> = line[mi..].split("{\"ply\":").skip(1).collect();
            if segs.len() != moves.len() {
                bad += 1;
                continue;
            }
            let (mut plies, mut best): (Vec<(u16, i16)>, Vec<(u16, i16)>) = (Vec::new(), Vec::new());
            let mut ok = true;
            for (seg, m) in segs.iter().zip(&moves) {
                let score = field(seg, "\"score\":").and_then(|v| v.parse::<i32>().ok());
                let depth = field(seg, "\"depth\":").and_then(|v| v.parse::<u16>().ok()).unwrap_or(0).min(63);
                let Some(score) = score else {
                    ok = false;
                    break;
                };
                plies.push((move_code(*m) | (depth << 10), enc_score(score)));
                match (field(seg, "\"engine_best\":").and_then(parse_move_text), field(seg, "\"engine_best_score\":").and_then(|v| v.parse::<i32>().ok())) {
                    (Some(bm), Some(bs)) => best.push((move_code(bm), enc_score(bs))),
                    _ => best.push((0, crate::gamedata::NO_SCORE)),
                }
            }
            if !ok {
                bad += 1;
                continue;
            }
            if best.iter().any(|x| x.0 == 0) {
                nolabel += 1;
            }
            let num = |k: &str| field(line, k).and_then(|v| v.parse::<u32>().ok());
            let pg = crate::gamedata::PackedGame {
                occ: start.occ,
                blockers: start.blockers,
                turn: start.turn,
                cand_seat: 0,
                forfeit: false,
                final_cnt: [num("\"final_x\":").unwrap_or(254).min(254) as u8, num("\"final_o\":").unwrap_or(254).min(254) as u8],
                result: res,
                src: 0,
                book: num("\"book\":").unwrap_or(0),
                pair: num("\"pair\":").unwrap_or(0),
                plies,
            };
            bin.append(&pg, gid as u32, &best).map_err(|e| format!("{out}: {e}"))?;
        }
    }
    println!(
        "jsonl-to-bin: {lines} lines read -> {} games, {} plies written to {out}.tgam + {out}.tbst; repeated sequences skipped {dups}, no DAG id {nogid}, unreadable {bad}; games without best-move labels {nolabel}",
        bin.games, bin.plies
    );
    Ok(())
}

/// `ti dag seq-import --db SEQ.db --dir DIR [--threads 4]`: read every `*.jsonl` game file under DIR (the files `ti gen`
/// wrote) into the sequence DAG. Each distinct game is stored once; the files are only read.
fn run_seq_import(db: &str, args: &[String]) -> Result<(), String> {
    let dir = crate::flag_str(args, "--dir").ok_or("usage: ti dag seq-import --db SEQ.db --dir DIR [--threads 4]")?;
    let threads = crate::flag_u64(args, "--threads").unwrap_or(4).max(1) as usize;
    let mut files: Vec<std::path::PathBuf> = Vec::new();
    let mut stack = vec![std::path::PathBuf::from(&dir)];
    while let Some(d) = stack.pop() {
        for e in std::fs::read_dir(&d).map_err(|e| format!("{}: {e}", d.display()))?.flatten() {
            let p = e.path();
            if p.is_dir() {
                stack.push(p);
            } else if p.extension().and_then(|x| x.to_str()) == Some("jsonl") {
                files.push(p);
            }
        }
    }
    files.sort();
    let dag = SeqDag::open(db)?;
    let next = AtomicU64::new(0);
    let (stored, dup, bad, retries) = (AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0));
    let err: Mutex<Option<String>> = Mutex::new(None);
    let t0 = Instant::now();
    std::thread::scope(|s| {
        for _ in 0..threads {
            s.spawn(|| {
                let run = || -> Result<(), String> {
                    let c = dag.connect()?;
                    loop {
                        let i = next.fetch_add(1, Ordering::Relaxed) as usize;
                        let Some(p) = files.get(i) else { return Ok(()) };
                        let src = format!(
                            "{}/{}",
                            p.parent().and_then(|x| x.file_name()).map(|n| n.to_string_lossy().into_owned()).unwrap_or_default(),
                            p.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default()
                        );
                        let text = std::fs::read_to_string(p).map_err(|e| format!("{}: {e}", p.display()))?;
                        let (mut fs, mut fd, mut fb) = (0u64, 0u64, 0u64);
                        for (n, line) in text.lines().enumerate() {
                            let g = parse_game_line(line);
                            let Some((start, moves, res)) = g else {
                                fb += 1;
                                continue;
                            };
                            match c.record(&src, n as i64, start, &moves, res, false)? {
                                Some((_, r)) => {
                                    fs += 1;
                                    retries.fetch_add(r as u64, Ordering::Relaxed);
                                }
                                None => fd += 1,
                            }
                        }
                        stored.fetch_add(fs, Ordering::Relaxed);
                        dup.fetch_add(fd, Ordering::Relaxed);
                        bad.fetch_add(fb, Ordering::Relaxed);
                        println!("{src}: stored {fs}, already present (same sequence) {fd}, unreadable {fb}  [{:.0}s]", t0.elapsed().as_secs_f64());
                    }
                };
                if let Err(e) = run() {
                    *err.lock().unwrap() = Some(e);
                }
            });
        }
    });
    if let Some(e) = err.lock().unwrap().take() {
        return Err(e);
    }
    println!(
        "done: {} files, stored {}, same sequence already present {}, unreadable {}, conflict retries {} in {:.0}s",
        files.len(),
        stored.load(Ordering::Relaxed),
        dup.load(Ordering::Relaxed),
        bad.load(Ordering::Relaxed),
        retries.load(Ordering::Relaxed),
        t0.elapsed().as_secs_f64()
    );
    run_seq_stat(db)
}

/// One `ti gen` / gate game line -> (start position, moves, result for side 0: 2 = x won, 1 = draw, 0 = o won).
fn parse_game_line(line: &str) -> Option<(Board, Vec<Move>, u8)> {
    let si = line.find("\"start\":\"")?;
    let rest = &line[si + 9..];
    let start = Board::from_ataxx_fen(&rest[..rest.find('"')?])?;
    let num = |key: &str| -> Option<i64> {
        let i = line.find(key)? + key.len();
        let t: String = line[i..].chars().take_while(|c| c.is_ascii_digit()).collect();
        t.parse().ok()
    };
    let (fx, fo) = (num("\"final_x\":")?, num("\"final_o\":")?);
    let res = if fx > fo { 2 } else if fx == fo { 1 } else { 0 };
    let mut moves = Vec::new();
    let mut tail = &line[line.find("\"moves\":[")?..];
    while let Some(i) = tail.find("\"move\":\"") {
        tail = &tail[i + 8..];
        let j = tail.find('"')?;
        moves.push(parse_move_text(&tail[..j])?);
        tail = &tail[j..];
    }
    (!moves.is_empty()).then_some((start, moves, res))
}

/// `ti dag probe --db X.db --jsonl FILE [--game N]`: read-only. Replays game N (0-based line) of a `ti gen` file from the
/// standard start and prints, per ply, how many legal moves the position has, how many of them the DAG already holds as
/// edges, the node visit count, and whether the move played is one of the held edges.
fn run_probe(db: &str, args: &[String]) -> Result<(), String> {
    let path = crate::flag_str(args, "--jsonl").ok_or("usage: ti dag probe --db X.db --jsonl FILE [--game N]")?;
    let game = crate::flag_u64(args, "--game").unwrap_or(0) as usize;
    let text = std::fs::read_to_string(&path).map_err(|e| format!("{path}: {e}"))?;
    let line = text.lines().nth(game).ok_or("no such game line")?;
    let mut moves: Vec<String> = Vec::new();
    let mut rest = line;
    while let Some(i) = rest.find("\"move\":\"") {
        rest = &rest[i + 8..];
        let j = rest.find('"').ok_or("bad json")?;
        moves.push(rest[..j].to_string());
        rest = &rest[j..];
    }
    let c = Dag::open_read(db)?.connect()?;
    let mut b = Board::start();
    let mut ml = titanium::MoveList::new();
    println!("ply legal held(edges) visits played_held");
    let (mut full, mut n_ply) = (0u64, 0u64);
    for (ply, t) in moves.iter().enumerate() {
        b.legal_moves_dedup_into(&mut ml);
        let edges = c.edge_options(&b)?;
        let held: std::collections::HashSet<String> = edges.iter().map(|e| e.mv.clone()).collect();
        let legal_held = ml.iter().filter(|m| held.contains(&crate::gen::edge_text(&b, *m))).count();
        let m = parse_move_text(t).ok_or_else(|| format!("bad move {t}"))?;
        let played_held = held.contains(&crate::gen::edge_text(&b, m));
        println!("{ply} {} {} {} {}", ml.len(), legal_held, c.visits(&b)?, played_held);
        n_ply += 1;
        if legal_held >= ml.len() {
            full += 1;
        }
        b = b.make_uai(m);
    }
    println!("plies {n_ply}, plies where every legal move is a DAG edge: {full}");
    Ok(())
}

fn run_add(db: &str, args: &[String]) -> Result<(), String> {
    let path = crate::flag_str(args, "--games").ok_or("usage: ti dag add --db X.db --games F.tgam [--threads 4]")?;
    let threads = crate::flag_u64(args, "--threads").unwrap_or(4).max(1) as usize;
    let src = crate::flag_str(args, "--src").unwrap_or_else(|| {
        std::path::Path::new(&path).file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_else(|| path.clone())
    });
    let dag = Dag::open(db)?;
    let rd = crate::gamedata::GameReader::open(&path)?;
    let total = rd.games;
    let rd = Mutex::new((rd, 0i64));
    let (added, skipped, unknown, plies, retries) = (AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0));
    let err: Mutex<Option<String>> = Mutex::new(None);
    let t0 = Instant::now();
    std::thread::scope(|s| {
        for _ in 0..threads {
            s.spawn(|| {
                let run = || -> Result<(), String> {
                    let c = dag.connect()?;
                    loop {
                        let (g, idx) = {
                            let mut l = rd.lock().unwrap();
                            let Some(g) = l.0.next() else { return Ok(()) };
                            l.1 += 1;
                            (g?, l.1 - 1)
                        };
                        if g.result > 2 {
                            unknown.fetch_add(1, Ordering::Relaxed);
                            continue;
                        }
                        let (pl, _) = g.replay().map_err(|e| format!("game {idx}: {e}"))?;
                        let moves: Vec<Move> = pl.iter().map(|p| p.mv).collect();
                        let header = format!("tgam {src} source {}", g.src);
                        let rec = GameRec {
                            src: &src,
                            idx,
                            header: &header,
                            book: g.book as i64,
                            titanium_is: match g.cand_seat {
                                0 => "x",
                                1 => "o",
                                _ => "both",
                            },
                            start: g.start(),
                            moves: &moves,
                            result_side0: g.result,
                        };
                        match c.record_game(&rec)? {
                            Some(r) => {
                                added.fetch_add(1, Ordering::Relaxed);
                                plies.fetch_add(moves.len() as u64, Ordering::Relaxed);
                                retries.fetch_add(r as u64, Ordering::Relaxed);
                            }
                            None => {
                                skipped.fetch_add(1, Ordering::Relaxed);
                            }
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
        return Err(e);
    }
    let dt = t0.elapsed().as_secs_f64();
    let (a, p) = (added.load(Ordering::Relaxed), plies.load(Ordering::Relaxed));
    println!(
        "{path}: {total} games -> added {a}, already stored {}, result unknown (skipped) {}",
        skipped.load(Ordering::Relaxed),
        unknown.load(Ordering::Relaxed)
    );
    println!(
        "{threads} writer threads: {dt:.1}s = {:.0} games/s, {:.0} row upserts/s, conflict retries {}",
        a as f64 / dt,
        (2 * p + a) as f64 / dt,
        retries.load(Ordering::Relaxed)
    );
    Ok(())
}

fn run_bench(db: &str, args: &[String]) -> Result<(), String> {
    let threads = crate::flag_u64(args, "--threads").unwrap_or(4).max(1) as usize;
    let dag = Dag::open_read(db)?;
    let c = dag.connect()?;
    let t0 = Instant::now();
    let (mut n, mut keys) = (0u64, Vec::new());
    block_on(async {
        let mut rows = c.conn.query("SELECT o0,o1,blk,stm FROM nodes", ()).await?;
        while let Some(r) = rows.next().await? {
            n += 1;
            if n % 13 == 0 && keys.len() < 200_000 {
                keys.push(Board::from_occ(row_u64(&r, 0)?, row_u64(&r, 1)?, row_u64(&r, 2)?, row_u64(&r, 3)? as u8));
            }
        }
        Ok::<(), Error>(())
    })
    .map_err(|e| format!("bench scan: {e}"))?;
    let dt = t0.elapsed().as_secs_f64();
    println!("scan: {n} node rows in {dt:.2}s = {:.0} rows/s", n as f64 / dt);
    for th in [1usize, threads] {
        let t0 = Instant::now();
        let missing = AtomicU64::new(0);
        std::thread::scope(|s| {
            for _ in 0..th {
                s.spawn(|| {
                    let c = dag.connect().unwrap();
                    for b in &keys {
                        if c.visits(b).unwrap() == 0 {
                            missing.fetch_add(1, Ordering::Relaxed);
                        }
                    }
                });
            }
        });
        let dt = t0.elapsed().as_secs_f64();
        println!("point lookups, {th} thread(s): {:.0}/s (missing {})", (th * keys.len()) as f64 / dt, missing.load(Ordering::Relaxed));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp(name: &str) -> String {
        let p = std::env::temp_dir().join(format!("ti_dag_test_{name}_{}.db", std::process::id()));
        for ext in ["", "-wal", "-log", "-shm"] {
            let _ = std::fs::remove_file(format!("{}{ext}", p.display()));
        }
        p.to_string_lossy().into_owned()
    }

    fn first_moves(n: usize) -> (Board, Vec<Move>) {
        let start = Board::start();
        let (mut b, mut out) = (start, Vec::new());
        for _ in 0..n {
            let m = (1..530u16).find_map(|c| titanium::ttcodec::decode_move(&b, c)).expect("a legal move");
            out.push(m);
            b = b.make_uai(m);
        }
        (start, out)
    }

    #[test]
    fn move_text_round_trip() {
        for sq in 0..49u8 {
            let m = Move { from: sq, to: 48 - sq };
            assert_eq!(parse_move_text(&move_text(m)), Some(m));
        }
        assert_eq!(sq_name(0), "a7");
        assert_eq!(sq_name(48), "g1");
        assert_eq!(parse_move_text("pass"), Some(Move::PASS));
        assert_eq!(parse_move_text("h1a1"), None);
    }

    #[test]
    fn record_is_counted_once_and_views_are_the_movers() {
        let path = tmp("record");
        let dag = Dag::open(&path).unwrap();
        let c = dag.connect().unwrap();
        let (start, moves) = first_moves(4);
        assert_eq!(c.visits(&start).unwrap(), 0);
        let rec = |idx| GameRec { src: "t", idx, header: "h", book: 0, titanium_is: "both", start, moves: &moves, result_side0: 2 };
        assert_eq!(c.record_game(&rec(0)).unwrap(), Some(0));
        assert_eq!(c.record_game(&rec(0)).unwrap(), None, "same (src, idx) is stored once");
        assert_eq!(c.visits(&start).unwrap(), 1);
        assert_eq!(c.record_game(&rec(1)).unwrap(), Some(0));
        assert_eq!(c.visits(&start).unwrap(), 2);
        // side 0 won: rows where side 0 moves count a win, rows where side 1 moves count a loss
        assert_eq!(c.one_i64("SELECT SUM(l) FROM nodes WHERE stm = 0").unwrap(), 0);
        assert_eq!(c.one_i64("SELECT SUM(w) FROM nodes WHERE stm = 1").unwrap(), 0);
        assert_eq!(c.one_i64("SELECT SUM(n) FROM nodes").unwrap(), 10);
        assert_eq!(c.one_i64("SELECT SUM(n) FROM edges").unwrap(), 8);
        assert_eq!(c.one_i64("SELECT COUNT(*) FROM nodes WHERE n != w + d + l").unwrap(), 0);
        assert_eq!(c.one_i64("SELECT MIN(min_ply) FROM nodes").unwrap(), 0);
    }

    #[test]
    fn parallel_writers_lose_nothing() {
        let path = tmp("parallel");
        let dag = Dag::open(&path).unwrap();
        let (start, moves) = first_moves(6);
        std::thread::scope(|s| {
            for t in 0..4i64 {
                let (dag, moves) = (&dag, &moves);
                s.spawn(move || {
                    let c = dag.connect().unwrap();
                    for i in 0..25 {
                        let rec = GameRec { src: "p", idx: t * 100 + i, header: "h", book: 0, titanium_is: "both", start, moves, result_side0: 1 };
                        c.record_game(&rec).unwrap();
                    }
                });
            }
        });
        let c = dag.connect().unwrap();
        assert_eq!(c.one_i64("SELECT COUNT(*) FROM games").unwrap(), 100);
        assert_eq!(c.visits(&start).unwrap(), 100);
        assert_eq!(c.one_i64("SELECT SUM(n) FROM nodes").unwrap(), 700);
        assert_eq!(c.one_i64("SELECT SUM(d) FROM nodes").unwrap(), 700);
    }
}

//! `ti gen`: self-play game generator for the data flywheel.
//!
//! Why it lives in the hub: the Modal image builds only `ti`. `ti sprt` needs an
//! external engine binary, so it cannot generate games there; this command plays
//! the games in-process against the linked engine crate and writes the same
//! `.jsonl` schema the gate writes, which `ti data pack-games` consumes as is.
//!
//! Rules: Botzone. The game ends the moment either side is stuck and the mobile
//! side absorbs the empties (`bz_over` / `bz_counts`), so no pass is ever emitted.
//!
//!   ti gen --out FILE [--games 500] [--time-ms 100] [--book FILE] [--book0 N]
//!          [--tt-bits 18] [--network FILE|bundled] [--threads 1]

use std::fs::File;
use std::io::{BufWriter, Write};
use std::process::ExitCode;
use std::time::Duration;

use titanium::nnue::Nnue;
use titanium::{Board, SearchLimits, Searcher};

/// UAI square name ("a7" .. "g1") for a 0..48 board index (row 0 = rank 7).
fn sq_name(sq: u8) -> String {
    let r = (sq / 7) as u8;
    let c = (sq % 7) as u8;
    format!("{}{}", (b'a' + c) as char, 7 - r)
}

pub fn run_gen(args: &[String]) -> ExitCode {
    let Some(out) = crate::flag_str(args, "--out") else {
        eprintln!("usage: ti gen --out FILE [--games 500] [--time-ms 100] [--book FILE] [--book0 N] [--tt-bits 18] [--network FILE|bundled]");
        return ExitCode::FAILURE;
    };
    let games = crate::flag_u64(args, "--games").unwrap_or(500) as u32;
    let time_ms = crate::flag_u64(args, "--time-ms").unwrap_or(100);
    let book0 = crate::flag_u64(args, "--book0").unwrap_or(0) as usize;
    let tt_bits = crate::flag_u64(args, "--tt-bits").unwrap_or(18) as usize;
    let network = crate::flag_str(args, "--network").unwrap_or_else(|| "bundled".into());

    let owned_net: Option<Nnue> = if network == "bundled" || network == "none" {
        None
    } else {
        match Nnue::load(&network) {
            Ok(n) => Some(n),
            Err(e) => {
                eprintln!("gen: cannot load network {network}: {e}");
                return ExitCode::FAILURE;
            }
        }
    };
    let net: Option<&Nnue> = match owned_net.as_ref() {
        Some(n) => Some(n),
        None if network == "none" => None,
        None => Some(Nnue::bundled_static()),
    };

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

    let mut w = match File::create(&out) {
        Ok(f) => BufWriter::with_capacity(1 << 20, f),
        Err(e) => {
            eprintln!("gen: cannot create {out}: {e}");
            return ExitCode::FAILURE;
        }
    };

    let t0 = std::time::Instant::now();
    for game in 0..games {
        let idx = match book.as_ref().filter(|l| !l.is_empty()) {
            Some(lines) => (book0 + game as usize) % lines.len(),
            None => 0,
        };
        let mut b = match book.as_ref().filter(|l| !l.is_empty()) {
            Some(lines) => match Board::from_str(&lines[idx]) {
                Some(x) => x,
                None => {
                    eprintln!("gen: bad book line: {}", lines[idx]);
                    return ExitCode::FAILURE;
                }
            },
            None => Board::start(),
        };
        let start_fen = b.to_ataxx_fen();
        let mut searcher = Searcher::with_tt_bits_unconfigured(tt_bits);
        searcher.set_nnue(net);
        let mut moves = String::new();
        let mut ply = 0u32;
        while !b.bz_over() && ply < 400 {
            if !b.has_moves(b.turn) {
                break;
            }
            let limits = SearchLimits { time: Some(Duration::from_millis(time_ms)), max_nodes: None, max_depth: 30 };
            let r = searcher.search(&b, &limits);
            let Some(mv) = r.best else { break };
            let side = if b.turn == 0 { 'x' } else { 'o' };
            if !moves.is_empty() {
                moves.push(',');
            }
            moves.push_str(&format!(
                "{{\"ply\":{ply},\"side\":\"{side}\",\"mover\":\"candidate\",\"move\":\"{}{}\",\"score\":{},\"mate\":null,\"depth\":{},\"nodes\":{},\"ms\":{}}}",
                sq_name(mv.from),
                sq_name(mv.to),
                r.score,
                r.depth,
                r.nodes,
                r.elapsed.as_millis()
            ));
            b = b.make(mv);
            ply += 1;
        }
        let (bx, bo) = b.bz_counts();
        let p0 = if bx > bo { 1.0 } else if bx < bo { 0.0 } else { 0.5 };
        let _ = writeln!(
            w,
            "{{\"pair\":{game},\"book\":{idx},\"start\":\"{start_fen}\",\"candidate_seat\":\"x\",\"candidate_points\":{p0},\"final_x\":{bx},\"final_o\":{bo},\"moves\":[{moves}]}}"
        );
        if game % 50 == 49 {
            let _ = w.flush();
            eprintln!("gen: {} games ({:.0}s elapsed)", game + 1, t0.elapsed().as_secs_f64());
        }
    }
    let _ = w.flush();
    eprintln!("gen: done {games} games -> {out} in {:.0}s", t0.elapsed().as_secs_f64());
    ExitCode::SUCCESS
}

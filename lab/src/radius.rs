//! `lab radius` (D961): the AREA OF EFFECT of a stone in the champion eval. For every pair of stones (A, B) of random
//! positions: interaction I(A,B) = f(A,B flipped) - f(A flipped) - f(B flipped) + f(nothing flipped), f = static eval
//! (side to move). I = how much B changes the value of A (and vice versa). Mean |I| by relative offset and by distance.
//!
//!   lab radius --tdat X.tdat [--pos 400] [--seed 5] [--out FILE]
use crate::hcefit::{phase, CLAMP, NPH, PH_NAME};
use crate::util::{arg, Rec, REC};
use std::fs::File;
use std::io::{Read, Seek, SeekFrom, Write};
use std::process::ExitCode;
use titanium::nnue::Nnue;
use titanium::{Board, MoveList};

pub fn run(args: &[String]) -> ExitCode {
    let Some(path) = arg(args, "--tdat") else {
        eprintln!("usage: lab radius --tdat X.tdat [--pos 400] [--seed 5] [--out FILE]");
        return ExitCode::from(2);
    };
    let npos: usize = arg(args, "--pos").and_then(|v| v.parse().ok()).unwrap_or(400);
    let mut seed: u64 = arg(args, "--seed").and_then(|v| v.parse::<u64>().ok()).unwrap_or(5) | 1;
    let out_path = arg(args, "--out");
    let net = Nnue::bundled();
    let mut f = File::open(&path).expect("tdat");
    let total = f.metadata().map(|m| m.len() as usize / REC).unwrap_or(0) as u64;
    let mut buf = [0u8; REC];
    let mut ml = MoveList::new();
    // [phase][same colour 0 / opposite 1][dr+6][dc+6] -> (sum |I|, sum I, n)
    let mut grid = vec![[[[(0f64, 0f64, 0u64); 13]; 13]; 2]; NPH];
    let mut by_d = vec![[[(0f64, 0u64); 7]; 2]; NPH];
    let t0 = std::time::Instant::now();
    let mut done = 0usize;
    let flip = |me: u64, op: u64, m: u64| -> (u64, u64) { ((me & !m) | (op & m), (op & !m) | (me & m)) };
    while done < npos {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        if f.seek(SeekFrom::Start((seed % total) * REC as u64)).is_err() || f.read_exact(&mut buf).is_err() {
            continue;
        }
        let r = Rec::from_bytes(&buf);
        let (me, op) = if r.stm == 0 { (r.occ0, r.occ1) } else { (r.occ1, r.occ0) };
        let b = Board::from_occ(me, op, 0, 0);
        b.legal_moves_dedup_into(&mut ml);
        if ml.len() < 2 {
            continue;
        }
        let ev = |m: u64| -> f64 {
            let (a, o) = flip(me, op, m);
            (net.eval(&Board::from_occ(a, o, 0, 0), 0) as f64).clamp(-CLAMP, CLAMP)
        };
        let f0 = ev(0);
        // skip decided positions: interactions there are game-result jumps, not patterns
        if f0.abs() > 600.0 {
            continue;
        }
        let ph = phase(b.empty());
        let stones: Vec<usize> = (0..49).filter(|&s| (me | op) >> s & 1 == 1).collect();
        let f1: Vec<f64> = stones.iter().map(|&s| ev(1u64 << s)).collect();
        for i in 0..stones.len() {
            for j in i + 1..stones.len() {
                let (a, bq) = (stones[i], stones[j]);
                let fij = ev((1u64 << a) | (1u64 << bq));
                if fij.abs() >= CLAMP || f1[i].abs() >= CLAMP || f1[j].abs() >= CLAMP {
                    continue;
                }
                let iv = fij - f1[i] - f1[j] + f0;
                let same = (((me >> a) & 1) == ((me >> bq) & 1)) as usize ^ 1; // 0 = same colour
                let (dr, dc) = ((bq / 7) as i32 - (a / 7) as i32, (bq % 7) as i32 - (a % 7) as i32);
                for (rr, cc) in [(dr, dc), (-dr, -dc)] {
                    let g = &mut grid[ph][same][(rr + 6) as usize][(cc + 6) as usize];
                    g.0 += iv.abs();
                    g.1 += iv;
                    g.2 += 1;
                }
                let d = dr.abs().max(dc.abs()) as usize;
                by_d[ph][same][d].0 += iv.abs();
                by_d[ph][same][d].1 += 1;
            }
        }
        done += 1;
    }
    let mut rep = String::new();
    let mut say = |s: String| {
        println!("{s}");
        rep.push_str(&s);
        rep.push('\n');
    };
    say(format!("radius: {npos} undecided positions (|eval| <= 600), every stone pair, {:.0}s", t0.elapsed().as_secs_f64()));
    say("mean |interaction| (cp) between two stones by CHEBYSHEV distance 1..6 (1 = touching, 2 = jump distance):".into());
    for ph in 0..NPH {
        for sm in 0..2 {
            let cells: Vec<String> = (1..7)
                .map(|d| {
                    let (s, n) = by_d[ph][sm][d];
                    if n > 0 { format!("d{d} {:>6.1}", s / n as f64) } else { format!("d{d}      .") }
                })
                .collect();
            say(format!("  {:<14} {:<9} {}", PH_NAME[ph], if sm == 0 { "same" } else { "opposite" }, cells.join("  ")));
        }
    }
    for ph in [1usize, 2] {
        for sm in 0..2 {
            say(format!("mean |interaction| by relative offset (centre = the stone), {}, {} colour:", PH_NAME[ph], if sm == 0 { "same" } else { "opposite" }));
            for rr in 0..13 {
                let line: String = (0..13)
                    .map(|cc| {
                        let (s, _, n) = grid[ph][sm][rr][cc];
                        if rr == 6 && cc == 6 { "     *".to_string() } else if n >= 20 { format!("{:>6.1}", s / n as f64) } else { "     .".to_string() }
                    })
                    .collect();
                say(format!("  {line}"));
            }
        }
    }
    if let Some(p) = out_path {
        if let Some(dir) = std::path::Path::new(&p).parent() {
            let _ = std::fs::create_dir_all(dir);
        }
        match File::create(&p).and_then(|mut f| f.write_all(rep.as_bytes())) {
            Ok(_) => println!("report written: {p}"),
            Err(e) => eprintln!("{p}: {e}"),
        }
    }
    ExitCode::SUCCESS
}

//! External-engine static evals on OUR validation rows, scored with the exact `lab hce-train` val metric (D993).
//!
//!   lab val-fen   --val V.tdat --out rows.fen          one FEN per val row (side to move = 'x'), same 600k cap as hce-train
//!   lab val-score --val V.tdat --scores S.txt [--col 0] [--name X]
//!                 S.txt: one line per FEN row, whitespace-separated integer cp columns from the side to move's view.
//!
//! Prints the val loss (wsum * |sigmoid(k * cp / 400) - target_of|^2.5) at k = 1 (the engine's own cp scale) and at the best
//! single scale k (grid), and the champion NNUE on the same rows. A scale is the only fair calibration between engines whose
//! cp units differ; nothing else is fitted.
use crate::hcetrain::{read_rows, sig, target};
use crate::util::{arg, Rec};
use std::io::{BufRead, BufReader, Write};
use std::process::ExitCode;
use titanium::nnue::Nnue;
use titanium::Board;

pub(crate) fn own_opp(r: &Rec) -> (u64, u64) {
    if r.stm == 0 { (r.occ0, r.occ1) } else { (r.occ1, r.occ0) }
}

pub fn fen(args: &[String]) -> ExitCode {
    let (Some(v), Some(o)) = (arg(args, "--val"), arg(args, "--out")) else {
        eprintln!("usage: lab val-fen --val V.tdat --out rows.fen");
        return ExitCode::from(2);
    };
    let rows = read_rows(&v, 600_000);
    let mut w = std::io::BufWriter::new(std::fs::File::create(&o).expect("out"));
    for r in &rows {
        let (me, op) = own_opp(r);
        let mut s = String::with_capacity(64);
        for row in (0..7).rev() {
            let mut run = 0;
            for col in 0..7 {
                let b = 1u64 << (row * 7 + col);
                let c = if me & b != 0 { 'x' } else if op & b != 0 { 'o' } else { ' ' };
                if c == ' ' {
                    run += 1;
                    continue;
                }
                if run > 0 {
                    s.push(char::from(b'0' + run));
                    run = 0;
                }
                s.push(c);
            }
            if run > 0 {
                s.push(char::from(b'0' + run));
            }
            if row > 0 {
                s.push('/');
            }
        }
        writeln!(w, "{s} x").unwrap();
    }
    println!("{} FEN rows -> {o}", rows.len());
    ExitCode::SUCCESS
}

fn loss(rows: &[Rec], cp: &[f32], k: f32) -> f64 {
    let (mut l, mut ws) = (0f64, 0f64);
    for (r, &c) in rows.iter().zip(cp) {
        let p = sig(k * c / 400.0);
        let t = target(r, false);
        let wt = r.wsum.max(0.0) as f64;
        l += wt * ((p - t).abs() as f64).powf(2.5);
        ws += wt;
    }
    l / ws
}

fn report(name: &str, rows: &[Rec], cp: &[f32]) -> String {
    let l1 = loss(rows, cp, 1.0);
    let mut best = (f64::MAX, 1.0f32);
    let mut k = 0.05f32;
    while k <= 8.0 {
        let l = loss(rows, cp, k);
        if l < best.0 {
            best = (l, k);
        }
        k *= 1.05;
    }
    format!("{name}: val loss {l1:.6} at its own cp scale | {:.6} at best scale k = {:.3}", best.0, best.1)
}

/// Zataxx (zzzzz151) nn.rs, bit-exact: 36 2x2 windows (3^4 states each, 2916 inputs) -> 256 i16 -> SCReLU -> 1, QA 255,
/// QB 64, SCALE 400; i16 accumulator wraps like the original release build. Weights read from its own net6.bin.
pub struct Zat {
    pub(crate) fw: Vec<i16>,
    pub(crate) fb: Vec<i16>,
    pub(crate) ow: Vec<i16>,
    pub(crate) obias: i16,
    pub(crate) trunc: bool,
}

impl Zat {
    pub(crate) fn load(p: &str) -> Zat {
        let b = std::fs::read(p).unwrap_or_else(|e| panic!("{p}: {e}"));
        assert_eq!(b.len(), (2916 * 256 + 256 + 256 + 1) * 2, "{p}: unexpected size");
        let v: Vec<i16> = b.chunks_exact(2).map(|c| i16::from_le_bytes([c[0], c[1]])).collect();
        let n = 2916 * 256;
        Zat { fw: v[..n].to_vec(), fb: v[n..n + 256].to_vec(), ow: v[n + 256..n + 512].to_vec(), obias: v[n + 512], trunc: false }
    }
    pub(crate) fn eval(&self, us: u64, them: u64) -> i32 {
        const POW: [usize; 4] = [1, 3, 9, 27];
        const MASK: u64 = 0b0001_1000_0011;
        let mut h: [i16; 256] = self.fb.clone().try_into().unwrap();
        for i in 0..6 {
            for j in 0..6 {
                let off = 7 * i + j;
                let mut idx = 81 * (6 * i + j);
                let (mut u, mut t) = ((us >> off) & MASK, (them >> off) & MASK);
                while u > 0 {
                    let mut sq = u.trailing_zeros() as usize;
                    u &= u - 1;
                    if sq > 6 {
                        sq -= 5;
                    }
                    idx += POW[sq];
                }
                while t > 0 {
                    let mut sq = t.trailing_zeros() as usize;
                    t &= t - 1;
                    if sq > 6 {
                        sq -= 5;
                    }
                    idx += 2 * POW[sq];
                }
                let row = &self.fw[idx * 256..idx * 256 + 256];
                for k in 0..256 {
                    h[k] = h[k].wrapping_add(row[k]);
                }
            }
        }
        let mut sum = 0i32;
        for k in 0..256 {
            let x = h[k].clamp(0, 255) as i32;
            sum += if self.trunc { x * ((x * self.ow[k] as i32) as i16) as i32 } else { x * x * self.ow[k] as i32 };
        }
        ((sum / 255 + self.obias as i32) * 400 / (255 * 64)).clamp(-30999, 30999)
    }
}

fn vflip(b: u64) -> u64 {
    (0..7).fold(0u64, |m, r| m | ((b >> (7 * r)) & 0x7F) << (7 * (6 - r)))
}

pub fn zataxx(args: &[String]) -> ExitCode {
    let (Some(v), Some(np)) = (arg(args, "--val"), arg(args, "--net")) else {
        eprintln!("usage: lab zataxx-val --val V.tdat --net C:/gitProjects/Zataxx/src/net6.bin");
        return ExitCode::from(2);
    };
    let mut z = Zat::load(&np);
    z.trunc = arg(args, "--avx2").is_some_and(|v| v == "1"); // AVX2 build: _mm256_mullo_epi16(v, w) keeps 16 bits of v*w
    let rows = read_rows(&v, 600_000);
    let t0 = std::time::Instant::now();
    let a: Vec<f32> = rows.iter().map(|r| { let (me, op) = own_opp(r); z.eval(me, op) as f32 }).collect();
    let ns = t0.elapsed().as_nanos() as f64 / rows.len() as f64;
    println!("parity sample (first 8 val rows, board as given): {:?}", &a[..8]);
    let f: Vec<f32> = rows.iter().map(|r| { let (me, op) = own_opp(r); z.eval(vflip(me), vflip(op)) as f32 }).collect();
    let net = Nnue::bundled();
    let t1 = std::time::Instant::now();
    let nn: Vec<f32> = rows.iter().map(|r| { let (me, op) = own_opp(r); net.eval(&Board::from_occ(me, op, 0, 0), 0) as f32 }).collect();
    let nn_ns = t1.elapsed().as_nanos() as f64 / rows.len() as f64;
    println!("{} val rows (val metric of lab hce-train); full-forward cost single thread: Zataxx {ns:.0} ns/eval (scalar port), champion NNUE {nn_ns:.0} ns/eval (net.eval, full refresh)", rows.len());
    println!("{}", report("champion NNUE 812C2118", &rows, &nn));
    println!("{}", report("Zataxx net6.bin (board as given)", &rows, &a));
    println!("{}", report("Zataxx net6.bin (board flipped vertically, orientation check)", &rows, &f));
    // sibling best-move agreement with the NNUE, same 2,000 val positions and rule as lab hce-train
    let (mut ag, mut n) = (0f64, 0f64);
    let mut ml = titanium::MoveList::new();
    for r in rows.iter().step_by((rows.len() / 2000).max(1)).take(2000) {
        let (me, op) = own_opp(r);
        let b = Board::from_occ(me, op, 0, 0);
        b.legal_moves_dedup_into(&mut ml);
        if ml.len() < 2 {
            continue;
        }
        let (mut bn, mut bz) = ((f32::MIN, 0usize), (f32::MIN, 0usize));
        for (i, mv) in ml.iter().enumerate() {
            let c = b.make(mv);
            let yn = -(net.eval(&c, 0) as f32);
            let yz = -(z.eval(c.me(), c.opp()) as f32);
            if yn > bn.0 {
                bn = (yn, i);
            }
            if yz > bz.0 {
                bz = (yz, i);
            }
        }
        ag += (bn.1 == bz.1) as u8 as f64;
        n += 1.0;
    }
    println!("sibling best-move agreement with the NNUE (2,000 val positions, as lab hce-train): Zataxx {:.2}%", 100.0 * ag / n);
    ExitCode::SUCCESS
}

pub fn score(args: &[String]) -> ExitCode {
    let (Some(v), Some(sp)) = (arg(args, "--val"), arg(args, "--scores")) else {
        eprintln!("usage: lab val-score --val V.tdat --scores S.txt [--name X]");
        return ExitCode::from(2);
    };
    let name = arg(args, "--name").unwrap_or_else(|| "external".into());
    let rows = read_rows(&v, 600_000);
    let rd = BufReader::new(std::fs::File::open(&sp).expect("scores"));
    let mut cols: Vec<Vec<f32>> = Vec::new();
    for line in rd.lines() {
        let line = line.unwrap();
        for (j, t) in line.split_whitespace().enumerate() {
            if cols.len() <= j {
                cols.push(Vec::new());
            }
            cols[j].push(t.parse::<f32>().expect("int"));
        }
    }
    assert!(!cols.is_empty() && cols[0].len() == rows.len(), "{} scores vs {} rows", cols.first().map_or(0, |c| c.len()), rows.len());
    let net = Nnue::bundled();
    let nn: Vec<f32> = rows
        .iter()
        .map(|r| {
            let (me, op) = own_opp(r);
            net.eval(&Board::from_occ(me, op, 0, 0), 0) as f32
        })
        .collect();
    println!("{} val rows (val metric of lab hce-train: wsum * |p - target_of|^2.5)", rows.len());
    println!("{}", report("champion NNUE 812C2118", &rows, &nn));
    for (j, c) in cols.iter().enumerate() {
        println!("{}", report(&format!("{name} column {j}"), &rows, c));
    }
    ExitCode::SUCCESS
}

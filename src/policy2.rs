//! `ti policy2` (D523): compact policy trainer for the FULL KataAtaxx set (~21.7M rows) + our gate-game labels on the
//! 16 GB PC. A sample stores only the board (own/opp bitboards) and its labels (the played move, or Kata's 200-visit
//! counts per square); moves, scalar features and 3x3 pattern rows are rebuilt from the board every epoch (~60 B per
//! sample instead of ~550 B). Head = what the engine runs on the non-pairwise champion (dim 0): slot bias + move scalars
//! + 3x3 pattern tables (k3 per square = TPOL2, k3c class-shared = TPOL3). Kata rows: stage 0 groups = clone destinations
//! and jump ORIGINS, stage 1 groups = destinations of the picked stone; soft loss -sum_g y_g ln(sum_{m in g} q_m).
//! Epochs run until the validation loss stops improving (early stop), the best epoch is exported.
//!
//!   ti policy2 --net NET.tnn --games a.jsonl,.. --kata DIR[,DIR] --pat k3c --init POL.bin --out-bin OUT.bin
//!              [--lr 0.003] [--pat-lr 0.01] [--kata-w 1] [--games-w 1] [--patience 4]
//!              [--adv-lambda L [--adv-scale 0.10] [--adv-clip 1]]
//!
//! `--adv-lambda L` (D765, off by default): weight every game move by exp(clip(advantage / scale, +-clip)), where
//! advantage = TD(L) return that followed the move - the mover's own logged score, in win-probability units. The
//! weights are scaled to mean 1. Needs game logs with a score per move (gate logs, `ti gen` output). Validation
//! numbers stay unweighted, so runs with and without it are comparable.
use crate::games;
use crate::policy::{arg, feats, k3code, npz_files, patidx, ring1, NONE, NSCAL, P3, SLOTS};
use std::fs;
use std::io::{BufRead, BufReader};
use std::process::ExitCode;
use titanium::{Board, Move, MoveList};

#[derive(Clone, Copy)]
struct S {
    own: u64,
    opp: u64,
    /// hard: played move from/to; soft: unused
    from: u8,
    to: u8,
    /// 0 = hard (game move), 1 = Kata stage 0, 2 = Kata stage 1 (picked stone `sel`)
    kind: u8,
    sel: u8,
    kn: u8,
    val: bool,
    ko: u32,
    /// loss weight of a hard (game move) sample: 1, or the advantage weight of `--adv-lambda` (D765)
    aw: f32,
    /// what followed the move in its game: 1 = advantage above +0.02, 2 = below -0.02, 0 = unknown / in between
    up: u8,
}

/// Advantage of each played move of a game (D765): `G_t - V_t` from the mover's view, in win-probability units.
/// `V_t` = sigmoid(the mover's logged score / 400). `G_t` = the TD(lambda) return over the LATER positions' values
/// and the game result, i.e. the value that actually followed the move. `None` where the ply has no logged score.
fn move_advantages(g: &games::Game, boards: &[Board], lambda: f32) -> Vec<Option<f32>> {
    let n = g.moves.len().min(boards.len());
    let sig = |cp: i32| 1.0 / (1.0 + (-(cp.clamp(-1500, 1500) as f32) / 400.0).exp());
    // everything from side 0's (x) view
    let vx: Vec<Option<f32>> = (0..n).map(|t| g.moves[t].score.map(|s| if boards[t].turn == 0 { sig(s) } else { 1.0 - sig(s) })).collect();
    let zx = if g.cand_seat == "x" { g.cand_points as f32 } else { 1.0 - g.cand_points as f32 };
    let mut out = vec![None; n];
    let mut ret = zx;
    for t in (0..n).rev() {
        let next = if t + 1 < n { vx[t + 1] } else { Some(zx) };
        if let Some(v) = next {
            ret = (1.0 - lambda) * v + lambda * ret;
        }
        if let Some(v) = vx[t] {
            out[t] = Some(if boards[t].turn == 0 { ret - v } else { v - ret });
        }
    }
    out
}

/// D537 jump ring: the 16 cells at distance 2 around a destination as 4 sides of 4 cells (clockwise, N E S W), base 3,
/// first cell most significant, 0 = empty or off-board, 1 = mover, 2 = opponent. MUST match the engine (policy.rs TPOL4).
pub(crate) const RING_SIDES: [[(i32, i32); 4]; 4] = [
    [(-2, -2), (-2, -1), (-2, 0), (-2, 1)],
    [(-2, 2), (-1, 2), (0, 2), (1, 2)],
    [(2, 2), (2, 1), (2, 0), (2, -1)],
    [(2, -2), (1, -2), (0, -2), (-1, -2)],
];
pub(crate) const RING_N: usize = 49 * 2 * 4 * 81;

const fn file_mask(f: usize) -> u64 {
    let mut m = 0u64;
    let mut r = 0usize;
    while r < 7 {
        m |= 1u64 << (r * 7 + f);
        r += 1;
    }
    m
}
const FULL49: u64 = (1u64 << 49) - 1;
const FILE_A: u64 = file_mask(0);
const FILE_G: u64 = file_mask(6);
pub(crate) const NSEE_BASE: usize = 18 * 19; // my_diff (0..17) x opp_state (0=stalemate, 1..18=1+max_opp_diff)

#[inline(always)]
fn box1(x: u64) -> u64 {
    let h = x | ((x & !FILE_A) >> 1) | ((x & !FILE_G) << 1);
    (h | (h >> 7) | (h << 7)) & FULL49
}

#[inline(always)]
fn victim_planes(v: u64) -> [u64; 4] {
    let s0 = (v >> 1) & !FILE_G;
    let s1 = (v << 1) & !FILE_A;
    let s2 = v >> 7;
    let s3 = (v << 7) & FULL49;
    let s4 = (v >> 8) & !FILE_G;
    let s5 = (v << 8) & !FILE_A & FULL49;
    let s6 = (v >> 6) & !FILE_A;
    let s7 = (v << 6) & !FILE_G & FULL49;
    let t0 = s0 ^ s1 ^ s2;
    let c0 = (s0 & s1) | (s2 & (s0 ^ s1));
    let t1 = s3 ^ s4 ^ s5;
    let c1 = (s3 & s4) | (s5 & (s3 ^ s4));
    let t2 = s6 ^ s7;
    let c2 = s6 & s7;
    let ones = t0 ^ t1 ^ t2;
    let k0 = (t0 & t1) | (t2 & (t0 ^ t1));
    let u0 = c0 ^ c1 ^ c2;
    let uc = (c0 & c1) | (c2 & (c0 ^ c1));
    [ones, u0 ^ k0, uc ^ (u0 & k0), uc & (u0 & k0)]
}

#[inline(always)]
fn planes_add(p: &mut [u64; 4], m: u64) {
    let c0 = p[0] & m;
    p[0] ^= m;
    let c1 = p[1] & c0;
    p[1] ^= c0;
    let c2 = p[2] & c1;
    p[2] ^= c1;
    p[3] ^= c2;
}

#[inline(always)]
fn planes_sub(p: &mut [u64; 4], m: u64) {
    let b0 = !p[0] & m;
    p[0] ^= m;
    let b1 = !p[1] & b0;
    p[1] ^= b0;
    let b2 = !p[2] & b1;
    p[2] ^= b1;
    p[3] ^= b2;
}

#[inline(always)]
fn planes_max(cand: u64, p: &[u64; 4]) -> u32 {
    let mut m = cand;
    let mut best = 0u32;
    for k in (0..4).rev() {
        let x = m & p[k];
        if x != 0 {
            best |= 1 << k;
            m = x;
        }
    }
    best
}

/// Exact 1-ply opponent reply differential state after our move:
/// 0 = opponent has no legal moves; 1..=18 = 1 + max_opp_diff (0..=17).
#[inline(always)]
fn opp_state(b1_empty: u64, b2_empty: u64, p: &[u64; 4]) -> usize {
    if b2_empty == 0 {
        return 0;
    }
    let cl = if b1_empty != 0 { 1 + 2 * planes_max(b1_empty, p) } else { 0 };
    let jp = 2 * planes_max(b2_empty, p);
    1 + cl.max(jp) as usize
}

/// ring table index of side `k` for a move to `to` (clone/jump), relative to the start of the ring block
pub(crate) fn ring_idx(me: u64, opp: u64, to: u8, jump: bool, k: usize) -> u32 {
    let (r, c) = ((to / 7) as i32, (to % 7) as i32);
    let mut code = 0u32;
    for &(dr, dc) in &RING_SIDES[k] {
        let (rr, cc) = (r + dr, c + dc);
        let d = if (0..7).contains(&rr) && (0..7).contains(&cc) {
            let bit = 1u64 << (rr * 7 + cc);
            if me & bit != 0 { 1 } else if opp & bit != 0 { 2 } else { 0 }
        } else {
            0
        };
        code = code * 3 + d;
    }
    (((to as u32 * 2 + jump as u32) * 4 + k as u32) * 81) + code
}

struct Scratch {
    ml: MoveList,
    mv: Vec<[u8; 5]>,
    pv: Vec<[u32; 2]>,
    /// D537 ring indices (absolute into the extended origin table), empty without --ring
    rv: Vec<[u32; 4]>,
    /// D741 2-ply SEE scalar indices into `g[NSCAL..]`, empty without --see
    sv: Vec<u16>,
    kg: Vec<i8>,
    ky: Vec<f32>,
    keys: Vec<u8>,
    z: Vec<f32>,
    dz: Vec<f32>,
    qg: Vec<f32>,
    /// D1320 phase bucket of the sample: base offsets into (bias, g, td, tor); all 0 with --buckets 1
    off: [usize; 4],
}

/// Rebuild moves / features / pattern rows / labels of a sample. Returns the hard label index (hard samples) or
/// Some(usize::MAX) for a soft sample with >= 2 groups; None = skip.
static RING_BASE: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(u32::MAX);
/// 0 = none, 1 = 2d (342), 2 = 2dp (4*342), 3 = 2dpt (2*4*342), 4 = 2d8 (8*342)
static SEE_MODE: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);

fn see_n(mode: u8) -> usize {
    match mode {
        1 => NSEE_BASE,
        2 => 4 * NSEE_BASE,
        3 | 4 => 8 * NSEE_BASE,
        _ => 0,
    }
}

/// D1320 (user D1312/D1313): --buckets N = one policy head per phase bucket, bucket = empties * N / 50 (the NNUE's and the
/// engine's formula); per-bucket sizes [N, bias, g, td, tor] (unset = 1 bucket, every offset 0 = the old trainer exactly)
static PB: std::sync::OnceLock<[usize; 5]> = std::sync::OnceLock::new();

fn bucket_off(s: &S) -> [usize; 4] {
    match PB.get() {
        Some(&[nb, sl, gl, tdl, tol]) if nb > 1 => {
            let e = 49 - (s.own | s.opp).count_ones() as usize;
            let k = (e * nb / 50).min(nb - 1);
            [k * sl, k * gl, k * tdl, k * tol]
        }
        _ => [0; 4],
    }
}

fn expand(s: &S, kpool: &[(u8, u16)], ring: &[u64; 49], k3c: bool, use_pat: bool, x: &mut Scratch) -> Option<usize> {
    x.off = bucket_off(s);
    let rb = RING_BASE.load(std::sync::atomic::Ordering::Relaxed);
    let sm = SEE_MODE.load(std::sync::atomic::Ordering::Relaxed);
    let b = Board::from_occ(s.own, s.opp, 0, 0);
    b.legal_moves_dedup_into(&mut x.ml);
    if x.ml.len() < 2 {
        return None;
    }
    x.mv.clear();
    x.pv.clear();
    x.rv.clear();
    x.sv.clear();
    x.kg.clear();
    x.ky.clear();
    x.keys.clear();

    let (me, opp, empty) = (b.me(), b.opp(), b.empty());
    let (p_base, b1_opp0, b2_opp0, see_prefix) = if sm != 0 {
        let pb = victim_planes(me);
        let b1 = box1(opp);
        let b2 = box1(b1);
        let ec = empty.count_ones() as usize;
        let blk = if sm == 4 {
            (ec.saturating_sub(1) / 6).min(7)
        } else if sm >= 2 {
            let ph = (ec.saturating_sub(1) / 12).min(3);
            let th = if sm == 3 {
                let os = opp_state(b1 & empty, b2 & empty, &pb);
                usize::from(os >= 8)
            } else {
                0
            };
            th * 4 + ph
        } else {
            0
        };
        (pb, b1, b2, blk * NSEE_BASE)
    } else {
        ([0; 4], 0, 0, 0)
    };
    let mut dest_ok = 0u64;
    let mut dest_p = [[0u64; 4]; 49];
    let mut dest_b1 = [0u64; 49];
    let mut dest_b2 = [0u64; 49];

    let mut label = None;
    for (i, m) in x.ml.iter().enumerate() {
        x.mv.push(feats(&b, m, ring));
        if use_pat {
            x.pv.push(patidx(&b, m, k3c));
        }
        if rb != u32::MAX {
            let j = !m.is_clone();
            x.rv.push([0, 1, 2, 3].map(|k| rb + ring_idx(b.me(), b.opp(), m.to, j, k)));
        }
        if sm != 0 {
            let tu = m.to as usize;
            if (dest_ok >> tu) & 1 == 0 {
                dest_ok |= 1u64 << tu;
                let flips_m = opp & ring[tu];
                let (b1o, b2o) = if flips_m == 0 {
                    (b1_opp0, b2_opp0)
                } else {
                    let b1 = box1(opp & !flips_m);
                    (b1, box1(b1))
                };
                let mut pt = p_base;
                planes_add(&mut pt, ring[tu]);
                let mut fm = flips_m;
                while fm != 0 {
                    let u = fm.trailing_zeros() as usize;
                    fm &= fm - 1;
                    planes_add(&mut pt, ring[u]);
                }
                dest_p[tu] = pt;
                dest_b1[tu] = b1o;
                dest_b2[tu] = b2o;
            }
            let clone = m.is_clone();
            let flips = (opp & ring[tu]).count_ones() as usize;
            let my_diff = clone as usize + 2 * flips;
            let tbit = 1u64 << tu;
            let oi = if clone {
                let ea = empty & !tbit;
                opp_state(dest_b1[tu] & ea, dest_b2[tu] & ea, &dest_p[tu])
            } else {
                let mut pa = dest_p[tu];
                planes_sub(&mut pa, ring[m.from as usize]);
                let ea = (empty & !tbit) | (1u64 << m.from);
                opp_state(dest_b1[tu] & ea, dest_b2[tu] & ea, &pa)
            };
            x.sv.push((NSCAL + see_prefix + my_diff * 19 + oi) as u16);
        }
        if s.kind == 0 {
            if m.to == s.to && (m.is_clone() || m.from == s.from) && m.is_clone() == (Move { from: s.from, to: s.to }).is_clone() {
                label = Some(i);
            }
        } else {
            let key = if s.kind == 2 {
                if !m.is_clone() && m.from == s.sel { Some(m.to) } else { None }
            } else if m.is_clone() {
                Some(m.to)
            } else {
                Some(m.from)
            };
            match key {
                None => x.kg.push(-1),
                Some(k) => {
                    let gi = match x.keys.iter().position(|&q| q == k) {
                        Some(p) => p,
                        None => {
                            x.keys.push(k);
                            let v = kpool[s.ko as usize..s.ko as usize + s.kn as usize].iter().find(|e| e.0 == k).map(|e| e.1).unwrap_or(0);
                            x.ky.push(v as f32);
                            x.keys.len() - 1
                        }
                    };
                    x.kg.push(gi as i8);
                }
            }
        }
    }
    if s.kind == 0 {
        return label;
    }
    let tot: f32 = x.ky.iter().sum();
    if x.keys.len() < 2 || tot <= 0.0 {
        return None;
    }
    for y in x.ky.iter_mut() {
        *y /= tot;
    }
    Some(usize::MAX)
}

/// D523: per-thread gradient buffers (dense; the k3c tables are 177k floats, cheap per thread)
struct Grad {
    gb: Vec<f32>,
    gg: Vec<f32>,
    gtd: Vec<f32>,
    gto: Vec<f32>,
    touched: Vec<u32>,
    touched_o: Vec<u32>,
    sc: Scratch,
    z: Vec<f32>,
}

fn new_scratch() -> Scratch {
    Scratch { ml: MoveList::new(), mv: vec![], pv: vec![], rv: vec![], sv: vec![], kg: vec![], ky: vec![], keys: vec![], z: vec![], dz: vec![], qg: vec![], off: [0; 4] }
}

#[allow(clippy::too_many_arguments)]
fn grad_one(s: &S, kpool: &[(u8, u16)], ring: &[u64; 49], k3c: bool, use_pat: bool, prm: (&[f32], &[f32], &[f32], &[f32]), w: (f32, f32), inv_n: f32, gr: &mut Grad) {
    let (bias, g, td, tor) = prm;
    let Some(lab) = expand(s, kpool, ring, k3c, use_pat, &mut gr.sc) else { return };
    let [ob, og, otd, oto] = gr.sc.off;
    let x = &mut gr.sc;
    let z = &mut gr.z;
    z.clear();
    for (i, m) in x.mv.iter().enumerate() {
        let mut v = bias[ob + m[0] as usize] + g[og + m[1] as usize] + g[og + m[2] as usize];
        if m[3] != 255 {
            v += g[og + m[3] as usize];
        }
        if let Some(&si) = x.sv.get(i) {
            v += g[og + si as usize];
        }
        if !td.is_empty() {
            let [d, o] = x.pv[i];
            v += td[otd + d as usize];
            if o != NONE {
                v += tor[oto + o as usize];
            }
        }
        if let Some(rr) = x.rv.get(i) {
            for &q in rr {
                v += tor[oto + q as usize];
            }
        }
        z.push(v);
    }
    x.dz.clear();
    if s.kind == 0 {
        let mx = z.iter().cloned().fold(f32::MIN, f32::max);
        let den: f32 = z.iter().map(|v| (v - mx).exp()).sum();
        for (i, v) in z.iter().enumerate() {
            x.dz.push(w.0 * s.aw * ((v - mx).exp() / den - (i == lab) as u32 as f32));
        }
    } else {
        let mx = z.iter().zip(&x.kg).filter(|(_, k)| **k >= 0).map(|(v, _)| *v).fold(f32::MIN, f32::max);
        x.qg.clear();
        x.qg.resize(x.ky.len(), 0.0);
        let mut den = 0f32;
        for (v, k) in z.iter().zip(&x.kg) {
            if *k >= 0 {
                let e = (v - mx).exp();
                x.qg[*k as usize] += e;
                den += e;
            }
        }
        for (v, k) in z.iter().zip(&x.kg) {
            if *k < 0 {
                x.dz.push(0.0);
            } else {
                let q = (v - mx).exp() / den;
                let gq = (x.qg[*k as usize] / den).max(1e-9);
                x.dz.push(w.1 * (q - x.ky[*k as usize] * q / gq));
            }
        }
    }
    for (i, m) in x.mv.iter().enumerate() {
        let d = x.dz[i] * inv_n;
        if d == 0.0 {
            continue;
        }
        gr.gb[ob + m[0] as usize] += d;
        gr.gg[og + m[1] as usize] += d;
        gr.gg[og + m[2] as usize] += d;
        if m[3] != 255 {
            gr.gg[og + m[3] as usize] += d;
        }
        if let Some(&si) = x.sv.get(i) {
            gr.gg[og + si as usize] += d;
        }
        if use_pat {
            let [pd, po] = x.pv[i];
            gr.touched.push(otd as u32 + pd);
            gr.gtd[otd + pd as usize] += d;
            if po != NONE {
                gr.touched_o.push(oto as u32 + po);
                gr.gto[oto + po as usize] += d;
            }
        }
        if let Some(rr) = x.rv.get(i) {
            for &q in rr {
                gr.touched_o.push(oto as u32 + q);
                gr.gto[oto + q as usize] += d;
            }
        }
    }
}

/// D528: decode one Kata tdata npz into compact samples (+ their visit entries). Returns (stage0, stage1) counts.
fn load_kata_file(path: &str, out: &mut Vec<S>, kpool: &mut Vec<(u8, u16)>, val: bool, bb: &mut Vec<u8>, pb: &mut Vec<u8>) -> Option<(usize, usize)> {
    let z = crate::data::Npz::open(path).ok()?;
    let (Ok(mut cb), Ok(mut cp)) = (z.col("binaryInputNCHWPacked"), z.col("policyTargetsNCMove")) else { return None };
    if cb.width * cb.elem != 154 || cp.width * cp.elem != 200 {
        return None;
    }
    let n = cb.rows;
    if cb.read(n, bb).is_err() || cp.read(n, pb).is_err() {
        return None;
    }
    let (mut n0, mut n1) = (0usize, 0usize);
    for r in 0..n {
        let row = &bb[r * 154..(r + 1) * 154];
        let plane = |c: usize| -> u64 {
            let mut x = 0u64;
            for i in 0..49 {
                if row[c * 7 + i / 8] >> (7 - i % 8) & 1 == 1 {
                    x |= 1u64 << i;
                }
            }
            x
        };
        let (own, opp, sel) = (plane(1), plane(2), plane(4));
        let ko = kpool.len() as u32;
        for i in 0..49 {
            let v = i16::from_le_bytes([pb[r * 200 + 2 * i], pb[r * 200 + 2 * i + 1]]);
            if v > 0 {
                kpool.push((i as u8, v as u16));
            }
        }
        let kn = kpool.len() as u32 - ko;
        if kn == 0 {
            continue;
        }
        let st1 = sel != 0;
        out.push(S { own, opp, from: 0, to: 0, kind: if st1 { 2 } else { 1 }, sel: if st1 { sel.trailing_zeros() as u8 } else { 255 }, kn: kn as u8, val, ko, aw: 1.0, up: 0 });
        if st1 {
            n1 += 1;
        } else {
            n0 += 1;
        }
    }
    Some((n0, n1))
}

/// D535: crash-safe write (temp file + rename); a crash or power cut mid-write never corrupts the previous file
fn write_atomic(path: &str, bytes: &[u8]) -> std::io::Result<()> {
    let tmp = format!("{path}.tmp");
    fs::write(&tmp, bytes)?;
    fs::rename(&tmp, path)
}

fn head_bytes(k3c: bool, use_pat: bool, parts: [&[f32]; 4]) -> Vec<u8> {
    if let Some(&[nb, sl, gl, tdl, tol]) = PB.get() {
        if nb > 1 {
            // D1320 TPOL6: magic, u32 dim 0, u32 nb, then per bucket [bias, g, td, tor] (the engine's policy.rs)
            let mut o = b"TPOL6".to_vec();
            o.extend_from_slice(&[0, 0, 0]);
            o.extend_from_slice(&0u32.to_le_bytes());
            o.extend_from_slice(&(nb as u32).to_le_bytes());
            for k in 0..nb {
                for (part, n) in parts.iter().zip([sl, gl, tdl, tol]) {
                    for v in &part[k * n..(k + 1) * n] {
                        o.extend_from_slice(&v.to_le_bytes());
                    }
                }
            }
            return o;
        }
    }
    let ring = RING_BASE.load(std::sync::atomic::Ordering::Relaxed) != u32::MAX;
    let see = SEE_MODE.load(std::sync::atomic::Ordering::Relaxed) != 0;
    let mut o = if see {
        b"TPOL5\0\0\0".to_vec()
    } else if ring {
        b"TPOL4\0\0\0".to_vec()
    } else if k3c {
        b"TPOL3\0\0\0".to_vec()
    } else if use_pat {
        b"TPOL2\0\0\0".to_vec()
    } else {
        b"TPOL1\0\0\0".to_vec()
    };
    o.extend_from_slice(&0u32.to_le_bytes());
    for part in parts {
        for v in part {
            o.extend_from_slice(&v.to_le_bytes());
        }
    }
    o
}

/// checkpoint: "PCK1", u32 next epoch, i32 adam step, f64 best score, u32 best epoch, u32 bad epochs, then 16 f32 vectors
/// (each u64 length + data): bias g td tor | mb vb mg vg mtd vtd mto vto | best bias g td tor
fn ckpt_bytes(hdr: (u32, i32, f64, u32, u32), vs: [&[f32]; 16]) -> Vec<u8> {
    let mut o = b"PCK1".to_vec();
    o.extend_from_slice(&hdr.0.to_le_bytes());
    o.extend_from_slice(&hdr.1.to_le_bytes());
    o.extend_from_slice(&hdr.2.to_le_bytes());
    o.extend_from_slice(&hdr.3.to_le_bytes());
    o.extend_from_slice(&hdr.4.to_le_bytes());
    for v in vs {
        o.extend_from_slice(&(v.len() as u64).to_le_bytes());
        for x in v {
            o.extend_from_slice(&x.to_le_bytes());
        }
    }
    o
}

#[allow(clippy::type_complexity)]
fn ckpt_read(path: &str) -> Option<((u32, i32, f64, u32, u32), Vec<Vec<f32>>)> {
    let d = fs::read(path).ok()?;
    if d.len() < 28 || &d[0..4] != b"PCK1" {
        return None;
    }
    let u32_at = |o: usize| u32::from_le_bytes(d[o..o + 4].try_into().unwrap());
    let hdr = (u32_at(4), i32::from_le_bytes(d[8..12].try_into().unwrap()), f64::from_le_bytes(d[12..20].try_into().unwrap()), u32_at(20), u32_at(24));
    let mut off = 28;
    let mut vs = Vec::new();
    for _ in 0..16 {
        let n = u64::from_le_bytes(d.get(off..off + 8)?.try_into().unwrap()) as usize;
        off += 8;
        let b = d.get(off..off + 4 * n)?;
        vs.push(b.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect());
        off += 4 * n;
    }
    Some((hdr, vs))
}

pub fn run(args: &[String]) -> ExitCode {
    let Some(net_path) = arg(args, "--net") else {
        eprintln!("usage: ti policy2 --net NET.tnn --games a.tgam|a.jsonl[,..] [--kata DIR[,DIR]] --pat k3c [--init POL.bin] --out-bin OUT.bin [--lr 0.003] [--pat-lr 0.01] [--kata-w 1] [--games-w 1] [--patience 4] [--max-samples N]");
        return ExitCode::from(2);
    };
    let _ = net_path; // dim 0: the head reads no accumulator (the net only fixes which head the file belongs to)
    let files: Vec<String> = arg(args, "--games").unwrap_or_default().split(',').filter(|s| !s.is_empty()).map(String::from).collect();
    let kata_dirs: Vec<String> = arg(args, "--kata").unwrap_or_default().split(',').filter(|s| !s.is_empty()).map(String::from).collect();
    let pat_kind = arg(args, "--pat").unwrap_or_else(|| "k3c".into());
    let use_pat = pat_kind == "k3" || pat_kind == "k3c";
    let k3c = pat_kind == "k3c";
    let (nd_slots, no_slots) = if k3c { (18usize, 9usize) } else { (SLOTS, 49usize) };
    // LAWS C15 (user D868): no epoch limit of any size; the run stops only by saturation (--patience epochs without a gain)
    if args.iter().any(|a| a == "--max-epochs") {
        eprintln!("--max-epochs is forbidden (LAWS C15, user D868: no epoch limit; every run trains to saturation)");
        return ExitCode::from(2);
    }
    let lr0: f32 = arg(args, "--lr").and_then(|v| v.parse().ok()).unwrap_or(0.003);
    let pat_lr: f32 = arg(args, "--pat-lr").and_then(|v| v.parse().ok()).unwrap_or(0.01);
    let kata_w: f32 = arg(args, "--kata-w").and_then(|v| v.parse().ok()).unwrap_or(1.0);
    let games_w: f32 = arg(args, "--games-w").and_then(|v| v.parse().ok()).unwrap_or(1.0);
    // C15 saturation: stop after `patience` consecutive epochs without a val gain (default 4 = "4 more epochs change nothing")
    let patience: usize = arg(args, "--patience").and_then(|v| v.parse().ok()).unwrap_or(4).max(1);
    let max_samples: usize = arg(args, "--max-samples").and_then(|v| v.parse().ok()).unwrap_or(usize::MAX);
    // D528: default 6 of the 8 hardware threads (the PC stays usable); Kata TRAIN files are streamed, never all in RAM
    let threads: usize = arg(args, "--threads").and_then(|v| v.parse().ok()).unwrap_or(6).max(1);
    // D537: --ring adds the jump-ring tables (TPOL4 = TPOL3 + ring [49][2][4][81]); needs --pat k3c
    let use_ring = args.iter().any(|a| a == "--ring");
    let see_arg = arg(args, "--see").unwrap_or_else(|| "none".into());
    // LAWS A12 (D930): only train a head main can load AND plays as trained: TPOL3 = --pat k3c, no ring, no SEE input
    // (main 0120fd12 reads TPOL3/TPOL5 only and has no SEE term at all; a SEE / ring / k3 / table-less head is either
    // refused on load or played with an input it never trained on, D891).
    let exp = crate::genguard::experiment(args);
    crate::genguard::announce("policy2", &exp);
    let out_bin_path = arg(args, "--out-bin").unwrap_or_default();
    if let Err(e) = crate::genguard::check_inputs("policy2", &exp, files.iter().map(String::as_str))
        .and_then(|_| crate::genguard::check_output("policy2", &exp, &out_bin_path))
    {
        eprintln!("{e}");
        return ExitCode::from(2);
    }
    let st = crate::genguard::standard();
    if exp.is_none() && (pat_kind != st.policy_pat || use_ring != st.policy_ring || see_arg != st.policy_see) {
        eprintln!("policy2: refused on the standard path (LAWS A12, D930/D932): the standard head is --pat {} ring {} --see {} ({}); got --pat {pat_kind} ring {use_ring} --see {see_arg}. Try it as --experiment NAME; a winner becomes the standard by changing data/STANDARD.txt on main.",
            st.policy_pat, st.policy_ring, st.policy_see, st.source);
        return ExitCode::from(2);
    }
    let see_mode: u8 = match see_arg.as_str() {
        "none" => 0,
        "2d" => 1,
        "2dp" => 2,
        "2dpt" => 3,
        "2d8" => 4,
        other => {
            eprintln!("bad --see {other} (expected none|2d|2dp|2dpt|2d8)");
            return ExitCode::from(2);
        }
    };
    let see_prior: f32 = arg(args, "--see-prior").and_then(|v| v.parse().ok()).unwrap_or(0.0);
    if see_mode != 0 && (use_ring || !k3c) {
        eprintln!("--see needs --pat k3c and cannot combine with --ring");
        return ExitCode::from(2);
    }
    SEE_MODE.store(see_mode, std::sync::atomic::Ordering::Relaxed);
    let nsee = see_n(see_mode);
    let kata_val_max: usize = arg(args, "--kata-val-max").and_then(|v| v.parse().ok()).unwrap_or(400_000);
    let ring = ring1();

    // Advantage-weighted game labels (D765, off by default): --adv-lambda L turns it on.
    let adv_lambda: Option<f32> = arg(args, "--adv-lambda").and_then(|v| v.parse().ok());
    let adv_scale: f32 = arg(args, "--adv-scale").and_then(|v| v.parse().ok()).unwrap_or(0.10);
    let adv_clip: f32 = arg(args, "--adv-clip").and_then(|v| v.parse().ok()).unwrap_or(1.0);
    let (mut adv_n, mut adv_clipped, mut adv_abs, mut adv_sum) = (0u64, 0u64, 0f64, 0f64);

    let mut ss: Vec<S> = Vec::new();
    let mut kpool: Vec<(u8, u16)> = Vec::new();
    // games: every played move is a hard label; val = game pair % 10 == 0
    let mut ml = MoveList::new();
    for f in &files {
        // D930: binary self-play / gate games (.tgam + .tbst, what ti gen and ti gate write). The label is the engine's
        // BEST move of each position (.tbst), never a novelty move that was only played to explore; the game goes on
        // with the move that was played. A .tgam without its .tbst is refused. Validation = DAG game id % 10 == 0.
        if f.ends_with(".tgam") {
            let tb_path = format!("{}.tbst", f.trim_end_matches(".tgam"));
            if !std::path::Path::new(&tb_path).exists() {
                eprintln!("policy2: {f}: no {tb_path}: best-move labels missing, refusing (D925/D930)");
                return ExitCode::from(2);
            }
            let (mut rd, mut tb) = match (crate::gamedata::GameReader::open(f), crate::gamedata::TbstReader::open(f)) {
                (Ok(r), Ok(t)) => (r, t),
                (Err(e), _) | (_, Err(e)) => {
                    eprintln!("policy2: {e}");
                    return ExitCode::from(2);
                }
            };
            let mut got = 0usize;
            for (gi, g) in rd.by_ref().enumerate() {
                let Ok(g) = g else {
                    eprintln!("policy2: {f}: unreadable game {gi}, refusing");
                    return ExitCode::from(2);
                };
                let Ok((gid, labels)) = tb.next_game(g.plies.len()) else {
                    eprintln!("policy2: {tb_path}: labels end early at game {gi}, refusing");
                    return ExitCode::from(2);
                };
                let val = gid % 10 == 0;
                let mut b = g.start();
                for (pi, ply) in g.plies.iter().enumerate() {
                    let code = ply.0 & 0x3ff;
                    if code == titanium::ttcodec::PASS_CODE {
                        b = b.make_uai(Move::PASS);
                        continue;
                    }
                    let Some(played) = titanium::ttcodec::decode_move(&b, code) else { break };
                    let best = labels.get(pi).and_then(|&(lc, _)| titanium::ttcodec::decode_move(&b, lc & 0x3ff));
                    b.legal_moves_dedup_into(&mut ml);
                    if let Some(bm) = best.filter(|m| !m.is_pass()) {
                        if ml.len() >= 2 {
                            ss.push(S { own: b.me(), opp: b.opp(), from: bm.from, to: bm.to, kind: 0, sel: 255, kn: 0, val, ko: 0, aw: 1.0, up: 0 });
                            got += 1;
                        }
                    }
                    b = b.make(played);
                }
            }
            eprintln!("{f}: {got} positions (label = engine best move from {tb_path})");
            continue;
        }
        let Ok(fh) = fs::File::open(f) else {
            eprintln!("{f}: cannot open");
            continue;
        };
        let mut got = 0usize;
        for line in BufReader::new(fh).lines().map_while(Result::ok) {
            let Ok(g) = games::parse_game(&line) else { continue };
            let Ok(boards) = games::replay(&g) else { continue };
            let val = g.pair % 10 == 0;
            // Always computed (the validation report splits by it); it changes the loss only with --adv-lambda.
            let adv: Vec<Option<f32>> = move_advantages(&g, &boards, adv_lambda.unwrap_or(0.9));
            for (t, (r, b)) in g.moves.iter().zip(&boards).enumerate() {
                if r.mv == "FORFEIT" {
                    break;
                }
                let pm = match b.parse_uai_move(r.label_mv()) {
                    Some(m) if !m.is_pass() => m,
                    _ => continue,
                };
                b.legal_moves_dedup_into(&mut ml);
                if ml.len() < 2 {
                    continue;
                }
                // A move after which the game went better than its own score promised gets more weight, a move
                // that was followed by a drop gets less: weight = exp(clip(advantage / scale)).
                let a_t = adv.get(t).copied().flatten();
                let up = match a_t {
                    Some(a) if a > 0.02 => 1u8,
                    Some(a) if a < -0.02 => 2u8,
                    _ => 0u8,
                };
                let aw = match a_t.filter(|_| adv_lambda.is_some()) {
                    Some(a) => {
                        adv_n += 1;
                        adv_abs += a.abs() as f64;
                        adv_sum += a as f64;
                        let x = a / adv_scale;
                        if x.abs() >= adv_clip {
                            adv_clipped += 1;
                        }
                        x.clamp(-adv_clip, adv_clip).exp()
                    }
                    None => 1.0,
                };
                ss.push(S { own: b.me(), opp: b.opp(), from: pm.from, to: pm.to, kind: 0, sel: 255, kn: 0, val, ko: 0, aw, up });
                got += 1;
            }
        }
        eprintln!("{f}: {got} positions");
    }
    if let (Some(l), true) = (adv_lambda, adv_n > 0) {
        // Scale to mean 1 so `--games-w` keeps its meaning against the Kata rows.
        let mean = ss.iter().map(|s| s.aw as f64).sum::<f64>() / ss.len() as f64;
        let (mut lo, mut hi) = (f32::MAX, f32::MIN);
        for s in ss.iter_mut() {
            s.aw = (s.aw as f64 / mean) as f32;
            lo = lo.min(s.aw);
            hi = hi.max(s.aw);
        }
        eprintln!(
            "advantage weights (--adv-lambda {l}, scale {adv_scale}, clip {adv_clip}): {adv_n} of {} game moves weighted; advantage mean {:+.4}, mean |.| {:.4}; at the clip {:.1}%; weight {lo:.2} .. {hi:.2} after scaling to mean 1",
            ss.len(),
            adv_sum / adv_n as f64,
            adv_abs / adv_n as f64,
            100.0 * adv_clipped as f64 / adv_n as f64
        );
    }
    let n_games = ss.len();
    // Kata tdata (all npz under the dirs); val = every 10th file
    let mut kf: Vec<std::path::PathBuf> = Vec::new();
    for d in &kata_dirs {
        npz_files(std::path::Path::new(d), &mut kf);
    }
    kf.sort();
    let (mut bb, mut pb) = (Vec::new(), Vec::new());
    let (mut nk0, mut nk1, mut bad) = (0usize, 0usize, 0usize);
    let mut train_files: Vec<String> = Vec::new();
    for (fi, fp) in kf.iter().enumerate() {
        let path = fp.to_string_lossy().to_string();
        if fi % 10 != 0 {
            train_files.push(path);
            continue;
        }
        if nk0 + nk1 >= kata_val_max {
            continue;
        }
        match load_kata_file(&path, &mut ss, &mut kpool, true, &mut bb, &mut pb) {
            Some((a, b)) => {
                nk0 += a;
                nk1 += b;
            }
            None => bad += 1,
        }
    }
    if max_samples < usize::MAX {
        // smoke tests: keep only as many train files as needed for ~max_samples rows (50k rows per file)
        train_files.truncate(max_samples.div_ceil(50_000).max(1));
    }
    println!("kata: {} files: {} train files STREAMED per epoch, val {} rows (stage0 {} stage1 {}) from every 10th file (cap {}), {} bad", kf.len(), train_files.len(), nk0 + nk1, nk0, nk1, kata_val_max, bad);
    let nval = ss.iter().filter(|s| s.val).count();
    println!(
        "in RAM: games {} + kata val {} = {} samples (val {}), kpool {} entries, ~{} MB; train blocks = 1 Kata file + its share of the game rows",
        n_games, nk0 + nk1, ss.len(), nval, kpool.len(),
        (ss.len() * std::mem::size_of::<S>() + kpool.len() * 4) / 1_000_000
    );

    // parameters (dim 0 head)
    let mut bias = vec![0f32; SLOTS];
    let mut g = vec![0f32; NSCAL + nsee];
    if see_prior != 0.0 {
        for blk in 0..(nsee / NSEE_BASE) {
            for md in 0..18usize {
                for oi in 1..19usize {
                    let opp_diff = oi - 1;
                    g[NSCAL + blk * NSEE_BASE + md * 19 + oi] = see_prior * (md as f32 - opp_diff as f32);
                }
            }
        }
    }
    let mut td = vec![0f32; if use_pat { nd_slots * P3 } else { 0 }];
    if use_ring && !k3c {
        eprintln!("--ring needs --pat k3c");
        return ExitCode::from(2);
    }
    let mut tor = vec![0f32; if use_pat { no_slots * P3 } else { 0 } + if use_ring { RING_N } else { 0 }];
    if use_ring {
        RING_BASE.store((no_slots * P3) as u32, std::sync::atomic::Ordering::Relaxed);
    }
    // D1323: a TPOL6 (phase-bucket) head as --init: kept aside, written into the buckets after --buckets sized them
    let mut pending6: Option<(usize, Vec<f32>)> = None;
    if let Some(ip) = arg(args, "--init") {
        let Ok(d) = fs::read(&ip) else {
            eprintln!("{ip}: cannot read");
            return ExitCode::from(1);
        };
        if d.len() >= 16 && &d[0..5] == b"TPOL6" {
            let fnb = u32::from_le_bytes(d[12..16].try_into().unwrap()) as usize;
            let per = SLOTS + NSCAL + 27 * P3;
            if fnb == 0 || d.len() != 16 + 4 * per * fnb || !k3c || !use_pat || nsee > 0 || use_ring {
                eprintln!("--init {ip}: bad TPOL6 file, or this run is not --pat k3c without --see / --ring");
                return ExitCode::from(2);
            }
            pending6 = Some((fnb, d[16..].chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect()));
            println!("init from {ip}: TPOL6 with {fnb} phase buckets");
        } else {
        let idim = u32::from_le_bytes(d[8..12].try_into().unwrap()) as usize;
        let v5 = &d[0..5] == b"TPOL5";
        let v4 = &d[0..5] == b"TPOL4";
        let (v2, v3) = (&d[0..5] == b"TPOL2", &d[0..5] == b"TPOL3" || v4 || v5);
        let ntab = if v4 { 27 * P3 + RING_N } else if v3 { 27 * P3 } else if v2 { (SLOTS + 49) * P3 } else { 0 };
        // D892: the FILE's SEE block size comes from the file (a TPOL5 init into a --see none run drops it: step-0 parity
        // with how the engine plays that file since D751, SEE off); a --see run needs the same block size as the file
        let total = (d.len().saturating_sub(12)) / 4;
        let file_see = if v5 { total.saturating_sub(SLOTS + NSCAL + SLOTS * idim + ntab) } else { 0 };
        let n1 = SLOTS + NSCAL + file_see + SLOTS * idim;
        if !(&d[0..5] == b"TPOL1" || v2 || v3) || d.len() != 12 + 4 * (n1 + ntab) || ((v2 || v3) && v3 != k3c) || (v4 && !use_ring) || (nsee > 0 && file_see != nsee) {
            eprintln!("--init {ip}: bad file or table kind does not match --pat/--see");
            return ExitCode::from(2);
        }
        let f: Vec<f32> = d[12..].chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
        bias.copy_from_slice(&f[..SLOTS]);
        let in_see = file_see.min(nsee);
        g[..NSCAL + in_see].copy_from_slice(&f[SLOTS..SLOTS + NSCAL + in_see]);
        if file_see > in_see {
            println!("init: dropped the file's SEE block ({file_see} floats; this run is --see {see_arg})");
        }
        if (v2 || v3) && use_pat {
            td.copy_from_slice(&f[n1..n1 + nd_slots * P3]);
            let rest = &f[n1 + nd_slots * P3..];
            tor[..rest.len()].copy_from_slice(rest); // a TPOL3 init leaves the ring block at 0 (step-0 parity, C14)
        }
        println!("init from {ip} (file dim {idim}{}, tables {}, see {see_arg})", if idim > 0 { ", U dropped as in the engine" } else { "" }, if v2 || v3 { "loaded" } else { "zero" });
        }
    }

    // D1320: --buckets N copies the (init) head into N phase buckets = exact step-0 parity (C14); TPOL6 export
    let nbk: usize = arg(args, "--buckets").and_then(|v| v.parse().ok()).unwrap_or(1);
    if nbk > 1 {
        if nsee > 0 || use_ring || !k3c || !use_pat {
            eprintln!("--buckets {nbk} needs --pat k3c without --see / --ring (the engine's TPOL6 holds TPOL3-style heads)");
            return ExitCode::from(2);
        }
        let _ = PB.set([nbk, bias.len(), g.len(), td.len(), tor.len()]);
        bias = bias.repeat(nbk);
        g = g.repeat(nbk);
        td = td.repeat(nbk);
        tor = tor.repeat(nbk);
        println!("phase buckets: {nbk} (bucket = empties * {nbk} / 50, as the NNUE); every bucket starts as the init head (step-0 parity)");
        if let Some((fnb, f)) = pending6.take() {
            // run bucket b starts as file bucket b / (nbk / fnb): the same head on the same positions, since
            // floor(e * fnb / 50) == floor(floor(e * nbk / 50) / (nbk / fnb))
            if nbk % fnb != 0 {
                eprintln!("--init TPOL6 with {fnb} buckets cannot seed --buckets {nbk} (needs a whole multiple)");
                return ExitCode::from(2);
            }
            let (sl, gl, tdl, tol) = (bias.len() / nbk, g.len() / nbk, td.len() / nbk, tor.len() / nbk);
            let per = sl + gl + tdl + tol;
            for b in 0..nbk {
                let s = b / (nbk / fnb);
                let blk = &f[s * per..(s + 1) * per];
                bias[b * sl..(b + 1) * sl].copy_from_slice(&blk[..sl]);
                g[b * gl..(b + 1) * gl].copy_from_slice(&blk[sl..sl + gl]);
                td[b * tdl..(b + 1) * tdl].copy_from_slice(&blk[sl + gl..sl + gl + tdl]);
                tor[b * tol..(b + 1) * tol].copy_from_slice(&blk[sl + gl + tdl..per]);
            }
            println!("phase buckets: seeded from the {fnb}-bucket head (bucket b <- b / {}): exact step-0 parity", nbk / fnb);
        }
    }
    if pending6.is_some() {
        eprintln!("--init TPOL6 needs --buckets N (a whole multiple of the file's buckets)");
        return ExitCode::from(2);
    }
    let logits = |x: &Scratch, bias: &[f32], g: &[f32], td: &[f32], tor: &[f32], out: &mut Vec<f32>| {
        out.clear();
        let [ob, og, otd, oto] = x.off;
        for (i, m) in x.mv.iter().enumerate() {
            let mut z = bias[ob + m[0] as usize] + g[og + m[1] as usize] + g[og + m[2] as usize];
            if m[3] != 255 {
                z += g[og + m[3] as usize];
            }
            if let Some(&si) = x.sv.get(i) {
                z += g[og + si as usize];
            }
            if !td.is_empty() {
                let [d, o] = x.pv[i];
                z += td[otd + d as usize];
                if o != NONE {
                    z += tor[oto + o as usize];
                }
            }
            if let Some(rr) = x.rv.get(i) {
                for &q in rr {
                    z += tor[oto + q as usize];
                }
            }
            out.push(z);
        }
    };
    let mut sc = Scratch { ml: MoveList::new(), mv: vec![], pv: vec![], rv: vec![], sv: vec![], kg: vec![], ky: vec![], keys: vec![], z: vec![], dz: vec![], qg: vec![], off: [0; 4] };
    // validation: (game CE, top1, top3, top5, n) and (kata soft CE, top1 most-visited group, n)
    // (top1 hits, n) on validation game moves with a positive advantage, then the same for a negative one
    let split = std::cell::Cell::new([0f64; 4]);
    let evaluate = |bias: &[f32], g: &[f32], td: &[f32], tor: &[f32], sc: &mut Scratch| -> ([f64; 5], [f64; 3]) {
        let (mut gm, mut km) = ([0f64; 5], [0f64; 3]);
        let mut sp = [0f64; 4];
        let mut z = std::mem::take(&mut sc.z);
        for s in ss.iter().filter(|s| s.val) {
            let Some(lab) = expand(s, &kpool, &ring, k3c, use_pat, sc) else { continue };
            logits(sc, bias, g, td, tor, &mut z);
            if s.kind == 0 {
                let mx = z.iter().cloned().fold(f32::MIN, f32::max);
                let den: f32 = z.iter().map(|v| (v - mx).exp()).sum();
                gm[0] += (den.ln() + mx - z[lab]) as f64;
                let rank = z.iter().filter(|v| **v > z[lab]).count();
                gm[1] += (rank == 0) as u32 as f64;
                if s.up == 1 {
                    sp[0] += (rank == 0) as u32 as f64;
                    sp[1] += 1.0;
                } else if s.up == 2 {
                    sp[2] += (rank == 0) as u32 as f64;
                    sp[3] += 1.0;
                }
                gm[2] += (rank < 3) as u32 as f64;
                gm[3] += (rank < 5) as u32 as f64;
                gm[4] += 1.0;
            } else {
                let mx = z.iter().zip(&sc.kg).filter(|(_, k)| **k >= 0).map(|(v, _)| *v).fold(f32::MIN, f32::max);
                let mut qg = vec![0f32; sc.ky.len()];
                let mut den = 0f32;
                for (v, k) in z.iter().zip(&sc.kg) {
                    if *k >= 0 {
                        let e = (v - mx).exp();
                        qg[*k as usize] += e;
                        den += e;
                    }
                }
                for (y, q) in sc.ky.iter().zip(&qg) {
                    if *y > 0.0 {
                        km[0] -= (*y as f64) * ((q / den).max(1e-9) as f64).ln();
                    }
                }
                let am = |v: &[f32]| (0..v.len()).max_by(|&a, &b| v[a].partial_cmp(&v[b]).unwrap()).unwrap();
                km[1] += (am(&qg) == am(&sc.ky)) as u32 as f64;
                km[2] += 1.0;
            }
        }
        sc.z = z;
        for i in 0..4 {
            gm[i] /= gm[4].max(1.0);
        }
        km[0] /= km[2].max(1.0);
        km[1] /= km[2].max(1.0);
        split.set(sp);
        (gm, km)
    };
    let show = |tag: &str, r: &([f64; 5], [f64; 3])| {
        println!(
            "{tag}: games CE {:.4} top1 {:.4} top3 {:.4} top5 {:.4} ({:.0}) | kata soft CE {:.4} top1 {:.4} ({:.0})",
            r.0[0], r.0[1], r.0[2], r.0[3], r.0[4], r.1[0], r.1[1], r.1[2]
        );
        // The same validation moves, split by what followed them in the game (advantage beyond +-0.02).
        let sp = split.get();
        if sp[1] > 0.0 && sp[3] > 0.0 {
            println!(
                "{tag}:   top1 on moves the game then proved RIGHT {:.4} ({:.0}) | on moves it proved WRONG {:.4} ({:.0}) | gap {:+.4}",
                sp[0] / sp[1],
                sp[1],
                sp[2] / sp[3],
                sp[3],
                sp[0] / sp[1] - sp[2] / sp[3]
            );
        }
    };
    let r0 = evaluate(&bias, &g, &td, &tor, &mut sc);
    show("val init", &r0);
    // D528: early-stop criterion: kata = Kata soft CE, games = our-label CE, sum = both (default)
    let stop_on = arg(args, "--stop-on").unwrap_or_else(|| "sum".into());
    let score = |r: &([f64; 5], [f64; 3])| -> f64 {
        let gce = if r.0[4] > 0.0 { r.0[0] } else { 0.0 };
        let kce = if r.1[2] > 0.0 { r.1[0] } else { 0.0 };
        match stop_on.as_str() { "kata" => kce, "games" => gce, _ => gce + kce }
    };
    let mut best = (score(&r0), bias.clone(), g.clone(), td.clone(), tor.clone(), 0usize);

    // Adam (dense on bias/g, lazy sparse on the tables)
    let mut tr: Vec<u32> = (0..ss.len() as u32).filter(|&i| !ss[i as usize].val && ss[i as usize].kind == 0).collect();
    let (b1, b2, eps) = (0.9f32, 0.999f32, 1e-8f32);
    let (mut mb, mut vb, mut mg, mut vg) = (vec![0f32; bias.len()], vec![0f32; bias.len()], vec![0f32; g.len()], vec![0f32; g.len()]);
    let (mut mtd, mut vtd, mut mto, mut vto) = (vec![0f32; td.len()], vec![0f32; td.len()], vec![0f32; tor.len()], vec![0f32; tor.len()]);
    let (mut gtd, mut gto) = (vec![0f32; td.len()], vec![0f32; tor.len()]);
    let (mut touched, mut touched_o): (Vec<u32>, Vec<u32>) = (Vec::new(), Vec::new());
    let mut t = 0i32;
    let bs = 256usize;
    let mut bad_epochs = 0usize;
    let _z: Vec<f32> = Vec::new();
    // D535: crash safety: --out-bin is rewritten (atomically) after every improving epoch, a full checkpoint
    // <out>.ckpt after every epoch, --resume continues from it, <out>.stop ends the run after the current epoch
    let out_bin = arg(args, "--out-bin");
    // LAWS A14 (user D1307): no run without a best snapshot + checkpoint, and a stop file (STOP in the --out-bin folder or
    // C:/gitProjects/STOP_ALL) is checked every 64 batches: the partial epoch is discarded, the last epoch's checkpoint and
    // the best snapshot stay, the same command + --resume restarts that epoch
    let Some(ob) = out_bin.clone() else {
        eprintln!("policy2: --out-bin FILE is required (LAWS A14: best snapshot + <FILE>.ckpt checkpoint)");
        return ExitCode::from(2);
    };
    let run_dir = std::path::Path::new(&ob).parent().map(|d| if d.as_os_str().is_empty() { std::path::PathBuf::from(".") } else { d.to_path_buf() }).unwrap_or_else(|| ".".into());
    if let Some(f) = crate::stopctl::requested(Some(&run_dir)) {
        eprintln!("policy2: stop file {f} exists: remove it to train");
        return ExitCode::from(2);
    }
    let ckpt_path = out_bin.as_ref().map(|p| format!("{p}.ckpt"));
    let stop_path = out_bin.as_ref().map(|p| format!("{p}.stop"));
    let mut ep0 = 0usize;
    if args.iter().any(|a| a == "--resume") {
        match ckpt_path.as_deref().and_then(ckpt_read) {
            Some((h, vs)) if vs[0].len() == bias.len() && vs[1].len() == g.len() && vs[2].len() == td.len() && vs[3].len() == tor.len() => {
                bias.copy_from_slice(&vs[0]);
                g.copy_from_slice(&vs[1]);
                td.copy_from_slice(&vs[2]);
                tor.copy_from_slice(&vs[3]);
                mb.copy_from_slice(&vs[4]);
                vb.copy_from_slice(&vs[5]);
                mg.copy_from_slice(&vs[6]);
                vg.copy_from_slice(&vs[7]);
                mtd.copy_from_slice(&vs[8]);
                vtd.copy_from_slice(&vs[9]);
                mto.copy_from_slice(&vs[10]);
                vto.copy_from_slice(&vs[11]);
                best = (h.2, vs[12].clone(), vs[13].clone(), vs[14].clone(), vs[15].clone(), h.3 as usize);
                ep0 = h.0 as usize;
                t = h.1;
                bad_epochs = h.4 as usize;
                println!("resumed from {} at epoch {} (best epoch {}, score {:.4})", ckpt_path.as_deref().unwrap_or(""), ep0 + 1, best.5, best.0);
            }
            _ => println!("--resume: no usable checkpoint, starting fresh"),
        }
    }
    if ep0 == 0 {
        // a best snapshot exists from the first second on (the start weights), however short the run (LAWS A14)
        if let Err(e) = write_atomic(&ob, &head_bytes(k3c, use_pat, [&best.1, &best.2, &best.3, &best.4])) {
            eprintln!("policy2: {ob}: {e}");
            return ExitCode::from(1);
        }
    }
    let mut abort: Option<String> = None;
    for ep in ep0.. {
        let lr = lr0 * (0.85f32).powi(ep as i32);
        let mut x: u64 = 0x9E37_79B9_7F4A_7C15 ^ (ep as u64 + 1).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        for i in (1..tr.len()).rev() {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            tr.swap(i, (x % (i as u64 + 1)) as usize);
        }
        let t0 = std::time::Instant::now();
        let (mut gb, mut gg) = (vec![0f32; bias.len()], vec![0f32; g.len()]);
        let mut forder: Vec<usize> = (0..train_files.len()).collect();
        for i in (1..forder.len()).rev() {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            forder.swap(i, (x % (i as u64 + 1)) as usize);
        }
        let nblocks = train_files.len().max(1);
        let gper = tr.len().div_ceil(nblocks);
        let mut rows_seen = 0usize;
        std::thread::scope(|ps| {
        // D528 producer: decodes the next Kata file while the workers train on the current block (2 blocks max in RAM)
        let (txb, rxb) = std::sync::mpsc::sync_channel::<(Vec<S>, Vec<(u8, u16)>)>(2);
        {
            let (tr, ss, train_files, forder) = (&tr, &ss, &train_files, forder);
            ps.spawn(move || {
                let (mut bb, mut pb) = (Vec::new(), Vec::new());
                for bi in 0..nblocks {
                    let (mut bss, mut bkp) = (Vec::new(), Vec::new());
                    if let Some(&fi) = forder.get(bi) {
                        let _ = load_kata_file(&train_files[fi], &mut bss, &mut bkp, false, &mut bb, &mut pb);
                    }
                    for &gi in &tr[(bi * gper).min(tr.len())..((bi + 1) * gper).min(tr.len())] {
                        bss.push(ss[gi as usize]);
                    }
                    if txb.send((bss, bkp)).is_err() {
                        break;
                    }
                }
            });
        }
        for (bss, bkp) in rxb {
        let mut btr: Vec<u32> = (0..bss.len() as u32).collect();
        for i in (1..btr.len()).rev() {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            btr.swap(i, (x % (i as u64 + 1)) as usize);
        }
        rows_seen += btr.len();
        // D526: PERSISTENT workers (one spawn per epoch, two barriers per batch) instead of 8 spawns per batch.
        // Workers read the parameters only between `start` and `done`; the main thread updates them only outside.
        let grs: Vec<std::sync::Mutex<Grad>> = (0..threads)
            .map(|_| std::sync::Mutex::new(Grad { gb: vec![0f32; bias.len()], gg: vec![0f32; g.len()], gtd: vec![0f32; td.len()], gto: vec![0f32; tor.len()], touched: vec![], touched_o: vec![], sc: new_scratch(), z: vec![] }))
            .collect();
        let bt = bs * threads;
        let nb = btr.len().div_ceil(bt);
        let (start_b, done_b) = (std::sync::Barrier::new(threads + 1), std::sync::Barrier::new(threads + 1));
        let cur = std::sync::atomic::AtomicUsize::new(0);
        struct Ptr(*mut f32, usize);
        unsafe impl Send for Ptr {}
        unsafe impl Sync for Ptr {}
        let pp = [Ptr(bias.as_mut_ptr(), bias.len()), Ptr(g.as_mut_ptr(), g.len()), Ptr(td.as_mut_ptr(), td.len()), Ptr(tor.as_mut_ptr(), tor.len())];
        let trr = &btr[..];
        std::thread::scope(|scp| {
            for ti in 0..threads {
                let (grs, start_b, done_b, cur, pp) = (&grs, &start_b, &done_b, &cur, &pp);
                let (kp, rg, ssr) = (&bkp[..], &ring, &bss[..]);
                scp.spawn(move || loop {
                    start_b.wait();
                    let b = cur.load(std::sync::atomic::Ordering::Acquire);
                    if b == usize::MAX {
                        break;
                    }
                    let chunk = &trr[b * bt..((b + 1) * bt).min(trr.len())];
                    let inv_n = 1.0 / chunk.len() as f32;
                    let per = chunk.len().div_ceil(threads);
                    let part = &chunk[(ti * per).min(chunk.len())..((ti + 1) * per).min(chunk.len())];
                    // SAFETY: between the barriers nobody writes the parameters
                    let prm = unsafe {
                        (
                            std::slice::from_raw_parts(pp[0].0 as *const f32, pp[0].1),
                            std::slice::from_raw_parts(pp[1].0 as *const f32, pp[1].1),
                            std::slice::from_raw_parts(pp[2].0 as *const f32, pp[2].1),
                            std::slice::from_raw_parts(pp[3].0 as *const f32, pp[3].1),
                        )
                    };
                    {
                        let mut gr = grs[ti].lock().unwrap();
                        for &si in part {
                            grad_one(&ssr[si as usize], kp, rg, k3c, use_pat, prm, (games_w, kata_w), inv_n, &mut gr);
                        }
                    }
                    done_b.wait();
                });
            }
            for bi in 0..nb {
                if bi % 64 == 0 {
                    if let Some(f) = crate::stopctl::requested(Some(&run_dir)) {
                        abort = Some(f);
                        break;
                    }
                }
                gb.iter_mut().for_each(|v| *v = 0.0);
                gg.iter_mut().for_each(|v| *v = 0.0);
                cur.store(bi, std::sync::atomic::Ordering::Release);
                start_b.wait();
                done_b.wait();
                for grm in grs.iter() {
                    let mut gr = grm.lock().unwrap();
                    let gr = &mut *gr;
                    for (a, b) in gb.iter_mut().zip(gr.gb.iter_mut()) {
                        *a += *b;
                        *b = 0.0;
                    }
                    for (a, b) in gg.iter_mut().zip(gr.gg.iter_mut()) {
                        *a += *b;
                        *b = 0.0;
                    }
                    for &k in &gr.touched {
                        let k = k as usize;
                        gtd[k] += gr.gtd[k];
                        gr.gtd[k] = 0.0;
                    }
                    for &k in &gr.touched_o {
                        let k = k as usize;
                        gto[k] += gr.gto[k];
                        gr.gto[k] = 0.0;
                    }
                    touched.append(&mut gr.touched);
                    touched_o.append(&mut gr.touched_o);
                }
                t += 1;
                let c1 = 1.0 - b1.powi(t);
                let c2 = 1.0 - b2.powi(t);
                // SAFETY: workers are parked at `start_b`; the main thread is the only accessor now
                let (pbias, pg, ptd, ptor) = unsafe {
                    (
                        std::slice::from_raw_parts_mut(pp[0].0, pp[0].1),
                        std::slice::from_raw_parts_mut(pp[1].0, pp[1].1),
                        std::slice::from_raw_parts_mut(pp[2].0, pp[2].1),
                        std::slice::from_raw_parts_mut(pp[3].0, pp[3].1),
                    )
                };
                for (p, gr, m, v) in [(pbias, &gb, &mut mb, &mut vb), (pg, &gg, &mut mg, &mut vg)] {
                    for i in 0..p.len() {
                        m[i] = b1 * m[i] + (1.0 - b1) * gr[i];
                        v[i] = b2 * v[i] + (1.0 - b2) * gr[i] * gr[i];
                        p[i] -= lr * (m[i] / c1) / ((v[i] / c2).sqrt() + eps);
                    }
                }
                if use_pat {
                    for (tab, gr, m, v, list) in [(ptd, &mut gtd, &mut mtd, &mut vtd, &mut touched), (ptor, &mut gto, &mut mto, &mut vto, &mut touched_o)] {
                        list.sort_unstable();
                        list.dedup();
                        for &k in list.iter() {
                            let k = k as usize;
                            m[k] = b1 * m[k] + (1.0 - b1) * gr[k];
                            v[k] = b2 * v[k] + (1.0 - b2) * gr[k] * gr[k];
                            tab[k] -= pat_lr * (lr / lr0) * (m[k] / c1) / ((v[k] / c2).sqrt() + eps);
                            gr[k] = 0.0;
                        }
                        list.clear();
                    }
                }
                if bi % 2000 == 1999 {
                    eprint!(".");
                }
            }
            cur.store(usize::MAX, std::sync::atomic::Ordering::Release);
            start_b.wait();
        });
        if abort.is_some() {
            break;
        }
        }
        });
        if let Some(f) = &abort {
            println!(
                "STOPPED by {f} during epoch {}: that partial epoch is discarded; kept: checkpoint {} (end of epoch {ep}) and best snapshot {ob} (epoch {}); the same command + --resume continues",
                ep + 1,
                ckpt_path.as_deref().unwrap_or(""),
                best.5
            );
            return ExitCode::SUCCESS;
        }
        eprintln!(" {} rows trained this epoch", rows_seen);
        let r = evaluate(&bias, &g, &td, &tor, &mut sc);
        show(&format!("epoch {} lr {:.5} ({:.0} s)", ep + 1, lr, t0.elapsed().as_secs_f64()), &r);
        let sc_now = score(&r);
        let mut stop = false;
        if sc_now < best.0 - 0.0005 {
            best = (sc_now, bias.clone(), g.clone(), td.clone(), tor.clone(), ep + 1);
            bad_epochs = 0;
            if let Some(p) = &out_bin {
                match write_atomic(p, &head_bytes(k3c, use_pat, [&best.1, &best.2, &best.3, &best.4])) {
                    Ok(()) => println!("  saved best (epoch {}) -> {p}", ep + 1),
                    Err(e) => eprintln!("  {p}: {e}"),
                }
            }
        } else {
            bad_epochs += 1;
            if bad_epochs >= patience {
                println!("early stop: no val improvement for {} epochs (best epoch {})", bad_epochs, best.5);
                stop = true;
            }
        }
        if let Some(cp) = &ckpt_path {
            let hdr = ((ep + 1) as u32, t, best.0, best.5 as u32, bad_epochs as u32);
            let vs: [&[f32]; 16] = [&bias, &g, &td, &tor, &mb, &vb, &mg, &vg, &mtd, &vtd, &mto, &vto, &best.1, &best.2, &best.3, &best.4];
            if let Err(e) = write_atomic(cp, &ckpt_bytes(hdr, vs)) {
                eprintln!("  {cp}: {e}");
            }
        }
        if stop_path.as_deref().map(|p| std::path::Path::new(p).exists()).unwrap_or(false) {
            println!("stop file found: ending after epoch {}", ep + 1);
            stop = true;
        }
        if stop {
            break;
        }
    }
    println!("best epoch {} (val score {:.4}, --stop-on {})", best.5, best.0, stop_on);
    if let Some(p) = &out_bin {
        let o = head_bytes(k3c, use_pat, [&best.1, &best.2, &best.3, &best.4]);
        match write_atomic(p, &o) {
            Ok(()) => println!("wrote {p} ({} B, best epoch {})", o.len(), best.5),
            Err(e) => {
                eprintln!("{p}: {e}");
                return ExitCode::from(1);
            }
        }
    }
    let _ = k3code; // shared definition lives in policy.rs
    ExitCode::SUCCESS
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mvv2_see_bitplanes_match_bruteforce_2ply() {
        let ring = ring1();
        let mut x: u64 = 0x1234_5678_9ABC_DEF0;
        let mut ml = MoveList::new();
        let mut ml2 = MoveList::new();
        for _ in 0..2000 {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            let occ = x & FULL49;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            let me = occ & x;
            let opp = occ & !me;
            if me == 0 || opp == 0 || (me | opp) == FULL49 {
                continue;
            }
            let b = Board::from_occ(me, opp, 0, 0);
            b.legal_moves_dedup_into(&mut ml);
            if ml.is_empty() {
                continue;
            }
            let pb = victim_planes(me);
            let b1_0 = box1(opp);
            let b2_0 = box1(b1_0);
            let mut dest_ok = 0u64;
            let mut dest_p = [[0u64; 4]; 49];
            let mut dest_b1 = [0u64; 49];
            let mut dest_b2 = [0u64; 49];
            let empty = b.empty();
            for m in ml.iter() {
                let tu = m.to as usize;
                if (dest_ok >> tu) & 1 == 0 {
                    dest_ok |= 1u64 << tu;
                    let flips_m = opp & ring[tu];
                    let (b1o, b2o) = if flips_m == 0 {
                        (b1_0, b2_0)
                    } else {
                        let b1 = box1(opp & !flips_m);
                        (b1, box1(b1))
                    };
                    let mut pt = pb;
                    planes_add(&mut pt, ring[tu]);
                    let mut fm = flips_m;
                    while fm != 0 {
                        let u = fm.trailing_zeros() as usize;
                        fm &= fm - 1;
                        planes_add(&mut pt, ring[u]);
                    }
                    dest_p[tu] = pt;
                    dest_b1[tu] = b1o;
                    dest_b2[tu] = b2o;
                }
                let clone = m.is_clone();
                let tbit = 1u64 << tu;
                let oi_fast = if clone {
                    let ea = empty & !tbit;
                    opp_state(dest_b1[tu] & ea, dest_b2[tu] & ea, &dest_p[tu])
                } else {
                    let mut pa = dest_p[tu];
                    planes_sub(&mut pa, ring[m.from as usize]);
                    let ea = (empty & !tbit) | (1u64 << m.from);
                    opp_state(dest_b1[tu] & ea, dest_b2[tu] & ea, &pa)
                };

                // Brute-force oracle: make(m), enumerate opp's legal moves
                let b2 = b.make(m);
                b2.legal_moves_dedup_into(&mut ml2);
                let oi_ref = if ml2.is_empty() {
                    0usize
                } else {
                    let mut best = 0usize;
                    for rm in ml2.iter() {
                        let c = (b2.opp() & ring[rm.to as usize]).count_ones() as usize;
                        best = best.max(rm.is_clone() as usize + 2 * c);
                    }
                    1 + best
                };
                assert_eq!(oi_fast, oi_ref, "mismatch on move {m:?} me={me:#x} opp={opp:#x}");
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// D1462 (user): `ti policy-rl` = the policy head plays ITSELF (pure policy, no search, Botzone rules) and learns after
// every batch of games, like a normal training run: Adam on the head (bias, scalars, 3x3 tables, per phase bucket),
// policy gradient loss = -(z - 0.5) ln pi(a) for every move a, z = the mover's game result (1 / 0.5 / 0; 0.5 is the
// exact self-play baseline), plus an optional decoupled pull toward the start head (--anchor, AdamW-style decay toward
// the init weights instead of toward 0). Muon is not used: the head has no weight matrices, only lookup tables.
// Openings: --open-plies uniformly random plies, then moves sampled from softmax(logits / --temp): games do not repeat.
// Every --arena-every steps: greedy current head vs the start head on fixed paired random openings (fixed seeds).
// Experiment only (--experiment NAME, outputs under experiments/); games go to <out dir>/games.bin, never the DAG.
//   ti policy-rl --experiment NAME --init POL.bin (TPOL6) --out-bin experiments/X/pol_rl.bin [--threads 4]
//     [--games-per-step 256] [--lr 0.001] [--temp 1.0] [--anchor 0.0] [--open-plies 4] [--arena-every 20]
//     [--arena-games 400] [--resume]
// Outputs: --out-bin (latest head, TPOL6, engine-loadable), <out>_best.bin (best arena score), <out>.ckpt (resume),
// <dir>/games.bin (u16 plies, u8 result 0 black / 1 white / 2 draw, then plies x (from, to)). STOP / STOP_ALL honoured.

struct RlRng(u64);
impl RlRng {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    fn unit(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 24) as f32
    }
}

type Prm<'a> = (&'a [f32], &'a [f32], &'a [f32], &'a [f32]);

/// logits of every legal move of `b` (side to move = own) into `z`; the moves are `x.ml` (same order as the trainer)
fn rl_logits(b: &Board, ring: &[u64; 49], prm: Prm, x: &mut Scratch, z: &mut Vec<f32>) -> usize {
    let s = S { own: b.me(), opp: b.opp(), from: 255, to: 255, kind: 0, sel: 255, kn: 0, val: false, ko: 0, aw: 1.0, up: 0 };
    let _ = expand(&s, &[], ring, true, true, x);
    let n = x.ml.len();
    z.clear();
    if n < 2 {
        return n;
    }
    let (bias, g, td, tor) = prm;
    let [ob, og, otd, oto] = x.off;
    for (i, m) in x.mv.iter().enumerate() {
        let mut v = bias[ob + m[0] as usize] + g[og + m[1] as usize] + g[og + m[2] as usize];
        if m[3] != 255 {
            v += g[og + m[3] as usize];
        }
        let [d, o] = x.pv[i];
        v += td[otd + d as usize];
        if o != NONE {
            v += tor[oto + o as usize];
        }
        z.push(v);
    }
    n
}

/// pick a move: temp <= 0 = argmax, else a sample of softmax(z / temp). Returns (index, entropy of softmax(z)).
fn rl_pick(z: &[f32], temp: f32, rng: &mut RlRng) -> (usize, f32) {
    let mx = z.iter().cloned().fold(f32::MIN, f32::max);
    let den: f32 = z.iter().map(|v| (v - mx).exp()).sum();
    let ent: f32 = -z
        .iter()
        .map(|v| {
            let p = (v - mx).exp() / den;
            if p > 0.0 { p * p.ln() } else { 0.0 }
        })
        .sum::<f32>();
    if temp <= 0.0 {
        let i = z.iter().enumerate().fold((0, f32::MIN), |a, (i, &v)| if v > a.1 { (i, v) } else { a }).0;
        return (i, ent);
    }
    let w: Vec<f32> = z.iter().map(|v| ((v - mx) / temp).exp()).collect();
    let tot: f32 = w.iter().sum();
    let mut r = rng.unit() * tot;
    for (i, wi) in w.iter().enumerate() {
        r -= wi;
        if r <= 0.0 {
            return (i, ent);
        }
    }
    (w.len() - 1, ent)
}

fn rl_random_opening(plies: usize, rng: &mut RlRng) -> Board {
    let mut b = Board::start();
    let mut ml = MoveList::new();
    for _ in 0..plies {
        if b.bz_over() {
            break;
        }
        b.legal_moves_dedup_into(&mut ml);
        if ml.is_empty() {
            break;
        }
        let m = ml.move_at((rng.next() % ml.len() as u64) as usize);
        b = b.make(m);
    }
    b
}

/// one game from `start`; side 0 (black) uses prm[0], side 1 prm[1]. Returns (result 0 black / 1 white / 2 draw, the
/// scored plies (own, opp, from, to, mover), entropy sum, scored plies count).
#[allow(clippy::type_complexity)]
fn rl_game(start: Board, ring: &[u64; 49], prm: [Prm; 2], temp: f32, rng: &mut RlRng, x: &mut Scratch, z: &mut Vec<f32>) -> (u8, Vec<(u64, u64, u8, u8, u8)>, f64, u32) {
    let mut b = start;
    let mut rec = Vec::new();
    let (mut ent, mut ne) = (0f64, 0u32);
    while b.ply < 400 && !b.bz_over() {
        let n = rl_logits(&b, ring, prm[b.turn as usize], x, z);
        if n == 0 {
            b = b.make_pass();
            continue;
        }
        let i = if n == 1 {
            0
        } else {
            let (i, e) = rl_pick(z, temp, rng);
            ent += e as f64;
            ne += 1;
            i
        };
        let m = x.ml.move_at(i);
        if n >= 2 {
            rec.push((b.me(), b.opp(), m.from, m.to, b.turn));
        }
        b = b.make(m);
    }
    let (c0, c1) = if b.bz_over() { b.bz_counts() } else { b.counts() };
    let res = match c0.cmp(&c1) {
        std::cmp::Ordering::Greater => 0,
        std::cmp::Ordering::Less => 1,
        std::cmp::Ordering::Equal => 2,
    };
    (res, rec, ent, ne)
}

pub fn run_rl(args: &[String]) -> ExitCode {
    let exp = crate::genguard::experiment(args);
    let (Some(_), Some(init), Some(out)) = (&exp, arg(args, "--init"), arg(args, "--out-bin")) else {
        eprintln!("usage: ti policy-rl --experiment NAME --init POL.bin (TPOL6) --out-bin experiments/X/pol_rl.bin [--threads 4] [--games-per-step 256] [--lr 0.001] [--temp 1.0] [--anchor 0.0] [--open-plies 4] [--arena-every 20] [--arena-games 400] [--resume]");
        return ExitCode::from(2);
    };
    crate::genguard::announce("policy-rl", &exp);
    if let Err(e) = crate::genguard::check_output("policy-rl", &exp, &out) {
        eprintln!("{e}");
        return ExitCode::from(2);
    }
    let num = |n: &str, d: f64| arg(args, n).and_then(|v| v.parse::<f64>().ok()).unwrap_or(d);
    let threads = num("--threads", 4.0).max(1.0) as usize;
    let gps = (num("--games-per-step", 256.0) as usize).max(threads);
    let lr = num("--lr", 0.001) as f32;
    let temp = num("--temp", 1.0) as f32;
    let anchor = num("--anchor", 0.0) as f32;
    let open_plies = num("--open-plies", 4.0) as usize;
    let arena_every = (num("--arena-every", 20.0) as usize).max(1);
    let arena_games = ((num("--arena-games", 400.0) as usize) / 2 * 2).max(2);

    // the start head: TPOL6 (phase buckets), file layout per bucket [bias, g, td, tor]
    let Ok(d) = fs::read(&init) else {
        eprintln!("{init}: cannot read");
        return ExitCode::from(1);
    };
    let per = SLOTS + NSCAL + 27 * P3;
    if d.len() < 16 || &d[0..5] != b"TPOL6" {
        eprintln!("{init}: not a TPOL6 policy head");
        return ExitCode::from(2);
    }
    let nb = u32::from_le_bytes(d[12..16].try_into().unwrap()) as usize;
    if nb == 0 || d.len() != 16 + 4 * per * nb {
        eprintln!("{init}: TPOL6 size does not match {nb} buckets of a --pat k3c head");
        return ExitCode::from(2);
    }
    let f: Vec<f32> = d[16..].chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
    let (sl, gl, tdl, tol) = (SLOTS, NSCAL, 18 * P3, 9 * P3);
    let _ = PB.set([nb, sl, gl, tdl, tol]);
    let (mut bias, mut g, mut td, mut tor) = (vec![0f32; sl * nb], vec![0f32; gl * nb], vec![0f32; tdl * nb], vec![0f32; tol * nb]);
    for k in 0..nb {
        let blk = &f[k * per..(k + 1) * per];
        bias[k * sl..(k + 1) * sl].copy_from_slice(&blk[..sl]);
        g[k * gl..(k + 1) * gl].copy_from_slice(&blk[sl..sl + gl]);
        td[k * tdl..(k + 1) * tdl].copy_from_slice(&blk[sl + gl..sl + gl + tdl]);
        tor[k * tol..(k + 1) * tol].copy_from_slice(&blk[sl + gl + tdl..per]);
    }
    let (b0, g0, td0, tor0) = (bias.clone(), g.clone(), td.clone(), tor.clone());
    let ring = ring1();
    let run_dir = std::path::Path::new(&out).parent().map(|p| p.to_path_buf()).unwrap_or_else(|| ".".into());
    let _ = fs::create_dir_all(&run_dir);
    if let Some(s) = crate::stopctl::requested(Some(&run_dir)) {
        eprintln!("policy-rl: stop file {s} exists: remove it to run");
        return ExitCode::from(2);
    }
    let best_path = format!("{}_best.bin", out.trim_end_matches(".bin"));
    let ckpt_path = format!("{out}.ckpt");
    let (mut mb, mut vb, mut mg, mut vg) = (vec![0f32; bias.len()], vec![0f32; bias.len()], vec![0f32; g.len()], vec![0f32; g.len()]);
    let (mut mtd, mut vtd, mut mto, mut vto) = (vec![0f32; td.len()], vec![0f32; td.len()], vec![0f32; tor.len()], vec![0f32; tor.len()]);
    let (mut step0, mut t, mut best_score, mut best_step) = (0usize, 0i32, f64::MIN, 0usize);
    if args.iter().any(|a| a == "--resume") {
        match ckpt_read(&ckpt_path) {
            Some((h, vs)) if vs[0].len() == bias.len() && vs[3].len() == tor.len() => {
                bias.copy_from_slice(&vs[0]);
                g.copy_from_slice(&vs[1]);
                td.copy_from_slice(&vs[2]);
                tor.copy_from_slice(&vs[3]);
                mb.copy_from_slice(&vs[4]);
                vb.copy_from_slice(&vs[5]);
                mg.copy_from_slice(&vs[6]);
                vg.copy_from_slice(&vs[7]);
                mtd.copy_from_slice(&vs[8]);
                vtd.copy_from_slice(&vs[9]);
                mto.copy_from_slice(&vs[10]);
                vto.copy_from_slice(&vs[11]);
                (step0, t, best_score, best_step) = (h.0 as usize, h.1, h.2, h.3 as usize);
                println!("resumed from {ckpt_path} at step {step0} (best arena {best_score:.4} at step {best_step})");
            }
            _ => println!("--resume: no usable checkpoint, starting fresh"),
        }
    }
    if step0 == 0 {
        let _ = write_atomic(&out, &head_bytes(true, true, [&bias, &g, &td, &tor]));
        let _ = write_atomic(&best_path, &head_bytes(true, true, [&bias, &g, &td, &tor]));
    }
    println!(
        "policy-rl: init {init} ({nb} buckets, {} params), {threads} threads, {gps} games/step, lr {lr}, temp {temp}, anchor {anchor}, open plies {open_plies}, arena {arena_games} g every {arena_every} steps",
        bias.len() + g.len() + td.len() + tor.len()
    );
    let mut games_f = fs::OpenOptions::new().create(true).append(true).open(run_dir.join("games.bin")).ok();
    let t_start = std::time::Instant::now();
    let mut games_total = (step0 * gps) as u64;
    let games_at_start = games_total;
    let (beta1, beta2, eps) = (0.9f32, 0.999f32, 1e-8f32);

    // greedy arena of `cur` vs `base` on fixed paired openings; returns (score of cur, wins, losses, draws)
    let arena = |cur: Prm, base: Prm| -> (f64, u32, u32, u32) {
        let pairs = arena_games / 2;
        let per_t = pairs.div_ceil(threads);
        let rs: Vec<(u32, u32, u32)> = std::thread::scope(|sc| {
            let hs: Vec<_> = (0..threads)
                .map(|ti| {
                    let ring = &ring;
                    sc.spawn(move || {
                        let (mut x, mut z) = (new_scratch(), Vec::new());
                        let (mut w, mut l, mut dr) = (0u32, 0u32, 0u32);
                        for k in ti * per_t..((ti + 1) * per_t).min(pairs) {
                            let mut orng = RlRng(0xA5A5_0000_0000_0001 ^ (k as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15));
                            let start = rl_random_opening(open_plies.max(2), &mut orng);
                            for cur_side in 0..2u8 {
                                let prm = if cur_side == 0 { [cur, base] } else { [base, cur] };
                                let mut r = RlRng(1);
                                let (res, ..) = rl_game(start, ring, prm, 0.0, &mut r, &mut x, &mut z);
                                if res == 2 {
                                    dr += 1;
                                } else if res == cur_side {
                                    w += 1;
                                } else {
                                    l += 1;
                                }
                            }
                        }
                        (w, l, dr)
                    })
                })
                .collect();
            hs.into_iter().map(|h| h.join().unwrap()).collect()
        });
        let (w, l, dr) = rs.iter().fold((0, 0, 0), |a, r| (a.0 + r.0, a.1 + r.1, a.2 + r.2));
        let n = (w + l + dr).max(1) as f64;
        ((w as f64 + 0.5 * dr as f64) / n, w, l, dr)
    };
    let elo = |s: f64| if s <= 0.0 || s >= 1.0 { f64::NAN } else { -400.0 * (1.0 / s - 1.0).log10() };

    for step in step0.. {
        if let Some(s) = crate::stopctl::requested(Some(&run_dir)) {
            println!("policy-rl: stop file {s}: saving and exiting at step {step}");
            let hdr = (step as u32, t, best_score, best_step as u32, 0u32);
            let _ = write_atomic(&ckpt_path, &ckpt_bytes(hdr, [&bias, &g, &td, &tor, &mb, &vb, &mg, &vg, &mtd, &vtd, &mto, &vto, &b0, &g0, &td0, &tor0]));
            break;
        }
        let per_t = gps.div_ceil(threads);
        let prm: Prm = (&bias, &g, &td, &tor);
        #[allow(clippy::type_complexity)]
        let outs: Vec<(Grad, [u64; 4], f64, u64, u64, Vec<u8>)> = std::thread::scope(|sc| {
            let hs: Vec<_> = (0..threads)
                .map(|ti| {
                    let ring = &ring;
                    sc.spawn(move || {
                        let mut rng = RlRng(0x1234_5678_9ABC_DEF1 ^ (((step as u64) << 8) | ti as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9));
                        let mut gr = Grad { gb: vec![0f32; prm.0.len()], gg: vec![0f32; prm.1.len()], gtd: vec![0f32; prm.2.len()], gto: vec![0f32; prm.3.len()], touched: vec![], touched_o: vec![], sc: new_scratch(), z: vec![] };
                        let (mut x, mut z) = (new_scratch(), Vec::new());
                        let mut res_n = [0u64; 4]; // black wins, white wins, draws, scored plies
                        let (mut ent, mut ne, mut nmoves) = (0f64, 0u64, 0u64);
                        let mut gbytes = Vec::new();
                        for _ in 0..per_t {
                            let start = rl_random_opening(open_plies, &mut rng);
                            let (res, rec, e, n) = rl_game(start, ring, [prm, prm], temp, &mut rng, &mut x, &mut z);
                            res_n[res as usize] += 1;
                            res_n[3] += rec.len() as u64;
                            ent += e;
                            ne += n as u64;
                            gbytes.extend_from_slice(&(rec.len() as u16).to_le_bytes());
                            gbytes.push(res);
                            for &(own, opp, from, to, turn) in &rec {
                                gbytes.push(from);
                                gbytes.push(to);
                                let zr = if res == 2 { 0.5 } else if res == turn { 1.0 } else { 0.0 };
                                let s = S { own, opp, from, to, kind: 0, sel: 255, kn: 0, val: false, ko: 0, aw: zr - 0.5, up: 0 };
                                if s.aw != 0.0 {
                                    grad_one(&s, &[], ring, true, true, prm, (1.0, 0.0), 1.0, &mut gr);
                                    gr.touched.clear();
                                    gr.touched_o.clear();
                                }
                                nmoves += 1;
                            }
                        }
                        (gr, res_n, ent, ne, nmoves, gbytes)
                    })
                })
                .collect();
            hs.into_iter().map(|h| h.join().unwrap()).collect()
        });
        let mut it = outs.into_iter();
        let (mut gsum, mut res_n, mut ent, mut ne, mut nmoves, mut gbytes) = it.next().unwrap();
        for (gr, r, e, n, nm, gb) in it {
            for (a, b) in gsum.gb.iter_mut().zip(&gr.gb) {
                *a += b;
            }
            for (a, b) in gsum.gg.iter_mut().zip(&gr.gg) {
                *a += b;
            }
            for (a, b) in gsum.gtd.iter_mut().zip(&gr.gtd) {
                *a += b;
            }
            for (a, b) in gsum.gto.iter_mut().zip(&gr.gto) {
                *a += b;
            }
            for k in 0..4 {
                res_n[k] += r[k];
            }
            ent += e;
            ne += n;
            nmoves += nm;
            gbytes.extend_from_slice(&gb);
        }
        if let Some(fh) = games_f.as_mut() {
            let _ = std::io::Write::write_all(fh, &gbytes);
        }
        let ngames = res_n[0] + res_n[1] + res_n[2];
        games_total += ngames;
        // Adam step on the mean gradient (+ decoupled pull toward the start head)
        t += 1;
        let inv = 1.0 / nmoves.max(1) as f32;
        let (bc1, bc2) = (1.0 - beta1.powi(t), 1.0 - beta2.powi(t));
        let mut gnorm = 0f64;
        let mut adam = |p: &mut [f32], p0: &[f32], gr: &[f32], m: &mut [f32], v: &mut [f32]| {
            for i in 0..p.len() {
                let gi = gr[i] * inv;
                if gi == 0.0 && m[i] == 0.0 {
                    continue;
                }
                gnorm += (gi * gi) as f64;
                m[i] = beta1 * m[i] + (1.0 - beta1) * gi;
                v[i] = beta2 * v[i] + (1.0 - beta2) * gi * gi;
                p[i] -= lr * ((m[i] / bc1) / ((v[i] / bc2).sqrt() + eps) + anchor * (p[i] - p0[i]));
            }
        };
        adam(&mut bias, &b0, &gsum.gb, &mut mb, &mut vb);
        adam(&mut g, &g0, &gsum.gg, &mut mg, &mut vg);
        adam(&mut td, &td0, &gsum.gtd, &mut mtd, &mut vtd);
        adam(&mut tor, &tor0, &gsum.gto, &mut mto, &mut vto);
        let drift: f64 = bias
            .iter()
            .zip(&b0)
            .chain(g.iter().zip(&g0))
            .chain(td.iter().zip(&td0))
            .chain(tor.iter().zip(&tor0))
            .map(|(a, b)| ((a - b) * (a - b)) as f64)
            .sum::<f64>()
            .sqrt();
        let el = t_start.elapsed().as_secs_f64();
        let pc = |k: usize| 100.0 * res_n[k] as f64 / ngames.max(1) as f64;
        println!(
            "[step {}] games {} (+{}) | black {:.1}% white {:.1}% draw {:.1}% | plies/game {:.1} | entropy {:.3} | grad {:.2e} | drift {:.3} | {:.0} games/s",
            step + 1,
            games_total,
            ngames,
            pc(0),
            pc(1),
            pc(2),
            res_n[3] as f64 / ngames.max(1) as f64,
            ent / ne.max(1) as f64,
            gnorm.sqrt(),
            drift,
            (games_total - games_at_start) as f64 / el.max(1e-9)
        );
        let _ = write_atomic(&out, &head_bytes(true, true, [&bias, &g, &td, &tor]));
        if (step + 1) % arena_every == 0 {
            let (s, w, l, dr) = arena((&bias, &g, &td, &tor), (&b0, &g0, &td0, &tor0));
            let mark = if s > best_score {
                best_score = s;
                best_step = step + 1;
                let _ = write_atomic(&best_path, &head_bytes(true, true, [&bias, &g, &td, &tor]));
                " NEW BEST"
            } else {
                ""
            };
            println!(
                "[arena step {}] greedy vs start head: +{w} -{l} ={dr} score {:.1}% Elo {:+.1}{mark} (best {:.1}% at step {best_step})",
                step + 1,
                100.0 * s,
                elo(s),
                100.0 * best_score
            );
            let hdr = ((step + 1) as u32, t, best_score, best_step as u32, 0u32);
            let _ = write_atomic(&ckpt_path, &ckpt_bytes(hdr, [&bias, &g, &td, &tor, &mb, &vb, &mg, &vg, &mtd, &vtd, &mto, &vto, &b0, &g0, &td0, &tor0]));
        }
    }
    ExitCode::SUCCESS
}

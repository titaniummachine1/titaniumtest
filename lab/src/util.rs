//! Small helpers copied from the hub (record reader, bitboard counts) so the lab builds against the engine alone.
use titanium::Move;

pub const REC: usize = 48;

/// the fields of a `.tdat` record the studies need (layout: hub src/data.rs Rec)
pub struct Rec {
    pub occ0: u64,
    pub occ1: u64,
    pub stm: u8,
}

impl Rec {
    pub fn from_bytes(b: &[u8]) -> Rec {
        Rec {
            occ0: u64::from_le_bytes(b[0..8].try_into().unwrap()),
            occ1: u64::from_le_bytes(b[8..16].try_into().unwrap()),
            stm: b[40],
        }
    }
}

pub fn arg(args: &[String], key: &str) -> Option<String> {
    args.iter().position(|a| a == key).and_then(|i| args.get(i + 1)).cloned()
}

pub fn ring1() -> [u64; 49] {
    let mut r = [0u64; 49];
    for i in 0..49u8 {
        for j in 0..49u8 {
            if i != j && (Move { from: i, to: j }).is_clone() {
                r[i as usize] |= 1u64 << j;
            }
        }
    }
    r
}

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

#[inline(always)]
pub fn box1(x: u64) -> u64 {
    let h = x | ((x & !FILE_A) >> 1) | ((x & !FILE_G) << 1);
    (h | (h >> 7) | (h << 7)) & FULL49
}

/// bit planes (1, 2, 4, 8) of the number of `v` stones around every square (carry-save adder over 8 shifts)
#[inline(always)]
pub fn victim_planes(v: u64) -> [u64; 4] {
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
pub fn planes_add(p: &mut [u64; 4], m: u64) {
    let c0 = p[0] & m;
    p[0] ^= m;
    let c1 = p[1] & c0;
    p[1] ^= c0;
    let c2 = p[2] & c1;
    p[2] ^= c1;
    p[3] ^= c2;
}

#[inline(always)]
pub fn planes_sub(p: &mut [u64; 4], m: u64) {
    let b0 = !p[0] & m;
    p[0] ^= m;
    let b1 = !p[1] & b0;
    p[1] ^= b0;
    let b2 = !p[2] & b1;
    p[2] ^= b1;
    p[3] ^= b2;
}

#[inline(always)]
pub fn planes_max(cand: u64, p: &[u64; 4]) -> u32 {
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

/// 0 = the opponent has no move; else 1 + its best reply gain (clone 1 + 2v, jump 2v)
#[inline(always)]
pub fn opp_state(b1_empty: u64, b2_empty: u64, p: &[u64; 4]) -> usize {
    if b2_empty == 0 {
        return 0;
    }
    let cl = if b1_empty != 0 { 1 + 2 * planes_max(b1_empty, p) } else { 0 };
    let jp = 2 * planes_max(b2_empty, p);
    1 + cl.max(jp) as usize
}

//! H4: position -> sparse features (+ bucket) with exact D4 augmentation done on the bitboards, then
//! the ENGINE's own `nnfeat::both` runs on the transformed board, so there is no feature permutation
//! table that could drift from the engine.
use crate::data::Rec;
use titanium::nnfeat::{self, ALL_GROUPS, MAX_ACTIVE};
use titanium::Board;

pub const MODES: usize = 8;

const fn square_map(mode: usize) -> [u8; 49] {
    let mut m = [0u8; 49];
    let mut r = 0;
    while r < 7 {
        let mut c = 0;
        while c < 7 {
            let (nr, nc) = match mode {
                0 => (r, c),
                1 => (c, 6 - r),
                2 => (6 - r, 6 - c),
                3 => (6 - c, r),
                4 => (r, 6 - c),
                5 => (6 - r, c),
                6 => (c, r),
                _ => (6 - c, 6 - r),
            };
            m[r * 7 + c] = (nr * 7 + nc) as u8;
            c += 1;
        }
        r += 1;
    }
    m
}

pub const MAPS: [[u8; 49]; MODES] = [
    square_map(0), square_map(1), square_map(2), square_map(3),
    square_map(4), square_map(5), square_map(6), square_map(7),
];

pub fn transform_bb(mut bb: u64, mode: usize) -> u64 {
    let m = &MAPS[mode];
    let mut out = 0u64;
    while bb != 0 {
        let sq = bb.trailing_zeros() as usize;
        out |= 1u64 << m[sq];
        bb &= bb - 1;
    }
    out
}

pub fn bucket_of(empties: usize, buckets: usize) -> usize {
    (empties * buckets / 50).min(buckets - 1)
}

pub struct Sample {
    pub us: Vec<u32>,
    pub them: Vec<u32>,
    pub bucket: usize,
}

pub fn board_of(r: &Rec, mode: usize) -> Board {
    Board::from_occ(transform_bb(r.occ0, mode), transform_bb(r.occ1, mode), 0, r.stm)
}

pub fn sample(r: &Rec, mode: usize, buckets: usize) -> Sample {
    let b = board_of(r, mode);
    let (us, nu, them, nt) = nnfeat::both(&b, ALL_GROUPS);
    Sample { us: us[..nu].to_vec(), them: them[..nt].to_vec(), bucket: bucket_of(b.empty().count_ones() as usize, buckets) }
}

#[allow(dead_code)]
pub const MAX: usize = MAX_ACTIVE;

#[cfg(test)]
mod tests {
    use super::*;
    use nnfeat::{G1_OFF, G2_OFF, SQ_STATES};

    fn rec(seed: &mut u64) -> Rec {
        let mut next = || {
            *seed ^= *seed << 13;
            *seed ^= *seed >> 7;
            *seed ^= *seed << 17;
            *seed
        };
        let full = (1u64 << 49) - 1;
        let a = next() & next() & full;
        let b = next() & next() & full & !a;
        Rec { occ0: a, occ1: b, outcome: 0.5, cp: 0.0, wsum: 1.0, p3: 0.0, cp5k: 0.0, zx: 0, stm: (next() & 1) as u8, teacher_kind: 0, proof_kind: 0, relabel_src: 0, ti_clash: 0 }
    }

    #[test]
    fn modes_are_a_group_and_identity_is_identity() {
        let mut s = 0x9E37_79B9_7F4A_7C15u64;
        for _ in 0..200 {
            let r = rec(&mut s);
            assert_eq!(transform_bb(r.occ0, 0), r.occ0);
            for m in 0..MODES {
                let t = transform_bb(r.occ0, m);
                assert_eq!(t.count_ones(), r.occ0.count_ones());
                // some mode inverts it
                assert!((0..MODES).any(|k| transform_bb(t, k) == r.occ0), "mode {m}");
            }
        }
    }

    #[test]
    fn g1_features_follow_the_square_map() {
        // G1 index = G1_OFF + sq*SQ_STATES + code: after a D4 transform the set of (code) per square must permute.
        let mut s = 0x1234_5678_9abc_def1u64;
        for _ in 0..200 {
            let r = rec(&mut s);
            let base = sample(&r, 0, 8);
            for m in 1..MODES {
                let t = sample(&r, m, 8);
                let g1 = |v: &Vec<u32>| -> Vec<(usize, usize)> {
                    let mut o: Vec<_> = v.iter().map(|&f| f as usize).filter(|&f| f >= G1_OFF && f < G2_OFF).map(|f| (f / SQ_STATES, f % SQ_STATES)).collect();
                    o.sort();
                    o
                };
                let mut want: Vec<_> = g1(&base.us).into_iter().map(|(sq, c)| (MAPS[m][sq] as usize, c)).collect();
                want.sort();
                assert_eq!(g1(&t.us), want, "mode {m}");
                assert_eq!(t.bucket, base.bucket);
            }
        }
    }

    #[test]
    fn bucket_rule_matches_engine() {
        assert_eq!(bucket_of(0, 8), 0);
        assert_eq!(bucket_of(49, 8), 7);
        assert_eq!(bucket_of(25, 8), 4);
    }
}

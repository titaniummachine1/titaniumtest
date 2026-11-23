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

/// First empties count of each bucket in map E16 (id 1). MUST equal `titanium::nnue::bucket_map` in the engine
/// (branch exp/bucketmap): fine buckets in the endgame where the loss is, even spread elsewhere.
pub const E16_STARTS: [usize; 16] = [0, 2, 3, 4, 5, 6, 7, 11, 15, 19, 22, 25, 29, 33, 37, 41];

/// Bucket of `empties` under map `id` (0 = the even formula; 1 = E16). The id lives in bits 8.. of the net's `groups` word.
pub fn bucket_of_id(empties: usize, buckets: usize, id: u32) -> usize {
    match id {
        1 => E16_STARTS.iter().filter(|&&s| s <= empties).count() - 1,
        _ => bucket_of(empties, buckets),
    }
}

pub struct Sample {
    pub us: Vec<u32>,
    pub them: Vec<u32>,
    pub bucket: usize,
}

pub fn board_of(r: &Rec, mode: usize) -> Board {
    Board::from_occ(transform_bb(r.occ0, mode), transform_bb(r.occ1, mode), 0, r.stm)
}

pub fn sample(r: &Rec, mode: usize, buckets: usize, map_id: u32) -> Sample {
    let b = board_of(r, mode);
    let (us, nu, them, nt) = nnfeat::both(&b, ALL_GROUPS);
    Sample { us: us[..nu].to_vec(), them: them[..nt].to_vec(), bucket: bucket_of_id(b.empty().count_ones() as usize, buckets, map_id) }
}

/// Full-dataset shuffled block permutation so `--d4-all` visits every 1024-row block once per pass without replacement,
/// ensuring all 8 D4 orientations of every position appear in one augmented epoch.
pub struct BlockPerm {
    perm: Vec<usize>,
    pos: usize,
}

impl BlockPerm {
    pub fn new(nblk: usize, next_u64: &mut impl FnMut() -> u64) -> Self {
        let mut perm: Vec<usize> = (0..nblk).collect();
        for i in (1..perm.len()).rev() {
            let j = (next_u64() % (i as u64 + 1)) as usize;
            perm.swap(i, j);
        }
        BlockPerm { perm, pos: 0 }
    }

    pub fn advance(&mut self, steps: usize, next_u64: &mut impl FnMut() -> u64) {
        for _ in 0..steps {
            let _ = self.next_block(next_u64);
        }
    }

    pub fn next_block(&mut self, next_u64: &mut impl FnMut() -> u64) -> usize {
        if self.perm.is_empty() {
            return 0;
        }
        if self.pos >= self.perm.len() {
            for i in (1..self.perm.len()).rev() {
                let j = (next_u64() % (i as u64 + 1)) as usize;
                self.perm.swap(i, j);
            }
            self.pos = 0;
        }
        let b = self.perm[self.pos];
        self.pos += 1;
        b
    }
}

/// D1630/D1631 (default on value training): expand every sampled position into all 8 D4 orientations (`transform_bb` `0..8`)
/// and Fisher-Yates scramble the entire queue buffer so the 8 copies of a position are never trained in sequence.
pub fn expand_and_scramble_d4(raw: &[Rec], out: &mut Vec<Rec>, mut next_u64: impl FnMut() -> u64) {
    out.clear();
    out.reserve(raw.len() * MODES);
    for &r in raw {
        for mode in 0..MODES {
            let mut o = r;
            o.occ0 = transform_bb(r.occ0, mode);
            o.occ1 = transform_bb(r.occ1, mode);
            out.push(o);
        }
    }
    for i in (1..out.len()).rev() {
        let j = (next_u64() % (i as u64 + 1)) as usize;
        out.swap(i, j);
    }
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
            let base = sample(&r, 0, 8, 0);
            for m in 1..MODES {
                let t = sample(&r, m, 8, 0);
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

    #[test]
    fn d4_all_expands_all_8_and_scrambles() {
        let mut s = 0x1234_5678_9ABC_DEF1u64;
        let mut raw = Vec::new();
        for idx in 0..32usize {
            let mut r = rec(&mut s);
            r.zx = idx as i32; // distinguish source board uniquely
            raw.push(r);
        }
        let mut rng = 0xDEAD_BEEF_CAFE_BABEu64;
        let mut next = || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            rng
        };
        let mut out = Vec::new();
        expand_and_scramble_d4(&raw, &mut out, &mut next);
        assert_eq!(out.len(), raw.len() * MODES);
        // Every source board must appear with all 8 transformed bitboards exactly once.
        for (idx, r) in raw.iter().enumerate() {
            let got: Vec<(u64, u64)> = out.iter().filter(|o| o.zx == idx as i32).map(|o| (o.occ0, o.occ1)).collect();
            assert_eq!(got.len(), MODES);
            for mode in 0..MODES {
                let want = (transform_bb(r.occ0, mode), transform_bb(r.occ1, mode));
                assert!(got.contains(&want), "board {idx} missing D4 mode {mode}");
            }
        }
        // Check that orientations are genuinely scrambled (not left in consecutive 8-item blocks).
        let consecutive_runs = out.windows(2).filter(|w| w[0].zx == w[1].zx).count();
        assert!(consecutive_runs < 10, "expected scrambled queue, found {consecutive_runs} adjacent pairs");

        // BlockPerm must visit every block 0..nblk exactly once before reshuffling.
        let mut perm = BlockPerm::new(17, &mut next);
        let mut seen: Vec<usize> = (0..17).map(|_| perm.next_block(&mut next)).collect();
        seen.sort_unstable();
        assert_eq!(seen, (0..17).collect::<Vec<_>>());
    }
}

#[cfg(test)]
mod bench {
    use super::*;
    /// `cargo test --release mapper_speed -- --ignored --nocapture` (needs the packed val file)
    #[test]
    #[ignore]
    fn mapper_speed() {
        let p = "C:/gitProjects/titanium-ataxx/data/nnue/packed/val_v9.tdat";
        let b = std::fs::read(p).unwrap();
        let rows: Vec<Rec> = b.chunks_exact(crate::data::REC).take(200_000).map(Rec::from_bytes).collect();
        let t = std::time::Instant::now();
        let mut n = 0usize;
        for (i, r) in rows.iter().enumerate() {
            n += sample(r, i & 7, 8, 0).us.len();
        }
        let s = t.elapsed().as_secs_f64();
        let t2 = std::time::Instant::now();
        let mut m = 0usize;
        for r in &rows {
            let (_, nu, _, nt) = nnfeat::both(&board_of(r, 0), ALL_GROUPS);
            m += nu + nt;
        }
        let s2 = t2.elapsed().as_secs_f64();
        println!("mapper (D4 + feats + Vec): {:.0} rows/s/core, avg active {:.1}; feats only {:.0} rows/s/core", rows.len() as f64 / s, n as f64 / rows.len() as f64, rows.len() as f64 / s2, );
        let _ = m;
    }
}

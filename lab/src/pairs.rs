//! gataxx's pair-pattern residual + mobility table (D991), as sparse features for `lab hce-train --pairs`.
//!
//! Same definition as gataxx 2.0.0 `PairPatterns.java` (MIT): every square is empty (0), side to move (1) or opponent (2);
//! a 2x2 block with origin s is the base-4 number of squares (s, s+1, s+7, s+8); a placement is a pair of blocks side by side
//! (4x2), one above the other (2x4) or touching at a corner (two diagonals): 80 placements. Each (placement, 8-square content)
//! has its own weight, tied across the 8 board symmetries. Mobility table: index = occupied count * 50 + empty squares within
//! two steps of a side's stones; feature +1 for the side to move's index, -1 for the opponent's (gataxx: mob[own] - mob[opp]).
//! Layout of the returned indices: 0..nclass = pair classes, nclass..nclass+2500 = mobility table.
use crate::hcefit::box2;

pub(crate) const PLACEMENTS: usize = 80;
pub(crate) const MOB: usize = 50 * 50;

pub(crate) struct Pairs {
    first: [u8; PLACEMENTS],
    second: [u8; PLACEMENTS],
    /// placement << 16 | pattern -> class id (u32::MAX = impossible pattern)
    class: Vec<u32>,
    pub(crate) nclass: usize,
    digits: [[u8; 16]; 16],
}

const ALL: u64 = (1u64 << 49) - 1;

pub(crate) fn transform(sq: usize, t: usize) -> usize {
    let n = 6;
    let (x, y) = (sq % 7, sq / 7);
    let (nx, ny) = match t {
        0 => (x, y),
        1 => (n - y, x),
        2 => (n - x, n - y),
        3 => (y, n - x),
        4 => (n - x, y),
        5 => (x, n - y),
        6 => (y, x),
        _ => (n - y, n - x),
    };
    ny * 7 + nx
}

impl Pairs {
    pub(crate) fn new() -> Pairs {
        let mut first = [0u8; PLACEMENTS];
        let mut second = [0u8; PLACEMENTS];
        let mut p = 0;
        let mut add = |ax: usize, ay: usize, bx: usize, by: usize| {
            first[p] = (ay * 7 + ax) as u8;
            second[p] = (by * 7 + bx) as u8;
            p += 1;
        };
        for y in 0..6 {
            for x in 0..=3 {
                add(x, y, x + 2, y);
            }
        }
        for y in 0..=3 {
            for x in 0..6 {
                add(x, y, x, y + 2);
            }
        }
        for y in 0..=3 {
            for x in 0..=3 {
                add(x, y, x + 2, y + 2);
            }
        }
        for y in 0..=3 {
            for x in 0..=3 {
                add(x, y + 2, x + 2, y);
            }
        }
        assert_eq!(p, PLACEMENTS);
        let sqs = |p: usize| -> [usize; 8] {
            let (a, b) = (first[p] as usize, second[p] as usize);
            [a, a + 1, a + 7, a + 8, b, b + 1, b + 7, b + 8]
        };
        let mut by_fp = std::collections::HashMap::new();
        for p in 0..PLACEMENTS {
            let fp = sqs(p).iter().fold(0u64, |m, &s| m | 1 << s);
            assert!(by_fp.insert(fp, p).is_none());
        }
        let mut digits = [[0u8; 16]; 16];
        for m in 0..16 {
            for t in 0..16 {
                if m & t != 0 {
                    continue;
                }
                let mut idx = 0u8;
                for i in 0..4 {
                    let d = if m >> i & 1 != 0 { 1 } else if t >> i & 1 != 0 { 2 } else { 0 };
                    idx |= d << (2 * i);
                }
                digits[m][t] = idx;
            }
        }
        // enumerate every possible pattern (digits 0..2 on 8 squares), canonical key = min over the 8 images
        let mut class = vec![u32::MAX; PLACEMENTS << 16];
        let mut ids: std::collections::HashMap<u32, u32> = std::collections::HashMap::new();
        for p in 0..PLACEMENTS {
            let sq = sqs(p);
            for code in 0..6561u32 {
                let mut st = [0u8; 8];
                let mut c = code;
                for s in st.iter_mut() {
                    *s = (c % 3) as u8;
                    c /= 3;
                }
                let enc = |sqv: &[usize; 8], at: &[u8; 49]| -> u32 {
                    let mut pat = 0u32;
                    for (i, &s) in sqv.iter().enumerate() {
                        pat |= (at[s] as u32) << (2 * (i % 4) + 8 * (i / 4));
                    }
                    pat
                };
                let mut at = [0u8; 49];
                for i in 0..8 {
                    at[sq[i]] = st[i];
                }
                let key = (p as u32) << 16 | enc(&sq, &at);
                let mut best = u32::MAX;
                for t in 0..8 {
                    let mut at2 = [0u8; 49];
                    let mut fp = 0u64;
                    for i in 0..8 {
                        let s2 = transform(sq[i], t);
                        at2[s2] = st[i];
                        fp |= 1 << s2;
                    }
                    let img = by_fp[&fp];
                    best = best.min((img as u32) << 16 | enc(&sqs(img), &at2));
                }
                let n = ids.len() as u32;
                let id = *ids.entry(best).or_insert(n);
                class[key as usize] = id;
            }
        }
        let nclass = ids.len();
        Pairs { first, second, class, nclass, digits }
    }

    pub(crate) fn dims(&self) -> usize {
        self.nclass + MOB
    }

    fn block(&self, me: u64, op: u64, s: u8) -> u32 {
        let s = s as u32;
        let m = ((me >> s) & 3) | ((me >> (s + 7)) & 3) << 2;
        let t = ((op >> s) & 3) | ((op >> (s + 7)) & 3) << 2;
        self.digits[m as usize][t as usize] as u32
    }

    fn mob_idx(own: u64, occ: u64) -> usize {
        occ.count_ones() as usize * 50 + (box2(own) & !occ & ALL).count_ones() as usize
    }

    /// appends (index + off, value) for the side to move `me` against `op`
    pub(crate) fn feats(&self, me: u64, op: u64, off: u32, out: &mut Vec<(u32, f32)>) {
        let mut bl = [0u32; 41];
        for y in 0..6u8 {
            for x in 0..6u8 {
                let s = y * 7 + x;
                bl[s as usize] = self.block(me, op, s);
            }
        }
        for p in 0..PLACEMENTS {
            let pat = bl[self.first[p] as usize] | bl[self.second[p] as usize] << 8;
            let c = self.class[p << 16 | pat as usize];
            debug_assert!(c != u32::MAX);
            out.push((off + c, 1.0));
        }
        let occ = me | op;
        let base = off + self.nclass as u32;
        let (a, b) = (Self::mob_idx(me, occ), Self::mob_idx(op, occ));
        if a != b {
            out.push((base + a as u32, 1.0));
            out.push((base + b as u32, -1.0));
        }
    }
}

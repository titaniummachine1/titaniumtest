//! Stone pairs by distance (D1072, user: "stone relations are local but leak into a wider area with fast-decaying
//! strength, like Stockfish's pawn pairs"). One sparse feature per pair of stones whose Chebyshev distance is in the
//! chosen bands, keyed by the pair's squares and colour combination (side to move's view: own-own, own-opp, opp-opp),
//! tied across the 8 board symmetries. Used by `lab hce-train --spairs "1,2,3,4"` (band 4 = distance 4..6) as a
//! residual group on the frozen NNUE, so each band's held-out gain shows how fast the missing pair information decays.
use crate::pairs::transform;

const N: usize = 49;

pub(crate) struct SPairs {
    /// distance d (1..6) in use
    pub(crate) keep_d: [bool; 7],
    /// per square: stones at a kept distance with a HIGHER square index (each unordered pair once)
    up: [u64; N],
    /// (a * 49 + b) * 3 + combo -> class id; combo 0 own-own, 1 own(a)-opp(b), 2 opp-opp; mixed pairs are ordered
    class: Vec<u32>,
    pub(crate) nclass: usize,
    pub(crate) bands: String,
}

fn cheb(a: usize, b: usize) -> usize {
    let (ax, ay, bx, by) = ((a % 7) as i32, (a / 7) as i32, (b % 7) as i32, (b / 7) as i32);
    (ax - bx).unsigned_abs().max((ay - by).unsigned_abs()) as usize
}

impl SPairs {
    pub(crate) fn new(bands: &str) -> SPairs {
        let mut keep_d = [false; 7];
        for s in bands.split(',').filter(|s| !s.trim().is_empty()) {
            match s.trim().parse::<usize>().expect("--spairs: distance bands 1,2,3,4") {
                b @ 1..=3 => keep_d[b] = true,
                4 => {
                    keep_d[4] = true;
                    keep_d[5] = true;
                    keep_d[6] = true;
                }
                b => panic!("--spairs: band {b} (use 1, 2, 3 or 4 = 4..6)"),
            }
        }
        let mut up = [0u64; N];
        for (a, u) in up.iter_mut().enumerate() {
            for b in a + 1..N {
                if keep_d[cheb(a, b)] {
                    *u |= 1u64 << b;
                }
            }
        }
        // canonical key per (ordered square pair, combo) = smallest image over the 8 symmetries; same-colour pairs unordered
        let mut ids = std::collections::HashMap::new();
        let mut class = vec![u32::MAX; N * N * 3];
        for a in 0..N {
            for b in 0..N {
                if a == b || !keep_d[cheb(a, b)] {
                    continue;
                }
                for combo in 0..3 {
                    let key = (0..8)
                        .map(|t| {
                            let (x, y) = (transform(a, t), transform(b, t));
                            if combo == 1 { (x, y) } else { (x.min(y), x.max(y)) }
                        })
                        .min()
                        .unwrap();
                    let n = ids.len() as u32;
                    let id = *ids.entry((key, combo)).or_insert(n);
                    class[(a * N + b) * 3 + combo] = id;
                }
            }
        }
        SPairs { keep_d, up, class, nclass: ids.len(), bands: bands.to_string() }
    }

    pub(crate) fn dims(&self) -> usize {
        self.nclass
    }

    /// appends (index + off, 1.0) for every kept stone pair, side to move `me` against `op`
    pub(crate) fn feats(&self, me: u64, op: u64, off: u32, out: &mut Vec<(u32, f32)>) {
        let occ = me | op;
        let mut s = occ;
        while s != 0 {
            let a = s.trailing_zeros() as usize;
            s &= s - 1;
            let a_me = (me >> a) & 1 != 0;
            let mut t = occ & self.up[a];
            while t != 0 {
                let b = t.trailing_zeros() as usize;
                t &= t - 1;
                let b_me = (me >> b) & 1 != 0;
                let idx = match (a_me, b_me) {
                    (true, true) => (a * N + b) * 3,
                    (false, false) => (a * N + b) * 3 + 2,
                    (true, false) => (a * N + b) * 3 + 1,
                    (false, true) => (b * N + a) * 3 + 1, // mixed pairs are keyed (own square, opp square)
                };
                out.push((off + self.class[idx], 1.0));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn classes_are_d4_invariant() {
        let p = SPairs::new("1,2,3,4");
        let (me, op) = (0x0000_1234_5678_9abc_u64 & ((1u64 << 49) - 1), 0x0001_8000_0300_0400_u64);
        let op = op & !me;
        let mut base = Vec::new();
        p.feats(me, op, 0, &mut base);
        base.sort_by_key(|x| x.0);
        for t in 1..8 {
            let tr = |x: u64| (0..49).filter(|&s| (x >> s) & 1 != 0).fold(0u64, |m, s| m | 1u64 << transform(s, t));
            let mut v = Vec::new();
            p.feats(tr(me), tr(op), 0, &mut v);
            v.sort_by_key(|x| x.0);
            assert_eq!(base, v, "symmetry {t}");
        }
    }
}

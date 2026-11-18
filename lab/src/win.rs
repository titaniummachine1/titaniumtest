//! Window tuples (D1117, user: "maybe 2x2 tuples could replace the expensive feature collection"). One sparse feature per
//! k x k window of the board (k = 2: 36 windows x 81 codes; k = 3: 25 windows x 19,683 codes), code = base-3 digits
//! (0 empty, 1 side to move, 2 opponent) of its cells, tied across the 8 board symmetries (a class = the smallest
//! (window, code) image). Used by `lab hce-train --win "2"` / "3" / "2,3" as a residual group on the frozen NNUE
//! (D1053 protocol): its held-out gain = the window information the champion lacks. Locality (D1111): 2x2 changes 15.6
//! rows per move, 3x3 20.1.
use crate::pairs::transform;

pub(crate) struct Win {
    /// per size in use: (k, windows (cells in row-major order), class table [window * 3^(k*k) + code], class offset)
    sizes: Vec<(usize, Vec<Vec<usize>>, Vec<u32>, u32)>,
    nclass: usize,
    pub(crate) spec: String,
}

fn windows(k: usize) -> Vec<Vec<usize>> {
    let mut v = Vec::new();
    for r in 0..=7 - k {
        for c in 0..=7 - k {
            v.push((0..k * k).map(|i| (r + i / k) * 7 + c + i % k).collect());
        }
    }
    v
}

impl Win {
    pub(crate) fn new(spec: &str) -> Win {
        let mut sizes = Vec::new();
        let mut nclass = 0usize;
        for s in spec.split(',').filter(|s| !s.trim().is_empty()) {
            let k: usize = s.trim().parse().expect("--win: window sizes 2 and/or 3");
            assert!(k == 2 || k == 3, "--win: sizes 2 or 3");
            let ws = windows(k);
            let ncode = 3usize.pow((k * k) as u32);
            // top-left square -> window index
            let tl: std::collections::HashMap<usize, usize> = ws.iter().enumerate().map(|(i, w)| (w[0], i)).collect();
            let mut ids = std::collections::HashMap::new();
            let mut class = vec![0u32; ws.len() * ncode];
            for (wi, cells) in ws.iter().enumerate() {
                for code in 0..ncode {
                    let digit = |i: usize| (code / 3usize.pow((k * k - 1 - i) as u32)) % 3;
                    let key = (0..8)
                        .map(|t| {
                            let img: Vec<usize> = cells.iter().map(|&q| transform(q, t)).collect();
                            let w2 = tl[img.iter().min().unwrap()];
                            let mut c2 = 0usize;
                            for &q2 in &ws[w2] {
                                let i = img.iter().position(|&x| x == q2).expect("D4 maps a window onto a window");
                                c2 = c2 * 3 + digit(i);
                            }
                            (w2, c2)
                        })
                        .min()
                        .unwrap();
                    let n = ids.len() as u32;
                    class[wi * ncode + code] = *ids.entry(key).or_insert(n);
                }
            }
            sizes.push((k, ws, class, nclass as u32));
            nclass += ids.len();
        }
        Win { sizes, nclass, spec: spec.to_string() }
    }

    /// D1142: the class table of the first window size (window * 3^(k*k) + code, row-major digits, first cell most significant)
    pub(crate) fn class_table(&self) -> &[u32] {
        &self.sizes[0].2
    }

    pub(crate) fn dims(&self) -> usize {
        self.nclass
    }

    /// appends (index + off, 1.0) for every window whose code is not all-empty, side to move `me` against `op`
    pub(crate) fn feats(&self, me: u64, op: u64, off: u32, out: &mut Vec<(u32, f32)>) {
        for (k, ws, class, base) in &self.sizes {
            let ncode = 3usize.pow((k * k) as u32);
            for (wi, cells) in ws.iter().enumerate() {
                let mut code = 0usize;
                for &q in cells {
                    code = code * 3 + if (me >> q) & 1 != 0 { 1 } else if (op >> q) & 1 != 0 { 2 } else { 0 };
                }
                if code != 0 {
                    out.push((off + base + class[wi * ncode + code], 1.0));
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn classes_are_d4_invariant() {
        let w = Win::new("2,3");
        let (me, op) = (0x0000_1234_5678_9abc_u64 & ((1u64 << 49) - 1), 0x0001_8000_0300_0400_u64);
        let op = op & !me;
        let mut base = Vec::new();
        w.feats(me, op, 0, &mut base);
        base.sort_by_key(|x| x.0);
        for t in 1..8 {
            let tr = |x: u64| (0..49).filter(|&s| (x >> s) & 1 != 0).fold(0u64, |m, s| m | 1u64 << transform(s, t));
            let mut v = Vec::new();
            w.feats(tr(me), tr(op), 0, &mut v);
            v.sort_by_key(|x| x.0);
            assert_eq!(base, v, "symmetry {t}");
        }
    }
}

//! D1028: gataxx 2.0.0's default static evaluation (FeatureEvaluator safe 4 / reach 4 / edge 8 / tempo 32 / cohesion -3,
//! material 16 per piece, + PairEvaluator built-in pair-patterns.bin at blend 1.0), ported from the Java source
//! (C:/gitProjects/engines-src/aoflaofl__gataxx/src/main/java/spamalot/gataxx/eval/*.java) so it can run inside Zataxx's
//! search for a same-search comparison with our HCE. Units: gataxx score (1/16 piece); `cp()` = score*100/16 as its UAI prints.
//! Walls are not supported (standard Ataxx board only, as in our data).
const SIZE: i32 = 7;
const ALL: u64 = (1u64 << 49) - 1;
const PLACEMENTS: usize = 80;
const MOB: usize = 50 * 50;

fn file_mask(f: u64) -> u64 {
    (0..7u64).fold(0, |m, r| m | 1 << (r * 7 + f))
}
fn north(b: u64) -> u64 {
    (b << 7) & ALL
}
fn south(b: u64) -> u64 {
    b >> 7
}
fn east(b: u64, not_a: u64) -> u64 {
    (b << 1) & not_a
}
fn west(b: u64, not_g: u64) -> u64 {
    (b >> 1) & not_g
}

pub struct GxEval {
    not_a: u64,
    not_g: u64,
    edge: u64,
    weights: Vec<i16>, // PLACEMENTS << 16 entries
    mobility: Vec<i16>,
    tempo: i32,
    scale: i32,
    first: [usize; PLACEMENTS],
    second: [usize; PLACEMENTS],
    digits: [[u32; 16]; 16],
}

impl GxEval {
    fn expand1(&self, b: u64) -> u64 {
        let row = b | east(b, self.not_a) | west(b, self.not_g);
        (row | (row << 7) | (row >> 7)) & ALL
    }
    fn adjacent_pairs_within(&self, a: u64) -> i32 {
        let e = east(a, self.not_a);
        let w = west(a, self.not_g);
        ((a & north(a)).count_ones() + (a & e).count_ones() + (a & north(e)).count_ones() + (a & north(w)).count_ones()) as i32
    }

    /// load gataxx's pair-pattern weight file (big endian, version 3 with mobility table)
    pub fn load(pair_file: &str) -> GxEval {
        let not_a = ALL & !file_mask(0);
        let not_g = ALL & !file_mask(6);
        let edge = (0..49u64).filter(|&s| { let (f, r) = (s % 7, s / 7); f == 0 || f == 6 || r == 0 || r == 6 }).fold(0, |m, s| m | 1 << s);
        // placements, in PairPatterns' static order
        let mut sq = [[0usize; 8]; PLACEMENTS];
        let (mut first, mut second) = ([0usize; PLACEMENTS], [0usize; PLACEMENTS]);
        let mut by_fp = std::collections::HashMap::new();
        let mut p = 0usize;
        let mut add = |ax: i32, ay: i32, bx: i32, by: i32, p: &mut usize| {
            let a = (ay * SIZE + ax) as usize;
            let b = (by * SIZE + bx) as usize;
            first[*p] = a;
            second[*p] = b;
            let s = [a, a + 1, a + 7, a + 8, b, b + 1, b + 7, b + 8];
            sq[*p] = s;
            let fp = s.iter().fold(0u64, |m, &q| m | 1 << q);
            assert!(by_fp.insert(fp, *p).is_none());
            *p += 1;
        };
        for y in 0..6 { for x in 0..=3 { add(x, y, x + 2, y, &mut p); } }
        for y in 0..=3 { for x in 0..6 { add(x, y, x, y + 2, &mut p); } }
        for y in 0..=3 { for x in 0..=3 { add(x, y, x + 2, y + 2, &mut p); } }
        for y in 0..=3 { for x in 0..=3 { add(x, y + 2, x + 2, y, &mut p); } }
        assert_eq!(p, PLACEMENTS);
        let mut digits = [[0u32; 16]; 16];
        for m in 0..16usize {
            for t in 0..16usize {
                let mut idx = 0u32;
                for i in 0..4 {
                    let d = if (m >> i) & 1 != 0 { 1 } else if (t >> i) & 1 != 0 { 2 } else { 0 };
                    idx |= d << (2 * i);
                }
                digits[m][t] = if (m | t) == 0 || (m & t) != 0 { 0 } else { idx };
            }
        }
        let transform = |s: usize, t: usize| -> usize {
            let n = 6i32;
            let (x, y) = ((s % 7) as i32, (s / 7) as i32);
            let (nx, ny) = match t { 0 => (x, y), 1 => (n - y, x), 2 => (n - x, n - y), 3 => (y, n - x), 4 => (n - x, y), 5 => (x, n - y), 6 => (y, x), _ => (n - y, n - x) };
            (ny * 7 + nx) as usize
        };
        let encode = |pl: usize, state_at: &[i32; 49]| -> u32 {
            let mut pat = 0u32;
            for b in 0..2 {
                let (mut idx, mut pieces) = (0u32, false);
                for i in 0..4 {
                    let s = state_at[sq[pl][4 * b + i]];
                    idx |= (s as u32) << (2 * i);
                    pieces |= s == 1 || s == 2;
                }
                pat |= (if pieces { idx } else { 0 }) << (8 * b);
            }
            pat
        };
        let data = std::fs::read(pair_file).unwrap_or_else(|e| panic!("{pair_file}: {e}"));
        let mut pos = 0usize;
        let mut rd = |n: usize| -> &[u8] { let s = &data[pos..pos + n]; pos += n; s };
        let i32be = |b: &[u8]| i32::from_be_bytes([b[0], b[1], b[2], b[3]]);
        assert_eq!(i32be(rd(4)), 0x5041_4952, "not a pair-pattern file");
        let version = i32be(rd(4));
        assert!((1..=3).contains(&version), "version {version}");
        let scale = if version == 1 { 8 } else { i32be(rd(4)) };
        let tb = rd(2);
        let tempo = i16::from_be_bytes([tb[0], tb[1]]) as i32;
        let classes = i32be(rd(4)) as usize;
        let mut weights = vec![0i16; PLACEMENTS << 16];
        for _ in 0..classes {
            let key = i32be(rd(4)) as u32;
            let wb = rd(2);
            let w = i16::from_be_bytes([wb[0], wb[1]]);
            let (pl, pattern) = ((key >> 16) as usize, key & 0xFFFF);
            let mut states = [0i32; 8];
            for b in 0..2 {
                let idx = (pattern >> (8 * b)) & 255;
                for i in 0..4 {
                    states[4 * b + i] = ((idx >> (2 * i)) & 3) as i32;
                }
            }
            for t in 0..8 {
                let mut fp = 0u64;
                let mut mapped = [0usize; 8];
                for i in 0..8 {
                    mapped[i] = transform(sq[pl][i], t);
                    fp |= 1 << mapped[i];
                }
                let image = by_fp[&fp];
                let mut st = [-1i32; 49];
                for i in 0..8 {
                    st[mapped[i]] = states[i];
                }
                weights[image << 16 | encode(image, &st) as usize] = w;
            }
        }
        let mut mobility = vec![0i16; MOB];
        if version >= 3 {
            let entries = i32be(rd(4)) as usize;
            for _ in 0..entries {
                let ib = rd(2);
                let index = u16::from_be_bytes([ib[0], ib[1]]) as usize;
                let vb = rd(2);
                mobility[index] = i16::from_be_bytes([vb[0], vb[1]]);
            }
        }
        GxEval { not_a, not_g, edge, weights, mobility, tempo, scale, first, second, digits }
    }

    fn block(&self, mine: u64, theirs: u64, origin: usize) -> u32 {
        let m = ((mine >> origin) & 3) | (((mine >> (origin + 7)) & 3) << 2);
        let t = ((theirs >> origin) & 3) | (((theirs >> (origin + 7)) & 3) << 2);
        if (m | t) == 0 { 0 } else { self.digits[m as usize][t as usize] }
    }
    fn mobility_index(&self, own: u64, other: u64) -> usize {
        let occ = own | other;
        let phase = occ.count_ones() as usize;
        let targets = (self.expand1(self.expand1(own)) & !occ & ALL).count_ones() as usize;
        phase * 50 + targets
    }

    /// gataxx score (1/16 piece) for the side to move: `mine` = side to move's stones
    pub fn eval(&self, mine: u64, theirs: u64) -> i32 {
        let empty = ALL & !(mine | theirs);
        // FeatureEvaluator (material 16, safe 4, reach 4, edge 8, tempo 32, cohesion -3, fade 0)
        let mut score = 32 + 16 * (mine.count_ones() as i32 - theirs.count_ones() as i32);
        let exposed = self.expand1(empty);
        let mut positional = 4 * ((mine & !exposed).count_ones() as i32 - (theirs & !exposed).count_ones() as i32);
        let my_reach = self.expand1(self.expand1(mine)) & empty;
        let their_reach = self.expand1(self.expand1(theirs)) & empty;
        let my_thr = mine & self.expand1(their_reach);
        let their_thr = theirs & self.expand1(my_reach);
        positional += 4 * (mine.count_ones() as i32 - my_thr.count_ones() as i32 - theirs.count_ones() as i32 + their_thr.count_ones() as i32);
        positional += (-3 * 2 * (self.adjacent_pairs_within(mine) - self.adjacent_pairs_within(theirs))) / 4; // Java int division truncates
        positional += 8 * ((mine & self.edge).count_ones() as i32 - (theirs & self.edge).count_ones() as i32);
        score += positional;
        // PairEvaluator correction, blend 1.0
        let mut blocks = [0u32; 49];
        for y in 0..6 {
            for x in 0..6 {
                let s = y * 7 + x;
                blocks[s] = self.block(mine, theirs, s);
            }
        }
        let mut sum = self.tempo;
        for p in 0..PLACEMENTS {
            sum += self.weights[p << 16 | (blocks[self.first[p]] | blocks[self.second[p]] << 8) as usize] as i32;
        }
        sum += self.mobility[self.mobility_index(mine, theirs)] as i32 - self.mobility[self.mobility_index(theirs, mine)] as i32;
        score + (sum as f64 / self.scale as f64 + 0.5).floor() as i32 // Math.round(blend * sum / scale)
    }

    /// what gataxx's UAI prints: floorDiv(score * 100 + 8, 16)
    pub fn cp(score: i32) -> i32 {
        (score * 100 + 8).div_euclid(16)
    }
}

/// `lab gx-eval --val V.tdat [--n 20]`: gataxx score + cp of the first N val rows (order of `lab val-fen`), for parity vs Java
pub fn cli(args: &[String]) -> std::process::ExitCode {
    let Some(v) = crate::util::arg(args, "--val") else {
        eprintln!("usage: lab gx-eval --val V.tdat [--n 20]");
        return std::process::ExitCode::from(2);
    };
    let n: usize = crate::util::arg(args, "--n").and_then(|s| s.parse().ok()).unwrap_or(20);
    let g = GxEval::load(GX_PAIRS);
    for (i, r) in crate::hcetrain::read_rows(&v, n).iter().enumerate() {
        let (me, op) = crate::extval::own_opp(r);
        let s = g.eval(me, op);
        println!("{i}\t{s}\t{}", GxEval::cp(s));
    }
    std::process::ExitCode::SUCCESS
}

pub const GX_PAIRS: &str = "C:/gitProjects/engines-src/aoflaofl__gataxx/src/main/resources/spamalot/gataxx/eval/pair-patterns.bin";

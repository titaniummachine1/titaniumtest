//! `ti train snarch`: SmallNet ARCHITECTURE research (user D826/D827/D829): the same samples, head and training as
//! `ti train smallnet`, only the INPUT layer changes, so the input designs are compared fairly.
//!
//! The futility SmallNet is a LOWER BOUND of the big net's eval of a search child (pinball loss at tau). A tighter bound
//! (less slack at the same violation rate) prunes more, so the search gets faster. Input designs (`--arch`):
//!   sq     today's engine net: one row per stone and perspective (147 features, own/enemy/blocker x 49 squares);
//!   w2     Zataxx-style: 36 overlapping 2x2 windows, each one of 81 states (empty/us/them per square) = 2,916 features,
//!          always exactly 36 rows per perspective (C:/gitProjects/Zataxx/src/nn.rs);
//!   w2sq   both of the above (36 + stones rows);
//!   delta  the user's parent+move idea (D826): only the 2x2 windows the move CHANGED, each as (child state) and
//!          (parent state) features, so the net learns what the move did; the parent's big-net eval comes in as the skip.
//!          About 9-16 rows instead of 36-49.
//!   dsq / dw2 / dw2sq  (user D835) the delta block PLUS a full child input (sq / w2 / w2sq): the net sees the whole child and
//!          what the move changed ("how much better or worse than the parent").
//! Head (all designs): two perspectives with shared weights -> CReLU -> `--buckets` empties buckets, each L1 (2W -> H)
//! CReLU -> 1, plus per-bucket skips of the parent eval and the child's hand eval (exactly the engine's head at 8 buckets).
//!
//! Report per design: RMSE, violation rate P(bound > truth), mean slack, and the FAIR number: mean slack after shifting the
//! bound so that exactly `tau` of the val children are violated (slack@tau), plus mean active rows per perspective (cost).
//!
//!   ti train snarch --net nnue_main.tnn --train data/nnue/packed/train_v10 --val data/nnue/packed/val_v9_clean \
//!       --arch sq,w2,w2sq,delta [--buckets 8] [--rows 1200000] [--val-rows 60000] [--child-every 3] [--epochs-mse 4]
//!       [--tau 0.02] [--lr 0.002] [--threads 8] [--seed 1] [--save-dir DIR]
//!       pinball phase: ALWAYS to SATURATION, NO epoch limit (LAWS C15, user D868; --patience 4 --max-decays 3 --decay 0.3
//!       --min-rel 2e-4), best epoch restored and saved as DIR/snarch_<arch>_b<buckets>.snar; --max-epochs/--fixed are refused
use std::process::ExitCode;
use std::time::Instant;

use crate::data::Rec;
use crate::mapper;
use crate::train::TrainableNet;
use crate::trainer::{load_val, Tdat};
use titanium::board::MoveList;

const W: usize = 32;
const H: usize = 16;
const D2: usize = 2 * W;
const NWIN: usize = 36;
const WSTATES: usize = 81;

#[derive(Clone, Copy)]
struct Sample {
    /// child, from the child's side to move
    own: u64,
    opp: u64,
    blk: u64,
    /// the parent position in the SAME colours (own = the child mover's stones before the move)
    pown: u64,
    popp: u64,
    parent: i16,
    hand: i16,
    target: i16,
    /// D847 (user D841 minimax): the move's policy log-probability among the parent's legal moves (x1000, bundled policy
    /// head 380F609C exactly as the engine's move ordering calls it) and its policy rank (0 = the policy's best move)
    lp: i16,
    rank: u8,
    /// D851 P2: the parent's big-net disagreement value (cp, exactly the engine's node_dis), the parent's policy entropy
    /// and top-1 log-prob (x1000), and the parent's big-net hidden layers h1||h2 (x127, first HMAX values)
    dis: i16,
    ent: i16,
    top1: i16,
    h: [i8; HMAX],
}

/// D851 P2 input groups, added after the arch's rows (suffixes +pol +dis +mv +ent +h1); every new weight starts at 0
#[derive(Clone, Copy, Default, PartialEq, Debug)]
struct Ext {
    pol: bool,
    dis: bool,
    mv: bool,
    ent: bool,
    h1: bool,
    /// D861 (user): move-source tuple: the 8-neighbour pattern of the square a JUMP left (the hole) or of a CLONE's
    /// destination, plus (mover pieces bordering it x can the side to move reach it) = the weakness the move created
    src: bool,
    /// D863 (user H12) move-relative input: every stone / empty within Chebyshev radius `rel` of the move square(s), as
    /// (role clone-dest / jump-dest / jump-source) x (offset from that square) x (mover / side to move / empty / blocked);
    /// 0 = off, "+rel" = radius 2 (the Ataxx move range), "+relf" = radius 6 (the whole board)
    rel: u8,
    /// D1295 (user): ZX head = the Zataxx net's SHAPE doing the SmallNet job: ONE accumulator of 256 (the side to move's
    /// view), SCReLU, per-bucket linear output + the same skips. Only with the w2 rows; start from a trained ZX net (--zx-init).
    zx: u16,
}

const ZX_W: usize = 256;
const NB_ZX: usize = 8;
const HMAX: usize = 48;
const REL_ROLES: usize = 3;
const REL_ST: usize = 4;
fn rel_noff(r: u8) -> usize {
    let d = 2 * r as usize + 1;
    d * d
}
fn rel_half(r: u8) -> usize {
    REL_ROLES * rel_noff(r) * REL_ST
}
/// D863 +rel rows of a child (indices within one perspective half), pushed onto `out` with offset `base`; returns the count.
fn rel_rows(s: &Sample, r: u8, base: usize, out: &mut Vec<u32>) -> usize {
    let left = s.popp & !s.opp;
    let dest = s.opp & !s.popp & !s.pown;
    if dest.count_ones() != 1 || r == 0 {
        return 0;
    }
    let jump = left != 0;
    let mut anchors = [(0usize, 0usize); 2];
    let mut na = 0;
    anchors[na] = (if jump { 1 } else { 0 }, dest.trailing_zeros() as usize);
    na += 1;
    if jump {
        anchors[na] = (2, left.trailing_zeros() as usize);
        na += 1;
    }
    let ri = r as i32;
    let d = 2 * ri + 1;
    let n0 = out.len();
    for &(role, a) in &anchors[..na] {
        let (ar, ac) = ((a / 7) as i32, (a % 7) as i32);
        for dr in -ri..=ri {
            for dc in -ri..=ri {
                let (rr, cc) = (ar + dr, ac + dc);
                if (dr, dc) == (0, 0) || !(0..7).contains(&rr) || !(0..7).contains(&cc) {
                    continue;
                }
                let q = (rr * 7 + cc) as usize;
                let st = if s.opp >> q & 1 == 1 {
                    0
                } else if s.own >> q & 1 == 1 {
                    1
                } else if s.blk >> q & 1 == 1 {
                    3
                } else {
                    2
                };
                let off = ((dr + ri) * d + (dc + ri)) as usize;
                out.push((base + (role * rel_noff(r) + off) * REL_ST + st) as u32);
            }
        }
    }
    out.len() - n0
}
/// D861 +src: 3^8 neighbour patterns per kind (clone dest / jump source) + 2 kinds x 9 adjacent-mover counts x 3 reach classes
const SRC_PAT: usize = 6561;
const SRC_CNT: usize = 54;
const SRC_HALF: usize = 2 * SRC_PAT + SRC_CNT;
const SRC_NF: usize = 2 * SRC_HALF;
const DIS_BINS: usize = 16;
const MV_ROWS: usize = 18;
const ENT_BINS: usize = 16;

impl Ext {
    fn parse(s: &str) -> Result<(String, Ext), String> {
        let mut parts = s.split('+');
        let base = parts.next().unwrap_or("").to_string();
        let mut e = Ext::default();
        for p in parts {
            match p {
                "pol" => e.pol = true,
                "dis" => e.dis = true,
                "mv" => e.mv = true,
                "ent" => e.ent = true,
                "h1" => e.h1 = true,
                "src" => e.src = true,
                "rel" => e.rel = 2,
                "relf" => e.rel = 6,
                "zx" => e.zx = 256,
                p if p.starts_with("zx") => e.zx = p[2..].parse::<u16>().map_err(|_| format!("bad +{p} in {s}"))?,
                _ => return Err(format!("unknown input group +{p} in {s}")),
            }
        }
        Ok((base, e))
    }
    fn suffix(self) -> String {
        let mut s = String::new();
        match self.zx {
            0 => {}
            256 => s.push_str("+zx"),
            w => s.push_str(&format!("+zx{w}")),
        }
        for (on, n) in [(self.pol, "+pol"), (self.dis, "+dis"), (self.mv, "+mv"), (self.ent, "+ent"), (self.h1, "+h1"), (self.src, "+src")] {
            if on {
                s.push_str(n);
            }
        }
        match self.rel {
            0 => {}
            2 => s.push_str("+rel"),
            r => s.push_str(&format!("+rel{r}")),
        }
        s
    }
    /// extra accumulator rows (both perspectives)
    fn nf(self) -> usize {
        (if self.pol { POL_NF } else { 0 }) + (if self.dis { 2 * DIS_BINS } else { 0 }) + (if self.mv { 2 * MV_ROWS } else { 0 }) + (if self.ent { 4 * ENT_BINS } else { 0 }) + (if self.src { SRC_NF } else { 0 }) + 2 * rel_half(self.rel) * (self.rel > 0) as usize
    }
    /// extra row updates per child per perspective in the engine (h1 is per node: 0 per child)
    fn touched(self) -> usize {
        2 * self.pol as usize + self.dis as usize + self.mv as usize + 2 * self.ent as usize + 2 * self.src as usize
    }
}

/// All squares at Chebyshev distance 1..=d of `sq` (d = 1: clone reach, d = 2: clone + jump reach).
fn ring_mask(sq: usize, d: i32) -> u64 {
    let (r, c) = ((sq / 7) as i32, (sq % 7) as i32);
    let mut m = 0u64;
    for dr in -d..=d {
        for dc in -d..=d {
            let (rr, cc) = (r + dr, c + dc);
            if (dr, dc) != (0, 0) && (0..7).contains(&rr) && (0..7).contains(&cc) {
                m |= 1u64 << (rr * 7 + cc);
            }
        }
    }
    m
}

/// The 8 neighbours of `sq` in fixed order (row-major, centre skipped); None = off the board.
fn nb8(sq: usize) -> [Option<usize>; 8] {
    let (r, c) = ((sq / 7) as i32, (sq % 7) as i32);
    let mut o = [None; 8];
    let mut k = 0;
    for dr in -1..=1 {
        for dc in -1..=1 {
            if (dr, dc) == (0, 0) {
                continue;
            }
            let (rr, cc) = (r + dr, c + dc);
            o[k] = if (0..7).contains(&rr) && (0..7).contains(&cc) { Some((rr * 7 + cc) as usize) } else { None };
            k += 1;
        }
    }
    o
}

/// Base-3 code of a list of cells: 0 = empty / blocked / off-board, 1 = `us`, 2 = `them` (same states as the 2x2 windows).
fn code3(cells: impl Iterator<Item = Option<usize>>, us: u64, them: u64) -> usize {
    let mut c = 0;
    let mut p = 1;
    for q in cells {
        let s = match q {
            Some(q) if us >> q & 1 == 1 => 1,
            Some(q) if them >> q & 1 == 1 => 2,
            _ => 0,
        };
        c += s * p;
        p *= 3;
    }
    c
}

/// D861 +src rows of a child: (pattern index in 0..2*SRC_PAT, count index in 0..SRC_CNT), None for a pass.
/// The mover is the child's OPPONENT (`opp`). Jump: the square it left (popp & !opp); clone: its destination.
/// Pattern = the 8 neighbours as 0 empty / 1 mover / 2 side to move; count = mover pieces bordering the square x reach
/// (2 = the side to move can clone there, 1 = only jump there, 0 = cannot reach it).
fn src_rows(s: &Sample) -> Option<(usize, usize)> {
    let left = s.popp & !s.opp;
    let dest = s.opp & !s.popp & !s.pown;
    if dest.count_ones() != 1 {
        return None;
    }
    let jump = left != 0;
    let sq = if jump { left.trailing_zeros() as usize } else { dest.trailing_zeros() as usize };
    let pat = code3(nb8(sq).into_iter(), s.opp, s.own);
    let adj = (ring_mask(sq, 1) & s.opp).count_ones() as usize;
    let reach = if ring_mask(sq, 1) & s.own != 0 { 2 } else if ring_mask(sq, 2) & s.own != 0 { 1 } else { 0 };
    Some((jump as usize * SRC_PAT + pat, jump as usize * 27 + adj * 3 + reach))
}

/// D861 (user) "artificial 3x3": per centre square a CROSS (centre + N S W E) and an X (centre + 4 diagonals), 5 cells each.
const NX3: usize = 49 * 2;
const X3S: usize = 243;
fn x3_cells(k: usize) -> [Option<usize>; 5] {
    let c = k / 2;
    let (r, col) = ((c / 7) as i32, (c % 7) as i32);
    let d: [(i32, i32); 4] = if k % 2 == 0 { [(-1, 0), (1, 0), (0, -1), (0, 1)] } else { [(-1, -1), (-1, 1), (1, -1), (1, 1)] };
    let mut o = [Some(c), None, None, None, None];
    for (i, (dr, dc)) in d.iter().enumerate() {
        let (rr, cc) = (r + dr, col + dc);
        if (0..7).contains(&rr) && (0..7).contains(&cc) {
            o[i + 1] = Some((rr * 7 + cc) as usize);
        }
    }
    o
}
fn x3_mask(k: usize) -> u64 {
    x3_cells(k).iter().flatten().fold(0u64, |m, &s| m | 1u64 << s)
}

/// D861 (user) full 3x3 windows: 25 overlapping windows x 3^9 states.
const NW3: usize = 25;
const W3S: usize = 19683;
fn w3_squares(w: usize) -> [usize; 9] {
    let (i, j) = (w / 5, w % 5);
    let a = i * 7 + j;
    [a, a + 1, a + 2, a + 7, a + 8, a + 9, a + 14, a + 15, a + 16]
}
fn w3_mask(w: usize) -> u64 {
    w3_squares(w).iter().fold(0u64, |m, &s| m | 1u64 << s)
}

fn dis_bin(d: i16) -> usize {
    ((d.max(0) as usize) / 50).min(DIS_BINS - 1)
}
fn ent_bin(x: i16) -> usize {
    // 0.25-nat bins of an entropy (x1000) or of -log p (x1000)
    ((x.unsigned_abs() as usize) / 250).min(ENT_BINS - 1)
}
/// move features from the bits alone: clone/jump x flips (0..8): the mover is the child's OPPONENT (it just moved)
fn mv_row(s: &Sample) -> usize {
    let jump = (s.popp & !s.opp) != 0;
    let flips = ((s.pown & s.opp).count_ones() as usize).min(8);
    jump as usize * 9 + flips
}

const POL_RANKS: usize = 16;
const POL_BINS: usize = 32;
/// extra accumulator rows of the policy input: (rank one-hot + log-prob bin) x 2 perspectives (separate tables)
const POL_NF: usize = 2 * (POL_RANKS + POL_BINS);

fn pol_bin(lp: i16) -> usize {
    // 0.5-nat bins of -log p: bin 0 = p > 0.61, bin 31 = p < 2e-7
    (((-(lp as f32) / 1000.0) * 2.0) as usize).min(POL_BINS - 1)
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Arch {
    Sq,
    W2,
    W2Sq,
    Delta,
    /// D835 (user): delta block + a full child input
    DSq,
    DW2,
    DW2Sq,
    /// D861 (user): cross + X per centre (artificial 3x3)
    X3,
    /// D861 (user): full 3x3 windows
    W3,
}

impl Arch {
    fn parse(s: &str) -> Option<Arch> {
        Some(match s {
            "sq" => Arch::Sq,
            "w2" => Arch::W2,
            "w2sq" => Arch::W2Sq,
            "delta" => Arch::Delta,
            "dsq" => Arch::DSq,
            "dw2" => Arch::DW2,
            "dw2sq" => Arch::DW2Sq,
            "x3" => Arch::X3,
            "w3" => Arch::W3,
            _ => return None,
        })
    }
    fn name(self) -> &'static str {
        match self {
            Arch::Sq => "sq",
            Arch::W2 => "w2",
            Arch::W2Sq => "w2sq",
            Arch::Delta => "delta",
            Arch::DSq => "dsq",
            Arch::DW2 => "dw2",
            Arch::DW2Sq => "dw2sq",
            Arch::X3 => "x3",
            Arch::W3 => "w3",
        }
    }
    /// The full-child-input part of a design (None for pure delta) and whether it carries the delta block (D835).
    fn split(self) -> (Option<Arch>, bool) {
        match self {
            Arch::Sq | Arch::W2 | Arch::W2Sq | Arch::X3 | Arch::W3 => (Some(self), false),
            Arch::Delta => (None, true),
            Arch::DSq => (Some(Arch::Sq), true),
            Arch::DW2 => (Some(Arch::W2), true),
            Arch::DW2Sq => (Some(Arch::W2Sq), true),
        }
    }
    fn nf(self) -> usize {
        let (base, d) = self.split();
        let b = match base {
            Some(Arch::Sq) => 147,
            Some(Arch::W2) => NWIN * WSTATES,
            Some(Arch::W2Sq) => NWIN * WSTATES + 147,
            Some(Arch::X3) => NX3 * X3S,
            Some(Arch::W3) => NW3 * W3S,
            _ => 0,
        };
        b + if d { DELTA_NF } else { 0 }
    }
    /// expected active rows per perspective (for the init scale)
    fn rows(self) -> f32 {
        let (base, d) = self.split();
        let b = match base {
            Some(Arch::Sq) => 25.0,
            Some(Arch::W2) => 36.0,
            Some(Arch::W2Sq) => 61.0,
            Some(Arch::X3) => NX3 as f32,
            Some(Arch::W3) => NW3 as f32,
            _ => 0.0,
        };
        b + if d { 24.0 } else { 0.0 }
    }
}

const DELTA_NF: usize = 2 * NWIN * WSTATES;

/// Squares of 2x2 window w (row i, col j = 0..6).
fn win_squares(w: usize) -> [usize; 4] {
    let (i, j) = (w / 6, w % 6);
    let a = i * 7 + j;
    [a, a + 1, a + 7, a + 8]
}

fn win_code(w: usize, us: u64, them: u64) -> usize {
    let mut c = 0;
    let mut p = 1;
    for sq in win_squares(w) {
        let s = if us >> sq & 1 == 1 {
            1
        } else if them >> sq & 1 == 1 {
            2
        } else {
            0
        };
        c += s * p;
        p *= 3;
    }
    c
}

fn win_mask(w: usize) -> u64 {
    win_squares(w).iter().fold(0u64, |m, &s| m | 1u64 << s)
}

/// Active features of one perspective: `us`/`them` are the stones as that perspective sees them.
fn feats(arch: Arch, us: u64, them: u64, blk: u64, pus: u64, pthem: u64, out: &mut Vec<u32>) {
    out.clear();
    let sq = |out: &mut Vec<u32>, off: usize| {
        let mut b = us;
        while b != 0 {
            out.push((off + b.trailing_zeros() as usize) as u32);
            b &= b - 1;
        }
        let mut b = them;
        while b != 0 {
            out.push((off + 49 + b.trailing_zeros() as usize) as u32);
            b &= b - 1;
        }
        let mut b = blk;
        while b != 0 {
            out.push((off + 98 + b.trailing_zeros() as usize) as u32);
            b &= b - 1;
        }
    };
    // layout: [delta block (2 x 2916) if any][full child input]
    let (base, delta) = arch.split();
    let mut off = 0;
    if delta {
        let changed = (us ^ pus) | (them ^ pthem);
        for w in 0..NWIN {
            if win_mask(w) & changed != 0 {
                out.push((w * WSTATES + win_code(w, us, them)) as u32);
                out.push((NWIN * WSTATES + w * WSTATES + win_code(w, pus, pthem)) as u32);
            }
        }
        off = DELTA_NF;
    }
    match base {
        Some(Arch::Sq) => sq(out, off),
        Some(b @ (Arch::W2 | Arch::W2Sq)) => {
            for w in 0..NWIN {
                out.push((off + w * WSTATES + win_code(w, us, them)) as u32);
            }
            if b == Arch::W2Sq {
                sq(out, off + NWIN * WSTATES);
            }
        }
        Some(Arch::X3) => {
            for k in 0..NX3 {
                out.push((off + k * X3S + code3(x3_cells(k).into_iter(), us, them)) as u32);
            }
        }
        Some(Arch::W3) => {
            for w in 0..NW3 {
                out.push((off + w * W3S + code3(w3_squares(w).into_iter().map(Some), us, them)) as u32);
            }
        }
        _ => {}
    }
}

struct Model {
    arch: Arch,
    /// D847/D851: extra input groups (rows after the arch's rows in the order pol, dis, mv, ent; h1 = dense projection)
    ext: Ext,
    /// first feature index of each extra row group
    o_pol: usize,
    o_dis: usize,
    o_mv: usize,
    o_ent: usize,
    o_src: usize,
    o_rel: usize,
    nb: usize,
    p: Vec<f32>,
    o_ftb: usize,
    o_l1w: usize,
    o_l1b: usize,
    o_ow: usize,
    o_ob: usize,
    /// per-bucket skips: [parent, hand, policy log-prob, disagreement] x nb
    o_sk: usize,
    /// h1: dense [2 halves][HMAX][W] projection of the parent's big-net hidden layers into the accumulator
    o_wh: usize,
    np: usize,
    /// accumulator width (W = 32 for the engine's head, ZX_W for the ZX head) and the head kind (D1295)
    w: usize,
    zx: bool,
}

struct Fwd {
    x: [f32; D2],
    hpre: [f32; H],
    h: [f32; H],
    y: f32,
    bk: usize,
    fu: Vec<u32>,
    ft: Vec<u32>,
    /// ZX head only: the clamped accumulator (before squaring)
    xz: Vec<f32>,
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn unit(&mut self) -> f32 {
        ((self.next() >> 11) as f64 / (1u64 << 53) as f64) as f32 * 2.0 - 1.0
    }
}

impl Model {
    fn new(arch: Arch, nb: usize, seed: u64, ext: Ext) -> Model {
        let pol_off = arch.nf();
        let o_pol = pol_off;
        let o_dis = o_pol + if ext.pol { POL_NF } else { 0 };
        let o_mv = o_dis + if ext.dis { 2 * DIS_BINS } else { 0 };
        let o_ent = o_mv + if ext.mv { 2 * MV_ROWS } else { 0 };
        let o_src = o_ent + if ext.ent { 4 * ENT_BINS } else { 0 };
        let o_rel = o_src + if ext.src { SRC_NF } else { 0 };
        let nf = pol_off + ext.nf();
        debug_assert_eq!(o_rel + if ext.rel > 0 { 2 * rel_half(ext.rel) } else { 0 }, nf);
        let zx = ext.zx > 0;
        let w = if zx { ext.zx as usize } else { W };
        let o_ftb = nf * w;
        let o_l1w = o_ftb + w;
        // ZX head: o_l1w holds the per-bucket output weights [nb][w]; there is no L1 layer (o_l1b = o_ow = o_ob)
        let o_l1b = if zx { o_l1w + nb * w } else { o_l1w + nb * H * D2 };
        let o_ow = if zx { o_l1b } else { o_l1b + nb * H };
        let o_ob = if zx { o_ow } else { o_ow + nb * H };
        let o_sk = o_ob + nb;
        let o_wh = o_sk + 4 * nb;
        let np = o_wh + if ext.h1 { 2 * HMAX * W } else { 0 };
        let mut r = Rng(seed | 1);
        let mut p = vec![0f32; np];
        // a feature's rows start small: the window designs have more rows per feature family than sq, scale by expected
        // active rows so the accumulator starts in the same range
        let scale = 0.15 * (25.0f32 / arch.rows()).sqrt();
        // D847: only the arch's rows draw from the RNG; policy rows and the policy skip start at 0, so a +pol model is
        // EXACTLY its control model at step 0 (paired comparison: any difference comes from the policy input)
        for v in &mut p[..pol_off * w] {
            *v = r.unit() * scale;
        }
        for v in &mut p[o_ftb..o_l1w] {
            *v = 0.25;
        }
        let fan = if zx { w } else { D2 };
        for v in &mut p[o_l1w..o_l1b] {
            *v = r.unit() * (3.0 / fan as f32).sqrt();
        }
        for v in &mut p[o_l1b..o_ow] {
            *v = 0.5;
        }
        for v in &mut p[o_ow..o_ob] {
            *v = r.unit() * 0.3;
        }
        for k in 0..nb {
            p[o_sk + k] = 0.8;
        }
        Model { arch, ext, o_pol, o_dis, o_mv, o_ent, o_src, o_rel, nb, p, o_ftb, o_l1w, o_l1b, o_ow, o_ob, o_sk, o_wh, np, w, zx }
    }

    /// D1295 (C14: never from random weights): start the ZX head from a TRAINED Zataxx-format net (i16, 2916 x 256 rows, 256
    /// bias, 256 output weights, 1 output bias; QA 255, QB 64). Step 0 = that net exactly (the same output for every bucket;
    /// output is cp/400, the unit of this trainer): the parent / hand skips start at 0 so nothing else is mixed in.
    fn load_zx(&mut self, path: &str) -> Result<(), String> {
        let b = std::fs::read(path).map_err(|e| format!("{path}: {e}"))?;
        let n = 2916 * 256;
        if !self.zx || self.w != ZX_W || self.arch != Arch::W2 || b.len() != (n + 256 + 256 + 1) * 2 {
            return Err(format!("{path}: --zx-init needs the w2+zx design and a Zataxx-format net ({} bytes)", (n + 513) * 2));
        }
        let v: Vec<f32> = b.chunks_exact(2).map(|c| i16::from_le_bytes([c[0], c[1]]) as f32).collect();
        for i in 0..n {
            self.p[i] = v[i] / 255.0;
        }
        for k in 0..ZX_W {
            self.p[self.o_ftb + k] = v[n + k] / 255.0;
        }
        for bk in 0..self.nb {
            for k in 0..ZX_W {
                self.p[self.o_l1w + bk * ZX_W + k] = v[n + 256 + k] / 64.0;
            }
            self.p[self.o_ob + bk] = v[n + 512] / (255.0 * 64.0);
            self.p[self.o_sk + bk] = 0.0;
            self.p[self.o_sk + self.nb + bk] = 0.0;
        }
        Ok(())
    }

    /// D1296: downscale a ZX head to `w` units. Keep the units with the largest (bucket-mean |output weight|) x (mean
    /// squared activation `msq` on real children); every dropped unit's MEAN contribution (output weight x msq) is folded
    /// into that bucket's output bias, so the narrow net starts as the wide one with the dropped units at their average.
    fn shrink_zx(&self, w: usize, msq: &[f64]) -> Model {
        let (wf, nb) = (self.w, self.nb);
        let imp = |k: usize| -> f64 { (0..nb).map(|bk| self.p[self.o_l1w + bk * wf + k].abs() as f64).sum::<f64>() / nb as f64 * msq[k] };
        let mut order: Vec<usize> = (0..wf).collect();
        order.sort_by(|&a, &b| imp(b).partial_cmp(&imp(a)).unwrap_or(std::cmp::Ordering::Equal).then(a.cmp(&b)));
        let mut keep: Vec<usize> = order[..w].to_vec();
        keep.sort_unstable();
        let mut m = Model::new(self.arch, nb, 1, Ext { zx: w as u16, ..self.ext });
        let nf = self.o_ftb / wf;
        for f in 0..nf {
            for (j, &k) in keep.iter().enumerate() {
                m.p[f * w + j] = self.p[f * wf + k];
            }
        }
        for (j, &k) in keep.iter().enumerate() {
            m.p[m.o_ftb + j] = self.p[self.o_ftb + k];
        }
        let kept: std::collections::HashSet<usize> = keep.iter().copied().collect();
        for bk in 0..nb {
            let mut ob = self.p[self.o_ob + bk];
            for k in 0..wf {
                if !kept.contains(&k) {
                    ob += self.p[self.o_l1w + bk * wf + k] * msq[k] as f32;
                }
            }
            m.p[m.o_ob + bk] = ob;
            for (j, &k) in keep.iter().enumerate() {
                m.p[m.o_l1w + bk * w + j] = self.p[self.o_l1w + bk * wf + k];
            }
        }
        let (a, b) = (self.o_sk, m.o_sk);
        for i in 0..4 * nb {
            m.p[b + i] = self.p[a + i];
        }
        m
    }

    /// D1313: load a trained ZX head (SNAR) of the same width; when the file has half this model's buckets, bucket b of
    /// this model takes the file's bucket b / 2 (output weights, bias, skips): the same output at step 0.
    fn load_expand(&mut self, path: &str) -> Result<(), String> {
        let d = std::fs::read(path).map_err(|e| format!("{path}: {e}"))?;
        if d.len() < 24 || &d[..4] != b"SNAR" || u32::from_le_bytes(d[4..8].try_into().unwrap()) != 2 {
            return Err(format!("{path}: not a SNAR v2 file"));
        }
        let nl = u32::from_le_bytes(d[8..12].try_into().unwrap()) as usize;
        let fnb = u32::from_le_bytes(d[12 + nl..16 + nl].try_into().unwrap()) as usize;
        if fnb == self.nb {
            return load_model(self, path);
        }
        if !self.zx || fnb == 0 || self.nb % fnb != 0 {
            return Err(format!("{path}: has {fnb} buckets, this run {} (a ZX head expands only by a whole factor)", self.nb));
        }
        let k = self.nb / fnb;
        let mut src = Model::new(self.arch, fnb, 1, self.ext);
        load_model(&mut src, path)?;
        let w = self.w;
        self.p[..self.o_l1w].copy_from_slice(&src.p[..src.o_l1w]);
        for b in 0..self.nb {
            let s = b / k;
            self.p[self.o_l1w + b * w..self.o_l1w + (b + 1) * w].copy_from_slice(&src.p[src.o_l1w + s * w..src.o_l1w + (s + 1) * w]);
            self.p[self.o_ob + b] = src.p[src.o_ob + s];
            for g in 0..4 {
                self.p[self.o_sk + g * self.nb + b] = src.p[src.o_sk + g * src.nb + s];
            }
        }
        Ok(())
    }

    fn forward_zx(&self, s: &Sample) -> Fwd {
        let p = &self.p;
        let w = self.w;
        let mut fu = Vec::with_capacity(48);
        let mut ft = Vec::new();
        feats(self.arch, s.own, s.opp, s.blk, s.pown, s.popp, &mut fu);
        self.ext_rows(s, &mut fu, &mut ft);
        let mut a = p[self.o_ftb..self.o_ftb + w].to_vec();
        for &f in &fu {
            let r = &p[f as usize * w..(f as usize + 1) * w];
            for j in 0..w {
                a[j] += r[j];
            }
        }
        for v in a.iter_mut() {
            *v = v.clamp(0.0, 1.0);
        }
        let bk = self.bucket(s);
        let mut y = p[self.o_ob + bk] + p[self.o_sk + bk] * (s.parent as f32 / 400.0) + p[self.o_sk + self.nb + bk] * (s.hand as f32 / 400.0);
        if self.ext.pol {
            y += p[self.o_sk + 2 * self.nb + bk] * (s.lp as f32 / 4000.0);
        }
        if self.ext.dis {
            y += p[self.o_sk + 3 * self.nb + bk] * (s.dis as f32 / 400.0);
        }
        let zo = &p[self.o_l1w + bk * w..self.o_l1w + (bk + 1) * w];
        for j in 0..w {
            y += zo[j] * a[j] * a[j];
        }
        Fwd { x: [0.0; D2], hpre: [0.0; H], h: [0.0; H], y, bk, fu, ft, xz: a }
    }

    fn backward_zx(&self, s: &Sample, f: &Fwd, dy: f32, g: &mut [f32]) {
        let p = &self.p;
        let (w, bk) = (self.w, f.bk);
        g[self.o_ob + bk] += dy;
        g[self.o_sk + bk] += dy * (s.parent as f32 / 400.0);
        g[self.o_sk + self.nb + bk] += dy * (s.hand as f32 / 400.0);
        if self.ext.pol {
            g[self.o_sk + 2 * self.nb + bk] += dy * (s.lp as f32 / 4000.0);
        }
        if self.ext.dis {
            g[self.o_sk + 3 * self.nb + bk] += dy * (s.dis as f32 / 400.0);
        }
        let mut da = vec![0f32; w];
        for j in 0..w {
            let c = f.xz[j];
            g[self.o_l1w + bk * w + j] += dy * c * c;
            if c > 0.0 && c < 1.0 {
                da[j] = dy * p[self.o_l1w + bk * w + j] * 2.0 * c;
            }
        }
        for j in 0..w {
            g[self.o_ftb + j] += da[j];
        }
        for &ff in &f.fu {
            let b = ff as usize * w;
            for j in 0..w {
                g[b + j] += da[j];
            }
        }
    }

    fn name(&self) -> String {
        format!("{}{}", self.arch.name(), self.ext.suffix())
    }

    fn bucket(&self, s: &Sample) -> usize {
        let empties = 49 - (s.own | s.opp | s.blk).count_ones() as usize;
        (empties * self.nb / 50).min(self.nb - 1)
    }

    fn forward(&self, s: &Sample) -> Fwd {
        if self.zx {
            return self.forward_zx(s);
        }
        let p = &self.p;
        let mut fu = Vec::with_capacity(64);
        let mut ft = Vec::with_capacity(64);
        feats(self.arch, s.own, s.opp, s.blk, s.pown, s.popp, &mut fu);
        feats(self.arch, s.opp, s.own, s.blk, s.popp, s.pown, &mut ft);
        self.ext_rows(s, &mut fu, &mut ft);
        self.forward_rest(s, fu, ft)
    }

    /// the extra input rows (policy, disagreement, move, entropy, source, relative) of both perspectives
    fn ext_rows(&self, s: &Sample, fu: &mut Vec<u32>, ft: &mut Vec<u32>) {
        let e = self.ext;
        if e.pol {
            // D847: the move's policy rank + log-prob bin, one row each per perspective (separate tables per half)
            let (r, l) = ((s.rank as usize).min(POL_RANKS - 1), pol_bin(s.lp));
            let o = self.o_pol;
            fu.push((o + r) as u32);
            fu.push((o + POL_RANKS + l) as u32);
            ft.push((o + POL_RANKS + POL_BINS + r) as u32);
            ft.push((o + 2 * POL_RANKS + POL_BINS + l) as u32);
        }
        if e.dis {
            let d = dis_bin(s.dis);
            fu.push((self.o_dis + d) as u32);
            ft.push((self.o_dis + DIS_BINS + d) as u32);
        }
        if e.mv {
            let r = mv_row(s);
            fu.push((self.o_mv + r) as u32);
            ft.push((self.o_mv + MV_ROWS + r) as u32);
        }
        if e.ent {
            let (en, t1) = (ent_bin(s.ent), ent_bin(s.top1));
            fu.push((self.o_ent + en) as u32);
            fu.push((self.o_ent + ENT_BINS + t1) as u32);
            ft.push((self.o_ent + 2 * ENT_BINS + en) as u32);
            ft.push((self.o_ent + 3 * ENT_BINS + t1) as u32);
        }
        if e.src {
            // D861: the move-source tuple (pattern + count row), separate tables per perspective half
            if let Some((pt, ct)) = src_rows(s) {
                let (o, o2) = (self.o_src, self.o_src + SRC_HALF);
                fu.push((o + pt) as u32);
                fu.push((o + 2 * SRC_PAT + ct) as u32);
                ft.push((o2 + pt) as u32);
                ft.push((o2 + 2 * SRC_PAT + ct) as u32);
            }
        }
        if e.rel > 0 {
            // D863: move-relative rows, separate tables per perspective half
            rel_rows(s, e.rel, self.o_rel, fu);
            rel_rows(s, e.rel, self.o_rel + rel_half(e.rel), ft);
        }
    }

    fn forward_rest(&self, s: &Sample, fu: Vec<u32>, ft: Vec<u32>) -> Fwd {
        let p = &self.p;
        let e = self.ext;
        let mut a = [0f32; D2];
        a[..W].copy_from_slice(&p[self.o_ftb..self.o_ftb + W]);
        a[W..].copy_from_slice(&p[self.o_ftb..self.o_ftb + W]);
        if e.h1 {
            // D851 I5: the parent's big-net hidden layers, projected densely into both halves (per node in the engine)
            for (k, &hv) in s.h.iter().enumerate() {
                if hv == 0 {
                    continue;
                }
                let hk = hv as f32 / 127.0;
                let ru = &p[self.o_wh + k * W..self.o_wh + (k + 1) * W];
                let rt = &p[self.o_wh + (HMAX + k) * W..self.o_wh + (HMAX + k + 1) * W];
                for j in 0..W {
                    a[j] += ru[j] * hk;
                    a[W + j] += rt[j] * hk;
                }
            }
        }
        for &f in &fu {
            let r = &p[f as usize * W..(f as usize + 1) * W];
            for j in 0..W {
                a[j] += r[j];
            }
        }
        for &f in &ft {
            let r = &p[f as usize * W..(f as usize + 1) * W];
            for j in 0..W {
                a[W + j] += r[j];
            }
        }
        let mut x = [0f32; D2];
        for j in 0..D2 {
            x[j] = a[j].clamp(0.0, 1.0);
        }
        let bk = self.bucket(s);
        let mut hpre = [0f32; H];
        let mut h = [0f32; H];
        let mut y = p[self.o_ob + bk] + p[self.o_sk + bk] * (s.parent as f32 / 400.0) + p[self.o_sk + self.nb + bk] * (s.hand as f32 / 400.0);
        if e.pol {
            y += p[self.o_sk + 2 * self.nb + bk] * (s.lp as f32 / 4000.0);
        }
        if e.dis {
            y += p[self.o_sk + 3 * self.nb + bk] * (s.dis as f32 / 400.0);
        }
        for i in 0..H {
            let w = &p[self.o_l1w + (bk * H + i) * D2..self.o_l1w + (bk * H + i + 1) * D2];
            let mut z = p[self.o_l1b + bk * H + i];
            for j in 0..D2 {
                z += w[j] * x[j];
            }
            hpre[i] = z;
            h[i] = z.clamp(0.0, 1.0);
            y += p[self.o_ow + bk * H + i] * h[i];
        }
        Fwd { x, hpre, h, y, bk, fu, ft, xz: Vec::new() }
    }

    fn backward(&self, s: &Sample, f: &Fwd, dy: f32, g: &mut [f32]) {
        if self.zx {
            return self.backward_zx(s, f, dy, g);
        }
        let p = &self.p;
        let bk = f.bk;
        g[self.o_ob + bk] += dy;
        g[self.o_sk + bk] += dy * (s.parent as f32 / 400.0);
        g[self.o_sk + self.nb + bk] += dy * (s.hand as f32 / 400.0);
        if self.ext.pol {
            g[self.o_sk + 2 * self.nb + bk] += dy * (s.lp as f32 / 4000.0);
        }
        if self.ext.dis {
            g[self.o_sk + 3 * self.nb + bk] += dy * (s.dis as f32 / 400.0);
        }
        let mut dx = [0f32; D2];
        for i in 0..H {
            g[self.o_ow + bk * H + i] += dy * f.h[i];
            if f.hpre[i] > 0.0 && f.hpre[i] < 1.0 {
                let dz = dy * p[self.o_ow + bk * H + i];
                g[self.o_l1b + bk * H + i] += dz;
                let base = self.o_l1w + (bk * H + i) * D2;
                for j in 0..D2 {
                    g[base + j] += dz * f.x[j];
                    dx[j] += dz * p[base + j];
                }
            }
        }
        let mut da = [0f32; D2];
        for j in 0..D2 {
            if f.x[j] > 0.0 && f.x[j] < 1.0 {
                da[j] = dx[j];
            }
        }
        for j in 0..W {
            g[self.o_ftb + j] += da[j] + da[W + j];
        }
        for &ff in &f.fu {
            let b = ff as usize * W;
            for j in 0..W {
                g[b + j] += da[j];
            }
        }
        for &ff in &f.ft {
            let b = ff as usize * W;
            for j in 0..W {
                g[b + j] += da[W + j];
            }
        }
        if self.ext.h1 {
            for (k, &hv) in s.h.iter().enumerate() {
                if hv == 0 {
                    continue;
                }
                let hk = hv as f32 / 127.0;
                let (bu, bt) = (self.o_wh + k * W, self.o_wh + (HMAX + k) * W);
                for j in 0..W {
                    g[bu + j] += da[j] * hk;
                    g[bt + j] += da[W + j] * hk;
                }
            }
        }
    }
}

/// D1296: mean squared activation of every hidden unit of a ZX head over `set`.
fn zx_mean_sq(m: &Model, set: &[Sample]) -> Vec<f64> {
    let mut acc = vec![0f64; m.w];
    for s in set {
        let f = m.forward(s);
        for (a, x) in acc.iter_mut().zip(&f.xz) {
            *a += (x * x) as f64;
        }
    }
    acc.iter().map(|v| v / set.len().max(1) as f64).collect()
}

fn build_samples(rows: &[Rec], big: &TrainableNet, every: u64, threads: usize, seed: u64) -> Vec<Sample> {
    let part = (rows.len() + threads - 1) / threads.max(1);
    let parts: Vec<Vec<Sample>> = std::thread::scope(|sc| {
        let hs: Vec<_> = rows
            .chunks(part.max(1))
            .enumerate()
            .map(|(ti, chunk)| {
                sc.spawn(move || {
                    let mut out = Vec::new();
                    for (li, r) in chunk.iter().enumerate() {
                        // D845: the D4 mode depends only on (seed, global row index), not on the thread split, so every
                        // machine and every --threads value builds the SAME samples (checked by the sample checksum)
                        let gi = (ti * part + li) as u64;
                        let mut rng = Rng((seed ^ (gi + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15)) | 1);
                        rng.next();
                        let mode = (rng.next() % mapper::MODES as u64) as usize;
                        let b = mapper::board_of(r, mode);
                        if b.game_over() {
                            continue;
                        }
                        // D851: one big-net forward of the parent gives its eval, disagreement value and hidden layers
                        let pf = big.forward_full(&b);
                        let parent = (400.0 * pf.y).round().clamp(-30000.0, 30000.0) as i16;
                        let dis = if big.has_dis {
                            let dim = big.l1 + big.l2;
                            let row = &big.dis_w[pf.bucket * dim..(pf.bucket + 1) * dim];
                            let d: f32 = row.iter().zip(pf.h1.iter().chain(pf.h2.iter())).map(|(a, b)| a * b).sum::<f32>() + big.dis_b[pf.bucket];
                            ((400.0 * d.max(0.0)).round() as i32).clamp(0, 30000) as i16
                        } else {
                            0
                        };
                        let mut hq = [0i8; HMAX];
                        for (k, v) in pf.h1.iter().chain(pf.h2.iter()).take(HMAX).enumerate() {
                            hq[k] = (v.clamp(0.0, 1.0) * 127.0).round() as i8;
                        }
                        let mut list = MoveList::new();
                        b.legal_moves_dedup_into(&mut list);
                        // D847/D892: policy log-softmax + rank over the legal moves. D892 FIX: the logit the SEARCH's move
                        // ordering computes (on main since 0120fd12 the SEE path is deleted, so logit_k IS the ordering logit, search.rs:2918; main runs pol_see_min 99 = SEE off since D751, policy
                        // dim 0, big net not pairwise -> no xin). D847-D891 used logit_k WITH the SEE term = an input the
                        // engine never computes (MISTAKES 2026-11-15).
                        let n = list.len();
                        let mut lg = vec![0f32; n];
                        if let Some(pol) = titanium::policy::global() {
                            let mut kc = titanium::policy::KCache::new();
                            for (i, l) in lg.iter_mut().enumerate() {
                                *l = pol.logit_k(&b, list.move_at(i), &mut kc);
                            }
                        }
                        let mx = lg.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                        let lse = mx + lg.iter().map(|&x| (x - mx).exp()).sum::<f32>().ln();
                        let mut ord: Vec<usize> = (0..n).collect();
                        ord.sort_by(|&a, &c| lg[c].partial_cmp(&lg[a]).unwrap_or(std::cmp::Ordering::Equal).then(a.cmp(&c)));
                        let mut rk = vec![0u8; n];
                        for (r, &i) in ord.iter().enumerate() {
                            rk[i] = r.min(255) as u8;
                        }
                        // parent policy shape (D851 I7): entropy and top-1 log-prob, x1000
                        let ent = if n > 0 { -lg.iter().map(|&x| { let lp = x - lse; lp.exp() * lp }).sum::<f32>() } else { 0.0 };
                        let ent = (ent * 1000.0).round().clamp(0.0, 30000.0) as i16;
                        let top1 = if n > 0 { ((mx - lse) * 1000.0).round().clamp(-30000.0, 0.0) as i16 } else { 0 };
                        for i in 0..list.len() {
                            let c = b.make(list.move_at(i));
                            if c.game_over() || c.hash % every != 0 {
                                continue;
                            }
                            let ct = c.turn as usize;
                            let hand = titanium::search::evaluate(&c, None, false).clamp(-30000, 30000) as i16;
                            let target = (400.0 * big.forward_y(&c)).round().clamp(-30000.0, 30000.0) as i16;
                            let lp = ((lg[i] - lse) * 1000.0).round().clamp(-30000.0, 0.0) as i16;
                            out.push(Sample { own: c.occ[ct], opp: c.occ[1 - ct], blk: c.blockers, pown: b.occ[ct], popp: b.occ[1 - ct], parent, hand, target, lp, rank: rk[i], dis, ent, top1, h: hq });
                        }
                    }
                    out
                })
            })
            .collect();
        hs.into_iter().map(|h| h.join().unwrap()).collect()
    });
    parts.into_iter().flatten().collect()
}

/// Row updates per child when the child accumulator is built from the parent's (the engine's way, smallnet.rs
/// child_acc_into): an incremental block costs one row op per changed feature per perspective; the delta block has no
/// parent accumulator to reuse, so it costs its active rows. Counted per perspective.
fn touched(arch: Arch, s: &Sample) -> usize {
    let changed = (s.own ^ s.pown) | (s.opp ^ s.popp);
    let cw = (0..NWIN).filter(|&w| win_mask(w) & changed != 0).count();
    let sq_ops = ((s.own ^ s.pown).count_ones() + (s.opp ^ s.popp).count_ones()) as usize;
    let (base, delta) = arch.split();
    let b = match base {
        Some(Arch::Sq) => sq_ops,
        Some(Arch::W2) => 2 * cw,
        Some(Arch::W2Sq) => 2 * cw + sq_ops,
        Some(Arch::X3) => 2 * (0..NX3).filter(|&k| x3_mask(k) & changed != 0).count(),
        Some(Arch::W3) => 2 * (0..NW3).filter(|&w| w3_mask(w) & changed != 0).count(),
        _ => 0,
    };
    b + if delta { 2 * cw } else { 0 }
}

/// Residuals t - y (cp) of a model on a sample set, in sample order.
fn residuals(m: &Model, set: &[Sample]) -> Vec<f64> {
    set.iter().map(|s| s.target as f64 - 400.0 * m.forward(s).y as f64).collect()
}

/// The shift c with P(t - y < c) = tau on `res`: the bound y + c is violated on exactly tau of those children.
fn tau_shift(res: &[f64], tau: f64) -> f64 {
    let mut r = res.to_vec();
    r.sort_by(|a, b| a.partial_cmp(b).unwrap());
    r[((tau * r.len() as f64) as usize).min(r.len().saturating_sub(1))]
}

/// D845 protocol: SELECT on half A (shift fitted on A, slack@tau on A), REPORT on half B with A's shift (out of sample).
struct Report {
    /// slack@tau on A with A's own shift (the selection number)
    a_slack: f64,
    /// val pinball on A (unshifted model, the training objective)
    a_pin: f64,
    /// on B with A's shift: mean slack, real violation rate, RMSE of the unshifted model
    b_slack: f64,
    b_viol: f64,
    b_rmse: f64,
    /// raw violation rate of the unshifted model on B
    b_viol_raw: f64,
    /// B slack quantiles (p10, p50, p90) after A's shift
    b_q: [f64; 3],
    /// per empties bucket (8): (children, slack, violation rate) on B with A's shift
    b_bk: Vec<(usize, f64, f64)>,
    rows: f64,
    touched: f64,
    shift: f64,
}

fn report(m: &Model, a: &[Sample], b: &[Sample], tau: f64) -> Report {
    let ra = residuals(m, a);
    let c = tau_shift(&ra, tau);
    let na = ra.len().max(1) as f64;
    let a_slack = ra.iter().sum::<f64>() / na - c;
    let a_pin = ra
        .iter()
        .map(|&r| {
            let d = r / 400.0;
            if d > 0.0 { tau * d } else { (tau - 1.0) * d }
        })
        .sum::<f64>()
        / na;
    let rb = residuals(m, b);
    let nb = rb.len().max(1) as f64;
    let mut sl: Vec<f64> = rb.iter().map(|&r| r - c).collect();
    let b_slack = sl.iter().sum::<f64>() / nb;
    let b_viol = sl.iter().filter(|&&x| x < 0.0).count() as f64 / nb;
    let b_viol_raw = rb.iter().filter(|&&x| x < 0.0).count() as f64 / nb;
    let b_rmse = (rb.iter().map(|r| r * r).sum::<f64>() / nb).sqrt();
    let mut bk = vec![(0usize, 0f64, 0usize); 8];
    let (mut rows, mut tch) = (0f64, 0f64);
    for (s, &x) in b.iter().zip(&sl) {
        let e = 49 - (s.own | s.opp | s.blk).count_ones() as usize;
        let k = (e * 8 / 50).min(7);
        bk[k].0 += 1;
        bk[k].1 += x;
        if x < 0.0 {
            bk[k].2 += 1;
        }
        let mut fu = Vec::new();
        feats(m.arch, s.own, s.opp, s.blk, s.pown, s.popp, &mut fu);
        rows += fu.len() as f64;
        // D863: +rel rows have no parent accumulator to reuse: every active rel row is a row op
        let rel_n = if m.ext.rel > 0 { rel_rows(s, m.ext.rel, 0, &mut Vec::new()) } else { 0 };
        tch += (touched(m.arch, s) + m.ext.touched() + rel_n) as f64;
    }
    sl.sort_by(|x, y| x.partial_cmp(y).unwrap());
    let q = |p: f64| sl[((p * nb) as usize).min(sl.len().saturating_sub(1))];
    Report {
        a_slack,
        a_pin,
        b_slack,
        b_viol,
        b_rmse,
        b_viol_raw,
        b_q: [q(0.1), q(0.5), q(0.9)],
        b_bk: bk.into_iter().map(|(n, s, v)| (n, if n > 0 { s / n as f64 } else { 0.0 }, if n > 0 { v as f64 / n as f64 } else { 0.0 })).collect(),
        rows: rows / nb,
        touched: tch / nb,
        shift: c,
    }
}

/// FNV-1a over every sample field: equal checksums = byte-identical sample sets (machines, thread counts).
fn checksum(set: &[Sample]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    let mut eat = |x: u64| {
        for i in 0..8 {
            h ^= (x >> (8 * i)) & 0xff;
            h = h.wrapping_mul(0x100_0000_01b3);
        }
    };
    for s in set {
        eat(s.own);
        eat(s.opp);
        eat(s.blk);
        eat(s.pown);
        eat(s.popp);
        eat((s.parent as u16 as u64) | (s.hand as u16 as u64) << 16 | (s.target as u16 as u64) << 32 | (s.rank as u64) << 56);
        eat(s.lp as u16 as u64 | (s.dis as u16 as u64) << 16 | (s.ent as u16 as u64) << 32 | (s.top1 as u16 as u64) << 48);
    }
    h
}

/// Saturation rule (user definition, memory feedback-saturation-definition): train at a constant LR; when the val pinball loss
/// has not improved by more than `min_rel` (relative) for `patience` epochs, multiply the LR by `decay`; after `max_decays`
/// decays, stop once `patience` more epochs bring no improvement. The best epoch's weights are restored at the end.
#[derive(Clone, Copy)]
struct Sat {
    patience: usize,
    max_decays: usize,
    decay: f32,
    min_rel: f64,
    /// D844: watch slack@tau (the compared number) instead of the val pinball loss
    on_slack: bool,
}

/// LAWS A14 (user D1305-D1307): one phase's checkpoint (full resumable state, every epoch + on STOP), best snapshot (SNAR,
/// the moment it improves) and the folder whose STOP file (or C:/gitProjects/STOP_ALL) stops the run.
struct Ck {
    state: String,
    best: String,
    run_dir: std::path::PathBuf,
    data_sum: u64,
    resume: bool,
}

#[derive(PartialEq, Debug)]
enum FitEnd {
    Done,
    /// stopped by a stop file: the state is saved; the same command + --resume continues
    Stopped(String),
}

/// SNAR of `m`'s layout with the parameters `p` (the best snapshot), written atomically.
fn save_params(m: &Model, p: &[f32], path: &str) -> Result<(), String> {
    let mut b = Vec::with_capacity(32 + 4 * p.len());
    b.extend_from_slice(b"SNAR");
    b.extend_from_slice(&2u32.to_le_bytes());
    let name = m.name();
    b.extend_from_slice(&(name.len() as u32).to_le_bytes());
    b.extend_from_slice(name.as_bytes());
    b.extend_from_slice(&(m.nb as u32).to_le_bytes());
    b.extend_from_slice(&(m.np as u32).to_le_bytes());
    for v in p {
        b.extend_from_slice(&v.to_le_bytes());
    }
    crate::stopctl::write_atomic(path, &b)
}

#[allow(clippy::too_many_arguments)]
fn fit(m: &mut Model, train: &[Sample], va: &[Sample], vb: &[Sample], epochs: usize, tau: Option<f32>, lr0: f32, threads: usize, seed: u64, tag: &str, tau_r: f64, sat: Option<Sat>, ck: &Ck) -> Result<FitEnd, String> {
    let batch = 16384usize;
    let np = m.np;
    let mut mm = vec![0f32; np];
    let mut vv = vec![0f32; np];
    let mut step = 0usize;
    // LAWS C15 (user D868): a saturation run has NO epoch limit; it stops only by the saturation rule below. The fixed
    // count `epochs` applies only to the MSE warm start (sat = None), which initialises the pinball phase.
    let epochs = if sat.is_some() { usize::MAX } else { epochs };
    let steps_total = if sat.is_some() { 1 } else { epochs * (train.len() / batch).max(1) };
    let mut lr_cur = lr0;
    let (mut best, mut best_ep, mut since, mut decays) = (f64::INFINITY, 0usize, 0usize, 0usize);
    let mut best_p: Vec<f32> = Vec::new();
    let mut grads: Vec<Vec<f32>> = (0..threads).map(|_| vec![0f32; np]).collect();
    let (b1, b2, eps) = (0.9f32, 0.999f32, 1e-8f32);
    let (mut ep0, mut chunk0, mut loss0) = (0usize, 0usize, 0f64);
    if ck.resume && std::path::Path::new(&ck.state).exists() {
        let d = std::fs::read(&ck.state).map_err(|e| format!("{}: {e}", ck.state))?;
        let mut r = crate::stopctl::Rd::new(&d);
        let (magic, ctag, sum) = (r.str()?, r.str()?, r.u64()?);
        if magic != "SACK1" || ctag != format!("{}/{tag}", m.name()) {
            return Err(format!("{}: not a checkpoint of {}/{tag}", ck.state, m.name()));
        }
        if sum != ck.data_sum {
            return Err(format!("{}: checkpoint samples {sum:016x} != this run {:016x} (other data / flags)", ck.state, ck.data_sum));
        }
        ep0 = r.u64()? as usize;
        chunk0 = r.u64()? as usize;
        step = r.u64()? as usize;
        lr_cur = r.f64()? as f32;
        best = r.f64()?;
        best_ep = r.u64()? as usize;
        since = r.u64()? as usize;
        decays = r.u64()? as usize;
        loss0 = r.f64()?;
        best_p = r.f32s()?;
        m.p = r.f32s()?;
        mm = r.f32s()?;
        vv = r.f32s()?;
        println!("[snarch] {} {tag} RESUMED from {}: epoch {} batch {chunk0}, lr {lr_cur:.2e}, best epoch {best_ep} = {best:.6}", m.name(), ck.state, ep0 + 1);
    } else {
        // a best snapshot exists from the first second on (the start weights), however short the run (LAWS A14)
        save_params(m, &m.p, &ck.best)?;
    }
    #[allow(clippy::too_many_arguments)]
    fn save_state(ck: &Ck, name: &str, tag: &str, ep: usize, chunk: usize, step: usize, lr: f32, best: f64, best_ep: usize, since: usize, decays: usize, loss: f64, best_p: &[f32], p: &[f32], mm: &[f32], vv: &[f32]) -> Result<(), String> {
        let mut w = crate::stopctl::Wr::default();
        w.str("SACK1").str(&format!("{name}/{tag}")).u64(ck.data_sum).u64(ep as u64).u64(chunk as u64).u64(step as u64).f64(lr as f64).f64(best).u64(best_ep as u64).u64(since as u64).u64(decays as u64).f64(loss).f32s(best_p).f32s(p).f32s(mm).f32s(vv);
        crate::stopctl::write_atomic(&ck.state, &w.0)
    }
    for ep in ep0..epochs {
        let t0 = Instant::now();
        // the epoch's order is a pure function of (seed, epoch): a resume replays it and skips the batches done
        let mut idx: Vec<u32> = (0..train.len() as u32).collect();
        let mut rng = Rng((seed ^ (ep as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15)) | 1);
        for i in (1..idx.len()).rev() {
            let j = (rng.next() % (i as u64 + 1)) as usize;
            idx.swap(i, j);
        }
        let mut loss_sum = if ep == ep0 { loss0 } else { 0.0 };
        let first = if ep == ep0 { chunk0 } else { 0 };
        for (ci, chunk) in idx.chunks(batch).enumerate().skip(first) {
            if chunk.len() < batch / 2 {
                break;
            }
            if let Some(f) = crate::stopctl::requested(Some(&ck.run_dir)) {
                save_state(ck, &m.name(), tag, ep, ci, step, lr_cur, best, best_ep, since, decays, loss_sum, &best_p, &m.p, &mm, &vv)?;
                println!("[snarch] {} {tag} STOPPED by {f} at epoch {} batch {ci}: state {} , best snapshot {} (epoch {best_ep}); the same command + --resume continues", m.name(), ep + 1, ck.state, ck.best);
                return Ok(FitEnd::Stopped(f));
            }
            step += 1;
            let lr = if sat.is_some() { lr_cur } else { lr0 * (0.05 + 0.95 * 0.5 * (1.0 + (std::f32::consts::PI * step as f32 / steps_total as f32).cos())) };
            let per = (chunk.len() + threads - 1) / threads;
            let mr = &*m;
            let ls: Vec<f64> = std::thread::scope(|sc| {
                let hs: Vec<_> = grads
                    .iter_mut()
                    .zip(chunk.chunks(per))
                    .map(|(g, part)| {
                        sc.spawn(move || {
                            g.iter_mut().for_each(|x| *x = 0.0);
                            let mut l = 0f64;
                            for &si in part {
                                let s = &train[si as usize];
                                let f = mr.forward(s);
                                let t = s.target as f32 / 400.0;
                                let d = t - f.y;
                                let dy = match tau {
                                    None => {
                                        l += (d * d) as f64;
                                        -2.0 * d
                                    }
                                    Some(q) => {
                                        l += (if d > 0.0 { q * d } else { (q - 1.0) * d }) as f64;
                                        if d > 0.0 {
                                            -q
                                        } else {
                                            1.0 - q
                                        }
                                    }
                                };
                                mr.backward(s, &f, dy / part.len() as f32, g);
                            }
                            l
                        })
                    })
                    .collect();
                hs.into_iter().map(|h| h.join().unwrap()).collect()
            });
            loss_sum += ls.iter().sum::<f64>();
            let used = chunk.chunks(per).count();
            let bc1 = 1.0 - b1.powi(step as i32);
            let bc2 = 1.0 - b2.powi(step as i32);
            for k in 0..np {
                let mut gk = 0f32;
                for g in grads.iter().take(used) {
                    gk += g[k];
                }
                if gk == 0.0 && mm[k] == 0.0 {
                    continue;
                }
                gk /= used as f32;
                mm[k] = b1 * mm[k] + (1.0 - b1) * gk;
                vv[k] = b2 * vv[k] + (1.0 - b2) * gk * gk;
                m.p[k] -= lr * (mm[k] / bc1) / ((vv[k] / bc2).sqrt() + eps);
            }
        }
        let r = report(m, va, vb, tau_r);
        println!(
            "[snarch] {} {tag} epoch {}/{} train {:.5} lr {:.2e} | A: slack@tau {:.2} pin {:.6} | B (A's shift {:+.1}): slack {:.2} viol {:.2}% (raw {:.2}%) RMSE {:.1} | {:.0}s",
            m.name(),
            ep + 1,
            if sat.is_some() { "sat".to_string() } else { epochs.to_string() },
            loss_sum / train.len() as f64,
            if sat.is_some() { lr_cur } else { lr0 },
            r.a_slack,
            r.a_pin,
            r.shift,
            r.b_slack,
            100.0 * r.b_viol,
            100.0 * r.b_viol_raw,
            r.b_rmse,
            t0.elapsed().as_secs_f64()
        );
        if let Some(s) = sat {
            // D844/D845: select on half A only (B is never used for any decision)
            let cur = if s.on_slack { r.a_slack } else { r.a_pin };
            if cur < best * (1.0 - s.min_rel) {
                best = cur;
                best_ep = ep + 1;
                since = 0;
                best_p.clone_from(&m.p);
                save_params(m, &best_p, &ck.best)?;
            } else {
                since += 1;
                if cur < best {
                    // a tiny gain still counts as the best weights, not as progress
                    best = cur;
                    best_ep = ep + 1;
                    best_p.clone_from(&m.p);
                    save_params(m, &best_p, &ck.best)?;
                }
            }
            let mut done = false;
            if since >= s.patience {
                if decays >= s.max_decays {
                    println!("[snarch] {} {tag} SATURATED: {} epochs without a {:.3}% gain of {} at the final lr; best epoch {best_ep} = {best:.6}", m.name(), s.patience, 100.0 * s.min_rel, if s.on_slack { "slack@tau" } else { "val_pin" });
                    done = true;
                } else {
                    decays += 1;
                    lr_cur *= s.decay;
                    since = 0;
                    println!("[snarch] {} {tag} plateau: lr -> {lr_cur:.2e} (decay {decays}/{})", m.name(), s.max_decays);
                }
            }
            save_state(ck, &m.name(), tag, ep + 1, 0, step, lr_cur, best, best_ep, since, decays, 0.0, &best_p, &m.p, &mm, &vv)?;
            if done {
                break;
            }
        } else {
            // fixed warm start (no selection): the current weights are the phase's result so far
            save_params(m, &m.p, &ck.best)?;
            save_state(ck, &m.name(), tag, ep + 1, 0, step, lr_cur, best, best_ep, since, decays, 0.0, &best_p, &m.p, &mm, &vv)?;
        }
    }
    if sat.is_some() && !best_p.is_empty() {
        m.p = best_p;
    }
    Ok(FitEnd::Done)
}

/// Binary dump of a trained research model (no JSON, D806): "SNAR" v2, arch name (u32 length + bytes), buckets u32, params u32,
/// then the f32 params little-endian in the Model layout (ft rows, ft bias, l1 w, l1 b, out w, out b, skips).
/// D892: load a SNAR v1 (16-byte name) or v2 (length + name) file into `m`; the design name, bucket count and parameter
/// count must match exactly.
fn load_model(m: &mut Model, path: &str) -> Result<(), String> {
    let d = std::fs::read(path).map_err(|e| format!("{path}: {e}"))?;
    if d.len() < 24 || &d[..4] != b"SNAR" {
        return Err(format!("{path}: not a SNAR file"));
    }
    let u = |o: usize| u32::from_le_bytes(d[o..o + 4].try_into().unwrap()) as usize;
    let (name, hdr) = match u(4) {
        1 => (String::from_utf8_lossy(&d[8..24]).trim_end_matches('\0').to_string(), 24),
        2 => {
            let nl = u(8);
            (String::from_utf8_lossy(&d[12..12 + nl]).to_string(), 12 + nl)
        }
        v => return Err(format!("{path}: SNAR version {v}")),
    };
    let (nb, np) = (u(hdr), u(hdr + 4));
    let legacy = np + m.nb == m.np && !m.ext.h1; // D1297: files before D851 have 3 skip groups, the 4th (disagreement) stays 0
    if name != m.name() || nb != m.nb || !(np == m.np || legacy) || d.len() != hdr + 8 + 4 * np {
        return Err(format!("{path}: is {name} nb {nb} np {np}, the run is {} nb {} np {}", m.name(), m.nb, m.np));
    }
    for (i, c) in d[hdr + 8..].chunks_exact(4).enumerate() {
        m.p[i] = f32::from_le_bytes(c.try_into().unwrap());
    }
    Ok(())
}

fn save_model(m: &Model, path: &str) -> Result<(), String> {
    let mut b = Vec::with_capacity(32 + 4 * m.np);
    b.extend_from_slice(b"SNAR");
    // v2 (D870): the arch name is stored as u32 length + bytes (v1's fixed 16 bytes panicked on longer names such as
    // w2+pol+dis+mv+ent+h1 after a full saturated run)
    b.extend_from_slice(&2u32.to_le_bytes());
    let name = m.name();
    b.extend_from_slice(&(name.len() as u32).to_le_bytes());
    b.extend_from_slice(name.as_bytes());
    b.extend_from_slice(&(m.nb as u32).to_le_bytes());
    b.extend_from_slice(&(m.np as u32).to_le_bytes());
    for v in &m.p {
        b.extend_from_slice(&v.to_le_bytes());
    }
    crate::stopctl::write_atomic(path, &b)
}

pub fn run_snarch(args: &[String]) -> ExitCode {
    match run(args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("snarch: {e}");
            ExitCode::FAILURE
        }
    }
}


/// D1303: `ti train snarch --export-zx FILE.snar --net big.tnn --val STEM --out OUT.tsn2 [--val-rows 400000] [--tau 0.02]
/// [--child-every 3] [--threads 7]`: write the saturated ZX head as the engine's "TSN2" blob. The 2% tau shift is fitted
/// PER EMPTIES BUCKET on val half A (the float model) and folded into that bucket's output bias; the weights are
/// quantised (ft x 255, output x 256, i16) and the blob is then scored THROUGH THE ENGINE'S OWN code (titanium::smallnet)
/// on val half B: violation rate and slack per bucket, and the float-vs-integer difference.
fn export_zx(args: &[String]) -> Result<(), String> {
    let flag = |n: &str| args.iter().position(|a| a == n).and_then(|i| args.get(i + 1)).cloned();
    let need = |n: &str| flag(n).ok_or_else(|| format!("{n} is required"));
    let num = |n: &str, d: f64| flag(n).and_then(|x| x.parse::<f64>().ok()).unwrap_or(d);
    let (snar, out, net_path, val_stem) = (need("--export-zx")?, need("--out")?, need("--net")?, need("--val")?);
    let (tau, threads, every, val_rows) = (num("--tau", 0.02), num("--threads", 7.0) as usize, num("--child-every", 3.0) as u64, num("--val-rows", 400_000.0) as usize);
    let big = TrainableNet::from_tnn_bytes(&std::fs::read(&net_path).map_err(|e| format!("{net_path}: {e}"))?)?;
    let mut va = Tdat::open(&val_stem)?;
    let vrows = load_val(&mut va, val_rows)?;
    let (mut ra, mut rb) = (Vec::new(), Vec::new());
    for (i, blk) in vrows.chunks(1024).enumerate() {
        if i % 2 == 0 { ra.extend_from_slice(blk) } else { rb.extend_from_slice(blk) }
    }
    let a = build_samples(&ra, &big, every, threads, 1 ^ 0x5555);
    let b = build_samples(&rb, &big, every, threads, 1 ^ 0xAAAA);
    println!("[export] val children: A {} ({:016x}), B {} ({:016x})", a.len(), checksum(&a), b.len(), checksum(&b));
    // the model's bucket count comes from the SNAR file (8 or 16 = the NNUE's buckets, D1313)
    let nbz = {
        let d = std::fs::read(&snar).map_err(|e| format!("{snar}: {e}"))?;
        let nl = u32::from_le_bytes(d[8..12].try_into().unwrap()) as usize;
        u32::from_le_bytes(d[12 + nl..16 + nl].try_into().unwrap()) as usize
    };
    let mut m = Model::new(Arch::W2, nbz, 1, Ext { zx: ZX_W as u16, ..Ext::default() });
    load_model(&mut m, &snar)?;
    // per-bucket shifts on A (cp): bound = 400 y + shift[bk] is violated on tau of the bucket's children
    let ra_res = residuals(&m, &a);
    let mut per: Vec<Vec<f64>> = vec![Vec::new(); nbz];
    for (s, r) in a.iter().zip(&ra_res) {
        per[m.bucket(s)].push(*r);
    }
    let global = tau_shift(&ra_res, tau);
    // a bucket with too few val children (e.g. the first moves at 32 buckets: >= 46 empties) takes the shift of the NEAREST
    // populated bucket (same game phase), never the global one: the opening needs a far larger shift than the middle game
    // (D1317: -377 cp at 8 buckets vs a global -10 cp)
    let own: Vec<Option<f64>> = per.iter().map(|v| if v.len() >= 400 { Some(tau_shift(v, tau)) } else { None }).collect();
    let shifts: Vec<f64> = (0..own.len())
        .map(|b| {
            own[b].unwrap_or_else(|| {
                (1..own.len()).find_map(|d| own.get(b.wrapping_sub(d)).copied().flatten().or_else(|| own.get(b + d).copied().flatten())).unwrap_or(global)
            })
        })
        .collect();
    for (b, o) in own.iter().enumerate() {
        if o.is_none() {
            println!("[export] bucket {b}: only {} val children, shift {:.1} cp taken from the nearest populated bucket", per[b].len(), shifts[b]);
        }
    }
    println!("[export] per-bucket shift cp (n): {}", shifts.iter().zip(&per).map(|(s, v)| format!("{s:.1} ({})", v.len())).collect::<Vec<_>>().join(" | "));
    // quantise
    let nrows = 2916;
    let mut blob: Vec<u8> = Vec::new();
    blob.extend_from_slice(b"TSN2");
    blob.extend_from_slice(&(nbz as u32).to_le_bytes());
    blob.extend_from_slice(&(ZX_W as u32).to_le_bytes());
    let mut worst = 0f32;
    let mut put = |v: f32, scale: f32, blob: &mut Vec<u8>| -> Result<(), String> {
        let q = (v * scale).round();
        worst = worst.max(q.abs());
        if q.abs() > 32767.0 {
            return Err(format!("weight {v} x {scale} does not fit i16"));
        }
        blob.extend_from_slice(&(q as i16).to_le_bytes());
        Ok(())
    };
    for i in 0..nrows * ZX_W {
        put(m.p[i], 255.0, &mut blob)?;
    }
    for k in 0..ZX_W {
        put(m.p[m.o_ftb + k], 255.0, &mut blob)?;
    }
    let mut zo_worst = 0f32;
    for bk in 0..nbz {
        let mut l1 = 0f64;
        for k in 0..ZX_W {
            let v = m.p[m.o_l1w + bk * ZX_W + k];
            zo_worst = zo_worst.max(v.abs());
            l1 += (v * 256.0).round().abs() as f64;
            put(v, 256.0, &mut blob)?;
        }
        if l1 * 32512.0 >= 2_147_483_647.0 {
            return Err(format!("bucket {bk}: sum of |zo_q| x 32512 overflows i32"));
        }
    }
    for bk in 0..nbz {
        let ob = m.p[m.o_ob + bk] + (shifts[bk] / 400.0) as f32;
        blob.extend_from_slice(&ob.to_le_bytes());
    }
    for k in 0..2 {
        for bk in 0..nbz {
            blob.extend_from_slice(&m.p[m.o_sk + k * nbz + bk].to_le_bytes());
        }
    }
    std::fs::write(&out, &blob).map_err(|e| format!("{out}: {e}"))?;
    println!("[export] wrote {out}: {} bytes, max |weight q| {worst:.0} (i16 32767), max |output weight| {zo_worst:.3} (x256 = {:.0})", blob.len(), zo_worst * 256.0);
    #[cfg(feature = "zxengine")]
    {
    // score the BLOB through the engine's own integer code on val half B
    let net = titanium::smallnet::SmallNet::parse(&blob)?;
    let (mut n_bk, mut viol_bk, mut slack_bk) = (vec![0usize; nbz], vec![0usize; nbz], vec![0f64; nbz]);
    let (mut dsum, mut dmax, mut ftsum) = (0f64, 0f64, 0f64);
    for s in &b {
        // engine call: parent = the side that moved to make this child (its stones before the move = child mover's `pown`)
        let pa = net.parent_acc(s.popp, s.pown);
        let empties = 49 - (s.own | s.opp | s.blk).count_ones();
        let eb = net.child_bound(&pa, s.popp, s.pown, s.own, s.opp, empties, s.parent as i32, s.hand as i32);
        let bk = m.bucket(s);
        let fb = 400.0 * m.forward(s).y as f64 + shifts[bk];
        let d = (eb as f64 - fb).abs();
        dsum += d;
        dmax = dmax.max(d);
        ftsum += fb;
        n_bk[bk] += 1;
        viol_bk[bk] += ((s.target as i32) < eb) as usize;
        slack_bk[bk] += s.target as f64 - eb as f64;
    }
    let nb_all = b.len().max(1) as f64;
    println!("[export] ENGINE integer bound vs float model on B: mean |diff| {:.3} cp, max {:.1} cp (float mean bound {:.1})", dsum / nb_all, dmax, ftsum / nb_all);
    let tv: usize = viol_bk.iter().sum();
    let ts: f64 = slack_bk.iter().sum();
    println!("[export] ENGINE bound on held-out B: violation {:.2}% slack {:.2} cp ({} children)", 100.0 * tv as f64 / nb_all, ts / nb_all, b.len());
    for bk in 0..nbz {
        let n = n_bk[bk].max(1) as f64;
        println!("[export]   bucket {bk}: n {:>7} violation {:5.2}% slack {:6.1} cp", n_bk[bk], 100.0 * viol_bk[bk] as f64 / n, slack_bk[bk] / n);
    }
    }
    #[cfg(not(feature = "zxengine"))]
    {
        let _ = (&b, &shifts);
        println!("[export] engine scoring skipped: build the hub with --features zxengine against the ZX engine to score the blob through the engine code");
    }
    Ok(())
}

fn run(args: &[String]) -> Result<(), String> {
    let flag = |n: &str| args.iter().position(|a| a == n).and_then(|i| args.get(i + 1)).cloned();
    let need = |n: &str| flag(n).ok_or_else(|| format!("{n} is required"));
    let num = |n: &str, d: f64| flag(n).and_then(|x| x.parse::<f64>().ok()).unwrap_or(d);
    if args.iter().any(|a| a == "--export-zx") {
        return export_zx(args);
    }
    if args.iter().any(|a| a == "--bench-heads") {
        bench_heads::run();
        return Ok(());
    }
    let net_path = need("--net")?;
    let train_stem = need("--train")?;
    let val_stem = need("--val")?;
    // D847: "w2+pol" = the arch plus the policy input
    let archs: Vec<(Arch, Ext)> = flag("--arch").unwrap_or_else(|| "sq,w2,w2sq,delta".into()).split(',').map(|s| { let s = s.trim(); let (a, e) = Ext::parse(s)?; Arch::parse(&a).map(|x| (x, e)).ok_or(format!("unknown arch {s}")) }).collect::<Result<_, _>>()?;
    let nb = num("--buckets", 8.0) as usize;
    // D1130: EVERY row of --train by default; fewer only with --experiment NAME (--save-dir under experiments/)
    let experiment = flag("--experiment");
    let rows_req = flag("--rows").map(|x| x.parse::<usize>().map_err(|e| format!("--rows {x}: {e}"))).transpose()?;
    let val_rows = num("--val-rows", 60_000.0) as usize;
    let every = num("--child-every", 3.0) as u64;
    let e_mse = num("--epochs-mse", 4.0) as usize;
    for banned in ["--max-epochs", "--epochs-pin", "--fixed"] {
        if args.iter().any(|a| a == banned) {
            return Err(format!("{banned} is forbidden (LAWS C15, user D868: no epoch limit; every run trains to saturation)"));
        }
    }
    let tau = num("--tau", 0.02) as f32;
    // D845: lists, one model per (arch, lr, seed): --lr 1e-3,2e-3,4e-3 --seed 1,2
    let list = |n: &str, d: &str| -> Result<Vec<f64>, String> {
        flag(n).unwrap_or_else(|| d.into()).split(',').map(|x| x.trim().parse::<f64>().map_err(|e| format!("{n} {x}: {e}"))).collect()
    };
    let lrs = list("--lr", "0.002")?;
    let seeds: Vec<u64> = list("--seed", "1")?.into_iter().map(|x| x as u64).collect();
    let threads = num("--threads", 8.0) as usize;
    // samples always come from seed 1 (D845: model seeds vary the init + shuffle only, never the data)
    let data_seed = num("--data-seed", 1.0) as u64;
    // LAWS C15 (user D868): the pinball phase ALWAYS trains to saturation; there is no epoch limit and no fixed mode
    let sat = Sat {
        patience: num("--patience", 4.0) as usize,
        max_decays: num("--max-decays", 3.0) as usize,
        decay: num("--decay", 0.3) as f32,
        min_rel: num("--min-rel", 2e-4),
        on_slack: flag("--sat-metric").map(|v| v != "pin").unwrap_or(true),
    };
    // LAWS A14 (user D1307): every run keeps its best snapshots + checkpoints in --save-dir; DIR/STOP (or
    // C:/gitProjects/STOP_ALL) stops it safely; the same command + --resume continues (finished arms / phases are skipped)
    let sd = flag("--save-dir").ok_or("--save-dir DIR is required (LAWS A14: best snapshots + checkpoints live there; DIR/STOP stops the run safely)")?;
    let resume = args.iter().any(|a| a == "--resume");
    let save_dir = Some(sd.clone());
    let init = flag("--init");
    let zx_init = flag("--zx-init");
    let zx_teacher = flag("--zx-teacher");
    std::fs::create_dir_all(&sd).map_err(|e| format!("{sd}: {e}"))?;
    if let Some(f) = crate::stopctl::requested(Some(std::path::Path::new(&sd))) {
        return Err(format!("stop file {f} exists: remove it to train"));
    }

    let bytes = std::fs::read(&net_path).map_err(|e| format!("{net_path}: {e}"))?;
    let big = TrainableNet::from_tnn_bytes(&bytes)?;
    let t0 = Instant::now();
    let mut tr = Tdat::open(&train_stem)?;
    let rows = crate::trainer::whole_rows_or_experiment("snarch", &tr, rows_req, &experiment, &save_dir.clone().unwrap_or_default(), every, std::mem::size_of::<Sample>())?;
    let mut va = Tdat::open(&val_stem)?;
    let trows = load_val(&mut tr, rows)?;
    let mut train = build_samples(&trows, &big, every, threads, data_seed);
    // D845: val split BY POSITION (alternating 1024-row blocks), so all children of one parent sit in one half
    let vrows = load_val(&mut va, val_rows)?;
    let (mut ra, mut rb) = (Vec::new(), Vec::new());
    for (i, blk) in vrows.chunks(1024).enumerate() {
        if i % 2 == 0 { ra.extend_from_slice(blk) } else { rb.extend_from_slice(blk) }
    }
    let mut val_a = build_samples(&ra, &big, (every / 2).max(1), threads, data_seed ^ 0x5555);
    let mut val_b = build_samples(&rb, &big, (every / 2).max(1), threads, data_seed ^ 0xAAAA);
    let pin_txt = format!("pinball to SATURATION on {} of val half A (patience {}, {} lr decays x{}, min gain {:.3}%, NO epoch limit)", if sat.on_slack { "slack@tau" } else { "val_pin" }, sat.patience, sat.max_decays, sat.decay, 100.0 * sat.min_rel);
    println!(
        "[snarch] big net {net_path}; rows {} -> train {} children (checksum {:016x}); val rows {} + {} -> A {} children ({:016x}), B {} children ({:016x}); {:.0}s",
        trows.len(),
        train.len(),
        checksum(&train),
        ra.len(),
        rb.len(),
        val_a.len(),
        checksum(&val_a),
        val_b.len(),
        checksum(&val_b),
        t0.elapsed().as_secs_f64()
    );
    // D851: verify (not assume) the stored big-net inputs: layer sizes, h range, disagreement range
    {
        let (mut hmin, mut hmax, mut hsum, mut nz) = (i8::MAX, i8::MIN, 0f64, 0usize);
        let (mut dsum, mut dmax) = (0f64, 0i16);
        for s in train.iter().step_by(97) {
            for &v in &s.h {
                hmin = hmin.min(v);
                hmax = hmax.max(v);
                hsum += v as f64;
                nz += (v != 0) as usize;
            }
            dsum += s.dis as f64;
            dmax = dmax.max(s.dis);
        }
        let ns = train.iter().step_by(97).count().max(1) as f64;
        println!(
            "[snarch] big net l1 {} l2 {} (h stored {} of {}), has_dis {}; h q127 min {hmin} max {hmax} mean {:.1} nonzero {:.1}/sample; parent dis mean {:.1} max {dmax} cp",
            big.l1,
            big.l2,
            (big.l1 + big.l2).min(HMAX),
            big.l1 + big.l2,
            big.has_dis,
            hsum / (ns * HMAX as f64),
            nz as f64 / ns,
            dsum / ns
        );
    }
    println!("[snarch] buckets {nb}, tau {tau}, lrs {lrs:?}, seeds {seeds:?}, threads {threads}, epochs {e_mse} MSE warm start + {pin_txt}");
    let mut summary = Vec::new();
    // the checkpoints are tied to THESE samples (a resume with other data / flags is refused)
    let data_sum = checksum(&train) ^ checksum(&val_a).rotate_left(1) ^ checksum(&val_b).rotate_left(2);
    for (arch, ext) in archs {
        for &lr in &lrs {
            for &seed in &seeds {
                let lr = lr as f32;
                let tag = format!("{}{}_lr{lr:.0e}_s{seed}_r{}", arch.name(), ext.suffix(), trows.len());
                if ext.zx > 0 && (arch != Arch::W2 || ext.h1) {
                    return Err("the +zx head needs the plain w2 rows (no +h1)".into());
                }
                let final_path = format!("{sd}/snarch_{tag}_b{nb}.snar");
                if std::path::Path::new(&final_path).exists() {
                    if resume {
                        println!("[snarch] {tag} already finished ({final_path}): skipped");
                        continue;
                    }
                    return Err(format!("{final_path} exists: use another --save-dir, or --resume to skip finished arms"));
                }
                let ck_of = |phase: &str| Ck { state: format!("{sd}/{tag}.{phase}.ckpt"), best: format!("{sd}/{tag}.{phase}.best.snar"), run_dir: std::path::PathBuf::from(&sd), data_sum, resume };
                let done_of = |phase: &str| format!("{sd}/{tag}.{phase}.done");
                let mut m = Model::new(arch, nb, seed, ext);
                let distill = ext.zx > 0 && ext.zx as usize != ZX_W && zx_teacher.is_some();
                if distill {
                    // D1301 (user): net-to-net downscale = OUR STANDARD distilled method (hub shrink.rs): (1) keep the channels that
                    // matter most (importance = mean act^2 x squared output weight, dropped ones folded into the bias) = the START,
                    // (2) DISTIL: train the narrow net to reproduce the saturated 256 head's own outputs, to saturation, (3) the
                    // normal pinball fit below. The teacher is the SATURATED 256 head, never the raw Zataxx net.
                    let tp = zx_teacher.as_ref().unwrap();
                    let mut teacher = Model::new(arch, nb, seed, Ext { zx: ZX_W as u16, ..ext });
                    load_model(&mut teacher, tp)?;
                    let rt = report(&teacher, &val_a, &val_b, tau as f64);
                    println!("[snarch] {tag} teacher {tp}: B slack {:.2} cp viol {:.2}% | A slack@tau {:.2} cp", rt.b_slack, 100.0 * rt.b_viol, rt.a_slack);
                    let step = (train.len() / 200_000).max(1);
                    let probe: Vec<Sample> = train.iter().step_by(step).copied().collect();
                    let msq = zx_mean_sq(&teacher, &probe);
                    m = teacher.shrink_zx(ext.zx as usize, &msq);
                    let r1 = report(&m, &val_a, &val_b, tau as f64);
                    println!("[snarch] {tag} shrunk {ZX_W} -> {} units (importance x mean act^2 on {} children, bias-compensated) = the distillation START, B: RMSE {:.1} cp, A slack@tau {:.2} cp", ext.zx, probe.len(), r1.b_rmse, r1.a_slack);
                    // teacher outputs as the targets for the distillation phase (originals restored afterwards)
                    let tout = |set: &[Sample]| -> Vec<i16> {
                        let mut o = vec![0i16; set.len()];
                        let per = (set.len() + threads - 1) / threads.max(1);
                        std::thread::scope(|sc| {
                            for (oc, sc_set) in o.chunks_mut(per.max(1)).zip(set.chunks(per.max(1))) {
                                let t = &teacher;
                                sc.spawn(move || {
                                    for (d, s) in oc.iter_mut().zip(sc_set) {
                                        *d = (400.0 * t.forward(s).y).round().clamp(-30000.0, 30000.0) as i16;
                                    }
                                });
                            }
                        });
                        o
                    };
                    let (o_tr, o_va, o_vb) = (tout(&train), tout(&val_a), tout(&val_b));
                    let (s_tr, s_va, s_vb): (Vec<i16>, Vec<i16>, Vec<i16>) = (train.iter().map(|s| s.target).collect(), val_a.iter().map(|s| s.target).collect(), val_b.iter().map(|s| s.target).collect());
                    for (s, t) in train.iter_mut().zip(&o_tr) { s.target = *t; }
                    for (s, t) in val_a.iter_mut().zip(&o_va) { s.target = *t; }
                    for (s, t) in val_b.iter_mut().zip(&o_vb) { s.target = *t; }
                    drop((o_tr, o_va, o_vb));
                    let ph = ck_of("distill");
                    if resume && std::path::Path::new(&done_of("distill")).exists() {
                        load_model(&mut m, &ph.best)?;
                        println!("[snarch] {tag} distill phase already finished: weights from {}", ph.best);
                    } else {
                        if let FitEnd::Stopped(f) = fit(&mut m, &train, &val_a, &val_b, 0, None, lr, threads, seed ^ 3, "distill", tau as f64, Some(sat), &ph)? {
                            println!("[snarch] run STOPPED by {f} (distill phase of {tag}); the same command + --resume continues");
                            return Ok(());
                        }
                        crate::stopctl::write_atomic(&done_of("distill"), b"done")?;
                    }
                    for (s, t) in train.iter_mut().zip(&s_tr) { s.target = *t; }
                    for (s, t) in val_a.iter_mut().zip(&s_va) { s.target = *t; }
                    for (s, t) in val_b.iter_mut().zip(&s_vb) { s.target = *t; }
                    let r2 = report(&m, &val_a, &val_b, tau as f64);
                    println!("[snarch] {tag} distilled to saturation, then measured against the TRUTH again: B RMSE {:.1} cp, A slack@tau {:.2} cp (before the pinball fit)", r2.b_rmse, r2.a_slack);
                } else if let (true, Some(ip)) = (ext.zx > 0, &init) {
                    // D1313 (user): a trained ZX head of the same width as the start; if it has half the buckets (8 -> 16,
                    // the NNUE's buckets), every bucket b16 starts as the trained bucket b16/2 = exact step-0 parity, since
                    // floor(e * 8 / 50) == floor(floor(e * 16 / 50) / 2)
                    m.load_expand(ip)?;
                    let r0 = report(&m, &val_a, &val_b, tau as f64);
                    println!("[snarch] {tag} initialised from {ip} ({} buckets), step 0: B slack {:.2} cp viol {:.2}%, A slack@tau {:.2} cp", m.nb, r0.b_slack, 100.0 * r0.b_viol, r0.a_slack);
                } else if ext.zx > 0 {
                    // C14 (user D1296): the ZX head ALWAYS starts from the trained Zataxx net; narrower widths are a
                    // downscale of it (most important units kept, the rest folded into the output bias)
                    let zp = zx_init.as_ref().ok_or("the +zx head needs --zx-init <Zataxx-format net> or --init <trained ZX .snar> (never random init, C14)")?;
                    let mut full = Model::new(arch, nb, seed, Ext { zx: ZX_W as u16, ..ext });
                    full.load_zx(zp)?;
                    let r0 = report(&full, &val_a, &val_b, tau as f64);
                    println!("[snarch] {tag} Zataxx net {zp} as the bound, step 0 (skips 0, no fine-tuning), B: RMSE {:.1} cp, A slack@tau {:.2} cp", r0.b_rmse, r0.a_slack);
                    if ext.zx as usize == ZX_W {
                        m = full;
                    } else {
                        let step = (train.len() / 200_000).max(1);
                        let probe: Vec<Sample> = train.iter().step_by(step).copied().collect();
                        let msq = zx_mean_sq(&full, &probe);
                        m = full.shrink_zx(ext.zx as usize, &msq);
                        let r1 = report(&m, &val_a, &val_b, tau as f64);
                        println!("[snarch] {tag} downscaled {ZX_W} -> {} units (importance |out w| x mean act^2 on {} children; dropped units folded into the output bias), step 0, B: RMSE {:.1} cp, A slack@tau {:.2} cp", ext.zx, probe.len(), r1.b_rmse, r1.a_slack);
                    }
                }
                println!("[snarch] {tag} : {} features, {} params", arch.nf(), m.np);
                // D892 (C14: never from random weights): --init FILE.snar starts from a trained model of the SAME design
                if let (Some(ip), 0) = (&init, ext.zx) {
                    load_model(&mut m, ip)?;
                    println!("[snarch] {tag} initialised from {ip}");
                }
                let ta = Instant::now();
                for (phase, on) in [("MSE", e_mse > 0 && !distill), ("pinball", true)] {
                    if !on {
                        continue;
                    }
                    let ph = ck_of(phase);
                    if resume && std::path::Path::new(&done_of(phase)).exists() {
                        load_model(&mut m, &ph.best)?;
                        println!("[snarch] {tag} {phase} phase already finished: weights from {}", ph.best);
                        continue;
                    }
                    let end = if phase == "MSE" {
                        fit(&mut m, &train, &val_a, &val_b, e_mse, None, lr, threads, seed, "MSE", tau as f64, None, &ph)?
                    } else {
                        fit(&mut m, &train, &val_a, &val_b, 0, Some(tau), lr, threads, seed ^ 7, "pinball", tau as f64, Some(sat), &ph)?
                    };
                    if let FitEnd::Stopped(f) = end {
                        println!("[snarch] run STOPPED by {f} ({phase} phase of {tag}); best snapshot {}; the same command + --resume continues", ph.best);
                        return Ok(());
                    }
                    crate::stopctl::write_atomic(&done_of(phase), b"done")?;
                }
                let r = report(&m, &val_a, &val_b, tau as f64);
                save_model(&m, &final_path)?;
                println!("[snarch] {tag} saved {final_path}");
                let bk: Vec<String> = r.b_bk.iter().map(|(n, s, v)| format!("{s:.0}/{:.1}%/{n}", 100.0 * v)).collect();
                summary.push(format!(
                    "{tag} | B slack {:.2} cp viol {:.2}% | A slack@tau {:.2} | B q10/50/90 {:.0}/{:.0}/{:.0} | B RMSE {:.1} | rows {:.1} touched {:.1} /persp | params {} | {:.0} min | B per empties bucket 0-7 slack/viol/n: {}",
                    r.b_slack,
                    100.0 * r.b_viol,
                    r.a_slack,
                    r.b_q[0],
                    r.b_q[1],
                    r.b_q[2],
                    r.b_rmse,
                    r.rows,
                    r.touched,
                    m.np,
                    ta.elapsed().as_secs_f64() / 60.0,
                    bk.join(" ")
                ));
                // print each model's line as soon as it is done (a long saturation run must not lose finished results)
                println!("[snarch] DONE {}", summary.last().unwrap());
            }
        }
    }
    println!("[snarch] SUMMARY (B = held-out half, never used for selection; lower B slack at ~tau viol = tighter bound):");
    for l in summary {
        println!("[snarch]   {l}");
    }
    Ok(())
}


/// D1298: `ti train snarch --bench-heads`: ns per CHILD BOUND CALL of the SmallNet head designs, same harness, AVX2 i16 (the
/// engine way: the parent accumulator is given, the child's is parent +- the changed rows, then the head). Weights are
/// random (the cost does not depend on their values); rows are random table rows (a 1.5 MB table does not sit in L1).
///   sq-like : 2 perspectives x W32, row ops each, CReLU, L1 64 -> 16 (per bucket), out 16 (today's engine design)
///   zx W    : ONE accumulator of W, 2 row ops per changed 2x2 window, SCReLU, dot with the output weights (no L1)
#[cfg(target_arch = "x86_64")]
mod bench_heads {
    use std::arch::x86_64::*;
    use std::hint::black_box;
    use std::time::Instant;

    pub fn run() {
        if !is_x86_feature_detected!("avx2") {
            println!("[bench] no AVX2 on this CPU");
            return;
        }
        let mut s = 0x9E37_79B9_7F4A_7C15u64;
        let mut rnd = move || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        };
        for &w in &[256usize, 128] {
            let table: Vec<i16> = (0..2916 * w).map(|_| (rnd() % 61) as i16 - 30).collect();
            let ow: Vec<i16> = (0..w).map(|_| (rnd() % 127) as i16 - 60).collect();
            let parent: Vec<i16> = (0..w).map(|_| (rnd() % 256) as i16).collect();
            for &cw in &[5usize, 8, 12] {
                let calls = 400_000usize;
                let rows: Vec<Vec<u32>> = (0..4096).map(|_| (0..2 * cw).map(|_| (rnd() % 2916) as u32).collect()).collect();
                let mut acc = vec![0i16; w];
                let mut sink = 0i64;
                for r in rows.iter().take(512) {
                    sink += unsafe { zx_call(w, &table, &ow, &parent, r, &mut acc) } as i64;
                }
                let t = Instant::now();
                for i in 0..calls {
                    sink += unsafe { zx_call(w, &table, &ow, &parent, &rows[i & 4095], &mut acc) } as i64;
                }
                let ns = t.elapsed().as_nanos() as f64 / calls as f64;
                black_box(sink);
                println!("[bench] zx W{w:<3} {cw:>2} changed windows ({} row ops): {ns:6.1} ns per child bound", 2 * cw);
            }
        }
        // the engine-like sq head: W32 x 2 perspectives, n changed feature rows per perspective, L1 64 -> 16
        let pt: Vec<i16> = (0..2 * 147 * 32).map(|_| (rnd() % 61) as i16 - 30).collect();
        let l1: Vec<i16> = (0..8 * 16 * 64).map(|_| (rnd() % 61) as i16 - 30).collect();
        let l1b: Vec<i32> = (0..8 * 16).map(|_| (rnd() % 100) as i32).collect();
        let ow2: Vec<i16> = (0..8 * 16).map(|_| (rnd() % 61) as i16 - 30).collect();
        let parent2: Vec<i16> = (0..64).map(|_| (rnd() % 256) as i16).collect();
        for &n in &[4usize, 6, 9] {
            let calls = 400_000usize;
            let rows: Vec<Vec<u32>> = (0..4096).map(|_| (0..2 * n).map(|_| (rnd() % 294) as u32).collect()).collect();
            let mut acc = vec![0i16; 64];
            let mut sink = 0i64;
            let t = Instant::now();
            for i in 0..calls {
                sink += unsafe { sq_call(&pt, &l1, &l1b, &ow2, &parent2, &rows[i & 4095], &mut acc, i & 7) } as i64;
            }
            let ns = t.elapsed().as_nanos() as f64 / calls as f64;
            black_box(sink);
            println!("[bench] sq-like W32x2 + L1 64->16, {n} row ops per perspective: {ns:6.1} ns per child bound (engine SmallNet measured 88 ns, D890)");
        }
    }

    #[target_feature(enable = "avx2")]
    unsafe fn zx_call(w: usize, table: &[i16], ow: &[i16], parent: &[i16], rows: &[u32], acc: &mut [i16]) -> i32 {
        let nv = w / 16;
        let ap = acc.as_mut_ptr();
        std::ptr::copy_nonoverlapping(parent.as_ptr(), ap, w);
        let mut i = 0;
        while i + 1 < rows.len() {
            let (po, pn) = (table.as_ptr().add(rows[i] as usize * w), table.as_ptr().add(rows[i + 1] as usize * w));
            for v in 0..nv {
                let a = _mm256_loadu_si256(ap.add(v * 16) as *const __m256i);
                let o = _mm256_loadu_si256(po.add(v * 16) as *const __m256i);
                let n = _mm256_loadu_si256(pn.add(v * 16) as *const __m256i);
                _mm256_storeu_si256(ap.add(v * 16) as *mut __m256i, _mm256_add_epi16(_mm256_sub_epi16(a, o), n));
            }
            i += 2;
        }
        let (zero, qa) = (_mm256_setzero_si256(), _mm256_set1_epi16(255));
        let mut sum = _mm256_setzero_si256();
        for v in 0..nv {
            let a = _mm256_loadu_si256(ap.add(v * 16) as *const __m256i);
            let c = _mm256_min_epi16(_mm256_max_epi16(a, zero), qa);
            let o = _mm256_loadu_si256(ow.as_ptr().add(v * 16) as *const __m256i);
            sum = _mm256_add_epi32(sum, _mm256_madd_epi16(_mm256_mullo_epi16(c, o), c));
        }
        hsum(sum)
    }

    #[target_feature(enable = "avx2")]
    unsafe fn hsum(v: __m256i) -> i32 {
        let s = _mm_add_epi32(_mm256_castsi256_si128(v), _mm256_extracti128_si256(v, 1));
        let s = _mm_hadd_epi32(s, s);
        let s = _mm_hadd_epi32(s, s);
        _mm_cvtsi128_si32(s)
    }

    #[target_feature(enable = "avx2")]
    #[allow(clippy::too_many_arguments)]
    unsafe fn sq_call(pt: &[i16], l1: &[i16], l1b: &[i32], ow: &[i16], parent: &[i16], rows: &[u32], acc: &mut [i16], bk: usize) -> i32 {
        let ap = acc.as_mut_ptr();
        std::ptr::copy_nonoverlapping(parent.as_ptr(), ap, 64);
        // rows: the first half update the side-to-move half (offset 0), the second half the other (offset 32)
        let n = rows.len() / 2;
        for (k, &r) in rows.iter().enumerate() {
            let half = (k >= n) as usize;
            let p = pt.as_ptr().add((r as usize) * 32);
            for v in 0..2 {
                let a = _mm256_loadu_si256(ap.add(half * 32 + v * 16) as *const __m256i);
                let x = _mm256_loadu_si256(p.add(v * 16) as *const __m256i);
                let y = if k & 1 == 0 { _mm256_add_epi16(a, x) } else { _mm256_sub_epi16(a, x) };
                _mm256_storeu_si256(ap.add(half * 32 + v * 16) as *mut __m256i, y);
            }
        }
        let (zero, one) = (_mm256_setzero_si256(), _mm256_set1_epi16(255));
        let mut x = [_mm256_setzero_si256(); 4];
        for v in 0..4 {
            x[v] = _mm256_min_epi16(_mm256_max_epi16(_mm256_loadu_si256(ap.add(v * 16) as *const __m256i), zero), one);
        }
        let mut out = 0i32;
        for i in 0..16 {
            let w = l1.as_ptr().add((bk * 16 + i) * 64);
            let mut s = _mm256_madd_epi16(x[0], _mm256_loadu_si256(w as *const __m256i));
            for v in 1..4 {
                s = _mm256_add_epi32(s, _mm256_madd_epi16(x[v], _mm256_loadu_si256(w.add(v * 16) as *const __m256i)));
            }
            let h = (hsum(s) + l1b[bk * 16 + i]).clamp(0, 255 * 255);
            out += h * ow[bk * 16 + i] as i32;
        }
        out
    }
}

#[cfg(not(target_arch = "x86_64"))]
mod bench_heads {
    pub fn run() {
        println!("[bench] x86_64 only");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn samp(own: u64, opp: u64, pown: u64, popp: u64) -> Sample {
        Sample { own, opp, blk: 0, pown, popp, parent: 0, hand: 0, target: 0, lp: 0, rank: 0, dis: 0, ent: 0, top1: 0, h: [0; HMAX] }
    }

    // D847: a +pol model must be EXACTLY its control at step 0 (zero policy rows + zero policy skip, same RNG stream)
    #[test]
    fn pol_model_equals_control_at_init() {
        for arch in [Arch::Sq, Arch::W2, Arch::DW2Sq, Arch::X3] {
            let a = Model::new(arch, 8, 3, Ext::default());
            let b = Model::new(arch, 8, 3, Ext { pol: true, dis: true, mv: true, ent: true, h1: true, src: true, rel: 2, zx: 0 });
            for (i, s) in [samp(0, 1 | 1 << 8, 1, 0), samp(1 << 20 | 1 << 21, 1 << 30, 1 << 20, 1 << 30 | 1 << 21)].iter().enumerate() {
                let mut s = *s;
                s.parent = 123;
                s.hand = -45;
                s.lp = -2345;
                s.rank = 7 + i as u8;
                s.dis = 160;
                s.ent = 1900;
                s.top1 = -700;
                s.h = [37; HMAX];
                assert_eq!(a.forward(&s).y.to_bits(), b.forward(&s).y.to_bits(), "{arch:?} sample {i}");
            }
            assert_eq!(b.np - a.np, (POL_NF + 2 * DIS_BINS + 2 * MV_ROWS + 4 * ENT_BINS + SRC_NF + 2 * rel_half(2)) * W + 2 * HMAX * W);
        }
    }

    #[test]
    fn rel_rows_match_hand() {
        // clone to b2 (8, row 1 col 1): radius 2 on the board = rows 0..3 x cols 0..3 = 16 cells minus the centre = 15 rows;
        // a1 (0) is the mover's stone at offset (-1,-1) -> offset index (1*5+1) = 6, role 0, state 0 -> row (0*25+6)*4+0 = 24
        let s = samp(0, 1 | 1 << 8, 1, 0);
        let mut v = Vec::new();
        assert_eq!(rel_rows(&s, 2, 0, &mut v), 15);
        assert!(v.contains(&24));
        assert!(v.iter().all(|&x| (x as usize) < rel_half(2)));
        // jump a1 (0) -> c3 (16): dest role 1 at row 2 col 2 = 5x5 - 1 = 24 rows; source role 2 at the corner = 3x3 - 1 = 8 rows
        let j = samp(1 << 14, 1 << 16, 1 << 14, 1);
        let mut v = Vec::new();
        assert_eq!(rel_rows(&j, 2, 0, &mut v), 32);
        // a3 (14) seen from the hole a1 at offset (+2, 0) -> offset (4*5+2) = 22, role 2, side to move (state 1)
        assert!(v.contains(&(((2 * 25 + 22) * 4 + 1) as u32)));
        // whole-board radius: the clone destination b2 sees all 48 other squares
        assert_eq!(rel_rows(&s, 6, 0, &mut Vec::new()), 48);
    }

    // D861: hand-worked shapes of the artificial 3x3 (cross + X) and the full 3x3 windows
    #[test]
    fn x3_and_w3_cells() {
        // square 24 (row 3, col 3): cross = centre, N 17, S 31, W 23, E 25; X = centre, NW 16, NE 18, SW 30, SE 32
        assert_eq!(x3_cells(48), [Some(24), Some(17), Some(31), Some(23), Some(25)]);
        assert_eq!(x3_cells(49), [Some(24), Some(16), Some(18), Some(30), Some(32)]);
        // corner a1 = square 0: cross has only S (7) and E (1); X only SE (8)
        assert_eq!(x3_cells(0), [Some(0), None, Some(7), None, Some(1)]);
        assert_eq!(x3_cells(1), [Some(0), None, None, None, Some(8)]);
        assert_eq!(w3_squares(0), [0, 1, 2, 7, 8, 9, 14, 15, 16]);
        assert_eq!(w3_squares(24), [32, 33, 34, 39, 40, 41, 46, 47, 48]);
        // codes stay inside their state range
        let all = (1u64 << 49) - 1;
        assert!(code3(x3_cells(48).into_iter(), all, 0) < X3S);
        assert!(code3(w3_squares(12).into_iter().map(Some), 0, all) < W3S);
        assert_eq!(code3(w3_squares(12).into_iter().map(Some), 0, all), W3S - 1);
    }

    #[test]
    fn touched_x3_w3_match_hand() {
        // same move as touched_counts_match_hand: changed squares a1 (0) and b2 (8)
        let s = samp(0, 1 | 1 << 8, 1, 0);
        // crosses containing 0: centres 0, 1, 7; containing 8: centres 8, 1, 7, 9, 15 -> {0,1,7,8,9,15} = 6
        // X shapes containing 0: centres 0, 8; containing 8: centres 8, 0, 2, 14, 16 -> {0,2,8,14,16} = 5 -> 11 shapes x 2
        assert_eq!(touched(Arch::X3, &s), 22);
        // 3x3 windows containing 0: (0,0); containing 8: (0,0),(0,1),(1,0),(1,1) -> 4 windows x 2
        assert_eq!(touched(Arch::W3, &s), 8);
    }

    #[test]
    fn src_rows_match_hand() {
        // clone: the mover (child opp) had nothing, clones to b2 (8) and converts a1 (0). Neighbours of b2 in order
        // [a1, b1, c1, a2, c2, a3, b3, c3]: only a1 = mover -> pattern 1; 1 mover piece borders b2; the side to move has no
        // stones -> reach 0 -> count 0*27 + 1*3 + 0 = 3
        let s = samp(0, 1 | 1 << 8, 1, 0);
        assert_eq!(src_rows(&s), Some((1, 3)));
        // jump: the mover jumps a1 (0) -> c3 (16), no flips; the side to move has a3 (14): distance 2 from a1 (jump reach)
        let j = samp(1 << 14, 1 << 16, 1 << 14, 1);
        // neighbours of a1: only b1 (1), a2 (7), b2 (8) are on the board, all empty -> pattern 0; 0 mover pieces border it
        assert_eq!(src_rows(&j), Some((SRC_PAT, 27 + 0 * 3 + 1)));
        // the side to move on b1 (1) = adjacent to the hole -> clone reach 2; b1 is neighbour index 4 (3 off-board above,
        // 1 off-board left) with state 2 -> pattern 2 * 3^4 = 162
        let j2 = samp(1 << 1, 1 << 16, 1 << 1, 1);
        assert_eq!(src_rows(&j2), Some((SRC_PAT + 162, 27 + 2)));
        // a pass (nothing changed) has no move square
        assert_eq!(src_rows(&samp(1, 2, 1, 2)), None);
    }

    #[test]
    fn pol_bins() {
        assert_eq!(pol_bin(0), 0);
        assert_eq!(pol_bin(-499), 0);
        assert_eq!(pol_bin(-500), 1);
        assert_eq!(pol_bin(-30000), POL_BINS - 1);
    }

    // hand-checked moves (child mover = side to move in the child; "own" = child mover's stones)
    #[test]
    fn touched_counts_match_hand() {
        // parent: child mover has a1 (bit 0); the opponent (who just moved) clones b2 (bit 8) and converts a1 -> opp.
        // child: own = {} ; opp = {a1, b2}. own^pown = a1 (1 bit), opp^popp = a1 + b2 (2 bits) -> sq 3 row ops.
        let s = samp(0, 1 | 1 << 8, 1, 0);
        assert_eq!(touched(Arch::Sq, &s), 3);
        // windows containing a1 or b2: a1 is only in window (0,0); b2 = square row 1 col 1 is in windows (0,0),(0,1),(1,0),(1,1)
        // -> 4 distinct changed windows -> w2 = 2 x 4 = 8 row ops, delta = 8, w2sq = 8 + 3, dsq = 8 + 3
        assert_eq!(touched(Arch::W2, &s), 8);
        assert_eq!(touched(Arch::Delta, &s), 8);
        assert_eq!(touched(Arch::W2Sq, &s), 11);
        assert_eq!(touched(Arch::DSq, &s), 11);
        assert_eq!(touched(Arch::DW2, &s), 16);
        assert_eq!(touched(Arch::DW2Sq, &s), 19);
    }

    fn zx_model_with_net() -> (Model, Vec<i16>) {
        let mut r = Rng(12345);
        let n = 2916 * 256;
        let net: Vec<i16> = (0..n + 513).map(|i| if i < n { (r.unit() * 60.0) as i16 } else if i < n + 256 { (r.unit() * 40.0) as i16 } else if i < n + 512 { (r.unit() * 50.0) as i16 } else { 123 }).collect();
        static N: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let path = std::env::temp_dir().join(format!("zx_test_{}_{}.bin", std::process::id(), N.fetch_add(1, std::sync::atomic::Ordering::Relaxed)));
        std::fs::write(&path, net.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>()).unwrap();
        let mut m = Model::new(Arch::W2, 8, 1, Ext { zx: 256, ..Ext::default() });
        m.load_zx(path.to_str().unwrap()).unwrap();
        let _ = std::fs::remove_file(path);
        (m, net)
    }

    // D1295: step 0 of the ZX head = the Zataxx float forward (clamp^2 dot, QA 255 / QB 64), whatever the bucket
    #[test]
    fn zx_head_step0_equals_zataxx_forward() {
        let (m, net) = zx_model_with_net();
        let n = 2916 * 256;
        let s = Sample { parent: 321, hand: -77, ..samp(1 | 1 << 9 | 1 << 20, 1 << 8 | 1 << 30 | 1 << 33, 0, 0) };
        let mut ix = Vec::new();
        feats(Arch::W2, s.own, s.opp, s.blk, s.pown, s.popp, &mut ix);
        assert_eq!(ix.len(), 36);
        let mut h: Vec<f32> = (0..256).map(|k| net[n + k] as f32 / 255.0).collect();
        for &i in &ix {
            for k in 0..256 {
                h[k] += net[i as usize * 256 + k] as f32 / 255.0;
            }
        }
        let want: f32 = (0..256).map(|k| { let a = h[k].clamp(0.0, 1.0); a * a * net[n + 256 + k] as f32 / 64.0 }).sum::<f32>() + 123.0 / (255.0 * 64.0);
        let got = m.forward(&s).y;
        assert!((got - want).abs() < 1e-4, "{got} vs {want}");
        // the unit-window state coding must be Zataxx's (us = 1, them = 2, powers of 3 over a1,b1,a2,b2)
        assert_eq!(win_code(0, 1, 1 << 1), 1 + 2 * 3);
    }

    // D1296: keeping every unit is the identity; dropping units keeps the head close (mean contribution folded into the bias)
    #[test]
    fn zx_shrink_identity_and_close() {
        let (m, _) = zx_model_with_net();
        let s = Sample { parent: 100, hand: 30, ..samp(1 | 1 << 9 | 1 << 20, 1 << 8 | 1 << 30 | 1 << 33, 0, 0) };
        let probe = vec![s; 4];
        let msq = zx_mean_sq(&m, &probe);
        let same = m.shrink_zx(256, &msq);
        assert_eq!(m.forward(&s).y.to_bits(), same.forward(&s).y.to_bits());
        let half = m.shrink_zx(128, &msq);
        assert_eq!(half.w, 128);
        // on the probe itself the folded bias makes the narrow output differ only by the dropped units' deviation from their mean
        let d = (m.forward(&s).y - half.forward(&s).y).abs();
        assert!(d < 0.5, "shrunk output moved by {d}");
    }

    // D1313: an 8-bucket ZX head expanded to 16 or 32 buckets gives the same output at every empties count
    #[test]
    fn zx_bucket_expansion_is_exact() {
        let (mut m8, _) = zx_model_with_net();
        for (i, v) in m8.p[m8.o_ob..m8.o_ob + 8].iter_mut().enumerate() {
            *v = 0.01 * i as f32;
        }
        for b in 0..8 {
            m8.p[m8.o_sk + b] = 0.1 * b as f32;
            m8.p[m8.o_l1w + b * 256 + 3] += 0.05 * b as f32;
        }
        static N: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let path = std::env::temp_dir().join(format!("zx_exp_{}_{}.snar", std::process::id(), N.fetch_add(1, std::sync::atomic::Ordering::Relaxed)));
        save_model(&m8, path.to_str().unwrap()).unwrap();
        for nb in [16usize, 32] {
            let mut mb = Model::new(Arch::W2, nb, 1, Ext { zx: 256, ..Ext::default() });
            mb.load_expand(path.to_str().unwrap()).unwrap();
            // positions with 0..=49 empties: own fills the first `f` squares in a stripe pattern, opp the next ones
            for filled in 0..=49usize {
                let (mut own, mut opp) = (0u64, 0u64);
                for sq in 0..filled {
                    if sq % 2 == 0 { own |= 1 << sq } else { opp |= 1 << sq }
                }
                let s = Sample { parent: 77, hand: -12, ..samp(own, opp, own, opp) };
                assert_eq!(m8.forward(&s).y.to_bits(), mb.forward(&s).y.to_bits(), "nb {nb} filled {filled}");
            }
        }
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn zx_head_gradient_matches_numeric() {
        let (mut m, _) = zx_model_with_net();
        // move the output skips and weights off the exact-parity point so every term is exercised
        for bk in 0..8 {
            m.p[m.o_sk + bk] = 0.3;
            m.p[m.o_sk + m.nb + bk] = -0.2;
        }
        let s = Sample { parent: 250, hand: -90, target: 40, ..samp(1 | 1 << 9 | 1 << 20, 1 << 8 | 1 << 30 | 1 << 33, 0, 0) };
        let f = m.forward(&s);
        let mut g = vec![0f32; m.np];
        m.backward(&s, &f, 1.0, &mut g);
        let bk = f.bk;
        let probes = [m.o_ob + bk, m.o_sk + bk, m.o_sk + m.nb + bk, m.o_l1w + bk * 256 + 5, m.o_ftb + 7, f.fu[3] as usize * 256 + 11, f.fu[10] as usize * 256 + 200];
        for &k in &probes {
            let old = m.p[k];
            let e = 1e-3f32;
            m.p[k] = old + e;
            let up = m.forward(&s).y;
            m.p[k] = old - e;
            let dn = m.forward(&s).y;
            m.p[k] = old;
            let num = (up - dn) / (2.0 * e);
            assert!((num - g[k]).abs() < 2e-2 * (1.0 + num.abs()), "param {k}: numeric {num} vs backward {}", g[k]);
        }
    }

    #[test]
    fn win_squares_are_2x2() {
        assert_eq!(win_squares(0), [0, 1, 7, 8]);
        assert_eq!(win_squares(35), [40, 41, 47, 48]);
    }

    #[test]
    fn tau_shift_gives_tau_violations() {
        let res: Vec<f64> = (0..10_000).map(|i| (i as f64 * 7919.0) % 1000.0 - 300.0).collect();
        let c = tau_shift(&res, 0.02);
        let v = res.iter().filter(|&&r| r < c).count();
        assert_eq!(v, 200, "exactly 2% of residuals below the shift");
    }
}

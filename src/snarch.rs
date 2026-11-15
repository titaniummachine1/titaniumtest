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
//!       pinball phase: to SATURATION by default (D839: --patience 4 --max-decays 3 --decay 0.3 --min-rel 2e-4 --max-epochs 400,
//!       best epoch restored and saved as DIR/snarch_<arch>_b<buckets>.snar); --fixed --epochs-pin 16 = old screen, never a result
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
}

const HMAX: usize = 48;
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
                _ => return Err(format!("unknown input group +{p} in {s}")),
            }
        }
        Ok((base, e))
    }
    fn suffix(self) -> String {
        let mut s = String::new();
        for (on, n) in [(self.pol, "+pol"), (self.dis, "+dis"), (self.mv, "+mv"), (self.ent, "+ent"), (self.h1, "+h1")] {
            if on {
                s.push_str(n);
            }
        }
        s
    }
    /// extra accumulator rows (both perspectives)
    fn nf(self) -> usize {
        (if self.pol { POL_NF } else { 0 }) + (if self.dis { 2 * DIS_BINS } else { 0 }) + (if self.mv { 2 * MV_ROWS } else { 0 }) + (if self.ent { 4 * ENT_BINS } else { 0 })
    }
    /// extra row updates per child per perspective in the engine (h1 is per node: 0 per child)
    fn touched(self) -> usize {
        2 * self.pol as usize + self.dis as usize + self.mv as usize + 2 * self.ent as usize
    }
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
        }
    }
    /// The full-child-input part of a design (None for pure delta) and whether it carries the delta block (D835).
    fn split(self) -> (Option<Arch>, bool) {
        match self {
            Arch::Sq | Arch::W2 | Arch::W2Sq => (Some(self), false),
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
}

struct Fwd {
    x: [f32; D2],
    hpre: [f32; H],
    h: [f32; H],
    y: f32,
    bk: usize,
    fu: Vec<u32>,
    ft: Vec<u32>,
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
        let nf = pol_off + ext.nf();
        debug_assert_eq!(o_ent + if ext.ent { 4 * ENT_BINS } else { 0 }, nf);
        let o_ftb = nf * W;
        let o_l1w = o_ftb + W;
        let o_l1b = o_l1w + nb * H * D2;
        let o_ow = o_l1b + nb * H;
        let o_ob = o_ow + nb * H;
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
        for v in &mut p[..pol_off * W] {
            *v = r.unit() * scale;
        }
        for v in &mut p[o_ftb..o_l1w] {
            *v = 0.25;
        }
        for v in &mut p[o_l1w..o_l1b] {
            *v = r.unit() * (3.0 / D2 as f32).sqrt();
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
        Model { arch, ext, o_pol, o_dis, o_mv, o_ent, nb, p, o_ftb, o_l1w, o_l1b, o_ow, o_ob, o_sk, o_wh, np }
    }

    fn name(&self) -> String {
        format!("{}{}", self.arch.name(), self.ext.suffix())
    }

    fn bucket(&self, s: &Sample) -> usize {
        let empties = 49 - (s.own | s.opp | s.blk).count_ones() as usize;
        (empties * self.nb / 50).min(self.nb - 1)
    }

    fn forward(&self, s: &Sample) -> Fwd {
        let p = &self.p;
        let mut fu = Vec::with_capacity(64);
        let mut ft = Vec::with_capacity(64);
        feats(self.arch, s.own, s.opp, s.blk, s.pown, s.popp, &mut fu);
        feats(self.arch, s.opp, s.own, s.blk, s.popp, s.pown, &mut ft);
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
        Fwd { x, hpre, h, y, bk, fu, ft }
    }

    fn backward(&self, s: &Sample, f: &Fwd, dy: f32, g: &mut [f32]) {
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
                        // D847: policy log-softmax + rank over the legal moves (engine call: logit_k(b, m, &[], kc))
                        let n = list.len();
                        let mut lg = vec![0f32; n];
                        if let Some(pol) = titanium::policy::global() {
                            let mut kc = titanium::policy::KCache::new();
                            for (i, l) in lg.iter_mut().enumerate() {
                                *l = pol.logit_k(&b, list.move_at(i), &[], &mut kc);
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
        tch += (touched(m.arch, s) + m.ext.touched()) as f64;
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
    max_epochs: usize,
    /// D844: watch slack@tau (the compared number) instead of the val pinball loss
    on_slack: bool,
}

#[allow(clippy::too_many_arguments)]
fn fit(m: &mut Model, train: &[Sample], va: &[Sample], vb: &[Sample], epochs: usize, tau: Option<f32>, lr0: f32, threads: usize, seed: u64, tag: &str, tau_r: f64, sat: Option<Sat>) {
    let batch = 16384usize;
    let np = m.np;
    let mut mm = vec![0f32; np];
    let mut vv = vec![0f32; np];
    let mut step = 0usize;
    let epochs = sat.map(|s| s.max_epochs).unwrap_or(epochs);
    let steps_total = epochs * (train.len() / batch).max(1);
    let mut lr_cur = lr0;
    let (mut best, mut best_ep, mut since, mut decays) = (f64::INFINITY, 0usize, 0usize, 0usize);
    let mut best_p: Vec<f32> = Vec::new();
    let mut idx: Vec<u32> = (0..train.len() as u32).collect();
    let mut rng = Rng(seed | 1);
    let mut grads: Vec<Vec<f32>> = (0..threads).map(|_| vec![0f32; np]).collect();
    let (b1, b2, eps) = (0.9f32, 0.999f32, 1e-8f32);
    for ep in 0..epochs {
        let t0 = Instant::now();
        for i in (1..idx.len()).rev() {
            let j = (rng.next() % (i as u64 + 1)) as usize;
            idx.swap(i, j);
        }
        let mut loss_sum = 0f64;
        for chunk in idx.chunks(batch) {
            if chunk.len() < batch / 2 {
                break;
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
            epochs,
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
            } else {
                since += 1;
                if cur < best {
                    // a tiny gain still counts as the best weights, not as progress
                    best = cur;
                    best_ep = ep + 1;
                    best_p.clone_from(&m.p);
                }
            }
            if since >= s.patience {
                if decays >= s.max_decays {
                    println!("[snarch] {} {tag} SATURATED: {} epochs without a {:.3}% gain of {} at the final lr; best epoch {best_ep} = {best:.6}", m.name(), s.patience, 100.0 * s.min_rel, if s.on_slack { "slack@tau" } else { "val_pin" });
                    break;
                }
                decays += 1;
                lr_cur *= s.decay;
                since = 0;
                println!("[snarch] {} {tag} plateau: lr -> {lr_cur:.2e} (decay {decays}/{})", m.name(), s.max_decays);
            }
            if ep + 1 == epochs {
                println!("[snarch] {} {tag} WARNING: hit --max-epochs {epochs} before saturation (not saturated)", m.name());
            }
        }
    }
    if sat.is_some() && !best_p.is_empty() {
        m.p = best_p;
    }
}

/// Binary dump of a trained research model (no JSON, D806): "SNAR" v1, arch name (16 B, zero-padded), buckets u32, params u32,
/// then the f32 params little-endian in the Model layout (ft rows, ft bias, l1 w, l1 b, out w, out b, skips).
fn save_model(m: &Model, path: &str) -> Result<(), String> {
    let mut b = Vec::with_capacity(32 + 4 * m.np);
    b.extend_from_slice(b"SNAR");
    b.extend_from_slice(&1u32.to_le_bytes());
    let mut name = [0u8; 16];
    name[..m.name().len()].copy_from_slice(m.name().as_bytes());
    b.extend_from_slice(&name);
    b.extend_from_slice(&(m.nb as u32).to_le_bytes());
    b.extend_from_slice(&(m.np as u32).to_le_bytes());
    for v in &m.p {
        b.extend_from_slice(&v.to_le_bytes());
    }
    std::fs::write(path, b).map_err(|e| format!("{path}: {e}"))
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

fn run(args: &[String]) -> Result<(), String> {
    let flag = |n: &str| args.iter().position(|a| a == n).and_then(|i| args.get(i + 1)).cloned();
    let need = |n: &str| flag(n).ok_or_else(|| format!("{n} is required"));
    let num = |n: &str, d: f64| flag(n).and_then(|x| x.parse::<f64>().ok()).unwrap_or(d);
    let net_path = need("--net")?;
    let train_stem = need("--train")?;
    let val_stem = need("--val")?;
    // D847: "w2+pol" = the arch plus the policy input
    let archs: Vec<(Arch, Ext)> = flag("--arch").unwrap_or_else(|| "sq,w2,w2sq,delta".into()).split(',').map(|s| { let s = s.trim(); let (a, e) = Ext::parse(s)?; Arch::parse(&a).map(|x| (x, e)).ok_or(format!("unknown arch {s}")) }).collect::<Result<_, _>>()?;
    let nb = num("--buckets", 8.0) as usize;
    let rows = num("--rows", 1_200_000.0) as usize;
    let val_rows = num("--val-rows", 60_000.0) as usize;
    let every = num("--child-every", 3.0) as u64;
    let e_mse = num("--epochs-mse", 4.0) as usize;
    let e_pin = num("--epochs-pin", 16.0) as usize;
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
    // D839 (user): the pinball phase trains to SATURATION by default; --fixed keeps the old fixed-epoch screen (never a result)
    let sat = if args.iter().any(|a| a == "--fixed") {
        None
    } else {
        Some(Sat {
            patience: num("--patience", 4.0) as usize,
            max_decays: num("--max-decays", 3.0) as usize,
            decay: num("--decay", 0.3) as f32,
            min_rel: num("--min-rel", 2e-4),
            max_epochs: num("--max-epochs", 400.0) as usize,
            on_slack: flag("--sat-metric").map(|v| v != "pin").unwrap_or(true),
        })
    };
    let save_dir = flag("--save-dir");
    if let Some(d) = &save_dir {
        std::fs::create_dir_all(d).map_err(|e| format!("{d}: {e}"))?;
    }

    let bytes = std::fs::read(&net_path).map_err(|e| format!("{net_path}: {e}"))?;
    let big = TrainableNet::from_tnn_bytes(&bytes)?;
    let t0 = Instant::now();
    let mut tr = Tdat::open(&train_stem)?;
    let mut va = Tdat::open(&val_stem)?;
    let trows = load_val(&mut tr, rows)?;
    let train = build_samples(&trows, &big, every, threads, data_seed);
    // D845: val split BY POSITION (alternating 1024-row blocks), so all children of one parent sit in one half
    let vrows = load_val(&mut va, val_rows)?;
    let (mut ra, mut rb) = (Vec::new(), Vec::new());
    for (i, blk) in vrows.chunks(1024).enumerate() {
        if i % 2 == 0 { ra.extend_from_slice(blk) } else { rb.extend_from_slice(blk) }
    }
    let val_a = build_samples(&ra, &big, (every / 2).max(1), threads, data_seed ^ 0x5555);
    let val_b = build_samples(&rb, &big, (every / 2).max(1), threads, data_seed ^ 0xAAAA);
    let pin_txt = match sat {
        Some(s) => format!("pinball to SATURATION on {} of val half A (patience {}, {} lr decays x{}, min gain {:.3}%, max {} epochs)", if s.on_slack { "slack@tau" } else { "val_pin" }, s.patience, s.max_decays, s.decay, 100.0 * s.min_rel, s.max_epochs),
        None => format!("{e_pin} pinball (FIXED screen, not a result)"),
    };
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
    for (arch, ext) in archs {
        for &lr in &lrs {
            for &seed in &seeds {
                let lr = lr as f32;
                let tag = format!("{}{}_lr{lr:.0e}_s{seed}_r{}", arch.name(), ext.suffix(), trows.len());
                let mut m = Model::new(arch, nb, seed, ext);
                println!("[snarch] {tag} : {} features, {} params", arch.nf(), m.np);
                let ta = Instant::now();
                fit(&mut m, &train, &val_a, &val_b, e_mse, None, lr, threads, seed, "MSE", tau as f64, None);
                fit(&mut m, &train, &val_a, &val_b, e_pin, Some(tau), lr, threads, seed ^ 7, "pinball", tau as f64, sat);
                let r = report(&m, &val_a, &val_b, tau as f64);
                if let Some(d) = &save_dir {
                    let path = format!("{d}/snarch_{tag}_b{nb}.snar");
                    save_model(&m, &path)?;
                    println!("[snarch] {tag} saved {path}");
                }
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

#[cfg(test)]
mod tests {
    use super::*;

    fn samp(own: u64, opp: u64, pown: u64, popp: u64) -> Sample {
        Sample { own, opp, blk: 0, pown, popp, parent: 0, hand: 0, target: 0, lp: 0, rank: 0, dis: 0, ent: 0, top1: 0, h: [0; HMAX] }
    }

    // D847: a +pol model must be EXACTLY its control at step 0 (zero policy rows + zero policy skip, same RNG stream)
    #[test]
    fn pol_model_equals_control_at_init() {
        for arch in [Arch::Sq, Arch::W2, Arch::DW2Sq] {
            let a = Model::new(arch, 8, 3, Ext::default());
            let b = Model::new(arch, 8, 3, Ext { pol: true, dis: true, mv: true, ent: true, h1: true });
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
            assert_eq!(b.np - a.np, (POL_NF + 2 * DIS_BINS + 2 * MV_ROWS + 4 * ENT_BINS) * W + 2 * HMAX * W);
        }
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

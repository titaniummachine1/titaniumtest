//! D1024: our trained HCE (lab hce-train weights JSON: bucketed term set + optional gataxx pair/mobility tables)
//! as a plain evaluator other engines can call (C:/gitProjects/zataxx-ab/hce swaps Zataxx's net for it).
//! Identical arithmetic to hce-train's validation scorer: cp = sum of weight * feature over fx(set, pairs, off, board).
use crate::hcecmp::{block, feats, masks, nbk, set_nbk, Masks};
use crate::pairs::Pairs;
use crate::util::ring1;
use std::cell::RefCell;
use titanium::Board;

pub struct HceEval {
    set: usize,
    w: Vec<f32>,
    pairs: Option<Pairs>,
    off: u32,
    ring: [u64; 49],
    m: Masks,
    // D1030 fast pair path (empty when the model has no pairs)
    pt: Vec<f32>,
    first: [u8; 80],
    second: [u8; 80],
    b3: [[u16; 16]; 16],
    mob_off: usize,
    cls: [u8; 49],
    edge: u64,
}

/// base-3 code of a 2x2 block from its own / enemy 4-bit masks (bit i = square s, s+1, s+7, s+8 -> digit x 3^i)
fn b3_table() -> [[u16; 16]; 16] {
    let mut t = [[0u16; 16]; 16];
    for m in 0..16 {
        for o in 0..16 {
            if m & o == 0 {
                t[m][o] = (0..4).map(|i| (if m >> i & 1 != 0 { 1 } else if o >> i & 1 != 0 { 2 } else { 0 }) * 3u16.pow(i as u32)).sum();
            }
        }
    }
    t
}

thread_local! {
    static BUF: RefCell<(Vec<(u16, f32)>, Vec<(u32, f32)>)> = const { RefCell::new((Vec::new(), Vec::new())) };
}

fn field<'a>(s: &'a str, key: &str) -> &'a str {
    let k = format!("\"{key}\":");
    let t = s.split(k.as_str()).nth(1).unwrap_or_else(|| panic!("weights JSON: no {key}"));
    t.split([',', '}']).next().unwrap().trim()
}

// D1029: C ABI for engines in other languages (gataxx/Java via java.lang.foreign): one HCE per process.
static FFI_HCE: std::sync::OnceLock<HceEval> = std::sync::OnceLock::new();

/// load the weights JSON (NUL-terminated UTF-8 path); returns 0 on success. Must be called before ti_hce_eval.
#[no_mangle]
pub extern "C" fn ti_hce_load(path: *const std::os::raw::c_char) -> i32 {
    if path.is_null() {
        return 1;
    }
    let p = unsafe { std::ffi::CStr::from_ptr(path) }.to_string_lossy().into_owned();
    match std::panic::catch_unwind(|| HceEval::load(&p)) {
        Ok(h) => {
            let _ = FFI_HCE.set(h);
            0
        }
        Err(_) => 2,
    }
}

/// HCE cp (our sigmoid(cp/400) scale) for the side to move; `me` / `op` in a1 = bit 0, row-major 7x7 (the HCE is D4-invariant)
#[no_mangle]
pub extern "C" fn ti_hce_eval(me: u64, op: u64) -> f32 {
    FFI_HCE.get().map_or(0.0, |h| h.eval(me, op))
}

/// `lab hce-eval --weights J --val V.tdat [--n 20]`: HCE cp of the first N val rows (same order as `lab val-fen`),
/// for parity checks of an engine that embeds HceEval (D1024)
pub fn cli(args: &[String]) -> std::process::ExitCode {
    let (Some(wp), Some(v)) = (crate::util::arg(args, "--weights"), crate::util::arg(args, "--val")) else {
        eprintln!("usage: lab hce-eval --weights J --val V.tdat [--n 20]");
        return std::process::ExitCode::from(2);
    };
    let n: usize = crate::util::arg(args, "--n").and_then(|s| s.parse().ok()).unwrap_or(20);
    let h = HceEval::load(&wp);
    if crate::util::arg(args, "--bench").is_some() {
        // D1030: parity fast vs reference + single-thread ns/eval (fast, reference, Zataxx net6 for scale)
        let rows = crate::hcetrain::read_rows(&v, n.max(1000));
        let pos: Vec<(u64, u64)> = rows.iter().map(crate::extval::own_opp).collect();
        let (mut md, mut sum) = (0f32, 0f64);
        for &(me, op) in &pos {
            md = md.max((h.eval(me, op) - h.eval_ref(me, op)).abs());
        }
        let time = |f: &dyn Fn(u64, u64) -> f32| -> f64 {
            let t = std::time::Instant::now();
            let mut acc = 0f64;
            for _ in 0..3 {
                for &(me, op) in &pos {
                    acc += f(me, op) as f64;
                }
            }
            std::hint::black_box(acc);
            t.elapsed().as_nanos() as f64 / (3 * pos.len()) as f64
        };
        let zp = "C:/gitProjects/zataxx-ab/default/src/net6.bin";
        let z = crate::extval::Zat::load(zp);
        let tf = time(&|a, b| h.eval(a, b));
        let tr = time(&|a, b| h.eval_ref(a, b));
        let tz = time(&|a, b| z.eval(a, b) as f32);
        let tt = time(&|a, b| {
            let bd = Board::from_occ(a, b, 0, 0);
            BUF.with(|c| {
                let (t, _) = &mut *c.borrow_mut();
                feats(h.set, &bd, &h.ring, &h.m, t);
                t.iter().map(|&(i, v)| h.w[i as usize] * v).sum::<f32>()
            })
        });
        println!("  of which HCE terms (feats + dot) {tt:.0} ns, pairs + mobility {:.0} ns", tf - tt);
        for &(me, op) in pos.iter().take(10) {
            sum += h.eval(me, op) as f64;
        }
        println!("hce-eval bench: {} positions | parity fast vs reference max |diff| {md:.4} cp | ns/eval: fast {tf:.0}, reference {tr:.0}, Zataxx net6 (lab port, full eval, no incremental) {tz:.0} | check {sum:.1}", pos.len());
        return std::process::ExitCode::SUCCESS;
    }
    for (i, r) in crate::hcetrain::read_rows(&v, n).iter().enumerate() {
        let (me, op) = crate::extval::own_opp(r);
        println!("{i}\t{:.2}", h.eval(me, op));
    }
    std::process::ExitCode::SUCCESS
}

impl HceEval {
    /// load a lab hce-train `*_weights.json`; sets the global bucket count (one HCE model per process)
    pub fn load(path: &str) -> HceEval {
        let s = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("{path}: {e}"));
        let set: usize = field(&s, "set").parse().unwrap();
        let buckets: usize = field(&s, "buckets").parse().unwrap();
        let base: usize = field(&s, "base_dims").parse().unwrap();
        let with_pairs = field(&s, "pairs") == "true";
        set_nbk(buckets);
        assert_eq!(base, nbk() * block(set), "{path}: base_dims {base} != buckets x block");
        let body = s.split("\"weights\":[").nth(1).and_then(|t| t.split(']').next()).expect("weights array");
        let w: Vec<f32> = body.split(',').map(|v| v.trim().parse::<f32>().unwrap()).collect();
        let pairs = if with_pairs { Some(Pairs::new()) } else { None };
        let dims = base + pairs.as_ref().map_or(0, |p| p.dims());
        assert_eq!(w.len(), dims, "{path}: {} weights, model needs {dims}", w.len());
        let (pt, first, second, mob_off) = match &pairs {
            Some(p) => {
                let (f, s) = p.first_second();
                (p.direct_table(&w, base), f, s, base + p.nclass)
            }
            None => (Vec::new(), [0; 80], [0; 80], 0),
        };
        HceEval { set, w, pairs, off: base as u32, ring: ring1(), m: masks(), pt, first, second, b3: b3_table(), mob_off, cls: std::array::from_fn(|q| crate::hcefit::class(q) as u8), edge: (0..49u64).filter(|&s| { let (r, c) = (s / 7, s % 7); r == 0 || r == 6 || c == 0 || c == 6 }).fold(0, |m, s| m | 1 << s) }
    }

    /// D1030: set-4 terms (hcecmp ours() + gataxx_terms()) summed straight into the weights, no feature list
    fn terms4(&self, me: u64, op: u64) -> f32 {
        use crate::util::{box1, victim_planes};
        const ALL: u64 = (1u64 << 49) - 1;
        const FA: u64 = 1 | 1 << 7 | 1 << 14 | 1 << 21 | 1 << 28 | 1 << 35 | 1 << 42; // file a
        const FG: u64 = FA << 6;
        let cnt = |p: &[u64; 4], q: usize| -> usize {
            (((p[0] >> q) & 1) | (((p[1] >> q) & 1) << 1) | (((p[2] >> q) & 1) << 2) | (((p[3] >> q) & 1) << 3)) as usize
        };
        let within = |a: u64| -> i32 {
            // unordered adjacent pairs inside `a` (8-neighbourhood) = hcecmp pairs(a, ring)
            let e = (a << 1) & !FA & ALL;
            let w = (a >> 1) & !FG;
            ((a & (a << 7) & ALL).count_ones() + (a & e).count_ones() + (a & (e << 7) & ALL).count_ones() + (a & (w << 7) & ALL).count_ones()) as i32
        };
        let e = ALL & !(me | op);
        let base = crate::hcecmp::bk(e) * block(4);
        let w = &self.w[base..base + block(4)];
        let box2 = |x: u64| box1(box1(x));
        let (bm, bo) = (box2(me), box2(op));
        let (lo, lm) = (bo & e, bm & e);
        let pe = victim_planes(e);
        let holes = e & !(pe[0] | pe[1] | pe[2] | pe[3]);
        let be = box1(e);
        let near_e = box2(e);
        let (pl_o, pl_m, ps_o, ps_m) = (victim_planes(lo), victim_planes(lm), victim_planes(op & near_e), victim_planes(me & near_e));
        let mat = me.count_ones() as f32 - op.count_ones() as f32;
        let mut s = w[0]
            + w[1] * mat
            + w[2] * ((me & be).count_ones() as f32 - (op & be).count_ones() as f32)
            + w[3] * ((holes & bm & !bo).count_ones() as f32 - (holes & bo & !bm).count_ones() as f32)
            + w[4] * (holes & bm & bo).count_ones() as f32
            + if lm == 0 { w[5] } else { 0.0 }
            + if lo == 0 { w[6] } else { 0.0 };
        let mut x = me;
        while x != 0 {
            let q = x.trailing_zeros() as usize;
            x &= x - 1;
            s += w[7 + self.cls[q] as usize * 9 + cnt(&pl_o, q)] + w[187 + cnt(&ps_o, q)];
        }
        let mut x = op;
        while x != 0 {
            let q = x.trailing_zeros() as usize;
            x &= x - 1;
            s += w[97 + self.cls[q] as usize * 9 + cnt(&pl_m, q)] + w[196 + cnt(&ps_m, q)];
        }
        // gataxx_terms: safe, edge, reach, cohesion
        let g = &w[crate::hcecmp::O_N..];
        let safe = (me & !be).count_ones() as f32 - (op & !be).count_ones() as f32;
        let ed = (me & self.edge).count_ones() as f32 - (op & self.edge).count_ones() as f32;
        let reach = (me & !box1(lo)).count_ones() as f32 - (op & !box1(lm)).count_ones() as f32;
        let coh = (within(me) - within(op)) as f32;
        s + g[0] + g[1] * mat + g[2] * safe + g[3] * ed + g[4] * reach + g[5] * coh
    }

    /// side-to-move view: `me` = stones of the side to move, `op` = the other side; returns cp (sigmoid(cp/400) scale).
    /// D1030 fast path: set-4 terms fused (no feature list), pair tables by one direct lookup per placement.
    pub fn eval(&self, me: u64, op: u64) -> f32 {
        let terms = if self.set == 4 {
            self.terms4(me, op)
        } else {
            let b = Board::from_occ(me, op, 0, 0);
            BUF.with(|c| {
                let (t, _) = &mut *c.borrow_mut();
                feats(self.set, &b, &self.ring, &self.m, t);
                t.iter().map(|&(i, v)| self.w[i as usize] * v).sum::<f32>()
            })
        };
        if self.pairs.is_none() {
            return terms;
        }
        let mut bl = [0u16; 41];
        for y in 0..6 {
            for x in 0..6 {
                let s = y * 7 + x;
                let m = ((me >> s) & 3) | ((me >> (s + 7)) & 3) << 2;
                let o = ((op >> s) & 3) | ((op >> (s + 7)) & 3) << 2;
                bl[s] = self.b3[m as usize][o as usize];
            }
        }
        let mut ps = 0f32;
        for p in 0..80 {
            ps += self.pt[p * 6561 + bl[self.first[p] as usize] as usize + 81 * bl[self.second[p] as usize] as usize];
        }
        let (a, bb) = Pairs::mob_index(me, op);
        terms + ps + self.w[self.mob_off + a] - self.w[self.mob_off + bb]
    }

    /// the reference scorer (hce-train's exact feature list); kept for parity checks of the fast path
    pub fn eval_ref(&self, me: u64, op: u64) -> f32 {
        let b = Board::from_occ(me, op, 0, 0);
        BUF.with(|c| {
            let (t, x) = &mut *c.borrow_mut();
            feats(self.set, &b, &self.ring, &self.m, t);
            x.clear();
            x.extend(t.iter().map(|&(i, v)| (i as u32, v)));
            if let Some(p) = &self.pairs {
                p.feats(me, op, self.off, x);
            }
            x.iter().map(|&(i, v)| self.w[i as usize] * v).sum::<f32>()
        })
    }
}

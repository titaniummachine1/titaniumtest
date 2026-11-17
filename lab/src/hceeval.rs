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
}

thread_local! {
    static BUF: RefCell<(Vec<(u16, f32)>, Vec<(u32, f32)>)> = const { RefCell::new((Vec::new(), Vec::new())) };
}

fn field<'a>(s: &'a str, key: &str) -> &'a str {
    let k = format!("\"{key}\":");
    let t = s.split(k.as_str()).nth(1).unwrap_or_else(|| panic!("weights JSON: no {key}"));
    t.split([',', '}']).next().unwrap().trim()
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
        HceEval { set, w, pairs, off: base as u32, ring: ring1(), m: masks() }
    }

    /// side-to-move view: `me` = stones of the side to move, `op` = the other side; returns cp (sigmoid(cp/400) scale)
    pub fn eval(&self, me: u64, op: u64) -> f32 {
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

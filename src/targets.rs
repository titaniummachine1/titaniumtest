//! H3: training targets. Port of `training/train_nnue.py::make_targets` (D162): proven mates are exact
//! 1/0 truth (no sigmoid squash, no outcome blend); everything else 0.75*sigmoid(clip(cp)/400)+0.25*outcome.
use crate::data::Rec;

pub const MATE: f32 = 100_000.0;
pub const MATE_BOUND_PLIES: f32 = 2000.0;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Target {
    pub target: f32,
    /// -1 unknown, 0 proven loss for side to move, 1 proven win
    pub proven: i8,
}

pub fn sigmoid_cp(cp: f32) -> f32 {
    1.0 / (1.0 + (-(cp.clamp(-1500.0, 1500.0)) / 400.0).exp())
}

/// `teacher_p3` replaces the sigmoid score part (teacher stage); proofs still override both.
pub fn make_target(cp: f32, outcome: f32, teacher_p3: Option<f32>) -> Target {
    let proven = if cp >= MATE - MATE_BOUND_PLIES {
        1
    } else if cp <= -(MATE - MATE_BOUND_PLIES) {
        0
    } else {
        -1
    };
    if proven >= 0 {
        return Target { target: proven as f32, proven };
    }
    let part = teacher_p3.unwrap_or_else(|| sigmoid_cp(cp));
    Target { target: 0.75 * part + 0.25 * outcome, proven }
}

pub fn target_of(r: &Rec) -> Target {
    make_target(r.cp, r.outcome, None)
}

/// Training target with soft proofs: a proven row becomes `blend * proof + (1 - blend) * sigmoid(cp5k)` (blend 1.0 = exact,
/// the D162 rule). Evaluation (`target_of`) always uses the exact target so val numbers stay comparable across runs.
pub fn target_train(r: &Rec, proof_blend: f32) -> Target {
    let t = target_of(r);
    if t.proven >= 0 && proof_blend < 1.0 {
        return Target { target: proof_blend * t.proven as f32 + (1.0 - proof_blend) * sigmoid_cp(r.cp5k), proven: t.proven };
    }
    t
}

/// Game-result weight by phase (engine D310, RESEARCH_2026_11_01 sec. 2): `w = [<=6 empties, 7..12, >=13]` replaces the fixed
/// 0.25 on non-proven rows, target = (1-w)*sigmoid(cp) + w*outcome. Training only; validation keeps `target_of`.
pub fn target_train_rw(r: &Rec, proof_blend: f32, rw: Option<[f32; 3]>) -> Target {
    let t = target_train(r, proof_blend);
    match rw {
        Some(w) if t.proven < 0 => {
            let empties = 49 - (r.occ0 | r.occ1).count_ones();
            let w = if empties <= 6 { w[0] } else if empties <= 12 { w[1] } else { w[2] };
            Target { target: (1.0 - w) * sigmoid_cp(r.cp) + w * r.outcome, proven: -1 }
        }
        _ => t,
    }
}

/// Val rows for a `--result-weight-phase` run: rewrites `outcome` so that `target_of` (0.75*sig + 0.25*outcome) equals the
/// run's training target; guard / plateau / early stop then watch the loss the run actually trains on.
pub fn rw_val_row(mut r: Rec, rw: [f32; 3]) -> Rec {
    let t = target_train_rw(&r, 1.0, Some(rw));
    if t.proven < 0 {
        r.outcome = (t.target - 0.75 * sigmoid_cp(r.cp)) / 0.25;
    }
    r
}

#[cfg(test)]
mod proof_blend_tests {
    use super::*;

    fn rec(cp: f32, cp5k: f32) -> Rec {
        Rec::from_bytes(&[0u8; 48]).tap(|r| {
            r.cp = cp;
            r.cp5k = cp5k;
            r.outcome = 1.0;
        })
    }
    trait Tap: Sized {
        fn tap(self, f: impl FnOnce(&mut Self)) -> Self;
    }
    impl Tap for Rec {
        fn tap(mut self, f: impl FnOnce(&mut Self)) -> Self {
            f(&mut self);
            self
        }
    }

    #[test]
    fn blend_one_is_exact_and_soft_pulls_toward_5k() {
        let proven_win = rec(99_000.0, 0.0);
        assert_eq!(target_train(&proven_win, 1.0).target, 1.0);
        assert!((target_train(&proven_win, 0.9).target - 0.95).abs() < 1e-6); // 0.9*1 + 0.1*0.5
        let proven_loss = rec(-99_000.0, 0.0);
        assert!((target_train(&proven_loss, 0.9).target - 0.05).abs() < 1e-6);
        let normal = rec(120.0, 120.0);
        assert_eq!(target_train(&normal, 0.9), target_of(&normal)); // non-proven rows untouched
    }

    #[test]
    fn result_weight_by_phase() {
        let mut open = rec(120.0, 120.0); // empty board: 49 empties -> w[2]
        assert_eq!(target_train_rw(&open, 1.0, None), target_of(&open));
        assert!((target_train_rw(&open, 1.0, Some([0.35, 0.04, 0.0])).target - sigmoid_cp(120.0)).abs() < 1e-6);
        open.occ0 = (1u64 << 44) - 1; // 44 stones -> 5 empties -> w[0]
        let t = target_train_rw(&open, 1.0, Some([0.35, 0.04, 0.0])).target;
        assert!((t - (0.65 * sigmoid_cp(120.0) + 0.35)).abs() < 1e-6);
        assert_eq!(target_train_rw(&rec(99_000.0, 0.0), 1.0, Some([0.35, 0.04, 0.0])).target, 1.0); // proofs untouched
        let v = rw_val_row(open, [0.35, 0.04, 0.0]);
        assert!((target_of(&v).target - t).abs() < 1e-6);
    }
}

/// `ti data check-targets --tdat X.tdat --ref ref.f32`: compare against the Python oracle's float32 targets.
pub fn run_check(args: &[String]) -> std::process::ExitCode {
    use std::io::Read;
    let (Some(t), Some(rf)) = (crate::flag_str(args, "--tdat"), crate::flag_str(args, "--ref")) else {
        eprintln!("usage: ti data check-targets --tdat <x.tdat> --ref <oracle targets f32 file>");
        return std::process::ExitCode::FAILURE;
    };
    let mut a = std::io::BufReader::new(std::fs::File::open(&t).unwrap());
    let mut b = std::io::BufReader::new(std::fs::File::open(&rf).unwrap());
    let (mut rec, mut rv) = ([0u8; crate::data::REC], [0u8; 4]);
    let (mut n, mut worst, mut proven, mut exact_bad) = (0u64, 0f32, 0u64, 0u64);
    while a.read_exact(&mut rec).is_ok() {
        if b.read_exact(&mut rv).is_err() {
            eprintln!("ref shorter than tdat");
            return std::process::ExitCode::FAILURE;
        }
        let want = f32::from_le_bytes(rv);
        let got = target_of(&Rec::from_bytes(&rec));
        if got.proven >= 0 {
            proven += 1;
            if got.target != want {
                exact_bad += 1;
            }
        }
        worst = worst.max((got.target - want).abs());
        n += 1;
    }
    println!("rows {n}  proven {proven}  proven-not-exact {exact_bad}  max|rust-oracle| {worst:.3e}");
    if exact_bad == 0 && worst < 2e-6 { std::process::ExitCode::SUCCESS } else { std::process::ExitCode::FAILURE }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn selftest_matches_python() {
        let cp = [0.0, 400.0, -1500.0, 99999.0, -99992.0, 98000.0, -97999.0, 99995.0f32];
        let oc = [0.5, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.5f32];
        let t: Vec<Target> = cp.iter().zip(oc).map(|(&c, o)| make_target(c, o, None)).collect();
        let pr: Vec<i8> = t.iter().map(|x| x.proven).collect();
        assert_eq!(pr, vec![-1, -1, -1, 1, 0, 1, -1, 1]);
        assert!(t[3].target == 1.0 && t[4].target == 0.0 && t[5].target == 1.0 && t[7].target == 1.0);
        assert!((t[0].target - 0.5).abs() < 1e-6);
        assert!((t[1].target - (0.75 / (1.0 + (-1.0f32).exp()) + 0.25)).abs() < 1e-6);
        assert!((t[6].target - 0.75 / (1.0 + (1500.0f32 / 400.0).exp())).abs() < 1e-6);
        let tt = make_target(0.0, 0.5, Some(0.3));
        assert!((tt.target - (0.75 * 0.3 + 0.25 * 0.5)).abs() < 1e-6);
        assert_eq!(make_target(99999.0, 0.0, Some(0.3)).target, 1.0);
    }
}

//! Stockfish Fishtest Pentanomial & Trinomial SPRT mathematics.
//!
//! Computes:
//! - Elo difference and 95% confidence interval from W-L-D or Pentanomial [LL, LD, DD+WL, WD, WW]
//! - Log-Likelihood Ratio (LLR) against [elo0, elo1] with bounds [ln(beta/(1-alpha)), ln((1-beta)/alpha)]
//!   For alpha = 0.05, beta = 0.05, bounds are [-2.9444, +2.9444].
//! - LOS (Likelihood Of Superiority).
//!
//! Gate rules (user order 2026-10-30): no verdict below `MIN_GAMES` (100 games = 50 complete pairs, both
//! seats of every line); pair-score variance floored at `VAR_FLOOR` so identical pairs (every line split
//! 1-1 by seat) cannot manufacture certainty; `seat_split` reports the share of 1-1 split pairs (> 0.5 =
//! the book is too unbalanced to measure strength).

/// No SPRT verdict below this many games.
pub const MIN_GAMES: u32 = 100;
/// Floor on the per-pair score variance (real gates measure ~0.10).
pub const VAR_FLOOR: f64 = 0.01;
/// Above this share of 1-1 split pairs the book lines are seat-decided.
pub const SEAT_SPLIT_WARN: f64 = 0.5;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SprtTally {
    pub wins: u32,
    pub losses: u32,
    pub draws: u32,
    /// Pentanomial pair buckets: [LL (0.0), LD (0.25), DD+WL (0.5), WD (0.75), WW (1.0)]
    pub penta: [u32; 5],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SprtVerdict {
    Continue,
    Pass,
    Fail,
}

#[derive(Clone, Copy, Debug)]
pub struct SprtStats {
    pub games: u32,
    pub pairs: u32,
    pub score_rate: f64,
    pub elo: f64,
    pub elo_ci95: f64,
    pub los: f64,
    pub llr: f64,
    pub lower_bound: f64,
    pub upper_bound: f64,
    /// Share of pairs that split 1-1 (`penta[2]` minus drawn pairs is not separable; DD is rare in Ataxx).
    pub seat_split: f64,
    pub verdict: SprtVerdict,
}

impl SprtTally {
    pub fn record_pair(&mut self, g1_pts_x2: u8, g2_pts_x2: u8) {
        for &pts in &[g1_pts_x2, g2_pts_x2] {
            match pts {
                2 => self.wins += 1,
                1 => self.draws += 1,
                0 => self.losses += 1,
                _ => panic!("record_pair: invalid game points {pts}"),
            }
        }
        let pair_pts = (g1_pts_x2 + g2_pts_x2) as usize;
        assert!(pair_pts <= 4, "record_pair: invalid pair sum {pair_pts}");
        self.penta[pair_pts] += 1;
    }

    pub fn games(&self) -> u32 {
        self.wins + self.losses + self.draws
    }

    pub fn pairs(&self) -> u32 {
        self.penta.iter().sum()
    }

    pub fn compute(&self, elo0: f64, elo1: f64, alpha: f64, beta: f64) -> SprtStats {
        let lower_bound = (beta / (1.0 - alpha)).ln();
        let upper_bound = ((1.0 - beta) / alpha).ln();
        let total_games = self.games();
        let total_pairs = self.pairs();

        if total_games == 0 {
            return SprtStats {
                games: 0,
                pairs: 0,
                score_rate: 0.5,
                elo: 0.0,
                elo_ci95: 0.0,
                los: 0.5,
                llr: 0.0,
                lower_bound,
                upper_bound,
                seat_split: 0.0,
                verdict: SprtVerdict::Continue,
            };
        }

        let n_g = total_games as f64;
        let score = (self.wins as f64 + 0.5 * self.draws as f64) / n_g;
        let clamped_s = score.clamp(1e-6, 1.0 - 1e-6);
        let elo = -400.0 * (1.0 / clamped_s - 1.0).log10();

        // Pentanomial mean and variance (Stockfish Fishtest regularized formulation)
        let (elo_ci95, los, llr) = if total_pairs >= 2 {
            let n_p = total_pairs as f64;
            let denom = n_p + 0.25;
            let probs: [f64; 5] = [
                (self.penta[0] as f64 + 0.05) / denom,
                (self.penta[1] as f64 + 0.05) / denom,
                (self.penta[2] as f64 + 0.05) / denom,
                (self.penta[3] as f64 + 0.05) / denom,
                (self.penta[4] as f64 + 0.05) / denom,
            ];
            let values: [f64; 5] = [0.0, 0.25, 0.5, 0.75, 1.0];
            let mu: f64 = probs.iter().zip(values.iter()).map(|(p, v)| p * v).sum();
            let var: f64 = probs
                .iter()
                .zip(values.iter())
                .map(|(p, v)| p * (v - mu) * (v - mu))
                .sum::<f64>()
                .max(VAR_FLOOR);
            let sigma_pair = (var / n_p).sqrt();

            let s_hi = (mu + 1.95996 * sigma_pair).clamp(1e-6, 1.0 - 1e-6);
            let s_lo = (mu - 1.95996 * sigma_pair).clamp(1e-6, 1.0 - 1e-6);
            let elo_hi = -400.0 * (1.0 / s_hi - 1.0).log10();
            let elo_lo = -400.0 * (1.0 / s_lo - 1.0).log10();
            let ci95 = (elo_hi - elo_lo) * 0.5;

            let z = (mu - 0.5) / sigma_pair;
            let los_val = 0.5 * (1.0 + erf_approx(z / std::f64::consts::SQRT_2));

            let s0 = 1.0 / (1.0 + 10.0f64.powf(-elo0 / 400.0));
            let s1 = 1.0 / (1.0 + 10.0f64.powf(-elo1 / 400.0));
            let llr_val = n_p * ((mu - s0) * (mu - s0) - (mu - s1) * (mu - s1)) * 0.5 / var;
            (ci95, los_val, llr_val)
        } else {
            (0.0, 0.5, 0.0)
        };

        let verdict = if total_games < MIN_GAMES {
            SprtVerdict::Continue
        } else if llr >= upper_bound {
            SprtVerdict::Pass
        } else if llr <= lower_bound {
            SprtVerdict::Fail
        } else {
            SprtVerdict::Continue
        };

        SprtStats {
            games: total_games,
            pairs: total_pairs,
            score_rate: score,
            elo,
            elo_ci95,
            los,
            llr,
            lower_bound,
            upper_bound,
            seat_split: if total_pairs > 0 { self.penta[2] as f64 / total_pairs as f64 } else { 0.0 },
            verdict,
        }
    }
}

// ---- D1346 (user): trinomial / pentanomial Elo + nElo and the fastchess "normalized" SPRT LLR ----
// Formulas: fastchess (Disservin) sprt.cpp / stats; normalized Elo and the MLE LLR after Michel Van den Bergh,
// "Comments on normalized Elo" (fishtest). The verdict of `compute` is unchanged; this is the extra, clearer readout.

/// nElo = 800 / ln 10 x (score - 0.5) / sigma (sigma = standard deviation of one GAME result).
const NELO_SCALE: f64 = 800.0 / std::f64::consts::LN_10;

/// Elo + nElo with 95% intervals for one sample model (games or pairs).
#[derive(Clone, Copy, Debug)]
pub struct EloNelo {
    pub elo: f64,
    pub elo_ci95: f64,
    pub nelo: f64,
    pub nelo_ci95: f64,
}

fn elo_of_score(s: f64) -> f64 {
    let s = s.clamp(1e-6, 1.0 - 1e-6);
    -400.0 * (1.0 / s - 1.0).log10()
}

/// `mu` = mean score, `se` = standard error of the mean, `sigma_game` = per-game standard deviation.
fn elo_nelo(mu: f64, se: f64, sigma_game: f64) -> EloNelo {
    let ci = (elo_of_score(mu + 1.95996 * se) - elo_of_score(mu - 1.95996 * se)) * 0.5;
    EloNelo { elo: elo_of_score(mu), elo_ci95: ci, nelo: (mu - 0.5) / sigma_game * NELO_SCALE, nelo_ci95: 1.95996 * se / sigma_game * NELO_SCALE }
}

fn regularize(count: u32) -> f64 {
    if count == 0 {
        1e-3
    } else {
        count as f64
    }
}

/// Maximum-likelihood distribution with normalized Elo `t_star` (Van den Bergh, section 4.1): the iterative scheme of
/// fastchess, the root of the (decreasing) equation 4.9 found by bisection.
fn mle_normalized(scores: &[f64], phat: &[f64], mu_ref: f64, t_star: f64) -> Vec<f64> {
    let n = scores.len();
    let mut p = vec![1.0 / n as f64; n];
    for _ in 0..10 {
        let mu: f64 = scores.iter().zip(&p).map(|(a, q)| a * q).sum();
        let var: f64 = scores.iter().zip(&p).map(|(a, q)| q * (a - mu) * (a - mu)).sum();
        let sigma = var.sqrt();
        let phi: Vec<f64> = scores.iter().map(|&a| a - mu_ref - 0.5 * t_star * sigma * (1.0 + ((a - mu) / sigma).powi(2))).collect();
        let u = phi.iter().cloned().fold(f64::INFINITY, f64::min);
        let v = phi.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        if !(u < 0.0 && v > 0.0) {
            break;
        }
        let f = |x: f64| -> f64 { phat.iter().zip(&phi).map(|(h, f)| h * f / (1.0 + x * f)).sum() };
        let (mut lo, mut hi) = (-1.0 / v, -1.0 / u);
        for _ in 0..200 {
            let mid = 0.5 * (lo + hi);
            if f(mid) > 0.0 {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        let theta = 0.5 * (lo + hi);
        let mut max_diff: f64 = 0.0;
        for i in 0..n {
            let np = phat[i] / (1.0 + theta * phi[i]);
            max_diff = max_diff.max((np - p[i]).abs());
            p[i] = np;
        }
        if max_diff < 1e-4 {
            break;
        }
    }
    p
}

/// fastchess `getLLR_normalized`: LLR of nElo `t0` against `t1` (already scaled to the sample model).
fn llr_normalized(counts: &[f64], scores: &[f64], t0: f64, t1: f64) -> f64 {
    let total: f64 = counts.iter().sum();
    let phat: Vec<f64> = counts.iter().map(|c| c / total).collect();
    let p0 = mle_normalized(scores, &phat, 0.5, t0);
    let p1 = mle_normalized(scores, &phat, 0.5, t1);
    total * phat.iter().zip(p0.iter().zip(&p1)).map(|(h, (a, b))| h * (b.ln() - a.ln())).sum::<f64>()
}

impl SprtTally {
    /// Trinomial (win / draw / loss games) Elo and nElo.
    pub fn trinomial(&self) -> Option<EloNelo> {
        let n = self.games() as f64;
        if n < 2.0 {
            return None;
        }
        let x = (self.wins as f64 + 0.5 * self.draws as f64) / n;
        let var = (self.wins as f64 * (1.0 - x).powi(2) + self.draws as f64 * (0.5 - x).powi(2) + self.losses as f64 * x * x) / n;
        if var <= 0.0 {
            return None;
        }
        Some(elo_nelo(x, (var / n).sqrt(), var.sqrt()))
    }

    /// Pentanomial (game pairs) Elo and nElo.
    pub fn pentanomial(&self) -> Option<EloNelo> {
        let n = self.pairs() as f64;
        if n < 2.0 {
            return None;
        }
        let values = [0.0, 0.25, 0.5, 0.75, 1.0];
        let mu: f64 = self.penta.iter().zip(values).map(|(c, v)| *c as f64 / n * v).sum();
        let var: f64 = self.penta.iter().zip(values).map(|(c, v)| *c as f64 / n * (v - mu) * (v - mu)).sum();
        if var <= 0.0 {
            return None;
        }
        Some(elo_nelo(mu, (var / n).sqrt(), (2.0 * var).sqrt()))
    }

    /// Normalized-model SPRT LLR on the game (trinomial) counts, bounds read as nElo.
    pub fn llr_trinomial_normalized(&self, nelo0: f64, nelo1: f64) -> f64 {
        let counts = [regularize(self.losses), regularize(self.draws), regularize(self.wins)];
        llr_normalized(&counts, &[0.0, 0.5, 1.0], nelo0 / NELO_SCALE, nelo1 / NELO_SCALE)
    }

    /// Normalized-model SPRT LLR on the pair (pentanomial) counts, bounds read as nElo.
    pub fn llr_pentanomial_normalized(&self, nelo0: f64, nelo1: f64) -> f64 {
        let counts = [regularize(self.penta[0]), regularize(self.penta[1]), regularize(self.penta[2]), regularize(self.penta[3]), regularize(self.penta[4])];
        let k = std::f64::consts::SQRT_2 / NELO_SCALE;
        llr_normalized(&counts, &[0.0, 0.25, 0.5, 0.75, 1.0], nelo0 * k, nelo1 * k)
    }

    /// The fastchess-style readout block (trinomial + pentanomial Elo / nElo and the two normalized SPRTs).
    pub fn report_block(&self, nelo0: f64, nelo1: f64, alpha: f64, beta: f64) -> String {
        let lower = (beta / (1.0 - alpha)).ln();
        let upper = ((1.0 - beta) / alpha).ln();
        let line = |name: &str, e: Option<EloNelo>| match e {
            Some(e) => format!("{name}: Elo {:+.1} +- {:.1}, nElo {:+.1} +- {:.1}\n", e.elo, e.elo_ci95, e.nelo, e.nelo_ci95),
            None => format!("{name}: n/a (too few samples)\n"),
        };
        let sprt = |name: &str, llr: f64| {
            let frac = if llr >= 0.0 { llr / upper } else { llr / lower };
            format!("SPRT ({name}, normalized): llr {llr:+.3} ({:.1}%), bounds [{lower:.2}, {upper:.2}], nElo [{nelo0:.1}, {nelo1:.1}]\n", frac * 100.0)
        };
        let mut s = String::new();
        s += &line("Trinomial   ", self.trinomial());
        s += &line("Pentanomial ", self.pentanomial());
        if self.games() >= 2 {
            s += &sprt("trinomial  ", self.llr_trinomial_normalized(nelo0, nelo1));
        }
        if self.pairs() >= 2 {
            s += &sprt("pentanomial", self.llr_pentanomial_normalized(nelo0, nelo1));
        }
        s.push_str("(readout only, fastchess formulas; the bounds above are read as nElo, the verdict line uses the Elo bounds)");
        s
    }
}

fn erf_approx(x: f64) -> f64 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.3275911 * x);
    let y = 1.0
        - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t
            + 0.254829592)
            * t
            * (-x * x).exp();
    sign * y
}

#[cfg(test)]
mod tests {
    use super::*;

    /// D1346: the fastchess readout reproduced from the user's pasted run (171-124-5, 300 games).
    #[test]
    fn trinomial_matches_the_pasted_fastchess_numbers() {
        let t = SprtTally { wins: 171, losses: 124, draws: 5, penta: [0; 5] };
        let e = t.trinomial().unwrap();
        assert!((e.elo - 54.9).abs() < 0.1, "elo {}", e.elo);
        assert!((e.elo_ci95 - 39.6).abs() < 0.15, "ci {}", e.elo_ci95);
        assert!((e.nelo - 55.6).abs() < 0.1, "nelo {}", e.nelo);
        assert!((e.nelo_ci95 - 39.3).abs() < 0.1, "nelo ci {}", e.nelo_ci95);
        let llr = t.llr_trinomial_normalized(0.0, 10.0);
        assert!((llr - 1.240).abs() < 0.005, "llr {llr}");
    }

    fn tally(pairs: &[((u8, u8), u32)]) -> SprtTally {
        let mut t = SprtTally::default();
        for &((a, b), n) in pairs {
            for _ in 0..n {
                t.record_pair(a, b);
            }
        }
        t
    }

    #[test]
    fn strong_candidate_passes_sprt() {
        let st = tally(&[((2, 2), 40), ((2, 0), 30), ((0, 0), 10)]).compute(0.0, 20.0, 0.05, 0.05);
        assert!(st.elo > 100.0);
        assert_eq!(st.verdict, SprtVerdict::Pass);
    }

    #[test]
    fn weaker_candidate_fails_sprt() {
        let st = tally(&[((0, 0), 40), ((2, 0), 30), ((2, 2), 10)]).compute(0.0, 20.0, 0.05, 0.05);
        assert!(st.elo < -100.0);
        assert_eq!(st.verdict, SprtVerdict::Fail);
    }

    #[test]
    fn no_verdict_below_min_games() {
        // 40 pairs = 80 games, all won: still no verdict (the old code decided after 8 pairs)
        let st = tally(&[((2, 2), 40)]).compute(0.0, 20.0, 0.05, 0.05);
        assert_eq!(st.games, 80);
        assert_eq!(st.verdict, SprtVerdict::Continue);
    }

    #[test]
    fn seat_split_book_never_passes_and_is_flagged() {
        // every line won by the same seat for both engines: no evidence of strength
        let st = tally(&[((2, 0), 100)]).compute(0.0, 20.0, 0.05, 0.05);
        assert_ne!(st.verdict, SprtVerdict::Pass);
        assert!(st.seat_split > SEAT_SPLIT_WARN);
        assert!(st.llr > -10.0, "variance floor must stop degenerate certainty, llr {}", st.llr);
    }

    #[test]
    fn matches_python_gate_summary_on_the_d154_regate() {
        // logs/gate_regate_d154_std_5k: penta [28, 8, 106, 10, 56] -> python gate_summary.py LLR +3.26 (elo1 20)
        let st = tally(&[((0, 0), 28), ((0, 1), 8), ((2, 0), 106), ((2, 1), 10), ((2, 2), 56)]).compute(0.0, 20.0, 0.05, 0.05);
        assert!((st.llr - 3.26).abs() < 0.05, "llr {}", st.llr);
        assert!((st.elo - 48.8).abs() < 1.0, "elo {}", st.elo);
        assert_eq!(st.verdict, SprtVerdict::Pass);
    }
}

//! Stockfish Fishtest Pentanomial & Trinomial SPRT mathematics.
//!
//! Computes:
//! - Elo difference and 95% confidence interval from W-L-D or Pentanomial [LL, LD, DD+WL, WD, WW]
//! - Log-Likelihood Ratio (LLR) against [elo0, elo1] with bounds [ln(beta/(1-alpha)), ln((1-beta)/alpha)]
//!   For alpha = 0.05, beta = 0.05, bounds are [-2.9444, +2.9444].
//! - LOS (Likelihood Of Superiority).

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
                .max(1e-4);
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

        let verdict = if total_pairs < 8 {
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
            verdict,
        }
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

    #[test]
    fn strong_candidate_passes_sprt() {
        let mut t = SprtTally::default();
        for _ in 0..40 {
            t.record_pair(2, 2); // WW
        }
        for _ in 0..10 {
            t.record_pair(2, 0); // WL
        }
        let st = t.compute(0.0, 5.0, 0.05, 0.05);
        assert!(st.elo > 100.0);
        assert!(st.llr > st.upper_bound);
        assert_eq!(st.verdict, SprtVerdict::Pass);
    }

    #[test]
    fn weaker_candidate_fails_sprt() {
        let mut t = SprtTally::default();
        for _ in 0..40 {
            t.record_pair(0, 0); // LL
        }
        for _ in 0..10 {
            t.record_pair(2, 0); // WL
        }
        let st = t.compute(0.0, 5.0, 0.05, 0.05);
        assert!(st.elo < -100.0);
        assert!(st.llr < st.lower_bound);
        assert_eq!(st.verdict, SprtVerdict::Fail);
    }
}

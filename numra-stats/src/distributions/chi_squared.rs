//! Chi-squared distribution.
//!
//! Author: Moussa Leblouba
//! Date: 9 February 2026
//! Modified: 2 May 2026

use numra_core::Scalar;
use rand::RngCore;

use super::gamma_dist::GammaDist;
use super::ContinuousDistribution;

/// Chi-squared distribution with k degrees of freedom.
///
/// This is a special case of Gamma(k/2, 1/2).
#[derive(Clone, Debug)]
pub struct ChiSquared<S: Scalar> {
    pub df: S,
    gamma: GammaDist<S>,
}

impl<S: Scalar> ChiSquared<S> {
    pub fn new(df: S) -> Self {
        let half = S::HALF;
        Self {
            df,
            gamma: GammaDist::new(df * half, half),
        }
    }
}

impl<S: Scalar> ContinuousDistribution<S> for ChiSquared<S> {
    fn pdf(&self, x: S) -> S {
        self.gamma.pdf(x)
    }

    fn cdf(&self, x: S) -> S {
        self.gamma.cdf(x)
    }

    fn quantile(&self, p: S) -> S {
        self.gamma.quantile(p)
    }

    fn mean(&self) -> S {
        self.df
    }

    fn variance(&self) -> S {
        self.df * S::TWO
    }

    fn sample(&self, rng: &mut dyn RngCore) -> S {
        self.gamma.sample(rng)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chi2_mean_variance() {
        let c = ChiSquared::new(5.0_f64);
        assert!((c.mean() - 5.0).abs() < 1e-14);
        assert!((c.variance() - 10.0).abs() < 1e-14);
    }

    #[test]
    fn test_chi2_cdf_at_zero() {
        let c = ChiSquared::new(3.0_f64);
        assert!(c.cdf(0.0).abs() < 1e-10);
    }

    #[test]
    fn test_chi2_quantile_roundtrip() {
        let c = ChiSquared::new(4.0_f64);
        for &p in &[0.1, 0.5, 0.9] {
            let x = c.quantile(p);
            let p2 = c.cdf(x);
            assert!((p - p2).abs() < 1e-5, "p={}, p2={}", p, p2);
        }
    }
}

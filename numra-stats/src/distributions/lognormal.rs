//! Log-normal distribution.
//!
//! Author: Moussa Leblouba
//! Date: 9 February 2026
//! Modified: 2 May 2026

use numra_core::Scalar;
use numra_special::{erf, erfinv};
use rand::RngCore;

use super::normal::Normal;
use super::ContinuousDistribution;

/// Log-normal distribution: ln(X) ~ N(mu, sigma^2).
#[derive(Clone, Debug)]
pub struct LogNormal<S: Scalar> {
    pub mu: S,
    pub sigma: S,
}

impl<S: Scalar> LogNormal<S> {
    pub fn new(mu: S, sigma: S) -> Self {
        Self { mu, sigma }
    }
}

impl<S: Scalar> ContinuousDistribution<S> for LogNormal<S> {
    fn pdf(&self, x: S) -> S {
        if x <= S::ZERO {
            return S::ZERO;
        }
        let two = S::TWO;
        let pi2 = S::from_f64(core::f64::consts::TAU);
        let z = (x.ln() - self.mu) / self.sigma;
        (S::ZERO - z * z / two).exp() / (x * self.sigma * pi2.sqrt())
    }

    fn cdf(&self, x: S) -> S {
        if x <= S::ZERO {
            return S::ZERO;
        }
        let sqrt2 = S::from_f64(core::f64::consts::SQRT_2);
        let half = S::HALF;
        let z = (x.ln() - self.mu) / (self.sigma * sqrt2);
        half * (S::ONE + erf(z))
    }

    fn quantile(&self, p: S) -> S {
        let sqrt2 = S::from_f64(core::f64::consts::SQRT_2);
        let two = S::TWO;
        (self.mu + self.sigma * sqrt2 * erfinv(two * p - S::ONE)).exp()
    }

    fn mean(&self) -> S {
        let half = S::HALF;
        (self.mu + half * self.sigma * self.sigma).exp()
    }

    fn variance(&self) -> S {
        let s2 = self.sigma * self.sigma;
        ((S::TWO * self.mu + s2).exp()) * (s2.exp() - S::ONE)
    }

    fn sample(&self, rng: &mut dyn RngCore) -> S {
        let normal = Normal::new(self.mu, self.sigma);
        normal.sample(rng).exp()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lognormal_pdf_positive_only() {
        let ln = LogNormal::new(0.0_f64, 1.0);
        assert!(ln.pdf(-1.0).abs() < 1e-14);
        assert!(ln.pdf(0.0).abs() < 1e-14);
        assert!(ln.pdf(1.0) > 0.0);
    }

    #[test]
    fn test_lognormal_cdf() {
        let ln = LogNormal::new(0.0_f64, 1.0);
        // CDF at x=1 should be 0.5 (since ln(1) = 0 = mean)
        assert!((ln.cdf(1.0) - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_lognormal_quantile_roundtrip() {
        let ln = LogNormal::new(1.0_f64, 0.5);
        for &p in &[0.1, 0.5, 0.9] {
            let x = ln.quantile(p);
            let p2 = ln.cdf(x);
            assert!((p - p2).abs() < 1e-8, "p={}, p2={}", p, p2);
        }
    }

    #[test]
    fn test_lognormal_mean() {
        let ln = LogNormal::new(0.0_f64, 1.0);
        // E[X] = exp(mu + sigma^2/2) = exp(0.5)
        assert!((ln.mean() - 0.5_f64.exp()).abs() < 1e-12);
    }
}

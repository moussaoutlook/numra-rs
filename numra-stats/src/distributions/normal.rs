//! Normal (Gaussian) distribution.
//!
//! Author: Moussa Leblouba
//! Date: 9 February 2026
//! Modified: 2 May 2026

use numra_core::Scalar;
use numra_special::{erf, erfinv};
use rand::RngCore;

use super::ContinuousDistribution;

/// Normal distribution N(mu, sigma^2).
#[derive(Clone, Debug)]
pub struct Normal<S: Scalar> {
    pub mu: S,
    pub sigma: S,
}

impl<S: Scalar> Normal<S> {
    pub fn new(mu: S, sigma: S) -> Self {
        Self { mu, sigma }
    }

    /// Standard normal N(0, 1).
    pub fn standard() -> Self {
        Self {
            mu: S::ZERO,
            sigma: S::ONE,
        }
    }
}

impl<S: Scalar> ContinuousDistribution<S> for Normal<S> {
    fn pdf(&self, x: S) -> S {
        let two = S::TWO;
        let pi2 = S::from_f64(core::f64::consts::TAU);
        let z = (x - self.mu) / self.sigma;
        (S::ZERO - z * z / two).exp() / (pi2.sqrt() * self.sigma)
    }

    fn cdf(&self, x: S) -> S {
        let sqrt2 = S::from_f64(core::f64::consts::SQRT_2);
        let half = S::HALF;
        let z = (x - self.mu) / (self.sigma * sqrt2);
        half * (S::ONE + erf(z))
    }

    fn quantile(&self, p: S) -> S {
        let two = S::TWO;
        let sqrt2 = S::from_f64(core::f64::consts::SQRT_2);
        self.mu + self.sigma * sqrt2 * erfinv(two * p - S::ONE)
    }

    fn mean(&self) -> S {
        self.mu
    }

    fn variance(&self) -> S {
        self.sigma * self.sigma
    }

    fn sample(&self, rng: &mut dyn RngCore) -> S {
        // Box-Muller transform
        let u1 = random_uniform_01::<S>(rng);
        let u2 = random_uniform_01::<S>(rng);
        let two = S::TWO;
        let pi2 = S::from_f64(core::f64::consts::TAU);
        let z = (S::ZERO - two * u1.ln()).sqrt() * (pi2 * u2).cos();
        self.mu + self.sigma * z
    }
}

/// Generate a uniform random number in (0, 1).
pub(crate) fn random_uniform_01<S: Scalar>(rng: &mut dyn RngCore) -> S {
    // Generate u32 and convert to (0, 1)
    let bits = rng.next_u32();
    // Avoid exactly 0 for log safety
    S::from_f64((bits as f64 + 1.0) / (u32::MAX as f64 + 2.0))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normal_pdf_at_mean() {
        let n = Normal::new(0.0_f64, 1.0);
        let peak = 1.0 / (2.0 * core::f64::consts::PI).sqrt();
        assert!((n.pdf(0.0) - peak).abs() < 1e-12);
    }

    #[test]
    fn test_normal_cdf_at_mean() {
        let n = Normal::new(0.0_f64, 1.0);
        assert!((n.cdf(0.0) - 0.5).abs() < 1e-12);
    }

    #[test]
    fn test_normal_cdf_tails() {
        let n = Normal::<f64>::standard();
        assert!(n.cdf(-5.0) < 1e-5);
        assert!(n.cdf(5.0) > 1.0 - 1e-5);
    }

    #[test]
    fn test_normal_quantile_roundtrip() {
        let n = Normal::new(2.0_f64, 3.0);
        for &p in &[0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99] {
            let x = n.quantile(p);
            let p2 = n.cdf(x);
            assert!((p - p2).abs() < 1e-10, "p={}, p2={}", p, p2);
        }
    }

    #[test]
    fn test_normal_mean_variance() {
        let n = Normal::new(3.0_f64, 2.0);
        assert!((n.mean() - 3.0).abs() < 1e-14);
        assert!((n.variance() - 4.0).abs() < 1e-14);
    }

    #[test]
    fn test_normal_sample() {
        use rand::SeedableRng;
        let n = Normal::new(0.0_f64, 1.0);
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let samples = n.sample_n(&mut rng, 10000);
        let mean: f64 = samples.iter().sum::<f64>() / samples.len() as f64;
        assert!(mean.abs() < 0.1, "sample mean = {}", mean);
    }
}

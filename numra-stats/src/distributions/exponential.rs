//! Exponential distribution.
//!
//! Author: Moussa Leblouba
//! Date: 9 February 2026
//! Modified: 2 May 2026

use numra_core::Scalar;
use rand::RngCore;

use super::normal::random_uniform_01;
use super::ContinuousDistribution;

/// Exponential distribution with rate parameter lambda.
#[derive(Clone, Debug)]
pub struct Exponential<S: Scalar> {
    pub lambda: S,
}

impl<S: Scalar> Exponential<S> {
    pub fn new(lambda: S) -> Self {
        Self { lambda }
    }
}

impl<S: Scalar> ContinuousDistribution<S> for Exponential<S> {
    fn pdf(&self, x: S) -> S {
        if x < S::ZERO {
            S::ZERO
        } else {
            self.lambda * (S::ZERO - self.lambda * x).exp()
        }
    }

    fn cdf(&self, x: S) -> S {
        if x < S::ZERO {
            S::ZERO
        } else {
            S::ONE - (S::ZERO - self.lambda * x).exp()
        }
    }

    fn quantile(&self, p: S) -> S {
        // Inverse CDF: -ln(1-p) / lambda
        (S::ZERO - (S::ONE - p).ln()) / self.lambda
    }

    fn mean(&self) -> S {
        S::ONE / self.lambda
    }

    fn variance(&self) -> S {
        S::ONE / (self.lambda * self.lambda)
    }

    fn sample(&self, rng: &mut dyn RngCore) -> S {
        let u = random_uniform_01::<S>(rng);
        self.quantile(u)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_exponential_pdf() {
        let e = Exponential::new(2.0_f64);
        assert!((e.pdf(0.0) - 2.0).abs() < 1e-12);
        assert!((e.pdf(1.0) - 2.0 * (-2.0_f64).exp()).abs() < 1e-12);
    }

    #[test]
    fn test_exponential_cdf() {
        let e = Exponential::new(1.0_f64);
        assert!(e.cdf(0.0).abs() < 1e-14);
        assert!((e.cdf(1.0) - (1.0 - (-1.0_f64).exp())).abs() < 1e-12);
    }

    #[test]
    fn test_exponential_quantile_roundtrip() {
        let e = Exponential::new(3.0_f64);
        for &p in &[0.1, 0.5, 0.9, 0.99] {
            let x = e.quantile(p);
            let p2 = e.cdf(x);
            assert!((p - p2).abs() < 1e-12, "p={}, p2={}", p, p2);
        }
    }

    #[test]
    fn test_exponential_mean_variance() {
        let e = Exponential::new(0.5_f64);
        assert!((e.mean() - 2.0).abs() < 1e-14);
        assert!((e.variance() - 4.0).abs() < 1e-14);
    }
}

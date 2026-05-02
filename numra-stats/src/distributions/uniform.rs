//! Continuous uniform distribution.
//!
//! Author: Moussa Leblouba
//! Date: 9 February 2026
//! Modified: 2 May 2026

use numra_core::Scalar;
use rand::RngCore;

use super::normal::random_uniform_01;
use super::ContinuousDistribution;

/// Continuous uniform distribution on [a, b].
#[derive(Clone, Debug)]
pub struct Uniform<S: Scalar> {
    pub a: S,
    pub b: S,
}

impl<S: Scalar> Uniform<S> {
    pub fn new(a: S, b: S) -> Self {
        Self { a, b }
    }
}

impl<S: Scalar> ContinuousDistribution<S> for Uniform<S> {
    fn pdf(&self, x: S) -> S {
        if x >= self.a && x <= self.b {
            S::ONE / (self.b - self.a)
        } else {
            S::ZERO
        }
    }

    fn cdf(&self, x: S) -> S {
        if x < self.a {
            S::ZERO
        } else if x > self.b {
            S::ONE
        } else {
            (x - self.a) / (self.b - self.a)
        }
    }

    fn quantile(&self, p: S) -> S {
        self.a + p * (self.b - self.a)
    }

    fn mean(&self) -> S {
        let half = S::HALF;
        (self.a + self.b) * half
    }

    fn variance(&self) -> S {
        let twelve = S::from_f64(12.0);
        let d = self.b - self.a;
        d * d / twelve
    }

    fn sample(&self, rng: &mut dyn RngCore) -> S {
        let u = random_uniform_01::<S>(rng);
        self.a + u * (self.b - self.a)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_uniform_pdf() {
        let u = Uniform::new(0.0_f64, 1.0);
        assert!((u.pdf(0.5) - 1.0).abs() < 1e-14);
        assert!((u.pdf(-0.1)).abs() < 1e-14);
        assert!((u.pdf(1.1)).abs() < 1e-14);
    }

    #[test]
    fn test_uniform_cdf() {
        let u = Uniform::new(2.0_f64, 4.0);
        assert!((u.cdf(2.0)).abs() < 1e-14);
        assert!((u.cdf(3.0) - 0.5).abs() < 1e-14);
        assert!((u.cdf(4.0) - 1.0).abs() < 1e-14);
    }

    #[test]
    fn test_uniform_quantile() {
        let u = Uniform::new(0.0_f64, 10.0);
        assert!((u.quantile(0.0)).abs() < 1e-14);
        assert!((u.quantile(0.5) - 5.0).abs() < 1e-14);
        assert!((u.quantile(1.0) - 10.0).abs() < 1e-14);
    }

    #[test]
    fn test_uniform_mean_variance() {
        let u = Uniform::new(0.0_f64, 12.0);
        assert!((u.mean() - 6.0).abs() < 1e-14);
        assert!((u.variance() - 12.0).abs() < 1e-14);
    }
}

//! F-distribution.
//!
//! Author: Moussa Leblouba
//! Date: 9 February 2026
//! Modified: 2 May 2026

use numra_core::Scalar;
use numra_special::{betainc, lgamma};
use rand::RngCore;

use super::gamma_dist::GammaDist;
use super::ContinuousDistribution;

/// F-distribution with d1 and d2 degrees of freedom.
#[derive(Clone, Debug)]
pub struct FDist<S: Scalar> {
    pub df1: S,
    pub df2: S,
}

impl<S: Scalar> FDist<S> {
    pub fn new(df1: S, df2: S) -> Self {
        Self { df1, df2 }
    }
}

impl<S: Scalar> ContinuousDistribution<S> for FDist<S> {
    fn pdf(&self, x: S) -> S {
        if x <= S::ZERO {
            return S::ZERO;
        }
        let half = S::HALF;
        let d1 = self.df1;
        let d2 = self.df2;
        let log_pdf = half * d1 * (d1 * x / (d1 * x + d2)).ln()
            + half * d2 * (d2 / (d1 * x + d2)).ln()
            - x.ln()
            - lbeta(d1 * half, d2 * half);
        log_pdf.exp()
    }

    fn cdf(&self, x: S) -> S {
        if x <= S::ZERO {
            return S::ZERO;
        }
        let half = S::HALF;
        let d1 = self.df1;
        let d2 = self.df2;
        let z = d1 * x / (d1 * x + d2);
        betainc(d1 * half, d2 * half, z)
    }

    fn quantile(&self, p: S) -> S {
        if p <= S::ZERO {
            return S::ZERO;
        }
        if p >= S::ONE {
            return S::INFINITY;
        }
        // Newton iteration
        let mut x = S::ONE; // initial guess
        for _ in 0..100 {
            let f_val = self.cdf(x) - p;
            let f_prime = self.pdf(x);
            if f_prime.to_f64().abs() < 1e-300 {
                break;
            }
            let step = f_val / f_prime;
            x -= step;
            if x <= S::ZERO {
                x = S::from_f64(1e-10);
            }
            if step.to_f64().abs() < 1e-12 * x.to_f64().abs() {
                break;
            }
        }
        x
    }

    fn mean(&self) -> S {
        let two = S::TWO;
        if self.df2 > two {
            self.df2 / (self.df2 - two)
        } else {
            S::INFINITY
        }
    }

    fn variance(&self) -> S {
        let two = S::TWO;
        let four = S::from_f64(4.0);
        if self.df2 > four {
            let d1 = self.df1;
            let d2 = self.df2;
            two * d2 * d2 * (d1 + d2 - two) / (d1 * (d2 - two) * (d2 - two) * (d2 - four))
        } else {
            S::INFINITY
        }
    }

    fn sample(&self, rng: &mut dyn RngCore) -> S {
        // F = (X1/d1) / (X2/d2) where Xi ~ Chi2(di)
        let half = S::HALF;
        let g1 = GammaDist::new(self.df1 * half, half);
        let g2 = GammaDist::new(self.df2 * half, half);
        let x1 = g1.sample(rng);
        let x2 = g2.sample(rng);
        (x1 / self.df1) / (x2 / self.df2)
    }
}

fn lbeta<S: Scalar>(a: S, b: S) -> S {
    lgamma(a) + lgamma(b) - lgamma(a + b)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_f_dist_mean() {
        let f = FDist::new(5.0_f64, 10.0);
        assert!((f.mean() - 10.0 / 8.0).abs() < 1e-12);
    }

    #[test]
    fn test_f_dist_cdf_at_zero() {
        let f = FDist::new(3.0_f64, 5.0);
        assert!(f.cdf(0.0).abs() < 1e-10);
    }

    #[test]
    fn test_f_dist_quantile_roundtrip() {
        let f = FDist::new(5.0_f64, 10.0);
        for &p in &[0.1, 0.5, 0.9] {
            let x = f.quantile(p);
            let p2 = f.cdf(x);
            assert!((p - p2).abs() < 1e-5, "p={}, p2={}", p, p2);
        }
    }
}

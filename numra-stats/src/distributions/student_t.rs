//! Student's t-distribution.
//!
//! Author: Moussa Leblouba
//! Date: 9 February 2026
//! Modified: 2 May 2026

use numra_core::Scalar;
use numra_special::{betainc, lgamma};
use rand::RngCore;

use super::gamma_dist::{sample_standard_normal, GammaDist};
use super::ContinuousDistribution;

/// Student's t-distribution with nu degrees of freedom.
#[derive(Clone, Debug)]
pub struct StudentT<S: Scalar> {
    pub df: S,
}

impl<S: Scalar> StudentT<S> {
    pub fn new(df: S) -> Self {
        Self { df }
    }
}

impl<S: Scalar> ContinuousDistribution<S> for StudentT<S> {
    fn pdf(&self, x: S) -> S {
        let half = S::HALF;
        let nu = self.df;
        let log_pdf = lgamma((nu + S::ONE) * half)
            - lgamma(nu * half)
            - half * (nu * S::from_f64(core::f64::consts::PI)).ln()
            - (nu + S::ONE) * half * (S::ONE + x * x / nu).ln();
        log_pdf.exp()
    }

    fn cdf(&self, x: S) -> S {
        let half = S::HALF;
        let nu = self.df;
        let t2 = x * x;
        let p = betainc(nu * half, half, nu / (nu + t2));
        if x >= S::ZERO {
            S::ONE - half * p
        } else {
            half * p
        }
    }

    fn quantile(&self, p: S) -> S {
        if p <= S::ZERO {
            return S::NEG_INFINITY;
        }
        if p >= S::ONE {
            return S::INFINITY;
        }
        // Newton iteration on CDF
        let mut x = super::gamma_dist::normal_quantile_approx(p);
        for _ in 0..100 {
            let f_val = self.cdf(x) - p;
            let f_prime = self.pdf(x);
            if f_prime.to_f64().abs() < 1e-300 {
                break;
            }
            let step = f_val / f_prime;
            x -= step;
            if step.to_f64().abs() < 1e-12 * (S::ONE + x.abs()).to_f64() {
                break;
            }
        }
        x
    }

    fn mean(&self) -> S {
        // Mean is 0 for df > 1, undefined otherwise
        S::ZERO
    }

    fn variance(&self) -> S {
        let two = S::TWO;
        if self.df > two {
            self.df / (self.df - two)
        } else {
            S::INFINITY
        }
    }

    fn sample(&self, rng: &mut dyn RngCore) -> S {
        // t = Z / sqrt(V/nu) where Z~N(0,1) and V~Chi2(nu)
        let z = sample_standard_normal::<S>(rng);
        let half = S::HALF;
        let chi2 = GammaDist::new(self.df * half, half);
        let v = chi2.sample(rng);
        z / (v / self.df).sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_student_t_symmetry() {
        let t = StudentT::new(5.0_f64);
        assert!((t.pdf(1.0) - t.pdf(-1.0)).abs() < 1e-12);
        assert!((t.cdf(0.0) - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_student_t_cdf_tails() {
        let t = StudentT::new(10.0_f64);
        assert!(t.cdf(-10.0) < 0.001);
        assert!(t.cdf(10.0) > 0.999);
    }

    #[test]
    fn test_student_t_quantile_roundtrip() {
        let t = StudentT::new(5.0_f64);
        for &p in &[0.05, 0.25, 0.5, 0.75, 0.95] {
            let x = t.quantile(p);
            let p2 = t.cdf(x);
            assert!((p - p2).abs() < 1e-6, "p={}, p2={}", p, p2);
        }
    }

    #[test]
    fn test_student_t_variance() {
        let t = StudentT::new(5.0_f64);
        assert!((t.variance() - 5.0 / 3.0).abs() < 1e-14);
    }
}

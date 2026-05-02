//! Beta distribution.
//!
//! Author: Moussa Leblouba
//! Date: 9 February 2026
//! Modified: 2 May 2026

use numra_core::Scalar;
use numra_special::{betainc, lgamma};
use rand::RngCore;

use super::gamma_dist::GammaDist;
use super::ContinuousDistribution;

/// Beta distribution on [0, 1] with parameters alpha and beta.
#[derive(Clone, Debug)]
pub struct BetaDist<S: Scalar> {
    pub alpha: S,
    pub beta: S,
}

impl<S: Scalar> BetaDist<S> {
    pub fn new(alpha: S, beta: S) -> Self {
        Self { alpha, beta }
    }
}

impl<S: Scalar> ContinuousDistribution<S> for BetaDist<S> {
    fn pdf(&self, x: S) -> S {
        if x < S::ZERO || x > S::ONE {
            return S::ZERO;
        }
        let log_pdf = (self.alpha - S::ONE) * x.ln() + (self.beta - S::ONE) * (S::ONE - x).ln()
            - lbeta(self.alpha, self.beta);
        log_pdf.exp()
    }

    fn cdf(&self, x: S) -> S {
        if x <= S::ZERO {
            return S::ZERO;
        }
        if x >= S::ONE {
            return S::ONE;
        }
        betainc(self.alpha, self.beta, x)
    }

    fn quantile(&self, p: S) -> S {
        if p <= S::ZERO {
            return S::ZERO;
        }
        if p >= S::ONE {
            return S::ONE;
        }
        // Initial guess: use mean of the distribution, adjusted toward p
        let mu = self.mean();
        let mut x = mu;
        // Refine initial guess: blend mean with p-based guess
        let p_f64 = p.to_f64();
        if p_f64 < 0.05 {
            x = S::from_f64(p_f64.powf(1.0 / self.alpha.to_f64()));
        } else if p_f64 > 0.95 {
            x = S::ONE - S::from_f64((1.0 - p_f64).powf(1.0 / self.beta.to_f64()));
        }
        // Clamp initial guess
        if x <= S::ZERO {
            x = S::from_f64(0.01);
        }
        if x >= S::ONE {
            x = S::from_f64(0.99);
        }
        // Newton iteration on CDF
        for _ in 0..100 {
            let f_val = self.cdf(x) - p;
            let f_prime = self.pdf(x);
            if f_prime.to_f64().abs() < 1e-300 {
                break;
            }
            let step = f_val / f_prime;
            // Halve step if it would move outside (0, 1)
            let mut s = step;
            for _ in 0..10 {
                let xn = x - s;
                if xn.to_f64() > 1e-10 && xn.to_f64() < 1.0 - 1e-10 {
                    break;
                }
                s *= S::HALF;
            }
            x -= s;
            // Hard clamp
            if x <= S::ZERO {
                x = S::from_f64(1e-10);
            }
            if x >= S::ONE {
                x = S::from_f64(1.0 - 1e-10);
            }
            if s.to_f64().abs() < 1e-12 {
                break;
            }
        }
        x
    }

    fn mean(&self) -> S {
        self.alpha / (self.alpha + self.beta)
    }

    fn variance(&self) -> S {
        let ab = self.alpha + self.beta;
        self.alpha * self.beta / (ab * ab * (ab + S::ONE))
    }

    fn sample(&self, rng: &mut dyn RngCore) -> S {
        // Sample via gamma: if X ~ Gamma(alpha, 1) and Y ~ Gamma(beta, 1),
        // then X/(X+Y) ~ Beta(alpha, beta)
        let gx = GammaDist::new(self.alpha, S::ONE);
        let gy = GammaDist::new(self.beta, S::ONE);
        let x = gx.sample(rng);
        let y = gy.sample(rng);
        x / (x + y)
    }
}

/// Log of the Beta function: ln(B(a, b)) = ln(Gamma(a)) + ln(Gamma(b)) - ln(Gamma(a+b))
fn lbeta<S: Scalar>(a: S, b: S) -> S {
    lgamma(a) + lgamma(b) - lgamma(a + b)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_beta_uniform_case() {
        // Beta(1, 1) = Uniform(0, 1)
        let b = BetaDist::new(1.0_f64, 1.0);
        assert!((b.pdf(0.5) - 1.0).abs() < 1e-12);
        assert!((b.cdf(0.5) - 0.5).abs() < 1e-8);
    }

    #[test]
    fn test_beta_symmetric() {
        let b = BetaDist::new(2.0_f64, 2.0);
        assert!((b.mean() - 0.5).abs() < 1e-14);
        // PDF should be symmetric about 0.5
        assert!((b.pdf(0.3) - b.pdf(0.7)).abs() < 1e-12);
    }

    #[test]
    fn test_beta_quantile_roundtrip() {
        let b = BetaDist::new(2.0_f64, 5.0);
        for &p in &[0.1, 0.5, 0.9] {
            let x = b.quantile(p);
            let p2 = b.cdf(x);
            assert!((p - p2).abs() < 1e-6, "p={}, p2={}", p, p2);
        }
    }

    #[test]
    fn test_beta_mean_variance() {
        let b = BetaDist::new(2.0_f64, 3.0);
        assert!((b.mean() - 0.4).abs() < 1e-14);
        // Var = 2*3 / (5*5*6) = 6/150 = 0.04
        assert!((b.variance() - 0.04).abs() < 1e-14);
    }
}

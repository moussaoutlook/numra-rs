//! Gamma distribution.
//!
//! Author: Moussa Leblouba
//! Date: 9 February 2026
//! Modified: 2 May 2026

use numra_core::Scalar;
use numra_special::{gammainc, lgamma};
use rand::RngCore;

use super::normal::random_uniform_01;
use super::ContinuousDistribution;

/// Gamma distribution with shape alpha and rate beta.
///
/// PDF: f(x) = beta^alpha * x^(alpha-1) * exp(-beta*x) / Gamma(alpha)
#[derive(Clone, Debug)]
pub struct GammaDist<S: Scalar> {
    pub shape: S,
    pub rate: S,
}

impl<S: Scalar> GammaDist<S> {
    pub fn new(shape: S, rate: S) -> Self {
        Self { shape, rate }
    }
}

impl<S: Scalar> ContinuousDistribution<S> for GammaDist<S> {
    fn pdf(&self, x: S) -> S {
        if x < S::ZERO {
            return S::ZERO;
        }
        if x == S::ZERO {
            return if self.shape == S::ONE {
                self.rate
            } else if self.shape > S::ONE {
                S::ZERO
            } else {
                S::INFINITY
            };
        }
        let log_pdf = self.shape * self.rate.ln() + (self.shape - S::ONE) * x.ln()
            - self.rate * x
            - lgamma(self.shape);
        log_pdf.exp()
    }

    fn cdf(&self, x: S) -> S {
        if x <= S::ZERO {
            return S::ZERO;
        }
        // CDF = P(a, b*x) = regularized lower incomplete gamma
        gammainc(self.shape, self.rate * x)
    }

    fn quantile(&self, p: S) -> S {
        if p <= S::ZERO {
            return S::ZERO;
        }
        if p >= S::ONE {
            return S::INFINITY;
        }
        // Newton iteration on CDF
        // Initial guess using Wilson-Hilferty approximation
        let mu = self.shape / self.rate;
        let sig = (self.shape / (self.rate * self.rate)).sqrt();
        let mut x = mu + sig * normal_quantile_approx(p);
        if x <= S::ZERO {
            x = mu * S::from_f64(0.01);
        }
        for _ in 0..50 {
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
        self.shape / self.rate
    }

    fn variance(&self) -> S {
        self.shape / (self.rate * self.rate)
    }

    fn sample(&self, rng: &mut dyn RngCore) -> S {
        // Marsaglia and Tsang's method for shape >= 1
        let one = S::ONE;
        let shape = if self.shape < one {
            self.shape + one
        } else {
            self.shape
        };

        let d = shape - S::from_f64(1.0 / 3.0);
        let c = S::ONE / (S::from_f64(9.0) * d).sqrt();

        loop {
            let x = sample_standard_normal::<S>(rng);
            let v = S::ONE + c * x;
            if v <= S::ZERO {
                continue;
            }
            let v = v * v * v;
            let u = random_uniform_01::<S>(rng);
            let x2 = x * x;
            if u < S::ONE - S::from_f64(0.0331) * x2 * x2 {
                let result = d * v / self.rate;
                if self.shape < one {
                    let u2 = random_uniform_01::<S>(rng);
                    return result * u2.ln().exp() / self.shape.ln().exp()
                        * (S::ONE / self.shape).ln().exp();
                }
                return result;
            }
            if u.ln() < S::HALF * x2 + d * (S::ONE - v + v.ln()) {
                let result = d * v / self.rate;
                if self.shape < one {
                    let u2 = random_uniform_01::<S>(rng);
                    return result * u2.powf(S::ONE / self.shape);
                }
                return result;
            }
        }
    }
}

/// Quick normal quantile approximation for initial guesses.
pub(crate) fn normal_quantile_approx<S: Scalar>(p: S) -> S {
    let p_f64 = p.to_f64();
    // Rational approximation from Abramowitz & Stegun 26.2.23
    let t = if p_f64 < 0.5 {
        (-2.0 * p_f64.ln()).sqrt()
    } else {
        (-2.0 * (1.0 - p_f64).ln()).sqrt()
    };
    let c0 = 2.515517;
    let c1 = 0.802853;
    let c2 = 0.010328;
    let d1 = 1.432788;
    let d2 = 0.189269;
    let d3 = 0.001308;
    let val = t - (c0 + c1 * t + c2 * t * t) / (1.0 + d1 * t + d2 * t * t + d3 * t * t * t);
    if p_f64 < 0.5 {
        S::from_f64(-val)
    } else {
        S::from_f64(val)
    }
}

/// Sample from standard normal using Box-Muller.
pub(crate) fn sample_standard_normal<S: Scalar>(rng: &mut dyn RngCore) -> S {
    let u1 = random_uniform_01::<S>(rng);
    let u2 = random_uniform_01::<S>(rng);
    let two = S::TWO;
    let pi2 = S::from_f64(core::f64::consts::TAU);
    (S::ZERO - two * u1.ln()).sqrt() * (pi2 * u2).cos()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gamma_pdf_exponential_case() {
        // Gamma(1, lambda) = Exponential(lambda)
        let g = GammaDist::new(1.0_f64, 2.0);
        assert!((g.pdf(0.0) - 2.0).abs() < 1e-12);
        assert!((g.pdf(1.0) - 2.0 * (-2.0_f64).exp()).abs() < 1e-12);
    }

    #[test]
    fn test_gamma_cdf() {
        let g = GammaDist::new(1.0_f64, 1.0);
        // CDF of Exp(1) at x=1 is 1-e^-1
        assert!((g.cdf(1.0) - (1.0 - (-1.0_f64).exp())).abs() < 1e-8);
    }

    #[test]
    fn test_gamma_quantile_roundtrip() {
        let g = GammaDist::new(3.0_f64, 2.0);
        for &p in &[0.1, 0.5, 0.9] {
            let x = g.quantile(p);
            let p2 = g.cdf(x);
            assert!((p - p2).abs() < 1e-6, "p={}, p2={}", p, p2);
        }
    }

    #[test]
    fn test_gamma_mean_variance() {
        let g = GammaDist::new(5.0_f64, 2.0);
        assert!((g.mean() - 2.5).abs() < 1e-14);
        assert!((g.variance() - 1.25).abs() < 1e-14);
    }

    #[test]
    fn test_gamma_sample_mean() {
        use rand::SeedableRng;
        let g = GammaDist::new(3.0_f64, 1.0);
        let mut rng = rand::rngs::StdRng::seed_from_u64(45);
        let samples = g.sample_n(&mut rng, 10000);
        let mean: f64 = samples.iter().sum::<f64>() / samples.len() as f64;
        assert!((mean - 3.0).abs() < 0.2, "sample mean = {}", mean);
    }
}

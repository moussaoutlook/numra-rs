//! Poisson distribution.
//!
//! Author: Moussa Leblouba
//! Date: 9 February 2026
//! Modified: 2 May 2026

use numra_core::Scalar;
use numra_special::gammaincc;
use rand::RngCore;

use super::normal::random_uniform_01;
use super::DiscreteDistribution;

/// Poisson distribution with rate parameter lambda.
#[derive(Clone, Debug)]
pub struct Poisson<S: Scalar> {
    pub lambda: S,
}

impl<S: Scalar> Poisson<S> {
    pub fn new(lambda: S) -> Self {
        Self { lambda }
    }
}

impl<S: Scalar> DiscreteDistribution<S> for Poisson<S> {
    fn pmf(&self, k: usize) -> S {
        // P(X=k) = lambda^k * exp(-lambda) / k!
        let k_s = S::from_usize(k);
        let log_pmf = k_s * self.lambda.ln() - self.lambda - log_factorial::<S>(k);
        log_pmf.exp()
    }

    fn cdf(&self, k: usize) -> S {
        // P(X <= k) = 1 - P(k+1, lambda) = gammaincc(k+1, lambda) [regularized upper incomplete gamma]
        // Actually, CDF = Q(k+1, lambda) = gammaincc(k+1, lambda)
        let k1 = S::from_usize(k + 1);
        gammaincc(k1, self.lambda)
    }

    fn mean(&self) -> S {
        self.lambda
    }

    fn variance(&self) -> S {
        self.lambda
    }

    fn sample(&self, rng: &mut dyn RngCore) -> usize {
        // Knuth's algorithm for small lambda
        let lam = self.lambda.to_f64();
        if lam < 30.0 {
            let l = (-lam).exp();
            let mut k = 0usize;
            let mut p = 1.0;
            loop {
                k += 1;
                let u = random_uniform_01::<f64>(rng);
                p *= u;
                if p <= l {
                    return k - 1;
                }
            }
        } else {
            // For large lambda, use normal approximation + rejection
            let mu = lam;
            let sigma = lam.sqrt();
            loop {
                let u1 = random_uniform_01::<f64>(rng);
                let u2 = random_uniform_01::<f64>(rng);
                let z = (-2.0 * u1.ln()).sqrt() * (core::f64::consts::TAU * u2).cos();
                let x = mu + sigma * z;
                if x >= 0.0 {
                    return x.round() as usize;
                }
            }
        }
    }
}

/// Compute ln(k!) using lgamma(k+1).
fn log_factorial<S: Scalar>(k: usize) -> S {
    if k <= 1 {
        return S::ZERO;
    }
    S::from_usize(k + 1).ln_gamma()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_poisson_pmf_sum() {
        let p = Poisson::new(3.0_f64);
        let sum: f64 = (0..30).map(|k| p.pmf(k)).sum();
        assert!((sum - 1.0).abs() < 1e-8, "sum = {}", sum);
    }

    #[test]
    fn test_poisson_cdf_monotone() {
        let p = Poisson::new(5.0_f64);
        let mut prev = 0.0_f64;
        for k in 0..20 {
            let c = p.cdf(k);
            assert!(c >= prev, "CDF not monotone at k={}", k);
            prev = c;
        }
        assert!((prev - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_poisson_mean_variance() {
        let p = Poisson::new(7.0_f64);
        assert!((p.mean() - 7.0).abs() < 1e-14);
        assert!((p.variance() - 7.0).abs() < 1e-14);
    }

    #[test]
    fn test_poisson_sample_mean() {
        let p = Poisson::new(5.0_f64);
        let mut rng = rand::thread_rng();
        let samples = p.sample_n(&mut rng, 10000);
        let mean = samples.iter().sum::<usize>() as f64 / samples.len() as f64;
        assert!((mean - 5.0).abs() < 0.3, "sample mean = {}", mean);
    }
}

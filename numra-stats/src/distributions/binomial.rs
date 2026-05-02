//! Binomial distribution.
//!
//! Author: Moussa Leblouba
//! Date: 9 February 2026
//! Modified: 2 May 2026

use numra_special::betainc;
use rand::RngCore;

use super::normal::random_uniform_01;
use super::DiscreteDistribution;

/// Binomial distribution B(n, p).
#[derive(Clone, Debug)]
pub struct Binomial {
    pub n: usize,
    pub p: f64,
}

impl Binomial {
    pub fn new(n: usize, p: f64) -> Self {
        Self { n, p }
    }
}

impl DiscreteDistribution<f64> for Binomial {
    fn pmf(&self, k: usize) -> f64 {
        if k > self.n {
            return 0.0;
        }
        let log_pmf = log_binom_coeff(self.n, k)
            + k as f64 * self.p.ln()
            + (self.n - k) as f64 * (1.0 - self.p).ln();
        log_pmf.exp()
    }

    fn cdf(&self, k: usize) -> f64 {
        if k >= self.n {
            return 1.0;
        }
        // CDF = I_{1-p}(n-k, k+1) using regularized incomplete beta
        betainc((self.n - k) as f64, (k + 1) as f64, 1.0 - self.p)
    }

    fn mean(&self) -> f64 {
        self.n as f64 * self.p
    }

    fn variance(&self) -> f64 {
        self.n as f64 * self.p * (1.0 - self.p)
    }

    fn sample(&self, rng: &mut dyn RngCore) -> usize {
        // Direct simulation for small n, normal approximation for large n
        if self.n < 50 {
            let mut count = 0;
            for _ in 0..self.n {
                let u = random_uniform_01::<f64>(rng);
                if u < self.p {
                    count += 1;
                }
            }
            count
        } else {
            // Normal approximation
            let mu = self.n as f64 * self.p;
            let sigma = (mu * (1.0 - self.p)).sqrt();
            let u1 = random_uniform_01::<f64>(rng);
            let u2 = random_uniform_01::<f64>(rng);
            let z = (-2.0 * u1.ln()).sqrt() * (core::f64::consts::TAU * u2).cos();
            let x = (mu + sigma * z).round().max(0.0).min(self.n as f64);
            x as usize
        }
    }
}

/// Compute ln(C(n, k)) = ln(n!) - ln(k!) - ln((n-k)!)
fn log_binom_coeff(n: usize, k: usize) -> f64 {
    let lf = |m: usize| -> f64 {
        if m <= 1 {
            0.0
        } else {
            numra_special::lgamma((m + 1) as f64)
        }
    };
    lf(n) - lf(k) - lf(n - k)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_binomial_pmf_sum() {
        let b = Binomial::new(10, 0.3);
        let sum: f64 = (0..=10).map(|k| b.pmf(k)).sum();
        assert!((sum - 1.0).abs() < 1e-10, "sum = {}", sum);
    }

    #[test]
    fn test_binomial_cdf_monotone() {
        let b = Binomial::new(10, 0.5);
        let mut prev = 0.0;
        for k in 0..=10 {
            let c = b.cdf(k);
            assert!(c >= prev - 1e-14, "CDF not monotone at k={}", k);
            prev = c;
        }
        assert!((prev - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_binomial_mean_variance() {
        let b = Binomial::new(20, 0.4);
        assert!((b.mean() - 8.0).abs() < 1e-14);
        assert!((b.variance() - 4.8).abs() < 1e-14);
    }

    #[test]
    fn test_binomial_coin_flip() {
        // B(1, 0.5) is a fair coin
        let b = Binomial::new(1, 0.5);
        assert!((b.pmf(0) - 0.5).abs() < 1e-14);
        assert!((b.pmf(1) - 0.5).abs() < 1e-14);
    }

    #[test]
    fn test_binomial_sample_mean() {
        let b = Binomial::new(20, 0.5);
        let mut rng = rand::thread_rng();
        let samples = b.sample_n(&mut rng, 5000);
        let mean = samples.iter().sum::<usize>() as f64 / samples.len() as f64;
        assert!((mean - 10.0).abs() < 1.0, "sample mean = {}", mean);
    }
}

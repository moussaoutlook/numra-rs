//! Probability distribution traits and implementations.
//!
//! Author: Moussa Leblouba
//! Date: 9 February 2026
//! Modified: 2 May 2026

use numra_core::Scalar;

pub mod beta_dist;
pub mod binomial;
pub mod chi_squared;
pub mod exponential;
pub mod f_dist;
pub mod gamma_dist;
pub mod lognormal;
pub mod normal;
pub mod poisson;
pub mod student_t;
pub mod uniform;

/// Trait for continuous probability distributions.
pub trait ContinuousDistribution<S: Scalar> {
    /// Probability density function.
    fn pdf(&self, x: S) -> S;

    /// Cumulative distribution function.
    fn cdf(&self, x: S) -> S;

    /// Inverse CDF (quantile function).
    fn quantile(&self, p: S) -> S;

    /// Distribution mean.
    fn mean(&self) -> S;

    /// Distribution variance.
    fn variance(&self) -> S;

    /// Standard deviation (default: sqrt(variance)).
    fn std_dev(&self) -> S {
        self.variance().sqrt()
    }

    /// Generate a single sample.
    fn sample(&self, rng: &mut dyn RngCore) -> S;

    /// Generate n samples.
    fn sample_n(&self, rng: &mut dyn RngCore, n: usize) -> Vec<S> {
        (0..n).map(|_| self.sample(rng)).collect()
    }
}

/// Trait for discrete probability distributions.
pub trait DiscreteDistribution<S: Scalar> {
    /// Probability mass function.
    fn pmf(&self, k: usize) -> S;

    /// Cumulative distribution function.
    fn cdf(&self, k: usize) -> S;

    /// Distribution mean.
    fn mean(&self) -> S;

    /// Distribution variance.
    fn variance(&self) -> S;

    /// Generate a single sample.
    fn sample(&self, rng: &mut dyn RngCore) -> usize;

    /// Generate n samples.
    fn sample_n(&self, rng: &mut dyn RngCore, n: usize) -> Vec<usize> {
        (0..n).map(|_| self.sample(rng)).collect()
    }
}

// Re-export rand trait for use by distributions
pub use rand::RngCore;

pub use beta_dist::BetaDist;
pub use binomial::Binomial;
pub use chi_squared::ChiSquared;
pub use exponential::Exponential;
pub use f_dist::FDist;
pub use gamma_dist::GammaDist;
pub use lognormal::LogNormal;
pub use normal::Normal;
pub use poisson::Poisson;
pub use student_t::StudentT;
pub use uniform::Uniform;

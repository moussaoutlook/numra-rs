//! Wiener process (Brownian motion) generator.
//!
//! Author: Moussa Leblouba
//! Date: 3 February 2026
//! Modified: 2 May 2026

use numra_core::Scalar;
use rand::prelude::*;
use rand_distr::StandardNormal;

/// Wiener process increment ΔW ~ N(0, Δt).
#[derive(Clone, Debug)]
pub struct WienerIncrement<S: Scalar> {
    /// The increment values (one per Wiener process)
    pub dw: Vec<S>,
    /// The time step
    pub dt: S,
}

impl<S: Scalar> WienerIncrement<S> {
    /// Create a new increment with given values.
    pub fn new(dw: Vec<S>, dt: S) -> Self {
        Self { dw, dt }
    }

    /// Create zero increment.
    pub fn zeros(n_wiener: usize, dt: S) -> Self {
        Self {
            dw: vec![S::ZERO; n_wiener],
            dt,
        }
    }
}

/// Wiener process generator for SDE simulation.
///
/// Generates increments ΔW(t) ~ N(0, Δt) for standard Brownian motion.
pub struct WienerProcess<R: Rng> {
    rng: R,
    n_wiener: usize,
}

impl<R: Rng> WienerProcess<R> {
    /// Create a new Wiener process with the given RNG.
    pub fn new(rng: R, n_wiener: usize) -> Self {
        Self { rng, n_wiener }
    }

    /// Generate a single increment ΔW ~ N(0, dt).
    pub fn increment<S: Scalar>(&mut self, dt: S) -> WienerIncrement<S> {
        let sqrt_dt = dt.sqrt();
        let dw: Vec<S> = (0..self.n_wiener)
            .map(|_| {
                let z: f64 = self.rng.sample(StandardNormal);
                S::from_f64(z) * sqrt_dt
            })
            .collect();
        WienerIncrement::new(dw, dt)
    }

    /// Generate multiple increments for a path from t0 to tf with step dt.
    pub fn path<S: Scalar>(&mut self, t0: S, tf: S, dt: S) -> Vec<WienerIncrement<S>> {
        let mut increments = Vec::new();
        let mut t = t0;
        while t < tf {
            let step = dt.min(tf - t);
            increments.push(self.increment(step));
            t += step;
        }
        increments
    }

    /// Generate correlated increments given a correlation matrix.
    ///
    /// The correlation matrix L should be the Cholesky factor of the
    /// correlation matrix: Corr = L * L^T.
    pub fn correlated_increment<S: Scalar>(
        &mut self,
        dt: S,
        cholesky: &[S], // n_wiener x n_wiener, row-major
    ) -> WienerIncrement<S> {
        let sqrt_dt = dt.sqrt();
        let n = self.n_wiener;

        // Generate independent standard normals
        let z: Vec<f64> = (0..n).map(|_| self.rng.sample(StandardNormal)).collect();

        // Apply Cholesky factor: dW = L * Z * sqrt(dt)
        let mut dw = vec![S::ZERO; n];
        for i in 0..n {
            for j in 0..=i {
                dw[i] += cholesky[i * n + j] * S::from_f64(z[j]);
            }
            dw[i] *= sqrt_dt;
        }

        WienerIncrement::new(dw, dt)
    }
}

impl WienerProcess<StdRng> {
    /// Create a Wiener process with a seed for reproducibility.
    pub fn with_seed(seed: u64, n_wiener: usize) -> Self {
        Self::new(StdRng::seed_from_u64(seed), n_wiener)
    }

    /// Create a Wiener process with random seed from system entropy.
    pub fn from_entropy(n_wiener: usize) -> Self {
        Self::new(StdRng::from_entropy(), n_wiener)
    }
}

/// Create a Wiener process with optional seed.
pub fn create_wiener(n_wiener: usize, seed: Option<u64>) -> WienerProcess<StdRng> {
    match seed {
        Some(s) => WienerProcess::with_seed(s, n_wiener),
        None => WienerProcess::from_entropy(n_wiener),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wiener_increment() {
        let mut wp = WienerProcess::with_seed(42, 2);
        let inc = wp.increment::<f64>(0.01);

        assert_eq!(inc.dw.len(), 2);
        assert!((inc.dt - 0.01).abs() < 1e-10);
        // Increments should be non-zero (with high probability)
        assert!(inc.dw[0].abs() > 0.0 || inc.dw[1].abs() > 0.0);
    }

    #[test]
    fn test_wiener_path() {
        let mut wp = WienerProcess::with_seed(42, 1);
        let path = wp.path::<f64>(0.0, 1.0, 0.1);

        // Should have approximately 10 increments (may vary by ±1 due to floating point)
        assert!(path.len() >= 10 && path.len() <= 11);

        // Most increments should be close to dt
        let full_steps = path
            .iter()
            .filter(|inc| (inc.dt - 0.1).abs() < 0.01)
            .count();
        assert!(full_steps >= 9);
    }

    #[test]
    fn test_wiener_variance() {
        // Check that variance of increments ≈ dt
        let mut wp = WienerProcess::with_seed(123, 1);
        let n = 10000;
        let dt = 0.01_f64;

        let increments: Vec<f64> = (0..n).map(|_| wp.increment::<f64>(dt).dw[0]).collect();

        let mean: f64 = increments.iter().sum::<f64>() / n as f64;
        let variance: f64 = increments.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n as f64;

        // Mean should be close to 0
        assert!(mean.abs() < 0.01);
        // Variance should be close to dt
        assert!((variance - dt).abs() < 0.001);
    }

    #[test]
    fn test_correlated_wiener() {
        // 2x2 correlation with ρ = 0.5
        // L = [[1, 0], [0.5, sqrt(0.75)]]
        let mut wp = WienerProcess::with_seed(42, 2);
        let cholesky = [1.0, 0.0, 0.5, 0.75_f64.sqrt()];
        let inc = wp.correlated_increment::<f64>(0.01, &cholesky);

        assert_eq!(inc.dw.len(), 2);
    }
}

//! Ensemble runner for Monte Carlo SDE simulations.
//!
//! Provides parallel execution of multiple SDE trajectories using rayon.
//!
//! Author: Moussa Leblouba
//! Date: 4 February 2026
//! Modified: 2 May 2026

use crate::system::{SdeOptions, SdeResult, SdeSolver, SdeSystem};
use numra_core::Scalar;
use rayon::prelude::*;

/// Result of ensemble simulation.
#[derive(Clone, Debug)]
pub struct EnsembleResult<S: Scalar> {
    /// Individual trajectory results
    pub trajectories: Vec<SdeResult<S>>,
    /// Number of successful simulations
    pub n_success: usize,
    /// Number of failed simulations
    pub n_failed: usize,
    /// Seeds used for each trajectory (for reproducibility)
    pub seeds: Vec<u64>,
}

impl<S: Scalar> EnsembleResult<S> {
    /// Create a new ensemble result.
    pub fn new(trajectories: Vec<SdeResult<S>>, seeds: Vec<u64>) -> Self {
        let n_success = trajectories.iter().filter(|r| r.success).count();
        let n_failed = trajectories.len() - n_success;
        Self {
            trajectories,
            n_success,
            n_failed,
            seeds,
        }
    }

    /// Get all final values for a specific component.
    ///
    /// Returns None for failed trajectories.
    pub fn final_values(&self, component: usize) -> Vec<Option<S>> {
        self.trajectories
            .iter()
            .map(|r| {
                r.y_final().map(|y| {
                    if component < y.len() {
                        y[component]
                    } else {
                        S::NAN
                    }
                })
            })
            .collect()
    }

    /// Get all successful final values for a component.
    pub fn successful_final_values(&self, component: usize) -> Vec<S> {
        self.trajectories
            .iter()
            .filter_map(|r| {
                if r.success {
                    r.y_final().map(|y| y[component])
                } else {
                    None
                }
            })
            .collect()
    }

    /// Iterate over successful trajectories.
    pub fn successful(&self) -> impl Iterator<Item = &SdeResult<S>> {
        self.trajectories.iter().filter(|r| r.success)
    }

    /// Get trajectory at index.
    pub fn get(&self, index: usize) -> Option<&SdeResult<S>> {
        self.trajectories.get(index)
    }

    /// Number of trajectories.
    pub fn len(&self) -> usize {
        self.trajectories.len()
    }

    /// Is the ensemble empty?
    pub fn is_empty(&self) -> bool {
        self.trajectories.is_empty()
    }
}

/// Ensemble runner for parallel Monte Carlo simulations.
pub struct EnsembleRunner;

impl EnsembleRunner {
    /// Run an ensemble of SDE simulations in parallel.
    ///
    /// # Arguments
    /// * `system` - The SDE system to simulate
    /// * `t0` - Initial time
    /// * `tf` - Final time
    /// * `x0` - Initial state
    /// * `options` - Solver options (seed is used as base seed)
    /// * `n_trajectories` - Number of Monte Carlo paths
    ///
    /// # Returns
    /// An `EnsembleResult` containing all trajectory results.
    pub fn run<S, Sys, Solver>(
        system: &Sys,
        t0: S,
        tf: S,
        x0: &[S],
        options: &SdeOptions<S>,
        n_trajectories: usize,
    ) -> EnsembleResult<S>
    where
        S: Scalar + Send + Sync,
        Sys: SdeSystem<S> + Sync,
        Solver: SdeSolver<S>,
    {
        // Generate seeds for each trajectory
        let base_seed = options.seed.unwrap_or(0);
        let seeds: Vec<u64> = (0..n_trajectories)
            .map(|i| base_seed.wrapping_add(i as u64))
            .collect();

        // Run simulations in parallel
        let results: Vec<SdeResult<S>> = seeds
            .par_iter()
            .map(|&seed| {
                Solver::solve(system, t0, tf, x0, options, Some(seed))
                    .unwrap_or_else(|msg| SdeResult::failed(msg, Default::default()))
            })
            .collect();

        EnsembleResult::new(results, seeds)
    }

    /// Run ensemble with custom seeds.
    ///
    /// Useful for resuming simulations or specific reproducibility needs.
    pub fn run_with_seeds<S, Sys, Solver>(
        system: &Sys,
        t0: S,
        tf: S,
        x0: &[S],
        options: &SdeOptions<S>,
        seeds: &[u64],
    ) -> EnsembleResult<S>
    where
        S: Scalar + Send + Sync,
        Sys: SdeSystem<S> + Sync,
        Solver: SdeSolver<S>,
    {
        let results: Vec<SdeResult<S>> = seeds
            .par_iter()
            .map(|&seed| {
                Solver::solve(system, t0, tf, x0, options, Some(seed))
                    .unwrap_or_else(|msg| SdeResult::failed(msg, Default::default()))
            })
            .collect();

        EnsembleResult::new(results, seeds.to_vec())
    }

    /// Run ensemble sequentially (for debugging or when parallelism isn't needed).
    pub fn run_sequential<S, Sys, Solver>(
        system: &Sys,
        t0: S,
        tf: S,
        x0: &[S],
        options: &SdeOptions<S>,
        n_trajectories: usize,
    ) -> EnsembleResult<S>
    where
        S: Scalar,
        Sys: SdeSystem<S>,
        Solver: SdeSolver<S>,
    {
        let base_seed = options.seed.unwrap_or(0);
        let seeds: Vec<u64> = (0..n_trajectories)
            .map(|i| base_seed.wrapping_add(i as u64))
            .collect();

        let results: Vec<SdeResult<S>> = seeds
            .iter()
            .map(|&seed| {
                Solver::solve(system, t0, tf, x0, options, Some(seed))
                    .unwrap_or_else(|msg| SdeResult::failed(msg, Default::default()))
            })
            .collect();

        EnsembleResult::new(results, seeds)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{EulerMaruyama, SdeSystem};

    #[allow(clippy::upper_case_acronyms)]
    struct GBM {
        mu: f64,
        sigma: f64,
    }

    impl SdeSystem<f64> for GBM {
        fn dim(&self) -> usize {
            1
        }
        fn drift(&self, _t: f64, x: &[f64], f: &mut [f64]) {
            f[0] = self.mu * x[0];
        }
        fn diffusion(&self, _t: f64, x: &[f64], g: &mut [f64]) {
            g[0] = self.sigma * x[0];
        }
    }

    #[test]
    fn test_ensemble_parallel() {
        let gbm = GBM {
            mu: 0.05,
            sigma: 0.2,
        };
        let options = SdeOptions::default().dt(0.01).seed(42);

        let result =
            EnsembleRunner::run::<_, _, EulerMaruyama>(&gbm, 0.0, 1.0, &[100.0], &options, 100);

        assert_eq!(result.len(), 100);
        assert_eq!(result.n_success, 100);
        assert_eq!(result.n_failed, 0);

        // All final prices should be positive
        let finals = result.successful_final_values(0);
        assert_eq!(finals.len(), 100);
        for &price in &finals {
            assert!(price > 0.0);
        }
    }

    #[test]
    fn test_ensemble_sequential() {
        let gbm = GBM {
            mu: 0.05,
            sigma: 0.2,
        };
        let options = SdeOptions::default().dt(0.01).seed(42);

        let result = EnsembleRunner::run_sequential::<_, _, EulerMaruyama>(
            &gbm,
            0.0,
            1.0,
            &[100.0],
            &options,
            10,
        );

        assert_eq!(result.len(), 10);
        assert_eq!(result.n_success, 10);
    }

    #[test]
    fn test_ensemble_reproducibility() {
        let gbm = GBM {
            mu: 0.05,
            sigma: 0.2,
        };
        let options = SdeOptions::default().dt(0.01).seed(12345);

        // Run twice with same base seed
        let r1 = EnsembleRunner::run_sequential::<_, _, EulerMaruyama>(
            &gbm,
            0.0,
            1.0,
            &[100.0],
            &options,
            5,
        );
        let r2 = EnsembleRunner::run_sequential::<_, _, EulerMaruyama>(
            &gbm,
            0.0,
            1.0,
            &[100.0],
            &options,
            5,
        );

        // Results should be identical
        for i in 0..5 {
            let y1 = r1.get(i).unwrap().y_final().unwrap()[0];
            let y2 = r2.get(i).unwrap().y_final().unwrap()[0];
            assert!((y1 - y2).abs() < 1e-10);
        }
    }

    #[test]
    fn test_ensemble_statistics_sample() {
        let gbm = GBM {
            mu: 0.05,
            sigma: 0.2,
        };
        let options = SdeOptions::default().dt(0.001).seed(0);

        // Large ensemble for statistical test
        let result =
            EnsembleRunner::run::<_, _, EulerMaruyama>(&gbm, 0.0, 1.0, &[100.0], &options, 1000);

        let finals = result.successful_final_values(0);

        // Compute sample mean and variance
        let mean: f64 = finals.iter().sum::<f64>() / finals.len() as f64;
        let variance: f64 =
            finals.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / finals.len() as f64;

        // For GBM: E[S_T] = S_0 * exp(μT)
        // Var[S_T] = S_0² * exp(2μT) * (exp(σ²T) - 1)
        let s0 = 100.0;
        let expected_mean = s0 * (0.05 * 1.0_f64).exp(); // ~105.13
        let expected_var =
            s0 * s0 * (2.0 * 0.05 * 1.0_f64).exp() * ((0.2 * 0.2 * 1.0_f64).exp() - 1.0); // ~454.9

        // Check within 3 standard errors
        let se_mean = (variance / finals.len() as f64).sqrt();
        assert!((mean - expected_mean).abs() < 3.0 * se_mean);

        // Variance estimate should be in the right ballpark
        assert!((variance - expected_var).abs() < expected_var * 0.2); // Within 20%
    }
}

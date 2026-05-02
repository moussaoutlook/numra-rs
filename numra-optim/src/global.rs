//! Global optimization via Differential Evolution (DE/rand/1/bin).
//!
//! Author: Moussa Leblouba
//! Date: 8 February 2026
//! Modified: 2 May 2026

use crate::error::OptimError;
use crate::types::{IterationRecord, OptimResult, OptimStatus};
use numra_core::Scalar;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

/// Options for Differential Evolution.
#[derive(Clone, Debug)]
pub struct DEOptions<S: Scalar> {
    /// Population size (default: None = 10 * dimension).
    pub pop_size: Option<usize>,
    /// Maximum generations.
    pub max_generations: usize,
    /// Differential weight F in [0, 2], typically 0.5-1.0.
    pub differential_weight: S,
    /// Crossover rate CR in [0, 1], typically 0.7-1.0.
    pub crossover_rate: S,
    /// Convergence tolerance on objective spread in population.
    pub tol: S,
    /// Random seed for reproducibility.
    pub seed: u64,
    /// Print progress.
    pub verbose: bool,
}

impl<S: Scalar> Default for DEOptions<S> {
    fn default() -> Self {
        Self {
            pop_size: None,
            max_generations: 1000,
            differential_weight: S::from_f64(0.8),
            crossover_rate: S::from_f64(0.9),
            tol: S::from_f64(1e-12),
            seed: 42,
            verbose: false,
        }
    }
}

/// Minimize `f` over the box defined by `bounds` using Differential Evolution
/// (DE/rand/1/bin strategy).
///
/// This is a derivative-free global optimizer suitable for non-convex, noisy, or
/// multimodal objective functions.
///
/// # Arguments
///
/// * `f` - Objective function mapping `&[S]` to `S`.
/// * `bounds` - Slice of `(lower, upper)` pairs, one per dimension.
/// * `opts` - Algorithm options (see [`DEOptions`]).
///
/// # Returns
///
/// An [`OptimResult`] with the best solution found.
pub fn de_minimize<S: Scalar, F>(
    f: F,
    bounds: &[(S, S)],
    opts: &DEOptions<S>,
) -> Result<OptimResult<S>, OptimError>
where
    F: Fn(&[S]) -> S,
{
    let start = std::time::Instant::now();
    let n = bounds.len();
    let np = opts.pop_size.unwrap_or(10 * n);
    let cr = opts.crossover_rate;
    let f_weight = opts.differential_weight;

    let mut rng = SmallRng::seed_from_u64(opts.seed);

    // ---- Initialize population uniformly within bounds ----
    let mut pop: Vec<Vec<S>> = (0..np)
        .map(|_| {
            bounds
                .iter()
                .map(|&(lo, hi)| lo + S::from_f64(rng.gen::<f64>()) * (hi - lo))
                .collect()
        })
        .collect();

    // Evaluate initial fitness
    let mut fitness: Vec<S> = pop.iter().map(|xi| f(xi)).collect();
    let mut n_feval: usize = np;

    // Track best
    let mut best_idx: usize = 0;
    for i in 1..np {
        if fitness[i] < fitness[best_idx] {
            best_idx = i;
        }
    }
    let mut best_x = pop[best_idx].clone();
    let mut best_f = fitness[best_idx];

    let mut history: Vec<IterationRecord<S>> = Vec::new();
    let mut converged = false;
    let mut gen = 0usize;

    // ---- Main loop ----
    for g in 0..opts.max_generations {
        gen = g + 1;

        for i in 0..np {
            // Pick 3 distinct indices != i
            let (a, b, c) = pick_three(&mut rng, np, i);

            // Mutation: v = pop[a] + F * (pop[b] - pop[c])
            let mut trial = vec![S::ZERO; n];
            let j_rand = rng.gen_range(0..n);

            for j in 0..n {
                let v_j = pop[a][j] + f_weight * (pop[b][j] - pop[c][j]);
                // Clip to bounds
                let v_j = v_j.clamp(bounds[j].0, bounds[j].1);

                // Binomial crossover
                if S::from_f64(rng.gen::<f64>()) < cr || j == j_rand {
                    trial[j] = v_j;
                } else {
                    trial[j] = pop[i][j];
                }
            }

            // Selection
            let trial_f = f(&trial);
            n_feval += 1;

            if trial_f <= fitness[i] {
                pop[i] = trial;
                fitness[i] = trial_f;

                if trial_f < best_f {
                    best_f = trial_f;
                    best_x = pop[i].clone();
                }
            }
        }

        // Record iteration
        history.push(IterationRecord {
            iteration: gen,
            objective: best_f,
            gradient_norm: S::ZERO,
            step_size: S::ZERO,
            constraint_violation: S::ZERO,
        });

        if opts.verbose {
            eprintln!("DE gen {}: best_f = {:.6e}", gen, best_f.to_f64());
        }

        // Check convergence: fitness spread
        let f_max = fitness.iter().copied().fold(S::NEG_INFINITY, S::max);
        let f_min = fitness.iter().copied().fold(S::INFINITY, S::min);
        if (f_max - f_min) < opts.tol {
            converged = true;
            break;
        }
    }

    let (status, message) = if converged {
        (
            OptimStatus::GradientConverged,
            format!(
                "Converged: fitness spread < tol ({:.2e}) at generation {}",
                opts.tol.to_f64(),
                gen
            ),
        )
    } else {
        (
            OptimStatus::MaxIterations,
            format!(
                "Reached maximum generations ({}), best f = {:.6e}",
                opts.max_generations,
                best_f.to_f64()
            ),
        )
    };

    let result = OptimResult {
        x: best_x,
        f: best_f,
        grad: Vec::new(),
        iterations: gen,
        n_feval,
        n_geval: 0,
        converged,
        message,
        status,
        history,
        lambda_eq: Vec::new(),
        lambda_ineq: Vec::new(),
        active_bounds: Vec::new(),
        constraint_violation: S::ZERO,
        wall_time_secs: 0.0,
        pareto: None,
        sensitivity: None,
    }
    .with_wall_time(start);

    Ok(result)
}

/// Pick 3 distinct random indices from 0..n, all different from `exclude`.
fn pick_three(rng: &mut SmallRng, n: usize, exclude: usize) -> (usize, usize, usize) {
    let mut a = rng.gen_range(0..n);
    while a == exclude {
        a = rng.gen_range(0..n);
    }
    let mut b = rng.gen_range(0..n);
    while b == exclude || b == a {
        b = rng.gen_range(0..n);
    }
    let mut c = rng.gen_range(0..n);
    while c == exclude || c == a || c == b {
        c = rng.gen_range(0..n);
    }
    (a, b, c)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_de_sphere() {
        // minimize sum(x_i^2), n=3, bounds [-5, 5]
        let bounds = vec![(-5.0, 5.0); 3];
        let result = de_minimize(
            |x: &[f64]| x.iter().map(|xi| xi * xi).sum::<f64>(),
            &bounds,
            &DEOptions {
                max_generations: 500,
                ..Default::default()
            },
        )
        .unwrap();
        assert!(result.converged, "did not converge: {}", result.message);
        for &xi in &result.x {
            assert!(xi.abs() < 0.1, "xi={}", xi);
        }
        assert!(result.f < 0.01, "f={}", result.f);
    }

    #[test]
    fn test_de_rastrigin() {
        // Global min at (0, 0) with f=0, many local minima
        let bounds = vec![(-5.12, 5.12); 2];
        let result = de_minimize(
            |x: &[f64]| {
                let n = x.len() as f64;
                10.0 * n
                    + x.iter()
                        .map(|xi| xi * xi - 10.0 * (2.0 * std::f64::consts::PI * xi).cos())
                        .sum::<f64>()
            },
            &bounds,
            &DEOptions {
                max_generations: 1000,
                ..Default::default()
            },
        )
        .unwrap();
        assert!(result.f < 1.0, "f={}", result.f);
    }

    #[test]
    fn test_de_rosenbrock() {
        let bounds = vec![(-5.0, 10.0), (-5.0, 10.0)];
        let result = de_minimize(
            |x: &[f64]| (1.0 - x[0]).powi(2) + 100.0 * (x[1] - x[0] * x[0]).powi(2),
            &bounds,
            &DEOptions {
                max_generations: 2000,
                ..Default::default()
            },
        )
        .unwrap();
        assert!(result.f < 0.01, "f={}", result.f);
        assert!((result.x[0] - 1.0).abs() < 0.1, "x0={}", result.x[0]);
    }

    #[test]
    fn test_de_deterministic() {
        let bounds = vec![(-5.0, 5.0); 2];
        let f = |x: &[f64]| x[0] * x[0] + x[1] * x[1];
        let r1 = de_minimize(f, &bounds, &DEOptions::default()).unwrap();
        let r2 = de_minimize(f, &bounds, &DEOptions::default()).unwrap();
        assert_eq!(r1.x, r2.x);
        assert_eq!(r1.f, r2.f);
    }

    #[test]
    fn test_de_1d() {
        let bounds = vec![(0.0, 10.0)];
        let result = de_minimize(
            |x: &[f64]| (x[0] - 3.0).powi(2),
            &bounds,
            &DEOptions::default(),
        )
        .unwrap();
        assert!((result.x[0] - 3.0).abs() < 0.1, "x={}", result.x[0]);
    }
}

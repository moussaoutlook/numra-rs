//! CMA-ES (Covariance Matrix Adaptation Evolution Strategy).
//!
//! A state-of-the-art derivative-free global optimizer for nonlinear,
//! non-convex optimization in continuous domains.
//!
//! Author: Moussa Leblouba
//! Date: 9 February 2026
//! Modified: 2 May 2026

use numra_core::Scalar;
use numra_linalg::{DenseMatrix, Matrix};
use rand::rngs::SmallRng;
use rand::SeedableRng;

use crate::error::OptimError;
use crate::types::{IterationRecord, OptimResult, OptimStatus};

/// Options for CMA-ES.
#[derive(Clone, Debug)]
pub struct CmaEsOptions<S: Scalar> {
    /// Population size lambda. Default: `4 + floor(3 * ln(n))`.
    pub population_size: Option<usize>,
    /// Initial step size (sigma). Default: 0.5.
    pub sigma0: S,
    /// Maximum iterations (generations).
    pub max_iter: usize,
    /// Convergence tolerance on fitness spread.
    pub tol_f: S,
    /// Convergence tolerance on sigma.
    pub tol_sigma: S,
    /// Random seed.
    pub seed: u64,
    /// Print progress.
    pub verbose: bool,
}

impl<S: Scalar> Default for CmaEsOptions<S> {
    fn default() -> Self {
        Self {
            population_size: None,
            sigma0: S::HALF,
            max_iter: 10_000,
            tol_f: S::from_f64(1e-12),
            tol_sigma: S::from_f64(1e-12),
            seed: 42,
            verbose: false,
        }
    }
}

#[allow(clippy::needless_range_loop)]
/// Minimize `f` using CMA-ES starting from `x0`.
///
/// CMA-ES samples a population from a multivariate normal distribution
/// N(mean, sigma^2 * C), ranks by fitness, and updates the mean,
/// step size sigma, and covariance matrix C.
///
/// # Arguments
///
/// * `f` - Objective function.
/// * `x0` - Initial mean.
/// * `opts` - Algorithm options.
pub fn cmaes_minimize<S, F>(
    f: F,
    x0: &[S],
    opts: &CmaEsOptions<S>,
) -> Result<OptimResult<S>, OptimError>
where
    S: Scalar + faer::SimpleEntity + faer::Conjugate<Canonical = S> + faer::ComplexField,
    F: Fn(&[S]) -> S,
{
    let start = std::time::Instant::now();
    let n = x0.len();
    if n == 0 {
        return Err(OptimError::DimensionMismatch {
            expected: 1,
            actual: 0,
        });
    }
    let nf = n as f64;

    // Population size
    let lambda = opts
        .population_size
        .unwrap_or((4.0 + (3.0 * nf.ln()).floor()) as usize);
    let lambda = lambda.max(4); // at least 4
    let mu = lambda / 2; // number of selected (parent) individuals

    // Recombination weights: w_i = ln(mu + 0.5) - ln(i)  for i = 1..mu
    let mut weights = Vec::with_capacity(mu);
    let log_mu_half = (mu as f64 + 0.5).ln();
    for i in 1..=mu {
        weights.push(log_mu_half - (i as f64).ln());
    }
    let w_sum: f64 = weights.iter().sum();
    for w in weights.iter_mut() {
        *w /= w_sum;
    }
    let w_sq_sum: f64 = weights.iter().map(|w| w * w).sum();
    let mu_eff = 1.0 / w_sq_sum;

    // Learning rates
    let cc = (4.0 + mu_eff / nf) / (nf + 4.0 + 2.0 * mu_eff / nf);
    let cs = (mu_eff + 2.0) / (nf + mu_eff + 5.0);
    let c1 = 2.0 / ((nf + 1.3).powi(2) + mu_eff);
    let cmu_raw = 2.0 * (mu_eff - 2.0 + 1.0 / mu_eff) / ((nf + 2.0).powi(2) + mu_eff);
    let cmu = cmu_raw.min(1.0 - c1);
    let damps = 1.0 + 2.0 * (0.0_f64).max(((mu_eff - 1.0) / (nf + 1.0)).sqrt() - 1.0) + cs;
    let chi_n = nf.sqrt() * (1.0 - 1.0 / (4.0 * nf) + 1.0 / (21.0 * nf * nf));

    // State variables
    let mut mean: Vec<S> = x0.to_vec();
    let mut sigma = opts.sigma0;

    // Covariance matrix C = I (stored as DenseMatrix)
    let mut c_mat = DenseMatrix::<S>::zeros(n, n);
    for i in 0..n {
        c_mat.set(i, i, S::ONE);
    }

    // Evolution paths
    let mut p_sigma = vec![S::ZERO; n]; // conjugate evolution path for sigma
    let mut p_c = vec![S::ZERO; n]; // evolution path for C

    // Eigendecomposition cache: C = B * D^2 * B^T
    // B = eigenvectors, D = sqrt(eigenvalues)
    let mut bd_mat = DenseMatrix::<S>::zeros(n, n);
    for i in 0..n {
        bd_mat.set(i, i, S::ONE);
    }
    let mut d_diag = vec![S::ONE; n]; // eigenvalues of C (not sqrt)
    let mut inv_sqrt_diag = vec![S::ONE; n]; // 1/sqrt(eigenvalues)

    let mut rng = SmallRng::seed_from_u64(opts.seed);
    let mut n_feval = 0_usize;
    let mut history: Vec<IterationRecord<S>> = Vec::new();
    let mut converged = false;
    let mut iterations = 0;
    let mut best_x = x0.to_vec();
    let mut best_f = f(x0);
    n_feval += 1;

    let mut eigen_update_gen: usize = 0;

    for gen in 0..opts.max_iter {
        iterations = gen + 1;

        // Sample lambda offspring: x_k = mean + sigma * B * D * z_k
        let mut population: Vec<Vec<S>> = Vec::with_capacity(lambda);
        let mut z_vectors: Vec<Vec<S>> = Vec::with_capacity(lambda);

        for _ in 0..lambda {
            // Sample z ~ N(0, I)
            let z: Vec<S> = (0..n).map(|_| sample_standard_normal(&mut rng)).collect();

            // x = mean + sigma * B * D * z
            let mut x = vec![S::ZERO; n];
            for i in 0..n {
                let mut val = S::ZERO;
                for j in 0..n {
                    val += bd_mat.get(i, j) * d_diag[j].sqrt() * z[j];
                }
                x[i] = mean[i] + sigma * val;
            }

            z_vectors.push(z);
            population.push(x);
        }

        // Evaluate fitness
        let mut fitness: Vec<(usize, S)> = population
            .iter()
            .enumerate()
            .map(|(i, x)| (i, f(x)))
            .collect();
        n_feval += lambda;

        // Sort by fitness (ascending = best first)
        fitness.sort_by(|a, b| a.1.to_f64().partial_cmp(&b.1.to_f64()).unwrap());

        // Track best ever
        if fitness[0].1 < best_f {
            best_f = fitness[0].1;
            best_x = population[fitness[0].0].clone();
        }

        if opts.verbose && gen % 50 == 0 {
            eprintln!(
                "CMA-ES gen {}: best_f={:.6e}, sigma={:.4e}",
                gen,
                best_f.to_f64(),
                sigma.to_f64()
            );
        }

        history.push(IterationRecord {
            iteration: gen,
            objective: best_f,
            gradient_norm: sigma,
            step_size: sigma,
            constraint_violation: S::ZERO,
        });

        // Check convergence
        let f_best_gen = fitness[0].1;
        let f_worst_gen = fitness[lambda - 1].1;
        if (f_worst_gen - f_best_gen).abs() < opts.tol_f && sigma < opts.tol_sigma {
            converged = true;
            break;
        }

        // ─── Update mean ───
        let old_mean = mean.clone();
        for j in 0..n {
            mean[j] = S::ZERO;
        }
        for i in 0..mu {
            let idx = fitness[i].0;
            let w_i = S::from_f64(weights[i]);
            for j in 0..n {
                mean[j] += w_i * population[idx][j];
            }
        }

        // ─── Update evolution paths ───
        // p_sigma = (1 - cs) * p_sigma + sqrt(cs * (2 - cs) * mu_eff) * C^{-1/2} * (mean - old_mean) / sigma
        let mean_shift: Vec<S> = (0..n).map(|j| (mean[j] - old_mean[j]) / sigma).collect();

        // C^{-1/2} * mean_shift = B * D^{-1} * B^T * mean_shift
        let mut c_inv_sqrt_shift = vec![S::ZERO; n];
        // temp = B^T * mean_shift
        let mut temp = vec![S::ZERO; n];
        for i in 0..n {
            let mut val = S::ZERO;
            for j in 0..n {
                val += bd_mat.get(j, i) * mean_shift[j]; // B^T: row i = col i of B
            }
            temp[i] = val;
        }
        // temp2 = D^{-1} * temp
        for i in 0..n {
            temp[i] *= inv_sqrt_diag[i];
        }
        // c_inv_sqrt_shift = B * temp2
        for i in 0..n {
            let mut val = S::ZERO;
            for j in 0..n {
                val += bd_mat.get(i, j) * temp[j];
            }
            c_inv_sqrt_shift[i] = val;
        }

        let cs_factor = S::from_f64((cs * (2.0 - cs) * mu_eff).sqrt());
        let one_minus_cs = S::from_f64(1.0 - cs);
        for i in 0..n {
            p_sigma[i] = one_minus_cs * p_sigma[i] + cs_factor * c_inv_sqrt_shift[i];
        }

        // ||p_sigma||
        let ps_norm: f64 = p_sigma
            .iter()
            .map(|&v| v.to_f64() * v.to_f64())
            .sum::<f64>()
            .sqrt();

        // h_sigma: stall indicator
        let gen_factor = 1.0 - (1.0 - cs).powi((2 * (gen + 1)) as i32);
        let threshold = (1.4 + 2.0 / (nf + 1.0)) * chi_n * gen_factor.sqrt();
        let h_sigma: f64 = if ps_norm < threshold { 1.0 } else { 0.0 };

        // p_c = (1 - cc) * p_c + h_sigma * sqrt(cc * (2 - cc) * mu_eff) * mean_shift
        let cc_factor = S::from_f64(h_sigma * (cc * (2.0 - cc) * mu_eff).sqrt());
        let one_minus_cc = S::from_f64(1.0 - cc);
        for i in 0..n {
            p_c[i] = one_minus_cc * p_c[i] + cc_factor * mean_shift[i];
        }

        // ─── Update covariance matrix ───
        // C = (1 - c1 - cmu) * C + c1 * (p_c * p_c^T + delta(h_sigma) * C)
        //   + cmu * sum_i w_i * (x_i - old_mean)*(x_i - old_mean)^T / sigma^2
        let delta_h = (1.0 - h_sigma) * cc * (2.0 - cc);
        let c_scale = S::from_f64(1.0 - c1 - cmu + c1 * delta_h);
        let c1_s = S::from_f64(c1);
        let cmu_s = S::from_f64(cmu);

        for i in 0..n {
            for j in 0..=i {
                let mut val = c_scale * c_mat.get(i, j);
                val += c1_s * p_c[i] * p_c[j];
                // Rank-mu update
                let mut rank_mu = S::ZERO;
                for k in 0..mu {
                    let idx = fitness[k].0;
                    let di = (population[idx][i] - old_mean[i]) / sigma;
                    let dj = (population[idx][j] - old_mean[j]) / sigma;
                    rank_mu += S::from_f64(weights[k]) * di * dj;
                }
                val += cmu_s * rank_mu;
                c_mat.set(i, j, val);
                c_mat.set(j, i, val);
            }
        }

        // ─── Update step size sigma ───
        sigma *= S::from_f64(((cs / damps) * (ps_norm / chi_n - 1.0)).exp());

        // ─── Eigendecomposition of C (every ~n/10 generations) ───
        let eigen_interval = (n / 10).max(1);
        if gen - eigen_update_gen >= eigen_interval {
            eigen_update_gen = gen;
            update_eigen(&c_mat, n, &mut bd_mat, &mut d_diag, &mut inv_sqrt_diag);
        }
    }

    let (status, message) = if converged {
        (
            OptimStatus::GradientConverged,
            format!("CMA-ES converged after {} generations", iterations),
        )
    } else {
        (
            OptimStatus::MaxIterations,
            format!(
                "CMA-ES: max generations ({}) reached, best f = {:.6e}",
                opts.max_iter,
                best_f.to_f64()
            ),
        )
    };

    Ok(OptimResult {
        x: best_x,
        f: best_f,
        grad: Vec::new(),
        iterations,
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
    .with_wall_time(start))
}

/// Update eigendecomposition of the covariance matrix.
/// C = B * diag(d) * B^T
fn update_eigen<S>(
    c_mat: &DenseMatrix<S>,
    n: usize,
    bd_mat: &mut DenseMatrix<S>,
    d_diag: &mut [S],
    inv_sqrt_diag: &mut [S],
) where
    S: Scalar + faer::SimpleEntity + faer::Conjugate<Canonical = S> + faer::ComplexField,
{
    // Use symmetric eigendecomposition
    match c_mat.eigh() {
        Ok(eig) => {
            let eigenvalues = eig.eigenvalues();
            let eigenvectors = eig.eigenvectors();

            for i in 0..n {
                let ev = eigenvalues[i];
                // Clamp eigenvalues to small positive value for numerical stability
                d_diag[i] = if ev > S::from_f64(1e-20) {
                    ev
                } else {
                    S::from_f64(1e-20)
                };
                inv_sqrt_diag[i] = S::ONE / d_diag[i].sqrt();
            }

            // Copy eigenvectors to bd_mat
            for i in 0..n {
                for j in 0..n {
                    bd_mat.set(i, j, eigenvectors.get(i, j));
                }
            }
        }
        Err(_) => {
            // If eigendecomposition fails, reset to identity
            for i in 0..n {
                d_diag[i] = S::ONE;
                inv_sqrt_diag[i] = S::ONE;
                for j in 0..n {
                    bd_mat.set(i, j, if i == j { S::ONE } else { S::ZERO });
                }
            }
        }
    }
}

/// Sample from standard normal using Box-Muller transform.
fn sample_standard_normal<S: Scalar>(rng: &mut SmallRng) -> S {
    use rand::Rng;
    let u1: f64 = rng.gen::<f64>().max(1e-300);
    let u2: f64 = rng.gen::<f64>();
    S::from_f64((-2.0 * u1.ln()).sqrt() * (core::f64::consts::TAU * u2).cos())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cmaes_sphere() {
        let result = cmaes_minimize(
            |x: &[f64]| x.iter().map(|xi| xi * xi).sum::<f64>(),
            &[5.0, 3.0, -2.0],
            &CmaEsOptions {
                max_iter: 2000,
                ..Default::default()
            },
        )
        .unwrap();
        assert!(result.f < 1e-6, "f={}", result.f);
        for &xi in &result.x {
            assert!(xi.abs() < 1e-3, "xi={}", xi);
        }
    }

    #[test]
    fn test_cmaes_rosenbrock() {
        let result = cmaes_minimize(
            |x: &[f64]| (1.0 - x[0]).powi(2) + 100.0 * (x[1] - x[0] * x[0]).powi(2),
            &[-1.0, 1.0],
            &CmaEsOptions {
                sigma0: 1.0,
                max_iter: 5000,
                ..Default::default()
            },
        )
        .unwrap();
        assert!(result.f < 0.01, "f={}", result.f);
    }

    #[test]
    fn test_cmaes_rastrigin() {
        // Global min at (0, 0) with f=0
        let result = cmaes_minimize(
            |x: &[f64]| {
                let n = x.len() as f64;
                10.0 * n
                    + x.iter()
                        .map(|xi| xi * xi - 10.0 * (2.0 * std::f64::consts::PI * xi).cos())
                        .sum::<f64>()
            },
            &[2.0, -2.0],
            &CmaEsOptions {
                sigma0: 2.0,
                max_iter: 5000,
                ..Default::default()
            },
        )
        .unwrap();
        assert!(result.f < 2.0, "f={}", result.f);
    }

    #[test]
    fn test_cmaes_1d() {
        let result = cmaes_minimize(
            |x: &[f64]| (x[0] - 7.0).powi(2),
            &[0.0],
            &CmaEsOptions::default(),
        )
        .unwrap();
        assert!((result.x[0] - 7.0).abs() < 0.1, "x={}", result.x[0]);
    }

    #[test]
    fn test_cmaes_deterministic() {
        let f = |x: &[f64]| x[0] * x[0] + x[1] * x[1];
        let r1 = cmaes_minimize(f, &[3.0, 4.0], &CmaEsOptions::default()).unwrap();
        let r2 = cmaes_minimize(f, &[3.0, 4.0], &CmaEsOptions::default()).unwrap();
        assert_eq!(r1.x, r2.x);
        assert_eq!(r1.f, r2.f);
    }
}

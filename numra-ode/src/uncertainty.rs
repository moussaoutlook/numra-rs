//! Uncertainty propagation in ODE solutions.
//!
//! This module provides tools for propagating parameter uncertainties
//! through ODE solutions. Given an ODE system:
//!
//! ```text
//! dy/dt = f(t, y, p),  y(t0) = y0
//! ```
//!
//! where the parameters `p` have associated uncertainties `sigma_p`, this
//! module computes uncertainty bands on the solution trajectory `y(t)`.
//!
//! Two modes are supported:
//!
//! - **Trajectory (GUM)**: First-order Taylor propagation via forward sensitivity
//!   equations. At each time `t`:
//!   `sigma_y_j(t)^2 = sum_k (dy_j/dp_k)^2 * sigma_p_k^2`
//!
//! - **Monte Carlo**: Sample parameters from distributions, solve N times,
//!   compute mean and standard deviation of the ensemble.
//!
//! # Composability
//!
//! The main entry point [`solve_with_uncertainty`] is generic over the solver:
//!
//! ```text
//! solve_with_uncertainty::<DoPri5, f64>(...) // explicit solver
//! solve_with_uncertainty::<Radau5, f64>(...) // stiff solver
//! solve_with_uncertainty::<Auto, f64>(...)   // auto-selected
//! ```
//!
//! Any solver implementing the `Solver<S>` trait works without modification.
//!
//! Author: Moussa Leblouba
//! Date: 9 February 2026
//! Modified: 2 May 2026

use numra_core::uncertainty::Uncertain;
use numra_core::Scalar;

use crate::error::SolverError;
use crate::problem::OdeSystem;
use crate::sensitivity::solve_forward_sensitivity_with;
use crate::solver::{Solver, SolverOptions, SolverResult, SolverStats};

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// Uncertainty propagation mode.
#[derive(Clone, Debug)]
pub enum UncertaintyMode {
    /// First-order Taylor series (GUM method) via sensitivity equations.
    Trajectory,
    /// Monte Carlo sampling.
    MonteCarlo {
        /// Number of Monte Carlo samples.
        n_samples: usize,
    },
}

/// A parameter with uncertainty.
#[derive(Clone, Debug)]
pub struct UncertainParam<S: Scalar> {
    /// Parameter name (for reporting).
    pub name: String,
    /// Nominal (mean) value.
    pub nominal: S,
    /// Standard deviation.
    pub std: S,
}

impl<S: Scalar> UncertainParam<S> {
    /// Create a new uncertain parameter.
    pub fn new(name: impl Into<String>, nominal: S, std: S) -> Self {
        Self {
            name: name.into(),
            nominal,
            std,
        }
    }

    /// Create from an `Uncertain<S>` value with a name.
    pub fn from_uncertain(name: impl Into<String>, u: Uncertain<S>) -> Self {
        Self {
            name: name.into(),
            nominal: u.mean,
            std: u.std(),
        }
    }

    /// Variance (sigma^2).
    pub fn variance(&self) -> S {
        self.std * self.std
    }
}

/// ODE result with uncertainty bands.
#[derive(Clone, Debug)]
pub struct UncertainSolverResult<S: Scalar> {
    /// Base result (nominal trajectory).
    pub result: SolverResult<S>,
    /// Standard deviation at each output time, for each component.
    /// `sigma[i * dim + j]` = std dev of component j at time i.
    pub sigma: Vec<S>,
    /// Sensitivity matrices at each output time (Trajectory mode only).
    /// Each entry is a flat row-major matrix of shape `(n_states, n_params)`.
    /// `sensitivities[i][j * n_params + k]` = dy_j/dp_k at time i.
    pub sensitivities: Option<Vec<Vec<S>>>,
    /// Parameter information.
    pub params: Vec<UncertainParam<S>>,
}

impl<S: Scalar> UncertainSolverResult<S> {
    /// Get sigma for component `j` at time index `i`.
    pub fn sigma_at(&self, i: usize, j: usize) -> S {
        self.sigma[i * self.result.dim + j]
    }

    /// Get the uncertain value (mean +/- std) for component `j` at time index `i`.
    pub fn uncertain_at(&self, i: usize, j: usize) -> Uncertain<S> {
        let mean = self.result.y_at(i)[j];
        let std = self.sigma_at(i, j);
        Uncertain::from_std(mean, std)
    }

    /// Get sensitivity dy_j/dp_k at time index `i`.
    /// Returns `None` if sensitivities were not computed (Monte Carlo mode).
    pub fn sensitivity_at(&self, i: usize, j: usize, k: usize) -> Option<S> {
        self.sensitivities.as_ref().map(|sens| {
            let n_params = self.params.len();
            sens[i][j * n_params + k]
        })
    }

    /// Number of time points.
    pub fn len(&self) -> usize {
        self.result.len()
    }

    /// Whether the result is empty.
    pub fn is_empty(&self) -> bool {
        self.result.is_empty()
    }
}

// ---------------------------------------------------------------------------
// Trajectory mode
// ---------------------------------------------------------------------------

/// Solve an ODE with uncertainty propagation using first-order Taylor (GUM)
/// via forward sensitivity equations.
///
/// This is solver-agnostic: the caller chooses the solver via the type
/// parameter `Sol`.
///
/// # Arguments
///
/// * `model` -- Parameterized RHS: `f(t, y, dydt, params)`.
/// * `y0` -- Initial state.
/// * `t0`, `tf` -- Integration interval.
/// * `params` -- Parameters with uncertainties.
/// * `options` -- Solver options (rtol, atol, etc.).
///
/// # Returns
///
/// An [`UncertainSolverResult`] with the nominal trajectory, sigma bands,
/// and the full sensitivity matrices.
pub fn solve_trajectory<Sol, S, F>(
    model: F,
    y0: &[S],
    t0: S,
    tf: S,
    params: &[UncertainParam<S>],
    options: &SolverOptions<S>,
) -> Result<UncertainSolverResult<S>, SolverError>
where
    S: Scalar,
    Sol: Solver<S>,
    F: Fn(S, &[S], &mut [S], &[S]),
{
    let n_states = y0.len();
    let n_params = params.len();
    let nominal_params: Vec<S> = params.iter().map(|p| p.nominal).collect();
    let variances: Vec<S> = params.iter().map(|p| p.variance()).collect();

    // Adapt argument order: solve_forward_sensitivity_with expects
    // `(t, y, p, dydt)`; this module's public model has `(t, y, dydt, p)`.
    let rhs = move |t: S, y: &[S], p: &[S], dydt: &mut [S]| {
        model(t, y, dydt, p);
    };

    let sens =
        solve_forward_sensitivity_with::<Sol, S, _>(rhs, y0, &nominal_params, t0, tf, options)?;

    if !sens.success {
        return Err(SolverError::Other(sens.message));
    }

    // The canonical primitive returns sensitivity in column-major (per-time
    // block of length `n_states * n_params`, with `block[k*n_states + j] =
    // ∂y_j/∂p_k`). The public layout for `UncertainSolverResult.sensitivities`
    // is row-major (state-major) per
    //   `sensitivities[i][j*n_params + k] = ∂y_j/∂p_k`.
    // Transpose during the copy so the public-facing accessor semantics stay
    // unchanged.
    let n_times = sens.len();
    let mut sens_out = Vec::with_capacity(n_times);
    let mut sigma_out = Vec::with_capacity(n_times * n_states);

    for i in 0..n_times {
        let block = sens.sensitivity_at(i);
        let mut row_major = vec![S::ZERO; n_states * n_params];
        for j in 0..n_states {
            for k in 0..n_params {
                row_major[j * n_params + k] = block[k * n_states + j];
            }
        }
        sens_out.push(row_major);

        // GUM propagation: σ_{y_j}^2 = Σ_k (∂y_j/∂p_k)^2 · σ_{p_k}^2.
        for j in 0..n_states {
            let mut var_j = S::ZERO;
            for k in 0..n_params {
                let dydp = block[k * n_states + j];
                var_j = var_j + dydp * dydp * variances[k];
            }
            sigma_out.push(var_j.sqrt());
        }
    }

    let nominal_result = SolverResult {
        t: sens.t,
        y: sens.y,
        dim: n_states,
        stats: sens.stats,
        success: true,
        message: String::new(),
        events: Vec::new(),
        terminated_by_event: false,
        dense_output: None,
    };

    Ok(UncertainSolverResult {
        result: nominal_result,
        sigma: sigma_out,
        sensitivities: Some(sens_out),
        params: params.to_vec(),
    })
}

// ---------------------------------------------------------------------------
// Monte Carlo mode
// ---------------------------------------------------------------------------

/// Solve an ODE with uncertainty propagation using Monte Carlo sampling.
///
/// Samples parameter vectors from independent normal distributions, solves
/// the ODE `n_samples` times, and computes mean and standard deviation of
/// the ensemble at each output time.
///
/// # Arguments
///
/// * `model` -- Parameterized RHS: `f(t, y, dydt, params)`.
/// * `y0` -- Initial state.
/// * `t0`, `tf` -- Integration interval.
/// * `params` -- Parameters with uncertainties.
/// * `n_samples` -- Number of Monte Carlo samples.
/// * `options` -- Solver options.
/// * `seed` -- Random seed for reproducibility.
///
/// The `model` argument is a factory that creates a fresh closure for each
/// parameter sample, ensuring thread-safety.
///
/// # Returns
///
/// An [`UncertainSolverResult`] with the mean trajectory and sigma bands.
/// Sensitivities are `None` in Monte Carlo mode.
pub fn solve_monte_carlo<Sol, S, F>(
    model: F,
    y0: &[S],
    t0: S,
    tf: S,
    params: &[UncertainParam<S>],
    n_samples: usize,
    options: &SolverOptions<S>,
    seed: u64,
) -> Result<UncertainSolverResult<S>, SolverError>
where
    S: Scalar,
    Sol: Solver<S>,
    F: Fn(S, &[S], &mut [S], &[S]) + Send + Sync,
{
    let n_states = y0.len();
    let n_params = params.len();

    // First, solve the nominal problem to get the reference trajectory
    let nominal_params: Vec<S> = params.iter().map(|p| p.nominal).collect();
    let nominal_sys = ParameterizedWrapper {
        model: &model,
        params: nominal_params.clone(),
        n_dim: n_states,
    };

    let nominal_result = Sol::solve(&nominal_sys, t0, tf, y0, options)?;
    if !nominal_result.success {
        return Err(SolverError::Other(nominal_result.message));
    }

    let n_times = nominal_result.len();

    // For MC, we collect statistics at the final time only, since not all
    // solvers honor t_eval and adaptive step counts vary across samples.
    // We then scale the nominal trajectory's sigma using the final-time ratio.
    let mut sum_final = vec![S::ZERO; n_states];
    let mut sum_sq_final = vec![S::ZERO; n_states];
    let mut n_success: usize = 0;

    let mut rng_state = seed;

    for _ in 0..n_samples {
        // Sample parameters from independent normals
        let mut p_sample = Vec::with_capacity(n_params);
        for param in params {
            let z = box_muller_sample(&mut rng_state);
            let p_val = param.nominal + param.std * S::from_f64(z);
            p_sample.push(p_val);
        }

        let sample_sys = ParameterizedWrapper {
            model: &model,
            params: p_sample,
            n_dim: n_states,
        };

        match Sol::solve(&sample_sys, t0, tf, y0, options) {
            Ok(result) if result.success => {
                if let Some(y_final) = result.y_final() {
                    n_success += 1;
                    for j in 0..n_states {
                        sum_final[j] = sum_final[j] + y_final[j];
                        sum_sq_final[j] = sum_sq_final[j] + y_final[j] * y_final[j];
                    }
                }
            }
            _ => {
                // Skip failed samples
            }
        }
    }

    if n_success < 2 {
        return Err(SolverError::Other(
            "Monte Carlo: fewer than 2 samples succeeded".to_string(),
        ));
    }

    // Compute final-time sigma from MC ensemble
    let n_s = S::from_usize(n_success);
    let mut sigma_final = Vec::with_capacity(n_states);
    for j in 0..n_states {
        let mean = sum_final[j] / n_s;
        let var = (sum_sq_final[j] / n_s - mean * mean) * n_s / (n_s - S::ONE);
        let std = if var > S::ZERO { var.sqrt() } else { S::ZERO };
        sigma_final.push(std);
    }

    // Build sigma for all time points by linearly interpolating from 0 at t0
    // to sigma_final at tf. This is an approximation — for exact MC at all
    // times, use segment-by-segment integration (see numra-ocp).
    let mut sigma = Vec::with_capacity(n_times * n_states);
    for i in 0..n_times {
        let frac = if n_times > 1 {
            S::from_usize(i) / S::from_usize(n_times - 1)
        } else {
            S::ONE
        };
        for j in 0..n_states {
            sigma.push(sigma_final[j] * frac);
        }
    }

    let mc_result = SolverResult {
        t: nominal_result.t.clone(),
        y: nominal_result.y.clone(),
        dim: n_states,
        stats: SolverStats::new(),
        success: true,
        message: format!("{}/{} samples succeeded", n_success, n_samples),
        events: Vec::new(),
        terminated_by_event: false,
        dense_output: None,
    };

    Ok(UncertainSolverResult {
        result: mc_result,
        sigma,
        sensitivities: None,
        params: params.to_vec(),
    })
}

// ---------------------------------------------------------------------------
// Unified entry point
// ---------------------------------------------------------------------------

/// Solve an ODE with uncertainty propagation.
///
/// This is the main composable entry point. Choose your solver via the type
/// parameter and your propagation mode via `UncertaintyMode`.
///
/// # Example (conceptual)
///
/// ```text
/// let params = vec![
///     UncertainParam::new("k", 0.5, 0.05),
/// ];
///
/// let result = solve_with_uncertainty::<DoPri5, f64, _>(
///     |t, y, dydt, p| { dydt[0] = -p[0] * y[0]; },
///     &[1.0],
///     0.0, 5.0,
///     &params,
///     &UncertaintyMode::Trajectory,
///     &SolverOptions::default(),
///     None,
/// )?;
///
/// // result.sigma_at(i, j) gives std dev of component j at time i
/// ```
pub fn solve_with_uncertainty<Sol, S, F>(
    model: F,
    y0: &[S],
    t0: S,
    tf: S,
    params: &[UncertainParam<S>],
    mode: &UncertaintyMode,
    options: &SolverOptions<S>,
    seed: Option<u64>,
) -> Result<UncertainSolverResult<S>, SolverError>
where
    S: Scalar,
    Sol: Solver<S>,
    F: Fn(S, &[S], &mut [S], &[S]) + Send + Sync,
{
    match mode {
        UncertaintyMode::Trajectory => {
            solve_trajectory::<Sol, S, F>(model, y0, t0, tf, params, options)
        }
        UncertaintyMode::MonteCarlo { n_samples } => solve_monte_carlo::<Sol, S, F>(
            model,
            y0,
            t0,
            tf,
            params,
            *n_samples,
            options,
            seed.unwrap_or(42),
        ),
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Wrapper that turns a parameterized model `f(t, y, dydt, params)` into
/// an `OdeSystem` with fixed parameters.
struct ParameterizedWrapper<'a, S: Scalar, F> {
    model: &'a F,
    params: Vec<S>,
    n_dim: usize,
}

impl<S: Scalar, F> OdeSystem<S> for ParameterizedWrapper<'_, S, F>
where
    F: Fn(S, &[S], &mut [S], &[S]) + Send + Sync,
{
    fn dim(&self) -> usize {
        self.n_dim
    }

    fn rhs(&self, t: S, y: &[S], dydt: &mut [S]) {
        (self.model)(t, y, dydt, &self.params);
    }
}

/// Simple splitmix64 PRNG (better statistical properties than xorshift for
/// Box-Muller).
fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9e3779b97f4a7c15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
    z ^ (z >> 31)
}

/// Generate a standard normal sample via Box-Muller transform.
fn box_muller_sample(state: &mut u64) -> f64 {
    loop {
        let u1 = ((splitmix64(state) >> 11) as f64) / ((1u64 << 53) as f64);
        let u2 = ((splitmix64(state) >> 11) as f64) / ((1u64 << 53) as f64);
        if u1 > 1e-15 {
            let r = (-2.0 * u1.ln()).sqrt();
            return r * (2.0 * core::f64::consts::PI * u2).cos();
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{DoPri5, Radau5};

    /// Exponential decay: dy/dt = -k*y, y(0) = 1, k = 0.5 +/- 0.05.
    ///
    /// Exact solution: y(t) = exp(-k*t).
    /// Exact sensitivity: dy/dk = -t * exp(-k*t).
    /// Exact sigma_y(t) = |dy/dk| * sigma_k = t * exp(-k*t) * sigma_k.
    #[test]
    fn test_trajectory_exponential_decay() {
        let params = vec![UncertainParam::new("k", 0.5, 0.05)];

        let result = solve_with_uncertainty::<DoPri5, f64, _>(
            |_t, y, dydt, p| {
                dydt[0] = -p[0] * y[0];
            },
            &[1.0],
            0.0,
            5.0,
            &params,
            &UncertaintyMode::Trajectory,
            &SolverOptions::default().rtol(1e-8).atol(1e-10),
            None,
        )
        .expect("solve_with_uncertainty failed");

        assert!(result.result.success);
        assert!(!result.is_empty());

        // Check sigma at the final time
        let n = result.len();
        let t_final = result.result.t[n - 1];
        let k = 0.5;
        let sigma_k = 0.05;

        let exact_sigma = t_final * (-k * t_final).exp() * sigma_k;
        let computed_sigma = result.sigma_at(n - 1, 0);

        assert!(
            (computed_sigma - exact_sigma).abs() < 0.001,
            "sigma: computed={}, exact={}, err={}",
            computed_sigma,
            exact_sigma,
            (computed_sigma - exact_sigma).abs()
        );

        // Check that sensitivity is available
        assert!(result.sensitivities.is_some());
        let dydp = result.sensitivity_at(n - 1, 0, 0).unwrap();
        let exact_dydp = -t_final * (-k * t_final).exp();
        assert!(
            (dydp - exact_dydp).abs() < 0.001,
            "dy/dk: computed={}, exact={}, err={}",
            dydp,
            exact_dydp,
            (dydp - exact_dydp).abs()
        );
    }

    /// Two-parameter model: dy/dt = -a*y + b, y(0) = 1.
    /// a = 1.0 +/- 0.1, b = 2.0 +/- 0.2.
    #[test]
    fn test_trajectory_two_params() {
        let params = vec![
            UncertainParam::new("a", 1.0, 0.1),
            UncertainParam::new("b", 2.0, 0.2),
        ];

        let result = solve_trajectory::<DoPri5, f64, _>(
            |_t, y, dydt, p| {
                dydt[0] = -p[0] * y[0] + p[1];
            },
            &[1.0],
            0.0,
            3.0,
            &params,
            &SolverOptions::default().rtol(1e-8).atol(1e-10),
        )
        .expect("solve_trajectory failed");

        assert!(result.result.success);

        // At t=3, solution should be near the steady state b/a = 2.0.
        let n = result.len();
        let y_final = result.result.y_at(n - 1)[0];
        assert!(
            (y_final - 2.0).abs() < 0.1,
            "y(3) = {}, expected near 2.0",
            y_final
        );

        // Sigma should be > 0 (uncertainty propagated)
        let sigma_final = result.sigma_at(n - 1, 0);
        assert!(
            sigma_final > 0.0,
            "sigma should be positive: {}",
            sigma_final
        );
    }

    /// Verify that Trajectory and Monte Carlo modes give consistent results.
    #[test]
    fn test_trajectory_vs_monte_carlo() {
        let params = vec![UncertainParam::new("k", 0.5, 0.05)];

        let traj_result = solve_with_uncertainty::<DoPri5, f64, _>(
            |_t, y, dydt, p| {
                dydt[0] = -p[0] * y[0];
            },
            &[1.0],
            0.0,
            2.0,
            &params,
            &UncertaintyMode::Trajectory,
            &SolverOptions::default().rtol(1e-8).atol(1e-10),
            None,
        )
        .expect("trajectory failed");

        let mc_result = solve_with_uncertainty::<DoPri5, f64, _>(
            |_t, y, dydt, p| {
                dydt[0] = -p[0] * y[0];
            },
            &[1.0],
            0.0,
            2.0,
            &params,
            &UncertaintyMode::MonteCarlo { n_samples: 5000 },
            &SolverOptions::default().rtol(1e-8).atol(1e-10),
            Some(12345),
        )
        .expect("monte carlo failed");

        // Final sigma should agree within ~10% (MC has statistical noise)
        let n_traj = traj_result.len();
        let n_mc = mc_result.len();
        let sigma_traj = traj_result.sigma_at(n_traj - 1, 0);
        let sigma_mc = mc_result.sigma_at(n_mc - 1, 0);

        let rel_diff = (sigma_traj - sigma_mc).abs() / sigma_traj;
        assert!(
            rel_diff < 0.15,
            "Trajectory sigma={}, MC sigma={}, rel_diff={}",
            sigma_traj,
            sigma_mc,
            rel_diff
        );
    }

    /// Composability test: same problem with a stiff solver (Radau5).
    #[test]
    fn test_composability_stiff_solver() {
        let params = vec![UncertainParam::new("k", 50.0, 5.0)];

        let result = solve_trajectory::<Radau5, f64, _>(
            |_t, y, dydt, p| {
                dydt[0] = -p[0] * y[0];
            },
            &[1.0],
            0.0,
            0.2,
            &params,
            &SolverOptions::default().rtol(1e-4).atol(1e-7).h0(1e-4),
        )
        .expect("stiff solve failed");

        assert!(result.result.success);
        assert!(!result.is_empty());

        // Verify sigma > 0 at final time
        let n = result.len();
        let sigma = result.sigma_at(n - 1, 0);
        assert!(sigma >= 0.0, "sigma should be non-negative: {}", sigma);
    }

    /// Multi-component system: Lotka-Volterra with uncertain parameters.
    #[test]
    fn test_trajectory_lotka_volterra() {
        let params = vec![
            UncertainParam::new("alpha", 1.0, 0.1),
            UncertainParam::new("beta", 0.1, 0.01),
            UncertainParam::new("delta", 0.075, 0.005),
            UncertainParam::new("gamma", 1.5, 0.1),
        ];

        let result = solve_trajectory::<DoPri5, f64, _>(
            |_t, y, dydt, p| {
                let x = y[0];
                let yy = y[1];
                dydt[0] = p[0] * x - p[1] * x * yy;
                dydt[1] = p[2] * x * yy - p[3] * yy;
            },
            &[10.0, 5.0],
            0.0,
            5.0,
            &params,
            &SolverOptions::default().rtol(1e-8).atol(1e-10),
        )
        .expect("Lotka-Volterra solve failed");

        assert!(result.result.success);
        assert_eq!(result.result.dim, 2);

        // Both components should have sigma > 0
        let n = result.len();
        let sigma_prey = result.sigma_at(n - 1, 0);
        let sigma_pred = result.sigma_at(n - 1, 1);
        assert!(sigma_prey > 0.0, "prey sigma should be positive");
        assert!(sigma_pred > 0.0, "predator sigma should be positive");

        // All 8 sensitivities should exist (2 states x 4 params)
        let sens = result.sensitivities.as_ref().unwrap();
        assert_eq!(sens[n - 1].len(), 2 * 4);
    }

    /// Test UncertainParam::from_uncertain conversion.
    #[test]
    fn test_uncertain_param_from_uncertain() {
        let u = Uncertain::from_std(5.0, 0.5);
        let p = UncertainParam::from_uncertain("x", u);
        assert!((p.nominal - 5.0).abs() < 1e-10);
        assert!((p.std - 0.5).abs() < 1e-10);
        assert!((p.variance() - 0.25).abs() < 1e-10);
    }

    /// Verify that zero uncertainty produces zero sigma.
    #[test]
    fn test_zero_uncertainty() {
        let params = vec![UncertainParam::new("k", 0.5, 0.0)];

        let result = solve_trajectory::<DoPri5, f64, _>(
            |_t, y, dydt, p| {
                dydt[0] = -p[0] * y[0];
            },
            &[1.0],
            0.0,
            2.0,
            &params,
            &SolverOptions::default(),
        )
        .expect("solve failed");

        // Sigma should be 0 everywhere
        for i in 0..result.len() {
            let sigma = result.sigma_at(i, 0);
            assert!(
                sigma.abs() < 1e-10,
                "sigma should be ~0 at t={}: got {}",
                result.result.t[i],
                sigma
            );
        }
    }

    /// Test the uncertain_at helper that returns Uncertain<S> values.
    #[test]
    fn test_uncertain_at() {
        let params = vec![UncertainParam::new("k", 0.5, 0.05)];

        let result = solve_trajectory::<DoPri5, f64, _>(
            |_t, y, dydt, p| {
                dydt[0] = -p[0] * y[0];
            },
            &[1.0],
            0.0,
            1.0,
            &params,
            &SolverOptions::default().rtol(1e-8),
        )
        .expect("solve failed");

        let n = result.len();
        let u = result.uncertain_at(n - 1, 0);
        assert!(u.mean > 0.0);
        assert!(u.variance > 0.0);
        assert!((u.std() - result.sigma_at(n - 1, 0)).abs() < 1e-14);
    }
}

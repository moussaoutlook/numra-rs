//! Forward sensitivity analysis for parameterized ODE models.
//!
//! Given an ODE model `dy/dt = f(t, y, p)` with parameters `p`, this module
//! computes the sensitivity matrix `S(t) = dy/dp` by solving the augmented
//! system:
//!
//! ```text
//! dS/dt = (df/dy) · S + df/dp,  S(t0) = 0
//! ```
//!
//! As of the v0.1 sensitivity unification, this module is a thin wrapper
//! over the canonical primitive in [`numra_ode::sensitivity`]. The
//! [`SensitivityResult`] type is re-exported from `numra-ode` so the OCP
//! and ODE layers share a single shape and accessor surface — no
//! duplicated row-major-vs-column-major conversions, no parallel test
//! suites. Jacobians `df/dy` and `df/dp` use forward finite differences
//! by default; users with stiff problems should implement
//! [`numra_ode::ParametricOdeSystem`] directly for analytical overrides.
//!
//! Author: Moussa Leblouba
//! Date: 9 February 2026
//! Modified: 6 May 2026

use numra_core::Scalar;
use numra_ode::sensitivity::solve_forward_sensitivity_with;
pub use numra_ode::SensitivityResult;
use numra_ode::{AugmentedSystem, ClosureSystem, DoPri5, Solver, SolverOptions};

use crate::error::OcpError;

/// ODE model closure signature: `(t, y, dydt, params)`.
type ModelFn<S> = dyn Fn(S, &[S], &mut [S], &[S]);

// ---------------------------------------------------------------------------
// Main entry point
// ---------------------------------------------------------------------------

/// Compute forward sensitivities of an ODE solution w.r.t. parameters.
///
/// Thin wrapper over [`numra_ode::sensitivity::solve_forward_sensitivity_with`].
/// Accepts the OCP-native `(t, y, dydt, params)` closure shape and adapts
/// to the canonical `(t, y, p, dydt)` primitive internally.
///
/// # Arguments
///
/// * `model` — ODE right-hand side `f(t, y, dydt, params)`.
/// * `y0` — Initial state.
/// * `params` — Parameter vector.
/// * `t0`, `tf` — Integration interval `[t0, tf]`.
/// * `output_times` — Optional output times. If `None`, the solver chooses
///   adaptively. If provided, integration runs segment-by-segment to land
///   exactly on each requested time.
/// * `rtol`, `atol` — Relative and absolute tolerances for the ODE solver.
///
/// # Returns
///
/// A [`SensitivityResult`] (re-exported from `numra-ode`) containing the
/// state trajectory and the sensitivity matrix `S(t) = dy/dp`. The
/// sensitivity layout is **column-major over parameters**:
/// `sensitivity[i*(N*N_s) + k*N + j] = ∂y_j(t_i)/∂p_k`. Use the typed
/// accessors (`sensitivity_at`, `sensitivity_for_param`, `dyi_dpj`,
/// `final_sensitivity`) instead of indexing the flat `Vec` directly.
#[allow(clippy::too_many_arguments)]
pub fn forward_sensitivity<S: Scalar>(
    model: &ModelFn<S>,
    y0: &[S],
    params: &[S],
    t0: S,
    tf: S,
    output_times: Option<&[S]>,
    rtol: S,
    atol: S,
) -> Result<SensitivityResult<S>, OcpError> {
    let opts = SolverOptions::default().rtol(rtol).atol(atol);

    match output_times {
        None => solve_forward_sensitivity_with::<DoPri5, S, _>(
            |t: S, y: &[S], p: &[S], dy: &mut [S]| model(t, y, dy, p),
            y0,
            params,
            t0,
            tf,
            &opts,
        )
        .map_err(|e| OcpError::IntegrationFailed(e.to_string())),
        Some(te) => integrate_at_output_times(model, y0, params, te, &opts),
    }
}

/// Segment-by-segment integration that lands exactly on each requested
/// output time. Drives the canonical [`AugmentedSystem`] directly so the
/// sensitivity state can be carried across segment boundaries without
/// re-initialising to zero.
fn integrate_at_output_times<S: Scalar>(
    model: &ModelFn<S>,
    y0: &[S],
    params: &[S],
    te: &[S],
    opts: &SolverOptions<S>,
) -> Result<SensitivityResult<S>, OcpError> {
    let n_states = y0.len();
    let n_params = params.len();

    if te.is_empty() {
        return Err(OcpError::IntegrationFailed(
            "output_times must contain at least one entry".to_string(),
        ));
    }

    let system = ClosureSystem::new(
        |t: S, y: &[S], p: &[S], dy: &mut [S]| model(t, y, dy, p),
        params.to_vec(),
        n_states,
    );
    let aug = AugmentedSystem::new(system);
    let aug_dim = aug.augmented_dim();
    let mut z_cur = aug.initial_augmented(y0);

    let tiny = S::from_f64(1e-15);

    let mut t_out = Vec::with_capacity(te.len());
    let mut y_out = Vec::with_capacity(te.len() * n_states);
    let mut sens_out = Vec::with_capacity(te.len() * n_states * n_params);

    // Record the state at the first requested time (no integration yet).
    t_out.push(te[0]);
    y_out.extend_from_slice(&z_cur[..n_states]);
    sens_out.extend_from_slice(&z_cur[n_states..aug_dim]);

    let mut last_stats = numra_ode::SolverStats::new();

    for seg in 0..(te.len() - 1) {
        let t_start = te[seg];
        let t_end = te[seg + 1];

        if (t_end - t_start).abs() < tiny {
            t_out.push(t_end);
            y_out.extend_from_slice(&z_cur[..n_states]);
            sens_out.extend_from_slice(&z_cur[n_states..aug_dim]);
            continue;
        }

        let result = DoPri5::solve(&aug, t_start, t_end, &z_cur, opts)
            .map_err(|e| OcpError::IntegrationFailed(e.to_string()))?;

        if !result.success {
            return Err(OcpError::IntegrationFailed(result.message));
        }

        z_cur = result
            .y_final()
            .ok_or_else(|| OcpError::IntegrationFailed("missing final state".to_string()))?;
        last_stats = result.stats;

        t_out.push(t_end);
        y_out.extend_from_slice(&z_cur[..n_states]);
        sens_out.extend_from_slice(&z_cur[n_states..aug_dim]);
    }

    Ok(SensitivityResult {
        t: t_out,
        y: y_out,
        sensitivity: sens_out,
        n_states,
        n_params,
        stats: last_stats,
        success: true,
        message: String::new(),
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use numra_ode::OdeProblem;

    /// Exponential decay: dy/dt = -k*y, y(0)=1, k=0.5.
    ///
    /// Analytical sensitivity: dy/dk(t) = -t * exp(-k*t).
    #[test]
    fn test_exponential_decay_sensitivity() {
        let k = 0.5_f64;
        let y0 = [1.0];
        let params = [k];

        let check_times = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let result = forward_sensitivity(
            &|_t: f64, y, dydt, p| {
                dydt[0] = -p[0] * y[0];
            },
            &y0,
            &params,
            0.0,
            5.0,
            Some(&check_times),
            1e-10,
            1e-12,
        )
        .expect("forward_sensitivity failed");

        assert_eq!(result.n_states, 1);
        assert_eq!(result.n_params, 1);

        // Check at t = 1, 2, 3, 4, 5 (indices 1..=5 in check_times).
        for (idx, &t) in check_times.iter().enumerate().skip(1) {
            let analytical = -t * (-k * t).exp();
            // Column-major over params: sensitivity_at(idx)[k*N + j] with
            // N = N_p = 1 → offset 0. Equivalent to dyi_dpj(idx, 0, 0).
            let computed = result.sensitivity_at(idx)[0];
            assert!(
                (computed - analytical).abs() < 1e-3,
                "t={t}: computed={computed}, analytical={analytical}, err={}",
                (computed - analytical).abs()
            );
        }
    }

    /// Two-parameter model: dy/dt = -a*y + b, y(0)=1, a=1, b=2.
    ///
    /// Analytical: y(t) = b/a + (y0 - b/a)*exp(-a*t) = 2 - exp(-t).
    /// Analytical dy/db(t) = (1/a)*(1 - exp(-a*t)) = 1 - exp(-t).
    #[test]
    fn test_two_param_sensitivity() {
        let a = 1.0_f64;
        let b = 2.0_f64;
        let y0 = [1.0];
        let params = [a, b];

        let check_times = vec![0.0, 1.0, 2.0, 3.0];
        let result = forward_sensitivity(
            &|_t: f64, y, dydt, p| {
                dydt[0] = -p[0] * y[0] + p[1];
            },
            &y0,
            &params,
            0.0,
            3.0,
            Some(&check_times),
            1e-10,
            1e-12,
        )
        .expect("forward_sensitivity failed");

        assert_eq!(result.n_states, 1);
        assert_eq!(result.n_params, 2);

        // Check dy/db at t = 1, 2, 3 (parameter index 1).
        for (idx, &t) in check_times.iter().enumerate().skip(1) {
            let analytical_dydb = 1.0 - (-t).exp();
            // Sensitivity is stored column-major over parameters:
            // s[k*N + j] = ∂y_j/∂p_k. Use the typed accessor instead of
            // raw indexing to make intent obvious and layout-independent.
            let computed = result.dyi_dpj(idx, 0, 1);
            assert!(
                (computed - analytical_dydb).abs() < 1e-3,
                "t={t}: computed dy/db={computed}, analytical={analytical_dydb}, err={}",
                (computed - analytical_dydb).abs()
            );
        }
    }

    /// Nonlinear model: dy/dt = -p*y^2, y(0)=1, p=0.5.
    ///
    /// Compare forward sensitivity S(T) with central finite differences
    /// of the solution: (y(T; p+h) - y(T; p-h)) / (2h).
    #[test]
    fn test_sensitivity_matches_finite_diff() {
        let p_val = 0.5_f64;
        let y0 = [1.0];
        let t_final = 2.0;

        let model = |_t: f64, y: &[f64], dydt: &mut [f64], p: &[f64]| {
            dydt[0] = -p[0] * y[0] * y[0];
        };

        // Forward sensitivity at p.
        let result = forward_sensitivity(
            &model,
            &y0,
            &[p_val],
            0.0,
            t_final,
            Some(&[0.0, t_final]),
            1e-10,
            1e-12,
        )
        .expect("forward_sensitivity failed");

        // Column-major: dyi_dpj(time_idx, state, param) = ∂y_state(t_i)/∂p_param.
        let sens_forward = result.dyi_dpj(1, 0, 0);

        // Central finite-difference estimate.
        let h = 1e-5;

        // y(T; p + h)
        let opts = SolverOptions::default().rtol(1e-12).atol(1e-14);
        let p_plus = p_val + h;
        let problem_plus = OdeProblem::new(
            move |_t: f64, y: &[f64], dydt: &mut [f64]| {
                dydt[0] = -p_plus * y[0] * y[0];
            },
            0.0,
            t_final,
            vec![1.0],
        );
        let res_plus = DoPri5::solve(&problem_plus, 0.0, t_final, &[1.0], &opts)
            .expect("integration p+h failed");
        let y_plus = res_plus.y_final().unwrap()[0];

        // y(T; p - h)
        let p_minus = p_val - h;
        let problem_minus = OdeProblem::new(
            move |_t: f64, y: &[f64], dydt: &mut [f64]| {
                dydt[0] = -p_minus * y[0] * y[0];
            },
            0.0,
            t_final,
            vec![1.0],
        );
        let res_minus = DoPri5::solve(&problem_minus, 0.0, t_final, &[1.0], &opts)
            .expect("integration p-h failed");
        let y_minus = res_minus.y_final().unwrap()[0];

        let fd_sens = (y_plus - y_minus) / (2.0 * h);

        assert!(
            (sens_forward - fd_sens).abs() < 1e-3,
            "forward sensitivity={sens_forward}, FD={fd_sens}, err={}",
            (sens_forward - fd_sens).abs()
        );
    }
}

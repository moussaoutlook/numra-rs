//! Forward sensitivity analysis for parameterized ODE models.
//!
//! Given an ODE model `dy/dt = f(t, y, p)` with parameters `p`, this module
//! computes the sensitivity matrix `S(t) = dy/dp` by solving the augmented
//! system:
//!
//! ```text
//! dS/dt = (df/dy) * S + df/dp,  S(t0) = 0
//! ```
//!
//! Jacobians `df/dy` and `df/dp` are computed via forward finite differences.
//!
//! Author: Moussa Leblouba
//! Date: 9 February 2026
//! Modified: 2 May 2026

use numra_core::Scalar;
use numra_ode::{DoPri5, OdeProblem, Solver, SolverOptions};

use crate::error::OcpError;

/// ODE model closure signature: `(t, y, dydt, params)`.
type ModelFn<S> = dyn Fn(S, &[S], &mut [S], &[S]);

// ---------------------------------------------------------------------------
// Result type
// ---------------------------------------------------------------------------

/// Result of forward sensitivity analysis.
///
/// Stores the state trajectory and sensitivity matrices at each output time.
#[derive(Clone, Debug)]
pub struct SensitivityResult<S: Scalar> {
    /// Time points.
    pub t: Vec<S>,
    /// State trajectory (flat row-major: `y[i * n_states + j]`).
    pub y: Vec<S>,
    /// Sensitivity matrices at each time (flat:
    /// `sens[i * n_states * n_params + state * n_params + param]`).
    pub sensitivity: Vec<S>,
    /// Number of state variables.
    pub n_states: usize,
    /// Number of parameters.
    pub n_params: usize,
}

impl<S: Scalar> SensitivityResult<S> {
    /// Return the sensitivity matrix at time index `i` as a slice of length
    /// `n_states * n_params`.
    pub fn sensitivity_at(&self, i: usize) -> &[S] {
        let block = self.n_states * self.n_params;
        let start = i * block;
        &self.sensitivity[start..start + block]
    }

    /// Return the state vector at time index `i` as a slice of length
    /// `n_states`.
    pub fn y_at(&self, i: usize) -> &[S] {
        let start = i * self.n_states;
        &self.y[start..start + self.n_states]
    }

    /// Number of output time points.
    pub fn len(&self) -> usize {
        self.t.len()
    }

    /// Whether the result is empty.
    pub fn is_empty(&self) -> bool {
        self.t.is_empty()
    }
}

// ---------------------------------------------------------------------------
// Augmented RHS helper
// ---------------------------------------------------------------------------

/// Evaluate the augmented right-hand side for the combined state+sensitivity
/// system.
///
/// `z = [y; S_flat]` where `S_flat[state * np + param]`.
/// `dz = [f(t,y,p); dS/dt_flat]`.
#[allow(clippy::too_many_arguments)]
fn augmented_rhs<S: Scalar>(
    model: &ModelFn<S>,
    t: S,
    z: &[S],
    dz: &mut [S],
    params: &[S],
    ns: usize,
    np: usize,
) {
    let fd_eps = S::from_f64(1e-7);
    let y = &z[..ns];

    let mut f0 = vec![S::ZERO; ns];
    let mut f_pert = vec![S::ZERO; ns];

    // (a) f(t, y, p)
    model(t, y, &mut f0, params);
    dz[..ns].copy_from_slice(&f0);

    // (b) df/dy via forward finite differences.
    let mut df_dy = vec![S::ZERO; ns * ns];
    let mut y_pert = y.to_vec();
    for j in 0..ns {
        let h_j = fd_eps * (S::ONE + y[j].abs());
        let y_j_orig = y_pert[j];
        y_pert[j] = y_j_orig + h_j;
        model(t, &y_pert, &mut f_pert, params);
        for i in 0..ns {
            df_dy[i * ns + j] = (f_pert[i] - f0[i]) / h_j;
        }
        y_pert[j] = y_j_orig;
    }

    // (c) df/dp via forward finite differences.
    let mut df_dp = vec![S::ZERO; ns * np];
    let mut p_pert = params.to_vec();
    for k in 0..np {
        let h_k = fd_eps * (S::ONE + params[k].abs());
        let p_k_orig = p_pert[k];
        p_pert[k] = p_k_orig + h_k;
        model(t, y, &mut f_pert, &p_pert);
        for i in 0..ns {
            df_dp[i * np + k] = (f_pert[i] - f0[i]) / h_k;
        }
        p_pert[k] = p_k_orig;
    }

    // (d) dS/dt = (df/dy) * S + df/dp
    let s_flat = &z[ns..];
    for i in 0..ns {
        for k in 0..np {
            let mut val = S::ZERO;
            for j in 0..ns {
                val += df_dy[i * ns + j] * s_flat[j * np + k];
            }
            val += df_dp[i * np + k];
            dz[ns + i * np + k] = val;
        }
    }
}

// ---------------------------------------------------------------------------
// Main entry point
// ---------------------------------------------------------------------------

/// Compute forward sensitivities of an ODE solution w.r.t. parameters.
///
/// # Arguments
///
/// * `model` -- ODE right-hand side `f(t, y, dydt, params)`.
/// * `y0` -- Initial state.
/// * `params` -- Parameter vector.
/// * `t0`, `tf` -- Integration interval `[t0, tf]`.
/// * `t_eval` -- Optional output times. If `None`, the solver chooses
///   adaptively.
/// * `rtol`, `atol` -- Relative and absolute tolerances for the ODE solver.
///
/// # Returns
///
/// A [`SensitivityResult`] containing the state trajectory and the
/// sensitivity matrix `S(t) = dy/dp` at each output time.
#[allow(clippy::too_many_arguments)]
pub fn forward_sensitivity<S: Scalar>(
    model: &ModelFn<S>,
    y0: &[S],
    params: &[S],
    t0: S,
    tf: S,
    t_eval: Option<&[S]>,
    rtol: S,
    atol: S,
) -> Result<SensitivityResult<S>, OcpError> {
    let n_states = y0.len();
    let n_params = params.len();
    let n_aug = n_states + n_states * n_params;

    let tiny = S::from_f64(1e-15);

    // -- Build augmented initial condition: [y0; 0 ... 0] ------------------
    let mut z0 = Vec::with_capacity(n_aug);
    z0.extend_from_slice(y0);
    z0.resize(n_aug, S::ZERO);

    let opts = SolverOptions::default().rtol(rtol).atol(atol);

    let mut t_out = Vec::new();
    let mut y_out = Vec::new();
    let mut sens_out = Vec::new();

    if let Some(te) = t_eval {
        // -- Segment-by-segment integration at exact output times ----------
        // DoPri5 does not respect `t_eval` in SolverOptions, so we
        // integrate between consecutive requested times to land exactly
        // on each one.
        let mut z_cur = z0;

        // Record the state at the first output time.
        t_out.push(te[0]);
        y_out.extend_from_slice(&z_cur[..n_states]);
        sens_out.extend_from_slice(&z_cur[n_states..n_states + n_states * n_params]);

        for seg in 0..(te.len() - 1) {
            let t_start = te[seg];
            let t_end = te[seg + 1];

            // Skip zero-length segments.
            if (t_end - t_start).abs() < tiny {
                t_out.push(t_end);
                y_out.extend_from_slice(&z_cur[..n_states]);
                sens_out.extend_from_slice(&z_cur[n_states..n_states + n_states * n_params]);
                continue;
            }

            let ns = n_states;
            let np = n_params;
            let p = params.to_vec();
            let rhs = move |t: S, z: &[S], dz: &mut [S]| {
                augmented_rhs(model, t, z, dz, &p, ns, np);
            };

            let problem = OdeProblem::new(rhs, t_start, t_end, z_cur.clone());
            let result = DoPri5::solve(&problem, t_start, t_end, &z_cur, &opts)
                .map_err(|e| OcpError::IntegrationFailed(e.to_string()))?;

            if !result.success {
                return Err(OcpError::IntegrationFailed(result.message));
            }

            z_cur = result.y_final().unwrap();

            t_out.push(t_end);
            y_out.extend_from_slice(&z_cur[..n_states]);
            sens_out.extend_from_slice(&z_cur[n_states..n_states + n_states * n_params]);
        }
    } else {
        // -- Single integration, keep all adaptive steps -------------------
        let ns = n_states;
        let np = n_params;
        let p = params.to_vec();
        let rhs = move |t: S, z: &[S], dz: &mut [S]| {
            augmented_rhs(model, t, z, dz, &p, ns, np);
        };

        let problem = OdeProblem::new(rhs, t0, tf, z0.clone());
        let result = DoPri5::solve(&problem, t0, tf, &z0, &opts)
            .map_err(|e| OcpError::IntegrationFailed(e.to_string()))?;

        if !result.success {
            return Err(OcpError::IntegrationFailed(result.message));
        }

        let n_times = result.len();
        t_out.reserve(n_times);
        y_out.reserve(n_times * n_states);
        sens_out.reserve(n_times * n_states * n_params);

        for i in 0..n_times {
            t_out.push(result.t[i]);
            let aug_i = result.y_at(i);
            y_out.extend_from_slice(&aug_i[..n_states]);
            sens_out.extend_from_slice(&aug_i[n_states..n_states + n_states * n_params]);
        }
    }

    Ok(SensitivityResult {
        t: t_out,
        y: y_out,
        sensitivity: sens_out,
        n_states,
        n_params,
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

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
            // Sensitivity is stored as S[state * n_params + param].
            // state=0, param=1 => offset 1.
            let computed = result.sensitivity_at(idx)[1];
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

        let sens_forward = result.sensitivity_at(1)[0];

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

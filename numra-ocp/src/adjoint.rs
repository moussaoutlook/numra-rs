//! Adjoint sensitivity analysis for parameterized ODE models.
//!
//! Given an ODE `dx/dt = f(t, x, p)` and a scalar objective
//!
//! ```text
//! J = phi(x(tf)) + integral_t0^tf L(t, x, p) dt
//! ```
//!
//! the adjoint method computes `dJ/dp` in O(n_outputs) cost, compared to
//! O(n_states * n_params) for forward sensitivity. This is much more
//! efficient when `n_params >> n_states`.
//!
//! # Algorithm
//!
//! 1. **Forward sweep**: Integrate `dx/dt = f(t, x, p)` to obtain the state
//!    trajectory `x(t)`.
//! 2. **Terminal condition**: `lambda(tf) = (d phi / d x)(x(tf))`
//! 3. **Backward sweep**: Integrate the costate equation backward in time:
//!    `d lambda/dt = -(df/dx)^T lambda - (dL/dx)^T`
//! 4. **Gradient**: `dJ/dp = (d phi / d p)(x(tf)) + integral_t0^tf [(df/dp)^T lambda + (dL/dp)] dt`
//!
//! Author: Moussa Leblouba
//! Date: 9 February 2026
//! Modified: 2 May 2026

use numra_core::Scalar;
use numra_ode::{DoPri5, OdeProblem, Solver, SolverOptions};

use crate::error::OcpError;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// ODE model closure: `f(t, x, dxdt, params)`.
type ModelFn<S> = dyn Fn(S, &[S], &mut [S], &[S]);

/// Terminal cost closure: `phi(x(tf)) -> S`.
type TerminalCostFn<S> = dyn Fn(&[S]) -> S;

/// Running cost closure: `L(t, x, params) -> S`.
type RunningCostFn<S> = dyn Fn(S, &[S], &[S]) -> S;

/// Result of adjoint sensitivity analysis.
#[derive(Clone, Debug)]
pub struct AdjointResult<S: Scalar> {
    /// Gradient `dJ/dp` (length `n_params`).
    pub gradient: Vec<S>,
    /// Objective value `J`.
    pub objective: S,
    /// Costate trajectory (flat row-major: `costate[i * n_states + j]`).
    /// Stored in forward time order (reversed from backward integration).
    pub costate: Vec<S>,
    /// Time points of the costate trajectory.
    pub costate_time: Vec<S>,
    /// Number of states.
    pub n_states: usize,
    /// Number of parameters.
    pub n_params: usize,
}

// ---------------------------------------------------------------------------
// Helper: evaluate model and FD Jacobians
// ---------------------------------------------------------------------------

/// Evaluate `f(t, x, p)` and return the result.
fn eval_dynamics<S: Scalar>(model: &ModelFn<S>, t: S, x: &[S], p: &[S], ns: usize) -> Vec<S> {
    let mut dxdt = vec![S::ZERO; ns];
    model(t, x, &mut dxdt, p);
    dxdt
}

/// Compute `df/dx` via forward finite differences.
fn compute_dfdx<S: Scalar>(
    model: &ModelFn<S>,
    t: S,
    x: &[S],
    p: &[S],
    f0: &[S],
    ns: usize,
) -> Vec<S> {
    let h_factor = S::EPSILON.sqrt();
    let mut dfdx = vec![S::ZERO; ns * ns]; // row-major: dfdx[i*ns + j] = df_i/dx_j
    let mut x_pert = x.to_vec();
    let mut f_pert = vec![S::ZERO; ns];
    for j in 0..ns {
        let h = h_factor * (S::ONE + x[j].abs());
        let x_orig = x_pert[j];
        x_pert[j] = x_orig + h;
        model(t, &x_pert, &mut f_pert, p);
        for i in 0..ns {
            dfdx[i * ns + j] = (f_pert[i] - f0[i]) / h;
        }
        x_pert[j] = x_orig;
    }
    dfdx
}

/// Compute `df/dp` via forward finite differences.
fn compute_dfdp<S: Scalar>(
    model: &ModelFn<S>,
    t: S,
    x: &[S],
    p: &[S],
    f0: &[S],
    ns: usize,
    np: usize,
) -> Vec<S> {
    let h_factor = S::EPSILON.sqrt();
    let mut dfdp = vec![S::ZERO; ns * np]; // row-major: dfdp[i*np + k] = df_i/dp_k
    let mut p_pert = p.to_vec();
    let mut f_pert = vec![S::ZERO; ns];
    for k in 0..np {
        let h = h_factor * (S::ONE + p[k].abs());
        let p_orig = p_pert[k];
        p_pert[k] = p_orig + h;
        model(t, x, &mut f_pert, &p_pert);
        for i in 0..ns {
            dfdp[i * np + k] = (f_pert[i] - f0[i]) / h;
        }
        p_pert[k] = p_orig;
    }
    dfdp
}

/// Compute gradient of a scalar function `g(x)` via FD.
fn grad_fd<S: Scalar>(g: &dyn Fn(&[S]) -> S, x: &[S], n: usize) -> Vec<S> {
    let h_factor = S::EPSILON.sqrt();
    let g0 = g(x);
    let mut grad = vec![S::ZERO; n];
    let mut x_pert = x.to_vec();
    for j in 0..n {
        let h = h_factor * (S::ONE + x[j].abs());
        let x_orig = x_pert[j];
        x_pert[j] = x_orig + h;
        grad[j] = (g(&x_pert) - g0) / h;
        x_pert[j] = x_orig;
    }
    grad
}

/// Type alias for a scalar function of `(t, x, p)`.
type ScalarFn3<S> = dyn Fn(S, &[S], &[S]) -> S;

/// Compute gradient of a scalar function `g(t, x, p)` w.r.t. `x` via FD.
fn grad_x_fd<S: Scalar>(g: &ScalarFn3<S>, t: S, x: &[S], p: &[S], ns: usize) -> Vec<S> {
    let h_factor = S::EPSILON.sqrt();
    let g0 = g(t, x, p);
    let mut grad = vec![S::ZERO; ns];
    let mut x_pert = x.to_vec();
    for j in 0..ns {
        let h = h_factor * (S::ONE + x[j].abs());
        let x_orig = x_pert[j];
        x_pert[j] = x_orig + h;
        grad[j] = (g(t, &x_pert, p) - g0) / h;
        x_pert[j] = x_orig;
    }
    grad
}

/// Compute gradient of a scalar function `g(t, x, p)` w.r.t. `p` via FD.
fn grad_p_fd<S: Scalar>(g: &ScalarFn3<S>, t: S, x: &[S], p: &[S], np: usize) -> Vec<S> {
    let h_factor = S::EPSILON.sqrt();
    let g0 = g(t, x, p);
    let mut grad = vec![S::ZERO; np];
    let mut p_pert = p.to_vec();
    for k in 0..np {
        let h = h_factor * (S::ONE + p[k].abs());
        let p_orig = p_pert[k];
        p_pert[k] = p_orig + h;
        grad[k] = (g(t, x, &p_pert) - g0) / h;
        p_pert[k] = p_orig;
    }
    grad
}

// ---------------------------------------------------------------------------
// Main entry point
// ---------------------------------------------------------------------------

/// Compute the gradient `dJ/dp` of a scalar objective via the adjoint method.
///
/// # Arguments
///
/// * `model` -- ODE right-hand side `f(t, x, dxdt, params)`.
/// * `terminal_cost` -- Terminal cost `phi(x(tf))`. Must be provided.
/// * `running_cost` -- Running cost `L(t, x, params)`. Optional.
/// * `x0` -- Initial state.
/// * `params` -- Parameter vector.
/// * `t_span` -- Integration interval `(t0, tf)`.
/// * `rtol`, `atol` -- ODE solver tolerances.
///
/// # Returns
///
/// An [`AdjointResult`] containing the gradient and costate trajectory.
#[allow(clippy::too_many_arguments)]
pub fn adjoint_gradient<S: Scalar>(
    model: &ModelFn<S>,
    terminal_cost: &TerminalCostFn<S>,
    running_cost: Option<&RunningCostFn<S>>,
    x0: &[S],
    params: &[S],
    t_span: (S, S),
    rtol: S,
    atol: S,
) -> Result<AdjointResult<S>, OcpError> {
    let ns = x0.len();
    let np = params.len();
    let (t0, tf) = t_span;
    let tiny = S::from_f64(1e-30);

    // -----------------------------------------------------------------------
    // Step 1: Forward integration to get state trajectory
    // -----------------------------------------------------------------------
    let p_fwd = params.to_vec();
    let fwd_rhs = move |t: S, x: &[S], dxdt: &mut [S]| {
        model(t, x, dxdt, &p_fwd);
    };

    let opts = SolverOptions::default().rtol(rtol).atol(atol);
    let problem_fwd = OdeProblem::new(fwd_rhs, t0, tf, x0.to_vec());
    let fwd_result = DoPri5::solve(&problem_fwd, t0, tf, x0, &opts)
        .map_err(|e| OcpError::IntegrationFailed(e.to_string()))?;

    if !fwd_result.success {
        return Err(OcpError::IntegrationFailed(fwd_result.message));
    }

    let n_fwd = fwd_result.t.len();
    if n_fwd == 0 {
        return Err(OcpError::IntegrationFailed(
            "empty forward trajectory".into(),
        ));
    }

    // Store the forward trajectory for interpolation during backward sweep.
    // We'll use piecewise linear interpolation.
    let fwd_t = &fwd_result.t;
    let fwd_y = &fwd_result.y;

    // Compute objective value.
    let x_tf = &fwd_y[(n_fwd - 1) * ns..n_fwd * ns];
    let mut objective = terminal_cost(x_tf);

    // Running cost integral via trapezoidal rule on forward trajectory.
    if let Some(rc) = running_cost {
        for i in 0..n_fwd.saturating_sub(1) {
            let ti = fwd_t[i];
            let ti1 = fwd_t[i + 1];
            let xi = &fwd_y[i * ns..(i + 1) * ns];
            let xi1 = &fwd_y[(i + 1) * ns..(i + 2) * ns];
            let li = rc(ti, xi, params);
            let li1 = rc(ti1, xi1, params);
            objective += S::HALF * (ti1 - ti) * (li + li1);
        }
    }

    // -----------------------------------------------------------------------
    // Step 2: Terminal condition for costate
    // -----------------------------------------------------------------------
    let lambda_tf = grad_fd(terminal_cost, x_tf, ns);

    // -----------------------------------------------------------------------
    // Step 3: Backward integration of costate
    // -----------------------------------------------------------------------
    // We integrate backward by substituting tau = tf - t (so tau goes from 0
    // to tf - t0), and the costate equation becomes:
    //
    //   d lambda / d tau = (df/dx)^T lambda + (dL/dx)^T
    //
    // (note the sign flip from d/dt to d/dtau).

    // We need to interpolate x(t) at arbitrary t during backward integration.
    // Precompute the forward trajectory for piecewise-linear interpolation.
    let fwd_t_clone = fwd_t.to_vec();
    let fwd_y_clone = fwd_y.to_vec();
    let p_bwd = params.to_vec();
    let ns_bwd = ns;

    let bwd_rhs = move |tau: S, lambda: &[S], dlambda: &mut [S]| {
        let t = tf - tau;

        // Interpolate x(t) from forward trajectory.
        let x_interp = interpolate_state(&fwd_t_clone, &fwd_y_clone, t, ns_bwd);

        // Evaluate dynamics.
        let f0 = eval_dynamics(model, t, &x_interp, &p_bwd, ns_bwd);

        // Compute df/dx.
        let dfdx = compute_dfdx(model, t, &x_interp, &p_bwd, &f0, ns_bwd);

        // (df/dx)^T * lambda
        for i in 0..ns_bwd {
            let mut val = S::ZERO;
            for j in 0..ns_bwd {
                // dfdx[j * ns + i] is df_j / dx_i
                val += dfdx[j * ns_bwd + i] * lambda[j];
            }
            dlambda[i] = val;
        }

        // Add running cost gradient contribution: (dL/dx)^T
        if let Some(rc) = running_cost {
            let dl_dx = grad_x_fd(rc, t, &x_interp, &p_bwd, ns_bwd);
            for i in 0..ns_bwd {
                dlambda[i] += dl_dx[i];
            }
        }
    };

    let tau_end = tf - t0;
    let bwd_problem = OdeProblem::new(bwd_rhs, S::ZERO, tau_end, lambda_tf);

    let bwd_result = DoPri5::solve(
        &bwd_problem,
        S::ZERO,
        tau_end,
        &grad_fd(terminal_cost, x_tf, ns),
        &opts,
    )
    .map_err(|e| OcpError::IntegrationFailed(format!("backward: {e}")))?;

    if !bwd_result.success {
        return Err(OcpError::IntegrationFailed(format!(
            "backward: {}",
            bwd_result.message
        )));
    }

    // Store costate in forward time order.
    let n_bwd = bwd_result.t.len();
    let mut costate_time = Vec::with_capacity(n_bwd);
    let mut costate = Vec::with_capacity(n_bwd * ns);

    for i in (0..n_bwd).rev() {
        let tau_i = bwd_result.t[i];
        costate_time.push(tf - tau_i);
        costate.extend_from_slice(&bwd_result.y[i * ns..(i + 1) * ns]);
    }

    // -----------------------------------------------------------------------
    // Step 4: Compute gradient dJ/dp via trapezoidal integration
    // -----------------------------------------------------------------------
    let mut gradient = vec![S::ZERO; np];

    // Gradient integral: trapezoidal rule over costate trajectory.
    let n_co = costate_time.len();
    for i in 0..n_co.saturating_sub(1) {
        let ti = costate_time[i];
        let ti1 = costate_time[i + 1];
        let dt = ti1 - ti;

        let lambda_i = &costate[i * ns..(i + 1) * ns];
        let lambda_i1 = &costate[(i + 1) * ns..(i + 2) * ns];

        // Interpolate x(t) at both times.
        let xi = interpolate_state(&fwd_result.t, &fwd_result.y, ti, ns);
        let xi1 = interpolate_state(&fwd_result.t, &fwd_result.y, ti1, ns);

        // Compute (df/dp)^T * lambda at both times.
        let f0_i = eval_dynamics(model, ti, &xi, params, ns);
        let dfdp_i = compute_dfdp(model, ti, &xi, params, &f0_i, ns, np);

        let f0_i1 = eval_dynamics(model, ti1, &xi1, params, ns);
        let dfdp_i1 = compute_dfdp(model, ti1, &xi1, params, &f0_i1, ns, np);

        for k in 0..np {
            let mut val_i = S::ZERO;
            let mut val_i1 = S::ZERO;
            for j in 0..ns {
                val_i += dfdp_i[j * np + k] * lambda_i[j];
                val_i1 += dfdp_i1[j * np + k] * lambda_i1[j];
            }

            // Add running cost gradient w.r.t. p.
            if let Some(rc) = running_cost {
                let dl_dp_i = grad_p_fd(rc, ti, &xi, params, np);
                let dl_dp_i1 = grad_p_fd(rc, ti1, &xi1, params, np);
                val_i += dl_dp_i[k];
                val_i1 += dl_dp_i1[k];
            }

            gradient[k] += S::HALF * dt * (val_i + val_i1);
        }
    }

    let _ = tiny; // suppress unused warning

    Ok(AdjointResult {
        gradient,
        objective,
        costate,
        costate_time,
        n_states: ns,
        n_params: np,
    })
}

// ---------------------------------------------------------------------------
// Interpolation helper
// ---------------------------------------------------------------------------

/// Piecewise-linear interpolation of a state trajectory.
fn interpolate_state<S: Scalar>(t_grid: &[S], y_flat: &[S], t: S, ns: usize) -> Vec<S> {
    let n = t_grid.len();
    if n == 0 {
        return vec![S::ZERO; ns];
    }

    // Clamp to grid bounds.
    if t <= t_grid[0] {
        return y_flat[..ns].to_vec();
    }
    if t >= t_grid[n - 1] {
        return y_flat[(n - 1) * ns..n * ns].to_vec();
    }

    // Binary search for interval.
    let mut lo = 0;
    let mut hi = n - 1;
    while hi - lo > 1 {
        let mid = (lo + hi) / 2;
        if t_grid[mid] <= t {
            lo = mid;
        } else {
            hi = mid;
        }
    }

    // Linear interpolation.
    let dt = t_grid[hi] - t_grid[lo];
    let tiny = S::from_f64(1e-30);
    if dt.abs() < tiny {
        return y_flat[lo * ns..(lo + 1) * ns].to_vec();
    }

    let alpha = (t - t_grid[lo]) / dt;
    let mut result = vec![S::ZERO; ns];
    for j in 0..ns {
        result[j] = (S::ONE - alpha) * y_flat[lo * ns + j] + alpha * y_flat[hi * ns + j];
    }
    result
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Exponential decay: dx/dt = -k*x, x(0) = 1, k = 0.5.
    /// Objective: J = x(T)^2.
    /// dJ/dk = 2 * x(T) * dx(T)/dk.
    /// x(T) = exp(-k*T), dx(T)/dk = -T * exp(-k*T).
    /// dJ/dk = 2 * exp(-k*T) * (-T * exp(-k*T)) = -2T * exp(-2kT).
    #[test]
    fn test_exponential_decay_gradient() {
        let k = 0.5_f64;
        let t_final = 2.0;

        let result = adjoint_gradient(
            &|_t: f64, x, dxdt, p| {
                dxdt[0] = -p[0] * x[0];
            },
            &|x| x[0] * x[0],
            None,
            &[1.0],
            &[k],
            (0.0, t_final),
            1e-10,
            1e-12,
        )
        .expect("adjoint failed");

        let analytical = -2.0 * t_final * (-2.0 * k * t_final).exp();
        assert!(
            (result.gradient[0] - analytical).abs() < 1e-3,
            "adjoint grad = {}, analytical = {}, err = {}",
            result.gradient[0],
            analytical,
            (result.gradient[0] - analytical).abs(),
        );

        // Check objective value.
        let x_tf = (-k * t_final).exp();
        let analytical_obj = x_tf * x_tf;
        assert!(
            (result.objective - analytical_obj).abs() < 1e-6,
            "obj = {}, analytical = {}",
            result.objective,
            analytical_obj,
        );
    }

    /// Compare adjoint gradient with forward sensitivity.
    /// dy/dt = -a*y + b, y(0) = 1, a = 1, b = 2.
    /// J = y(T)^2.
    #[test]
    fn test_adjoint_vs_forward_sensitivity() {
        let a = 1.0_f64;
        let b = 2.0_f64;
        let t_final = 2.0;
        let params = [a, b];

        let model = |_t: f64, y: &[f64], dydt: &mut [f64], p: &[f64]| {
            dydt[0] = -p[0] * y[0] + p[1];
        };

        // Adjoint gradient.
        let adj = adjoint_gradient(
            &model,
            &|x| x[0] * x[0],
            None,
            &[1.0],
            &params,
            (0.0, t_final),
            1e-10,
            1e-12,
        )
        .expect("adjoint failed");

        // Central finite-difference gradient for validation.
        let h = 1e-5;
        let opts = SolverOptions::default().rtol(1e-12).atol(1e-14);

        let mut fd_grad = vec![0.0; 2];
        for k in 0..2 {
            let mut p_plus = params;
            let mut p_minus = params;
            p_plus[k] += h;
            p_minus[k] -= h;

            let p_p = p_plus;
            let rhs_plus = move |_t: f64, y: &[f64], dydt: &mut [f64]| {
                dydt[0] = -p_p[0] * y[0] + p_p[1];
            };
            let prob_plus = OdeProblem::new(rhs_plus, 0.0, t_final, vec![1.0]);
            let res_plus = DoPri5::solve(&prob_plus, 0.0, t_final, &[1.0], &opts).unwrap();
            let y_plus = res_plus.y_final().unwrap()[0];

            let p_m = p_minus;
            let rhs_minus = move |_t: f64, y: &[f64], dydt: &mut [f64]| {
                dydt[0] = -p_m[0] * y[0] + p_m[1];
            };
            let prob_minus = OdeProblem::new(rhs_minus, 0.0, t_final, vec![1.0]);
            let res_minus = DoPri5::solve(&prob_minus, 0.0, t_final, &[1.0], &opts).unwrap();
            let y_minus = res_minus.y_final().unwrap()[0];

            fd_grad[k] = (y_plus * y_plus - y_minus * y_minus) / (2.0 * h);
        }

        for k in 0..2 {
            assert!(
                (adj.gradient[k] - fd_grad[k]).abs() < 1e-2,
                "param {k}: adjoint = {}, FD = {}, err = {}",
                adj.gradient[k],
                fd_grad[k],
                (adj.gradient[k] - fd_grad[k]).abs(),
            );
        }
    }

    /// Test with running cost.
    /// dx/dt = -k*x, J = integral x^2 dt + x(T)^2.
    #[test]
    fn test_with_running_cost() {
        let k = 0.5_f64;
        let t_final = 1.0;

        let result = adjoint_gradient(
            &|_t: f64, x, dxdt, p| {
                dxdt[0] = -p[0] * x[0];
            },
            &|x| x[0] * x[0],
            Some(&|_t, x, _p| x[0] * x[0]),
            &[1.0],
            &[k],
            (0.0, t_final),
            1e-8,
            1e-10,
        )
        .expect("adjoint failed");

        // Validate with central FD.
        let h = 1e-5;
        let opts = SolverOptions::default().rtol(1e-12).atol(1e-14);

        let compute_j = |kval: f64| -> f64 {
            let rhs = move |_t: f64, y: &[f64], dydt: &mut [f64]| {
                dydt[0] = -kval * y[0];
            };
            let prob = OdeProblem::new(rhs, 0.0, t_final, vec![1.0]);
            let res = DoPri5::solve(&prob, 0.0, t_final, &[1.0], &opts).unwrap();
            let y_tf = res.y_final().unwrap()[0];
            let mut cost = y_tf * y_tf;
            // Running cost integral via trapezoidal rule.
            for i in 0..res.t.len().saturating_sub(1) {
                let ti = res.t[i];
                let ti1 = res.t[i + 1];
                let yi = res.y[i];
                let yi1 = res.y[i + 1];
                cost += 0.5 * (ti1 - ti) * (yi * yi + yi1 * yi1);
            }
            cost
        };

        let fd_grad = (compute_j(k + h) - compute_j(k - h)) / (2.0 * h);
        assert!(
            (result.gradient[0] - fd_grad).abs() < 0.05,
            "adjoint = {}, FD = {}, err = {}",
            result.gradient[0],
            fd_grad,
            (result.gradient[0] - fd_grad).abs(),
        );
    }

    /// Costate trajectory structure.
    #[test]
    fn test_costate_structure() {
        let result = adjoint_gradient(
            &|_t: f64, x, dxdt, p| {
                dxdt[0] = -p[0] * x[0];
            },
            &|x| x[0] * x[0],
            None,
            &[1.0],
            &[0.5],
            (0.0, 2.0),
            1e-8,
            1e-10,
        )
        .expect("adjoint failed");

        assert!(!result.costate_time.is_empty());
        assert_eq!(
            result.costate.len(),
            result.costate_time.len() * result.n_states,
        );

        // Costate time should be in forward order.
        for i in 0..result.costate_time.len().saturating_sub(1) {
            assert!(
                result.costate_time[i + 1] >= result.costate_time[i],
                "costate_time not monotonic at i={}",
                i,
            );
        }
    }
}

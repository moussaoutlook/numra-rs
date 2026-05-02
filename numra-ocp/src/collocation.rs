//! Direct collocation for optimal control.
//!
//! Transcribes an optimal control problem into a nonlinear programme (NLP)
//! by parameterising states and controls at mesh nodes and enforcing
//! dynamics via collocation defect constraints.
//!
//! Two collocation schemes are supported:
//!
//! - **Trapezoidal**: 2nd-order, simplest. Defect constraint:
//!   `x_{k+1} - x_k - (h/2)*(f_k + f_{k+1}) = 0`
//!
//! - **Hermite-Simpson**: 3rd-order, the industry standard for direct
//!   collocation. Uses a midpoint and cubic Hermite interpolation.
//!
//! Author: Moussa Leblouba
//! Date: 9 February 2026
//! Modified: 2 May 2026

use std::sync::Arc;
use std::time::Instant;

use numra_core::Scalar;
use numra_optim::OptimProblem;

use crate::error::OcpError;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// Dynamics closure: `f(t, x, u, p, dxdt)`.
type DynamicsFn<S> = dyn Fn(S, &[S], &[S], &[S], &mut [S]) + Send + Sync;

/// Terminal cost closure: `phi(x(tf), tf) -> S`.
type TerminalCostFn<S> = dyn Fn(&[S], S) -> S + Send + Sync;

/// Running cost closure: `L(t, x, u) -> S`.
type RunningCostFn<S> = dyn Fn(S, &[S], &[S]) -> S + Send + Sync;

/// Terminal constraint closure: `psi(x(tf)) -> Vec<S>`, each component = 0.
type TerminalConstraintFn<S> = dyn Fn(&[S]) -> Vec<S> + Send + Sync;

/// Path constraint closure: `c(t, x, u) -> Vec<S>`.
type PathConstraintFn<S> = dyn Fn(S, &[S], &[S]) -> Vec<S> + Send + Sync;

/// Path constraints with bounds.
type PathConstraints<S> = (Box<PathConstraintFn<S>>, Vec<(S, S)>);

/// Collocation scheme.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum CollocationScheme {
    /// Trapezoidal rule (2nd order).
    Trapezoidal,
    /// Hermite-Simpson (3rd order) -- the recommended default.
    HermiteSimpson,
}

/// Result of a direct collocation solve.
#[derive(Clone, Debug)]
pub struct CollocationResult<S: Scalar> {
    /// Time grid at mesh nodes.
    pub time: Vec<S>,
    /// States at each mesh node (flat row-major: `states[i * nx + j]`).
    pub states: Vec<S>,
    /// Controls at each mesh node (flat row-major: `controls[i * nu + j]`).
    pub controls: Vec<S>,
    /// Optimal parameters (empty if no free parameters).
    pub parameters: Vec<S>,
    /// Optimal objective value.
    pub objective: S,
    /// Final time (may differ from initial if free).
    pub final_time: S,
    /// Whether the optimizer converged.
    pub converged: bool,
    /// Human-readable status message.
    pub message: String,
    /// Number of optimizer iterations.
    pub iterations: usize,
    /// Wall-clock time in seconds.
    pub wall_time_secs: f64,
    /// Number of states (for interpreting flat arrays).
    pub n_states: usize,
    /// Number of controls (for interpreting flat arrays).
    pub n_controls: usize,
}

// ---------------------------------------------------------------------------
// NLP layout helpers
// ---------------------------------------------------------------------------

/// NLP variable layout information. All offsets and dimensions are computed
/// once and shared (via `Copy`) across all closures.
///
/// This struct is non-generic -- it only stores usize offsets and a few
/// f64 values for the fixed time parameters.
#[derive(Clone, Copy)]
struct NlpLayout {
    nx: usize,
    nu: usize,
    np: usize,
    n_int: usize,
    x_offset: usize,
    u_offset: usize,
    p_offset: usize,
    tf_offset: usize,
    has_free_tf: bool,
    tf_fixed: f64,
    t0: f64,
}

impl NlpLayout {
    fn n_decision(&self) -> usize {
        self.tf_offset + if self.has_free_tf { 1 } else { 0 }
    }

    fn tf<S: Scalar>(&self, z: &[S]) -> S {
        if self.has_free_tf {
            z[self.tf_offset]
        } else {
            S::from_f64(self.tf_fixed)
        }
    }

    fn h<S: Scalar>(&self, z: &[S]) -> S {
        (self.tf(z) - S::from_f64(self.t0)) / S::from_usize(self.n_int)
    }

    fn time_at<S: Scalar>(&self, z: &[S], k: usize) -> S {
        S::from_f64(self.t0) + S::from_usize(k) * self.h(z)
    }

    fn x_range(&self, k: usize) -> (usize, usize) {
        let start = self.x_offset + k * self.nx;
        (start, start + self.nx)
    }

    fn u_range(&self, k: usize) -> (usize, usize) {
        let start = self.u_offset + k * self.nu;
        (start, start + self.nu)
    }

    fn p_range(&self) -> (usize, usize) {
        (self.p_offset, self.p_offset + self.np)
    }
}

// ---------------------------------------------------------------------------
// Builder
// ---------------------------------------------------------------------------

/// Builder for direct collocation optimal control problems.
pub struct CollocationProblem<S: Scalar> {
    n_states: usize,
    n_controls: usize,
    n_params: usize,
    dynamics: Option<Box<DynamicsFn<S>>>,
    x0: Option<Vec<S>>,
    t0: S,
    tf: S,
    free_final_time: Option<(S, S, S)>, // (tf_init, tf_lo, tf_hi)
    n_intervals: usize,
    scheme: CollocationScheme,
    terminal_cost: Option<Box<TerminalCostFn<S>>>,
    running_cost: Option<Box<RunningCostFn<S>>>,
    terminal_constraints: Option<Box<TerminalConstraintFn<S>>>,
    path_constraints: Option<PathConstraints<S>>,
    control_bounds: Vec<Option<(S, S)>>,
    state_bounds: Vec<Option<(S, S)>>,
    params0: Option<Vec<S>>,
    param_bounds: Vec<Option<(S, S)>>,
    max_iter: usize,
}

impl<S: Scalar> CollocationProblem<S> {
    /// Create a new collocation problem.
    pub fn new(n_states: usize, n_controls: usize) -> Self {
        Self {
            n_states,
            n_controls,
            n_params: 0,
            dynamics: None,
            x0: None,
            t0: S::ZERO,
            tf: S::ONE,
            free_final_time: None,
            n_intervals: 20,
            scheme: CollocationScheme::HermiteSimpson,
            terminal_cost: None,
            running_cost: None,
            terminal_constraints: None,
            path_constraints: None,
            control_bounds: vec![None; n_controls],
            state_bounds: vec![None; n_states],
            params0: None,
            param_bounds: Vec::new(),
            max_iter: 500,
        }
    }

    /// Set the dynamics: `f(t, x, u, p, dxdt)`.
    pub fn dynamics<F>(mut self, f: F) -> Self
    where
        F: Fn(S, &[S], &[S], &[S], &mut [S]) + Send + Sync + 'static,
    {
        self.dynamics = Some(Box::new(f));
        self
    }

    /// Set the number of free parameters.
    pub fn n_params(mut self, n: usize) -> Self {
        self.n_params = n;
        self.param_bounds = vec![None; n];
        self
    }

    /// Set the initial state.
    pub fn x0(mut self, x0: &[S]) -> Self {
        self.x0 = Some(x0.to_vec());
        self
    }

    /// Set the time span `[t0, tf]`.
    pub fn time_span(mut self, t0: S, tf: S) -> Self {
        self.t0 = t0;
        self.tf = tf;
        self
    }

    /// Enable free final time.
    pub fn free_final_time(mut self, tf_init: S, tf_bounds: (S, S)) -> Self {
        self.free_final_time = Some((tf_init, tf_bounds.0, tf_bounds.1));
        self
    }

    /// Set the number of collocation intervals.
    pub fn n_intervals(mut self, n: usize) -> Self {
        self.n_intervals = n;
        self
    }

    /// Set the collocation scheme.
    pub fn scheme(mut self, scheme: CollocationScheme) -> Self {
        self.scheme = scheme;
        self
    }

    /// Set the terminal cost `phi(x(tf), tf)`.
    pub fn terminal_cost<F>(mut self, f: F) -> Self
    where
        F: Fn(&[S], S) -> S + Send + Sync + 'static,
    {
        self.terminal_cost = Some(Box::new(f));
        self
    }

    /// Set the running cost `L(t, x, u)`.
    pub fn running_cost<F>(mut self, f: F) -> Self
    where
        F: Fn(S, &[S], &[S]) -> S + Send + Sync + 'static,
    {
        self.running_cost = Some(Box::new(f));
        self
    }

    /// Set terminal equality constraints `psi(x(tf)) = 0`.
    pub fn terminal_constraint<F>(mut self, f: F) -> Self
    where
        F: Fn(&[S]) -> Vec<S> + Send + Sync + 'static,
    {
        self.terminal_constraints = Some(Box::new(f));
        self
    }

    /// Set path constraints with bounds: `lo <= c(t, x, u) <= hi`.
    pub fn path_constraint<F>(mut self, f: F, bounds: &[(S, S)]) -> Self
    where
        F: Fn(S, &[S], &[S]) -> Vec<S> + Send + Sync + 'static,
    {
        self.path_constraints = Some((Box::new(f), bounds.to_vec()));
        self
    }

    /// Set bounds for each control variable (applied at every node).
    pub fn control_bounds(mut self, bounds: &[(S, S)]) -> Self {
        self.control_bounds = bounds.iter().map(|&b| Some(b)).collect();
        self
    }

    /// Set bounds for each state variable (applied at every node).
    pub fn state_bounds(mut self, bounds: &[(S, S)]) -> Self {
        self.state_bounds = bounds.iter().map(|&b| Some(b)).collect();
        self
    }

    /// Set initial guess for free parameters.
    pub fn params0(mut self, p0: &[S]) -> Self {
        self.params0 = Some(p0.to_vec());
        self
    }

    /// Set bounds for a specific parameter.
    pub fn param_bound(mut self, idx: usize, lo: S, hi: S) -> Self {
        if idx < self.param_bounds.len() {
            self.param_bounds[idx] = Some((lo, hi));
        }
        self
    }

    /// Set maximum optimizer iterations.
    pub fn max_iter(mut self, n: usize) -> Self {
        self.max_iter = n;
        self
    }

    // -----------------------------------------------------------------------
    // Solve
    // -----------------------------------------------------------------------

    /// Solve the collocation problem.
    pub fn solve(self) -> Result<CollocationResult<S>, OcpError>
    where
        S: faer::SimpleEntity + faer::Conjugate<Canonical = S> + faer::ComplexField,
    {
        let start = Instant::now();

        // -- Validate -------------------------------------------------------
        let dynamics = self.dynamics.ok_or(OcpError::NoDynamics)?;
        let x0_val = self.x0.ok_or(OcpError::NoInitialState)?;

        if x0_val.len() != self.n_states {
            return Err(OcpError::DimensionMismatch(format!(
                "x0 length {} != n_states {}",
                x0_val.len(),
                self.n_states,
            )));
        }

        if self.terminal_cost.is_none() && self.running_cost.is_none() {
            return Err(OcpError::Other(
                "at least one of terminal_cost or running_cost must be set".into(),
            ));
        }

        let nx = self.n_states;
        let nu = self.n_controls;
        let np = self.n_params;
        let n_int = self.n_intervals;
        let n_nodes = n_int + 1;
        let scheme = self.scheme;

        let has_free_tf = self.free_final_time.is_some();
        let tf_fixed = if let Some((tfi, _, _)) = self.free_final_time {
            tfi.to_f64()
        } else {
            self.tf.to_f64()
        };

        let lay = NlpLayout {
            nx,
            nu,
            np,
            n_int,
            x_offset: 0,
            u_offset: n_nodes * nx,
            p_offset: n_nodes * nx + n_nodes * nu,
            tf_offset: n_nodes * nx + n_nodes * nu + np,
            has_free_tf,
            tf_fixed,
            t0: self.t0.to_f64(),
        };

        let n_decision = lay.n_decision();

        // -- Build initial guess -------------------------------------------
        let mut z0 = vec![S::ZERO; n_decision];

        // Constant state initial guess (x0 everywhere).
        for k in 0..n_nodes {
            let (s, e) = lay.x_range(k);
            z0[s..e].copy_from_slice(&x0_val);
        }

        // Parameters initial guess.
        if let Some(ref p0) = self.params0 {
            let (s, e) = lay.p_range();
            z0[s..e].copy_from_slice(p0);
        }

        // Free final time initial.
        if has_free_tf {
            z0[lay.tf_offset] = S::from_f64(tf_fixed);
        }

        // -- Shared state for closures -------------------------------------
        let dynamics = Arc::new(dynamics);
        let terminal_cost: Option<Arc<Box<TerminalCostFn<S>>>> = self.terminal_cost.map(Arc::new);
        let running_cost: Option<Arc<Box<RunningCostFn<S>>>> = self.running_cost.map(Arc::new);

        // -- Objective function --------------------------------------------
        let dyn_obj = Arc::clone(&dynamics);
        let tc_obj = terminal_cost.clone();
        let rc_obj = running_cost.clone();

        let objective_fn = move |z: &[S]| -> S {
            let tf_val = lay.tf(z);
            let mut cost = S::ZERO;

            if let Some(ref rc) = rc_obj {
                for k in 0..n_int {
                    let tk = lay.time_at(z, k);
                    let tk1 = lay.time_at(z, k + 1);
                    let h = tk1 - tk;
                    let (xs, xe) = lay.x_range(k);
                    let (us, ue) = lay.u_range(k);
                    let (xs1, xe1) = lay.x_range(k + 1);
                    let (us1, ue1) = lay.u_range(k + 1);
                    let lk = rc(tk, &z[xs..xe], &z[us..ue]);
                    let lk1 = rc(tk1, &z[xs1..xe1], &z[us1..ue1]);

                    if scheme == CollocationScheme::HermiteSimpson {
                        let t_mid = S::HALF * (tk + tk1);
                        let mut x_mid = vec![S::ZERO; nx];
                        let mut u_mid = vec![S::ZERO; nu];
                        for j in 0..nx {
                            x_mid[j] = S::HALF * (z[xs + j] + z[xs1 + j]);
                        }
                        for j in 0..nu {
                            u_mid[j] = S::HALF * (z[us + j] + z[us1 + j]);
                        }
                        let (ps, pe) = lay.p_range();
                        let mut fk = vec![S::ZERO; nx];
                        let mut fk1 = vec![S::ZERO; nx];
                        dyn_obj(tk, &z[xs..xe], &z[us..ue], &z[ps..pe], &mut fk);
                        dyn_obj(tk1, &z[xs1..xe1], &z[us1..ue1], &z[ps..pe], &mut fk1);
                        let eighth_h = h / S::from_f64(8.0);
                        for j in 0..nx {
                            x_mid[j] += eighth_h * (fk[j] - fk1[j]);
                        }
                        let l_mid = rc(t_mid, &x_mid, &u_mid);
                        let sixth_h = h / S::from_f64(6.0);
                        cost += sixth_h * (lk + S::from_f64(4.0) * l_mid + lk1);
                    } else {
                        cost += S::HALF * h * (lk + lk1);
                    }
                }
            }

            if let Some(ref tc) = tc_obj {
                let (xs, xe) = lay.x_range(n_int);
                cost += tc(&z[xs..xe], tf_val);
            }

            cost
        };

        // -- Build OptimProblem --------------------------------------------
        let mut prob = OptimProblem::new(n_decision)
            .x0(&z0)
            .objective(objective_fn)
            .max_iter(self.max_iter);

        // Initial state equality constraints: x_0 = x0_val.
        for j in 0..nx {
            let x0_j = x0_val[j];
            let x_off = lay.x_offset;
            prob = prob.constraint_eq(move |z: &[S]| -> S { z[x_off + j] - x0_j });
        }

        // -- Defect constraints --------------------------------------------
        match scheme {
            CollocationScheme::Trapezoidal => {
                for k in 0..n_int {
                    for j in 0..nx {
                        let dyn_c = Arc::clone(&dynamics);
                        prob = prob.constraint_eq(move |z: &[S]| -> S {
                            let h = lay.h(z);
                            let tk = lay.time_at(z, k);
                            let tk1 = lay.time_at(z, k + 1);
                            let (xs, xe) = lay.x_range(k);
                            let (us, ue) = lay.u_range(k);
                            let (xs1, xe1) = lay.x_range(k + 1);
                            let (us1, ue1) = lay.u_range(k + 1);
                            let (ps, pe) = lay.p_range();

                            let mut fk = vec![S::ZERO; nx];
                            let mut fk1 = vec![S::ZERO; nx];
                            dyn_c(tk, &z[xs..xe], &z[us..ue], &z[ps..pe], &mut fk);
                            dyn_c(tk1, &z[xs1..xe1], &z[us1..ue1], &z[ps..pe], &mut fk1);

                            z[xs1 + j] - z[xs + j] - S::HALF * h * (fk[j] + fk1[j])
                        });
                    }
                }
            }
            CollocationScheme::HermiteSimpson => {
                for k in 0..n_int {
                    for j in 0..nx {
                        let dyn_c = Arc::clone(&dynamics);
                        prob = prob.constraint_eq(move |z: &[S]| -> S {
                            let h = lay.h(z);
                            let tk = lay.time_at(z, k);
                            let tk1 = lay.time_at(z, k + 1);
                            let t_mid = S::HALF * (tk + tk1);
                            let (xs, xe) = lay.x_range(k);
                            let (us, ue) = lay.u_range(k);
                            let (xs1, xe1) = lay.x_range(k + 1);
                            let (us1, ue1) = lay.u_range(k + 1);
                            let (ps, pe) = lay.p_range();

                            let mut fk = vec![S::ZERO; nx];
                            let mut fk1 = vec![S::ZERO; nx];
                            dyn_c(tk, &z[xs..xe], &z[us..ue], &z[ps..pe], &mut fk);
                            dyn_c(tk1, &z[xs1..xe1], &z[us1..ue1], &z[ps..pe], &mut fk1);

                            // Hermite midpoint state.
                            let mut x_mid = vec![S::ZERO; nx];
                            let mut u_mid = vec![S::ZERO; nu];
                            let eighth_h = h / S::from_f64(8.0);
                            for i in 0..nx {
                                x_mid[i] = S::HALF * (z[xs + i] + z[xs1 + i])
                                    + eighth_h * (fk[i] - fk1[i]);
                            }
                            for i in 0..nu {
                                u_mid[i] = S::HALF * (z[us + i] + z[us1 + i]);
                            }

                            let mut f_mid = vec![S::ZERO; nx];
                            dyn_c(t_mid, &x_mid, &u_mid, &z[ps..pe], &mut f_mid);

                            // Simpson defect.
                            let sixth_h = h / S::from_f64(6.0);
                            z[xs1 + j]
                                - z[xs + j]
                                - sixth_h * (fk[j] + S::from_f64(4.0) * f_mid[j] + fk1[j])
                        });
                    }
                }
            }
        }

        // -- Terminal constraints ------------------------------------------
        if let Some(tc_fn) = self.terminal_constraints {
            let tc_fn = Arc::new(tc_fn);
            let dummy = vec![S::ZERO; nx];
            let n_tc = tc_fn(&dummy).len();

            for ci in 0..n_tc {
                let tc_c = Arc::clone(&tc_fn);
                prob = prob.constraint_eq(move |z: &[S]| -> S {
                    let (xs, xe) = lay.x_range(n_int);
                    tc_c(&z[xs..xe])[ci]
                });
            }
        }

        // -- Path constraints (inequality, enforced at each node) ----------
        if let Some((pc_fn, pc_bounds)) = self.path_constraints {
            let pc_fn = Arc::new(pc_fn);
            let n_pc = pc_bounds.len();

            for k in 0..n_nodes {
                #[allow(clippy::needless_range_loop)]
                for ci in 0..n_pc {
                    let (lo, hi) = pc_bounds[ci];
                    let pc_lo = Arc::clone(&pc_fn);
                    let pc_hi = Arc::clone(&pc_fn);

                    // c(t, x, u) - lo >= 0
                    prob = prob.constraint_ineq(move |z: &[S]| -> S {
                        let tk = lay.time_at(z, k);
                        let (xs, xe) = lay.x_range(k);
                        let (us, ue) = lay.u_range(k);
                        pc_lo(tk, &z[xs..xe], &z[us..ue])[ci] - lo
                    });

                    // hi - c(t, x, u) >= 0
                    prob = prob.constraint_ineq(move |z: &[S]| -> S {
                        let tk = lay.time_at(z, k);
                        let (xs, xe) = lay.x_range(k);
                        let (us, ue) = lay.u_range(k);
                        hi - pc_hi(tk, &z[xs..xe], &z[us..ue])[ci]
                    });
                }
            }
        }

        // -- Variable bounds -----------------------------------------------
        for k in 0..n_nodes {
            for j in 0..nx {
                if let Some(&Some((lo, hi))) = self.state_bounds.get(j) {
                    let (s, _) = lay.x_range(k);
                    prob = prob.bounds(s + j, (lo, hi));
                }
            }
            for j in 0..nu {
                if let Some(&Some((lo, hi))) = self.control_bounds.get(j) {
                    let (s, _) = lay.u_range(k);
                    prob = prob.bounds(s + j, (lo, hi));
                }
            }
        }

        for j in 0..np {
            if let Some(&Some((lo, hi))) = self.param_bounds.get(j) {
                prob = prob.bounds(lay.p_offset + j, (lo, hi));
            }
        }

        if let Some((_, tf_lo, tf_hi)) = self.free_final_time {
            prob = prob.bounds(lay.tf_offset, (tf_lo, tf_hi));
        }

        // -- Solve ----------------------------------------------------------
        let optim_result = prob.solve().map_err(OcpError::OptimFailed)?;
        let z_opt = &optim_result.x;

        // -- Extract results ------------------------------------------------
        let tf_opt = lay.tf(z_opt);
        let mut time = Vec::with_capacity(n_nodes);
        let mut states = Vec::with_capacity(n_nodes * nx);
        let mut controls = Vec::with_capacity(n_nodes * nu);

        for k in 0..n_nodes {
            time.push(lay.time_at(z_opt, k));
            let (xs, xe) = lay.x_range(k);
            states.extend_from_slice(&z_opt[xs..xe]);
            let (us, ue) = lay.u_range(k);
            controls.extend_from_slice(&z_opt[us..ue]);
        }

        let parameters = if np > 0 {
            let (ps, pe) = lay.p_range();
            z_opt[ps..pe].to_vec()
        } else {
            Vec::new()
        };

        Ok(CollocationResult {
            time,
            states,
            controls,
            parameters,
            objective: optim_result.f,
            final_time: tf_opt,
            converged: optim_result.converged,
            message: optim_result.message.clone(),
            iterations: optim_result.iterations,
            wall_time_secs: start.elapsed().as_secs_f64(),
            n_states: nx,
            n_controls: nu,
        })
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Double integrator: dx/dt = v, dv/dt = u.
    /// Minimize integral u^2 dt, x(0)=0, v(0)=0, x(1)=1, v(1)=0.
    /// Analytical cost = 12.
    #[test]
    fn test_double_integrator_hermite_simpson() {
        let result = CollocationProblem::new(2, 1)
            .dynamics(|_t, x, u, _p, dxdt| {
                dxdt[0] = x[1];
                dxdt[1] = u[0];
            })
            .x0(&[0.0, 0.0])
            .time_span(0.0, 1.0)
            .n_intervals(20)
            .scheme(CollocationScheme::HermiteSimpson)
            .running_cost(|_t, _x, u| u[0] * u[0])
            .terminal_constraint(|x| vec![x[0] - 1.0, x[1]])
            .max_iter(500)
            .solve()
            .expect("collocation solve failed");

        assert!(result.converged, "should converge, got: {}", result.message);

        let n = result.time.len();
        let xf = result.states[(n - 1) * 2];
        let vf = result.states[(n - 1) * 2 + 1];
        assert!((xf - 1.0).abs() < 0.1, "x(T) = {xf}, expected ~1.0");
        assert!(vf.abs() < 0.1, "v(T) = {vf}, expected ~0.0");
        assert!(
            (result.objective - 12.0).abs() < 2.0,
            "cost = {}, expected ~12.0",
            result.objective,
        );
    }

    /// Same problem with trapezoidal scheme.
    #[test]
    fn test_double_integrator_trapezoidal() {
        let result = CollocationProblem::new(2, 1)
            .dynamics(|_t, x, u, _p, dxdt| {
                dxdt[0] = x[1];
                dxdt[1] = u[0];
            })
            .x0(&[0.0, 0.0])
            .time_span(0.0, 1.0)
            .n_intervals(20)
            .scheme(CollocationScheme::Trapezoidal)
            .running_cost(|_t, _x, u| u[0] * u[0])
            .terminal_constraint(|x| vec![x[0] - 1.0, x[1]])
            .max_iter(500)
            .solve()
            .expect("collocation solve failed");

        assert!(result.converged, "should converge, got: {}", result.message);

        let n = result.time.len();
        let xf = result.states[(n - 1) * 2];
        assert!((xf - 1.0).abs() < 0.1, "x(T) = {xf}, expected ~1.0");
    }

    /// Minimum energy: dx/dt = u, x(0) = 0, terminal cost: 100*(x(1)-1)^2.
    #[test]
    fn test_minimum_energy_collocation() {
        let result = CollocationProblem::new(1, 1)
            .dynamics(|_t, _x, u, _p, dxdt| {
                dxdt[0] = u[0];
            })
            .x0(&[0.0])
            .time_span(0.0, 1.0)
            .n_intervals(10)
            .scheme(CollocationScheme::HermiteSimpson)
            .terminal_cost(|x, _tf| 100.0 * (x[0] - 1.0).powi(2))
            .running_cost(|_t, _x, u| u[0] * u[0])
            .max_iter(300)
            .solve()
            .expect("collocation solve failed");

        let n = result.time.len();
        let xf = result.states[n - 1];
        assert!((xf - 1.0).abs() < 0.3, "x(T) = {xf}, expected ~1.0");
    }

    /// Control-bounded: dx/dt = u, -1 <= u <= 1, target x(2) = 3.
    /// With T=2 and |u|<=1, best is u=1 => x(2)=2.
    #[test]
    fn test_control_bounded() {
        let result = CollocationProblem::new(1, 1)
            .dynamics(|_t, _x, u, _p, dxdt| {
                dxdt[0] = u[0];
            })
            .x0(&[0.0])
            .time_span(0.0, 2.0)
            .n_intervals(10)
            .scheme(CollocationScheme::HermiteSimpson)
            .terminal_cost(|x, _tf| (x[0] - 3.0).powi(2))
            .control_bounds(&[(-1.0, 1.0)])
            .max_iter(300)
            .solve()
            .expect("collocation solve failed");

        let n = result.time.len();
        let xf = result.states[n - 1];
        assert!((xf - 2.0).abs() < 0.3, "x(T) = {xf}, expected ~2.0");
    }

    /// Result structure: time grid, states, controls dimensions.
    #[test]
    fn test_result_structure() {
        let result = CollocationProblem::new(2, 1)
            .dynamics(|_t, x, u, _p, dxdt| {
                dxdt[0] = x[1];
                dxdt[1] = u[0];
            })
            .x0(&[0.0, 0.0])
            .time_span(0.0, 1.0)
            .n_intervals(5)
            .scheme(CollocationScheme::HermiteSimpson)
            .terminal_cost(|x, _tf| x[0].powi(2))
            .max_iter(100)
            .solve()
            .expect("collocation solve failed");

        assert_eq!(result.time.len(), 6);
        assert_eq!(result.states.len(), 6 * 2);
        assert_eq!(result.controls.len(), 6 * 1);
        assert_eq!(result.n_states, 2);
        assert_eq!(result.n_controls, 1);

        for k in 0..5 {
            assert!(result.time[k + 1] > result.time[k]);
        }

        assert!((result.time[0] - 0.0).abs() < 1e-12);
        assert!((result.time[5] - 1.0).abs() < 1e-12);
    }
}

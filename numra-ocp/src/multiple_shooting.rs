//! Multiple shooting optimal control.
//!
//! Multiple shooting divides the time horizon into segments and treats the
//! state at each segment boundary as a decision variable. Each segment is
//! integrated independently and continuity is enforced via equality
//! constraints. This provides much better numerical conditioning than single
//! shooting for long time horizons.
//!
//! # Decision variables
//!
//! `[x_0, x_1, ..., x_{N-1}, u_0, u_1, ..., u_{N-1}]`
//!
//! where `x_k` is the state at the start of segment `k` and `u_k` is the
//! piecewise-constant control on segment `k`. `x_0` is fixed via equality
//! constraints to the given initial condition.
//!
//! # Continuity constraints
//!
//! For each segment boundary `k = 0, ..., N-2`:
//!
//! `phi_k(x_k, u_k) - x_{k+1} = 0`
//!
//! where `phi_k` integrates the ODE from `t_k` to `t_{k+1}` starting at
//! `x_k` with control `u_k`.
//!
//! Author: Moussa Leblouba
//! Date: 9 February 2026
//! Modified: 2 May 2026

use std::sync::Arc;
use std::time::Instant;

use numra_core::Scalar;
use numra_ode::{DoPri5, OdeProblem, Solver, SolverOptions};
use numra_optim::OptimProblem;

use crate::error::OcpError;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// Dynamics closure: `f(t, x, dxdt, u)`.
type DynamicsFn<S> = dyn Fn(S, &[S], &mut [S], &[S]) + Send + Sync;

/// Terminal cost closure: `phi(x(T)) -> S`.
type TerminalCostFn<S> = dyn Fn(&[S]) -> S + Send + Sync;

/// Running cost closure: `L(t, x, u) -> S`.
type RunningCostFn<S> = dyn Fn(S, &[S], &[S]) -> S + Send + Sync;

/// Terminal constraint closure: `h(x(T)) -> Vec<S>`, each component = 0.
type TerminalConstraintFn<S> = dyn Fn(&[S]) -> Vec<S> + Send + Sync;

/// Result of a multiple shooting optimal control solve.
#[derive(Clone, Debug)]
pub struct MultipleShootingResult<S: Scalar> {
    /// Optimal control vector (flat: `n_controls * n_segments`).
    pub controls: Vec<S>,
    /// Final state `x(T)` at the optimum.
    pub final_state: Vec<S>,
    /// Optimal objective value.
    pub objective: S,
    /// Whether the optimizer converged.
    pub converged: bool,
    /// Human-readable status message.
    pub message: String,
    /// Number of optimizer iterations.
    pub iterations: usize,
    /// Wall-clock time in seconds.
    pub wall_time_secs: f64,
    /// Time grid of the reconstructed trajectory.
    pub t_trajectory: Vec<S>,
    /// State trajectory (flat row-major: `y[i * n_states + j]`).
    pub y_trajectory: Vec<S>,
    /// Number of states.
    pub n_states: usize,
}

// ---------------------------------------------------------------------------
// NLP layout
// ---------------------------------------------------------------------------

/// NLP variable layout for multiple shooting.
#[derive(Clone, Copy)]
struct MsLayout<S: Scalar> {
    nx: usize,
    nu: usize,
    n_seg: usize,
    x_offset: usize, // always 0
    u_offset: usize, // n_seg * nx
    t0: S,
    dt: S,
    ode_rtol: S,
    ode_atol: S,
}

impl<S: Scalar> MsLayout<S> {
    fn n_decision(&self) -> usize {
        self.u_offset + self.n_seg * self.nu
    }

    fn x_start(&self, k: usize) -> usize {
        self.x_offset + k * self.nx
    }

    fn u_start(&self, k: usize) -> usize {
        self.u_offset + k * self.nu
    }

    fn t_start(&self, k: usize) -> S {
        self.t0 + S::from_usize(k) * self.dt
    }

    fn t_end(&self, k: usize) -> S {
        self.t0 + S::from_usize(k + 1) * self.dt
    }
}

/// Integrate a single segment and return the final state.
fn integrate_segment<S: Scalar>(
    dynamics: &Arc<Box<DynamicsFn<S>>>,
    x_k: &[S],
    u_k: &[S],
    t_start: S,
    t_end: S,
    rtol: S,
    atol: S,
) -> Result<Vec<S>, String> {
    let dyn_ref = Arc::clone(dynamics);
    let u_seg = u_k.to_vec();
    let rhs = move |t: S, y: &[S], dydt: &mut [S]| {
        dyn_ref(t, y, dydt, &u_seg);
    };
    let opts = SolverOptions::default().rtol(rtol).atol(atol);
    let problem = OdeProblem::new(rhs, t_start, t_end, x_k.to_vec());
    let result = DoPri5::solve(&problem, t_start, t_end, x_k, &opts).map_err(|e| e.to_string())?;
    if !result.success {
        return Err(result.message.clone());
    }
    result.y_final().ok_or_else(|| "empty result".into())
}

/// Integrate a single segment returning the full trajectory and running cost.
#[allow(clippy::too_many_arguments)]
fn integrate_segment_full<S: Scalar>(
    dynamics: &Arc<Box<DynamicsFn<S>>>,
    x_k: &[S],
    u_k: &[S],
    t_start: S,
    t_end: S,
    rtol: S,
    atol: S,
    running_cost: Option<&RunningCostFn<S>>,
    nx: usize,
) -> Result<(Vec<S>, Vec<S>, S), String> {
    let dyn_ref = Arc::clone(dynamics);
    let u_seg = u_k.to_vec();
    let u_for_cost = u_k.to_vec();
    let rhs = move |t: S, y: &[S], dydt: &mut [S]| {
        dyn_ref(t, y, dydt, &u_seg);
    };
    let opts = SolverOptions::default().rtol(rtol).atol(atol);
    let problem = OdeProblem::new(rhs, t_start, t_end, x_k.to_vec());
    let result = DoPri5::solve(&problem, t_start, t_end, x_k, &opts).map_err(|e| e.to_string())?;
    if !result.success {
        return Err(result.message);
    }

    let mut cost = S::ZERO;
    if let Some(rc) = running_cost {
        let n_pts = result.t.len();
        for i in 0..n_pts.saturating_sub(1) {
            let ti = result.t[i];
            let ti1 = result.t[i + 1];
            let yi = &result.y[i * nx..(i + 1) * nx];
            let yi1 = &result.y[(i + 1) * nx..(i + 2) * nx];
            let li = rc(ti, yi, &u_for_cost);
            let li1 = rc(ti1, yi1, &u_for_cost);
            cost += S::HALF * (ti1 - ti) * (li + li1);
        }
    }

    Ok((result.t, result.y, cost))
}

// ---------------------------------------------------------------------------
// Builder
// ---------------------------------------------------------------------------

/// Builder for multiple shooting optimal control problems.
pub struct MultipleShootingProblem<S: Scalar> {
    n_states: usize,
    n_controls: usize,
    dynamics: Option<Box<DynamicsFn<S>>>,
    y0: Option<Vec<S>>,
    t0: S,
    tf: S,
    n_segments: usize,
    control_bounds: Vec<Option<(S, S)>>,
    terminal_cost: Option<Box<TerminalCostFn<S>>>,
    running_cost: Option<Box<RunningCostFn<S>>>,
    terminal_constraints: Option<Box<TerminalConstraintFn<S>>>,
    ode_rtol: S,
    ode_atol: S,
    max_iter: usize,
}

impl<S: Scalar> MultipleShootingProblem<S> {
    /// Create a new multiple shooting problem.
    pub fn new(n_states: usize, n_controls: usize) -> Self {
        Self {
            n_states,
            n_controls,
            dynamics: None,
            y0: None,
            t0: S::ZERO,
            tf: S::ONE,
            n_segments: 10,
            control_bounds: vec![None; n_controls],
            terminal_cost: None,
            running_cost: None,
            terminal_constraints: None,
            ode_rtol: S::from_f64(1e-8),
            ode_atol: S::from_f64(1e-10),
            max_iter: 200,
        }
    }

    /// Set the controlled ODE right-hand side: `f(t, x, dxdt, u)`.
    pub fn dynamics<F>(mut self, f: F) -> Self
    where
        F: Fn(S, &[S], &mut [S], &[S]) + Send + Sync + 'static,
    {
        self.dynamics = Some(Box::new(f));
        self
    }

    /// Set the initial state `y(t0)`.
    pub fn initial_state(mut self, y0: Vec<S>) -> Self {
        self.y0 = Some(y0);
        self
    }

    /// Set the time interval `[t0, tf]`.
    pub fn time_span(mut self, t0: S, tf: S) -> Self {
        self.t0 = t0;
        self.tf = tf;
        self
    }

    /// Set the number of shooting segments.
    pub fn n_segments(mut self, n: usize) -> Self {
        self.n_segments = n;
        self
    }

    /// Set bounds for each control variable (applied to every segment).
    pub fn control_bounds(mut self, bounds: Vec<Option<(S, S)>>) -> Self {
        self.control_bounds = bounds;
        self
    }

    /// Set the terminal cost `phi(y(T))`.
    pub fn terminal_cost<F>(mut self, f: F) -> Self
    where
        F: Fn(&[S]) -> S + Send + Sync + 'static,
    {
        self.terminal_cost = Some(Box::new(f));
        self
    }

    /// Set the running cost `L(t, y, u)`.
    pub fn running_cost<F>(mut self, f: F) -> Self
    where
        F: Fn(S, &[S], &[S]) -> S + Send + Sync + 'static,
    {
        self.running_cost = Some(Box::new(f));
        self
    }

    /// Set terminal equality constraints `h(y(T)) = 0`.
    pub fn terminal_constraint<F>(mut self, f: F) -> Self
    where
        F: Fn(&[S]) -> Vec<S> + Send + Sync + 'static,
    {
        self.terminal_constraints = Some(Box::new(f));
        self
    }

    /// Set ODE solver tolerances.
    pub fn ode_tolerances(mut self, rtol: S, atol: S) -> Self {
        self.ode_rtol = rtol;
        self.ode_atol = atol;
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

    /// Execute the multiple shooting optimal control solve.
    pub fn solve(self) -> Result<MultipleShootingResult<S>, OcpError>
    where
        S: faer::SimpleEntity + faer::Conjugate<Canonical = S> + faer::ComplexField,
    {
        let start = Instant::now();

        // -- Validate -------------------------------------------------------
        let dynamics = self.dynamics.ok_or(OcpError::NoDynamics)?;
        let y0 = self.y0.ok_or(OcpError::NoInitialState)?;

        if y0.len() != self.n_states {
            return Err(OcpError::DimensionMismatch(format!(
                "y0 length {} != n_states {}",
                y0.len(),
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
        let n_seg = self.n_segments;
        let dt = (self.tf - self.t0) / S::from_usize(n_seg);

        let lay = MsLayout {
            nx,
            nu,
            n_seg,
            x_offset: 0,
            u_offset: n_seg * nx,
            t0: self.t0,
            dt,
            ode_rtol: self.ode_rtol,
            ode_atol: self.ode_atol,
        };

        let n_decision = lay.n_decision();

        // -- Build initial guess -------------------------------------------
        // Propagate y0 forward to initialize segment states.
        let dynamics = Arc::new(dynamics);
        let mut z0 = vec![S::ZERO; n_decision];

        // Set x_0 = y0.
        z0[..nx].copy_from_slice(&y0);

        // Propagate forward with u = 0 to get initial state guesses.
        let mut x_cur = y0.clone();
        for k in 1..n_seg {
            match integrate_segment(
                &dynamics,
                &x_cur,
                &vec![S::ZERO; nu],
                lay.t_start(k - 1),
                lay.t_end(k - 1),
                lay.ode_rtol,
                lay.ode_atol,
            ) {
                Ok(x_next) => {
                    z0[lay.x_start(k)..lay.x_start(k) + nx].copy_from_slice(&x_next);
                    x_cur = x_next;
                }
                Err(_) => {
                    // If integration fails, just use y0 as guess.
                    z0[lay.x_start(k)..lay.x_start(k) + nx].copy_from_slice(&y0);
                }
            }
        }

        // Controls: zero initial guess (already 0).

        // -- Shared state ---------------------------------------------------
        let terminal_cost: Option<Arc<Box<TerminalCostFn<S>>>> = self.terminal_cost.map(Arc::new);
        let running_cost: Option<Arc<Box<RunningCostFn<S>>>> = self.running_cost.map(Arc::new);

        // -- Objective function --------------------------------------------
        let dyn_obj = Arc::clone(&dynamics);
        let tc_obj = terminal_cost.clone();
        let rc_obj = running_cost.clone();

        let big = S::from_f64(1e20);
        let objective_fn = move |z: &[S]| -> S {
            let mut total_cost = S::ZERO;
            let mut x_final = vec![S::ZERO; nx];

            for k in 0..n_seg {
                let x_k = &z[lay.x_start(k)..lay.x_start(k) + nx];
                let u_k = &z[lay.u_start(k)..lay.u_start(k) + nu];
                let rc_ref = rc_obj.as_ref().map(|b| &***b as &RunningCostFn<S>);

                match integrate_segment_full(
                    &dyn_obj,
                    x_k,
                    u_k,
                    lay.t_start(k),
                    lay.t_end(k),
                    lay.ode_rtol,
                    lay.ode_atol,
                    rc_ref,
                    nx,
                ) {
                    Ok((_t, y, seg_cost)) => {
                        total_cost += seg_cost;
                        let n_pts = _t.len();
                        if n_pts > 0 {
                            x_final.copy_from_slice(&y[(n_pts - 1) * nx..n_pts * nx]);
                        }
                    }
                    Err(_) => return big,
                }
            }

            if let Some(ref tc) = tc_obj {
                total_cost += tc(&x_final);
            }

            total_cost
        };

        // -- Build OptimProblem --------------------------------------------
        let mut prob = OptimProblem::new(n_decision)
            .x0(&z0)
            .objective(objective_fn)
            .max_iter(self.max_iter);

        // Initial state constraints: x_0 = y0.
        for j in 0..nx {
            let y0_j = y0[j];
            prob = prob.constraint_eq(move |z: &[S]| -> S { z[j] - y0_j });
        }

        // -- Continuity constraints ----------------------------------------
        // For k = 0, ..., N-2: phi_k(x_k, u_k) - x_{k+1} = 0
        let big_c = S::from_f64(1e10);
        for k in 0..(n_seg - 1) {
            for j in 0..nx {
                let dyn_c = Arc::clone(&dynamics);
                prob = prob.constraint_eq(move |z: &[S]| -> S {
                    let x_k = &z[lay.x_start(k)..lay.x_start(k) + nx];
                    let u_k = &z[lay.u_start(k)..lay.u_start(k) + nu];

                    match integrate_segment(
                        &dyn_c,
                        x_k,
                        u_k,
                        lay.t_start(k),
                        lay.t_end(k),
                        lay.ode_rtol,
                        lay.ode_atol,
                    ) {
                        Ok(x_end) => x_end[j] - z[lay.x_start(k + 1) + j],
                        Err(_) => big_c,
                    }
                });
            }
        }

        // -- Terminal constraints ------------------------------------------
        if let Some(tc_fn) = self.terminal_constraints {
            let tc_fn = Arc::new(tc_fn);
            let dummy = vec![S::ZERO; nx];
            let n_tc = tc_fn(&dummy).len();

            for ci in 0..n_tc {
                let tc_c = Arc::clone(&tc_fn);
                let dyn_c = Arc::clone(&dynamics);

                prob = prob.constraint_eq(move |z: &[S]| -> S {
                    // Integrate last segment to get final state.
                    let k = n_seg - 1;
                    let x_k = &z[lay.x_start(k)..lay.x_start(k) + nx];
                    let u_k = &z[lay.u_start(k)..lay.u_start(k) + nu];

                    match integrate_segment(
                        &dyn_c,
                        x_k,
                        u_k,
                        lay.t_start(k),
                        lay.t_end(k),
                        lay.ode_rtol,
                        lay.ode_atol,
                    ) {
                        Ok(x_final) => tc_c(&x_final)[ci],
                        Err(_) => big_c,
                    }
                });
            }
        }

        // -- Control bounds ------------------------------------------------
        for seg in 0..n_seg {
            for ctrl in 0..nu {
                if let Some(&Some((lo, hi))) = self.control_bounds.get(ctrl) {
                    prob = prob.bounds(lay.u_start(seg) + ctrl, (lo, hi));
                }
            }
        }

        // -- Solve ----------------------------------------------------------
        let optim_result = prob.solve().map_err(OcpError::OptimFailed)?;
        let z_opt = &optim_result.x;

        // -- Reconstruct trajectory ----------------------------------------
        let mut traj_t: Vec<S> = Vec::new();
        let mut traj_y: Vec<S> = Vec::new();
        let mut x_final = y0.clone();
        let mut total_obj = S::ZERO;

        let rc_ref = running_cost.as_ref().map(|b| &***b as &RunningCostFn<S>);

        for k in 0..n_seg {
            let x_k = &z_opt[lay.x_start(k)..lay.x_start(k) + nx];
            let u_k = &z_opt[lay.u_start(k)..lay.u_start(k) + nu];

            let (seg_t, seg_y, seg_cost) = integrate_segment_full(
                &dynamics,
                x_k,
                u_k,
                lay.t_start(k),
                lay.t_end(k),
                lay.ode_rtol,
                lay.ode_atol,
                rc_ref,
                nx,
            )
            .map_err(OcpError::IntegrationFailed)?;

            total_obj += seg_cost;

            // Skip first point for segments > 0 to avoid duplicate boundary.
            let skip = if k == 0 { 0 } else { 1 };
            for i in skip..seg_t.len() {
                traj_t.push(seg_t[i]);
                traj_y.extend_from_slice(&seg_y[i * nx..(i + 1) * nx]);
            }

            let n_pts = seg_t.len();
            if n_pts > 0 {
                x_final = seg_y[(n_pts - 1) * nx..n_pts * nx].to_vec();
            }
        }

        if let Some(ref tc) = terminal_cost {
            total_obj += tc(&x_final);
        }

        // Extract flat controls from decision vector.
        let controls = z_opt[lay.u_offset..].to_vec();

        Ok(MultipleShootingResult {
            controls,
            final_state: x_final,
            objective: total_obj,
            converged: optim_result.converged,
            message: optim_result.message.clone(),
            iterations: optim_result.iterations,
            wall_time_secs: start.elapsed().as_secs_f64(),
            t_trajectory: traj_t,
            y_trajectory: traj_y,
            n_states: nx,
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
    /// Terminal cost: 100*((x-1)^2 + v^2) + running cost: 0.01*u^2.
    #[test]
    fn test_double_integrator() {
        let result = MultipleShootingProblem::new(2, 1)
            .dynamics(|_t, y, dydt, u| {
                dydt[0] = y[1];
                dydt[1] = u[0];
            })
            .initial_state(vec![0.0, 0.0])
            .time_span(0.0, 2.0)
            .n_segments(10)
            .terminal_cost(|y| 100.0 * ((y[0] - 1.0).powi(2) + y[1].powi(2)))
            .running_cost(|_t, _y, u| 0.01 * u[0].powi(2))
            .max_iter(200)
            .solve()
            .expect("multiple shooting solve failed");

        let x_final = result.final_state[0];
        assert!(
            (x_final - 1.0).abs() < 0.3,
            "x(T) = {x_final}, expected within 0.3 of 1.0"
        );
    }

    /// Minimum energy: dx/dt = u, terminal cost: 1000*(x-1)^2, running: u^2.
    #[test]
    fn test_minimum_energy() {
        let result = MultipleShootingProblem::new(1, 1)
            .dynamics(|_t, _y, dydt, u| {
                dydt[0] = u[0];
            })
            .initial_state(vec![0.0])
            .time_span(0.0, 1.0)
            .n_segments(10)
            .terminal_cost(|y| 1000.0 * (y[0] - 1.0).powi(2))
            .running_cost(|_t, _y, u| u[0].powi(2))
            .max_iter(200)
            .solve()
            .expect("multiple shooting solve failed");

        let x_final = result.final_state[0];
        assert!(
            (x_final - 1.0).abs() < 0.3,
            "x(T) = {x_final}, expected within 0.3 of 1.0"
        );
    }

    /// Compare multiple shooting with single shooting on a simple problem.
    /// Multiple shooting should produce similar results.
    #[test]
    fn test_vs_single_shooting() {
        let ms_result = MultipleShootingProblem::new(1, 1)
            .dynamics(|_t, _y, dydt, u| {
                dydt[0] = u[0];
            })
            .initial_state(vec![0.0])
            .time_span(0.0, 1.0)
            .n_segments(5)
            .terminal_cost(|y| (y[0] - 2.0).powi(2))
            .max_iter(100)
            .solve()
            .expect("multiple shooting solve failed");

        let ss_result = crate::ShootingProblem::new(1, 1)
            .dynamics(|_t, _y, dydt, u| {
                dydt[0] = u[0];
            })
            .initial_state(vec![0.0])
            .time_span(0.0, 1.0)
            .n_segments(5)
            .terminal_cost(|y| (y[0] - 2.0).powi(2))
            .max_iter(100)
            .solve()
            .expect("single shooting solve failed");

        // Both should reach x(T) ~ 2.
        assert!(
            (ms_result.final_state[0] - ss_result.final_state[0]).abs() < 0.5,
            "MS x(T) = {}, SS x(T) = {}",
            ms_result.final_state[0],
            ss_result.final_state[0],
        );
    }

    /// Trajectory output structure check.
    #[test]
    fn test_trajectory_structure() {
        let result = MultipleShootingProblem::new(1, 1)
            .dynamics(|_t, _y, dydt, u| {
                dydt[0] = u[0];
            })
            .initial_state(vec![0.0])
            .time_span(0.0, 1.0)
            .n_segments(5)
            .terminal_cost(|y| y[0].powi(2))
            .max_iter(50)
            .solve()
            .expect("multiple shooting solve failed");

        assert!(
            !result.t_trajectory.is_empty(),
            "t_trajectory should be non-empty"
        );
        assert!(
            !result.y_trajectory.is_empty(),
            "y_trajectory should be non-empty"
        );

        // First time is t0.
        assert!(
            (result.t_trajectory[0] - 0.0).abs() < 1e-12,
            "first time should be 0.0"
        );

        // Last time is ~tf.
        let t_last = *result.t_trajectory.last().unwrap();
        assert!(
            (t_last - 1.0).abs() < 1e-6,
            "last time should be ~1.0, got {t_last}"
        );

        // Consistent lengths.
        assert_eq!(
            result.y_trajectory.len(),
            result.t_trajectory.len() * result.n_states,
        );
    }
}

//! Single-shooting optimal control.
//!
//! Given a controlled ODE `dy/dt = f(t, y, u)`, an initial state `y(t0)`,
//! and a cost functional (terminal and/or running cost), find the piecewise-
//! constant control sequence `u_0, u_1, ..., u_{N-1}` that minimizes the
//! total cost subject to optional terminal equality constraints.
//!
//! The control vector is parameterised as `N` segments of `n_controls` each,
//! yielding `n_decision = n_controls * n_segments` decision variables.
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

/// Dynamics closure: `(t, y, dydt, u)`.
type DynamicsFn<S> = dyn Fn(S, &[S], &mut [S], &[S]) + Send + Sync;

/// Terminal cost closure: `phi(y(T)) -> S`.
type TerminalCostFn<S> = dyn Fn(&[S]) -> S + Send + Sync;

/// Running cost closure: `L(t, y, u) -> S`.
type RunningCostFn<S> = dyn Fn(S, &[S], &[S]) -> S + Send + Sync;

/// Terminal constraint closure: `h(y(T)) -> Vec<S>`, each component = 0.
type TerminalConstraintFn<S> = dyn Fn(&[S]) -> Vec<S> + Send + Sync;

/// Result of a single-shooting optimal control solve.
#[derive(Clone, Debug)]
pub struct ShootingResult<S: Scalar> {
    /// Optimal control vector (flat: `n_controls * n_segments`).
    pub controls: Vec<S>,
    /// Final state `y(T)` at the optimum.
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
    /// Number of states (useful for interpreting `y_trajectory`).
    pub n_states: usize,
}

// ---------------------------------------------------------------------------
// Builder
// ---------------------------------------------------------------------------

/// Builder for single-shooting optimal control problems.
pub struct ShootingProblem<S: Scalar> {
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

impl<S: Scalar> ShootingProblem<S> {
    /// Create a new shooting problem.
    ///
    /// - `n_states`: dimension of the ODE state vector.
    /// - `n_controls`: dimension of the control vector per segment.
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

    /// Set the controlled ODE right-hand side: `f(t, y, dydt, u)`.
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

    /// Set the number of control intervals.
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

    /// Execute the single-shooting optimal control solve.
    pub fn solve(self) -> Result<ShootingResult<S>, OcpError>
    where
        S: faer::SimpleEntity + faer::Conjugate<Canonical = S> + faer::ComplexField,
    {
        let start = Instant::now();

        // -- Validate inputs ------------------------------------------------
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

        let n_states = self.n_states;
        let n_controls = self.n_controls;
        let n_segments = self.n_segments;
        let n_decision = n_controls * n_segments;
        let t0 = self.t0;
        let tf = self.tf;
        let dt = (tf - t0) / S::from_usize(n_segments);
        let ode_rtol = self.ode_rtol;
        let ode_atol = self.ode_atol;

        // -- Shared state ---------------------------------------------------
        let dynamics = Arc::new(dynamics);
        let y0 = Arc::new(y0);
        let terminal_cost: Option<Arc<Box<TerminalCostFn<S>>>> = self.terminal_cost.map(Arc::new);
        let running_cost: Option<Arc<Box<RunningCostFn<S>>>> = self.running_cost.map(Arc::new);

        let params = SimParams {
            n_states,
            n_controls,
            n_segments,
            t0,
            dt,
            ode_rtol,
            ode_atol,
        };

        // -- Build objective ------------------------------------------------
        let dyn_obj = Arc::clone(&dynamics);
        let y0_obj = Arc::clone(&y0);
        let tc_obj = terminal_cost.clone();
        let rc_obj = running_cost.clone();
        let p_obj = params;

        let big = S::from_f64(1e20);
        let objective_fn = move |u: &[S]| -> S {
            let rc_ref = rc_obj.as_ref().map(|b| &***b as &RunningCostFn<S>);
            let tc_ref = tc_obj.as_ref().map(|b| &***b as &TerminalCostFn<S>);
            match simulate(&dyn_obj, &y0_obj, u, &p_obj, rc_ref, tc_ref) {
                Ok((_traj_t, _traj_y, cost)) => cost,
                Err(_) => big,
            }
        };

        // -- Build OptimProblem ---------------------------------------------
        let u0 = vec![S::ZERO; n_decision];
        let mut prob = OptimProblem::new(n_decision)
            .x0(&u0)
            .objective(objective_fn)
            .max_iter(self.max_iter);

        // Apply control bounds to every segment.
        for seg in 0..n_segments {
            for ctrl in 0..n_controls {
                if let Some(&Some((lo, hi))) = self.control_bounds.get(ctrl) {
                    prob = prob.bounds(seg * n_controls + ctrl, (lo, hi));
                }
            }
        }

        // -- Terminal constraints -------------------------------------------
        if let Some(tc_fn) = self.terminal_constraints {
            let tc_fn = Arc::new(tc_fn);

            // Probe to determine number of constraints.
            let dummy = vec![S::ZERO; n_states];
            let n_constraints = tc_fn(&dummy).len();

            let big_c = S::from_f64(1e20);
            for ci in 0..n_constraints {
                let dyn_c = Arc::clone(&dynamics);
                let y0_c = Arc::clone(&y0);
                let tc_c = Arc::clone(&tc_fn);
                let p_c = params;

                prob = prob.constraint_eq(move |u: &[S]| -> S {
                    match simulate_final_state(&dyn_c, &y0_c, u, &p_c) {
                        Ok(y_final) => tc_c(&y_final)[ci],
                        Err(_) => big_c,
                    }
                });
            }
        }

        // -- Solve ----------------------------------------------------------
        let optim_result = prob.solve().map_err(OcpError::OptimFailed)?;

        // -- Reconstruct trajectory at optimal controls ---------------------
        let optimal_u = &optim_result.x;
        let rc_final = running_cost.as_ref().map(|b| &***b as &RunningCostFn<S>);
        let tc_final = terminal_cost.as_ref().map(|b| &***b as &TerminalCostFn<S>);
        let (traj_t, traj_y, obj) =
            simulate(&dynamics, &y0, optimal_u, &params, rc_final, tc_final)
                .map_err(OcpError::IntegrationFailed)?;

        let final_state = if traj_t.is_empty() {
            y0.as_ref().clone()
        } else {
            let last_idx = traj_t.len() - 1;
            traj_y[last_idx * n_states..(last_idx + 1) * n_states].to_vec()
        };

        Ok(ShootingResult {
            controls: optimal_u.clone(),
            final_state,
            objective: obj,
            converged: optim_result.converged,
            message: optim_result.message.clone(),
            iterations: optim_result.iterations,
            wall_time_secs: start.elapsed().as_secs_f64(),
            t_trajectory: traj_t,
            y_trajectory: traj_y,
            n_states,
        })
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Simulation parameters shared across closures.
#[derive(Clone, Copy)]
struct SimParams<S: Scalar> {
    n_states: usize,
    n_controls: usize,
    n_segments: usize,
    t0: S,
    dt: S,
    ode_rtol: S,
    ode_atol: S,
}

/// Simulate the full trajectory under piecewise-constant controls and
/// return `(t_grid, y_flat, total_cost)`.
///
/// `t_grid` and `y_flat` concatenate the ODE output of all segments.
/// Duplicate boundary points between consecutive segments are removed
/// (only the first segment keeps its initial point; subsequent segments
/// skip the duplicated initial state).
fn simulate<S: Scalar>(
    dynamics: &Arc<Box<DynamicsFn<S>>>,
    y0: &Arc<Vec<S>>,
    u: &[S],
    p: &SimParams<S>,
    running_cost: Option<&RunningCostFn<S>>,
    terminal_cost: Option<&TerminalCostFn<S>>,
) -> Result<(Vec<S>, Vec<S>, S), String> {
    let options = SolverOptions::default().rtol(p.ode_rtol).atol(p.ode_atol);

    let mut traj_t: Vec<S> = Vec::new();
    let mut traj_y: Vec<S> = Vec::new();
    let mut y_cur = y0.as_ref().clone();
    let mut total_cost = S::ZERO;

    for seg in 0..p.n_segments {
        let t_start = p.t0 + S::from_usize(seg) * p.dt;
        let t_end = p.t0 + S::from_usize(seg + 1) * p.dt;
        let u_seg: Vec<S> = u[seg * p.n_controls..(seg + 1) * p.n_controls].to_vec();

        // Build ODE RHS with this segment's control baked in.
        let dyn_ref = Arc::clone(dynamics);
        let u_seg_clone = u_seg.clone();
        let rhs = move |t: S, y: &[S], dydt: &mut [S]| {
            dyn_ref(t, y, dydt, &u_seg_clone);
        };

        let problem = OdeProblem::new(rhs, t_start, t_end, y_cur.clone());
        let result = DoPri5::solve(&problem, t_start, t_end, &y_cur, &options)
            .map_err(|e| format!("segment {seg}: {e}"))?;

        if !result.success {
            return Err(format!("segment {seg}: {}", result.message));
        }

        // Accumulate running cost via trapezoidal rule.
        if let Some(rc) = running_cost {
            let n_pts = result.t.len();
            for k in 0..n_pts.saturating_sub(1) {
                let tk = result.t[k];
                let tk1 = result.t[k + 1];
                let yk = &result.y[k * p.n_states..(k + 1) * p.n_states];
                let yk1 = &result.y[(k + 1) * p.n_states..(k + 2) * p.n_states];
                let lk = rc(tk, yk, &u_seg);
                let lk1 = rc(tk1, yk1, &u_seg);
                total_cost += S::HALF * (tk1 - tk) * (lk + lk1);
            }
        }

        // Append trajectory, skipping the first point for segments > 0
        // to avoid duplicating the boundary.
        let skip = if seg == 0 { 0 } else { 1 };
        for k in skip..result.t.len() {
            traj_t.push(result.t[k]);
            traj_y.extend_from_slice(&result.y[k * p.n_states..(k + 1) * p.n_states]);
        }

        // Chain: final state of this segment is initial state of the next.
        y_cur = result
            .y_final()
            .ok_or_else(|| format!("segment {seg}: empty result"))?;
    }

    // Add terminal cost.
    if let Some(tc) = terminal_cost {
        total_cost += tc(&y_cur);
    }

    Ok((traj_t, traj_y, total_cost))
}

/// Simulate only to obtain the final state `y(T)`.
fn simulate_final_state<S: Scalar>(
    dynamics: &Arc<Box<DynamicsFn<S>>>,
    y0: &Arc<Vec<S>>,
    u: &[S],
    p: &SimParams<S>,
) -> Result<Vec<S>, String> {
    let options = SolverOptions::default().rtol(p.ode_rtol).atol(p.ode_atol);
    let mut y_cur = y0.as_ref().clone();

    for seg in 0..p.n_segments {
        let t_start = p.t0 + S::from_usize(seg) * p.dt;
        let t_end = p.t0 + S::from_usize(seg + 1) * p.dt;
        let u_seg: Vec<S> = u[seg * p.n_controls..(seg + 1) * p.n_controls].to_vec();

        let dyn_ref = Arc::clone(dynamics);
        let rhs = move |t: S, y: &[S], dydt: &mut [S]| {
            dyn_ref(t, y, dydt, &u_seg);
        };

        let problem = OdeProblem::new(rhs, t_start, t_end, y_cur.clone());
        let result = DoPri5::solve(&problem, t_start, t_end, &y_cur, &options)
            .map_err(|e| format!("segment {seg}: {e}"))?;

        if !result.success {
            return Err(format!("segment {seg}: {}", result.message));
        }

        y_cur = result
            .y_final()
            .ok_or_else(|| format!("segment {seg}: empty result"))?;
    }

    Ok(y_cur)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Double integrator: dx/dt = v, dv/dt = u.
    /// Terminal cost: 100*((x-1)^2 + v^2).
    /// Running cost: 0.01*u^2.
    /// T = 2.0, 10 segments.
    #[test]
    fn test_double_integrator_terminal_cost() {
        let result = ShootingProblem::new(2, 1)
            .dynamics(|_t, y, dydt, u| {
                dydt[0] = y[1]; // dx/dt = v
                dydt[1] = u[0]; // dv/dt = u
            })
            .initial_state(vec![0.0, 0.0])
            .time_span(0.0, 2.0)
            .n_segments(10)
            .terminal_cost(|y| 100.0 * ((y[0] - 1.0).powi(2) + y[1].powi(2)))
            .running_cost(|_t, _y, u| 0.01 * u[0].powi(2))
            .max_iter(200)
            .solve()
            .expect("shooting solve failed");

        let x_final = result.final_state[0];
        assert!(
            (x_final - 1.0).abs() < 0.3,
            "x(T) = {x_final}, expected within 0.3 of 1.0"
        );
    }

    /// Minimum-energy control: 1-state dx/dt = u.
    /// Terminal cost: 1000*(x-1)^2.
    /// Running cost: u^2.
    /// T = 1.0, 10 segments.
    #[test]
    fn test_minimum_energy_control() {
        let result = ShootingProblem::new(1, 1)
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
            .expect("shooting solve failed");

        let x_final = result.final_state[0];
        assert!(
            (x_final - 1.0).abs() < 0.3,
            "x(T) = {x_final}, expected within 0.3 of 1.0"
        );
    }

    /// Pure terminal cost (no running cost).
    /// 1-state dx/dt = u, terminal cost: (x-3)^2.
    /// T = 1.0, 5 segments.
    #[test]
    fn test_pure_terminal_cost() {
        let result = ShootingProblem::new(1, 1)
            .dynamics(|_t, _y, dydt, u| {
                dydt[0] = u[0];
            })
            .initial_state(vec![0.0])
            .time_span(0.0, 1.0)
            .n_segments(5)
            .terminal_cost(|y| (y[0] - 3.0).powi(2))
            .max_iter(200)
            .solve()
            .expect("shooting solve failed");

        let x_final = result.final_state[0];
        assert!(
            (x_final - 3.0).abs() < 0.5,
            "x(T) = {x_final}, expected within 0.5 of 3.0"
        );
    }

    /// Trajectory output structure check.
    /// 1-state dx/dt = u, terminal cost: x^2.
    #[test]
    fn test_trajectory_output() {
        let result = ShootingProblem::new(1, 1)
            .dynamics(|_t, _y, dydt, u| {
                dydt[0] = u[0];
            })
            .initial_state(vec![0.0])
            .time_span(0.0, 1.0)
            .n_segments(5)
            .terminal_cost(|y| y[0].powi(2))
            .max_iter(50)
            .solve()
            .expect("shooting solve failed");

        // Trajectory arrays are populated.
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
            "first time should be t0=0.0, got {}",
            result.t_trajectory[0],
        );

        // Last time is approximately tf.
        let t_last = *result.t_trajectory.last().unwrap();
        assert!(
            (t_last - 1.0).abs() < 1e-6,
            "last time should be ~tf=1.0, got {t_last}"
        );

        // y_trajectory length is consistent.
        assert!(
            !result.y_trajectory.is_empty(),
            "y_trajectory should have entries"
        );
        assert_eq!(
            result.y_trajectory.len(),
            result.t_trajectory.len() * result.n_states,
            "y_trajectory length mismatch"
        );

        // n_states matches.
        assert_eq!(result.n_states, 1);
    }
}

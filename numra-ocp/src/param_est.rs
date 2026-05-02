//! Parameter estimation for ODE models.
//!
//! Given an ODE model `dy/dt = f(t, y; p)` and observed data `(t_i, y_i)`,
//! find the parameters `p` that minimize the residual between predicted and
//! observed states.
//!
//! Author: Moussa Leblouba
//! Date: 9 February 2026
//! Modified: 2 May 2026

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Instant;

use numra_core::Scalar;
use numra_ode::{DoPri5, OdeProblem, Solver, SolverOptions};
use numra_optim::OptimProblem;

use crate::error::OcpError;

/// ODE model closure: `(t, y, dydt, params)`.
type ModelFn<S> = dyn Fn(S, &[S], &mut [S], &[S]) + Send + Sync;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// Which ODE solver to use for forward integrations.
#[derive(Clone, Debug, Default)]
pub enum OdeSolverChoice {
    /// Dormand-Prince 5(4) explicit method (non-stiff).
    #[default]
    DoPri5,
}

/// Result of a parameter estimation run.
#[derive(Clone, Debug)]
pub struct ParamEstResult<S: Scalar> {
    /// Estimated parameters.
    pub params: Vec<S>,
    /// Final residual norm (L2).
    pub residual_norm: S,
    /// Optimizer iterations.
    pub iterations: usize,
    /// Whether the optimizer converged.
    pub converged: bool,
    /// Human-readable status message.
    pub message: String,
    /// Predicted observations at the data times (flat row-major).
    pub predicted: Vec<S>,
    /// Total number of ODE integrations performed.
    pub n_integrations: usize,
    /// Wall-clock time in seconds.
    pub wall_time_secs: f64,
}

// ---------------------------------------------------------------------------
// Builder
// ---------------------------------------------------------------------------

/// Builder for ODE parameter estimation problems.
pub struct ParamEstProblem<S: Scalar> {
    n_params: usize,
    n_states: usize,
    model: Option<Box<ModelFn<S>>>,
    y0: Option<Vec<S>>,
    params0: Option<Vec<S>>,
    param_bounds: Vec<Option<(S, S)>>,
    t_data: Vec<S>,
    y_data: Vec<S>,
    observed_indices: Option<Vec<usize>>,
    solver: OdeSolverChoice,
    ode_rtol: S,
    ode_atol: S,
    max_iter: usize,
}

impl<S: Scalar> ParamEstProblem<S> {
    /// Create a new parameter estimation problem.
    ///
    /// - `n_params`: number of parameters to estimate.
    /// - `n_states`: dimension of the ODE state vector.
    pub fn new(n_params: usize, n_states: usize) -> Self {
        Self {
            n_params,
            n_states,
            model: None,
            y0: None,
            params0: None,
            param_bounds: vec![None; n_params],
            t_data: Vec::new(),
            y_data: Vec::new(),
            observed_indices: None,
            solver: OdeSolverChoice::default(),
            ode_rtol: S::from_f64(1e-8),
            ode_atol: S::from_f64(1e-10),
            max_iter: 100,
        }
    }

    /// Set the ODE right-hand side: `f(t, y, dydt, params)`.
    pub fn model<F>(mut self, f: F) -> Self
    where
        F: Fn(S, &[S], &mut [S], &[S]) + Send + Sync + 'static,
    {
        self.model = Some(Box::new(f));
        self
    }

    /// Set the initial state `y(t0)`.
    pub fn initial_state(mut self, y0: Vec<S>) -> Self {
        self.y0 = Some(y0);
        self
    }

    /// Set the initial parameter guess.
    pub fn params(mut self, p0: Vec<S>) -> Self {
        self.params0 = Some(p0);
        self
    }

    /// Set bounds for parameter `i`.
    pub fn param_bounds(mut self, i: usize, bounds: (S, S)) -> Self {
        self.param_bounds[i] = Some(bounds);
        self
    }

    /// Set bounds for all parameters at once.
    pub fn all_param_bounds(mut self, bounds: Vec<Option<(S, S)>>) -> Self {
        self.param_bounds = bounds;
        self
    }

    /// Set measurement data.
    ///
    /// `y_data` is flat row-major: `y_data[i * n_observed + j]` for time
    /// index `i` and observed state `j`.
    pub fn data(mut self, t_data: Vec<S>, y_data: Vec<S>) -> Self {
        self.t_data = t_data;
        self.y_data = y_data;
        self
    }

    /// Specify which state indices are observed.
    ///
    /// If not called, all states are assumed observed.
    pub fn observed(mut self, indices: Vec<usize>) -> Self {
        self.observed_indices = Some(indices);
        self
    }

    /// Choose the ODE solver for forward integrations.
    pub fn ode_solver(mut self, choice: OdeSolverChoice) -> Self {
        self.solver = choice;
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

    /// Run parameter estimation.
    pub fn solve(self) -> Result<ParamEstResult<S>, OcpError>
    where
        S: faer::SimpleEntity + faer::Conjugate<Canonical = S> + faer::ComplexField,
    {
        let start = Instant::now();

        // -- Validate -------------------------------------------------------
        let model = self.model.ok_or(OcpError::NoModel)?;
        let y0 = self.y0.ok_or(OcpError::NoInitialState)?;
        let params0 = self
            .params0
            .ok_or(OcpError::Other("no initial parameter guess".to_string()))?;
        if self.t_data.is_empty() || self.y_data.is_empty() {
            return Err(OcpError::NoData);
        }
        if y0.len() != self.n_states {
            return Err(OcpError::DimensionMismatch(format!(
                "y0 length {} != n_states {}",
                y0.len(),
                self.n_states
            )));
        }
        if params0.len() != self.n_params {
            return Err(OcpError::DimensionMismatch(format!(
                "params0 length {} != n_params {}",
                params0.len(),
                self.n_params
            )));
        }

        let obs_idx: Vec<usize> = self
            .observed_indices
            .unwrap_or_else(|| (0..self.n_states).collect());
        let n_observed = obs_idx.len();
        let n_data = self.t_data.len();
        let n_residuals = n_data * n_observed;

        if self.y_data.len() != n_residuals {
            return Err(OcpError::DimensionMismatch(format!(
                "y_data length {} != n_data({}) * n_observed({})",
                self.y_data.len(),
                n_data,
                n_observed,
            )));
        }

        // -- Shared state ---------------------------------------------------
        let model = Arc::new(model);
        let y0 = Arc::new(y0);
        let t_data = Arc::new(self.t_data);
        let y_data = Arc::new(self.y_data);
        let obs_idx = Arc::new(obs_idx);
        let n_states = self.n_states;
        let ode_rtol = self.ode_rtol;
        let ode_atol = self.ode_atol;
        let counter = Arc::new(AtomicUsize::new(0));
        let has_bounds = self.param_bounds.iter().any(|b| b.is_some());

        // -- Optimize -------------------------------------------------------
        let optim_result = if has_bounds {
            // L-BFGS-B with scalar sum-of-squares objective.
            let m = Arc::clone(&model);
            let y0c = Arc::clone(&y0);
            let td = Arc::clone(&t_data);
            let yd = Arc::clone(&y_data);
            let oi = Arc::clone(&obs_idx);
            let ctr = Arc::clone(&counter);

            let mut prob = OptimProblem::new(self.n_params)
                .x0(&params0)
                .objective(move |p: &[S]| {
                    let pred = integrate_at_params(&m, &y0c, &td, p, n_states, ode_rtol, ode_atol);
                    ctr.fetch_add(1, Ordering::Relaxed);
                    let mut sos = S::ZERO;
                    for i in 0..td.len() {
                        for (j, &idx) in oi.iter().enumerate() {
                            let r = pred[i * n_states + idx] - yd[i * oi.len() + j];
                            sos += r * r;
                        }
                    }
                    sos
                })
                .max_iter(self.max_iter);

            for (i, b) in self.param_bounds.iter().enumerate() {
                if let Some(&(lo, hi)) = b.as_ref() {
                    prob = prob.bounds(i, (lo, hi));
                }
            }
            prob.solve().map_err(OcpError::OptimFailed)?
        } else {
            // LM (least squares).
            let m = Arc::clone(&model);
            let y0c = Arc::clone(&y0);
            let td = Arc::clone(&t_data);
            let yd = Arc::clone(&y_data);
            let oi = Arc::clone(&obs_idx);
            let ctr = Arc::clone(&counter);

            OptimProblem::new(self.n_params)
                .x0(&params0)
                .least_squares(n_residuals, move |p: &[S], r: &mut [S]| {
                    let pred = integrate_at_params(&m, &y0c, &td, p, n_states, ode_rtol, ode_atol);
                    ctr.fetch_add(1, Ordering::Relaxed);
                    for i in 0..td.len() {
                        for (j, &idx) in oi.iter().enumerate() {
                            r[i * oi.len() + j] = pred[i * n_states + idx] - yd[i * oi.len() + j];
                        }
                    }
                })
                .max_iter(self.max_iter)
                .solve()
                .map_err(OcpError::OptimFailed)?
        };

        // -- Final integration at optimal params ----------------------------
        let optimal_params = &optim_result.x;
        let pred_full = integrate_at_params(
            &model,
            &y0,
            &t_data,
            optimal_params,
            n_states,
            ode_rtol,
            ode_atol,
        );
        counter.fetch_add(1, Ordering::Relaxed);

        // Extract predicted observations (only observed indices).
        let mut predicted = Vec::with_capacity(n_residuals);
        for i in 0..n_data {
            for &idx in obs_idx.iter() {
                predicted.push(pred_full[i * n_states + idx]);
            }
        }

        // Residual norm.
        let mut rnorm2 = S::ZERO;
        for k in 0..n_residuals {
            let r = predicted[k] - y_data[k];
            rnorm2 += r * r;
        }
        let residual_norm = rnorm2.sqrt();

        Ok(ParamEstResult {
            params: optimal_params.clone(),
            residual_norm,
            iterations: optim_result.iterations,
            converged: optim_result.converged,
            message: optim_result.message.clone(),
            predicted,
            n_integrations: counter.load(Ordering::Relaxed),
            wall_time_secs: start.elapsed().as_secs_f64(),
        })
    }
}

// ---------------------------------------------------------------------------
// Helper: forward integration and flat extraction
// ---------------------------------------------------------------------------

/// Integrate the ODE at the given parameters and return the full state
/// at each data time as a flat vector of length `n_data * n_states`.
///
/// We integrate segment-by-segment between successive data times so
/// the solver lands exactly on each measurement time.
///
/// If the ODE integration fails, returns a vector filled with `1e10` to
/// steer the optimizer away from this region.
fn integrate_at_params<S: Scalar>(
    model: &Arc<Box<ModelFn<S>>>,
    y0: &Arc<Vec<S>>,
    t_data: &Arc<Vec<S>>,
    params: &[S],
    n_states: usize,
    rtol: S,
    atol: S,
) -> Vec<S> {
    let n_data = t_data.len();
    let total = n_data * n_states;

    let options = SolverOptions::default().rtol(rtol).atol(atol);

    // Store the state at each data time.
    let mut out = Vec::with_capacity(total);

    // Current state, starting from y0.
    let mut y_cur = y0.as_ref().clone();

    // First data point: the initial state itself.
    out.extend_from_slice(&y_cur);

    let big = S::from_f64(1e10);
    let tiny = S::from_f64(1e-15);

    // Integrate from t_data[i] to t_data[i+1] for each segment.
    for i in 0..(n_data - 1) {
        let t_start = t_data[i];
        let t_end = t_data[i + 1];

        // Skip zero-length segments.
        if (t_end - t_start).abs() < tiny {
            out.extend_from_slice(&y_cur);
            continue;
        }

        let p = params.to_vec();
        let model_ref = Arc::clone(model);
        let rhs = move |t: S, y: &[S], dydt: &mut [S]| {
            model_ref(t, y, dydt, &p);
        };

        let problem = OdeProblem::new(rhs, t_start, t_end, y_cur.clone());

        match DoPri5::solve(&problem, t_start, t_end, &y_cur, &options) {
            Ok(result) if result.success => {
                // The last time point should be t_end; grab the final state.
                if let Some(y_final) = result.y_final() {
                    y_cur = y_final.to_vec();
                    out.extend_from_slice(&y_cur);
                } else {
                    return vec![big; total];
                }
            }
            _ => return vec![big; total],
        }
    }

    out
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Exponential decay: dy/dt = -k*y, true k=0.5.
    #[test]
    fn test_exponential_decay() {
        let k_true = 0.5;
        let y0_val = 1.0;
        let t_data: Vec<f64> = (0..=10).map(|i| i as f64 * 0.5).collect();
        let y_data: Vec<f64> = t_data
            .iter()
            .map(|&t| y0_val * (-k_true * t).exp())
            .collect();

        let result = ParamEstProblem::new(1, 1)
            .model(|_t: f64, y, dydt, p| {
                dydt[0] = -p[0] * y[0];
            })
            .initial_state(vec![y0_val])
            .params(vec![1.0])
            .data(t_data, y_data)
            .solve()
            .expect("parameter estimation failed");

        assert!(
            result.converged,
            "optimizer did not converge: {}",
            result.message
        );
        let k_est = result.params[0];
        assert!(
            (k_est - k_true).abs() < 0.01,
            "k_est = {k_est}, expected ~{k_true}"
        );
        assert!(
            result.residual_norm < 1e-4,
            "residual_norm = {}",
            result.residual_norm
        );
        assert!(result.n_integrations > 0);
    }

    /// Two-parameter model: dy/dt = -a*y + b, true a=1, b=2.
    /// Analytical: y(t) = b/a + (y0 - b/a)*exp(-a*t) = 2 + (y0-2)*exp(-t).
    #[test]
    fn test_two_param_model() {
        let a_true = 1.0;
        let b_true = 2.0;
        let y0_val = 1.0;

        let t_data: Vec<f64> = (0..=20).map(|i| i as f64 * 0.25).collect();
        let y_data: Vec<f64> = t_data
            .iter()
            .map(|&t| b_true / a_true + (y0_val - b_true / a_true) * (-a_true * t).exp())
            .collect();

        let result = ParamEstProblem::new(2, 1)
            .model(|_t: f64, y, dydt, p| {
                dydt[0] = -p[0] * y[0] + p[1];
            })
            .initial_state(vec![y0_val])
            .params(vec![0.5, 1.0])
            .data(t_data, y_data)
            .solve()
            .expect("parameter estimation failed");

        assert!(
            result.converged,
            "optimizer did not converge: {}",
            result.message
        );
        assert!(
            (result.params[0] - a_true).abs() < 0.1,
            "a_est = {}, expected ~{a_true}",
            result.params[0]
        );
        assert!(
            (result.params[1] - b_true).abs() < 0.1,
            "b_est = {}, expected ~{b_true}",
            result.params[1]
        );
    }

    /// Exponential decay with bounds: k in [0.01, 5.0].
    #[test]
    fn test_param_est_with_bounds() {
        let k_true = 0.5;
        let y0_val = 1.0;
        let t_data: Vec<f64> = (0..=10).map(|i| i as f64 * 0.5).collect();
        let y_data: Vec<f64> = t_data
            .iter()
            .map(|&t| y0_val * (-k_true * t).exp())
            .collect();

        let result = ParamEstProblem::new(1, 1)
            .model(|_t: f64, y, dydt, p| {
                dydt[0] = -p[0] * y[0];
            })
            .initial_state(vec![y0_val])
            .params(vec![3.0])
            .param_bounds(0, (0.01, 5.0))
            .data(t_data, y_data)
            .solve()
            .expect("parameter estimation failed");

        assert!(
            result.converged,
            "optimizer did not converge: {}",
            result.message
        );
        let k_est = result.params[0];
        assert!(
            (k_est - k_true).abs() < 0.05,
            "k_est = {k_est}, expected ~{k_true}"
        );
        assert!(
            (0.01..=5.0).contains(&k_est),
            "k_est out of bounds: {k_est}"
        );
    }

    /// Partial observation: 2-state coupled system, observe only state 0.
    ///   dx/dt = -a*x + y
    ///   dy/dt = x - b*y
    /// True: a=0.5, b=1.0, x0=1, y0=0.
    /// Both parameters influence x(t) through coupling.
    #[test]
    fn test_partial_observation() {
        let a_true = 0.5;
        let b_true = 1.0;
        let x0 = 1.0;
        let y0_val = 0.0;

        // Generate "exact" data by integrating the ODE with the true params.
        let t_data: Vec<f64> = (0..=20).map(|i| i as f64 * 0.5).collect();

        // Integrate with true parameters to get reference data.
        let opts = numra_ode::SolverOptions::default().rtol(1e-12).atol(1e-14);

        // Integrate segment by segment to get exact values at data times.
        let mut y_data = Vec::new();
        let mut y_cur = vec![x0, y0_val];
        y_data.push(y_cur[0]); // state 0 at t=0
        for i in 0..(t_data.len() - 1) {
            let t_s = t_data[i];
            let t_e = t_data[i + 1];
            let prob = numra_ode::OdeProblem::new(
                move |_t: f64, y: &[f64], dydt: &mut [f64]| {
                    dydt[0] = -a_true * y[0] + y[1];
                    dydt[1] = y[0] - b_true * y[1];
                },
                t_s,
                t_e,
                y_cur.clone(),
            );
            let res = numra_ode::DoPri5::solve(&prob, t_s, t_e, &y_cur, &opts).unwrap();
            y_cur = res.y_final().unwrap().to_vec();
            y_data.push(y_cur[0]); // only state 0
        }

        let result = ParamEstProblem::new(2, 2)
            .model(|_t: f64, y, dydt, p| {
                dydt[0] = -p[0] * y[0] + y[1];
                dydt[1] = y[0] - p[1] * y[1];
            })
            .initial_state(vec![x0, y0_val])
            .params(vec![0.8, 1.5]) // initial guess
            .observed(vec![0]) // only observe x
            .data(t_data, y_data)
            .max_iter(200)
            .solve()
            .expect("parameter estimation failed");

        assert!(
            result.converged,
            "optimizer did not converge: {}",
            result.message
        );
        assert!(
            (result.params[0] - a_true).abs() < 0.2,
            "a_est = {}, expected ~{a_true}",
            result.params[0]
        );
        assert!(
            (result.params[1] - b_true).abs() < 0.2,
            "b_est = {}, expected ~{b_true}",
            result.params[1]
        );
    }
}

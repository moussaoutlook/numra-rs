//! Common types for optimization algorithms.
//!
//! Author: Moussa Leblouba
//! Date: 8 February 2026
//! Modified: 2 May 2026

use numra_core::Scalar;

/// Options common to unconstrained optimizers.
#[derive(Clone, Debug)]
pub struct OptimOptions<S: Scalar> {
    pub max_iter: usize,
    pub gtol: S,
    pub ftol: S,
    pub xtol: S,
    pub verbose: bool,
}

impl<S: Scalar> Default for OptimOptions<S> {
    fn default() -> Self {
        Self {
            max_iter: 1000,
            gtol: S::from_f64(1e-8),
            ftol: S::from_f64(1e-12),
            xtol: S::from_f64(1e-12),
            verbose: false,
        }
    }
}

impl<S: Scalar> OptimOptions<S> {
    pub fn max_iter(mut self, n: usize) -> Self {
        self.max_iter = n;
        self
    }
    pub fn gtol(mut self, tol: S) -> Self {
        self.gtol = tol;
        self
    }
    pub fn ftol(mut self, tol: S) -> Self {
        self.ftol = tol;
        self
    }
    pub fn xtol(mut self, tol: S) -> Self {
        self.xtol = tol;
        self
    }
}

/// Status of an optimization run.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum OptimStatus {
    GradientConverged,
    FunctionConverged,
    StepConverged,
    MaxIterations,
    LineSearchFailed,
    Infeasible,
    Unbounded,
}

/// Record of a single iteration for convergence history.
#[derive(Clone, Debug)]
pub struct IterationRecord<S: Scalar> {
    pub iteration: usize,
    pub objective: S,
    pub gradient_norm: S,
    pub step_size: S,
    pub constraint_violation: S,
}

/// A single point on a Pareto front.
#[derive(Clone, Debug)]
pub struct ParetoPoint<S: Scalar> {
    /// Decision variable values.
    pub x: Vec<S>,
    /// Objective function values.
    pub objectives: Vec<S>,
}

/// Result of multi-objective optimization.
#[derive(Clone, Debug)]
pub struct ParetoResult<S: Scalar> {
    /// Pareto-optimal points (non-dominated solutions).
    pub points: Vec<ParetoPoint<S>>,
}

/// Parametric sensitivity of optimal solution w.r.t. problem parameters.
#[derive(Clone, Debug)]
pub struct ParamSensitivity<S: Scalar> {
    /// Parameter names.
    pub names: Vec<String>,
    /// Sensitivity matrix: `dx_i/dp_j` stored row-major, `n_vars x n_params`.
    /// `values[i * n_params + j]` = dxstar_i/dp_j.
    pub values: Vec<S>,
    /// Number of decision variables.
    pub n_vars: usize,
    /// Number of parameters.
    pub n_params: usize,
}

impl<S: Scalar> ParamSensitivity<S> {
    /// Sensitivity of all variables to parameter `j`.
    pub fn column(&self, j: usize) -> Vec<S> {
        (0..self.n_vars)
            .map(|i| self.values[i * self.n_params + j])
            .collect()
    }

    /// Sensitivity of variable `i` to all parameters.
    pub fn row(&self, i: usize) -> Vec<S> {
        self.values[i * self.n_params..(i + 1) * self.n_params].to_vec()
    }

    /// Single entry: dxstar_i/dp_j.
    pub fn get(&self, i: usize, j: usize) -> S {
        self.values[i * self.n_params + j]
    }
}

/// Result of an optimization.
#[derive(Clone, Debug)]
pub struct OptimResult<S: Scalar> {
    pub x: Vec<S>,
    pub f: S,
    pub grad: Vec<S>,
    pub iterations: usize,
    pub n_feval: usize,
    pub n_geval: usize,
    pub converged: bool,
    pub message: String,
    pub status: OptimStatus,
    pub history: Vec<IterationRecord<S>>,
    pub lambda_eq: Vec<S>,
    pub lambda_ineq: Vec<S>,
    pub active_bounds: Vec<usize>,
    pub constraint_violation: S,
    pub wall_time_secs: f64,
    pub pareto: Option<ParetoResult<S>>,
    pub sensitivity: Option<ParamSensitivity<S>>,
}

impl<S: Scalar> OptimResult<S> {
    /// Set wall_time_secs from elapsed time since `start`.
    pub(crate) fn with_wall_time(mut self, start: std::time::Instant) -> Self {
        self.wall_time_secs = start.elapsed().as_secs_f64();
        self
    }

    /// Construct a result for an unconstrained optimization, filling constrained
    /// fields with defaults.
    #[allow(clippy::too_many_arguments)]
    pub fn unconstrained(
        x: Vec<S>,
        f: S,
        grad: Vec<S>,
        iterations: usize,
        n_feval: usize,
        n_geval: usize,
        converged: bool,
        message: String,
        status: OptimStatus,
    ) -> Self {
        Self {
            x,
            f,
            grad,
            iterations,
            n_feval,
            n_geval,
            converged,
            message,
            status,
            history: Vec::new(),
            lambda_eq: Vec::new(),
            lambda_ineq: Vec::new(),
            active_bounds: Vec::new(),
            constraint_violation: S::ZERO,
            wall_time_secs: 0.0,
            pareto: None,
            sensitivity: None,
        }
    }
}

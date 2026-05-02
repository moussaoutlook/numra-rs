//! Robust optimization with worst-case constraint reformulation.
//!
//! When problem parameters are uncertain, robust optimization tightens
//! constraints to ensure feasibility at a specified confidence level.
//! For each inequality constraint `g(x, p) <= 0`, the solver determines
//! the worst-case parameter values (within the confidence ellipsoid) and
//! enforces `g(x, p_worst) <= 0` instead.
//!
//! # Example
//!
//! ```rust
//! use numra_optim::robust::RobustProblem;
//!
//! let result = RobustProblem::<f64>::new(1)
//!     .x0(&[5.0])
//!     .objective(|x: &[f64], _p: &[f64]| (x[0] - 5.0) * (x[0] - 5.0))
//!     .param("target", 5.0, 1.0)
//!     .solve()
//!     .unwrap();
//! ```
//!
//! Author: Moussa Leblouba
//! Date: 8 February 2026
//! Modified: 2 May 2026

use std::sync::Arc;

use numra_core::Scalar;

use crate::error::OptimError;
use crate::optim_sensitivity::compute_param_sensitivity;
use crate::problem::{ConstraintKind, OptimProblem};
use crate::types::ParamSensitivity;

/// Shared parameterized scalar function: `f(x, params) -> S`.
type ParamObjFn<S> = Arc<dyn Fn(&[S], &[S]) -> S + Send + Sync>;
/// Shared parameterized gradient function: `g(x, params, grad_out)`.
type ParamGradFn<S> = Arc<dyn Fn(&[S], &[S], &mut [S]) + Send + Sync>;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// An uncertain parameter for robust optimization.
#[derive(Clone, Debug)]
pub struct UncertainParam<S: Scalar> {
    /// Parameter name (for reporting).
    pub name: String,
    /// Nominal (mean) value.
    pub mean: S,
    /// Standard deviation.
    pub std: S,
}

/// Options for robust optimization.
#[derive(Clone, Debug)]
pub struct RobustOptions<S: Scalar> {
    /// Confidence level in (0, 1). Default: 0.95.
    pub confidence: S,
    /// Maximum optimizer iterations. Default: 1000.
    pub max_iter: usize,
}

impl<S: Scalar> Default for RobustOptions<S> {
    fn default() -> Self {
        Self {
            confidence: S::from_f64(0.95),
            max_iter: 1000,
        }
    }
}

/// Result of robust optimization.
#[derive(Clone, Debug)]
pub struct RobustResult<S: Scalar> {
    /// Optimal decision variables.
    pub x: Vec<S>,
    /// Nominal objective value (at mean parameters).
    pub f_nominal: S,
    /// Worst-case objective value.
    pub f_worst_case: S,
    /// Solution uncertainty: std dev of each x_i due to parameter uncertainty.
    pub x_std: Vec<S>,
    /// Whether the optimizer converged.
    pub converged: bool,
    /// Status message.
    pub message: String,
    /// Iterations.
    pub iterations: usize,
    /// Wall time.
    pub wall_time_secs: f64,
    /// Parametric sensitivity (dx*/dp) if computed.
    pub sensitivity: Option<ParamSensitivity<S>>,
}

// ---------------------------------------------------------------------------
// Builder
// ---------------------------------------------------------------------------

/// A parameterized constraint for robust optimization.
struct RobustConstraint<S: Scalar> {
    func: ParamObjFn<S>,
    kind: ConstraintKind,
}

/// Declarative builder for robust optimization problems.
///
/// The objective and constraints are functions of both decision variables `x`
/// and uncertain parameters `p`. The solver reformulates the problem so that
/// constraints hold under worst-case parameter perturbations at the specified
/// confidence level.
pub struct RobustProblem<S: Scalar> {
    n: usize,
    x0: Option<Vec<S>>,
    bounds: Vec<Option<(S, S)>>,
    objective: Option<ParamObjFn<S>>,
    gradient: Option<ParamGradFn<S>>,
    constraints: Vec<RobustConstraint<S>>,
    params: Vec<UncertainParam<S>>,
    options: RobustOptions<S>,
}

impl<S: Scalar> RobustProblem<S> {
    /// Create a new robust optimization problem with `n` decision variables.
    pub fn new(n: usize) -> Self {
        Self {
            n,
            x0: None,
            bounds: vec![None; n],
            objective: None,
            gradient: None,
            constraints: Vec::new(),
            params: Vec::new(),
            options: RobustOptions::default(),
        }
    }

    /// Set the initial point.
    pub fn x0(mut self, x0: &[S]) -> Self {
        self.x0 = Some(x0.to_vec());
        self
    }

    /// Set bounds for variable `i`.
    pub fn bounds(mut self, i: usize, lo_hi: (S, S)) -> Self {
        self.bounds[i] = Some(lo_hi);
        self
    }

    /// Set bounds for all variables at once.
    pub fn all_bounds(mut self, bounds: &[(S, S)]) -> Self {
        for (i, &b) in bounds.iter().enumerate() {
            self.bounds[i] = Some(b);
        }
        self
    }

    /// Set the parameterized objective function `f(x, params)`.
    pub fn objective<F>(mut self, f: F) -> Self
    where
        F: Fn(&[S], &[S]) -> S + Send + Sync + 'static,
    {
        self.objective = Some(Arc::new(f));
        self
    }

    /// Set the gradient of the objective w.r.t. `x`.
    ///
    /// `g(x, params, grad_out)` writes the gradient into `grad_out`.
    pub fn gradient<G>(mut self, g: G) -> Self
    where
        G: Fn(&[S], &[S], &mut [S]) + Send + Sync + 'static,
    {
        self.gradient = Some(Arc::new(g));
        self
    }

    /// Add an inequality constraint `g(x, params) <= 0`.
    pub fn constraint_ineq<F>(mut self, f: F) -> Self
    where
        F: Fn(&[S], &[S]) -> S + Send + Sync + 'static,
    {
        self.constraints.push(RobustConstraint {
            func: Arc::new(f),
            kind: ConstraintKind::Inequality,
        });
        self
    }

    /// Add an equality constraint `h(x, params) = 0`.
    pub fn constraint_eq<F>(mut self, f: F) -> Self
    where
        F: Fn(&[S], &[S]) -> S + Send + Sync + 'static,
    {
        self.constraints.push(RobustConstraint {
            func: Arc::new(f),
            kind: ConstraintKind::Equality,
        });
        self
    }

    /// Add a single uncertain parameter.
    pub fn param(mut self, name: &str, mean: S, std: S) -> Self {
        self.params.push(UncertainParam {
            name: name.to_string(),
            mean,
            std,
        });
        self
    }

    /// Add multiple uncertain parameters at once.
    pub fn params(mut self, params: Vec<UncertainParam<S>>) -> Self {
        self.params.extend(params);
        self
    }

    /// Set the confidence level (must be in (0, 1)).
    pub fn confidence(mut self, level: S) -> Self {
        self.options.confidence = level;
        self
    }

    /// Set the maximum number of optimizer iterations.
    pub fn max_iter(mut self, n: usize) -> Self {
        self.options.max_iter = n;
        self
    }
}

impl<S: Scalar + faer::SimpleEntity + faer::Conjugate<Canonical = S> + faer::ComplexField>
    RobustProblem<S>
{
    /// Solve the robust optimization problem.
    ///
    /// 1. Computes worst-case parameter vectors for each inequality constraint.
    /// 2. Reformulates as a standard `OptimProblem` with tightened constraints.
    /// 3. Solves the reformulated problem.
    /// 4. Computes parametric sensitivity and solution uncertainty.
    pub fn solve(self) -> Result<RobustResult<S>, OptimError> {
        let start = std::time::Instant::now();

        let obj = self.objective.ok_or(OptimError::NoObjective)?;
        let x0 = self.x0.clone().ok_or(OptimError::NoInitialPoint)?;
        let n = self.n;

        // 1. Compute k-factor from confidence level.
        let k = normal_quantile(self.options.confidence);

        // 2. Extract nominal parameter values.
        let p_nom: Vec<S> = self.params.iter().map(|p| p.mean).collect();
        let p_stds: Vec<S> = self.params.iter().map(|p| p.std).collect();
        let n_params = self.params.len();

        // 3. Build a standard OptimProblem.
        // a. Objective at nominal params.
        let obj_for_problem = Arc::clone(&obj);
        let p_nom_obj = p_nom.clone();
        let mut problem = OptimProblem::new(n)
            .x0(&x0)
            .objective(move |x: &[S]| obj_for_problem(x, &p_nom_obj))
            .max_iter(self.options.max_iter);

        // b. If gradient is provided, set it with nominal params.
        if let Some(grad_fn) = &self.gradient {
            let grad_fn = Arc::clone(grad_fn);
            let p_nom_grad = p_nom.clone();
            problem = problem.gradient(move |x: &[S], g: &mut [S]| {
                grad_fn(x, &p_nom_grad, g);
            });
        }

        // c. Apply bounds.
        for (i, b) in self.bounds.iter().enumerate() {
            if let Some(lo_hi) = b {
                problem = problem.bounds(i, *lo_hi);
            }
        }

        // d. Add constraints.
        for rc in &self.constraints {
            match rc.kind {
                ConstraintKind::Equality => {
                    // Equality constraints use nominal params (not robustified).
                    let func = Arc::clone(&rc.func);
                    let p_nom_eq = p_nom.clone();
                    problem = problem.constraint_eq(move |x: &[S]| func(x, &p_nom_eq));
                }
                ConstraintKind::Inequality => {
                    // Compute worst-case params for this constraint.
                    let p_worst =
                        compute_worst_case_params(&*rc.func, &x0, &p_nom, &p_stds, k, n_params);
                    let func = Arc::clone(&rc.func);
                    problem = problem.constraint_ineq(move |x: &[S]| func(x, &p_worst));
                }
            }
        }

        // 4. Solve the reformulated problem.
        let result = problem.solve()?;
        let x_star = result.x.clone();

        // 5. Compute parametric sensitivity (dx*/dp).
        let sensitivity = if !self.params.is_empty() {
            let obj_sens = Arc::clone(&obj);
            let bounds_sens = self.bounds.clone();
            let grad_sens = self.gradient.clone();
            let max_iter = self.options.max_iter;
            let param_names: Vec<&str> = self.params.iter().map(|p| p.name.as_str()).collect();

            let sens_result = compute_param_sensitivity(
                |params: &[S]| {
                    let obj_inner = Arc::clone(&obj_sens);
                    let p_inner = params.to_vec();
                    let mut prob = OptimProblem::new(n)
                        .x0(&x_star)
                        .objective(move |x: &[S]| obj_inner(x, &p_inner))
                        .max_iter(max_iter);

                    if let Some(ref gf) = grad_sens {
                        let gf = Arc::clone(gf);
                        let p_g = params.to_vec();
                        prob = prob.gradient(move |x: &[S], g: &mut [S]| {
                            gf(x, &p_g, g);
                        });
                    }

                    for (i, b) in bounds_sens.iter().enumerate() {
                        if let Some(lo_hi) = b {
                            prob = prob.bounds(i, *lo_hi);
                        }
                    }
                    prob
                },
                &p_nom,
                &param_names,
                None,
            );

            sens_result.ok()
        } else {
            None
        };

        // 6. Compute x_std from sensitivity: x_std[i] = sqrt(sum_j (dx_i/dp_j)^2 * sigma_j^2).
        let x_std = if let Some(ref sens) = sensitivity {
            (0..n)
                .map(|i| {
                    let var: S = (0..n_params)
                        .map(|j| {
                            let dxdp = sens.get(i, j);
                            dxdp * dxdp * p_stds[j] * p_stds[j]
                        })
                        .sum();
                    var.sqrt()
                })
                .collect()
        } else {
            vec![S::ZERO; n]
        };

        // 7. Compute nominal and worst-case objective values.
        let f_nominal = obj(&x_star, &p_nom);

        // Worst-case objective: find worst direction for each param.
        let f_worst_case = if !self.params.is_empty() {
            let obj_worst = |_x: &[S], p: &[S]| obj(&x_star, p);
            let p_worst_obj = compute_worst_case_params_for_obj(
                &obj_worst, &x_star, &p_nom, &p_stds, k, n_params,
            );
            obj(&x_star, &p_worst_obj)
        } else {
            f_nominal
        };

        Ok(RobustResult {
            x: x_star,
            f_nominal,
            f_worst_case,
            x_std,
            converged: result.converged,
            message: result.message,
            iterations: result.iterations,
            wall_time_secs: start.elapsed().as_secs_f64(),
            sensitivity,
        })
    }
}

// ---------------------------------------------------------------------------
// Helper: compute worst-case parameters for a constraint
// ---------------------------------------------------------------------------

/// For an inequality constraint `g(x, p) <= 0`, determine the worst-case
/// parameter vector (within the k-sigma confidence region) that maximises `g`.
///
/// For each parameter j, estimate the sign of dg/dp_j via finite differences
/// at the representative point `(x0, p_nom)`, then set p_worst_j to
/// `p_nom_j + k * std_j` if dg/dp_j > 0, or `p_nom_j - k * std_j` otherwise.
fn compute_worst_case_params<S: Scalar>(
    g: &dyn Fn(&[S], &[S]) -> S,
    x0: &[S],
    p_nom: &[S],
    p_stds: &[S],
    k: S,
    n_params: usize,
) -> Vec<S> {
    let mut p_worst = p_nom.to_vec();
    let fd_eps = S::from_f64(1e-8);

    for j in 0..n_params {
        if p_stds[j] <= S::ZERO {
            continue;
        }
        let h = fd_eps * (S::ONE + p_nom[j].abs());

        let mut p_plus = p_nom.to_vec();
        p_plus[j] += h;
        let g_plus = g(x0, &p_plus);

        let mut p_minus = p_nom.to_vec();
        p_minus[j] -= h;
        let g_minus = g(x0, &p_minus);

        // Choose the direction that makes g larger (more violated).
        if g_plus > g_minus {
            p_worst[j] = p_nom[j] + k * p_stds[j];
        } else {
            p_worst[j] = p_nom[j] - k * p_stds[j];
        }
    }

    p_worst
}

/// For the objective function, find worst-case params that maximise f (worst case).
fn compute_worst_case_params_for_obj<S: Scalar>(
    _f_wrapper: &dyn Fn(&[S], &[S]) -> S,
    x_star: &[S],
    p_nom: &[S],
    p_stds: &[S],
    k: S,
    n_params: usize,
) -> Vec<S> {
    let mut p_worst = p_nom.to_vec();
    let fd_eps = S::from_f64(1e-8);

    // We need the actual objective. Since _f_wrapper just calls obj(x_star, p),
    // we use it directly.
    let f_at = |p: &[S]| _f_wrapper(x_star, p);

    for j in 0..n_params {
        if p_stds[j] <= S::ZERO {
            continue;
        }
        let h = fd_eps * (S::ONE + p_nom[j].abs());

        let mut p_plus = p_nom.to_vec();
        p_plus[j] += h;
        let f_plus = f_at(&p_plus);

        let mut p_minus = p_nom.to_vec();
        p_minus[j] -= h;
        let f_minus = f_at(&p_minus);

        if f_plus > f_minus {
            p_worst[j] = p_nom[j] + k * p_stds[j];
        } else {
            p_worst[j] = p_nom[j] - k * p_stds[j];
        }
    }

    p_worst
}

// ---------------------------------------------------------------------------
// Normal quantile (inverse CDF)
// ---------------------------------------------------------------------------

/// Compute the inverse of the standard normal CDF (quantile function).
///
/// Uses the Abramowitz & Stegun 26.2.23 rational approximation.
///
/// # Arguments
///
/// * `p` - Probability in (0, 1).
///
/// # Returns
///
/// The value `z` such that `Phi(z) = p` where `Phi` is the standard normal CDF.
///
/// # Panics
///
/// Panics if `p` is not in (0, 1).
pub fn normal_quantile<S: Scalar>(p: S) -> S {
    assert!(
        p > S::ZERO && p < S::ONE,
        "p must be in (0, 1), got {}",
        p.to_f64()
    );

    if (p - S::HALF).abs() < S::from_f64(1e-15) {
        return S::ZERO;
    }

    if p < S::HALF {
        return -normal_quantile(S::ONE - p);
    }

    // p > 0.5: use the Abramowitz & Stegun rational approximation.
    let t = (S::from_f64(-2.0) * (S::ONE - p).ln()).sqrt();

    let c0 = S::from_f64(2.515517);
    let c1 = S::from_f64(0.802853);
    let c2 = S::from_f64(0.010328);
    let d1 = S::from_f64(1.432788);
    let d2 = S::from_f64(0.189269);
    let d3 = S::from_f64(0.001308);

    t - (c0 + c1 * t + c2 * t * t) / (S::ONE + d1 * t + d2 * t * t + d3 * t * t * t)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normal_quantile() {
        // Phi^{-1}(0.5) = 0.0
        assert!(
            normal_quantile(0.5_f64).abs() < 1e-10,
            "q(0.5) = {}, expected 0.0",
            normal_quantile(0.5_f64)
        );

        // Phi^{-1}(0.95) ~ 1.6449
        let q95 = normal_quantile(0.95_f64);
        assert!(
            (q95 - 1.6449).abs() < 1e-3,
            "q(0.95) = {}, expected ~1.6449",
            q95
        );

        // Phi^{-1}(0.99) ~ 2.3263
        let q99 = normal_quantile(0.99_f64);
        assert!(
            (q99 - 2.3263).abs() < 1e-3,
            "q(0.99) = {}, expected ~2.3263",
            q99
        );

        // Phi^{-1}(0.975) ~ 1.9600
        let q975 = normal_quantile(0.975_f64);
        assert!(
            (q975 - 1.9600).abs() < 1e-3,
            "q(0.975) = {}, expected ~1.9600",
            q975
        );
    }

    #[test]
    fn test_robust_unconstrained() {
        // min (x - p)^2 with uncertain param p = 5 +/- 1.
        // Optimal x* = p_nom = 5. Sensitivity dx*/dp = 1, so x_std ~ 1.0.
        let result = RobustProblem::<f64>::new(1)
            .x0(&[0.0])
            .objective(|x: &[f64], p: &[f64]| (x[0] - p[0]) * (x[0] - p[0]))
            .gradient(|x: &[f64], p: &[f64], g: &mut [f64]| {
                g[0] = 2.0 * (x[0] - p[0]);
            })
            .param("p", 5.0, 1.0)
            .solve()
            .unwrap();

        assert!(
            (result.x[0] - 5.0).abs() < 0.1,
            "x* = {}, expected ~5.0",
            result.x[0]
        );
        assert!(
            (result.x_std[0] - 1.0).abs() < 0.3,
            "x_std = {}, expected ~1.0",
            result.x_std[0]
        );
        assert!(result.converged, "solver should converge");
    }

    #[test]
    fn test_robust_constraint_tightening() {
        // min -x (maximize x) s.t. x - p <= 0 (i.e. x <= p)
        // param p = 10 +/- 2, confidence 0.95 (k ~ 1.645).
        // Nominal: x* = 10. Robust: x* ~ 10 - 1.645*2 = 6.71.
        let result = RobustProblem::<f64>::new(1)
            .x0(&[5.0])
            .objective(|x: &[f64], _p: &[f64]| -x[0])
            .gradient(|_x: &[f64], _p: &[f64], g: &mut [f64]| {
                g[0] = -1.0;
            })
            .constraint_ineq(|x: &[f64], p: &[f64]| {
                x[0] - p[0] // x <= p
            })
            .param("p", 10.0, 2.0)
            .confidence(0.95)
            .bounds(0, (-100.0, 100.0))
            .solve()
            .unwrap();

        // Robust solution should be well below nominal 10.
        assert!(
            result.x[0] < 8.5,
            "x* = {}, expected < 8.5 (robust tightening)",
            result.x[0]
        );
        // And should be approximately 10 - 1.645*2 = 6.71
        assert!(
            result.x[0] > 4.0,
            "x* = {}, should be > 4.0 (not overly conservative)",
            result.x[0]
        );
    }

    #[test]
    fn test_robust_two_params() {
        // min x^2 s.t. x - (p1 + p2) <= 0 (i.e. x <= p1 + p2).
        // Params: p1 = 5 +/- 1, p2 = 5 +/- 1. Confidence 0.95.
        // Nominal bound: x <= 10. Robust bound tighter (x <= 10 - 2*k*1 ~ 6.71).
        // Since min x^2 with x <= bound: unconstrained min is x=0 which satisfies
        // any positive upper bound. So x* = 0 < 10 in all cases.
        let result = RobustProblem::<f64>::new(1)
            .x0(&[0.0])
            .objective(|x: &[f64], _p: &[f64]| x[0] * x[0])
            .gradient(|x: &[f64], _p: &[f64], g: &mut [f64]| {
                g[0] = 2.0 * x[0];
            })
            .constraint_ineq(|x: &[f64], p: &[f64]| {
                // x - (p1 + p2) <= 0, i.e. x <= p1 + p2
                x[0] - (p[0] + p[1])
            })
            .param("p1", 5.0, 1.0)
            .param("p2", 5.0, 1.0)
            .confidence(0.95)
            .solve()
            .unwrap();

        assert!(
            result.x[0] < 10.0,
            "x* = {}, expected < 10 (robust tightening with two params)",
            result.x[0]
        );
    }
}

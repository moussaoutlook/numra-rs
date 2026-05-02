//! Declarative optimization problem builder.
//!
//! Author: Moussa Leblouba
//! Date: 5 March 2026
//! Modified: 2 May 2026

use numra_core::Scalar;

use crate::augmented_lagrangian::AugLagOptions;
use crate::error::OptimError;
use crate::global::DEOptions;
use crate::types::{OptimOptions, OptimResult};

/// Boxed scalar function `&[S] -> S`.
pub type ScalarFn<S> = Box<dyn Fn(&[S]) -> S>;
/// Boxed vector function `(&[S], &mut [S])`.
pub type VectorFn<S> = Box<dyn Fn(&[S], &mut [S])>;

/// Variable type for mixed-integer problems.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VarType {
    Continuous,
    Integer,
    Binary,
}

/// Hint about problem structure for closure-based objectives.
///
/// When the objective is provided as a closure (`ObjectiveKind::Minimize`),
/// the auto-selector cannot determine if the underlying function is linear
/// or quadratic. Setting a hint informs solver selection without requiring
/// coefficient extraction.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ProblemHint {
    /// No structural information (default).
    #[default]
    None,
    /// The closure implements a linear function.
    Linear,
    /// The closure implements a quadratic function.
    Quadratic,
}

/// Kind of constraint: equality h(x)=0 or inequality g(x)<=0.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ConstraintKind {
    Equality,
    Inequality,
}

/// A single constraint with optional gradient.
pub struct Constraint<S: Scalar> {
    pub(crate) func: ScalarFn<S>,
    pub(crate) grad: Option<VectorFn<S>>,
    pub(crate) kind: ConstraintKind,
}

/// A linear constraint: a^T x {<=, =} b.
#[derive(Clone, Debug)]
pub struct LinearConstraint<S: Scalar> {
    pub(crate) a: Vec<S>,
    pub(crate) b: S,
    pub(crate) kind: ConstraintKind,
}

/// Whether the objective is a scalar function or a least-squares residual.
pub enum ObjectiveKind<S: Scalar> {
    /// Linear objective: min c^T x
    Linear { c: Vec<S> },
    /// Quadratic objective: min 1/2 x^T H x + c^T x
    /// H stored row-major as `Vec<S>` (n*n), must be positive semi-definite.
    Quadratic {
        h_row_major: Vec<S>,
        c: Vec<S>,
        n: usize,
    },
    Minimize {
        func: ScalarFn<S>,
        grad: Option<VectorFn<S>>,
    },
    LeastSquares {
        residual: VectorFn<S>,
        jacobian: Option<VectorFn<S>>,
        n_residuals: usize,
    },
    /// Multi-objective: minimize [f1(x), f2(x), ...] simultaneously.
    MultiObjective { funcs: Vec<ScalarFn<S>> },
}

/// Declarative optimization problem.
///
/// Build a problem with the fluent API, then call `.solve()` to automatically
/// dispatch to the best solver, or `.solve_with(choice)` for explicit control.
pub struct OptimProblem<S: Scalar> {
    pub(crate) n: usize,
    pub(crate) x0: Option<Vec<S>>,
    pub(crate) bounds: Vec<Option<(S, S)>>,
    pub(crate) var_types: Vec<VarType>,
    pub(crate) objective: Option<ObjectiveKind<S>>,
    pub(crate) constraints: Vec<Constraint<S>>,
    pub(crate) linear_constraints: Vec<LinearConstraint<S>>,
    pub(crate) options: OptimOptions<S>,
    pub(crate) aug_lag_options: Option<AugLagOptions<S>>,
    pub(crate) global: bool,
    pub(crate) de_options: Option<DEOptions<S>>,
    pub(crate) negate_result: bool,
    pub(crate) hint: ProblemHint,
}

impl<S: Scalar> OptimProblem<S> {
    /// Create a new problem with `n` decision variables.
    pub fn new(n: usize) -> Self {
        Self {
            n,
            x0: None,
            bounds: vec![None; n],
            var_types: vec![VarType::Continuous; n],
            objective: None,
            constraints: Vec::new(),
            linear_constraints: Vec::new(),
            options: OptimOptions::default(),
            aug_lag_options: None,
            global: false,
            de_options: None,
            negate_result: false,
            hint: ProblemHint::None,
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

    /// Mark variable `i` as integer-valued.
    pub fn integer_var(mut self, i: usize) -> Self {
        self.var_types[i] = VarType::Integer;
        self
    }

    /// Mark variable `i` as binary (0 or 1).
    pub fn binary_var(mut self, i: usize) -> Self {
        self.var_types[i] = VarType::Binary;
        self.bounds[i] = Some((S::ZERO, S::ONE));
        self
    }

    /// Set a scalar objective function to minimize.
    pub fn objective<F: Fn(&[S]) -> S + 'static>(mut self, f: F) -> Self {
        match &mut self.objective {
            Some(ObjectiveKind::Minimize { func, .. }) => {
                *func = Box::new(f);
            }
            _ => {
                self.objective = Some(ObjectiveKind::Minimize {
                    func: Box::new(f),
                    grad: None,
                });
            }
        }
        self
    }

    /// Set the gradient for the scalar objective.
    pub fn gradient<G: Fn(&[S], &mut [S]) + 'static>(mut self, g: G) -> Self {
        match &mut self.objective {
            Some(ObjectiveKind::Minimize { grad, .. }) => {
                *grad = Some(Box::new(g));
            }
            _ => {
                self.objective = Some(ObjectiveKind::Minimize {
                    func: Box::new(|_| S::ZERO),
                    grad: Some(Box::new(g)),
                });
            }
        }
        self
    }

    /// Set a scalar objective function to maximize.
    ///
    /// Internally converts to minimization by negating the objective and gradient.
    pub fn maximize<F: Fn(&[S]) -> S + 'static>(mut self, f: F) -> Self {
        self.objective = Some(ObjectiveKind::Minimize {
            func: Box::new(move |x: &[S]| -f(x)),
            grad: None,
        });
        self.negate_result = true;
        self
    }

    /// Set the gradient for a maximization objective.
    ///
    /// Note: provide the gradient of the ORIGINAL function (not negated).
    /// The builder handles negation internally.
    pub fn maximize_gradient<G: Fn(&[S], &mut [S]) + 'static>(mut self, g: G) -> Self {
        if let Some(ObjectiveKind::Minimize { grad, .. }) = &mut self.objective {
            if self.negate_result {
                *grad = Some(Box::new(move |x: &[S], gout: &mut [S]| {
                    g(x, gout);
                    for gi in gout.iter_mut() {
                        *gi = -*gi;
                    }
                }));
            }
        }
        self
    }

    /// Set a least-squares objective: minimize ||r(x)||^2.
    pub fn least_squares<R: Fn(&[S], &mut [S]) + 'static>(
        mut self,
        n_residuals: usize,
        residual: R,
    ) -> Self {
        self.objective = Some(ObjectiveKind::LeastSquares {
            residual: Box::new(residual),
            jacobian: None,
            n_residuals,
        });
        self
    }

    /// Set the Jacobian for a least-squares objective.
    pub fn jacobian<J: Fn(&[S], &mut [S]) + 'static>(mut self, j: J) -> Self {
        if let Some(ObjectiveKind::LeastSquares { jacobian, .. }) = &mut self.objective {
            *jacobian = Some(Box::new(j));
        }
        self
    }

    /// Set a linear objective: minimize c^T x.
    pub fn linear_objective(mut self, c: &[S]) -> Self {
        assert_eq!(c.len(), self.n, "linear objective dimension mismatch");
        self.objective = Some(ObjectiveKind::Linear { c: c.to_vec() });
        self
    }

    /// Set a quadratic objective: minimize 1/2 x^T H x + c^T x.
    /// `h_row_major` is the n x n Hessian in row-major order.
    pub fn quadratic_objective_dense(mut self, h_row_major: &[S], c: &[S]) -> Self {
        let n = self.n;
        assert_eq!(h_row_major.len(), n * n, "Hessian dimension mismatch");
        assert_eq!(c.len(), n, "linear term dimension mismatch");
        self.objective = Some(ObjectiveKind::Quadratic {
            h_row_major: h_row_major.to_vec(),
            c: c.to_vec(),
            n,
        });
        self
    }

    /// Add a linear inequality constraint: a^T x <= b.
    pub fn linear_constraint_ineq(mut self, a: &[S], b: S) -> Self {
        assert_eq!(a.len(), self.n, "constraint dimension mismatch");
        self.linear_constraints.push(LinearConstraint {
            a: a.to_vec(),
            b,
            kind: ConstraintKind::Inequality,
        });
        self
    }

    /// Add a linear equality constraint: a^T x = b.
    pub fn linear_constraint_eq(mut self, a: &[S], b: S) -> Self {
        assert_eq!(a.len(), self.n, "constraint dimension mismatch");
        self.linear_constraints.push(LinearConstraint {
            a: a.to_vec(),
            b,
            kind: ConstraintKind::Equality,
        });
        self
    }

    /// Add an inequality constraint g(x) <= 0.
    pub fn constraint_ineq<F: Fn(&[S]) -> S + 'static>(mut self, f: F) -> Self {
        self.constraints.push(Constraint {
            func: Box::new(f),
            grad: None,
            kind: ConstraintKind::Inequality,
        });
        self
    }

    /// Add an equality constraint h(x) = 0.
    pub fn constraint_eq<F: Fn(&[S]) -> S + 'static>(mut self, f: F) -> Self {
        self.constraints.push(Constraint {
            func: Box::new(f),
            grad: None,
            kind: ConstraintKind::Equality,
        });
        self
    }

    /// Add an inequality constraint with gradient.
    pub fn constraint_ineq_with_grad<F, G>(mut self, f: F, g: G) -> Self
    where
        F: Fn(&[S]) -> S + 'static,
        G: Fn(&[S], &mut [S]) + 'static,
    {
        self.constraints.push(Constraint {
            func: Box::new(f),
            grad: Some(Box::new(g)),
            kind: ConstraintKind::Inequality,
        });
        self
    }

    /// Add an equality constraint with gradient.
    pub fn constraint_eq_with_grad<F, G>(mut self, f: F, g: G) -> Self
    where
        F: Fn(&[S]) -> S + 'static,
        G: Fn(&[S], &mut [S]) + 'static,
    {
        self.constraints.push(Constraint {
            func: Box::new(f),
            grad: Some(Box::new(g)),
            kind: ConstraintKind::Equality,
        });
        self
    }

    /// Set maximum iterations.
    pub fn max_iter(mut self, n: usize) -> Self {
        self.options.max_iter = n;
        self
    }

    /// Set gradient tolerance.
    pub fn gtol(mut self, tol: S) -> Self {
        self.options.gtol = tol;
        self
    }

    /// Override all options.
    pub fn options(mut self, opts: OptimOptions<S>) -> Self {
        self.options = opts;
        self
    }

    /// Set augmented Lagrangian options for constrained problems.
    pub fn aug_lag_options(mut self, opts: AugLagOptions<S>) -> Self {
        self.aug_lag_options = Some(opts);
        self
    }

    /// Set the constraint tolerance for constrained problems.
    pub fn ctol(mut self, tol: S) -> Self {
        let opts = self
            .aug_lag_options
            .get_or_insert_with(AugLagOptions::default);
        opts.ctol = tol;
        self
    }

    /// Enable global optimization (Differential Evolution).
    pub fn global(mut self, enabled: bool) -> Self {
        self.global = enabled;
        self
    }

    /// Set Differential Evolution options.
    pub fn de_options(mut self, opts: DEOptions<S>) -> Self {
        self.de_options = Some(opts);
        self
    }

    /// Set multiple objectives for multi-objective optimization (all minimized).
    pub fn multi_objective(mut self, funcs: Vec<ScalarFn<S>>) -> Self {
        self.objective = Some(ObjectiveKind::MultiObjective { funcs });
        self
    }

    /// Provide a structural hint about the closure-based objective.
    ///
    /// This informs auto solver selection when the objective is a closure
    /// that implements a linear or quadratic function but cannot be decomposed
    /// into coefficient vectors.
    pub fn hint(mut self, hint: ProblemHint) -> Self {
        self.hint = hint;
        self
    }

    // ── query methods ──

    /// Returns `true` if any variable has bounds.
    pub fn has_bounds(&self) -> bool {
        self.bounds.iter().any(|b| b.is_some())
    }

    /// Returns `true` if constraints have been added.
    pub fn has_constraints(&self) -> bool {
        !self.constraints.is_empty()
    }

    /// Returns `true` if the objective is a least-squares residual.
    pub fn is_least_squares(&self) -> bool {
        matches!(self.objective, Some(ObjectiveKind::LeastSquares { .. }))
    }

    /// Returns `true` if the objective is structurally linear (coefficient vector).
    pub fn is_linear(&self) -> bool {
        matches!(self.objective, Some(ObjectiveKind::Linear { .. }))
    }

    /// Returns `true` if the objective is structurally quadratic (Hessian + linear).
    pub fn is_quadratic(&self) -> bool {
        matches!(self.objective, Some(ObjectiveKind::Quadratic { .. }))
    }

    /// Returns `true` if a linear hint is set (closure known to be linear).
    pub fn is_hint_linear(&self) -> bool {
        self.hint == ProblemHint::Linear
    }

    /// Returns `true` if a quadratic hint is set (closure known to be quadratic).
    pub fn is_hint_quadratic(&self) -> bool {
        self.hint == ProblemHint::Quadratic
    }

    /// Returns `true` if the objective is multi-objective.
    pub fn is_multi_objective(&self) -> bool {
        matches!(self.objective, Some(ObjectiveKind::MultiObjective { .. }))
    }

    /// Returns `true` if linear constraints have been added.
    pub fn has_linear_constraints(&self) -> bool {
        !self.linear_constraints.is_empty()
    }

    /// Returns `true` if any variable is integer or binary.
    pub fn has_integer_vars(&self) -> bool {
        self.var_types.iter().any(|v| *v != VarType::Continuous)
    }

    /// Number of equality constraints.
    pub fn n_eq_constraints(&self) -> usize {
        self.constraints
            .iter()
            .filter(|c| c.kind == ConstraintKind::Equality)
            .count()
    }

    /// Number of inequality constraints.
    pub fn n_ineq_constraints(&self) -> usize {
        self.constraints
            .iter()
            .filter(|c| c.kind == ConstraintKind::Inequality)
            .count()
    }
}

impl<S: Scalar + faer::SimpleEntity + faer::Conjugate<Canonical = S> + faer::ComplexField>
    OptimProblem<S>
{
    /// Solve using automatic solver selection.
    pub fn solve(self) -> Result<OptimResult<S>, OptimError> {
        crate::auto::auto_minimize(self)
    }

    /// Solve using an explicitly chosen solver.
    pub fn solve_with(
        self,
        choice: crate::auto::SolverChoice,
    ) -> Result<OptimResult<S>, OptimError> {
        crate::auto::dispatch(self, choice)
    }
}

/// Compute a finite-difference gradient of `f` at `x` using central differences.
pub fn finite_diff_gradient<S: Scalar>(f: &dyn Fn(&[S]) -> S, x: &[S], g: &mut [S]) {
    let n = x.len();
    let eps = S::from_f64(1e-8);
    let mut xp = x.to_vec();
    for i in 0..n {
        let xi_orig = xp[i];
        xp[i] = xi_orig + eps;
        let fp = f(&xp);
        xp[i] = xi_orig - eps;
        let fm = f(&xp);
        g[i] = (fp - fm) / (S::TWO * eps);
        xp[i] = xi_orig;
    }
}

/// Compute a finite-difference Jacobian of residual `r` at `x`.
/// `jac` is row-major m*n.
pub fn finite_diff_jacobian<S: Scalar>(
    r: &dyn Fn(&[S], &mut [S]),
    x: &[S],
    m: usize,
    jac: &mut [S],
) {
    let n = x.len();
    let eps = S::from_f64(1e-8);
    let mut xp = x.to_vec();
    let mut rp = vec![S::ZERO; m];
    let mut rm = vec![S::ZERO; m];
    for j in 0..n {
        let xj_orig = xp[j];
        xp[j] = xj_orig + eps;
        r(&xp, &mut rp);
        xp[j] = xj_orig - eps;
        r(&xp, &mut rm);
        for i in 0..m {
            jac[i * n + j] = (rp[i] - rm[i]) / (S::TWO * eps);
        }
        xp[j] = xj_orig;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_builder_construction() {
        let problem = OptimProblem::<f64>::new(2)
            .x0(&[1.0, 2.0])
            .objective(|x: &[f64]| x[0] * x[0] + x[1] * x[1])
            .gradient(|x: &[f64], g: &mut [f64]| {
                g[0] = 2.0 * x[0];
                g[1] = 2.0 * x[1];
            })
            .gtol(1e-10);

        assert!(!problem.has_bounds());
        assert!(!problem.has_constraints());
        assert!(!problem.is_least_squares());
        assert_eq!(problem.n_eq_constraints(), 0);
        assert_eq!(problem.n_ineq_constraints(), 0);
    }

    #[test]
    fn test_builder_with_bounds() {
        let problem = OptimProblem::<f64>::new(3)
            .x0(&[0.0, 0.0, 0.0])
            .objective(|x: &[f64]| x.iter().copied().map(|xi| xi * xi).sum::<f64>())
            .bounds(0, (-1.0, 1.0))
            .bounds(2, (0.0, 10.0));

        assert!(problem.has_bounds());
        assert!(problem.bounds[0].is_some());
        assert!(problem.bounds[1].is_none());
        assert!(problem.bounds[2].is_some());
    }

    #[test]
    fn test_builder_with_constraints() {
        let problem = OptimProblem::<f64>::new(2)
            .x0(&[1.0, 1.0])
            .objective(|x: &[f64]| x[0] + x[1])
            .constraint_eq(|x: &[f64]| x[0] * x[0] + x[1] * x[1] - 1.0)
            .constraint_ineq(|x: &[f64]| x[0] - 0.5);

        assert!(problem.has_constraints());
        assert_eq!(problem.n_eq_constraints(), 1);
        assert_eq!(problem.n_ineq_constraints(), 1);
    }

    #[test]
    fn test_builder_least_squares() {
        let problem = OptimProblem::<f64>::new(2).x0(&[0.0, 0.0]).least_squares(
            3,
            |x: &[f64], r: &mut [f64]| {
                r[0] = x[0] - 1.0;
                r[1] = x[1] - 2.0;
                r[2] = x[0] + x[1] - 3.0;
            },
        );

        assert!(problem.is_least_squares());
    }

    #[test]
    fn test_builder_linear_objective() {
        let p = OptimProblem::<f64>::new(3).linear_objective(&[2.0, 3.0, 1.0]);
        assert!(p.is_linear());
        assert!(!p.is_quadratic());
        assert!(!p.is_least_squares());
    }

    #[test]
    fn test_builder_quadratic_objective() {
        let p = OptimProblem::<f64>::new(2)
            .quadratic_objective_dense(&[2.0, 0.0, 0.0, 4.0], &[1.0, 1.0]);
        assert!(p.is_quadratic());
        assert!(!p.is_linear());
    }

    #[test]
    fn test_builder_linear_constraints() {
        let p = OptimProblem::<f64>::new(2)
            .linear_objective(&[1.0, 2.0])
            .linear_constraint_ineq(&[1.0, 1.0], 10.0)
            .linear_constraint_eq(&[1.0, -1.0], 0.0);
        assert!(p.has_linear_constraints());
        assert_eq!(p.linear_constraints.len(), 2);
    }

    #[test]
    fn test_builder_integer_vars() {
        let p = OptimProblem::<f64>::new(3)
            .linear_objective(&[1.0, 2.0, 3.0])
            .integer_var(0)
            .binary_var(2);
        assert!(p.has_integer_vars());
        assert_eq!(p.var_types[0], VarType::Integer);
        assert_eq!(p.var_types[1], VarType::Continuous);
        assert_eq!(p.var_types[2], VarType::Binary);
        assert_eq!(p.bounds[2], Some((0.0, 1.0)));
    }

    #[test]
    fn test_finite_diff_gradient() {
        let f = |x: &[f64]| x[0] * x[0] + 4.0 * x[1] * x[1];
        let x = [3.0, 2.0];
        let mut g = [0.0; 2];
        finite_diff_gradient(&f, &x, &mut g);
        // df/dx0 = 2*x0 = 6.0, df/dx1 = 8*x1 = 16.0
        assert!((g[0] - 6.0).abs() < 1e-5);
        assert!((g[1] - 16.0).abs() < 1e-5);
    }
}

//! Newton-Raphson solver with optional line search.
//!
//! Solves systems of nonlinear equations F(x) = 0 using Newton's method:
//!   x_{n+1} = x_n - J^{-1} F(x_n)
//!
//! where J = ∂F/∂x is the Jacobian matrix.
//!
//! Author: Moussa Leblouba
//! Date: 5 March 2026
//! Modified: 2 May 2026

use faer::{ComplexField, Conjugate, SimpleEntity};
use numra_core::Scalar;
use numra_linalg::{DenseMatrix, LUFactorization, LinalgError, Matrix};

/// Options for the Newton solver.
#[derive(Clone, Debug)]
pub struct NewtonOptions<S: Scalar> {
    /// Maximum number of iterations.
    pub max_iter: usize,
    /// Absolute tolerance for residual norm.
    pub atol: S,
    /// Relative tolerance for residual norm.
    pub rtol: S,
    /// Enable line search for globalization.
    pub line_search: bool,
    /// Minimum step size in line search.
    pub min_step: S,
    /// Maximum step reduction iterations.
    pub max_backsteps: usize,
    /// Armijo parameter (sufficient decrease).
    pub armijo: S,
}

impl<S: Scalar> Default for NewtonOptions<S> {
    fn default() -> Self {
        Self {
            max_iter: 50,
            atol: S::from_f64(1e-10),
            rtol: S::from_f64(1e-10),
            line_search: true,
            min_step: S::from_f64(1e-4),
            max_backsteps: 10,
            armijo: S::from_f64(1e-4),
        }
    }
}

impl<S: Scalar> NewtonOptions<S> {
    /// Create new options with default values.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set maximum iterations.
    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set absolute tolerance.
    pub fn atol(mut self, atol: S) -> Self {
        self.atol = atol;
        self
    }

    /// Set relative tolerance.
    pub fn rtol(mut self, rtol: S) -> Self {
        self.rtol = rtol;
        self
    }

    /// Enable or disable line search.
    pub fn line_search(mut self, enable: bool) -> Self {
        self.line_search = enable;
        self
    }
}

/// Result of Newton iteration.
#[derive(Clone, Debug)]
pub struct NewtonResult<S: Scalar> {
    /// Solution vector.
    pub x: Vec<S>,
    /// Final residual norm.
    pub residual_norm: S,
    /// Number of iterations performed.
    pub iterations: usize,
    /// Number of function evaluations.
    pub n_eval: usize,
    /// Number of Jacobian evaluations.
    pub n_jac: usize,
    /// Whether the solver converged.
    pub converged: bool,
    /// Convergence message.
    pub message: String,
}

/// Trait for nonlinear systems F(x) = 0.
pub trait NonlinearSystem<S: Scalar>: Send + Sync {
    /// Dimension of the system.
    fn dim(&self) -> usize;

    /// Evaluate F(x) and store result in f.
    fn eval(&self, x: &[S], f: &mut [S]);

    /// Evaluate the Jacobian J = ∂F/∂x.
    /// If None, finite differences will be used.
    fn jacobian(&self, x: &[S], jac: &mut [S]) {
        let _ = (x, jac);
        // Default: not implemented, use finite differences
    }

    /// Whether analytical Jacobian is available.
    fn has_jacobian(&self) -> bool {
        false
    }
}

/// Newton-Raphson solver.
pub struct Newton<S: Scalar> {
    options: NewtonOptions<S>,
}

impl<S: Scalar + SimpleEntity + Conjugate<Canonical = S> + ComplexField> Newton<S> {
    /// Create a new Newton solver with given options.
    pub fn new(options: NewtonOptions<S>) -> Self {
        Self { options }
    }

    /// Create a Newton solver with default options.
    pub fn default_solver() -> Self {
        Self::new(NewtonOptions::default())
    }

    /// Solve the nonlinear system F(x) = 0 starting from x0.
    pub fn solve<Sys: NonlinearSystem<S>>(
        &self,
        system: &Sys,
        x0: &[S],
    ) -> Result<NewtonResult<S>, LinalgError> {
        let n = system.dim();
        if x0.len() != n {
            return Err(LinalgError::DimensionMismatch {
                expected: (n, 1),
                actual: (x0.len(), 1),
            });
        }

        let mut x = x0.to_vec();
        let mut f = vec![S::ZERO; n];
        let mut jac_data = vec![S::ZERO; n * n];
        let mut dx = vec![S::ZERO; n];

        let mut n_eval = 0;
        let mut n_jac = 0;
        let mut converged = false;
        let mut message = String::new();

        // Evaluate initial residual
        system.eval(&x, &mut f);
        n_eval += 1;
        let mut f_norm = self.norm(&f);
        let f0_norm = f_norm;

        for iter in 0..self.options.max_iter {
            // Check convergence
            let tol = self.options.atol + self.options.rtol * f0_norm;
            if f_norm < tol {
                converged = true;
                message = format!("Converged after {} iterations", iter);
                break;
            }

            // Compute Jacobian
            if system.has_jacobian() {
                system.jacobian(&x, &mut jac_data);
            } else {
                self.finite_difference_jacobian(system, &x, &f, &mut jac_data);
                n_eval += n; // n extra evaluations for FD
            }
            n_jac += 1;

            // Build Jacobian matrix and solve J * dx = -f
            let jac_matrix = self.build_matrix(&jac_data, n);
            let lu = LUFactorization::new(&jac_matrix)?;

            // Negate f for solve
            let neg_f: Vec<S> = f.iter().map(|&v| -v).collect();
            dx = lu.solve(&neg_f)?;

            // Update x and f: line search atomically updates both,
            // plain Newton steps and evaluates separately.
            if self.options.line_search {
                self.line_search_update(system, &mut x, &dx, f_norm, &mut f, &mut n_eval);
            } else {
                for i in 0..n {
                    x[i] += dx[i];
                }
                system.eval(&x, &mut f);
                n_eval += 1;
            }
            f_norm = self.norm(&f);
        }

        if !converged {
            message = format!(
                "Did not converge after {} iterations, residual = {:.2e}",
                self.options.max_iter,
                f_norm.to_f64()
            );
        }

        Ok(NewtonResult {
            x,
            residual_norm: f_norm,
            iterations: if converged {
                self.options.max_iter.min(n_jac)
            } else {
                self.options.max_iter
            },
            n_eval,
            n_jac,
            converged,
            message,
        })
    }

    /// Compute 2-norm of a vector.
    fn norm(&self, v: &[S]) -> S {
        let sum: S = v.iter().fold(S::ZERO, |acc, &x| acc + x * x);
        sum.sqrt()
    }

    /// Finite difference Jacobian approximation.
    fn finite_difference_jacobian<Sys: NonlinearSystem<S>>(
        &self,
        system: &Sys,
        x: &[S],
        f0: &[S],
        jac: &mut [S],
    ) {
        let n = system.dim();
        /// Step size for finite difference Jacobian approximation.
        const FD_JACOBIAN_EPS: f64 = 1e-8;
        let eps = S::from_f64(FD_JACOBIAN_EPS);
        let mut x_pert = x.to_vec();
        let mut f_pert = vec![S::ZERO; n];

        for j in 0..n {
            let x_j = x[j];
            let h = eps * (S::ONE + x_j.abs());

            x_pert[j] = x_j + h;
            system.eval(&x_pert, &mut f_pert);
            x_pert[j] = x_j;

            // Column j of Jacobian (row-major storage)
            for i in 0..n {
                jac[i * n + j] = (f_pert[i] - f0[i]) / h;
            }
        }
    }

    /// Build DenseMatrix from row-major data.
    fn build_matrix(&self, data: &[S], n: usize) -> DenseMatrix<S> {
        let mut m = DenseMatrix::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                m.set(i, j, data[i * n + j]);
            }
        }
        m
    }

    /// Armijo line search that atomically updates x and f.
    ///
    /// On return, `x` holds the accepted trial point and `f` holds the
    /// residual at that point.  This eliminates the previous fragile
    /// coupling where `x` and `f` were updated in separate locations.
    fn line_search_update<Sys: NonlinearSystem<S>>(
        &self,
        system: &Sys,
        x: &mut [S],
        dx: &[S],
        f_norm: S,
        f: &mut [S],
        n_eval_count: &mut usize,
    ) {
        let n = x.len();
        let mut step = S::ONE;
        let mut x_trial = vec![S::ZERO; n];
        let mut f_trial = vec![S::ZERO; n];

        let mut best_step = S::ONE;
        let mut best_norm = S::INFINITY;

        for _ in 0..self.options.max_backsteps {
            for i in 0..n {
                x_trial[i] = x[i] + step * dx[i];
            }

            system.eval(&x_trial, &mut f_trial);
            *n_eval_count += 1;

            let f_trial_norm = self.norm(&f_trial);

            if f_trial_norm < best_norm {
                best_norm = f_trial_norm;
                best_step = step;
            }

            if f_trial_norm <= f_norm * (S::ONE - self.options.armijo * step) {
                x.copy_from_slice(&x_trial);
                f.copy_from_slice(&f_trial);
                return;
            }

            step *= S::from_f64(0.5);
            if step < self.options.min_step {
                break;
            }
        }

        // Fall back to best step found
        for i in 0..n {
            x[i] += best_step * dx[i];
        }
        system.eval(x, f);
        *n_eval_count += 1;
    }
}

/// Convenience function to solve F(x) = 0.
pub fn newton_solve<S, Sys>(
    system: &Sys,
    x0: &[S],
    options: Option<NewtonOptions<S>>,
) -> Result<NewtonResult<S>, LinalgError>
where
    S: Scalar + SimpleEntity + Conjugate<Canonical = S> + ComplexField,
    Sys: NonlinearSystem<S>,
{
    let opts = options.unwrap_or_default();
    let solver = Newton::new(opts);
    solver.solve(system, x0)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Simple 1D system: f(x) = x^2 - 2 (roots at ±√2)
    struct SquareRoot;

    impl NonlinearSystem<f64> for SquareRoot {
        fn dim(&self) -> usize {
            1
        }

        fn eval(&self, x: &[f64], f: &mut [f64]) {
            f[0] = x[0] * x[0] - 2.0;
        }

        fn jacobian(&self, x: &[f64], jac: &mut [f64]) {
            jac[0] = 2.0 * x[0];
        }

        fn has_jacobian(&self) -> bool {
            true
        }
    }

    /// 2D system: x^2 + y^2 = 1, x - y = 0 (roots at ±(√2/2, √2/2))
    struct CircleLine;

    impl NonlinearSystem<f64> for CircleLine {
        fn dim(&self) -> usize {
            2
        }

        fn eval(&self, x: &[f64], f: &mut [f64]) {
            f[0] = x[0] * x[0] + x[1] * x[1] - 1.0;
            f[1] = x[0] - x[1];
        }

        fn jacobian(&self, x: &[f64], jac: &mut [f64]) {
            // Row-major: [df1/dx, df1/dy, df2/dx, df2/dy]
            jac[0] = 2.0 * x[0];
            jac[1] = 2.0 * x[1];
            jac[2] = 1.0;
            jac[3] = -1.0;
        }

        fn has_jacobian(&self) -> bool {
            true
        }
    }

    /// Rosenbrock function minimum: f1 = 1-x, f2 = 10(y-x^2)
    struct Rosenbrock;

    impl NonlinearSystem<f64> for Rosenbrock {
        fn dim(&self) -> usize {
            2
        }

        fn eval(&self, x: &[f64], f: &mut [f64]) {
            f[0] = 1.0 - x[0];
            f[1] = 10.0 * (x[1] - x[0] * x[0]);
        }

        fn jacobian(&self, x: &[f64], jac: &mut [f64]) {
            jac[0] = -1.0;
            jac[1] = 0.0;
            jac[2] = -20.0 * x[0];
            jac[3] = 10.0;
        }

        fn has_jacobian(&self) -> bool {
            true
        }
    }

    #[test]
    fn test_newton_sqrt2() {
        let system = SquareRoot;
        let options = NewtonOptions::default();
        let solver = Newton::new(options);

        let result = solver.solve(&system, &[1.5]).unwrap();

        assert!(result.converged);
        assert!((result.x[0] - 2.0_f64.sqrt()).abs() < 1e-10);
    }

    #[test]
    fn test_newton_sqrt2_negative() {
        let system = SquareRoot;
        let solver = Newton::default_solver();

        let result = solver.solve(&system, &[-1.5]).unwrap();

        assert!(result.converged);
        assert!((result.x[0] + 2.0_f64.sqrt()).abs() < 1e-10);
    }

    #[test]
    fn test_newton_circle_line() {
        let system = CircleLine;
        let solver = Newton::default_solver();

        let result = solver.solve(&system, &[0.5, 0.5]).unwrap();

        assert!(result.converged);
        let expected = (0.5_f64).sqrt();
        assert!((result.x[0] - expected).abs() < 1e-10);
        assert!((result.x[1] - expected).abs() < 1e-10);
    }

    #[test]
    fn test_newton_rosenbrock() {
        let system = Rosenbrock;
        let options = NewtonOptions::default().line_search(true);
        let solver = Newton::new(options);

        let result = solver.solve(&system, &[-0.5, 0.5]).unwrap();

        assert!(result.converged);
        assert!((result.x[0] - 1.0).abs() < 1e-8);
        assert!((result.x[1] - 1.0).abs() < 1e-8);
    }

    #[test]
    fn test_newton_finite_difference() {
        // Same as SquareRoot but without analytical Jacobian
        struct SquareRootFD;

        impl NonlinearSystem<f64> for SquareRootFD {
            fn dim(&self) -> usize {
                1
            }
            fn eval(&self, x: &[f64], f: &mut [f64]) {
                f[0] = x[0] * x[0] - 2.0;
            }
        }

        let system = SquareRootFD;
        let solver = Newton::default_solver();

        let result = solver.solve(&system, &[1.5]).unwrap();

        assert!(result.converged);
        assert!((result.x[0] - 2.0_f64.sqrt()).abs() < 1e-8);
        assert!(result.n_eval > result.n_jac); // FD uses more evals
    }

    #[test]
    fn test_newton_convenience_function() {
        let system = SquareRoot;
        let result = newton_solve(&system, &[1.5], None).unwrap();

        assert!(result.converged);
        assert!((result.x[0] - 2.0_f64.sqrt()).abs() < 1e-10);
    }

    #[test]
    fn test_newton_no_line_search() {
        let system = Rosenbrock;
        let options = NewtonOptions::default().line_search(false);
        let solver = Newton::new(options);

        // Without line search, may still converge for good initial guess
        let result = solver.solve(&system, &[0.9, 0.9]).unwrap();

        assert!(result.converged);
    }

    // ============================================================================
    // f32 Scalar Tests
    // ============================================================================

    #[test]
    fn test_newton_sqrt2_f32() {
        struct SquareRootF32;
        impl NonlinearSystem<f32> for SquareRootF32 {
            fn dim(&self) -> usize {
                1
            }
            fn eval(&self, x: &[f32], f: &mut [f32]) {
                f[0] = x[0] * x[0] - 2.0;
            }
            fn jacobian(&self, x: &[f32], jac: &mut [f32]) {
                jac[0] = 2.0 * x[0];
            }
            fn has_jacobian(&self) -> bool {
                true
            }
        }

        // f32 has ~7 digits of precision, so use appropriate tolerances
        let options: NewtonOptions<f32> = NewtonOptions::default().atol(1e-6).rtol(1e-6);
        let solver = Newton::new(options);
        let result = solver.solve(&SquareRootF32, &[1.5f32]).unwrap();

        assert!(result.converged);
        assert!((result.x[0] - 2.0f32.sqrt()).abs() < 1e-5);
    }

    // ============================================================================
    // NaN / Infinity Tests
    // ============================================================================

    #[test]
    fn test_newton_nan_initial_guess() {
        let result = Newton::default_solver().solve(&SquareRoot, &[f64::NAN]);

        // NaN input should either fail or not converge
        match result {
            Ok(r) => assert!(!r.converged || r.x[0].is_nan()),
            Err(_) => {} // Also acceptable
        }
    }

    #[test]
    fn test_newton_dimension_mismatch() {
        let result = Newton::default_solver().solve(&SquareRoot, &[1.0, 2.0]);
        assert!(result.is_err());
    }
}

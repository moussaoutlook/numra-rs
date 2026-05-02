//! BFGS quasi-Newton optimizer.
//!
//! Implements the Broyden-Fletcher-Goldfarb-Shanno (BFGS) algorithm for
//! unconstrained minimization. The method maintains an approximation to the
//! inverse Hessian matrix and updates it using rank-2 corrections derived
//! from gradient differences.
//!
//! Line search is performed using the strong Wolfe conditions via
//! [`numra_nonlinear::line_search::wolfe_line_search`].
//!
//! Author: Moussa Leblouba
//! Date: 8 February 2026
//! Modified: 2 May 2026

use numra_core::Scalar;

use crate::error::OptimError;
use crate::types::{IterationRecord, OptimOptions, OptimResult, OptimStatus};
use numra_nonlinear::line_search::{wolfe_line_search, WolfeOptions};

/// Inner product of two slices.
fn dot<S: Scalar>(a: &[S], b: &[S]) -> S {
    a.iter().zip(b.iter()).map(|(ai, bi)| *ai * *bi).sum()
}

/// Euclidean norm of a slice.
fn norm<S: Scalar>(a: &[S]) -> S {
    dot(a, a).sqrt()
}

/// BFGS quasi-Newton optimizer (struct API).
///
/// # Example
///
/// ```
/// use numra_optim::{Bfgs, OptimOptions};
///
/// let bfgs = Bfgs::new(OptimOptions::default().gtol(1e-10));
/// let f = |x: &[f64]| x[0] * x[0] + x[1] * x[1];
/// let g = |x: &[f64], grad: &mut [f64]| { grad[0] = 2.0 * x[0]; grad[1] = 2.0 * x[1]; };
/// let result = bfgs.minimize(f, g, &[3.0, 4.0]).unwrap();
/// assert!(result.converged);
/// ```
pub struct Bfgs<S: Scalar> {
    pub options: OptimOptions<S>,
}

impl<S: Scalar> Bfgs<S> {
    /// Create a new BFGS optimizer with the given options.
    pub fn new(options: OptimOptions<S>) -> Self {
        Self { options }
    }

    /// Minimize `f` starting from `x0`.
    ///
    /// `grad` writes the gradient of `f` at `x` into the provided buffer.
    pub fn minimize<F, G>(&self, f: F, grad: G, x0: &[S]) -> Result<OptimResult<S>, OptimError>
    where
        F: Fn(&[S]) -> S,
        G: Fn(&[S], &mut [S]),
    {
        bfgs_minimize(f, grad, x0, &self.options)
    }
}

/// Minimize `f` using the BFGS quasi-Newton method.
///
/// # Arguments
///
/// * `f` - Objective function mapping `&[S]` to `S`.
/// * `grad` - Gradient of `f`. Takes `(x, g)` and writes the gradient into `g`.
/// * `x0` - Initial guess.
/// * `opts` - Optimizer options (tolerances, max iterations, etc.).
///
/// # Returns
///
/// An [`OptimResult`] containing the minimizer, minimum value, final gradient,
/// iteration count, and convergence status.
pub fn bfgs_minimize<S: Scalar, F, G>(
    f: F,
    grad: G,
    x0: &[S],
    opts: &OptimOptions<S>,
) -> Result<OptimResult<S>, OptimError>
where
    F: Fn(&[S]) -> S,
    G: Fn(&[S], &mut [S]),
{
    let start = std::time::Instant::now();
    let n = x0.len();
    if n == 0 {
        return Err(OptimError::DimensionMismatch {
            expected: 1,
            actual: 0,
        });
    }

    // Current iterate
    let mut x = x0.to_vec();
    let mut f_val = f(&x);
    let mut n_feval: usize = 1;
    let mut n_geval: usize = 0;

    // Current gradient
    let mut g = vec![S::ZERO; n];
    grad(&x, &mut g);
    n_geval += 1;

    // Check if already at minimum
    let g_norm = norm(&g);
    if g_norm <= opts.gtol {
        return Ok(OptimResult::unconstrained(
            x,
            f_val,
            g,
            0,
            n_feval,
            n_geval,
            true,
            "gradient norm below tolerance at start".into(),
            OptimStatus::GradientConverged,
        )
        .with_wall_time(start));
    }

    // Initialize inverse Hessian approximation as identity (row-major flat vec)
    let mut h_inv = vec![S::ZERO; n * n];
    for i in 0..n {
        h_inv[i * n + i] = S::ONE;
    }

    let mut history = Vec::new();

    // Wolfe line search options (quasi-Newton defaults)
    let wolfe_opts = WolfeOptions::default();

    // Working buffers
    let mut d = vec![S::ZERO; n]; // search direction
    let mut g_new = vec![S::ZERO; n];
    let mut s = vec![S::ZERO; n]; // step: x_new - x
    let mut y = vec![S::ZERO; n]; // gradient difference: g_new - g
    let mut hy = vec![S::ZERO; n]; // H_inv * y

    for iter in 0..opts.max_iter {
        // Compute search direction: d = -H_inv * g
        for i in 0..n {
            let mut sum = S::ZERO;
            for j in 0..n {
                sum += h_inv[i * n + j] * g[j];
            }
            d[i] = -sum;
        }

        // Wolfe line search
        let ls_result = wolfe_line_search(&f, &grad, &x, &d, f_val, &g, &wolfe_opts)?;
        let alpha = ls_result.step;
        let f_new = ls_result.f_new;
        n_feval += ls_result.n_eval;

        // Compute step s = alpha * d and update x
        for i in 0..n {
            s[i] = alpha * d[i];
            x[i] += s[i];
        }

        // Evaluate gradient at new point
        grad(&x, &mut g_new);
        n_geval += 1;

        // Compute gradient difference y = g_new - g
        for i in 0..n {
            y[i] = g_new[i] - g[i];
        }

        // Check convergence: gradient norm
        let g_new_norm = norm(&g_new);

        history.push(IterationRecord {
            iteration: iter,
            objective: f_new,
            gradient_norm: g_new_norm,
            step_size: alpha,
            constraint_violation: S::ZERO,
        });

        if g_new_norm <= opts.gtol {
            g.copy_from_slice(&g_new);
            return Ok((OptimResult {
                history,
                ..OptimResult::unconstrained(
                    x,
                    f_new,
                    g,
                    iter + 1,
                    n_feval,
                    n_geval,
                    true,
                    "gradient norm below tolerance".into(),
                    OptimStatus::GradientConverged,
                )
            })
            .with_wall_time(start));
        }

        // Check convergence: function value change
        let f_change = (f_new - f_val).abs();
        if f_change <= opts.ftol * (S::ONE + f_val.abs()) {
            g.copy_from_slice(&g_new);
            return Ok((OptimResult {
                history,
                ..OptimResult::unconstrained(
                    x,
                    f_new,
                    g,
                    iter + 1,
                    n_feval,
                    n_geval,
                    true,
                    "function change below tolerance".into(),
                    OptimStatus::FunctionConverged,
                )
            })
            .with_wall_time(start));
        }

        // Check convergence: step size
        let s_norm = norm(&s);
        let x_norm = norm(&x);
        if s_norm <= opts.xtol * (S::ONE + x_norm) {
            g.copy_from_slice(&g_new);
            return Ok((OptimResult {
                history,
                ..OptimResult::unconstrained(
                    x,
                    f_new,
                    g,
                    iter + 1,
                    n_feval,
                    n_geval,
                    true,
                    "step size below tolerance".into(),
                    OptimStatus::StepConverged,
                )
            })
            .with_wall_time(start));
        }

        // BFGS update of inverse Hessian
        let sy = dot(&s, &y);
        if sy > S::from_f64(1e-16) {
            let rho = S::ONE / sy;

            // Compute hy = H_inv * y
            for i in 0..n {
                let mut sum = S::ZERO;
                for j in 0..n {
                    sum += h_inv[i * n + j] * y[j];
                }
                hy[i] = sum;
            }

            let yhy = dot(&y, &hy);

            // Sherman-Morrison-Woodbury BFGS update:
            // H_new = (I - rho*s*y^T) * H * (I - rho*y*s^T) + rho*s*s^T
            // Expanded: H_new = H + rho*(1 + rho*y^T*H*y)*s*s^T - rho*(H*y*s^T + s*y^T*H)
            for i in 0..n {
                for j in 0..n {
                    h_inv[i * n + j] +=
                        rho * ((S::ONE + rho * yhy) * s[i] * s[j] - hy[i] * s[j] - s[i] * hy[j]);
                }
            }
        }

        // Update state for next iteration
        f_val = f_new;
        g.copy_from_slice(&g_new);
    }

    Ok((OptimResult {
        history,
        ..OptimResult::unconstrained(
            x,
            f_val,
            g,
            opts.max_iter,
            n_feval,
            n_geval,
            false,
            "maximum iterations reached".into(),
            OptimStatus::MaxIterations,
        )
    })
    .with_wall_time(start))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_bfgs_quadratic() {
        // f(x) = x0^2 + 4*x1^2, minimum at (0, 0)
        let f = |x: &[f64]| x[0] * x[0] + 4.0 * x[1] * x[1];
        let grad = |x: &[f64], g: &mut [f64]| {
            g[0] = 2.0 * x[0];
            g[1] = 8.0 * x[1];
        };

        let x0 = [5.0, 3.0];
        let opts = OptimOptions::default().gtol(1e-10);
        let result = bfgs_minimize(f, grad, &x0, &opts).unwrap();

        assert!(result.converged, "should converge: {}", result.message);
        assert_abs_diff_eq!(result.x[0], 0.0, epsilon = 1e-8);
        assert_abs_diff_eq!(result.x[1], 0.0, epsilon = 1e-8);
        assert_abs_diff_eq!(result.f, 0.0, epsilon = 1e-14);
        assert!(
            result.wall_time_secs > 0.0,
            "wall_time_secs should be positive"
        );
    }

    #[test]
    fn test_bfgs_rosenbrock() {
        // f(x) = (1 - x0)^2 + 100*(x1 - x0^2)^2, minimum at (1, 1)
        let f = |x: &[f64]| {
            let a = 1.0 - x[0];
            let b = x[1] - x[0] * x[0];
            a * a + 100.0 * b * b
        };
        let grad = |x: &[f64], g: &mut [f64]| {
            g[0] = -2.0 * (1.0 - x[0]) - 400.0 * x[0] * (x[1] - x[0] * x[0]);
            g[1] = 200.0 * (x[1] - x[0] * x[0]);
        };

        let x0 = [-1.0, 1.0];
        let opts = OptimOptions::default().gtol(1e-8).max_iter(2000);
        let result = bfgs_minimize(f, grad, &x0, &opts).unwrap();

        assert!(result.converged, "should converge: {}", result.message);
        assert_abs_diff_eq!(result.x[0], 1.0, epsilon = 1e-5);
        assert_abs_diff_eq!(result.x[1], 1.0, epsilon = 1e-5);
    }

    #[test]
    fn test_bfgs_struct_api() {
        // Simple quadratic via the Bfgs struct API
        let f = |x: &[f64]| x[0] * x[0] + x[1] * x[1];
        let grad = |x: &[f64], g: &mut [f64]| {
            g[0] = 2.0 * x[0];
            g[1] = 2.0 * x[1];
        };

        let bfgs = Bfgs::new(OptimOptions::default().gtol(1e-10));
        let result = bfgs.minimize(f, grad, &[3.0, 4.0]).unwrap();

        assert!(result.converged);
        assert_abs_diff_eq!(result.x[0], 0.0, epsilon = 1e-8);
        assert_abs_diff_eq!(result.x[1], 0.0, epsilon = 1e-8);
        assert!(result.iterations > 0);
        assert!(result.n_feval > 0);
        assert!(result.n_geval > 0);
    }

    #[test]
    fn test_bfgs_history() {
        let result = bfgs_minimize(
            |x: &[f64]| x[0] * x[0] + x[1] * x[1],
            |x: &[f64], g: &mut [f64]| {
                g[0] = 2.0 * x[0];
                g[1] = 2.0 * x[1];
            },
            &[5.0, 3.0],
            &OptimOptions::default(),
        )
        .unwrap();
        assert!(!result.history.is_empty(), "history should not be empty");
        // Objective should decrease
        for w in result.history.windows(2) {
            assert!(
                w[1].objective <= w[0].objective + 1e-10,
                "objective should decrease: {} -> {}",
                w[0].objective,
                w[1].objective
            );
        }
    }

    #[test]
    fn test_bfgs_high_dim() {
        // f(x) = sum(x_i^2), n=20, start from [1, 2, ..., 20]
        let n = 20;
        let f = |x: &[f64]| x.iter().copied().map(|xi| xi * xi).sum::<f64>();
        let grad = |x: &[f64], g: &mut [f64]| {
            for i in 0..x.len() {
                g[i] = 2.0 * x[i];
            }
        };

        let x0: Vec<f64> = (1..=n).map(|i| i as f64).collect();
        let opts = OptimOptions::default().gtol(1e-10);
        let result = bfgs_minimize(f, grad, &x0, &opts).unwrap();

        assert!(result.converged, "should converge: {}", result.message);
        for i in 0..n {
            assert_abs_diff_eq!(result.x[i], 0.0, epsilon = 1e-6);
        }
        assert_abs_diff_eq!(result.f, 0.0, epsilon = 1e-10);
    }
}

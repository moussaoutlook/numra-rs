//! L-BFGS-B: bound-constrained limited-memory BFGS.
//!
//! Implements a projected L-BFGS method for minimization with simple box
//! constraints on variables: `lo_i <= x_i <= hi_i`.
//!
//! Author: Moussa Leblouba
//! Date: 8 February 2026
//! Modified: 2 May 2026

use numra_core::Scalar;

use crate::error::OptimError;
use crate::lbfgs::two_loop_recursion;
use crate::types::{IterationRecord, OptimOptions, OptimResult, OptimStatus};
use std::collections::VecDeque;

/// Options for L-BFGS-B.
#[derive(Clone, Debug)]
pub struct LbfgsBOptions<S: Scalar> {
    pub base: OptimOptions<S>,
    pub memory: usize,
}

impl<S: Scalar> Default for LbfgsBOptions<S> {
    fn default() -> Self {
        Self {
            base: OptimOptions::default(),
            memory: 10,
        }
    }
}

impl<S: Scalar> LbfgsBOptions<S> {
    pub fn memory(mut self, m: usize) -> Self {
        self.memory = m;
        self
    }
    pub fn max_iter(mut self, n: usize) -> Self {
        self.base.max_iter = n;
        self
    }
    pub fn gtol(mut self, tol: S) -> Self {
        self.base.gtol = tol;
        self
    }
}

/// Project `x` onto the feasible box defined by `bounds`.
pub fn project<S: Scalar>(x: &mut [S], bounds: &[Option<(S, S)>]) {
    for (xi, bi) in x.iter_mut().zip(bounds.iter()) {
        if let Some((lo, hi)) = bi {
            *xi = xi.clamp(*lo, *hi);
        }
    }
}

/// Compute the projected gradient norm: ||P(x - g) - x||_inf.
pub fn projected_gradient_norm<S: Scalar>(x: &[S], g: &[S], bounds: &[Option<(S, S)>]) -> S {
    let mut norm = S::ZERO;
    for i in 0..x.len() {
        let mut xi_step = x[i] - g[i];
        if let Some((lo, hi)) = bounds[i] {
            xi_step = xi_step.clamp(lo, hi);
        }
        norm = norm.max((xi_step - x[i]).abs());
    }
    norm
}

/// Minimize `f` subject to box constraints using projected L-BFGS.
///
/// Uses the L-BFGS two-loop recursion for search direction and
/// projected Armijo backtracking for step acceptance.
pub fn lbfgsb_minimize<S: Scalar, F, G>(
    f: F,
    grad: G,
    x0: &[S],
    bounds: &[Option<(S, S)>],
    opts: &LbfgsBOptions<S>,
) -> Result<OptimResult<S>, OptimError>
where
    F: Fn(&[S]) -> S,
    G: Fn(&[S], &mut [S]),
{
    let start = std::time::Instant::now();
    let n = x0.len();
    if n != bounds.len() {
        return Err(OptimError::DimensionMismatch {
            expected: n,
            actual: bounds.len(),
        });
    }

    let m = opts.memory;
    let mut x = x0.to_vec();
    project(&mut x, bounds);

    let mut g_buf = vec![S::ZERO; n];
    let mut fval = f(&x);
    grad(&x, &mut g_buf);
    let mut n_feval = 1_usize;
    let mut n_geval = 1_usize;

    let mut s_hist: VecDeque<Vec<S>> = VecDeque::with_capacity(m);
    let mut y_hist: VecDeque<Vec<S>> = VecDeque::with_capacity(m);
    let mut rho_hist: VecDeque<S> = VecDeque::with_capacity(m);

    let c1 = S::from_f64(1e-4);
    let mut f_prev;
    let mut history = Vec::new();

    for iter in 0..opts.base.max_iter {
        f_prev = fval;
        let pg_norm = projected_gradient_norm(&x, &g_buf, bounds);
        if pg_norm < opts.base.gtol {
            let active = active_bounds_at(&x, bounds);
            return Ok((OptimResult {
                active_bounds: active,
                history,
                ..OptimResult::unconstrained(
                    x,
                    fval,
                    g_buf,
                    iter,
                    n_feval,
                    n_geval,
                    true,
                    format!(
                        "Converged: projected gradient norm {:.2e}",
                        pg_norm.to_f64()
                    ),
                    OptimStatus::GradientConverged,
                )
            })
            .with_wall_time(start));
        }

        // Compute L-BFGS direction d = -H*g (already negated by two_loop_recursion)
        let d = two_loop_recursion::<S>(&g_buf, &s_hist, &y_hist, &rho_hist);

        // Projected backtracking line search along d
        // Find alpha such that f(P(x + alpha*d)) <= f(x) + c1 * g^T * (P(x+alpha*d) - x)
        let mut alpha = S::ONE;
        let mut x_new = vec![S::ZERO; n];
        let mut f_new;
        let mut accepted = false;

        for _ls in 0..30 {
            for i in 0..n {
                x_new[i] = x[i] + alpha * d[i];
            }
            project(&mut x_new, bounds);

            f_new = f(&x_new);
            n_feval += 1;

            // Directional derivative along projected step
            let dg: S = g_buf
                .iter()
                .zip(x_new.iter().zip(x.iter()))
                .map(|(gi, (xn, xo))| *gi * (*xn - *xo))
                .sum();

            if f_new <= fval + c1 * dg {
                accepted = true;
                fval = f_new;
                break;
            }

            alpha *= S::from_f64(0.5);
        }

        if !accepted {
            // Fall back to projected gradient step with backtracking
            alpha = S::ONE;
            for _ls in 0..30 {
                for i in 0..n {
                    x_new[i] = x[i] - alpha * g_buf[i];
                }
                project(&mut x_new, bounds);

                f_new = f(&x_new);
                n_feval += 1;

                let dg: S = g_buf
                    .iter()
                    .zip(x_new.iter().zip(x.iter()))
                    .map(|(gi, (xn, xo))| *gi * (*xn - *xo))
                    .sum();

                if f_new <= fval + c1 * dg {
                    fval = f_new;
                    accepted = true;
                    break;
                }
                alpha *= S::from_f64(0.5);
            }

            if !accepted {
                // Take the smallest projected gradient step anyway
                fval = f(&x_new);
                n_feval += 1;
            }
        }

        // Compute step
        let s: Vec<S> = x_new.iter().zip(x.iter()).map(|(a, b)| *a - *b).collect();
        let s_norm: S = s.iter().copied().map(|si| si * si).sum::<S>().sqrt();

        history.push(IterationRecord {
            iteration: iter,
            objective: fval,
            gradient_norm: pg_norm,
            step_size: alpha,
            constraint_violation: S::ZERO,
        });

        // Check step convergence
        let x_norm: S = x.iter().copied().map(|xi| xi * xi).sum::<S>().sqrt();
        if s_norm < opts.base.xtol * (S::ONE + x_norm) {
            let active = active_bounds_at(&x_new, bounds);
            grad(&x_new, &mut g_buf);
            n_geval += 1;
            return Ok((OptimResult {
                active_bounds: active,
                history,
                ..OptimResult::unconstrained(
                    x_new,
                    fval,
                    g_buf,
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

        // Check function convergence
        let f_change = (f_prev - fval).abs();
        if f_change < opts.base.ftol * (S::ONE + f_prev.abs()) && iter > 0 {
            grad(&x_new, &mut g_buf);
            n_geval += 1;
            let active = active_bounds_at(&x_new, bounds);
            return Ok((OptimResult {
                active_bounds: active,
                history,
                ..OptimResult::unconstrained(
                    x_new,
                    fval,
                    g_buf,
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

        x = x_new;

        // New gradient
        let mut g_new = vec![S::ZERO; n];
        grad(&x, &mut g_new);
        n_geval += 1;

        // L-BFGS update
        let y: Vec<S> = g_new
            .iter()
            .zip(g_buf.iter())
            .map(|(gn, go)| *gn - *go)
            .collect();
        let sy: S = s.iter().zip(y.iter()).map(|(si, yi)| *si * *yi).sum();

        if sy > S::from_f64(1e-16) {
            if s_hist.len() == m {
                s_hist.pop_front();
                y_hist.pop_front();
                rho_hist.pop_front();
            }
            rho_hist.push_back(S::ONE / sy);
            s_hist.push_back(s);
            y_hist.push_back(y);
        }

        g_buf = g_new;
    }

    let active = active_bounds_at(&x, bounds);
    Ok((OptimResult {
        active_bounds: active,
        history,
        ..OptimResult::unconstrained(
            x,
            fval,
            g_buf,
            opts.base.max_iter,
            n_feval,
            n_geval,
            false,
            format!("Maximum iterations ({}) reached", opts.base.max_iter),
            OptimStatus::MaxIterations,
        )
    })
    .with_wall_time(start))
}

/// Return indices of variables that are at their bounds.
fn active_bounds_at<S: Scalar>(x: &[S], bounds: &[Option<(S, S)>]) -> Vec<usize> {
    let tol = S::from_f64(1e-12);
    let mut active = Vec::new();
    for (i, bi) in bounds.iter().enumerate() {
        if let Some((lo, hi)) = bi {
            if (x[i] - *lo).abs() < tol || (x[i] - *hi).abs() < tol {
                active.push(i);
            }
        }
    }
    active
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_project() {
        let mut x = vec![-2.0, 5.0, 0.5];
        let bounds = vec![Some((0.0, 1.0)), Some((0.0, 3.0)), None];
        project(&mut x, &bounds);
        assert_eq!(x, vec![0.0, 3.0, 0.5]);
    }

    #[test]
    fn test_projected_gradient_norm() {
        let x = [0.0, 1.0];
        let g = [-1.0, 2.0];
        let bounds = vec![Some((0.0, 1.0)), Some((0.0, 1.0))];
        let pgn = projected_gradient_norm(&x, &g, &bounds);
        assert!((pgn - 1.0).abs() < 1e-14);
    }

    #[test]
    fn test_lbfgsb_quadratic_at_boundary() {
        // f(x) = (x0-2)^2 + (x1-2)^2, min at (2,2)
        // bounds: 0 <= x0 <= 1, 0 <= x1 <= 3
        // Constrained min: (1, 2)
        let f = |x: &[f64]| (x[0] - 2.0).powi(2) + (x[1] - 2.0).powi(2);
        let g = |x: &[f64], grad: &mut [f64]| {
            grad[0] = 2.0 * (x[0] - 2.0);
            grad[1] = 2.0 * (x[1] - 2.0);
        };
        let bounds = vec![Some((0.0, 1.0)), Some((0.0, 3.0))];
        let opts = LbfgsBOptions {
            base: OptimOptions::default().gtol(1e-10),
            memory: 5,
        };

        let result = lbfgsb_minimize(f, g, &[0.5, 0.5], &bounds, &opts).unwrap();
        assert!(result.converged, "did not converge: {}", result.message);
        assert!((result.x[0] - 1.0).abs() < 1e-6, "x0={}", result.x[0]);
        assert!((result.x[1] - 2.0).abs() < 1e-6, "x1={}", result.x[1]);
        assert!(result.active_bounds.contains(&0));
    }

    #[test]
    fn test_lbfgsb_rosenbrock_bounded() {
        // Rosenbrock with bounds: 0 <= x0 <= 2, 0 <= x1 <= 2
        // Unconstrained min at (1,1), which is inside bounds
        let f = |x: &[f64]| (1.0 - x[0]).powi(2) + 100.0 * (x[1] - x[0] * x[0]).powi(2);
        let g = |x: &[f64], grad: &mut [f64]| {
            grad[0] = -2.0 * (1.0 - x[0]) - 400.0 * x[0] * (x[1] - x[0] * x[0]);
            grad[1] = 200.0 * (x[1] - x[0] * x[0]);
        };
        let bounds = vec![Some((0.0, 2.0)), Some((0.0, 2.0))];
        let opts = LbfgsBOptions {
            base: OptimOptions::default().gtol(1e-6).max_iter(2000),
            memory: 10,
        };

        let result = lbfgsb_minimize(f, g, &[0.1, 0.1], &bounds, &opts).unwrap();
        assert!(result.converged, "did not converge: {}", result.message);
        assert!(result.x[0] >= -1e-10 && result.x[0] <= 2.0 + 1e-10);
        assert!(result.x[1] >= -1e-10 && result.x[1] <= 2.0 + 1e-10);
        assert!((result.x[0] - 1.0).abs() < 1e-2, "x0={}", result.x[0]);
        assert!((result.x[1] - 1.0).abs() < 1e-2, "x1={}", result.x[1]);
    }

    #[test]
    fn test_lbfgsb_sphere_all_bounded() {
        // f(x) = sum(x_i^2), n=5, all bounded [1, 10]
        // Minimum at x = [1,1,1,1,1] (all at lower bound)
        let f = |x: &[f64]| x.iter().map(|xi| xi * xi).sum::<f64>();
        let g = |x: &[f64], grad: &mut [f64]| {
            for i in 0..x.len() {
                grad[i] = 2.0 * x[i];
            }
        };
        let bounds: Vec<Option<(f64, f64)>> = vec![Some((1.0, 10.0)); 5];
        let opts = LbfgsBOptions::default().gtol(1e-10);

        let result = lbfgsb_minimize(f, g, &[5.0; 5], &bounds, &opts).unwrap();
        assert!(result.converged, "did not converge: {}", result.message);
        for &xi in &result.x {
            assert!((xi - 1.0).abs() < 1e-6, "x_i={}", xi);
        }
    }
}

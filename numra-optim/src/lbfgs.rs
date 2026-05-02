//! L-BFGS (Limited-memory BFGS) optimizer.
//!
//! Uses the two-loop recursion to compute search directions using only
//! the last `m` (s, y) correction pairs, requiring O(mn) storage instead of O(n²).
//!
//! Author: Moussa Leblouba
//! Date: 8 February 2026
//! Modified: 2 May 2026

use numra_core::Scalar;

use crate::error::OptimError;
use crate::types::{IterationRecord, OptimOptions, OptimResult, OptimStatus};
use numra_nonlinear::line_search::{wolfe_line_search, WolfeOptions};
use std::collections::VecDeque;

/// Options specific to L-BFGS.
#[derive(Clone, Debug)]
pub struct LbfgsOptions<S: Scalar> {
    /// Base optimization options.
    pub base: OptimOptions<S>,
    /// Number of correction pairs to store (default: 10).
    pub memory: usize,
}

impl<S: Scalar> Default for LbfgsOptions<S> {
    fn default() -> Self {
        Self {
            base: OptimOptions::default(),
            memory: 10,
        }
    }
}

impl<S: Scalar> LbfgsOptions<S> {
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

/// L-BFGS optimizer.
pub struct Lbfgs<S: Scalar> {
    options: LbfgsOptions<S>,
}

impl<S: Scalar> Lbfgs<S> {
    pub fn new(options: LbfgsOptions<S>) -> Self {
        Self { options }
    }

    pub fn minimize<F, G>(&self, f: F, grad: G, x0: &[S]) -> Result<OptimResult<S>, OptimError>
    where
        F: Fn(&[S]) -> S,
        G: Fn(&[S], &mut [S]),
    {
        lbfgs_minimize(f, grad, x0, &self.options)
    }
}

/// Minimize f(x) using L-BFGS.
pub fn lbfgs_minimize<S: Scalar, F, G>(
    f: F,
    grad: G,
    x0: &[S],
    opts: &LbfgsOptions<S>,
) -> Result<OptimResult<S>, OptimError>
where
    F: Fn(&[S]) -> S,
    G: Fn(&[S], &mut [S]),
{
    let start = std::time::Instant::now();
    let n = x0.len();
    let m = opts.memory;
    let mut x = x0.to_vec();
    let mut g = vec![S::ZERO; n];
    let mut g_new = vec![S::ZERO; n];

    // History buffers
    let mut s_hist: VecDeque<Vec<S>> = VecDeque::with_capacity(m);
    let mut y_hist: VecDeque<Vec<S>> = VecDeque::with_capacity(m);
    let mut rho_hist: VecDeque<S> = VecDeque::with_capacity(m);

    let mut fval = f(&x);
    grad(&x, &mut g);
    let mut n_feval = 1_usize;
    let mut n_geval = 1_usize;

    let wolfe_opts = WolfeOptions::default();
    let mut history = Vec::new();

    for iter in 0..opts.base.max_iter {
        let g_norm = g.iter().copied().map(|gi| gi * gi).sum::<S>().sqrt();
        if g_norm < opts.base.gtol {
            return Ok((OptimResult {
                history,
                ..OptimResult::unconstrained(
                    x,
                    fval,
                    g,
                    iter,
                    n_feval,
                    n_geval,
                    true,
                    format!("Converged: gradient norm {:.2e}", g_norm.to_f64()),
                    OptimStatus::GradientConverged,
                )
            })
            .with_wall_time(start));
        }

        // Two-loop recursion to compute d = -H_k * g
        let d = two_loop_recursion(&g, &s_hist, &y_hist, &rho_hist);

        // Line search
        let ls_result = wolfe_line_search(&f, &grad, &x, &d, fval, &g, &wolfe_opts)?;
        n_feval += ls_result.n_eval;

        let alpha = ls_result.step;

        // s = alpha * d
        let s: Vec<S> = d.iter().map(|di| alpha * *di).collect();

        // Update x
        for i in 0..n {
            x[i] += s[i];
        }

        let f_new = ls_result.f_new;

        if (fval - f_new).abs() < opts.base.ftol * (S::ONE + fval.abs()) {
            grad(&x, &mut g);
            n_geval += 1;
            let g_norm_new = g.iter().copied().map(|gi| gi * gi).sum::<S>().sqrt();
            history.push(IterationRecord {
                iteration: iter,
                objective: f_new,
                gradient_norm: g_norm_new,
                step_size: alpha,
                constraint_violation: S::ZERO,
            });
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
                    format!(
                        "Converged: function change {:.2e}",
                        (fval - f_new).abs().to_f64()
                    ),
                    OptimStatus::FunctionConverged,
                )
            })
            .with_wall_time(start));
        }

        fval = f_new;

        // New gradient
        grad(&x, &mut g_new);
        n_geval += 1;

        let g_new_norm = g_new.iter().copied().map(|gi| gi * gi).sum::<S>().sqrt();
        history.push(IterationRecord {
            iteration: iter,
            objective: fval,
            gradient_norm: g_new_norm,
            step_size: alpha,
            constraint_violation: S::ZERO,
        });

        // y = g_new - g
        let y: Vec<S> = g_new
            .iter()
            .zip(g.iter())
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

        g.copy_from_slice(&g_new);
    }

    Ok((OptimResult {
        history,
        ..OptimResult::unconstrained(
            x,
            fval,
            g,
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

/// Two-loop recursion for L-BFGS direction computation.
pub(crate) fn two_loop_recursion<S: Scalar>(
    g: &[S],
    s_hist: &VecDeque<Vec<S>>,
    y_hist: &VecDeque<Vec<S>>,
    rho_hist: &VecDeque<S>,
) -> Vec<S> {
    let k = s_hist.len();
    let n = g.len();
    let mut q = g.to_vec();
    let mut alpha_vec = vec![S::ZERO; k];

    // Forward pass
    for i in (0..k).rev() {
        let a: S = rho_hist[i]
            * s_hist[i]
                .iter()
                .zip(q.iter())
                .map(|(si, qi)| *si * *qi)
                .sum::<S>();
        alpha_vec[i] = a;
        for j in 0..n {
            q[j] -= a * y_hist[i][j];
        }
    }

    // Initial Hessian scaling: H_0 = gamma * I
    let gamma = if k > 0 {
        let sy: S = s_hist[k - 1]
            .iter()
            .zip(y_hist[k - 1].iter())
            .map(|(s, y)| *s * *y)
            .sum();
        let yy: S = y_hist[k - 1].iter().copied().map(|y| y * y).sum();
        sy / yy
    } else {
        S::ONE
    };

    let mut r: Vec<S> = q.iter().map(|qi| gamma * *qi).collect();

    // Backward pass
    for i in 0..k {
        let b: S = rho_hist[i]
            * y_hist[i]
                .iter()
                .zip(r.iter())
                .map(|(yi, ri)| *yi * *ri)
                .sum::<S>();
        for j in 0..n {
            r[j] += s_hist[i][j] * (alpha_vec[i] - b);
        }
    }

    // Negate for descent direction
    for ri in r.iter_mut() {
        *ri = -*ri;
    }

    r
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lbfgs_quadratic() {
        let f = |x: &[f64]| x[0] * x[0] + 4.0 * x[1] * x[1];
        let g = |x: &[f64], grad: &mut [f64]| {
            grad[0] = 2.0 * x[0];
            grad[1] = 8.0 * x[1];
        };

        let result = lbfgs_minimize(f, g, &[5.0, 3.0], &LbfgsOptions::default()).unwrap();
        assert!(result.converged);
        assert!(result.f < 1e-12);
    }

    #[test]
    fn test_lbfgs_rosenbrock() {
        let f = |x: &[f64]| (1.0 - x[0]).powi(2) + 100.0 * (x[1] - x[0] * x[0]).powi(2);
        let g = |x: &[f64], grad: &mut [f64]| {
            grad[0] = -2.0 * (1.0 - x[0]) - 400.0 * x[0] * (x[1] - x[0] * x[0]);
            grad[1] = 200.0 * (x[1] - x[0] * x[0]);
        };

        let result = lbfgs_minimize(f, g, &[-1.0, 1.0], &LbfgsOptions::default()).unwrap();
        assert!(
            result.converged,
            "L-BFGS did not converge: {}",
            result.message
        );
        assert!((result.x[0] - 1.0).abs() < 1e-4);
        assert!((result.x[1] - 1.0).abs() < 1e-4);
    }

    #[test]
    fn test_lbfgs_large_scale() {
        // n=100, f(x) = sum(x_i^2)
        let n = 100;
        let f = |x: &[f64]| x.iter().copied().map(|xi| xi * xi).sum::<f64>();
        let g = |x: &[f64], grad: &mut [f64]| {
            for i in 0..x.len() {
                grad[i] = 2.0 * x[i];
            }
        };

        let x0: Vec<f64> = (1..=n).map(|i| i as f64 * 0.1).collect();
        let result = lbfgs_minimize(f, g, &x0, &LbfgsOptions::default()).unwrap();
        assert!(result.converged);
        assert!(result.f < 1e-10);
    }
}

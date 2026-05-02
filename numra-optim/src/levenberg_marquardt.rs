//! Levenberg-Marquardt algorithm for nonlinear least squares.
//!
//! Solves min_x ||r(x)||^2 where r: R^n -> R^m is a residual vector.
//!
//! Author: Moussa Leblouba
//! Date: 8 February 2026
//! Modified: 2 May 2026

use numra_core::Scalar;

use crate::error::OptimError;
use crate::types::{IterationRecord, OptimResult, OptimStatus};
use numra_linalg::{DenseMatrix, Matrix};

/// Options for Levenberg-Marquardt.
#[derive(Clone, Debug)]
pub struct LmOptions<S: Scalar> {
    pub max_iter: usize,
    pub gtol: S,
    pub xtol: S,
    pub ftol: S,
    /// Initial damping parameter.
    pub lambda_init: S,
    /// Damping increase factor.
    pub lambda_up: S,
    /// Damping decrease factor.
    pub lambda_down: S,
}

impl<S: Scalar> Default for LmOptions<S> {
    fn default() -> Self {
        Self {
            max_iter: 200,
            gtol: S::from_f64(1e-10),
            xtol: S::from_f64(1e-10),
            ftol: S::from_f64(1e-12),
            lambda_init: S::from_f64(1e-3),
            lambda_up: S::from_f64(10.0),
            lambda_down: S::from_f64(0.1),
        }
    }
}

/// Result of LM optimization, extending OptimResult with residual info.
pub type LmResult<S> = OptimResult<S>;

/// Minimize ||r(x)||^2 using Levenberg-Marquardt.
///
/// - `residual`: computes r(x) into output slice of length `m`
/// - `jacobian`: computes J(x) into row-major slice of length `m * n`
/// - `x0`: initial guess of length `n`
/// - `m`: number of residuals
pub fn lm_minimize<S, R, J>(
    residual: R,
    jacobian: J,
    x0: &[S],
    m: usize,
    opts: &LmOptions<S>,
) -> Result<LmResult<S>, OptimError>
where
    S: Scalar + faer::SimpleEntity + faer::Conjugate<Canonical = S> + faer::ComplexField,
    R: Fn(&[S], &mut [S]),
    J: Fn(&[S], &mut [S]),
{
    let start = std::time::Instant::now();
    let n = x0.len();
    let mut x = x0.to_vec();
    let mut r = vec![S::ZERO; m];
    let mut jac = vec![S::ZERO; m * n];
    let mut n_feval = 0_usize;
    let mut n_geval = 0_usize;

    // Evaluate initial residual
    residual(&x, &mut r);
    n_feval += 1;
    let mut cost = r.iter().copied().map(|ri| ri * ri).sum::<S>();

    let mut lambda = opts.lambda_init;
    let mut history = Vec::new();

    let small_diag = S::from_f64(1e-12);
    let small_pred = S::from_f64(1e-30);

    for iter in 0..opts.max_iter {
        // Compute Jacobian
        jacobian(&x, &mut jac);
        n_geval += 1;

        // Compute J^T r (gradient of 0.5*||r||^2)
        let mut jtr = vec![S::ZERO; n];
        for i in 0..m {
            for j in 0..n {
                jtr[j] += jac[i * n + j] * r[i];
            }
        }

        // Check gradient convergence
        let g_norm = jtr.iter().copied().map(|gi| gi * gi).sum::<S>().sqrt();

        history.push(IterationRecord {
            iteration: iter,
            objective: cost,
            gradient_norm: g_norm,
            step_size: lambda,
            constraint_violation: S::ZERO,
        });

        if g_norm < opts.gtol {
            return Ok((OptimResult {
                history,
                ..OptimResult::unconstrained(
                    x,
                    cost,
                    jtr,
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

        // Compute J^T J + lambda * I
        let mut jtj = vec![S::ZERO; n * n];
        for i in 0..m {
            for j in 0..n {
                for k in 0..n {
                    jtj[j * n + k] += jac[i * n + j] * jac[i * n + k];
                }
            }
        }

        // Solve (J^T J + lambda * diag(J^T J)) * dx = -J^T r
        // Use adaptive damping: scale by diagonal of J^T J
        let mut a = jtj.clone();
        for i in 0..n {
            let diag = if jtj[i * n + i] > small_diag {
                jtj[i * n + i]
            } else {
                small_diag
            };
            a[i * n + i] += lambda * diag;
        }

        let neg_jtr: Vec<S> = jtr.iter().map(|g| -*g).collect();
        let dx = solve_via_lu(&a, &neg_jtr, n)?;

        // Trial point
        let x_new: Vec<S> = x.iter().zip(dx.iter()).map(|(xi, di)| *xi + *di).collect();
        let mut r_new = vec![S::ZERO; m];
        residual(&x_new, &mut r_new);
        n_feval += 1;
        let cost_new = r_new.iter().copied().map(|ri| ri * ri).sum::<S>();

        // Gain ratio
        let predicted = {
            let mut pred = S::ZERO;
            for i in 0..n {
                let diag = if jtj[i * n + i] > small_diag {
                    jtj[i * n + i]
                } else {
                    small_diag
                };
                pred += dx[i] * (lambda * diag * dx[i] + jtr[i]);
            }
            -pred
        };

        if predicted.abs() < small_pred {
            // Nearly zero predicted improvement, likely converged
            return Ok((OptimResult {
                history,
                ..OptimResult::unconstrained(
                    x,
                    cost,
                    jtr,
                    iter,
                    n_feval,
                    n_geval,
                    true,
                    "Converged: negligible predicted improvement".to_string(),
                    OptimStatus::FunctionConverged,
                )
            })
            .with_wall_time(start));
        }

        let actual = cost - cost_new;
        let rho = actual / predicted.abs();

        if rho > S::ZERO && cost_new < cost {
            // Accept step
            let dx_norm = dx.iter().copied().map(|d| d * d).sum::<S>().sqrt();

            x = x_new;
            r = r_new;
            let f_change = cost - cost_new;
            cost = cost_new;

            // Decrease damping
            lambda *= opts.lambda_down;
            let lambda_floor = S::from_f64(1e-20);
            lambda = if lambda > lambda_floor {
                lambda
            } else {
                lambda_floor
            };

            // Check convergence
            if dx_norm
                < opts.xtol * (S::ONE + x.iter().copied().map(|xi| xi * xi).sum::<S>().sqrt())
            {
                let mut g = vec![S::ZERO; n];
                jacobian(&x, &mut jac);
                for i in 0..m {
                    for j in 0..n {
                        g[j] += jac[i * n + j] * r[i];
                    }
                }
                return Ok((OptimResult {
                    history,
                    ..OptimResult::unconstrained(
                        x,
                        cost,
                        g,
                        iter + 1,
                        n_feval,
                        n_geval + 1,
                        true,
                        format!("Converged: step size {:.2e}", dx_norm.to_f64()),
                        OptimStatus::StepConverged,
                    )
                })
                .with_wall_time(start));
            }

            if f_change < opts.ftol * (S::ONE + cost) {
                let mut g = vec![S::ZERO; n];
                jacobian(&x, &mut jac);
                for i in 0..m {
                    for j in 0..n {
                        g[j] += jac[i * n + j] * r[i];
                    }
                }
                return Ok((OptimResult {
                    history,
                    ..OptimResult::unconstrained(
                        x,
                        cost,
                        g,
                        iter + 1,
                        n_feval,
                        n_geval + 1,
                        true,
                        format!("Converged: function change {:.2e}", f_change.to_f64()),
                        OptimStatus::FunctionConverged,
                    )
                })
                .with_wall_time(start));
            }
        } else {
            // Reject step, increase damping
            lambda *= opts.lambda_up;
        }
    }

    // Final gradient
    let mut grad = vec![S::ZERO; n];
    jacobian(&x, &mut jac);
    for i in 0..m {
        for j in 0..n {
            grad[j] += jac[i * n + j] * r[i];
        }
    }

    Ok((OptimResult {
        history,
        ..OptimResult::unconstrained(
            x,
            cost,
            grad,
            opts.max_iter,
            n_feval,
            n_geval + 1,
            false,
            format!("Maximum iterations ({}) reached", opts.max_iter),
            OptimStatus::MaxIterations,
        )
    })
    .with_wall_time(start))
}

/// Solve dense linear system Ax = b via LU factorization (numra-linalg/faer backend).
/// A is n*n in row-major, b is length n.
fn solve_via_lu<S>(a: &[S], b: &[S], n: usize) -> Result<Vec<S>, OptimError>
where
    S: Scalar + faer::SimpleEntity + faer::Conjugate<Canonical = S> + faer::ComplexField,
{
    let mat = DenseMatrix::<S>::from_row_major(n, n, a);
    mat.solve(b).map_err(|_| OptimError::SingularMatrix)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lm_linear_fit() {
        // Fit y = a*x + b to data: (0,1), (1,3), (2,5)
        // True: a=2, b=1
        let t_data = [0.0, 1.0, 2.0];
        let y_data = [1.0, 3.0, 5.0];

        let residual = |params: &[f64], r: &mut [f64]| {
            for i in 0..t_data.len() {
                r[i] = params[0] * t_data[i] + params[1] - y_data[i];
            }
        };

        let jacobian = |params: &[f64], jac: &mut [f64]| {
            let m = t_data.len();
            for i in 0..m {
                jac[i * 2] = t_data[i]; // dr_i/da
                jac[i * 2 + 1] = 1.0; // dr_i/db
            }
            let _ = params; // params unused for linear model Jacobian
        };

        let result =
            lm_minimize(residual, jacobian, &[0.0, 0.0], 3, &LmOptions::default()).unwrap();
        assert!(result.converged);
        assert!((result.x[0] - 2.0).abs() < 1e-8);
        assert!((result.x[1] - 1.0).abs() < 1e-8);
        assert!(
            result.wall_time_secs > 0.0,
            "wall_time_secs should be positive"
        );
    }

    #[test]
    fn test_lm_exponential_fit() {
        // Fit y = a * exp(b * x) to data
        // True: a=2.0, b=-0.5
        let t_data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let y_data: Vec<f64> = t_data.iter().map(|&t| 2.0 * (-0.5 * t).exp()).collect();

        let m = t_data.len();

        let residual = |p: &[f64], r: &mut [f64]| {
            for i in 0..m {
                r[i] = p[0] * (p[1] * t_data[i]).exp() - y_data[i];
            }
        };

        let jacobian = |p: &[f64], jac: &mut [f64]| {
            for i in 0..m {
                let e = (p[1] * t_data[i]).exp();
                jac[i * 2] = e; // dr_i/da
                jac[i * 2 + 1] = p[0] * t_data[i] * e; // dr_i/db
            }
        };

        let result =
            lm_minimize(residual, jacobian, &[1.0, -1.0], m, &LmOptions::default()).unwrap();
        assert!(result.converged, "LM did not converge: {}", result.message);
        assert!((result.x[0] - 2.0).abs() < 1e-4);
        assert!((result.x[1] + 0.5).abs() < 1e-4);
    }

    #[test]
    fn test_lm_circle_fit() {
        // Fit circle (cx, cy, r) to noisy points on unit circle at origin
        use std::f64::consts::PI;
        let n_pts = 20;
        let angles: Vec<f64> = (0..n_pts)
            .map(|i| 2.0 * PI * i as f64 / n_pts as f64)
            .collect();
        let x_data: Vec<f64> = angles.iter().map(|a| a.cos()).collect();
        let y_data: Vec<f64> = angles.iter().map(|a| a.sin()).collect();

        let m = n_pts;

        let residual = |p: &[f64], r: &mut [f64]| {
            let cx = p[0];
            let cy = p[1];
            let rad = p[2];
            for i in 0..m {
                let dx = x_data[i] - cx;
                let dy = y_data[i] - cy;
                r[i] = (dx * dx + dy * dy).sqrt() - rad;
            }
        };

        let jacobian = |p: &[f64], jac: &mut [f64]| {
            let cx = p[0];
            let cy = p[1];
            for i in 0..m {
                let dx = x_data[i] - cx;
                let dy = y_data[i] - cy;
                let dist = (dx * dx + dy * dy).sqrt();
                jac[i * 3] = -dx / dist; // dr_i/dcx
                jac[i * 3 + 1] = -dy / dist; // dr_i/dcy
                jac[i * 3 + 2] = -1.0; // dr_i/dr
            }
        };

        let result = lm_minimize(
            residual,
            jacobian,
            &[0.5, 0.5, 0.5],
            m,
            &LmOptions::default(),
        )
        .unwrap();
        assert!(result.converged, "LM did not converge: {}", result.message);
        assert!(result.x[0].abs() < 1e-6); // cx ~ 0
        assert!(result.x[1].abs() < 1e-6); // cy ~ 0
        assert!((result.x[2] - 1.0).abs() < 1e-6); // r ~ 1
    }
}

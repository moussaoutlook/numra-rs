//! Augmented Lagrangian method for constrained optimization.
//!
//! Converts a constrained problem into a sequence of unconstrained (or
//! bound-constrained) subproblems by penalizing constraint violations
//! and maintaining Lagrange multiplier estimates.
//!
//! Author: Moussa Leblouba
//! Date: 8 February 2026
//! Modified: 2 May 2026

use numra_core::Scalar;

use crate::error::OptimError;
use crate::lbfgs::{lbfgs_minimize, LbfgsOptions};
use crate::lbfgsb::{lbfgsb_minimize, LbfgsBOptions};
use crate::problem::{
    finite_diff_gradient, Constraint, ConstraintKind, ObjectiveKind, OptimProblem,
};
use crate::types::{IterationRecord, OptimOptions, OptimResult, OptimStatus};

/// Options for the Augmented Lagrangian outer loop.
#[derive(Clone, Debug)]
pub struct AugLagOptions<S: Scalar> {
    pub inner_opts: OptimOptions<S>,
    pub max_outer_iter: usize,
    pub sigma_init: S,
    pub sigma_factor: S,
    pub sigma_max: S,
    pub ctol: S,
}

impl<S: Scalar> Default for AugLagOptions<S> {
    fn default() -> Self {
        Self {
            inner_opts: OptimOptions::default().max_iter(500),
            max_outer_iter: 50,
            sigma_init: S::ONE,
            sigma_factor: S::from_f64(10.0),
            sigma_max: S::from_f64(1e12),
            ctol: S::from_f64(1e-6),
        }
    }
}

/// Solve a constrained optimization problem using the Augmented Lagrangian method.
///
/// Expects `problem` to have a `Minimize` objective and constraints.
pub fn augmented_lagrangian_minimize<S: Scalar>(
    problem: OptimProblem<S>,
    opts: &AugLagOptions<S>,
) -> Result<OptimResult<S>, OptimError> {
    let start = std::time::Instant::now();
    // Destructure to avoid partial move issues
    let OptimProblem {
        n,
        x0,
        bounds,
        objective,
        constraints,
        ..
    } = problem;

    let x0 = x0.ok_or(OptimError::NoInitialPoint)?;

    // Extract objective
    let (obj_func, obj_grad) = match objective {
        Some(ObjectiveKind::Minimize { func, grad }) => (func, grad),
        Some(ObjectiveKind::LeastSquares { .. }) => {
            return Err(OptimError::Other(
                "augmented Lagrangian requires scalar objective, not least squares".into(),
            ));
        }
        Some(ObjectiveKind::Linear { .. }) | Some(ObjectiveKind::Quadratic { .. }) => {
            return Err(OptimError::Other(
                "augmented Lagrangian requires scalar objective; use Simplex for LP or ActiveSetQP for QP".into(),
            ));
        }
        Some(ObjectiveKind::MultiObjective { .. }) => {
            return Err(OptimError::Other(
                "augmented Lagrangian requires scalar objective; use NSGA-II for multi-objective"
                    .into(),
            ));
        }
        None => return Err(OptimError::NoObjective),
    };

    let has_bounds = bounds.iter().any(|b| b.is_some());

    // Separate equality and inequality constraints
    let eq_constraints: Vec<&Constraint<S>> = constraints
        .iter()
        .filter(|c| c.kind == ConstraintKind::Equality)
        .collect();
    let ineq_constraints: Vec<&Constraint<S>> = constraints
        .iter()
        .filter(|c| c.kind == ConstraintKind::Inequality)
        .collect();

    let n_eq = eq_constraints.len();
    let n_ineq = ineq_constraints.len();

    // Multiplier estimates
    let mut lambda_eq = vec![S::ZERO; n_eq];
    let mut mu_ineq = vec![S::ZERO; n_ineq];
    let mut sigma = opts.sigma_init;
    let mut x = x0;

    let mut total_feval = 0_usize;
    let mut total_geval = 0_usize;
    let mut history = Vec::new();

    let two = S::TWO;

    for outer in 0..opts.max_outer_iter {
        // Build augmented Lagrangian subproblem
        let lam_eq = lambda_eq.clone();
        let mu_in = mu_ineq.clone();
        let sig = sigma;

        let aug_f = |xv: &[S]| -> S {
            let mut val = (obj_func)(xv);

            // Equality: lambda_j * h_j(x) + (sigma/2) * h_j(x)^2
            for (j, c) in eq_constraints.iter().enumerate() {
                let h = (c.func)(xv);
                val = val + lam_eq[j] * h + (sig / two) * h * h;
            }

            // Inequality: (sigma/2) * max(0, g_i(x) + mu_i/sigma)^2 - mu_i^2/(2*sigma)
            for (i, c) in ineq_constraints.iter().enumerate() {
                let g = (c.func)(xv);
                let shifted = g + mu_in[i] / sig;
                if shifted > S::ZERO {
                    val = val + (sig / two) * shifted * shifted - mu_in[i] * mu_in[i] / (two * sig);
                }
            }

            val
        };

        let aug_grad = |xv: &[S], gout: &mut [S]| {
            // Base gradient
            if let Some(ref og) = obj_grad {
                og(xv, gout);
            } else {
                finite_diff_gradient(&*obj_func, xv, gout);
            }

            let mut cgrad = vec![S::ZERO; n];

            // Equality constraint gradients
            for (j, c) in eq_constraints.iter().enumerate() {
                let h = (c.func)(xv);
                let mult = lam_eq[j] + sig * h;
                if let Some(ref cg) = c.grad {
                    cg(xv, &mut cgrad);
                } else {
                    finite_diff_gradient(&*c.func, xv, &mut cgrad);
                }
                for k in 0..n {
                    gout[k] += mult * cgrad[k];
                }
            }

            // Inequality constraint gradients
            for (i, c) in ineq_constraints.iter().enumerate() {
                let g_val = (c.func)(xv);
                let shifted = g_val + mu_in[i] / sig;
                if shifted > S::ZERO {
                    let mult = sig * shifted;
                    if let Some(ref cg) = c.grad {
                        cg(xv, &mut cgrad);
                    } else {
                        finite_diff_gradient(&*c.func, xv, &mut cgrad);
                    }
                    for k in 0..n {
                        gout[k] += mult * cgrad[k];
                    }
                }
            }
        };

        // Solve subproblem
        let sub_result = if has_bounds {
            let sub_opts = LbfgsBOptions {
                base: opts.inner_opts.clone(),
                memory: 10,
            };
            lbfgsb_minimize(aug_f, aug_grad, &x, &bounds, &sub_opts)?
        } else {
            let sub_opts = LbfgsOptions {
                base: opts.inner_opts.clone(),
                memory: 10,
            };
            lbfgs_minimize(aug_f, aug_grad, &x, &sub_opts)?
        };

        total_feval += sub_result.n_feval;
        total_geval += sub_result.n_geval;
        x = sub_result.x;

        // Compute constraint violations and update multipliers
        let mut max_violation = S::ZERO;

        for (j, c) in eq_constraints.iter().enumerate() {
            let h = (c.func)(&x);
            let abs_h = h.abs();
            if abs_h > max_violation {
                max_violation = abs_h;
            }
            lambda_eq[j] += sigma * h;
        }

        for (i, c) in ineq_constraints.iter().enumerate() {
            let g_val = (c.func)(&x);
            let shifted = g_val + mu_ineq[i] / sigma;
            if shifted > S::ZERO {
                let g_pos = if g_val > S::ZERO { g_val } else { S::ZERO };
                if g_pos > max_violation {
                    max_violation = g_pos;
                }
                let new_mu = mu_ineq[i] + sigma * g_val;
                mu_ineq[i] = if new_mu > S::ZERO { new_mu } else { S::ZERO };
            } else {
                mu_ineq[i] = S::ZERO;
            }
        }

        history.push(IterationRecord {
            iteration: outer,
            objective: (obj_func)(&x),
            gradient_norm: S::ZERO,
            step_size: sigma,
            constraint_violation: max_violation,
        });

        // Check convergence
        if max_violation < opts.ctol {
            let fval = (obj_func)(&x);
            let mut g_buf = vec![S::ZERO; n];
            if let Some(ref og) = obj_grad {
                og(&x, &mut g_buf);
            } else {
                finite_diff_gradient(&*obj_func, &x, &mut g_buf);
            }

            return Ok((OptimResult {
                lambda_eq,
                lambda_ineq: mu_ineq,
                constraint_violation: max_violation,
                history,
                ..OptimResult::unconstrained(
                    x,
                    fval,
                    g_buf,
                    outer + 1,
                    total_feval,
                    total_geval,
                    true,
                    format!(
                        "Converged: constraint violation {:.2e} after {} outer iterations",
                        max_violation.to_f64(),
                        outer + 1
                    ),
                    OptimStatus::GradientConverged,
                )
            })
            .with_wall_time(start));
        }

        // Increase penalty
        sigma *= opts.sigma_factor;
        if sigma > opts.sigma_max {
            sigma = opts.sigma_max;
        }
    }

    // Did not converge
    let max_violation: S = eq_constraints
        .iter()
        .map(|c| (c.func)(&x).abs())
        .chain(ineq_constraints.iter().map(|c| {
            let v = (c.func)(&x);
            if v > S::ZERO {
                v
            } else {
                S::ZERO
            }
        }))
        .fold(S::ZERO, |a, b| if b > a { b } else { a });

    if max_violation.to_f64() > 0.1 {
        return Err(OptimError::Infeasible {
            violation: max_violation.to_f64(),
        });
    }

    let fval = (obj_func)(&x);
    let mut g_buf = vec![S::ZERO; n];
    if let Some(ref og) = obj_grad {
        og(&x, &mut g_buf);
    } else {
        finite_diff_gradient(&*obj_func, &x, &mut g_buf);
    }

    Ok((OptimResult {
        lambda_eq,
        lambda_ineq: mu_ineq,
        constraint_violation: max_violation,
        history,
        ..OptimResult::unconstrained(
            x,
            fval,
            g_buf,
            opts.max_outer_iter,
            total_feval,
            total_geval,
            false,
            format!(
                "Maximum outer iterations ({}) reached, violation={:.2e}",
                opts.max_outer_iter,
                max_violation.to_f64()
            ),
            OptimStatus::MaxIterations,
        )
    })
    .with_wall_time(start))
}

#[cfg(test)]
mod tests {
    use crate::problem::OptimProblem;

    #[test]
    fn test_equality_constrained_circle() {
        // minimize x0 + x1 subject to x0^2 + x1^2 = 1
        // Lagrangian: min at x = (-1/sqrt(2), -1/sqrt(2))
        let result = OptimProblem::new(2)
            .x0(&[1.0, 0.0])
            .objective(|x: &[f64]| x[0] + x[1])
            .gradient(|x: &[f64], g: &mut [f64]| {
                g[0] = 1.0;
                g[1] = 1.0;
                let _ = x;
            })
            .constraint_eq_with_grad(
                |x: &[f64]| x[0] * x[0] + x[1] * x[1] - 1.0,
                |x: &[f64], g: &mut [f64]| {
                    g[0] = 2.0 * x[0];
                    g[1] = 2.0 * x[1];
                },
            )
            .solve()
            .unwrap();

        assert!(result.converged, "did not converge: {}", result.message);
        let expected = -1.0 / 2.0_f64.sqrt();
        assert!(
            (result.x[0] - expected).abs() < 1e-3,
            "x0={}, expected {}",
            result.x[0],
            expected
        );
        assert!(
            (result.x[1] - expected).abs() < 1e-3,
            "x1={}, expected {}",
            result.x[1],
            expected
        );
        assert!(result.constraint_violation < 1e-5);
    }

    #[test]
    fn test_inequality_constrained() {
        // minimize (x0-2)^2 + (x1-2)^2 subject to x0 + x1 <= 2
        // i.e. constraint: x0 + x1 - 2 <= 0
        // Unconstrained min at (2,2), but constrained to x0+x1=2 => min at (1,1)
        let result = OptimProblem::new(2)
            .x0(&[0.0, 0.0])
            .objective(|x: &[f64]| (x[0] - 2.0).powi(2) + (x[1] - 2.0).powi(2))
            .gradient(|x: &[f64], g: &mut [f64]| {
                g[0] = 2.0 * (x[0] - 2.0);
                g[1] = 2.0 * (x[1] - 2.0);
            })
            .constraint_ineq_with_grad(
                |x: &[f64]| x[0] + x[1] - 2.0,
                |_x: &[f64], g: &mut [f64]| {
                    g[0] = 1.0;
                    g[1] = 1.0;
                },
            )
            .solve()
            .unwrap();

        assert!(result.converged, "did not converge: {}", result.message);
        assert!(
            (result.x[0] - 1.0).abs() < 1e-2,
            "x0={}, expected 1.0",
            result.x[0]
        );
        assert!(
            (result.x[1] - 1.0).abs() < 1e-2,
            "x1={}, expected 1.0",
            result.x[1]
        );
    }

    #[test]
    fn test_mixed_constraints() {
        // minimize x0^2 + x1^2
        // subject to: x0 + x1 = 1 (equality)
        //             x0 >= 0.6   i.e. 0.6 - x0 <= 0 (inequality, active)
        // Without ineq: x0=0.5, x1=0.5
        // With ineq (active): x0=0.6, x1=0.4
        let result = OptimProblem::new(2)
            .x0(&[1.0, 1.0])
            .objective(|x: &[f64]| x[0] * x[0] + x[1] * x[1])
            .gradient(|x: &[f64], g: &mut [f64]| {
                g[0] = 2.0 * x[0];
                g[1] = 2.0 * x[1];
            })
            .constraint_eq_with_grad(
                |x: &[f64]| x[0] + x[1] - 1.0,
                |_x: &[f64], g: &mut [f64]| {
                    g[0] = 1.0;
                    g[1] = 1.0;
                },
            )
            .constraint_ineq_with_grad(
                |x: &[f64]| 0.6 - x[0],
                |_x: &[f64], g: &mut [f64]| {
                    g[0] = -1.0;
                    g[1] = 0.0;
                },
            )
            .solve()
            .unwrap();

        assert!(result.converged, "did not converge: {}", result.message);
        assert!(
            (result.x[0] - 0.6).abs() < 5e-2,
            "x0={}, expected 0.6",
            result.x[0]
        );
        assert!(
            (result.x[1] - 0.4).abs() < 5e-2,
            "x1={}, expected 0.4",
            result.x[1]
        );
        assert!(result.constraint_violation < 1e-3);
    }

    #[test]
    fn test_aug_lag_custom_options() {
        use crate::augmented_lagrangian::AugLagOptions;
        let opts = AugLagOptions {
            sigma_init: 10.0,
            ctol: 1e-8,
            ..AugLagOptions::default()
        };
        let result = OptimProblem::new(2)
            .x0(&[1.0, 0.0])
            .objective(|x: &[f64]| x[0] + x[1])
            .gradient(|x: &[f64], g: &mut [f64]| {
                g[0] = 1.0;
                g[1] = 1.0;
                let _ = x;
            })
            .constraint_eq(|x: &[f64]| x[0] * x[0] + x[1] * x[1] - 1.0)
            .aug_lag_options(opts)
            .solve()
            .unwrap();
        assert!(result.converged, "did not converge: {}", result.message);
        assert!(result.constraint_violation < 1e-7);
    }
}

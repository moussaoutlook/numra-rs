//! Automatic solver selection for optimization problems.
//!
//! Author: Moussa Leblouba
//! Date: 5 March 2026
//! Modified: 2 May 2026

use std::sync::Arc;

use numra_core::Scalar;

use crate::augmented_lagrangian::augmented_lagrangian_minimize;
use crate::bfgs::bfgs_minimize;
use crate::error::OptimError;
use crate::lbfgs::{lbfgs_minimize, LbfgsOptions};
use crate::lbfgsb::{lbfgsb_minimize, LbfgsBOptions};
use crate::levenberg_marquardt::{lm_minimize, LmOptions};
use crate::problem::{
    finite_diff_gradient, finite_diff_jacobian, ConstraintKind, ObjectiveKind, OptimProblem,
    VarType,
};
use crate::types::OptimResult;

/// Explicit solver choice for `OptimProblem::solve_with`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SolverChoice {
    Simplex,
    ActiveSetQP,
    MixedIntegerLP,
    DifferentialEvolution,
    NsgaII,
    Bfgs,
    Lbfgs,
    LbfgsB,
    LevenbergMarquardt,
    AugmentedLagrangian,
    NelderMead,
    Powell,
    CmaEs,
    Sqp,
}

/// Select the best solver for the given problem structure.
pub fn select_solver<S: Scalar>(problem: &OptimProblem<S>) -> SolverChoice {
    if problem.is_multi_objective() {
        return SolverChoice::NsgaII;
    }
    if problem.global {
        return SolverChoice::DifferentialEvolution;
    }

    // Structural variants: exact coefficient access available
    if problem.is_linear() && problem.has_integer_vars() {
        return SolverChoice::MixedIntegerLP;
    }
    if problem.is_linear() {
        return SolverChoice::Simplex;
    }
    if problem.is_quadratic() {
        return SolverChoice::ActiveSetQP;
    }

    // Hint-based dispatch: closure is known to be linear/quadratic but
    // coefficients aren't available, so use NLP solvers that work well
    // on these problem classes.
    if problem.is_hint_linear() || problem.is_hint_quadratic() {
        if problem.has_constraints() {
            return SolverChoice::Sqp;
        }
        return SolverChoice::LbfgsB;
    }

    if problem.is_least_squares() && !problem.has_constraints() && !problem.has_bounds() {
        SolverChoice::LevenbergMarquardt
    } else if problem.has_constraints() {
        SolverChoice::AugmentedLagrangian
    } else if problem.has_bounds() {
        SolverChoice::LbfgsB
    } else {
        SolverChoice::Lbfgs
    }
}

/// Round integer and binary variables to their nearest feasible integer values.
///
/// Note: `result.f` is NOT re-evaluated (the objective closure may have been consumed).
/// The caller (bridge) can re-evaluate if needed.
fn round_integer_vars<S: Scalar>(x: &mut [S], var_types: &[VarType], bounds: &[Option<(S, S)>]) {
    for (i, vt) in var_types.iter().enumerate() {
        match vt {
            VarType::Integer => {
                let rounded = x[i].round();
                x[i] = match bounds.get(i).copied().flatten() {
                    Some((lo, hi)) => rounded.max(lo).min(hi),
                    None => rounded,
                };
            }
            VarType::Binary => {
                x[i] = if x[i] > S::from_f64(0.5) {
                    S::ONE
                } else {
                    S::ZERO
                };
            }
            VarType::Continuous => {}
        }
    }
}

/// Solve a problem using automatic solver selection.
pub fn auto_minimize<
    S: Scalar + faer::SimpleEntity + faer::Conjugate<Canonical = S> + faer::ComplexField,
>(
    problem: OptimProblem<S>,
) -> Result<OptimResult<S>, OptimError> {
    let negate = problem.negate_result;
    let has_int = problem.has_integer_vars();
    let var_types = problem.var_types.clone();
    let bounds = problem.bounds.clone();
    let choice = select_solver(&problem);
    let mut result = dispatch(problem, choice)?;

    // Post-solve integer rounding when NLP solver was used for integer problems
    if has_int && choice != SolverChoice::MixedIntegerLP {
        round_integer_vars(&mut result.x, &var_types, &bounds);
        result
            .message
            .push_str(" (integer vars rounded post-solve; f is pre-round value)");
    }

    if negate {
        result.f = -result.f;
        for gi in result.grad.iter_mut() {
            *gi = -*gi;
        }
        // Also negate objective in history records
        for rec in result.history.iter_mut() {
            rec.objective = -rec.objective;
        }
    }
    Ok(result)
}

/// Dispatch to a specific solver.
#[allow(clippy::type_complexity)]
pub fn dispatch<
    S: Scalar + faer::SimpleEntity + faer::Conjugate<Canonical = S> + faer::ComplexField,
>(
    problem: OptimProblem<S>,
    choice: SolverChoice,
) -> Result<OptimResult<S>, OptimError> {
    // Augmented Lagrangian consumes the whole problem
    if choice == SolverChoice::AugmentedLagrangian {
        let aug_opts = problem.aug_lag_options.clone().unwrap_or_default();
        return augmented_lagrangian_minimize(problem, &aug_opts);
    }

    // SQP consumes the whole problem (it needs the constraints)
    if choice == SolverChoice::Sqp {
        let x0 = problem.x0.ok_or(OptimError::NoInitialPoint)?;
        let objective = problem.objective.ok_or(OptimError::NoObjective)?;
        let (func, grad) = match objective {
            ObjectiveKind::Minimize { func, grad } => (func, grad),
            _ => return Err(OptimError::Other("SQP requires scalar objective".into())),
        };
        let func_arc: Arc<dyn Fn(&[S]) -> S> = Arc::from(func);
        let grad_fn: Box<dyn Fn(&[S], &mut [S])> = match grad {
            Some(g) => g,
            None => {
                let fa = Arc::clone(&func_arc);
                Box::new(move |x: &[S], g: &mut [S]| {
                    finite_diff_gradient(&*fa, x, g);
                })
            }
        };
        let sqp_opts = crate::sqp::SqpOptions {
            max_iter: problem.options.max_iter,
            tol: problem.options.gtol,
            ..Default::default()
        };
        return crate::sqp::sqp_minimize(
            &*func_arc,
            &*grad_fn,
            &problem.constraints,
            &x0,
            &sqp_opts,
        );
    }

    // Destructure problem
    let OptimProblem {
        x0,
        bounds,
        var_types,
        objective,
        options: opts,
        linear_constraints,
        global: _global,
        de_options,
        ..
    } = problem;

    // DE and NSGA-II don't require x0 -- handle them before the x0 check
    if choice == SolverChoice::DifferentialEvolution {
        let objective = objective.ok_or(OptimError::NoObjective)?;
        let func = match objective {
            ObjectiveKind::Minimize { func, .. } => func,
            _ => return Err(OptimError::Other("DE requires scalar objective".into())),
        };
        let de_bounds: Vec<(S, S)> = bounds
            .iter()
            .map(|b: &Option<(S, S)>| {
                b.ok_or(OptimError::Other(
                    "DE requires all variables to be bounded".into(),
                ))
            })
            .collect::<Result<_, _>>()?;
        let de_opts = de_options.unwrap_or_default();
        return crate::global::de_minimize(&*func, &de_bounds, &de_opts);
    }

    if choice == SolverChoice::NsgaII {
        let objective = objective.ok_or(OptimError::NoObjective)?;
        let funcs = match objective {
            ObjectiveKind::MultiObjective { funcs } => funcs,
            _ => return Err(OptimError::Other("NSGA-II requires multi-objective".into())),
        };
        let nsga_bounds: Vec<(S, S)> = bounds
            .iter()
            .map(|b: &Option<(S, S)>| {
                b.ok_or(OptimError::Other(
                    "NSGA-II requires all variables to be bounded".into(),
                ))
            })
            .collect::<Result<_, _>>()?;
        let obj_refs: Vec<&dyn Fn(&[S]) -> S> =
            funcs.iter().map(|f| &**f as &dyn Fn(&[S]) -> S).collect();
        let nsga_opts = crate::multiobjective::NsgaIIOptions::default();
        return crate::multiobjective::nsga2_optimize(&obj_refs, &nsga_bounds, &nsga_opts);
    }

    // LP, QP, and MILP solvers don't require x0 -- handle them before the x0 check
    if choice == SolverChoice::MixedIntegerLP {
        let objective = objective.ok_or(OptimError::NoObjective)?;
        let c = match objective {
            ObjectiveKind::Linear { c } => c,
            _ => return Err(OptimError::Other("MILP requires linear objective".into())),
        };

        let mut a_ineq: Vec<Vec<S>> = Vec::new();
        let mut b_ineq_vec: Vec<S> = Vec::new();
        let mut a_eq: Vec<Vec<S>> = Vec::new();
        let mut b_eq_vec: Vec<S> = Vec::new();

        for lc in &linear_constraints {
            match lc.kind {
                ConstraintKind::Inequality => {
                    a_ineq.push(lc.a.clone());
                    b_ineq_vec.push(lc.b);
                }
                ConstraintKind::Equality => {
                    a_eq.push(lc.a.clone());
                    b_eq_vec.push(lc.b);
                }
            }
        }

        let min_lp_tol = S::from_f64(1e-10);
        let milp_opts = crate::milp::MILPOptions {
            max_nodes: opts.max_iter,
            lp_tol: if opts.gtol > min_lp_tol {
                opts.gtol
            } else {
                min_lp_tol
            },
            verbose: opts.verbose,
            ..Default::default()
        };
        return crate::milp::milp_solve(
            &c,
            &a_ineq,
            &b_ineq_vec,
            &a_eq,
            &b_eq_vec,
            &var_types,
            &bounds,
            &milp_opts,
        );
    }

    if choice == SolverChoice::Simplex {
        let objective = objective.ok_or(OptimError::NoObjective)?;
        let c = match objective {
            ObjectiveKind::Linear { c } => c,
            _ => {
                return Err(OptimError::Other(
                    "Simplex requires linear objective".into(),
                ))
            }
        };

        let mut a_ineq: Vec<Vec<S>> = Vec::new();
        let mut b_ineq_vec: Vec<S> = Vec::new();
        let mut a_eq: Vec<Vec<S>> = Vec::new();
        let mut b_eq_vec: Vec<S> = Vec::new();

        for lc in &linear_constraints {
            match lc.kind {
                ConstraintKind::Inequality => {
                    a_ineq.push(lc.a.clone());
                    b_ineq_vec.push(lc.b);
                }
                ConstraintKind::Equality => {
                    a_eq.push(lc.a.clone());
                    b_eq_vec.push(lc.b);
                }
            }
        }

        let min_lp_tol = S::from_f64(1e-10);
        let lp_opts = crate::lp::LPOptions {
            max_iter: opts.max_iter,
            tol: if opts.gtol > min_lp_tol {
                opts.gtol
            } else {
                min_lp_tol
            },
            verbose: opts.verbose,
        };
        return crate::lp::simplex_solve(&c, &a_ineq, &b_ineq_vec, &a_eq, &b_eq_vec, &lp_opts);
    }

    if choice == SolverChoice::ActiveSetQP {
        let objective = objective.ok_or(OptimError::NoObjective)?;
        let (h_row_major, qp_c, qp_n) = match objective {
            ObjectiveKind::Quadratic { h_row_major, c, n } => (h_row_major, c, n),
            _ => {
                return Err(OptimError::Other(
                    "ActiveSetQP requires quadratic objective".into(),
                ))
            }
        };

        let mut a_ineq: Vec<Vec<S>> = Vec::new();
        let mut b_ineq_vec: Vec<S> = Vec::new();
        let mut a_eq: Vec<Vec<S>> = Vec::new();
        let mut b_eq_vec: Vec<S> = Vec::new();

        for lc in &linear_constraints {
            match lc.kind {
                ConstraintKind::Inequality => {
                    a_ineq.push(lc.a.clone());
                    b_ineq_vec.push(lc.b);
                }
                ConstraintKind::Equality => {
                    a_eq.push(lc.a.clone());
                    b_eq_vec.push(lc.b);
                }
            }
        }

        let min_qp_tol = S::from_f64(1e-10);
        let qp_opts = crate::qp::QPOptions {
            max_iter: opts.max_iter,
            tol: if opts.gtol > min_qp_tol {
                opts.gtol
            } else {
                min_qp_tol
            },
            verbose: opts.verbose,
        };
        return crate::qp::active_set_qp_solve(
            &h_row_major,
            &qp_c,
            qp_n,
            &a_ineq,
            &b_ineq_vec,
            &a_eq,
            &b_eq_vec,
            &bounds,
            &qp_opts,
        );
    }

    // Derivative-free and CMA-ES require x0 but handle objective extraction differently
    if choice == SolverChoice::NelderMead {
        let x0 = x0.ok_or(OptimError::NoInitialPoint)?;
        let objective = objective.ok_or(OptimError::NoObjective)?;
        let func = match objective {
            ObjectiveKind::Minimize { func, .. } => func,
            _ => {
                return Err(OptimError::Other(
                    "Nelder-Mead requires scalar objective".into(),
                ))
            }
        };
        let nm_opts = crate::derivative_free::NelderMeadOptions {
            max_iter: opts.max_iter,
            ..Default::default()
        };
        return crate::derivative_free::nelder_mead(&*func, &x0, &nm_opts);
    }

    if choice == SolverChoice::Powell {
        let x0 = x0.ok_or(OptimError::NoInitialPoint)?;
        let objective = objective.ok_or(OptimError::NoObjective)?;
        let func = match objective {
            ObjectiveKind::Minimize { func, .. } => func,
            _ => return Err(OptimError::Other("Powell requires scalar objective".into())),
        };
        let p_opts = crate::derivative_free::PowellOptions {
            max_iter: opts.max_iter,
            ..Default::default()
        };
        return crate::derivative_free::powell(&*func, &x0, &p_opts);
    }

    if choice == SolverChoice::CmaEs {
        let x0 = x0.ok_or(OptimError::NoInitialPoint)?;
        let objective = objective.ok_or(OptimError::NoObjective)?;
        let func = match objective {
            ObjectiveKind::Minimize { func, .. } => func,
            _ => return Err(OptimError::Other("CMA-ES requires scalar objective".into())),
        };
        let cma_opts = crate::cmaes::CmaEsOptions {
            max_iter: opts.max_iter,
            ..Default::default()
        };
        return crate::cmaes::cmaes_minimize(&*func, &x0, &cma_opts);
    }

    let x0 = x0.ok_or(OptimError::NoInitialPoint)?;
    let objective = objective.ok_or(OptimError::NoObjective)?;

    match choice {
        SolverChoice::LevenbergMarquardt => {
            let (residual, jacobian, m) = match objective {
                ObjectiveKind::LeastSquares {
                    residual,
                    jacobian,
                    n_residuals,
                } => (residual, jacobian, n_residuals),
                _ => {
                    return Err(OptimError::Other(
                        "LM requires least-squares objective".into(),
                    ))
                }
            };

            let res_arc: Arc<dyn Fn(&[S], &mut [S])> = Arc::from(residual);
            let jac_fn: Box<dyn Fn(&[S], &mut [S])> = match jacobian {
                Some(j) => j,
                None => {
                    let r = Arc::clone(&res_arc);
                    Box::new(move |x: &[S], jac: &mut [S]| {
                        finite_diff_jacobian(&*r, x, m, jac);
                    })
                }
            };

            let lm_opts = LmOptions {
                max_iter: opts.max_iter,
                gtol: opts.gtol,
                xtol: opts.xtol,
                ftol: opts.ftol,
                ..LmOptions::default()
            };
            lm_minimize(&*res_arc, &*jac_fn, &x0, m, &lm_opts)
        }

        SolverChoice::Bfgs => {
            let (func, grad) = match objective {
                ObjectiveKind::Minimize { func, grad } => (func, grad),
                _ => return Err(OptimError::Other("BFGS requires scalar objective".into())),
            };

            let func_arc: Arc<dyn Fn(&[S]) -> S> = Arc::from(func);
            let grad_fn: Box<dyn Fn(&[S], &mut [S])> = match grad {
                Some(g) => g,
                None => {
                    let f = Arc::clone(&func_arc);
                    Box::new(move |x: &[S], g: &mut [S]| {
                        finite_diff_gradient(&*f, x, g);
                    })
                }
            };

            bfgs_minimize(&*func_arc, &*grad_fn, &x0, &opts)
        }

        SolverChoice::Lbfgs => {
            let (func, grad) = match objective {
                ObjectiveKind::Minimize { func, grad } => (func, grad),
                _ => return Err(OptimError::Other("L-BFGS requires scalar objective".into())),
            };

            let func_arc: Arc<dyn Fn(&[S]) -> S> = Arc::from(func);
            let grad_fn: Box<dyn Fn(&[S], &mut [S])> = match grad {
                Some(g) => g,
                None => {
                    let f = Arc::clone(&func_arc);
                    Box::new(move |x: &[S], g: &mut [S]| {
                        finite_diff_gradient(&*f, x, g);
                    })
                }
            };

            let lbfgs_opts = LbfgsOptions {
                base: opts,
                memory: 10,
            };
            lbfgs_minimize(&*func_arc, &*grad_fn, &x0, &lbfgs_opts)
        }

        SolverChoice::LbfgsB => {
            let (func, grad) = match objective {
                ObjectiveKind::Minimize { func, grad } => (func, grad),
                _ => {
                    return Err(OptimError::Other(
                        "L-BFGS-B requires scalar objective".into(),
                    ))
                }
            };

            let func_arc: Arc<dyn Fn(&[S]) -> S> = Arc::from(func);
            let grad_fn: Box<dyn Fn(&[S], &mut [S])> = match grad {
                Some(g) => g,
                None => {
                    let f = Arc::clone(&func_arc);
                    Box::new(move |x: &[S], g: &mut [S]| {
                        finite_diff_gradient(&*f, x, g);
                    })
                }
            };

            let lbfgsb_opts = LbfgsBOptions {
                base: opts,
                memory: 10,
            };
            lbfgsb_minimize(&*func_arc, &*grad_fn, &x0, &bounds, &lbfgsb_opts)
        }

        SolverChoice::AugmentedLagrangian
        | SolverChoice::Simplex
        | SolverChoice::ActiveSetQP
        | SolverChoice::MixedIntegerLP
        | SolverChoice::DifferentialEvolution
        | SolverChoice::NsgaII
        | SolverChoice::NelderMead
        | SolverChoice::Powell
        | SolverChoice::CmaEs
        | SolverChoice::Sqp => unreachable!(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::problem::OptimProblem;

    #[test]
    fn test_select_solver_unconstrained() {
        let p = OptimProblem::new(2)
            .x0(&[0.0, 0.0])
            .objective(|x: &[f64]| x[0] * x[0] + x[1] * x[1]);
        assert_eq!(select_solver(&p), SolverChoice::Lbfgs);
    }

    #[test]
    fn test_select_solver_bounded() {
        let p = OptimProblem::new(2)
            .x0(&[0.0, 0.0])
            .objective(|x: &[f64]| x[0] * x[0] + x[1] * x[1])
            .bounds(0, (0.0, 1.0));
        assert_eq!(select_solver(&p), SolverChoice::LbfgsB);
    }

    #[test]
    fn test_select_solver_constrained() {
        let p = OptimProblem::new(2)
            .x0(&[0.0, 0.0])
            .objective(|x: &[f64]| x[0] + x[1])
            .constraint_eq(|x: &[f64]| x[0] * x[0] + x[1] * x[1] - 1.0);
        assert_eq!(select_solver(&p), SolverChoice::AugmentedLagrangian);
    }

    #[test]
    fn test_select_solver_least_squares() {
        let p =
            OptimProblem::new(2)
                .x0(&[0.0, 0.0])
                .least_squares(3, |_x: &[f64], r: &mut [f64]| {
                    r[0] = 0.0;
                    r[1] = 0.0;
                    r[2] = 0.0;
                });
        assert_eq!(select_solver(&p), SolverChoice::LevenbergMarquardt);
    }

    #[test]
    fn test_select_solver_lp() {
        let p = OptimProblem::new(2)
            .x0(&[0.0, 0.0])
            .linear_objective(&[1.0, 2.0])
            .linear_constraint_ineq(&[1.0, 1.0], 10.0);
        assert_eq!(select_solver(&p), SolverChoice::Simplex);
    }

    #[test]
    fn test_select_solver_qp() {
        let p = OptimProblem::new(2)
            .x0(&[0.0, 0.0])
            .quadratic_objective_dense(&[2.0, 0.0, 0.0, 2.0], &[0.0, 0.0]);
        assert_eq!(select_solver(&p), SolverChoice::ActiveSetQP);
    }

    #[test]
    fn test_auto_unconstrained_solve() {
        let result = OptimProblem::new(2)
            .x0(&[5.0, 3.0])
            .objective(|x: &[f64]| x[0] * x[0] + 4.0 * x[1] * x[1])
            .gradient(|x: &[f64], g: &mut [f64]| {
                g[0] = 2.0 * x[0];
                g[1] = 8.0 * x[1];
            })
            .solve()
            .unwrap();

        assert!(result.converged, "did not converge: {}", result.message);
        assert!(result.x[0].abs() < 1e-6);
        assert!(result.x[1].abs() < 1e-6);
    }

    #[test]
    fn test_auto_bounded_solve() {
        let result = OptimProblem::new(2)
            .x0(&[0.5, 0.5])
            .objective(|x: &[f64]| (x[0] - 2.0).powi(2) + (x[1] - 2.0).powi(2))
            .gradient(|x: &[f64], g: &mut [f64]| {
                g[0] = 2.0 * (x[0] - 2.0);
                g[1] = 2.0 * (x[1] - 2.0);
            })
            .all_bounds(&[(0.0, 1.0), (0.0, 3.0)])
            .solve()
            .unwrap();

        assert!(result.converged, "did not converge: {}", result.message);
        assert!((result.x[0] - 1.0).abs() < 1e-4, "x0={}", result.x[0]);
        assert!((result.x[1] - 2.0).abs() < 1e-4, "x1={}", result.x[1]);
    }

    #[test]
    fn test_auto_lp_solve() {
        let result = OptimProblem::new(2)
            .x0(&[0.0, 0.0])
            .linear_objective(&[-1.0, -1.0])
            .linear_constraint_ineq(&[1.0, 1.0], 4.0)
            .linear_constraint_ineq(&[1.0, 0.0], 3.0)
            .linear_constraint_ineq(&[0.0, 1.0], 3.0)
            .solve()
            .unwrap();
        assert!(result.converged, "LP did not converge: {}", result.message);
        assert!((result.f - (-4.0)).abs() < 1e-6, "f={}", result.f);
    }

    #[test]
    fn test_solve_with_explicit_simplex() {
        let result = OptimProblem::new(2)
            .x0(&[0.0, 0.0])
            .linear_objective(&[1.0, 2.0])
            .linear_constraint_eq(&[1.0, 1.0], 3.0)
            .solve_with(SolverChoice::Simplex)
            .unwrap();
        assert!(result.converged);
        assert!((result.f - 3.0).abs() < 1e-6, "f={}", result.f);
    }

    #[test]
    fn test_auto_qp_solve() {
        let result = OptimProblem::new(2)
            .x0(&[0.0, 0.0])
            .quadratic_objective_dense(&[1.0, 0.0, 0.0, 1.0], &[-3.0, -3.0])
            .all_bounds(&[(0.0, 1.0), (0.0, 1.0)])
            .solve()
            .unwrap();
        assert!(result.converged, "QP did not converge: {}", result.message);
        assert!((result.x[0] - 1.0).abs() < 1e-4, "x0={}", result.x[0]);
        assert!((result.x[1] - 1.0).abs() < 1e-4, "x1={}", result.x[1]);
    }

    #[test]
    fn test_auto_qp_equality() {
        let result = OptimProblem::new(2)
            .x0(&[0.0, 0.0])
            .quadratic_objective_dense(&[1.0, 0.0, 0.0, 1.0], &[0.0, 0.0])
            .linear_constraint_eq(&[1.0, 1.0], 1.0)
            .solve()
            .unwrap();
        assert!(result.converged, "QP did not converge: {}", result.message);
        assert!((result.x[0] - 0.5).abs() < 1e-4, "x0={}", result.x[0]);
        assert!((result.x[1] - 0.5).abs() < 1e-4, "x1={}", result.x[1]);
    }

    #[test]
    fn test_solve_with_explicit_active_set() {
        let result = OptimProblem::new(2)
            .x0(&[0.0, 0.0])
            .quadratic_objective_dense(&[2.0, 0.0, 0.0, 2.0], &[-2.0, -4.0])
            .solve_with(SolverChoice::ActiveSetQP)
            .unwrap();
        assert!(result.converged);
        assert!((result.x[0] - 1.0).abs() < 1e-6, "x0={}", result.x[0]);
        assert!((result.x[1] - 2.0).abs() < 1e-6, "x1={}", result.x[1]);
    }

    #[test]
    fn test_select_solver_milp() {
        let p = OptimProblem::new(2)
            .linear_objective(&[1.0, 2.0])
            .integer_var(0)
            .linear_constraint_ineq(&[1.0, 1.0], 10.0);
        assert_eq!(select_solver(&p), SolverChoice::MixedIntegerLP);
    }

    #[test]
    fn test_auto_milp_solve() {
        // minimize -3x1 - 5x2 s.t. x1+2x2<=6, 2x1+x2<=8, integer
        let result = OptimProblem::new(2)
            .linear_objective(&[-3.0, -5.0])
            .linear_constraint_ineq(&[1.0, 2.0], 6.0)
            .linear_constraint_ineq(&[2.0, 1.0], 8.0)
            .integer_var(0)
            .integer_var(1)
            .solve()
            .unwrap();
        assert!(result.converged);
        assert!((result.x[0] - result.x[0].round()).abs() < 1e-6);
        assert!((result.x[1] - result.x[1].round()).abs() < 1e-6);
    }

    #[test]
    fn test_milp_via_solve_with() {
        let result = OptimProblem::new(2)
            .linear_objective(&[-1.0, -1.0])
            .linear_constraint_ineq(&[1.0, 1.0], 3.5)
            .integer_var(0)
            .solve_with(SolverChoice::MixedIntegerLP)
            .unwrap();
        assert!(result.converged);
        assert!((result.x[0] - result.x[0].round()).abs() < 1e-6);
    }

    #[test]
    fn test_select_solver_global() {
        let p = OptimProblem::new(2)
            .x0(&[0.0, 0.0])
            .objective(|x: &[f64]| x[0] * x[0] + x[1] * x[1])
            .all_bounds(&[(-5.0, 5.0), (-5.0, 5.0)])
            .global(true);
        assert_eq!(select_solver(&p), SolverChoice::DifferentialEvolution);
    }

    #[test]
    fn test_auto_de_solve() {
        let result = OptimProblem::new(2)
            .objective(|x: &[f64]| x[0] * x[0] + x[1] * x[1])
            .all_bounds(&[(-5.0, 5.0), (-5.0, 5.0)])
            .global(true)
            .solve()
            .unwrap();
        assert!(result.f < 0.1, "f={}", result.f);
    }

    #[test]
    fn test_select_solver_multi_objective() {
        let p = OptimProblem::new(2)
            .multi_objective(vec![
                Box::new(|x: &[f64]| x[0] * x[0]),
                Box::new(|x: &[f64]| (x[0] - 2.0).powi(2)),
            ])
            .all_bounds(&[(0.0, 4.0), (0.0, 4.0)]);
        assert_eq!(select_solver(&p), SolverChoice::NsgaII);
    }

    #[test]
    fn test_auto_nsga2_solve() {
        let result = OptimProblem::new(1)
            .multi_objective(vec![
                Box::new(|x: &[f64]| x[0] * x[0]),
                Box::new(|x: &[f64]| (x[0] - 2.0).powi(2)),
            ])
            .all_bounds(&[(0.0, 4.0)])
            .solve()
            .unwrap();
        assert!(result.pareto.is_some());
        let pareto = result.pareto.unwrap();
        assert!(!pareto.points.is_empty());
    }

    #[test]
    fn test_maximize_quadratic() {
        // maximize -(x0^2 + x1^2) => maximum at (0, 0) with value 0
        let result = OptimProblem::new(2)
            .x0(&[5.0, 3.0])
            .maximize(|x: &[f64]| -(x[0] * x[0] + x[1] * x[1]))
            .solve()
            .unwrap();
        assert!(result.converged, "did not converge: {}", result.message);
        assert!(result.x[0].abs() < 1e-6);
        assert!(result.x[1].abs() < 1e-6);
        assert!(result.f.abs() < 1e-10, "f={}", result.f); // Maximum value is 0
    }

    #[test]
    fn test_maximize_with_bounds() {
        // maximize x0 + x1 subject to 0 <= xi <= 1
        // Maximum at (1, 1) with value 2
        let result = OptimProblem::new(2)
            .x0(&[0.5, 0.5])
            .maximize(|x: &[f64]| x[0] + x[1])
            .maximize_gradient(|_x: &[f64], g: &mut [f64]| {
                g[0] = 1.0;
                g[1] = 1.0;
            })
            .all_bounds(&[(0.0, 1.0), (0.0, 1.0)])
            .solve()
            .unwrap();
        assert!(result.converged, "did not converge: {}", result.message);
        assert!((result.x[0] - 1.0).abs() < 1e-4, "x0={}", result.x[0]);
        assert!((result.x[1] - 1.0).abs() < 1e-4, "x1={}", result.x[1]);
        assert!((result.f - 2.0).abs() < 1e-4, "f={}", result.f);
    }
}

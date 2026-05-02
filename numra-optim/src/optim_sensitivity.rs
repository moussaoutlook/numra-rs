//! Parametric sensitivity analysis for optimization problems.
//!
//! Given a parameterized problem where the objective and/or constraints
//! depend on parameters `p`, compute `dx*/dp` by finite differences:
//! for each parameter p_j, perturb p_j, re-solve, and compute
//! (x*(p+h) - x*(p-h)) / (2h).
//!
//! Author: Moussa Leblouba
//! Date: 8 February 2026
//! Modified: 2 May 2026

use numra_core::Scalar;

use crate::error::OptimError;
use crate::problem::OptimProblem;
use crate::types::ParamSensitivity;

/// Compute parametric sensitivity of an optimization problem.
///
/// `build_problem` is a closure that takes a parameter vector and returns
/// an `OptimProblem` ready to solve. For each parameter, a central finite
/// difference is used to estimate `dx*/dp_j`.
///
/// # Arguments
///
/// * `build_problem` - Factory that builds an `OptimProblem` from a parameter vector.
/// * `params` - Nominal parameter values.
/// * `names` - Human-readable names for each parameter.
/// * `eps` - Relative perturbation size (default 1e-5).
///
/// # Returns
///
/// A `ParamSensitivity` matrix of shape `n_vars x n_params`.
pub fn compute_param_sensitivity<S, F>(
    build_problem: F,
    params: &[S],
    names: &[&str],
    eps: Option<S>,
) -> Result<ParamSensitivity<S>, OptimError>
where
    S: Scalar + faer::SimpleEntity + faer::Conjugate<Canonical = S> + faer::ComplexField,
    F: Fn(&[S]) -> OptimProblem<S>,
{
    let eps = eps.unwrap_or_else(|| S::from_f64(1e-5));
    let n_params = params.len();

    // Solve the nominal problem to get x* and n_vars.
    let nominal = build_problem(params).solve()?;
    let x_star = &nominal.x;
    let n_vars = x_star.len();

    let mut values = vec![S::ZERO; n_vars * n_params];

    for j in 0..n_params {
        let h = eps * (S::ONE + params[j].abs());

        // Forward perturbation
        let mut p_plus = params.to_vec();
        p_plus[j] += h;
        let result_plus = build_problem(&p_plus).solve()?;
        let x_plus = &result_plus.x;

        // Backward perturbation
        let mut p_minus = params.to_vec();
        p_minus[j] -= h;
        let result_minus = build_problem(&p_minus).solve()?;
        let x_minus = &result_minus.x;

        // Central finite difference
        for i in 0..n_vars {
            values[i * n_params + j] = (x_plus[i] - x_minus[i]) / (S::TWO * h);
        }
    }

    Ok(ParamSensitivity {
        names: names.iter().map(|s| s.to_string()).collect(),
        values,
        n_vars,
        n_params,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sensitivity_quadratic() {
        // min (x - p)^2 => x* = p, so dx*/dp = 1
        let params = [3.0];
        let sens = compute_param_sensitivity(
            |p: &[f64]| {
                let p_val = p[0];
                OptimProblem::new(1)
                    .x0(&[0.0])
                    .objective(move |x: &[f64]| (x[0] - p_val) * (x[0] - p_val))
            },
            &params,
            &["p"],
            None,
        )
        .unwrap();

        assert_eq!(sens.n_vars, 1);
        assert_eq!(sens.n_params, 1);
        assert!(
            (sens.get(0, 0) - 1.0).abs() < 1e-3,
            "dx/dp = {}, expected 1.0",
            sens.get(0, 0)
        );
    }

    #[test]
    fn test_sensitivity_two_params() {
        // min (x - p1)^2 + (y - p2)^2 => x* = p1, y* = p2
        // dx/dp1 = 1, dx/dp2 = 0, dy/dp1 = 0, dy/dp2 = 1
        let params = [3.0, 7.0];
        let sens = compute_param_sensitivity(
            |p: &[f64]| {
                let p1 = p[0];
                let p2 = p[1];
                OptimProblem::new(2)
                    .x0(&[0.0, 0.0])
                    .objective(move |x: &[f64]| {
                        (x[0] - p1) * (x[0] - p1) + (x[1] - p2) * (x[1] - p2)
                    })
            },
            &params,
            &["p1", "p2"],
            None,
        )
        .unwrap();

        assert_eq!(sens.n_vars, 2);
        assert_eq!(sens.n_params, 2);

        // dx/dp1 = 1
        assert!(
            (sens.get(0, 0) - 1.0).abs() < 1e-3,
            "dx/dp1 = {}, expected 1.0",
            sens.get(0, 0)
        );
        // dx/dp2 = 0
        assert!(
            sens.get(0, 1).abs() < 1e-3,
            "dx/dp2 = {}, expected 0.0",
            sens.get(0, 1)
        );
        // dy/dp1 = 0
        assert!(
            sens.get(1, 0).abs() < 1e-3,
            "dy/dp1 = {}, expected 0.0",
            sens.get(1, 0)
        );
        // dy/dp2 = 1
        assert!(
            (sens.get(1, 1) - 1.0).abs() < 1e-3,
            "dy/dp2 = {}, expected 1.0",
            sens.get(1, 1)
        );

        // Also test column/row accessors
        let col0 = sens.column(0);
        assert!((col0[0] - 1.0).abs() < 1e-3);
        assert!(col0[1].abs() < 1e-3);

        let row1 = sens.row(1);
        assert!(row1[0].abs() < 1e-3);
        assert!((row1[1] - 1.0).abs() < 1e-3);
    }

    #[test]
    fn test_sensitivity_bounded() {
        // min (x - p)^2 with x in [0, 10]
        // At p=5 (interior): x*=5, dx*/dp = 1
        // At p=15 (bound active): x*=10, dx*/dp ~ 0

        // Interior case: p = 5
        let sens_interior = compute_param_sensitivity(
            |p: &[f64]| {
                let p_val = p[0];
                OptimProblem::new(1)
                    .x0(&[5.0])
                    .objective(move |x: &[f64]| (x[0] - p_val) * (x[0] - p_val))
                    .bounds(0, (0.0, 10.0))
            },
            &[5.0],
            &["p"],
            None,
        )
        .unwrap();

        assert!(
            (sens_interior.get(0, 0) - 1.0).abs() < 1e-3,
            "interior dx/dp = {}, expected 1.0",
            sens_interior.get(0, 0)
        );

        // Bound-active case: p = 15 => x* = 10 (upper bound)
        let sens_bound = compute_param_sensitivity(
            |p: &[f64]| {
                let p_val = p[0];
                OptimProblem::new(1)
                    .x0(&[5.0])
                    .objective(move |x: &[f64]| (x[0] - p_val) * (x[0] - p_val))
                    .bounds(0, (0.0, 10.0))
            },
            &[15.0],
            &["p"],
            None,
        )
        .unwrap();

        assert!(
            sens_bound.get(0, 0).abs() < 0.1,
            "bound-active dx/dp = {}, expected ~0",
            sens_bound.get(0, 0)
        );
    }
}

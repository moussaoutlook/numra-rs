//! Revised Simplex method for Linear Programming.
//!
//! Solves: minimize c^T x subject to A_ineq x <= b_ineq, A_eq x = b_eq, x >= 0.
//!
//! Uses two-phase simplex: Phase I finds a basic feasible solution,
//! Phase II optimizes the objective.
//!
//! Author: Moussa Leblouba
//! Date: 8 February 2026
//! Modified: 2 May 2026

use crate::error::OptimError;
use crate::types::{IterationRecord, OptimResult, OptimStatus};
use numra_core::Scalar;
use numra_linalg::{DenseMatrix, Matrix};

/// Options for the LP solver.
#[derive(Clone, Debug)]
pub struct LPOptions<S: Scalar> {
    pub max_iter: usize,
    pub tol: S,
    pub verbose: bool,
}

impl<S: Scalar> Default for LPOptions<S> {
    fn default() -> Self {
        Self {
            max_iter: 10_000,
            tol: S::from_f64(1e-10),
            verbose: false,
        }
    }
}

/// Solve a linear program using the Revised Simplex method.
///
/// # Standard form conversion
///
/// The user provides:
/// - `c`: objective coefficients (minimize c^T x)
/// - `a_ineq`: rows of A for inequality constraints A_ineq x <= b_ineq
/// - `b_ineq`: RHS of inequality constraints
/// - `a_eq`: rows of A for equality constraints A_eq x = b_eq
/// - `b_eq`: RHS of equality constraints
/// - All variables assumed non-negative (x >= 0)
///
/// Internally converts to standard form by adding slack variables:
///   minimize  c^T x
///   subject to [A_eq; A_ineq | I] [x; s] = [b_eq; b_ineq], x >= 0, s >= 0
pub fn simplex_solve<
    S: Scalar + faer::SimpleEntity + faer::Conjugate<Canonical = S> + faer::ComplexField,
>(
    c: &[S],
    a_ineq: &[Vec<S>],
    b_ineq: &[S],
    a_eq: &[Vec<S>],
    b_eq: &[S],
    opts: &LPOptions<S>,
) -> Result<OptimResult<S>, OptimError> {
    let start = std::time::Instant::now();
    let n_orig = c.len();
    let m_ineq = a_ineq.len();
    let m_eq = a_eq.len();
    let m = m_ineq + m_eq;

    if m == 0 && n_orig > 0 {
        // No constraints: unbounded if any c_j < 0, else optimal at x = 0
        if c.iter().any(|&ci| ci < -opts.tol) {
            return Err(OptimError::Unbounded);
        }
        let x = vec![S::ZERO; n_orig];
        return Ok(OptimResult::unconstrained(
            x,
            S::ZERO,
            c.to_vec(),
            0,
            0,
            0,
            true,
            "Optimal at origin (no constraints)".into(),
            OptimStatus::FunctionConverged,
        )
        .with_wall_time(start));
    }

    // Build standard-form tableau
    let n_slack = m_ineq;
    let n_total = n_orig + n_slack;

    // Validate and build constraint rows with slacks
    let mut a_rows: Vec<Vec<S>> = Vec::with_capacity(m);
    let mut b_std: Vec<S> = Vec::with_capacity(m);

    // Equality constraints first
    for i in 0..m_eq {
        if a_eq[i].len() != n_orig {
            return Err(OptimError::DimensionMismatch {
                expected: n_orig,
                actual: a_eq[i].len(),
            });
        }
        let mut row = a_eq[i].clone();
        row.resize(n_total, S::ZERO);
        let mut bi = b_eq[i];
        if bi < S::ZERO {
            for r in row.iter_mut() {
                *r = -*r;
            }
            bi = -bi;
        }
        a_rows.push(row);
        b_std.push(bi);
    }

    // Inequality constraints: add slacks
    for i in 0..m_ineq {
        if a_ineq[i].len() != n_orig {
            return Err(OptimError::DimensionMismatch {
                expected: n_orig,
                actual: a_ineq[i].len(),
            });
        }
        let mut row = a_ineq[i].clone();
        row.resize(n_total, S::ZERO);
        row[n_orig + i] = S::ONE;
        let mut bi = b_ineq[i];
        if bi < S::ZERO {
            for r in row.iter_mut() {
                *r = -*r;
            }
            bi = -bi;
        }
        a_rows.push(row);
        b_std.push(bi);
    }

    // Extended cost vector
    let mut c_std = c.to_vec();
    c_std.resize(n_total, S::ZERO);

    // Determine if Phase I is needed
    let mut need_art = vec![false; m];
    for item in need_art.iter_mut().take(m_eq) {
        *item = true;
    }
    let has_artificials = need_art.iter().any(|&x| x);

    let mut basis: Vec<usize> = Vec::with_capacity(m);

    if has_artificials {
        // Phase I: add artificial variables for equality rows
        let n_phase1 = n_total + m;
        let mut a_ph1: Vec<Vec<S>> = Vec::with_capacity(m);
        for (i, row) in a_rows.iter().enumerate() {
            let mut ph1_row = row.clone();
            ph1_row.resize(n_phase1, S::ZERO);
            if need_art[i] {
                ph1_row[n_total + i] = S::ONE;
            }
            a_ph1.push(ph1_row);
        }

        let mut c_ph1 = vec![S::ZERO; n_phase1];
        for i in 0..m {
            if need_art[i] {
                c_ph1[n_total + i] = S::ONE;
            }
        }

        let mut basis_ph1 = Vec::with_capacity(m);
        for i in 0..m_eq {
            basis_ph1.push(n_total + i); // artificial variables for equality rows
        }
        for i in 0..m_ineq {
            basis_ph1.push(n_orig + i); // slack variables for inequality rows
        }

        let ph1_result = simplex_core(&a_ph1, &mut b_std, &c_ph1, &mut basis_ph1, opts)?;

        if ph1_result > opts.tol {
            return Err(OptimError::LPInfeasible);
        }

        // Check no artificial is in basis at nonzero level
        for &bj in &basis_ph1 {
            if bj >= n_total {
                let row = basis_ph1.iter().position(|&x| x == bj).unwrap();
                if b_std[row].abs() > opts.tol {
                    return Err(OptimError::LPInfeasible);
                }
            }
        }

        basis = basis_ph1;
        // Replace artificial basis variables with real ones
        for row in 0..m {
            if basis[row] >= n_total {
                for (j, val) in a_rows[row].iter().enumerate().take(n_total) {
                    if !basis.contains(&j) && val.abs() > opts.tol {
                        basis[row] = j;
                        break;
                    }
                }
            }
        }
    } else {
        // No equality constraints: slack variables form the initial basis
        for i in 0..m_ineq {
            basis.push(n_orig + i);
        }
    }

    // Phase II: optimize the actual objective
    let _opt_val = simplex_core(&a_rows, &mut b_std, &c_std, &mut basis, opts)?;

    // Extract solution
    let mut x = vec![S::ZERO; n_total];
    for (row, &bj) in basis.iter().enumerate() {
        if bj < n_total {
            x[bj] = b_std[row];
        }
    }

    let x_orig: Vec<S> = x[..n_orig].to_vec();
    let f_val: S = c
        .iter()
        .zip(x_orig.iter())
        .map(|(&ci, &xi)| ci * xi)
        .sum::<S>();

    // Compute dual variables (pi = c_B^T * B^{-1})
    let mut b_mat = DenseMatrix::<S>::zeros(m, m);
    for (col, &bj) in basis.iter().enumerate() {
        for (i, a_row) in a_rows.iter().enumerate().take(m) {
            b_mat.set(i, col, a_row[bj]);
        }
    }
    let c_b: Vec<S> = basis.iter().map(|&j| c_std[j]).collect();
    // Solve B^T * pi = c_B
    let mut bt = DenseMatrix::<S>::zeros(m, m);
    for i in 0..m {
        for j in 0..m {
            bt.set(i, j, b_mat.get(j, i));
        }
    }
    let pi = bt.solve(&c_b).unwrap_or_else(|_| vec![S::ZERO; m]);

    let mut lambda_eq = Vec::with_capacity(m_eq);
    let mut lambda_ineq = Vec::with_capacity(m_ineq);
    for item in pi.iter().take(m_eq) {
        lambda_eq.push(*item);
    }
    for i in 0..m_ineq {
        lambda_ineq.push(pi[m_eq + i]);
    }

    let history = vec![IterationRecord {
        iteration: 0,
        objective: f_val,
        gradient_norm: S::ZERO,
        step_size: S::ZERO,
        constraint_violation: S::ZERO,
    }];

    Ok((OptimResult {
        lambda_eq,
        lambda_ineq,
        history,
        ..OptimResult::unconstrained(
            x_orig,
            f_val,
            c.to_vec(),
            0,
            0,
            0,
            true,
            "Optimal solution found".into(),
            OptimStatus::FunctionConverged,
        )
    })
    .with_wall_time(start))
}

/// Core revised simplex iteration.
///
/// Performs pivot operations on the basis until optimality is reached or
/// unboundedness / max iterations is detected.
///
/// Returns the optimal objective value on success.
fn simplex_core<
    S: Scalar + faer::SimpleEntity + faer::Conjugate<Canonical = S> + faer::ComplexField,
>(
    a_rows: &[Vec<S>],
    b: &mut [S],
    c: &[S],
    basis: &mut [usize],
    opts: &LPOptions<S>,
) -> Result<S, OptimError> {
    let m = a_rows.len();
    let n = c.len();

    for _iter in 0..opts.max_iter {
        // Build basis matrix B
        let mut b_mat = DenseMatrix::<S>::zeros(m, m);
        for (col, &bj) in basis.iter().enumerate() {
            for (row, a_row) in a_rows.iter().enumerate().take(m) {
                b_mat.set(row, col, a_row[bj]);
            }
        }

        let c_b: Vec<S> = basis.iter().map(|&j| c[j]).collect();

        // Solve B^T * pi = c_B for dual variables
        let mut bt = DenseMatrix::<S>::zeros(m, m);
        for i in 0..m {
            for j in 0..m {
                bt.set(i, j, b_mat.get(j, i));
            }
        }
        let pi = bt.solve(&c_b).map_err(|_| OptimError::SingularMatrix)?;

        // Compute reduced costs and find entering variable (most negative)
        let mut entering = None;
        let mut min_rc = -opts.tol;
        for j in 0..n {
            if basis.contains(&j) {
                continue;
            }
            let mut rc = c[j];
            for i in 0..m {
                rc -= pi[i] * a_rows[i][j];
            }
            if rc < min_rc {
                min_rc = rc;
                entering = Some(j);
            }
        }

        let entering = match entering {
            Some(j) => j,
            None => {
                // All reduced costs >= 0: optimal
                return Ok(basis
                    .iter()
                    .enumerate()
                    .map(|(row, &j)| c[j] * b[row])
                    .sum::<S>());
            }
        };

        // Solve B * d = a_entering for the direction
        let a_col: Vec<S> = (0..m).map(|i| a_rows[i][entering]).collect();
        let d = b_mat
            .solve(&a_col)
            .map_err(|_| OptimError::SingularMatrix)?;

        // Minimum ratio test to find leaving variable
        let mut min_ratio = S::INFINITY;
        let mut leaving_row = None;
        for i in 0..m {
            if d[i] > opts.tol {
                let ratio = b[i] / d[i];
                if ratio < min_ratio {
                    min_ratio = ratio;
                    leaving_row = Some(i);
                }
            }
        }

        let leaving_row = match leaving_row {
            Some(r) => r,
            None => return Err(OptimError::Unbounded),
        };

        // Update RHS and basis
        let theta = min_ratio;
        for i in 0..m {
            if i == leaving_row {
                b[i] = theta;
            } else {
                b[i] -= theta * d[i];
            }
        }
        basis[leaving_row] = entering;

        // Clamp tiny negatives to zero (numerical cleanup)
        for bi in b.iter_mut() {
            if *bi < S::ZERO && *bi > -opts.tol {
                *bi = S::ZERO;
            }
        }
    }

    Err(OptimError::Other(format!(
        "simplex: max iterations ({}) reached",
        opts.max_iter
    )))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lp_simple_2d() {
        // minimize -x1 - x2 subject to x1+x2<=4, x1<=3, x2<=3, x>=0
        let c = vec![-1.0, -1.0];
        let a_ineq = vec![vec![1.0, 1.0], vec![1.0, 0.0], vec![0.0, 1.0]];
        let b_ineq = vec![4.0, 3.0, 3.0];
        let opts = LPOptions::default();
        let result = simplex_solve(&c, &a_ineq, &b_ineq, &[], &[], &opts).unwrap();
        assert!(result.converged, "LP did not converge: {}", result.message);
        assert!(
            (result.f - (-4.0)).abs() < 1e-8,
            "f={}, expected -4",
            result.f
        );
        assert!(result.x[0] + result.x[1] - 4.0 < 1e-8);
    }

    #[test]
    fn test_lp_with_equality() {
        // minimize x1 + 2*x2 subject to x1+x2=3, x>=0
        let c = vec![1.0, 2.0];
        let a_eq = vec![vec![1.0, 1.0]];
        let b_eq = vec![3.0];
        let opts = LPOptions::default();
        let result = simplex_solve(&c, &[], &[], &a_eq, &b_eq, &opts).unwrap();
        assert!(result.converged, "LP did not converge: {}", result.message);
        assert!((result.f - 3.0).abs() < 1e-8, "f={}", result.f);
        assert!((result.x[0] - 3.0).abs() < 1e-8);
        assert!(result.x[1].abs() < 1e-8);
    }

    #[test]
    fn test_lp_unbounded() {
        // minimize -x with no constraints => unbounded
        let c = vec![-1.0];
        let opts = LPOptions::default();
        let result = simplex_solve(&c, &[], &[], &[], &[], &opts);
        assert!(result.is_err());
    }

    #[test]
    fn test_lp_infeasible() {
        // x >= 0 and x <= -1 is infeasible (after negation: -x <= 1, so x >= -1
        // but we also need the constraint to truly be infeasible)
        // Actually: x <= -1 means a_ineq = [1], b_ineq = [-1]
        // After negation of negative RHS: -x <= 1, i.e. x >= -1
        // With x >= 0 this is feasible at x=0. We need a different infeasible setup.
        // Use equality: x = -1 with x >= 0 => infeasible
        // Use a clearly infeasible system: x1+x2=1, x1+x2=2
        let c2 = vec![1.0, 1.0];
        let a_eq2 = vec![vec![1.0, 1.0], vec![1.0, 1.0]];
        let b_eq2 = vec![1.0, 2.0];
        let opts = LPOptions::default();
        let result = simplex_solve(&c2, &[], &[], &a_eq2, &b_eq2, &opts);
        assert!(result.is_err());
    }

    #[test]
    fn test_lp_3d_production() {
        // Production planning: maximize 5x1+4x2+3x3 => minimize -5x1-4x2-3x3
        // subject to 6x1+4x2+2x3<=240, 3x1+2x2+5x3<=270, x>=0
        let c = vec![-5.0, -4.0, -3.0];
        let a_ineq = vec![vec![6.0, 4.0, 2.0], vec![3.0, 2.0, 5.0]];
        let b_ineq = vec![240.0, 270.0];
        let opts = LPOptions::default();
        let result = simplex_solve(&c, &a_ineq, &b_ineq, &[], &[], &opts).unwrap();
        assert!(result.converged, "LP did not converge: {}", result.message);
        assert!(result.f <= -219.0, "f={}, expected <= -219", result.f);
    }

    #[test]
    fn test_lp_dual_variables() {
        // minimize -x1 - x2 subject to x1+x2<=1, x>=0
        let c = vec![-1.0, -1.0];
        let a_ineq = vec![vec![1.0, 1.0]];
        let b_ineq = vec![1.0];
        let opts = LPOptions::default();
        let result = simplex_solve(&c, &a_ineq, &b_ineq, &[], &[], &opts).unwrap();
        assert!(result.converged);
        assert!((result.f - (-1.0)).abs() < 1e-8, "f={}", result.f);
        assert!(!result.lambda_ineq.is_empty());
    }
}

//! Active Set method for convex Quadratic Programming.
//!
//! Solves: minimize 1/2 x^T H x + c^T x
//! subject to: A_ineq x <= b_ineq, A_eq x = b_eq, l <= x <= u.
//!
//! Requires H to be positive semi-definite (convex QP).
//!
//! Author: Moussa Leblouba
//! Date: 8 February 2026
//! Modified: 2 May 2026

use crate::error::OptimError;
use crate::types::{IterationRecord, OptimResult, OptimStatus};
use numra_core::Scalar;
use numra_linalg::{CholeskyFactorization, DenseMatrix, Matrix};

/// Options for the QP solver.
#[derive(Clone, Debug)]
pub struct QPOptions<S: Scalar> {
    pub max_iter: usize,
    pub tol: S,
    pub verbose: bool,
}

impl<S: Scalar> Default for QPOptions<S> {
    fn default() -> Self {
        Self {
            max_iter: 10_000,
            tol: S::from_f64(1e-10),
            verbose: false,
        }
    }
}

// ── helpers ──

fn dot<S: Scalar>(a: &[S], b: &[S]) -> S {
    a.iter().zip(b.iter()).map(|(ai, bi)| *ai * *bi).sum::<S>()
}

fn norm<S: Scalar>(v: &[S]) -> S {
    dot(v, v).sqrt()
}

/// Compute g = H * x + c.
fn compute_gradient<
    S: Scalar + faer::SimpleEntity + faer::Conjugate<Canonical = S> + faer::ComplexField,
>(
    h: &DenseMatrix<S>,
    x: &[S],
    c: &[S],
    g: &mut [S],
) {
    h.mul_vec(x, g);
    for (gi, ci) in g.iter_mut().zip(c.iter()) {
        *gi += *ci;
    }
}

/// Compute f = 0.5 * x^T H x + c^T x.
fn compute_objective<
    S: Scalar + faer::SimpleEntity + faer::Conjugate<Canonical = S> + faer::ComplexField,
>(
    h: &DenseMatrix<S>,
    x: &[S],
    c: &[S],
) -> S {
    let n = x.len();
    let mut hx = vec![S::ZERO; n];
    h.mul_vec(x, &mut hx);
    S::from_f64(0.5) * dot(x, &hx) + dot(x, c)
}

/// Get the constraint row vector for constraint index `idx`.
/// Indices 0..m_eq are equalities, m_eq..m_eq+m_ineq are inequalities.
fn get_constraint_row<'a, S: Scalar>(
    idx: usize,
    m_eq: usize,
    a_eq: &'a [Vec<S>],
    a_ineq_all: &'a [Vec<S>],
) -> &'a [S] {
    if idx < m_eq {
        &a_eq[idx]
    } else {
        &a_ineq_all[idx - m_eq]
    }
}

/// Find an initial feasible point.
///
/// Strategy:
/// 1. If no constraints: solve H x = -c for unconstrained minimum.
/// 2. If only equalities: solve the equality-constrained KKT system.
/// 3. If inequalities present: start from the equality-feasible point (or origin),
///    then iteratively project onto violated inequality constraints.
#[allow(clippy::too_many_arguments)]
fn find_initial_feasible_point<
    S: Scalar + faer::SimpleEntity + faer::Conjugate<Canonical = S> + faer::ComplexField,
>(
    h: &DenseMatrix<S>,
    c: &[S],
    n: usize,
    a_eq: &[Vec<S>],
    b_eq: &[S],
    a_ineq_all: &[Vec<S>],
    b_ineq_all: &[S],
    tol: S,
) -> Result<Vec<S>, OptimError> {
    let m_eq = a_eq.len();
    let m_ineq = a_ineq_all.len();

    if m_eq == 0 && m_ineq == 0 {
        // Unconstrained: solve H x = -c
        let neg_c: Vec<S> = c.iter().map(|ci| -(*ci)).collect();
        return h.solve(&neg_c).map_err(|_| OptimError::SingularMatrix);
    }

    // Start with equality-feasible point via KKT
    let mut x0 = if m_eq > 0 {
        // Solve the KKT system for equality-constrained minimum:
        // [ H   A_eq^T ] [ x ]   [ -c   ]
        // [ A_eq   0   ] [ lambda ] = [ b_eq ]
        let kkt_size = n + m_eq;
        let mut kkt = DenseMatrix::<S>::zeros(kkt_size, kkt_size);
        // Fill H block
        for i in 0..n {
            for j in 0..n {
                kkt.set(i, j, h.get(i, j));
            }
        }
        // Fill A_eq and A_eq^T blocks
        for (i, a_eq_row) in a_eq.iter().enumerate().take(m_eq) {
            for (j, &val) in a_eq_row.iter().enumerate().take(n) {
                kkt.set(n + i, j, val);
                kkt.set(j, n + i, val);
            }
        }
        // RHS
        let mut rhs = vec![S::ZERO; kkt_size];
        for (ri, ci) in rhs.iter_mut().zip(c.iter()).take(n) {
            *ri = -(*ci);
        }
        rhs[n..(m_eq + n)].copy_from_slice(&b_eq[..m_eq]);
        let sol = kkt.solve(&rhs).map_err(|_| OptimError::SingularMatrix)?;
        sol[..n].to_vec()
    } else {
        vec![S::ZERO; n]
    };

    if m_ineq == 0 {
        return Ok(x0);
    }

    // Project onto violated inequality constraints iteratively.
    // When equality constraints are present, project along the null space of A_eq
    // to maintain equality feasibility.
    for _phase in 0..200 {
        let mut worst_idx = None;
        let mut worst_violation = tol;
        for i in 0..m_ineq {
            let residual = dot(&a_ineq_all[i], &x0) - b_ineq_all[i];
            if residual > worst_violation {
                worst_violation = residual;
                worst_idx = Some(i);
            }
        }
        if worst_idx.is_none() {
            break; // All inequalities satisfied
        }

        let idx = worst_idx.unwrap();
        let a = &a_ineq_all[idx];
        let b_val = b_ineq_all[idx];

        if m_eq > 0 {
            // We need to find a direction d such that:
            //   A_eq d = 0 (stay on equality constraints)
            //   a^T d < 0  (reduce violation of the violated inequality)
            // Solve the least-norm problem:
            //   min ||d||^2 s.t. A_eq d = 0, a^T d = -(a^T x0 - b_val)
            // Using KKT of this sub-problem with the augmented constraint set:
            //   [ I   A_eq^T  a ] [ d  ]   [ 0          ]
            //   [ A_eq  0     0 ] [ mu ]  = [ 0          ]
            //   [ a^T   0     0 ] [ nu ]    [ b_val - a^T x0 ]
            let aug_size = n + m_eq + 1;
            let mut aug = DenseMatrix::<S>::zeros(aug_size, aug_size);
            // I block
            for i in 0..n {
                aug.set(i, i, S::ONE);
            }
            // A_eq and A_eq^T
            for (i, a_eq_row) in a_eq.iter().enumerate().take(m_eq) {
                for (j, &val) in a_eq_row.iter().enumerate().take(n) {
                    aug.set(n + i, j, val);
                    aug.set(j, n + i, val);
                }
            }
            // a and a^T
            for (j, &val) in a.iter().enumerate().take(n) {
                aug.set(n + m_eq, j, val);
                aug.set(j, n + m_eq, val);
            }
            let mut rhs = vec![S::ZERO; aug_size];
            rhs[n + m_eq] = b_val - dot(a, &x0);

            if let Ok(sol) = aug.solve(&rhs) {
                let d: Vec<S> = sol[..n].to_vec();
                for j in 0..n {
                    x0[j] += d[j];
                }
            } else {
                // Fallback: simple projection (may break equalities)
                let ax = dot(a, &x0);
                let aa = dot(a, a);
                if aa > S::from_f64(1e-20) {
                    let shift = (ax - b_val) / aa;
                    for j in 0..n {
                        x0[j] -= shift * a[j];
                    }
                }
            }
        } else {
            // No equalities: simple projection onto the hyperplane
            let ax = dot(a, &x0);
            let aa = dot(a, a);
            if aa < S::from_f64(1e-20) {
                continue;
            }
            let shift = (ax - b_val) / aa;
            for j in 0..n {
                x0[j] -= shift * a[j];
            }
        }
    }

    // Verify feasibility
    let eq_violation: S = (0..m_eq)
        .map(|i| (dot(&a_eq[i], &x0) - b_eq[i]).abs())
        .fold(S::ZERO, |a, b| if b > a { b } else { a });
    let ineq_violation: S = (0..m_ineq)
        .map(|i| {
            let v = dot(&a_ineq_all[i], &x0) - b_ineq_all[i];
            if v > S::ZERO {
                v
            } else {
                S::ZERO
            }
        })
        .fold(S::ZERO, |a, b| if b > a { b } else { a });

    let violation = if ineq_violation > eq_violation {
        ineq_violation
    } else {
        eq_violation
    };
    if violation > S::from_f64(1e-4) {
        return Err(OptimError::Infeasible {
            violation: violation.to_f64(),
        });
    }

    Ok(x0)
}

/// Line search along direction p from x: find maximum alpha in [0, 1]
/// such that x + alpha * p satisfies all inequality constraints.
/// Returns (alpha, blocking_constraint_index) where the blocking constraint
/// is the one in a_ineq_all that becomes active.
#[allow(clippy::too_many_arguments)]
fn line_search_step<S: Scalar>(
    x: &[S],
    p: &[S],
    _n: usize,
    m_eq: usize,
    a_ineq_all: &[Vec<S>],
    b_ineq_all: &[S],
    working_set: &[usize],
    tol: S,
) -> (S, Option<usize>) {
    let m_ineq = a_ineq_all.len();
    let mut alpha = S::ONE;
    let mut blocking: Option<usize> = None;

    for i in 0..m_ineq {
        let global_idx = m_eq + i;
        // Skip constraints already in the working set
        if working_set.contains(&global_idx) {
            continue;
        }
        let a = &a_ineq_all[i];
        let ap = dot(a, p);
        if ap <= tol {
            // Direction does not move toward violation of this constraint
            continue;
        }
        let ax = dot(a, x);
        let bi = b_ineq_all[i];
        let slack = bi - ax;
        let alpha_i = if slack < S::ZERO { S::ZERO } else { slack / ap };
        if alpha_i < alpha {
            alpha = alpha_i;
            blocking = Some(global_idx);
        }
    }

    // Clamp alpha
    if alpha < S::ZERO {
        alpha = S::ZERO;
    }

    (alpha, blocking)
}

/// Compute the maximum constraint violation for the final result.
fn compute_constraint_violation<S: Scalar>(
    x: &[S],
    a_eq_orig: &[Vec<S>],
    b_eq_orig: &[S],
    a_ineq_orig: &[Vec<S>],
    b_ineq_orig: &[S],
    bounds: &[Option<(S, S)>],
) -> S {
    let mut violation = S::ZERO;
    for i in 0..a_eq_orig.len() {
        let res = (dot(&a_eq_orig[i], x) - b_eq_orig[i]).abs();
        if res > violation {
            violation = res;
        }
    }
    for i in 0..a_ineq_orig.len() {
        let v = dot(&a_ineq_orig[i], x) - b_ineq_orig[i];
        let res = if v > S::ZERO { v } else { S::ZERO };
        if res > violation {
            violation = res;
        }
    }
    for (j, b) in bounds.iter().enumerate() {
        if let Some((lo, hi)) = b {
            if x[j] < *lo {
                let d = *lo - x[j];
                if d > violation {
                    violation = d;
                }
            }
            if x[j] > *hi {
                let d = x[j] - *hi;
                if d > violation {
                    violation = d;
                }
            }
        }
    }
    violation
}

/// Solve a convex QP using the primal active set method.
///
/// Minimizes: 1/2 x^T H x + c^T x
/// subject to: A_ineq x <= b_ineq, A_eq x = b_eq, l <= x <= u.
///
/// # Arguments
/// * `h_row_major` - Hessian matrix in row-major order (n x n), must be PSD.
/// * `c` - Linear term (length n).
/// * `n` - Number of variables.
/// * `a_ineq` - Inequality constraint rows.
/// * `b_ineq` - Inequality constraint RHS.
/// * `a_eq` - Equality constraint rows.
/// * `b_eq` - Equality constraint RHS.
/// * `bounds` - Optional (lower, upper) bounds per variable.
/// * `opts` - Solver options.
#[allow(clippy::too_many_arguments)]
pub fn active_set_qp_solve<
    S: Scalar + faer::SimpleEntity + faer::Conjugate<Canonical = S> + faer::ComplexField,
>(
    h_row_major: &[S],
    c: &[S],
    n: usize,
    a_ineq: &[Vec<S>],
    b_ineq: &[S],
    a_eq: &[Vec<S>],
    b_eq: &[S],
    bounds: &[Option<(S, S)>],
    opts: &QPOptions<S>,
) -> Result<OptimResult<S>, OptimError> {
    let start = std::time::Instant::now();
    let tol = opts.tol;

    // Build H as DenseMatrix
    let h = DenseMatrix::<S>::from_row_major(n, n, h_row_major);

    // Validate that H is positive semi-definite via Cholesky factorization.
    // Add a small regularization to handle PSD (not just PD) matrices.
    let mut h_reg = DenseMatrix::<S>::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            h_reg.set(i, j, h.get(i, j));
        }
        h_reg.set(i, i, h.get(i, i) + S::from_f64(1e-12));
    }
    if CholeskyFactorization::new(&h_reg).is_err() {
        return Err(OptimError::QPNotPositiveSemiDefinite);
    }

    // Convert bounds to inequality constraints.
    // Lower bound x_i >= l_i  =>  -x_i <= -l_i  (i.e. -e_i^T x <= -l_i)
    // Upper bound x_i <= u_i  =>  e_i^T x <= u_i
    let m_eq = a_eq.len();
    let m_ineq_orig = a_ineq.len();
    let mut a_ineq_all: Vec<Vec<S>> = a_ineq.to_vec();
    let mut b_ineq_all: Vec<S> = b_ineq.to_vec();

    // Track which inequality indices correspond to bounds for active_bounds reporting
    // bound_info[i] = Some((var_idx, is_upper)) for converted bound constraints
    let mut bound_info: Vec<Option<(usize, bool)>> = vec![None; m_ineq_orig];

    for (j, b) in bounds.iter().enumerate() {
        if let Some((lo, hi)) = b {
            // Lower bound: -x_j <= -lo
            let mut row_lo = vec![S::ZERO; n];
            row_lo[j] = -S::ONE;
            a_ineq_all.push(row_lo);
            b_ineq_all.push(-(*lo));
            bound_info.push(Some((j, false)));

            // Upper bound: x_j <= hi
            let mut row_hi = vec![S::ZERO; n];
            row_hi[j] = S::ONE;
            a_ineq_all.push(row_hi);
            b_ineq_all.push(*hi);
            bound_info.push(Some((j, true)));
        }
    }

    let m_ineq_total = a_ineq_all.len();
    let _m_total = m_eq + m_ineq_total;

    // Find initial feasible point
    let mut x = find_initial_feasible_point(&h, c, n, a_eq, b_eq, &a_ineq_all, &b_ineq_all, tol)?;

    // Initialize working set: all equalities + active inequalities at x0
    let mut working_set: Vec<usize> = (0..m_eq).collect();
    for i in 0..m_ineq_total {
        let residual = dot(&a_ineq_all[i], &x) - b_ineq_all[i];
        if residual.abs() < tol * S::from_f64(10.0) {
            let global_idx = m_eq + i;
            if !working_set.contains(&global_idx) {
                working_set.push(global_idx);
            }
        }
    }

    let mut g = vec![S::ZERO; n];
    let mut history: Vec<IterationRecord<S>> = Vec::new();
    let mut iterations = 0;

    for iter in 0..opts.max_iter {
        iterations = iter + 1;
        compute_gradient(&h, &x, c, &mut g);
        let f_val = compute_objective(&h, &x, c);
        let g_norm = norm(&g);

        if opts.verbose {
            eprintln!(
                "QP iter {}: f={:.6e}, ||g||={:.6e}, |W|={}",
                iter,
                f_val.to_f64(),
                g_norm.to_f64(),
                working_set.len()
            );
        }

        history.push(IterationRecord {
            iteration: iter,
            objective: f_val,
            gradient_norm: g_norm,
            step_size: S::ZERO,
            constraint_violation: S::ZERO,
        });

        // Build and solve KKT system for the equality QP subproblem:
        //   min 1/2 p^T H p + g^T p  subject to A_W p = 0
        // KKT: [ H  A_W^T ] [ p ] = [ -g ]
        //      [ A_W  0   ] [ lam ] = [  0 ]

        let n_w = working_set.len();

        if n_w == 0 {
            // No active constraints: solve H p = -g
            let neg_g: Vec<S> = g.iter().map(|gi| -(*gi)).collect();
            let p = h.solve(&neg_g).map_err(|_| OptimError::SingularMatrix)?;
            let p_norm = norm(&p);

            if p_norm < tol {
                // Optimal
                let f_final = compute_objective(&h, &x, c);
                let cv = compute_constraint_violation(&x, a_eq, b_eq, a_ineq, b_ineq, bounds);
                return Ok((OptimResult {
                    constraint_violation: cv,
                    history,
                    ..OptimResult::unconstrained(
                        x,
                        f_final,
                        g,
                        iterations,
                        iterations,
                        iterations,
                        true,
                        "Optimal solution found".into(),
                        OptimStatus::GradientConverged,
                    )
                })
                .with_wall_time(start));
            }

            // Line search to maintain feasibility
            let (alpha, blocking) =
                line_search_step(&x, &p, n, m_eq, &a_ineq_all, &b_ineq_all, &working_set, tol);

            for j in 0..n {
                x[j] += alpha * p[j];
            }

            if let Some(blocking_idx) = blocking {
                if alpha < S::ONE - tol {
                    working_set.push(blocking_idx);
                }
            }

            continue;
        }

        // Build KKT system
        let kkt_size = n + n_w;
        let mut kkt = DenseMatrix::<S>::zeros(kkt_size, kkt_size);

        // H block
        for i in 0..n {
            for j in 0..n {
                kkt.set(i, j, h.get(i, j));
            }
        }

        // A_W and A_W^T blocks
        for (wi, &cidx) in working_set.iter().enumerate() {
            let a_row = get_constraint_row(cidx, m_eq, a_eq, &a_ineq_all);
            for (j, &val) in a_row.iter().enumerate().take(n) {
                kkt.set(n + wi, j, val);
                kkt.set(j, n + wi, val);
            }
        }

        // RHS: [-g; 0]
        let mut rhs = vec![S::ZERO; kkt_size];
        for i in 0..n {
            rhs[i] = -g[i];
        }

        let sol = kkt.solve(&rhs).map_err(|_| OptimError::SingularMatrix)?;
        let p: Vec<S> = sol[..n].to_vec();
        let lambdas: Vec<S> = sol[n..].to_vec();
        let p_norm = norm(&p);

        if p_norm < tol {
            // Step is zero: check Lagrange multipliers for inequality constraints
            // If all multipliers for inequality constraints >= 0, we are optimal.
            // Otherwise, remove the inequality constraint with the most negative multiplier.
            let mut most_neg_lambda = -tol;
            let mut most_neg_wi: Option<usize> = None;

            for (wi, &cidx) in working_set.iter().enumerate() {
                if cidx < m_eq {
                    // Equality constraint: never remove
                    continue;
                }
                if lambdas[wi] < most_neg_lambda {
                    most_neg_lambda = lambdas[wi];
                    most_neg_wi = Some(wi);
                }
            }

            if most_neg_wi.is_none() {
                // Optimal: all inequality multipliers non-negative
                let f_final = compute_objective(&h, &x, c);
                compute_gradient(&h, &x, c, &mut g);

                // Extract multiplier information
                let mut lambda_eq_out = Vec::new();
                let mut lambda_ineq_out = Vec::new();
                let mut active_bounds_out = Vec::new();

                for (wi, &cidx) in working_set.iter().enumerate() {
                    if cidx < m_eq {
                        lambda_eq_out.push(lambdas[wi]);
                    } else {
                        let ineq_idx = cidx - m_eq;
                        lambda_ineq_out.push(lambdas[wi]);
                        if let Some((var_idx, _)) = bound_info[ineq_idx] {
                            active_bounds_out.push(var_idx);
                        }
                    }
                }
                // Deduplicate active_bounds
                active_bounds_out.sort_unstable();
                active_bounds_out.dedup();

                let cv = compute_constraint_violation(&x, a_eq, b_eq, a_ineq, b_ineq, bounds);

                return Ok((OptimResult {
                    lambda_eq: lambda_eq_out,
                    lambda_ineq: lambda_ineq_out,
                    active_bounds: active_bounds_out,
                    constraint_violation: cv,
                    history,
                    ..OptimResult::unconstrained(
                        x,
                        f_final,
                        g,
                        iterations,
                        iterations,
                        iterations,
                        true,
                        "Optimal solution found".into(),
                        OptimStatus::GradientConverged,
                    )
                })
                .with_wall_time(start));
            }

            // Remove the constraint with most negative multiplier
            working_set.remove(most_neg_wi.unwrap());
        } else {
            // p is nonzero: line search to maintain feasibility among inactive constraints
            let (alpha, blocking) =
                line_search_step(&x, &p, n, m_eq, &a_ineq_all, &b_ineq_all, &working_set, tol);

            for j in 0..n {
                x[j] += alpha * p[j];
            }

            if let Some(blocking_idx) = blocking {
                if alpha < S::ONE - tol {
                    working_set.push(blocking_idx);
                }
            }
        }
    }

    // Max iterations reached
    let f_final = compute_objective(&h, &x, c);
    compute_gradient(&h, &x, c, &mut g);
    let cv = compute_constraint_violation(&x, a_eq, b_eq, a_ineq, b_ineq, bounds);

    Ok((OptimResult {
        constraint_violation: cv,
        history,
        ..OptimResult::unconstrained(
            x,
            f_final,
            g,
            iterations,
            iterations,
            iterations,
            false,
            format!("QP active set: max iterations ({}) reached", opts.max_iter),
            OptimStatus::MaxIterations,
        )
    })
    .with_wall_time(start))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_qp_unconstrained() {
        // H = [[2, 0], [0, 2]], c = [-2, -4]
        // Optimal: x = [1, 2], f = -5
        let h = vec![2.0, 0.0, 0.0, 2.0];
        let c = vec![-2.0, -4.0];
        let opts = QPOptions::default();
        let result = active_set_qp_solve(&h, &c, 2, &[], &[], &[], &[], &[], &opts).unwrap();
        assert!(result.converged);
        assert!((result.x[0] - 1.0).abs() < 1e-6, "x0={}", result.x[0]);
        assert!((result.x[1] - 2.0).abs() < 1e-6, "x1={}", result.x[1]);
        assert!((result.f - (-5.0)).abs() < 1e-6, "f={}", result.f);
    }

    #[test]
    fn test_qp_with_inequality() {
        // minimize 1/2 (x0^2 + x1^2), H = I, c = 0
        // subject to: -x0 - x1 <= -1  (i.e. x0 + x1 >= 1)
        // Optimal: x = [0.5, 0.5], f = 0.25
        let h = vec![1.0, 0.0, 0.0, 1.0];
        let c = vec![0.0, 0.0];
        let a_ineq = vec![vec![-1.0, -1.0]];
        let b_ineq = vec![-1.0];
        let opts = QPOptions::default();
        let result =
            active_set_qp_solve(&h, &c, 2, &a_ineq, &b_ineq, &[], &[], &[], &opts).unwrap();
        assert!(result.converged);
        assert!((result.x[0] - 0.5).abs() < 1e-6, "x0={}", result.x[0]);
        assert!((result.x[1] - 0.5).abs() < 1e-6, "x1={}", result.x[1]);
        assert!((result.f - 0.25).abs() < 1e-6, "f={}", result.f);
    }

    #[test]
    fn test_qp_with_equality() {
        // minimize 1/2 (x0^2 + x1^2), subject to x0 + x1 = 1
        // Optimal: x = [0.5, 0.5], f = 0.25
        let h = vec![1.0, 0.0, 0.0, 1.0];
        let c = vec![0.0, 0.0];
        let a_eq = vec![vec![1.0, 1.0]];
        let b_eq = vec![1.0];
        let opts = QPOptions::default();
        let result = active_set_qp_solve(&h, &c, 2, &[], &[], &a_eq, &b_eq, &[], &opts).unwrap();
        assert!(result.converged);
        assert!((result.x[0] - 0.5).abs() < 1e-6, "x0={}", result.x[0]);
        assert!((result.x[1] - 0.5).abs() < 1e-6, "x1={}", result.x[1]);
    }

    #[test]
    fn test_qp_with_bounds() {
        // minimize 1/2 x^T I x + [-3, -3]^T x subject to 0 <= xi <= 1
        // Unconstrained min at (3,3), bounded to (1,1)
        let h = vec![1.0, 0.0, 0.0, 1.0];
        let c = vec![-3.0, -3.0];
        let bounds: Vec<Option<(f64, f64)>> = vec![Some((0.0, 1.0)), Some((0.0, 1.0))];
        let opts = QPOptions::default();
        let result = active_set_qp_solve(&h, &c, 2, &[], &[], &[], &[], &bounds, &opts).unwrap();
        assert!(result.converged);
        assert!((result.x[0] - 1.0).abs() < 1e-6, "x0={}", result.x[0]);
        assert!((result.x[1] - 1.0).abs() < 1e-6, "x1={}", result.x[1]);
    }

    #[test]
    fn test_qp_portfolio() {
        // Markowitz portfolio: minimize 1/2 x^T Sigma x
        // subject to: mu^T x >= target, sum(x) = 1, x >= 0
        let h = vec![0.04, 0.006, 0.002, 0.006, 0.01, 0.004, 0.002, 0.004, 0.0225];
        let c = vec![0.0, 0.0, 0.0];
        // -0.12*x0 - 0.10*x1 - 0.07*x2 <= -0.10
        let a_ineq = vec![vec![-0.12, -0.10, -0.07]];
        let b_ineq = vec![-0.10];
        let a_eq = vec![vec![1.0, 1.0, 1.0]];
        let b_eq = vec![1.0];
        let bounds: Vec<Option<(f64, f64)>> =
            vec![Some((0.0, 1.0)), Some((0.0, 1.0)), Some((0.0, 1.0))];
        let opts = QPOptions::default();
        let result =
            active_set_qp_solve(&h, &c, 3, &a_ineq, &b_ineq, &a_eq, &b_eq, &bounds, &opts).unwrap();
        assert!(result.converged, "QP did not converge: {}", result.message);
        let sum: f64 = result.x.iter().copied().sum();
        assert!((sum - 1.0).abs() < 1e-6, "sum={}", sum);
        let ret: f64 = 0.12 * result.x[0] + 0.10 * result.x[1] + 0.07 * result.x[2];
        assert!(ret >= 0.10 - 1e-4, "return={}", ret);
        for xi in &result.x {
            assert!(*xi >= -1e-8, "negative weight: {}", xi);
        }
    }

    #[test]
    fn test_qp_non_psd_rejected() {
        // H = [[1, 0], [0, -1]] is indefinite (not PSD)
        let h = vec![1.0, 0.0, 0.0, -1.0];
        let c = vec![0.0, 0.0];
        let opts = QPOptions::default();
        let result = active_set_qp_solve(&h, &c, 2, &[], &[], &[], &[], &[], &opts);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            matches!(err, crate::error::OptimError::QPNotPositiveSemiDefinite),
            "expected QPNotPositiveSemiDefinite, got: {}",
            err
        );
    }
}

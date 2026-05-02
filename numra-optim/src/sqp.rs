//! Sequential Quadratic Programming (SQP) for constrained nonlinear optimization.
//!
//! At each iteration: (1) form a QP subproblem using a BFGS Hessian approximation,
//! (2) solve the QP for a search direction, (3) line search with L1 merit function.
//!
//! Author: Moussa Leblouba
//! Date: 5 March 2026
//! Modified: 2 May 2026

use numra_core::Scalar;
use numra_linalg::{DenseMatrix, Matrix};

use crate::error::OptimError;
use crate::problem::{finite_diff_gradient, Constraint, ConstraintKind};
use crate::types::{IterationRecord, OptimResult, OptimStatus};

/// Options for the SQP solver.
#[derive(Clone, Debug)]
pub struct SqpOptions<S: Scalar> {
    pub max_iter: usize,
    pub tol: S,
    pub verbose: bool,
}

impl<S: Scalar> Default for SqpOptions<S> {
    fn default() -> Self {
        Self {
            max_iter: 500,
            tol: S::from_f64(1e-6),
            verbose: false,
        }
    }
}

/// SQP solver for nonlinearly constrained optimization.
///
/// Minimizes `f(x)` subject to equality and inequality constraints.
pub fn sqp_minimize<S, F, G>(
    f: F,
    grad_f: G,
    constraints: &[Constraint<S>],
    x0: &[S],
    opts: &SqpOptions<S>,
) -> Result<OptimResult<S>, OptimError>
where
    S: Scalar + faer::SimpleEntity + faer::Conjugate<Canonical = S> + faer::ComplexField,
    F: Fn(&[S]) -> S,
    G: Fn(&[S], &mut [S]),
{
    let start = std::time::Instant::now();
    let n = x0.len();

    let eq_indices: Vec<usize> = constraints
        .iter()
        .enumerate()
        .filter(|(_, c)| c.kind == ConstraintKind::Equality)
        .map(|(i, _)| i)
        .collect();
    let ineq_indices: Vec<usize> = constraints
        .iter()
        .enumerate()
        .filter(|(_, c)| c.kind == ConstraintKind::Inequality)
        .map(|(i, _)| i)
        .collect();
    let n_eq = eq_indices.len();
    let n_ineq = ineq_indices.len();

    let mut x = x0.to_vec();
    let mut grad = vec![S::ZERO; n];
    grad_f(&x, &mut grad);

    // BFGS Hessian approximation (starts at identity)
    let mut hess = DenseMatrix::<S>::zeros(n, n);
    for i in 0..n {
        hess.set(i, i, S::ONE);
    }

    let mut n_feval = 1_usize;
    let mut n_geval = 1_usize;
    let mut history: Vec<IterationRecord<S>> = Vec::new();
    let mut converged = false;
    let mut iterations = 0;

    // L1 merit penalty parameter — start with a moderate value and increase as needed
    let mut mu = S::from_f64(10.0);
    let mut prev_step_norm = S::INFINITY;
    let mut prev_violation = S::INFINITY;
    let mut stagnation_count = 0_usize;

    for iter in 0..opts.max_iter {
        iterations = iter + 1;
        let f_val = f(&x);
        n_feval += 1;

        // Evaluate constraints
        let c_vals: Vec<S> = constraints.iter().map(|c| (c.func)(&x)).collect();

        // Constraint violation
        let max_violation = constraint_violation(&c_vals, &eq_indices, &ineq_indices);

        // Compute constraint Jacobians
        let mut c_grads: Vec<Vec<S>> = Vec::with_capacity(constraints.len());
        for c in constraints {
            let mut cg = vec![S::ZERO; n];
            if let Some(ref g) = c.grad {
                g(&x, &mut cg);
            } else {
                finite_diff_gradient(&*c.func, &x, &mut cg);
            }
            c_grads.push(cg);
        }

        // Compute Lagrangian gradient for KKT check
        let grad_norm = grad.iter().map(|&g| g * g).sum::<S>().sqrt();
        let kkt_norm = if !eq_indices.is_empty() || !ineq_indices.is_empty() {
            lagrangian_gradient_norm(&grad, &c_vals, &c_grads, &eq_indices, &ineq_indices, n)
        } else {
            grad_norm
        };

        if opts.verbose {
            eprintln!(
                "SQP iter {}: f={:.6e}, ||kkt||={:.2e}, cv={:.2e}, mu={:.1e}, step={:.2e}",
                iter,
                f_val.to_f64(),
                kkt_norm.to_f64(),
                max_violation.to_f64(),
                mu.to_f64(),
                prev_step_norm.to_f64()
            );
        }

        history.push(IterationRecord {
            iteration: iter,
            objective: f_val,
            gradient_norm: kkt_norm,
            step_size: prev_step_norm,
            constraint_violation: max_violation,
        });

        // Check convergence: KKT optimality + feasibility
        if kkt_norm < opts.tol && max_violation < opts.tol {
            converged = true;
            break;
        }
        // Also converge if the actual step is tiny and we're feasible
        if prev_step_norm < opts.tol && max_violation < opts.tol {
            converged = true;
            break;
        }

        // Feasibility restoration: if constraint violation stagnates, take a Newton step
        // toward the constraint surface (ignoring the objective temporarily)
        if max_violation > opts.tol {
            let cv_decrease = prev_violation - max_violation;
            if cv_decrease < S::from_f64(1e-8) * max_violation {
                stagnation_count += 1;
            } else {
                stagnation_count = 0;
            }

            if stagnation_count >= 5 {
                // Feasibility restoration: minimize sum c_i^2 via one Gauss-Newton step
                // J^T J d = -J^T c  (normal equations)
                let m = constraints.len();
                let mut jtj = vec![S::ZERO; n * n];
                let mut jtc = vec![S::ZERO; n];
                for i in 0..m {
                    let ci = c_vals[i];
                    // Skip inactive inequalities
                    if constraints[i].kind == ConstraintKind::Inequality && ci < S::ZERO {
                        continue;
                    }
                    for j in 0..n {
                        jtc[j] += c_grads[i][j] * ci;
                        for k in 0..n {
                            jtj[j * n + k] += c_grads[i][j] * c_grads[i][k];
                        }
                    }
                }
                // Regularize and solve
                for j in 0..n {
                    jtj[j * n + j] += S::from_f64(1e-6);
                }
                let jtj_mat = DenseMatrix::<S>::from_row_major(n, n, &jtj);
                let neg_jtc: Vec<S> = jtc.iter().map(|&v| -v).collect();
                if let Ok(d_feas) = jtj_mat.solve(&neg_jtc) {
                    // Take a damped feasibility step
                    let mut alpha_f = S::ONE;
                    for _ in 0..10 {
                        let x_trial: Vec<S> = (0..n).map(|j| x[j] + alpha_f * d_feas[j]).collect();
                        let c_trial: Vec<S> =
                            constraints.iter().map(|c| (c.func)(&x_trial)).collect();
                        let cv_trial = constraint_violation(&c_trial, &eq_indices, &ineq_indices);
                        if cv_trial < max_violation {
                            x = x_trial;
                            grad_f(&x, &mut grad);
                            n_geval += 1;
                            n_feval += 1;
                            // Reset Hessian to identity after restoration
                            for i in 0..n {
                                for j in 0..n {
                                    hess.set(i, j, if i == j { S::ONE } else { S::ZERO });
                                }
                            }
                            stagnation_count = 0;
                            break;
                        }
                        alpha_f *= S::HALF;
                    }
                }
                prev_violation = max_violation;
                continue;
            }
        } else {
            stagnation_count = 0;
        }
        prev_violation = max_violation;

        // Solve QP subproblem for search direction and multiplier estimates
        let (d, max_mult) = solve_qp_subproblem(
            &hess,
            &grad,
            &c_vals,
            &c_grads,
            &eq_indices,
            &ineq_indices,
            n,
        );

        let d_norm = d.iter().map(|&di| di * di).sum::<S>().sqrt();
        if d_norm < S::from_f64(1e-15) {
            if max_violation < opts.tol {
                converged = true;
            }
            break;
        }

        // Update mu: must be > max |λ_i| for L1 merit to be exact
        let mu_min = max_mult + S::from_f64(0.1);
        if mu < mu_min {
            mu = mu_min;
        }

        // Compute directional derivative of L1 merit function
        let df_dd: S = grad.iter().zip(d.iter()).map(|(&gi, &di)| gi * di).sum();
        let merit_deriv = df_dd - mu * l1_penalty(&c_vals, &eq_indices, &ineq_indices);

        // Further increase mu if merit is not descending
        if merit_deriv > S::ZERO {
            mu *= S::TWO;
        }

        let merit_x = f_val + mu * l1_penalty(&c_vals, &eq_indices, &ineq_indices);

        // Backtracking line search on L1 merit function
        let mut alpha = S::ONE;
        let eta = S::from_f64(1e-4);
        let mut step_evals = 0;

        for _ in 0..30 {
            let x_trial: Vec<S> = (0..n).map(|j| x[j] + alpha * d[j]).collect();
            let f_trial = f(&x_trial);
            let c_trial: Vec<S> = constraints.iter().map(|c| (c.func)(&x_trial)).collect();
            step_evals += 1;

            let merit_trial = f_trial + mu * l1_penalty(&c_trial, &eq_indices, &ineq_indices);

            // Accept if sufficient decrease in merit
            if merit_trial <= merit_x + eta * alpha * merit_deriv {
                break;
            }
            // Also accept if merit decreased at all and alpha is small enough
            if merit_trial < merit_x && alpha < S::from_f64(0.1) {
                break;
            }

            alpha *= S::HALF;
        }
        n_feval += step_evals;

        // Update x
        let old_x = x.clone();
        let old_grad = grad.clone();
        for j in 0..n {
            x[j] += alpha * d[j];
        }
        prev_step_norm = alpha * d_norm;

        // Update gradient
        grad_f(&x, &mut grad);
        n_geval += 1;

        // BFGS Hessian update (Powell's damped variant)
        let s: Vec<S> = (0..n).map(|j| x[j] - old_x[j]).collect();
        let y: Vec<S> = (0..n).map(|j| grad[j] - old_grad[j]).collect();

        let ss: S = s.iter().map(|&si| si * si).sum();
        if ss > S::from_f64(1e-20) {
            let sy: S = s.iter().zip(y.iter()).map(|(&si, &yi)| si * yi).sum();
            let mut hs = vec![S::ZERO; n];
            hess.mul_vec(&s, &mut hs);
            let shs: S = s.iter().zip(hs.iter()).map(|(&si, &hi)| si * hi).sum();

            if shs > S::from_f64(1e-20) {
                // Powell's damping
                let theta = if sy >= S::from_f64(0.2) * shs {
                    S::ONE
                } else if (shs - sy).to_f64().abs() > 1e-20 {
                    S::from_f64(0.8) * shs / (shs - sy)
                } else {
                    S::ONE
                };

                let r: Vec<S> = (0..n)
                    .map(|j| theta * y[j] + (S::ONE - theta) * hs[j])
                    .collect();
                let sr: S = s.iter().zip(r.iter()).map(|(&si, &ri)| si * ri).sum();

                if sr > S::from_f64(1e-20) {
                    for i in 0..n {
                        for j in 0..n {
                            let val = hess.get(i, j) - hs[i] * hs[j] / shs + r[i] * r[j] / sr;
                            hess.set(i, j, val);
                        }
                    }
                }
            }
        }
    }

    let f_final = f(&x);
    let c_final: Vec<S> = constraints.iter().map(|c| (c.func)(&x)).collect();
    let final_violation = constraint_violation(&c_final, &eq_indices, &ineq_indices);

    let (status, message) = if converged {
        (
            OptimStatus::GradientConverged,
            format!("SQP converged after {} iterations", iterations),
        )
    } else {
        (
            OptimStatus::MaxIterations,
            format!("SQP: max iterations ({}) reached", opts.max_iter),
        )
    };

    Ok(OptimResult {
        x,
        f: f_final,
        grad,
        iterations,
        n_feval,
        n_geval,
        converged,
        message,
        status,
        history,
        lambda_eq: vec![S::ZERO; n_eq],
        lambda_ineq: vec![S::ZERO; n_ineq],
        active_bounds: Vec::new(),
        constraint_violation: final_violation,
        wall_time_secs: 0.0,
        pareto: None,
        sensitivity: None,
    }
    .with_wall_time(start))
}

/// Compute the Lagrangian gradient norm as a KKT optimality measure.
///
/// For equality constraints: solve min ||∇f + A_eq^T λ||^2 for λ.
/// For a practical approximation: compute the component of ∇f orthogonal to the
/// constraint gradient space. If ∇f is in span(∇c_i), KKT norm ≈ 0.
fn lagrangian_gradient_norm<S: Scalar>(
    grad: &[S],
    c_vals: &[S],
    c_grads: &[Vec<S>],
    eq_indices: &[usize],
    ineq_indices: &[usize],
    n: usize,
) -> S {
    // Collect active constraint gradients (equality + active inequality)
    let mut active_grads: Vec<&Vec<S>> = Vec::new();
    for &i in eq_indices {
        active_grads.push(&c_grads[i]);
    }
    for &i in ineq_indices {
        if c_vals[i] > S::from_f64(-1e-6) {
            active_grads.push(&c_grads[i]);
        }
    }

    if active_grads.is_empty() {
        return grad.iter().map(|&g| g * g).sum::<S>().sqrt();
    }

    // Compute lambda via normal equations: A * A^T * lambda = -A * grad
    let m = active_grads.len();
    // For small m, direct computation
    // Project grad onto the space spanned by active constraint gradients
    // Residual = grad - A^T * (A * A^T)^{-1} * A * grad
    // For m=1: residual = grad - (a . grad)/(a . a) * a
    if m == 1 {
        let a = &active_grads[0];
        let ag: S = a.iter().zip(grad.iter()).map(|(&ai, &gi)| ai * gi).sum();
        let aa: S = a.iter().map(|&ai| ai * ai).sum();
        if aa < S::from_f64(1e-20) {
            return grad.iter().map(|&g| g * g).sum::<S>().sqrt();
        }
        // KKT: ∇f + λ∇c = 0  =>  λ = -(a·g)/(a·a)
        let lambda = -ag / aa;
        let mut residual_sq = S::ZERO;
        for j in 0..n {
            let r = grad[j] + lambda * a[j];
            residual_sq += r * r;
        }
        return residual_sq.sqrt();
    }

    // General case: form A (m x n), compute A*A^T (m x m), solve for lambda
    // Then residual = grad + A^T * lambda
    let mut aat = vec![S::ZERO; m * m];
    let mut ag = vec![S::ZERO; m];
    for i in 0..m {
        for j in 0..m {
            let mut dot = S::ZERO;
            for (ak_i, ak_j) in active_grads[i].iter().zip(active_grads[j].iter()).take(n) {
                dot += *ak_i * *ak_j;
            }
            aat[i * m + j] = dot;
        }
        let mut dot = S::ZERO;
        for k in 0..n {
            dot += active_grads[i][k] * grad[k];
        }
        ag[i] = -dot;
    }

    // Solve A*A^T * lambda = -A * grad via Gaussian elimination (small system)
    // Augmented matrix [aat | ag]
    let mut aug = vec![S::ZERO; m * (m + 1)];
    for i in 0..m {
        for j in 0..m {
            aug[i * (m + 1) + j] = aat[i * m + j];
        }
        aug[i * (m + 1) + m] = ag[i];
    }

    for col in 0..m {
        // Find pivot
        let mut max_abs = S::ZERO;
        let mut pivot_row = col;
        for row in col..m {
            let v = aug[row * (m + 1) + col].abs();
            if v > max_abs {
                max_abs = v;
                pivot_row = row;
            }
        }
        if max_abs < S::from_f64(1e-20) {
            // Singular — just return gradient norm
            return grad.iter().map(|&g| g * g).sum::<S>().sqrt();
        }
        // Swap rows
        if pivot_row != col {
            for j in 0..=m {
                aug.swap(col * (m + 1) + j, pivot_row * (m + 1) + j);
            }
        }
        // Eliminate
        let pivot = aug[col * (m + 1) + col];
        for row in (col + 1)..m {
            let factor = aug[row * (m + 1) + col] / pivot;
            for j in col..=m {
                let val = aug[col * (m + 1) + j];
                aug[row * (m + 1) + j] -= factor * val;
            }
        }
    }

    // Back-substitute
    let mut lambda = vec![S::ZERO; m];
    for i in (0..m).rev() {
        let mut s = aug[i * (m + 1) + m];
        for j in (i + 1)..m {
            s -= aug[i * (m + 1) + j] * lambda[j];
        }
        lambda[i] = s / aug[i * (m + 1) + i];
    }

    // Compute Lagrangian gradient: ∇f + Σ λ_i ∇c_i
    let mut residual_sq = S::ZERO;
    for j in 0..n {
        let mut r = grad[j];
        for (i, ag_i) in active_grads.iter().enumerate() {
            r += lambda[i] * ag_i[j];
        }
        residual_sq += r * r;
    }
    residual_sq.sqrt()
}

/// Compute max constraint violation.
fn constraint_violation<S: Scalar>(
    c_vals: &[S],
    eq_indices: &[usize],
    ineq_indices: &[usize],
) -> S {
    let mut v = S::ZERO;
    for &i in eq_indices {
        let cv = c_vals[i].abs();
        if cv > v {
            v = cv;
        }
    }
    for &i in ineq_indices {
        if c_vals[i] > v {
            v = c_vals[i];
        }
    }
    v
}

/// Compute L1 penalty: sum|h_eq| + sum max(0, g_ineq).
fn l1_penalty<S: Scalar>(c_vals: &[S], eq_indices: &[usize], ineq_indices: &[usize]) -> S {
    let mut p = S::ZERO;
    for &i in eq_indices {
        p += c_vals[i].abs();
    }
    for &i in ineq_indices {
        if c_vals[i] > S::ZERO {
            p += c_vals[i];
        }
    }
    p
}

/// Solve the QP subproblem via KKT system.
/// Returns (search_direction, max_abs_multiplier).
fn solve_qp_subproblem<S>(
    hess: &DenseMatrix<S>,
    grad: &[S],
    c_vals: &[S],
    c_grads: &[Vec<S>],
    eq_indices: &[usize],
    ineq_indices: &[usize],
    n: usize,
) -> (Vec<S>, S)
where
    S: Scalar + faer::SimpleEntity + faer::Conjugate<Canonical = S> + faer::ComplexField,
{
    // Active inequalities: violated or nearly active (tighter threshold)
    let active_ineq: Vec<usize> = ineq_indices
        .iter()
        .copied()
        .filter(|&i| c_vals[i] > S::from_f64(-1e-6))
        .collect();

    // Filter linearly dependent constraints via incremental independence check.
    // Accept a constraint gradient only if it adds significant new direction.
    let mut independent_eq: Vec<usize> = Vec::new();
    let mut independent_ineq: Vec<usize> = Vec::new();
    let mut accepted_grads: Vec<&Vec<S>> = Vec::new();

    for &ci in eq_indices {
        if is_independent(&c_grads[ci], &accepted_grads, n) {
            accepted_grads.push(&c_grads[ci]);
            independent_eq.push(ci);
        }
    }
    for &ci in &active_ineq {
        if is_independent(&c_grads[ci], &accepted_grads, n) {
            accepted_grads.push(&c_grads[ci]);
            independent_ineq.push(ci);
        }
    }

    let n_ind_eq = independent_eq.len();
    let n_ind_ineq = independent_ineq.len();
    let n_total = n_ind_eq + n_ind_ineq;

    if n_total == 0 {
        // Unconstrained QP: H * d = -grad
        let neg_grad: Vec<S> = grad.iter().map(|&g| -g).collect();
        let d = hess.solve(&neg_grad).unwrap_or(neg_grad);
        return (d, S::ZERO);
    }

    // Build KKT system
    let kkt_n = n + n_total;
    let mut kkt = DenseMatrix::<S>::zeros(kkt_n, kkt_n);
    let mut rhs = vec![S::ZERO; kkt_n];

    // H block with scale-dependent regularization
    let diag_max = (0..n)
        .map(|i| hess.get(i, i).abs())
        .fold(S::ZERO, |a, b| if b > a { b } else { a });
    let reg = S::from_f64(1e-8) * (S::ONE + diag_max);
    for i in 0..n {
        for j in 0..n {
            kkt.set(i, j, hess.get(i, j));
        }
        kkt.set(i, i, kkt.get(i, i) + reg);
        rhs[i] = -grad[i];
    }

    // Equality constraints
    for (row, &ci) in independent_eq.iter().enumerate() {
        let cg = &c_grads[ci];
        for (j, &cgj) in cg.iter().enumerate().take(n) {
            kkt.set(n + row, j, cgj);
            kkt.set(j, n + row, cgj);
        }
        rhs[n + row] = -c_vals[ci];
    }

    // Active inequality constraints
    for (row, &ci) in independent_ineq.iter().enumerate() {
        let kkt_row = n + n_ind_eq + row;
        let cg = &c_grads[ci];
        for (j, &cgj) in cg.iter().enumerate().take(n) {
            kkt.set(kkt_row, j, cgj);
            kkt.set(j, kkt_row, cgj);
        }
        rhs[kkt_row] = -c_vals[ci];
    }

    // Small negative diagonal on constraint block for numerical regularization
    for i in 0..n_total {
        kkt.set(n + i, n + i, kkt.get(n + i, n + i) - S::from_f64(1e-10));
    }

    match kkt.solve(&rhs) {
        Ok(sol) => {
            let d = sol[..n].to_vec();
            // Extract max absolute multiplier
            let max_mult =
                sol[n..]
                    .iter()
                    .map(|&v| v.abs())
                    .fold(S::ZERO, |a, b| if b > a { b } else { a });
            (d, max_mult)
        }
        Err(_) => {
            // Fallback: steepest descent toward feasibility
            let mut d: Vec<S> = grad.iter().map(|&g| -g).collect();
            for &i in eq_indices {
                let cv = c_vals[i];
                for j in 0..n {
                    d[j] -= cv * c_grads[i][j];
                }
            }
            for &i in ineq_indices {
                if c_vals[i] > S::ZERO {
                    for j in 0..n {
                        d[j] -= c_vals[i] * c_grads[i][j];
                    }
                }
            }
            (d, S::from_f64(100.0))
        }
    }
}

/// Check if a constraint gradient is linearly independent from already-accepted ones.
/// Uses a Gram-Schmidt-like residual check: project `g` onto the span of `accepted`,
/// and reject if the residual is too small relative to `g`.
fn is_independent<S: Scalar>(g: &[S], accepted: &[&Vec<S>], n: usize) -> bool {
    let g_norm_sq: S = g.iter().map(|&v| v * v).sum();
    if g_norm_sq < S::from_f64(1e-20) {
        return false; // zero gradient
    }

    if accepted.is_empty() {
        return true;
    }

    // Compute residual after projecting out accepted directions
    let mut residual: Vec<S> = g.to_vec();
    for &a in accepted {
        let a_norm_sq: S = a.iter().map(|&v| v * v).sum();
        if a_norm_sq < S::from_f64(1e-20) {
            continue;
        }
        let dot: S = residual
            .iter()
            .zip(a.iter())
            .take(n)
            .map(|(&ri, &ai)| ri * ai)
            .sum();
        let coeff = dot / a_norm_sq;
        for j in 0..n.min(residual.len()) {
            residual[j] -= coeff * a[j];
        }
    }

    let res_norm_sq: S = residual.iter().map(|&v| v * v).sum();
    // Independent if residual retains at least 1% of original norm
    res_norm_sq > S::from_f64(1e-4) * g_norm_sq
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::problem::Constraint;

    #[test]
    fn test_sqp_equality_circle() {
        // minimize x0 + x1  s.t.  x0^2 + x1^2 = 1
        // Optimal: (-1/sqrt(2), -1/sqrt(2))
        let constraints = vec![Constraint {
            func: Box::new(|x: &[f64]| x[0] * x[0] + x[1] * x[1] - 1.0),
            grad: Some(Box::new(|x: &[f64], g: &mut [f64]| {
                g[0] = 2.0 * x[0];
                g[1] = 2.0 * x[1];
            })),
            kind: ConstraintKind::Equality,
        }];

        let result = sqp_minimize(
            |x: &[f64]| x[0] + x[1],
            |_x: &[f64], g: &mut [f64]| {
                g[0] = 1.0;
                g[1] = 1.0;
            },
            &constraints,
            &[1.0, 0.0],
            &SqpOptions::default(),
        )
        .unwrap();

        assert!(result.converged, "SQP did not converge: {}", result.message);
        let expected = -1.0 / 2.0_f64.sqrt();
        assert!(
            (result.x[0] - expected).abs() < 0.05,
            "x0={}, expected {}",
            result.x[0],
            expected
        );
        assert!(result.constraint_violation < 1e-3);
    }

    #[test]
    fn test_sqp_inequality() {
        // minimize (x0-2)^2 + (x1-2)^2  s.t.  x0 + x1 <= 2
        // Optimal at (1, 1)
        let constraints = vec![Constraint {
            func: Box::new(|x: &[f64]| x[0] + x[1] - 2.0),
            grad: Some(Box::new(|_x: &[f64], g: &mut [f64]| {
                g[0] = 1.0;
                g[1] = 1.0;
            })),
            kind: ConstraintKind::Inequality,
        }];

        let result = sqp_minimize(
            |x: &[f64]| (x[0] - 2.0).powi(2) + (x[1] - 2.0).powi(2),
            |x: &[f64], g: &mut [f64]| {
                g[0] = 2.0 * (x[0] - 2.0);
                g[1] = 2.0 * (x[1] - 2.0);
            },
            &constraints,
            &[0.0, 0.0],
            &SqpOptions::default(),
        )
        .unwrap();

        assert!(result.converged, "SQP did not converge: {}", result.message);
        assert!(
            (result.x[0] - 1.0).abs() < 0.1,
            "x0={}, expected ~1.0",
            result.x[0]
        );
    }

    #[test]
    fn test_sqp_mixed_constraints() {
        // minimize x0^2 + x1^2
        // s.t. x0 + x1 = 1, 0.6 - x0 <= 0
        // Optimal: x0=0.6, x1=0.4
        let constraints = vec![
            Constraint {
                func: Box::new(|x: &[f64]| x[0] + x[1] - 1.0),
                grad: Some(Box::new(|_x: &[f64], g: &mut [f64]| {
                    g[0] = 1.0;
                    g[1] = 1.0;
                })),
                kind: ConstraintKind::Equality,
            },
            Constraint {
                func: Box::new(|x: &[f64]| 0.6 - x[0]),
                grad: Some(Box::new(|_x: &[f64], g: &mut [f64]| {
                    g[0] = -1.0;
                    g[1] = 0.0;
                })),
                kind: ConstraintKind::Inequality,
            },
        ];

        let result = sqp_minimize(
            |x: &[f64]| x[0] * x[0] + x[1] * x[1],
            |x: &[f64], g: &mut [f64]| {
                g[0] = 2.0 * x[0];
                g[1] = 2.0 * x[1];
            },
            &constraints,
            &[1.0, 1.0],
            &SqpOptions::default(),
        )
        .unwrap();

        assert!(result.converged, "SQP did not converge: {}", result.message);
        assert!(
            (result.x[0] - 0.6).abs() < 0.1,
            "x0={}, expected ~0.6",
            result.x[0]
        );
    }

    #[test]
    fn test_sqp_multiple_inequality_constraints() {
        // minimize x0^2 + x1^2 s.t. x0 >= 1, x1 >= 1, x0 + x1 <= 4
        // 3 active or near-active inequality constraints
        // Optimal at (1, 1)
        let constraints = vec![
            Constraint {
                func: Box::new(|x: &[f64]| -x[0] + 1.0), // 1 - x0 <= 0
                grad: Some(Box::new(|_x: &[f64], g: &mut [f64]| {
                    g[0] = -1.0;
                    g[1] = 0.0;
                })),
                kind: ConstraintKind::Inequality,
            },
            Constraint {
                func: Box::new(|x: &[f64]| -x[1] + 1.0), // 1 - x1 <= 0
                grad: Some(Box::new(|_x: &[f64], g: &mut [f64]| {
                    g[0] = 0.0;
                    g[1] = -1.0;
                })),
                kind: ConstraintKind::Inequality,
            },
            Constraint {
                func: Box::new(|x: &[f64]| x[0] + x[1] - 4.0), // x0+x1 <= 4
                grad: Some(Box::new(|_x: &[f64], g: &mut [f64]| {
                    g[0] = 1.0;
                    g[1] = 1.0;
                })),
                kind: ConstraintKind::Inequality,
            },
        ];

        let result = sqp_minimize(
            |x: &[f64]| x[0] * x[0] + x[1] * x[1],
            |x: &[f64], g: &mut [f64]| {
                g[0] = 2.0 * x[0];
                g[1] = 2.0 * x[1];
            },
            &constraints,
            &[0.5, 0.5],
            &SqpOptions::default(),
        )
        .unwrap();

        assert!(result.converged, "SQP did not converge: {}", result.message);
        assert!(
            (result.x[0] - 1.0).abs() < 0.15,
            "x0={}, expected ~1.0",
            result.x[0]
        );
        assert!(
            (result.x[1] - 1.0).abs() < 0.15,
            "x1={}, expected ~1.0",
            result.x[1]
        );
    }

    #[test]
    fn test_sqp_dependent_constraints() {
        // Nearly parallel constraints: x0 + x1 <= 2 and x0 + 1.001*x1 <= 2.001
        // minimize (x0-3)^2 + (x1-3)^2 -> optimal at (1, 1)
        let constraints = vec![
            Constraint {
                func: Box::new(|x: &[f64]| x[0] + x[1] - 2.0),
                grad: Some(Box::new(|_x: &[f64], g: &mut [f64]| {
                    g[0] = 1.0;
                    g[1] = 1.0;
                })),
                kind: ConstraintKind::Inequality,
            },
            Constraint {
                func: Box::new(|x: &[f64]| x[0] + 1.001 * x[1] - 2.001),
                grad: Some(Box::new(|_x: &[f64], g: &mut [f64]| {
                    g[0] = 1.0;
                    g[1] = 1.001;
                })),
                kind: ConstraintKind::Inequality,
            },
        ];

        let result = sqp_minimize(
            |x: &[f64]| (x[0] - 3.0).powi(2) + (x[1] - 3.0).powi(2),
            |x: &[f64], g: &mut [f64]| {
                g[0] = 2.0 * (x[0] - 3.0);
                g[1] = 2.0 * (x[1] - 3.0);
            },
            &constraints,
            &[0.0, 0.0],
            &SqpOptions::default(),
        )
        .unwrap();

        // Should not crash, and should find a feasible point near (1, 1)
        assert!(result.converged, "SQP did not converge: {}", result.message);
        assert!(
            result.x[0] + result.x[1] <= 2.0 + 0.01,
            "constraint violated: x0+x1={}",
            result.x[0] + result.x[1]
        );
    }

    #[test]
    fn test_sqp_many_constraints() {
        // 5 constraints (2 eq + 3 ineq): min x0^2 + x1^2 + x2^2
        // s.t. x0+x1+x2 = 3, x0-x1 = 0, x0>=0, x1>=0, x2>=0
        // Optimal: (1, 1, 1)
        let constraints = vec![
            Constraint {
                func: Box::new(|x: &[f64]| x[0] + x[1] + x[2] - 3.0),
                grad: Some(Box::new(|_x: &[f64], g: &mut [f64]| {
                    g[0] = 1.0;
                    g[1] = 1.0;
                    g[2] = 1.0;
                })),
                kind: ConstraintKind::Equality,
            },
            Constraint {
                func: Box::new(|x: &[f64]| x[0] - x[1]),
                grad: Some(Box::new(|_x: &[f64], g: &mut [f64]| {
                    g[0] = 1.0;
                    g[1] = -1.0;
                    g[2] = 0.0;
                })),
                kind: ConstraintKind::Equality,
            },
            Constraint {
                func: Box::new(|x: &[f64]| -x[0]),
                grad: Some(Box::new(|_x: &[f64], g: &mut [f64]| {
                    g[0] = -1.0;
                    g[1] = 0.0;
                    g[2] = 0.0;
                })),
                kind: ConstraintKind::Inequality,
            },
            Constraint {
                func: Box::new(|x: &[f64]| -x[1]),
                grad: Some(Box::new(|_x: &[f64], g: &mut [f64]| {
                    g[0] = 0.0;
                    g[1] = -1.0;
                    g[2] = 0.0;
                })),
                kind: ConstraintKind::Inequality,
            },
            Constraint {
                func: Box::new(|x: &[f64]| -x[2]),
                grad: Some(Box::new(|_x: &[f64], g: &mut [f64]| {
                    g[0] = 0.0;
                    g[1] = 0.0;
                    g[2] = -1.0;
                })),
                kind: ConstraintKind::Inequality,
            },
        ];

        let result = sqp_minimize(
            |x: &[f64]| x[0] * x[0] + x[1] * x[1] + x[2] * x[2],
            |x: &[f64], g: &mut [f64]| {
                g[0] = 2.0 * x[0];
                g[1] = 2.0 * x[1];
                g[2] = 2.0 * x[2];
            },
            &constraints,
            &[2.0, 1.0, 0.5],
            &SqpOptions::default(),
        )
        .unwrap();

        assert!(result.converged, "SQP did not converge: {}", result.message);
        assert!(
            (result.x[0] - 1.0).abs() < 0.15,
            "x0={}, expected ~1.0",
            result.x[0]
        );
        assert!(
            (result.x[1] - 1.0).abs() < 0.15,
            "x1={}, expected ~1.0",
            result.x[1]
        );
        assert!(
            (result.x[2] - 1.0).abs() < 0.15,
            "x2={}, expected ~1.0",
            result.x[2]
        );
    }

    #[test]
    fn test_sqp_unconstrained() {
        let result = sqp_minimize(
            |x: &[f64]| x[0] * x[0] + 4.0 * x[1] * x[1],
            |x: &[f64], g: &mut [f64]| {
                g[0] = 2.0 * x[0];
                g[1] = 8.0 * x[1];
            },
            &[],
            &[5.0, 3.0],
            &SqpOptions::default(),
        )
        .unwrap();

        assert!(result.converged, "did not converge: {}", result.message);
        assert!(result.x[0].abs() < 1e-3, "x0={}", result.x[0]);
        assert!(result.x[1].abs() < 1e-3, "x1={}", result.x[1]);
    }
}

//! Iterative (Krylov) linear solvers for sparse systems.
//!
//! Provides CG, GMRES, BiCGSTAB, and MINRES — the standard Krylov methods
//! for large sparse systems where direct factorization is too expensive.
//!
//! Author: Moussa Leblouba
//! Date: 9 February 2026
//! Modified: 2 May 2026

use crate::preconditioner::{IdentityPreconditioner, Preconditioner};
use crate::sparse::SparseMatrix;
use crate::Scalar;
use faer::{ComplexField, Conjugate, Entity, SimpleEntity};
use numra_core::LinalgError;

/// Options for iterative solvers.
pub struct IterativeOptions<S: Scalar> {
    /// Relative residual tolerance (stop when ||r|| / ||b|| < tol).
    pub tol: S,
    /// Maximum number of iterations.
    pub max_iter: usize,
    /// Restart parameter for GMRES (default 30).
    pub restart: usize,
}

impl<S: Scalar> Default for IterativeOptions<S> {
    fn default() -> Self {
        Self {
            tol: S::from_f64(1e-8),
            max_iter: 1000,
            restart: 30,
        }
    }
}

impl<S: Scalar> IterativeOptions<S> {
    /// Set tolerance.
    pub fn tol(mut self, tol: S) -> Self {
        self.tol = tol;
        self
    }

    /// Set maximum iterations.
    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set GMRES restart parameter.
    pub fn restart(mut self, restart: usize) -> Self {
        self.restart = restart;
        self
    }
}

/// Result of an iterative solve.
pub struct IterativeResult<S: Scalar> {
    /// Solution vector.
    pub x: Vec<S>,
    /// Final residual norm ||b - Ax||.
    pub residual_norm: S,
    /// Number of iterations performed.
    pub iterations: usize,
    /// Whether the solver converged to the requested tolerance.
    pub converged: bool,
}

// ============================================================================
// Vector helpers
// ============================================================================

fn dot<S: Scalar>(a: &[S], b: &[S]) -> S {
    a.iter()
        .zip(b.iter())
        .map(|(&ai, &bi)| ai * bi)
        .fold(S::ZERO, |acc, x| acc + x)
}

fn norm<S: Scalar>(v: &[S]) -> S {
    S::from_f64(dot(v, v).to_f64().sqrt())
}

/// y = a*x + y (AXPY)
fn axpy<S: Scalar>(a: S, x: &[S], y: &mut [S]) {
    for (yi, &xi) in y.iter_mut().zip(x.iter()) {
        *yi += a * xi;
    }
}

// ============================================================================
// Conjugate Gradient (CG) — for SPD matrices
// ============================================================================

/// Conjugate Gradient solver for symmetric positive definite systems.
///
/// Solves Ax = b where A is SPD. Convergence is guaranteed in at most n
/// iterations in exact arithmetic; in practice converges much faster for
/// well-conditioned or preconditioned systems.
pub fn cg<S: Scalar + SimpleEntity + Conjugate<Canonical = S> + ComplexField + Entity>(
    a: &SparseMatrix<S>,
    b: &[S],
    x0: Option<&[S]>,
    opts: &IterativeOptions<S>,
) -> Result<IterativeResult<S>, LinalgError> {
    pcg(a, b, &IdentityPreconditioner, x0, opts)
}

/// Preconditioned Conjugate Gradient.
///
/// Solves Ax = b using preconditioner M ≈ A. The effective system is
/// M^{-1}Ax = M^{-1}b with improved condition number.
pub fn pcg<S: Scalar + SimpleEntity + Conjugate<Canonical = S> + ComplexField + Entity>(
    a: &SparseMatrix<S>,
    b: &[S],
    precond: &dyn Preconditioner<S>,
    x0: Option<&[S]>,
    opts: &IterativeOptions<S>,
) -> Result<IterativeResult<S>, LinalgError> {
    let n = b.len();
    if a.nrows() != n || a.ncols() != n {
        return Err(LinalgError::DimensionMismatch {
            expected: (n, n),
            actual: (a.nrows(), a.ncols()),
        });
    }

    let mut x = match x0 {
        Some(x0) => x0.to_vec(),
        None => vec![S::ZERO; n],
    };

    // r = b - A*x
    let ax = a.mul_vec(&x)?;
    let mut r: Vec<S> = b.iter().zip(ax.iter()).map(|(&bi, &ai)| bi - ai).collect();

    let b_norm = norm(b);
    if b_norm.to_f64() < S::EPSILON.to_f64() {
        return Ok(IterativeResult {
            x: vec![S::ZERO; n],
            residual_norm: S::ZERO,
            iterations: 0,
            converged: true,
        });
    }

    // z = M^{-1} r
    let mut z = vec![S::ZERO; n];
    precond.apply(&r, &mut z);

    let mut p = z.clone();
    let mut rz = dot(&r, &z);

    for iter in 0..opts.max_iter {
        let r_norm = norm(&r);
        if r_norm.to_f64() / b_norm.to_f64() < opts.tol.to_f64() {
            return Ok(IterativeResult {
                x,
                residual_norm: r_norm,
                iterations: iter,
                converged: true,
            });
        }

        // ap = A * p
        let ap = a.mul_vec(&p)?;
        let pap = dot(&p, &ap);

        if pap.to_f64().abs() < S::EPSILON.to_f64() * 100.0 {
            return Ok(IterativeResult {
                x,
                residual_norm: r_norm,
                iterations: iter,
                converged: false,
            });
        }

        let alpha = S::from_f64(rz.to_f64() / pap.to_f64());

        // x += alpha * p
        axpy(alpha, &p, &mut x);
        // r -= alpha * ap
        axpy(S::ZERO - alpha, &ap, &mut r);

        // z = M^{-1} r
        precond.apply(&r, &mut z);
        let rz_new = dot(&r, &z);

        let beta = S::from_f64(rz_new.to_f64() / rz.to_f64());
        // p = z + beta * p
        for i in 0..n {
            p[i] = z[i] + beta * p[i];
        }
        rz = rz_new;
    }

    let r_norm = norm(&r);
    Ok(IterativeResult {
        x,
        residual_norm: r_norm,
        iterations: opts.max_iter,
        converged: r_norm.to_f64() / b_norm.to_f64() < opts.tol.to_f64(),
    })
}

// ============================================================================
// GMRES — for general (non-symmetric) matrices
// ============================================================================

/// Restarted GMRES solver for general non-symmetric systems.
///
/// Uses the Arnoldi iteration with modified Gram-Schmidt and Givens
/// rotations for the least-squares subproblem. Restarts after `opts.restart`
/// iterations to bound memory usage.
pub fn gmres<S: Scalar + SimpleEntity + Conjugate<Canonical = S> + ComplexField + Entity>(
    a: &SparseMatrix<S>,
    b: &[S],
    x0: Option<&[S]>,
    opts: &IterativeOptions<S>,
) -> Result<IterativeResult<S>, LinalgError> {
    let n = b.len();
    if a.nrows() != n || a.ncols() != n {
        return Err(LinalgError::DimensionMismatch {
            expected: (n, n),
            actual: (a.nrows(), a.ncols()),
        });
    }

    let b_norm = norm(b);
    if b_norm.to_f64() < S::EPSILON.to_f64() {
        return Ok(IterativeResult {
            x: vec![S::ZERO; n],
            residual_norm: S::ZERO,
            iterations: 0,
            converged: true,
        });
    }

    let mut x = match x0 {
        Some(x0) => x0.to_vec(),
        None => vec![S::ZERO; n],
    };

    let m = opts.restart.min(n); // Krylov subspace dimension
    let mut total_iter = 0;

    while total_iter < opts.max_iter {
        // r = b - A*x
        let ax = a.mul_vec(&x)?;
        let r: Vec<S> = b.iter().zip(ax.iter()).map(|(&bi, &ai)| bi - ai).collect();
        let r_norm = norm(&r);

        if r_norm.to_f64() / b_norm.to_f64() < opts.tol.to_f64() {
            return Ok(IterativeResult {
                x,
                residual_norm: r_norm,
                iterations: total_iter,
                converged: true,
            });
        }

        // Arnoldi iteration
        let mut v: Vec<Vec<S>> = Vec::with_capacity(m + 1); // Orthonormal basis
        let inv_r_norm = S::from_f64(1.0 / r_norm.to_f64());
        v.push(r.iter().map(|&ri| ri * inv_r_norm).collect());

        // Upper Hessenberg matrix H (stored column-major, (m+1) x m)
        let mut h = vec![vec![S::ZERO; m + 1]; m];

        // Givens rotation parameters
        let mut cs = vec![S::ZERO; m]; // cosines
        let mut sn = vec![S::ZERO; m]; // sines
        let mut g = vec![S::ZERO; m + 1]; // transformed RHS
        g[0] = r_norm;

        let mut k = 0; // actual Krylov dimension built
        for j in 0..m {
            if total_iter >= opts.max_iter {
                break;
            }

            // w = A * v[j]
            let w_tmp = a.mul_vec(&v[j])?;
            let mut w = w_tmp;

            // Modified Gram-Schmidt
            for i in 0..=j {
                h[j][i] = dot(&w, &v[i]);
                axpy(S::ZERO - h[j][i], &v[i], &mut w);
            }
            h[j][j + 1] = norm(&w);

            // Check for breakdown (lucky convergence or linear dependence)
            if h[j][j + 1].to_f64().abs() < S::EPSILON.to_f64() * 100.0 {
                k = j + 1;
                // Apply previous Givens rotations to column j
                apply_givens_to_col(&mut h[j], &cs, &sn, j);
                // Apply new Givens rotation
                let (c, s) = givens_rotation(h[j][j], h[j][j + 1]);
                cs[j] = c;
                sn[j] = s;
                apply_givens_pair(&mut h[j], j, c, s);
                let gj = g[j];
                let gjp1 = g[j + 1];
                g[j] = c * gj + s * gjp1;
                g[j + 1] = S::ZERO - s * gj + c * gjp1;
                total_iter += 1;
                break;
            }

            // Normalize
            let inv_hjj1 = S::from_f64(1.0 / h[j][j + 1].to_f64());
            v.push(w.iter().map(|&wi| wi * inv_hjj1).collect());

            // Apply previous Givens rotations to column j of H
            apply_givens_to_col(&mut h[j], &cs, &sn, j);

            // Compute new Givens rotation for (h[j][j], h[j][j+1])
            let (c, s) = givens_rotation(h[j][j], h[j][j + 1]);
            cs[j] = c;
            sn[j] = s;
            apply_givens_pair(&mut h[j], j, c, s);

            // Apply to g
            let gj = g[j];
            let gjp1 = g[j + 1];
            g[j] = c * gj + s * gjp1;
            g[j + 1] = S::ZERO - s * gj + c * gjp1;

            total_iter += 1;
            k = j + 1;

            // Check convergence
            if g[j + 1].abs().to_f64() / b_norm.to_f64() < opts.tol.to_f64() {
                break;
            }
        }

        if k == 0 {
            break;
        }

        // Solve the upper triangular system H[0..k, 0..k] * y = g[0..k]
        let mut y = vec![S::ZERO; k];
        for i in (0..k).rev() {
            let mut sum = g[i];
            for j in (i + 1)..k {
                sum -= h[j][i] * y[j];
            }
            if h[i][i].to_f64().abs() < S::EPSILON.to_f64() * 100.0 {
                break;
            }
            y[i] = S::from_f64(sum.to_f64() / h[i][i].to_f64());
        }

        // Update x = x + V * y
        for j in 0..k {
            axpy(y[j], &v[j], &mut x);
        }

        // Check final residual
        let ax = a.mul_vec(&x)?;
        let r_final: Vec<S> = b.iter().zip(ax.iter()).map(|(&bi, &ai)| bi - ai).collect();
        let r_final_norm = norm(&r_final);
        if r_final_norm.to_f64() / b_norm.to_f64() < opts.tol.to_f64() {
            return Ok(IterativeResult {
                x,
                residual_norm: r_final_norm,
                iterations: total_iter,
                converged: true,
            });
        }
    }

    // Final residual
    let ax = a.mul_vec(&x)?;
    let r: Vec<S> = b.iter().zip(ax.iter()).map(|(&bi, &ai)| bi - ai).collect();
    let r_norm = norm(&r);
    Ok(IterativeResult {
        x,
        residual_norm: r_norm,
        iterations: total_iter,
        converged: r_norm.to_f64() / b_norm.to_f64() < opts.tol.to_f64(),
    })
}

/// Compute Givens rotation parameters to zero out b in [a; b].
fn givens_rotation<S: Scalar>(a: S, b: S) -> (S, S) {
    let af = a.to_f64();
    let bf = b.to_f64();
    if bf.abs() < f64::EPSILON * 100.0 {
        return (S::ONE, S::ZERO);
    }
    if af.abs() < f64::EPSILON * 100.0 {
        return (S::ZERO, S::ONE);
    }
    let r = (af * af + bf * bf).sqrt();
    (S::from_f64(af / r), S::from_f64(bf / r))
}

/// Apply previously computed Givens rotations to column j of H.
fn apply_givens_to_col<S: Scalar>(col: &mut [S], cs: &[S], sn: &[S], j: usize) {
    for i in 0..j {
        let a = col[i];
        let b = col[i + 1];
        col[i] = cs[i] * a + sn[i] * b;
        col[i + 1] = (S::ZERO - sn[i]) * a + cs[i] * b;
    }
}

/// Apply a single Givens rotation to two elements at indices i, i+1 in a slice.
fn apply_givens_pair<S: Scalar>(col: &mut [S], i: usize, c: S, s: S) {
    let a = col[i];
    let b = col[i + 1];
    col[i] = c * a + s * b;
    col[i + 1] = (S::ZERO - s) * a + c * b;
}

// ============================================================================
// BiCGSTAB — for general (non-symmetric) matrices, no restart needed
// ============================================================================

/// BiCGSTAB (Bi-Conjugate Gradient Stabilized) solver.
///
/// For general non-symmetric systems. Unlike GMRES, does not require restart
/// and has fixed memory cost per iteration.
pub fn bicgstab<S: Scalar + SimpleEntity + Conjugate<Canonical = S> + ComplexField + Entity>(
    a: &SparseMatrix<S>,
    b: &[S],
    x0: Option<&[S]>,
    opts: &IterativeOptions<S>,
) -> Result<IterativeResult<S>, LinalgError> {
    let n = b.len();
    if a.nrows() != n || a.ncols() != n {
        return Err(LinalgError::DimensionMismatch {
            expected: (n, n),
            actual: (a.nrows(), a.ncols()),
        });
    }

    let b_norm = norm(b);
    if b_norm.to_f64() < S::EPSILON.to_f64() {
        return Ok(IterativeResult {
            x: vec![S::ZERO; n],
            residual_norm: S::ZERO,
            iterations: 0,
            converged: true,
        });
    }

    let mut x = match x0 {
        Some(x0) => x0.to_vec(),
        None => vec![S::ZERO; n],
    };

    // r = b - A*x
    let ax = a.mul_vec(&x)?;
    let mut r: Vec<S> = b.iter().zip(ax.iter()).map(|(&bi, &ai)| bi - ai).collect();

    let r0_hat = r.clone(); // shadow residual (fixed)
    let mut rho = S::ONE;
    let mut alpha = S::ONE;
    let mut omega = S::ONE;

    let mut p = vec![S::ZERO; n];
    let mut v = vec![S::ZERO; n];

    for iter in 0..opts.max_iter {
        let r_norm = norm(&r);
        if r_norm.to_f64() / b_norm.to_f64() < opts.tol.to_f64() {
            return Ok(IterativeResult {
                x,
                residual_norm: r_norm,
                iterations: iter,
                converged: true,
            });
        }

        let rho_new = dot(&r0_hat, &r);
        if rho_new.to_f64().abs() < S::EPSILON.to_f64() * 100.0 {
            // Breakdown: r0_hat ⊥ r
            return Ok(IterativeResult {
                x,
                residual_norm: r_norm,
                iterations: iter,
                converged: false,
            });
        }

        let beta =
            S::from_f64((rho_new.to_f64() / rho.to_f64()) * (alpha.to_f64() / omega.to_f64()));

        // p = r + beta * (p - omega * v)
        for i in 0..n {
            p[i] = r[i] + beta * (p[i] - omega * v[i]);
        }

        // v = A * p
        v = a.mul_vec(&p)?;

        let r0v = dot(&r0_hat, &v);
        if r0v.to_f64().abs() < S::EPSILON.to_f64() * 100.0 {
            return Ok(IterativeResult {
                x,
                residual_norm: r_norm,
                iterations: iter,
                converged: false,
            });
        }
        alpha = S::from_f64(rho_new.to_f64() / r0v.to_f64());

        // s = r - alpha * v
        let s: Vec<S> = r
            .iter()
            .zip(v.iter())
            .map(|(&ri, &vi)| ri - alpha * vi)
            .collect();

        let s_norm = norm(&s);
        if s_norm.to_f64() / b_norm.to_f64() < opts.tol.to_f64() {
            // x += alpha * p
            axpy(alpha, &p, &mut x);
            return Ok(IterativeResult {
                x,
                residual_norm: s_norm,
                iterations: iter + 1,
                converged: true,
            });
        }

        // t = A * s
        let t = a.mul_vec(&s)?;

        let tt = dot(&t, &t);
        if tt.to_f64().abs() < S::EPSILON.to_f64() * 100.0 {
            axpy(alpha, &p, &mut x);
            r = s;
        } else {
            omega = S::from_f64(dot(&t, &s).to_f64() / tt.to_f64());

            // x += alpha * p + omega * s
            axpy(alpha, &p, &mut x);
            axpy(omega, &s, &mut x);

            // r = s - omega * t
            for i in 0..n {
                r[i] = s[i] - omega * t[i];
            }
        }

        rho = rho_new;
    }

    let r_norm = norm(&r);
    Ok(IterativeResult {
        x,
        residual_norm: r_norm,
        iterations: opts.max_iter,
        converged: r_norm.to_f64() / b_norm.to_f64() < opts.tol.to_f64(),
    })
}

// ============================================================================
// MINRES — for symmetric (possibly indefinite) matrices
// ============================================================================

/// MINRES solver for symmetric (possibly indefinite) systems.
///
/// Based on the Lanczos process with Givens QR factorization.
/// Minimizes ||r|| = ||b - Ax|| over the Krylov subspace.
/// Unlike CG, does not require positive definiteness.
///
/// Reference: Paige & Saunders (1975), "Solution of sparse indefinite
/// systems of linear equations."
pub fn minres<S: Scalar + SimpleEntity + Conjugate<Canonical = S> + ComplexField + Entity>(
    a: &SparseMatrix<S>,
    b: &[S],
    x0: Option<&[S]>,
    opts: &IterativeOptions<S>,
) -> Result<IterativeResult<S>, LinalgError> {
    let n = b.len();
    if a.nrows() != n || a.ncols() != n {
        return Err(LinalgError::DimensionMismatch {
            expected: (n, n),
            actual: (a.nrows(), a.ncols()),
        });
    }

    let b_norm = norm(b);
    if b_norm.to_f64() < S::EPSILON.to_f64() {
        return Ok(IterativeResult {
            x: vec![S::ZERO; n],
            residual_norm: S::ZERO,
            iterations: 0,
            converged: true,
        });
    }

    let mut x = match x0 {
        Some(x0) => x0.to_vec(),
        None => vec![S::ZERO; n],
    };

    // r = b - A*x
    let ax = a.mul_vec(&x)?;
    let r: Vec<S> = b.iter().zip(ax.iter()).map(|(&bi, &ai)| bi - ai).collect();

    let mut beta_curr = norm(&r);
    if beta_curr.to_f64() / b_norm.to_f64() < opts.tol.to_f64() {
        return Ok(IterativeResult {
            x,
            residual_norm: beta_curr,
            iterations: 0,
            converged: true,
        });
    }

    // v_{k-1} = 0, v_k = r / beta_1
    let mut v_prev = vec![S::ZERO; n];
    let mut v_curr: Vec<S> = r
        .iter()
        .map(|&ri| S::from_f64(ri.to_f64() / beta_curr.to_f64()))
        .collect();

    // Givens rotation parameters (two previous rotations needed)
    let mut c_old = S::ONE;
    let mut s_old = S::ZERO;
    let mut c_curr = S::ONE;
    let mut s_curr = S::ZERO;

    // Direction vectors
    let mut w_curr = vec![S::ZERO; n];
    let mut w_prev = vec![S::ZERO; n];

    // Residual norm tracking
    let mut phi_bar = beta_curr;

    for iter in 0..opts.max_iter {
        // Lanczos step: compute A*v_curr, orthogonalize
        let av = a.mul_vec(&v_curr)?;
        let alpha = dot(&v_curr, &av);

        // v_next = A*v_curr - alpha*v_curr - beta_curr*v_prev
        let mut v_next: Vec<S> = Vec::with_capacity(n);
        for i in 0..n {
            v_next.push(av[i] - alpha * v_curr[i] - beta_curr * v_prev[i]);
        }
        let beta_next = norm(&v_next);

        // Normalize v_next
        if beta_next.to_f64().abs() > S::EPSILON.to_f64() * 100.0 {
            let inv_bn = S::from_f64(1.0 / beta_next.to_f64());
            for vi in &mut v_next {
                *vi *= inv_bn;
            }
        }

        // Apply previous Givens rotations to the new column of the tridiagonal
        // New column before rotations: [..., 0, beta_curr, alpha, beta_next]
        // Only the last 3 entries above the subdiagonal matter.

        // Step 1: Apply G_{k-2} to (0, beta_curr) at rows (k-2, k-1)
        let epsilon = s_old * beta_curr;
        let delta_hat = c_old * beta_curr;

        // Step 2: Apply G_{k-1} to (delta_hat, alpha) at rows (k-1, k)
        let delta_tilde = c_curr * delta_hat + s_curr * alpha;
        let gamma_bar = (S::ZERO - s_curr) * delta_hat + c_curr * alpha;

        // Step 3: Compute new Givens to zero beta_next from (gamma_bar, beta_next)
        let gb = gamma_bar.to_f64();
        let bn = beta_next.to_f64();
        let gamma = (gb * gb + bn * bn).sqrt();

        let (c_new, s_new) = if gamma.abs() < f64::EPSILON * 100.0 {
            (S::ONE, S::ZERO)
        } else {
            (S::from_f64(gb / gamma), S::from_f64(bn / gamma))
        };
        let gamma_s = S::from_f64(gamma);

        // Update residual norm
        let phi = c_new * phi_bar;
        let phi_bar_new = (S::ZERO - s_new) * phi_bar;

        // Direction vector: w_new = (v_curr - delta_tilde*w_curr - epsilon*w_prev) / gamma
        if gamma.abs() < f64::EPSILON * 100.0 {
            // Breakdown
            return Ok(IterativeResult {
                x,
                residual_norm: phi_bar.abs(),
                iterations: iter + 1,
                converged: false,
            });
        }

        let mut w_next = Vec::with_capacity(n);
        for i in 0..n {
            w_next.push((v_curr[i] - delta_tilde * w_curr[i] - epsilon * w_prev[i]) / gamma_s);
        }

        // Update solution: x += phi * w_next
        axpy(phi, &w_next, &mut x);

        // Shift for next iteration
        phi_bar = phi_bar_new;
        v_prev = v_curr;
        v_curr = v_next;
        w_prev = w_curr;
        w_curr = w_next;
        beta_curr = beta_next;
        c_old = c_curr;
        s_old = s_curr;
        c_curr = c_new;
        s_curr = s_new;

        // Check convergence (phi_bar tracks the residual norm)
        if phi_bar.abs().to_f64() / b_norm.to_f64() < opts.tol.to_f64() {
            return Ok(IterativeResult {
                x,
                residual_norm: phi_bar.abs(),
                iterations: iter + 1,
                converged: true,
            });
        }
    }

    Ok(IterativeResult {
        x,
        residual_norm: phi_bar.abs(),
        iterations: opts.max_iter,
        converged: phi_bar.abs().to_f64() / b_norm.to_f64() < opts.tol.to_f64(),
    })
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::preconditioner::{Ilu0, Jacobi, Ssor};

    /// Build n×n 1D Laplacian: tridiag(-1, 2, -1).
    fn laplacian_1d(n: usize) -> SparseMatrix<f64> {
        let mut triplets = Vec::new();
        for i in 0..n {
            triplets.push((i, i, 2.0));
            if i > 0 {
                triplets.push((i, i - 1, -1.0));
            }
            if i < n - 1 {
                triplets.push((i, i + 1, -1.0));
            }
        }
        SparseMatrix::from_triplets(n, n, &triplets).unwrap()
    }

    /// Build n×n non-symmetric convection-diffusion: tridiag(-1-eps, 2, -1+eps).
    fn convection_diffusion(n: usize, eps: f64) -> SparseMatrix<f64> {
        let mut triplets = Vec::new();
        for i in 0..n {
            triplets.push((i, i, 2.0));
            if i > 0 {
                triplets.push((i, i - 1, -1.0 - eps));
            }
            if i < n - 1 {
                triplets.push((i, i + 1, -1.0 + eps));
            }
        }
        SparseMatrix::from_triplets(n, n, &triplets).unwrap()
    }

    #[test]
    fn test_cg_laplacian() {
        let n = 50;
        let a = laplacian_1d(n);
        let x_exact: Vec<f64> = (0..n).map(|i| (i + 1) as f64).collect();
        let b = a.mul_vec(&x_exact).unwrap();

        let opts = IterativeOptions::default().tol(1e-10);
        let result = cg(&a, &b, None, &opts).unwrap();

        assert!(
            result.converged,
            "CG did not converge after {} iters",
            result.iterations
        );
        for i in 0..n {
            assert!(
                (result.x[i] - x_exact[i]).abs() < 1e-6,
                "CG error at {}: {} vs {}",
                i,
                result.x[i],
                x_exact[i]
            );
        }
    }

    #[test]
    fn test_cg_with_initial_guess() {
        let n = 20;
        let a = laplacian_1d(n);
        let x_exact: Vec<f64> = (0..n).map(|i| (i + 1) as f64).collect();
        let b = a.mul_vec(&x_exact).unwrap();

        // Good initial guess should need fewer iterations
        let x0: Vec<f64> = (0..n).map(|i| i as f64 + 0.9).collect();
        let opts = IterativeOptions::default().tol(1e-10);
        let result = cg(&a, &b, Some(&x0), &opts).unwrap();

        assert!(result.converged);
    }

    #[test]
    fn test_pcg_jacobi() {
        let n = 50;
        let a = laplacian_1d(n);
        let x_exact: Vec<f64> = (0..n).map(|i| (i + 1) as f64).collect();
        let b = a.mul_vec(&x_exact).unwrap();

        let jac = Jacobi::new(&a);
        let opts = IterativeOptions::default().tol(1e-10);

        let result_plain = cg(&a, &b, None, &opts).unwrap();
        let result_pcg = pcg(&a, &b, &jac, None, &opts).unwrap();

        assert!(result_pcg.converged);
        // Jacobi preconditioning should converge in similar or fewer iterations
        // for Laplacian (already diag dominant)
        assert!(
            result_pcg.iterations <= result_plain.iterations + 5,
            "PCG: {} vs CG: {}",
            result_pcg.iterations,
            result_plain.iterations
        );
    }

    #[test]
    fn test_pcg_ilu0() {
        let n = 50;
        let a = laplacian_1d(n);
        let x_exact: Vec<f64> = (0..n).map(|i| (i + 1) as f64).collect();
        let b = a.mul_vec(&x_exact).unwrap();

        let ilu = Ilu0::new(&a).unwrap();
        let opts = IterativeOptions::default().tol(1e-10);

        let result_plain = cg(&a, &b, None, &opts).unwrap();
        let result_pcg = pcg(&a, &b, &ilu, None, &opts).unwrap();

        assert!(result_pcg.converged);
        // ILU(0) on tridiagonal == exact LU, so should converge in 1 iteration
        assert!(
            result_pcg.iterations <= 2,
            "ILU(0) PCG should converge immediately on tridiagonal: {} iters",
            result_pcg.iterations
        );
        assert!(
            result_pcg.iterations < result_plain.iterations,
            "ILU(0) should beat unpreconditioned CG"
        );
    }

    #[test]
    fn test_gmres_nonsymmetric() {
        let n = 30;
        let a = convection_diffusion(n, 0.1);
        let x_exact: Vec<f64> = (0..n).map(|i| (i as f64).sin()).collect();
        let b = a.mul_vec(&x_exact).unwrap();

        let opts = IterativeOptions::default().tol(1e-10).restart(30);
        let result = gmres(&a, &b, None, &opts).unwrap();

        assert!(
            result.converged,
            "GMRES did not converge after {} iterations",
            result.iterations
        );
        for i in 0..n {
            assert!(
                (result.x[i] - x_exact[i]).abs() < 1e-6,
                "GMRES error at {}: {} vs {}",
                i,
                result.x[i],
                x_exact[i]
            );
        }
    }

    #[test]
    fn test_gmres_spd_matches_cg() {
        // GMRES on SPD should give same answer as CG
        let n = 20;
        let a = laplacian_1d(n);
        let b: Vec<f64> = (0..n).map(|i| (i + 1) as f64).collect();

        let opts = IterativeOptions::default().tol(1e-10);
        let cg_result = cg(&a, &b, None, &opts).unwrap();
        let gmres_result = gmres(&a, &b, None, &opts).unwrap();

        assert!(cg_result.converged);
        assert!(gmres_result.converged);
        for i in 0..n {
            assert!(
                (cg_result.x[i] - gmres_result.x[i]).abs() < 1e-6,
                "CG vs GMRES mismatch at {}: {} vs {}",
                i,
                cg_result.x[i],
                gmres_result.x[i]
            );
        }
    }

    #[test]
    fn test_bicgstab_nonsymmetric() {
        let n = 30;
        let a = convection_diffusion(n, 0.2);
        let x_exact: Vec<f64> = (0..n).map(|i| (i as f64).cos()).collect();
        let b = a.mul_vec(&x_exact).unwrap();

        let opts = IterativeOptions::default().tol(1e-10);
        let result = bicgstab(&a, &b, None, &opts).unwrap();

        assert!(
            result.converged,
            "BiCGSTAB did not converge after {} iters",
            result.iterations
        );
        for i in 0..n {
            assert!(
                (result.x[i] - x_exact[i]).abs() < 1e-6,
                "BiCGSTAB error at {}: {} vs {}",
                i,
                result.x[i],
                x_exact[i]
            );
        }
    }

    #[test]
    fn test_bicgstab_spd_matches_cg() {
        let n = 20;
        let a = laplacian_1d(n);
        let b: Vec<f64> = (0..n).map(|i| (i + 1) as f64).collect();

        let opts = IterativeOptions::default().tol(1e-10);
        let cg_result = cg(&a, &b, None, &opts).unwrap();
        let bicg_result = bicgstab(&a, &b, None, &opts).unwrap();

        assert!(cg_result.converged);
        assert!(bicg_result.converged);
        for i in 0..n {
            assert!(
                (cg_result.x[i] - bicg_result.x[i]).abs() < 1e-6,
                "CG vs BiCGSTAB mismatch at {}: {} vs {}",
                i,
                cg_result.x[i],
                bicg_result.x[i]
            );
        }
    }

    #[test]
    fn test_minres_symmetric_indefinite() {
        // Symmetric indefinite: diag(-1, 2, -1, 2, -1) — has both positive and negative eigenvalues
        let n = 5;
        let mut triplets = Vec::new();
        for i in 0..n {
            let sign = if i % 2 == 0 { -1.0 } else { 2.0 };
            triplets.push((i, i, sign * 3.0));
            if i > 0 {
                triplets.push((i, i - 1, 1.0));
            }
            if i < n - 1 {
                triplets.push((i, i + 1, 1.0));
            }
        }
        let a = SparseMatrix::from_triplets(n, n, &triplets).unwrap();
        let x_exact = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = a.mul_vec(&x_exact).unwrap();

        let opts = IterativeOptions::default().tol(1e-10);
        let result = minres(&a, &b, None, &opts).unwrap();

        assert!(
            result.converged,
            "MINRES did not converge after {} iters",
            result.iterations
        );
        for i in 0..n {
            assert!(
                (result.x[i] - x_exact[i]).abs() < 1e-4,
                "MINRES error at {}: {} vs {}",
                i,
                result.x[i],
                x_exact[i]
            );
        }
    }

    #[test]
    fn test_minres_spd() {
        let n = 20;
        let a = laplacian_1d(n);
        let x_exact: Vec<f64> = (0..n).map(|i| (i + 1) as f64).collect();
        let b = a.mul_vec(&x_exact).unwrap();

        let opts = IterativeOptions::default().tol(1e-10);
        let result = minres(&a, &b, None, &opts).unwrap();

        assert!(result.converged, "MINRES did not converge");
        for i in 0..n {
            assert!(
                (result.x[i] - x_exact[i]).abs() < 1e-6,
                "MINRES error at {}: {} vs {}",
                i,
                result.x[i],
                x_exact[i]
            );
        }
    }

    #[test]
    fn test_cg_zero_rhs() {
        let n = 10;
        let a = laplacian_1d(n);
        let b = vec![0.0; n];
        let opts = IterativeOptions::default();
        let result = cg(&a, &b, None, &opts).unwrap();
        assert!(result.converged);
        assert_eq!(result.iterations, 0);
    }

    #[test]
    fn test_dimension_mismatch() {
        let a = laplacian_1d(5);
        let b = vec![1.0; 3]; // wrong size
        let opts = IterativeOptions::default();
        assert!(cg(&a, &b, None, &opts).is_err());
        assert!(gmres(&a, &b, None, &opts).is_err());
        assert!(bicgstab(&a, &b, None, &opts).is_err());
        assert!(minres(&a, &b, None, &opts).is_err());
    }

    #[test]
    fn test_pcg_ssor() {
        let n = 20;
        let a = laplacian_1d(n);
        let x_exact: Vec<f64> = (0..n).map(|i| (i + 1) as f64).collect();
        let b = a.mul_vec(&x_exact).unwrap();

        // Use omega=1.0 (symmetric Gauss-Seidel) for stability
        let ssor = Ssor::new(&a, 1.0);
        let opts = IterativeOptions::default().tol(1e-8);
        let result = pcg(&a, &b, &ssor, None, &opts).unwrap();

        assert!(
            result.converged,
            "SSOR-PCG did not converge after {} iters",
            result.iterations
        );
        for i in 0..n {
            assert!(
                (result.x[i] - x_exact[i]).abs() < 1e-4,
                "SSOR-PCG error at {}: {} vs {}",
                i,
                result.x[i],
                x_exact[i]
            );
        }
    }

    #[test]
    fn test_iterative_vs_direct() {
        // Compare iterative and direct solutions on the same sparse system
        let n = 100;
        let a = laplacian_1d(n);
        let b: Vec<f64> = (0..n).map(|i| (i as f64 * 0.1).sin()).collect();

        // Direct solve
        let lu = crate::SparseLU::new(&a).unwrap();
        let x_direct = lu.solve(&b).unwrap();

        // CG
        let opts = IterativeOptions::default().tol(1e-12);
        let x_cg = cg(&a, &b, None, &opts).unwrap();
        assert!(x_cg.converged);

        for i in 0..n {
            assert!(
                (x_cg.x[i] - x_direct[i]).abs() < 1e-8,
                "CG vs direct mismatch at {}: {} vs {}",
                i,
                x_cg.x[i],
                x_direct[i]
            );
        }
    }

    #[test]
    fn test_cg_f32() {
        let n = 10;
        let mut triplets: Vec<(usize, usize, f32)> = Vec::new();
        for i in 0..n {
            triplets.push((i, i, 2.0f32));
            if i > 0 {
                triplets.push((i, i - 1, -1.0f32));
            }
            if i < n - 1 {
                triplets.push((i, i + 1, -1.0f32));
            }
        }
        let a = SparseMatrix::from_triplets(n, n, &triplets).unwrap();
        let b: Vec<f32> = (0..n).map(|i| (i + 1) as f32).collect();

        let opts = IterativeOptions::default().tol(1e-5f32);
        let result = cg(&a, &b, None, &opts).unwrap();
        assert!(result.converged);
    }

    #[test]
    fn test_gmres_restart() {
        // Use small restart to force restart behavior
        let n = 30;
        let a = convection_diffusion(n, 0.1);
        let x_exact: Vec<f64> = (0..n).map(|i| (i as f64).sin()).collect();
        let b = a.mul_vec(&x_exact).unwrap();

        let opts = IterativeOptions::default().tol(1e-10).restart(5); // Small restart
        let result = gmres(&a, &b, None, &opts).unwrap();

        assert!(result.converged, "Restarted GMRES did not converge");
        for i in 0..n {
            assert!(
                (result.x[i] - x_exact[i]).abs() < 1e-5,
                "Restarted GMRES error at {}",
                i
            );
        }
    }
}

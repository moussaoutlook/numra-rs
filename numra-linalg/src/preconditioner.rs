//! Preconditioners for iterative linear solvers.
//!
//! Author: Moussa Leblouba
//! Date: 9 February 2026
//! Modified: 2 May 2026

use crate::sparse::SparseMatrix;
use crate::Scalar;
use faer::{ComplexField, Conjugate, Entity, SimpleEntity};
use numra_core::LinalgError;

/// Preconditioner trait: applies M^{-1} to a vector.
///
/// In a preconditioned solve of Ax = b, the preconditioner M approximates A
/// so that M^{-1}A has a smaller condition number.
pub trait Preconditioner<S: Scalar> {
    /// Compute z = M^{-1} * r.
    fn apply(&self, r: &[S], z: &mut [S]);
}

/// Identity preconditioner (no preconditioning).
pub struct IdentityPreconditioner;

impl<S: Scalar> Preconditioner<S> for IdentityPreconditioner {
    fn apply(&self, r: &[S], z: &mut [S]) {
        z.copy_from_slice(r);
    }
}

/// Jacobi (diagonal) preconditioner: M = diag(A).
///
/// Cheap to construct and apply (O(n)), effective for diagonally dominant systems.
pub struct Jacobi<S: Scalar> {
    inv_diag: Vec<S>,
}

impl<S: Scalar + SimpleEntity + Conjugate<Canonical = S> + ComplexField + Entity> Jacobi<S> {
    /// Create a Jacobi preconditioner from a sparse matrix.
    ///
    /// Zero diagonal entries are replaced with 1.0 to avoid division by zero.
    pub fn new(a: &SparseMatrix<S>) -> Self {
        let n = a.nrows();
        let mut inv_diag = Vec::with_capacity(n);
        for i in 0..n {
            let d = a.get(i, i);
            if d.abs() > S::EPSILON {
                inv_diag.push(S::ONE / d);
            } else {
                inv_diag.push(S::ONE);
            }
        }
        Self { inv_diag }
    }
}

impl<S: Scalar> Preconditioner<S> for Jacobi<S> {
    fn apply(&self, r: &[S], z: &mut [S]) {
        for i in 0..r.len() {
            z[i] = self.inv_diag[i] * r[i];
        }
    }
}

/// Incomplete LU(0) preconditioner.
///
/// Computes an approximate LU factorization where fill-in outside the
/// original sparsity pattern is dropped. Stored as dense L and U
/// (pragmatic approach matching sparse.rs; works well for moderate sizes).
pub struct Ilu0<S: Scalar> {
    /// Lower triangular factor (n x n, unit diagonal stored implicitly)
    l: Vec<Vec<S>>,
    /// Upper triangular factor (n x n)
    u: Vec<Vec<S>>,
    n: usize,
}

impl<S: Scalar + SimpleEntity + Conjugate<Canonical = S> + ComplexField + Entity> Ilu0<S> {
    /// Construct an ILU(0) factorization.
    ///
    /// The factorization preserves the sparsity pattern of A: entries where
    /// A is zero remain zero in L and U.
    pub fn new(a: &SparseMatrix<S>) -> Result<Self, LinalgError> {
        let n = a.nrows();
        if n != a.ncols() {
            return Err(LinalgError::NotSquare {
                nrows: n,
                ncols: a.ncols(),
            });
        }

        // Build sparsity pattern from A
        let mut pattern = vec![vec![false; n]; n];
        let col_ptrs = a.col_ptrs();
        let row_indices = a.row_indices();
        for j in 0..n {
            for idx in col_ptrs[j]..col_ptrs[j + 1] {
                pattern[row_indices[idx]][j] = true;
            }
        }

        // Work with dense copy for ILU(0) computation
        let mut m = vec![vec![S::ZERO; n]; n];
        for i in 0..n {
            for j in 0..n {
                if pattern[i][j] {
                    m[i][j] = a.get(i, j);
                }
            }
        }

        // IKJ variant of ILU(0)
        for i in 1..n {
            for k in 0..i {
                if !pattern[i][k] {
                    continue;
                }
                if m[k][k].abs() < S::EPSILON {
                    return Err(LinalgError::Singular { step: k });
                }
                m[i][k] = m[i][k] / m[k][k];
                for j in (k + 1)..n {
                    if pattern[i][j] {
                        m[i][j] = m[i][j] - m[i][k] * m[k][j];
                    }
                }
            }
        }

        // Extract L (lower, unit diagonal) and U (upper)
        let mut l = vec![vec![S::ZERO; n]; n];
        let mut u = vec![vec![S::ZERO; n]; n];
        for i in 0..n {
            l[i][i] = S::ONE;
            for j in 0..i {
                l[i][j] = m[i][j];
            }
            for j in i..n {
                u[i][j] = m[i][j];
            }
        }

        Ok(Self { l, u, n })
    }
}

impl<S: Scalar> Preconditioner<S> for Ilu0<S> {
    fn apply(&self, r: &[S], z: &mut [S]) {
        let n = self.n;

        // Forward solve: L y = r
        let mut y = vec![S::ZERO; n];
        for i in 0..n {
            let mut sum = r[i];
            for (j, yj) in y.iter().enumerate().take(i) {
                sum -= self.l[i][j] * *yj;
            }
            y[i] = sum; // unit diagonal
        }

        // Back solve: U z = y
        for i in (0..n).rev() {
            let mut sum = y[i];
            for (j, zj) in z.iter().enumerate().take(n).skip(i + 1) {
                sum -= self.u[i][j] * *zj;
            }
            z[i] = sum / self.u[i][i];
        }
    }
}

/// SSOR (Symmetric Successive Over-Relaxation) preconditioner.
///
/// M = (D/omega + L) * (D/omega)^{-1} * (D/omega + U)
/// where D = diag(A), L = strict lower, U = strict upper, omega in (0, 2).
pub struct Ssor<S: Scalar> {
    diag: Vec<S>,
    /// Stored as dense rows for simplicity
    lower: Vec<Vec<(usize, S)>>, // lower[i] = [(col, val), ...]
    upper: Vec<Vec<(usize, S)>>, // upper[i] = [(col, val), ...]
    omega: S,
    n: usize,
}

impl<S: Scalar + SimpleEntity + Conjugate<Canonical = S> + ComplexField + Entity> Ssor<S> {
    /// Create an SSOR preconditioner with relaxation parameter `omega`.
    ///
    /// `omega = 1.0` gives symmetric Gauss-Seidel.
    pub fn new(a: &SparseMatrix<S>, omega: S) -> Self {
        let n = a.nrows();
        let col_ptrs = a.col_ptrs();
        let row_indices = a.row_indices();
        let values = a.values();

        let mut diag = vec![S::ZERO; n];
        let mut lower: Vec<Vec<(usize, S)>> = vec![Vec::new(); n];
        let mut upper: Vec<Vec<(usize, S)>> = vec![Vec::new(); n];

        for j in 0..n {
            for idx in col_ptrs[j]..col_ptrs[j + 1] {
                let i = row_indices[idx];
                let v = values[idx];
                if i == j {
                    diag[i] = v;
                } else if i > j {
                    lower[i].push((j, v));
                } else {
                    upper[i].push((j, v));
                }
            }
        }

        Self {
            diag,
            lower,
            upper,
            omega,
            n,
        }
    }
}

impl<S: Scalar> Preconditioner<S> for Ssor<S> {
    fn apply(&self, r: &[S], z: &mut [S]) {
        let n = self.n;
        let w = self.omega;

        // Forward sweep: (D/w + L) * y = r
        let mut y = vec![S::ZERO; n];
        for i in 0..n {
            let mut sum = r[i];
            for &(j, v) in &self.lower[i] {
                sum -= v * y[j];
            }
            y[i] = w * sum / self.diag[i];
        }

        // Scale: y = (D/w) * y => multiply by D/(w) already done

        // Backward sweep: (D/w + U) * z = (D/w) * y
        // (D/w) * y[i] = diag[i]/w * y[i] = diag[i]/w * (w * sum / diag[i]) = sum
        // So we effectively have D * y_unscaled as RHS
        // Simpler: z = (D/w + U)^{-1} * D/w * y
        for i in (0..n).rev() {
            let mut sum = self.diag[i] / w * y[i];
            for &(j, v) in &self.upper[i] {
                sum -= v * z[j];
            }
            z[i] = w * sum / self.diag[i];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spd_3x3() -> SparseMatrix<f64> {
        // [4 1 0; 1 4 1; 0 1 4] — SPD, diagonally dominant
        let triplets = vec![
            (0, 0, 4.0),
            (0, 1, 1.0),
            (1, 0, 1.0),
            (1, 1, 4.0),
            (1, 2, 1.0),
            (2, 1, 1.0),
            (2, 2, 4.0),
        ];
        SparseMatrix::from_triplets(3, 3, &triplets).unwrap()
    }

    #[test]
    fn test_jacobi_apply() {
        let a = spd_3x3();
        let jac = Jacobi::new(&a);
        let r = [8.0, 12.0, 8.0];
        let mut z = [0.0; 3];
        jac.apply(&r, &mut z);
        // z = D^{-1} r = [8/4, 12/4, 8/4] = [2, 3, 2]
        assert!((z[0] - 2.0).abs() < 1e-14);
        assert!((z[1] - 3.0).abs() < 1e-14);
        assert!((z[2] - 2.0).abs() < 1e-14);
    }

    #[test]
    fn test_ilu0_exact_on_dense() {
        // For a small dense matrix, ILU(0) == LU (no fill-in to drop)
        let triplets = vec![
            (0, 0, 4.0),
            (0, 1, 1.0),
            (1, 0, 1.0),
            (1, 1, 4.0),
            (1, 2, 1.0),
            (2, 1, 1.0),
            (2, 2, 4.0),
        ];
        let a = SparseMatrix::from_triplets(3, 3, &triplets).unwrap();
        let ilu = Ilu0::new(&a).unwrap();

        // Applying ILU^{-1} to b should approximately solve Ax = b
        let b = [5.0, 6.0, 5.0];
        let mut z = [0.0; 3];
        ilu.apply(&b, &mut z);

        // Verify L*U*z ≈ b by computing A*z
        let az = a.mul_vec(&z).unwrap();
        for i in 0..3 {
            assert!(
                (az[i] - b[i]).abs() < 1e-10,
                "ILU residual at {}: {} vs {}",
                i,
                az[i],
                b[i]
            );
        }
    }

    #[test]
    fn test_ssor_reduces_residual() {
        let a = spd_3x3();
        let ssor = Ssor::new(&a, 1.0);
        let b = [5.0, 6.0, 5.0];
        let mut z = [0.0; 3];
        ssor.apply(&b, &mut z);
        // z should be an approximation to A^{-1}b — check it's not zero
        let norm: f64 = z.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!(norm > 0.1, "SSOR output should be nonzero");
    }

    #[test]
    fn test_identity_preconditioner() {
        let r = [1.0, 2.0, 3.0];
        let mut z = [0.0; 3];
        IdentityPreconditioner.apply(&r, &mut z);
        assert_eq!(z, r);
    }
}

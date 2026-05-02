//! LU factorization and solver.
//!
//! Author: Moussa Leblouba
//! Date: 8 February 2026
//! Modified: 2 May 2026

use crate::matrix::{DenseMatrix, Matrix};
use crate::Scalar;
use faer::linalg::solvers::PartialPivLu;
use faer::prelude::*;
use faer::{ComplexField, Conjugate, Entity, Mat, SimpleEntity};
use numra_core::LinalgError;

/// LU factorization of a matrix.
///
/// Caches the decomposition computed in `new()` and reuses it in every `solve()` call,
/// avoiding redundant recomputation.
pub struct LUFactorization<S: Scalar + Entity> {
    /// Cached LU decomposition from faer
    lu: PartialPivLu<S>,
    n: usize,
}

impl<S: Scalar + SimpleEntity + Conjugate<Canonical = S> + ComplexField> LUFactorization<S> {
    /// Compute LU factorization of a matrix.
    ///
    /// The decomposition is cached and reused for subsequent `solve()` calls.
    pub fn new(matrix: &DenseMatrix<S>) -> Result<Self, LinalgError> {
        if !matrix.is_square() {
            return Err(LinalgError::NotSquare {
                nrows: matrix.rows(),
                ncols: matrix.cols(),
            });
        }

        let n = matrix.rows();
        // Compute and cache the LU factorization
        let lu = matrix.as_faer().partial_piv_lu();

        Ok(Self { lu, n })
    }

    /// Dimension of the factorized matrix.
    pub fn dim(&self) -> usize {
        self.n
    }

    /// Solve Ax = b using the cached LU factorization.
    pub fn solve(&self, b: &[S]) -> Result<Vec<S>, LinalgError> {
        if b.len() != self.n {
            return Err(LinalgError::DimensionMismatch {
                expected: (self.n, 1),
                actual: (b.len(), 1),
            });
        }

        // Create column vector from b
        let mut b_mat = Mat::zeros(self.n, 1);
        for (i, &val) in b.iter().enumerate() {
            b_mat.write(i, 0, val);
        }

        // Solve using the cached LU factorization (no recomputation)
        let x_mat = self.lu.solve(&b_mat);

        // Extract result
        let mut x = Vec::with_capacity(self.n);
        for i in 0..self.n {
            x.push(x_mat.read(i, 0));
        }

        Ok(x)
    }

    /// Solve Ax = b in-place (b is overwritten with x).
    pub fn solve_inplace(&self, b: &mut [S]) -> Result<(), LinalgError> {
        let x = self.solve(b)?;
        b.copy_from_slice(&x);
        Ok(())
    }

    /// Solve multiple right-hand sides: AX = B.
    pub fn solve_multi(&self, b: &[S], nrhs: usize) -> Result<Vec<S>, LinalgError> {
        if b.len() != self.n * nrhs {
            return Err(LinalgError::DimensionMismatch {
                expected: (self.n, nrhs),
                actual: (b.len(), 1),
            });
        }

        // Create matrix from b (column-major)
        let mut b_mat = Mat::zeros(self.n, nrhs);
        for j in 0..nrhs {
            for i in 0..self.n {
                b_mat.write(i, j, b[j * self.n + i]);
            }
        }

        // Solve using the cached LU factorization (no recomputation)
        let x_mat = self.lu.solve(&b_mat);

        // Extract result (column-major)
        let mut x = vec![S::ZERO; self.n * nrhs];
        for j in 0..nrhs {
            for i in 0..self.n {
                x[j * self.n + i] = x_mat.read(i, j);
            }
        }

        Ok(x)
    }
}

/// Trait for types that can solve linear systems.
pub trait LUSolver<S: Scalar> {
    /// Solve Ax = b.
    fn lu_solve(&self, b: &[S]) -> Result<Vec<S>, LinalgError>;

    /// Compute and store LU factorization.
    fn lu_factor(&self) -> Result<LUFactorization<S>, LinalgError>
    where
        S: Entity + SimpleEntity + Conjugate<Canonical = S> + ComplexField;
}

impl<S: Scalar + SimpleEntity + Conjugate<Canonical = S> + ComplexField> LUSolver<S>
    for DenseMatrix<S>
{
    fn lu_solve(&self, b: &[S]) -> Result<Vec<S>, LinalgError> {
        self.solve(b)
    }

    fn lu_factor(&self) -> Result<LUFactorization<S>, LinalgError> {
        LUFactorization::new(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Matrix;

    #[test]
    fn test_lu_factorization() {
        let mut m: DenseMatrix<f64> = DenseMatrix::zeros(3, 3);
        m.set(0, 0, 2.0);
        m.set(1, 1, 3.0);
        m.set(2, 2, 4.0);

        let lu = LUFactorization::new(&m).unwrap();
        assert_eq!(lu.dim(), 3);
    }

    #[test]
    fn test_lu_solve() {
        let mut m: DenseMatrix<f64> = DenseMatrix::zeros(2, 2);
        m.set(0, 0, 1.0);
        m.set(0, 1, 2.0);
        m.set(1, 0, 3.0);
        m.set(1, 1, 4.0);

        let lu = LUFactorization::new(&m).unwrap();
        let b = vec![5.0, 11.0];
        let x = lu.solve(&b).unwrap();

        assert!((x[0] - 1.0).abs() < 1e-10);
        assert!((x[1] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_lu_solve_inplace() {
        let mut m: DenseMatrix<f64> = DenseMatrix::zeros(2, 2);
        m.set(0, 0, 2.0);
        m.set(0, 1, 0.0);
        m.set(1, 0, 0.0);
        m.set(1, 1, 3.0);

        let lu = LUFactorization::new(&m).unwrap();
        let mut b = vec![4.0, 9.0];
        lu.solve_inplace(&mut b).unwrap();

        assert!((b[0] - 2.0).abs() < 1e-10);
        assert!((b[1] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_lu_solve_multi() {
        let mut m: DenseMatrix<f64> = DenseMatrix::zeros(2, 2);
        m.set(0, 0, 2.0);
        m.set(1, 1, 3.0);

        let lu = LUFactorization::new(&m).unwrap();

        // Two right-hand sides (column-major): [2, 6] and [4, 9]
        let b = vec![2.0, 6.0, 4.0, 9.0];
        let x = lu.solve_multi(&b, 2).unwrap();

        // First solution: [1, 2]
        assert!((x[0] - 1.0).abs() < 1e-10);
        assert!((x[1] - 2.0).abs() < 1e-10);
        // Second solution: [2, 3]
        assert!((x[2] - 2.0).abs() < 1e-10);
        assert!((x[3] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_lu_solver_trait() {
        let mut m: DenseMatrix<f64> = DenseMatrix::zeros(2, 2);
        m.set(0, 0, 1.0);
        m.set(0, 1, 2.0);
        m.set(1, 0, 3.0);
        m.set(1, 1, 4.0);

        let b = vec![5.0, 11.0];
        let x = m.lu_solve(&b).unwrap();

        assert!((x[0] - 1.0).abs() < 1e-10);
        assert!((x[1] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_lu_repeated_solve() {
        // Verify that two different solve() calls on the same factorization
        // both produce correct results, confirming the cached decomposition works.
        let mut m: DenseMatrix<f64> = DenseMatrix::zeros(2, 2);
        m.set(0, 0, 1.0);
        m.set(0, 1, 2.0);
        m.set(1, 0, 3.0);
        m.set(1, 1, 4.0);

        let lu = LUFactorization::new(&m).unwrap();

        // First solve: A * [1, 2]^T = [5, 11]^T
        let b1 = vec![5.0, 11.0];
        let x1 = lu.solve(&b1).unwrap();
        assert!((x1[0] - 1.0).abs() < 1e-10);
        assert!((x1[1] - 2.0).abs() < 1e-10);

        // Second solve with different RHS: A * [3, 1]^T = [5, 13]^T
        let b2 = vec![5.0, 13.0];
        let x2 = lu.solve(&b2).unwrap();
        assert!((x2[0] - 3.0).abs() < 1e-10);
        assert!((x2[1] - 1.0).abs() < 1e-10);
    }
}

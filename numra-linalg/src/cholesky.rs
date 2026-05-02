//! Cholesky factorization for symmetric positive-definite matrices.
//!
//! Author: Moussa Leblouba
//! Date: 8 February 2026
//! Modified: 2 May 2026

use crate::matrix::DenseMatrix;
use crate::Scalar;
use faer::linalg::solvers::Cholesky;
use faer::prelude::*;
use faer::{ComplexField, Conjugate, Entity, Mat, SimpleEntity};
use numra_core::LinalgError;

/// Cholesky factorization A = L L^T for symmetric positive-definite matrices.
///
/// Caches the decomposition computed in `new()` and reuses it in every `solve()` call,
/// avoiding redundant recomputation.
pub struct CholeskyFactorization<S: Scalar + Entity> {
    /// Cached Cholesky decomposition from faer
    chol: Cholesky<S>,
    n: usize,
}

impl<S: Scalar + SimpleEntity + Conjugate<Canonical = S> + ComplexField> CholeskyFactorization<S> {
    /// Compute Cholesky factorization of a matrix.
    ///
    /// Returns an error if the matrix is not square or not positive definite.
    /// The decomposition is cached and reused for subsequent `solve()` calls.
    pub fn new(matrix: &DenseMatrix<S>) -> Result<Self, LinalgError> {
        let nrows = matrix.rows();
        let ncols = matrix.cols();

        if nrows != ncols {
            return Err(LinalgError::NotSquare { nrows, ncols });
        }

        let n = nrows;

        // Compute and cache the Cholesky factorization
        let chol = Cholesky::try_new(matrix.as_faer(), faer::Side::Lower)
            .map_err(|_| LinalgError::NotPositiveDefinite)?;

        Ok(Self { chol, n })
    }

    /// Dimension of the factorized matrix.
    pub fn dim(&self) -> usize {
        self.n
    }

    /// Solve Ax = b using the cached Cholesky factorization.
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

        // Solve using the cached Cholesky factorization (no recomputation)
        let x_mat = self.chol.solve(&b_mat);

        // Extract result
        let mut x = Vec::with_capacity(self.n);
        for i in 0..self.n {
            x.push(x_mat.read(i, 0));
        }

        Ok(x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Matrix;

    #[test]
    fn test_cholesky_solve() {
        // A = [4 2; 2 3], b = [8, 7], x = [1, 1]
        let mut a: DenseMatrix<f64> = DenseMatrix::zeros(2, 2);
        a.set(0, 0, 4.0);
        a.set(0, 1, 2.0);
        a.set(1, 0, 2.0);
        a.set(1, 1, 3.0);

        let chol = CholeskyFactorization::new(&a).unwrap();
        assert_eq!(chol.dim(), 2);

        let b = vec![6.0, 5.0];
        let x = chol.solve(&b).unwrap();

        assert!((x[0] - 1.0).abs() < 1e-10);
        assert!((x[1] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_cholesky_3x3() {
        // A = [25 15 -5; 15 18 0; -5 0 11], b = [35, 33, 6], x = [1, 1, 1]
        let mut a: DenseMatrix<f64> = DenseMatrix::zeros(3, 3);
        a.set(0, 0, 25.0);
        a.set(0, 1, 15.0);
        a.set(0, 2, -5.0);
        a.set(1, 0, 15.0);
        a.set(1, 1, 18.0);
        a.set(1, 2, 0.0);
        a.set(2, 0, -5.0);
        a.set(2, 1, 0.0);
        a.set(2, 2, 11.0);

        let chol = CholeskyFactorization::new(&a).unwrap();
        assert_eq!(chol.dim(), 3);

        let b = vec![35.0, 33.0, 6.0];
        let x = chol.solve(&b).unwrap();

        assert!((x[0] - 1.0).abs() < 1e-10);
        assert!((x[1] - 1.0).abs() < 1e-10);
        assert!((x[2] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_cholesky_not_positive_definite() {
        // A = [1 2; 2 1] is not positive definite
        let mut a: DenseMatrix<f64> = DenseMatrix::zeros(2, 2);
        a.set(0, 0, 1.0);
        a.set(0, 1, 2.0);
        a.set(1, 0, 2.0);
        a.set(1, 1, 1.0);

        assert!(CholeskyFactorization::new(&a).is_err());
    }

    #[test]
    fn test_cholesky_not_square() {
        // 2x3 matrix -> error
        let a: DenseMatrix<f64> = DenseMatrix::zeros(2, 3);
        assert!(CholeskyFactorization::new(&a).is_err());
    }

    #[test]
    fn test_cholesky_repeated_solve() {
        // A = [4 2; 2 3] (SPD)
        // Verify that two different solve() calls on the same factorization
        // both produce correct results, confirming the cached decomposition works.
        let mut a: DenseMatrix<f64> = DenseMatrix::zeros(2, 2);
        a.set(0, 0, 4.0);
        a.set(0, 1, 2.0);
        a.set(1, 0, 2.0);
        a.set(1, 1, 3.0);

        let chol = CholeskyFactorization::new(&a).unwrap();

        // First solve: A * [1, 1]^T = [6, 5]^T
        let b1 = vec![6.0, 5.0];
        let x1 = chol.solve(&b1).unwrap();
        assert!((x1[0] - 1.0).abs() < 1e-10);
        assert!((x1[1] - 1.0).abs() < 1e-10);

        // Second solve with different RHS: A * [2, -1]^T = [6, 1]^T
        let b2 = vec![6.0, 1.0];
        let x2 = chol.solve(&b2).unwrap();
        assert!((x2[0] - 2.0).abs() < 1e-10);
        assert!((x2[1] - (-1.0)).abs() < 1e-10);
    }
}

//! QR factorization and solver.
//!
//! Author: Moussa Leblouba
//! Date: 8 February 2026
//! Modified: 2 May 2026

use crate::matrix::DenseMatrix;
use crate::Scalar;
use faer::linalg::solvers::Qr;
use faer::prelude::*;
use faer::{ComplexField, Conjugate, Entity, Mat, SimpleEntity};
use numra_core::LinalgError;

/// QR factorization of a matrix.
///
/// Caches the decomposition computed in `new()` and reuses it in every `solve()` /
/// `solve_least_squares()` call, avoiding redundant recomputation.
pub struct QRFactorization<S: Scalar + Entity> {
    /// Cached QR decomposition from faer
    qr: Qr<S>,
    m: usize,
    n: usize,
}

impl<S: Scalar + SimpleEntity + Conjugate<Canonical = S> + ComplexField> QRFactorization<S> {
    /// Compute QR factorization of a matrix.
    ///
    /// Requires `m >= n` (the matrix must have at least as many rows as columns).
    /// The decomposition is cached and reused for subsequent `solve()` /
    /// `solve_least_squares()` calls.
    pub fn new(matrix: &DenseMatrix<S>) -> Result<Self, LinalgError> {
        let m = matrix.rows();
        let n = matrix.cols();

        if m < n {
            return Err(LinalgError::DimensionMismatch {
                expected: (n, n),
                actual: (m, n),
            });
        }

        // Compute and cache the QR factorization
        let qr = Qr::new(matrix.as_faer());

        Ok(Self { qr, m, n })
    }

    /// Number of rows of the original matrix.
    pub fn nrows(&self) -> usize {
        self.m
    }

    /// Number of columns of the original matrix.
    pub fn ncols(&self) -> usize {
        self.n
    }

    /// Solve Ax = b using the cached QR factorization (square systems only).
    pub fn solve(&self, b: &[S]) -> Result<Vec<S>, LinalgError> {
        if self.m != self.n {
            return Err(LinalgError::NotSquare {
                nrows: self.m,
                ncols: self.n,
            });
        }
        if b.len() != self.m {
            return Err(LinalgError::DimensionMismatch {
                expected: (self.m, 1),
                actual: (b.len(), 1),
            });
        }

        // Create column vector from b
        let mut b_mat = Mat::zeros(self.m, 1);
        for (i, &val) in b.iter().enumerate() {
            b_mat.write(i, 0, val);
        }

        // Solve using the cached QR factorization (no recomputation)
        let x_mat = self.qr.solve(&b_mat);

        // Extract result
        let mut x = Vec::with_capacity(self.n);
        for i in 0..self.n {
            x.push(x_mat.read(i, 0));
        }

        Ok(x)
    }

    /// Solve the least-squares problem min ||Ax - b||_2.
    ///
    /// Works for both square and overdetermined systems (m >= n).
    /// Uses the cached QR factorization.
    pub fn solve_least_squares(&self, b: &[S]) -> Result<Vec<S>, LinalgError> {
        if b.len() != self.m {
            return Err(LinalgError::DimensionMismatch {
                expected: (self.m, 1),
                actual: (b.len(), 1),
            });
        }

        // Create column vector from b
        let mut b_mat = Mat::zeros(self.m, 1);
        for (i, &val) in b.iter().enumerate() {
            b_mat.write(i, 0, val);
        }

        // Solve least-squares using the cached QR factorization (no recomputation)
        let x_mat = self.qr.solve_lstsq(&b_mat);

        // Extract result (x_mat has n rows after lstsq resize)
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
    fn test_qr_square() {
        // Solve [1 2; 3 4] * x = [5; 11]
        // x = [1; 2]
        let mut m: DenseMatrix<f64> = DenseMatrix::zeros(2, 2);
        m.set(0, 0, 1.0);
        m.set(0, 1, 2.0);
        m.set(1, 0, 3.0);
        m.set(1, 1, 4.0);

        let qr = QRFactorization::new(&m).unwrap();
        assert_eq!(qr.nrows(), 2);
        assert_eq!(qr.ncols(), 2);

        let b = vec![5.0, 11.0];
        let x = qr.solve(&b).unwrap();

        assert!((x[0] - 1.0).abs() < 1e-10);
        assert!((x[1] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_qr_overdetermined() {
        // Least-squares: A = [1 1; 1 2; 1 3], b = [1; 2; 2]
        // Normal equations: A^T A x = A^T b
        // A^T A = [3 6; 6 14], A^T b = [5; 11]
        // x = [2/3; 1/2]
        let mut m: DenseMatrix<f64> = DenseMatrix::zeros(3, 2);
        m.set(0, 0, 1.0);
        m.set(0, 1, 1.0);
        m.set(1, 0, 1.0);
        m.set(1, 1, 2.0);
        m.set(2, 0, 1.0);
        m.set(2, 1, 3.0);

        let qr = QRFactorization::new(&m).unwrap();
        assert_eq!(qr.nrows(), 3);
        assert_eq!(qr.ncols(), 2);

        let b = vec![1.0, 2.0, 2.0];
        let x = qr.solve_least_squares(&b).unwrap();

        assert!((x[0] - 2.0 / 3.0).abs() < 1e-10);
        assert!((x[1] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_qr_dimension_mismatch() {
        let mut m: DenseMatrix<f64> = DenseMatrix::zeros(2, 2);
        m.set(0, 0, 1.0);
        m.set(0, 1, 2.0);
        m.set(1, 0, 3.0);
        m.set(1, 1, 4.0);

        let qr = QRFactorization::new(&m).unwrap();

        // Wrong b size for solve
        let b = vec![1.0, 2.0, 3.0];
        let result = qr.solve(&b);
        assert!(result.is_err());

        // Wrong b size for solve_least_squares
        let result = qr.solve_least_squares(&b);
        assert!(result.is_err());
    }
}

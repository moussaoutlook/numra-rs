//! Matrix trait and implementations.
//!
//! Author: Moussa Leblouba
//! Date: 8 February 2026
//! Modified: 2 May 2026

use crate::Scalar;
use faer::prelude::*;
use faer::{ComplexField, Conjugate, Entity, Mat, MatMut, MatRef, SimpleEntity};
use numra_core::LinalgError;

/// Trait for matrix types.
///
/// Provides a backend-agnostic interface for matrix operations needed by ODE solvers.
pub trait Matrix<S: Scalar>: Clone + Sized {
    /// Create a zero matrix with given dimensions.
    fn zeros(rows: usize, cols: usize) -> Self;

    /// Create an identity matrix.
    fn identity(n: usize) -> Self;

    /// Number of rows.
    fn nrows(&self) -> usize;

    /// Number of columns.
    fn ncols(&self) -> usize;

    /// Get element at (i, j).
    fn get(&self, i: usize, j: usize) -> S;

    /// Set element at (i, j).
    fn set(&mut self, i: usize, j: usize, value: S);

    /// Fill matrix with zeros.
    fn fill_zero(&mut self);

    /// Scale all elements by a constant.
    fn scale(&mut self, alpha: S);

    /// Compute y = self * x (matrix-vector product).
    fn mul_vec(&self, x: &[S], y: &mut [S]);

    /// Add another matrix: self += alpha * other.
    fn add_scaled(&mut self, alpha: S, other: &Self);

    /// Solve the linear system Ax = b, returning x.
    fn solve(&self, b: &[S]) -> Result<Vec<S>, LinalgError>;

    /// Check if matrix is square.
    fn is_square(&self) -> bool {
        self.nrows() == self.ncols()
    }
}

/// Dense matrix backed by faer.
#[derive(Clone, Debug)]
pub struct DenseMatrix<S: Scalar + Entity> {
    data: Mat<S>,
}

impl<S: Scalar + Entity> DenseMatrix<S> {
    /// Create from a faer Mat.
    pub fn from_faer(mat: Mat<S>) -> Self {
        Self { data: mat }
    }

    /// Get reference to underlying faer matrix.
    pub fn as_faer(&self) -> MatRef<'_, S> {
        self.data.as_ref()
    }

    /// Get mutable reference to underlying faer matrix.
    pub fn as_faer_mut(&mut self) -> MatMut<'_, S> {
        self.data.as_mut()
    }

    /// Create from row-major data.
    pub fn from_row_major(rows: usize, cols: usize, data: &[S]) -> Self {
        assert_eq!(data.len(), rows * cols);
        let mut mat = Mat::zeros(rows, cols);
        for i in 0..rows {
            for j in 0..cols {
                mat.write(i, j, data[i * cols + j]);
            }
        }
        Self { data: mat }
    }

    /// Create from column-major data.
    pub fn from_col_major(rows: usize, cols: usize, data: &[S]) -> Self {
        assert_eq!(data.len(), rows * cols);
        let mut mat = Mat::zeros(rows, cols);
        for j in 0..cols {
            for i in 0..rows {
                mat.write(i, j, data[j * rows + i]);
            }
        }
        Self { data: mat }
    }

    /// Convert to row-major vector.
    pub fn to_row_major(&self) -> Vec<S> {
        let (rows, cols) = (self.data.nrows(), self.data.ncols());
        let mut data = Vec::with_capacity(rows * cols);
        for i in 0..rows {
            for j in 0..cols {
                data.push(self.data.read(i, j));
            }
        }
        data
    }

    /// Frobenius norm.
    pub fn norm_frobenius(&self) -> S {
        let mut sum = S::ZERO;
        for i in 0..self.data.nrows() {
            for j in 0..self.data.ncols() {
                let v = self.data.read(i, j);
                sum += v * v;
            }
        }
        sum.sqrt()
    }

    /// Infinity norm (max row sum).
    pub fn norm_inf(&self) -> S {
        let mut max_sum = S::ZERO;
        for i in 0..self.data.nrows() {
            let mut row_sum = S::ZERO;
            for j in 0..self.data.ncols() {
                row_sum += self.data.read(i, j).abs();
            }
            max_sum = max_sum.max(row_sum);
        }
        max_sum
    }

    /// Number of rows (direct access without trait).
    pub fn rows(&self) -> usize {
        self.data.nrows()
    }

    /// Number of columns (direct access without trait).
    pub fn cols(&self) -> usize {
        self.data.ncols()
    }

    /// Check if square (direct access without trait).
    pub fn is_square(&self) -> bool {
        self.data.nrows() == self.data.ncols()
    }
}

impl<S: Scalar + SimpleEntity + Conjugate<Canonical = S> + ComplexField> Matrix<S>
    for DenseMatrix<S>
{
    fn zeros(rows: usize, cols: usize) -> Self {
        Self {
            data: Mat::zeros(rows, cols),
        }
    }

    fn identity(n: usize) -> Self {
        let mut mat = Mat::zeros(n, n);
        for i in 0..n {
            mat.write(i, i, S::ONE);
        }
        Self { data: mat }
    }

    fn nrows(&self) -> usize {
        self.data.nrows()
    }

    fn ncols(&self) -> usize {
        self.data.ncols()
    }

    fn get(&self, i: usize, j: usize) -> S {
        self.data.read(i, j)
    }

    fn set(&mut self, i: usize, j: usize, value: S) {
        self.data.write(i, j, value);
    }

    fn fill_zero(&mut self) {
        for i in 0..self.nrows() {
            for j in 0..self.ncols() {
                self.data.write(i, j, S::ZERO);
            }
        }
    }

    fn scale(&mut self, alpha: S) {
        for i in 0..self.nrows() {
            for j in 0..self.ncols() {
                let v = self.data.read(i, j);
                self.data.write(i, j, alpha * v);
            }
        }
    }

    fn mul_vec(&self, x: &[S], y: &mut [S]) {
        assert_eq!(x.len(), self.ncols());
        assert_eq!(y.len(), self.nrows());

        for (i, y_i) in y.iter_mut().enumerate().take(self.nrows()) {
            let mut sum = S::ZERO;
            for (j, &x_j) in x.iter().enumerate().take(self.ncols()) {
                sum += self.data.read(i, j) * x_j;
            }
            *y_i = sum;
        }
    }

    fn add_scaled(&mut self, alpha: S, other: &Self) {
        assert_eq!(self.nrows(), other.nrows());
        assert_eq!(self.ncols(), other.ncols());

        for i in 0..self.nrows() {
            for j in 0..self.ncols() {
                let v = self.data.read(i, j) + alpha * other.data.read(i, j);
                self.data.write(i, j, v);
            }
        }
    }

    fn solve(&self, b: &[S]) -> Result<Vec<S>, LinalgError> {
        if !self.is_square() {
            return Err(LinalgError::NotSquare {
                nrows: self.nrows(),
                ncols: self.ncols(),
            });
        }
        if b.len() != self.nrows() {
            return Err(LinalgError::DimensionMismatch {
                expected: (self.nrows(), 1),
                actual: (b.len(), 1),
            });
        }

        // Use faer's LU solver
        let lu = self.data.as_ref().partial_piv_lu();

        // Create column vector from b
        let mut b_mat = Mat::zeros(b.len(), 1);
        for (i, &val) in b.iter().enumerate() {
            b_mat.write(i, 0, val);
        }

        // Solve
        let x_mat = lu.solve(&b_mat);

        // Extract result
        let mut x = Vec::with_capacity(b.len());
        for i in 0..b.len() {
            x.push(x_mat.read(i, 0));
        }

        Ok(x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zeros() {
        let m: DenseMatrix<f64> = DenseMatrix::zeros(3, 4);
        assert_eq!(m.nrows(), 3);
        assert_eq!(m.ncols(), 4);
        for i in 0..3 {
            for j in 0..4 {
                assert!((m.get(i, j) - 0.0).abs() < 1e-15);
            }
        }
    }

    #[test]
    fn test_identity() {
        let m: DenseMatrix<f64> = DenseMatrix::identity(3);
        assert_eq!(m.nrows(), 3);
        assert_eq!(m.ncols(), 3);
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!((m.get(i, j) - expected).abs() < 1e-15);
            }
        }
    }

    #[test]
    fn test_set_get() {
        let mut m: DenseMatrix<f64> = DenseMatrix::zeros(2, 2);
        m.set(0, 0, 1.0);
        m.set(0, 1, 2.0);
        m.set(1, 0, 3.0);
        m.set(1, 1, 4.0);

        assert!((m.get(0, 0) - 1.0).abs() < 1e-15);
        assert!((m.get(0, 1) - 2.0).abs() < 1e-15);
        assert!((m.get(1, 0) - 3.0).abs() < 1e-15);
        assert!((m.get(1, 1) - 4.0).abs() < 1e-15);
    }

    #[test]
    fn test_mul_vec() {
        // [1 2] * [1]   [5]
        // [3 4]   [2] = [11]
        let mut m: DenseMatrix<f64> = DenseMatrix::zeros(2, 2);
        m.set(0, 0, 1.0);
        m.set(0, 1, 2.0);
        m.set(1, 0, 3.0);
        m.set(1, 1, 4.0);

        let x = [1.0, 2.0];
        let mut y = [0.0, 0.0];
        m.mul_vec(&x, &mut y);

        assert!((y[0] - 5.0).abs() < 1e-10);
        assert!((y[1] - 11.0).abs() < 1e-10);
    }

    #[test]
    fn test_scale() {
        let mut m: DenseMatrix<f64> = DenseMatrix::identity(2);
        m.scale(3.0);
        assert!((m.get(0, 0) - 3.0).abs() < 1e-15);
        assert!((m.get(1, 1) - 3.0).abs() < 1e-15);
    }

    #[test]
    fn test_add_scaled() {
        let mut a: DenseMatrix<f64> = DenseMatrix::identity(2);
        let b: DenseMatrix<f64> = DenseMatrix::identity(2);
        a.add_scaled(2.0, &b);

        // a should now be 3*I
        assert!((a.get(0, 0) - 3.0).abs() < 1e-15);
        assert!((a.get(1, 1) - 3.0).abs() < 1e-15);
    }

    #[test]
    fn test_solve_diagonal() {
        // Solve diag(2, 3, 4) * x = [1, 2, 3]
        // x = [0.5, 2/3, 0.75]
        let mut m: DenseMatrix<f64> = DenseMatrix::zeros(3, 3);
        m.set(0, 0, 2.0);
        m.set(1, 1, 3.0);
        m.set(2, 2, 4.0);

        let b = vec![1.0, 2.0, 3.0];
        let x = m.solve(&b).unwrap();

        assert!((x[0] - 0.5).abs() < 1e-10);
        assert!((x[1] - 2.0 / 3.0).abs() < 1e-10);
        assert!((x[2] - 0.75).abs() < 1e-10);
    }

    #[test]
    fn test_solve_general() {
        // Solve [1 2; 3 4] * x = [5; 11]
        // x = [1; 2]
        let mut m: DenseMatrix<f64> = DenseMatrix::zeros(2, 2);
        m.set(0, 0, 1.0);
        m.set(0, 1, 2.0);
        m.set(1, 0, 3.0);
        m.set(1, 1, 4.0);

        let b = vec![5.0, 11.0];
        let x = m.solve(&b).unwrap();

        assert!((x[0] - 1.0).abs() < 1e-10);
        assert!((x[1] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_from_row_major() {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let m: DenseMatrix<f64> = DenseMatrix::from_row_major(2, 3, &data);

        assert_eq!(m.nrows(), 2);
        assert_eq!(m.ncols(), 3);
        assert!((m.get(0, 0) - 1.0).abs() < 1e-15);
        assert!((m.get(0, 2) - 3.0).abs() < 1e-15);
        assert!((m.get(1, 0) - 4.0).abs() < 1e-15);
        assert!((m.get(1, 2) - 6.0).abs() < 1e-15);
    }

    #[test]
    fn test_norm_frobenius() {
        let mut m: DenseMatrix<f64> = DenseMatrix::zeros(2, 2);
        m.set(0, 0, 1.0);
        m.set(0, 1, 2.0);
        m.set(1, 0, 3.0);
        m.set(1, 1, 4.0);

        // sqrt(1 + 4 + 9 + 16) = sqrt(30)
        let norm = m.norm_frobenius();
        assert!((norm - 30.0_f64.sqrt()).abs() < 1e-10);
    }

    // ============================================================================
    // Edge Case Tests
    // ============================================================================

    #[test]
    fn test_1x1_matrix() {
        let mut m: DenseMatrix<f64> = DenseMatrix::zeros(1, 1);
        m.set(0, 0, 5.0);
        assert!(m.is_square());
        assert!((m.get(0, 0) - 5.0).abs() < 1e-15);

        let b = vec![10.0];
        let x = m.solve(&b).unwrap();
        assert!((x[0] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_identity_1x1() {
        let m: DenseMatrix<f64> = DenseMatrix::identity(1);
        assert!((m.get(0, 0) - 1.0).abs() < 1e-15);
    }

    #[test]
    fn test_rectangular_not_square() {
        let m: DenseMatrix<f64> = DenseMatrix::zeros(2, 3);
        assert!(!m.is_square());
    }

    #[test]
    fn test_solve_non_square_error() {
        let m: DenseMatrix<f64> = DenseMatrix::zeros(2, 3);
        let b = vec![1.0, 2.0];
        let result = m.solve(&b);
        assert!(result.is_err());
    }

    #[test]
    fn test_solve_dimension_mismatch() {
        let m: DenseMatrix<f64> = DenseMatrix::zeros(2, 2);
        let b = vec![1.0, 2.0, 3.0]; // Wrong size
        let result = m.solve(&b);
        assert!(result.is_err());
    }

    #[test]
    fn test_fill_zero() {
        let mut m: DenseMatrix<f64> = DenseMatrix::identity(3);
        m.fill_zero();
        for i in 0..3 {
            for j in 0..3 {
                assert!(m.get(i, j).abs() < 1e-15);
            }
        }
    }

    #[test]
    fn test_scale_by_zero() {
        let mut m: DenseMatrix<f64> = DenseMatrix::identity(2);
        m.scale(0.0);
        for i in 0..2 {
            for j in 0..2 {
                assert!(m.get(i, j).abs() < 1e-15);
            }
        }
    }

    #[test]
    fn test_scale_by_negative() {
        let mut m: DenseMatrix<f64> = DenseMatrix::identity(2);
        m.scale(-1.0);
        assert!((m.get(0, 0) + 1.0).abs() < 1e-15);
        assert!((m.get(1, 1) + 1.0).abs() < 1e-15);
    }

    #[test]
    fn test_mul_vec_with_zeros() {
        let m: DenseMatrix<f64> = DenseMatrix::zeros(2, 2);
        let x = [100.0, 200.0];
        let mut y = [999.0, 999.0];
        m.mul_vec(&x, &mut y);
        assert!(y[0].abs() < 1e-15);
        assert!(y[1].abs() < 1e-15);
    }

    #[test]
    fn test_norm_inf() {
        let mut m: DenseMatrix<f64> = DenseMatrix::zeros(2, 2);
        m.set(0, 0, -1.0);
        m.set(0, 1, 2.0);
        m.set(1, 0, 3.0);
        m.set(1, 1, -4.0);

        // Row 0: |−1| + |2| = 3
        // Row 1: |3| + |−4| = 7
        // Max = 7
        assert!((m.norm_inf() - 7.0).abs() < 1e-10);
    }

    #[test]
    fn test_zeros_large() {
        let m: DenseMatrix<f64> = DenseMatrix::zeros(100, 100);
        assert_eq!(m.nrows(), 100);
        assert_eq!(m.ncols(), 100);
    }

    #[test]
    fn test_from_col_major() {
        // Column-major: col0 = [1, 3], col1 = [2, 4]
        let data = [1.0, 3.0, 2.0, 4.0];
        let m: DenseMatrix<f64> = DenseMatrix::from_col_major(2, 2, &data);

        assert!((m.get(0, 0) - 1.0).abs() < 1e-15);
        assert!((m.get(1, 0) - 3.0).abs() < 1e-15);
        assert!((m.get(0, 1) - 2.0).abs() < 1e-15);
        assert!((m.get(1, 1) - 4.0).abs() < 1e-15);
    }

    #[test]
    fn test_to_row_major() {
        let mut m: DenseMatrix<f64> = DenseMatrix::zeros(2, 3);
        m.set(0, 0, 1.0);
        m.set(0, 1, 2.0);
        m.set(0, 2, 3.0);
        m.set(1, 0, 4.0);
        m.set(1, 1, 5.0);
        m.set(1, 2, 6.0);

        let data = m.to_row_major();
        assert_eq!(data, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_solve_ill_conditioned() {
        // Near-singular matrix (Hilbert-like)
        let mut m: DenseMatrix<f64> = DenseMatrix::zeros(2, 2);
        m.set(0, 0, 1.0);
        m.set(0, 1, 0.5);
        m.set(1, 0, 0.5);
        m.set(1, 1, 0.333333333333);

        let b = vec![1.5, 0.833333333333];
        let result = m.solve(&b);
        // Should still produce a result (may be less accurate)
        assert!(result.is_ok());
    }

    // ============================================================================
    // f32 Scalar Tests
    // ============================================================================

    #[test]
    fn test_f32_solve() {
        let mut m: DenseMatrix<f32> = DenseMatrix::zeros(2, 2);
        m.set(0, 0, 2.0);
        m.set(0, 1, 0.0);
        m.set(1, 0, 0.0);
        m.set(1, 1, 3.0);

        let b = vec![4.0f32, 9.0f32];
        let x = m.solve(&b).unwrap();

        assert!((x[0] - 2.0).abs() < 1e-5);
        assert!((x[1] - 3.0).abs() < 1e-5);
    }

    #[test]
    fn test_f32_identity() {
        let m: DenseMatrix<f32> = DenseMatrix::identity(3);
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0f32 } else { 0.0f32 };
                assert!((m.get(i, j) - expected).abs() < 1e-7);
            }
        }
    }

    #[test]
    fn test_f32_mul_vec() {
        let mut m: DenseMatrix<f32> = DenseMatrix::zeros(2, 2);
        m.set(0, 0, 1.0);
        m.set(0, 1, 2.0);
        m.set(1, 0, 3.0);
        m.set(1, 1, 4.0);

        let x = [1.0f32, 2.0f32];
        let mut y = [0.0f32, 0.0f32];
        m.mul_vec(&x, &mut y);

        assert!((y[0] - 5.0).abs() < 1e-5);
        assert!((y[1] - 11.0).abs() < 1e-5);
    }

    // ============================================================================
    // Error Type Tests
    // ============================================================================

    #[test]
    fn test_solve_non_square_returns_not_square_error() {
        let m: DenseMatrix<f64> = DenseMatrix::zeros(2, 3);
        let b = vec![1.0, 2.0];
        match m.solve(&b) {
            Err(LinalgError::NotSquare { nrows: 2, ncols: 3 }) => {}
            other => panic!("Expected NotSquare error, got {:?}", other),
        }
    }

    #[test]
    fn test_solve_dimension_mismatch_returns_typed_error() {
        let m: DenseMatrix<f64> = DenseMatrix::zeros(2, 2);
        let b = vec![1.0, 2.0, 3.0];
        match m.solve(&b) {
            Err(LinalgError::DimensionMismatch { .. }) => {}
            other => panic!("Expected DimensionMismatch error, got {:?}", other),
        }
    }
}

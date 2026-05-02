//! Eigenvalue decomposition.
//!
//! Author: Moussa Leblouba
//! Date: 9 February 2026
//! Modified: 2 May 2026

use crate::matrix::DenseMatrix;
use crate::Scalar;
use faer::linalg::solvers::SelfAdjointEigendecomposition;
use faer::{ComplexField, Conjugate, Entity, SimpleEntity};
use numra_core::LinalgError;

/// Symmetric/Hermitian eigendecomposition (real eigenvalues).
///
/// For a symmetric matrix A, computes A = U * diag(lambda) * U^T where U is orthogonal.
pub struct SymEigenDecomposition<S: Scalar + Entity> {
    evd: SelfAdjointEigendecomposition<S>,
    n: usize,
}

impl<S: Scalar + SimpleEntity + Conjugate<Canonical = S> + ComplexField> SymEigenDecomposition<S> {
    /// Compute symmetric eigendecomposition.
    ///
    /// Uses the lower triangle of the matrix.
    pub fn new(matrix: &DenseMatrix<S>) -> Result<Self, LinalgError> {
        let nrows = matrix.rows();
        let ncols = matrix.cols();
        if nrows != ncols {
            return Err(LinalgError::NotSquare { nrows, ncols });
        }
        let n = nrows;
        let evd = SelfAdjointEigendecomposition::new(matrix.as_faer(), faer::Side::Lower);
        Ok(Self { evd, n })
    }

    /// Real eigenvalues in ascending order.
    pub fn eigenvalues(&self) -> Vec<S> {
        let s = self.evd.s();
        (0..self.n).map(|i| s.column_vector().read(i)).collect()
    }

    /// Orthogonal eigenvector matrix (n x n).
    pub fn eigenvectors(&self) -> DenseMatrix<S> {
        DenseMatrix::from_faer(self.evd.u().to_owned())
    }

    /// Dimension of the decomposed matrix.
    pub fn dim(&self) -> usize {
        self.n
    }
}

/// General eigendecomposition (complex eigenvalues for real matrices).
///
/// Stores real and imaginary parts of eigenvalues separately to avoid
/// exposing complex number types in the public API.
pub struct EigenDecomposition<S: Scalar> {
    eigenvalues_re: Vec<S>,
    eigenvalues_im: Vec<S>,
    n: usize,
}

impl EigenDecomposition<f64> {
    /// Compute general eigenvalues of a real matrix.
    ///
    /// Returns eigenvalues as separate real and imaginary parts.
    /// Complex eigenvalues come in conjugate pairs.
    pub fn new(matrix: &DenseMatrix<f64>) -> Result<Self, LinalgError> {
        let nrows = matrix.rows();
        let ncols = matrix.cols();
        if nrows != ncols {
            return Err(LinalgError::NotSquare { nrows, ncols });
        }
        let n = nrows;

        let evals: Vec<faer::complex_native::c64> =
            matrix.as_faer().eigenvalues::<faer::complex_native::c64>();

        let eigenvalues_re: Vec<f64> = evals.iter().map(|c| c.re).collect();
        let eigenvalues_im: Vec<f64> = evals.iter().map(|c| c.im).collect();

        Ok(Self {
            eigenvalues_re,
            eigenvalues_im,
            n,
        })
    }
}

impl EigenDecomposition<f32> {
    /// Compute general eigenvalues of a real matrix (f32).
    pub fn new(matrix: &DenseMatrix<f32>) -> Result<Self, LinalgError> {
        let nrows = matrix.rows();
        let ncols = matrix.cols();
        if nrows != ncols {
            return Err(LinalgError::NotSquare { nrows, ncols });
        }
        let n = nrows;

        let evals: Vec<faer::complex_native::c32> =
            matrix.as_faer().eigenvalues::<faer::complex_native::c32>();

        let eigenvalues_re: Vec<f32> = evals.iter().map(|c| c.re).collect();
        let eigenvalues_im: Vec<f32> = evals.iter().map(|c| c.im).collect();

        Ok(Self {
            eigenvalues_re,
            eigenvalues_im,
            n,
        })
    }
}

impl<S: Scalar> EigenDecomposition<S> {
    /// Real parts of eigenvalues.
    pub fn eigenvalues_real(&self) -> &[S] {
        &self.eigenvalues_re
    }

    /// Imaginary parts of eigenvalues.
    pub fn eigenvalues_imag(&self) -> &[S] {
        &self.eigenvalues_im
    }

    /// Dimension of the decomposed matrix.
    pub fn dim(&self) -> usize {
        self.n
    }
}

// Convenience methods on DenseMatrix
impl<S: Scalar + SimpleEntity + Conjugate<Canonical = S> + ComplexField> DenseMatrix<S> {
    /// Symmetric eigendecomposition.
    pub fn eigh(&self) -> Result<SymEigenDecomposition<S>, LinalgError> {
        SymEigenDecomposition::new(self)
    }

    /// Symmetric eigenvalues only (ascending order).
    pub fn eigvalsh(&self) -> Result<Vec<S>, LinalgError> {
        let evd = SymEigenDecomposition::new(self)?;
        Ok(evd.eigenvalues())
    }
}

impl DenseMatrix<f64> {
    /// General eigenvalues as (real_parts, imag_parts).
    pub fn eigvals(&self) -> Result<(Vec<f64>, Vec<f64>), LinalgError> {
        let evd = EigenDecomposition::<f64>::new(self)?;
        Ok((evd.eigenvalues_re, evd.eigenvalues_im))
    }
}

impl DenseMatrix<f32> {
    /// General eigenvalues as (real_parts, imag_parts).
    pub fn eigvals(&self) -> Result<(Vec<f32>, Vec<f32>), LinalgError> {
        let evd = EigenDecomposition::<f32>::new(self)?;
        Ok((evd.eigenvalues_re, evd.eigenvalues_im))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Matrix;

    #[test]
    fn test_sym_eigen_2x2() {
        // A = [2 1; 1 2], eigenvalues = 1, 3
        let mut a: DenseMatrix<f64> = DenseMatrix::zeros(2, 2);
        a.set(0, 0, 2.0);
        a.set(0, 1, 1.0);
        a.set(1, 0, 1.0);
        a.set(1, 1, 2.0);

        let evd = SymEigenDecomposition::<f64>::new(&a).unwrap();
        let eigenvalues = evd.eigenvalues();

        // Ascending order
        assert!((eigenvalues[0] - 1.0).abs() < 1e-10);
        assert!((eigenvalues[1] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_sym_eigen_3x3() {
        // Diagonal: eigenvalues are the diagonal entries
        let mut a: DenseMatrix<f64> = DenseMatrix::zeros(3, 3);
        a.set(0, 0, 5.0);
        a.set(1, 1, 2.0);
        a.set(2, 2, 8.0);

        let evd = SymEigenDecomposition::<f64>::new(&a).unwrap();
        let eigenvalues = evd.eigenvalues();

        // Ascending: 2, 5, 8
        assert!((eigenvalues[0] - 2.0).abs() < 1e-10);
        assert!((eigenvalues[1] - 5.0).abs() < 1e-10);
        assert!((eigenvalues[2] - 8.0).abs() < 1e-10);
    }

    #[test]
    fn test_eigenvectors_orthogonal() {
        // A = [2 1; 1 2], V should be orthogonal: V^T V ≈ I
        let mut a: DenseMatrix<f64> = DenseMatrix::zeros(2, 2);
        a.set(0, 0, 2.0);
        a.set(0, 1, 1.0);
        a.set(1, 0, 1.0);
        a.set(1, 1, 2.0);

        let evd = SymEigenDecomposition::<f64>::new(&a).unwrap();
        let v = evd.eigenvectors();

        // V^T V should be identity
        let n = 2;
        for i in 0..n {
            for j in 0..n {
                let mut dot = 0.0;
                for k in 0..n {
                    dot += v.get(k, i) * v.get(k, j);
                }
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (dot - expected).abs() < 1e-10,
                    "V^T V not identity at ({}, {}): {} vs {}",
                    i,
                    j,
                    dot,
                    expected
                );
            }
        }
    }

    #[test]
    fn test_sym_eigen_reconstruction() {
        // V * diag(lambda) * V^T ≈ A
        let mut a: DenseMatrix<f64> = DenseMatrix::zeros(3, 3);
        a.set(0, 0, 4.0);
        a.set(0, 1, 1.0);
        a.set(0, 2, 0.0);
        a.set(1, 0, 1.0);
        a.set(1, 1, 3.0);
        a.set(1, 2, 1.0);
        a.set(2, 0, 0.0);
        a.set(2, 1, 1.0);
        a.set(2, 2, 2.0);

        let evd = SymEigenDecomposition::<f64>::new(&a).unwrap();
        let eigenvalues = evd.eigenvalues();
        let v = evd.eigenvectors();

        let n = 3;
        for i in 0..n {
            for j in 0..n {
                let mut val = 0.0;
                for k in 0..n {
                    val += v.get(i, k) * eigenvalues[k] * v.get(j, k);
                }
                assert!(
                    (val - a.get(i, j)).abs() < 1e-10,
                    "Reconstruction failed at ({}, {}): {} vs {}",
                    i,
                    j,
                    val,
                    a.get(i, j)
                );
            }
        }
    }

    #[test]
    fn test_general_eigenvalues_rotation() {
        // 90-degree rotation: [[0, -1], [1, 0]]
        // Eigenvalues: +i, -i
        let mut a: DenseMatrix<f64> = DenseMatrix::zeros(2, 2);
        a.set(0, 0, 0.0);
        a.set(0, 1, -1.0);
        a.set(1, 0, 1.0);
        a.set(1, 1, 0.0);

        let evd = EigenDecomposition::<f64>::new(&a).unwrap();
        let re = evd.eigenvalues_real();
        let im = evd.eigenvalues_imag();

        // Both real parts should be 0
        assert!(re[0].abs() < 1e-10);
        assert!(re[1].abs() < 1e-10);

        // Imaginary parts should be ±1 (conjugate pair)
        let mut im_sorted = vec![im[0], im[1]];
        im_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert!((im_sorted[0] - (-1.0)).abs() < 1e-10);
        assert!((im_sorted[1] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_general_eigenvalues_diagonal() {
        // Diagonal matrix: eigenvalues are the diagonal entries
        let mut a: DenseMatrix<f64> = DenseMatrix::zeros(3, 3);
        a.set(0, 0, 2.0);
        a.set(1, 1, 5.0);
        a.set(2, 2, -1.0);

        let evd = EigenDecomposition::<f64>::new(&a).unwrap();
        let re = evd.eigenvalues_real();
        let im = evd.eigenvalues_imag();

        // All imaginary parts should be 0
        for &v in im.iter() {
            assert!(v.abs() < 1e-10);
        }

        // Real parts should be {-1, 2, 5} in some order
        let mut re_sorted = re.to_vec();
        re_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert!((re_sorted[0] - (-1.0)).abs() < 1e-10);
        assert!((re_sorted[1] - 2.0).abs() < 1e-10);
        assert!((re_sorted[2] - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_convenience_eigh() {
        let mut a: DenseMatrix<f64> = DenseMatrix::zeros(2, 2);
        a.set(0, 0, 3.0);
        a.set(1, 1, 7.0);

        let eigenvalues = a.eigvalsh().unwrap();
        assert!((eigenvalues[0] - 3.0).abs() < 1e-10);
        assert!((eigenvalues[1] - 7.0).abs() < 1e-10);
    }

    #[test]
    fn test_convenience_eigvals() {
        let mut a: DenseMatrix<f64> = DenseMatrix::zeros(2, 2);
        a.set(0, 0, 1.0);
        a.set(1, 1, 2.0);

        let (re, im) = a.eigvals().unwrap();
        for &v in im.iter() {
            assert!(v.abs() < 1e-10);
        }
        let mut re_sorted = re.clone();
        re_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert!((re_sorted[0] - 1.0).abs() < 1e-10);
        assert!((re_sorted[1] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_not_square_error() {
        let a: DenseMatrix<f64> = DenseMatrix::zeros(2, 3);
        assert!(SymEigenDecomposition::new(&a).is_err());
        assert!(EigenDecomposition::<f64>::new(&a).is_err());
    }

    #[test]
    fn test_sym_eigen_f32() {
        let mut a: DenseMatrix<f32> = DenseMatrix::zeros(2, 2);
        a.set(0, 0, 4.0);
        a.set(0, 1, 0.0);
        a.set(1, 0, 0.0);
        a.set(1, 1, 9.0);

        let evd = SymEigenDecomposition::new(&a).unwrap();
        let eigenvalues = evd.eigenvalues();

        assert!((eigenvalues[0] - 4.0f32).abs() < 1e-5);
        assert!((eigenvalues[1] - 9.0f32).abs() < 1e-5);
    }

    #[test]
    fn test_general_eigen_f32() {
        let mut a: DenseMatrix<f32> = DenseMatrix::zeros(2, 2);
        a.set(0, 0, 0.0);
        a.set(0, 1, -1.0);
        a.set(1, 0, 1.0);
        a.set(1, 1, 0.0);

        let evd = EigenDecomposition::<f32>::new(&a).unwrap();
        let re = evd.eigenvalues_real();
        let im = evd.eigenvalues_imag();

        assert!(re[0].abs() < 1e-5);
        assert!(re[1].abs() < 1e-5);

        let mut im_sorted: Vec<f32> = vec![im[0], im[1]];
        im_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert!((im_sorted[0] - (-1.0f32)).abs() < 1e-5);
        assert!((im_sorted[1] - 1.0f32).abs() < 1e-5);
    }
}

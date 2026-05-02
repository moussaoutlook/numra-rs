//! Singular Value Decomposition (SVD).
//!
//! Author: Moussa Leblouba
//! Date: 9 February 2026
//! Modified: 2 May 2026

use crate::matrix::DenseMatrix;
use crate::Scalar;
use faer::linalg::solvers::{Svd, ThinSvd};
use faer::{ComplexField, Conjugate, Entity, Mat, RealField, SimpleEntity};
use numra_core::LinalgError;

/// Full SVD decomposition: A = U * diag(S) * V^T.
///
/// Caches the decomposition computed in `new()`.
pub struct SvdDecomposition<S: Scalar + Entity> {
    svd: Svd<S>,
    m: usize,
    n: usize,
}

impl<S: Scalar + SimpleEntity + Conjugate<Canonical = S> + ComplexField> SvdDecomposition<S>
where
    S::Real: RealField,
{
    /// Compute full SVD of a matrix.
    pub fn new(matrix: &DenseMatrix<S>) -> Result<Self, LinalgError> {
        let m = matrix.rows();
        let n = matrix.cols();
        let svd = Svd::new(matrix.as_faer());
        Ok(Self { svd, m, n })
    }

    /// Left singular vectors (m x m).
    pub fn u(&self) -> DenseMatrix<S> {
        DenseMatrix::from_faer(self.svd.u().to_owned())
    }

    /// Right singular vectors (n x n).
    pub fn v(&self) -> DenseMatrix<S> {
        DenseMatrix::from_faer(self.svd.v().to_owned())
    }

    /// Singular values in decreasing order.
    pub fn singular_values(&self) -> Vec<S> {
        let s = self.svd.s_diagonal();
        (0..s.nrows()).map(|i| s.read(i)).collect()
    }

    /// Moore-Penrose pseudoinverse.
    pub fn pseudoinverse(&self) -> DenseMatrix<S> {
        DenseMatrix::from_faer(self.svd.pseudoinverse())
    }

    /// Numerical rank: count of singular values > tol.
    pub fn rank(&self, tol: S) -> usize {
        let s = self.svd.s_diagonal();
        (0..s.nrows()).filter(|&i| s.read(i).abs() > tol).count()
    }

    /// Condition number: sigma_max / sigma_min.
    pub fn cond(&self) -> S {
        let s = self.svd.s_diagonal();
        let k = s.nrows();
        if k == 0 {
            return S::ZERO;
        }
        let s_max = s.read(0).abs();
        let s_min = s.read(k - 1).abs();
        if s_min == S::ZERO {
            return S::INFINITY;
        }
        s_max / s_min
    }

    /// Number of rows of the original matrix.
    pub fn nrows(&self) -> usize {
        self.m
    }

    /// Number of columns of the original matrix.
    pub fn ncols(&self) -> usize {
        self.n
    }
}

/// Thin SVD decomposition: A = U * diag(S) * V^T.
///
/// U is m x min(m,n), V is n x min(m,n).
pub struct ThinSvdDecomposition<S: Scalar + Entity> {
    svd: ThinSvd<S>,
    m: usize,
    n: usize,
}

impl<S: Scalar + SimpleEntity + Conjugate<Canonical = S> + ComplexField> ThinSvdDecomposition<S>
where
    S::Real: RealField,
{
    /// Compute thin SVD of a matrix.
    pub fn new(matrix: &DenseMatrix<S>) -> Result<Self, LinalgError> {
        let m = matrix.rows();
        let n = matrix.cols();
        let svd = ThinSvd::new(matrix.as_faer());
        Ok(Self { svd, m, n })
    }

    /// Left singular vectors (m x min(m,n)).
    pub fn u(&self) -> DenseMatrix<S> {
        DenseMatrix::from_faer(self.svd.u().to_owned())
    }

    /// Right singular vectors (n x min(m,n)).
    pub fn v(&self) -> DenseMatrix<S> {
        DenseMatrix::from_faer(self.svd.v().to_owned())
    }

    /// Singular values in decreasing order.
    pub fn singular_values(&self) -> Vec<S> {
        let s = self.svd.s_diagonal();
        (0..s.nrows()).map(|i| s.read(i)).collect()
    }

    /// Moore-Penrose pseudoinverse.
    pub fn pseudoinverse(&self) -> DenseMatrix<S> {
        DenseMatrix::from_faer(self.svd.pseudoinverse())
    }

    /// Numerical rank: count of singular values > tol.
    pub fn rank(&self, tol: S) -> usize {
        let s = self.svd.s_diagonal();
        (0..s.nrows()).filter(|&i| s.read(i).abs() > tol).count()
    }

    /// Condition number: sigma_max / sigma_min.
    pub fn cond(&self) -> S {
        let s = self.svd.s_diagonal();
        let k = s.nrows();
        if k == 0 {
            return S::ZERO;
        }
        let s_max = s.read(0).abs();
        let s_min = s.read(k - 1).abs();
        if s_min == S::ZERO {
            return S::INFINITY;
        }
        s_max / s_min
    }

    /// Number of rows of the original matrix.
    pub fn nrows(&self) -> usize {
        self.m
    }

    /// Number of columns of the original matrix.
    pub fn ncols(&self) -> usize {
        self.n
    }
}

// Convenience methods on DenseMatrix
impl<S: Scalar + SimpleEntity + Conjugate<Canonical = S> + ComplexField> DenseMatrix<S>
where
    S::Real: RealField,
{
    /// Compute full SVD.
    pub fn svd(&self) -> Result<SvdDecomposition<S>, LinalgError> {
        SvdDecomposition::new(self)
    }

    /// Compute thin SVD.
    pub fn thin_svd(&self) -> Result<ThinSvdDecomposition<S>, LinalgError> {
        ThinSvdDecomposition::new(self)
    }

    /// Compute singular values only (decreasing order).
    pub fn singular_values(&self) -> Vec<S> {
        let svd = ThinSvd::new(self.as_faer());
        let s = svd.s_diagonal();
        (0..s.nrows()).map(|i| s.read(i)).collect()
    }

    /// Moore-Penrose pseudoinverse via thin SVD.
    pub fn pinv(&self) -> Result<DenseMatrix<S>, LinalgError> {
        let svd = ThinSvdDecomposition::new(self)?;
        Ok(svd.pseudoinverse())
    }

    /// Condition number via SVD (sigma_max / sigma_min).
    pub fn cond(&self) -> S {
        let svd = ThinSvd::new(self.as_faer());
        let s = svd.s_diagonal();
        let k = s.nrows();
        if k == 0 {
            return S::ZERO;
        }
        let s_max = s.read(0).abs();
        let s_min = s.read(k - 1).abs();
        if s_min == S::ZERO {
            return S::INFINITY;
        }
        s_max / s_min
    }

    /// Numerical rank: count of singular values > tol.
    pub fn rank(&self, tol: S) -> usize {
        let svd = ThinSvd::new(self.as_faer());
        let s = svd.s_diagonal();
        (0..s.nrows()).filter(|&i| s.read(i).abs() > tol).count()
    }

    /// Least-squares solve via thin SVD pseudoinverse: min ||Ax - b||_2.
    pub fn lstsq(&self, b: &[S]) -> Result<Vec<S>, LinalgError> {
        if b.len() != self.rows() {
            return Err(LinalgError::DimensionMismatch {
                expected: (self.rows(), 1),
                actual: (b.len(), 1),
            });
        }
        let pinv = self.pinv()?;
        // pinv is n x m, b is m x 1 → x is n x 1
        let mut b_mat = Mat::zeros(self.rows(), 1);
        for (i, &val) in b.iter().enumerate() {
            b_mat.write(i, 0, val);
        }
        let pinv_ref = pinv.as_faer();
        let result = pinv_ref * b_mat.as_ref();
        let x: Vec<S> = (0..self.cols()).map(|i| result.read(i, 0)).collect();
        Ok(x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Matrix;

    #[test]
    fn test_svd_diagonal() {
        // SVD of diagonal matrix → singular values are the diagonal entries (sorted desc)
        let mut a: DenseMatrix<f64> = DenseMatrix::zeros(3, 3);
        a.set(0, 0, 3.0);
        a.set(1, 1, 1.0);
        a.set(2, 2, 2.0);

        let svd = SvdDecomposition::new(&a).unwrap();
        let s = svd.singular_values();

        assert!((s[0] - 3.0).abs() < 1e-10);
        assert!((s[1] - 2.0).abs() < 1e-10);
        assert!((s[2] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_svd_rectangular_reconstruction() {
        // SVD of 3x2 matrix: U * S * V^T ≈ A
        let mut a: DenseMatrix<f64> = DenseMatrix::zeros(3, 2);
        a.set(0, 0, 1.0);
        a.set(0, 1, 2.0);
        a.set(1, 0, 3.0);
        a.set(1, 1, 4.0);
        a.set(2, 0, 5.0);
        a.set(2, 1, 6.0);

        let svd = SvdDecomposition::new(&a).unwrap();
        let u = svd.u();
        let v = svd.v();
        let s = svd.singular_values();

        // Reconstruct: U * diag(s) * V^T
        let m = a.rows();
        let n = a.cols();
        let k = s.len();
        for i in 0..m {
            for j in 0..n {
                let mut val = 0.0;
                for p in 0..k {
                    val += u.get(i, p) * s[p] * v.get(j, p);
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
    fn test_pseudoinverse() {
        // A * pinv(A) * A ≈ A
        let mut a: DenseMatrix<f64> = DenseMatrix::zeros(3, 2);
        a.set(0, 0, 1.0);
        a.set(0, 1, 2.0);
        a.set(1, 0, 3.0);
        a.set(1, 1, 4.0);
        a.set(2, 0, 5.0);
        a.set(2, 1, 6.0);

        let svd = SvdDecomposition::new(&a).unwrap();
        let pinv = svd.pseudoinverse();

        // Compute A * pinv(A) * A
        // pinv is 2x3, A is 3x2
        // A * pinv = 3x3, then * A = 3x2
        let m = a.rows();
        let n = a.cols();
        assert_eq!(pinv.rows(), n); // pinv is n x m
        assert_eq!(pinv.cols(), m); // pinv is n x m = 2 x 3

        // A * pinv(A) → m x m
        let mut a_pinv: DenseMatrix<f64> = DenseMatrix::zeros(m, m);
        for i in 0..m {
            for j in 0..m {
                let mut val = 0.0;
                for k in 0..n {
                    val += a.get(i, k) * pinv.get(k, j);
                }
                a_pinv.set(i, j, val);
            }
        }

        // (A * pinv(A)) * A → m x n
        for i in 0..m {
            for j in 0..n {
                let mut val = 0.0;
                for k in 0..m {
                    val += a_pinv.get(i, k) * a.get(k, j);
                }
                assert!(
                    (val - a.get(i, j)).abs() < 1e-10,
                    "A * pinv(A) * A != A at ({}, {}): {} vs {}",
                    i,
                    j,
                    val,
                    a.get(i, j)
                );
            }
        }
    }

    #[test]
    fn test_condition_number_identity() {
        let a: DenseMatrix<f64> = DenseMatrix::identity(4);
        let svd = SvdDecomposition::new(&a).unwrap();
        assert!((svd.cond() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_rank_deficient() {
        // Rank-1 matrix: [1 2; 2 4]
        let mut a: DenseMatrix<f64> = DenseMatrix::zeros(2, 2);
        a.set(0, 0, 1.0);
        a.set(0, 1, 2.0);
        a.set(1, 0, 2.0);
        a.set(1, 1, 4.0);

        let svd = SvdDecomposition::new(&a).unwrap();
        assert_eq!(svd.rank(1e-10), 1);
    }

    #[test]
    fn test_lstsq() {
        // Overdetermined: A = [1 1; 1 2; 1 3], b = [1; 2; 2]
        let mut a: DenseMatrix<f64> = DenseMatrix::zeros(3, 2);
        a.set(0, 0, 1.0);
        a.set(0, 1, 1.0);
        a.set(1, 0, 1.0);
        a.set(1, 1, 2.0);
        a.set(2, 0, 1.0);
        a.set(2, 1, 3.0);

        let b = vec![1.0, 2.0, 2.0];
        let x = a.lstsq(&b).unwrap();

        // QR solution: x ≈ [2/3, 1/2]
        assert!((x[0] - 2.0 / 3.0).abs() < 1e-10);
        assert!((x[1] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_thin_svd() {
        let mut a: DenseMatrix<f64> = DenseMatrix::zeros(4, 2);
        a.set(0, 0, 1.0);
        a.set(0, 1, 0.0);
        a.set(1, 0, 0.0);
        a.set(1, 1, 2.0);
        a.set(2, 0, 0.0);
        a.set(2, 1, 0.0);
        a.set(3, 0, 0.0);
        a.set(3, 1, 0.0);

        let svd = ThinSvdDecomposition::new(&a).unwrap();
        let u = svd.u();
        let v = svd.v();
        let s = svd.singular_values();

        // Thin: U is 4x2, V is 2x2
        assert_eq!(u.rows(), 4);
        assert_eq!(u.cols(), 2);
        assert_eq!(v.rows(), 2);
        assert_eq!(v.cols(), 2);

        assert!((s[0] - 2.0).abs() < 1e-10);
        assert!((s[1] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_convenience_methods() {
        let mut a: DenseMatrix<f64> = DenseMatrix::zeros(2, 2);
        a.set(0, 0, 3.0);
        a.set(1, 1, 1.0);

        let s = a.singular_values();
        assert!((s[0] - 3.0).abs() < 1e-10);
        assert!((s[1] - 1.0).abs() < 1e-10);

        assert!((a.cond() - 3.0).abs() < 1e-10);
        assert_eq!(a.rank(1e-10), 2);
    }

    #[test]
    fn test_svd_f32() {
        let mut a: DenseMatrix<f32> = DenseMatrix::zeros(2, 2);
        a.set(0, 0, 3.0);
        a.set(0, 1, 0.0);
        a.set(1, 0, 0.0);
        a.set(1, 1, 2.0);

        let svd = SvdDecomposition::new(&a).unwrap();
        let s = svd.singular_values();

        assert!((s[0] - 3.0).abs() < 1e-5);
        assert!((s[1] - 2.0).abs() < 1e-5);
    }
}

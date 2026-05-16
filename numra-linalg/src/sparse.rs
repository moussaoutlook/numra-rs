//! Sparse matrix types and solvers.
//!
//! Author: Moussa Leblouba
//! Date: 9 February 2026
//! Modified: 2 May 2026

use crate::cholesky::CholeskyFactorization;
use crate::lu::LUFactorization;
use crate::matrix::DenseMatrix;
use crate::Scalar;
use faer::sparse::SparseColMat;
use faer::{ComplexField, Conjugate, Entity, SimpleEntity};
use numra_core::LinalgError;

/// CSC sparse matrix wrapping faer's `SparseColMat`.
///
/// **Trait relationship**: `SparseMatrix<S>` does not currently implement
/// the [`crate::Matrix`] trait. Sparse-aware solvers in this crate
/// (`crate::iterative`, `crate::preconditioner`) consume `&SparseMatrix<S>`
/// concretely. Whether sparse should join the [`crate::Matrix`] trait is an
/// open foundation question deferred per Foundation Spec §7 #5 — see
/// F-MATRIX-SHAPE in `docs/internal-followups.md`.
pub struct SparseMatrix<S: Scalar + Entity> {
    inner: SparseColMat<usize, S>,
    nrows: usize,
    ncols: usize,
}

impl<S: Scalar + SimpleEntity + Conjugate<Canonical = S> + ComplexField> SparseMatrix<S> {
    /// Build from COO (triplet) format.
    ///
    /// Duplicate entries at the same (row, col) are summed.
    pub fn from_triplets(
        nrows: usize,
        ncols: usize,
        triplets: &[(usize, usize, S)],
    ) -> Result<Self, LinalgError> {
        let faer_triplets: Vec<(usize, usize, S)> = triplets.to_vec();
        let inner =
            SparseColMat::try_new_from_triplets(nrows, ncols, &faer_triplets).map_err(|_| {
                LinalgError::DimensionMismatch {
                    expected: (nrows, ncols),
                    actual: (0, 0),
                }
            })?;
        Ok(Self {
            inner,
            nrows,
            ncols,
        })
    }

    /// Number of rows.
    pub fn nrows(&self) -> usize {
        self.nrows
    }

    /// Number of columns.
    pub fn ncols(&self) -> usize {
        self.ncols
    }

    /// Number of structural non-zeros.
    pub fn nnz(&self) -> usize {
        self.inner.compute_nnz()
    }

    /// Element access (returns zero for missing entries).
    pub fn get(&self, row: usize, col: usize) -> S {
        let col_ptrs = self.inner.col_ptrs();
        let row_indices = self.inner.row_indices();
        let values = self.inner.values();

        let start = col_ptrs[col];
        let end = col_ptrs[col + 1];
        for idx in start..end {
            if row_indices[idx] == row {
                return values[idx];
            }
        }
        S::ZERO
    }

    /// Convert to dense matrix.
    pub fn to_dense(&self) -> DenseMatrix<S> {
        DenseMatrix::from_faer(self.inner.to_dense())
    }

    /// CSC column pointers.
    pub fn col_ptrs(&self) -> Vec<usize> {
        self.inner.col_ptrs().to_vec()
    }

    /// CSC row indices.
    pub fn row_indices(&self) -> Vec<usize> {
        self.inner.row_indices().to_vec()
    }

    /// CSC values.
    pub fn values(&self) -> Vec<S> {
        let vals = self.inner.values();
        (0..vals.len()).map(|i| vals[i]).collect()
    }

    /// Sparse matrix-vector product: y = A * x.
    pub fn mul_vec(&self, x: &[S]) -> Result<Vec<S>, LinalgError> {
        if x.len() != self.ncols {
            return Err(LinalgError::DimensionMismatch {
                expected: (self.ncols, 1),
                actual: (x.len(), 1),
            });
        }

        let col_ptrs = self.inner.col_ptrs();
        let row_indices = self.inner.row_indices();
        let values = self.inner.values();

        let mut y = vec![S::ZERO; self.nrows];

        for j in 0..self.ncols {
            let start = col_ptrs[j];
            let end = col_ptrs[j + 1];
            for idx in start..end {
                let i = row_indices[idx];
                y[i] += values[idx] * x[j];
            }
        }

        Ok(y)
    }

    /// Transpose of the sparse matrix.
    pub fn transpose(&self) -> Result<SparseMatrix<S>, LinalgError> {
        // Build triplets with swapped row/col
        let col_ptrs = self.inner.col_ptrs();
        let row_indices = self.inner.row_indices();
        let values = self.inner.values();

        let mut triplets = Vec::with_capacity(self.nnz());
        for j in 0..self.ncols {
            let start = col_ptrs[j];
            let end = col_ptrs[j + 1];
            for idx in start..end {
                let i = row_indices[idx];
                triplets.push((j, i, values[idx]));
            }
        }

        SparseMatrix::from_triplets(self.ncols, self.nrows, &triplets)
    }
}

/// Sparse LU factorization for solving Ax = b.
///
/// Converts to dense internally (Phase 1 pragmatic approach).
/// A future phase will use native faer sparse solvers for O(nnz) performance.
pub struct SparseLU<S: Scalar + Entity> {
    lu: LUFactorization<S>,
    n: usize,
}

impl<S: Scalar + SimpleEntity + Conjugate<Canonical = S> + ComplexField> SparseLU<S> {
    /// Factor the sparse matrix (converts to dense internally).
    pub fn new(matrix: &SparseMatrix<S>) -> Result<Self, LinalgError> {
        if matrix.nrows() != matrix.ncols() {
            return Err(LinalgError::NotSquare {
                nrows: matrix.nrows(),
                ncols: matrix.ncols(),
            });
        }
        let n = matrix.nrows();
        let dense = matrix.to_dense();
        let lu = LUFactorization::new(&dense)?;
        Ok(Self { lu, n })
    }

    /// Solve Ax = b.
    pub fn solve(&self, b: &[S]) -> Result<Vec<S>, LinalgError> {
        self.lu.solve(b)
    }

    /// Dimension of the factorized matrix.
    pub fn dim(&self) -> usize {
        self.n
    }
}

/// Sparse Cholesky factorization for SPD systems.
///
/// Converts to dense internally (Phase 1 pragmatic approach).
pub struct SparseCholesky<S: Scalar + Entity> {
    chol: CholeskyFactorization<S>,
    n: usize,
}

impl<S: Scalar + SimpleEntity + Conjugate<Canonical = S> + ComplexField> SparseCholesky<S> {
    /// Factor the sparse SPD matrix (converts to dense internally).
    pub fn new(matrix: &SparseMatrix<S>) -> Result<Self, LinalgError> {
        if matrix.nrows() != matrix.ncols() {
            return Err(LinalgError::NotSquare {
                nrows: matrix.nrows(),
                ncols: matrix.ncols(),
            });
        }
        let n = matrix.nrows();
        let dense = matrix.to_dense();
        let chol = CholeskyFactorization::new(&dense)?;
        Ok(Self { chol, n })
    }

    /// Solve Ax = b.
    pub fn solve(&self, b: &[S]) -> Result<Vec<S>, LinalgError> {
        self.chol.solve(b)
    }

    /// Dimension of the factorized matrix.
    pub fn dim(&self) -> usize {
        self.n
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Matrix;

    #[test]
    fn test_identity_from_triplets() {
        let triplets = vec![(0, 0, 1.0), (1, 1, 1.0), (2, 2, 1.0)];
        let m = SparseMatrix::from_triplets(3, 3, &triplets).unwrap();

        assert_eq!(m.nrows(), 3);
        assert_eq!(m.ncols(), 3);
        assert_eq!(m.nnz(), 3);

        assert!((m.get(0, 0) - 1.0).abs() < 1e-15);
        assert!((m.get(1, 1) - 1.0).abs() < 1e-15);
        assert!((m.get(2, 2) - 1.0).abs() < 1e-15);
        assert!(m.get(0, 1).abs() < 1e-15);
    }

    #[test]
    fn test_tridiagonal() {
        // [-2 1 0; 1 -2 1; 0 1 -2]
        let triplets = vec![
            (0, 0, -2.0),
            (0, 1, 1.0),
            (1, 0, 1.0),
            (1, 1, -2.0),
            (1, 2, 1.0),
            (2, 1, 1.0),
            (2, 2, -2.0),
        ];
        let m = SparseMatrix::from_triplets(3, 3, &triplets).unwrap();

        assert_eq!(m.nnz(), 7);
        assert!((m.get(0, 0) - (-2.0)).abs() < 1e-15);
        assert!((m.get(0, 1) - 1.0).abs() < 1e-15);
        assert!((m.get(2, 2) - (-2.0)).abs() < 1e-15);
        assert!(m.get(0, 2).abs() < 1e-15);
    }

    #[test]
    fn test_duplicate_entries_summed() {
        // Two entries at (0,0) with values 3.0 and 4.0 should sum to 7.0
        let triplets = vec![(0, 0, 3.0), (0, 0, 4.0), (1, 1, 1.0)];
        let m = SparseMatrix::from_triplets(2, 2, &triplets).unwrap();

        assert!((m.get(0, 0) - 7.0).abs() < 1e-15);
        assert!((m.get(1, 1) - 1.0).abs() < 1e-15);
    }

    #[test]
    fn test_to_dense_roundtrip() {
        let triplets = vec![(0, 0, 1.0), (0, 1, 2.0), (1, 0, 3.0), (1, 1, 4.0)];
        let sparse = SparseMatrix::from_triplets(2, 2, &triplets).unwrap();
        let dense = sparse.to_dense();

        assert!((dense.get(0, 0) - 1.0).abs() < 1e-15);
        assert!((dense.get(0, 1) - 2.0).abs() < 1e-15);
        assert!((dense.get(1, 0) - 3.0).abs() < 1e-15);
        assert!((dense.get(1, 1) - 4.0).abs() < 1e-15);
    }

    #[test]
    fn test_spmv_tridiagonal() {
        // [2 -1 0; -1 2 -1; 0 -1 2] * [1; 1; 1] = [1; 0; 1]
        let triplets = vec![
            (0, 0, 2.0),
            (0, 1, -1.0),
            (1, 0, -1.0),
            (1, 1, 2.0),
            (1, 2, -1.0),
            (2, 1, -1.0),
            (2, 2, 2.0),
        ];
        let m = SparseMatrix::from_triplets(3, 3, &triplets).unwrap();
        let x = vec![1.0, 1.0, 1.0];
        let y = m.mul_vec(&x).unwrap();

        assert!((y[0] - 1.0).abs() < 1e-10);
        assert!((y[1] - 0.0).abs() < 1e-10);
        assert!((y[2] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_sparse_lu_solve() {
        // Same system as dense: [1 2; 3 4] x = [5; 11] → x = [1; 2]
        let triplets = vec![(0, 0, 1.0), (0, 1, 2.0), (1, 0, 3.0), (1, 1, 4.0)];
        let sparse = SparseMatrix::from_triplets(2, 2, &triplets).unwrap();
        let lu = SparseLU::new(&sparse).unwrap();

        let b = vec![5.0, 11.0];
        let x = lu.solve(&b).unwrap();

        assert!((x[0] - 1.0).abs() < 1e-10);
        assert!((x[1] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_sparse_lu_matches_dense() {
        // Tridiagonal 4x4 system
        let triplets = vec![
            (0, 0, 4.0),
            (0, 1, -1.0),
            (1, 0, -1.0),
            (1, 1, 4.0),
            (1, 2, -1.0),
            (2, 1, -1.0),
            (2, 2, 4.0),
            (2, 3, -1.0),
            (3, 2, -1.0),
            (3, 3, 4.0),
        ];
        let sparse = SparseMatrix::from_triplets(4, 4, &triplets).unwrap();
        let dense = sparse.to_dense();

        let b = vec![1.0, 2.0, 3.0, 4.0];

        let x_dense = dense.solve(&b).unwrap();
        let lu = SparseLU::new(&sparse).unwrap();
        let x_sparse = lu.solve(&b).unwrap();

        for i in 0..4 {
            assert!(
                (x_dense[i] - x_sparse[i]).abs() < 1e-10,
                "Mismatch at {}: {} vs {}",
                i,
                x_dense[i],
                x_sparse[i]
            );
        }
    }

    #[test]
    fn test_sparse_cholesky_solve() {
        // SPD: [4 2; 2 3]
        let triplets = vec![(0, 0, 4.0), (0, 1, 2.0), (1, 0, 2.0), (1, 1, 3.0)];
        let sparse = SparseMatrix::from_triplets(2, 2, &triplets).unwrap();
        let chol = SparseCholesky::new(&sparse).unwrap();

        let b = vec![6.0, 5.0];
        let x = chol.solve(&b).unwrap();

        assert!((x[0] - 1.0).abs() < 1e-10);
        assert!((x[1] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_transpose() {
        let triplets = vec![(0, 0, 1.0), (0, 1, 2.0), (1, 0, 3.0), (1, 1, 4.0)];
        let m = SparseMatrix::from_triplets(2, 2, &triplets).unwrap();
        let mt = m.transpose().unwrap();

        assert!((mt.get(0, 0) - 1.0).abs() < 1e-15);
        assert!((mt.get(0, 1) - 3.0).abs() < 1e-15);
        assert!((mt.get(1, 0) - 2.0).abs() < 1e-15);
        assert!((mt.get(1, 1) - 4.0).abs() < 1e-15);
    }

    #[test]
    fn test_spmv_dimension_mismatch() {
        let triplets = vec![(0, 0, 1.0), (1, 1, 1.0)];
        let m = SparseMatrix::from_triplets(2, 2, &triplets).unwrap();

        let x = vec![1.0, 2.0, 3.0]; // Wrong size
        assert!(m.mul_vec(&x).is_err());
    }

    #[test]
    fn test_sparse_f32() {
        let triplets: Vec<(usize, usize, f32)> = vec![(0, 0, 2.0), (1, 1, 3.0)];
        let m = SparseMatrix::from_triplets(2, 2, &triplets).unwrap();
        let lu = SparseLU::new(&m).unwrap();

        let b = vec![4.0f32, 9.0f32];
        let x = lu.solve(&b).unwrap();

        assert!((x[0] - 2.0).abs() < 1e-5);
        assert!((x[1] - 3.0).abs() < 1e-5);
    }
}

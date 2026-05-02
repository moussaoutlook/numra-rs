#![allow(clippy::needless_range_loop)]
#![allow(clippy::useless_vec)]

//! Linear algebra abstractions for Numra.
//!
//! This crate provides matrix operations and linear solvers built on [faer].
//!
//! # Design
//!
//! The [`Matrix`] trait provides a backend-agnostic interface for matrix operations.
//! The primary implementation wraps faer's `Mat<S>` type.
//!
//! # Example
//!
//! ```rust
//! use numra_linalg::{Matrix, DenseMatrix};
//!
//! // Create a 3x3 matrix
//! let mut a: DenseMatrix<f64> = DenseMatrix::zeros(3, 3);
//! a.set(0, 0, 2.0);
//! a.set(1, 1, 3.0);
//! a.set(2, 2, 4.0);
//!
//! // Solve Ax = b
//! let b = vec![1.0, 2.0, 3.0];
//! let x = a.solve(&b).unwrap();
//!
//! assert!((x[0] - 0.5).abs() < 1e-10);
//! assert!((x[1] - 2.0/3.0).abs() < 1e-10);
//! assert!((x[2] - 0.75).abs() < 1e-10);
//! ```
//!
//! Author: Moussa Leblouba
//! Date: 9 February 2026
//! Modified: 2 May 2026

pub use numra_core::LinalgError;
pub use numra_core::Scalar;

mod cholesky;
mod eigen;
pub mod iterative;
mod lu;
mod matrix;
pub mod preconditioner;
mod qr;
mod sparse;
mod svd;

pub use cholesky::CholeskyFactorization;
pub use eigen::{EigenDecomposition, SymEigenDecomposition};
pub use iterative::{bicgstab, cg, gmres, minres, pcg, IterativeOptions, IterativeResult};
pub use lu::{LUFactorization, LUSolver};
pub use matrix::{DenseMatrix, Matrix};
pub use preconditioner::{IdentityPreconditioner, Ilu0, Jacobi, Preconditioner, Ssor};
pub use qr::QRFactorization;
pub use sparse::{SparseCholesky, SparseLU, SparseMatrix};
pub use svd::{SvdDecomposition, ThinSvdDecomposition};

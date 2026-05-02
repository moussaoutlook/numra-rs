#![allow(clippy::single_match)]

//! Nonlinear solvers for Numra.
//!
//! This crate provides iterative methods for solving systems of nonlinear equations:
//!
//! - **Newton-Raphson** with line search for globalization
//! - Support for both analytical and numerical (finite difference) Jacobians
//!
//! # Example
//!
//! ```rust
//! use numra_nonlinear::{Newton, NewtonOptions, NonlinearSystem};
//!
//! // Solve x^2 = 2 (find sqrt(2))
//! struct SquareRoot;
//!
//! impl NonlinearSystem<f64> for SquareRoot {
//!     fn dim(&self) -> usize { 1 }
//!     fn eval(&self, x: &[f64], f: &mut [f64]) {
//!         f[0] = x[0] * x[0] - 2.0;
//!     }
//! }
//!
//! let solver = Newton::default_solver();
//! let result = solver.solve(&SquareRoot, &[1.5]).unwrap();
//! assert!((result.x[0] - std::f64::consts::SQRT_2).abs() < 1e-10);
//! ```
//!
//! Author: Moussa Leblouba
//! Date: 8 February 2026
//! Modified: 2 May 2026

pub mod line_search;
pub mod newton;

pub use line_search::{wolfe_line_search, LineSearchError, LineSearchResult, WolfeOptions};
pub use newton::{newton_solve, Newton, NewtonOptions, NewtonResult, NonlinearSystem};
pub use numra_linalg::LinalgError;

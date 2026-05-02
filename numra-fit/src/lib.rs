//! # Numra Curve Fitting
//!
//! Nonlinear curve fitting, polynomial fitting, and evaluation.
//!
//! This crate provides tools for fitting parametric models to data,
//! built on top of `numra-optim`'s Levenberg-Marquardt solver and
//! `numra-linalg`'s SVD-based least squares.
//!
//! ## Key functions
//!
//! - [`curve_fit()`] - Fit any `y = f(x, params)` model to data
//! - [`curve_fit_weighted()`] - Weighted nonlinear least squares
//! - [`polyfit()`] - Fit polynomial of given degree
//! - [`polyval()`] - Evaluate polynomial at given points
//!
//! ## Example
//!
//! ```rust
//! use numra_fit::{curve_fit, polyfit, polyval};
//!
//! // Fit an exponential decay: y = a * exp(-b * x)
//! let x = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
//! let y = vec![5.0, 3.03, 1.84, 1.11, 0.67, 0.41];
//!
//! let result = curve_fit(
//!     |x: f64, p: &[f64]| p[0] * (-p[1] * x).exp(),
//!     &x, &y, &[4.0, 0.3], None,
//! ).unwrap();
//!
//! println!("a = {:.3}, b = {:.3}", result.params[0], result.params[1]);
//! println!("R^2 = {:.6}", result.r_squared);
//! ```
//!
//! Author: Moussa Leblouba
//! Date: 9 February 2026
//! Modified: 2 May 2026

// Allow some clippy lints prevalent in numerical code
#![allow(clippy::assign_op_pattern)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::excessive_precision)]
#![allow(clippy::type_complexity)]

pub mod curve_fit;
pub mod error;
pub mod polynomial;
pub mod types;

pub use curve_fit::{curve_fit, curve_fit_weighted, curve_fit_with_jacobian};
pub use error::FitError;
pub use polynomial::{polyfit, polyval};
pub use types::{FitOptions, FitResult};

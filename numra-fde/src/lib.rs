// Clippy: numerical code uses index-heavy loops and literals from references.
#![allow(clippy::needless_range_loop)]

//! FDE (Fractional Differential Equation) solvers for Numra.
//!
//! This crate provides methods for solving fractional differential equations
//! using the Caputo derivative:
//!
//! ```text
//! D^α y(t) = f(t, y),  y(0) = y₀
//! ```
//!
//! where D^α is the Caputo fractional derivative of order α ∈ (0, 1].
//!
//! # The Caputo Derivative
//!
//! The Caputo fractional derivative of order α is defined as:
//!
//! ```text
//! D^α y(t) = (1/Γ(1-α)) ∫₀ᵗ (t-s)^(-α) y'(s) ds
//! ```
//!
//! Key properties:
//! - Reduces to ordinary derivative when α = 1
//! - D^α c = 0 for constants (unlike Riemann-Liouville)
//! - Has "memory" - depends on full history of y
//!
//! # Solvers
//!
//! - [`L1Solver`] - L1 scheme for Caputo derivative (order 2-α accuracy)
//!
//! # Example
//!
//! ```
//! use numra_fde::{FdeSystem, L1Solver, FdeSolver, FdeOptions};
//!
//! // Fractional relaxation: D^α y = -λy, y(0) = 1
//! struct FractionalRelaxation { lambda: f64 }
//!
//! impl FdeSystem<f64> for FractionalRelaxation {
//!     fn dim(&self) -> usize { 1 }
//!     fn alpha(&self) -> f64 { 0.5 }  // Half-order derivative
//!     fn rhs(&self, _t: f64, y: &[f64], f: &mut [f64]) {
//!         f[0] = -self.lambda * y[0];
//!     }
//! }
//!
//! let system = FractionalRelaxation { lambda: 1.0 };
//! let opts = FdeOptions::default().dt(0.01);
//! let result = L1Solver::solve(&system, 0.0, 1.0, &[1.0], &opts);
//! ```
//!
//! Author: Moussa Leblouba
//! Date: 2 February 2026
//! Modified: 2 May 2026

pub use numra_core::Scalar;

mod caputo;
mod l1_solver;
mod system;

pub use caputo::{caputo_weights, gamma, mittag_leffler, mittag_leffler_1};
pub use l1_solver::L1Solver;
pub use system::{FdeOptions, FdeResult, FdeSolver, FdeSystem};

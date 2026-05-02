//! IDE (Integro-Differential Equation) solvers for Numra.
//!
//! This crate provides methods for solving Volterra integro-differential equations:
//!
//! ```text
//! y'(t) = f(t, y) + ∫₀ᵗ K(t, s, y(s)) ds
//! ```
//!
//! where K is the memory kernel representing history-dependent effects.
//!
//! # Memory Kernels
//!
//! The crate supports various kernel types:
//! - **Exponential kernels**: K(t,s) = a * exp(-b*(t-s)), efficient via Prony series
//! - **Power-law kernels**: K(t,s) = (t-s)^(-α), fractional memory
//! - **Custom kernels**: User-defined kernel functions
//!
//! # Solvers
//!
//! - [`VolterraSolver`] - General Volterra IDE solver using quadrature
//! - [`PronySolver`] - Efficient solver for sum-of-exponentials kernels
//!
//! # Example
//!
//! ```
//! use numra_ide::{IdeSystem, VolterraSolver, IdeSolver, IdeOptions, Kernel};
//! use numra_ide::kernels::ExponentialKernel;
//!
//! // Viscoelastic material: y' = -k*y + ∫ K(t-s)*y(s) ds
//! struct Viscoelastic {
//!     k: f64,
//!     kernel: ExponentialKernel<f64>,
//! }
//!
//! impl IdeSystem<f64> for Viscoelastic {
//!     fn dim(&self) -> usize { 1 }
//!
//!     fn rhs(&self, _t: f64, y: &[f64], f: &mut [f64]) {
//!         f[0] = -self.k * y[0];
//!     }
//!
//!     fn kernel(&self, t: f64, s: f64, y: &[f64], k: &mut [f64]) {
//!         k[0] = self.kernel.evaluate(t - s) * y[0];
//!     }
//! }
//! ```
//!
//! Author: Moussa Leblouba
//! Date: 2 February 2026
//! Modified: 2 May 2026

pub use numra_core::Scalar;

pub mod kernels;
pub mod prony;
mod system;
mod volterra;

pub use kernels::{ExponentialKernel, Kernel, PowerLawKernel, PronyKernel};
pub use prony::{PronySolver, PronySystem};
pub use system::{IdeOptions, IdeResult, IdeSolver, IdeStats, IdeSystem};
pub use volterra::{VolterraRK4Solver, VolterraSolver};

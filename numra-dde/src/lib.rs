// Allow some clippy lints that are prevalent in numerical code
#![allow(clippy::assign_op_pattern)]
#![allow(clippy::needless_range_loop)]

//! DDE (Delay Differential Equation) solvers for Numra.
//!
//! This crate provides methods for solving delay differential equations
//! of the form:
//!
//! ```text
//! y'(t) = f(t, y(t), y(t - τ₁), y(t - τ₂), ...)
//! ```
//!
//! where τᵢ are the delays (constant or state-dependent).
//!
//! # Solvers
//!
//! - [`MethodOfSteps`] - Method of steps using embedded RK solver
//!
//! # Example
//!
//! ```
//! use numra_dde::{DdeSystem, MethodOfSteps, DdeSolver, DdeOptions};
//!
//! // Mackey-Glass equation: y'(t) = β * y(t-τ) / (1 + y(t-τ)^n) - γ * y(t)
//! struct MackeyGlass {
//!     beta: f64,
//!     gamma: f64,
//!     n: f64,
//!     tau: f64,
//! }
//!
//! impl DdeSystem<f64> for MackeyGlass {
//!     fn dim(&self) -> usize { 1 }
//!     fn delays(&self) -> Vec<f64> { vec![self.tau] }
//!     fn rhs(&self, _t: f64, y: &[f64], y_delayed: &[&[f64]], dydt: &mut [f64]) {
//!         let y_tau = y_delayed[0][0];  // y(t - tau)
//!         dydt[0] = self.beta * y_tau / (1.0 + y_tau.powf(self.n)) - self.gamma * y[0];
//!     }
//! }
//!
//! let mg = MackeyGlass { beta: 2.0, gamma: 1.0, n: 9.65, tau: 2.0 };
//! let history = |_t: f64| vec![0.5];  // Constant history
//! let opts = DdeOptions::default();
//! let result = MethodOfSteps::solve(&mg, 0.0, 100.0, &history, &opts);
//! ```
//!
//! Author: Moussa Leblouba
//! Date: 3 February 2026
//! Modified: 2 May 2026

pub use numra_core::Scalar;

mod history;
mod method_of_steps;
mod system;

pub use history::{HermiteInterpolator, History, HistoryFunction};
pub use method_of_steps::MethodOfSteps;
pub use system::{DdeOptions, DdeResult, DdeSolver, DdeStats, DdeSystem};

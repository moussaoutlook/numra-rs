//! SDE (Stochastic Differential Equation) solvers for Numra.
//!
//! This crate provides methods for solving stochastic differential equations
//! of the form:
//!
//! ```text
//! dX(t) = f(t, X) dt + g(t, X) dW(t)
//! ```
//!
//! where `f` is the drift, `g` is the diffusion, and `W(t)` is a Wiener process.
//!
//! # Solvers
//!
//! - [`EulerMaruyama`] - Simple fixed-step solver (strong order 0.5)
//! - [`Milstein`] - Higher order solver (strong order 1.0)
//! - [`Sra1`] - Adaptive strong order 1.5 method
//! - [`Sra2`] - Adaptive weak order 2.0 method
//!
//! # Example
//!
//! ```
//! use numra_sde::{SdeSystem, EulerMaruyama, SdeSolver, SdeOptions};
//!
//! // Geometric Brownian Motion: dS = μS dt + σS dW
//! struct GBM { mu: f64, sigma: f64 }
//!
//! impl SdeSystem<f64> for GBM {
//!     fn dim(&self) -> usize { 1 }
//!     fn drift(&self, _t: f64, x: &[f64], f: &mut [f64]) {
//!         f[0] = self.mu * x[0];
//!     }
//!     fn diffusion(&self, _t: f64, x: &[f64], g: &mut [f64]) {
//!         g[0] = self.sigma * x[0];
//!     }
//! }
//!
//! let gbm = GBM { mu: 0.05, sigma: 0.2 };
//! let opts = SdeOptions::default().dt(0.01);
//! let result = EulerMaruyama::solve(&gbm, 0.0, 1.0, &[100.0], &opts, None);
//! assert!(result.is_ok());
//! ```
//!
//! Author: Moussa Leblouba
//! Date: 2 February 2026
//! Modified: 2 May 2026

pub use numra_core::Scalar;

mod ensemble;
mod euler_maruyama;
mod milstein;
mod sra;
mod stats;
mod system;
pub mod wiener;

pub use ensemble::{EnsembleResult, EnsembleRunner};
pub use euler_maruyama::EulerMaruyama;
pub use milstein::Milstein;
pub use sra::{Sra1, Sra2};
pub use stats::{
    mean, median, percentile, std, variance, EnsembleStats, Percentiles, RunningStats,
};
pub use system::{NoiseType, SdeOptions, SdeResult, SdeSolver, SdeSystem};
pub use wiener::{create_wiener, WienerIncrement, WienerProcess};

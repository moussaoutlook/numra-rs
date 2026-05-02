// Allow some clippy lints that are prevalent in numerical code
#![allow(clippy::assign_op_pattern)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::manual_abs_diff)]

//! Stochastic Partial Differential Equation (SPDE) solvers for Numra.
//!
//! This crate provides solvers for SPDEs of the form:
//!
//! ```text
//! ∂u/∂t = L[u] + σ(u) ξ(x,t)
//! ```
//!
//! where L is a spatial differential operator, σ(u) is the diffusion coefficient,
//! and ξ(x,t) is space-time white noise.
//!
//! # Method
//!
//! We use the Method of Lines (MOL) approach:
//! 1. Discretize the spatial operator L using finite differences
//! 2. Convert to a system of SDEs
//! 3. Solve using SDE solvers from `numra-sde`
//!
//! # Example: Stochastic Heat Equation
//!
//! ```
//! use numra_spde::{SpdeSystem, SpdeSolver, SpdeOptions, SpdeResult};
//! use numra_pde::Grid1D;
//! use numra_core::Scalar;
//!
//! // Stochastic heat equation: ∂T/∂t = α ∂²T/∂x² + σ dW(x,t)
//! struct StochasticHeat {
//!     alpha: f64,  // Thermal diffusivity
//!     sigma: f64,  // Noise intensity
//! }
//!
//! impl SpdeSystem<f64> for StochasticHeat {
//!     fn dim(&self) -> usize { 1 }  // 1D spatial
//!
//!     fn drift(&self, _t: f64, u: &[f64], du: &mut [f64], grid: &Grid1D<f64>) {
//!         let dx = grid.dx_uniform();
//!         let n = u.len();
//!         for i in 0..n {
//!             let u_left = if i == 0 { 0.0 } else { u[i - 1] };  // Dirichlet BC
//!             let u_right = if i == n - 1 { 0.0 } else { u[i + 1] };  // Dirichlet BC
//!             du[i] = self.alpha * (u_left - 2.0 * u[i] + u_right) / (dx * dx);
//!         }
//!     }
//!
//!     fn diffusion(&self, _t: f64, u: &[f64], sigma: &mut [f64], _grid: &Grid1D<f64>) {
//!         for i in 0..u.len() {
//!             sigma[i] = self.sigma;  // Additive noise
//!         }
//!     }
//! }
//! ```
//!
//! Author: Moussa Leblouba
//! Date: 3 February 2026
//! Modified: 2 May 2026

pub use numra_core::Scalar;
pub use numra_pde::Grid1D;

mod noise;
mod solver;
mod system;

pub use noise::{ColoredNoise, ColoredNoiseGenerator, SpaceTimeNoise, WhiteNoise};
pub use solver::{
    MolSdeSolver, SpdeEnsemble, SpdeMethod, SpdeOptions, SpdeResult, SpdeSolver, SpdeStats,
};
pub use system::{NoiseCorrelation, SpdeSystem};

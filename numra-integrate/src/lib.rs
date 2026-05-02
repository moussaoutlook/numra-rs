//! Numerical integration and quadrature for Numra.
//!
//! This crate provides numerical integration methods:
//!
//! - **Adaptive quadrature** ([`quad`]): Gauss-Kronrod G7K15 with adaptive subdivision
//! - **Fixed-order Gaussian** ([`gauss_legendre`], [`gauss_laguerre`], [`gauss_hermite`]):
//!   Precomputed nodes/weights for finite, semi-infinite, and infinite intervals
//! - **Composite rules** ([`trapezoid`], [`simpson`], [`romberg`]):
//!   Classical rules for sampled data and function-based integration
//! - **Multi-dimensional** ([`dblquad`]): Iterated 1D quadrature for double integrals
//!
//! Author: Moussa Leblouba
//! Date: 9 February 2026
//! Modified: 2 May 2026

pub mod adaptive;
pub mod composite;
pub mod error;
pub mod fixed;
pub mod multidim;

pub use adaptive::{quad, QuadOptions, QuadResult};
pub use composite::{cumulative_trapezoid, romberg, simpson, trapezoid, trapezoid_nonuniform};
pub use error::IntegrationError;
pub use fixed::{gauss_hermite, gauss_laguerre, gauss_legendre};
pub use multidim::dblquad;

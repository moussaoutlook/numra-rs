//! Automatic differentiation for Numra: forward-mode and reverse-mode.
//!
//! This crate provides two AD modes:
//!
//! - **Forward-mode** ([`Dual`]): Augments values with directional derivatives.
//!   Cost: O(n) passes for n inputs. Best when outputs >> inputs.
//! - **Reverse-mode** ([`reverse::Var`]): Tape-based computation graph with backward pass.
//!   Cost: O(m) passes for m outputs. Best when inputs >> outputs (optimization).
//!
//! # Forward-mode example
//!
//! ```rust
//! use numra_autodiff::{Dual, gradient};
//!
//! let grad = gradient(|x: &[Dual<f64>]| x[0] * x[0] + x[1] * x[1], &[3.0, 4.0]);
//! assert!((grad[0] - 6.0).abs() < 1e-12); // df/dx0 = 2*x0 = 6
//! assert!((grad[1] - 8.0).abs() < 1e-12); // df/dx1 = 2*x1 = 8
//! ```
//!
//! # Reverse-mode example
//!
//! ```rust
//! use numra_autodiff::reverse::grad;
//!
//! let g = grad(|x| x[0].clone() * x[0].clone() + x[1].clone() * x[1].clone(), &[3.0, 4.0]);
//! assert!((g[0] - 6.0).abs() < 1e-12);
//! assert!((g[1] - 8.0).abs() < 1e-12);
//! ```
//!
//! Author: Moussa Leblouba
//! Date: 9 February 2026
//! Modified: 2 May 2026

pub mod bridge;
pub mod dual;
pub mod gradient;
pub mod reverse;
pub mod tape;

pub use bridge::{gradient_closure, model_jacobian_closure};
pub use dual::Dual;
pub use gradient::{gradient, jacobian};

//! # Numra Core
//!
//! Core traits and types for the Numra numerical methods library.
//!
//! This crate provides the fundamental abstractions that all other Numra crates build upon:
//!
//! - [`Scalar`] - Trait for scalar numeric types (f32, f64)
//! - [`Vector`] - Trait for vector types
//! - [`Signal`] - Trait for time-dependent forcing functions
//! - Error types and result definitions
//!
//! ## Design Philosophy
//!
//! Numra Core is designed to be:
//! - **Backend-agnostic**: Works with faer, nalgebra, or pure Rust
//! - **No-std compatible**: Can be used in embedded systems (with alloc)
//! - **Zero-cost abstractions**: Compiles to optimal machine code
//!
//! ## Example
//!
//! ```rust
//! use numra_core::{Scalar, Signal};
//! use numra_core::signal::Harmonic;
//!
//! // Define a harmonic forcing function
//! let forcing = Harmonic::new(1.0, 2.0, 0.0);  // amplitude=1, freq=2 Hz, phase=0
//!
//! // Evaluate at t=0.25s (should be sin(2π*2*0.25) = sin(π) = 0)
//! let value = forcing.eval(0.25);
//! assert!(value.abs() < 1e-10);
//! ```
//!
//! Author: Moussa Leblouba
//! Date: 9 February 2026
//! Modified: 2 May 2026

#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(not(feature = "std"))]
extern crate alloc;

pub mod error;
pub mod scalar;
pub mod signal;
pub mod uncertainty;
pub mod vector;

pub use error::{LinalgError, NumraError, NumraResult};
pub use scalar::{from_f64_vec, to_f64_vec, Scalar};
pub use signal::Signal;
pub use uncertainty::{compute_sensitivities, Interval, Sensitivity, SensitivityResult, Uncertain};
pub use vector::Vector;

/// Commonly used items
pub mod prelude {
    pub use crate::error::{NumraError, NumraResult};
    #[cfg(feature = "std")]
    pub use crate::signal::FromFile;
    pub use crate::signal::{Chirp, Harmonic, Interpolation, Pulse, Ramp, Step, Tabulated};
    pub use crate::uncertainty::{Interval, Uncertain};
    pub use crate::Scalar;
    pub use crate::Signal;
    pub use crate::Vector;
}

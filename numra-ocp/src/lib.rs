#![allow(clippy::needless_range_loop)]
#![allow(clippy::identity_op)]
#![allow(clippy::useless_vec)]

//! ODE-constrained optimization for Numra.
//!
//! # Modules
//!
//! - [`param_est`] -- Parameter estimation for ODE models
//! - [`shooting`] -- Single-shooting optimal control
//!
//! Author: Moussa Leblouba
//! Date: 9 February 2026
//! Modified: 2 May 2026

pub mod adjoint;
pub mod collocation;
pub mod error;
pub mod multiple_shooting;
pub mod param_est;
pub mod sensitivity;
pub mod shooting;

pub use adjoint::{adjoint_gradient, AdjointResult};
pub use collocation::{CollocationProblem, CollocationResult, CollocationScheme};
pub use error::OcpError;
pub use multiple_shooting::{MultipleShootingProblem, MultipleShootingResult};
pub use param_est::{OdeSolverChoice, ParamEstProblem, ParamEstResult};
pub use sensitivity::{forward_sensitivity, SensitivityResult};
pub use shooting::{ShootingProblem, ShootingResult};

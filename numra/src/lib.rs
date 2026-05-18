//! # Numra - Numerical Methods Library
//!
//! Numra is a comprehensive numerical methods library for Rust.
//!
//! This crate provides a unified interface to all Numra components:
//! - `numra_core`: Core traits and utilities (Scalar, Vector, Signal, Uncertainty)
//! - `numra_linalg`: Linear algebra (Matrix, LU decomposition)
//! - `numra_nonlinear`: Nonlinear solvers (Newton iteration)
//! - `numra_ode`: ODE solvers (DoPri5, Radau5, BDF, ESDIRK, Verner, etc.) and
//!   forward sensitivity analysis (`ParametricOdeSystem`, `solve_forward_sensitivity`)
//! - `numra_sde`: SDE solvers (Euler-Maruyama, Milstein, SRA1/SRA2)
//! - `numra_dde`: DDE solvers (Method of Steps)
//! - `numra_fde`: FDE solvers (L1 scheme for Caputo derivative)
//! - `numra_ide`: IDE solvers (Volterra, Prony series)
//! - `numra_pde`: PDE solvers (Method of Lines, moving boundaries)
//! - `numra_spde`: SPDE solvers (MOL + SDE steppers)
//!
//! ## Sensitivity analysis — two distinct concepts
//!
//! - [`compute_sensitivities`] / [`ParameterSensitivity`] /
//!   [`ParameterSensitivityResult`] (re-exported from `numra-core`):
//!   parameter-uncertainty / Sobol-style sensitivity for arbitrary scalar
//!   functions of uncertain inputs.
//! - [`solve_forward_sensitivity`] / [`solve_forward_sensitivity_with`] /
//!   [`ParametricOdeSystem`] / [`SensitivityResult`] (re-exported from
//!   `numra-ode`): forward sensitivity for ODE/DAE systems. Solves the
//!   augmented `[y; S]` system to compute `S(t) = ∂y/∂p` alongside the
//!   state trajectory.
//!
//! Author: Moussa Leblouba
//! Date: 9 February 2026
//! Modified: 6 May 2026

// Selective re-exports from numra-core to avoid namespace pollution.
pub use numra_autodiff as autodiff;
pub use numra_core::error;
pub use numra_core::scalar;
pub use numra_core::signal;
pub use numra_core::uncertainty;
pub use numra_core::vector;
pub use numra_core::{
    compute_sensitivities, Interval, LinalgError, NumraError, NumraResult, ParameterSensitivity,
    ParameterSensitivityResult, Scalar, Signal, Uncertain, Vector,
};

pub use numra_dde as dde;
pub use numra_fde as fde;
pub use numra_fft as fft;
pub use numra_fit as fit;
pub use numra_ide as ide;
pub use numra_integrate as integrate;
pub use numra_interp as interp;
pub use numra_linalg as linalg;
pub use numra_nonlinear as nonlinear;
pub use numra_ocp as ocp;
pub use numra_ode as ode;
// Top-level promotion of the canonical forward-sensitivity surface so users
// can write `numra::solve_forward_sensitivity` symmetrically with
// `numra::compute_sensitivities`. Advanced building blocks (`AugmentedSystem`,
// `ClosureSystem`) remain in `numra::ode` for callers who need them.
pub use numra_ode::sensitivity::{
    solve_forward_sensitivity, solve_forward_sensitivity_with, solve_initial_condition_sensitivity,
    solve_initial_condition_sensitivity_with,
};
pub use numra_ode::{ParametricOdeSystem, SensitivityResult, StateTransitionResult};
pub use numra_optim as optim;
pub use numra_pde as pde;
pub use numra_sde as sde;
pub use numra_signal as dsp;
pub use numra_spde as spde;
pub use numra_special as special;
pub use numra_stats as stats;

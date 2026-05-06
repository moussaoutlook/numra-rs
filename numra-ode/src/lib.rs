// Allow some clippy lints that are prevalent in numerical code
#![allow(clippy::assign_op_pattern)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::manual_memcpy)]
#![allow(clippy::collapsible_else_if)]
#![allow(clippy::single_match)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::excessive_precision)]
#![allow(clippy::manual_clamp)]
#![allow(clippy::identity_op)]
#![allow(clippy::erasing_op)]

//! # Numra ODE Solvers
//!
//! This crate provides a comprehensive suite of ODE solvers for solving
//! initial value problems of the form:
//!
//! ```text
//! dy/dt = f(t, y),  y(t0) = y0
//! ```
//!
//! ## Available Solvers
//!
//! ### Explicit Runge-Kutta (Non-stiff)
//! - [`DoPri5`] - Dormand-Prince 5(4) with dense output
//! - [`Tsit5`] - Tsitouras 5(4), efficient with FSAL
//! - [`Vern6`] - Verner 6(5), high accuracy
//! - [`Vern7`] - Verner 7(6), higher accuracy
//! - [`Vern8`] - Verner 8(7), very high accuracy
//!
//! ### Implicit Runge-Kutta (Stiff)
//! - [`Radau5`] - Radau IIA 3-stage, L-stable (5th order)
//! - [`Esdirk32`] - ESDIRK 3-stage, 2nd order
//! - [`Esdirk43`] - ESDIRK 4-stage, 3rd order
//! - [`Esdirk54`] - ESDIRK 5-stage, 4th order (L-stable)
//!
//! ### Multistep Methods (Stiff)
//! - [`Bdf`] - BDF orders 1-5 with variable order
//!
//! ### Automatic Selection
//! - [`Auto`] - Automatic solver selection based on problem characteristics
//!
//! ## Example
//!
//! ```rust
//! use numra_ode::{OdeProblem, DoPri5, Solver, SolverOptions};
//!
//! // Define the Lorenz system
//! fn lorenz(t: f64, y: &[f64], dydt: &mut [f64]) {
//!     let sigma = 10.0;
//!     let rho = 28.0;
//!     let beta = 8.0 / 3.0;
//!     dydt[0] = sigma * (y[1] - y[0]);
//!     dydt[1] = y[0] * (rho - y[2]) - y[1];
//!     dydt[2] = y[0] * y[1] - beta * y[2];
//! }
//!
//! // Create problem
//! let y0 = vec![1.0, 1.0, 1.0];
//! let problem = OdeProblem::new(lorenz, 0.0, 20.0, y0.clone());
//!
//! // Solve with DoPri5
//! let options = SolverOptions::default();
//! let result = DoPri5::solve(&problem, 0.0, 20.0, &y0, &options).unwrap();
//!
//! println!("Final state: {:?}", result.y_final());
//! ```
//!
//! Author: Moussa Leblouba
//! Date: 9 February 2026
//! Modified: 2 May 2026

pub mod auto;
pub mod bdf;
pub mod dae_init;
pub mod dense;
pub mod dopri5;
pub mod error;
pub mod esdirk;
pub mod events;
pub mod index_reduction;
pub mod problem;
pub mod radau5;
pub mod sensitivity;
pub mod solver;
pub mod step_control;
pub mod tsit5;
pub mod uncertainty;
pub mod verner;

pub use dense::DenseOutput;
pub use error::SolverError;
pub use problem::{DaeProblem, OdeProblem, OdeSystem};
pub use solver::{Solver, SolverOptions, SolverResult, SolverStats};
pub use step_control::{PIController, StepController};

// Explicit methods
pub use dopri5::DoPri5;
pub use tsit5::Tsit5;
pub use verner::{Vern6, Vern7, Vern8};

// Implicit methods
pub use bdf::Bdf;
pub use esdirk::{Esdirk32, Esdirk43, Esdirk54};
pub use radau5::Radau5;

// Automatic selection
pub use auto::{auto_solve, auto_solve_with_hints, Accuracy, Auto, SolverHints, Stiffness};

pub use sensitivity::{
    solve_forward_sensitivity, solve_forward_sensitivity_with, AugmentedSystem, ClosureSystem,
    ParametricOdeSystem, SensitivityResult,
};

pub use dae_init::{compute_consistent_initial, compute_consistent_initial_tol};

pub use index_reduction::{
    analyze_dae_index, analyze_system, detect_structure, reduce_dae_problem, reduce_index,
    DaeIndexInfo, DaeStructure, ReducedDaeSystem,
};

pub use uncertainty::{
    solve_monte_carlo, solve_trajectory, solve_with_uncertainty, UncertainParam,
    UncertainSolverResult, UncertaintyMode,
};

pub use numra_core::{Scalar, Vector};

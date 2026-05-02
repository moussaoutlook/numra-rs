// Allow some clippy lints that are prevalent in numerical code
#![allow(clippy::assign_op_pattern)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::manual_range_contains)]
#![allow(clippy::identity_op)]

//! PDE (Partial Differential Equation) solvers for Numra.
//!
//! This crate provides the Method of Lines (MOL) approach for solving PDEs:
//!
//! 1. Discretize spatial derivatives using finite differences (FDM)
//! 2. Convert the PDE to a system of ODEs in time
//! 3. Solve using ODE solvers from `numra-ode`
//!
//! # Supported Equation Types
//!
//! - **Parabolic PDEs**: Heat equation, diffusion equations
//! - **Moving boundary problems**: Stefan problem, phase change
//! - **Reaction-diffusion**: Turing patterns, combustion
//!
//! # Example: 1D Heat Equation
//!
//! ```
//! use numra_pde::{Grid1D, HeatEquation1D, MOLSystem};
//! use numra_pde::boundary::DirichletBC;
//! use numra_ode::{DoPri5, Solver, SolverOptions};
//!
//! // Create spatial grid
//! let grid = Grid1D::uniform(0.0_f64, 1.0, 51);
//!
//! // Heat equation with diffusivity α = 0.01
//! let pde = HeatEquation1D::new(0.01);
//!
//! // Boundary conditions: T(0) = 1, T(1) = 0
//! let bc_left = DirichletBC::new(1.0);
//! let bc_right = DirichletBC::new(0.0);
//!
//! // Create MOL system
//! let mol = MOLSystem::new(pde, grid.clone(), bc_left, bc_right);
//!
//! // Initial condition: T(x,0) = sin(πx)
//! let u0: Vec<f64> = grid.points()
//!     .iter()
//!     .map(|&x| (std::f64::consts::PI * x).sin())
//!     .collect();
//!
//! // Solve (interior points only)
//! let options = SolverOptions::default().rtol(1e-6);
//! let result = DoPri5::solve(&mol, 0.0, 0.5, &u0[1..u0.len()-1], &options);
//! ```
//!
//! # Moving Boundary Problems
//!
//! For problems like the Stefan problem where the domain boundary moves:
//!
//! ```text
//! ∂T/∂t = α ∂²T/∂x²,  0 < x < s(t)
//! T(0,t) = T_hot
//! T(s(t),t) = T_melt
//! ds/dt = -k ∂T/∂x |_{x=s(t)}  (Stefan condition)
//! ```
//!
//! Use the `moving` module with coordinate transforms.
//!
//! Author: Moussa Leblouba
//! Date: 9 February 2026
//! Modified: 2 May 2026

pub use numra_core::Scalar;

pub mod boundary;
pub mod boundary2d;
mod discretize;
mod equations;
mod equations2d;
mod grid;
mod mol;
mod mol2d;
pub mod moving;
mod sparse_assembly;

pub use boundary::{BoundaryCondition, DirichletBC, NeumannBC, PeriodicBC, RobinBC};
pub use boundary2d::{BoundaryConditions2D, BoundaryConditions3D};
pub use discretize::{DifferenceScheme, Stencil, FDM};
pub use equations::{DiffusionReaction1D, HeatEquation1D};
pub use equations2d::{AdvectionDiffusion2D, HeatEquation2D, ReactionDiffusion2D};
pub use grid::{Grid1D, Grid2D, Grid3D};
pub use mol::{MOLSystem, PdeSystem};
pub use mol2d::MOLSystem2D;
pub use moving::{Bound, CoordinateTransform, Domain1D, MovingBound, StefanCondition};
pub use sparse_assembly::{
    assemble_laplacian_2d, assemble_laplacian_3d, assemble_operator_2d, Operator2DCoefficients,
    SparseScalar,
};

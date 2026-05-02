//! Optimization algorithms for Numra.
//!
//! # Algorithms
//!
//! - [`Bfgs`] -- BFGS quasi-Newton method (unconstrained)
//! - [`Lbfgs`] -- L-BFGS limited-memory quasi-Newton method (unconstrained)
//! - [`lm_minimize`] -- Levenberg-Marquardt nonlinear least squares
//! - [`lbfgsb_minimize`] -- L-BFGS-B bound-constrained optimizer
//! - [`augmented_lagrangian_minimize`] -- Augmented Lagrangian constrained optimizer
//! - [`OptimProblem`] -- Declarative problem builder with auto solver selection
//!
//! Author: Moussa Leblouba
//! Date: 5 March 2026
//! Modified: 2 May 2026

pub mod augmented_lagrangian;
pub mod auto;
pub mod bfgs;
pub mod cmaes;
pub mod derivative_free;
pub mod error;
pub mod global;
pub mod lbfgs;
pub mod lbfgsb;
pub mod levenberg_marquardt;
pub mod lp;
pub mod milp;
pub mod multiobjective;
pub mod optim_sensitivity;
pub mod problem;
pub mod qp;
pub mod robust;
pub mod sqp;
pub mod stochastic;
pub mod types;

pub use augmented_lagrangian::{augmented_lagrangian_minimize, AugLagOptions};
pub use auto::SolverChoice;
pub use bfgs::{bfgs_minimize, Bfgs};
pub use cmaes::{cmaes_minimize, CmaEsOptions};
pub use derivative_free::{nelder_mead, powell, NelderMeadOptions, PowellOptions};
pub use error::OptimError;
pub use global::{de_minimize, DEOptions};
pub use lbfgs::{lbfgs_minimize, Lbfgs, LbfgsOptions};
pub use lbfgsb::{lbfgsb_minimize, LbfgsBOptions};
pub use levenberg_marquardt::{lm_minimize, LmOptions};
pub use lp::{simplex_solve, LPOptions};
pub use milp::{milp_solve, MILPOptions};
pub use multiobjective::{nsga2_optimize, NsgaIIOptions};
pub use optim_sensitivity::compute_param_sensitivity;
pub use problem::{
    Constraint, ConstraintKind, LinearConstraint, ObjectiveKind, OptimProblem, ProblemHint, VarType,
};
pub use qp::{active_set_qp_solve, QPOptions};
pub use robust::{RobustOptions, RobustProblem, RobustResult, UncertainParam};
pub use sqp::{sqp_minimize, SqpOptions};
pub use stochastic::{StochasticOptions, StochasticParam, StochasticProblem, StochasticResult};
pub use types::{
    IterationRecord, OptimOptions, OptimResult, OptimStatus, ParamSensitivity, ParetoPoint,
    ParetoResult,
};

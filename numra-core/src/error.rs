//! Error types for Numra.
//!
//! This module defines the error types used throughout the Numra library.
//!
//! Author: Moussa Leblouba
//! Date: 9 February 2026
//! Modified: 2 May 2026

use core::fmt;

/// Result type alias for Numra operations.
pub type NumraResult<T> = Result<T, NumraError>;

/// Errors that can occur in Numra operations.
///
/// `NumraError` is the workspace-wide error type into which every fallible
/// crate's error converts via a `From<…> for NumraError` impl. This lets a
/// user write `solve_ode(...)?.integrate(...)?.fit(...)?` and have `?` work
/// uniformly across crate boundaries.
///
/// Variants fall into two tiers:
///
/// - **Low-level numerics-failure modes from `numra-core`** (structurally
///   preserved): [`Linalg`], [`Convergence`], [`NumericalOptim`].
/// - **Crate-level user-facing errors** (stringified, tagged by source crate):
///   [`Ode`], [`Optim`], [`Ocp`], [`Fit`], [`Signal`], [`LineSearch`],
///   [`Interp`], [`Integrate`], [`Special`], [`Stats`].
///
/// The [`NumericalOptim`] variant wraps `numra-core`'s
/// [`OptimizationError`] (line-search / descent / convergence failures that
/// happen *inside* algorithms regardless of which crate hosts them); the
/// [`Optim`] variant carries `numra-optim::OptimError` (the optimization
/// crate's API-level failures: missing objective, infeasibility, unbounded,
/// etc.). The two are distinct concerns; do not confuse them.
///
/// `NumraError` is `#[non_exhaustive]`; new variants may be added in minor
/// releases. Match arms must include a `_ => …` catch-all.
///
/// [`Linalg`]: NumraError::Linalg
/// [`Convergence`]: NumraError::Convergence
/// [`NumericalOptim`]: NumraError::NumericalOptim
/// [`Ode`]: NumraError::Ode
/// [`Optim`]: NumraError::Optim
/// [`Ocp`]: NumraError::Ocp
/// [`Fit`]: NumraError::Fit
/// [`Signal`]: NumraError::Signal
/// [`LineSearch`]: NumraError::LineSearch
/// [`Interp`]: NumraError::Interp
/// [`Integrate`]: NumraError::Integrate
/// [`Special`]: NumraError::Special
/// [`Stats`]: NumraError::Stats
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum NumraError {
    /// Linear algebra error from `numra-linalg` / `numra-core`.
    Linalg(LinalgError),
    /// Convergence failure from a Newton or fixed-point iteration.
    Convergence(ConvergenceError),
    /// Invalid input that doesn't fit a more specific variant.
    InvalidInput(String),
    /// Step size became too small.
    StepSizeTooSmall { h: f64, h_min: f64 },
    /// Maximum iterations exceeded.
    MaxIterations { iterations: usize, max: usize },
    /// Integration reached a singularity or stiffness.
    Stiffness { t: f64, message: String },
    /// Event caused termination.
    EventTermination { t: f64, event_index: usize },
    /// Numerical-optimization failure from inside an algorithm
    /// (line search, descent, convergence). See [`OptimizationError`].
    NumericalOptim(OptimizationError),
    /// ODE solver error from `numra-ode`.
    Ode(String),
    /// Optimization error from `numra-optim` (API-level failures).
    Optim(String),
    /// ODE-constrained-optimization error from `numra-ocp`.
    Ocp(String),
    /// Curve-fitting error from `numra-fit`.
    Fit(String),
    /// Signal-processing error from `numra-signal`.
    Signal(String),
    /// Line-search error from `numra-nonlinear`.
    LineSearch(String),
    /// Interpolation error from `numra-interp`.
    Interp(String),
    /// Numerical-integration (quadrature) error from `numra-integrate`.
    Integrate(String),
    /// Special-function evaluation error from `numra-special`.
    Special(String),
    /// Statistics error from `numra-stats`.
    Stats(String),
}

impl fmt::Display for NumraError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            NumraError::Linalg(e) => write!(f, "Linear algebra error: {}", e),
            NumraError::Convergence(e) => write!(f, "Convergence error: {}", e),
            NumraError::InvalidInput(msg) => write!(f, "Invalid input: {}", msg),
            NumraError::StepSizeTooSmall { h, h_min } => {
                write!(f, "Step size {} below minimum {}", h, h_min)
            }
            NumraError::MaxIterations { iterations, max } => {
                write!(f, "Maximum iterations ({}) exceeded at {}", max, iterations)
            }
            NumraError::Stiffness { t, message } => {
                write!(f, "Stiffness detected at t={}: {}", t, message)
            }
            NumraError::EventTermination { t, event_index } => {
                write!(f, "Event {} terminated integration at t={}", event_index, t)
            }
            NumraError::NumericalOptim(e) => write!(f, "Numerical optimization error: {}", e),
            NumraError::Ode(s) => write!(f, "ODE error: {}", s),
            NumraError::Optim(s) => write!(f, "Optimization error: {}", s),
            NumraError::Ocp(s) => write!(f, "OCP error: {}", s),
            NumraError::Fit(s) => write!(f, "Fit error: {}", s),
            NumraError::Signal(s) => write!(f, "Signal error: {}", s),
            NumraError::LineSearch(s) => write!(f, "Line search error: {}", s),
            NumraError::Interp(s) => write!(f, "Interpolation error: {}", s),
            NumraError::Integrate(s) => write!(f, "Integration error: {}", s),
            NumraError::Special(s) => write!(f, "Special function error: {}", s),
            NumraError::Stats(s) => write!(f, "Statistics error: {}", s),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for NumraError {}

/// Linear algebra specific errors.
#[derive(Clone, Debug, PartialEq)]
pub enum LinalgError {
    /// Matrix is singular
    Singular { step: usize },
    /// Dimensions mismatch
    DimensionMismatch {
        expected: (usize, usize),
        actual: (usize, usize),
    },
    /// Matrix is not square
    NotSquare { nrows: usize, ncols: usize },
    /// Matrix is not positive definite
    NotPositiveDefinite,
    /// Iterative solver did not converge
    IterativeNotConverged { iterations: usize, residual: f64 },
    /// Eigenvalue decomposition failed
    EigenDecompositionFailed,
    /// SVD computation failed
    SvdFailed,
}

impl fmt::Display for LinalgError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LinalgError::Singular { step } => {
                write!(f, "Matrix is singular (detected at step {})", step)
            }
            LinalgError::DimensionMismatch { expected, actual } => {
                write!(
                    f,
                    "Dimension mismatch: expected {:?}, got {:?}",
                    expected, actual
                )
            }
            LinalgError::NotSquare { nrows, ncols } => {
                write!(f, "Matrix is not square: {}x{}", nrows, ncols)
            }
            LinalgError::NotPositiveDefinite => {
                write!(f, "Matrix is not positive definite")
            }
            LinalgError::IterativeNotConverged {
                iterations,
                residual,
            } => {
                write!(
                    f,
                    "Iterative solver did not converge after {} iterations (residual: {})",
                    iterations, residual
                )
            }
            LinalgError::EigenDecompositionFailed => {
                write!(f, "Eigenvalue decomposition failed")
            }
            LinalgError::SvdFailed => {
                write!(f, "SVD computation failed")
            }
        }
    }
}

impl From<LinalgError> for NumraError {
    fn from(e: LinalgError) -> Self {
        NumraError::Linalg(e)
    }
}

/// Convergence-related errors.
#[derive(Clone, Debug, PartialEq)]
pub enum ConvergenceError {
    /// Newton iteration did not converge
    Newton { iterations: usize, residual: f64 },
    /// Fixed point iteration did not converge
    FixedPoint { iterations: usize, delta: f64 },
}

impl fmt::Display for ConvergenceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ConvergenceError::Newton {
                iterations,
                residual,
            } => {
                write!(
                    f,
                    "Newton iteration did not converge after {} iterations (residual: {})",
                    iterations, residual
                )
            }
            ConvergenceError::FixedPoint { iterations, delta } => {
                write!(
                    f,
                    "Fixed point iteration did not converge after {} iterations (delta: {})",
                    iterations, delta
                )
            }
        }
    }
}

impl From<ConvergenceError> for NumraError {
    fn from(e: ConvergenceError) -> Self {
        NumraError::Convergence(e)
    }
}

/// Optimization-related errors.
#[derive(Clone, Debug, PartialEq)]
pub enum OptimizationError {
    /// Line search failed to find sufficient decrease
    LineSearchFailed { iterations: usize, step_size: f64 },
    /// Search direction is not a descent direction
    NotDescentDirection { directional_derivative: f64 },
    /// Optimization did not converge
    NotConverged {
        iterations: usize,
        gradient_norm: f64,
    },
    /// Function evaluation returned NaN or infinity
    InvalidFunctionValue { value: f64 },
}

impl fmt::Display for OptimizationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OptimizationError::LineSearchFailed {
                iterations,
                step_size,
            } => {
                write!(
                    f,
                    "Line search failed after {} iterations (step size: {})",
                    iterations, step_size
                )
            }
            OptimizationError::NotDescentDirection {
                directional_derivative,
            } => {
                write!(
                    f,
                    "Search direction is not a descent direction (directional derivative: {})",
                    directional_derivative
                )
            }
            OptimizationError::NotConverged {
                iterations,
                gradient_norm,
            } => {
                write!(
                    f,
                    "Optimization did not converge after {} iterations (gradient norm: {})",
                    iterations, gradient_norm
                )
            }
            OptimizationError::InvalidFunctionValue { value } => {
                write!(f, "Function evaluation returned invalid value: {}", value)
            }
        }
    }
}

impl From<OptimizationError> for NumraError {
    fn from(e: OptimizationError) -> Self {
        NumraError::NumericalOptim(e)
    }
}

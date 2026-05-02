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
#[derive(Clone, Debug, PartialEq)]
pub enum NumraError {
    /// Linear algebra error
    Linalg(LinalgError),
    /// Convergence failure
    Convergence(ConvergenceError),
    /// Invalid input
    InvalidInput(String),
    /// Step size became too small
    StepSizeTooSmall { h: f64, h_min: f64 },
    /// Maximum iterations exceeded
    MaxIterations { iterations: usize, max: usize },
    /// Integration reached a singularity or stiffness
    Stiffness { t: f64, message: String },
    /// Event caused termination
    EventTermination { t: f64, event_index: usize },
    /// Optimization error
    Optimization(OptimizationError),
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
            NumraError::Optimization(e) => write!(f, "Optimization error: {}", e),
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
        NumraError::Optimization(e)
    }
}

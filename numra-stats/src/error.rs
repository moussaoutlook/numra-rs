//! Error types for statistical computations.
//!
//! Author: Moussa Leblouba
//! Date: 9 February 2026
//! Modified: 2 May 2026

use core::fmt;

/// Errors that can occur during statistical computations.
#[derive(Clone, Debug, PartialEq)]
pub enum StatsError {
    /// Empty data set.
    EmptyData,
    /// Data sets have mismatched lengths.
    LengthMismatch { expected: usize, got: usize },
    /// Invalid parameter value.
    InvalidParameter(String),
    /// Dimension mismatch.
    DimensionMismatch(String),
    /// Singular matrix encountered.
    SingularMatrix,
    /// Computation did not converge.
    ConvergenceFailure(String),
}

impl fmt::Display for StatsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            StatsError::EmptyData => write!(f, "empty data set"),
            StatsError::LengthMismatch { expected, got } => {
                write!(f, "length mismatch: expected {}, got {}", expected, got)
            }
            StatsError::InvalidParameter(msg) => write!(f, "invalid parameter: {}", msg),
            StatsError::DimensionMismatch(msg) => write!(f, "dimension mismatch: {}", msg),
            StatsError::SingularMatrix => write!(f, "singular matrix"),
            StatsError::ConvergenceFailure(msg) => write!(f, "convergence failure: {}", msg),
        }
    }
}

impl From<StatsError> for numra_core::NumraError {
    fn from(e: StatsError) -> Self {
        numra_core::NumraError::Stats(e.to_string())
    }
}

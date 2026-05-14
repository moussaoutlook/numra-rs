//! Error types for curve fitting.
//!
//! Author: Moussa Leblouba
//! Date: 9 February 2026
//! Modified: 2 May 2026

use core::fmt;
use numra_core::NumraError;

/// Error type for curve fitting operations.
#[derive(Clone, Debug, PartialEq)]
pub enum FitError {
    /// Not enough data points.
    InsufficientData { need: usize, got: usize },
    /// More parameters than data points.
    Underdetermined { n_params: usize, n_data: usize },
    /// Optimization failed to converge.
    OptimizationFailed(String),
    /// Dimension mismatch between x and y data.
    LengthMismatch { expected: usize, got: usize },
    /// Invalid parameter (e.g., negative polynomial degree).
    InvalidParameter(String),
    /// Singular or near-singular Jacobian (covariance unreliable).
    SingularJacobian,
}

impl fmt::Display for FitError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InsufficientData { need, got } => {
                write!(f, "insufficient data: need at least {need}, got {got}")
            }
            Self::Underdetermined { n_params, n_data } => {
                write!(
                    f,
                    "underdetermined: {n_params} parameters but only {n_data} data points"
                )
            }
            Self::OptimizationFailed(msg) => write!(f, "optimization failed: {msg}"),
            Self::LengthMismatch { expected, got } => {
                write!(f, "length mismatch: expected {expected}, got {got}")
            }
            Self::InvalidParameter(msg) => write!(f, "invalid parameter: {msg}"),
            Self::SingularJacobian => write!(f, "singular Jacobian: covariance unreliable"),
        }
    }
}

impl std::error::Error for FitError {}

impl From<FitError> for NumraError {
    fn from(e: FitError) -> Self {
        NumraError::Fit(e.to_string())
    }
}

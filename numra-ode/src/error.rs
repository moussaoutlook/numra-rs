//! Error types for ODE solvers.
//!
//! Author: Moussa Leblouba
//! Date: 8 February 2026
//! Modified: 2 May 2026

use numra_core::LinalgError;
use thiserror::Error;

/// Errors that can occur during ODE solving.
#[derive(Debug, Error)]
pub enum SolverError {
    #[error("dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },

    #[error("step size too small at t = {t}: h = {h} < h_min = {h_min}")]
    StepSizeTooSmall { t: f64, h: f64, h_min: f64 },

    #[error("maximum iterations exceeded at t = {t}")]
    MaxIterationsExceeded { t: f64 },

    #[error("Newton iteration failed to converge at t = {t}")]
    NewtonConvergenceFailed { t: f64 },

    #[error("singular mass matrix detected")]
    SingularMassMatrix,

    #[error("inconsistent initial conditions for DAE")]
    InconsistentInitialConditions,

    #[error("LU factorization failed")]
    LuFactorizationFailed,

    #[error("{0}")]
    Other(String),
}

impl From<String> for SolverError {
    fn from(s: String) -> Self {
        SolverError::Other(s)
    }
}

impl From<&str> for SolverError {
    fn from(s: &str) -> Self {
        SolverError::Other(s.to_string())
    }
}

impl From<LinalgError> for SolverError {
    fn from(e: LinalgError) -> Self {
        match e {
            LinalgError::Singular { .. }
            | LinalgError::NotSquare { .. }
            | LinalgError::NotPositiveDefinite => SolverError::LuFactorizationFailed,
            _ => SolverError::Other(e.to_string()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = SolverError::DimensionMismatch {
            expected: 3,
            actual: 5,
        };
        assert!(err.to_string().contains("3"));
        assert!(err.to_string().contains("5"));
    }

    #[test]
    fn test_error_from_string() {
        let err = SolverError::from("custom error");
        matches!(err, SolverError::Other(_));
    }
}

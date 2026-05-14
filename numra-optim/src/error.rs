//! Error types for optimization algorithms.
//!
//! Author: Moussa Leblouba
//! Date: 8 February 2026
//! Modified: 2 May 2026

use numra_core::NumraError;
use thiserror::Error;

/// Errors that can occur during optimization.
#[derive(Debug, Error)]
pub enum OptimError {
    #[error("line search failed at iteration {iteration}: {reason}")]
    LineSearchFailed { iteration: usize, reason: String },
    #[error("not a descent direction (dg = {directional_derivative:.2e})")]
    NotDescentDirection { directional_derivative: f64 },
    #[error("invalid function value: {value}")]
    InvalidFunctionValue { value: f64 },
    #[error("singular matrix in solver")]
    SingularMatrix,
    #[error("dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },
    #[error("no objective function specified")]
    NoObjective,
    #[error("no initial point specified")]
    NoInitialPoint,
    #[error("infeasible: constraint violation = {violation:.2e}")]
    Infeasible { violation: f64 },
    #[error("problem is unbounded")]
    Unbounded,
    #[error("linear program is infeasible")]
    LPInfeasible,
    #[error("QP Hessian is not positive semi-definite")]
    QPNotPositiveSemiDefinite,
    #[error("mixed-integer linear program is infeasible")]
    MILPInfeasible,
    #[error("{0}")]
    Other(String),
}

impl From<String> for OptimError {
    fn from(s: String) -> Self {
        OptimError::Other(s)
    }
}

impl From<&str> for OptimError {
    fn from(s: &str) -> Self {
        OptimError::Other(s.to_string())
    }
}

impl From<numra_nonlinear::LineSearchError> for OptimError {
    fn from(e: numra_nonlinear::LineSearchError) -> Self {
        OptimError::Other(e.to_string())
    }
}

impl From<numra_nonlinear::LinalgError> for OptimError {
    fn from(e: numra_nonlinear::LinalgError) -> Self {
        match e {
            numra_nonlinear::LinalgError::Singular { .. }
            | numra_nonlinear::LinalgError::NotSquare { .. }
            | numra_nonlinear::LinalgError::NotPositiveDefinite => OptimError::SingularMatrix,
            numra_nonlinear::LinalgError::DimensionMismatch { expected, actual } => {
                OptimError::DimensionMismatch {
                    expected: expected.0,
                    actual: actual.0,
                }
            }
            _ => OptimError::Other(e.to_string()),
        }
    }
}

impl From<OptimError> for NumraError {
    fn from(e: OptimError) -> Self {
        NumraError::Optim(e.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_display_formatting() {
        let e = OptimError::LineSearchFailed {
            iteration: 5,
            reason: "step too small".into(),
        };
        assert_eq!(
            e.to_string(),
            "line search failed at iteration 5: step too small"
        );

        let e = OptimError::NotDescentDirection {
            directional_derivative: 1.5,
        };
        assert!(e.to_string().contains("1.50e0"));

        let e = OptimError::SingularMatrix;
        assert_eq!(e.to_string(), "singular matrix in solver");

        let e = OptimError::DimensionMismatch {
            expected: 3,
            actual: 5,
        };
        assert_eq!(e.to_string(), "dimension mismatch: expected 3, got 5");
    }

    #[test]
    fn test_lp_qp_error_variants() {
        let e = OptimError::Unbounded;
        assert_eq!(e.to_string(), "problem is unbounded");

        let e = OptimError::LPInfeasible;
        assert_eq!(e.to_string(), "linear program is infeasible");

        let e = OptimError::QPNotPositiveSemiDefinite;
        assert_eq!(e.to_string(), "QP Hessian is not positive semi-definite");

        let e = OptimError::MILPInfeasible;
        assert_eq!(e.to_string(), "mixed-integer linear program is infeasible");
    }

    #[test]
    fn test_from_string() {
        let e: OptimError = "some error".into();
        assert_eq!(e.to_string(), "some error");

        let e: OptimError = String::from("another error").into();
        assert_eq!(e.to_string(), "another error");
    }
}

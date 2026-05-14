//! Error types for special functions.
//!
//! Author: Moussa Leblouba
//! Date: 9 February 2026
//! Modified: 2 May 2026

use core::fmt;

/// Errors from special function evaluation.
#[derive(Clone, Debug, PartialEq)]
pub enum SpecialError {
    /// Argument is outside the domain of the function.
    Domain(String),
    /// Series did not converge within the allowed iterations.
    Convergence {
        function: &'static str,
        iterations: usize,
    },
}

impl fmt::Display for SpecialError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Domain(msg) => write!(f, "domain error: {msg}"),
            Self::Convergence {
                function,
                iterations,
            } => {
                write!(
                    f,
                    "{function} did not converge after {iterations} iterations"
                )
            }
        }
    }
}

impl From<SpecialError> for numra_core::NumraError {
    fn from(e: SpecialError) -> Self {
        numra_core::NumraError::Special(e.to_string())
    }
}

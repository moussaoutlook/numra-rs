//!
//! Author: Moussa Leblouba
//! Date: 9 February 2026
//! Modified: 2 May 2026

use core::fmt;
use numra_core::NumraError;

/// Errors that can occur during numerical integration.
#[derive(Clone, Debug, PartialEq)]
pub enum IntegrationError {
    /// Maximum subdivisions reached without achieving tolerance.
    MaxSubdivisions {
        subdivisions: usize,
        error_estimate: f64,
    },
    /// Roundoff error detected.
    RoundoffDetected,
    /// Integrand returned NaN or infinity.
    InvalidValue { x: f64 },
    /// Bad input (e.g., a >= b for finite interval).
    InvalidInput(String),
}

impl fmt::Display for IntegrationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            IntegrationError::MaxSubdivisions {
                subdivisions,
                error_estimate,
            } => {
                write!(
                    f,
                    "Maximum subdivisions ({}) reached, error estimate: {:.2e}",
                    subdivisions, error_estimate
                )
            }
            IntegrationError::RoundoffDetected => {
                write!(f, "Roundoff error detected during integration")
            }
            IntegrationError::InvalidValue { x } => {
                write!(f, "Integrand returned invalid value at x = {}", x)
            }
            IntegrationError::InvalidInput(msg) => {
                write!(f, "Invalid input: {}", msg)
            }
        }
    }
}

impl From<IntegrationError> for NumraError {
    fn from(e: IntegrationError) -> Self {
        NumraError::Integrate(e.to_string())
    }
}

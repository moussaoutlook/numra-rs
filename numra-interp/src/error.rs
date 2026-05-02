//!
//! Author: Moussa Leblouba
//! Date: 9 February 2026
//! Modified: 2 May 2026

use core::fmt;
use numra_core::NumraError;

/// Errors that can occur during interpolation.
#[derive(Clone, Debug, PartialEq)]
pub enum InterpError {
    /// Too few data points for the requested method.
    TooFewPoints { got: usize, min: usize },
    /// Data x-coordinates are not strictly increasing.
    UnsortedData,
    /// x and y arrays have different lengths.
    DimensionMismatch { x_len: usize, y_len: usize },
    /// Bad input.
    InvalidInput(String),
}

impl fmt::Display for InterpError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            InterpError::TooFewPoints { got, min } => {
                write!(f, "Too few data points: got {}, need at least {}", got, min)
            }
            InterpError::UnsortedData => {
                write!(f, "Data x-coordinates must be strictly increasing")
            }
            InterpError::DimensionMismatch { x_len, y_len } => {
                write!(
                    f,
                    "Dimension mismatch: x has {} elements, y has {}",
                    x_len, y_len
                )
            }
            InterpError::InvalidInput(msg) => {
                write!(f, "Invalid input: {}", msg)
            }
        }
    }
}

impl From<InterpError> for NumraError {
    fn from(e: InterpError) -> Self {
        NumraError::InvalidInput(e.to_string())
    }
}

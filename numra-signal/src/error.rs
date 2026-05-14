//! Error types for signal processing.
//!
//! Author: Moussa Leblouba
//! Date: 9 February 2026
//! Modified: 2 May 2026

use core::fmt;
use numra_core::NumraError;

/// Errors that can occur in signal processing operations.
#[derive(Clone, Debug, PartialEq)]
pub enum SignalError {
    /// Filter order must be positive.
    InvalidOrder(usize),
    /// Cutoff frequency must be in (0, fs/2).
    InvalidCutoff { cutoff: f64, nyquist: f64 },
    /// Ripple must be positive.
    InvalidRipple(f64),
    /// Input signal is empty or too short.
    InsufficientData { need: usize, got: usize },
    /// Invalid parameter.
    InvalidParameter(String),
}

impl fmt::Display for SignalError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidOrder(n) => write!(f, "invalid filter order: {n}"),
            Self::InvalidCutoff { cutoff, nyquist } => {
                write!(f, "cutoff {cutoff} must be in (0, {nyquist})")
            }
            Self::InvalidRipple(r) => write!(f, "ripple must be positive, got {r}"),
            Self::InsufficientData { need, got } => {
                write!(f, "need at least {need} samples, got {got}")
            }
            Self::InvalidParameter(msg) => write!(f, "invalid parameter: {msg}"),
        }
    }
}

impl std::error::Error for SignalError {}

impl From<SignalError> for NumraError {
    fn from(e: SignalError) -> Self {
        NumraError::Signal(e.to_string())
    }
}

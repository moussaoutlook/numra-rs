//! Error types for ODE-constrained optimization.
//!
//! Author: Moussa Leblouba
//! Date: 8 February 2026
//! Modified: 2 May 2026

use thiserror::Error;

/// Errors from ODE-constrained optimization.
#[derive(Debug, Error)]
pub enum OcpError {
    #[error("ODE integration failed: {0}")]
    IntegrationFailed(String),
    #[error("optimization failed: {0}")]
    OptimFailed(#[from] numra_optim::OptimError),
    #[error("no model specified")]
    NoModel,
    #[error("no data specified")]
    NoData,
    #[error("no initial state specified")]
    NoInitialState,
    #[error("no dynamics specified")]
    NoDynamics,
    #[error("dimension mismatch: {0}")]
    DimensionMismatch(String),
    #[error("{0}")]
    Other(String),
}

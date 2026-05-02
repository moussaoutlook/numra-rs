//! Types for curve fitting results and options.
//!
//! Author: Moussa Leblouba
//! Date: 9 February 2026
//! Modified: 2 May 2026

use numra_core::Scalar;

/// Result of a curve fit.
#[derive(Clone, Debug)]
pub struct FitResult<S: Scalar> {
    /// Optimized parameters.
    pub params: Vec<S>,
    /// Covariance matrix of parameters (row-major, n_params x n_params).
    /// `None` if covariance could not be estimated (singular Jacobian).
    pub covariance: Option<Vec<S>>,
    /// Standard errors of parameters (sqrt of diagonal of covariance).
    /// `None` if covariance could not be estimated.
    pub std_errors: Option<Vec<S>>,
    /// Residuals at the optimized parameters (y_data - model(x, params)).
    pub residuals: Vec<S>,
    /// Sum of squared residuals.
    pub chi_squared: S,
    /// Reduced chi-squared: chi_squared / dof.
    pub reduced_chi_squared: S,
    /// Coefficient of determination (R-squared).
    pub r_squared: S,
    /// Degrees of freedom (n_data - n_params).
    pub dof: usize,
    /// Number of function evaluations.
    pub n_evaluations: usize,
    /// Whether the optimizer converged.
    pub converged: bool,
}

/// Options for curve fitting.
#[derive(Clone, Debug)]
pub struct FitOptions<S: Scalar> {
    /// Maximum iterations.
    pub max_iter: usize,
    /// Function tolerance.
    pub ftol: S,
    /// Gradient tolerance.
    pub gtol: S,
    /// Step tolerance.
    pub xtol: S,
}

impl<S: Scalar> Default for FitOptions<S> {
    fn default() -> Self {
        Self {
            max_iter: 200,
            ftol: S::from_f64(1e-12),
            gtol: S::from_f64(1e-10),
            xtol: S::from_f64(1e-10),
        }
    }
}

impl<S: Scalar> FitOptions<S> {
    /// Set maximum iterations.
    pub fn max_iter(mut self, n: usize) -> Self {
        self.max_iter = n;
        self
    }

    /// Set function tolerance.
    pub fn ftol(mut self, v: S) -> Self {
        self.ftol = v;
        self
    }

    /// Set gradient tolerance.
    pub fn gtol(mut self, v: S) -> Self {
        self.gtol = v;
        self
    }

    /// Set step tolerance.
    pub fn xtol(mut self, v: S) -> Self {
        self.xtol = v;
        self
    }
}

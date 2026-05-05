//! ODE solver infrastructure.
//!
//! This module defines the common traits and types for ODE solvers.
//!
//! Author: Moussa Leblouba
//! Date: 30 April 2026
//! Modified: 2 May 2026

use crate::dense::DenseOutput;
use crate::error::SolverError;
use crate::events::{Event, EventFunction};
use crate::problem::OdeSystem;
use core::fmt;
use numra_core::Scalar;
use std::sync::Arc;

/// Solver options and tolerances.
///
/// Cloneable thanks to `Arc`-wrapped event functions.
pub struct SolverOptions<S: Scalar> {
    /// Relative tolerance
    pub rtol: S,
    /// Absolute tolerance (scalar)
    pub atol: S,
    /// Initial step size (None = auto)
    pub h0: Option<S>,
    /// Maximum step size
    pub h_max: S,
    /// Minimum step size
    pub h_min: S,
    /// Maximum number of steps
    pub max_steps: usize,
    /// Save solution at these times (None = save all steps)
    pub t_eval: Option<Vec<S>>,
    /// Enable dense output
    pub dense_output: bool,
    /// Event functions for zero-crossing detection (Arc enables Clone)
    pub events: Vec<Arc<dyn EventFunction<S>>>,
}

impl<S: Scalar> Clone for SolverOptions<S> {
    fn clone(&self) -> Self {
        Self {
            rtol: self.rtol,
            atol: self.atol,
            h0: self.h0,
            h_max: self.h_max,
            h_min: self.h_min,
            max_steps: self.max_steps,
            t_eval: self.t_eval.clone(),
            dense_output: self.dense_output,
            events: self.events.clone(),
        }
    }
}

impl<S: Scalar> fmt::Debug for SolverOptions<S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SolverOptions")
            .field("rtol", &self.rtol)
            .field("atol", &self.atol)
            .field("h0", &self.h0)
            .field("h_max", &self.h_max)
            .field("h_min", &self.h_min)
            .field("max_steps", &self.max_steps)
            .field("t_eval", &self.t_eval)
            .field("dense_output", &self.dense_output)
            .field("events", &format!("[{} event(s)]", self.events.len()))
            .finish()
    }
}

impl<S: Scalar> Default for SolverOptions<S> {
    fn default() -> Self {
        Self {
            rtol: S::from_f64(1e-6),
            atol: S::from_f64(1e-9),
            h0: None,
            h_max: S::INFINITY,
            // Scale h_min with machine epsilon to support both f32 and f64.
            // f64: 100 * EPSILON ~ 2.2e-14 (close to previous fixed 1e-14)
            // f32: 100 * EPSILON ~ 1.2e-5  (meaningful for f32 precision)
            // A fixed 1e-14 was below f32 machine epsilon (~1.2e-7), making it useless.
            h_min: S::EPSILON * S::from_f64(100.0),
            max_steps: 100_000,
            t_eval: None,
            dense_output: false,
            events: Vec::new(),
        }
    }
}

impl<S: Scalar> SolverOptions<S> {
    /// Set relative tolerance.
    pub fn rtol(mut self, rtol: S) -> Self {
        self.rtol = rtol;
        self
    }

    /// Set absolute tolerance.
    pub fn atol(mut self, atol: S) -> Self {
        self.atol = atol;
        self
    }

    /// Set initial step size.
    pub fn h0(mut self, h0: S) -> Self {
        self.h0 = Some(h0);
        self
    }

    /// Set maximum step size.
    pub fn h_max(mut self, h_max: S) -> Self {
        self.h_max = h_max;
        self
    }

    /// Set evaluation times.
    pub fn t_eval(mut self, t_eval: Vec<S>) -> Self {
        self.t_eval = Some(t_eval);
        self
    }

    /// Enable dense output.
    pub fn dense(mut self) -> Self {
        self.dense_output = true;
        self
    }

    /// Set maximum number of steps.
    pub fn max_steps(mut self, max_steps: usize) -> Self {
        self.max_steps = max_steps;
        self
    }

    /// Set minimum step size.
    pub fn h_min(mut self, h_min: S) -> Self {
        self.h_min = h_min;
        self
    }

    /// Add an event function for zero-crossing detection.
    ///
    /// Internally converts to `Arc` to enable `Clone` on `SolverOptions`.
    pub fn event(mut self, event: Box<dyn EventFunction<S>>) -> Self {
        self.events.push(Arc::from(event));
        self
    }
}

/// Solver statistics.
#[derive(Clone, Debug, Default)]
pub struct SolverStats {
    /// Number of function evaluations
    pub n_eval: usize,
    /// Number of Jacobian evaluations
    pub n_jac: usize,
    /// Number of accepted steps
    pub n_accept: usize,
    /// Number of rejected steps
    pub n_reject: usize,
    /// Number of LU decompositions (for implicit methods)
    pub n_lu: usize,
}

impl SolverStats {
    pub fn new() -> Self {
        Self::default()
    }
}

/// Result of ODE integration.
#[derive(Clone, Debug)]
pub struct SolverResult<S: Scalar> {
    /// Time points
    pub t: Vec<S>,
    /// Solution at each time point (row-major: y[i*dim + j] = y_j(t_i))
    pub y: Vec<S>,
    /// Dimension of the system
    pub dim: usize,
    /// Solver statistics
    pub stats: SolverStats,
    /// Was integration successful?
    pub success: bool,
    /// Message (error description if failed)
    pub message: String,
    /// Detected events during integration
    pub events: Vec<Event<S>>,
    /// Whether integration was terminated by a Stop event
    pub terminated_by_event: bool,
    /// Dense output for continuous interpolation (populated when `SolverOptions::dense()` was set).
    pub dense_output: Option<DenseOutput<S>>,
}

impl<S: Scalar> SolverResult<S> {
    /// Create a new successful result.
    pub fn new(t: Vec<S>, y: Vec<S>, dim: usize, stats: SolverStats) -> Self {
        Self {
            t,
            y,
            dim,
            stats,
            success: true,
            message: String::new(),
            events: Vec::new(),
            terminated_by_event: false,
            dense_output: None,
        }
    }

    /// Create a failed result.
    pub fn failed(message: String, stats: SolverStats) -> Self {
        Self {
            t: Vec::new(),
            y: Vec::new(),
            dim: 0,
            stats,
            success: false,
            message,
            events: Vec::new(),
            terminated_by_event: false,
            dense_output: None,
        }
    }

    /// Number of time points.
    pub fn len(&self) -> usize {
        self.t.len()
    }

    /// Is result empty?
    pub fn is_empty(&self) -> bool {
        self.t.is_empty()
    }

    /// Get final time.
    pub fn t_final(&self) -> Option<S> {
        self.t.last().copied()
    }

    /// Get final state.
    pub fn y_final(&self) -> Option<Vec<S>> {
        if self.t.is_empty() {
            None
        } else {
            let start = (self.t.len() - 1) * self.dim;
            Some(self.y[start..start + self.dim].to_vec())
        }
    }

    /// Get state at index i.
    pub fn y_at(&self, i: usize) -> &[S] {
        let start = i * self.dim;
        &self.y[start..start + self.dim]
    }

    /// Number of time steps in the solution.
    pub fn n_steps(&self) -> usize {
        self.y.len().checked_div(self.dim).unwrap_or(0)
    }

    /// Extract the j-th state variable as a time series.
    ///
    /// Returns `Some(Vec<S>)` containing `y_j(t_0), y_j(t_1), ..., y_j(t_N)`,
    /// or `None` if `j >= self.dim`.
    /// Useful for feeding a single component into FFT, statistics, or plotting.
    pub fn component(&self, j: usize) -> Option<Vec<S>> {
        if j >= self.dim {
            return None;
        }
        Some(
            (0..self.n_steps())
                .map(|i| self.y[i * self.dim + j])
                .collect(),
        )
    }

    /// Iterate over (t, y) pairs.
    pub fn iter(&self) -> impl Iterator<Item = (S, &[S])> {
        self.t
            .iter()
            .enumerate()
            .map(move |(i, &t)| (t, self.y_at(i)))
    }
}

/// Trait for ODE solvers.
pub trait Solver<S: Scalar> {
    /// Solve the ODE problem.
    fn solve<Sys: OdeSystem<S>>(
        problem: &Sys,
        t0: S,
        tf: S,
        y0: &[S],
        options: &SolverOptions<S>,
    ) -> Result<SolverResult<S>, SolverError>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_solver_options_default() {
        let opts: SolverOptions<f64> = SolverOptions::default();
        assert!((opts.rtol - 1e-6).abs() < 1e-10);
        assert!((opts.atol - 1e-9).abs() < 1e-15);
    }

    #[test]
    fn test_solver_options_builder() {
        let opts: SolverOptions<f64> = SolverOptions::default().rtol(1e-8).atol(1e-10).h0(0.01);
        assert!((opts.rtol - 1e-8).abs() < 1e-15);
        assert!((opts.atol - 1e-10).abs() < 1e-15);
        assert!((opts.h0.unwrap() - 0.01).abs() < 1e-15);
    }

    #[test]
    fn test_solver_result() {
        let t = vec![0.0, 0.5, 1.0];
        let y = vec![1.0, 2.0, 0.5, 1.5, 0.2, 1.0]; // 2D system
        let result = SolverResult::new(t, y, 2, SolverStats::new());

        assert_eq!(result.len(), 3);
        assert!((result.t_final().unwrap() - 1.0).abs() < 1e-10);

        let y_final = result.y_final().unwrap();
        assert!((y_final[0] - 0.2).abs() < 1e-10);
        assert!((y_final[1] - 1.0).abs() < 1e-10);

        assert_eq!(result.y_at(0), &[1.0, 2.0]);
        assert_eq!(result.y_at(1), &[0.5, 1.5]);
    }

    #[test]
    fn test_n_steps() {
        let t = vec![0.0, 0.5, 1.0];
        let y = vec![1.0, 2.0, 0.5, 1.5, 0.2, 1.0];
        let result = SolverResult::new(t, y, 2, SolverStats::new());
        assert_eq!(result.n_steps(), 3);

        let empty = SolverResult::<f64>::failed("err".to_string(), SolverStats::new());
        assert_eq!(empty.n_steps(), 0);
    }

    #[test]
    fn test_component() {
        let t = vec![0.0, 0.5, 1.0];
        // 2D system: y0 = [1.0, 0.5, 0.2], y1 = [2.0, 1.5, 1.0]
        let y = vec![1.0, 2.0, 0.5, 1.5, 0.2, 1.0];
        let result = SolverResult::new(t, y, 2, SolverStats::new());

        let comp0 = result.component(0).unwrap();
        assert_eq!(comp0, vec![1.0, 0.5, 0.2]);

        let comp1 = result.component(1).unwrap();
        assert_eq!(comp1, vec![2.0, 1.5, 1.0]);
    }

    #[test]
    fn test_component_out_of_bounds() {
        let t = vec![0.0];
        let y = vec![1.0, 2.0];
        let result = SolverResult::new(t, y, 2, SolverStats::new());
        assert!(result.component(2).is_none());
    }
}

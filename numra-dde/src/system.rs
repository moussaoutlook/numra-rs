//! DDE system trait and solver infrastructure.
//!
//! Author: Moussa Leblouba
//! Date: 4 February 2026
//! Modified: 2 May 2026

use numra_core::Scalar;

/// Trait for delay differential equation systems.
///
/// Defines a DDE of the form:
/// ```text
/// y'(t) = f(t, y(t), y(t - τ₁), y(t - τ₂), ...)
/// ```
pub trait DdeSystem<S: Scalar> {
    /// Dimension of the state space.
    fn dim(&self) -> usize;

    /// Get the delays τᵢ.
    ///
    /// For constant delays, this returns a fixed vector.
    /// For state-dependent delays, implement `delays_at` instead.
    fn delays(&self) -> Vec<S>;

    /// Get delays at a specific state (for state-dependent delays).
    ///
    /// Default implementation returns constant delays.
    fn delays_at(&self, _t: S, _y: &[S]) -> Vec<S> {
        self.delays()
    }

    /// Number of delays.
    fn n_delays(&self) -> usize {
        self.delays().len()
    }

    /// Evaluate the right-hand side f(t, y(t), y(t-τ₁), ...).
    ///
    /// # Arguments
    /// * `t` - Current time
    /// * `y` - Current state y(t)
    /// * `y_delayed` - Delayed states: `y_delayed[i] = y(t - τᵢ)`
    /// * `dydt` - Output buffer for y'(t)
    fn rhs(&self, t: S, y: &[S], y_delayed: &[&[S]], dydt: &mut [S]);

    /// Whether delays are state-dependent.
    fn has_state_dependent_delays(&self) -> bool {
        false
    }
}

/// Options for DDE solvers.
///
/// **Divergence from `numra_ode::SolverOptions`** (per Foundation Spec §2.5):
/// the field set most closely mirrors `SolverOptions` (`rtol`, `atol`, `h0`,
/// `h_max`, `h_min`, `max_steps`, `t_eval`, `dense_output`) but diverges on
/// two DDE-specific concerns: (1) `dense_output` defaults to `true` here
/// because the Method-of-Steps solver evaluates `y(t − τ)` at delay-shifted
/// times that don't generally land on integration grid points; (2)
/// `track_discontinuities` + `discontinuity_order` encode the inherent DDE
/// discontinuity-propagation structure (initial-history non-smoothness
/// propagates forward at integer multiples of the delays), which has no
/// ODE analog. See `docs/architecture/foundation-specification.md` §2.5.
#[derive(Clone, Debug)]
pub struct DdeOptions<S: Scalar> {
    /// Relative tolerance
    pub rtol: S,
    /// Absolute tolerance
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
    /// Enable dense output (required for accurate history interpolation)
    pub dense_output: bool,
    /// Track discontinuities in the solution
    pub track_discontinuities: bool,
    /// Track discontinuities up to this derivative order
    pub discontinuity_order: usize,
}

impl<S: Scalar> Default for DdeOptions<S> {
    fn default() -> Self {
        Self {
            rtol: S::from_f64(1e-6),
            atol: S::from_f64(1e-9),
            h0: None,
            h_max: S::INFINITY,
            h_min: S::from_f64(1e-14),
            max_steps: 100_000,
            t_eval: None,
            dense_output: true, // Usually needed for DDEs
            track_discontinuities: false,
            discontinuity_order: 0,
        }
    }
}

impl<S: Scalar> DdeOptions<S> {
    pub fn rtol(mut self, rtol: S) -> Self {
        self.rtol = rtol;
        self
    }

    pub fn atol(mut self, atol: S) -> Self {
        self.atol = atol;
        self
    }

    pub fn h0(mut self, h0: S) -> Self {
        self.h0 = Some(h0);
        self
    }

    pub fn h_max(mut self, h_max: S) -> Self {
        self.h_max = h_max;
        self
    }

    pub fn max_steps(mut self, max_steps: usize) -> Self {
        self.max_steps = max_steps;
        self
    }

    /// Enable discontinuity tracking.
    ///
    /// When enabled, the solver will detect and step exactly to
    /// discontinuity points that propagate from t0 at each delay interval.
    pub fn track_discontinuities(mut self, track: bool) -> Self {
        self.track_discontinuities = track;
        self
    }

    /// Set the discontinuity order to track.
    ///
    /// Discontinuities propagate: the initial discontinuity at t0 propagates
    /// to t0 + tau for each delay tau. Setting order=3 tracks up to 3 levels
    /// of propagation (e.g., t0, t0+tau, t0+2*tau, t0+3*tau for single delay).
    pub fn discontinuity_order(mut self, order: usize) -> Self {
        self.discontinuity_order = order;
        self
    }
}

/// Statistics from DDE solver.
#[derive(Clone, Debug, Default)]
pub struct DdeStats {
    /// Number of RHS evaluations
    pub n_eval: usize,
    /// Number of accepted steps
    pub n_accept: usize,
    /// Number of rejected steps
    pub n_reject: usize,
    /// Number of discontinuity crossings handled
    pub n_discontinuities: usize,
}

/// Result of DDE integration.
#[derive(Clone, Debug)]
pub struct DdeResult<S: Scalar> {
    /// Time points
    pub t: Vec<S>,
    /// Solution at each time point (row-major: y[i*dim + j] = y_j(t_i))
    pub y: Vec<S>,
    /// Dimension of the system
    pub dim: usize,
    /// Solver statistics
    pub stats: DdeStats,
    /// Was integration successful?
    pub success: bool,
    /// Message (error description if failed)
    pub message: String,
}

impl<S: Scalar> DdeResult<S> {
    pub fn new(t: Vec<S>, y: Vec<S>, dim: usize, stats: DdeStats) -> Self {
        Self {
            t,
            y,
            dim,
            stats,
            success: true,
            message: String::new(),
        }
    }

    pub fn failed(message: String, stats: DdeStats) -> Self {
        Self {
            t: Vec::new(),
            y: Vec::new(),
            dim: 0,
            stats,
            success: false,
            message,
        }
    }

    pub fn len(&self) -> usize {
        self.t.len()
    }

    pub fn is_empty(&self) -> bool {
        self.t.is_empty()
    }

    pub fn t_final(&self) -> Option<S> {
        self.t.last().copied()
    }

    pub fn y_final(&self) -> Option<Vec<S>> {
        if self.t.is_empty() {
            None
        } else {
            let start = (self.t.len() - 1) * self.dim;
            Some(self.y[start..start + self.dim].to_vec())
        }
    }

    pub fn y_at(&self, i: usize) -> &[S] {
        let start = i * self.dim;
        &self.y[start..start + self.dim]
    }

    pub fn iter(&self) -> impl Iterator<Item = (S, &[S])> {
        self.t
            .iter()
            .enumerate()
            .map(move |(i, &t)| (t, self.y_at(i)))
    }
}

/// Trait for DDE solvers.
pub trait DdeSolver<S: Scalar> {
    /// Solve the DDE problem.
    ///
    /// # Arguments
    /// * `system` - The DDE system to solve
    /// * `t0` - Initial time
    /// * `tf` - Final time
    /// * `history` - History function φ(t) for t ≤ t0
    /// * `options` - Solver options
    fn solve<Sys, H>(
        system: &Sys,
        t0: S,
        tf: S,
        history: &H,
        options: &DdeOptions<S>,
    ) -> Result<DdeResult<S>, String>
    where
        Sys: DdeSystem<S>,
        H: Fn(S) -> Vec<S>;
}

#[cfg(test)]
mod tests {
    use super::*;

    struct TestDde;

    impl DdeSystem<f64> for TestDde {
        fn dim(&self) -> usize {
            1
        }
        fn delays(&self) -> Vec<f64> {
            vec![1.0]
        }
        fn rhs(&self, _t: f64, y: &[f64], y_delayed: &[&[f64]], dydt: &mut [f64]) {
            dydt[0] = -y[0] + y_delayed[0][0];
        }
    }

    #[test]
    fn test_dde_system_trait() {
        let sys = TestDde;
        assert_eq!(sys.dim(), 1);
        assert_eq!(sys.n_delays(), 1);
        assert_eq!(sys.delays(), vec![1.0]);

        let mut dydt = [0.0];
        let y_delayed = [&[0.5][..]];
        sys.rhs(0.0, &[1.0], &y_delayed, &mut dydt);
        assert!((dydt[0] - (-0.5)).abs() < 1e-10);
    }

    #[test]
    fn test_dde_options() {
        let opts: DdeOptions<f64> = DdeOptions::default().rtol(1e-8).atol(1e-10);
        assert!((opts.rtol - 1e-8).abs() < 1e-15);
        assert!((opts.atol - 1e-10).abs() < 1e-15);
    }
}

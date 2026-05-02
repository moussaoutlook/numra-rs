//! SDE system trait and solver infrastructure.
//!
//! Author: Moussa Leblouba
//! Date: 3 February 2026
//! Modified: 2 May 2026

use numra_core::Scalar;

/// Type of noise in the SDE.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum NoiseType {
    /// Diagonal noise: each component has independent Wiener process
    #[default]
    Diagonal,
    /// Scalar noise: single Wiener process affects all components
    Scalar,
    /// General noise: full noise matrix (m Wiener processes, n state dims)
    General { n_wiener: usize },
}

/// Trait for stochastic differential equation systems.
///
/// Defines an SDE of the form:
/// ```text
/// dX(t) = f(t, X) dt + g(t, X) dW(t)
/// ```
pub trait SdeSystem<S: Scalar>: Sync {
    /// Dimension of the state space.
    fn dim(&self) -> usize;

    /// Evaluate the drift function f(t, x).
    ///
    /// # Arguments
    /// * `t` - Current time
    /// * `x` - Current state
    /// * `f` - Output buffer for drift (length = dim)
    fn drift(&self, t: S, x: &[S], f: &mut [S]);

    /// Evaluate the diffusion function g(t, x).
    ///
    /// For diagonal noise, `g` has length `dim`.
    /// For scalar noise, `g` has length `dim` (same noise scaled differently).
    /// For general noise, `g` has length `dim * n_wiener`.
    ///
    /// # Arguments
    /// * `t` - Current time
    /// * `x` - Current state
    /// * `g` - Output buffer for diffusion
    fn diffusion(&self, t: S, x: &[S], g: &mut [S]);

    /// Type of noise (default: diagonal).
    fn noise_type(&self) -> NoiseType {
        NoiseType::Diagonal
    }

    /// Number of Wiener processes.
    fn n_wiener(&self) -> usize {
        match self.noise_type() {
            NoiseType::Diagonal => self.dim(),
            NoiseType::Scalar => 1,
            NoiseType::General { n_wiener } => n_wiener,
        }
    }

    /// Derivative of diffusion w.r.t. state: ∂g/∂x * g
    ///
    /// Required for Milstein method. Default implementation uses finite differences.
    fn diffusion_derivative(&self, t: S, x: &[S], gdg: &mut [S]) {
        let dim = self.dim();
        let eps = S::from_f64(1e-8);

        let mut g = vec![S::ZERO; dim];
        let mut g_plus = vec![S::ZERO; dim];
        let mut x_pert = x.to_vec();

        self.diffusion(t, x, &mut g);

        for i in 0..dim {
            x_pert[i] = x[i] + eps;
            self.diffusion(t, &x_pert, &mut g_plus);
            x_pert[i] = x[i];

            // (∂g_i/∂x_i) * g_i for diagonal case
            gdg[i] = (g_plus[i] - g[i]) / eps * g[i];
        }
    }
}

/// Options for SDE solvers.
#[derive(Clone, Debug)]
pub struct SdeOptions<S: Scalar> {
    /// Fixed time step (for non-adaptive methods)
    pub dt: S,
    /// Relative tolerance (for adaptive methods)
    pub rtol: S,
    /// Absolute tolerance (for adaptive methods)
    pub atol: S,
    /// Maximum time step
    pub dt_max: S,
    /// Minimum time step
    pub dt_min: S,
    /// Maximum number of steps
    pub max_steps: usize,
    /// Save solution at all steps (vs. just final)
    pub save_trajectory: bool,
    /// Random seed (None = use system entropy)
    pub seed: Option<u64>,
}

impl<S: Scalar> Default for SdeOptions<S> {
    fn default() -> Self {
        Self {
            dt: S::from_f64(0.01),
            rtol: S::from_f64(1e-3),
            atol: S::from_f64(1e-6),
            dt_max: S::INFINITY,
            dt_min: S::from_f64(1e-10),
            max_steps: 1_000_000,
            save_trajectory: true,
            seed: None,
        }
    }
}

impl<S: Scalar> SdeOptions<S> {
    /// Set fixed time step.
    pub fn dt(mut self, dt: S) -> Self {
        self.dt = dt;
        self
    }

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

    /// Set maximum time step.
    pub fn dt_max(mut self, dt_max: S) -> Self {
        self.dt_max = dt_max;
        self
    }

    /// Set random seed for reproducibility.
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Enable/disable trajectory saving.
    pub fn save_trajectory(mut self, save: bool) -> Self {
        self.save_trajectory = save;
        self
    }
}

/// Statistics from SDE solver.
#[derive(Clone, Debug, Default)]
pub struct SdeStats {
    /// Number of drift evaluations
    pub n_drift: usize,
    /// Number of diffusion evaluations
    pub n_diffusion: usize,
    /// Number of accepted steps
    pub n_accept: usize,
    /// Number of rejected steps (for adaptive methods)
    pub n_reject: usize,
}

/// Result of SDE integration.
#[derive(Clone, Debug)]
pub struct SdeResult<S: Scalar> {
    /// Time points
    pub t: Vec<S>,
    /// Solution at each time point (row-major: y[i*dim + j] = y_j(t_i))
    pub y: Vec<S>,
    /// Dimension of the system
    pub dim: usize,
    /// Solver statistics
    pub stats: SdeStats,
    /// Was integration successful?
    pub success: bool,
    /// Message (error description if failed)
    pub message: String,
}

impl<S: Scalar> SdeResult<S> {
    /// Create a new successful result.
    pub fn new(t: Vec<S>, y: Vec<S>, dim: usize, stats: SdeStats) -> Self {
        Self {
            t,
            y,
            dim,
            stats,
            success: true,
            message: String::new(),
        }
    }

    /// Create a failed result.
    pub fn failed(message: String, stats: SdeStats) -> Self {
        Self {
            t: Vec::new(),
            y: Vec::new(),
            dim: 0,
            stats,
            success: false,
            message,
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

    /// Iterate over (t, y) pairs.
    pub fn iter(&self) -> impl Iterator<Item = (S, &[S])> {
        self.t
            .iter()
            .enumerate()
            .map(move |(i, &t)| (t, self.y_at(i)))
    }
}

/// Trait for SDE solvers.
pub trait SdeSolver<S: Scalar> {
    /// Solve the SDE problem.
    ///
    /// # Arguments
    /// * `system` - The SDE system to solve
    /// * `t0` - Initial time
    /// * `tf` - Final time
    /// * `x0` - Initial state
    /// * `options` - Solver options
    /// * `seed` - Optional random seed (overrides options.seed)
    fn solve<Sys: SdeSystem<S>>(
        system: &Sys,
        t0: S,
        tf: S,
        x0: &[S],
        options: &SdeOptions<S>,
        seed: Option<u64>,
    ) -> Result<SdeResult<S>, String>;
}

#[cfg(test)]
mod tests {
    use super::*;

    struct TestSde;

    impl SdeSystem<f64> for TestSde {
        fn dim(&self) -> usize {
            1
        }
        fn drift(&self, _t: f64, x: &[f64], f: &mut [f64]) {
            f[0] = -x[0];
        }
        fn diffusion(&self, _t: f64, x: &[f64], g: &mut [f64]) {
            g[0] = 0.1 * x[0];
        }
    }

    #[test]
    fn test_sde_system_trait() {
        let sys = TestSde;
        assert_eq!(sys.dim(), 1);
        assert_eq!(sys.n_wiener(), 1);

        let mut f = [0.0];
        let mut g = [0.0];
        sys.drift(0.0, &[1.0], &mut f);
        sys.diffusion(0.0, &[1.0], &mut g);
        assert!((f[0] - (-1.0)).abs() < 1e-10);
        assert!((g[0] - 0.1).abs() < 1e-10);
    }

    #[test]
    fn test_sde_options() {
        let opts: SdeOptions<f64> = SdeOptions::default().dt(0.001).seed(42);
        assert!((opts.dt - 0.001).abs() < 1e-10);
        assert_eq!(opts.seed, Some(42));
    }
}

//! IDE system trait and solver infrastructure.
//!
//! Author: Moussa Leblouba
//! Date: 5 March 2026
//! Modified: 2 May 2026

use numra_core::Scalar;

/// Trait for integro-differential equation systems.
///
/// Defines a Volterra IDE of the form:
/// ```text
/// y'(t) = f(t, y) + ∫₀ᵗ K(t, s, y(s)) ds,  y(0) = y₀
/// ```
pub trait IdeSystem<S: Scalar> {
    /// Dimension of the state space.
    fn dim(&self) -> usize;

    /// Evaluate the local right-hand side f(t, y).
    ///
    /// This is the non-integral part of the equation.
    fn rhs(&self, t: S, y: &[S], f: &mut [S]);

    /// Evaluate the memory kernel K(t, s, y(s)).
    ///
    /// # Arguments
    /// * `t` - Current time
    /// * `s` - Integration variable (past time)
    /// * `y_s` - Solution at time s
    /// * `k` - Output buffer for kernel values
    fn kernel(&self, t: S, s: S, y_s: &[S], k: &mut [S]);

    /// Whether the kernel depends only on (t-s), not t and s separately.
    ///
    /// Convolution kernels K(t-s) can be computed more efficiently.
    fn is_convolution_kernel(&self) -> bool {
        false
    }
}

/// Options for IDE solvers.
#[derive(Clone, Debug)]
pub struct IdeOptions<S: Scalar> {
    /// Time step size
    pub dt: S,
    /// Maximum number of steps
    pub max_steps: usize,
    /// Tolerance for iterative methods
    pub tol: S,
    /// Maximum iterations for implicit methods
    pub max_iter: usize,
    /// Number of quadrature points per step for integral
    pub quad_points: usize,
}

impl<S: Scalar> Default for IdeOptions<S> {
    fn default() -> Self {
        Self {
            dt: S::from_f64(0.01),
            max_steps: 100_000,
            tol: S::from_f64(1e-10),
            max_iter: 100,
            quad_points: 4, // Gauss-Legendre 4-point by default
        }
    }
}

impl<S: Scalar> IdeOptions<S> {
    pub fn dt(mut self, dt: S) -> Self {
        self.dt = dt;
        self
    }

    pub fn max_steps(mut self, max_steps: usize) -> Self {
        self.max_steps = max_steps;
        self
    }

    pub fn tol(mut self, tol: S) -> Self {
        self.tol = tol;
        self
    }

    pub fn quad_points(mut self, n: usize) -> Self {
        self.quad_points = n;
        self
    }
}

/// Statistics from IDE solver.
#[derive(Clone, Debug, Default)]
pub struct IdeStats {
    /// Number of RHS evaluations
    pub n_rhs: usize,
    /// Number of kernel evaluations
    pub n_kernel: usize,
    /// Number of steps taken
    pub n_steps: usize,
}

/// Result of IDE integration.
#[derive(Clone, Debug)]
pub struct IdeResult<S: Scalar> {
    /// Time points
    pub t: Vec<S>,
    /// Solution at each time point (row-major)
    pub y: Vec<S>,
    /// Dimension of the system
    pub dim: usize,
    /// Solver statistics
    pub stats: IdeStats,
    /// Was integration successful?
    pub success: bool,
    /// Message
    pub message: String,
}

impl<S: Scalar> IdeResult<S> {
    pub fn new(t: Vec<S>, y: Vec<S>, dim: usize, stats: IdeStats) -> Self {
        Self {
            t,
            y,
            dim,
            stats,
            success: true,
            message: String::new(),
        }
    }

    pub fn failed(message: String, stats: IdeStats) -> Self {
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
}

/// Trait for IDE solvers.
pub trait IdeSolver<S: Scalar> {
    /// Solve the IDE problem.
    fn solve<Sys: IdeSystem<S>>(
        system: &Sys,
        t0: S,
        tf: S,
        y0: &[S],
        options: &IdeOptions<S>,
    ) -> Result<IdeResult<S>, String>;
}

#[cfg(test)]
mod tests {
    use super::*;

    struct TestIde;

    impl IdeSystem<f64> for TestIde {
        fn dim(&self) -> usize {
            1
        }

        fn rhs(&self, _t: f64, y: &[f64], f: &mut [f64]) {
            f[0] = -y[0];
        }

        fn kernel(&self, t: f64, s: f64, y_s: &[f64], k: &mut [f64]) {
            // Exponential kernel: K(t,s) = exp(-(t-s)) * y(s)
            k[0] = (-(t - s)).exp() * y_s[0];
        }

        fn is_convolution_kernel(&self) -> bool {
            true
        }
    }

    #[test]
    fn test_ide_system_trait() {
        let sys = TestIde;
        assert_eq!(sys.dim(), 1);
        assert!(sys.is_convolution_kernel());

        let mut f = [0.0];
        sys.rhs(0.0, &[1.0], &mut f);
        assert!((f[0] - (-1.0)).abs() < 1e-10);

        let mut k = [0.0];
        sys.kernel(1.0, 0.5, &[2.0], &mut k);
        // K(1, 0.5, 2) = exp(-0.5) * 2 ≈ 1.2131
        assert!((k[0] - 2.0 * (-0.5_f64).exp()).abs() < 1e-10);
    }

    #[test]
    fn test_ide_options() {
        let opts: IdeOptions<f64> = IdeOptions::default().dt(0.001).quad_points(8);
        assert!((opts.dt - 0.001).abs() < 1e-10);
        assert_eq!(opts.quad_points, 8);
    }

    #[test]
    fn test_ide_result_accessors() {
        let t = vec![0.0, 0.5, 1.0];
        let y = vec![1.0, 2.0, 0.8, 1.5, 0.6, 1.2]; // 3 time points, dim=2
        let stats = IdeStats {
            n_rhs: 10,
            n_kernel: 20,
            n_steps: 2,
        };
        let result = IdeResult::new(t, y, 2, stats);

        // len and is_empty
        assert_eq!(result.len(), 3);
        assert!(!result.is_empty());

        // t_final
        assert!((result.t_final().unwrap() - 1.0).abs() < 1e-15);

        // y_final
        let yf = result.y_final().unwrap();
        assert_eq!(yf.len(), 2);
        assert!((yf[0] - 0.6).abs() < 1e-15);
        assert!((yf[1] - 1.2).abs() < 1e-15);

        // y_at
        let y0 = result.y_at(0);
        assert!((y0[0] - 1.0).abs() < 1e-15);
        assert!((y0[1] - 2.0).abs() < 1e-15);

        let y1 = result.y_at(1);
        assert!((y1[0] - 0.8).abs() < 1e-15);
        assert!((y1[1] - 1.5).abs() < 1e-15);
    }

    #[test]
    fn test_ide_result_failed() {
        let stats = IdeStats {
            n_rhs: 5,
            n_kernel: 3,
            n_steps: 1,
        };
        let result: IdeResult<f64> = IdeResult::failed("something went wrong".to_string(), stats);

        assert!(!result.success);
        assert!(result.message.contains("something went wrong"));
        assert!(result.is_empty());
        assert_eq!(result.len(), 0);
        assert!(result.t_final().is_none());
        assert!(result.y_final().is_none());
    }

    #[test]
    fn test_ide_options_max_steps() {
        let opts: IdeOptions<f64> = IdeOptions::default().max_steps(42);
        assert_eq!(opts.max_steps, 42);

        // Also verify the default
        let default_opts: IdeOptions<f64> = IdeOptions::default();
        assert_eq!(default_opts.max_steps, 100_000);
    }
}

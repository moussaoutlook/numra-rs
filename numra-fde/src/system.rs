//! FDE system trait and solver infrastructure.
//!
//! Author: Moussa Leblouba
//! Date: 5 March 2026
//! Modified: 2 May 2026

use numra_core::Scalar;

/// Trait for fractional differential equation systems.
///
/// Defines an FDE of the form:
/// ```text
/// D^α y(t) = f(t, y),  y(0) = y₀
/// ```
/// where D^α is the Caputo fractional derivative of order α.
pub trait FdeSystem<S: Scalar> {
    /// Dimension of the state space.
    fn dim(&self) -> usize;

    /// Fractional order α ∈ (0, 1].
    ///
    /// When α = 1, this reduces to an ordinary ODE.
    fn alpha(&self) -> S;

    /// Evaluate the right-hand side f(t, y).
    fn rhs(&self, t: S, y: &[S], f: &mut [S]);

    /// Check if the order is valid (0 < α ≤ 1).
    fn is_valid_order(&self) -> bool {
        let alpha = self.alpha();
        alpha > S::ZERO && alpha <= S::ONE
    }
}

/// Options for FDE solvers.
#[derive(Clone, Debug)]
pub struct FdeOptions<S: Scalar> {
    /// Time step size
    pub dt: S,
    /// Maximum number of steps
    pub max_steps: usize,
    /// Tolerance for iterative methods (if applicable)
    pub tol: S,
    /// Maximum iterations for implicit methods
    pub max_iter: usize,
}

impl<S: Scalar> Default for FdeOptions<S> {
    fn default() -> Self {
        Self {
            dt: S::from_f64(0.01),
            max_steps: 100_000,
            tol: S::from_f64(1e-10),
            max_iter: 100,
        }
    }
}

impl<S: Scalar> FdeOptions<S> {
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
}

/// Statistics from FDE solver.
#[derive(Clone, Debug, Default)]
pub struct FdeStats {
    /// Number of RHS evaluations
    pub n_rhs: usize,
    /// Number of steps taken
    pub n_steps: usize,
}

/// Result of FDE integration.
#[derive(Clone, Debug)]
pub struct FdeResult<S: Scalar> {
    /// Time points
    pub t: Vec<S>,
    /// Solution at each time point (row-major)
    pub y: Vec<S>,
    /// Dimension of the system
    pub dim: usize,
    /// Solver statistics
    pub stats: FdeStats,
    /// Was integration successful?
    pub success: bool,
    /// Message
    pub message: String,
}

impl<S: Scalar> FdeResult<S> {
    pub fn new(t: Vec<S>, y: Vec<S>, dim: usize, stats: FdeStats) -> Self {
        Self {
            t,
            y,
            dim,
            stats,
            success: true,
            message: String::new(),
        }
    }

    pub fn failed(message: String, stats: FdeStats) -> Self {
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

/// Trait for FDE solvers.
pub trait FdeSolver<S: Scalar> {
    /// Solve the FDE problem.
    fn solve<Sys: FdeSystem<S>>(
        system: &Sys,
        t0: S,
        tf: S,
        y0: &[S],
        options: &FdeOptions<S>,
    ) -> Result<FdeResult<S>, String>;
}

#[cfg(test)]
mod tests {
    use super::*;

    struct TestFde;

    impl FdeSystem<f64> for TestFde {
        fn dim(&self) -> usize {
            1
        }
        fn alpha(&self) -> f64 {
            0.5
        }
        fn rhs(&self, _t: f64, y: &[f64], f: &mut [f64]) {
            f[0] = -y[0];
        }
    }

    #[test]
    fn test_fde_system_trait() {
        let sys = TestFde;
        assert_eq!(sys.dim(), 1);
        assert!((sys.alpha() - 0.5).abs() < 1e-10);
        assert!(sys.is_valid_order());

        let mut f = [0.0];
        sys.rhs(0.0, &[1.0], &mut f);
        assert!((f[0] - (-1.0)).abs() < 1e-10);
    }

    #[test]
    fn test_fde_options() {
        let opts: FdeOptions<f64> = FdeOptions::default().dt(0.001);
        assert!((opts.dt - 0.001).abs() < 1e-10);
    }

    #[test]
    fn test_invalid_alpha_zero() {
        struct AlphaZero;
        impl FdeSystem<f64> for AlphaZero {
            fn dim(&self) -> usize {
                1
            }
            fn alpha(&self) -> f64 {
                0.0
            }
            fn rhs(&self, _t: f64, _y: &[f64], f: &mut [f64]) {
                f[0] = 0.0;
            }
        }
        assert!(!AlphaZero.is_valid_order(), "alpha=0 should be invalid");
    }

    #[test]
    fn test_invalid_alpha_negative() {
        struct AlphaNeg;
        impl FdeSystem<f64> for AlphaNeg {
            fn dim(&self) -> usize {
                1
            }
            fn alpha(&self) -> f64 {
                -0.5
            }
            fn rhs(&self, _t: f64, _y: &[f64], f: &mut [f64]) {
                f[0] = 0.0;
            }
        }
        assert!(!AlphaNeg.is_valid_order(), "alpha=-0.5 should be invalid");
    }

    #[test]
    fn test_alpha_exactly_one() {
        struct AlphaOne;
        impl FdeSystem<f64> for AlphaOne {
            fn dim(&self) -> usize {
                1
            }
            fn alpha(&self) -> f64 {
                1.0
            }
            fn rhs(&self, _t: f64, _y: &[f64], f: &mut [f64]) {
                f[0] = 0.0;
            }
        }
        assert!(AlphaOne.is_valid_order(), "alpha=1.0 should be valid");
    }
}

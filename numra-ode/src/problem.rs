//! ODE problem definition.
//!
//! This module defines the traits and types for representing ODE initial value problems.
//! Also supports DAEs (Differential-Algebraic Equations) via mass matrices.
//!
//! Author: Moussa Leblouba
//! Date: 4 February 2026
//! Modified: 2 May 2026

use numra_core::Scalar;

/// A system of ordinary differential equations.
///
/// Represents: dy/dt = f(t, y)
pub trait OdeSystem<S: Scalar> {
    /// Dimension of the system.
    fn dim(&self) -> usize;

    /// Compute the right-hand side: dydt = f(t, y)
    fn rhs(&self, t: S, y: &[S], dydt: &mut [S]);

    /// Optionally compute the Jacobian: J = ∂f/∂y
    /// Default implementation uses finite differences.
    fn jacobian(&self, t: S, y: &[S], jac: &mut [S]) {
        let n = self.dim();
        let eps = S::from_f64(1e-8);
        let mut y_pert = y.to_vec();
        let mut f0 = vec![S::ZERO; n];
        let mut f1 = vec![S::ZERO; n];

        self.rhs(t, y, &mut f0);

        for j in 0..n {
            let yj_save = y_pert[j];
            let h = eps * (S::ONE + yj_save.abs());
            y_pert[j] = yj_save + h;
            self.rhs(t, &y_pert, &mut f1);
            y_pert[j] = yj_save;

            for i in 0..n {
                jac[i * n + j] = (f1[i] - f0[i]) / h;
            }
        }
    }

    /// Is the system autonomous? (f does not depend on t explicitly)
    fn is_autonomous(&self) -> bool {
        false
    }

    /// Does this system have a mass matrix?
    fn has_mass_matrix(&self) -> bool {
        false
    }

    /// Get the mass matrix M for the DAE: M * y' = f(t, y)
    ///
    /// Default returns identity (standard ODE).
    /// For DAEs, override to return the (possibly singular) mass matrix.
    ///
    /// The matrix is stored in row-major order: mass[i * n + j] = M[i, j]
    fn mass_matrix(&self, mass: &mut [S]) {
        let n = self.dim();
        for i in 0..n {
            for j in 0..n {
                mass[i * n + j] = if i == j { S::ONE } else { S::ZERO };
            }
        }
    }

    /// Is the mass matrix singular? (i.e., is this a DAE?)
    fn is_singular_mass(&self) -> bool {
        false
    }

    /// Return indices of algebraic variables (where `M[i,i] = 0`).
    ///
    /// Default returns empty (all differential).
    fn algebraic_indices(&self) -> Vec<usize> {
        Vec::new()
    }
}

/// ODE initial value problem.
///
/// Represents: dy/dt = f(t, y), y(t0) = y0, solve from t0 to tf
pub struct OdeProblem<S: Scalar, F>
where
    F: Fn(S, &[S], &mut [S]),
{
    /// Right-hand side function
    pub f: F,
    /// Initial time
    pub t0: S,
    /// Final time
    pub tf: S,
    /// Initial state
    pub y0: Vec<S>,
}

impl<S: Scalar, F> OdeProblem<S, F>
where
    F: Fn(S, &[S], &mut [S]),
{
    /// Create a new ODE problem.
    pub fn new(f: F, t0: S, tf: S, y0: Vec<S>) -> Self {
        Self { f, t0, tf, y0 }
    }

    /// Dimension of the system.
    pub fn dim(&self) -> usize {
        self.y0.len()
    }

    /// Time span.
    pub fn tspan(&self) -> (S, S) {
        (self.t0, self.tf)
    }

    /// Integration direction: 1 for forward, -1 for backward.
    pub fn direction(&self) -> S {
        if self.tf >= self.t0 {
            S::ONE
        } else {
            -S::ONE
        }
    }
}

impl<S: Scalar, F> OdeSystem<S> for OdeProblem<S, F>
where
    F: Fn(S, &[S], &mut [S]),
{
    fn dim(&self) -> usize {
        self.y0.len()
    }

    fn rhs(&self, t: S, y: &[S], dydt: &mut [S]) {
        (self.f)(t, y, dydt);
    }
}

/// DAE (Differential-Algebraic Equation) problem.
///
/// Represents: M * y' = f(t, y)
///
/// where M is a (possibly singular) mass matrix.
/// - When M is nonsingular, this is equivalent to y' = M⁻¹ f(t, y)
/// - When M is singular (index-1 DAE), some equations are algebraic constraints
///
/// # Example: Pendulum DAE
///
/// The pendulum in Cartesian coordinates:
/// ```text
/// x'' = -λx         (x-component of constraint force)
/// y'' = -λy - g     (y-component + gravity)
/// x² + y² = L²      (constraint: pendulum length)
/// ```
///
/// Written as first-order DAE with y = [x, y, vx, vy, λ]:
/// ```text
/// x'  = vx           (differential)
/// y'  = vy           (differential)
/// vx' = -λx          (differential)
/// vy' = -λy - g      (differential)
/// 0   = x² + y² - L² (algebraic)
/// ```
pub struct DaeProblem<S: Scalar, F, M>
where
    F: Fn(S, &[S], &mut [S]),
    M: Fn(&mut [S]),
{
    /// Right-hand side function
    pub f: F,
    /// Mass matrix function
    pub mass: M,
    /// Initial time
    pub t0: S,
    /// Final time
    pub tf: S,
    /// Initial state (must be consistent!)
    pub y0: Vec<S>,
    /// Indices of algebraic variables (where `M[i,i] = 0`)
    pub alg_indices: Vec<usize>,
}

impl<S: Scalar, F, M> DaeProblem<S, F, M>
where
    F: Fn(S, &[S], &mut [S]),
    M: Fn(&mut [S]),
{
    /// Create a new DAE problem.
    ///
    /// # Arguments
    /// * `f` - RHS function
    /// * `mass` - Mass matrix function (fills row-major array)
    /// * `t0` - Initial time
    /// * `tf` - Final time
    /// * `y0` - Initial state (must satisfy algebraic constraints!)
    /// * `alg_indices` - Indices of algebraic equations
    pub fn new(f: F, mass: M, t0: S, tf: S, y0: Vec<S>, alg_indices: Vec<usize>) -> Self {
        Self {
            f,
            mass,
            t0,
            tf,
            y0,
            alg_indices,
        }
    }

    /// Dimension of the system.
    pub fn dim(&self) -> usize {
        self.y0.len()
    }
}

impl<S: Scalar, F, M> OdeSystem<S> for DaeProblem<S, F, M>
where
    F: Fn(S, &[S], &mut [S]),
    M: Fn(&mut [S]),
{
    fn dim(&self) -> usize {
        self.y0.len()
    }

    fn rhs(&self, t: S, y: &[S], dydt: &mut [S]) {
        (self.f)(t, y, dydt);
    }

    fn has_mass_matrix(&self) -> bool {
        true
    }

    fn mass_matrix(&self, mass: &mut [S]) {
        (self.mass)(mass);
    }

    fn is_singular_mass(&self) -> bool {
        !self.alg_indices.is_empty()
    }

    fn algebraic_indices(&self) -> Vec<usize> {
        self.alg_indices.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ode_problem_creation() {
        let f = |_t: f64, y: &[f64], dydt: &mut [f64]| {
            dydt[0] = -y[0];
        };
        let problem = OdeProblem::new(f, 0.0, 1.0, vec![1.0]);
        assert_eq!(problem.dim(), 1);
        assert_eq!(problem.tspan(), (0.0, 1.0));
    }

    #[test]
    fn test_rhs_evaluation() {
        let f = |_t: f64, y: &[f64], dydt: &mut [f64]| {
            dydt[0] = -2.0 * y[0];
        };
        let problem = OdeProblem::new(f, 0.0, 1.0, vec![1.0]);

        let mut dydt = vec![0.0];
        problem.rhs(0.0, &[1.0], &mut dydt);
        assert!((dydt[0] + 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_jacobian_finite_diff() {
        // dy/dt = -2*y, so J = -2
        let f = |_t: f64, y: &[f64], dydt: &mut [f64]| {
            dydt[0] = -2.0 * y[0];
        };
        let problem = OdeProblem::new(f, 0.0, 1.0, vec![1.0]);

        let mut jac = vec![0.0];
        problem.jacobian(0.0, &[1.0], &mut jac);
        assert!((jac[0] + 2.0).abs() < 1e-5);
    }
}

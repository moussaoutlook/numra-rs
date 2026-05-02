//! Boundary conditions for PDE problems.
//!
//! Author: Moussa Leblouba
//! Date: 3 February 2026
//! Modified: 2 May 2026

use numra_core::Scalar;

/// Trait for boundary conditions.
pub trait BoundaryCondition<S: Scalar>: Clone {
    /// Apply the boundary condition at the given time.
    ///
    /// Returns the boundary value (for Dirichlet) or modifies the ghost point (for Neumann).
    fn apply(&self, t: S, interior_value: S, dx: S) -> S;

    /// Get the boundary value directly (for Dirichlet-type BCs).
    fn value(&self, _t: S) -> Option<S> {
        None
    }

    /// Get the flux value (for Neumann-type BCs).
    fn flux(&self, _t: S) -> Option<S> {
        None
    }

    /// Whether this BC specifies the solution value (Dirichlet-type).
    fn is_dirichlet(&self) -> bool;
}

/// Dirichlet boundary condition: u = g(t) at the boundary.
#[derive(Clone, Debug)]
pub struct DirichletBC<S: Scalar> {
    /// Boundary value (constant or function of time)
    value: DirichletValue<S>,
}

#[derive(Clone, Debug)]
enum DirichletValue<S: Scalar> {
    Constant(S),
    // For time-varying BC, we'd need a Box<dyn Fn(S) -> S + Send + Sync>
    // but for simplicity, keeping constant for now
}

impl<S: Scalar> DirichletBC<S> {
    /// Create a constant Dirichlet BC.
    pub fn new(value: S) -> Self {
        Self {
            value: DirichletValue::Constant(value),
        }
    }
}

impl<S: Scalar> BoundaryCondition<S> for DirichletBC<S> {
    fn apply(&self, t: S, _interior_value: S, _dx: S) -> S {
        self.value(t).unwrap()
    }

    fn value(&self, _t: S) -> Option<S> {
        match &self.value {
            DirichletValue::Constant(v) => Some(*v),
        }
    }

    fn is_dirichlet(&self) -> bool {
        true
    }
}

/// Neumann boundary condition: ∂u/∂n = g(t) at the boundary.
#[derive(Clone, Debug)]
pub struct NeumannBC<S: Scalar> {
    /// Normal derivative value (positive = outward flux)
    flux: S,
}

impl<S: Scalar> NeumannBC<S> {
    /// Create a constant Neumann BC.
    pub fn new(flux: S) -> Self {
        Self { flux }
    }

    /// Create a zero-flux (insulated) BC.
    pub fn zero_flux() -> Self {
        Self { flux: S::ZERO }
    }
}

impl<S: Scalar> BoundaryCondition<S> for NeumannBC<S> {
    fn apply(&self, _t: S, interior_value: S, dx: S) -> S {
        // Ghost point value for central difference:
        // (u[1] - u[-1]) / (2*dx) = flux
        // => u[-1] = u[1] - 2*dx*flux
        interior_value - S::from_f64(2.0) * dx * self.flux
    }

    fn flux(&self, _t: S) -> Option<S> {
        Some(self.flux)
    }

    fn is_dirichlet(&self) -> bool {
        false
    }
}

/// Robin boundary condition: α*u + β*∂u/∂n = g at the boundary.
#[derive(Clone, Debug)]
pub struct RobinBC<S: Scalar> {
    /// Coefficient for u
    pub alpha: S,
    /// Coefficient for ∂u/∂n
    pub beta: S,
    /// Right-hand side value
    pub gamma: S,
}

impl<S: Scalar> RobinBC<S> {
    /// Create a Robin BC: α*u + β*∂u/∂n = γ.
    pub fn new(alpha: S, beta: S, gamma: S) -> Self {
        Self { alpha, beta, gamma }
    }

    /// Create a convective BC: h*(u - u_ambient) = -k*∂u/∂n.
    /// This is equivalent to: (h/k)*u + ∂u/∂n = (h/k)*u_ambient.
    pub fn convective(h_over_k: S, u_ambient: S) -> Self {
        Self {
            alpha: h_over_k,
            beta: S::ONE,
            gamma: h_over_k * u_ambient,
        }
    }
}

impl<S: Scalar> BoundaryCondition<S> for RobinBC<S> {
    fn apply(&self, _t: S, interior_value: S, dx: S) -> S {
        // At left boundary: α*u[0] + β*(u[1] - u[-1])/(2*dx) = γ
        // Solve for ghost point u[-1]:
        // u[-1] = u[1] - (2*dx/β)*(γ - α*u[0])
        // We don't have u[0] here, so this is an approximation
        // using interior_value ≈ u[1]:
        //
        // For accuracy, the MOL solver should handle Robin BCs specially.
        // This is a simple approximation.
        let flux_approx = (self.gamma - self.alpha * interior_value) / self.beta;
        interior_value - S::from_f64(2.0) * dx * flux_approx
    }

    fn is_dirichlet(&self) -> bool {
        // Robin is neither pure Dirichlet nor pure Neumann
        self.beta.abs() < S::from_f64(1e-10)
    }
}

/// Periodic boundary condition.
#[derive(Clone, Debug)]
pub struct PeriodicBC;

impl<S: Scalar> BoundaryCondition<S> for PeriodicBC {
    fn apply(&self, _t: S, interior_value: S, _dx: S) -> S {
        // For periodic BC, the ghost point equals the value from the other end.
        // The MOL solver handles this by wrapping indices.
        interior_value
    }

    fn is_dirichlet(&self) -> bool {
        false
    }
}

/// Generic boundary condition that boxes the specific type.
#[derive(Clone)]
pub struct BoxedBC<S: Scalar> {
    inner: BcType<S>,
}

#[derive(Clone)]
enum BcType<S: Scalar> {
    Dirichlet(DirichletBC<S>),
    Neumann(NeumannBC<S>),
    Robin(RobinBC<S>),
    Periodic,
}

impl<S: Scalar> BoxedBC<S> {
    pub fn dirichlet(value: S) -> Self {
        Self {
            inner: BcType::Dirichlet(DirichletBC::new(value)),
        }
    }

    pub fn neumann(flux: S) -> Self {
        Self {
            inner: BcType::Neumann(NeumannBC::new(flux)),
        }
    }

    pub fn robin(alpha: S, beta: S, gamma: S) -> Self {
        Self {
            inner: BcType::Robin(RobinBC::new(alpha, beta, gamma)),
        }
    }

    pub fn periodic() -> Self {
        Self {
            inner: BcType::Periodic,
        }
    }
}

impl<S: Scalar> BoundaryCondition<S> for BoxedBC<S> {
    fn apply(&self, t: S, interior_value: S, dx: S) -> S {
        match &self.inner {
            BcType::Dirichlet(bc) => bc.apply(t, interior_value, dx),
            BcType::Neumann(bc) => bc.apply(t, interior_value, dx),
            BcType::Robin(bc) => bc.apply(t, interior_value, dx),
            BcType::Periodic => interior_value,
        }
    }

    fn value(&self, t: S) -> Option<S> {
        match &self.inner {
            BcType::Dirichlet(bc) => bc.value(t),
            _ => None,
        }
    }

    fn flux(&self, t: S) -> Option<S> {
        match &self.inner {
            BcType::Neumann(bc) => bc.flux(t),
            _ => None,
        }
    }

    fn is_dirichlet(&self) -> bool {
        matches!(&self.inner, BcType::Dirichlet(_))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dirichlet_bc() {
        let bc = DirichletBC::new(100.0_f64);
        assert!(bc.is_dirichlet());
        assert!((bc.value(0.0).unwrap() - 100.0).abs() < 1e-10);
        assert!((bc.apply(0.0, 50.0, 0.1) - 100.0).abs() < 1e-10);
    }

    #[test]
    fn test_neumann_bc() {
        let bc = NeumannBC::new(10.0_f64);
        assert!(!bc.is_dirichlet());
        assert!((bc.flux(0.0).unwrap() - 10.0).abs() < 1e-10);

        // Ghost point: u[-1] = u[1] - 2*dx*flux
        // With interior_value = 50, dx = 0.1, flux = 10:
        // ghost = 50 - 2*0.1*10 = 50 - 2 = 48
        let ghost = bc.apply(0.0, 50.0, 0.1);
        assert!((ghost - 48.0).abs() < 1e-10);
    }

    #[test]
    fn test_zero_flux_bc() {
        let bc = NeumannBC::<f64>::zero_flux();
        assert!((bc.flux(0.0).unwrap()).abs() < 1e-10);

        // Zero flux: ghost = interior
        let ghost = bc.apply(0.0, 50.0, 0.1);
        assert!((ghost - 50.0).abs() < 1e-10);
    }

    #[test]
    fn test_boxed_bc() {
        let bc = BoxedBC::dirichlet(100.0_f64);
        assert!(bc.is_dirichlet());
        assert!((bc.value(0.0).unwrap() - 100.0).abs() < 1e-10);

        let bc = BoxedBC::neumann(10.0_f64);
        assert!(!bc.is_dirichlet());
        assert!((bc.flux(0.0).unwrap() - 10.0).abs() < 1e-10);
    }
}

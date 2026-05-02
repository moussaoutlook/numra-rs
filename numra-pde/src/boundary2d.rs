//! Boundary conditions for 2D and 3D PDE problems.
//!
//! Author: Moussa Leblouba
//! Date: 9 February 2026
//! Modified: 2 May 2026

use crate::boundary::BoxedBC;
use numra_core::Scalar;

/// Boundary conditions for a 2D rectangular domain.
///
/// Specifies one BC per edge of the rectangle.
#[derive(Clone)]
pub struct BoundaryConditions2D<S: Scalar> {
    /// Left boundary (x = x_min)
    pub x_min: BoxedBC<S>,
    /// Right boundary (x = x_max)
    pub x_max: BoxedBC<S>,
    /// Bottom boundary (y = y_min)
    pub y_min: BoxedBC<S>,
    /// Top boundary (y = y_max)
    pub y_max: BoxedBC<S>,
}

impl<S: Scalar> BoundaryConditions2D<S> {
    /// Create BCs with Dirichlet = 0 on all sides.
    pub fn all_zero_dirichlet() -> Self {
        Self {
            x_min: BoxedBC::dirichlet(S::ZERO),
            x_max: BoxedBC::dirichlet(S::ZERO),
            y_min: BoxedBC::dirichlet(S::ZERO),
            y_max: BoxedBC::dirichlet(S::ZERO),
        }
    }

    /// Create BCs with the same Dirichlet value on all sides.
    pub fn all_dirichlet(value: S) -> Self {
        Self {
            x_min: BoxedBC::dirichlet(value),
            x_max: BoxedBC::dirichlet(value),
            y_min: BoxedBC::dirichlet(value),
            y_max: BoxedBC::dirichlet(value),
        }
    }

    /// Create BCs with zero-flux Neumann on all sides.
    pub fn all_zero_neumann() -> Self {
        Self {
            x_min: BoxedBC::neumann(S::ZERO),
            x_max: BoxedBC::neumann(S::ZERO),
            y_min: BoxedBC::neumann(S::ZERO),
            y_max: BoxedBC::neumann(S::ZERO),
        }
    }
}

/// Boundary conditions for a 3D rectangular domain.
///
/// Specifies one BC per face of the box.
#[derive(Clone)]
pub struct BoundaryConditions3D<S: Scalar> {
    /// Left face (x = x_min)
    pub x_min: BoxedBC<S>,
    /// Right face (x = x_max)
    pub x_max: BoxedBC<S>,
    /// Front face (y = y_min)
    pub y_min: BoxedBC<S>,
    /// Back face (y = y_max)
    pub y_max: BoxedBC<S>,
    /// Bottom face (z = z_min)
    pub z_min: BoxedBC<S>,
    /// Top face (z = z_max)
    pub z_max: BoxedBC<S>,
}

impl<S: Scalar> BoundaryConditions3D<S> {
    /// Create BCs with Dirichlet = 0 on all faces.
    pub fn all_zero_dirichlet() -> Self {
        Self {
            x_min: BoxedBC::dirichlet(S::ZERO),
            x_max: BoxedBC::dirichlet(S::ZERO),
            y_min: BoxedBC::dirichlet(S::ZERO),
            y_max: BoxedBC::dirichlet(S::ZERO),
            z_min: BoxedBC::dirichlet(S::ZERO),
            z_max: BoxedBC::dirichlet(S::ZERO),
        }
    }

    /// Create BCs with the same Dirichlet value on all faces.
    pub fn all_dirichlet(value: S) -> Self {
        Self {
            x_min: BoxedBC::dirichlet(value),
            x_max: BoxedBC::dirichlet(value),
            y_min: BoxedBC::dirichlet(value),
            y_max: BoxedBC::dirichlet(value),
            z_min: BoxedBC::dirichlet(value),
            z_max: BoxedBC::dirichlet(value),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::boundary::BoundaryCondition;

    #[test]
    fn test_bc2d_all_zero_dirichlet() {
        let bc = BoundaryConditions2D::<f64>::all_zero_dirichlet();
        assert!(bc.x_min.is_dirichlet());
        assert!(bc.x_max.is_dirichlet());
        assert!(bc.y_min.is_dirichlet());
        assert!(bc.y_max.is_dirichlet());
        assert!((bc.x_min.value(0.0).unwrap()).abs() < 1e-10);
    }

    #[test]
    fn test_bc2d_all_dirichlet() {
        let bc = BoundaryConditions2D::all_dirichlet(5.0_f64);
        assert!((bc.x_min.value(0.0).unwrap() - 5.0).abs() < 1e-10);
        assert!((bc.y_max.value(0.0).unwrap() - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_bc2d_all_zero_neumann() {
        let bc = BoundaryConditions2D::<f64>::all_zero_neumann();
        assert!(!bc.x_min.is_dirichlet());
        assert!(!bc.y_max.is_dirichlet());
    }

    #[test]
    fn test_bc3d_all_zero_dirichlet() {
        let bc = BoundaryConditions3D::<f64>::all_zero_dirichlet();
        assert!(bc.x_min.is_dirichlet());
        assert!(bc.z_max.is_dirichlet());
        assert!((bc.z_min.value(0.0).unwrap()).abs() < 1e-10);
    }
}

//! Moving boundary infrastructure for problems like the Stefan problem.
//!
//! Supports:
//! - Fixed and moving boundaries
//! - Stefan condition (phase change)
//! - Coordinate transforms to fixed computational domain
//!
//! Author: Moussa Leblouba
//! Date: 2 February 2026
//! Modified: 2 May 2026

use numra_core::Scalar;

/// 1D domain with possibly moving boundaries.
#[derive(Clone, Debug)]
pub struct Domain1D<S: Scalar> {
    /// Left boundary
    pub left: Bound<S>,
    /// Right boundary
    pub right: Bound<S>,
    /// Number of grid points
    pub n_points: usize,
}

impl<S: Scalar> Domain1D<S> {
    /// Create a fixed domain.
    pub fn fixed(x_min: S, x_max: S, n_points: usize) -> Self {
        Self {
            left: Bound::Fixed(x_min),
            right: Bound::Fixed(x_max),
            n_points,
        }
    }

    /// Create a domain with moving right boundary (common for Stefan problems).
    pub fn with_moving_right(x_min: S, s_initial: S, n_points: usize) -> Self {
        Self {
            left: Bound::Fixed(x_min),
            right: Bound::Moving(MovingBound::new(s_initial)),
            n_points,
        }
    }

    /// Get current domain bounds.
    pub fn bounds(&self) -> (S, S) {
        (self.left.position(), self.right.position())
    }

    /// Get current domain length.
    pub fn length(&self) -> S {
        let (lo, hi) = self.bounds();
        hi - lo
    }

    /// Generate grid points for current domain configuration.
    pub fn grid_points(&self) -> Vec<S> {
        let (lo, hi) = self.bounds();
        let n = self.n_points;
        (0..n)
            .map(|i| lo + (hi - lo) * S::from_usize(i) / S::from_usize(n - 1))
            .collect()
    }

    /// Get uniform grid spacing.
    pub fn dx(&self) -> S {
        self.length() / S::from_usize(self.n_points - 1)
    }
}

/// A boundary that can be fixed or moving.
#[derive(Clone, Debug)]
pub enum Bound<S: Scalar> {
    /// Fixed boundary at constant position.
    Fixed(S),
    /// Moving boundary.
    Moving(MovingBound<S>),
}

impl<S: Scalar> Bound<S> {
    /// Get current position.
    pub fn position(&self) -> S {
        match self {
            Bound::Fixed(x) => *x,
            Bound::Moving(mb) => mb.position,
        }
    }

    /// Check if boundary is fixed.
    pub fn is_fixed(&self) -> bool {
        matches!(self, Bound::Fixed(_))
    }

    /// Check if boundary is moving.
    pub fn is_moving(&self) -> bool {
        matches!(self, Bound::Moving(_))
    }

    /// Update position (only for moving boundaries).
    pub fn set_position(&mut self, new_pos: S) {
        if let Bound::Moving(mb) = self {
            mb.position = new_pos;
        }
    }
}

/// Moving boundary specification.
#[derive(Clone, Debug)]
pub struct MovingBound<S: Scalar> {
    /// Current position
    pub position: S,
    /// Stefan condition if applicable
    pub stefan: Option<StefanCondition<S>>,
}

impl<S: Scalar> MovingBound<S> {
    /// Create a moving boundary at given position.
    pub fn new(position: S) -> Self {
        Self {
            position,
            stefan: None,
        }
    }

    /// Create a moving boundary with Stefan condition.
    pub fn with_stefan(position: S, stefan: StefanCondition<S>) -> Self {
        Self {
            position,
            stefan: Some(stefan),
        }
    }
}

/// Stefan condition: ds/dt = -κ ∂u/∂n at the moving boundary.
///
/// This models phase change where:
/// - ρL ds/dt = -k ∂T/∂x (energy balance)
/// - κ = k / (ρL) is the Stefan coefficient
#[derive(Clone, Debug)]
pub struct StefanCondition<S: Scalar> {
    /// Stefan coefficient κ = k / (ρL)
    /// where k = thermal conductivity, ρ = density, L = latent heat
    pub coefficient: S,
}

impl<S: Scalar> StefanCondition<S> {
    /// Create a Stefan condition with given coefficient.
    pub fn new(coefficient: S) -> Self {
        Self { coefficient }
    }

    /// Compute boundary velocity from flux.
    ///
    /// ds/dt = -κ * ∂u/∂n
    /// where ∂u/∂n is the normal derivative (positive outward)
    pub fn velocity(&self, flux: S) -> S {
        -self.coefficient * flux
    }
}

/// Coordinate transformation for moving boundary problems.
///
/// Maps physical domain [0, s(t)] to fixed computational domain [0, 1].
/// This avoids remeshing as the boundary moves.
#[derive(Clone)]
pub struct CoordinateTransform<S: Scalar> {
    /// Current boundary position
    pub s: S,
}

impl<S: Scalar> CoordinateTransform<S> {
    /// Create a transform for domain [0, s].
    pub fn new(s: S) -> Self {
        Self { s }
    }

    /// Update boundary position.
    pub fn update(&mut self, s: S) {
        self.s = s;
    }

    /// Transform physical coordinate x to computational ξ.
    /// ξ = x / s(t)
    pub fn physical_to_computational(&self, x: S) -> S {
        x / self.s
    }

    /// Transform computational coordinate ξ to physical x.
    /// x = ξ * s(t)
    pub fn computational_to_physical(&self, xi: S) -> S {
        xi * self.s
    }

    /// Jacobian of the transform: ∂x/∂ξ = s
    pub fn jacobian(&self) -> S {
        self.s
    }

    /// Transform first derivative: ∂u/∂x = (1/s) ∂u/∂ξ
    pub fn transform_d1(&self, du_dxi: S) -> S {
        du_dxi / self.s
    }

    /// Transform second derivative: ∂²u/∂x² = (1/s²) ∂²u/∂ξ²
    pub fn transform_d2(&self, d2u_dxi2: S) -> S {
        d2u_dxi2 / (self.s * self.s)
    }

    /// Additional convective term from moving boundary.
    ///
    /// When using transformed coordinates, the heat equation becomes:
    /// ∂u/∂t = α/s² ∂²u/∂ξ² + (ds/dt * ξ / s) ∂u/∂ξ
    ///
    /// This returns the coefficient for the convective term.
    pub fn convection_coefficient(&self, dsdt: S, xi: S) -> S {
        dsdt * xi / self.s
    }
}

/// Field state for moving boundary problems.
///
/// Contains the solution field along with boundary information.
#[derive(Clone)]
pub struct FieldState1D<S: Scalar> {
    /// Solution values at grid points
    pub values: Vec<S>,
    /// Domain specification
    pub domain: Domain1D<S>,
}

impl<S: Scalar> FieldState1D<S> {
    /// Create a new field state.
    pub fn new(values: Vec<S>, domain: Domain1D<S>) -> Self {
        assert_eq!(values.len(), domain.n_points);
        Self { values, domain }
    }

    /// Get value at index i.
    pub fn get(&self, i: usize) -> S {
        self.values[i]
    }

    /// Set value at index i.
    pub fn set(&mut self, i: usize, value: S) {
        self.values[i] = value;
    }

    /// Compute boundary flux at left boundary.
    /// Uses second-order one-sided difference.
    pub fn flux_left(&self) -> S {
        let dx = self.domain.dx();
        let v = &self.values;
        let n = v.len();
        if n < 3 {
            return (v[1] - v[0]) / dx;
        }
        (S::from_f64(-3.0) * v[0] + S::from_f64(4.0) * v[1] - v[2]) / (S::from_f64(2.0) * dx)
    }

    /// Compute boundary flux at right boundary.
    /// Uses second-order one-sided difference.
    pub fn flux_right(&self) -> S {
        let dx = self.domain.dx();
        let v = &self.values;
        let n = v.len();
        if n < 3 {
            return (v[n - 1] - v[n - 2]) / dx;
        }
        (S::from_f64(3.0) * v[n - 1] - S::from_f64(4.0) * v[n - 2] + v[n - 3])
            / (S::from_f64(2.0) * dx)
    }

    /// Evaluate field at arbitrary point using linear interpolation.
    pub fn evaluate(&self, x: S) -> S {
        let points = self.domain.grid_points();
        let (lo, hi) = self.domain.bounds();

        if x <= lo {
            return self.values[0];
        }
        if x >= hi {
            return *self.values.last().unwrap();
        }

        // Find bracketing indices
        let dx = self.domain.dx();
        let i = ((x - lo) / dx).to_f64().floor() as usize;
        let i = i.min(self.values.len() - 2);

        let x0 = points[i];
        let x1 = points[i + 1];
        let t = (x - x0) / (x1 - x0);

        self.values[i] * (S::ONE - t) + self.values[i + 1] * t
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fixed_domain() {
        let domain = Domain1D::fixed(0.0_f64, 1.0, 11);
        assert!((domain.length() - 1.0).abs() < 1e-10);
        assert!((domain.dx() - 0.1).abs() < 1e-10);

        let points = domain.grid_points();
        assert_eq!(points.len(), 11);
        assert!((points[0]).abs() < 1e-10);
        assert!((points[10] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_moving_domain() {
        let mut domain = Domain1D::with_moving_right(0.0_f64, 0.5, 11);
        assert!((domain.length() - 0.5).abs() < 1e-10);

        // Move boundary
        domain.right.set_position(0.8);
        assert!((domain.length() - 0.8).abs() < 1e-10);
    }

    #[test]
    fn test_stefan_condition() {
        let stefan = StefanCondition::new(0.1_f64);

        // flux = 10 (into the domain) → ds/dt = -0.1 * 10 = -1 (boundary retracts)
        let velocity = stefan.velocity(10.0);
        assert!((velocity - (-1.0)).abs() < 1e-10);

        // flux = -5 (out of domain) → ds/dt = -0.1 * (-5) = 0.5 (boundary advances)
        let velocity = stefan.velocity(-5.0);
        assert!((velocity - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_coordinate_transform() {
        let transform = CoordinateTransform::new(2.0_f64);

        // Physical to computational: x = 1 → ξ = 0.5
        let xi = transform.physical_to_computational(1.0);
        assert!((xi - 0.5).abs() < 1e-10);

        // Computational to physical: ξ = 0.5 → x = 1
        let x = transform.computational_to_physical(0.5);
        assert!((x - 1.0).abs() < 1e-10);

        // Jacobian
        assert!((transform.jacobian() - 2.0).abs() < 1e-10);

        // Derivative transforms
        // ∂u/∂x = (1/2) ∂u/∂ξ
        let du_dx = transform.transform_d1(4.0);
        assert!((du_dx - 2.0).abs() < 1e-10);

        // ∂²u/∂x² = (1/4) ∂²u/∂ξ²
        let d2u_dx2 = transform.transform_d2(8.0);
        assert!((d2u_dx2 - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_field_state_flux() {
        // Linear field: u = x on [0, 1]
        let domain = Domain1D::fixed(0.0_f64, 1.0, 11);
        let values: Vec<f64> = (0..11).map(|i| i as f64 * 0.1).collect();
        let field = FieldState1D::new(values, domain);

        // For u = x, du/dx = 1 everywhere
        let flux_left = field.flux_left();
        let flux_right = field.flux_right();

        assert!((flux_left - 1.0).abs() < 0.01);
        assert!((flux_right - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_field_evaluate() {
        let domain = Domain1D::fixed(0.0_f64, 1.0, 11);
        let values: Vec<f64> = (0..11).map(|i| i as f64 * 0.1).collect();
        let field = FieldState1D::new(values, domain);

        // Evaluate at grid points
        assert!((field.evaluate(0.0) - 0.0).abs() < 1e-10);
        assert!((field.evaluate(0.5) - 0.5).abs() < 1e-10);
        assert!((field.evaluate(1.0) - 1.0).abs() < 1e-10);

        // Evaluate between grid points
        assert!((field.evaluate(0.25) - 0.25).abs() < 0.01);
        assert!((field.evaluate(0.75) - 0.75).abs() < 0.01);
    }
}

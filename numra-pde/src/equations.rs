//! Common PDE equations.
//!
//! Author: Moussa Leblouba
//! Date: 3 February 2026
//! Modified: 2 May 2026

use crate::mol::PdeSystem;
use numra_core::Scalar;

/// 1D Heat (diffusion) equation: ∂u/∂t = α ∂²u/∂x².
#[derive(Clone, Debug)]
pub struct HeatEquation1D<S: Scalar> {
    /// Thermal diffusivity α
    pub alpha: S,
}

impl<S: Scalar> HeatEquation1D<S> {
    pub fn new(alpha: S) -> Self {
        Self { alpha }
    }
}

impl<S: Scalar> PdeSystem<S> for HeatEquation1D<S> {
    fn rhs(&self, _t: S, _x: S, _u: S, _du_dx: S, d2u_dx2: S) -> S {
        self.alpha * d2u_dx2
    }
}

/// 1D Diffusion-Reaction equation: ∂u/∂t = D ∂²u/∂x² + f(u).
///
/// Common examples:
/// - Fisher equation: f(u) = u(1-u)
/// - Allen-Cahn: f(u) = u - u³
/// - Logistic: f(u) = ru(1 - u/K)
#[derive(Clone)]
pub struct DiffusionReaction1D<S: Scalar, R>
where
    R: Fn(S) -> S + Clone,
{
    /// Diffusion coefficient D
    pub diffusion: S,
    /// Reaction term f(u)
    pub reaction: R,
}

impl<S: Scalar, R: Fn(S) -> S + Clone> DiffusionReaction1D<S, R> {
    pub fn new(diffusion: S, reaction: R) -> Self {
        Self {
            diffusion,
            reaction,
        }
    }

    /// Create Fisher equation: D ∂²u/∂x² + ru(1-u).
    pub fn fisher(diffusion: S, growth_rate: S) -> DiffusionReaction1D<S, impl Fn(S) -> S + Clone>
    where
        S: Copy,
    {
        let r = growth_rate;
        DiffusionReaction1D::new(diffusion, move |u: S| r * u * (S::ONE - u))
    }

    /// Create Allen-Cahn equation: D ∂²u/∂x² + u - u³.
    pub fn allen_cahn(diffusion: S) -> DiffusionReaction1D<S, impl Fn(S) -> S + Clone>
    where
        S: Copy,
    {
        DiffusionReaction1D::new(diffusion, |u: S| u - u * u * u)
    }
}

impl<S: Scalar, R: Fn(S) -> S + Clone> PdeSystem<S> for DiffusionReaction1D<S, R> {
    fn rhs(&self, _t: S, _x: S, u: S, _du_dx: S, d2u_dx2: S) -> S {
        self.diffusion * d2u_dx2 + (self.reaction)(u)
    }
}

/// 1D Advection-Diffusion equation: ∂u/∂t = D ∂²u/∂x² - v ∂u/∂x.
///
/// Infrastructure for advection-diffusion problems.
#[allow(dead_code)]
#[derive(Clone, Debug)]
pub struct AdvectionDiffusion1D<S: Scalar> {
    /// Diffusion coefficient
    pub diffusion: S,
    /// Advection velocity
    pub velocity: S,
}

#[allow(dead_code)]
impl<S: Scalar> AdvectionDiffusion1D<S> {
    pub fn new(diffusion: S, velocity: S) -> Self {
        Self {
            diffusion,
            velocity,
        }
    }
}

impl<S: Scalar> PdeSystem<S> for AdvectionDiffusion1D<S> {
    fn rhs(&self, _t: S, _x: S, _u: S, du_dx: S, d2u_dx2: S) -> S {
        self.diffusion * d2u_dx2 - self.velocity * du_dx
    }
}

/// 1D Burgers' equation: ∂u/∂t + u ∂u/∂x = ν ∂²u/∂x².
///
/// Infrastructure for Burgers' equation problems.
#[allow(dead_code)]
#[derive(Clone, Debug)]
pub struct Burgers1D<S: Scalar> {
    /// Viscosity coefficient
    pub viscosity: S,
}

#[allow(dead_code)]
impl<S: Scalar> Burgers1D<S> {
    pub fn new(viscosity: S) -> Self {
        Self { viscosity }
    }
}

impl<S: Scalar> PdeSystem<S> for Burgers1D<S> {
    fn rhs(&self, _t: S, _x: S, u: S, du_dx: S, d2u_dx2: S) -> S {
        self.viscosity * d2u_dx2 - u * du_dx
    }
}

/// 1D Wave equation (first-order form):
/// ∂u/∂t = v
/// ∂v/∂t = c² ∂²u/∂x²
///
/// For MOL, we treat this as a system of equations.
///
/// Infrastructure for wave equation problems.
#[allow(dead_code)]
#[derive(Clone, Debug)]
pub struct Wave1D<S: Scalar> {
    /// Wave speed c
    pub wave_speed: S,
}

#[allow(dead_code)]
impl<S: Scalar> Wave1D<S> {
    pub fn new(wave_speed: S) -> Self {
        Self { wave_speed }
    }
}

// Note: Wave equation requires special handling as a system
// Implementation would need the PdeSystem trait to support systems

/// Heat equation with source term: ∂u/∂t = α ∂²u/∂x² + Q(x,t).
///
/// Infrastructure for heat equation with source term problems.
#[allow(dead_code)]
#[derive(Clone)]
pub struct HeatWithSource1D<S: Scalar, Q>
where
    Q: Fn(S, S) -> S + Clone,
{
    /// Thermal diffusivity
    pub alpha: S,
    /// Source term Q(x, t)
    pub source: Q,
}

#[allow(dead_code)]
impl<S: Scalar, Q: Fn(S, S) -> S + Clone> HeatWithSource1D<S, Q> {
    pub fn new(alpha: S, source: Q) -> Self {
        Self { alpha, source }
    }
}

impl<S: Scalar, Q: Fn(S, S) -> S + Clone> PdeSystem<S> for HeatWithSource1D<S, Q> {
    fn rhs(&self, t: S, x: S, _u: S, _du_dx: S, d2u_dx2: S) -> S {
        self.alpha * d2u_dx2 + (self.source)(x, t)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_heat_equation() {
        let heat = HeatEquation1D::new(0.1_f64);
        // ∂²u/∂x² = 2 (for u = x²)
        let rhs = heat.rhs(0.0, 0.5, 0.25, 1.0, 2.0);
        assert!((rhs - 0.2).abs() < 1e-10);
    }

    #[test]
    fn test_fisher_equation() {
        // Fisher equation manually: D=0.1, r=1.0
        let fisher = DiffusionReaction1D::new(0.1_f64, |u: f64| u * (1.0 - u));
        // At u = 0.5: reaction = 0.5 * (1 - 0.5) = 0.25
        // d2u = 2, diffusion = 0.1 * 2 = 0.2
        // Total = 0.2 + 0.25 = 0.45
        let rhs = fisher.rhs(0.0, 0.5, 0.5, 0.0, 2.0);
        assert!((rhs - 0.45).abs() < 1e-10);
    }

    #[test]
    fn test_burgers_equation() {
        let burgers = Burgers1D::new(0.1_f64);
        // At u = 1, du/dx = 2, d2u/dx2 = 0
        // RHS = 0.1 * 0 - 1 * 2 = -2
        let rhs = burgers.rhs(0.0, 0.5, 1.0, 2.0, 0.0);
        assert!((rhs - (-2.0)).abs() < 1e-10);
    }

    #[test]
    fn test_heat_with_source() {
        let heat = HeatWithSource1D::new(0.1_f64, |x: f64, _t: f64| x * x);
        // At x = 0.5: source = 0.25
        // d2u = 2, diffusion = 0.1 * 2 = 0.2
        // Total = 0.2 + 0.25 = 0.45
        let rhs = heat.rhs(0.0, 0.5, 0.0, 0.0, 2.0);
        assert!((rhs - 0.45).abs() < 1e-10);
    }
}

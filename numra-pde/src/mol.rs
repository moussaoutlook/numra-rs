//! Method of Lines (MOL) for converting PDEs to ODE systems.
//!
//! The Method of Lines discretizes spatial derivatives while leaving
//! time continuous, converting a PDE to a system of ODEs.
//!
//! Author: Moussa Leblouba
//! Date: 4 February 2026
//! Modified: 2 May 2026

use crate::boundary::BoundaryCondition;
use crate::discretize::FDM;
use crate::grid::Grid1D;
use numra_core::Scalar;
use numra_ode::OdeSystem;

/// Trait for 1D PDE systems that can be solved via MOL.
///
/// The PDE has the form:
/// ```text
/// ∂u/∂t = F(t, x, u, ∂u/∂x, ∂²u/∂x², ...)
/// ```
pub trait PdeSystem<S: Scalar> {
    /// Compute the RHS of the PDE at a single spatial point.
    ///
    /// # Arguments
    /// * `t` - Current time
    /// * `x` - Spatial coordinate
    /// * `u` - Solution value at this point
    /// * `du_dx` - First spatial derivative
    /// * `d2u_dx2` - Second spatial derivative
    fn rhs(&self, t: S, x: S, u: S, du_dx: S, d2u_dx2: S) -> S;

    /// Number of solution components (for systems).
    fn n_components(&self) -> usize {
        1
    }
}

/// Method of Lines system that wraps a PDE for use with ODE solvers.
///
/// This converts a PDE to an ODE system by:
/// 1. Storing solution values at interior grid points
/// 2. Computing spatial derivatives using FDM
/// 3. Applying boundary conditions
#[derive(Clone)]
pub struct MOLSystem<
    S: Scalar,
    P: PdeSystem<S>,
    BcL: BoundaryCondition<S>,
    BcR: BoundaryCondition<S>,
> {
    /// The PDE system
    pde: P,
    /// Spatial grid
    grid: Grid1D<S>,
    /// Left boundary condition
    bc_left: BcL,
    /// Right boundary condition
    bc_right: BcR,
}

impl<S: Scalar, P: PdeSystem<S>, BcL: BoundaryCondition<S>, BcR: BoundaryCondition<S>>
    MOLSystem<S, P, BcL, BcR>
{
    /// Create a new MOL system.
    pub fn new(pde: P, grid: Grid1D<S>, bc_left: BcL, bc_right: BcR) -> Self {
        Self {
            pde,
            grid,
            bc_left,
            bc_right,
        }
    }

    /// Get the spatial grid.
    pub fn grid(&self) -> &Grid1D<S> {
        &self.grid
    }

    /// Number of interior points (ODE dimension).
    pub fn n_interior(&self) -> usize {
        self.grid.n_interior()
    }

    /// Build full solution array including boundaries.
    ///
    /// Takes interior values and adds boundary values.
    pub fn build_full_solution(&self, t: S, u_interior: &[S]) -> Vec<S> {
        let n = self.grid.len();
        let mut u_full = vec![S::ZERO; n];

        // Left boundary
        if self.bc_left.is_dirichlet() {
            u_full[0] = self.bc_left.value(t).unwrap();
        } else {
            // Ghost point approach for Neumann
            let dx = self.grid.dx_uniform();
            u_full[0] = self.bc_left.apply(t, u_interior[0], dx);
        }

        // Interior points
        for (i, &val) in u_interior.iter().enumerate() {
            u_full[i + 1] = val;
        }

        // Right boundary
        if self.bc_right.is_dirichlet() {
            u_full[n - 1] = self.bc_right.value(t).unwrap();
        } else {
            let dx = self.grid.dx_uniform();
            u_full[n - 1] = self.bc_right.apply(t, u_interior[u_interior.len() - 1], dx);
        }

        u_full
    }
}

impl<S: Scalar, P: PdeSystem<S>, BcL: BoundaryCondition<S>, BcR: BoundaryCondition<S>> OdeSystem<S>
    for MOLSystem<S, P, BcL, BcR>
{
    fn dim(&self) -> usize {
        self.n_interior() * self.pde.n_components()
    }

    fn rhs(&self, t: S, y: &[S], dydt: &mut [S]) {
        let n_interior = self.n_interior();
        let _n_full = self.grid.len();
        let dx = self.grid.dx_uniform();

        // Build full solution with boundary values
        let u_full = self.build_full_solution(t, y);
        let points = self.grid.points();

        // Compute derivatives at each interior point
        for i in 0..n_interior {
            let idx_full = i + 1; // Index in full array

            let x = points[idx_full];
            let u = u_full[idx_full];

            // Compute spatial derivatives
            let du_dx = FDM::d1_central(&u_full, dx, idx_full);
            let d2u_dx2 = FDM::d2_central(&u_full, dx, idx_full);

            // Evaluate PDE RHS
            dydt[i] = self.pde.rhs(t, x, u, du_dx, d2u_dx2);
        }
    }
}

/// Simple heat equation PDE: ∂u/∂t = α ∂²u/∂x².
///
/// Infrastructure for MOL-based heat equation solvers.
#[allow(dead_code)]
#[derive(Clone, Debug)]
pub struct HeatEquationPde<S: Scalar> {
    /// Thermal diffusivity
    pub alpha: S,
}

#[allow(dead_code)]
impl<S: Scalar> HeatEquationPde<S> {
    pub fn new(alpha: S) -> Self {
        Self { alpha }
    }
}

impl<S: Scalar> PdeSystem<S> for HeatEquationPde<S> {
    fn rhs(&self, _t: S, _x: S, _u: S, _du_dx: S, d2u_dx2: S) -> S {
        self.alpha * d2u_dx2
    }
}

/// Diffusion-reaction PDE: ∂u/∂t = D ∂²u/∂x² + R(u).
///
/// Infrastructure for MOL-based diffusion-reaction solvers.
#[allow(dead_code)]
#[derive(Clone)]
pub struct DiffusionReactionPde<S: Scalar, F>
where
    F: Fn(S) -> S + Clone,
{
    /// Diffusion coefficient
    pub diffusion: S,
    /// Reaction term R(u)
    pub reaction: F,
}

#[allow(dead_code)]
impl<S: Scalar, F: Fn(S) -> S + Clone> DiffusionReactionPde<S, F> {
    pub fn new(diffusion: S, reaction: F) -> Self {
        Self {
            diffusion,
            reaction,
        }
    }
}

impl<S: Scalar, F: Fn(S) -> S + Clone> PdeSystem<S> for DiffusionReactionPde<S, F> {
    fn rhs(&self, _t: S, _x: S, u: S, _du_dx: S, d2u_dx2: S) -> S {
        self.diffusion * d2u_dx2 + (self.reaction)(u)
    }
}

/// Advection-diffusion PDE: ∂u/∂t = D ∂²u/∂x² - v ∂u/∂x.
///
/// Infrastructure for MOL-based advection-diffusion solvers.
#[allow(dead_code)]
#[derive(Clone, Debug)]
pub struct AdvectionDiffusionPde<S: Scalar> {
    /// Diffusion coefficient
    pub diffusion: S,
    /// Advection velocity
    pub velocity: S,
}

#[allow(dead_code)]
impl<S: Scalar> AdvectionDiffusionPde<S> {
    pub fn new(diffusion: S, velocity: S) -> Self {
        Self {
            diffusion,
            velocity,
        }
    }
}

impl<S: Scalar> PdeSystem<S> for AdvectionDiffusionPde<S> {
    fn rhs(&self, _t: S, _x: S, _u: S, du_dx: S, d2u_dx2: S) -> S {
        self.diffusion * d2u_dx2 - self.velocity * du_dx
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::boundary::DirichletBC;
    use numra_ode::{DoPri5, Solver, SolverOptions};

    #[test]
    fn test_mol_system_dim() {
        let grid = Grid1D::uniform(0.0, 1.0, 11);
        let pde = HeatEquationPde::new(0.01_f64);
        let bc_left = DirichletBC::new(1.0);
        let bc_right = DirichletBC::new(0.0);

        let mol = MOLSystem::new(pde, grid, bc_left, bc_right);
        assert_eq!(mol.dim(), 9); // 11 - 2 boundary points
    }

    #[test]
    fn test_heat_equation_dirichlet() {
        // Heat equation with T(0) = 1, T(1) = 0
        // Initial: T(x,0) = 1 - x (linear)
        // Should diffuse toward steady state (linear)

        let grid = Grid1D::uniform(0.0, 1.0, 21);
        let pde = HeatEquationPde::new(0.01_f64);
        let bc_left = DirichletBC::new(1.0);
        let bc_right = DirichletBC::new(0.0);

        let mol = MOLSystem::new(pde, grid.clone(), bc_left, bc_right);

        // Initial condition: T = 1 - x (interior points only)
        let u0: Vec<f64> = grid.interior_points().iter().map(|&x| 1.0 - x).collect();

        let options = SolverOptions::default().rtol(1e-6).atol(1e-9);
        let result = DoPri5::solve(&mol, 0.0, 1.0, &u0, &options).unwrap();

        // After diffusion, should still be approximately linear
        let y_final = result.y_final().unwrap();
        for (i, &y) in y_final.iter().enumerate() {
            let x = grid.interior_points()[i];
            let expected = 1.0 - x;
            assert!(
                (y - expected).abs() < 0.1,
                "At x={}: y={}, expected={}",
                x,
                y,
                expected
            );
        }
    }

    #[test]
    fn test_diffusion_reaction() {
        // Fisher equation: ∂u/∂t = D ∂²u/∂x² + u(1-u)
        let grid = Grid1D::uniform(0.0, 1.0, 21);
        let pde = DiffusionReactionPde::new(0.01_f64, |u: f64| u * (1.0 - u));
        let bc_left = DirichletBC::new(1.0);
        let bc_right = DirichletBC::new(0.0);

        let mol = MOLSystem::new(pde, grid.clone(), bc_left, bc_right);

        let u0: Vec<f64> = grid
            .interior_points()
            .iter()
            .map(|&x| if x < 0.3 { 1.0 } else { 0.0 })
            .collect();

        let options = SolverOptions::default().rtol(1e-6);
        let result = DoPri5::solve(&mol, 0.0, 0.5, &u0, &options).unwrap();

        assert!(result.success);
        // Just check it ran successfully
        let y_final = result.y_final().unwrap();
        for &y in &y_final {
            assert!(
                (-0.1..=1.1).contains(&y),
                "Solution out of expected range: {}",
                y
            );
        }
    }
}

//! Common 2D PDE equation builders.
//!
//! Convenience constructors for standard 2D PDEs using [`MOLSystem2D`].
//!
//! Author: Moussa Leblouba
//! Date: 9 February 2026
//! Modified: 2 May 2026

use crate::boundary2d::BoundaryConditions2D;
use crate::grid::Grid2D;
use crate::mol2d::MOLSystem2D;
use crate::sparse_assembly::{Operator2DCoefficients, SparseScalar};

/// 2D Heat (diffusion) equation: u_t = alpha * (u_xx + u_yy).
pub struct HeatEquation2D;

impl HeatEquation2D {
    /// Build a 2D heat equation MOL system.
    pub fn build<S: SparseScalar>(
        grid: Grid2D<S>,
        alpha: S,
        bc: &BoundaryConditions2D<S>,
    ) -> MOLSystem2D<S> {
        MOLSystem2D::heat(grid, alpha, bc)
    }
}

/// 2D Advection-Diffusion: u_t = D*(u_xx + u_yy) - vx*u_x - vy*u_y.
pub struct AdvectionDiffusion2D;

impl AdvectionDiffusion2D {
    /// Build a 2D advection-diffusion MOL system.
    pub fn build<S: SparseScalar>(
        grid: Grid2D<S>,
        diffusion: S,
        vx: S,
        vy: S,
        bc: &BoundaryConditions2D<S>,
    ) -> MOLSystem2D<S> {
        let coeffs = Operator2DCoefficients::advection_diffusion(diffusion, vx, vy);
        MOLSystem2D::with_operator(grid, &coeffs, bc)
    }
}

/// 2D Reaction-Diffusion: u_t = D*(u_xx + u_yy) + R(t, x, y, u).
pub struct ReactionDiffusion2D;

impl ReactionDiffusion2D {
    /// Build a 2D reaction-diffusion MOL system.
    pub fn build<S, R>(
        grid: Grid2D<S>,
        diffusion: S,
        bc: &BoundaryConditions2D<S>,
        reaction: R,
    ) -> MOLSystem2D<S>
    where
        S: SparseScalar,
        R: Fn(S, S, S, S) -> S + Send + Sync + 'static,
    {
        MOLSystem2D::heat(grid, diffusion, bc).with_reaction(reaction)
    }

    /// Build a 2D Fisher equation: u_t = D*(u_xx + u_yy) + r*u*(1-u).
    pub fn fisher<S: SparseScalar>(
        grid: Grid2D<S>,
        diffusion: S,
        growth_rate: S,
        bc: &BoundaryConditions2D<S>,
    ) -> MOLSystem2D<S> {
        let r = growth_rate;
        MOLSystem2D::heat(grid, diffusion, bc)
            .with_reaction(move |_t, _x, _y, u| r * u * (S::ONE - u))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use numra_ode::{DoPri5, OdeSystem, Solver, SolverOptions};

    #[test]
    fn test_heat_equation_2d() {
        let grid = Grid2D::uniform(0.0, 1.0, 11, 0.0, 1.0, 11);
        let bc = BoundaryConditions2D::all_zero_dirichlet();
        let mol = HeatEquation2D::build(grid, 0.01_f64, &bc);
        assert_eq!(mol.dim(), 81);
    }

    #[test]
    fn test_advection_diffusion_2d() {
        let grid = Grid2D::uniform(0.0, 1.0, 11, 0.0, 1.0, 11);
        let bc = BoundaryConditions2D::all_zero_dirichlet();
        let mol = AdvectionDiffusion2D::build(grid, 0.01, 1.0, 0.0, &bc);
        assert_eq!(mol.dim(), 81);

        // Should run without error
        let u0 = vec![0.0; 81];
        let options = SolverOptions::default().rtol(1e-4);
        let result = DoPri5::solve(&mol, 0.0, 0.01, &u0, &options).unwrap();
        assert!(result.success);
    }

    #[test]
    fn test_fisher_2d() {
        let n = 11;
        let grid = Grid2D::uniform(0.0, 1.0, n, 0.0, 1.0, n);
        let bc = BoundaryConditions2D::all_zero_dirichlet();
        let mol = ReactionDiffusion2D::fisher(grid.clone(), 0.01, 1.0, &bc);

        let nx_int = n - 2;
        let ny_int = n - 2;
        let n_int = nx_int * ny_int;

        // IC: small perturbation in center
        let mut u0 = vec![0.0; n_int];
        for jj in 0..ny_int {
            for ii in 0..nx_int {
                let x = grid.x_grid.points()[ii + 1];
                let y = grid.y_grid.points()[jj + 1];
                let r2 = (x - 0.5) * (x - 0.5) + (y - 0.5) * (y - 0.5);
                if r2 < 0.04 {
                    u0[jj * nx_int + ii] = 0.5;
                }
            }
        }

        let options = SolverOptions::default().rtol(1e-4);
        let result = DoPri5::solve(&mol, 0.0, 0.1, &u0, &options).unwrap();
        assert!(result.success);
    }
}

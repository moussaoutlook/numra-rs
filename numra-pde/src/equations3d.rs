//! Common 3D PDE equation builders.
//!
//! Convenience constructors for standard 3D PDEs using [`MOLSystem3D`].
//! Mirrors `equations2d.rs` with a third spatial axis added.
//!
//! Author: Moussa Leblouba
//! Date: 7 May 2026

use crate::boundary2d::BoundaryConditions3D;
use crate::grid::Grid3D;
use crate::mol3d::MOLSystem3D;
use crate::sparse_assembly::{Operator3DCoefficients, SparseScalar};

/// 3D Heat (diffusion) equation: u_t = alpha * (u_xx + u_yy + u_zz).
pub struct HeatEquation3D;

impl HeatEquation3D {
    /// Build a 3D heat equation MOL system.
    pub fn build<S: SparseScalar>(
        grid: Grid3D<S>,
        alpha: S,
        bc: &BoundaryConditions3D<S>,
    ) -> MOLSystem3D<S> {
        MOLSystem3D::heat(grid, alpha, bc)
    }
}

/// 3D Advection-Diffusion: u_t = D*(u_xx + u_yy + u_zz) - vx*u_x - vy*u_y - vz*u_z.
pub struct AdvectionDiffusion3D;

impl AdvectionDiffusion3D {
    /// Build a 3D advection-diffusion MOL system.
    #[allow(clippy::too_many_arguments)]
    pub fn build<S: SparseScalar>(
        grid: Grid3D<S>,
        diffusion: S,
        vx: S,
        vy: S,
        vz: S,
        bc: &BoundaryConditions3D<S>,
    ) -> MOLSystem3D<S> {
        let coeffs = Operator3DCoefficients::advection_diffusion(diffusion, vx, vy, vz);
        MOLSystem3D::with_operator(grid, &coeffs, bc)
    }
}

/// 3D Reaction-Diffusion: u_t = D*(u_xx + u_yy + u_zz) + R(t, x, y, z, u).
pub struct ReactionDiffusion3D;

impl ReactionDiffusion3D {
    /// Build a 3D reaction-diffusion MOL system.
    pub fn build<S, R>(
        grid: Grid3D<S>,
        diffusion: S,
        bc: &BoundaryConditions3D<S>,
        reaction: R,
    ) -> MOLSystem3D<S>
    where
        S: SparseScalar,
        R: Fn(S, S, S, S, S) -> S + Send + Sync + 'static,
    {
        MOLSystem3D::heat(grid, diffusion, bc).with_reaction(reaction)
    }

    /// Build a 3D Fisher-KPP equation: u_t = D*(u_xx + u_yy + u_zz) + r*u*(1-u).
    pub fn fisher<S: SparseScalar>(
        grid: Grid3D<S>,
        diffusion: S,
        growth_rate: S,
        bc: &BoundaryConditions3D<S>,
    ) -> MOLSystem3D<S> {
        let r = growth_rate;
        MOLSystem3D::heat(grid, diffusion, bc)
            .with_reaction(move |_t, _x, _y, _z, u| r * u * (S::ONE - u))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use numra_ode::{DoPri5, OdeSystem, Solver, SolverOptions};

    #[test]
    fn test_heat_equation_3d() {
        let grid = Grid3D::uniform(0.0, 1.0, 7, 0.0, 1.0, 7, 0.0, 1.0, 7);
        let bc = BoundaryConditions3D::all_zero_dirichlet();
        let mol = HeatEquation3D::build(grid, 0.01_f64, &bc);
        assert_eq!(mol.dim(), 125); // 5*5*5
    }

    #[test]
    fn test_advection_diffusion_3d() {
        let grid = Grid3D::uniform(0.0, 1.0, 7, 0.0, 1.0, 7, 0.0, 1.0, 7);
        let bc = BoundaryConditions3D::all_zero_dirichlet();
        let mol = AdvectionDiffusion3D::build(grid, 0.01, 1.0, 0.0, 0.0, &bc);
        assert_eq!(mol.dim(), 125);

        // Should run without error
        let u0 = vec![0.0; 125];
        let options = SolverOptions::default().rtol(1e-4);
        let result = DoPri5::solve(&mol, 0.0, 0.01, &u0, &options).unwrap();
        assert!(result.success);
    }

    #[test]
    fn test_reaction_diffusion_3d_custom() {
        // u_t = D*Laplacian(u) - 0.5 * u (linear decay)
        let n = 7;
        let grid = Grid3D::uniform(0.0, 1.0, n, 0.0, 1.0, n, 0.0, 1.0, n);
        let bc = BoundaryConditions3D::all_zero_dirichlet();
        let mol = ReactionDiffusion3D::build(grid, 0.01, &bc, |_t, _x, _y, _z, u| -0.5 * u);

        let nx_int = n - 2;
        let n_int = nx_int * nx_int * nx_int;
        let u0 = vec![0.1; n_int];

        let options = SolverOptions::default().rtol(1e-4);
        let result = DoPri5::solve(&mol, 0.0, 0.1, &u0, &options).unwrap();
        assert!(result.success);

        // With negative reaction + zero Dirichlet, solution should decay
        let y_final = result.y_final().unwrap();
        for &v in &y_final {
            assert!(v < 0.1, "Expected decay, got {}", v);
        }
    }

    #[test]
    fn test_fisher_3d() {
        let n = 9;
        let grid = Grid3D::uniform(0.0, 1.0, n, 0.0, 1.0, n, 0.0, 1.0, n);
        let bc = BoundaryConditions3D::all_zero_dirichlet();
        let mol = ReactionDiffusion3D::fisher(grid.clone(), 0.01, 1.0, &bc);

        let nx_int = n - 2;
        let ny_int = n - 2;
        let nz_int = n - 2;
        let n_int = nx_int * ny_int * nz_int;

        // IC: small spherical bump in centre
        let mut u0 = vec![0.0; n_int];
        for kk in 0..nz_int {
            for jj in 0..ny_int {
                for ii in 0..nx_int {
                    let x = grid.x_grid.points()[ii + 1];
                    let y = grid.y_grid.points()[jj + 1];
                    let z = grid.z_grid.points()[kk + 1];
                    let r2 = (x - 0.5) * (x - 0.5) + (y - 0.5) * (y - 0.5) + (z - 0.5) * (z - 0.5);
                    if r2 < 0.05 {
                        u0[kk * (nx_int * ny_int) + jj * nx_int + ii] = 0.5;
                    }
                }
            }
        }

        let options = SolverOptions::default().rtol(1e-4);
        let result = DoPri5::solve(&mol, 0.0, 0.1, &u0, &options).unwrap();
        assert!(result.success);

        // Solution stays in [-0.1, 1.1] for Fisher-KPP
        let y_final = result.y_final().unwrap();
        for &v in &y_final {
            assert!(
                (-0.1..=1.1).contains(&v),
                "Solution out of expected range: {}",
                v
            );
        }
    }
}

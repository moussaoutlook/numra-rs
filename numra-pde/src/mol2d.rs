//! 2D Method of Lines for converting 2D PDEs to ODE systems.
//!
//! Uses sparse matrix assembly for the spatial operator, then
//! the ODE RHS is simply a sparse matrix-vector product.
//!
//! Author: Moussa Leblouba
//! Date: 9 February 2026
//! Modified: 2 May 2026

use crate::boundary2d::BoundaryConditions2D;
use crate::grid::Grid2D;
use crate::sparse_assembly::{
    assemble_laplacian_2d, assemble_operator_2d, Operator2DCoefficients, SparseScalar,
};
use numra_linalg::SparseMatrix;
use numra_ode::OdeSystem;

/// Type alias for a reaction term closure: (t, x, y, u) -> f(u).
type ReactionFn<S> = Box<dyn Fn(S, S, S, S) -> S + Send + Sync>;

/// 2D Method of Lines system.
///
/// Converts a 2D PDE of the form `u_t = L[u] + R(t, x, y, u)`
/// into an ODE system, where L is a linear spatial operator assembled
/// as a sparse matrix and R is an optional nonlinear reaction term.
pub struct MOLSystem2D<S: SparseScalar> {
    /// Spatial grid
    grid: Grid2D<S>,
    /// Assembled sparse operator matrix (n_int x n_int)
    operator: SparseMatrix<S>,
    /// RHS contribution from boundary conditions
    bc_rhs: Vec<S>,
    /// Optional nonlinear reaction term R(t, x, y, u)
    reaction: Option<ReactionFn<S>>,
}

impl<S: SparseScalar> MOLSystem2D<S> {
    /// Create a 2D MOL system for the heat equation: u_t = alpha * laplacian(u).
    pub fn heat(grid: Grid2D<S>, alpha: S, bc: &BoundaryConditions2D<S>) -> Self {
        let coeffs = Operator2DCoefficients::scaled_laplacian(alpha);
        let (operator, bc_rhs) =
            assemble_operator_2d(&grid, &coeffs, bc).expect("Failed to assemble 2D operator");
        Self {
            grid,
            operator,
            bc_rhs,
            reaction: None,
        }
    }

    /// Create a 2D MOL system for the Laplacian: u_t = laplacian(u).
    pub fn laplacian(grid: Grid2D<S>, bc: &BoundaryConditions2D<S>) -> Self {
        let (operator, bc_rhs) =
            assemble_laplacian_2d(&grid, bc).expect("Failed to assemble 2D Laplacian");
        Self {
            grid,
            operator,
            bc_rhs,
            reaction: None,
        }
    }

    /// Create a 2D MOL system with a general linear operator.
    pub fn with_operator(
        grid: Grid2D<S>,
        coeffs: &Operator2DCoefficients<S>,
        bc: &BoundaryConditions2D<S>,
    ) -> Self {
        let (operator, bc_rhs) =
            assemble_operator_2d(&grid, coeffs, bc).expect("Failed to assemble 2D operator");
        Self {
            grid,
            operator,
            bc_rhs,
            reaction: None,
        }
    }

    /// Add a nonlinear reaction term R(t, x, y, u) to the system.
    ///
    /// The full PDE becomes: `u_t = L[u] + R(t, x, y, u)`.
    pub fn with_reaction<F>(mut self, reaction: F) -> Self
    where
        F: Fn(S, S, S, S) -> S + Send + Sync + 'static,
    {
        self.reaction = Some(Box::new(reaction));
        self
    }

    /// Get the spatial grid.
    pub fn grid(&self) -> &Grid2D<S> {
        &self.grid
    }

    /// Number of interior points (ODE dimension).
    pub fn n_interior(&self) -> usize {
        self.grid.n_interior()
    }

    /// Build the full solution array including boundaries.
    ///
    /// Interior values are stored in column-major order: u[jj * nx_int + ii].
    /// The full array has nx*ny entries in the same column-major layout.
    pub fn build_full_solution(&self, u_interior: &[S]) -> Vec<S> {
        let nx = self.grid.nx();
        let ny = self.grid.ny();
        let nx_int = self.grid.nx_interior();

        let mut u_full = vec![S::ZERO; nx * ny];

        // Fill interior
        for jj in 0..self.grid.ny_interior() {
            for ii in 0..nx_int {
                let full_idx = (jj + 1) * nx + (ii + 1);
                let int_idx = jj * nx_int + ii;
                u_full[full_idx] = u_interior[int_idx];
            }
        }

        // Boundaries are zero (Dirichlet=0) or not needed for output
        u_full
    }
}

impl<S: SparseScalar> OdeSystem<S> for MOLSystem2D<S> {
    fn dim(&self) -> usize {
        self.n_interior()
    }

    fn rhs(&self, t: S, y: &[S], dydt: &mut [S]) {
        // Sparse matvec: dydt = operator * y + bc_rhs
        let matvec = self.operator.mul_vec(y).expect("Sparse matvec failed");

        let n = self.n_interior();
        for i in 0..n {
            dydt[i] = matvec[i] + self.bc_rhs[i];
        }

        // Add reaction term if present
        if let Some(ref reaction) = self.reaction {
            let nx_int = self.grid.nx_interior();
            for jj in 0..self.grid.ny_interior() {
                for ii in 0..nx_int {
                    let idx = jj * nx_int + ii;
                    let x = self.grid.x_grid.points()[ii + 1];
                    let y_coord = self.grid.y_grid.points()[jj + 1];
                    dydt[idx] = dydt[idx] + reaction(t, x, y_coord, y[idx]);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use numra_ode::{DoPri5, Solver, SolverOptions};

    #[test]
    fn test_mol2d_heat_steady_state() {
        // 2D heat equation with zero Dirichlet BCs and zero initial condition
        // Should stay at zero (trivial test)
        let grid = Grid2D::uniform(0.0, 1.0, 11, 0.0, 1.0, 11);
        let bc = BoundaryConditions2D::all_zero_dirichlet();
        let mol = MOLSystem2D::heat(grid, 0.01_f64, &bc);

        assert_eq!(mol.dim(), 81); // 9*9

        let u0 = vec![0.0; 81];
        let options = SolverOptions::default().rtol(1e-6);
        let result = DoPri5::solve(&mol, 0.0, 0.1, &u0, &options).unwrap();
        assert!(result.success);

        let y_final = result.y_final().unwrap();
        for &v in &y_final {
            assert!(v.abs() < 1e-10, "Expected zero, got {}", v);
        }
    }

    #[test]
    fn test_mol2d_heat_decay() {
        // 2D heat equation: u_t = alpha*(u_xx + u_yy)
        // IC: u(x,y,0) = sin(pi*x)*sin(pi*y)
        // BC: zero Dirichlet
        // Exact: u(x,y,t) = sin(pi*x)*sin(pi*y)*exp(-2*pi^2*alpha*t)
        let alpha = 0.01_f64;
        let n = 21;
        let grid = Grid2D::uniform(0.0, 1.0, n, 0.0, 1.0, n);
        let bc = BoundaryConditions2D::all_zero_dirichlet();
        let mol = MOLSystem2D::heat(grid.clone(), alpha, &bc);

        let nx_int = n - 2;
        let ny_int = n - 2;
        let n_int = nx_int * ny_int;

        // IC
        let mut u0 = vec![0.0; n_int];
        let pi = std::f64::consts::PI;
        for jj in 0..ny_int {
            for ii in 0..nx_int {
                let x = grid.x_grid.points()[ii + 1];
                let y = grid.y_grid.points()[jj + 1];
                u0[jj * nx_int + ii] = (pi * x).sin() * (pi * y).sin();
            }
        }

        let t_final = 0.5;
        let options = SolverOptions::default().rtol(1e-6).atol(1e-9);
        let result = DoPri5::solve(&mol, 0.0, t_final, &u0, &options).unwrap();
        assert!(result.success);

        let y_final = result.y_final().unwrap();
        let decay = (-2.0 * pi * pi * alpha * t_final).exp();

        for jj in 0..ny_int {
            for ii in 0..nx_int {
                let idx = jj * nx_int + ii;
                let x = grid.x_grid.points()[ii + 1];
                let y = grid.y_grid.points()[jj + 1];
                let exact = (pi * x).sin() * (pi * y).sin() * decay;
                assert!(
                    (y_final[idx] - exact).abs() < 0.02,
                    "At ({:.2}, {:.2}): computed={:.6}, exact={:.6}",
                    x,
                    y,
                    y_final[idx],
                    exact
                );
            }
        }
    }

    #[test]
    fn test_mol2d_reaction_diffusion() {
        // u_t = D*laplacian(u) + u*(1-u) (Fisher equation in 2D)
        // Just verify it runs and solution stays in [0, 1] range
        let d = 0.01_f64;
        let n = 11;
        let grid = Grid2D::uniform(0.0, 1.0, n, 0.0, 1.0, n);
        let bc = BoundaryConditions2D::all_zero_dirichlet();
        let mol =
            MOLSystem2D::heat(grid.clone(), d, &bc).with_reaction(|_t, _x, _y, u| u * (1.0 - u));

        let nx_int = n - 2;
        let ny_int = n - 2;
        let n_int = nx_int * ny_int;

        // IC: small bump in center
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
        let result = DoPri5::solve(&mol, 0.0, 0.5, &u0, &options).unwrap();
        assert!(result.success);

        let y_final = result.y_final().unwrap();
        for &v in &y_final {
            assert!(v >= -0.1 && v <= 1.1, "Solution out of range: {}", v);
        }
    }

    #[test]
    fn test_mol2d_nonzero_dirichlet() {
        // Heat equation with T=1 on left, T=0 on other sides
        // After long time, should approach Laplace equation solution
        let n = 11;
        let grid = Grid2D::uniform(0.0, 1.0, n, 0.0, 1.0, n);
        let bc = BoundaryConditions2D {
            x_min: crate::boundary::BoxedBC::dirichlet(1.0),
            x_max: crate::boundary::BoxedBC::dirichlet(0.0),
            y_min: crate::boundary::BoxedBC::dirichlet(0.0),
            y_max: crate::boundary::BoxedBC::dirichlet(0.0),
        };
        let mol = MOLSystem2D::heat(grid.clone(), 0.1_f64, &bc);

        let u0 = vec![0.0; mol.dim()];
        let options = SolverOptions::default().rtol(1e-6);
        let result = DoPri5::solve(&mol, 0.0, 5.0, &u0, &options).unwrap();
        assert!(result.success);

        let y_final = result.y_final().unwrap();
        // Near the left boundary (ii=0), values should be close to 1
        let nx_int = n - 2;
        let mid_j = (n - 2) / 2;
        let left_val = y_final[mid_j * nx_int + 0];
        let right_val = y_final[mid_j * nx_int + (nx_int - 1)];
        assert!(left_val > 0.3, "Near left should be warm: {}", left_val);
        assert!(right_val < 0.3, "Near right should be cool: {}", right_val);
    }

    #[test]
    fn test_mol2d_build_full_solution() {
        let grid = Grid2D::uniform(0.0, 1.0, 5, 0.0, 1.0, 5);
        let bc = BoundaryConditions2D::all_zero_dirichlet();
        let mol = MOLSystem2D::heat(grid, 0.01_f64, &bc);

        let u_int = vec![1.0; 9]; // 3x3 interior = 9
        let u_full = mol.build_full_solution(&u_int);
        assert_eq!(u_full.len(), 25); // 5x5

        // Interior should be 1.0
        assert!((u_full[1 * 5 + 1] - 1.0).abs() < 1e-10);
        assert!((u_full[2 * 5 + 2] - 1.0).abs() < 1e-10);
        // Boundary should be 0.0
        assert!(u_full[0].abs() < 1e-10);
        assert!(u_full[4].abs() < 1e-10);
    }
}

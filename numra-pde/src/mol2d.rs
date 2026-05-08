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

    /// Analytical Jacobian: copy the assembled sparse spatial operator
    /// (which already equals `∂(L[u])/∂u` by construction) into the row-major
    /// dense buffer the solver expects, then add the reaction term's
    /// contribution to the diagonal.
    ///
    /// The reaction Jacobian is diagonal *because* the reaction is
    /// pointwise: `R(t, x_i, y_j, u_i)` depends only on the local state
    /// `u_i`, so `∂R_i/∂u_k` is identically zero for any `k != i`. Off-
    /// diagonal entries cannot be populated by a pointwise reaction, no
    /// matter what the closure contains. This is what makes the
    /// FD-on-the-diagonal fallback affordable: one extra closure call per
    /// interior point versus the `O(N)` rhs evaluations the trait-default
    /// FD would need to populate the same entries through perturbation.
    /// If a future reaction model couples grid points (nonlocal /
    /// integro-PDE / multi-component), the trait default's full FD path is
    /// the correct fallback — but that's out of scope for v1.
    fn jacobian(&self, t: S, y: &[S], jac: &mut [S]) {
        let n = self.n_interior();
        let nn = n * n;

        // Zero the dense buffer; the sparse operator only fills nonzero
        // entries below.
        for v in jac.iter_mut().take(nn) {
            *v = S::ZERO;
        }

        // Linear operator: walk the CSC representation directly, writing
        // each (row, col) pair to its row-major position. Avoids the
        // intermediate DenseMatrix allocation that .to_dense() would do.
        let col_ptrs = self.operator.col_ptrs();
        let row_indices = self.operator.row_indices();
        let values = self.operator.values();
        for j in 0..n {
            let start = col_ptrs[j];
            let end = col_ptrs[j + 1];
            for idx in start..end {
                let i = row_indices[idx];
                jac[i * n + j] = values[idx];
            }
        }

        // Reaction: diagonal-only FD. Same eps formula as the trait default
        // (eps * (1 + |u|), eps = 1e-8) so the partial-FD-partial-analytical
        // mix stays numerically consistent.
        if let Some(ref reaction) = self.reaction {
            let eps = S::from_f64(1e-8);
            let nx_int = self.grid.nx_interior();
            for jj in 0..self.grid.ny_interior() {
                for ii in 0..nx_int {
                    let idx = jj * nx_int + ii;
                    let x = self.grid.x_grid.points()[ii + 1];
                    let y_coord = self.grid.y_grid.points()[jj + 1];
                    let u = y[idx];
                    let h = eps * (S::ONE + u.abs());
                    let r0 = reaction(t, x, y_coord, u);
                    let r1 = reaction(t, x, y_coord, u + h);
                    let dr_du = (r1 - r0) / h;
                    jac[idx * n + idx] = jac[idx * n + idx] + dr_du;
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

    /// Helper: trait-default FD Jacobian, used as the agreement reference
    /// for the analytical-override regression. Inlined here rather than
    /// pulled from numra-ode to avoid a circular test dep; bit-identical
    /// to the OdeSystem::jacobian default.
    fn fd_jacobian<Sys: numra_ode::OdeSystem<f64>>(sys: &Sys, t: f64, y: &[f64]) -> Vec<f64> {
        let n = sys.dim();
        let eps = 1e-8;
        let mut jac = vec![0.0; n * n];
        let mut y_pert = y.to_vec();
        let mut f0 = vec![0.0; n];
        let mut f1 = vec![0.0; n];
        sys.rhs(t, y, &mut f0);
        for j in 0..n {
            let yj = y_pert[j];
            let h = eps * (1.0 + yj.abs());
            y_pert[j] = yj + h;
            sys.rhs(t, &y_pert, &mut f1);
            y_pert[j] = yj;
            for i in 0..n {
                jac[i * n + j] = (f1[i] - f0[i]) / h;
            }
        }
        jac
    }

    #[test]
    fn test_mol2d_jacobian_agrees_with_fd_no_reaction() {
        // Pure linear PDE — analytical and FD must agree to roundoff plus
        // FD truncation, well below 1e-5 in absolute terms.
        let grid = Grid2D::uniform(0.0, 1.0, 7, 0.0, 1.0, 7);
        let bc = BoundaryConditions2D::all_zero_dirichlet();
        let mol = MOLSystem2D::heat(grid, 0.01_f64, &bc);
        let n = mol.dim();

        // Non-trivial state so the FD perturbations matter.
        let y: Vec<f64> = (0..n).map(|i| ((i + 1) as f64).sin()).collect();

        let mut jac_analytical = vec![0.0; n * n];
        OdeSystem::jacobian(&mol, 0.0, &y, &mut jac_analytical);
        let jac_fd = fd_jacobian(&mol, 0.0, &y);

        for i in 0..n {
            for j in 0..n {
                let a = jac_analytical[i * n + j];
                let f = jac_fd[i * n + j];
                let tol = 1e-5_f64.max(1e-5 * a.abs());
                assert!(
                    (a - f).abs() < tol,
                    "Jacobian mismatch at ({},{}): analytical={}, fd={}",
                    i,
                    j,
                    a,
                    f
                );
            }
        }
    }

    #[test]
    fn test_mol2d_jacobian_agrees_with_fd_with_reaction() {
        // Reaction R(u) = -u^3, so dR/du = -3 u^2. The diagonal-FD path
        // should land within FD truncation of this analytical value, and
        // off-diagonal entries should match the operator-only baseline.
        let grid = Grid2D::uniform(0.0, 1.0, 5, 0.0, 1.0, 5);
        let bc = BoundaryConditions2D::all_zero_dirichlet();
        let mol =
            MOLSystem2D::heat(grid, 0.05_f64, &bc).with_reaction(|_t, _x, _y, u: f64| -u * u * u);
        let n = mol.dim();
        let y: Vec<f64> = (0..n).map(|i| 0.1 + (i as f64) * 0.01).collect();

        let mut jac_analytical = vec![0.0; n * n];
        OdeSystem::jacobian(&mol, 0.0, &y, &mut jac_analytical);
        let jac_fd = fd_jacobian(&mol, 0.0, &y);

        for i in 0..n {
            for j in 0..n {
                let a = jac_analytical[i * n + j];
                let f = jac_fd[i * n + j];
                assert!(
                    (a - f).abs() < 1e-4,
                    "Jacobian mismatch at ({},{}): analytical={}, fd={}",
                    i,
                    j,
                    a,
                    f
                );
            }
        }
    }

    #[test]
    fn test_mol2d_radau5_uses_analytical_jacobian() {
        // End-to-end composability: solve a stiff 2D heat problem with
        // Radau5 and confirm success. The analytical-Jacobian override
        // means Radau5 stops paying for the N+1 rhs evaluations FD would
        // need per Jacobian rebuild; the test asserts it still produces
        // a correct answer — the perf win is verified separately by the
        // numra-bench harness.
        use numra_ode::{Radau5, Solver, SolverOptions};
        let n = 9;
        let grid = Grid2D::uniform(0.0, 1.0, n, 0.0, 1.0, n);
        let bc = BoundaryConditions2D::all_zero_dirichlet();
        let mol = MOLSystem2D::heat(grid.clone(), 0.5_f64, &bc); // large alpha => stiff

        let nx_int = n - 2;
        let pi = std::f64::consts::PI;
        let mut u0 = vec![0.0; nx_int * nx_int];
        for jj in 0..nx_int {
            for ii in 0..nx_int {
                let x = grid.x_grid.points()[ii + 1];
                let yc = grid.y_grid.points()[jj + 1];
                u0[jj * nx_int + ii] = (pi * x).sin() * (pi * yc).sin();
            }
        }

        let options = SolverOptions::default().rtol(1e-6).atol(1e-9);
        let result = Radau5::solve(&mol, 0.0, 0.05, &u0, &options).unwrap();
        assert!(result.success);

        // Compare to analytical decay at the centre.
        let y_final = result.y_final().unwrap();
        let mid = (nx_int / 2) * nx_int + (nx_int / 2);
        let exact = (-2.0 * pi * pi * 0.5_f64 * 0.05).exp();
        // Coarse 9² grid limits FD accuracy to a few percent at the centre.
        let rel_err = (y_final[mid] - exact).abs() / exact;
        assert!(
            rel_err < 0.05,
            "computed={}, exact={}, rel_err={}",
            y_final[mid],
            exact,
            rel_err
        );
    }
}

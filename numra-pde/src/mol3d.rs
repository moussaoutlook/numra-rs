//! 3D Method of Lines for converting 3D PDEs to ODE systems.
//!
//! Uses sparse matrix assembly for the spatial operator (7-point stencil),
//! then the ODE RHS is a sparse matrix-vector product plus an optional
//! pointwise reaction term.
//!
//! Author: Moussa Leblouba
//! Date: 7 May 2026

use crate::boundary2d::BoundaryConditions3D;
use crate::grid::Grid3D;
use crate::sparse_assembly::{
    assemble_laplacian_3d, assemble_operator_3d, Operator3DCoefficients, SparseScalar,
};
use numra_linalg::SparseMatrix;
use numra_ode::OdeSystem;

/// Type alias for a reaction term closure: (t, x, y, z, u) -> f(u).
type ReactionFn<S> = Box<dyn Fn(S, S, S, S, S) -> S + Send + Sync>;

/// 3D Method of Lines system.
///
/// Converts a 3D PDE of the form `u_t = L[u] + R(t, x, y, z, u)`
/// into an ODE system, where L is a linear spatial operator assembled
/// as a sparse matrix and R is an optional nonlinear reaction term.
pub struct MOLSystem3D<S: SparseScalar> {
    /// Spatial grid
    grid: Grid3D<S>,
    /// Assembled sparse operator matrix (n_int x n_int)
    operator: SparseMatrix<S>,
    /// RHS contribution from boundary conditions
    bc_rhs: Vec<S>,
    /// Optional nonlinear reaction term R(t, x, y, z, u)
    reaction: Option<ReactionFn<S>>,
}

impl<S: SparseScalar> MOLSystem3D<S> {
    /// Create a 3D MOL system for the heat equation: u_t = alpha * laplacian(u).
    pub fn heat(grid: Grid3D<S>, alpha: S, bc: &BoundaryConditions3D<S>) -> Self {
        let coeffs = Operator3DCoefficients::scaled_laplacian(alpha);
        let (operator, bc_rhs) =
            assemble_operator_3d(&grid, &coeffs, bc).expect("Failed to assemble 3D operator");
        Self {
            grid,
            operator,
            bc_rhs,
            reaction: None,
        }
    }

    /// Create a 3D MOL system for the Laplacian: u_t = laplacian(u).
    pub fn laplacian(grid: Grid3D<S>, bc: &BoundaryConditions3D<S>) -> Self {
        let (operator, bc_rhs) =
            assemble_laplacian_3d(&grid, bc).expect("Failed to assemble 3D Laplacian");
        Self {
            grid,
            operator,
            bc_rhs,
            reaction: None,
        }
    }

    /// Create a 3D MOL system with a general linear operator.
    pub fn with_operator(
        grid: Grid3D<S>,
        coeffs: &Operator3DCoefficients<S>,
        bc: &BoundaryConditions3D<S>,
    ) -> Self {
        let (operator, bc_rhs) =
            assemble_operator_3d(&grid, coeffs, bc).expect("Failed to assemble 3D operator");
        Self {
            grid,
            operator,
            bc_rhs,
            reaction: None,
        }
    }

    /// Add a nonlinear reaction term R(t, x, y, z, u) to the system.
    ///
    /// The full PDE becomes: `u_t = L[u] + R(t, x, y, z, u)`.
    pub fn with_reaction<F>(mut self, reaction: F) -> Self
    where
        F: Fn(S, S, S, S, S) -> S + Send + Sync + 'static,
    {
        self.reaction = Some(Box::new(reaction));
        self
    }

    /// Get the spatial grid.
    pub fn grid(&self) -> &Grid3D<S> {
        &self.grid
    }

    /// Number of interior points (ODE dimension).
    pub fn n_interior(&self) -> usize {
        self.grid.n_interior()
    }

    /// Build the full solution array including boundaries.
    ///
    /// Interior values are stored in column-major order:
    ///   `u[kk * (nx_int * ny_int) + jj * nx_int + ii]`.
    /// The full array has `nx*ny*nz` entries in the same column-major
    /// layout used by `Grid3D::linear_index`.
    pub fn build_full_solution(&self, u_interior: &[S]) -> Vec<S> {
        let nx = self.grid.x_grid.len();
        let ny = self.grid.y_grid.len();
        let nz = self.grid.z_grid.len();
        let nx_int = self.grid.x_grid.n_interior();
        let ny_int = self.grid.y_grid.n_interior();
        let nz_int = self.grid.z_grid.n_interior();

        let mut u_full = vec![S::ZERO; nx * ny * nz];

        for kk in 0..nz_int {
            for jj in 0..ny_int {
                for ii in 0..nx_int {
                    let full_idx = self.grid.linear_index(ii + 1, jj + 1, kk + 1);
                    let int_idx = kk * (nx_int * ny_int) + jj * nx_int + ii;
                    u_full[full_idx] = u_interior[int_idx];
                }
            }
        }

        u_full
    }
}

impl<S: SparseScalar> OdeSystem<S> for MOLSystem3D<S> {
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
            let nx_int = self.grid.x_grid.n_interior();
            let ny_int = self.grid.y_grid.n_interior();
            let nz_int = self.grid.z_grid.n_interior();
            for kk in 0..nz_int {
                for jj in 0..ny_int {
                    for ii in 0..nx_int {
                        let idx = kk * (nx_int * ny_int) + jj * nx_int + ii;
                        let x = self.grid.x_grid.points()[ii + 1];
                        let y_coord = self.grid.y_grid.points()[jj + 1];
                        let z_coord = self.grid.z_grid.points()[kk + 1];
                        dydt[idx] = dydt[idx] + reaction(t, x, y_coord, z_coord, y[idx]);
                    }
                }
            }
        }
    }

    /// Analytical Jacobian: copy the assembled sparse spatial operator
    /// (which already equals ∂(L[u])/∂u by construction) into the row-major
    /// dense buffer the solver expects, then add the reaction term's
    /// contribution to the diagonal.
    ///
    /// The reaction Jacobian is diagonal *because* the reaction is
    /// pointwise: `R(t, x_i, y_j, z_k, u_i)` depends only on the local
    /// state `u_i`, so `∂R_i/∂u_m` is identically zero for any `m != i`.
    /// Off-diagonal entries cannot be populated by a pointwise reaction,
    /// no matter what the closure contains. This is what makes the
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

        // Linear operator: walk the CSC representation directly into the
        // row-major buffer. Avoids the intermediate DenseMatrix allocation
        // that .to_dense() would do.
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
            let nx_int = self.grid.x_grid.n_interior();
            let ny_int = self.grid.y_grid.n_interior();
            let nz_int = self.grid.z_grid.n_interior();
            for kk in 0..nz_int {
                for jj in 0..ny_int {
                    for ii in 0..nx_int {
                        let idx = kk * (nx_int * ny_int) + jj * nx_int + ii;
                        let x = self.grid.x_grid.points()[ii + 1];
                        let y_coord = self.grid.y_grid.points()[jj + 1];
                        let z_coord = self.grid.z_grid.points()[kk + 1];
                        let u = y[idx];
                        let h = eps * (S::ONE + u.abs());
                        let r0 = reaction(t, x, y_coord, z_coord, u);
                        let r1 = reaction(t, x, y_coord, z_coord, u + h);
                        let dr_du = (r1 - r0) / h;
                        jac[idx * n + idx] = jac[idx * n + idx] + dr_du;
                    }
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
    fn test_mol3d_dim_and_zero_state() {
        // Trivial: zero IC + zero Dirichlet BCs => stays at zero.
        let grid = Grid3D::uniform(0.0, 1.0, 5, 0.0, 1.0, 5, 0.0, 1.0, 5);
        let bc = BoundaryConditions3D::all_zero_dirichlet();
        let mol = MOLSystem3D::heat(grid, 0.01_f64, &bc);

        // 3*3*3 = 27 interior points
        assert_eq!(mol.dim(), 27);

        let u0 = vec![0.0; 27];
        let options = SolverOptions::default().rtol(1e-6);
        let result = DoPri5::solve(&mol, 0.0, 0.1, &u0, &options).unwrap();
        assert!(result.success);

        let y_final = result.y_final().unwrap();
        for &v in &y_final {
            assert!(v.abs() < 1e-10, "Expected zero, got {}", v);
        }
    }

    #[test]
    fn test_mol3d_heat_decay() {
        // 3D heat equation: u_t = alpha*(u_xx + u_yy + u_zz)
        // IC: u(x,y,z,0) = sin(pi*x)*sin(pi*y)*sin(pi*z) on [0,1]^3
        // BC: zero Dirichlet
        // Exact: u(x,y,z,t) = IC * exp(-3*pi^2*alpha*t)
        let alpha = 0.01_f64;
        let n = 13;
        let grid = Grid3D::uniform(0.0, 1.0, n, 0.0, 1.0, n, 0.0, 1.0, n);
        let bc = BoundaryConditions3D::all_zero_dirichlet();
        let mol = MOLSystem3D::heat(grid.clone(), alpha, &bc);

        let nx_int = n - 2;
        let ny_int = n - 2;
        let nz_int = n - 2;
        let n_int = nx_int * ny_int * nz_int;

        // IC
        let mut u0 = vec![0.0; n_int];
        let pi = std::f64::consts::PI;
        for kk in 0..nz_int {
            for jj in 0..ny_int {
                for ii in 0..nx_int {
                    let x = grid.x_grid.points()[ii + 1];
                    let y = grid.y_grid.points()[jj + 1];
                    let z = grid.z_grid.points()[kk + 1];
                    u0[kk * (nx_int * ny_int) + jj * nx_int + ii] =
                        (pi * x).sin() * (pi * y).sin() * (pi * z).sin();
                }
            }
        }

        let t_final = 0.5;
        let options = SolverOptions::default().rtol(1e-6).atol(1e-9);
        let result = DoPri5::solve(&mol, 0.0, t_final, &u0, &options).unwrap();
        assert!(result.success);

        let y_final = result.y_final().unwrap();
        let decay = (-3.0 * pi * pi * alpha * t_final).exp();

        // Check at the center of the cube (max amplitude); coarse FD on a
        // 13^3 grid limits accuracy to ~3% at the center.
        let mid_i = nx_int / 2;
        let mid_j = ny_int / 2;
        let mid_k = nz_int / 2;
        let idx = mid_k * (nx_int * ny_int) + mid_j * nx_int + mid_i;
        let x = grid.x_grid.points()[mid_i + 1];
        let y = grid.y_grid.points()[mid_j + 1];
        let z = grid.z_grid.points()[mid_k + 1];
        let exact = (pi * x).sin() * (pi * y).sin() * (pi * z).sin() * decay;
        let rel_err = (y_final[idx] - exact).abs() / exact.abs();
        assert!(
            rel_err < 0.05,
            "Center: computed={:.6}, exact={:.6}, rel_err={:.4}",
            y_final[idx],
            exact,
            rel_err
        );
    }

    #[test]
    fn test_mol3d_reaction_diffusion() {
        // u_t = D*laplacian(u) + u*(1-u) (Fisher-KPP in 3D)
        // Just verify it runs and stays in a sensible range.
        let d = 0.01_f64;
        let n = 7;
        let grid = Grid3D::uniform(0.0, 1.0, n, 0.0, 1.0, n, 0.0, 1.0, n);
        let bc = BoundaryConditions3D::all_zero_dirichlet();
        let mol = MOLSystem3D::heat(grid.clone(), d, &bc)
            .with_reaction(|_t, _x, _y, _z, u| u * (1.0 - u));

        let nx_int = n - 2;
        let ny_int = n - 2;
        let nz_int = n - 2;
        let n_int = nx_int * ny_int * nz_int;

        // IC: bump in center
        let mut u0 = vec![0.0; n_int];
        for kk in 0..nz_int {
            for jj in 0..ny_int {
                for ii in 0..nx_int {
                    let x = grid.x_grid.points()[ii + 1];
                    let y = grid.y_grid.points()[jj + 1];
                    let z = grid.z_grid.points()[kk + 1];
                    let r2 = (x - 0.5).powi(2) + (y - 0.5).powi(2) + (z - 0.5).powi(2);
                    if r2 < 0.05 {
                        u0[kk * (nx_int * ny_int) + jj * nx_int + ii] = 0.5;
                    }
                }
            }
        }

        let options = SolverOptions::default().rtol(1e-4);
        let result = DoPri5::solve(&mol, 0.0, 0.5, &u0, &options).unwrap();
        assert!(result.success);

        let y_final = result.y_final().unwrap();
        for &v in &y_final {
            assert!(
                (-0.1..=1.1).contains(&v),
                "Solution out of expected range: {}",
                v
            );
        }
    }

    #[test]
    fn test_mol3d_nonzero_dirichlet() {
        // Hot left face, cold elsewhere.
        // After enough time, near-left interior should warm up,
        // far-right should stay cool.
        let n = 9;
        let grid = Grid3D::uniform(0.0, 1.0, n, 0.0, 1.0, n, 0.0, 1.0, n);
        let bc = BoundaryConditions3D {
            x_min: crate::boundary::BoxedBC::dirichlet(1.0),
            x_max: crate::boundary::BoxedBC::dirichlet(0.0),
            y_min: crate::boundary::BoxedBC::dirichlet(0.0),
            y_max: crate::boundary::BoxedBC::dirichlet(0.0),
            z_min: crate::boundary::BoxedBC::dirichlet(0.0),
            z_max: crate::boundary::BoxedBC::dirichlet(0.0),
        };
        let mol = MOLSystem3D::heat(grid, 0.1_f64, &bc);

        let u0 = vec![0.0; mol.dim()];
        let options = SolverOptions::default().rtol(1e-6);
        let result = DoPri5::solve(&mol, 0.0, 5.0, &u0, &options).unwrap();
        assert!(result.success);

        let y_final = result.y_final().unwrap();
        let nx_int = n - 2;
        let ny_int = n - 2;
        let mid_j = ny_int / 2;
        let mid_k = (n - 2) / 2;
        let left = y_final[mid_k * (nx_int * ny_int) + mid_j * nx_int];
        let right = y_final[mid_k * (nx_int * ny_int) + mid_j * nx_int + (nx_int - 1)];
        assert!(left > 0.3, "Near left face should be warm: {}", left);
        assert!(right < 0.3, "Near right face should be cool: {}", right);
    }

    #[test]
    fn test_mol3d_build_full_solution() {
        let grid = Grid3D::uniform(0.0, 1.0, 4, 0.0, 1.0, 4, 0.0, 1.0, 4);
        let bc = BoundaryConditions3D::all_zero_dirichlet();
        let mol = MOLSystem3D::heat(grid, 0.01_f64, &bc);

        let u_int = vec![1.0; 8]; // 2*2*2
        let u_full = mol.build_full_solution(&u_int);
        assert_eq!(u_full.len(), 64); // 4*4*4

        // Interior cell (1,1,1) should be 1.0
        let center = mol.grid().linear_index(1, 1, 1);
        assert!((u_full[center] - 1.0).abs() < 1e-10);

        // Corner (0,0,0) should be 0.0 (boundary)
        let corner = mol.grid().linear_index(0, 0, 0);
        assert!(u_full[corner].abs() < 1e-10);
    }

    /// Trait-default FD Jacobian helper, identical to the one in mol2d.rs.
    /// Used as the agreement reference for the analytical-override
    /// regression — keeps the test self-contained.
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
    fn test_mol3d_jacobian_agrees_with_fd_no_reaction() {
        // Pure linear 3D PDE — analytical and FD must agree.
        let grid = Grid3D::uniform(0.0, 1.0, 5, 0.0, 1.0, 5, 0.0, 1.0, 5);
        let bc = BoundaryConditions3D::all_zero_dirichlet();
        let mol = MOLSystem3D::heat(grid, 0.01_f64, &bc);
        let n = mol.dim();
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
    fn test_mol3d_jacobian_agrees_with_fd_with_reaction() {
        // Reaction R(u) = -u^3, so dR/du = -3 u^2. Diagonal-FD path
        // matches; off-diagonals come from the operator only.
        let grid = Grid3D::uniform(0.0, 1.0, 4, 0.0, 1.0, 4, 0.0, 1.0, 4);
        let bc = BoundaryConditions3D::all_zero_dirichlet();
        let mol = MOLSystem3D::heat(grid, 0.05_f64, &bc)
            .with_reaction(|_t, _x, _y, _z, u: f64| -u * u * u);
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
    fn test_mol3d_radau5_uses_analytical_jacobian() {
        // End-to-end composability: solve a stiff 3D heat problem with
        // Radau5. Confirms the analytical Jacobian path doesn't break
        // the solver's Newton convergence on a real workload.
        use numra_ode::{Radau5, Solver, SolverOptions};
        let n = 7;
        let grid = Grid3D::uniform(0.0, 1.0, n, 0.0, 1.0, n, 0.0, 1.0, n);
        let bc = BoundaryConditions3D::all_zero_dirichlet();
        let mol = MOLSystem3D::heat(grid.clone(), 0.5_f64, &bc); // large alpha => stiff

        let nx_int = n - 2;
        let n_int = nx_int * nx_int * nx_int;
        let pi = std::f64::consts::PI;
        let mut u0 = vec![0.0; n_int];
        for kk in 0..nx_int {
            for jj in 0..nx_int {
                for ii in 0..nx_int {
                    let x = grid.x_grid.points()[ii + 1];
                    let yc = grid.y_grid.points()[jj + 1];
                    let zc = grid.z_grid.points()[kk + 1];
                    u0[kk * (nx_int * nx_int) + jj * nx_int + ii] =
                        (pi * x).sin() * (pi * yc).sin() * (pi * zc).sin();
                }
            }
        }

        let options = SolverOptions::default().rtol(1e-5).atol(1e-8);
        let result = Radau5::solve(&mol, 0.0, 0.02, &u0, &options).unwrap();
        assert!(result.success);

        // Compare against analytical decay at the cube centre.
        let y_final = result.y_final().unwrap();
        let mid = (nx_int / 2) * (nx_int * nx_int) + (nx_int / 2) * nx_int + (nx_int / 2);
        let exact = (-3.0 * pi * pi * 0.5_f64 * 0.02).exp();
        // Coarse 7³ grid; centre relative error should still come in well
        // under 10%.
        let rel_err = (y_final[mid] - exact).abs() / exact;
        assert!(
            rel_err < 0.1,
            "computed={}, exact={}, rel_err={}",
            y_final[mid],
            exact,
            rel_err
        );
    }
}

//! Parametric 3D Method of Lines for forward sensitivity analysis.
//!
//! Mirror of [`crate::ParametricMOLSystem2D`] with a third spatial axis
//! added. Same parameter layout (`[α, reaction_p_0, reaction_p_1, ...]`),
//! same linearity argument for the operator scaling, same flag contract.
//!
//! Author: Moussa Leblouba
//! Date: 8 May 2026

use crate::boundary2d::BoundaryConditions3D;
use crate::grid::Grid3D;
use crate::sparse_assembly::{assemble_operator_3d, Operator3DCoefficients, SparseScalar};
use numra_linalg::SparseMatrix;
use numra_ode::ParametricOdeSystem;

/// Parametric reaction closure: `(t, x, y, z, u, p) -> R`.
type ParametricReactionFn<S> = Box<dyn Fn(S, S, S, S, S, &[S]) -> S + Send + Sync>;

/// Parametric 3D Method of Lines system. See
/// [`crate::ParametricMOLSystem2D`] for the parameter layout, linearity
/// argument, and BC handling notes — they apply identically here.
pub struct ParametricMOLSystem3D<S: SparseScalar> {
    grid: Grid3D<S>,
    /// Un-scaled Laplacian (built with α = 1).
    l0: SparseMatrix<S>,
    /// Un-scaled boundary RHS contribution (built with α = 1).
    bc_rhs_0: Vec<S>,
    /// Nominal parameter vector. `params[0]` = α, `params[1..]` = reaction.
    nominal_params: Vec<S>,
    reaction: Option<ParametricReactionFn<S>>,
}

impl<S: SparseScalar> ParametricMOLSystem3D<S> {
    /// Build a parametric heat equation `u_t = α∇²u`.
    pub fn heat(grid: Grid3D<S>, alpha_nominal: S, bc: &BoundaryConditions3D<S>) -> Self {
        let coeffs = Operator3DCoefficients::laplacian();
        let (l0, bc_rhs_0) =
            assemble_operator_3d(&grid, &coeffs, bc).expect("Failed to assemble 3D Laplacian");
        Self {
            grid,
            l0,
            bc_rhs_0,
            nominal_params: vec![alpha_nominal],
            reaction: None,
        }
    }

    /// Build a parametric heat equation with a parametric reaction term.
    pub fn heat_with_reaction<R>(
        grid: Grid3D<S>,
        alpha_nominal: S,
        bc: &BoundaryConditions3D<S>,
        nominal_reaction_params: Vec<S>,
        reaction: R,
    ) -> Self
    where
        R: Fn(S, S, S, S, S, &[S]) -> S + Send + Sync + 'static,
    {
        let coeffs = Operator3DCoefficients::laplacian();
        let (l0, bc_rhs_0) =
            assemble_operator_3d(&grid, &coeffs, bc).expect("Failed to assemble 3D Laplacian");
        let mut nominal_params = Vec::with_capacity(1 + nominal_reaction_params.len());
        nominal_params.push(alpha_nominal);
        nominal_params.extend(nominal_reaction_params);
        Self {
            grid,
            l0,
            bc_rhs_0,
            nominal_params,
            reaction: Some(Box::new(reaction)),
        }
    }

    pub fn grid(&self) -> &Grid3D<S> {
        &self.grid
    }

    pub fn n_interior(&self) -> usize {
        self.grid.n_interior()
    }
}

impl<S: SparseScalar> ParametricOdeSystem<S> for ParametricMOLSystem3D<S> {
    fn n_states(&self) -> usize {
        self.grid.n_interior()
    }

    fn n_params(&self) -> usize {
        self.nominal_params.len()
    }

    fn params(&self) -> &[S] {
        &self.nominal_params
    }

    fn rhs_with_params(&self, t: S, y: &[S], p: &[S], dydt: &mut [S]) {
        let alpha = p[0];
        let matvec = self.l0.mul_vec(y).expect("Sparse matvec failed");
        let n = self.grid.n_interior();
        for i in 0..n {
            dydt[i] = alpha * (matvec[i] + self.bc_rhs_0[i]);
        }
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
                        dydt[idx] = dydt[idx] + reaction(t, x, y_coord, z_coord, y[idx], p);
                    }
                }
            }
        }
    }

    fn jacobian_y(&self, t: S, y: &[S], jac: &mut [S]) {
        let n = self.n_states();
        let nn = n * n;
        let alpha = self.nominal_params[0];

        for v in jac.iter_mut().take(nn) {
            *v = S::ZERO;
        }

        let col_ptrs = self.l0.col_ptrs();
        let row_indices = self.l0.row_indices();
        let values = self.l0.values();
        for j in 0..n {
            for idx in col_ptrs[j]..col_ptrs[j + 1] {
                let i = row_indices[idx];
                jac[i * n + j] = alpha * values[idx];
            }
        }

        // Diagonal reaction Jacobian; same pointwise-reaction reasoning as
        // the 2D variant — see ParametricMOLSystem2D::jacobian_y.
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
                        let r0 = reaction(t, x, y_coord, z_coord, u, &self.nominal_params);
                        let r1 = reaction(t, x, y_coord, z_coord, u + h, &self.nominal_params);
                        let dr_du = (r1 - r0) / h;
                        jac[idx * n + idx] = jac[idx * n + idx] + dr_du;
                    }
                }
            }
        }
    }

    fn jacobian_p(&self, t: S, y: &[S], jp: &mut [S]) {
        let n = self.n_states();
        let np = self.n_params();

        for v in jp.iter_mut().take(n * np) {
            *v = S::ZERO;
        }

        // Column 0 (α): linear part L0·y + bc_rhs_0.
        let lin = self.l0.mul_vec(y).expect("Sparse matvec failed");
        for i in 0..n {
            jp[i] = lin[i] + self.bc_rhs_0[i];
        }

        // Reaction contribution to every column. FD on the closure with
        // the matching parameter slot perturbed.
        if let Some(ref reaction) = self.reaction {
            let eps = S::from_f64(1e-8);
            let nx_int = self.grid.x_grid.n_interior();
            let ny_int = self.grid.y_grid.n_interior();
            let nz_int = self.grid.z_grid.n_interior();
            let mut p_pert = self.nominal_params.clone();
            for k in 0..np {
                let pk = self.nominal_params[k];
                let h = eps * (S::ONE + pk.abs());
                p_pert[k] = pk + h;
                for kk in 0..nz_int {
                    for jj in 0..ny_int {
                        for ii in 0..nx_int {
                            let idx = kk * (nx_int * ny_int) + jj * nx_int + ii;
                            let x = self.grid.x_grid.points()[ii + 1];
                            let y_coord = self.grid.y_grid.points()[jj + 1];
                            let z_coord = self.grid.z_grid.points()[kk + 1];
                            let u = y[idx];
                            let r0 = reaction(t, x, y_coord, z_coord, u, &self.nominal_params);
                            let r1 = reaction(t, x, y_coord, z_coord, u, &p_pert);
                            let dr_dpk = (r1 - r0) / h;
                            jp[k * n + idx] = jp[k * n + idx] + dr_dpk;
                        }
                    }
                }
                p_pert[k] = pk;
            }
        }
    }

    fn has_analytical_jacobian_y(&self) -> bool {
        true
    }

    fn has_analytical_jacobian_p(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use numra_ode::sensitivity::solve_forward_sensitivity;
    use numra_ode::Radau5;

    fn fd_jacobian_y<Sys: ParametricOdeSystem<f64>>(sys: &Sys, t: f64, y: &[f64]) -> Vec<f64> {
        let n = sys.n_states();
        let p = sys.params().to_vec();
        let h_factor = (1e-15_f64).sqrt() * 1.5;
        let mut jac = vec![0.0; n * n];
        let mut f0 = vec![0.0; n];
        let mut f1 = vec![0.0; n];
        let mut y_pert = y.to_vec();
        sys.rhs_with_params(t, y, &p, &mut f0);
        for j in 0..n {
            let yj = y_pert[j];
            let h = h_factor * (1.0 + yj.abs());
            y_pert[j] = yj + h;
            sys.rhs_with_params(t, &y_pert, &p, &mut f1);
            y_pert[j] = yj;
            for i in 0..n {
                jac[i * n + j] = (f1[i] - f0[i]) / h;
            }
        }
        jac
    }

    fn fd_jacobian_p<Sys: ParametricOdeSystem<f64>>(sys: &Sys, t: f64, y: &[f64]) -> Vec<f64> {
        let n = sys.n_states();
        let np = sys.n_params();
        let p_nom = sys.params().to_vec();
        let h_factor = (1e-15_f64).sqrt() * 1.5;
        let mut jp = vec![0.0; n * np];
        let mut f0 = vec![0.0; n];
        let mut f1 = vec![0.0; n];
        let mut p_pert = p_nom.clone();
        sys.rhs_with_params(t, y, &p_nom, &mut f0);
        for k in 0..np {
            let pk = p_pert[k];
            let h = h_factor * (1.0 + pk.abs());
            p_pert[k] = pk + h;
            sys.rhs_with_params(t, y, &p_pert, &mut f1);
            p_pert[k] = pk;
            for i in 0..n {
                jp[k * n + i] = (f1[i] - f0[i]) / h;
            }
        }
        jp
    }

    #[test]
    fn test_jacobian_y_agrees_with_fd_no_reaction() {
        let grid = Grid3D::uniform(0.0, 1.0, 5, 0.0, 1.0, 5, 0.0, 1.0, 5);
        let bc = BoundaryConditions3D::all_zero_dirichlet();
        let mol = ParametricMOLSystem3D::heat(grid, 0.05_f64, &bc);
        let n = mol.n_states();
        let y: Vec<f64> = (0..n).map(|i| ((i + 1) as f64).sin()).collect();

        let mut jac_a = vec![0.0; n * n];
        mol.jacobian_y(0.0, &y, &mut jac_a);
        let jac_fd = fd_jacobian_y(&mol, 0.0, &y);

        for i in 0..n {
            for j in 0..n {
                let a = jac_a[i * n + j];
                let f = jac_fd[i * n + j];
                let tol = 1e-5_f64.max(1e-5 * a.abs());
                assert!(
                    (a - f).abs() < tol,
                    "jac_y mismatch at ({},{}): analytical={}, fd={}",
                    i,
                    j,
                    a,
                    f
                );
            }
        }
    }

    #[test]
    fn test_jacobian_p_agrees_with_fd_with_reaction() {
        let grid = Grid3D::uniform(0.0, 1.0, 4, 0.0, 1.0, 4, 0.0, 1.0, 4);
        let bc = BoundaryConditions3D::all_zero_dirichlet();
        let mol = ParametricMOLSystem3D::heat_with_reaction(
            grid,
            0.05_f64,
            &bc,
            vec![0.5, 0.1],
            |_t, _x, _y, _z, u, p: &[f64]| -p[1] * u + p[2],
        );
        let n = mol.n_states();
        let np = mol.n_params();
        assert_eq!(np, 3);

        let y: Vec<f64> = (0..n).map(|i| 0.1 + (i as f64) * 0.01).collect();
        let mut jp_a = vec![0.0; n * np];
        mol.jacobian_p(0.0, &y, &mut jp_a);
        let jp_fd = fd_jacobian_p(&mol, 0.0, &y);

        for k in 0..np {
            for i in 0..n {
                let a = jp_a[k * n + i];
                let f = jp_fd[k * n + i];
                assert!(
                    (a - f).abs() < 1e-4,
                    "jac_p[k={},i={}] mismatch: analytical={}, fd={}",
                    k,
                    i,
                    a,
                    f
                );
            }
        }
    }

    #[test]
    fn test_forward_sensitivity_matches_analytical_decay() {
        // 3D heat equation with the analytic exp(-3π²αt) decay.
        // Sensitivity ∂u/∂α = -3π²t · u at cube centre.
        use numra_ode::SolverOptions;
        let alpha = 0.01_f64;
        let n = 11;
        let grid = Grid3D::uniform(0.0, 1.0, n, 0.0, 1.0, n, 0.0, 1.0, n);
        let bc = BoundaryConditions3D::all_zero_dirichlet();
        let mol = ParametricMOLSystem3D::heat(grid.clone(), alpha, &bc);

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

        let t_final = 0.3;
        let opts = SolverOptions::default().rtol(1e-6).atol(1e-9);
        let result =
            solve_forward_sensitivity::<Radau5, f64, _>(&mol, 0.0, t_final, &u0, &opts).unwrap();

        let mid = (nx_int / 2) * (nx_int * nx_int) + (nx_int / 2) * nx_int + (nx_int / 2);
        let exact_u = (pi * 0.5).sin().powi(3) * (-3.0 * pi * pi * alpha * t_final).exp();
        let exact_du_dalpha = -3.0 * pi * pi * t_final * exact_u;

        let computed_u = result.final_state()[mid];
        let last_t = result.t.len() - 1;
        let computed_du_dalpha = result.dyi_dpj(last_t, mid, 0);

        // Coarser 11³ grid + larger spatial discretisation error in 3D:
        // allow 5% rel error.
        let rel_err_u = (computed_u - exact_u).abs() / exact_u;
        let rel_err_s = (computed_du_dalpha - exact_du_dalpha).abs() / exact_du_dalpha.abs();
        assert!(
            rel_err_u < 0.05,
            "state: computed={}, exact={}, rel_err={}",
            computed_u,
            exact_u,
            rel_err_u
        );
        assert!(
            rel_err_s < 0.05,
            "∂u/∂α: computed={}, exact={}, rel_err={}",
            computed_du_dalpha,
            exact_du_dalpha,
            rel_err_s
        );
    }
}

//! Parametric 2D Method of Lines for forward sensitivity analysis.
//!
//! Wraps a 2D heat-equation MOL discretisation as a [`ParametricOdeSystem`],
//! letting users compute `∂y/∂p` for parameters `p = [α, reaction_p_0, ...]`
//! via [`numra_ode::sensitivity::solve_forward_sensitivity`] without
//! hand-rolling the parametric system.
//!
//! # Design
//!
//! For the equation `u_t = α∇²u + R(t, x, y, u, p_react)` discretised on a
//! uniform 2D grid by 5-point central differences, the assembled spatial
//! operator scales linearly with α. So the parametric system stores the
//! *un-scaled* Laplacian `L0` and the un-scaled boundary contribution
//! `bc_rhs_0` once; at each rhs / jacobian call it scales by the current
//! parameter `α`. No per-call matrix re-assembly.
//!
//! Mixed Dirichlet / Neumann boundary conditions both work automatically:
//! the Neumann ghost-point flux contribution `x_minus * 2*dx*flux` carries
//! `x_minus = α / dx²` (for a pure-Laplacian operator), so its α scaling
//! propagates through the full assembly. No split between α-scaled and
//! α-independent boundary terms is needed at v1 scope.
//!
//! # Parameter layout
//!
//! `params() = [α, reaction_p_0, reaction_p_1, ...]`. Slot 0 is reserved
//! for the diffusion coefficient α; the remaining slots are reaction
//! parameters. The reaction closure receives the *full* parameter slice
//! `&[S]` so it can index `p[1..]` for its own params (or `p[0]` for α
//! if the reaction depends on diffusion).
//!
//! Author: Moussa Leblouba
//! Date: 8 May 2026

use crate::boundary2d::BoundaryConditions2D;
use crate::grid::Grid2D;
use crate::sparse_assembly::{assemble_operator_2d, Operator2DCoefficients, SparseScalar};
use numra_linalg::SparseMatrix;
use numra_ode::ParametricOdeSystem;

/// Type alias for a parametric reaction closure: `(t, x, y, u, p) -> R`.
///
/// The full parameter slice is passed; the closure can read `p[0]` for the
/// diffusion coefficient α or `p[1..]` for its reaction-specific parameters.
type ParametricReactionFn<S> = Box<dyn Fn(S, S, S, S, &[S]) -> S + Send + Sync>;

/// Parametric 2D Method of Lines system.
///
/// See the module-level docs for the parameter layout, the linearity
/// argument that makes the scaled-operator path correct, and the
/// boundary-condition handling.
pub struct ParametricMOLSystem2D<S: SparseScalar> {
    grid: Grid2D<S>,
    /// Un-scaled Laplacian (built with α = 1).
    l0: SparseMatrix<S>,
    /// Un-scaled boundary RHS contribution (built with α = 1).
    bc_rhs_0: Vec<S>,
    /// Nominal parameter vector. `params[0]` = α, `params[1..]` = reaction.
    nominal_params: Vec<S>,
    /// Optional parametric reaction term `R(t, x, y, u, &p)`.
    reaction: Option<ParametricReactionFn<S>>,
}

impl<S: SparseScalar> ParametricMOLSystem2D<S> {
    /// Build a parametric heat equation `u_t = α∇²u` with the diffusion
    /// coefficient α as the single parameter.
    pub fn heat(grid: Grid2D<S>, alpha_nominal: S, bc: &BoundaryConditions2D<S>) -> Self {
        let coeffs = Operator2DCoefficients::laplacian();
        let (l0, bc_rhs_0) =
            assemble_operator_2d(&grid, &coeffs, bc).expect("Failed to assemble 2D Laplacian");
        Self {
            grid,
            l0,
            bc_rhs_0,
            nominal_params: vec![alpha_nominal],
            reaction: None,
        }
    }

    /// Build a parametric heat equation with a parametric reaction term.
    ///
    /// The full parameter vector is `[α, reaction_params...]` —
    /// `nominal_reaction_params` supplies nominal values for the reaction
    /// parameters only; α goes in slot 0 separately.
    ///
    /// The reaction closure receives `(t, x, y, u, &p_full)` where `p_full`
    /// is the full parameter slice (`p_full[0]` is α; `p_full[1..]` are the
    /// reaction parameters supplied here).
    pub fn heat_with_reaction<R>(
        grid: Grid2D<S>,
        alpha_nominal: S,
        bc: &BoundaryConditions2D<S>,
        nominal_reaction_params: Vec<S>,
        reaction: R,
    ) -> Self
    where
        R: Fn(S, S, S, S, &[S]) -> S + Send + Sync + 'static,
    {
        let coeffs = Operator2DCoefficients::laplacian();
        let (l0, bc_rhs_0) =
            assemble_operator_2d(&grid, &coeffs, bc).expect("Failed to assemble 2D Laplacian");
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

    /// Get the spatial grid.
    pub fn grid(&self) -> &Grid2D<S> {
        &self.grid
    }

    /// Number of interior points (the ODE state dimension).
    pub fn n_interior(&self) -> usize {
        self.grid.n_interior()
    }
}

impl<S: SparseScalar> ParametricOdeSystem<S> for ParametricMOLSystem2D<S> {
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
        // α · (L0·y + bc_rhs_0). Both terms scale linearly with α; see the
        // module-level docs for the linearity argument.
        let matvec = self.l0.mul_vec(y).expect("Sparse matvec failed");
        let n = self.grid.n_interior();
        for i in 0..n {
            dydt[i] = alpha * (matvec[i] + self.bc_rhs_0[i]);
        }

        // Pointwise reaction term, if present. The closure sees the full
        // parameter slice so it can index p[0] for α or p[1..] for its
        // own params.
        if let Some(ref reaction) = self.reaction {
            let nx_int = self.grid.nx_interior();
            for jj in 0..self.grid.ny_interior() {
                for ii in 0..nx_int {
                    let idx = jj * nx_int + ii;
                    let x = self.grid.x_grid.points()[ii + 1];
                    let y_coord = self.grid.y_grid.points()[jj + 1];
                    dydt[idx] = dydt[idx] + reaction(t, x, y_coord, y[idx], p);
                }
            }
        }
    }

    /// Analytical state Jacobian: `α · L0` plus the diagonal reaction
    /// contribution. Same structure as `MOLSystem2D::jacobian`, but the
    /// linear part is scaled by `α` from the parameter vector — and the
    /// reaction sees the full parameter slice.
    fn jacobian_y(&self, t: S, y: &[S], jac: &mut [S]) {
        let n = self.n_states();
        let nn = n * n;
        let alpha = self.nominal_params[0];

        for v in jac.iter_mut().take(nn) {
            *v = S::ZERO;
        }

        // α · L0 → row-major dense.
        let col_ptrs = self.l0.col_ptrs();
        let row_indices = self.l0.row_indices();
        let values = self.l0.values();
        for j in 0..n {
            for idx in col_ptrs[j]..col_ptrs[j + 1] {
                let i = row_indices[idx];
                jac[i * n + j] = alpha * values[idx];
            }
        }

        // Diagonal reaction Jacobian: `∂R_i/∂u_j = δ_ij · ∂R_i/∂u_i`
        // (pointwise reaction ⇒ no off-diagonal entries by structure).
        // FD on the closure at the nominal parameter vector. Same step
        // formula as `ParametricOdeSystem::jacobian_y` default.
        if let Some(ref reaction) = self.reaction {
            let h_factor = S::EPSILON.sqrt();
            let nx_int = self.grid.nx_interior();
            for jj in 0..self.grid.ny_interior() {
                for ii in 0..nx_int {
                    let idx = jj * nx_int + ii;
                    let x = self.grid.x_grid.points()[ii + 1];
                    let y_coord = self.grid.y_grid.points()[jj + 1];
                    let u = y[idx];
                    let h = h_factor * (S::ONE + u.abs());
                    let r0 = reaction(t, x, y_coord, u, &self.nominal_params);
                    let r1 = reaction(t, x, y_coord, u + h, &self.nominal_params);
                    let dr_du = (r1 - r0) / h;
                    jac[idx * n + idx] = jac[idx * n + idx] + dr_du;
                }
            }
        }
    }

    /// Analytical parameter Jacobian. Column 0 (the α slot) is `L0·y +
    /// bc_rhs_0` analytically, plus any `∂R/∂α` contribution from the
    /// reaction closure (FD-detected). Columns 1..N_s are pure reaction
    /// contributions (FD on the closure with the relevant slot perturbed).
    /// The reaction is pointwise so each column has only diagonal entries
    /// (one nonzero per grid point).
    fn jacobian_p(&self, t: S, y: &[S], jp: &mut [S]) {
        let n = self.n_states();
        let np = self.n_params();

        for v in jp.iter_mut().take(n * np) {
            *v = S::ZERO;
        }

        // Column 0 (α): linear part. ∂(α · L0·y + α · bc_rhs_0)/∂α
        //                          = L0·y + bc_rhs_0.
        let lin = self.l0.mul_vec(y).expect("Sparse matvec failed");
        for i in 0..n {
            jp[i] = lin[i] + self.bc_rhs_0[i];
        }

        // Reaction contribution to every column (including column 0 if R
        // depends on α). FD with the matching parameter slot perturbed.
        // Each grid point contributes only its own row — pointwise R.
        // Same step formula as `ParametricOdeSystem::jacobian_p` default.
        if let Some(ref reaction) = self.reaction {
            let h_factor = S::EPSILON.sqrt();
            let nx_int = self.grid.nx_interior();
            let mut p_pert = self.nominal_params.clone();
            for k in 0..np {
                let pk = self.nominal_params[k];
                let h = h_factor * (S::ONE + pk.abs());
                p_pert[k] = pk + h;
                for jj in 0..self.grid.ny_interior() {
                    for ii in 0..nx_int {
                        let idx = jj * nx_int + ii;
                        let x = self.grid.x_grid.points()[ii + 1];
                        let y_coord = self.grid.y_grid.points()[jj + 1];
                        let u = y[idx];
                        let r0 = reaction(t, x, y_coord, u, &self.nominal_params);
                        let r1 = reaction(t, x, y_coord, u, &p_pert);
                        let dr_dpk = (r1 - r0) / h;
                        jp[k * n + idx] = jp[k * n + idx] + dr_dpk;
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

    /// Trait-default FD Jacobian helper. Used for analytical-vs-FD
    /// agreement regression. Identical to the FD path in
    /// `ParametricOdeSystem::jacobian_y`/`jacobian_p` defaults
    /// (`sqrt(EPSILON) * (1 + |y|)`).
    fn fd_jacobian_y<Sys: ParametricOdeSystem<f64>>(sys: &Sys, t: f64, y: &[f64]) -> Vec<f64> {
        let n = sys.n_states();
        let p = sys.params().to_vec();
        let h_factor = f64::EPSILON.sqrt();
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
        let h_factor = f64::EPSILON.sqrt();
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
        let grid = Grid2D::uniform(0.0, 1.0, 7, 0.0, 1.0, 7);
        let bc = BoundaryConditions2D::all_zero_dirichlet();
        let mol = ParametricMOLSystem2D::heat(grid, 0.05_f64, &bc);
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
    fn test_jacobian_y_agrees_with_fd_with_reaction() {
        // Parametric reaction: R(u, p) = -p[1] * u^3   (Allen-Cahn-like with
        // a parametric reaction rate at slot 1).
        let grid = Grid2D::uniform(0.0, 1.0, 5, 0.0, 1.0, 5);
        let bc = BoundaryConditions2D::all_zero_dirichlet();
        let mol = ParametricMOLSystem2D::heat_with_reaction(
            grid,
            0.05_f64,
            &bc,
            vec![1.0],
            |_t, _x, _y, u, p: &[f64]| -p[1] * u * u * u,
        );
        let n = mol.n_states();
        let y: Vec<f64> = (0..n).map(|i| 0.1 + (i as f64) * 0.01).collect();

        let mut jac_a = vec![0.0; n * n];
        mol.jacobian_y(0.0, &y, &mut jac_a);
        let jac_fd = fd_jacobian_y(&mol, 0.0, &y);

        for i in 0..n {
            for j in 0..n {
                let a = jac_a[i * n + j];
                let f = jac_fd[i * n + j];
                assert!(
                    (a - f).abs() < 1e-4,
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
    fn test_jacobian_p_agrees_with_fd_no_reaction() {
        // Pure heat equation: only one parameter, α. ∂rhs/∂α = L·y + bc_rhs.
        let grid = Grid2D::uniform(0.0, 1.0, 7, 0.0, 1.0, 7);
        let bc = BoundaryConditions2D::all_zero_dirichlet();
        let mol = ParametricMOLSystem2D::heat(grid, 0.05_f64, &bc);
        let n = mol.n_states();
        let np = mol.n_params();
        assert_eq!(np, 1);

        let y: Vec<f64> = (0..n).map(|i| ((i + 1) as f64).sin()).collect();
        let mut jp_a = vec![0.0; n * np];
        mol.jacobian_p(0.0, &y, &mut jp_a);
        let jp_fd = fd_jacobian_p(&mol, 0.0, &y);

        for i in 0..n {
            let a = jp_a[i];
            let f = jp_fd[i];
            let tol = 1e-5_f64.max(1e-5 * a.abs());
            assert!(
                (a - f).abs() < tol,
                "jac_p[α] mismatch at i={}: analytical={}, fd={}",
                i,
                a,
                f
            );
        }
    }

    #[test]
    fn test_jacobian_p_agrees_with_fd_with_reaction() {
        // Two reaction params at slots 1 and 2: R(u, p) = -p[1] * u + p[2].
        let grid = Grid2D::uniform(0.0, 1.0, 5, 0.0, 1.0, 5);
        let bc = BoundaryConditions2D::all_zero_dirichlet();
        let mol = ParametricMOLSystem2D::heat_with_reaction(
            grid,
            0.05_f64,
            &bc,
            vec![0.5, 0.1],
            |_t, _x, _y, u, p: &[f64]| -p[1] * u + p[2],
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
        // Heat equation u_t = α∇²u on [0,1]² with zero Dirichlet BCs.
        // IC: u(x,y,0) = sin(πx)sin(πy).
        // Exact: u(x,y,t) = sin(πx)sin(πy)·exp(-2π²·α·t).
        // Sensitivity ∂u/∂α = -2π²t · u.
        // We compute the numerical sensitivity at t=0.5, compare to the
        // closed form at the cube centre.
        use numra_ode::SolverOptions;
        let alpha = 0.01_f64;
        let n = 17;
        let grid = Grid2D::uniform(0.0, 1.0, n, 0.0, 1.0, n);
        let bc = BoundaryConditions2D::all_zero_dirichlet();
        let mol = ParametricMOLSystem2D::heat(grid.clone(), alpha, &bc);

        let nx_int = n - 2;
        let n_int = nx_int * nx_int;

        let pi = std::f64::consts::PI;
        let mut u0 = vec![0.0; n_int];
        for jj in 0..nx_int {
            for ii in 0..nx_int {
                let x = grid.x_grid.points()[ii + 1];
                let yc = grid.y_grid.points()[jj + 1];
                u0[jj * nx_int + ii] = (pi * x).sin() * (pi * yc).sin();
            }
        }

        let t_final = 0.5;
        let opts = SolverOptions::default().rtol(1e-7).atol(1e-10);
        let result =
            solve_forward_sensitivity::<Radau5, f64, _>(&mol, 0.0, t_final, &u0, &opts).unwrap();

        // Centre cell.
        let mid = (nx_int / 2) * nx_int + (nx_int / 2);
        let exact_u =
            (pi * 0.5).sin() * (pi * 0.5).sin() * (-2.0 * pi * pi * alpha * t_final).exp();
        let exact_du_dalpha = -2.0 * pi * pi * t_final * exact_u;

        let computed_u = result.final_state()[mid];
        let last_t = result.t.len() - 1;
        let computed_du_dalpha = result.dyi_dpj(last_t, mid, 0);

        // Coarse 17² grid + low-order spatial discretisation: a few percent
        // error at the centre is normal.
        let rel_err_u = (computed_u - exact_u).abs() / exact_u;
        let rel_err_s = (computed_du_dalpha - exact_du_dalpha).abs() / exact_du_dalpha.abs();
        assert!(
            rel_err_u < 0.02,
            "state: computed={}, exact={}, rel_err={}",
            computed_u,
            exact_u,
            rel_err_u
        );
        assert!(
            rel_err_s < 0.02,
            "∂u/∂α: computed={}, exact={}, rel_err={}",
            computed_du_dalpha,
            exact_du_dalpha,
            rel_err_s
        );
    }
}

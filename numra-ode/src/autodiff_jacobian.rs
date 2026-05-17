//! Autodiff-derived state Jacobian for ODE systems.
//!
//! This module ships behind the `autodiff` cargo feature; callers who do
//! not opt in do not compile or pull in `numra-autodiff`. The feature gate
//! is the discoverability surface — there is no silent-substitution path
//! for the analytical-Jacobian default.
//!
//! [`AutodiffJacobianSystem`] adapts an RHS closure written against
//! `numra_autodiff::Dual<S>` into a plain [`crate::OdeSystem<S>`] whose
//! `jacobian()` is computed by forward-mode automatic differentiation —
//! exact to round-off, no `~sqrt(ε)` truncation error.
//!
//! # Composition with the IC API
//!
//! The adapter is an `OdeSystem<S>`. It plugs directly into both
//! [`crate::sensitivity::solve_initial_condition_sensitivity`] and any
//! [`crate::Solver`]; the IC API's internal wrapper forwards
//! `OdeSystem::jacobian` for its variational `J_y`, so passing an
//! `AutodiffJacobianSystem` to the IC API obtains the state-transition
//! matrix with AD-exact `J_y`.
//!
//! Author: Moussa Leblouba
//! Date: 17 May 2026

use std::cell::RefCell;

use numra_autodiff::Dual;
use numra_core::Scalar;

use crate::problem::OdeSystem;

/// `OdeSystem<S>` adapter whose `jacobian()` is computed by forward-mode AD.
///
/// The adapter owns an RHS closure typed against `Dual<S>` and a small
/// pre-allocated dual-buffer pair for the sweep. `dim` is the state
/// dimension `N` (the closure must read / write slices of length `N`).
///
/// # Cost
///
/// Per integrator step, the adapter performs:
///
/// - **1** `Dual<S>`-RHS evaluation for the primal [`OdeSystem::rhs`]
///   (tangents seeded to zero).
/// - **N** `Dual<S>`-RHS evaluations for [`OdeSystem::jacobian`], one per
///   input direction `e_j`.
///
/// The trait's forward-FD default ([`crate::problem::OdeSystem::jacobian`])
/// performs **N + 1** `S`-RHS evaluations per Jacobian. This adapter is
/// therefore in the **same complexity class** (`O(N)` RHS evals per
/// Jacobian), at a ~2× per-eval cost (each `Dual<S>` op carries one extra
/// fused-multiply-add for the tangent), in exchange for a Jacobian
/// accurate to round-off rather than `~sqrt(S::EPSILON)` (~`1.5e-8` for
/// `f64`). The accuracy delta materially benefits stiff implicit solvers
/// (Radau5, BDF, ESDIRK*) whose Newton convergence depends on Jacobian
/// fidelity.
///
/// **Zero allocation per call** after construction: the dual buffers are
/// owned by the adapter behind [`RefCell`], mirroring `AugmentedSystem`'s
/// lifted-scratch pattern (`sensitivity.rs:296-299`).
///
/// # Layout
///
/// `jacobian` writes a row-major `N × N` matrix `J[i*N + j] = ∂f_i/∂y_j`
/// — bit-for-bit compatible with [`crate::OdeSystem::jacobian`] and the
/// variational integrator's `J_y` consumption layout.
///
/// # Example
///
/// ```
/// # #[cfg(feature = "autodiff")] {
/// use numra_autodiff::Dual;
/// use numra_ode::AutodiffJacobianSystem;
/// use numra_ode::sensitivity::solve_initial_condition_sensitivity;
/// use numra_ode::{DoPri5, SolverOptions};
///
/// // Lotka-Volterra written once, against Dual<f64>, used for both the
/// // primal RHS and the AD-derived J_y.
/// let sys = AutodiffJacobianSystem::new(
///     |_t: f64, y: &[Dual<f64>], dy: &mut [Dual<f64>]| {
///         let a = Dual::constant(1.5);
///         let b = Dual::constant(1.0);
///         let c = Dual::constant(3.0);
///         let d = Dual::constant(1.0);
///         dy[0] = a * y[0] - b * y[0] * y[1];
///         dy[1] = c * y[0] * y[1] - d * y[1];
///     },
///     2,
/// );
///
/// let result = solve_initial_condition_sensitivity::<DoPri5, f64, _>(
///     &sys,
///     0.0, 0.5,
///     &[1.0, 1.0],
///     &SolverOptions::default().rtol(1e-8).atol(1e-11),
/// ).unwrap();
///
/// assert!(result.success());
/// assert_eq!(result.dim(), 2);
/// # }
/// ```
pub struct AutodiffJacobianSystem<S: Scalar, F>
where
    F: Fn(S, &[Dual<S>], &mut [Dual<S>]),
{
    rhs_dual: F,
    dim: usize,
    dual_y: RefCell<Vec<Dual<S>>>,
    dual_dy: RefCell<Vec<Dual<S>>>,
}

impl<S: Scalar, F> AutodiffJacobianSystem<S, F>
where
    F: Fn(S, &[Dual<S>], &mut [Dual<S>]),
{
    /// Build the adapter from a `Dual<S>`-typed RHS closure and the state
    /// dimension `N`.
    pub fn new(rhs_dual: F, dim: usize) -> Self {
        let zero = Dual::constant(S::ZERO);
        Self {
            rhs_dual,
            dim,
            dual_y: RefCell::new(vec![zero; dim]),
            dual_dy: RefCell::new(vec![zero; dim]),
        }
    }
}

impl<S: Scalar, F> OdeSystem<S> for AutodiffJacobianSystem<S, F>
where
    F: Fn(S, &[Dual<S>], &mut [Dual<S>]),
{
    fn dim(&self) -> usize {
        self.dim
    }

    fn rhs(&self, t: S, y: &[S], dy: &mut [S]) {
        let n = self.dim;
        let zero = Dual::constant(S::ZERO);
        let mut dual_y = self.dual_y.borrow_mut();
        let mut dual_dy = self.dual_dy.borrow_mut();
        for i in 0..n {
            dual_y[i] = Dual::constant(y[i]); // primal only — zero tangent
            dual_dy[i] = zero;
        }
        (self.rhs_dual)(t, &dual_y, &mut dual_dy);
        for i in 0..n {
            dy[i] = dual_dy[i].value();
        }
    }

    fn jacobian(&self, t: S, y: &[S], jac: &mut [S]) {
        let n = self.dim;
        let zero = Dual::constant(S::ZERO);
        let mut dual_y = self.dual_y.borrow_mut();
        let mut dual_dy = self.dual_dy.borrow_mut();
        // Initialise primals once.
        for i in 0..n {
            dual_y[i] = Dual::constant(y[i]);
        }
        // Sweep N directions e_j. Each pass reads `dual_y` with one entry
        // bumped to `Dual::variable(y[j])` (unit tangent), all others
        // `Dual::constant(y[i])`. After the pass we restore that entry to
        // primal-only before bumping the next, so the buffer stays in the
        // "all-constant" state between sweeps.
        for j in 0..n {
            dual_y[j] = Dual::variable(y[j]); // seed e_j
            for i in 0..n {
                dual_dy[i] = zero;
            }
            (self.rhs_dual)(t, &dual_y, &mut dual_dy);
            // Extract column j of J: row-major J[i*n + j] = ∂f_i/∂y_j.
            for i in 0..n {
                jac[i * n + j] = dual_dy[i].deriv();
            }
            dual_y[j] = Dual::constant(y[j]); // restore primal-only
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// AD Jacobian of `f(y) = [y_0^2 + y_1, y_0 * y_1]` at (3, 4):
    ///   J = [[2*y_0, 1], [y_1, y_0]] = [[6, 1], [4, 3]]   (row-major)
    #[test]
    fn ad_jacobian_2x2_nonlinear() {
        let sys = AutodiffJacobianSystem::<f64, _>::new(
            |_t, y: &[Dual<f64>], dy: &mut [Dual<f64>]| {
                dy[0] = y[0] * y[0] + y[1];
                dy[1] = y[0] * y[1];
            },
            2,
        );
        let y = [3.0_f64, 4.0];
        let mut jac = [0.0_f64; 4];
        sys.jacobian(0.0, &y, &mut jac);
        // J[i*n + j] = ∂f_i/∂y_j
        assert!((jac[0] - 6.0).abs() < 1e-14, "∂f0/∂y0 = 2*y0");
        assert!((jac[1] - 1.0).abs() < 1e-14, "∂f0/∂y1 = 1");
        assert!((jac[2] - 4.0).abs() < 1e-14, "∂f1/∂y0 = y1");
        assert!((jac[3] - 3.0).abs() < 1e-14, "∂f1/∂y1 = y0");
    }

    /// AD primal RHS agrees with the f64 equivalent within round-off.
    #[test]
    fn ad_rhs_matches_f64_primal() {
        let sys = AutodiffJacobianSystem::<f64, _>::new(
            |_t, y: &[Dual<f64>], dy: &mut [Dual<f64>]| {
                dy[0] = y[0] * y[0] + y[1];
                dy[1] = y[0] * y[1];
            },
            2,
        );
        let y = [3.0_f64, 4.0];
        let mut dy = [0.0_f64; 2];
        sys.rhs(0.0, &y, &mut dy);
        assert!((dy[0] - (9.0 + 4.0)).abs() < 1e-14);
        assert!((dy[1] - (3.0 * 4.0)).abs() < 1e-14);
    }

    /// Successive `jacobian()` calls reuse the same scratch buffers — no
    /// per-call allocation visible at the public API (verified indirectly
    /// by exercising the call repeatedly without panic or growth).
    #[test]
    fn ad_jacobian_reentrant() {
        let sys = AutodiffJacobianSystem::<f64, _>::new(
            |_t, y: &[Dual<f64>], dy: &mut [Dual<f64>]| {
                dy[0] = y[0] * y[1];
                dy[1] = y[1] - y[0];
            },
            2,
        );
        let mut jac = [0.0_f64; 4];
        for k in 0..10_000 {
            let y = [k as f64 * 0.01, (k as f64) * 0.02 - 1.0];
            sys.jacobian(0.0, &y, &mut jac);
            // ∂f0/∂y0 = y[1]; ∂f0/∂y1 = y[0]; ∂f1/∂y0 = -1; ∂f1/∂y1 = 1.
            assert!((jac[0] - y[1]).abs() < 1e-14);
            assert!((jac[1] - y[0]).abs() < 1e-14);
            assert!((jac[2] - (-1.0)).abs() < 1e-14);
            assert!((jac[3] - 1.0).abs() < 1e-14);
        }
    }
}

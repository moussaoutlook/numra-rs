//! F-IC-SENS-MASS-SURFACE regression suite (Foundation Spec §6 #14, 0.1.5).
//!
//! Six tests pinning the M-aware variational integration end-to-end:
//!
//!  1. `path_a_non_identity_mass_param_sensitivity`        — forward sensitivity on `2y' = -p y`
//!  2. `path_a_singular_mass_dae_param_sensitivity`        — forward sensitivity on index-1 linear DAE
//!  3. `path_b_non_identity_mass_ic_sensitivity`           — IC sensitivity on `2y' = -y`, before/after
//!  4. `path_b_singular_mass_dae_ic_sensitivity`           — IC sensitivity on index-1 linear DAE, before/after
//!  5. `augmented_system_lifts_mass_surface_block_diagonally` — structural test on `AugmentedSystem`
//!  6. The matching structural test on `IcAsParametric` lives in
//!     `numra-ode/src/sensitivity.rs::tests::ic_as_parametric_forwards_mass_surface`
//!     because `IcAsParametric` is private.
//!
//! ## Path-typed compile-time / runtime-fact distinction
//!
//! **Path A tests (1, 2) would not compile against 0.1.4.** The pre-0.1.5
//! `ParametricOdeSystem` trait at `HEAD~..:numra-ode/src/sensitivity.rs:152-281`
//! (commit `98ee6ca`, 0.1.4 head) contained exactly: `n_states`, `n_params`,
//! `params`, `rhs_with_params`, `rhs`, `jacobian_y`, `jacobian_p`,
//! `initial_sensitivity`, `has_analytical_jacobian_y`,
//! `has_analytical_jacobian_p` — and none of `has_mass_matrix`,
//! `mass_matrix`, `is_singular_mass`, `algebraic_indices`, `is_autonomous`.
//! A user wanting to declare M on a parametric system literally could not
//! express the problem. The Path A "before" state is therefore a
//! **compile-time fact**, not a runtime numerical failure.
//!
//! **Path B tests (3, 4) compile against 0.1.4** and exhibit the documented
//! runtime failure shapes (M⁻¹-factor scaling for non-identity-M;
//! algebraic-as-differential dynamics for singular-M) because the wrapped
//! `OdeSystem` *can* declare M but `IcAsParametric` silently drops it on the
//! way into the augmented integration. The Path B post-fix assertions are
//! complemented by falsifiability-gate assertions on the pre-fix-bug values
//! — if the `IcAsParametric` forwarding or the `AugmentedSystem` overrides
//! are reverted, the post-fix assertion fails AND the gate assertion fires
//! with the diagnostic pre-fix-bug value. Same revert-confirm-restore
//! discipline as F-SOLVER-FIELDS and F-IC-SENS.
//!
//! ## IC-projection caveat for singular-M tests
//!
//! Tests 2 and 4 pin the M-aware variational integration only. The
//! finite-difference IC perturbation in `fd_jacobian_y_inline`
//! (sensitivity.rs:394) does not project the perturbed initial condition
//! onto the constraint manifold for singular-M systems; the resulting Φ
//! is "the sensitivity given a manifold-inconsistent IC perturbation,
//! correctly integrated against M-aware variational dynamics" — strictly
//! improved over pre-0.1.5 but not manifold-projected. Tracked as
//! Foundation Spec §7 #9.

#![allow(clippy::identity_op, clippy::erasing_op, clippy::needless_range_loop)]

use numra_ode::sensitivity::{
    solve_forward_sensitivity, solve_initial_condition_sensitivity, AugmentedSystem,
    ParametricOdeSystem,
};
use numra_ode::{OdeSystem, Radau5, SolverOptions};

// ============================================================================
// Test 1 — Path A non-identity-M forward sensitivity (post-fix only)
//
// System:  2 y' = -p y                                  (scalar, M = 2)
// Equiv ODE:  y' = -(p/2) y
// State:      y(t)        = y₀ · exp(-p t / 2)
// Sens:       ∂y/∂p (t)   = -(t/2) · y(t)
//
// At p = 1, y₀ = 1, t = 2:  y(2)        = exp(-1)        ≈ 0.36788
//                            ∂y/∂p (2)   = -exp(-1)       ≈ -0.36788
//
// **Compile-time fact**: this test would not compile against 0.1.4
// because `ParametricOdeSystem` lacked the M surface. There is no
// pre-fix runtime "bug" to compare against on Path A — the bug is the
// inability to express the problem at all.
// ============================================================================

struct ScalarParametricMassOde {
    p: [f64; 1],
}

impl ParametricOdeSystem<f64> for ScalarParametricMassOde {
    fn n_states(&self) -> usize {
        1
    }
    fn n_params(&self) -> usize {
        1
    }
    fn params(&self) -> &[f64] {
        &self.p
    }
    fn rhs_with_params(&self, _t: f64, y: &[f64], p: &[f64], dy: &mut [f64]) {
        // M y' = f   with  f = -p y  (and M = 2 below).
        dy[0] = -p[0] * y[0];
    }
    fn jacobian_y(&self, _t: f64, _y: &[f64], jy: &mut [f64]) {
        jy[0] = -self.p[0];
    }
    fn jacobian_p(&self, _t: f64, y: &[f64], jp: &mut [f64]) {
        jp[0] = -y[0];
    }
    fn has_analytical_jacobian_y(&self) -> bool {
        true
    }
    fn has_analytical_jacobian_p(&self) -> bool {
        true
    }
    fn has_mass_matrix(&self) -> bool {
        true
    }
    fn mass_matrix(&self, m: &mut [f64]) {
        m[0] = 2.0;
    }
}

#[test]
fn path_a_non_identity_mass_param_sensitivity() {
    let sys = ScalarParametricMassOde { p: [1.0] };
    let y0 = [1.0];
    let opts = SolverOptions::default().rtol(1e-10).atol(1e-12);
    let r = solve_forward_sensitivity::<Radau5, _, _>(&sys, 0.0, 2.0, &y0, &opts).unwrap();
    assert!(r.success, "solver failed: {}", r.message);

    let last = r.len() - 1;
    let t_f = r.t[last];
    let p_val = 1.0_f64;
    let y_truth = (-p_val * t_f / 2.0).exp();
    let dydp_truth = -(t_f / 2.0) * y_truth;

    let y_final = r.y_at(last)[0];
    let dydp_final = r.dyi_dpj(last, 0, 0);

    assert!(
        (y_final - y_truth).abs() < 1e-6,
        "post-fix state mismatch: y(2)={} vs analytic exp(-1)={}",
        y_final,
        y_truth,
    );
    assert!(
        (dydp_final - dydp_truth).abs() < 1e-6,
        "post-fix sensitivity mismatch: ∂y/∂p(2)={} vs analytic -exp(-1)={}",
        dydp_final,
        dydp_truth,
    );
}

// ============================================================================
// Test 2 — Path A singular-M DAE forward sensitivity (post-fix only)
//
// System:  y₁' = -p y₂                                  (M[0,0] = 1)
//          0   = sin(t) - y₂                            (M[1,1] = 0, algebraic)
// Reduced ODE on manifold (y₂ = sin(t)):
//          y₁(t) = y₁(0) + p · (cos(t) - 1)
// Sens:    ∂y₁/∂p (t) = cos(t) - 1
//          ∂y₂/∂p (t) = 0          (algebraic, no parameter dependence)
//
// At p = 1, y₁(0) = 0, y₂(0) = 0, t = π/2:
//   y₁(π/2) = cos(π/2) - 1 = -1
//   ∂y₁/∂p (π/2) = cos(π/2) - 1 = -1
//
// **Compile-time fact** comment from Test 1 applies — this test would
// not compile against 0.1.4.
//
// **IC-projection caveat**: this test pins M-aware variational
// integration only; the FD-IC-perturbation manifold-projection question
// is Foundation Spec §7 #9.
// ============================================================================

struct LinearDaeParametric {
    p: [f64; 1],
}

impl ParametricOdeSystem<f64> for LinearDaeParametric {
    fn n_states(&self) -> usize {
        2
    }
    fn n_params(&self) -> usize {
        1
    }
    fn params(&self) -> &[f64] {
        &self.p
    }
    fn rhs_with_params(&self, t: f64, y: &[f64], p: &[f64], dy: &mut [f64]) {
        // Row 0 (differential): f₀ = -p · y₁
        // Row 1 (algebraic):    f₁ = sin(t) - y₁
        dy[0] = -p[0] * y[1];
        dy[1] = t.sin() - y[1];
    }
    fn jacobian_y(&self, _t: f64, _y: &[f64], jy: &mut [f64]) {
        // Row-major 2×2.
        jy[0 * 2 + 0] = 0.0;
        jy[0 * 2 + 1] = -self.p[0];
        jy[1 * 2 + 0] = 0.0;
        jy[1 * 2 + 1] = -1.0;
    }
    fn jacobian_p(&self, _t: f64, y: &[f64], jp: &mut [f64]) {
        // Column-major 2×1.
        jp[0] = -y[1];
        jp[1] = 0.0;
    }
    fn has_analytical_jacobian_y(&self) -> bool {
        true
    }
    fn has_analytical_jacobian_p(&self) -> bool {
        true
    }
    fn has_mass_matrix(&self) -> bool {
        true
    }
    fn mass_matrix(&self, m: &mut [f64]) {
        // diag(1, 0) row-major.
        m[0] = 1.0;
        m[1] = 0.0;
        m[2] = 0.0;
        m[3] = 0.0;
    }
    fn is_singular_mass(&self) -> bool {
        true
    }
    fn algebraic_indices(&self) -> Vec<usize> {
        vec![1]
    }
}

#[test]
fn path_a_singular_mass_dae_param_sensitivity() {
    use std::f64::consts::FRAC_PI_2;
    let sys = LinearDaeParametric { p: [1.0] };
    let y0 = [0.0, 0.0]; // y₂(0) = sin(0) = 0 is consistent.
    let opts = SolverOptions::default().rtol(1e-10).atol(1e-12);
    let r = solve_forward_sensitivity::<Radau5, _, _>(&sys, 0.0, FRAC_PI_2, &y0, &opts).unwrap();
    assert!(r.success, "solver failed: {}", r.message);

    let last = r.len() - 1;
    let t_f = r.t[last];
    let y1_truth = t_f.cos() - 1.0;
    let y2_truth = t_f.sin();
    let dy1_dp_truth = t_f.cos() - 1.0;

    let y_final = r.y_at(last);
    assert!(
        (y_final[0] - y1_truth).abs() < 1e-5,
        "post-fix y₁(π/2) mismatch: {} vs cos(π/2)-1 = {}",
        y_final[0],
        y1_truth,
    );
    assert!(
        (y_final[1] - y2_truth).abs() < 1e-5,
        "post-fix y₂(π/2) mismatch: {} vs sin(π/2) = {} \
         — algebraic constraint not enforced, M-awareness broken",
        y_final[1],
        y2_truth,
    );

    let dy1_dp = r.dyi_dpj(last, 0, 0);
    assert!(
        (dy1_dp - dy1_dp_truth).abs() < 1e-3,
        "post-fix ∂y₁/∂p(π/2) mismatch: {} vs cos(π/2)-1 = {}",
        dy1_dp,
        dy1_dp_truth,
    );
}

// ============================================================================
// Test 3 — Path B non-identity-M IC sensitivity (before/after)
//
// System (as OdeSystem):  2 y' = -y         (scalar, M = 2)
// True trajectory:        y(t) = y₀ · exp(-t/2)
// True STM:               Φ(t) = ∂y/∂y₀ = exp(-t/2)
//
// At t = 2:  Φ_truth(2) = exp(-1) ≈ 0.36788.
//
// **Construction note.** This test deliberately uses a *scalar* system
// (1×1 M) so the M-blind reading reduces to a clean closed-form
// exponential `exp(A·t)`. Higher-dimensional non-identity-M systems
// (e.g. FE-PDE mass matrices) exhibit the *same* precise bug shape —
// the matrix-exponential's argument is off by a factor of M⁻¹
// (Φ_truth = expm(M⁻¹·A·t), Φ_bug = expm(A·t)) — but produce no scalar
// reduction. Scalar choice keeps the falsifiability gate's pre-fix-bug
// value derivable in closed form, mirroring Test 4's closed-form
// derivation of the algebraic-as-differential bug for the DAE case.
//
// **Pre-fix bug runtime fact**: pre-0.1.5 `IcAsParametric` silently
// dropped `M = 2`, presenting an M-blind system to the augmented
// integration. The augmented `rhs` then filled the state block with
// `f(t, y) = -y` and `AugmentedSystem::has_mass_matrix()` returned
// `false`, so the M-aware solver took the identity-M path. Net result:
// the augmented state was integrated as `y' = -y`, giving
//   y_bug(t) = y₀ · exp(A · t) = exp(-t)              [= expm(A·t) for scalar A]
//   Φ_bug(t) = exp(A · t)      = exp(-t)              [variational of the wrong ODE]
// vs the truth, which solves `M · Φ' = A · Φ`:
//   Φ_truth(t) = exp(M⁻¹ · A · t) = exp(-t/2).
// At t = 2:  Φ_bug(2) = exp(-2) ≈ 0.13534, Φ_truth(2) = exp(-1) ≈
// 0.36788. The discrepancy is the M⁻¹-factor in the matrix-exponential
// argument (general DAE statement); for this scalar system the ratio
// collapses to Φ_bug = Φ_truth² (1×1 specific shape).
//
// **Falsifiability gate**: reverting the `IcAsParametric::mass_matrix`
// forwarding or the `AugmentedSystem::mass_matrix` lift produces
// Φ(2) ≈ 0.13534 (the pre-fix-bug value). The second assertion below
// fires in that case, with a clearer diagnostic than the first.
// ============================================================================

struct ScalarMassOde;

impl OdeSystem<f64> for ScalarMassOde {
    fn dim(&self) -> usize {
        1
    }
    fn rhs(&self, _t: f64, y: &[f64], dydt: &mut [f64]) {
        dydt[0] = -y[0]; // f, not M⁻¹ f.
    }
    fn jacobian(&self, _t: f64, _y: &[f64], jac: &mut [f64]) {
        jac[0] = -1.0;
    }
    fn has_mass_matrix(&self) -> bool {
        true
    }
    fn mass_matrix(&self, m: &mut [f64]) {
        m[0] = 2.0;
    }
}

#[test]
fn path_b_non_identity_mass_ic_sensitivity() {
    let sys = ScalarMassOde;
    let y0 = [1.0];
    let opts = SolverOptions::default().rtol(1e-10).atol(1e-12);
    let r =
        solve_initial_condition_sensitivity::<Radau5, _, _>(&sys, 0.0, 2.0, &y0, &opts).unwrap();
    assert!(r.success(), "solver failed: {}", r.message());

    let last = r.len() - 1;
    let t_f = r.t()[last];

    let y_truth = (-t_f / 2.0).exp();
    let phi_truth = (-t_f / 2.0).exp();
    let phi_bug = (-t_f).exp(); // Pre-fix M-blind value, for the falsifiability gate.

    let y_final = r.y(last)[0];
    let phi_final = r.phi_ij(last, 0, 0);

    // Post-fix correctness.
    assert!(
        (y_final - y_truth).abs() < 1e-6,
        "post-fix state mismatch: y(2)={} vs analytic exp(-1)={}",
        y_final,
        y_truth,
    );
    assert!(
        (phi_final - phi_truth).abs() < 1e-6,
        "post-fix Φ mismatch: Φ(2)={} vs analytic exp(-1)={}",
        phi_final,
        phi_truth,
    );

    // Falsifiability gate: post-fix Φ must NOT equal the pre-fix-bug
    // value. Reverting the M-forwarding wiring would make Φ(2) match
    // exp(-2) ≈ 0.13534 — this assertion catches that silent reversion.
    assert!(
        (phi_final - phi_bug).abs() > 0.1,
        "Φ matches the pre-fix M-blind bug value ({} ≈ {}); IcAsParametric or \
         AugmentedSystem M-forwarding is silently broken",
        phi_final,
        phi_bug,
    );
}

// ============================================================================
// Test 4 — Path B singular-M DAE IC sensitivity (before/after, state only)
//
// System (as OdeSystem):  y₁' = -y₂           (M[0,0] = 1)
//                          0   = sin(t) - y₂   (M[1,1] = 0, algebraic)
// IC consistent: y₂(0) = sin(0) = 0; y₁(0) is free, pick 1.
// True trajectory (on manifold):
//   y₂(t) = sin(t)
//   y₁(t) = y₁(0) + cos(t) - 1 = cos(t)        (with y₁(0) = 1)
//
// At t = 2:  y₁_truth(2) = cos(2) ≈ -0.41615
//            y₂_truth(2) = sin(2) ≈  0.90930
//
// **Pre-fix bug runtime fact**: M dropped, the augmented integration
// runs `y' = f(t, y)` for both rows, treating the algebraic equation
// as differential:
//   y₁' = -y₂
//   y₂' = sin(t) - y₂        ← was meant to be `0 = sin(t) - y₂`
// The y₂ equation is a forced linear ODE; with y₂(0) = 0 its solution
// is y₂_bug(t) = (1/2)·(sin(t) - cos(t) + e^(-t)).
// At t = 2: y₂_bug(2) = 0.5·(sin(2) - cos(2) + e^(-2))
//                     = 0.5·(0.90930 - (-0.41615) + 0.13534)
//                     ≈ 0.73039
// That's the "algebraic-as-differential" failure shape — the algebraic
// row's behavior is qualitatively wrong, not merely scaled.
//
// **Falsifiability gate** on y₂(2) as above for Test 3.
//
// **IC-projection scope note**: this test asserts on the state
// trajectory, not on Φ. The IC API initialises Φ(0) = I, which is
// inconsistent with the algebraic constraint ∂y₂/∂y₀_k = 0 — the DAE
// solver's first-step behavior under this inconsistency is the IC-
// manifold-projection issue tracked as Foundation Spec §7 #9. After
// that issue is resolved, Test 4 can be extended with Φ assertions.
// ============================================================================

struct LinearDaeOde;

impl OdeSystem<f64> for LinearDaeOde {
    fn dim(&self) -> usize {
        2
    }
    fn rhs(&self, t: f64, y: &[f64], dydt: &mut [f64]) {
        dydt[0] = -y[1];
        dydt[1] = t.sin() - y[1];
    }
    fn jacobian(&self, _t: f64, _y: &[f64], jac: &mut [f64]) {
        // Row-major 2×2.
        jac[0 * 2 + 0] = 0.0;
        jac[0 * 2 + 1] = -1.0;
        jac[1 * 2 + 0] = 0.0;
        jac[1 * 2 + 1] = -1.0;
    }
    fn has_mass_matrix(&self) -> bool {
        true
    }
    fn mass_matrix(&self, m: &mut [f64]) {
        m[0] = 1.0;
        m[1] = 0.0;
        m[2] = 0.0;
        m[3] = 0.0;
    }
    fn is_singular_mass(&self) -> bool {
        true
    }
    fn algebraic_indices(&self) -> Vec<usize> {
        vec![1]
    }
}

#[test]
fn path_b_singular_mass_dae_ic_sensitivity() {
    let sys = LinearDaeOde;
    let y0 = [1.0, 0.0]; // sin(0) - 0 = 0 → consistent.
    let opts = SolverOptions::default().rtol(1e-8).atol(1e-10);
    let r =
        solve_initial_condition_sensitivity::<Radau5, _, _>(&sys, 0.0, 2.0, &y0, &opts).unwrap();
    assert!(r.success(), "solver failed: {}", r.message());

    let last = r.len() - 1;
    let t_f = r.t()[last];
    let y1_truth = t_f.cos();
    let y2_truth = t_f.sin();
    let y2_bug = 0.5 * (t_f.sin() - t_f.cos() + (-t_f).exp());

    let y_final = r.y(last);

    // Post-fix correctness — algebraic constraint enforced.
    assert!(
        (y_final[1] - y2_truth).abs() < 1e-4,
        "post-fix y₂(2) mismatch: {} vs sin(2) = {} \
         — algebraic constraint not enforced, M-awareness broken",
        y_final[1],
        y2_truth,
    );
    assert!(
        (y_final[0] - y1_truth).abs() < 1e-4,
        "post-fix y₁(2) mismatch: {} vs cos(2) = {}",
        y_final[0],
        y1_truth,
    );

    // Falsifiability gate: post-fix y₂ must NOT equal the pre-fix
    // algebraic-as-differential bug value.
    assert!(
        (y_final[1] - y2_bug).abs() > 0.05,
        "y₂ matches the pre-fix algebraic-as-differential bug value \
         ({} ≈ {}); IcAsParametric or AugmentedSystem M-forwarding is \
         silently broken",
        y_final[1],
        y2_bug,
    );
}

// ============================================================================
// Test 5 — Structural: AugmentedSystem lifts the M surface block-diagonally
//
// Construct a `ParametricOdeSystem` mock declaring:
//   n_states = 3, n_params = 2
//   has_mass_matrix = true
//   mass_matrix = diag(2, 3, 0)
//   is_singular_mass = true
//   algebraic_indices = [2]
//   is_autonomous = true
//
// Wrap in `AugmentedSystem`. Assert the `OdeSystem` impl reports the
// correctly-lifted values:
//   augmented dim = 3 · (1 + 2) = 9
//   augmented M = block_diag(M, M, M)  (3 copies)
//   algebraic_indices_aug = {2, 5, 8}
//   is_singular_mass, has_mass_matrix, is_autonomous all forward `true`.
//
// This is the inward-direction structural check on Step B's lift —
// completes the boundary-symmetry pair with Test 6 (`IcAsParametric`
// forwarding, in `src/sensitivity.rs::tests`).
// ============================================================================

struct MockParametric {
    p: [f64; 2],
}

impl ParametricOdeSystem<f64> for MockParametric {
    fn n_states(&self) -> usize {
        3
    }
    fn n_params(&self) -> usize {
        2
    }
    fn params(&self) -> &[f64] {
        &self.p
    }
    fn rhs_with_params(&self, _t: f64, _y: &[f64], _p: &[f64], dy: &mut [f64]) {
        for slot in dy.iter_mut() {
            *slot = 0.0;
        }
    }
    fn is_autonomous(&self) -> bool {
        true
    }
    fn has_mass_matrix(&self) -> bool {
        true
    }
    fn mass_matrix(&self, m: &mut [f64]) {
        // diag(2, 3, 0) row-major 3×3.
        for slot in m.iter_mut() {
            *slot = 0.0;
        }
        m[0 * 3 + 0] = 2.0;
        m[1 * 3 + 1] = 3.0;
        m[2 * 3 + 2] = 0.0;
    }
    fn is_singular_mass(&self) -> bool {
        true
    }
    fn algebraic_indices(&self) -> Vec<usize> {
        vec![2]
    }
}

#[test]
fn augmented_system_lifts_mass_surface_block_diagonally() {
    let aug = AugmentedSystem::new(MockParametric { p: [0.0, 0.0] });

    assert!(aug.is_autonomous(), "is_autonomous not forwarded");
    assert!(aug.has_mass_matrix(), "has_mass_matrix not forwarded");
    assert!(aug.is_singular_mass(), "is_singular_mass not forwarded");
    assert_eq!(
        aug.algebraic_indices(),
        vec![2, 5, 8],
        "algebraic indices not lifted to {{2, 5, 8}}",
    );

    // Augmented mass matrix: dim = 3 * (1 + 2) = 9; block_diag(M, M, M).
    let dim = aug.dim();
    assert_eq!(dim, 9);
    let mut m_aug = vec![0.0_f64; dim * dim];
    aug.mass_matrix(&mut m_aug);

    // Diagonal entries: positions (0,0), (3,3), (6,6) → 2; (1,1), (4,4), (7,7) → 3;
    // (2,2), (5,5), (8,8) → 0 (algebraic). All other entries → 0.
    for b in 0..3 {
        let off = b * 3;
        assert_eq!(
            m_aug[(off + 0) * dim + (off + 0)],
            2.0,
            "block {} M[0,0]",
            b
        );
        assert_eq!(
            m_aug[(off + 1) * dim + (off + 1)],
            3.0,
            "block {} M[1,1]",
            b
        );
        assert_eq!(
            m_aug[(off + 2) * dim + (off + 2)],
            0.0,
            "block {} M[2,2]",
            b
        );
    }
    // Spot-check off-block zero entries (cross-block coupling must be zero).
    assert_eq!(m_aug[0 * dim + 3], 0.0, "cross-block (0,3) should be 0");
    assert_eq!(m_aug[3 * dim + 0], 0.0, "cross-block (3,0) should be 0");
    assert_eq!(m_aug[0 * dim + 6], 0.0, "cross-block (0,6) should be 0");
    // Off-diagonal within a block (M is diagonal in this mock).
    assert_eq!(m_aug[0 * dim + 1], 0.0, "intra-block (0,1) should be 0");
}

// ============================================================================
// Sanity test: identity-M parametric host inherits trait defaults and
// the augmented system reports has_mass_matrix = false, dim-sized
// identity, no algebraic indices. Pins the "0.1.4 users with identity-M
// systems are unaffected" claim from the CHANGELOG affected-scope.
// ============================================================================

struct IdentityMockParametric {
    p: [f64; 1],
}

impl ParametricOdeSystem<f64> for IdentityMockParametric {
    fn n_states(&self) -> usize {
        2
    }
    fn n_params(&self) -> usize {
        1
    }
    fn params(&self) -> &[f64] {
        &self.p
    }
    fn rhs_with_params(&self, _t: f64, _y: &[f64], _p: &[f64], dy: &mut [f64]) {
        for slot in dy.iter_mut() {
            *slot = 0.0;
        }
    }
    // All M-surface methods inherit defaults: has_mass_matrix = false,
    // mass_matrix = identity, is_singular_mass = false, algebraic_indices = [].
}

#[test]
fn augmented_system_identity_m_default_path_unchanged() {
    let aug = AugmentedSystem::new(IdentityMockParametric { p: [0.0] });
    assert!(!aug.has_mass_matrix());
    assert!(!aug.is_singular_mass());
    assert!(aug.algebraic_indices().is_empty());

    // Identity-M path: aug.mass_matrix() returns the full dim × dim
    // identity (semantically distinct from the M-aware identity branch
    // — see `AugmentedSystem::mass_matrix` rustdoc).
    let dim = aug.dim(); // 2 * (1 + 1) = 4.
    assert_eq!(dim, 4);
    let mut m = vec![0.0_f64; dim * dim];
    aug.mass_matrix(&mut m);
    for i in 0..dim {
        for j in 0..dim {
            let expected = if i == j { 1.0 } else { 0.0 };
            assert_eq!(
                m[i * dim + j],
                expected,
                "identity-M default at ({}, {})",
                i,
                j
            );
        }
    }
}

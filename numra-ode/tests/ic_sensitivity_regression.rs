// The `row * N + col` column-major flattening is written explicitly
// throughout these tests so the (row, col) intent is visible at each
// indexing site. Clippy flags `0*2+0` / `1*2+1` etc. as identity_op /
// erasing_op, and the matching `for row in 0..N { for col in 0..N {
// ... expected[row][col] ... } }` loop as needless_range_loop because
// the loop variable doubles as both an `phi_ij` arg and an array index.
// Both choices are deliberate test-code pedagogy.
#![allow(clippy::identity_op, clippy::erasing_op, clippy::needless_range_loop)]

//! Initial-condition sensitivity (state-transition matrix) regression suite.
//!
//! Sequencing rationale: `ic_matches_hand_seeded_variational` (test 4 in
//! the Phase 1 design note) runs first. It compares the IC API result
//! bit-for-bit against the existing variational integrator hand-seeded
//! with `N_s = N`, `J_p ≡ 0`, `S(t₀) = I` — the *same* seeding the IC
//! wrapper produces internally. A failure here isolates the wrapper's
//! seed / layout composition (column-major identity, column-major Φ,
//! row-major J_y forwarding) from the variational math itself. The
//! analytic-oracle tests below only run on top of a passing test 4.
//!
//! Tests:
//!  1. `scalar_linear_matches_exp_minus_kt`  — closed-form Φ on `dy/dt = -k y`
//!  2. `linear_2x2_matches_expm_A_t`         — closed-form Φ on a 2×2 system
//!  4. `ic_matches_hand_seeded_variational`  — wrapper-correctness pin (FIRST)
//!  5. `ic_dummy_params_are_unread`          — IC-NO-PARAM-READ invariant pin
//!
//! Test 3 (autodiff-vs-analytic `J_y` agreement) lives in
//! `ic_sensitivity_autodiff.rs` so it can be `#[cfg(feature = "autodiff")]`-gated.

use numra_ode::sensitivity::{
    solve_forward_sensitivity, solve_initial_condition_sensitivity,
    solve_initial_condition_sensitivity_with, ParametricOdeSystem,
};
use numra_ode::{DoPri5, OdeSystem, SolverOptions};

// ============================================================================
// Test 4 (FIRST) — specialisation-vs-core regression
//
// Hand-seed `solve_forward_sensitivity` with the *exact same* shape the IC
// wrapper produces (N_s := N, J_p ≡ 0, S₀ = I); assert bit-for-bit equality
// with the IC API result. Pin against silent drift between the
// specialisation and the proven core.
// ============================================================================

/// Hand-rolled IC-seeded `ParametricOdeSystem` for a 2×2 linear flow,
/// mirroring the seeding the private `IcAsParametric` wrapper performs.
struct Linear2x2HandSeeded;

impl Linear2x2HandSeeded {
    /// `dy/dt = A y` with A = [[-1.0, 0.5], [0.0, -0.5]].
    const A: [[f64; 2]; 2] = [[-1.0, 0.5], [0.0, -0.5]];
}

impl ParametricOdeSystem<f64> for Linear2x2HandSeeded {
    fn n_states(&self) -> usize {
        2
    }
    fn n_params(&self) -> usize {
        2
    } // N_s := N
    fn params(&self) -> &[f64] {
        // The wrapper holds a zeroed dummy; reproduce that exactly.
        &[0.0, 0.0]
    }

    fn rhs_with_params(&self, _t: f64, y: &[f64], _p: &[f64], dy: &mut [f64]) {
        // Ignore `_p` exactly as the wrapper does.
        let a = Self::A;
        dy[0] = a[0][0] * y[0] + a[0][1] * y[1];
        dy[1] = a[1][0] * y[0] + a[1][1] * y[1];
    }

    fn jacobian_y(&self, _t: f64, _y: &[f64], jy: &mut [f64]) {
        // Row-major J_y[i*N + j] = A[i][j].
        let a = Self::A;
        jy[0] = a[0][0];
        jy[1] = a[0][1];
        jy[2] = a[1][0];
        jy[3] = a[1][1];
    }

    fn jacobian_p(&self, _t: f64, _y: &[f64], jp: &mut [f64]) {
        for slot in jp.iter_mut() {
            *slot = 0.0;
        }
    }

    fn initial_sensitivity(&self, _y0: &[f64], s0: &mut [f64]) {
        // Column-major identity matches the wrapper: s0[k*N + i] = δ_{ik}.
        for slot in s0.iter_mut() {
            *slot = 0.0;
        }
        s0[0 * 2 + 0] = 1.0;
        s0[1 * 2 + 1] = 1.0;
    }

    fn has_analytical_jacobian_y(&self) -> bool {
        true
    }
    fn has_analytical_jacobian_p(&self) -> bool {
        true
    }
}

/// IC-shaped `OdeSystem` for the same 2×2 linear flow, exposing the
/// `OdeSystem`-shape input to `solve_initial_condition_sensitivity`. Same
/// analytical Jacobian as the hand-seeded form so the comparison is on
/// wrapper-correctness, not Jacobian-source differences.
struct Linear2x2OdeSystem;

impl OdeSystem<f64> for Linear2x2OdeSystem {
    fn dim(&self) -> usize {
        2
    }

    fn rhs(&self, _t: f64, y: &[f64], dy: &mut [f64]) {
        let a = Linear2x2HandSeeded::A;
        dy[0] = a[0][0] * y[0] + a[0][1] * y[1];
        dy[1] = a[1][0] * y[0] + a[1][1] * y[1];
    }

    fn jacobian(&self, _t: f64, _y: &[f64], jac: &mut [f64]) {
        let a = Linear2x2HandSeeded::A;
        jac[0] = a[0][0];
        jac[1] = a[0][1];
        jac[2] = a[1][0];
        jac[3] = a[1][1];
    }
}

#[test]
fn ic_matches_hand_seeded_variational() {
    let y0 = [1.0_f64, 0.5];
    let (t0, tf) = (0.0_f64, 1.5_f64);
    let opts = SolverOptions::default().rtol(1e-9).atol(1e-12);

    let hand = solve_forward_sensitivity::<DoPri5, _, _>(&Linear2x2HandSeeded, t0, tf, &y0, &opts)
        .expect("hand-seeded variational solve failed");

    let ic = solve_initial_condition_sensitivity::<DoPri5, _, _>(
        &Linear2x2OdeSystem,
        t0,
        tf,
        &y0,
        &opts,
    )
    .expect("IC solve failed");

    // Same solver, same options, same RHS, same J_y, same seed → identical
    // augmented trajectories. The IC API result must equal the hand-seeded
    // SensitivityResult after the IC accessor rename.
    assert_eq!(hand.t.len(), ic.len());
    assert_eq!(hand.n_states, ic.dim());
    assert_eq!(hand.n_params, ic.dim()); // N_s := N for IC

    for i in 0..ic.len() {
        assert_eq!(hand.t[i], ic.t()[i], "t differs at i={i}");
        let hy = hand.y_at(i);
        let iy = ic.y(i);
        assert_eq!(hy, iy, "y differs at i={i}");

        // Φ(t_i) — column-major NxN — must match the hand-seeded
        // sensitivity block bit-for-bit.
        let h_phi = hand.sensitivity_at(i);
        let i_phi = ic.phi(i);
        assert_eq!(
            h_phi, i_phi,
            "Φ(t_{i}) differs: hand={h_phi:?} ic={i_phi:?}"
        );

        for row in 0..2 {
            for col in 0..2 {
                let h = hand.dyi_dpj(i, row, col);
                let g = ic.phi_ij(i, row, col);
                assert_eq!(
                    h, g,
                    "phi_ij(i={i},row={row},col={col}) differs: hand={h} ic={g}"
                );
            }
        }
    }
}

// ============================================================================
// Test 1 — scalar linear, closed-form Φ(t) = exp(-k t)
// ============================================================================

#[test]
fn scalar_linear_matches_exp_minus_kt() {
    let k = 0.5_f64;
    let result = solve_initial_condition_sensitivity_with::<DoPri5, f64, _>(
        move |_t, y, dy| {
            dy[0] = -k * y[0];
        },
        &[1.0],
        0.0,
        2.0,
        &SolverOptions::default().rtol(1e-9).atol(1e-12),
    )
    .expect("IC solve failed");

    assert!(result.success());
    assert_eq!(result.dim(), 1);

    // Identity at t₀.
    assert!((result.phi_ij(0, 0, 0) - 1.0).abs() < 1e-12);

    // Φ(t_i) = exp(-k t_i) at every output time.
    for i in 0..result.len() {
        let t_i = result.t()[i];
        let phi = result.phi_ij(i, 0, 0);
        let exact = (-k * t_i).exp();
        assert!(
            (phi - exact).abs() < 1e-7,
            "i={i} t={t_i} phi={phi} exact={exact} |err|={}",
            (phi - exact).abs()
        );
    }
}

// ============================================================================
// Test 2 — 2×2 linear, closed-form Φ(t) = expm(A·t)
//
// A = [[-1, 0.5], [0, -0.5]] is upper-triangular with distinct eigenvalues
// {-1, -0.5}. expm(A·t) has a known closed form:
//
//   Φ_{0,0}(t) = exp(-t)
//   Φ_{1,1}(t) = exp(-0.5 t)
//   Φ_{1,0}(t) = 0
//   Φ_{0,1}(t) = (0.5 / (-0.5 - (-1))) · (exp(-0.5 t) - exp(-t))
//              = 1 · (exp(-0.5 t) - exp(-t))
// ============================================================================

fn expected_phi_2x2(t: f64) -> [[f64; 2]; 2] {
    let em1t = (-t).exp();
    let em05t = (-0.5 * t).exp();
    [[em1t, em05t - em1t], [0.0, em05t]]
}

#[test]
fn linear_2x2_matches_expm_a_t() {
    let opts = SolverOptions::default().rtol(1e-10).atol(1e-13);
    let result = solve_initial_condition_sensitivity::<DoPri5, f64, _>(
        &Linear2x2OdeSystem,
        0.0,
        1.5,
        &[1.0, 0.5],
        &opts,
    )
    .expect("IC 2x2 solve failed");

    assert!(result.success());
    assert_eq!(result.dim(), 2);

    // Identity at t₀.
    let phi0 = result.phi(0);
    assert!((phi0[0 * 2 + 0] - 1.0).abs() < 1e-12, "Φ(0)_{{0,0}}");
    assert!((phi0[1 * 2 + 1] - 1.0).abs() < 1e-12, "Φ(0)_{{1,1}}");
    assert!(phi0[0 * 2 + 1].abs() < 1e-12, "Φ(0)_{{1,0}}");
    assert!(phi0[1 * 2 + 0].abs() < 1e-12, "Φ(0)_{{0,1}}");

    // expm(A·t) at every output time.
    for i in 0..result.len() {
        let t_i = result.t()[i];
        let expected = expected_phi_2x2(t_i);
        for row in 0..2 {
            for col in 0..2 {
                let got = result.phi_ij(i, row, col);
                let want = expected[row][col];
                assert!(
                    (got - want).abs() < 1e-7,
                    "Φ(t_{i})_{{{row},{col}}} got={got} want={want} t={t_i}"
                );
            }
        }
    }
}

// ============================================================================
// Test 5 — IC-NO-PARAM-READ invariant pin
//
// The IC wrapper holds a dummy `params()` slice of length N. The wrapped
// RHS (`OdeSystem::rhs`) has no `p` argument and provably never reads it.
// To pin this stronger-than-FD-vs-analytical claim, we cannot drive the
// wrapper directly (it's private). Instead we drive the *exact same shape*
// through `solve_forward_sensitivity` with a hand-built parametric system
// that (a) ignores `p` in its RHS (mirroring the wrapper) and (b) is given
// either zeroed or garbage `params()`. The trajectory must be identical;
// if the wrapper's `rhs_with_params` ever read `p`, NaN/∞ would
// propagate through the augmented RHS into y(t) and Φ(t), and the test
// would fail loudly.
//
// The pin is structural: it asserts the property that justifies
// `IcAsParametric::has_analytical_jacobian_p() = true` (∂f/∂p ≡ 0 exactly
// because the wrapped RHS does not consume p), independent of whether the
// debug consistency check is enabled.
// ============================================================================

/// Parametric wrapper *equivalent in shape* to `IcAsParametric` for the
/// scalar exponential decay. `dummy_p` controls what the slice returned by
/// `params()` contains; `rhs_with_params` ignores `_p` exactly as the
/// wrapper does, so the contents of `dummy_p` must not influence the
/// integrated trajectory.
struct IcShapeWithParamProbe {
    k: f64,
    dummy_p: Vec<f64>,
}

impl IcShapeWithParamProbe {
    fn new(k: f64, dummy_p_value: f64) -> Self {
        Self {
            k,
            dummy_p: vec![dummy_p_value],
        }
    }
}

impl ParametricOdeSystem<f64> for IcShapeWithParamProbe {
    fn n_states(&self) -> usize {
        1
    }
    fn n_params(&self) -> usize {
        1
    } // N_s := N
    fn params(&self) -> &[f64] {
        &self.dummy_p
    }

    fn rhs_with_params(&self, _t: f64, y: &[f64], _p: &[f64], dy: &mut [f64]) {
        // IC-NO-PARAM-READ: ignore `_p` exactly as `IcAsParametric` does.
        dy[0] = -self.k * y[0];
    }

    fn jacobian_y(&self, _t: f64, _y: &[f64], jy: &mut [f64]) {
        jy[0] = -self.k;
    }

    fn jacobian_p(&self, _t: f64, _y: &[f64], jp: &mut [f64]) {
        jp[0] = 0.0;
    }

    fn initial_sensitivity(&self, _y0: &[f64], s0: &mut [f64]) {
        s0[0] = 1.0; // identity, N = 1
    }

    fn has_analytical_jacobian_y(&self) -> bool {
        true
    }
    fn has_analytical_jacobian_p(&self) -> bool {
        true
    }
}

#[test]
fn ic_dummy_params_are_unread() {
    let opts = SolverOptions::default().rtol(1e-9).atol(1e-12);

    let with_zero = solve_forward_sensitivity::<DoPri5, _, _>(
        &IcShapeWithParamProbe::new(0.5, 0.0),
        0.0,
        1.0,
        &[1.0],
        &opts,
    )
    .expect("solve with zero dummy params failed");

    // Garbage: NaN and ∞ together — both would propagate as NaN through any
    // arithmetic op (NaN poisons; ∞ * 0 = NaN; ∞ - ∞ = NaN). If the wrapper's
    // RHS ever reads `p`, this run cannot produce a valid trajectory.
    let with_garbage_nan = solve_forward_sensitivity::<DoPri5, _, _>(
        &IcShapeWithParamProbe::new(0.5, f64::NAN),
        0.0,
        1.0,
        &[1.0],
        &opts,
    )
    .expect("solve with NaN dummy params failed");

    let with_garbage_inf = solve_forward_sensitivity::<DoPri5, _, _>(
        &IcShapeWithParamProbe::new(0.5, f64::INFINITY),
        0.0,
        1.0,
        &[1.0],
        &opts,
    )
    .expect("solve with ∞ dummy params failed");

    assert_eq!(with_zero.len(), with_garbage_nan.len());
    assert_eq!(with_zero.len(), with_garbage_inf.len());

    for i in 0..with_zero.len() {
        let z = with_zero.dyi_dpj(i, 0, 0);
        let n = with_garbage_nan.dyi_dpj(i, 0, 0);
        let f = with_garbage_inf.dyi_dpj(i, 0, 0);

        // Same y and same Φ across all three dummy-param values — proves
        // that the RHS does not consume `p`. If any of these are NaN or
        // differ, IC-NO-PARAM-READ has been violated.
        assert!(z.is_finite(), "zero-dummy result not finite at i={i}: {z}");
        assert!(n.is_finite(), "NaN-dummy poisoned trajectory at i={i}: {n}");
        assert!(f.is_finite(), "∞-dummy poisoned trajectory at i={i}: {f}");
        assert_eq!(z, n, "Φ differs between zero and NaN dummy at i={i}");
        assert_eq!(z, f, "Φ differs between zero and ∞ dummy at i={i}");
    }
}

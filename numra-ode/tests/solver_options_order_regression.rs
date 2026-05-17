//! Regression test pinning F-SOLVER-FIELDS.
//!
//! Before the F-SOLVER-FIELDS fix, `Bdf::with_max_order` / `Bdf::fixed_order`
//! were silently non-functional: they constructed a `Bdf` value whose
//! `max_order` / `min_order` fields were unreadable from the static
//! `Solver<S>::solve` trait method, which discarded `self` and called
//! `Bdf::new()` internally. Order control was effectively dead code.
//!
//! The fix wires order control through `SolverOptions::max_order(n)` /
//! `SolverOptions::min_order(n)`. This test pins that wiring: pinning BDF
//! to order 1 (Backward Euler) must observably take more accepted steps
//! than the unrestricted-order run (which adaptively reaches order ≥ 3
//! on this smooth problem). Without the wiring, the two `n_accept` counts
//! would be identical (since `max_order` would be silently ignored).
//!
//! Revert-confirm-restore protocol: temporarily reverting the wiring in
//! `bdf.rs::solve_internal` (replacing `resolve_order_bounds(options)` with
//! the old hard-coded `(1, MAX_ORDER)`) must make this test fail. Restoring
//! the wiring must make it pass.

use numra_ode::{Bdf, OdeProblem, Solver, SolverOptions};

#[test]
fn bdf_max_order_pinning_observable_via_step_count() {
    // Smooth, well-conditioned problem where BDF naturally climbs to
    // order ≥ 3 at the chosen tolerances. The chosen tf is long enough
    // that order selection has many steps to act on.
    let make_problem = || {
        OdeProblem::new(
            |_t, y: &[f64], dydt: &mut [f64]| {
                dydt[0] = -y[0];
            },
            0.0,
            10.0,
            vec![1.0],
        )
    };

    let opts_default: SolverOptions<f64> = SolverOptions::default().rtol(1e-8).atol(1e-11);
    let opts_pinned_bdf1 = opts_default.clone().max_order(1).min_order(1);

    let r_default = Bdf::solve(&make_problem(), 0.0, 10.0, &[1.0], &opts_default).unwrap();
    let r_pinned = Bdf::solve(&make_problem(), 0.0, 10.0, &[1.0], &opts_pinned_bdf1).unwrap();

    // Both runs must succeed and converge to the analytical answer.
    let expected = (-10.0_f64).exp();
    assert!(r_default.success);
    assert!(r_pinned.success);
    assert!(
        (r_default.y_final().unwrap()[0] - expected).abs() < 1e-5,
        "default-order BDF must remain accurate"
    );
    assert!(
        (r_pinned.y_final().unwrap()[0] - expected).abs() < 1e-3,
        "BDF1 must remain accurate at this tolerance (looser bound: BDF1 \
         is O(h) so the same rtol budget supports a larger absolute error)"
    );

    // The core pin: BDF1 takes substantially more accepted steps than the
    // unrestricted run. Concretely, BDF1 on a smooth problem at rtol=1e-8
    // needs O(thousands) of steps; the unrestricted run reaches BDF4-5 and
    // finishes in ~tens of steps. A factor-of-5 ratio is far below the
    // expected ratio (which is order 100x) but well above any noise floor.
    assert!(
        r_pinned.stats.n_accept > 5 * r_default.stats.n_accept,
        "BDF1 should take ≫5x more steps than unrestricted BDF on smooth \
         exponential decay; got pinned={}, default={}. If these are similar, \
         the SolverOptions::max_order/min_order wiring through Bdf::solve is \
         not taking effect (F-SOLVER-FIELDS regression).",
        r_pinned.stats.n_accept,
        r_default.stats.n_accept,
    );
}

#[test]
fn bdf_max_order_cap_clamps_to_algorithmic_range() {
    // Values outside [1, MAX_ORDER=5] must be clamped silently. Passing
    // max_order(99) should behave like the natural cap (5). Passing
    // max_order(0) should behave like max_order(1).
    let make_problem = || {
        OdeProblem::new(
            |_t, y: &[f64], dydt: &mut [f64]| {
                dydt[0] = -y[0];
            },
            0.0,
            5.0,
            vec![1.0],
        )
    };

    let opts_high = SolverOptions::<f64>::default()
        .rtol(1e-6)
        .atol(1e-9)
        .max_order(99);
    let opts_default: SolverOptions<f64> = SolverOptions::default().rtol(1e-6).atol(1e-9);

    let r_high = Bdf::solve(&make_problem(), 0.0, 5.0, &[1.0], &opts_high).unwrap();
    let r_default = Bdf::solve(&make_problem(), 0.0, 5.0, &[1.0], &opts_default).unwrap();

    // max_order(99) clamped to 5 ≡ the natural default cap → same n_accept.
    assert_eq!(
        r_high.stats.n_accept, r_default.stats.n_accept,
        "max_order(99) must clamp to the natural cap (5) and behave \
         identically to default options"
    );

    // max_order(0) clamps up to 1, equivalent to BDF1 pinning.
    let opts_zero = SolverOptions::<f64>::default()
        .rtol(1e-6)
        .atol(1e-9)
        .max_order(0);
    let opts_bdf1 = SolverOptions::<f64>::default()
        .rtol(1e-6)
        .atol(1e-9)
        .max_order(1)
        .min_order(1);

    let r_zero = Bdf::solve(&make_problem(), 0.0, 5.0, &[1.0], &opts_zero).unwrap();
    let r_bdf1 = Bdf::solve(&make_problem(), 0.0, 5.0, &[1.0], &opts_bdf1).unwrap();
    assert_eq!(
        r_zero.stats.n_accept, r_bdf1.stats.n_accept,
        "max_order(0) must clamp up to 1 and behave identically to \
         max_order(1).min_order(1)"
    );
}

//! Regression suite for `SolverOptions::t_eval`.
//!
//! GitHub issue #1: `SolverOptions::t_eval` was documented and exposed but
//! ignored by every solver — `solve` returned the natural step grid no matter
//! what the caller requested. This file pins the contract:
//!
//! - The returned `t` vector must be exactly `t_eval` (same length, same
//!   values), not the solver's natural step grid.
//! - Interpolated `y` values must match the analytical solution to within the
//!   solver's working tolerance.
//! - `solve_forward_sensitivity` inherits the fix because its augmented
//!   solver call goes through the same `Solver::solve` pathway.
//!
//! The 1-D exponential decay `dy/dt = -k y, y(0)=1` is the workhorse here:
//! the analytical solution `y(t) = exp(-k t)` is smooth enough that any
//! reasonable interpolant (Hermite cubic in our case) is well within solver
//! tolerance.

use numra_ode::sensitivity::{solve_forward_sensitivity, ParametricOdeSystem};
use numra_ode::{
    Bdf, DoPri5, Esdirk32, Esdirk43, Esdirk54, OdeProblem, Radau5, Solver, SolverOptions, Tsit5,
    Vern6, Vern7, Vern8,
};

fn decay_problem() -> OdeProblem<f64, impl Fn(f64, &[f64], &mut [f64]) + Clone> {
    OdeProblem::new(
        |_t, y: &[f64], dy: &mut [f64]| {
            dy[0] = -0.5 * y[0];
        },
        0.0,
        4.0,
        vec![1.0],
    )
}

fn assert_grid_matches(result_t: &[f64], requested: &[f64]) {
    assert_eq!(
        result_t.len(),
        requested.len(),
        "requested {} points, solver returned {}",
        requested.len(),
        result_t.len()
    );
    for (got, want) in result_t.iter().zip(requested.iter()) {
        assert!(
            (got - want).abs() < 1e-12,
            "grid mismatch: got {got}, want {want}"
        );
    }
}

fn assert_decay_values(result_t: &[f64], result_y: &[f64], k: f64, tol: f64) {
    for (i, &t) in result_t.iter().enumerate() {
        let want = (-k * t).exp();
        let got = result_y[i];
        assert!(
            (got - want).abs() < tol,
            "y({t}) = {got}, expected {want} (tol={tol})"
        );
    }
}

fn opts_with_grid(grid: Vec<f64>, rtol: f64, atol: f64) -> SolverOptions<f64> {
    let mut o = SolverOptions::default().rtol(rtol).atol(atol);
    o.t_eval = Some(grid);
    o
}

// ---------------------------------------------------------------------------
// One canonical case per solver to confirm the wiring is in place everywhere.
// ---------------------------------------------------------------------------

#[test]
fn dopri5_returns_requested_grid() {
    let problem = decay_problem();
    let requested = vec![0.0, 1.0, 2.0, 3.0, 4.0];
    let opts = opts_with_grid(requested.clone(), 1e-10, 1e-12);

    let r = DoPri5::solve(&problem, 0.0, 4.0, &[1.0], &opts).unwrap();
    assert_grid_matches(&r.t, &requested);
    assert_decay_values(&r.t, &r.y, 0.5, 1e-7);
}

#[test]
fn tsit5_returns_requested_grid() {
    let problem = decay_problem();
    let requested = vec![0.0, 0.7, 1.4, 2.1, 2.8, 3.5, 4.0];
    let opts = opts_with_grid(requested.clone(), 1e-10, 1e-12);

    let r = Tsit5::solve(&problem, 0.0, 4.0, &[1.0], &opts).unwrap();
    assert_grid_matches(&r.t, &requested);
    assert_decay_values(&r.t, &r.y, 0.5, 1e-7);
}

#[test]
fn vern6_returns_requested_grid() {
    let problem = decay_problem();
    let requested = vec![0.5, 1.5, 2.5, 3.5];
    let opts = opts_with_grid(requested.clone(), 1e-10, 1e-12);

    let r = Vern6::solve(&problem, 0.0, 4.0, &[1.0], &opts).unwrap();
    assert_grid_matches(&r.t, &requested);
    assert_decay_values(&r.t, &r.y, 0.5, 1e-7);
}

#[test]
fn vern7_returns_requested_grid() {
    let problem = decay_problem();
    let requested = vec![0.0, 2.0, 4.0];
    let opts = opts_with_grid(requested.clone(), 1e-10, 1e-12);

    let r = Vern7::solve(&problem, 0.0, 4.0, &[1.0], &opts).unwrap();
    assert_grid_matches(&r.t, &requested);
    // Vern7 is 7th-order, so the controller picks large steps; the requested
    // points sit at step boundaries here, so Hermite reproduces them exactly.
    assert_decay_values(&r.t, &r.y, 0.5, 1e-7);
}

#[test]
fn vern8_returns_requested_grid() {
    let problem = decay_problem();
    let requested = vec![0.25, 1.25, 2.25, 3.25];
    let opts = opts_with_grid(requested.clone(), 1e-10, 1e-12);

    let r = Vern8::solve(&problem, 0.0, 4.0, &[1.0], &opts).unwrap();
    assert_grid_matches(&r.t, &requested);
    // Vern8 takes very large steps (~1.0) at these tolerances, and the
    // requested points sit mid-step, so Hermite cubic O(h^4) interpolation
    // dominates the error budget. 1e-4 reflects that floor.
    assert_decay_values(&r.t, &r.y, 0.5, 1e-4);
}

#[test]
fn radau5_returns_requested_grid() {
    let problem = decay_problem();
    let requested = vec![0.0, 1.0, 2.0, 3.0, 4.0];
    let opts = opts_with_grid(requested.clone(), 1e-8, 1e-10);

    let r = Radau5::solve(&problem, 0.0, 4.0, &[1.0], &opts).unwrap();
    assert_grid_matches(&r.t, &requested);
    assert_decay_values(&r.t, &r.y, 0.5, 1e-5);
}

#[test]
fn bdf_returns_requested_grid() {
    let problem = decay_problem();
    let requested = vec![0.0, 1.0, 2.0, 3.0, 4.0];
    let opts = opts_with_grid(requested.clone(), 1e-8, 1e-10);

    let r = Bdf::solve(&problem, 0.0, 4.0, &[1.0], &opts).unwrap();
    assert_grid_matches(&r.t, &requested);
    assert_decay_values(&r.t, &r.y, 0.5, 1e-5);
}

#[test]
fn esdirk32_returns_requested_grid() {
    let problem = decay_problem();
    let requested = vec![0.0, 1.0, 2.0, 3.0, 4.0];
    let opts = opts_with_grid(requested.clone(), 1e-8, 1e-10);

    let r = Esdirk32::solve(&problem, 0.0, 4.0, &[1.0], &opts).unwrap();
    assert_grid_matches(&r.t, &requested);
    assert_decay_values(&r.t, &r.y, 0.5, 1e-4);
}

#[test]
fn esdirk43_returns_requested_grid() {
    let problem = decay_problem();
    let requested = vec![0.0, 1.0, 2.0, 3.0, 4.0];
    let opts = opts_with_grid(requested.clone(), 1e-8, 1e-10);

    let r = Esdirk43::solve(&problem, 0.0, 4.0, &[1.0], &opts).unwrap();
    assert_grid_matches(&r.t, &requested);
    assert_decay_values(&r.t, &r.y, 0.5, 1e-5);
}

#[test]
fn esdirk54_returns_requested_grid() {
    let problem = decay_problem();
    let requested = vec![0.0, 1.0, 2.0, 3.0, 4.0];
    let opts = opts_with_grid(requested.clone(), 1e-8, 1e-10);

    let r = Esdirk54::solve(&problem, 0.0, 4.0, &[1.0], &opts).unwrap();
    assert_grid_matches(&r.t, &requested);
    assert_decay_values(&r.t, &r.y, 0.5, 1e-5);
}

// ---------------------------------------------------------------------------
// Edge cases (exercised on DoPri5; same code path applies to all solvers)
// ---------------------------------------------------------------------------

#[test]
fn endpoints_are_bit_exact() {
    // Hermite cubic with theta=0 reproduces y_old, theta=1 reproduces y_new,
    // so the start-of-interval entry must come back bit-exact equal to y0.
    let problem = decay_problem();
    let requested = vec![0.0, 4.0];
    let opts = opts_with_grid(requested.clone(), 1e-10, 1e-12);

    let r = DoPri5::solve(&problem, 0.0, 4.0, &[1.0], &opts).unwrap();
    assert_eq!(r.y[0], 1.0, "y(t0) must equal y0 exactly");
    let want = (-0.5_f64 * 4.0).exp();
    assert!((r.y[1] - want).abs() < 1e-7);
}

#[test]
fn dense_grid_within_steps() {
    // 41 points covering the same interval. Catches any "off by one step"
    // bug where the emitter advances incorrectly across step boundaries.
    let problem = decay_problem();
    let requested: Vec<f64> = (0..=40).map(|i| (i as f64) * 0.1).collect();
    let opts = opts_with_grid(requested.clone(), 1e-10, 1e-12);

    let r = DoPri5::solve(&problem, 0.0, 4.0, &[1.0], &opts).unwrap();
    assert_grid_matches(&r.t, &requested);
    assert_decay_values(&r.t, &r.y, 0.5, 1e-7);
}

#[test]
fn backward_integration() {
    // dy/dt = -y solved backward from t=4 to t=0. Grid is sorted in the
    // integration direction (descending here).
    let problem = OdeProblem::new(
        |_t, y: &[f64], dy: &mut [f64]| {
            dy[0] = -y[0];
        },
        4.0,
        0.0,
        vec![(-4.0_f64).exp()],
    );
    let requested = vec![4.0, 3.0, 2.0, 1.0, 0.0];
    let opts = opts_with_grid(requested.clone(), 1e-10, 1e-12);

    let y0 = (-4.0_f64).exp();
    let r = DoPri5::solve(&problem, 4.0, 0.0, &[y0], &opts).unwrap();
    assert_grid_matches(&r.t, &requested);
    for (i, &t) in requested.iter().enumerate() {
        let want = (-t).exp();
        assert!(
            (r.y[i] - want).abs() < 1e-7,
            "backward y({t}) = {}, expected {want}",
            r.y[i]
        );
    }
}

#[test]
fn none_returns_natural_step_grid() {
    // Sanity: when the option is None (default), the solver still returns
    // its natural step grid as before. This is the legacy behavior.
    let problem = decay_problem();
    let opts = SolverOptions::default().rtol(1e-10).atol(1e-12);

    let r = DoPri5::solve(&problem, 0.0, 4.0, &[1.0], &opts).unwrap();
    assert!(r.t.len() > 5, "natural grid should produce many points");
    assert!((r.t[0] - 0.0).abs() < 1e-12);
    assert!((r.t[r.t.len() - 1] - 4.0).abs() < 1e-12);
}

// ---------------------------------------------------------------------------
// The exact reproducer from the issue: forward sensitivity should also
// honor the requested grid, since it goes through Solver::solve internally.
// ---------------------------------------------------------------------------

#[test]
fn solve_forward_sensitivity_returns_requested_grid() {
    struct Decay {
        k: f64,
    }
    impl ParametricOdeSystem<f64> for Decay {
        fn n_states(&self) -> usize {
            1
        }
        fn n_params(&self) -> usize {
            1
        }
        fn params(&self) -> &[f64] {
            std::slice::from_ref(&self.k)
        }
        fn rhs_with_params(&self, _t: f64, y: &[f64], p: &[f64], dy: &mut [f64]) {
            dy[0] = -p[0] * y[0];
        }
        fn jacobian_y(&self, _t: f64, _y: &[f64], jy: &mut [f64]) {
            jy[0] = -self.k;
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
    }

    let requested = vec![0.0, 1.0, 2.0, 3.0, 4.0];
    let opts = opts_with_grid(requested.clone(), 1e-10, 1e-12);

    let r = solve_forward_sensitivity::<DoPri5, f64, _>(&Decay { k: 0.5 }, 0.0, 4.0, &[1.0], &opts)
        .unwrap();

    assert_eq!(
        r.len(),
        requested.len(),
        "solve_forward_sensitivity should return {} points (got {})",
        requested.len(),
        r.len()
    );
    for (i, &t) in requested.iter().enumerate() {
        let want_y = (-0.5_f64 * t).exp();
        let want_dy_dk = -t * (-0.5_f64 * t).exp();
        assert!(
            (r.t[i] - t).abs() < 1e-12,
            "t mismatch at index {i}: got {}, want {t}",
            r.t[i]
        );
        assert!(
            (r.y_at(i)[0] - want_y).abs() < 1e-7,
            "y at t={t}: got {}, want {want_y}",
            r.y_at(i)[0]
        );
        let s_jk = r.dyi_dpj(i, 0, 0);
        assert!(
            (s_jk - want_dy_dk).abs() < 1e-6,
            "ds/dk at t={t}: got {s_jk}, want {want_dy_dk}"
        );
    }
}

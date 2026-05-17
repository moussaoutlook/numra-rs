//! Autodiff-vs-analytic `J_y` agreement on a nonlinear system.
//!
//! Test 3 of the IC sensitivity Phase-1 plan: the AD path is *correct*,
//! not merely present. Runs `solve_initial_condition_sensitivity` twice on
//! Lotka-Volterra, once with an analytical `OdeSystem::jacobian` override
//! and once with `AutodiffJacobianSystem` wrapping the same RHS typed
//! against `Dual<f64>`. The two state-transition trajectories must agree.
//!
//! Behind the `autodiff` cargo feature so callers who do not opt in do
//! not compile or pull `numra-autodiff`.

#![cfg(feature = "autodiff")]

use numra_autodiff::Dual;
use numra_ode::sensitivity::solve_initial_condition_sensitivity;
use numra_ode::{AutodiffJacobianSystem, DoPri5, OdeSystem, SolverOptions};

// --- Lotka-Volterra parameters (a, b, c, d) = (1.5, 1.0, 3.0, 1.0) ---

const A: f64 = 1.5;
const B: f64 = 1.0;
const C: f64 = 3.0;
const D: f64 = 1.0;

/// `OdeSystem` impl with a hand-written analytical Jacobian.
struct LvAnalyticalJ;

impl OdeSystem<f64> for LvAnalyticalJ {
    fn dim(&self) -> usize {
        2
    }

    fn rhs(&self, _t: f64, y: &[f64], dy: &mut [f64]) {
        dy[0] = A * y[0] - B * y[0] * y[1];
        dy[1] = C * y[0] * y[1] - D * y[1];
    }

    fn jacobian(&self, _t: f64, y: &[f64], jac: &mut [f64]) {
        // Row-major J[i*2 + j] = ∂f_i/∂y_j.
        jac[0] = A - B * y[1]; // ∂f0/∂y0
        jac[1] = -B * y[0]; // ∂f0/∂y1
        jac[2] = C * y[1]; // ∂f1/∂y0
        jac[3] = C * y[0] - D; // ∂f1/∂y1
    }
}

/// Same Lotka-Volterra RHS, typed against `Dual<f64>` for AD-derived `J_y`.
fn lv_dual(_t: f64, y: &[Dual<f64>], dy: &mut [Dual<f64>]) {
    let a = Dual::constant(A);
    let b = Dual::constant(B);
    let c = Dual::constant(C);
    let d = Dual::constant(D);
    dy[0] = a * y[0] - b * y[0] * y[1];
    dy[1] = c * y[0] * y[1] - d * y[1];
}

#[test]
fn autodiff_jy_matches_analytic_jy_on_lotka_volterra() {
    let y0 = [1.0_f64, 1.0];
    let (t0, tf) = (0.0_f64, 1.0_f64);
    // Tight tolerances so the integrator's local error doesn't dominate
    // the AD-vs-analytic difference we're measuring.
    let opts = SolverOptions::default().rtol(1e-10).atol(1e-13);

    let analytic = solve_initial_condition_sensitivity::<DoPri5, f64, _>(
        &LvAnalyticalJ,
        t0,
        tf,
        &y0,
        &opts,
    )
    .expect("analytic IC solve failed");

    let ad_sys = AutodiffJacobianSystem::<f64, _>::new(lv_dual, 2);
    let ad = solve_initial_condition_sensitivity::<DoPri5, f64, _>(
        &ad_sys, t0, tf, &y0, &opts,
    )
    .expect("autodiff IC solve failed");

    // Solver-adaptive output grids may differ in step count and exact
    // breakpoints between the two runs (the AD path's Jacobian is exact
    // to round-off and the analytic path's Jacobian is exact too, but
    // floating-point round-off in different evaluation orders can produce
    // bit-different step proposals). We compare at the final time, which
    // both solvers land on, with a tolerance well below the integrator's
    // truncation error.
    assert!(analytic.success());
    assert!(ad.success());

    let af = analytic.final_phi();
    let df = ad.final_phi();
    assert_eq!(af.len(), 4);
    assert_eq!(df.len(), 4);

    for k in 0..4 {
        let diff = (af[k] - df[k]).abs();
        assert!(
            diff < 1e-8,
            "Φ(t_f)[{k}] differs: analytic={} autodiff={} |Δ|={diff}",
            af[k],
            df[k],
        );
    }

    // Also check y(t_f): the primal trajectory must match within
    // integrator tolerance regardless of which Jacobian path drove the
    // augmented step controller.
    let ay = analytic.final_y();
    let dy = ad.final_y();
    for i in 0..2 {
        let diff = (ay[i] - dy[i]).abs();
        assert!(
            diff < 1e-9,
            "y(t_f)[{i}] differs: analytic={} autodiff={} |Δ|={diff}",
            ay[i],
            dy[i],
        );
    }
}

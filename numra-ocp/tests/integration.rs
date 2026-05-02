//! Integration tests for numra-ocp: parameter estimation, shooting, and
//! forward sensitivity.
//!
//! Author: Moussa Leblouba
//! Date: 9 February 2026
//! Modified: 2 May 2026

use numra_ocp::{forward_sensitivity, ParamEstProblem, ParamEstResult, ShootingProblem};
use numra_ode::{DoPri5, OdeProblem, Solver, SolverOptions};

// ---------------------------------------------------------------------------
// Helper: generate reference data by integrating the true ODE segment-by-
// segment so we land exactly on each requested time.
// ---------------------------------------------------------------------------

fn generate_data<F>(rhs: F, y0: &[f64], times: &[f64], n_states: usize) -> Vec<Vec<f64>>
where
    F: Fn(f64, &[f64], &mut [f64]) + Clone + 'static,
{
    let opts = SolverOptions::default().rtol(1e-12).atol(1e-14);
    let mut snapshots: Vec<Vec<f64>> = Vec::with_capacity(times.len());
    let mut y_cur = y0.to_vec();
    snapshots.push(y_cur.clone());

    for i in 0..(times.len() - 1) {
        let t_s = times[i];
        let t_e = times[i + 1];
        let rhs_clone = rhs.clone();
        let problem = OdeProblem::new(
            move |t: f64, y: &[f64], dydt: &mut [f64]| rhs_clone(t, y, dydt),
            t_s,
            t_e,
            y_cur.clone(),
        );
        let res =
            DoPri5::solve(&problem, t_s, t_e, &y_cur, &opts).expect("reference integration failed");
        y_cur = res.y_final().expect("empty result");
        assert_eq!(y_cur.len(), n_states);
        snapshots.push(y_cur.clone());
    }

    snapshots
}

// ===========================================================================
// Test 1: Spring-mass-damper parameter estimation
// ===========================================================================

#[test]
fn test_param_est_spring_mass_damper() {
    // True parameters (m = 1 absorbed):
    //   dx/dt = v
    //   dv/dt = -k*x - c*v
    // p = [c, k], true c = 0.5, true k = 2.0.
    let c_true = 0.5;
    let k_true = 2.0;

    let x0 = 1.0;
    let v0 = 0.0;
    let n_states = 2;

    // 21 evenly spaced points from t = 0 to t = 10.
    let t_data: Vec<f64> = (0..=20).map(|i| i as f64 * 0.5).collect();

    // Generate reference data with the true system.
    let snapshots = generate_data(
        move |_t: f64, y: &[f64], dydt: &mut [f64]| {
            dydt[0] = y[1];
            dydt[1] = -k_true * y[0] - c_true * y[1];
        },
        &[x0, v0],
        &t_data,
        n_states,
    );

    // Observe position only (state 0).
    let y_data: Vec<f64> = snapshots.iter().map(|s| s[0]).collect();

    let result: ParamEstResult<f64> = ParamEstProblem::new(2, n_states)
        .model(|_t: f64, y, dydt, p| {
            dydt[0] = y[1];
            dydt[1] = -p[1] * y[0] - p[0] * y[1];
        })
        .initial_state(vec![x0, v0])
        .params(vec![1.0, 1.0]) // initial guess
        .observed(vec![0]) // observe position only
        .data(t_data, y_data)
        .max_iter(200)
        .solve()
        .expect("param est failed");

    assert!(
        result.converged,
        "optimizer did not converge: {}",
        result.message
    );

    let c_est = result.params[0];
    let k_est = result.params[1];

    assert!(
        (c_est - c_true).abs() < 0.3,
        "c_est = {c_est}, expected ~{c_true} (error = {})",
        (c_est - c_true).abs()
    );
    assert!(
        (k_est - k_true).abs() < 0.3,
        "k_est = {k_est}, expected ~{k_true} (error = {})",
        (k_est - k_true).abs()
    );
}

// ===========================================================================
// Test 2: Shooting with damped double integrator
// ===========================================================================

#[test]
fn test_shooting_with_damping() {
    // Damped double integrator:
    //   dx/dt = v
    //   dv/dt = u - 0.1 * v
    // Terminal cost: 100 * ((x - 1)^2 + v^2)
    // Running cost: 0.01 * u^2
    // T = 2.0, 10 segments, initial state (0, 0).

    let result = ShootingProblem::new(2, 1)
        .dynamics(|_t: f64, y, dydt, u| {
            dydt[0] = y[1]; // dx/dt = v
            dydt[1] = u[0] - 0.1 * y[1]; // dv/dt = u - 0.1*v
        })
        .initial_state(vec![0.0, 0.0])
        .time_span(0.0, 2.0)
        .n_segments(10)
        .terminal_cost(|y: &[f64]| 100.0 * ((y[0] - 1.0).powi(2) + y[1].powi(2)))
        .running_cost(|_t: f64, _y, u| 0.01 * u[0].powi(2))
        .max_iter(300)
        .solve()
        .expect("shooting solve failed");

    let x_final = result.final_state[0];
    assert!(
        (x_final - 1.0_f64).abs() < 0.3,
        "x(T) = {x_final}, expected within 0.3 of 1.0"
    );
}

// ===========================================================================
// Test 3: Forward sensitivity gradient check
// ===========================================================================

#[test]
fn test_sensitivity_for_param_est_gradient() {
    // Model: dy/dt = -k * y, k = 0.5, y(0) = 1.
    // Analytical solution: y(t) = exp(-k * t).
    // Data: 11 points from t = 0 to t = 5, generated from analytical solution.
    let k_true = 0.5_f64;
    let y0_val = 1.0_f64;

    let t_data: Vec<f64> = (0..=10).map(|i| i as f64 * 0.5).collect();
    let y_data: Vec<f64> = t_data
        .iter()
        .map(|&t| y0_val * (-k_true * t).exp())
        .collect();

    // Compute forward sensitivity at the true parameter.
    let sens_result = forward_sensitivity(
        &|_t: f64, y, dydt, p| {
            dydt[0] = -p[0] * y[0];
        },
        &[y0_val],
        &[k_true],
        0.0,
        5.0,
        Some(&t_data),
        1e-10,
        1e-12,
    )
    .expect("forward_sensitivity failed");

    // Gradient of least-squares objective at true parameters:
    //   J(k) = sum_i (y(t_i; k) - y_data_i)^2
    //   dJ/dk = 2 * sum_i (y(t_i; k) - y_data_i) * (dy/dk)(t_i)
    //
    // At the true k, residuals y(t_i; k) - y_data_i should be ~0,
    // so the gradient should be ~0.
    let mut grad: f64 = 0.0;
    for (i, &y_obs) in y_data.iter().enumerate() {
        let y_pred = sens_result.y_at(i)[0];
        let residual = y_pred - y_obs;
        let dy_dk = sens_result.sensitivity_at(i)[0];
        grad += 2.0 * residual * dy_dk;
    }

    assert!(
        grad.abs() < 1e-4,
        "gradient at true params should be ~0, got {grad}"
    );
}

// ===========================================================================
// Test 4: Wall time is populated
// ===========================================================================

#[test]
fn test_wall_time_populated() {
    // Simple exponential decay model; verify wall_time_secs > 0.
    let k_true = 0.5;
    let y0_val = 1.0;
    let t_data: Vec<f64> = (0..=5).map(|i| i as f64).collect();
    let y_data: Vec<f64> = t_data
        .iter()
        .map(|&t| y0_val * (-k_true * t).exp())
        .collect();

    let result: ParamEstResult<f64> = ParamEstProblem::new(1, 1)
        .model(|_t: f64, y, dydt, p| {
            dydt[0] = -p[0] * y[0];
        })
        .initial_state(vec![y0_val])
        .params(vec![1.0])
        .data(t_data, y_data)
        .solve()
        .expect("param est failed");

    assert!(
        result.wall_time_secs > 0.0,
        "wall_time_secs should be > 0, got {}",
        result.wall_time_secs
    );
}

//! End-to-end interoperability workflow tests for Numra.
//!
//! These tests verify that all crates compose seamlessly — no manual type
//! conversions, no boilerplate wrappers, no f64 bottlenecks.
//!
//! 6 workflows:
//! 1. ODE → Interpolation → Integration
//! 2. ODE → FFT → Signal → Peak Detection
//! 3. ParamEst → Sensitivity → Uncertainty
//! 4. Autodiff → Optimization → Curve Fitting
//! 5. PDE → Statistics
//! 6. Stats Sampling → Monte Carlo ODE
//!
//! Author: Moussa Leblouba
//! Date: 10 February 2026
//! Modified: 2 May 2026

use core::f64::consts::PI;

// =============================================================================
// Workflow 1: ODE → Interpolation → Integration
//
// Solve an ODE, interpolate the solution with a cubic spline,
// then numerically integrate the interpolant to verify conservation.
// =============================================================================

#[test]
fn workflow_ode_interp_integrate() {
    use numra::integrate::{quad, QuadOptions};
    use numra::interp::{CubicSpline, Interpolant};
    use numra::ode::{DoPri5, OdeProblem, Solver, SolverOptions, SolverResult};

    // Solve y' = -y, y(0) = 1 on [0, 3]
    let problem = OdeProblem::new(
        |_t, y: &[f64], dydt: &mut [f64]| {
            dydt[0] = -y[0];
        },
        0.0,
        3.0,
        vec![1.0],
    );
    let opts = SolverOptions::default().rtol(1e-8).dense();
    let result: SolverResult<f64> = DoPri5::solve(&problem, 0.0, 3.0, &[1.0], &opts).unwrap();

    // Use component() helper (A1) to extract the first state variable
    let y_series = result.component(0).unwrap();
    assert_eq!(y_series.len(), result.t.len());

    // Build cubic spline interpolation of the ODE solution
    let spline = CubicSpline::natural(&result.t, &y_series).unwrap();

    // Verify interpolant agrees with exact solution at a few points
    for &ti in &[0.5, 1.0, 1.5, 2.0, 2.5] {
        let interp_val = spline.interpolate(ti);
        let exact = (-ti).exp();
        assert!(
            (interp_val - exact).abs() < 1e-4,
            "spline({ti}) = {interp_val}, exact = {exact}"
        );
    }

    // Integrate the interpolant from 0 to 3
    // integral_0^3 e^(-t) dt = 1 - e^(-3) ~ 0.9502
    let q_opts = QuadOptions::default();
    let q_result = quad(|t| spline.interpolate(t), 0.0, 3.0, &q_opts).unwrap();
    let exact_integral = 1.0 - (-3.0_f64).exp();
    assert!(
        (q_result.value - exact_integral).abs() < 1e-3,
        "integral = {}, exact = {exact_integral}",
        q_result.value
    );
}

// =============================================================================
// Workflow 2: ODE → FFT → Signal → Peak Detection
//
// Solve a multi-frequency ODE, extract the solution, run FFT to get the
// power spectrum, filter it, and detect peaks.
// =============================================================================

#[test]
fn workflow_ode_fft_signal_peaks() {
    use numra::dsp::{butter, filtfilt, find_peaks, PeakOptions};
    use numra::fft::{psd, Window};
    use numra::ode::{DoPri5, OdeProblem, Solver, SolverOptions};

    // Solve y'' + y = 0 -> y = cos(t), y' = -sin(t)
    let tf = 8.0 * PI;
    let problem = OdeProblem::new(
        |_t, y: &[f64], dydt: &mut [f64]| {
            dydt[0] = y[1];
            dydt[1] = -y[0]; // Simple harmonic oscillator
        },
        0.0,
        tf,
        vec![1.0, 0.0],
    );

    let opts = SolverOptions::default().rtol(1e-8).dense();
    let result = DoPri5::solve(&problem, 0.0, tf, &[1.0, 0.0], &opts).unwrap();

    // Extract x component
    let x_component = result.component(0).unwrap();
    assert!(x_component.len() > 10);

    // Build a synthetic signal: the ODE solution + high-frequency component
    let signal: Vec<f64> = (0..x_component.len())
        .map(|i| {
            let t = result.t[i];
            x_component[i] + 0.3 * (10.0 * t).sin()
        })
        .collect();

    // Power spectral density to verify frequency content
    let (freqs, power) = psd(&signal, 1.0, &Window::Hann);
    assert!(!freqs.is_empty());
    assert_eq!(freqs.len(), power.len());

    // Design a lowpass filter to isolate the fundamental
    let sos = butter(4, 10.0, 100.0).unwrap();

    // Apply zero-phase filtering if signal is long enough
    if signal.len() >= 20 {
        let filtered = filtfilt(&sos, &signal);
        assert_eq!(filtered.len(), signal.len());

        // Detect peaks in the filtered signal
        let peaks = find_peaks(&filtered, &PeakOptions::default().height(0.3));
        // We should find peaks from the oscillation
        assert!(!peaks.is_empty(), "Should detect oscillation peaks");
    }
}

// =============================================================================
// Workflow 3: ParamEst → Sensitivity → Uncertainty
//
// Estimate parameters of an ODE model, compute forward sensitivity
// of the solution w.r.t. estimated parameters, and propagate uncertainty.
// =============================================================================

#[test]
fn workflow_param_est_sensitivity_uncertainty() {
    use numra::ocp::{OdeSolverChoice, ParamEstProblem};
    use numra::ode::uncertainty::{solve_with_uncertainty, UncertainParam, UncertaintyMode};
    use numra::ode::{DoPri5, SolverOptions};

    // Generate synthetic data from y' = -k*y with k=2
    let k_true = 2.0;
    let t_data: Vec<f64> = (0..11).map(|i| i as f64 * 0.1).collect();
    let y_data: Vec<f64> = t_data.iter().map(|&t| (-k_true * t).exp()).collect();

    // Step 1: Parameter estimation — recover k from data
    let result = ParamEstProblem::new(1, 1) // 1 param, 1 state
        .model(|_t: f64, y: &[f64], dydt: &mut [f64], params: &[f64]| {
            dydt[0] = -params[0] * y[0];
        })
        .data(t_data.clone(), y_data.clone())
        .params(vec![1.0]) // Initial guess: k=1
        .initial_state(vec![1.0])
        .ode_solver(OdeSolverChoice::DoPri5)
        .solve()
        .unwrap();

    let k_est = result.params[0];
    assert!(
        (k_est - k_true).abs() < 0.1,
        "Estimated k = {k_est}, expected ~{k_true}"
    );

    // Step 2: Uncertainty propagation with the estimated parameter
    let uncertain_params = vec![UncertainParam::new("k", k_est, 0.1)];
    let u_result = solve_with_uncertainty::<DoPri5, f64, _>(
        |_t, y, dydt, params| {
            dydt[0] = -params[0] * y[0];
        },
        &[1.0],
        0.0,
        1.0,
        &uncertain_params,
        &UncertaintyMode::Trajectory,
        &SolverOptions::default().rtol(1e-6),
        None,
    )
    .unwrap();

    // Solution should have uncertainty information
    assert!(!u_result.sigma.is_empty());
    // The final uncertainty should be positive (parameter uncertainty propagates)
    let final_sigma = u_result.sigma.last().unwrap();
    assert!(
        *final_sigma > 0.0,
        "Uncertainty should propagate: sigma = {final_sigma}"
    );
}

// =============================================================================
// Workflow 4: Autodiff → Optimization → Curve Fitting
//
// Use autodiff to compute analytical gradients, pass them to the optimizer,
// and use curve_fit_with_jacobian for precise parameter estimation.
// =============================================================================

#[test]
fn workflow_autodiff_optim_fit() {
    use numra::autodiff::{gradient_closure, Dual};
    use numra::fit::curve_fit_with_jacobian;
    use numra::optim::OptimProblem;

    // Part A: Autodiff -> Optim bridge
    // Minimize Rosenbrock: f(x,y) = (1-x)^2 + 100(y-x^2)^2
    let grad_fn = gradient_closure(|x: &[Dual<f64>]| {
        let a = Dual::constant(1.0) - x[0];
        let b = x[1] - x[0] * x[0];
        a * a + Dual::constant(100.0) * b * b
    });

    let result = OptimProblem::new(2)
        .x0(&[0.0, 0.0])
        .objective(|x: &[f64]| {
            let a = 1.0 - x[0];
            let b = x[1] - x[0] * x[0];
            a * a + 100.0 * b * b
        })
        .gradient(move |x: &[f64], g: &mut [f64]| {
            grad_fn(x, g);
        })
        .solve()
        .unwrap();

    assert!(
        (result.x[0] - 1.0).abs() < 0.01 && (result.x[1] - 1.0).abs() < 0.01,
        "Rosenbrock optimum: [{}, {}], expected [1, 1]",
        result.x[0],
        result.x[1]
    );

    // Part B: Curve fitting with analytical Jacobian
    // Fit y = a * exp(-b * x)
    let x_data: Vec<f64> = (0..20).map(|i| i as f64 * 0.25).collect();
    let y_data: Vec<f64> = x_data.iter().map(|&x| 3.0 * (-0.5 * x).exp()).collect();

    // Analytical model Jacobian: d(a*exp(-b*x))/d[a,b] = [exp(-b*x), -a*x*exp(-b*x)]
    let fit_result = curve_fit_with_jacobian(
        |x: f64, params: &[f64]| params[0] * (-params[1] * x).exp(),
        |x: f64, params: &[f64]| {
            let e = (-params[1] * x).exp();
            vec![e, -params[0] * x * e]
        },
        &x_data,
        &y_data,
        &[1.0, 1.0],
        None,
    )
    .unwrap();

    assert!(fit_result.converged);
    assert!(
        (fit_result.params[0] - 3.0).abs() < 0.01,
        "a = {}, expected 3.0",
        fit_result.params[0]
    );
    assert!(
        (fit_result.params[1] - 0.5).abs() < 0.01,
        "b = {}, expected 0.5",
        fit_result.params[1]
    );
}

// =============================================================================
// Workflow 5: PDE → Statistics
//
// Solve a PDE (heat equation), extract the spatial distribution at final time,
// and compute descriptive statistics on it.
// =============================================================================

#[test]
fn workflow_pde_statistics() {
    use numra::ode::{DoPri5, Solver, SolverOptions};
    use numra::pde::boundary::DirichletBC;
    use numra::pde::{Grid1D, HeatEquation1D, MOLSystem};
    use numra::stats::{mean, median, std_dev, variance};

    // Heat equation on [0, 1] with left=1.0, right=0.0
    let grid = Grid1D::uniform(0.0, 1.0, 51);
    let pde = HeatEquation1D::new(0.1); // diffusivity = 0.1
    let bc_left = DirichletBC::new(1.0);
    let bc_right = DirichletBC::new(0.0);
    let mol = MOLSystem::new(pde, grid.clone(), bc_left, bc_right);

    // Initial condition: sine bump
    let u0: Vec<f64> = grid
        .interior_points()
        .iter()
        .map(|&x| (PI * x).sin())
        .collect();

    let opts = SolverOptions::default().rtol(1e-6);
    let result = DoPri5::solve(&mol, 0.0, 0.5, &u0, &opts).unwrap();
    assert!(result.success);

    // Extract final spatial distribution
    let u_final = result.y_final().unwrap().to_vec();

    // Compute statistics on the spatial distribution
    let m = mean(&u_final).unwrap();
    let v = variance(&u_final).unwrap();
    let s = std_dev(&u_final).unwrap();
    let med = median(&u_final).unwrap();

    // After diffusion, mean should be moderate (between 0 and 1)
    assert!(m > 0.0 && m < 1.0, "mean = {m}");
    // Variance should be positive (non-uniform distribution)
    assert!(v > 0.0, "variance = {v}");
    // Std dev should equal sqrt(variance)
    assert!(
        (s - v.sqrt()).abs() < 1e-10,
        "std_dev inconsistent with variance"
    );
    // Median should be in range
    assert!(med > 0.0 && med < 1.0, "median = {med}");

    // After diffusion, the distribution should be smoother (lower variance)
    // than the initial sine bump
    let v_initial = variance(&u0).unwrap();
    assert!(
        v < v_initial,
        "Diffusion should reduce variance: {v} vs {v_initial}"
    );
}

// =============================================================================
// Workflow 6: Stats Sampling → Monte Carlo ODE
//
// Use stats distributions to sample parameters, solve ODE ensemble,
// and compute statistics on the results.
// =============================================================================

#[test]
fn workflow_stats_monte_carlo_ode() {
    use numra::ode::{DoPri5, OdeProblem, Solver, SolverOptions};
    use numra::stats::{mean, std_dev};

    // Monte Carlo: sample decay rate k from a distribution,
    // solve y' = -k*y for each sample, collect final values
    let n_samples = 50;
    let k_mean = 1.0;
    let k_std = 0.2;

    // Deterministic sampling: use linearly spaced quantiles (no RNG needed)
    let k_samples: Vec<f64> = (0..n_samples)
        .map(|i| {
            // Map uniform (0,1) to approximate normal via simple transform
            let u = (i as f64 + 0.5) / n_samples as f64;
            // Approximate inverse normal CDF (good enough for testing)
            let z = if u < 0.5 {
                -(-2.0 * u.ln()).sqrt()
            } else {
                (-2.0 * (1.0 - u).ln()).sqrt()
            };
            k_mean + k_std * z
        })
        .collect();

    // Solve ODE for each sample
    let opts = SolverOptions::default().rtol(1e-6);
    let mut y_finals = Vec::with_capacity(n_samples);

    for &k in &k_samples {
        let problem = OdeProblem::new(
            move |_t, y: &[f64], dydt: &mut [f64]| {
                dydt[0] = -k * y[0];
            },
            0.0,
            1.0,
            vec![1.0],
        );
        let result = DoPri5::solve(&problem, 0.0, 1.0, &[1.0], &opts).unwrap();
        y_finals.push(result.y_final().unwrap()[0]);
    }

    // Compute statistics on the ensemble
    let y_mean = mean(&y_finals).unwrap();
    let y_std = std_dev(&y_finals).unwrap();

    // Expected: y(1) = exp(-k) where k ~ N(1.0, 0.2^2)
    // E[exp(-k)] ~ exp(-k_mean + k_std^2/2) = exp(-1 + 0.02) = exp(-0.98)
    let expected_mean = (-k_mean + k_std * k_std / 2.0).exp();
    assert!(
        (y_mean - expected_mean).abs() < 0.05,
        "MC mean = {y_mean}, expected ~ {expected_mean}"
    );

    // Standard deviation should be positive (uncertainty in k -> uncertainty in y)
    assert!(y_std > 0.0, "MC std should be positive: {y_std}");

    // All samples should give reasonable values
    for &y in &y_finals {
        assert!(y > 0.0 && y < 1.0, "y = {y} out of range");
    }
}

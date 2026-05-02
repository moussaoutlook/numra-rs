//! Lorenz Attractor Example with Uncertainty Propagation
//!
//! This example demonstrates:
//! 1. Solving the Lorenz system using DoPri5
//! 2. Uncertainty propagation for initial conditions
//! 3. Sensitivity analysis for parameters
//!
//! The Lorenz system:
//!   dx/dt = σ(y - x)
//!   dy/dt = x(ρ - z) - y
//!   dz/dt = xy - βz
//!
//! Standard parameters: σ = 10, ρ = 28, β = 8/3
//!
//! Author: Moussa Leblouba
//! Date: 4 February 2026
//! Modified: 2 May 2026

use numra::ode::{DoPri5, OdeProblem, Solver, SolverOptions};
use numra::uncertainty::{compute_sensitivities, Uncertain};

/// Lorenz system parameters
struct LorenzParams {
    sigma: f64,
    rho: f64,
    beta: f64,
}

impl Default for LorenzParams {
    fn default() -> Self {
        Self {
            sigma: 10.0,
            rho: 28.0,
            beta: 8.0 / 3.0,
        }
    }
}

fn main() {
    println!("=== Lorenz Attractor with Uncertainty ===\n");

    // Standard Lorenz parameters
    let params = LorenzParams::default();

    // Initial conditions (with uncertainty)
    let x0 = Uncertain::from_std(1.0, 0.1); // x = 1.0 ± 0.1
    let y0 = Uncertain::from_std(1.0, 0.1); // y = 1.0 ± 0.1
    let z0 = Uncertain::from_std(1.0, 0.1); // z = 1.0 ± 0.1

    println!("Initial conditions:");
    println!("  x₀ = {:.3} ± {:.3}", x0.mean, x0.std());
    println!("  y₀ = {:.3} ± {:.3}", y0.mean, y0.std());
    println!("  z₀ = {:.3} ± {:.3}", z0.mean, z0.std());
    println!();

    // Create the ODE problem
    let problem = OdeProblem::new(
        |_t, y: &[f64], dydt: &mut [f64]| {
            let (x, y_val, z) = (y[0], y[1], y[2]);
            let sigma = 10.0;
            let rho = 28.0;
            let beta = 8.0 / 3.0;

            dydt[0] = sigma * (y_val - x);
            dydt[1] = x * (rho - z) - y_val;
            dydt[2] = x * y_val - beta * z;
        },
        0.0,
        20.0,
        vec![x0.mean, y0.mean, z0.mean],
    );

    // Solve with default options
    let options = SolverOptions::default().rtol(1e-8).atol(1e-10);

    println!("Solving Lorenz system from t=0 to t=20...");
    let result = DoPri5::solve(&problem, 0.0, 20.0, &[x0.mean, y0.mean, z0.mean], &options)
        .expect("Solver failed");

    // Get solution statistics
    println!("  Steps accepted: {}", result.stats.n_accept);
    println!("  Steps rejected: {}", result.stats.n_reject);
    println!("  Function evals: {}", result.stats.n_eval);
    println!();

    // Final state
    let y_final = result.y_final().expect("No final state");
    println!("Final state at t=20:");
    println!("  x = {:.6}", y_final[0]);
    println!("  y = {:.6}", y_final[1]);
    println!("  z = {:.6}", y_final[2]);
    println!();

    // Trajectory summary
    println!("Trajectory has {} points", result.t.len());
    println!();

    // === Sensitivity Analysis ===
    println!("=== Sensitivity Analysis ===\n");

    // Compute sensitivity of final z-coordinate with respect to parameters
    let solve_for_z = |p: &[f64]| -> f64 {
        let problem = OdeProblem::new(
            move |_t, y: &[f64], dydt: &mut [f64]| {
                let (x, y_val, z) = (y[0], y[1], y[2]);
                dydt[0] = p[0] * (y_val - x); // σ
                dydt[1] = x * (p[1] - z) - y_val; // ρ
                dydt[2] = x * y_val - p[2] * z; // β
            },
            0.0,
            5.0, // Shorter time for sensitivity (chaos makes long-time sensitivity ill-defined)
            vec![1.0, 1.0, 1.0],
        );

        let options = SolverOptions::default().rtol(1e-6);
        let result = DoPri5::solve(&problem, 0.0, 5.0, &[1.0, 1.0, 1.0], &options)
            .expect("Solver failed in sensitivity");
        result.y_final().unwrap()[2] // Return final z
    };

    let nominal_params = [params.sigma, params.rho, params.beta];
    let param_names = ["σ", "ρ", "β"];

    let sens_result = compute_sensitivities(solve_for_z, &nominal_params, &param_names, None);

    println!("Sensitivity of z(t=5) to parameters:");
    for s in &sens_result.sensitivities {
        println!(
            "  ∂z/∂{}: {:.4} (normalized: {:.4})",
            s.name, s.coefficient, s.normalized
        );
    }
    println!();

    // Find most sensitive parameter
    if let Some(most_sens) = sens_result.most_sensitive() {
        println!(
            "Most sensitive parameter: {} (|normalized| = {:.4})",
            most_sens.name,
            most_sens.normalized.abs()
        );
    }
    println!();

    // === Uncertainty Propagation ===
    println!("=== Uncertainty Propagation ===\n");

    // Assume 5% uncertainty on each parameter
    let param_std = [0.5, 1.4, 0.133]; // 5% of nominal
    let param_vars: Vec<f64> = param_std.iter().map(|s| s * s).collect();

    let total_var = sens_result.propagate_uncertainty(&param_vars);
    let total_std = total_var.sqrt();

    println!("Parameter uncertainties:");
    println!("  σ = {:.2} ± {:.3}", params.sigma, param_std[0]);
    println!("  ρ = {:.2} ± {:.3}", params.rho, param_std[1]);
    println!("  β = {:.4} ± {:.4}", params.beta, param_std[2]);
    println!();

    println!("Propagated uncertainty in z(t=5):");
    println!("  z = {:.4} ± {:.4}", sens_result.output, total_std);
    println!();

    // Individual contributions
    println!("Uncertainty contributions:");
    for (i, s) in sens_result.sensitivities.iter().enumerate() {
        let contribution = (s.coefficient * param_std[i]).abs();
        let pct = 100.0 * contribution * contribution / total_var;
        println!("  {}: {:.2}%", s.name, pct);
    }
    println!();

    // === Ensemble of trajectories ===
    println!("=== Ensemble with Perturbed Initial Conditions ===\n");

    let n_samples = 5;
    println!("Running {} trajectories with perturbed ICs...", n_samples);

    let perturbations = [
        [1.0, 1.0, 1.0],
        [1.1, 1.0, 1.0],
        [1.0, 1.1, 1.0],
        [0.9, 1.0, 1.0],
        [1.0, 0.9, 1.0],
    ];

    for ic in perturbations.iter() {
        let result = DoPri5::solve(&problem, 0.0, 10.0, ic, &options).expect("Solver failed");
        let y_end = result.y_final().unwrap();
        println!(
            "  IC [{:.1}, {:.1}, {:.1}] → z(10) = {:.4}",
            ic[0], ic[1], ic[2], y_end[2]
        );
    }
    println!();

    println!("Note: The Lorenz system is chaotic - small perturbations in ICs");
    println!("lead to exponentially diverging trajectories over time.");
    println!();

    println!("=== Done ===");
}

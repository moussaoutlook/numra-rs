//! SPDE Example: Stochastic Heat Equation with Adaptive Time Stepping
//!
//! Solves the 1D stochastic heat equation:
//!
//! ```text
//! du/dt = alpha * d^2u/dx^2 + sigma * dW(x,t)
//! ```
//!
//! on [0, 1] with homogeneous Dirichlet boundary conditions u(0,t) = u(1,t) = 0.
//!
//! This example demonstrates:
//! 1. Setting up an SPDE system with the SpdeSystem trait
//! 2. Comparing fixed vs. adaptive time stepping
//! 3. Convergence towards the deterministic solution as noise decreases
//!
//! The deterministic solution with IC u(x,0) = sin(pi*x) is:
//!   u(x,t) = sin(pi*x) * exp(-alpha * pi^2 * t)
//!
//! Run with: cargo run --example spde_heat
//!
//! Author: Moussa Leblouba
//! Date: 4 February 2026
//! Modified: 2 May 2026

use numra::pde::Grid1D;
use numra::spde::{MolSdeSolver, SpdeOptions, SpdeSolver, SpdeSystem};

/// Stochastic heat equation: du = alpha * u_xx * dt + sigma * dW
struct StochasticHeat {
    alpha: f64,
    sigma: f64,
}

impl SpdeSystem<f64> for StochasticHeat {
    fn drift(&self, _t: f64, u: &[f64], du: &mut [f64], grid: &Grid1D<f64>) {
        let dx = grid.dx_uniform();
        let dx2 = dx * dx;
        let n = u.len();

        for i in 0..n {
            let u_left = if i == 0 { 0.0 } else { u[i - 1] };
            let u_right = if i == n - 1 { 0.0 } else { u[i + 1] };
            du[i] = self.alpha * (u_left - 2.0 * u[i] + u_right) / dx2;
        }
    }

    fn diffusion(&self, _t: f64, _u: &[f64], sigma: &mut [f64], _grid: &Grid1D<f64>) {
        for s in sigma.iter_mut() {
            *s = self.sigma;
        }
    }

    fn is_additive(&self) -> bool {
        true
    }
}

/// Compute the deterministic (exact) solution at time t and spatial points.
fn exact_solution(x: &[f64], t: f64, alpha: f64) -> Vec<f64> {
    let pi = std::f64::consts::PI;
    let decay = (-alpha * pi * pi * t).exp();
    x.iter().map(|&xi| (pi * xi).sin() * decay).collect()
}

fn main() {
    println!("=== Stochastic Heat Equation: Adaptive vs. Fixed Stepping ===\n");

    let alpha = 0.01;
    let length = 1.0;
    let t_final = 0.5;
    let n_points = 31; // total grid points (including boundaries)

    println!("PDE: du/dt = alpha * d^2u/dx^2 + sigma * dW");
    println!(
        "Domain: [0, {}], Dirichlet BCs: u(0)=u({})=0",
        length, length
    );
    println!("IC: u(x,0) = sin(pi*x)\n");

    // Create spatial grid
    let grid = Grid1D::uniform(0.0, length, n_points);
    let dx = grid.dx_uniform();

    println!(
        "Spatial grid: {} total points, {} interior, dx = {:.5}",
        n_points,
        n_points - 2,
        dx
    );

    // Initial condition on interior points
    let interior = grid.interior_points();
    let u0: Vec<f64> = interior
        .iter()
        .map(|&x| (std::f64::consts::PI * x / length).sin())
        .collect();

    // Stability limit for explicit Euler: dt < dx^2 / (2*alpha)
    let dt_stability = dx * dx / (2.0 * alpha);
    let dt_fixed = (dt_stability * 0.4).min(0.001);
    println!("Stability limit dt_max = {:.6}", dt_stability);
    println!("Fixed time step dt = {:.6}\n", dt_fixed);

    // =============================================
    // Part 1: Convergence as noise goes to zero
    // =============================================
    println!("=== Part 1: Noise-to-Zero Convergence ===\n");

    let noise_levels = [0.0, 0.01, 0.05, 0.1, 0.2];
    let exact_final = exact_solution(interior, t_final, alpha);

    println!(
        "{:>8} | {:>12} | {:>12} | {:>8}",
        "sigma", "L2 error", "Linf error", "steps"
    );
    println!("{:-<8}|{:-<13}|{:-<13}|{:-<9}", "", "", "", "");

    for &sigma in &noise_levels {
        let system = StochasticHeat { alpha, sigma };

        let options = SpdeOptions::default().dt(dt_fixed).n_output(50).seed(42);

        let result = MolSdeSolver::solve(&system, 0.0, t_final, &u0, &grid, &options)
            .expect("Solver failed");

        let y_final = result.y_final().unwrap();

        // Compute errors vs. deterministic solution
        let mut l2_err = 0.0;
        let mut linf_err: f64 = 0.0;
        for (i, &exact_val) in exact_final.iter().enumerate() {
            let diff = (y_final[i] - exact_val).abs();
            l2_err += diff * diff;
            linf_err = linf_err.max(diff);
        }
        l2_err = (l2_err / exact_final.len() as f64).sqrt();

        println!(
            "{:>8.3} | {:>12.6} | {:>12.6} | {:>8}",
            sigma, l2_err, linf_err, result.stats.n_steps
        );
    }

    println!("\nNote: With sigma=0 the error is purely from spatial/temporal discretization.");
    println!("As sigma increases, stochastic fluctuations dominate the error.\n");

    // =============================================
    // Part 2: Fixed vs. Adaptive Time Stepping
    // =============================================
    println!("=== Part 2: Fixed vs. Adaptive Time Stepping ===\n");

    let sigma = 0.05;
    let system = StochasticHeat { alpha, sigma };

    // Fixed time stepping
    let options_fixed = SpdeOptions::default().dt(dt_fixed).n_output(50).seed(42);

    println!("Solving with fixed time step (dt = {:.6})...", dt_fixed);
    let result_fixed = MolSdeSolver::solve(&system, 0.0, t_final, &u0, &grid, &options_fixed)
        .expect("Fixed solver failed");

    println!("  Steps: {}", result_fixed.stats.n_steps);
    println!("  Drift evals: {}", result_fixed.stats.n_drift);
    println!("  Diffusion evals: {}", result_fixed.stats.n_diffusion);

    // Adaptive time stepping
    let options_adaptive = SpdeOptions::default()
        .adaptive(true)
        .dt(dt_fixed)
        .rtol(1e-3)
        .atol(1e-6)
        .dt_min(1e-8)
        .dt_max(dt_stability * 0.8)
        .n_output(50)
        .seed(42);

    println!("\nSolving with adaptive time stepping...");
    let result_adaptive = MolSdeSolver::solve(&system, 0.0, t_final, &u0, &grid, &options_adaptive)
        .expect("Adaptive solver failed");

    println!("  Steps: {}", result_adaptive.stats.n_steps);
    println!("  Rejected steps: {}", result_adaptive.stats.n_reject);
    println!("  Drift evals: {}", result_adaptive.stats.n_drift);
    println!("  Diffusion evals: {}", result_adaptive.stats.n_diffusion);

    // Compare final solutions
    let y_fixed = result_fixed.y_final().unwrap();
    let y_adaptive = result_adaptive.y_final().unwrap();

    println!("\n  Comparison at final time t = {}:", t_final);
    println!(
        "  {:>8} | {:>10} | {:>10} | {:>10}",
        "x", "Fixed", "Adaptive", "Exact"
    );
    println!("  {:-<8}|{:-<11}|{:-<11}|{:-<11}", "", "", "", "");

    for i in (0..interior.len()).step_by((interior.len() / 8).max(1)) {
        let x = interior[i];
        println!(
            "  {:>8.4} | {:>10.5} | {:>10.5} | {:>10.5}",
            x, y_fixed[i], y_adaptive[i], exact_final[i]
        );
    }

    // =============================================
    // Part 3: Spatial Profile Evolution
    // =============================================
    println!("\n=== Part 3: Spatial Profile at Selected Times ===\n");

    let system = StochasticHeat { alpha, sigma };
    let options = SpdeOptions::default().dt(dt_fixed).n_output(100).seed(42);

    let result =
        MolSdeSolver::solve(&system, 0.0, t_final, &u0, &grid, &options).expect("Solver failed");

    // Print profiles at selected times
    let target_times = [0.0, 0.1, 0.2, 0.3, 0.5];
    let mid_idx = interior.len() / 2;

    println!(
        "  {:>8} | {:>10} | {:>10} | {:>10}",
        "t", "u(0.5) num", "u(0.5) exact", "difference"
    );
    println!("  {:-<8}|{:-<11}|{:-<11}|{:-<11}", "", "", "", "");

    let mut target_idx = 0;
    for (i, &t) in result.t.iter().enumerate() {
        if target_idx < target_times.len() && t >= target_times[target_idx] {
            let u_mid = result.y[i][mid_idx];
            let exact_mid = (std::f64::consts::PI * interior[mid_idx]).sin()
                * (-alpha * std::f64::consts::PI * std::f64::consts::PI * t).exp();
            let diff = u_mid - exact_mid;
            println!(
                "  {:>8.3} | {:>10.5} | {:>10.5} | {:>+10.5}",
                t, u_mid, exact_mid, diff
            );
            target_idx += 1;
        }
    }

    // Summary
    println!("\n=== Summary ===\n");
    println!("The stochastic heat equation adds white noise to the classical heat equation.");
    println!("Key observations:");
    println!("  - The deterministic solution decays exponentially: u ~ exp(-alpha*pi^2*t)");
    println!("  - Noise causes fluctuations around the deterministic trajectory");
    println!("  - With sigma=0, only discretization error remains");
    println!("  - Adaptive time stepping adjusts dt based on local error estimates");

    println!();
    println!("=== Done ===");
}

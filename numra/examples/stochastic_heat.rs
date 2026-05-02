//! Stochastic Heat Equation Example
//!
//! Demonstrates solving the stochastic heat equation using SPDE solvers:
//!
//! ```text
//! ∂T/∂t = α ∂²T/∂x² + σ ξ(x,t)
//! T(0, t) = T(L, t) = 0   (Dirichlet BCs)
//! T(x, 0) = sin(πx/L)     (initial condition)
//! ```
//!
//! where ξ(x,t) is space-time white noise with intensity σ.
//!
//! The deterministic solution decays exponentially:
//! T(x, t) = sin(πx/L) exp(-α π² t / L²)
//!
//! The stochastic perturbations cause the solution to fluctuate around this.
//!
//! Author: Moussa Leblouba
//! Date: 4 February 2026
//! Modified: 2 May 2026

use numra_pde::Grid1D;
use numra_spde::{MolSdeSolver, SpdeEnsemble, SpdeOptions, SpdeSolver, SpdeSystem};

/// Stochastic heat equation system.
struct StochasticHeat {
    /// Thermal diffusivity
    alpha: f64,
    /// Noise intensity
    sigma: f64,
}

impl SpdeSystem<f64> for StochasticHeat {
    fn drift(&self, _t: f64, u: &[f64], du: &mut [f64], grid: &Grid1D<f64>) {
        let dx = grid.dx_uniform();
        let dx2 = dx * dx;
        let n = u.len();

        // Heat equation: du/dt = α d²u/dx²
        // Using central differences with Dirichlet BCs (T=0 at boundaries)
        for i in 0..n {
            let u_left = if i == 0 { 0.0 } else { u[i - 1] }; // Left BC
            let u_right = if i == n - 1 { 0.0 } else { u[i + 1] }; // Right BC
            du[i] = self.alpha * (u_left - 2.0 * u[i] + u_right) / dx2;
        }
    }

    fn diffusion(&self, _t: f64, _u: &[f64], sigma: &mut [f64], _grid: &Grid1D<f64>) {
        // Additive noise: constant diffusion coefficient
        for s in sigma.iter_mut() {
            *s = self.sigma;
        }
    }

    fn is_additive(&self) -> bool {
        true
    }
}

fn main() {
    println!("=== Stochastic Heat Equation (SPDE) ===\n");

    // Physical parameters
    let alpha = 0.01; // Thermal diffusivity
    let sigma = 0.05; // Noise intensity
    let length = 1.0; // Domain length
    let t_final = 0.5; // Simulation time

    println!("Parameters:");
    println!("  Thermal diffusivity α = {}", alpha);
    println!("  Noise intensity σ = {}", sigma);
    println!("  Domain length L = {} m", length);
    println!("  Final time = {} s\n", t_final);

    // Create spatial grid
    let n_points = 41;
    let grid = Grid1D::uniform(0.0, length, n_points);
    let dx = grid.dx_uniform();
    println!("Grid: {} points, Δx = {:.4} m", n_points, dx);

    // Create SPDE system
    let system = StochasticHeat { alpha, sigma };

    // Initial condition: sin(πx)
    let u0: Vec<f64> = grid
        .interior_points()
        .iter()
        .map(|&x| (std::f64::consts::PI * x / length).sin())
        .collect();

    // Time step (stability: dt < dx² / (2α) for explicit methods)
    let dt_stability = dx * dx / (2.0 * alpha);
    let dt = (dt_stability * 0.5).min(0.001); // Conservative time step
    println!(
        "Time step: {:.6} s (stability limit: {:.6} s)\n",
        dt, dt_stability
    );

    // Solver options
    let options = SpdeOptions::default().dt(dt).n_output(50).seed(12345);

    // Single trajectory
    println!("Solving single trajectory...\n");
    let result = MolSdeSolver::solve(&system, 0.0, t_final, &u0, &grid, &options).unwrap();

    println!("Solver Statistics:");
    println!("  Steps taken: {}", result.stats.n_steps);
    println!("  Drift evaluations: {}", result.stats.n_drift);
    println!("  Diffusion evaluations: {}", result.stats.n_diffusion);

    // Compare with deterministic solution
    let y_final = result.y_final().unwrap();
    let decay_rate = alpha * std::f64::consts::PI * std::f64::consts::PI / (length * length);
    let decay_factor = (-decay_rate * t_final).exp();

    println!("\n=== Solution at t = {} s ===", t_final);
    println!("x (m)\t\tStochastic\tDeterministic");
    println!("{}\t{}\t{}", "-".repeat(6), "-".repeat(10), "-".repeat(13));

    let interior_pts = grid.interior_points();
    for i in (0..y_final.len()).step_by(y_final.len() / 8) {
        let x = interior_pts[i];
        let deterministic = (std::f64::consts::PI * x / length).sin() * decay_factor;
        println!("{:.4}\t\t{:.5}\t\t{:.5}", x, y_final[i], deterministic);
    }

    // Monte Carlo ensemble
    println!("\n=== Monte Carlo Ensemble (20 trajectories) ===\n");
    let n_trajectories = 20;
    let ensemble_options = SpdeOptions::default().dt(dt).n_output(20).seed(42);

    let results = SpdeEnsemble::solve(
        &system,
        0.0,
        t_final,
        &u0,
        &grid,
        &ensemble_options,
        n_trajectories,
    )
    .unwrap();

    // Compute ensemble statistics
    let mean = SpdeEnsemble::mean(&results);
    let std = SpdeEnsemble::std(&results, &mean);

    // Get final time statistics
    let final_idx = mean.len() - 1;
    let mean_final = &mean[final_idx];
    let std_final = &std[final_idx];

    println!("Final time statistics at midpoint (x = L/2):");
    let mid_idx = y_final.len() / 2;
    let det_mid = (std::f64::consts::PI * 0.5).sin() * decay_factor;

    println!("  Deterministic: {:.5}", det_mid);
    println!("  Ensemble mean: {:.5}", mean_final[mid_idx]);
    println!("  Ensemble std:  {:.5}", std_final[mid_idx]);
    println!(
        "  Mean error:    {:.5}",
        (mean_final[mid_idx] - det_mid).abs()
    );

    // Time evolution at midpoint
    println!("\n=== Time Evolution at x = L/2 ===");
    println!("t (s)\t\tMean\t\tStd\t\tDeterministic");

    for (i, mean_t) in mean.iter().enumerate().step_by(mean.len() / 5) {
        let t = results[0].t[i];
        let det = (std::f64::consts::PI * 0.5).sin() * (-decay_rate * t).exp();
        println!(
            "{:.3}\t\t{:.4}\t\t{:.4}\t\t{:.4}",
            t, mean_t[mid_idx], std[i][mid_idx], det
        );
    }

    // Final row
    let t_last = t_final;
    let det_last = (std::f64::consts::PI * 0.5).sin() * (-decay_rate * t_last).exp();
    println!(
        "{:.3}\t\t{:.4}\t\t{:.4}\t\t{:.4}",
        t_last, mean_final[mid_idx], std_final[mid_idx], det_last
    );

    // Spatial variance profile
    println!("\n=== Final Spatial Variance Profile ===");
    println!("x (m)\t\tStd");

    for i in (0..std_final.len()).step_by(std_final.len() / 8) {
        let x = interior_pts[i];
        println!("{:.4}\t\t{:.5}", x, std_final[i]);
    }

    // Summary
    println!("\n=== Summary ===");
    let max_std: f64 = std_final.iter().cloned().fold(0.0, f64::max);
    let mean_std: f64 = std_final.iter().sum::<f64>() / std_final.len() as f64;

    println!("Noise intensity σ = {}", sigma);
    println!("Maximum spatial std at t_final: {:.5}", max_std);
    println!("Mean spatial std at t_final:    {:.5}", mean_std);
    println!("Std / σ ratio (spatial avg):    {:.2}", mean_std / sigma);
}

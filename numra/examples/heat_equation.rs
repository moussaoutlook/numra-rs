//! 1D Heat Equation Example
//!
//! Demonstrates solving the heat equation using Method of Lines:
//!
//! ```text
//! ∂T/∂t = α ∂²T/∂x²,  0 < x < L,  t > 0
//! T(0, t) = T_hot
//! T(L, t) = T_cold
//! T(x, 0) = f(x)
//! ```
//!
//! The analytical solution for sine initial condition is:
//! T(x, t) = T_cold + (T_hot - T_cold)(1 - x/L) + Σ Bₙ sin(nπx/L) exp(-αn²π²t/L²)
//!
//! Author: Moussa Leblouba
//! Date: 2 February 2026
//! Modified: 2 May 2026

use numra_ode::{DoPri5, Solver, SolverOptions};
use numra_pde::boundary::DirichletBC;
use numra_pde::{Grid1D, HeatEquation1D, MOLSystem};

fn main() {
    println!("=== 1D Heat Equation Example ===\n");

    // Problem parameters
    let alpha = 0.01; // Thermal diffusivity (m²/s)
    let length = 1.0; // Domain length (m)
    let t_hot = 100.0; // Hot side temperature (°C)
    let t_cold = 0.0; // Cold side temperature (°C)
    let t_final = 2.0; // Simulation time (s)

    println!("Parameters:");
    println!("  Thermal diffusivity α = {} m²/s", alpha);
    println!("  Domain length L = {} m", length);
    println!("  T(0,t) = {}°C, T(L,t) = {}°C", t_hot, t_cold);
    println!("  Final time = {} s\n", t_final);

    // Create spatial grid
    let n_points = 51;
    let grid = Grid1D::uniform(0.0, length, n_points);
    let dx = grid.dx_uniform();
    println!("Grid: {} points, Δx = {:.4} m", n_points, dx);

    // Create heat equation PDE
    let pde = HeatEquation1D::new(alpha);

    // Boundary conditions
    let bc_left = DirichletBC::new(t_hot);
    let bc_right = DirichletBC::new(t_cold);

    // Create MOL system
    let mol = MOLSystem::new(pde, grid.clone(), bc_left, bc_right);

    // Initial condition: sinusoidal perturbation on linear background
    let u0: Vec<f64> = grid
        .points()
        .iter()
        .map(|&x| {
            // Linear steady state + sinusoidal perturbation
            let steady = t_hot * (1.0 - x / length) + t_cold * (x / length);
            let perturbation = 20.0 * (std::f64::consts::PI * x / length).sin();
            steady + perturbation
        })
        .collect();

    // Solve (interior points only)
    let u0_interior = &u0[1..u0.len() - 1];
    let options = SolverOptions::default().rtol(1e-6).atol(1e-9);

    println!("\nSolving with DoPri5 (adaptive RK5(4))...\n");
    let result = DoPri5::solve(&mol, 0.0, t_final, u0_interior, &options).expect("Solver failed");

    // Display results
    println!("Solution statistics:");
    println!("  Steps accepted: {}", result.stats.n_accept);
    println!("  Steps rejected: {}", result.stats.n_reject);
    println!("  RHS evaluations: {}", result.stats.n_eval);
    println!("  Success: {}", result.success);

    // Build full solution with boundaries
    let y_final = result.y_final().unwrap();
    let t_final_actual = result.t_final().unwrap();

    println!("\nTemperature profile at t = {:.3} s:", t_final_actual);
    println!("  x (m)\t\tT (°C)\t\tSteady State");
    println!(
        "  {}\t\t{}\t\t{}",
        "-".repeat(6),
        "-".repeat(6),
        "-".repeat(12)
    );

    // Print with boundary values
    let x_points = grid.points();
    println!("  {:.4}\t\t{:.2}\t\t{:.2}", x_points[0], t_hot, t_hot);

    for (i, &y) in y_final.iter().enumerate() {
        let x = x_points[i + 1];
        let steady = t_hot * (1.0 - x / length);
        if i % 10 == 4 {
            // Print every 10th point
            println!("  {:.4}\t\t{:.2}\t\t{:.2}", x, y, steady);
        }
    }

    println!(
        "  {:.4}\t\t{:.2}\t\t{:.2}",
        x_points[n_points - 1],
        t_cold,
        t_cold
    );

    // Verify convergence to steady state
    println!("\n=== Convergence Check ===");
    let max_deviation: f64 = y_final
        .iter()
        .enumerate()
        .map(|(i, &y)| {
            let x = x_points[i + 1];
            let steady = t_hot * (1.0 - x / length);
            (y - steady).abs()
        })
        .fold(0.0, f64::max);

    println!("Max deviation from steady state: {:.4}°C", max_deviation);

    if max_deviation < 1.0 {
        println!("Solution has nearly reached steady state!");
    } else {
        println!("Solution is still evolving toward steady state.");
    }

    // Demonstrate time evolution
    println!("\n=== Time Evolution at x = L/2 ===");
    let mid_idx = n_points / 2 - 1; // Interior index for midpoint
    println!("t (s)\t\tT(L/2, t)");

    for (i, &t) in result.t.iter().enumerate().step_by(result.t.len() / 10) {
        let y = result.y_at(i);
        if mid_idx < y.len() {
            println!("{:.3}\t\t{:.2}", t, y[mid_idx]);
        }
    }

    // Final value at midpoint
    if mid_idx < y_final.len() {
        println!("{:.3}\t\t{:.2}", t_final_actual, y_final[mid_idx]);
    }
}

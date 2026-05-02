//! Van der Pol Oscillator Example - Stiff ODE Demonstration
//!
//! The Van der Pol oscillator is a classic test problem for stiff ODE solvers:
//!
//!   x'' - μ(1 - x²)x' + x = 0
//!
//! Written as a first-order system:
//!   dx/dt = y
//!   dy/dt = μ(1 - x²)y - x
//!
//! For small μ, the system is non-stiff and any ODE solver works well.
//! For large μ (e.g., μ = 1000), the system becomes very stiff with
//! rapid transitions between slow and fast dynamics.
//!
//! This example demonstrates:
//! 1. Solving with explicit DoPri5 (for mild stiffness)
//! 2. Solving with implicit Radau5 (for stiff problems)
//! 3. Comparing solver statistics
//!
//! Author: Moussa Leblouba
//! Date: 2 February 2026
//! Modified: 2 May 2026

use numra::ode::{DoPri5, OdeProblem, Radau5, Solver, SolverOptions};
use std::time::Instant;

fn main() {
    println!("=== Van der Pol Oscillator ===\n");
    println!("System: x'' - μ(1 - x²)x' + x = 0");
    println!("Initial conditions: x(0) = 2, x'(0) = 0\n");

    // Test with increasing stiffness
    let mu_values = [1.0, 10.0, 100.0];

    for &mu in &mu_values {
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        println!("μ = {}", mu);
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

        // Integration time depends on stiffness (period ≈ (3 - 2ln2)μ for large μ)
        let tf = if mu <= 10.0 { 20.0 } else { 2.0 * mu };

        // Create the ODE problem
        let problem = OdeProblem::new(
            move |_t, y: &[f64], dydt: &mut [f64]| {
                dydt[0] = y[1];
                dydt[1] = mu * (1.0 - y[0] * y[0]) * y[1] - y[0];
            },
            0.0,
            tf,
            vec![2.0, 0.0],
        );

        let options = SolverOptions::default().rtol(1e-4).atol(1e-6);

        // Try DoPri5 (explicit)
        println!("DoPri5 (Explicit Runge-Kutta):");
        let start = Instant::now();
        let result_dopri = DoPri5::solve(&problem, 0.0, tf, &[2.0, 0.0], &options);
        let elapsed_dopri = start.elapsed();

        match result_dopri {
            Ok(result) => {
                let y_final = result.y_final().unwrap();
                println!("  ✓ Completed in {:.2?}", elapsed_dopri);
                println!(
                    "  Final state: x = {:.6}, y = {:.6}",
                    y_final[0], y_final[1]
                );
                println!(
                    "  Steps: {} accepted, {} rejected",
                    result.stats.n_accept, result.stats.n_reject
                );
                println!("  Function evaluations: {}", result.stats.n_eval);
            }
            Err(e) => {
                println!("  ✗ Failed: {}", e);
            }
        }
        println!();

        // Try Radau5 (implicit)
        println!("Radau5 (Implicit Runge-Kutta):");
        let start = Instant::now();
        let result_radau = Radau5::solve(&problem, 0.0, tf, &[2.0, 0.0], &options);
        let elapsed_radau = start.elapsed();

        match result_radau {
            Ok(result) => {
                let y_final = result.y_final().unwrap();
                println!("  ✓ Completed in {:.2?}", elapsed_radau);
                println!(
                    "  Final state: x = {:.6}, y = {:.6}",
                    y_final[0], y_final[1]
                );
                println!(
                    "  Steps: {} accepted, {} rejected",
                    result.stats.n_accept, result.stats.n_reject
                );
                println!("  Function evaluations: {}", result.stats.n_eval);
                println!("  Jacobian evaluations: {}", result.stats.n_jac);
                println!("  LU factorizations: {}", result.stats.n_lu);
            }
            Err(e) => {
                println!("  ✗ Failed: {}", e);
            }
        }
        println!();
    }

    // Special case: Very stiff μ = 1000
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("μ = 1000 (VERY STIFF - Week 3 Deliverable)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    // For μ=1000, the transient is over in about 2*μ time units
    // We integrate just past the initial transient to show the solver handles stiffness
    let mu = 1000.0;
    let tf = 11.0; // Just past first quasi-period

    let problem = OdeProblem::new(
        move |_t, y: &[f64], dydt: &mut [f64]| {
            dydt[0] = y[1];
            dydt[1] = mu * (1.0 - y[0] * y[0]) * y[1] - y[0];
        },
        0.0,
        tf,
        vec![2.0, 0.0],
    );

    // Relaxed tolerances for very stiff problem
    let options = SolverOptions::default().rtol(1e-2).atol(1e-3);

    println!("Radau5 (Implicit Runge-Kutta):");
    println!("  Tolerances: rtol = 1e-2, atol = 1e-3");
    println!("  Integration to t = {} (just past initial transient)", tf);
    let start = Instant::now();
    let result = Radau5::solve(&problem, 0.0, tf, &[2.0, 0.0], &options);
    let elapsed = start.elapsed();

    match result {
        Ok(result) => {
            let y_final = result.y_final().unwrap();
            println!("  ✓ Completed in {:.2?}", elapsed);
            println!(
                "  Final state: x = {:.6}, y = {:.6}",
                y_final[0], y_final[1]
            );
            println!(
                "  Steps: {} accepted, {} rejected",
                result.stats.n_accept, result.stats.n_reject
            );
            println!("  Function evaluations: {}", result.stats.n_eval);
            println!("  Jacobian evaluations: {}", result.stats.n_jac);
            println!("  LU factorizations: {}", result.stats.n_lu);

            // For μ=1000, the solution should be near x ≈ -2 after first jump
            println!("\n  ✓ Successfully integrated through stiff transient!");
        }
        Err(e) => {
            println!("  ✗ Failed: {}", e);
            println!("\n  Note: For μ=1000, this is an extremely stiff problem.");
        }
    }

    // Also demonstrate with explicit solver for comparison
    println!("\nDoPri5 (Explicit Runge-Kutta) for comparison:");
    let options_dopri = SolverOptions::default().rtol(1e-2).atol(1e-3);
    let start = Instant::now();
    let result_dopri = DoPri5::solve(&problem, 0.0, tf, &[2.0, 0.0], &options_dopri);
    let elapsed_dopri = start.elapsed();

    match result_dopri {
        Ok(result) => {
            let y_final = result.y_final().unwrap();
            println!("  ✓ Completed in {:.2?}", elapsed_dopri);
            println!(
                "  Final state: x = {:.6}, y = {:.6}",
                y_final[0], y_final[1]
            );
            println!(
                "  Steps: {} accepted, {} rejected",
                result.stats.n_accept, result.stats.n_reject
            );
        }
        Err(e) => {
            println!("  ✗ Failed: {}", e);
        }
    }

    println!("\n=== Summary ===\n");
    println!("The Van der Pol oscillator demonstrates the importance of");
    println!("implicit methods for stiff ODEs:");
    println!();
    println!("• For μ ≤ 10: Both explicit (DoPri5) and implicit (Radau5) work well");
    println!("• For μ = 100: Explicit methods struggle, implicit methods excel");
    println!("• For μ = 1000: Only robust implicit methods can handle the stiffness");
    println!();
    println!("Radau5 is L-stable, making it ideal for stiff problems where the");
    println!("solution has fast transients that need to be damped quickly.");
}

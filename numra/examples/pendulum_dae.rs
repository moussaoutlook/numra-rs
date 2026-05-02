//! Pendulum as a Differential-Algebraic Equation (DAE)
//!
//! This example demonstrates DAE concepts using a simple pendulum.
//!
//! ## Standard ODE Formulation (what we solve)
//!
//! In polar coordinates, the pendulum is a simple ODE:
//! ```text
//! θ'' = -(g/L) * sin(θ)
//! ```
//!
//! ## DAE Formulation (for illustration)
//!
//! In Cartesian coordinates with constraint x² + y² = L², it becomes a DAE:
//! ```text
//! M * y' = f(t, y)
//! ```
//! where M has zeros for algebraic equations (the constraint force λ).
//!
//! Run with: cargo run --example pendulum_dae
//!
//! Author: Moussa Leblouba
//! Date: 4 February 2026
//! Modified: 2 May 2026

use numra::ode::{DaeProblem, DoPri5, OdeProblem, Solver, SolverOptions};

fn main() {
    println!("=== Simple Pendulum Example ===\n");

    // Physical parameters
    let l = 1.0; // Pendulum length
    let g = 9.81; // Gravity

    println!("Parameters:");
    println!("  Length L: {} m", l);
    println!("  Gravity g: {} m/s²", g);
    println!();

    // Initial conditions: pendulum at angle θ₀ = π/4, released from rest
    let theta0 = std::f64::consts::FRAC_PI_4; // 45 degrees
    let omega0 = 0.0; // Angular velocity

    println!("Initial conditions:");
    println!("  θ₀ = {:.2}° = {:.6} rad", theta0.to_degrees(), theta0);
    println!("  ω₀ = {} rad/s", omega0);
    println!();

    // ========================================
    // Part 1: Standard ODE formulation
    // ========================================
    println!("=== Part 1: Standard ODE (polar coordinates) ===\n");

    // State: [θ, ω] where ω = dθ/dt
    // θ' = ω
    // ω' = -(g/L) * sin(θ)
    let y0_ode = vec![theta0, omega0];

    let pendulum_ode = move |_t: f64, y: &[f64], dydt: &mut [f64]| {
        let theta = y[0];
        let omega = y[1];
        dydt[0] = omega;
        dydt[1] = -(g / l) * theta.sin();
    };

    let problem = OdeProblem::new(pendulum_ode, 0.0, 10.0, y0_ode.clone());
    let options = SolverOptions::default().rtol(1e-10).atol(1e-12);

    let result = DoPri5::solve(&problem, 0.0, 10.0, &y0_ode, &options)
        .expect("Failed to solve pendulum ODE");

    println!("ODE solved successfully!");
    println!("  Time points: {}", result.len());
    println!("  Accepted steps: {}", result.stats.n_accept);
    println!();

    // Compute period from small-angle approximation: T = 2π√(L/g)
    let period_approx = 2.0 * std::f64::consts::PI * (l / g).sqrt();
    println!("Theoretical period (small angle): {:.4} s", period_approx);
    println!();

    // Display trajectory
    println!("Trajectory (θ in degrees):");
    println!(
        "  {:>6} | {:>10} | {:>10} | {:>10}",
        "t", "θ (deg)", "ω (rad/s)", "E/E₀"
    );
    println!("  {:-<6}|{:-<11}|{:-<11}|{:-<11}", "", "", "", "");

    // Initial energy
    let e0 = 0.5 * l * l * omega0 * omega0 + g * l * (1.0 - theta0.cos());

    let sample_step = (result.len() / 15).max(1);
    for (i, &t) in result.t.iter().enumerate() {
        if i % sample_step == 0 || i == result.len() - 1 {
            let state = result.y_at(i);
            let theta = state[0];
            let omega = state[1];
            let energy = 0.5 * l * l * omega * omega + g * l * (1.0 - theta.cos());

            println!(
                "  {:>6.2} | {:>10.4} | {:>10.4} | {:>10.8}",
                t,
                theta.to_degrees(),
                omega,
                energy / e0
            );
        }
    }

    // ========================================
    // Part 2: Cartesian formulation (demonstrates DAE structure)
    // ========================================
    println!("\n=== Part 2: Cartesian DAE Structure ===\n");

    // Convert to Cartesian for DAE demonstration
    let x0 = l * theta0.sin();
    let y0_cart = -l * theta0.cos();
    let vx0 = 0.0;
    let vy0 = 0.0;

    println!("Cartesian initial conditions:");
    println!("  x₀ = {:.6}", x0);
    println!("  y₀ = {:.6}", y0_cart);
    println!(
        "  Constraint check: x² + y² = {:.6} (should be L² = {})",
        x0 * x0 + y0_cart * y0_cart,
        l * l
    );
    println!();

    // DAE structure: M * [x', y', vx', vy', λ']ᵀ = [vx, vy, -λx, -λy-g, constraint]ᵀ
    // where M = diag(1,1,1,1,0) - the last row has M[4,4]=0 (algebraic)

    // For a true DAE solver, we would use:
    let rhs = move |_t: f64, y: &[f64], dydt: &mut [f64]| {
        let x = y[0];
        let y_pos = y[1];
        let vx = y[2];
        let vy = y[3];
        let lambda = y[4];

        dydt[0] = vx;
        dydt[1] = vy;
        dydt[2] = -lambda * x;
        dydt[3] = -lambda * y_pos - g;
        // Constraint: x*vx + y*vy = 0 (velocity perpendicular to position)
        // This is the differentiated constraint from x² + y² = L²
        dydt[4] = x * vx + y_pos * vy;
    };

    let mass = |m: &mut [f64]| {
        for val in m.iter_mut().take(25) {
            *val = 0.0;
        }
        m[0] = 1.0; // M[0,0] = 1
        m[6] = 1.0; // M[1,1] = 1
        m[12] = 1.0; // M[2,2] = 1
        m[18] = 1.0; // M[3,3] = 1
                     // m[24] = 0.0 - algebraic equation for λ
    };

    // Initial λ from constraint at rest: vx' = vy' = 0
    // From 0 = -λ*y - g => λ = -g/y (at rest)
    let lambda0 = -g / y0_cart;

    let _dae_problem = DaeProblem::new(
        rhs,
        mass,
        0.0,
        10.0,
        vec![x0, y0_cart, vx0, vy0, lambda0],
        vec![4], // λ (index 4) is algebraic
    );

    println!("DAE structure created:");
    println!("  State dimension: 5 (x, y, vx, vy, λ)");
    println!("  Algebraic indices: [4] (constraint force λ)");
    println!("  Mass matrix: diag(1, 1, 1, 1, 0)");
    println!();
    println!("Note: Full DAE support with singular mass matrices in implicit solvers");
    println!("like Radau5 requires additional implementation (LU with pivoting for");
    println!("the augmented system). The ODE formulation above demonstrates the physics.");
    println!();

    // Verify conservation
    let final_state = result.y_final().unwrap();
    let theta_f = final_state[0];
    let omega_f = final_state[1];
    let e_final = 0.5 * l * l * omega_f * omega_f + g * l * (1.0 - theta_f.cos());

    println!("Final state verification:");
    println!("  Final θ: {:.4}°", theta_f.to_degrees());
    println!("  Final ω: {:.4} rad/s", omega_f);
    println!("  Energy conservation: E/E₀ = {:.10}", e_final / e0);
    println!("  (Should be 1.0 for a conservative system)");
}

//! Week 4 Deliverable: Full ODE Solver Suite with Auto-Selection
//!
//! This example demonstrates:
//! 1. All available ODE solvers (explicit and implicit)
//! 2. Automatic solver selection based on problem characteristics
//! 3. Comparison of solver performance on different problem types
//!
//! Run with: cargo run --example solver_zoo
//!
//! Author: Moussa Leblouba
//! Date: 4 February 2026
//! Modified: 2 May 2026

use numra::ode::{
    // Automatic selection
    auto_solve,
    auto_solve_with_hints,
    Accuracy,
    Bdf,
    // Explicit Runge-Kutta methods
    DoPri5,
    Esdirk32,
    Esdirk43,
    Esdirk54,
    OdeProblem,
    // Implicit methods
    Radau5,
    Solver,
    SolverHints,
    SolverOptions,
    Stiffness,
    Tsit5,
    Vern6,
    Vern7,
    Vern8,
};

fn main() {
    println!("╔═══════════════════════════════════════════════════════════════╗");
    println!("║        Numra ODE Solver Suite - Week 4 Deliverable            ║");
    println!("╚═══════════════════════════════════════════════════════════════╝\n");

    // === Part 1: Non-stiff problem comparison ===
    println!("━━━ Part 1: Non-stiff Problem (Harmonic Oscillator) ━━━\n");

    let problem_nonstiff = OdeProblem::new(
        |_t, y: &[f64], dydt: &mut [f64]| {
            dydt[0] = y[1]; // dx/dt = v
            dydt[1] = -y[0]; // dv/dt = -x
        },
        0.0,
        10.0,
        vec![1.0, 0.0],
    );
    let options = SolverOptions::default().rtol(1e-6);

    println!("Solving: dx/dt = v, dv/dt = -x (harmonic oscillator)");
    println!("Initial: x=1, v=0, solving to t=10\n");

    let methods = [
        (
            "DoPri5",
            solve_with::<DoPri5, _>(&problem_nonstiff, &options),
        ),
        (
            "Tsit5 ",
            solve_with::<Tsit5, _>(&problem_nonstiff, &options),
        ),
        (
            "Vern6 ",
            solve_with::<Vern6, _>(&problem_nonstiff, &options),
        ),
        (
            "Vern7 ",
            solve_with::<Vern7, _>(&problem_nonstiff, &options),
        ),
        (
            "Vern8 ",
            solve_with::<Vern8, _>(&problem_nonstiff, &options),
        ),
    ];

    println!("  Method    x(10)        Error        Steps   Evals");
    println!("  ────────────────────────────────────────────────────");
    let expected_x = 10.0_f64.cos();

    for (name, result) in methods {
        if let Ok(r) = result {
            let y = r.y_final().unwrap();
            let err = (y[0] - expected_x).abs();
            println!(
                "  {}   {:+.8}   {:.2e}   {:>5}   {:>5}",
                name, y[0], err, r.stats.n_accept, r.stats.n_eval
            );
        } else {
            println!("  {}   FAILED", name);
        }
    }
    println!();

    // === Part 2: Stiff problem comparison ===
    println!("━━━ Part 2: Stiff Problem (Fast Decay) ━━━\n");

    let problem_stiff = OdeProblem::new(
        |_t, y: &[f64], dydt: &mut [f64]| {
            dydt[0] = -100.0 * y[0]; // Very fast decay
        },
        0.0,
        0.2,
        vec![1.0],
    );
    let options_stiff = SolverOptions::default().rtol(1e-4).atol(1e-6);

    println!("Solving: dy/dt = -100*y (stiff decay)");
    println!("Initial: y=1, solving to t=0.2\n");

    let methods_stiff = [
        (
            "Radau5  ",
            solve_with::<Radau5, _>(&problem_stiff, &options_stiff),
        ),
        (
            "BDF     ",
            solve_with::<Bdf, _>(&problem_stiff, &options_stiff),
        ),
        (
            "ESDIRK32",
            solve_with::<Esdirk32, _>(&problem_stiff, &options_stiff),
        ),
        (
            "ESDIRK43",
            solve_with::<Esdirk43, _>(&problem_stiff, &options_stiff),
        ),
        (
            "ESDIRK54",
            solve_with::<Esdirk54, _>(&problem_stiff, &options_stiff),
        ),
    ];

    println!("  Method      y(0.2)       Error        Steps   LU");
    println!("  ────────────────────────────────────────────────────");
    let expected_y = (-20.0_f64).exp();

    for (name, result) in methods_stiff {
        if let Ok(r) = result {
            let y = r.y_final().unwrap();
            let err = (y[0] - expected_y).abs();
            println!(
                "  {}   {:.6e}   {:.2e}   {:>5}   {:>3}",
                name, y[0], err, r.stats.n_accept, r.stats.n_lu
            );
        } else {
            println!("  {}   FAILED", name);
        }
    }
    println!();

    // === Part 3: Automatic Solver Selection ===
    println!("━━━ Part 3: Automatic Solver Selection ━━━\n");

    // Non-stiff problem with auto
    println!("Test 1: Non-stiff problem (auto should choose explicit method)");
    let result_auto1 = auto_solve(&problem_nonstiff, 0.0, 10.0, &[1.0, 0.0], &options);
    if let Ok(r) = result_auto1 {
        let y = r.y_final().unwrap();
        println!(
            "  Result: x(10) = {:.8}, steps = {}",
            y[0], r.stats.n_accept
        );
        println!("  ✓ Auto-selection successful\n");
    }

    // Stiff problem with auto
    println!("Test 2: Stiff problem with hint");
    let hints_stiff = SolverHints::new().stiffness(Stiffness::VeryStiff);
    let result_auto2 = auto_solve_with_hints(
        &problem_stiff,
        0.0,
        0.2,
        &[1.0],
        &options_stiff,
        &hints_stiff,
    );
    if let Ok(r) = result_auto2 {
        let y = r.y_final().unwrap();
        println!(
            "  Result: y(0.2) = {:.6e}, LU decompositions = {}",
            y[0], r.stats.n_lu
        );
        println!("  ✓ Auto-selection with stiffness hint successful\n");
    }

    // High accuracy with auto
    println!("Test 3: High accuracy requirement");
    let options_high = SolverOptions::default().rtol(1e-10).atol(1e-12);
    let hints_high = SolverHints::new()
        .stiffness(Stiffness::NonStiff)
        .accuracy(Accuracy::VeryHigh);

    let result_auto3 = auto_solve_with_hints(
        &problem_nonstiff,
        0.0,
        10.0,
        &[1.0, 0.0],
        &options_high,
        &hints_high,
    );
    if let Ok(r) = result_auto3 {
        let y = r.y_final().unwrap();
        let err = (y[0] - expected_x).abs();
        println!("  Result: x(10) = {:.12}, error = {:.2e}", y[0], err);
        println!("  ✓ High-accuracy auto-selection successful\n");
    }

    // === Part 4: Van der Pol Oscillator ===
    println!("━━━ Part 4: Van der Pol Oscillator (Increasing Stiffness) ━━━\n");

    for mu in [1.0, 10.0, 100.0] {
        let problem = OdeProblem::new(
            move |_t, y: &[f64], dydt: &mut [f64]| {
                dydt[0] = y[1];
                dydt[1] = mu * (1.0 - y[0] * y[0]) * y[1] - y[0];
            },
            0.0,
            20.0,
            vec![2.0, 0.0],
        );
        let options_vdp = SolverOptions::default().rtol(1e-4).atol(1e-6);

        // Choose appropriate method based on stiffness
        let stiffness = if mu >= 100.0 {
            Stiffness::VeryStiff
        } else if mu >= 10.0 {
            Stiffness::ModeratelyStiff
        } else {
            Stiffness::NonStiff
        };

        let hints = SolverHints::new().stiffness(stiffness);
        let result = auto_solve_with_hints(&problem, 0.0, 20.0, &[2.0, 0.0], &options_vdp, &hints);

        match result {
            Ok(r) => {
                let y = r.y_final().unwrap();
                println!(
                    "  μ = {:>3}: x(20) = {:+.4}, v(20) = {:+.4}, steps = {}",
                    mu as i32, y[0], y[1], r.stats.n_accept
                );
            }
            Err(e) => {
                println!("  μ = {:>3}: FAILED - {}", mu as i32, e);
            }
        }
    }
    println!();

    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Week 4 Deliverable: Full ODE Solver Suite ✓");
    println!("  - 5 Explicit RK methods: DoPri5, Tsit5, Vern6, Vern7, Vern8");
    println!("  - 5 Implicit methods: Radau5, BDF, ESDIRK32, ESDIRK43, ESDIRK54");
    println!("  - Automatic solver selection with stiffness detection");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
}

fn solve_with<M: Solver<f64>, F>(
    problem: &OdeProblem<f64, F>,
    options: &SolverOptions<f64>,
) -> Result<numra::ode::SolverResult<f64>, numra::ode::SolverError>
where
    F: Fn(f64, &[f64], &mut [f64]),
{
    M::solve(problem, problem.t0, problem.tf, &problem.y0, options)
}

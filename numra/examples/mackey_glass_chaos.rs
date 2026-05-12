//! DDE Example: Mackey-Glass Equation -- Parameter Study
//!
//! The Mackey-Glass equation is a classical delay differential equation:
//!
//! ```text
//! y'(t) = beta * y(t-tau) / (1 + y(t-tau)^n) - gamma * y(t)
//! ```
//!
//! This example complements the basic `mackey_glass` example by exploring
//! how different delay values (tau) affect the dynamics:
//!
//! - tau = 6:  periodic (limit cycle)
//! - tau = 17: chaotic (classic chaos parameter)
//! - tau = 30: hyperchaotic
//!
//! Run with: cargo run --example mackey_glass_chaos
//!
//! Author: Moussa Leblouba
//! Date: 3 February 2026
//! Modified: 2 May 2026

use numra::dde::{DdeOptions, DdeSolver, DdeSystem, MethodOfSteps};

/// Mackey-Glass DDE system parameterized for easy reuse.
struct MackeyGlass {
    beta: f64,
    gamma: f64,
    n: f64,
    tau: f64,
}

impl DdeSystem<f64> for MackeyGlass {
    fn dim(&self) -> usize {
        1
    }

    fn delays(&self) -> Vec<f64> {
        vec![self.tau]
    }

    fn rhs(&self, _t: f64, y: &[f64], y_delayed: &[&[f64]], dydt: &mut [f64]) {
        let y_tau = y_delayed[0][0]; // y(t - tau)
        dydt[0] = self.beta * y_tau / (1.0 + y_tau.powf(self.n)) - self.gamma * y[0];
    }
}

/// Solve the Mackey-Glass equation and return summary statistics.
fn solve_and_analyze(tau: f64, t_final: f64) -> Result<(f64, f64, f64, usize), String> {
    let system = MackeyGlass {
        beta: 2.0,
        gamma: 1.0,
        n: 9.65,
        tau,
    };

    let history = |_t: f64| vec![0.5];

    let options = DdeOptions::default()
        .rtol(1e-6)
        .atol(1e-9)
        .max_steps(200_000);

    let result = MethodOfSteps::solve(&system, 0.0, t_final, &history, &options)?;

    // Analyze only the second half of the trajectory (after transients)
    let half = result.t.len() / 2;
    let mut y_min = f64::INFINITY;
    let mut y_max = f64::NEG_INFINITY;
    for i in half..result.y.len() {
        let val = result.y[i];
        if val < y_min {
            y_min = val;
        }
        if val > y_max {
            y_max = val;
        }
    }

    let y_final = result.y_final().unwrap()[0];
    let n_steps = result.stats.n_accept;

    Ok((y_min, y_max, y_final, n_steps))
}

fn main() {
    println!("=== Mackey-Glass DDE: Parameter Study ===\n");
    println!("Equation: y'(t) = beta*y(t-tau)/(1+y(t-tau)^n) - gamma*y(t)");
    println!("Fixed parameters: beta=2, gamma=1, n=9.65");
    println!("History: y(t) = 0.5 for t <= 0\n");

    // Study how the delay tau affects dynamics
    let delays = [2.0, 6.0, 10.0, 17.0, 23.0, 30.0];
    let t_final = 500.0;

    println!(
        "Solving for {} different delay values (tau)...\n",
        delays.len()
    );
    println!(
        "{:>6} | {:>8} | {:>8} | {:>8} | {:>8} | {:>12}",
        "tau", "y_min", "y_max", "range", "y_final", "steps"
    );
    println!(
        "{:-<6}|{:-<9}|{:-<9}|{:-<9}|{:-<9}|{:-<13}",
        "", "", "", "", "", ""
    );

    let mut results_data: Vec<(f64, f64, f64, f64)> = Vec::new();

    for &tau in &delays {
        match solve_and_analyze(tau, t_final) {
            Ok((y_min, y_max, y_final, n_steps)) => {
                let range = y_max - y_min;
                println!(
                    "{:>6.1} | {:>8.4} | {:>8.4} | {:>8.4} | {:>8.4} | {:>12}",
                    tau, y_min, y_max, range, y_final, n_steps
                );
                results_data.push((tau, y_min, y_max, range));
            }
            Err(e) => {
                println!("{:>6.1} | FAILED: {}", tau, e);
            }
        }
    }

    println!();

    // Classify dynamics based on range
    println!("=== Dynamics Classification ===\n");
    for &(tau, _y_min, _y_max, range) in &results_data {
        let classification = if range < 0.01 {
            "Stable equilibrium"
        } else if range < 0.3 {
            "Small oscillations"
        } else if range < 0.6 {
            "Periodic (limit cycle)"
        } else if range < 1.0 {
            "Quasi-periodic / Chaotic"
        } else {
            "Strongly chaotic"
        };
        println!(
            "  tau = {:>5.1}: range = {:.4} => {}",
            tau, range, classification
        );
    }

    println!();

    // Detailed trajectory for the classic chaotic case (tau=17)
    println!("=== Detailed Trajectory for tau = 17 (classic chaos) ===\n");

    let system = MackeyGlass {
        beta: 2.0,
        gamma: 1.0,
        n: 9.65,
        tau: 17.0,
    };

    let history = |_t: f64| vec![0.5];

    let options = DdeOptions::default()
        .rtol(1e-8)
        .atol(1e-10)
        .max_steps(200_000);

    let result = MethodOfSteps::solve(&system, 0.0, t_final, &history, &options)
        .expect("Failed to solve with tau=17");

    println!("Solver statistics:");
    println!("  Time points: {}", result.len());
    println!("  Accepted steps: {}", result.stats.n_accept);
    println!("  Rejected steps: {}", result.stats.n_reject);
    println!();

    // Sample trajectory
    println!("Trajectory samples (tau=17):");
    println!("  {:>10} | {:>12}", "t", "y(t)");
    println!("  {:-<10}|{:-<13}", "", "");

    let sample_times = [
        0.0, 25.0, 50.0, 100.0, 150.0, 200.0, 250.0, 300.0, 350.0, 400.0, 450.0, 500.0,
    ];
    let mut sample_idx = 0;

    for (i, &t) in result.t.iter().enumerate() {
        if sample_idx < sample_times.len() && t >= sample_times[sample_idx] {
            let y = result.y_at(i)[0];
            println!("  {:>10.1} | {:>12.6}", t, y);
            sample_idx += 1;
        }
    }

    // Sensitivity to initial history
    println!("\n=== Sensitivity to Initial History ===\n");
    println!("Comparing y(0)=0.5 vs y(0)=0.51 at t=500 (tau=17):\n");

    let history_b = |_t: f64| vec![0.51];

    let result_b = MethodOfSteps::solve(&system, 0.0, t_final, &history_b, &options)
        .expect("Failed to solve perturbed system");

    let y_a = result.y_final().unwrap()[0];
    let y_b = result_b.y_final().unwrap()[0];

    println!("  y(500) with history=0.50: {:.8}", y_a);
    println!("  y(500) with history=0.51: {:.8}", y_b);
    println!("  Difference: {:.8}", (y_a - y_b).abs());
    println!();
    println!("A large difference from a small initial perturbation is a hallmark");
    println!("of chaotic dynamics (sensitive dependence on initial conditions).");

    println!();
    println!("=== Done ===");
}

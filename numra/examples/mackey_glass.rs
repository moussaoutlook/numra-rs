//! Mackey-Glass Delay Differential Equation Example
//!
//! The Mackey-Glass equation is a classic DDE that exhibits chaotic behavior:
//!
//! ```text
//! y'(t) = β * y(t-τ) / (1 + y(t-τ)^n) - γ * y(t)
//! ```
//!
//! With parameters β=2, γ=1, n=9.65, τ=2, the system exhibits chaotic dynamics.
//!
//! Run with: cargo run --example mackey_glass
//!
//! Author: Moussa Leblouba
//! Date: 2 February 2026
//! Modified: 2 May 2026

use numra::dde::{DdeOptions, DdeSolver, DdeSystem, MethodOfSteps};

/// Mackey-Glass equation
struct MackeyGlass {
    /// Production rate
    beta: f64,
    /// Decay rate
    gamma: f64,
    /// Hill coefficient (nonlinearity)
    n: f64,
    /// Time delay
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
                                     // dy/dt = β * y(t-τ) / (1 + y(t-τ)^n) - γ * y(t)
        dydt[0] = self.beta * y_tau / (1.0 + y_tau.powf(self.n)) - self.gamma * y[0];
    }
}

fn main() {
    println!("=== Mackey-Glass Delay Differential Equation ===\n");

    // Standard chaotic parameters
    let system = MackeyGlass {
        beta: 2.0,
        gamma: 1.0,
        n: 9.65,
        tau: 2.0,
    };

    println!("Parameters:");
    println!("  β (production rate): {}", system.beta);
    println!("  γ (decay rate): {}", system.gamma);
    println!("  n (Hill coefficient): {}", system.n);
    println!("  τ (delay): {}", system.tau);
    println!();

    // Constant history function
    let history = |_t: f64| vec![0.5];

    // Solver options
    let options = DdeOptions::default()
        .rtol(1e-6)
        .atol(1e-9)
        .max_steps(100000);

    println!("Solving from t=0 to t=500...\n");

    let result = MethodOfSteps::solve(&system, 0.0, 500.0, &history, &options)
        .expect("Failed to solve Mackey-Glass equation");

    println!("Solution computed successfully!");
    println!("  Number of time points: {}", result.len());
    println!("  Accepted steps: {}", result.stats.n_accept);
    println!("  Rejected steps: {}", result.stats.n_reject);
    println!();

    // Find min, max, and final value
    let mut y_min = f64::INFINITY;
    let mut y_max = f64::NEG_INFINITY;
    for y in result.y.iter() {
        if *y < y_min {
            y_min = *y;
        }
        if *y > y_max {
            y_max = *y;
        }
    }

    let y_final = result.y_final().unwrap()[0];

    println!("Solution statistics:");
    println!("  Minimum y: {:.6}", y_min);
    println!("  Maximum y: {:.6}", y_max);
    println!("  Range: {:.6}", y_max - y_min);
    println!("  Final y(500): {:.6}", y_final);
    println!();

    // Sample some values for the trajectory
    println!("Sample trajectory points:");
    println!("  {:>10} | {:>12}", "t", "y(t)");
    println!("  {:-<10}|{:-<13}", "", "");

    let sample_times = [0.0, 50.0, 100.0, 200.0, 300.0, 400.0, 500.0];
    let mut sample_idx = 0;

    for (i, &t) in result.t.iter().enumerate() {
        if sample_idx < sample_times.len() && t >= sample_times[sample_idx] {
            let y = result.y_at(i)[0];
            println!("  {:>10.1} | {:>12.6}", t, y);
            sample_idx += 1;
        }
    }

    println!();
    println!("The Mackey-Glass equation with τ=2 exhibits chaotic oscillations.");
    println!(
        "The solution oscillates irregularly between approximately {} and {}.",
        y_min.round() as i32,
        y_max.round() as i32
    );
}

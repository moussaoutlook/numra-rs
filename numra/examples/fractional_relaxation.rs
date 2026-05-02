//! Fractional Relaxation Example
//!
//! Demonstrates solving fractional differential equations using the L1 scheme.
//!
//! The fractional relaxation equation is:
//!   D^α y(t) = -λ y(t),  y(0) = 1
//!
//! where D^α is the Caputo fractional derivative of order α ∈ (0, 1].
//!
//! The exact solution is the Mittag-Leffler function:
//!   y(t) = E_α(-λ t^α)
//!
//! Key properties:
//! - When α = 1: reduces to exponential decay y(t) = exp(-λt)
//! - When α < 1: exhibits slower-than-exponential decay (heavy tail)
//! - Describes anomalous relaxation in viscoelastic materials
//!
//! Author: Moussa Leblouba
//! Date: 2 February 2026
//! Modified: 2 May 2026

use numra_fde::{mittag_leffler_1, FdeOptions, FdeSolver, FdeSystem, L1Solver};

/// Fractional relaxation system: D^α y = -λy
struct FractionalRelaxation {
    lambda: f64,
    alpha: f64,
}

impl FractionalRelaxation {
    fn new(lambda: f64, alpha: f64) -> Self {
        Self { lambda, alpha }
    }

    /// Exact solution using Mittag-Leffler function
    fn exact_solution(&self, t: f64) -> f64 {
        mittag_leffler_1(-self.lambda * t.powf(self.alpha), self.alpha)
    }
}

impl FdeSystem<f64> for FractionalRelaxation {
    fn dim(&self) -> usize {
        1
    }

    fn alpha(&self) -> f64 {
        self.alpha
    }

    fn rhs(&self, _t: f64, y: &[f64], f: &mut [f64]) {
        f[0] = -self.lambda * y[0];
    }
}

fn main() {
    println!("=== Fractional Relaxation Example ===\n");

    let lambda = 1.0;
    let t_final = 3.0;
    let y0 = [1.0];

    // Compare different fractional orders
    let alphas = [1.0, 0.9, 0.7, 0.5];

    println!("Fractional relaxation: D^α y = -{:.1} y, y(0) = 1", lambda);
    println!("Comparing orders α = {:?}\n", alphas);

    for alpha in alphas {
        let system = FractionalRelaxation::new(lambda, alpha);
        let options = FdeOptions::default().dt(0.01);

        let result =
            L1Solver::solve(&system, 0.0, t_final, &y0, &options).expect("FDE solve failed");

        // Get solution at specific time points
        let check_times = [0.5, 1.0, 2.0, 3.0];

        println!("α = {:.1}:", alpha);
        println!("  t\t  Computed\t  Exact\t\t  Error");
        println!(
            "  {}\t  {}\t  {}\t  {}",
            "-".repeat(4),
            "-".repeat(8),
            "-".repeat(8),
            "-".repeat(8)
        );

        for &t_check in &check_times {
            // Find closest index
            let idx = (t_check / 0.01_f64).round() as usize;
            if idx < result.t.len() {
                let y_computed = result.y_at(idx)[0];
                let y_exact = system.exact_solution(t_check);
                let error = (y_computed - y_exact).abs();

                println!(
                    "  {:.1}\t  {:.6}\t  {:.6}\t  {:.2e}",
                    t_check, y_computed, y_exact, error
                );
            }
        }
        println!();
    }

    // Demonstrate the "heavy tail" property
    println!("=== Heavy Tail Behavior ===");
    println!("At t = 3.0, comparing decay rates:\n");

    for alpha in alphas {
        let system = FractionalRelaxation::new(lambda, alpha);
        let y_alpha = system.exact_solution(3.0);
        let y_exp = (-lambda * 3.0_f64).exp(); // Standard exponential

        println!(
            "α = {:.1}: y(3) = {:.6}, ratio to exp(-3) = {:.2}",
            alpha,
            y_alpha,
            y_alpha / y_exp
        );
    }

    println!("\nSmaller α = slower decay (longer memory)");
    println!("This models anomalous diffusion and viscoelastic relaxation.");

    // Solver statistics
    println!("\n=== Solver Statistics (α = 0.5) ===");
    let system = FractionalRelaxation::new(lambda, 0.5);
    let options = FdeOptions::default().dt(0.01);
    let result = L1Solver::solve(&system, 0.0, t_final, &y0, &options).unwrap();

    println!("Steps taken: {}", result.stats.n_steps);
    println!("RHS evaluations: {}", result.stats.n_rhs);
    println!("Final time: {:.2}", result.t_final().unwrap());
    println!("Final value: {:.6}", result.y_final().unwrap()[0]);
}

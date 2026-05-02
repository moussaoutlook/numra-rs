//! Viscoelastic Material Example with Prony Series
//!
//! Demonstrates solving integro-differential equations using the efficient
//! Prony series (Sum of Exponentials) approach.
//!
//! Models a viscoelastic material under stress:
//!   σ'(t) = E₀ ε'(t) - ∫₀ᵗ K(t-s) ε'(s) ds
//!
//! where:
//! - σ(t) is the stress response
//! - ε(t) is the applied strain
//! - E₀ is the instantaneous modulus
//! - K(t) is the relaxation kernel (Prony series)
//!
//! For a generalized Maxwell model with multiple relaxation times:
//!   K(t) = Σᵢ Eᵢ/τᵢ exp(-t/τᵢ)
//!
//! The Prony solver computes the memory integral recursively in O(1) memory
//! per step, avoiding the O(N²) cost of full quadrature.
//!
//! Author: Moussa Leblouba
//! Date: 2 February 2026
//! Modified: 2 May 2026

use numra_ide::kernels::PronyKernel;
use numra_ide::prony::{PronySolver, PronySystem};
use numra_ide::{IdeOptions, IdeSolver, IdeSystem, VolterraSolver};

/// Simple viscoelastic stress-strain response using Volterra formulation.
///
/// The equation is:
///   y'(t) = -k*y(t) + ∫₀ᵗ a*exp(-b*(t-s)) * y(s) ds
struct SimpleViscoelastic {
    k: f64,         // Local relaxation rate
    amplitude: f64, // Kernel amplitude
    rate: f64,      // Kernel decay rate
}

impl IdeSystem<f64> for SimpleViscoelastic {
    fn dim(&self) -> usize {
        1
    }

    fn rhs(&self, _t: f64, y: &[f64], f: &mut [f64]) {
        f[0] = -self.k * y[0];
    }

    fn kernel(&self, t: f64, s: f64, y_s: &[f64], k: &mut [f64]) {
        let tau = t - s;
        k[0] = self.amplitude * (-self.rate * tau).exp() * y_s[0];
    }

    fn is_convolution_kernel(&self) -> bool {
        true
    }
}

/// Viscoelastic material with Prony series kernel.
///
/// This uses the efficient recursive formulation.
struct PronyViscoelastic {
    k: f64,
    kernel: PronyKernel<f64>,
}

impl PronyViscoelastic {
    fn with_single_mode(k: f64, a: f64, b: f64) -> Self {
        Self {
            k,
            kernel: PronyKernel::single(a, b),
        }
    }

    fn with_maxwell_modes(k: f64, moduli: Vec<f64>, relaxation_times: Vec<f64>) -> Self {
        // Convert moduli and relaxation times to Prony coefficients
        // For Maxwell: amplitude = E_i / τ_i, rate = 1 / τ_i
        let amplitudes: Vec<f64> = moduli
            .iter()
            .zip(relaxation_times.iter())
            .map(|(e, tau)| e / tau)
            .collect();
        let rates: Vec<f64> = relaxation_times.iter().map(|tau| 1.0 / tau).collect();

        Self {
            k,
            kernel: PronyKernel::new(amplitudes, rates),
        }
    }
}

impl PronySystem<f64> for PronyViscoelastic {
    fn dim(&self) -> usize {
        1
    }

    fn rhs(&self, _t: f64, y: &[f64], f: &mut [f64]) {
        f[0] = -self.k * y[0];
    }

    fn kernel(&self) -> &PronyKernel<f64> {
        &self.kernel
    }
}

fn main() {
    println!("=== Viscoelastic Material with Prony Series ===\n");

    // Material parameters
    let k = 1.0; // Local relaxation rate
    let a = 0.5; // Memory amplitude
    let b = 0.3; // Memory decay rate
    let t_final = 5.0;
    let y0 = [1.0];

    // Compare Volterra (full quadrature) vs Prony (recursive)
    println!("Comparing Volterra solver (full quadrature) vs Prony solver (recursive)\n");

    let volterra_system = SimpleViscoelastic {
        k,
        amplitude: a,
        rate: b,
    };
    let prony_system = PronyViscoelastic::with_single_mode(k, a, b);

    let options = IdeOptions::default().dt(0.01);

    // Solve with Volterra
    let volterra_result = VolterraSolver::solve(&volterra_system, 0.0, t_final, &y0, &options)
        .expect("Volterra solve failed");

    // Solve with Prony
    let prony_result =
        PronySolver::solve(&prony_system, 0.0, t_final, &y0, &options).expect("Prony solve failed");

    // Compare results
    println!("t\t  Volterra\t  Prony\t\t  Difference");
    println!(
        "{}\t  {}\t  {}\t  {}",
        "-".repeat(4),
        "-".repeat(8),
        "-".repeat(8),
        "-".repeat(10)
    );

    let check_times = [0.5, 1.0, 2.0, 3.0, 5.0];
    for &t in &check_times {
        let idx = (t / 0.01_f64).round() as usize;
        if idx < volterra_result.t.len() && idx < prony_result.t.len() {
            let y_v = volterra_result.y_at(idx)[0];
            let y_p = prony_result.y_at(idx)[0];
            let diff = (y_v - y_p).abs();
            println!("{:.1}\t  {:.6}\t  {:.6}\t  {:.2e}", t, y_v, y_p, diff);
        }
    }

    // Statistics comparison
    println!("\n=== Solver Statistics ===");
    println!(
        "Volterra: {} steps, {} RHS, {} kernel evals",
        volterra_result.stats.n_steps, volterra_result.stats.n_rhs, volterra_result.stats.n_kernel
    );
    println!(
        "Prony:    {} steps, {} RHS, {} kernel evals",
        prony_result.stats.n_steps, prony_result.stats.n_rhs, prony_result.stats.n_kernel
    );

    // Memory effect demonstration
    println!("\n=== Memory Effect ===");
    println!("Without memory integral (pure relaxation y' = -ky):");
    let y_no_memory = (-k * 2.0_f64).exp();
    println!("  y(2) = exp(-2) = {:.6}", y_no_memory);

    let idx = (2.0_f64 / 0.01_f64).round() as usize;
    let y_with_memory = prony_result.y_at(idx)[0];
    println!("\nWith memory integral:");
    println!("  y(2) = {:.6}", y_with_memory);
    println!("\nThe memory integral adds positive feedback, slowing decay.");
    println!("Ratio: {:.2}x slower", y_with_memory / y_no_memory);

    // Generalized Maxwell model with multiple relaxation times
    println!("\n=== Generalized Maxwell Model ===");
    println!("Multiple relaxation modes: τ₁=0.5s, τ₂=2.0s, τ₃=10.0s\n");

    let multi_mode = PronyViscoelastic::with_maxwell_modes(
        1.0,
        vec![0.3, 0.5, 0.2],  // Moduli E_i
        vec![0.5, 2.0, 10.0], // Relaxation times τ_i
    );

    let options_long = IdeOptions::default().dt(0.05);
    let multi_result = PronySolver::solve(&multi_mode, 0.0, 20.0, &y0, &options_long)
        .expect("Multi-mode solve failed");

    println!("Stress response at different times:");
    println!("t\t  σ(t)/σ₀");
    for &t in &[0.5, 1.0, 2.0, 5.0, 10.0, 20.0] {
        let idx = (t / 0.05_f64).round() as usize;
        if idx < multi_result.t.len() {
            let y = multi_result.y_at(idx)[0];
            println!("{:.1}\t  {:.4}", t, y);
        }
    }

    println!("\n=== Efficiency Demonstration ===");
    println!("Kernel evaluations scale linearly with Prony (not quadratically).");
    println!("Number of Prony terms: {}", multi_mode.kernel.num_terms());
    println!(
        "Total kernel evaluations for 400 steps: {}",
        multi_result.stats.n_kernel
    );
    println!(
        "Per-step kernel evals: {:.1}",
        multi_result.stats.n_kernel as f64 / multi_result.stats.n_steps as f64
    );
}

//! Geometric Brownian Motion Monte Carlo Example
//!
//! Demonstrates SDE solving with ensemble statistics for option pricing.
//!
//! GBM: dS = μS dt + σS dW
//! Solution: S(T) = S(0) * exp((μ - σ²/2)T + σW(T))
//!
//! Run with: cargo run --example gbm_monte_carlo
//!
//! Author: Moussa Leblouba
//! Date: 4 February 2026
//! Modified: 2 May 2026

use numra::sde::{
    EnsembleRunner, EnsembleStats, EulerMaruyama, Milstein, SdeOptions, SdeSystem, Sra1,
};

/// Geometric Brownian Motion model for stock prices.
#[allow(clippy::upper_case_acronyms)]
struct GBM {
    /// Drift rate (expected return)
    mu: f64,
    /// Volatility
    sigma: f64,
}

impl SdeSystem<f64> for GBM {
    fn dim(&self) -> usize {
        1
    }

    fn drift(&self, _t: f64, x: &[f64], f: &mut [f64]) {
        f[0] = self.mu * x[0];
    }

    fn diffusion(&self, _t: f64, x: &[f64], g: &mut [f64]) {
        g[0] = self.sigma * x[0];
    }

    fn diffusion_derivative(&self, _t: f64, x: &[f64], gdg: &mut [f64]) {
        // g = σx, dg/dx = σ, so g * dg/dx = σ²x
        gdg[0] = self.sigma * self.sigma * x[0];
    }
}

fn main() {
    println!("=== Geometric Brownian Motion Monte Carlo ===\n");

    // Model parameters
    let s0 = 100.0; // Initial stock price
    let mu = 0.05; // 5% expected annual return
    let sigma = 0.2; // 20% annual volatility
    let t_final = 1.0; // 1 year

    // Number of Monte Carlo paths
    let n_paths = 10_000;

    let gbm = GBM { mu, sigma };

    println!("Parameters:");
    println!("  Initial price: ${:.2}", s0);
    println!("  Drift (μ): {:.1}%", mu * 100.0);
    println!("  Volatility (σ): {:.1}%", sigma * 100.0);
    println!("  Time horizon: {:.1} year", t_final);
    println!("  Monte Carlo paths: {}", n_paths);
    println!();

    // Analytical solution for comparison
    // E[S(T)] = S(0) * exp(μT)
    // Var[S(T)] = S(0)² * exp(2μT) * (exp(σ²T) - 1)
    let expected_mean = s0 * (mu * t_final).exp();
    let expected_var =
        s0 * s0 * (2.0 * mu * t_final).exp() * ((sigma * sigma * t_final).exp() - 1.0);
    let expected_std = expected_var.sqrt();

    println!("Analytical (exact) results:");
    println!("  E[S(T)] = ${:.2}", expected_mean);
    println!("  Std[S(T)] = ${:.2}", expected_std);
    println!();

    // Compare different solvers
    let options = SdeOptions::default()
        .dt(0.001) // Small step for accuracy
        .seed(42)
        .save_trajectory(false); // Only need final values

    println!("Running Monte Carlo simulations...\n");

    // Euler-Maruyama
    let em_result =
        EnsembleRunner::run::<_, _, EulerMaruyama>(&gbm, 0.0, t_final, &[s0], &options, n_paths);
    let em_finals = em_result.successful_final_values(0);
    let em_stats = EnsembleStats::from_samples(&em_finals).unwrap();

    println!("Euler-Maruyama (strong order 0.5):");
    println!(
        "  Mean: ${:.2} (error: {:.2}%)",
        em_stats.mean,
        (em_stats.mean - expected_mean).abs() / expected_mean * 100.0
    );
    println!(
        "  Std:  ${:.2} (error: {:.2}%)",
        em_stats.std,
        (em_stats.std - expected_std).abs() / expected_std * 100.0
    );
    println!(
        "  95% CI: [${:.2}, ${:.2}]",
        em_stats.percentiles.p5, em_stats.percentiles.p95
    );
    println!();

    // Milstein
    let mil_result =
        EnsembleRunner::run::<_, _, Milstein>(&gbm, 0.0, t_final, &[s0], &options, n_paths);
    let mil_finals = mil_result.successful_final_values(0);
    let mil_stats = EnsembleStats::from_samples(&mil_finals).unwrap();

    println!("Milstein (strong order 1.0):");
    println!(
        "  Mean: ${:.2} (error: {:.2}%)",
        mil_stats.mean,
        (mil_stats.mean - expected_mean).abs() / expected_mean * 100.0
    );
    println!(
        "  Std:  ${:.2} (error: {:.2}%)",
        mil_stats.std,
        (mil_stats.std - expected_std).abs() / expected_std * 100.0
    );
    println!(
        "  95% CI: [${:.2}, ${:.2}]",
        mil_stats.percentiles.p5, mil_stats.percentiles.p95
    );
    println!();

    // SRA1 (adaptive)
    let sra_options = SdeOptions::default()
        .dt(0.01) // Larger initial step (adaptive will adjust)
        .rtol(1e-3)
        .atol(1e-6)
        .seed(42)
        .save_trajectory(false);

    let sra_result =
        EnsembleRunner::run::<_, _, Sra1>(&gbm, 0.0, t_final, &[s0], &sra_options, n_paths);
    let sra_finals = sra_result.successful_final_values(0);
    let sra_stats = EnsembleStats::from_samples(&sra_finals).unwrap();

    println!("SRA1 (adaptive, strong order 1.5):");
    println!(
        "  Mean: ${:.2} (error: {:.2}%)",
        sra_stats.mean,
        (sra_stats.mean - expected_mean).abs() / expected_mean * 100.0
    );
    println!(
        "  Std:  ${:.2} (error: {:.2}%)",
        sra_stats.std,
        (sra_stats.std - expected_std).abs() / expected_std * 100.0
    );
    println!(
        "  95% CI: [${:.2}, ${:.2}]",
        sra_stats.percentiles.p5, sra_stats.percentiles.p95
    );
    println!();

    // Option pricing example: European call option
    println!("=== European Call Option Pricing ===\n");

    let strike = 105.0;
    let r = 0.03; // Risk-free rate

    // For risk-neutral pricing, use μ = r (risk-neutral drift)
    let gbm_rn = GBM { mu: r, sigma };

    let rn_result =
        EnsembleRunner::run::<_, _, Milstein>(&gbm_rn, 0.0, t_final, &[s0], &options, n_paths);
    let rn_finals = rn_result.successful_final_values(0);

    // Payoff: max(S(T) - K, 0)
    let payoffs: Vec<f64> = rn_finals.iter().map(|&s| (s - strike).max(0.0)).collect();
    let payoff_stats = EnsembleStats::from_samples(&payoffs).unwrap();

    // Discounted expected payoff = option price
    let discount = (-r * t_final).exp();
    let mc_price = discount * payoff_stats.mean;
    let mc_std_err = discount * payoff_stats.standard_error();

    println!("Call option parameters:");
    println!("  Strike: ${:.2}", strike);
    println!("  Risk-free rate: {:.1}%", r * 100.0);
    println!();
    println!(
        "Monte Carlo price: ${:.4} ± ${:.4} (95% CI)",
        mc_price,
        1.96 * mc_std_err
    );

    // Black-Scholes analytical price for comparison
    let bs_price = black_scholes_call(s0, strike, r, sigma, t_final);
    println!("Black-Scholes price: ${:.4}", bs_price);
    println!(
        "Pricing error: ${:.4} ({:.2}%)",
        (mc_price - bs_price).abs(),
        (mc_price - bs_price).abs() / bs_price * 100.0
    );
}

/// Black-Scholes formula for European call option.
fn black_scholes_call(s: f64, k: f64, r: f64, sigma: f64, t: f64) -> f64 {
    let d1 = ((s / k).ln() + (r + 0.5 * sigma * sigma) * t) / (sigma * t.sqrt());
    let d2 = d1 - sigma * t.sqrt();
    s * norm_cdf(d1) - k * (-r * t).exp() * norm_cdf(d2)
}

/// Standard normal CDF approximation (Abramowitz & Stegun).
fn norm_cdf(x: f64) -> f64 {
    // Approximation using the error function via polynomial
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let p = 0.3275911;

    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs() / std::f64::consts::SQRT_2;

    let t = 1.0 / (1.0 + p * x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();

    0.5 * (1.0 + sign * y)
}

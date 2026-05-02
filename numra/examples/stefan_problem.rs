//! Stefan Problem Example (Moving Boundary)
//!
//! The one-phase Stefan problem models melting/solidification:
//!
//! ```text
//! ∂T/∂t = α ∂²T/∂x²,  0 < x < s(t)
//! T(0, t) = T_hot > T_melt
//! T(s(t), t) = T_melt
//! ds/dt = -κ ∂T/∂x |_{x=s(t)}  (Stefan condition)
//! ```
//!
//! We use coordinate transformation ξ = x/s(t) to map to fixed domain [0, 1].
//!
//! Transformed equation:
//! ∂T/∂t = (α/s²) ∂²T/∂ξ² + (ξ * ds/dt / s) ∂T/∂ξ
//!
//! Author: Moussa Leblouba
//! Date: 4 February 2026
//! Modified: 2 May 2026

use numra_ode::{DoPri5, OdeSystem, Solver, SolverOptions};
use numra_pde::moving::CoordinateTransform;

/// Stefan problem system.
///
/// State vector: [T_1, T_2, ..., T_{n-1}, s]
/// where T_i are interior temperatures and s is the interface position.
struct StefanSystem {
    /// Thermal diffusivity
    alpha: f64,
    /// Stefan coefficient κ = k / (ρ * L)
    kappa: f64,
    /// Hot wall temperature
    t_hot: f64,
    /// Melting temperature
    t_melt: f64,
    /// Number of interior grid points
    n_interior: usize,
}

impl StefanSystem {
    fn new(alpha: f64, kappa: f64, t_hot: f64, t_melt: f64, n_interior: usize) -> Self {
        Self {
            alpha,
            kappa,
            t_hot,
            t_melt,
            n_interior,
        }
    }

    /// Build full temperature profile including boundaries.
    fn build_full_profile(&self, y: &[f64]) -> Vec<f64> {
        let mut full = Vec::with_capacity(self.n_interior + 2);
        full.push(self.t_hot); // Left BC
        full.extend_from_slice(&y[..self.n_interior]);
        full.push(self.t_melt); // Right BC at interface
        full
    }

    /// Get interface position from state vector.
    fn _interface_position(&self, y: &[f64]) -> f64 {
        y[self.n_interior]
    }
}

impl OdeSystem<f64> for StefanSystem {
    fn dim(&self) -> usize {
        self.n_interior + 1 // Temperature at interior points + interface position
    }

    fn rhs(&self, _t: f64, y: &[f64], dydt: &mut [f64]) {
        let n = self.n_interior;
        let s = y[n]; // Interface position

        // Grid spacing in computational domain [0, 1]
        let dxi = 1.0 / (n as f64 + 1.0);
        let dxi2 = dxi * dxi;

        // Build full temperature array
        let t_full = self.build_full_profile(y);

        // Coordinate transform
        let transform = CoordinateTransform::new(s);

        // Compute interface velocity (Stefan condition)
        // ds/dt = -κ * (∂T/∂x at x=s) = -κ * (1/s) * (∂T/∂ξ at ξ=1)
        let dt_dxi_interface =
            (3.0 * t_full[n + 1] - 4.0 * t_full[n] + t_full[n - 1]) / (2.0 * dxi);
        let dt_dx_interface = transform.transform_d1(dt_dxi_interface);
        let dsdt = -self.kappa * dt_dx_interface;

        // Heat equation in transformed coordinates
        // ∂T/∂t = (α/s²) ∂²T/∂ξ² + (ξ * ds/dt / s) ∂T/∂ξ
        let alpha_s2 = self.alpha / (s * s);

        for (i, dydt_i) in dydt.iter_mut().enumerate().take(n) {
            let idx = i + 1; // Index in full array
            let xi = (i as f64 + 1.0) * dxi; // Computational coordinate

            // Second derivative in ξ
            let d2t_dxi2 = (t_full[idx + 1] - 2.0 * t_full[idx] + t_full[idx - 1]) / dxi2;

            // First derivative in ξ
            let dt_dxi = (t_full[idx + 1] - t_full[idx - 1]) / (2.0 * dxi);

            // Convective term from moving boundary
            let convection = transform.convection_coefficient(dsdt, xi) * dt_dxi;

            // Full RHS
            *dydt_i = alpha_s2 * d2t_dxi2 + convection;
        }

        // Interface velocity
        dydt[n] = dsdt;
    }
}

fn main() {
    println!("=== One-Phase Stefan Problem (Melting) ===\n");

    // Physical parameters (ice melting)
    let alpha = 1.0e-6; // Thermal diffusivity (m²/s) - water/ice
    let kappa = 1.0e-6; // Stefan coefficient (m²/s/K)
    let t_hot = 20.0; // Hot wall temperature (°C)
    let t_melt = 0.0; // Melting temperature (°C)

    // Initial conditions
    let s_initial = 0.001; // Initial melt thickness (1 mm)
    let t_final = 100.0; // Simulation time (s)

    // Grid
    let n_interior = 20;
    let n_total = n_interior + 2;

    println!("Physical Parameters:");
    println!("  Thermal diffusivity α = {:.2e} m²/s", alpha);
    println!("  Stefan coefficient κ = {:.2e} m²/s/K", kappa);
    println!("  Hot wall T = {}°C", t_hot);
    println!("  Melting point T_melt = {}°C", t_melt);
    println!("\nInitial Conditions:");
    println!("  Initial melt thickness s₀ = {:.1} mm", s_initial * 1000.0);
    println!("  Simulation time = {} s", t_final);
    println!("  Grid points: {} interior\n", n_interior);

    // Create system
    let system = StefanSystem::new(alpha, kappa, t_hot, t_melt, n_interior);

    // Initial state: linear temperature profile + interface position
    let mut y0 = vec![0.0; n_interior + 1];
    for (i, y0_i) in y0.iter_mut().enumerate().take(n_interior) {
        let xi = (i as f64 + 1.0) / (n_interior as f64 + 1.0);
        *y0_i = t_hot * (1.0 - xi) + t_melt * xi;
    }
    y0[n_interior] = s_initial;

    // Solve
    let options = SolverOptions::default()
        .rtol(1e-6)
        .atol(1e-9)
        .max_steps(100000);

    println!("Solving with DoPri5...\n");
    let result = DoPri5::solve(&system, 0.0, t_final, &y0, &options).expect("Solver failed");

    println!("Solver Statistics:");
    println!("  Steps accepted: {}", result.stats.n_accept);
    println!("  RHS evaluations: {}", result.stats.n_eval);

    // Extract results
    let y_final = result.y_final().unwrap();
    let s_final = y_final[n_interior];

    println!("\n=== Results ===");
    println!("Final interface position: {:.3} mm", s_final * 1000.0);
    println!(
        "Interface velocity: {:.3} μm/s",
        (s_final - s_initial) / t_final * 1e6
    );

    // Analytical solution for comparison (Neumann solution)
    // s(t) = 2 * λ * sqrt(α * t)
    // where λ satisfies: λ * exp(λ²) * erf(λ) = St / sqrt(π)
    // St = c_p * (T_hot - T_melt) / L (Stefan number)
    println!("\n=== Interface Evolution ===");
    println!("t (s)\t\ts (mm)\t\tds/dt (μm/s)");

    let mut prev_s = s_initial;
    let mut prev_t = 0.0;

    for (i, &t) in result.t.iter().enumerate() {
        if i % (result.t.len() / 10).max(1) == 0 || i == result.t.len() - 1 {
            let s = result.y_at(i)[n_interior];
            let dsdt = if t > prev_t {
                (s - prev_s) / (t - prev_t) * 1e6
            } else {
                0.0
            };
            println!("{:.1}\t\t{:.3}\t\t{:.3}", t, s * 1000.0, dsdt);
            prev_s = s;
            prev_t = t;
        }
    }

    // Temperature profile at final time
    println!("\n=== Final Temperature Profile ===");
    println!("x (mm)\t\tT (°C)");
    let t_profile = system.build_full_profile(&y_final);
    for (i, &t_val) in t_profile.iter().enumerate().take(n_total) {
        let xi = i as f64 / (n_total as f64 - 1.0);
        let x = xi * s_final;
        if i % 4 == 0 || i == n_total - 1 {
            println!("{:.3}\t\t{:.2}", x * 1000.0, t_val);
        }
    }

    // Verify Stefan condition is satisfied
    let dxi = 1.0 / (n_interior as f64 + 1.0);
    let dt_dxi = (3.0 * t_profile[n_total - 1] - 4.0 * t_profile[n_total - 2]
        + t_profile[n_total - 3])
        / (2.0 * dxi);
    let dt_dx = dt_dxi / s_final;
    let expected_dsdt = -kappa * dt_dx;

    println!("\n=== Stefan Condition Verification ===");
    println!("∂T/∂x at interface: {:.2} K/m", dt_dx);
    println!("Expected ds/dt: {:.3} μm/s", expected_dsdt * 1e6);
}

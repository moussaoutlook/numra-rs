//! Robertson Stiff Kinetics — Forward Sensitivity Analysis
//!
//! The Robertson chemical kinetics system is the canonical stiff ODE benchmark
//! and a textbook test case for sensitivity analysis. The reaction network
//!
//!     A  -> B          (rate k1 = 0.04)
//!     2B -> C + B      (rate k2 = 3.0e7)
//!     B + C -> A + C   (rate k3 = 1.0e4)
//!
//! gives the ODE system
//!
//!     dy0/dt = -k1·y0 + k3·y1·y2
//!     dy1/dt =  k1·y0 - k2·y1^2 - k3·y1·y2
//!     dy2/dt =  k2·y1^2
//!
//! with y(0) = [1, 0, 0]. The rate constants span eight orders of magnitude,
//! so the system is severely stiff — `Radau5` or `Bdf` is required.
//!
//! This example walks through:
//!
//!   1. Implementing `ParametricOdeSystem` with analytical state and
//!      parameter Jacobians, and *correctly* flagging both as analytical
//!      (forgetting either flag silently degrades to FD; in debug builds
//!      `AugmentedSystem` panics with a clear message — see commit
//!      `34d81b6`'s safety net).
//!   2. Calling `solve_forward_sensitivity::<Radau5, _, _>` to compute
//!      `S(t) = dy/dk` alongside the state trajectory.
//!   3. Reading individual components via `dyi_dpj(time_idx, state, param)`,
//!      contiguous per-parameter slices via `sensitivity_for_param`, and
//!      the dimensionless logarithmic sensitivity via
//!      `normalized_sensitivity_at`.
//!   4. Ranking parameters by influence on each state — the standard
//!      first-pass insight from a sensitivity run.
//!
//! Run:
//!
//!     cargo run --release --example robertson_sensitivity -p numra
//!
//! Author: Moussa Leblouba
//! Date: 6 May 2026

use numra::ode::{Radau5, SolverOptions};
use numra::{solve_forward_sensitivity, ParametricOdeSystem};

/// Robertson stiff kinetics, parameterised by the three rate constants.
///
/// Implements [`ParametricOdeSystem`] with hand-coded Jacobians. State
/// dimension `N = 3`, parameter dimension `N_s = 3`.
struct Robertson {
    k: [f64; 3],
}

impl ParametricOdeSystem<f64> for Robertson {
    fn n_states(&self) -> usize {
        3
    }

    fn n_params(&self) -> usize {
        3
    }

    fn params(&self) -> &[f64] {
        &self.k
    }

    /// `f(t, y, k) = dy/dt` for arbitrary `k`. Used by both the integrator
    /// and the FD-default Jacobians (which we override below).
    fn rhs_with_params(&self, _t: f64, y: &[f64], k: &[f64], dy: &mut [f64]) {
        let (y0, y1, y2) = (y[0], y[1], y[2]);
        let (k1, k2, k3) = (k[0], k[1], k[2]);
        dy[0] = -k1 * y0 + k3 * y1 * y2;
        dy[1] = k1 * y0 - k2 * y1 * y1 - k3 * y1 * y2;
        dy[2] = k2 * y1 * y1;
    }

    /// State Jacobian `J_y[i, j] = ∂f_i/∂y_j` in **row-major** order.
    fn jacobian_y(&self, _t: f64, y: &[f64], jy: &mut [f64]) {
        let (y1, y2) = (y[1], y[2]);
        let (k1, k2, k3) = (self.k[0], self.k[1], self.k[2]);
        // Row 0: ∂(dy0/dt)/∂y_*
        jy[0] = -k1; // ∂/∂y0
        jy[1] = k3 * y2; // ∂/∂y1
        jy[2] = k3 * y1; // ∂/∂y2
                         // Row 1: ∂(dy1/dt)/∂y_*
        jy[3] = k1; // ∂/∂y0
        jy[4] = -2.0 * k2 * y1 - k3 * y2; // ∂/∂y1
        jy[5] = -k3 * y1; // ∂/∂y2
                          // Row 2: ∂(dy2/dt)/∂y_*
        jy[6] = 0.0; // ∂/∂y0
        jy[7] = 2.0 * k2 * y1; // ∂/∂y1
        jy[8] = 0.0; // ∂/∂y2
    }

    /// Parameter Jacobian `J_p[i, k] = ∂f_i/∂p_k` in **column-major** order
    /// (`jp[k * N + i]`). Note the asymmetry vs `jacobian_y` — column-major
    /// here lets `AugmentedSystem` slice off `J_p[:, k]` as a contiguous
    /// vector when assembling the per-parameter sensitivity column ODE.
    fn jacobian_p(&self, _t: f64, y: &[f64], jp: &mut [f64]) {
        let (y0, y1, y2) = (y[0], y[1], y[2]);
        // Column 0: ∂f/∂k1
        jp[0] = -y0; // ∂(dy0/dt)/∂k1
        jp[1] = y0; // ∂(dy1/dt)/∂k1
        jp[2] = 0.0; // ∂(dy2/dt)/∂k1
                     // Column 1: ∂f/∂k2
        jp[3] = 0.0; // ∂(dy0/dt)/∂k2
        jp[4] = -y1 * y1; // ∂(dy1/dt)/∂k2
        jp[5] = y1 * y1; // ∂(dy2/dt)/∂k2
                         // Column 2: ∂f/∂k3
        jp[6] = y1 * y2; // ∂(dy0/dt)/∂k3
        jp[7] = -y1 * y2; // ∂(dy1/dt)/∂k3
        jp[8] = 0.0; // ∂(dy2/dt)/∂k3
    }

    // Both Jacobians are analytical — flag accordingly so `AugmentedSystem`
    // calls our overrides instead of running its own forward FD. Forgetting
    // either flag is silent in release builds and a debug-build panic
    // (commit `34d81b6` safety net), so always pair the override with the
    // matching flag.
    fn has_analytical_jacobian_y(&self) -> bool {
        true
    }
    fn has_analytical_jacobian_p(&self) -> bool {
        true
    }
}

fn main() {
    println!("=== Robertson Stiff Kinetics — Forward Sensitivity ===\n");
    println!("Reaction network:");
    println!("  A  -> B          (k1 = 0.04)");
    println!("  2B -> C + B      (k2 = 3.0e7)");
    println!("  B + C -> A + C   (k3 = 1.0e4)\n");

    let k_nominal = [0.04_f64, 3.0e7, 1.0e4];
    let system = Robertson { k: k_nominal };
    let y0 = [1.0_f64, 0.0, 0.0];
    let t_final = 40.0;

    // Tight tolerances — sensitivity equations inherit the same.
    let opts = SolverOptions::default().rtol(1e-8).atol(1e-12);

    // Solve the augmented (state + sensitivity) system. Radau5 is required:
    // BDF works too, but at this stiffness Radau5 typically takes fewer steps.
    let result = solve_forward_sensitivity::<Radau5, f64, _>(&system, 0.0, t_final, &y0, &opts)
        .expect("Radau5 augmented solve failed");
    assert!(result.success, "{}", result.message);

    println!("Integration: t in [0, {}]", t_final);
    println!(
        "  output points : {}\n  RHS evals     : {}\n  Jacobian evals: {}\n  accepted steps: {}\n  rejected steps: {}\n  LU decomps    : {}",
        result.len(),
        result.stats.n_eval,
        result.stats.n_jac,
        result.stats.n_accept,
        result.stats.n_reject,
        result.stats.n_lu,
    );

    // ---- State at the final time ------------------------------------------
    let y_final = result.final_state();
    println!("\nState at t = {}:", t_final);
    println!("  y0 (A)  = {:.6e}", y_final[0]);
    println!("  y1 (B)  = {:.6e}", y_final[1]);
    println!("  y2 (C)  = {:.6e}", y_final[2]);
    let mass = y_final[0] + y_final[1] + y_final[2];
    println!("  sum     = {:.12} (should be 1.0)", mass);

    // ---- Raw sensitivity matrix at final time -----------------------------
    // Layout: column-major over parameters. `dyi_dpj(i, state, param)` is
    // the stable accessor; never index `result.sensitivity` directly.
    let last = result.len() - 1;
    println!("\nSensitivity matrix S(t={}) = dy/dk:", t_final);
    println!("                    ∂/∂k1          ∂/∂k2          ∂/∂k3");
    for state in 0..3 {
        print!("  ∂y{}/∂k_*  =", state);
        for param in 0..3 {
            print!("  {:>13.4e}", result.dyi_dpj(last, state, param));
        }
        println!();
    }

    // ---- Per-parameter slice ----------------------------------------------
    // `sensitivity_for_param(t_idx, k)` returns a contiguous `&[f64]` of
    // length `N` — handy when you want to feed `dy/dp_k` into a downstream
    // gradient calculation without copying.
    println!("\nPer-parameter sensitivity vectors (final time):");
    for param in 0..3 {
        let col = result.sensitivity_for_param(last, param);
        println!(
            "  ∂y/∂k{} = [{:>11.4e}, {:>11.4e}, {:>11.4e}]",
            param + 1,
            col[0],
            col[1],
            col[2],
        );
    }

    // ---- Logarithmic (normalised) sensitivity -----------------------------
    // S_norm[j, k] = (k_k / y_j) · ∂y_j/∂k_k. Dimensionless and directly
    // comparable across parameters with wildly different magnitudes — the
    // standard first-pass tool for ranking parameter influence.
    println!("\nNormalised sensitivity (k_k / y_j) · (∂y_j/∂k_k):");
    println!("                    k1             k2             k3");
    let s_norm = result.normalized_sensitivity_at(last, &k_nominal);
    let n = system.n_states();
    for state in 0..3 {
        print!("  y{} normalised =", state);
        for param in 0..3 {
            print!("  {:>13.4e}", s_norm[param * n + state]);
        }
        println!();
    }

    // ---- Influence ranking -------------------------------------------------
    // For each state, sort the three parameters by absolute normalised
    // sensitivity. This is the actionable summary a calibration workflow
    // would consume to decide which constants to refit and which to fix.
    println!("\nInfluence ranking (largest |normalised sensitivity| first):");
    let names = ["k1", "k2", "k3"];
    for state in 0..3 {
        let mut ranked: Vec<(usize, f64)> = (0..3)
            .map(|param| (param, s_norm[param * n + state].abs()))
            .collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let formatted: Vec<String> = ranked
            .iter()
            .map(|(p, mag)| format!("{} ({:.2e})", names[*p], mag))
            .collect();
        println!("  y{}: {}", state, formatted.join(" > "));
    }

    println!("\nDone.");
}

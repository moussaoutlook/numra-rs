//! Event Detection Example: Bouncing Ball
//!
//! Simulates a bouncing ball under gravity with coefficient of restitution:
//!
//! ```text
//! y'' = -g
//! ```
//!
//! The system is integrated as a first-order ODE:
//!   y[0]' = y[1]       (height' = velocity)
//!   y[1]' = -g          (velocity' = -gravity)
//!
//! Event detection triggers when height y[0] = 0 (ground contact, falling direction).
//! At each bounce, the velocity is reversed and scaled by the coefficient of
//! restitution e: v_after = -e * v_before.
//!
//! The simulation chains multiple integration segments, stopping at each event
//! and restarting with modified initial conditions.
//!
//! Run with: cargo run --example bouncing_ball
//!
//! Author: Moussa Leblouba
//! Date: 4 February 2026
//! Modified: 2 May 2026

use numra::ode::events::{EventAction, EventDirection, EventFunction};
use numra::ode::{DoPri5, OdeProblem, Solver, SolverOptions};

/// Event function that detects when the ball hits the ground (height = 0).
struct GroundContact;

impl EventFunction<f64> for GroundContact {
    fn evaluate(&self, _t: f64, y: &[f64]) -> f64 {
        y[0] // height; event triggers when this crosses zero
    }

    fn direction(&self) -> EventDirection {
        EventDirection::Falling // only when height goes from positive to negative
    }

    fn action(&self) -> EventAction {
        EventAction::Stop // stop integration so we can apply the bounce
    }
}

fn main() {
    println!("=== Bouncing Ball with Event Detection ===\n");

    // Physical parameters
    let g = 9.81; // gravitational acceleration (m/s^2)
    let e = 0.8; // coefficient of restitution (energy loss per bounce)
    let h0 = 10.0; // initial height (m)
    let v0 = 0.0; // initial velocity (m/s), released from rest
    let t_end = 10.0; // maximum simulation time (s)
    let max_bounces = 20; // stop after this many bounces

    println!("Parameters:");
    println!("  Gravity g = {} m/s^2", g);
    println!("  Coefficient of restitution e = {}", e);
    println!("  Initial height h0 = {} m", h0);
    println!("  Initial velocity v0 = {} m/s", v0);
    println!("  Max simulation time = {} s", t_end);
    println!();

    // Analytical time to first bounce: t = sqrt(2*h0/g)
    let t_first_bounce = (2.0_f64 * h0 / g).sqrt();
    println!(
        "Analytical first bounce time: t = sqrt(2*h0/g) = {:.6} s",
        t_first_bounce
    );
    println!();

    // Storage for the full trajectory
    let mut all_t: Vec<f64> = Vec::new();
    let mut all_h: Vec<f64> = Vec::new();
    let mut all_v: Vec<f64> = Vec::new();

    // Bounce tracking
    let mut bounce_times: Vec<f64> = Vec::new();
    let mut bounce_velocities: Vec<f64> = Vec::new();

    // Current state
    let mut t_current = 0.0;
    let mut height = h0;
    let mut velocity = v0;
    let mut bounce_count = 0;

    println!("Simulating bounces...\n");
    println!(
        "  {:>6} | {:>10} | {:>12} | {:>12}",
        "Bounce", "Time (s)", "Impact v (m/s)", "Rebound v (m/s)"
    );
    println!("  {:-<6}|{:-<11}|{:-<13}|{:-<13}", "", "", "", "");

    while t_current < t_end && bounce_count < max_bounces {
        // Create ODE problem for this segment
        let problem = OdeProblem::new(
            |_t, y: &[f64], dydt: &mut [f64]| {
                dydt[0] = y[1]; // dh/dt = v
                dydt[1] = -g; // dv/dt = -g
            },
            t_current,
            t_end,
            vec![height, velocity],
        );

        let y0 = vec![height, velocity];

        let options = SolverOptions::default()
            .rtol(1e-10)
            .atol(1e-12)
            .event(Box::new(GroundContact));

        let result =
            DoPri5::solve(&problem, t_current, t_end, &y0, &options).expect("Solver failed");

        // Store trajectory points from this segment
        for (t, state) in result.iter() {
            all_t.push(t);
            all_h.push(state[0]);
            all_v.push(state[1]);
        }

        if result.terminated_by_event && !result.events.is_empty() {
            let ev = &result.events[0];
            bounce_count += 1;

            let impact_velocity = ev.y[1];
            let rebound_velocity = -e * impact_velocity;

            println!(
                "  {:>6} | {:>10.6} | {:>12.6} | {:>12.6}",
                bounce_count, ev.t, impact_velocity, rebound_velocity
            );

            bounce_times.push(ev.t);
            bounce_velocities.push(impact_velocity);

            // Update state for next segment
            t_current = ev.t;
            height = 0.0; // on the ground
            velocity = rebound_velocity;

            // Stop if rebound velocity is very small (ball has essentially stopped)
            if rebound_velocity.abs() < 0.01 {
                println!("\n  Ball has essentially stopped (rebound v < 0.01 m/s).");
                break;
            }
        } else {
            // No event: integration reached t_end
            t_current = t_end;
        }
    }

    println!();

    // Summary statistics
    println!("=== Summary ===\n");
    println!("  Total bounces detected: {}", bounce_count);
    println!("  Total trajectory points: {}", all_t.len());

    if let Some(&last_bounce_t) = bounce_times.last() {
        println!("  Time of last bounce: {:.6} s", last_bounce_t);
    }
    println!();

    // Verify first bounce against analytical solution
    if !bounce_times.is_empty() {
        let numerical_first = bounce_times[0];
        let error = (numerical_first - t_first_bounce).abs();
        println!("Verification (first bounce):");
        println!("  Analytical: t = {:.10} s", t_first_bounce);
        println!("  Numerical:  t = {:.10} s", numerical_first);
        println!("  Error:      {:.2e} s", error);
        println!();
    }

    // Energy analysis
    println!("Energy analysis:");
    println!(
        "  {:>6} | {:>12} | {:>12} | {:>8}",
        "Bounce", "Max height (m)", "KE at impact", "KE ratio"
    );
    println!("  {:-<6}|{:-<13}|{:-<13}|{:-<9}", "", "", "", "");

    let initial_ke = 0.5 * (2.0 * g * h0); // KE at first impact = mgh0
    for (i, &v_impact) in bounce_velocities.iter().enumerate() {
        let ke = 0.5 * v_impact * v_impact;
        let max_h = v_impact * v_impact / (2.0 * g); // height before this bounce
        let ke_ratio = ke / initial_ke;
        println!(
            "  {:>6} | {:>12.6} | {:>12.6} | {:>8.4}",
            i + 1,
            max_h,
            ke,
            ke_ratio
        );
    }
    println!();

    // Theoretical energy: at bounce n, the impact KE ratio is e^(2*(n-1))
    // because the first bounce has full energy, and each bounce loses factor e^2.
    println!("Energy loss verification:");
    println!("  After {} bounces:", bounce_count);
    let theoretical_ratio = e.powi(2 * (bounce_count - 1));
    let actual_ratio = if !bounce_velocities.is_empty() {
        let last_ke = 0.5 * bounce_velocities.last().unwrap().powi(2);
        last_ke / initial_ke
    } else {
        1.0
    };
    println!(
        "  Theoretical impact KE ratio at last bounce (e^{{2(n-1)}}): {:.8}",
        theoretical_ratio
    );
    println!(
        "  Numerical KE ratio:                                     {:.8}",
        actual_ratio
    );
    println!(
        "  Match: {}",
        if (theoretical_ratio - actual_ratio).abs() < 1e-4 {
            "YES"
        } else {
            "NO"
        }
    );
    println!();

    // Print a few sample trajectory points
    println!("Sample trajectory:");
    println!(
        "  {:>10} | {:>12} | {:>12}",
        "t (s)", "height (m)", "velocity (m/s)"
    );
    println!("  {:-<10}|{:-<13}|{:-<13}", "", "", "");

    let n_samples = 20;
    let step = (all_t.len() / n_samples).max(1);
    for i in (0..all_t.len()).step_by(step) {
        println!(
            "  {:>10.4} | {:>12.6} | {:>12.6}",
            all_t[i], all_h[i], all_v[i]
        );
    }
    // Always print the last point
    if let Some(last_idx) = all_t.len().checked_sub(1) {
        if last_idx % step != 0 {
            println!(
                "  {:>10.4} | {:>12.6} | {:>12.6}",
                all_t[last_idx], all_h[last_idx], all_v[last_idx]
            );
        }
    }

    println!();
    println!("=== Done ===");
}

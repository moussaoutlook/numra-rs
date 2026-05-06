//! Standalone perf harness for `numra-ode::uncertainty::solve_trajectory`.
//!
//! Run with:
//!   cargo test --release -p numra-ode --test perf_solve_trajectory \
//!     -- --ignored --nocapture
//!
//! Prints median of N reps wall-clock; not a CI gate. Used during the
//! commit-4 port to measure baseline-vs-after delta.

use std::time::Instant;

use numra_ode::uncertainty::{solve_with_uncertainty, UncertainParam, UncertaintyMode};
use numra_ode::{DoPri5, SolverOptions};

#[test]
#[ignore]
fn perf_lv_trajectory() {
    let params = vec![
        UncertainParam::new("alpha", 1.0_f64, 0.1),
        UncertainParam::new("beta", 0.1, 0.01),
        UncertainParam::new("delta", 0.075, 0.005),
        UncertainParam::new("gamma", 1.5, 0.1),
    ];
    let model = |_t: f64, y: &[f64], dy: &mut [f64], p: &[f64]| {
        let x = y[0];
        let yy = y[1];
        dy[0] = p[0] * x - p[1] * x * yy;
        dy[1] = p[2] * x * yy - p[3] * yy;
    };
    let opts = SolverOptions::default().rtol(1e-9).atol(1e-12);

    // Warm up.
    for _ in 0..3 {
        let _ = solve_with_uncertainty::<DoPri5, f64, _>(
            model,
            &[10.0, 5.0],
            0.0,
            5.0,
            &params,
            &UncertaintyMode::Trajectory,
            &opts,
            None,
        )
        .expect("warmup failed");
    }

    let reps = 401;
    let mut samples_ns: Vec<u128> = Vec::with_capacity(reps);
    for _ in 0..reps {
        let t0 = Instant::now();
        let r = solve_with_uncertainty::<DoPri5, f64, _>(
            model,
            &[10.0, 5.0],
            0.0,
            5.0,
            &params,
            &UncertaintyMode::Trajectory,
            &opts,
            None,
        )
        .expect("solve failed");
        let elapsed_ns = t0.elapsed().as_nanos();
        samples_ns.push(elapsed_ns);
        // Touch the result so optimisations can't dead-code-eliminate it.
        std::hint::black_box(&r);
    }
    samples_ns.sort_unstable();

    let median_ns = samples_ns[reps / 2];
    let mean_ns: u128 = samples_ns.iter().sum::<u128>() / (reps as u128);
    let min_ns = samples_ns[0];
    let max_ns = samples_ns[reps - 1];

    eprintln!("\n  solve_trajectory  (LV, 4 params, rtol=1e-9, n_reps={reps})");
    eprintln!("  ------------------------------------------------------------");
    eprintln!("  median   = {:.3} ms", median_ns as f64 / 1e6);
    eprintln!("  mean     = {:.3} ms", mean_ns as f64 / 1e6);
    eprintln!(
        "  min/max  = {:.3} / {:.3} ms",
        min_ns as f64 / 1e6,
        max_ns as f64 / 1e6
    );
}

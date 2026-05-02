//! Method of Steps DDE solver.
//!
//! Solves DDEs by treating them as ODEs with interpolated history.
//!
//! Author: Moussa Leblouba
//! Date: 4 February 2026
//! Modified: 2 May 2026

use crate::history::{History, HistoryStep};
use crate::system::{DdeOptions, DdeResult, DdeSolver, DdeStats, DdeSystem};
use numra_core::Scalar;

/// Method of Steps DDE solver.
///
/// Uses an embedded Runge-Kutta method (similar to DoPri5) with
/// history interpolation for the delayed terms.
pub struct MethodOfSteps;

/// Propagate discontinuities from the initial time.
///
/// Discontinuities in DDEs propagate: if there's a discontinuity at t_d,
/// then derivative discontinuity occurs at t_d + tau for each delay tau.
/// This function computes all discontinuity points up to a given order.
/// Maximum number of discontinuity points to track.
/// Prevents combinatorial explosion when there are many delays and high order.
const MAX_DISCONTINUITIES: usize = 1000;

fn propagate_discontinuities<S: Scalar>(t0: S, delays: &[S], tf: S, order: usize) -> Vec<S> {
    let mut discs = vec![t0]; // Initial discontinuity at t0

    for _ in 0..order {
        let mut new_discs = Vec::new();
        for &d in &discs {
            for &tau in delays {
                let t_new = d + tau;
                if t_new <= tf && t_new > t0 {
                    new_discs.push(t_new);
                }
            }
        }
        discs.extend(new_discs);

        // Cap total discontinuities to prevent combinatorial explosion
        if discs.len() > MAX_DISCONTINUITIES {
            break;
        }
    }

    // Sort and remove duplicates
    discs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    discs.dedup_by(|a, b| (*a - *b).abs() < S::from_f64(1e-14));

    // Filter to only return points > t0 and <= tf, capped
    let mut result: Vec<S> = discs.into_iter().filter(|&d| d > t0 && d <= tf).collect();
    result.truncate(MAX_DISCONTINUITIES);
    result
}

impl<S: Scalar> DdeSolver<S> for MethodOfSteps {
    fn solve<Sys, H>(
        system: &Sys,
        t0: S,
        tf: S,
        history: &H,
        options: &DdeOptions<S>,
    ) -> Result<DdeResult<S>, String>
    where
        Sys: DdeSystem<S>,
        H: Fn(S) -> Vec<S>,
    {
        let dim = system.dim();
        let n_delays = system.n_delays();

        // Initialize history
        let mut hist = History::new(history, t0, dim);

        // Get initial state from history at t0
        let y0 = history(t0);
        if y0.len() != dim {
            return Err(format!(
                "History function returned dimension {} but system has dimension {}",
                y0.len(),
                dim
            ));
        }

        // Initialize state
        let mut t = t0;
        let mut y = y0.clone();

        // Initial step size
        let mut h = options.h0.unwrap_or_else(|| {
            let span = tf - t0;
            (span * S::from_f64(0.001)).min(S::from_f64(0.1))
        });

        // Working arrays
        let mut f = vec![S::ZERO; dim];
        let mut y_delayed: Vec<Vec<S>> = vec![vec![S::ZERO; dim]; n_delays];
        let has_state_dependent_delays = system.has_state_dependent_delays();
        let constant_delays = system.delays();

        // Helper to get delays at a given state, with positivity validation
        let get_delays = |sys: &Sys, t: S, y: &[S]| -> Result<Vec<S>, String> {
            let delays = if has_state_dependent_delays {
                sys.delays_at(t, y)
            } else {
                constant_delays.clone()
            };
            for (i, &tau) in delays.iter().enumerate() {
                if tau < S::ZERO {
                    return Err(format!(
                        "Delay {} is negative ({}) at t = {}. Delays must be non-negative.",
                        i,
                        tau.to_f64(),
                        t.to_f64()
                    ));
                }
            }
            Ok(delays)
        };

        // Evaluate initial derivative
        let delays = get_delays(system, t, &y)?;
        for (i, &tau) in delays.iter().enumerate() {
            y_delayed[i] = hist.evaluate(t - tau);
        }
        let y_delayed_refs: Vec<&[S]> = y_delayed.iter().map(|v| v.as_slice()).collect();
        system.rhs(t, &y, &y_delayed_refs, &mut f);

        // Output storage
        let mut t_out = vec![t];
        let mut y_out = y.clone();
        let mut stats = DdeStats::default();
        stats.n_eval += 1;

        // Dormand-Prince 5(4) coefficients
        let a21 = S::from_f64(1.0 / 5.0);
        let a31 = S::from_f64(3.0 / 40.0);
        let a32 = S::from_f64(9.0 / 40.0);
        let a41 = S::from_f64(44.0 / 45.0);
        let a42 = S::from_f64(-56.0 / 15.0);
        let a43 = S::from_f64(32.0 / 9.0);
        let a51 = S::from_f64(19372.0 / 6561.0);
        let a52 = S::from_f64(-25360.0 / 2187.0);
        let a53 = S::from_f64(64448.0 / 6561.0);
        let a54 = S::from_f64(-212.0 / 729.0);
        let a61 = S::from_f64(9017.0 / 3168.0);
        let a62 = S::from_f64(-355.0 / 33.0);
        let a63 = S::from_f64(46732.0 / 5247.0);
        let a64 = S::from_f64(49.0 / 176.0);
        let a65 = S::from_f64(-5103.0 / 18656.0);
        let a71 = S::from_f64(35.0 / 384.0);
        let a73 = S::from_f64(500.0 / 1113.0);
        let a74 = S::from_f64(125.0 / 192.0);
        let a75 = S::from_f64(-2187.0 / 6784.0);
        let a76 = S::from_f64(11.0 / 84.0);

        let c2 = S::from_f64(1.0 / 5.0);
        let c3 = S::from_f64(3.0 / 10.0);
        let c4 = S::from_f64(4.0 / 5.0);
        let c5 = S::from_f64(8.0 / 9.0);
        let c6 = S::ONE;
        let c7 = S::ONE;

        // Error estimation coefficients
        let e1 = S::from_f64(71.0 / 57600.0);
        let e3 = S::from_f64(-71.0 / 16695.0);
        let e4 = S::from_f64(71.0 / 1920.0);
        let e5 = S::from_f64(-17253.0 / 339200.0);
        let e6 = S::from_f64(22.0 / 525.0);
        let e7 = S::from_f64(-1.0 / 40.0);

        // Stage values
        let mut k1 = f.clone();
        let mut k2 = vec![S::ZERO; dim];
        let mut k3 = vec![S::ZERO; dim];
        let mut k4 = vec![S::ZERO; dim];
        let mut k5 = vec![S::ZERO; dim];
        let mut k6 = vec![S::ZERO; dim];
        let mut k7 = vec![S::ZERO; dim];
        let mut y_stage = vec![S::ZERO; dim];
        let mut y_new = vec![S::ZERO; dim];
        let mut y_err = vec![S::ZERO; dim];

        // Step size control parameters
        let safety = S::from_f64(0.9);
        let fac_min = S::from_f64(0.2);
        let fac_max = S::from_f64(10.0);
        let order = S::from_f64(5.0);

        // Compute discontinuity points if tracking is enabled
        let discontinuities = if options.track_discontinuities && options.discontinuity_order > 0 {
            propagate_discontinuities(t0, &delays, tf, options.discontinuity_order)
        } else {
            Vec::new()
        };
        let mut disc_idx = 0; // Index into discontinuities vector

        let mut step = 0;
        while t < tf && step < options.max_steps {
            // Limit step size
            h = h.min(tf - t).min(options.h_max).max(options.h_min);

            // Check if we would cross a discontinuity
            if options.track_discontinuities && disc_idx < discontinuities.len() {
                let next_disc = discontinuities[disc_idx];
                let t_end = t + h;
                // If we would cross or overshoot the discontinuity, reduce step to land exactly on it
                if next_disc > t && next_disc <= t_end {
                    let h_to_disc = next_disc - t;
                    if h_to_disc < options.h_min {
                        // h_min is too large and would skip this discontinuity
                        return Err(format!(
                            "h_min ({}) is larger than distance to discontinuity at t = {} (distance = {}). \
                             Reduce h_min or disable discontinuity tracking.",
                            options.h_min.to_f64(), next_disc.to_f64(), h_to_disc.to_f64()
                        ));
                    }
                    h = h_to_disc;
                }
            }

            // k1 is already computed (FSAL)

            // k2
            let t2 = t + c2 * h;
            for j in 0..dim {
                y_stage[j] = y[j] + h * a21 * k1[j];
            }
            let delays_k2 = get_delays(system, t2, &y_stage)?;
            for (i, &tau) in delays_k2.iter().enumerate() {
                y_delayed[i] = hist.evaluate(t2 - tau);
            }
            let y_delayed_refs: Vec<&[S]> = y_delayed.iter().map(|v| v.as_slice()).collect();
            system.rhs(t2, &y_stage, &y_delayed_refs, &mut k2);

            // k3
            let t3 = t + c3 * h;
            for j in 0..dim {
                y_stage[j] = y[j] + h * (a31 * k1[j] + a32 * k2[j]);
            }
            let delays_k3 = get_delays(system, t3, &y_stage)?;
            for (i, &tau) in delays_k3.iter().enumerate() {
                y_delayed[i] = hist.evaluate(t3 - tau);
            }
            let y_delayed_refs: Vec<&[S]> = y_delayed.iter().map(|v| v.as_slice()).collect();
            system.rhs(t3, &y_stage, &y_delayed_refs, &mut k3);

            // k4
            let t4 = t + c4 * h;
            for j in 0..dim {
                y_stage[j] = y[j] + h * (a41 * k1[j] + a42 * k2[j] + a43 * k3[j]);
            }
            let delays_k4 = get_delays(system, t4, &y_stage)?;
            for (i, &tau) in delays_k4.iter().enumerate() {
                y_delayed[i] = hist.evaluate(t4 - tau);
            }
            let y_delayed_refs: Vec<&[S]> = y_delayed.iter().map(|v| v.as_slice()).collect();
            system.rhs(t4, &y_stage, &y_delayed_refs, &mut k4);

            // k5
            let t5 = t + c5 * h;
            for j in 0..dim {
                y_stage[j] = y[j] + h * (a51 * k1[j] + a52 * k2[j] + a53 * k3[j] + a54 * k4[j]);
            }
            let delays_k5 = get_delays(system, t5, &y_stage)?;
            for (i, &tau) in delays_k5.iter().enumerate() {
                y_delayed[i] = hist.evaluate(t5 - tau);
            }
            let y_delayed_refs: Vec<&[S]> = y_delayed.iter().map(|v| v.as_slice()).collect();
            system.rhs(t5, &y_stage, &y_delayed_refs, &mut k5);

            // k6
            let t6 = t + c6 * h;
            for j in 0..dim {
                y_stage[j] = y[j]
                    + h * (a61 * k1[j] + a62 * k2[j] + a63 * k3[j] + a64 * k4[j] + a65 * k5[j]);
            }
            let delays_k6 = get_delays(system, t6, &y_stage)?;
            for (i, &tau) in delays_k6.iter().enumerate() {
                y_delayed[i] = hist.evaluate(t6 - tau);
            }
            let y_delayed_refs: Vec<&[S]> = y_delayed.iter().map(|v| v.as_slice()).collect();
            system.rhs(t6, &y_stage, &y_delayed_refs, &mut k6);

            // Compute 5th order solution
            for j in 0..dim {
                y_new[j] = y[j]
                    + h * (a71 * k1[j] + a73 * k3[j] + a74 * k4[j] + a75 * k5[j] + a76 * k6[j]);
            }

            // k7 (at new point, for FSAL and error)
            let t7 = t + c7 * h;
            let delays_k7 = get_delays(system, t7, &y_new)?;
            for (i, &tau) in delays_k7.iter().enumerate() {
                y_delayed[i] = hist.evaluate(t7 - tau);
            }
            let y_delayed_refs: Vec<&[S]> = y_delayed.iter().map(|v| v.as_slice()).collect();
            system.rhs(t7, &y_new, &y_delayed_refs, &mut k7);

            stats.n_eval += 6;

            // Error estimate
            for j in 0..dim {
                y_err[j] = h
                    * (e1 * k1[j] + e3 * k3[j] + e4 * k4[j] + e5 * k5[j] + e6 * k6[j] + e7 * k7[j]);
            }

            // Compute error norm
            let mut err_sq = S::ZERO;
            for j in 0..dim {
                let scale = options.atol + options.rtol * y[j].abs().max(y_new[j].abs());
                let ratio = y_err[j] / scale;
                err_sq = err_sq + ratio * ratio;
            }
            let err = (err_sq / S::from_usize(dim)).sqrt();

            // Accept or reject
            if err <= S::ONE {
                // Accept step
                let t_new = t + h;

                // Add step to history
                hist.add_step(HistoryStep::new(
                    t,
                    y.clone(),
                    k1.clone(),
                    t_new,
                    y_new.clone(),
                    k7.clone(),
                ));

                t = t_new;
                y.clone_from(&y_new);
                k1.clone_from(&k7); // FSAL

                // Save output
                t_out.push(t);
                y_out.extend_from_slice(&y);
                stats.n_accept += 1;
                step += 1;

                // Check if we hit a discontinuity point
                if options.track_discontinuities && disc_idx < discontinuities.len() {
                    let next_disc = discontinuities[disc_idx];
                    if (t - next_disc).abs() < S::from_f64(1e-10) {
                        // We've hit this discontinuity, move to the next one
                        stats.n_discontinuities += 1;
                        disc_idx += 1;
                    }
                }
            } else {
                stats.n_reject += 1;
            }

            // Compute new step size
            let err_safe = err.max(S::from_f64(1e-10));
            let fac = safety * err_safe.powf(-S::ONE / (order + S::ONE));
            h = h * fac.max(fac_min).min(fac_max);
        }

        if step >= options.max_steps && t < tf {
            return Err(format!(
                "Maximum steps ({}) exceeded at t = {}",
                options.max_steps,
                t.to_f64()
            ));
        }

        Ok(DdeResult::new(t_out, y_out, dim, stats))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DdeSystem;

    /// Simple test DDE: y'(t) = -y(t-1), y(t) = 1 for t <= 0
    struct SimpleDelay;

    impl DdeSystem<f64> for SimpleDelay {
        fn dim(&self) -> usize {
            1
        }
        fn delays(&self) -> Vec<f64> {
            vec![1.0]
        }
        fn rhs(&self, _t: f64, _y: &[f64], y_delayed: &[&[f64]], dydt: &mut [f64]) {
            dydt[0] = -y_delayed[0][0];
        }
    }

    #[test]
    fn test_simple_delay() {
        let system = SimpleDelay;
        let history = |_t: f64| vec![1.0];
        let options = DdeOptions::default().rtol(1e-6).atol(1e-9);

        let result =
            MethodOfSteps::solve(&system, 0.0, 5.0, &history, &options).expect("Solve failed");

        assert!(result.success);
        assert!(!result.t.is_empty());

        // The solution should decay (oscillatory decay for this equation)
        let y_final = result.y_final().unwrap()[0];
        assert!(y_final.abs() < 2.0); // Should be bounded
    }

    /// Mackey-Glass equation: y'(t) = β * y(t-τ) / (1 + y(t-τ)^n) - γ * y(t)
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
            let y_tau = y_delayed[0][0];
            dydt[0] = self.beta * y_tau / (1.0 + y_tau.powf(self.n)) - self.gamma * y[0];
        }
    }

    #[test]
    fn test_mackey_glass() {
        let system = MackeyGlass {
            beta: 2.0,
            gamma: 1.0,
            n: 9.65,
            tau: 2.0,
        };
        let history = |_t: f64| vec![0.5];
        let options = DdeOptions::default().rtol(1e-4).atol(1e-6).max_steps(50000);

        let result =
            MethodOfSteps::solve(&system, 0.0, 50.0, &history, &options).expect("Solve failed");

        assert!(result.success);
        // Mackey-Glass with these parameters is chaotic
        // Just verify the solution stays bounded and positive
        for &y in result.y.iter() {
            assert!(y > 0.0, "Solution should stay positive");
            assert!(y < 3.0, "Solution should stay bounded");
        }
    }

    #[test]
    fn test_two_delays() {
        struct TwoDelays;
        impl DdeSystem<f64> for TwoDelays {
            fn dim(&self) -> usize {
                1
            }
            fn delays(&self) -> Vec<f64> {
                vec![0.5, 1.0]
            }
            fn rhs(&self, _t: f64, y: &[f64], y_delayed: &[&[f64]], dydt: &mut [f64]) {
                dydt[0] = -y[0] + 0.5 * y_delayed[0][0] + 0.3 * y_delayed[1][0];
            }
        }

        let system = TwoDelays;
        let history = |_t: f64| vec![1.0];
        let options = DdeOptions::default();

        let result =
            MethodOfSteps::solve(&system, 0.0, 10.0, &history, &options).expect("Solve failed");

        assert!(result.success);
        assert!(result.stats.n_accept > 0);
    }

    #[test]
    fn test_2d_system() {
        struct TwoD;
        impl DdeSystem<f64> for TwoD {
            fn dim(&self) -> usize {
                2
            }
            fn delays(&self) -> Vec<f64> {
                vec![1.0]
            }
            fn rhs(&self, _t: f64, y: &[f64], y_delayed: &[&[f64]], dydt: &mut [f64]) {
                dydt[0] = -y[0] + y_delayed[0][1];
                dydt[1] = -y[1] + y_delayed[0][0];
            }
        }

        let system = TwoD;
        let history = |_t: f64| vec![1.0, 0.5];
        let options = DdeOptions::default();

        let result =
            MethodOfSteps::solve(&system, 0.0, 5.0, &history, &options).expect("Solve failed");

        assert!(result.success);
        assert_eq!(result.dim, 2);
    }

    #[test]
    fn test_dde_discontinuity_tracking() {
        // DDE with discontinuous history
        // y'(t) = -y(t-1) for t > 0
        // y(t) = 1 for t <= 0 (constant history)
        //
        // The solution has derivative discontinuity at t=1, t=2, ...
        // Solver should hit these points exactly

        struct SimpleDelaySystem;

        impl DdeSystem<f64> for SimpleDelaySystem {
            fn dim(&self) -> usize {
                1
            }
            fn delays(&self) -> Vec<f64> {
                vec![1.0]
            }
            fn rhs(&self, _t: f64, _y: &[f64], y_delayed: &[&[f64]], dydt: &mut [f64]) {
                dydt[0] = -y_delayed[0][0];
            }
        }

        let system = SimpleDelaySystem;
        let history = |_t: f64| vec![1.0];
        let options = DdeOptions::default()
            .track_discontinuities(true)
            .discontinuity_order(3);

        let result =
            MethodOfSteps::solve(&system, 0.0, 3.5, &history, &options).expect("Solve failed");

        // Check that t=1.0, t=2.0, t=3.0 are in the output times
        let has_t1 = result.t.iter().any(|&t| (t - 1.0).abs() < 1e-10);
        let has_t2 = result.t.iter().any(|&t| (t - 2.0).abs() < 1e-10);
        let has_t3 = result.t.iter().any(|&t| (t - 3.0).abs() < 1e-10);

        assert!(has_t1, "Discontinuity at t=1 not tracked");
        assert!(has_t2, "Discontinuity at t=2 not tracked");
        assert!(has_t3, "Discontinuity at t=3 not tracked");

        // Verify that discontinuity count is correct
        assert_eq!(
            result.stats.n_discontinuities, 3,
            "Should have tracked 3 discontinuities"
        );
    }

    #[test]
    fn test_propagate_discontinuities() {
        // Test the discontinuity propagation function directly
        let delays = vec![1.0];
        let discs = propagate_discontinuities(0.0, &delays, 5.0, 3);

        // Should have discontinuities at 1, 2, 3
        assert_eq!(discs.len(), 3);
        assert!((discs[0] - 1.0).abs() < 1e-10);
        assert!((discs[1] - 2.0).abs() < 1e-10);
        assert!((discs[2] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_propagate_discontinuities_multiple_delays() {
        // With two delays, discontinuities propagate differently
        let delays = vec![0.5, 1.0];
        let discs = propagate_discontinuities(0.0, &delays, 3.0, 2);

        // Should have discontinuities at 0.5, 1.0, 1.5, 2.0
        // (0.5+0.5=1.0 is duplicate, 0.5+1.0=1.5, 1.0+0.5=1.5 is duplicate, 1.0+1.0=2.0)
        assert!(
            discs.iter().any(|&d| (d - 0.5).abs() < 1e-10),
            "Should have 0.5"
        );
        assert!(
            discs.iter().any(|&d| (d - 1.0).abs() < 1e-10),
            "Should have 1.0"
        );
        assert!(
            discs.iter().any(|&d| (d - 1.5).abs() < 1e-10),
            "Should have 1.5"
        );
        assert!(
            discs.iter().any(|&d| (d - 2.0).abs() < 1e-10),
            "Should have 2.0"
        );
    }

    #[test]
    fn test_state_dependent_delay() {
        // y'(t) = -y(t - tau(y))
        // where tau(y) = 0.5 + 0.1 * y
        //
        // This is a simple state-dependent delay where the delay
        // increases with y. For y(0) = 1, initial delay is 0.6.

        struct StateDependentSystem;

        impl DdeSystem<f64> for StateDependentSystem {
            fn dim(&self) -> usize {
                1
            }

            fn delays(&self) -> Vec<f64> {
                // Return a nominal delay for n_delays() to work
                vec![0.5]
            }

            fn delays_at(&self, _t: f64, y: &[f64]) -> Vec<f64> {
                vec![0.5 + 0.1 * y[0]]
            }

            fn has_state_dependent_delays(&self) -> bool {
                true
            }

            fn rhs(&self, _t: f64, _y: &[f64], y_delayed: &[&[f64]], dydt: &mut [f64]) {
                dydt[0] = -y_delayed[0][0];
            }
        }

        let system = StateDependentSystem;
        let history = |_t: f64| vec![1.0];
        let options = DdeOptions::default().rtol(1e-6).atol(1e-8);

        let result = MethodOfSteps::solve(&system, 0.0, 2.0, &history, &options);

        assert!(
            result.is_ok(),
            "State-dependent delay solve failed: {:?}",
            result.err()
        );
        let sol = result.unwrap();

        // The solution should decay (since RHS is negative of delayed value)
        // and remain bounded
        let y_final = sol.y_final().unwrap()[0];
        assert!(y_final < 1.0, "Solution should decay from initial y=1");
        assert!(y_final > -10.0, "Solution should remain bounded");

        // Verify the delay varies during integration
        // At t=0, y=1, delay=0.6
        // As y decreases, delay decreases toward 0.5
    }

    #[test]
    fn test_state_dependent_delay_vs_constant() {
        // Compare state-dependent delay vs constant delay to verify
        // the state-dependent delay is actually being used
        //
        // y'(t) = -y(t - tau)
        // For state-dependent: tau(y) = 0.2 + 0.5 * y^2
        // For constant: tau = 0.5
        //
        // With y(0) = 1, initial state-dependent delay is 0.7
        // As solution decays, delay decreases toward 0.2

        struct StateDependentDelay;
        impl DdeSystem<f64> for StateDependentDelay {
            fn dim(&self) -> usize {
                1
            }
            fn delays(&self) -> Vec<f64> {
                vec![0.5]
            }
            fn delays_at(&self, _t: f64, y: &[f64]) -> Vec<f64> {
                vec![0.2 + 0.5 * y[0] * y[0]]
            }
            fn has_state_dependent_delays(&self) -> bool {
                true
            }
            fn rhs(&self, _t: f64, _y: &[f64], y_delayed: &[&[f64]], dydt: &mut [f64]) {
                dydt[0] = -y_delayed[0][0];
            }
        }

        struct ConstantDelay;
        impl DdeSystem<f64> for ConstantDelay {
            fn dim(&self) -> usize {
                1
            }
            fn delays(&self) -> Vec<f64> {
                vec![0.5]
            }
            fn rhs(&self, _t: f64, _y: &[f64], y_delayed: &[&[f64]], dydt: &mut [f64]) {
                dydt[0] = -y_delayed[0][0];
            }
        }

        let history = |_t: f64| vec![1.0];
        let options = DdeOptions::default().rtol(1e-8).atol(1e-10);

        let result_sd = MethodOfSteps::solve(&StateDependentDelay, 0.0, 3.0, &history, &options)
            .expect("State-dependent solve failed");
        let result_const = MethodOfSteps::solve(&ConstantDelay, 0.0, 3.0, &history, &options)
            .expect("Constant delay solve failed");

        // Solutions should be different because delays are different
        let y_sd = result_sd.y_final().unwrap()[0];
        let y_const = result_const.y_final().unwrap()[0];

        // They should differ by a meaningful amount (not just floating point noise)
        let diff = (y_sd - y_const).abs();
        assert!(diff > 0.01,
            "State-dependent and constant delay solutions should differ significantly, but diff = {}",
            diff);
    }

    #[test]
    fn test_dde_negative_delay_rejected() {
        // State-dependent delay that returns a negative value should be rejected
        struct NegativeDelaySystem;

        impl DdeSystem<f64> for NegativeDelaySystem {
            fn dim(&self) -> usize {
                1
            }
            fn delays(&self) -> Vec<f64> {
                vec![1.0]
            }
            fn delays_at(&self, _t: f64, _y: &[f64]) -> Vec<f64> {
                vec![-0.5] // Negative delay - invalid
            }
            fn has_state_dependent_delays(&self) -> bool {
                true
            }
            fn rhs(&self, _t: f64, _y: &[f64], y_delayed: &[&[f64]], dydt: &mut [f64]) {
                dydt[0] = -y_delayed[0][0];
            }
        }

        let system = NegativeDelaySystem;
        let history = |_t: f64| vec![1.0];
        let options = DdeOptions::default();

        let result = MethodOfSteps::solve(&system, 0.0, 1.0, &history, &options);
        assert!(result.is_err(), "Should reject negative delay");
        assert!(
            result.unwrap_err().contains("negative"),
            "Error should mention negative delay"
        );
    }

    #[test]
    fn test_dde_discontinuity_hmin_too_large() {
        // When h_min is larger than the distance to a discontinuity,
        // the solver should return an error instead of skipping it
        struct SimpleDelaySystem;

        impl DdeSystem<f64> for SimpleDelaySystem {
            fn dim(&self) -> usize {
                1
            }
            fn delays(&self) -> Vec<f64> {
                vec![0.001]
            } // Very small delay
            fn rhs(&self, _t: f64, _y: &[f64], y_delayed: &[&[f64]], dydt: &mut [f64]) {
                dydt[0] = -y_delayed[0][0];
            }
        }

        let system = SimpleDelaySystem;
        let history = |_t: f64| vec![1.0];
        // h_min is larger than the delay, so we can't step to the discontinuity
        let options = DdeOptions::default()
            .track_discontinuities(true)
            .discontinuity_order(1)
            .h_max(0.1);

        // Set h_min very large
        let mut opts = options;
        opts.h_min = 0.01; // Larger than delay of 0.001

        let result = MethodOfSteps::solve(&system, 0.0, 0.1, &history, &opts);
        assert!(
            result.is_err(),
            "Should error when h_min would skip discontinuity"
        );
        assert!(
            result.unwrap_err().contains("h_min"),
            "Error should mention h_min"
        );
    }

    #[test]
    fn test_dde_discontinuity_cap() {
        // Many delays with high order should be capped
        let delays = vec![0.01, 0.02, 0.03, 0.04, 0.05];
        let discs = propagate_discontinuities(0.0, &delays, 100.0, 10);

        // Should be capped at MAX_DISCONTINUITIES
        assert!(
            discs.len() <= MAX_DISCONTINUITIES,
            "Discontinuities should be capped, got {}",
            discs.len()
        );
    }

    #[test]
    fn test_dde_solution_accuracy() {
        // y'(t) = -y(t-1), y(t) = 1 for t <= 0
        // Exact solution for 0 <= t <= 1: y(t) = 1 - t
        // Exact solution for 1 <= t <= 2: y(t) = 1 - t + (t-1)^2/2
        let system = SimpleDelay;
        let history = |_t: f64| vec![1.0];
        let options = DdeOptions::default()
            .rtol(1e-8)
            .atol(1e-10)
            .track_discontinuities(true)
            .discontinuity_order(3);

        let result =
            MethodOfSteps::solve(&system, 0.0, 2.0, &history, &options).expect("Solve failed");

        // Helper: find the closest time point to target
        let find_nearest = |target: f64| -> usize {
            result
                .t
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| {
                    (a.to_f64() - target)
                        .abs()
                        .partial_cmp(&(b.to_f64() - target).abs())
                        .unwrap()
                })
                .unwrap()
                .0
        };

        // Check at t ~ 0.5: y(0.5) = 1 - 0.5 = 0.5
        let idx_half = find_nearest(0.5);
        let t_half = result.t[idx_half];
        let y_half = result.y_at(idx_half)[0];
        let exact_half = 1.0 - t_half;
        assert!(
            (y_half - exact_half).abs() < 1e-4,
            "At t={}: expected ~{}, got {}",
            t_half,
            exact_half,
            y_half
        );

        // Check at t = 1.0 (discontinuity tracking should land exactly here)
        let idx_one = find_nearest(1.0);
        let y_one = result.y_at(idx_one)[0];
        assert!(y_one.abs() < 1e-5, "At t=1.0: expected ~0, got {}", y_one);

        // Check at t ~ 1.5: y(t) = 1 - t + (t-1)^2/2
        let idx_1_5 = find_nearest(1.5);
        let t_1_5 = result.t[idx_1_5];
        let y_1_5 = result.y_at(idx_1_5)[0];
        let exact_1_5 = 1.0 - t_1_5 + (t_1_5 - 1.0) * (t_1_5 - 1.0) / 2.0;
        assert!(
            (y_1_5 - exact_1_5).abs() < 1e-4,
            "At t={}: expected ~{}, got {}",
            t_1_5,
            exact_1_5,
            y_1_5
        );
    }
}

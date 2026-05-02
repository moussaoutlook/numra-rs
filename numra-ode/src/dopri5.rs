//! Dormand-Prince 5(4) explicit Runge-Kutta solver.
//!
//! DoPri5 is one of the most widely used adaptive RK methods for non-stiff ODEs.
//! It uses a 5th order method for advancing the solution and a 4th order embedded
//! method for error estimation.
//!
//! ## Features
//! - 5th order accuracy with embedded 4th order error estimator
//! - 7 stages with FSAL (First Same As Last) property
//! - Free 4th order interpolant for dense output
//! - PI step size control for smooth adaptation
//!
//! ## Reference
//! Dormand, J. R.; Prince, P. J. (1980), "A family of embedded Runge-Kutta formulae",
//! Journal of Computational and Applied Mathematics, 6 (1): 19–26
//!
//! Author: Moussa Leblouba
//! Date: 5 March 2026
//! Modified: 2 May 2026

use crate::dense::{DenseOutput, DenseSegment, DoPri5Interpolant};
use crate::error::SolverError;
use crate::events::{find_event_time, Event, EventAction};
use crate::problem::OdeSystem;
use crate::solver::{Solver, SolverOptions, SolverResult, SolverStats};
use crate::step_control::{PIController, StepController};
use numra_core::Scalar;

/// Dormand-Prince 5(4) solver.
///
/// # Example
///
/// ```rust
/// use numra_ode::{OdeProblem, DoPri5, Solver, SolverOptions};
///
/// // Exponential decay: dy/dt = -y
/// let problem = OdeProblem::new(
///     |_t, y: &[f64], dydt: &mut [f64]| { dydt[0] = -y[0]; },
///     0.0,
///     5.0,
///     vec![1.0]
/// );
///
/// let options = SolverOptions::default();
/// let result = DoPri5::solve(&problem, 0.0, 5.0, &[1.0], &options).unwrap();
///
/// // y(5) ≈ e^(-5) ≈ 0.00674
/// let y_final = result.y_final().unwrap();
/// assert!((y_final[0] - (-5.0_f64).exp()).abs() < 1e-5);
/// ```
#[derive(Clone, Debug, Default)]
pub struct DoPri5;

impl DoPri5 {
    /// Create a new DoPri5 solver.
    pub fn new() -> Self {
        Self
    }
}

/// Butcher tableau coefficients for DoPri5.
///
/// The tableau is:
/// ```text
/// 0    |
/// 1/5  | 1/5
/// 3/10 | 3/40      9/40
/// 4/5  | 44/45     -56/15    32/9
/// 8/9  | 19372/6561 -25360/2187 64448/6561 -212/729
/// 1    | 9017/3168  -355/33    46732/5247  49/176   -5103/18656
/// 1    | 35/384     0          500/1113    125/192  -2187/6784   11/84
/// -----+-------------------------------------------------------------------
/// b    | 35/384     0          500/1113    125/192  -2187/6784   11/84    0
/// b*   | 5179/57600 0          7571/16695  393/640  -92097/339200 187/2100 1/40
/// ```
#[allow(dead_code)]
mod tableau {
    // Nodes (c coefficients)
    pub const C2: f64 = 1.0 / 5.0;
    pub const C3: f64 = 3.0 / 10.0;
    pub const C4: f64 = 4.0 / 5.0;
    pub const C5: f64 = 8.0 / 9.0;
    pub const C6: f64 = 1.0;
    pub const C7: f64 = 1.0;

    // A matrix coefficients (row by row)
    pub const A21: f64 = 1.0 / 5.0;

    pub const A31: f64 = 3.0 / 40.0;
    pub const A32: f64 = 9.0 / 40.0;

    pub const A41: f64 = 44.0 / 45.0;
    pub const A42: f64 = -56.0 / 15.0;
    pub const A43: f64 = 32.0 / 9.0;

    pub const A51: f64 = 19372.0 / 6561.0;
    pub const A52: f64 = -25360.0 / 2187.0;
    pub const A53: f64 = 64448.0 / 6561.0;
    pub const A54: f64 = -212.0 / 729.0;

    pub const A61: f64 = 9017.0 / 3168.0;
    pub const A62: f64 = -355.0 / 33.0;
    pub const A63: f64 = 46732.0 / 5247.0;
    pub const A64: f64 = 49.0 / 176.0;
    pub const A65: f64 = -5103.0 / 18656.0;

    pub const A71: f64 = 35.0 / 384.0;
    pub const A72: f64 = 0.0;
    pub const A73: f64 = 500.0 / 1113.0;
    pub const A74: f64 = 125.0 / 192.0;
    pub const A75: f64 = -2187.0 / 6784.0;
    pub const A76: f64 = 11.0 / 84.0;

    // 5th order weights (same as A7*)
    pub const B1: f64 = 35.0 / 384.0;
    pub const B2: f64 = 0.0;
    pub const B3: f64 = 500.0 / 1113.0;
    pub const B4: f64 = 125.0 / 192.0;
    pub const B5: f64 = -2187.0 / 6784.0;
    pub const B6: f64 = 11.0 / 84.0;
    pub const B7: f64 = 0.0;

    // 4th order weights (for error estimation)
    pub const B1_HAT: f64 = 5179.0 / 57600.0;
    pub const B2_HAT: f64 = 0.0;
    pub const B3_HAT: f64 = 7571.0 / 16695.0;
    pub const B4_HAT: f64 = 393.0 / 640.0;
    pub const B5_HAT: f64 = -92097.0 / 339200.0;
    pub const B6_HAT: f64 = 187.0 / 2100.0;
    pub const B7_HAT: f64 = 1.0 / 40.0;

    // Error coefficients: e = b - b_hat
    pub const E1: f64 = B1 - B1_HAT;
    pub const E2: f64 = B2 - B2_HAT;
    pub const E3: f64 = B3 - B3_HAT;
    pub const E4: f64 = B4 - B4_HAT;
    pub const E5: f64 = B5 - B5_HAT;
    pub const E6: f64 = B6 - B6_HAT;
    pub const E7: f64 = B7 - B7_HAT;
}

impl<S: Scalar> Solver<S> for DoPri5 {
    fn solve<Sys: OdeSystem<S>>(
        problem: &Sys,
        t0: S,
        tf: S,
        y0: &[S],
        options: &SolverOptions<S>,
    ) -> Result<SolverResult<S>, SolverError> {
        let dim = problem.dim();
        if y0.len() != dim {
            return Err(SolverError::DimensionMismatch {
                expected: dim,
                actual: y0.len(),
            });
        }

        // Direction of integration
        let direction = if tf >= t0 { S::ONE } else { -S::ONE };

        // Initialize step size controller
        let mut controller = PIController::for_order(5);

        // Initialize step size
        let mut h = match options.h0 {
            Some(h0) => direction * h0.abs(),
            None => estimate_initial_step(problem, t0, y0, direction, options),
        };

        // Clamp step size
        h = direction * h.abs().min(options.h_max).max(options.h_min);

        // Working arrays
        let mut t = t0;
        let mut y = y0.to_vec();
        let mut y_new = vec![S::ZERO; dim];

        // Stage vectors (FSAL: k7 of one step = k1 of next)
        let mut k1 = vec![S::ZERO; dim];
        let mut k2 = vec![S::ZERO; dim];
        let mut k3 = vec![S::ZERO; dim];
        let mut k4 = vec![S::ZERO; dim];
        let mut k5 = vec![S::ZERO; dim];
        let mut k6 = vec![S::ZERO; dim];
        let mut k7 = vec![S::ZERO; dim];

        // Temporary vector for stage computation
        let mut y_stage = vec![S::ZERO; dim];

        // Error vector
        let mut err = vec![S::ZERO; dim];

        // Pre-allocated workspace for dense output (avoids allocation per step)
        let mut k_all = if options.dense_output {
            vec![S::ZERO; 7 * dim]
        } else {
            Vec::new()
        };

        // Output storage
        let mut t_out = vec![t0];
        let mut y_out = y0.to_vec();

        // Event tracking
        let has_events = !options.events.is_empty();
        let mut detected_events: Vec<Event<S>> = Vec::new();

        // Previous event function values (for sign-change detection)
        let mut g_prev: Vec<S> = options
            .events
            .iter()
            .map(|ef| ef.evaluate(t0, y0))
            .collect();

        // Statistics
        let mut stats = SolverStats::new();

        // Dense output
        let mut dense = if options.dense_output {
            DenseOutput::new(dim, direction)
        } else {
            DenseOutput::new(0, direction)
        };

        // Compute initial k1
        problem.rhs(t, &y, &mut k1);
        stats.n_eval += 1;

        // Pre-allocated tolerance weights buffer (avoids allocation per step)
        let mut tol_weights = vec![S::ZERO; dim];
        let update_tol_weights = |weights: &mut [S], y: &[S]| {
            for (w, &yi) in weights.iter_mut().zip(y.iter()) {
                *w = options.atol + options.rtol * yi.abs();
            }
        };

        // Main integration loop
        let mut step_count = 0;
        let mut last_step = false;

        while !last_step {
            // Check step limit
            if step_count >= options.max_steps {
                return Err(SolverError::MaxIterationsExceeded { t: t.to_f64() });
            }

            // Adjust final step to hit tf exactly
            if direction * (t + h - tf) > S::ZERO {
                h = tf - t;
                last_step = true;
            }

            // Compute stages
            // k1 is already computed (FSAL)

            // k2
            for i in 0..dim {
                y_stage[i] = y[i] + h * S::from_f64(tableau::A21) * k1[i];
            }
            problem.rhs(t + h * S::from_f64(tableau::C2), &y_stage, &mut k2);

            // k3
            for i in 0..dim {
                y_stage[i] = y[i]
                    + h * (S::from_f64(tableau::A31) * k1[i] + S::from_f64(tableau::A32) * k2[i]);
            }
            problem.rhs(t + h * S::from_f64(tableau::C3), &y_stage, &mut k3);

            // k4
            for i in 0..dim {
                y_stage[i] = y[i]
                    + h * (S::from_f64(tableau::A41) * k1[i]
                        + S::from_f64(tableau::A42) * k2[i]
                        + S::from_f64(tableau::A43) * k3[i]);
            }
            problem.rhs(t + h * S::from_f64(tableau::C4), &y_stage, &mut k4);

            // k5
            for i in 0..dim {
                y_stage[i] = y[i]
                    + h * (S::from_f64(tableau::A51) * k1[i]
                        + S::from_f64(tableau::A52) * k2[i]
                        + S::from_f64(tableau::A53) * k3[i]
                        + S::from_f64(tableau::A54) * k4[i]);
            }
            problem.rhs(t + h * S::from_f64(tableau::C5), &y_stage, &mut k5);

            // k6
            for i in 0..dim {
                y_stage[i] = y[i]
                    + h * (S::from_f64(tableau::A61) * k1[i]
                        + S::from_f64(tableau::A62) * k2[i]
                        + S::from_f64(tableau::A63) * k3[i]
                        + S::from_f64(tableau::A64) * k4[i]
                        + S::from_f64(tableau::A65) * k5[i]);
            }
            problem.rhs(t + h * S::from_f64(tableau::C6), &y_stage, &mut k6);

            // Compute y_new (5th order)
            for i in 0..dim {
                y_new[i] = y[i]
                    + h * (S::from_f64(tableau::B1) * k1[i]
                        + S::from_f64(tableau::B3) * k3[i]
                        + S::from_f64(tableau::B4) * k4[i]
                        + S::from_f64(tableau::B5) * k5[i]
                        + S::from_f64(tableau::B6) * k6[i]);
            }

            // k7 (FSAL: this will be k1 of next step)
            problem.rhs(t + h, &y_new, &mut k7);

            stats.n_eval += 6; // We computed k2..k7

            // Error estimate
            for i in 0..dim {
                err[i] = h
                    * (S::from_f64(tableau::E1) * k1[i]
                        + S::from_f64(tableau::E3) * k3[i]
                        + S::from_f64(tableau::E4) * k4[i]
                        + S::from_f64(tableau::E5) * k5[i]
                        + S::from_f64(tableau::E6) * k6[i]
                        + S::from_f64(tableau::E7) * k7[i]);
            }

            // Compute scaled error norm using pre-allocated buffer
            update_tol_weights(&mut tol_weights, &y);
            let err_norm = weighted_rms_norm(&err, &tol_weights);

            // Detect NaN in error norm (propagated from NaN inputs/RHS)
            if err_norm.is_nan() {
                return Err(SolverError::Other(
                    "NaN detected in error estimate (check inputs and RHS function)".to_string(),
                ));
            }

            // Step size control
            let proposal = controller.propose(h, err_norm, 5);

            if proposal.accept {
                // Accept step
                stats.n_accept += 1;
                controller.accept(h, err_norm);

                // Build interpolation coefficients (needed for dense output)
                // Uses pre-allocated k_all buffer to avoid per-step allocation.
                let interp_coeffs = if options.dense_output {
                    k_all[0..dim].copy_from_slice(&k1);
                    k_all[dim..2 * dim].copy_from_slice(&k2);
                    k_all[2 * dim..3 * dim].copy_from_slice(&k3);
                    k_all[3 * dim..4 * dim].copy_from_slice(&k4);
                    k_all[4 * dim..5 * dim].copy_from_slice(&k5);
                    k_all[5 * dim..6 * dim].copy_from_slice(&k6);
                    k_all[6 * dim..7 * dim].copy_from_slice(&k7);
                    Some(DoPri5Interpolant::build_coefficients(
                        &y, &y_new, &k_all, h, dim,
                    ))
                } else {
                    None
                };

                // Store dense output if enabled
                if options.dense_output {
                    if let Some(ref coeffs) = interp_coeffs {
                        dense.add_segment(DenseSegment::new(t, t + h, coeffs.clone(), dim));
                    }
                }

                // Event detection
                if has_events {
                    let t_new = t + h;

                    let mut stop_event = false;
                    let mut earliest_event_t = t_new;
                    let mut earliest_event_y: Option<Vec<S>> = None;

                    for (idx, event_fn) in options.events.iter().enumerate() {
                        let g_curr = event_fn.evaluate(t_new, &y_new);

                        // Check for sign change
                        if g_prev[idx] * g_curr < S::ZERO {
                            // Build Hermite cubic interpolation between step endpoints.
                            // Uses y, y_new (values) and k1, k7 (derivatives) for
                            // accurate cubic interpolation.
                            let y_ref = &y;
                            let y_new_ref = &y_new;
                            let k1_ref = &k1;
                            let k7_ref = &k7;
                            let t_start = t;
                            let h_step = h;
                            let interpolate = move |t_interp: S| -> Vec<S> {
                                let theta = (t_interp - t_start) / h_step;
                                let theta2 = theta * theta;
                                let theta3 = theta2 * theta;
                                // Hermite basis functions
                                let h00 = S::TWO * theta3 - S::from_f64(3.0) * theta2 + S::ONE;
                                let h10 = theta3 - S::TWO * theta2 + theta;
                                let h01 = -S::TWO * theta3 + S::from_f64(3.0) * theta2;
                                let h11 = theta3 - theta2;
                                let mut y_interp = vec![S::ZERO; dim];
                                for i in 0..dim {
                                    y_interp[i] = h00 * y_ref[i]
                                        + h10 * h_step * k1_ref[i]
                                        + h01 * y_new_ref[i]
                                        + h11 * h_step * k7_ref[i];
                                }
                                y_interp
                            };

                            if let Some((t_event, y_event)) = find_event_time(
                                event_fn.as_ref(),
                                t,
                                &y,
                                t_new,
                                &y_new,
                                &interpolate,
                            ) {
                                // Track the earliest event in case of multiple
                                if earliest_event_y.is_none()
                                    || (direction * (t_event - earliest_event_t) < S::ZERO)
                                {
                                    earliest_event_t = t_event;
                                    earliest_event_y = Some(y_event.clone());
                                }

                                detected_events.push(Event {
                                    t: t_event,
                                    y: y_event,
                                    event_index: idx,
                                });

                                if event_fn.action() == EventAction::Stop {
                                    stop_event = true;
                                }
                            }
                        }

                        g_prev[idx] = g_curr;
                    }

                    if stop_event {
                        // Terminate integration at the earliest stop event.
                        // Filter out events that occur after the earliest stop event.
                        let ev_t = earliest_event_t;
                        // Safety: stop_event is only set when an event is found,
                        // which guarantees earliest_event_y is Some.
                        let ev_y = match earliest_event_y {
                            Some(y) => y,
                            None => {
                                return Err(SolverError::Other(
                                    "Internal error: stop event without event data".into(),
                                ))
                            }
                        };

                        detected_events.retain(|e| direction * (e.t - ev_t) <= S::ZERO);

                        t_out.push(ev_t);
                        y_out.extend_from_slice(&ev_y);

                        let mut result = SolverResult::new(t_out, y_out, dim, stats);
                        result.events = detected_events;
                        result.terminated_by_event = true;
                        return Ok(result);
                    }
                }

                // Update state
                t = t + h;
                y.copy_from_slice(&y_new);

                // FSAL: k7 becomes k1
                k1.copy_from_slice(&k7);

                // Store output
                t_out.push(t);
                y_out.extend_from_slice(&y);

                step_count += 1;
            } else {
                // Reject step
                stats.n_reject += 1;
                controller.reject(h, err_norm);
                last_step = false; // Need to retry
            }

            // Update step size
            h = direction * proposal.h_new.abs().min(options.h_max).max(options.h_min);
        }

        let mut result = SolverResult::new(t_out, y_out, dim, stats);
        result.events = detected_events;
        Ok(result)
    }
}

/// Compute weighted RMS norm for error control.
fn weighted_rms_norm<S: Scalar>(err: &[S], weights: &[S]) -> S {
    let n = S::from_usize(err.len());
    let mut sum = S::ZERO;
    for (e, w) in err.iter().zip(weights.iter()) {
        let scaled = *e / *w;
        sum = sum + scaled * scaled;
    }
    (sum / n).sqrt()
}

/// Estimate initial step size using the algorithm from Hairer-Wanner.
fn estimate_initial_step<S: Scalar, Sys: OdeSystem<S>>(
    problem: &Sys,
    t0: S,
    y0: &[S],
    direction: S,
    options: &SolverOptions<S>,
) -> S {
    let dim = problem.dim();

    // Compute f0 = f(t0, y0)
    let mut f0 = vec![S::ZERO; dim];
    problem.rhs(t0, y0, &mut f0);

    // Compute scale = atol + |y0| * rtol
    let scale: Vec<S> = y0
        .iter()
        .map(|&yi| options.atol + options.rtol * yi.abs())
        .collect();

    // d0 = ||y0 / scale||
    let d0 = weighted_rms_norm(y0, &scale);

    // d1 = ||f0 / scale||
    let d1 = weighted_rms_norm(&f0, &scale);

    // First guess: h0 = 0.01 * d0/d1
    let h0 = if d0 < S::EPSILON.sqrt() || d1 < S::EPSILON.sqrt() {
        S::from_f64(1e-6)
    } else {
        S::from_f64(0.01) * d0 / d1
    };

    // Perform explicit Euler step
    let mut y1 = vec![S::ZERO; dim];
    for i in 0..dim {
        y1[i] = y0[i] + direction * h0 * f0[i];
    }

    // Compute f1 = f(t0 + h0, y1)
    let mut f1 = vec![S::ZERO; dim];
    problem.rhs(t0 + direction * h0, &y1, &mut f1);

    // d2 = ||f1 - f0|| / h0
    let mut df = vec![S::ZERO; dim];
    for i in 0..dim {
        df[i] = (f1[i] - f0[i]) / h0;
    }
    let d2 = weighted_rms_norm(&df, &scale);

    // h1 = (0.01 / max(d1, d2))^(1/5)
    let max_d = d1.max(d2);
    let h1 = if max_d <= S::from_f64(1e-15) {
        (h0 * S::from_f64(1e-3)).max(S::from_f64(1e-6))
    } else {
        (S::from_f64(0.01) / max_d).powf(S::from_f64(0.2))
    };

    // h = min(100 * h0, h1)
    let h = (S::from_f64(100.0) * h0).min(h1);

    direction * h
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::problem::OdeProblem;

    #[test]
    fn test_exponential_decay() {
        // dy/dt = -y, y(0) = 1, exact: y(t) = e^(-t)
        let problem = OdeProblem::new(
            |_t: f64, y: &[f64], dydt: &mut [f64]| {
                dydt[0] = -y[0];
            },
            0.0,
            5.0,
            vec![1.0],
        );

        let options = SolverOptions::default().rtol(1e-8).atol(1e-10);
        let result = DoPri5::solve(&problem, 0.0, 5.0, &[1.0], &options).unwrap();

        assert!(result.success);
        let y_final = result.y_final().unwrap();
        let exact = (-5.0_f64).exp();
        let error = (y_final[0] - exact).abs();
        assert!(error < 1e-7, "Error {} too large", error);
    }

    #[test]
    fn test_harmonic_oscillator() {
        // d²x/dt² = -x => y' = [y[1], -y[0]]
        // x(0) = 1, x'(0) = 0
        // Exact: x(t) = cos(t)
        let problem = OdeProblem::new(
            |_t: f64, y: &[f64], dydt: &mut [f64]| {
                dydt[0] = y[1];
                dydt[1] = -y[0];
            },
            0.0,
            10.0,
            vec![1.0, 0.0],
        );

        let options = SolverOptions::default().rtol(1e-8).atol(1e-10);
        let result = DoPri5::solve(&problem, 0.0, 10.0, &[1.0, 0.0], &options).unwrap();

        assert!(result.success);
        let y_final = result.y_final().unwrap();
        let exact_x = 10.0_f64.cos();
        let exact_v = -10.0_f64.sin();

        let error_x = (y_final[0] - exact_x).abs();
        let error_v = (y_final[1] - exact_v).abs();

        assert!(error_x < 1e-6, "Position error {} too large", error_x);
        assert!(error_v < 1e-6, "Velocity error {} too large", error_v);
    }

    #[test]
    fn test_lorenz_stability() {
        // Lorenz system - just check that it runs without blowing up
        let problem = OdeProblem::new(
            |_t: f64, y: &[f64], dydt: &mut [f64]| {
                let sigma = 10.0;
                let rho = 28.0;
                let beta = 8.0 / 3.0;
                dydt[0] = sigma * (y[1] - y[0]);
                dydt[1] = y[0] * (rho - y[2]) - y[1];
                dydt[2] = y[0] * y[1] - beta * y[2];
            },
            0.0,
            20.0,
            vec![1.0, 1.0, 1.0],
        );

        let options = SolverOptions::default();
        let result = DoPri5::solve(&problem, 0.0, 20.0, &[1.0, 1.0, 1.0], &options).unwrap();

        assert!(result.success);
        let y_final = result.y_final().unwrap();

        // Solution should remain bounded
        for &yi in y_final.iter() {
            assert!(yi.abs() < 100.0, "Solution blew up");
        }
    }

    #[test]
    fn test_backward_integration() {
        // dy/dt = -y, solve backward from t=5 to t=0
        // y(5) = e^(-5), exact: y(0) = 1
        let y5 = (-5.0_f64).exp();

        let problem = OdeProblem::new(
            |_t: f64, y: &[f64], dydt: &mut [f64]| {
                dydt[0] = -y[0];
            },
            5.0,
            0.0,
            vec![y5],
        );

        let options = SolverOptions::default().rtol(1e-8).atol(1e-10);
        let result = DoPri5::solve(&problem, 5.0, 0.0, &[y5], &options).unwrap();

        assert!(result.success);
        let y_final = result.y_final().unwrap();
        let error = (y_final[0] - 1.0).abs();
        assert!(error < 1e-6, "Error {} too large", error);
    }

    #[test]
    fn test_stats() {
        let problem = OdeProblem::new(
            |_t: f64, y: &[f64], dydt: &mut [f64]| {
                dydt[0] = -y[0];
            },
            0.0,
            1.0,
            vec![1.0],
        );

        let options = SolverOptions::default();
        let result = DoPri5::solve(&problem, 0.0, 1.0, &[1.0], &options).unwrap();

        assert!(result.stats.n_accept > 0);
        assert!(result.stats.n_eval > 0);
    }

    // ============================================================================
    // Edge Case Tests
    // ============================================================================

    #[test]
    fn test_zero_interval() {
        // t0 == t_end, should return immediately
        let problem = OdeProblem::new(
            |_t: f64, y: &[f64], dydt: &mut [f64]| {
                dydt[0] = -y[0];
            },
            0.0,
            0.0,
            vec![1.0],
        );

        let options = SolverOptions::default();
        let result = DoPri5::solve(&problem, 0.0, 0.0, &[1.0], &options).unwrap();

        assert!(result.success);
        let y_final = result.y_final().unwrap();
        assert!((y_final[0] - 1.0).abs() < 1e-15);
    }

    #[test]
    fn test_very_short_interval() {
        // Very small integration interval
        let problem = OdeProblem::new(
            |_t: f64, y: &[f64], dydt: &mut [f64]| {
                dydt[0] = -y[0];
            },
            0.0,
            1e-10,
            vec![1.0],
        );

        let options = SolverOptions::default();
        let result = DoPri5::solve(&problem, 0.0, 1e-10, &[1.0], &options).unwrap();

        assert!(result.success);
        let y_final = result.y_final().unwrap();
        // Should be very close to initial
        assert!((y_final[0] - 1.0).abs() < 1e-8);
    }

    #[test]
    fn test_constant_zero_rhs() {
        // dy/dt = 0, solution should stay constant
        let problem = OdeProblem::new(
            |_t: f64, _y: &[f64], dydt: &mut [f64]| {
                dydt[0] = 0.0;
            },
            0.0,
            10.0,
            vec![42.0],
        );

        let options = SolverOptions::default();
        let result = DoPri5::solve(&problem, 0.0, 10.0, &[42.0], &options).unwrap();

        assert!(result.success);
        let y_final = result.y_final().unwrap();
        assert!((y_final[0] - 42.0).abs() < 1e-12);
    }

    #[test]
    fn test_single_step_only() {
        // Use max_steps = 1 to force early termination
        let problem = OdeProblem::new(
            |_t: f64, y: &[f64], dydt: &mut [f64]| {
                dydt[0] = -y[0];
            },
            0.0,
            10.0,
            vec![1.0],
        );

        let options = SolverOptions::default().max_steps(1);
        let result = DoPri5::solve(&problem, 0.0, 10.0, &[1.0], &options);

        // Should return an error when max_steps is exceeded
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            crate::error::SolverError::MaxIterationsExceeded { .. }
        ));
    }

    #[test]
    fn test_tight_tolerance() {
        // Very tight tolerances
        let problem = OdeProblem::new(
            |_t: f64, y: &[f64], dydt: &mut [f64]| {
                dydt[0] = -y[0];
            },
            0.0,
            1.0,
            vec![1.0],
        );

        let options = SolverOptions::default().rtol(1e-12).atol(1e-14);
        let result = DoPri5::solve(&problem, 0.0, 1.0, &[1.0], &options).unwrap();

        assert!(result.success);
        let y_final = result.y_final().unwrap();
        let exact = (-1.0_f64).exp();
        let error = (y_final[0] - exact).abs();
        assert!(
            error < 1e-11,
            "Error {} too large for tight tolerance",
            error
        );
    }

    #[test]
    fn test_loose_tolerance() {
        // Very loose tolerances - should complete quickly
        let problem = OdeProblem::new(
            |_t: f64, y: &[f64], dydt: &mut [f64]| {
                dydt[0] = -y[0];
            },
            0.0,
            1.0,
            vec![1.0],
        );

        let options = SolverOptions::default().rtol(1e-2).atol(1e-3);
        let result = DoPri5::solve(&problem, 0.0, 1.0, &[1.0], &options).unwrap();

        assert!(result.success);
        // Should have fewer steps than tight tolerance
        assert!(result.stats.n_accept < 50);
    }

    #[test]
    fn test_zero_initial_condition() {
        // y(0) = 0, dy/dt = 1, exact: y(t) = t
        let problem = OdeProblem::new(
            |_t: f64, _y: &[f64], dydt: &mut [f64]| {
                dydt[0] = 1.0;
            },
            0.0,
            5.0,
            vec![0.0],
        );

        let options = SolverOptions::default();
        let result = DoPri5::solve(&problem, 0.0, 5.0, &[0.0], &options).unwrap();

        assert!(result.success);
        let y_final = result.y_final().unwrap();
        assert!((y_final[0] - 5.0).abs() < 1e-8);
    }

    #[test]
    fn test_large_initial_condition() {
        // Large initial value
        let problem = OdeProblem::new(
            |_t: f64, y: &[f64], dydt: &mut [f64]| {
                dydt[0] = -0.1 * y[0];
            },
            0.0,
            1.0,
            vec![1e10],
        );

        let options = SolverOptions::default();
        let result = DoPri5::solve(&problem, 0.0, 1.0, &[1e10], &options).unwrap();

        assert!(result.success);
        let y_final = result.y_final().unwrap();
        let exact = 1e10 * (-0.1_f64).exp();
        let rel_error = (y_final[0] - exact).abs() / exact;
        assert!(rel_error < 1e-5, "Relative error {} too large", rel_error);
    }

    #[test]
    fn test_high_dimension() {
        // Higher dimensional system (10D linear decay)
        let problem = OdeProblem::new(
            |_t: f64, y: &[f64], dydt: &mut [f64]| {
                for (i, &yi) in y.iter().enumerate() {
                    dydt[i] = -(i as f64 + 1.0) * 0.1 * yi;
                }
            },
            0.0,
            1.0,
            vec![1.0; 10],
        );

        let options = SolverOptions::default();
        let result = DoPri5::solve(&problem, 0.0, 1.0, &[1.0; 10], &options).unwrap();

        assert!(result.success);
        let y_final = result.y_final().unwrap();
        assert_eq!(y_final.len(), 10);

        // Check each component
        for (i, &yi) in y_final.iter().enumerate() {
            let rate = (i as f64 + 1.0) * 0.1;
            let exact = (-rate).exp();
            let error = (yi - exact).abs();
            assert!(error < 1e-5, "Component {} error {} too large", i, error);
        }
    }

    // ============================================================================
    // Event Detection Tests
    // ============================================================================

    #[test]
    fn test_event_detection_bouncing_ball() {
        // Bouncing ball: y'' = -g  =>  y[0]' = y[1], y[1]' = -g
        // Event: y[0] = 0 (ground contact, falling direction)
        use crate::events::{EventAction, EventDirection, EventFunction};

        struct GroundContact;

        impl EventFunction<f64> for GroundContact {
            fn evaluate(&self, _t: f64, y: &[f64]) -> f64 {
                y[0] // Event when height = 0
            }

            fn direction(&self) -> EventDirection {
                EventDirection::Falling // Only when falling (y goes from + to -)
            }

            fn action(&self) -> EventAction {
                EventAction::Stop // Stop at first bounce
            }
        }

        let g = 9.81_f64;
        let problem = OdeProblem::new(
            |_t, y: &[f64], dydt: &mut [f64]| {
                dydt[0] = y[1]; // dy/dt = v
                dydt[1] = -g; // dv/dt = -g
            },
            0.0,
            10.0,
            vec![10.0, 0.0],
        );

        let y0 = vec![10.0, 0.0]; // height=10, velocity=0

        let options = SolverOptions::default()
            .rtol(1e-8)
            .atol(1e-10)
            .event(Box::new(GroundContact));

        let result = DoPri5::solve(&problem, 0.0, 10.0, &y0, &options).unwrap();

        // Should stop at first bounce
        assert!(
            result.terminated_by_event,
            "Should have terminated by event"
        );
        assert!(!result.events.is_empty(), "Should have detected events");

        // At event, height should be ~0
        let event = &result.events[0];
        assert!(
            event.y[0].abs() < 1e-4,
            "Event should occur at y=0, got y={}",
            event.y[0]
        );

        // Time to fall from height 10: t = sqrt(2h/g) ~ 1.428
        let expected_t = (2.0 * 10.0 / g).sqrt();
        assert!(
            (event.t - expected_t).abs() < 0.01,
            "Expected t={:.3}, got t={:.3}",
            expected_t,
            event.t
        );
    }

    #[test]
    fn test_event_continue_action() {
        // Event with Continue action should record but not stop
        use crate::events::{EventAction, EventDirection, EventFunction};

        struct ZeroCrossing;

        impl EventFunction<f64> for ZeroCrossing {
            fn evaluate(&self, _t: f64, y: &[f64]) -> f64 {
                y[0] // Event when y[0] crosses zero
            }

            fn direction(&self) -> EventDirection {
                EventDirection::Both
            }

            fn action(&self) -> EventAction {
                EventAction::Continue // Record but continue
            }
        }

        // Harmonic oscillator: y(t) = cos(t), crosses zero at pi/2, 3pi/2, etc.
        let problem = OdeProblem::new(
            |_t: f64, y: &[f64], dydt: &mut [f64]| {
                dydt[0] = y[1];
                dydt[1] = -y[0];
            },
            0.0,
            10.0,
            vec![1.0, 0.0],
        );

        let options = SolverOptions::default()
            .rtol(1e-8)
            .atol(1e-10)
            .event(Box::new(ZeroCrossing));

        let result = DoPri5::solve(&problem, 0.0, 10.0, &[1.0, 0.0], &options).unwrap();

        // Should NOT terminate by event
        assert!(
            !result.terminated_by_event,
            "Should not have terminated by event"
        );

        // Should have detected multiple zero crossings of cos(t)
        // In [0, 10], cos(t) = 0 at t = pi/2, 3pi/2, 5pi/2, 7pi/2, 9pi/2
        // That's roughly at t = 1.571, 4.712, 7.854 (and 3 more approaching 10)
        assert!(
            result.events.len() >= 3,
            "Should have detected at least 3 events, got {}",
            result.events.len()
        );

        // Check first event is near pi/2
        let first = &result.events[0];
        let expected_t = std::f64::consts::FRAC_PI_2;
        assert!(
            (first.t - expected_t).abs() < 0.01,
            "First event expected at t={:.3}, got t={:.3}",
            expected_t,
            first.t
        );
    }

    #[test]
    fn test_event_rising_only_integration() {
        // Harmonic oscillator: y(t) = cos(t), y'(t) = -sin(t)
        // y(t) crosses zero at pi/2 (falling) and 3pi/2 (rising).
        // With Rising direction, should only detect rising crossings.
        use crate::events::{EventAction, EventDirection, EventFunction};

        struct RisingZeroCrossing;

        impl EventFunction<f64> for RisingZeroCrossing {
            fn evaluate(&self, _t: f64, y: &[f64]) -> f64 {
                y[0]
            }
            fn direction(&self) -> EventDirection {
                EventDirection::Rising
            }
            fn action(&self) -> EventAction {
                EventAction::Continue
            }
        }

        let problem = OdeProblem::new(
            |_t: f64, y: &[f64], dydt: &mut [f64]| {
                dydt[0] = y[1];
                dydt[1] = -y[0];
            },
            0.0,
            10.0,
            vec![1.0, 0.0],
        );

        let options = SolverOptions::default()
            .rtol(1e-8)
            .atol(1e-10)
            .event(Box::new(RisingZeroCrossing));

        let result = DoPri5::solve(&problem, 0.0, 10.0, &[1.0, 0.0], &options).unwrap();

        // cos(t) = 0 rising at t = 3pi/2, 7pi/2 (approximately 4.712, 10.996)
        // In [0, 10], only 3pi/2 ≈ 4.712 qualifies as rising
        // At pi/2, cos goes from + to - (falling), should be filtered out
        for event in &result.events {
            // At each event, y[0] ~ 0 and velocity y[1] should be positive (rising)
            assert!(
                event.y[1] > -0.1,
                "Rising event should have positive velocity, got y[1]={}",
                event.y[1]
            );
        }
        assert!(
            !result.events.is_empty(),
            "Should detect at least one rising zero crossing"
        );
    }

    #[test]
    fn test_event_simultaneous_events() {
        // Two event functions that detect the same zero crossing
        use crate::events::{EventAction, EventDirection, EventFunction};

        struct ZeroCross1;
        impl EventFunction<f64> for ZeroCross1 {
            fn evaluate(&self, _t: f64, y: &[f64]) -> f64 {
                y[0]
            }
            fn direction(&self) -> EventDirection {
                EventDirection::Both
            }
            fn action(&self) -> EventAction {
                EventAction::Continue
            }
        }

        struct ZeroCross2;
        impl EventFunction<f64> for ZeroCross2 {
            fn evaluate(&self, _t: f64, y: &[f64]) -> f64 {
                y[0]
            }
            fn direction(&self) -> EventDirection {
                EventDirection::Both
            }
            fn action(&self) -> EventAction {
                EventAction::Continue
            }
        }

        let problem = OdeProblem::new(
            |_t: f64, y: &[f64], dydt: &mut [f64]| {
                dydt[0] = y[1];
                dydt[1] = -y[0];
            },
            0.0,
            5.0,
            vec![1.0, 0.0],
        );

        let options = SolverOptions::default()
            .rtol(1e-8)
            .atol(1e-10)
            .event(Box::new(ZeroCross1))
            .event(Box::new(ZeroCross2));

        let result = DoPri5::solve(&problem, 0.0, 5.0, &[1.0, 0.0], &options).unwrap();

        // Both event functions detect the same crossings, so we expect
        // pairs of events at each crossing time
        assert!(
            result.events.len() >= 4,
            "Should detect events from both functions, got {}",
            result.events.len()
        );

        // Check that events from both indices are present
        let has_idx_0 = result.events.iter().any(|e| e.event_index == 0);
        let has_idx_1 = result.events.iter().any(|e| e.event_index == 1);
        assert!(has_idx_0, "Should have events from function 0");
        assert!(has_idx_1, "Should have events from function 1");
    }

    #[test]
    fn test_event_backward_integration() {
        // Event detection during backward integration
        use crate::events::{EventAction, EventDirection, EventFunction};

        struct ZeroCross;
        impl EventFunction<f64> for ZeroCross {
            fn evaluate(&self, _t: f64, y: &[f64]) -> f64 {
                y[0]
            }
            fn direction(&self) -> EventDirection {
                EventDirection::Both
            }
            fn action(&self) -> EventAction {
                EventAction::Stop
            }
        }

        // Harmonic oscillator backward from t=5 to t=0
        // cos(t) crosses zero going backward
        let y5 = [5.0_f64.cos(), -5.0_f64.sin()];
        let problem = OdeProblem::new(
            |_t: f64, y: &[f64], dydt: &mut [f64]| {
                dydt[0] = y[1];
                dydt[1] = -y[0];
            },
            5.0,
            0.0,
            y5.to_vec(),
        );

        let options = SolverOptions::default()
            .rtol(1e-8)
            .atol(1e-10)
            .event(Box::new(ZeroCross));

        let result = DoPri5::solve(&problem, 5.0, 0.0, &y5, &options).unwrap();

        // Should detect a zero crossing during backward integration
        assert!(
            result.terminated_by_event,
            "Should terminate at event during backward integration"
        );
        assert!(
            !result.events.is_empty(),
            "Should detect events during backward integration"
        );

        // The event time should be between 0 and 5
        let event = &result.events[0];
        assert!(
            event.t > 0.0 && event.t < 5.0,
            "Event time {} should be between 0 and 5",
            event.t
        );
        assert!(
            event.y[0].abs() < 0.01,
            "y at event should be ~0, got {}",
            event.y[0]
        );
    }

    #[test]
    fn test_no_event_when_no_crossing() {
        // Exponential decay never crosses zero, so no events should be detected
        use crate::events::{EventAction, EventDirection, EventFunction};

        struct ZeroCheck;

        impl EventFunction<f64> for ZeroCheck {
            fn evaluate(&self, _t: f64, y: &[f64]) -> f64 {
                y[0] // y[0] = e^(-t) > 0 always
            }

            fn direction(&self) -> EventDirection {
                EventDirection::Both
            }

            fn action(&self) -> EventAction {
                EventAction::Stop
            }
        }

        let problem = OdeProblem::new(
            |_t: f64, y: &[f64], dydt: &mut [f64]| {
                dydt[0] = -y[0];
            },
            0.0,
            5.0,
            vec![1.0],
        );

        let options = SolverOptions::default().event(Box::new(ZeroCheck));

        let result = DoPri5::solve(&problem, 0.0, 5.0, &[1.0], &options).unwrap();

        assert!(!result.terminated_by_event);
        assert!(result.events.is_empty());
    }

    // ============================================================================
    // f32 Scalar Tests
    // ============================================================================

    #[test]
    fn test_exponential_decay_f32() {
        // Verify DoPri5 works with f32 scalars
        let problem = OdeProblem::new(
            |_t: f32, y: &[f32], dydt: &mut [f32]| {
                dydt[0] = -y[0];
            },
            0.0f32,
            5.0f32,
            vec![1.0f32],
        );

        let options: SolverOptions<f32> = SolverOptions::default().rtol(1e-4).atol(1e-6);
        let result = DoPri5::solve(&problem, 0.0f32, 5.0f32, &[1.0f32], &options).unwrap();

        assert!(result.success);
        let y_final = result.y_final().unwrap();
        let exact = (-5.0f32).exp();
        let error = (y_final[0] - exact).abs();
        assert!(error < 1e-3, "f32 error {} too large", error);
    }

    #[test]
    fn test_harmonic_oscillator_f32() {
        let problem = OdeProblem::new(
            |_t: f32, y: &[f32], dydt: &mut [f32]| {
                dydt[0] = y[1];
                dydt[1] = -y[0];
            },
            0.0f32,
            6.0f32,
            vec![1.0f32, 0.0f32],
        );

        let options: SolverOptions<f32> = SolverOptions::default().rtol(1e-4).atol(1e-6);
        let result = DoPri5::solve(&problem, 0.0f32, 6.0f32, &[1.0f32, 0.0f32], &options).unwrap();

        assert!(result.success);
        let y_final = result.y_final().unwrap();
        let exact_x = 6.0f32.cos();
        let error = (y_final[0] - exact_x).abs();
        assert!(error < 1e-3, "f32 harmonic error {} too large", error);
    }

    // ============================================================================
    // NaN / Infinity Input Tests
    // ============================================================================

    #[test]
    fn test_nan_initial_condition() {
        // NaN in initial condition — solver should detect and return error
        let problem = OdeProblem::new(
            |_t: f64, y: &[f64], dydt: &mut [f64]| {
                dydt[0] = -y[0];
            },
            0.0,
            1.0,
            vec![f64::NAN],
        );

        let options = SolverOptions::default();
        let result = DoPri5::solve(&problem, 0.0, 1.0, &[f64::NAN], &options);
        assert!(
            result.is_err(),
            "NaN initial condition should produce error"
        );
    }

    #[test]
    fn test_infinity_initial_condition() {
        // Infinity in initial condition — solver should detect NaN from computation
        let problem = OdeProblem::new(
            |_t: f64, y: &[f64], dydt: &mut [f64]| {
                dydt[0] = -y[0];
            },
            0.0,
            1.0,
            vec![f64::INFINITY],
        );

        let options = SolverOptions::default();
        let result = DoPri5::solve(&problem, 0.0, 1.0, &[f64::INFINITY], &options);
        // Infinity * 0 = NaN in error estimate, so should fail
        assert!(
            result.is_err(),
            "Infinity initial condition should produce error"
        );
    }

    #[test]
    fn test_rhs_produces_nan() {
        // RHS function produces NaN — solver should detect and return error
        let problem = OdeProblem::new(
            |_t: f64, _y: &[f64], dydt: &mut [f64]| {
                dydt[0] = f64::NAN;
            },
            0.0,
            1.0,
            vec![1.0],
        );

        let options = SolverOptions::default();
        let result = DoPri5::solve(&problem, 0.0, 1.0, &[1.0], &options);
        assert!(result.is_err(), "NaN in RHS should produce error");
    }
}

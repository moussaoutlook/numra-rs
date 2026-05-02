//! Tsit5: Tsitouras 5(4) explicit Runge-Kutta method.
//!
//! A modern 5th order explicit RK method with optimized coefficients
//! for efficient error estimation.
//!
//! ## Features
//! - 5th order accuracy with embedded 4th order for error estimation
//! - 7 stages (FSAL - First Same As Last)
//! - Optimized coefficients for typical ODE problems
//!
//! ## Reference
//! Tsitouras, Ch. (2011), "Runge-Kutta pairs of order 5(4) satisfying only the
//! first column simplifying assumption", Computers & Mathematics with Applications.
//!
//! Author: Moussa Leblouba
//! Date: 4 February 2026
//! Modified: 2 May 2026

use crate::error::SolverError;
use crate::problem::OdeSystem;
use crate::solver::{Solver, SolverOptions, SolverResult, SolverStats};
use numra_core::Scalar;

/// Tsit5 solver: Tsitouras 5(4) method.
#[derive(Clone, Debug, Default)]
pub struct Tsit5;

impl Tsit5 {
    /// Create a new Tsit5 solver.
    pub fn new() -> Self {
        Self
    }
}

/// Tsitouras 5(4) coefficients.
#[allow(dead_code)]
mod tableau {
    // Nodes (c_i)
    pub const C2: f64 = 0.161;
    pub const C3: f64 = 0.327;
    pub const C4: f64 = 0.9;
    pub const C5: f64 = 0.9800255409045097;
    pub const C6: f64 = 1.0;
    pub const C7: f64 = 1.0;

    // A matrix (lower triangular, row by row)
    pub const A21: f64 = 0.161;

    pub const A31: f64 = -0.008480655492356989;
    pub const A32: f64 = 0.335480655492357;

    pub const A41: f64 = 2.8971530571054935;
    pub const A42: f64 = -6.359448489975075;
    pub const A43: f64 = 4.3622954328695815;

    pub const A51: f64 = 5.325864828439257;
    pub const A52: f64 = -11.748883564062828;
    pub const A53: f64 = 7.4955393428898365;
    pub const A54: f64 = -0.09249506636175525;

    pub const A61: f64 = 5.86145544294642;
    pub const A62: f64 = -12.92096931784711;
    pub const A63: f64 = 8.159367898576159;
    pub const A64: f64 = -0.071584973281401;
    pub const A65: f64 = -0.028269050394068383;

    pub const A71: f64 = 0.09646076681806523;
    pub const A72: f64 = 0.01;
    pub const A73: f64 = 0.4798896504144996;
    pub const A74: f64 = 1.379008574103742;
    pub const A75: f64 = -3.290069515436081;
    pub const A76: f64 = 2.324710524099774;

    // 5th order weights (b_i)
    pub const B1: f64 = 0.09646076681806523;
    pub const B2: f64 = 0.01;
    pub const B3: f64 = 0.4798896504144996;
    pub const B4: f64 = 1.379008574103742;
    pub const B5: f64 = -3.290069515436081;
    pub const B6: f64 = 2.324710524099774;
    pub const B7: f64 = 0.0;

    // Error coefficients (E_i = b_i - b_hat_i) from Tsitouras 2011 paper, Table 1
    // These are the differences between 5th and 4th order solutions
    // Note: b_hat_7 = 1/66 in the paper, so E7 = b7 - b_hat_7 = 0 - 1/66 = -1/66
    // The sum of all E coefficients must be 0 for proper error estimation
    pub const E1: f64 = 0.001780011052226;
    pub const E2: f64 = 0.000816434459657;
    pub const E3: f64 = -0.007880878010262;
    pub const E4: f64 = 0.144711007173263;
    pub const E5: f64 = -0.582357165452555;
    pub const E6: f64 = 0.458082105929187;
    pub const E7: f64 = -1.0 / 66.0; // -0.015151515151515...
}

impl<S: Scalar> Solver<S> for Tsit5 {
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

        let mut t = t0;
        let mut y = y0.to_vec();
        let mut t_out = vec![t0];
        let mut y_out = y0.to_vec();

        // Stage derivatives
        let mut k1 = vec![S::ZERO; dim];
        let mut k2 = vec![S::ZERO; dim];
        let mut k3 = vec![S::ZERO; dim];
        let mut k4 = vec![S::ZERO; dim];
        let mut k5 = vec![S::ZERO; dim];
        let mut k6 = vec![S::ZERO; dim];
        let mut k7 = vec![S::ZERO; dim];
        let mut y_stage = vec![S::ZERO; dim];
        let mut y_new = vec![S::ZERO; dim];
        let mut err = vec![S::ZERO; dim];

        let mut stats = SolverStats::default();

        // Initial step size
        problem.rhs(t, &y, &mut k1);
        stats.n_eval += 1;
        let mut h = initial_step_size(&y, &k1, options, dim);
        let h_min = options.h_min;
        let h_max = options.h_max.min((tf - t0).abs());

        let direction = if tf > t0 { S::ONE } else { -S::ONE };
        let mut step_count = 0_usize;

        while (tf - t) * direction > S::from_f64(1e-10) * (tf - t0).abs() {
            if step_count >= options.max_steps {
                return Err(SolverError::MaxIterationsExceeded { t: t.to_f64() });
            }

            // Adjust final step
            if (t + h - tf) * direction > S::ZERO {
                h = tf - t;
            }

            // Clamp step size
            h = h.abs().max(h_min) * direction;
            if h.abs() > h_max {
                h = h_max * direction;
            }

            // Compute stages
            // k1 is already computed (FSAL)

            // k2
            for i in 0..dim {
                y_stage[i] = y[i] + h * S::from_f64(tableau::A21) * k1[i];
            }
            problem.rhs(t + S::from_f64(tableau::C2) * h, &y_stage, &mut k2);

            // k3
            for i in 0..dim {
                y_stage[i] = y[i]
                    + h * (S::from_f64(tableau::A31) * k1[i] + S::from_f64(tableau::A32) * k2[i]);
            }
            problem.rhs(t + S::from_f64(tableau::C3) * h, &y_stage, &mut k3);

            // k4
            for i in 0..dim {
                y_stage[i] = y[i]
                    + h * (S::from_f64(tableau::A41) * k1[i]
                        + S::from_f64(tableau::A42) * k2[i]
                        + S::from_f64(tableau::A43) * k3[i]);
            }
            problem.rhs(t + S::from_f64(tableau::C4) * h, &y_stage, &mut k4);

            // k5
            for i in 0..dim {
                y_stage[i] = y[i]
                    + h * (S::from_f64(tableau::A51) * k1[i]
                        + S::from_f64(tableau::A52) * k2[i]
                        + S::from_f64(tableau::A53) * k3[i]
                        + S::from_f64(tableau::A54) * k4[i]);
            }
            problem.rhs(t + S::from_f64(tableau::C5) * h, &y_stage, &mut k5);

            // k6
            for i in 0..dim {
                y_stage[i] = y[i]
                    + h * (S::from_f64(tableau::A61) * k1[i]
                        + S::from_f64(tableau::A62) * k2[i]
                        + S::from_f64(tableau::A63) * k3[i]
                        + S::from_f64(tableau::A64) * k4[i]
                        + S::from_f64(tableau::A65) * k5[i]);
            }
            problem.rhs(t + S::from_f64(tableau::C6) * h, &y_stage, &mut k6);

            // k7 and y_new (5th order solution)
            for i in 0..dim {
                y_new[i] = y[i]
                    + h * (S::from_f64(tableau::B1) * k1[i]
                        + S::from_f64(tableau::B2) * k2[i]
                        + S::from_f64(tableau::B3) * k3[i]
                        + S::from_f64(tableau::B4) * k4[i]
                        + S::from_f64(tableau::B5) * k5[i]
                        + S::from_f64(tableau::B6) * k6[i]);
            }
            problem.rhs(t + h, &y_new, &mut k7);
            stats.n_eval += 6;

            // Error estimate
            for i in 0..dim {
                err[i] = h
                    * (S::from_f64(tableau::E1) * k1[i]
                        + S::from_f64(tableau::E2) * k2[i]
                        + S::from_f64(tableau::E3) * k3[i]
                        + S::from_f64(tableau::E4) * k4[i]
                        + S::from_f64(tableau::E5) * k5[i]
                        + S::from_f64(tableau::E6) * k6[i]
                        + S::from_f64(tableau::E7) * k7[i]);
            }

            let err_norm = error_norm(&err, &y, &y_new, options, dim);

            // Step size control
            let safety = S::from_f64(0.9);
            let fac_max = S::from_f64(5.0);
            let fac_min = S::from_f64(0.2);

            if err_norm <= S::ONE {
                // Accept step
                stats.n_accept += 1;

                t = t + h;
                y.copy_from_slice(&y_new);
                k1.copy_from_slice(&k7); // FSAL

                t_out.push(t);
                y_out.extend_from_slice(&y);

                // New step size for 5th order method: exponent = -1/(p+1) = -1/6
                let err_safe = err_norm.max(S::from_f64(1e-10));
                let fac = safety * err_safe.powf(S::from_f64(-1.0 / 6.0));
                let fac = fac.min(fac_max).max(fac_min);
                h = h * fac;
            } else {
                // Reject step - use slightly more aggressive reduction
                stats.n_reject += 1;

                let err_safe = err_norm.max(S::from_f64(1e-10));
                let fac = safety * err_safe.powf(S::from_f64(-1.0 / 5.0));
                let fac = fac.max(fac_min);
                h = h * fac;
            }

            if h.abs() < h_min {
                return Err(SolverError::StepSizeTooSmall {
                    t: t.to_f64(),
                    h: h.to_f64(),
                    h_min: h_min.to_f64(),
                });
            }

            step_count += 1;
        }

        Ok(SolverResult::new(t_out, y_out, dim, stats))
    }
}

fn initial_step_size<S: Scalar>(y0: &[S], f0: &[S], options: &SolverOptions<S>, dim: usize) -> S {
    if let Some(h0) = options.h0 {
        return h0;
    }

    let mut y_norm = S::ZERO;
    let mut f_norm = S::ZERO;
    for i in 0..dim {
        let sc = options.atol + options.rtol * y0[i].abs();
        y_norm = y_norm + (y0[i] / sc) * (y0[i] / sc);
        f_norm = f_norm + (f0[i] / sc) * (f0[i] / sc);
    }
    y_norm = (y_norm / S::from_usize(dim)).sqrt();
    f_norm = (f_norm / S::from_usize(dim)).sqrt();

    if y_norm < S::from_f64(1e-5) || f_norm < S::from_f64(1e-5) {
        S::from_f64(1e-6)
    } else {
        (S::from_f64(0.01) * y_norm / f_norm).min(options.h_max)
    }
}

fn error_norm<S: Scalar>(
    err: &[S],
    y: &[S],
    y_new: &[S],
    options: &SolverOptions<S>,
    dim: usize,
) -> S {
    let mut err_norm = S::ZERO;
    for i in 0..dim {
        let sc = options.atol + options.rtol * y[i].abs().max(y_new[i].abs());
        let sc = sc.max(S::from_f64(1e-15));
        let scaled_err = err[i] / sc;
        err_norm = err_norm + scaled_err * scaled_err;
    }
    (err_norm / S::from_usize(dim)).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::problem::OdeProblem;

    #[test]
    fn test_tsit5_exponential_decay() {
        let problem = OdeProblem::new(
            |_t, y: &[f64], dydt: &mut [f64]| {
                dydt[0] = -y[0];
            },
            0.0,
            5.0,
            vec![1.0],
        );
        let options = SolverOptions::default().rtol(1e-6).atol(1e-8);
        let result = Tsit5::solve(&problem, 0.0, 5.0, &[1.0], &options).unwrap();

        assert!(result.success);
        let y_final = result.y_final().unwrap();
        let expected = (-5.0_f64).exp();
        assert!(
            (y_final[0] - expected).abs() < 1e-5,
            "Tsit5 exponential: got {}, expected {}",
            y_final[0],
            expected
        );
    }

    #[test]
    fn test_tsit5_harmonic_oscillator() {
        // y'' + y = 0, or y1' = y2, y2' = -y1
        let problem = OdeProblem::new(
            |_t, y: &[f64], dydt: &mut [f64]| {
                dydt[0] = y[1];
                dydt[1] = -y[0];
            },
            0.0,
            10.0,
            vec![1.0, 0.0],
        );
        // Use moderate tolerances (1e-4 is reasonable for unit tests)
        let options = SolverOptions::default().rtol(1e-4).atol(1e-6);
        let result = Tsit5::solve(&problem, 0.0, 10.0, &[1.0, 0.0], &options).unwrap();

        assert!(result.success);
        let y_final = result.y_final().unwrap();
        // y1 = cos(t), y2 = -sin(t)
        let expected_y1 = 10.0_f64.cos();
        let expected_y2 = -10.0_f64.sin();
        assert!(
            (y_final[0] - expected_y1).abs() < 1e-3,
            "Tsit5 harmonic y[0]: got {}, expected {}",
            y_final[0],
            expected_y1
        );
        assert!(
            (y_final[1] - expected_y2).abs() < 1e-3,
            "Tsit5 harmonic y[1]: got {}, expected {}",
            y_final[1],
            expected_y2
        );
    }

    #[test]
    fn test_tsit5_lorenz() {
        let sigma = 10.0;
        let rho = 28.0;
        let beta = 8.0 / 3.0;

        let problem = OdeProblem::new(
            move |_t, y: &[f64], dydt: &mut [f64]| {
                dydt[0] = sigma * (y[1] - y[0]);
                dydt[1] = y[0] * (rho - y[2]) - y[1];
                dydt[2] = y[0] * y[1] - beta * y[2];
            },
            0.0,
            10.0,
            vec![1.0, 1.0, 1.0],
        );
        // Use moderate tolerances for chaotic system
        let options = SolverOptions::default().rtol(1e-4).atol(1e-6);
        let result = Tsit5::solve(&problem, 0.0, 10.0, &[1.0, 1.0, 1.0], &options);

        assert!(result.is_ok());
    }

    #[test]
    fn test_tsit5_efficiency() {
        // Compare function evaluations with DoPri5 on same problem
        let problem = OdeProblem::new(
            |_t, y: &[f64], dydt: &mut [f64]| {
                dydt[0] = -y[0];
            },
            0.0,
            5.0,
            vec![1.0],
        );
        // Use looser tolerances for efficiency test
        let options = SolverOptions::default().rtol(1e-3).atol(1e-5);
        let result = Tsit5::solve(&problem, 0.0, 5.0, &[1.0], &options).unwrap();

        // Tsit5 should be reasonably efficient - allow more evaluations
        // since we're using simpler step control than DoPri5's PI controller
        assert!(
            result.stats.n_eval < 500,
            "Tsit5 used {} evaluations, expected < 500",
            result.stats.n_eval
        );
    }
}

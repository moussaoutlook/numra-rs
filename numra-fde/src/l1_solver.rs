//! L1 scheme solver for fractional differential equations.
//!
//! The L1 scheme discretizes the Caputo derivative as:
//!
//! ```text
//! D^α y(tₙ) ≈ (Δt^(-α) / Γ(2-α)) Σⱼ₌₀ⁿ⁻¹ bⱼ (y(tₙ₋ⱼ) - y(tₙ₋ⱼ₋₁))
//! ```
//!
//! For the FDE: D^α y = f(t, y), we get an implicit equation:
//!
//! ```text
//! c * (yₙ - yₙ₋₁) + c * Σⱼ₌₁ⁿ⁻¹ bⱼ (yₙ₋ⱼ - yₙ₋ⱼ₋₁) = f(tₙ, yₙ)
//! ```
//!
//! This is solved using fixed-point iteration (for explicit f) or Newton's method.
//!
//! Author: Moussa Leblouba
//! Date: 5 March 2026
//! Modified: 2 May 2026

use crate::caputo::{caputo_weights, l1_coefficient};
use crate::system::{FdeOptions, FdeResult, FdeSolver, FdeStats, FdeSystem};
use numra_core::Scalar;

/// L1 scheme solver for Caputo fractional differential equations.
///
/// Convergence order: O(Δt^(2-α))
pub struct L1Solver;

impl<S: Scalar> FdeSolver<S> for L1Solver {
    fn solve<Sys: FdeSystem<S>>(
        system: &Sys,
        t0: S,
        tf: S,
        y0: &[S],
        options: &FdeOptions<S>,
    ) -> Result<FdeResult<S>, String> {
        let dim = system.dim();
        let alpha = system.alpha();

        if !system.is_valid_order() {
            return Err(format!(
                "Invalid fractional order α = {}. Must be in (0, 1].",
                alpha.to_f64()
            ));
        }

        if y0.len() != dim {
            return Err(format!(
                "Initial state dimension {} doesn't match system dimension {}",
                y0.len(),
                dim
            ));
        }

        let dt = options.dt;
        let n_steps = ((tf - t0) / dt).to_f64().ceil() as usize;

        if n_steps > options.max_steps {
            return Err(format!(
                "Required steps {} exceeds maximum {}",
                n_steps, options.max_steps
            ));
        }

        // Precompute L1 coefficient: c = Δt^(-α) / Γ(2-α)
        let c = l1_coefficient(dt, alpha);

        // Storage for solution history (needed for memory effect)
        let mut y_history: Vec<Vec<S>> = Vec::with_capacity(n_steps + 1);
        y_history.push(y0.to_vec());

        let mut t_out = vec![t0];
        let mut y_out = y0.to_vec();
        let mut stats = FdeStats::default();

        let mut f_buf = vec![S::ZERO; dim];

        for n in 1..=n_steps {
            let t = t0 + S::from_usize(n) * dt;

            // Compute weights for this step
            let b = caputo_weights(n, alpha);

            // Compute the history sum: Σⱼ₌₁ⁿ⁻¹ bⱼ (yₙ₋ⱼ - yₙ₋ⱼ₋₁)
            let mut history_sum = vec![S::ZERO; dim];
            for j in 1..n {
                let y_nmj = &y_history[n - j];
                let y_nmjm1 = &y_history[n - j - 1];
                for i in 0..dim {
                    history_sum[i] += b[j] * (y_nmj[i] - y_nmjm1[i]);
                }
            }

            // Get previous solution
            let y_prev = &y_history[n - 1];

            // Fixed-point iteration to solve:
            // c * b[0] * (yₙ - yₙ₋₁) + c * history_sum = f(tₙ, yₙ)
            // => yₙ = yₙ₋₁ + (f(tₙ, yₙ) - c * history_sum) / (c * b[0])
            //
            // For explicit f, we can use simple iteration.
            // Start with predictor: yₙ ≈ yₙ₋₁
            let mut y_new = y_prev.clone();

            let c_b0 = c * b[0];
            let inv_c_b0 = S::ONE / c_b0;

            for _iter in 0..options.max_iter {
                // Evaluate f at current guess
                system.rhs(t, &y_new, &mut f_buf);
                stats.n_rhs += 1;

                // Compute new estimate
                let mut y_next = vec![S::ZERO; dim];
                let mut converged = true;

                for i in 0..dim {
                    let rhs = f_buf[i] - c * history_sum[i];
                    y_next[i] = y_prev[i] + rhs * inv_c_b0;

                    // Check convergence
                    let diff = (y_next[i] - y_new[i]).abs();
                    let scale = options.tol * (S::ONE + y_new[i].abs());
                    if diff > scale {
                        converged = false;
                    }
                }

                y_new = y_next;

                if converged {
                    break;
                }
            }

            // Store solution
            y_history.push(y_new.clone());
            t_out.push(t);
            y_out.extend_from_slice(&y_new);
            stats.n_steps += 1;
        }

        Ok(FdeResult::new(t_out, y_out, dim, stats))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::caputo::mittag_leffler_1;

    /// Fractional relaxation: D^α y = -λy, y(0) = 1
    /// Exact solution: y(t) = E_α(-λ * t^α)
    struct FractionalRelaxation {
        lambda: f64,
        alpha: f64,
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

    #[test]
    fn test_l1_fractional_relaxation_alpha_1() {
        // When α = 1, reduces to ordinary ODE: y' = -y, y(0) = 1
        // Solution: y(t) = exp(-t)
        let system = FractionalRelaxation {
            lambda: 1.0,
            alpha: 1.0,
        };
        let options = FdeOptions::default().dt(0.01);

        let result = L1Solver::solve(&system, 0.0, 1.0, &[1.0], &options).expect("Solve failed");

        assert!(result.success);
        let y_final = result.y_final().unwrap()[0];
        let y_exact = (-1.0_f64).exp();

        // Should be close to exp(-1) ≈ 0.3679
        assert!((y_final - y_exact).abs() < 0.01);
    }

    #[test]
    fn test_l1_fractional_relaxation_alpha_half() {
        // D^0.5 y = -y, y(0) = 1
        // Solution: y(t) = E_{0.5}(-t^0.5)
        let system = FractionalRelaxation {
            lambda: 1.0,
            alpha: 0.5,
        };
        let options = FdeOptions::default().dt(0.01);

        let result = L1Solver::solve(&system, 0.0, 1.0, &[1.0], &options).expect("Solve failed");

        assert!(result.success);

        // Check a few points against Mittag-Leffler
        for (i, &t) in result.t.iter().enumerate().step_by(10) {
            if t > 0.0 {
                let y_computed = result.y_at(i)[0];
                let y_exact = mittag_leffler_1(-t.powf(0.5), 0.5);

                // L1 scheme has order (2-α) = 1.5, so error ~ O(dt^1.5)
                // With dt=0.01, expect error ~ 0.001
                let error = (y_computed - y_exact).abs();
                assert!(
                    error < 0.05,
                    "At t={}: computed={}, exact={}, error={}",
                    t,
                    y_computed,
                    y_exact,
                    error
                );
            }
        }
    }

    #[test]
    fn test_l1_2d_system() {
        struct TwoD;
        impl FdeSystem<f64> for TwoD {
            fn dim(&self) -> usize {
                2
            }
            fn alpha(&self) -> f64 {
                0.8
            }
            fn rhs(&self, _t: f64, y: &[f64], f: &mut [f64]) {
                f[0] = -y[0];
                f[1] = -2.0 * y[1];
            }
        }

        let options = FdeOptions::default().dt(0.01);
        let result = L1Solver::solve(&TwoD, 0.0, 1.0, &[1.0, 1.0], &options).expect("Solve failed");

        assert!(result.success);
        let y_final = result.y_final().unwrap();

        // Second component should decay faster
        assert!(y_final[0] > y_final[1]);
    }

    #[test]
    fn test_invalid_alpha() {
        struct InvalidAlpha;
        impl FdeSystem<f64> for InvalidAlpha {
            fn dim(&self) -> usize {
                1
            }
            fn alpha(&self) -> f64 {
                1.5
            } // Invalid: > 1
            fn rhs(&self, _t: f64, _y: &[f64], f: &mut [f64]) {
                f[0] = 0.0;
            }
        }

        let options = FdeOptions::default();
        let result = L1Solver::solve(&InvalidAlpha, 0.0, 1.0, &[1.0], &options);

        assert!(result.is_err());
    }

    #[test]
    fn test_l1_convergence_order() {
        // D^0.5 y = -y, y(0) = 1. L1 has order 2-alpha = 1.5.
        // Error ratio at halved dt should be ~2^1.5 ≈ 2.83.
        let system = FractionalRelaxation {
            lambda: 1.0,
            alpha: 0.5,
        };
        let tf = 1.0;

        // Coarse: dt = 0.02
        let opts_coarse = FdeOptions::default().dt(0.02);
        let res_coarse =
            L1Solver::solve(&system, 0.0, tf, &[1.0], &opts_coarse).expect("Coarse solve failed");
        let y_coarse = res_coarse.y_final().unwrap()[0];

        // Fine: dt = 0.01
        let opts_fine = FdeOptions::default().dt(0.01);
        let res_fine =
            L1Solver::solve(&system, 0.0, tf, &[1.0], &opts_fine).expect("Fine solve failed");
        let y_fine = res_fine.y_final().unwrap()[0];

        // Reference at very fine dt
        let opts_ref = FdeOptions::default().dt(0.001);
        let res_ref =
            L1Solver::solve(&system, 0.0, tf, &[1.0], &opts_ref).expect("Ref solve failed");
        let y_ref = res_ref.y_final().unwrap()[0];

        let err_coarse = (y_coarse - y_ref).abs();
        let err_fine = (y_fine - y_ref).abs();

        let ratio = err_coarse / err_fine;
        // Expected ratio ~ 2^1.5 ≈ 2.83, allow range [2.0, 4.0]
        assert!(
            ratio > 2.0 && ratio < 4.0,
            "Convergence ratio: got {}, expected ~2.83 (errors: coarse={}, fine={})",
            ratio,
            err_coarse,
            err_fine
        );
    }

    #[test]
    fn test_l1_alpha_near_1() {
        // alpha = 0.99 should produce results close to exp(-t)
        let system = FractionalRelaxation {
            lambda: 1.0,
            alpha: 0.99,
        };
        let options = FdeOptions::default().dt(0.01);

        let result = L1Solver::solve(&system, 0.0, 1.0, &[1.0], &options).expect("Solve failed");

        let y_final = result.y_final().unwrap()[0];
        let y_exact = (-1.0_f64).exp();

        assert!(
            (y_final - y_exact).abs() < 0.05,
            "alpha=0.99: got {}, expected ~{}",
            y_final,
            y_exact
        );
    }

    #[test]
    fn test_l1_zero_rhs() {
        // D^alpha y = 0, y(0) = 5.0 => y(t) = 5.0 for all t
        struct ZeroRhs;
        impl FdeSystem<f64> for ZeroRhs {
            fn dim(&self) -> usize {
                1
            }
            fn alpha(&self) -> f64 {
                0.7
            }
            fn rhs(&self, _t: f64, _y: &[f64], f: &mut [f64]) {
                f[0] = 0.0;
            }
        }

        let options = FdeOptions::default().dt(0.01);
        let result = L1Solver::solve(&ZeroRhs, 0.0, 1.0, &[5.0], &options).expect("Solve failed");

        for (i, &t) in result.t.iter().enumerate() {
            let y = result.y_at(i)[0];
            assert!(
                (y - 5.0).abs() < 1e-10,
                "At t={}: y should be 5.0, got {}",
                t,
                y
            );
        }
    }

    #[test]
    fn test_l1_dimension_mismatch() {
        // y0 has 2 elements but system has dim=1
        let system = FractionalRelaxation {
            lambda: 1.0,
            alpha: 0.5,
        };
        let options = FdeOptions::default();

        let result = L1Solver::solve(&system, 0.0, 1.0, &[1.0, 2.0], &options);
        assert!(result.is_err(), "Should error on dimension mismatch");
    }
}

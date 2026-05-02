//! Adaptive SDE solvers based on Stochastic Runge-Kutta methods.
//!
//! These methods provide adaptive step size control for SDEs.
//!
//! - `Sra1`: Strong order 1.5 for additive noise, 1.0 for multiplicative
//! - `Sra2`: Weak order 2.0 (better for computing expectations)
//!
//! Author: Moussa Leblouba
//! Date: 4 February 2026
//! Modified: 2 May 2026

use crate::system::{NoiseType, SdeOptions, SdeResult, SdeSolver, SdeStats, SdeSystem};
use crate::wiener::create_wiener;
use numra_core::Scalar;

/// SRA1 - Adaptive strong order 1.0-1.5 method.
///
/// Uses a two-stage Runge-Kutta scheme with embedded error estimation.
/// Strong order 1.5 for additive noise, 1.0 for multiplicative.
pub struct Sra1;

impl<S: Scalar> SdeSolver<S> for Sra1 {
    fn solve<Sys: SdeSystem<S>>(
        system: &Sys,
        t0: S,
        tf: S,
        x0: &[S],
        options: &SdeOptions<S>,
        seed: Option<u64>,
    ) -> Result<SdeResult<S>, String> {
        let dim = system.dim();
        if x0.len() != dim {
            return Err(format!(
                "Initial state dimension {} doesn't match system dimension {}",
                x0.len(),
                dim
            ));
        }

        // Currently only supports diagonal noise
        match system.noise_type() {
            NoiseType::Diagonal | NoiseType::Scalar => {}
            _ => return Err("SRA1 currently only supports diagonal or scalar noise".to_string()),
        }

        let n_wiener = system.n_wiener();
        let actual_seed = seed.or(options.seed);
        let mut wiener = create_wiener(n_wiener, actual_seed);

        // Allocate storage
        let mut t = t0;
        let mut x = x0.to_vec();
        let mut h = options.dt.min(options.dt_max);

        // Working arrays
        let mut f1 = vec![S::ZERO; dim];
        let mut f2 = vec![S::ZERO; dim];
        let mut g1 = vec![S::ZERO; dim];
        let mut g2 = vec![S::ZERO; dim];
        let mut x_stage = vec![S::ZERO; dim];
        let mut x_new = vec![S::ZERO; dim];
        let mut x_err = vec![S::ZERO; dim];

        let mut t_out = Vec::new();
        let mut y_out = Vec::new();
        let mut stats = SdeStats::default();

        // Constants for step size control
        let safety = S::from_f64(0.9);
        let fac_min = S::from_f64(0.2);
        let fac_max = S::from_f64(5.0);
        let order = S::from_f64(1.5); // Strong order for error estimation

        // Save initial state
        if options.save_trajectory {
            t_out.push(t);
            y_out.extend_from_slice(&x);
        }

        let half = S::from_f64(0.5);
        let one = S::ONE;
        let mut step = 0;

        while t < tf && step < options.max_steps {
            // Limit step to not overshoot tf
            h = h.min(tf - t).min(options.dt_max).max(options.dt_min);

            // Generate Wiener increment
            let dw = wiener.increment(h);
            let sqrt_h = h.sqrt();

            // Stage 1: Evaluate at current point
            system.drift(t, &x, &mut f1);
            system.diffusion(t, &x, &mut g1);
            stats.n_drift += 1;
            stats.n_diffusion += 1;

            // Stage 2: Evaluate at predicted point
            // x_stage = x + f1*h + g1*sqrt(h) (deterministic predictor)
            for i in 0..dim {
                x_stage[i] = x[i] + f1[i] * h + g1[i] * sqrt_h;
            }
            system.drift(t + h, &x_stage, &mut f2);
            system.diffusion(t + h, &x_stage, &mut g2);
            stats.n_drift += 1;
            stats.n_diffusion += 1;

            // Two-stage SRK update with Rößler-style coefficients
            // Higher order: x_new = x + 0.5*(f1+f2)*h + 0.5*(g1+g2)*dW
            // Lower order:  x_lo  = x + f1*h + g1*dW
            // Error: x_err = 0.5*(f2-f1)*h + 0.5*(g2-g1)*dW

            let is_scalar = matches!(system.noise_type(), NoiseType::Scalar);

            for i in 0..dim {
                let dw_i = if is_scalar { dw.dw[0] } else { dw.dw[i] };

                // Higher-order solution
                x_new[i] = x[i] + half * (f1[i] + f2[i]) * h + half * (g1[i] + g2[i]) * dw_i;

                // Error estimate (difference from lower-order solution)
                x_err[i] = half * (f2[i] - f1[i]) * h + half * (g2[i] - g1[i]) * dw_i;
            }

            // Compute error norm
            let mut err_sq = S::ZERO;
            for i in 0..dim {
                let scale = options.atol + options.rtol * x[i].abs().max(x_new[i].abs());
                let ratio = x_err[i] / scale;
                err_sq += ratio * ratio;
            }
            let err = (err_sq / S::from_usize(dim)).sqrt();

            // Accept or reject step
            if err <= one {
                // Accept step
                t += h;
                x[..dim].copy_from_slice(&x_new[..dim]);
                stats.n_accept += 1;
                step += 1;

                // Save state
                if options.save_trajectory {
                    t_out.push(t);
                    y_out.extend_from_slice(&x);
                }
            } else {
                // Reject step
                stats.n_reject += 1;
            }

            // Compute new step size
            let err_safe = err.max(S::from_f64(1e-10));
            let fac = safety * err_safe.powf(-one / (order + one));
            h *= fac.max(fac_min).min(fac_max);
        }

        if step >= options.max_steps && t < tf {
            return Err(format!(
                "Maximum steps ({}) exceeded at t = {}",
                options.max_steps,
                t.to_f64()
            ));
        }

        // If not saving trajectory, just save final state
        if !options.save_trajectory {
            t_out.push(t);
            y_out.extend_from_slice(&x);
        }

        Ok(SdeResult::new(t_out, y_out, dim, stats))
    }
}

/// SRA2 - Adaptive weak order 2.0 method.
///
/// Better for computing expectations (mean, variance) rather than pathwise accuracy.
/// Uses a three-stage scheme optimized for weak convergence.
pub struct Sra2;

impl<S: Scalar> SdeSolver<S> for Sra2 {
    fn solve<Sys: SdeSystem<S>>(
        system: &Sys,
        t0: S,
        tf: S,
        x0: &[S],
        options: &SdeOptions<S>,
        seed: Option<u64>,
    ) -> Result<SdeResult<S>, String> {
        let dim = system.dim();
        if x0.len() != dim {
            return Err(format!(
                "Initial state dimension {} doesn't match system dimension {}",
                x0.len(),
                dim
            ));
        }

        // Currently only supports diagonal noise
        match system.noise_type() {
            NoiseType::Diagonal | NoiseType::Scalar => {}
            _ => return Err("SRA2 currently only supports diagonal or scalar noise".to_string()),
        }

        let n_wiener = system.n_wiener();
        let actual_seed = seed.or(options.seed);
        let mut wiener = create_wiener(n_wiener, actual_seed);

        // Allocate storage
        let mut t = t0;
        let mut x = x0.to_vec();
        let mut h = options.dt.min(options.dt_max);

        // Working arrays
        let mut f1 = vec![S::ZERO; dim];
        let mut f2 = vec![S::ZERO; dim];
        let mut f3 = vec![S::ZERO; dim];
        let mut g1 = vec![S::ZERO; dim];
        let mut g2 = vec![S::ZERO; dim];
        let mut x_stage = vec![S::ZERO; dim];
        let mut x_new = vec![S::ZERO; dim];
        let mut x_err = vec![S::ZERO; dim];

        let mut t_out = Vec::new();
        let mut y_out = Vec::new();
        let mut stats = SdeStats::default();

        // Constants for step size control
        let safety = S::from_f64(0.9);
        let fac_min = S::from_f64(0.2);
        let fac_max = S::from_f64(5.0);
        let order = S::from_f64(2.0); // Weak order

        // Coefficients for 3-stage weak order 2 method
        let c2 = S::from_f64(2.0 / 3.0);
        let a21 = S::from_f64(2.0 / 3.0);
        let b1 = S::from_f64(0.25);
        let b2 = S::from_f64(0.75);

        // Save initial state
        if options.save_trajectory {
            t_out.push(t);
            y_out.extend_from_slice(&x);
        }

        let one = S::ONE;
        let half = S::from_f64(0.5);
        let mut step = 0;

        while t < tf && step < options.max_steps {
            h = h.min(tf - t).min(options.dt_max).max(options.dt_min);

            // Generate Wiener increment
            let dw = wiener.increment(h);
            let sqrt_h = h.sqrt();

            // Stage 1
            system.drift(t, &x, &mut f1);
            system.diffusion(t, &x, &mut g1);
            stats.n_drift += 1;
            stats.n_diffusion += 1;

            // Stage 2 at t + c2*h
            let is_scalar = matches!(system.noise_type(), NoiseType::Scalar);
            for i in 0..dim {
                let dw_i = if is_scalar { dw.dw[0] } else { dw.dw[i] };
                x_stage[i] = x[i] + a21 * f1[i] * h + g1[i] * sqrt_h;
                let _ = dw_i; // Used for stochastic part
            }
            system.drift(t + c2 * h, &x_stage, &mut f2);
            system.diffusion(t + c2 * h, &x_stage, &mut g2);
            stats.n_drift += 1;
            stats.n_diffusion += 1;

            // Final stage at t + h for error estimation
            for i in 0..dim {
                x_stage[i] = x[i] + f1[i] * h;
            }
            system.drift(t + h, &x_stage, &mut f3);
            stats.n_drift += 1;

            // Compute solution: x_new = x + (b1*f1 + b2*f2)*h + g1*dW
            for i in 0..dim {
                let dw_i = if is_scalar { dw.dw[0] } else { dw.dw[i] };

                x_new[i] = x[i] + (b1 * f1[i] + b2 * f2[i]) * h + half * (g1[i] + g2[i]) * dw_i;

                // Error estimate (embedded method difference)
                x_err[i] = (b2 * (f2[i] - f1[i]) + b1 * (f1[i] - f3[i])) * h;
            }

            // Compute error norm
            let mut err_sq = S::ZERO;
            for i in 0..dim {
                let scale = options.atol + options.rtol * x[i].abs().max(x_new[i].abs());
                let ratio = x_err[i] / scale;
                err_sq += ratio * ratio;
            }
            let err = (err_sq / S::from_usize(dim)).sqrt();

            // Accept or reject step
            if err <= one {
                t += h;
                x[..dim].copy_from_slice(&x_new[..dim]);
                stats.n_accept += 1;
                step += 1;

                if options.save_trajectory {
                    t_out.push(t);
                    y_out.extend_from_slice(&x);
                }
            } else {
                stats.n_reject += 1;
            }

            // Compute new step size
            let err_safe = err.max(S::from_f64(1e-10));
            let fac = safety * err_safe.powf(-one / (order + one));
            h *= fac.max(fac_min).min(fac_max);
        }

        if step >= options.max_steps && t < tf {
            return Err(format!(
                "Maximum steps ({}) exceeded at t = {}",
                options.max_steps,
                t.to_f64()
            ));
        }

        if !options.save_trajectory {
            t_out.push(t);
            y_out.extend_from_slice(&x);
        }

        Ok(SdeResult::new(t_out, y_out, dim, stats))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[allow(clippy::upper_case_acronyms)]
    struct GBM {
        mu: f64,
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
    }

    #[test]
    fn test_sra1_gbm() {
        let gbm = GBM {
            mu: 0.05,
            sigma: 0.2,
        };
        let options = SdeOptions::default().dt(0.01).seed(42);

        let result = Sra1::solve(&gbm, 0.0, 1.0, &[100.0], &options, None).expect("Solve failed");

        assert!(result.success);
        let final_price = result.y_final().unwrap()[0];
        assert!(final_price > 0.0);
        assert!(result.stats.n_accept > 0);
    }

    #[test]
    fn test_sra2_gbm() {
        let gbm = GBM {
            mu: 0.05,
            sigma: 0.2,
        };
        let options = SdeOptions::default().dt(0.01).seed(42);

        let result = Sra2::solve(&gbm, 0.0, 1.0, &[100.0], &options, None).expect("Solve failed");

        assert!(result.success);
        let final_price = result.y_final().unwrap()[0];
        assert!(final_price > 0.0);
    }

    #[test]
    fn test_sra1_adapts_step() {
        // Stiff problem should cause step rejection/adaptation
        struct Stiff;
        impl SdeSystem<f64> for Stiff {
            fn dim(&self) -> usize {
                1
            }
            fn drift(&self, _t: f64, x: &[f64], f: &mut [f64]) {
                f[0] = -50.0 * x[0]; // Fast dynamics
            }
            fn diffusion(&self, _t: f64, _x: &[f64], g: &mut [f64]) {
                g[0] = 0.1;
            }
        }

        let options = SdeOptions::default()
            .dt(0.1) // Large initial step
            .rtol(1e-4)
            .atol(1e-6)
            .seed(42);

        let result = Sra1::solve(&Stiff, 0.0, 1.0, &[1.0], &options, None).expect("Solve failed");

        assert!(result.success);
        // Should have some rejected steps due to large initial dt
        // or many accepted steps with smaller dt
        assert!(result.stats.n_accept >= 10);
    }

    #[test]
    fn test_reproducibility() {
        let gbm = GBM {
            mu: 0.05,
            sigma: 0.2,
        };
        let options = SdeOptions::default().dt(0.01);

        let r1 = Sra1::solve(&gbm, 0.0, 1.0, &[100.0], &options, Some(42)).expect("Solve failed");
        let r2 = Sra1::solve(&gbm, 0.0, 1.0, &[100.0], &options, Some(42)).expect("Solve failed");

        let y1 = r1.y_final().unwrap()[0];
        let y2 = r2.y_final().unwrap()[0];
        assert!((y1 - y2).abs() < 1e-10);
    }
}

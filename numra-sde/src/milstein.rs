//! Milstein method for SDEs.
//!
//! Higher-order SDE solver with strong order 1.0.
//!
//! For an SDE: dX = f(t,X) dt + g(t,X) dW
//!
//! The Milstein update is:
//! X_{n+1} = X_n + f(t_n, X_n) * dt + g(t_n, X_n) * ΔW_n
//!           + 0.5 * g(t_n, X_n) * g'(t_n, X_n) * (ΔW_n² - dt)
//!
//! The extra term accounts for the Itô-Stratonovich correction.
//!
//! Author: Moussa Leblouba
//! Date: 4 February 2026
//! Modified: 2 May 2026

use crate::system::{NoiseType, SdeOptions, SdeResult, SdeSolver, SdeStats, SdeSystem};
use crate::wiener::create_wiener;
use numra_core::Scalar;

/// Milstein SDE solver.
///
/// Fixed time step solver with strong order 1.0 and weak order 1.0.
/// Requires diffusion derivative ∂g/∂x (computed via finite differences by default).
pub struct Milstein;

impl<S: Scalar> SdeSolver<S> for Milstein {
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

        // Milstein currently only supports diagonal noise
        match system.noise_type() {
            NoiseType::Diagonal => {}
            _ => return Err("Milstein currently only supports diagonal noise".to_string()),
        }

        let dt = options.dt;
        let n_wiener = system.n_wiener();
        let actual_seed = seed.or(options.seed);
        let mut wiener = create_wiener(n_wiener, actual_seed);

        // Allocate storage
        let mut t = t0;
        let mut x = x0.to_vec();
        let mut f = vec![S::ZERO; dim];
        let mut g = vec![S::ZERO; dim];
        let mut gdg = vec![S::ZERO; dim]; // g * dg/dx

        let mut t_out = Vec::new();
        let mut y_out = Vec::new();
        let mut stats = SdeStats::default();

        // Save initial state
        if options.save_trajectory {
            t_out.push(t);
            y_out.extend_from_slice(&x);
        }

        let half = S::from_f64(0.5);
        let mut step = 0;
        while t < tf && step < options.max_steps {
            // Adjust final step
            let h = dt.min(tf - t);

            // Evaluate drift, diffusion, and diffusion derivative
            system.drift(t, &x, &mut f);
            system.diffusion(t, &x, &mut g);
            system.diffusion_derivative(t, &x, &mut gdg);
            stats.n_drift += 1;
            stats.n_diffusion += 2; // diffusion_derivative calls diffusion

            // Generate Wiener increment
            let dw = wiener.increment(h);

            // Milstein update for diagonal noise:
            // X_{n+1} = X_n + f * dt + g * ΔW + 0.5 * g * g' * (ΔW² - dt)
            for i in 0..dim {
                let dw_i = dw.dw[i];
                let correction = half * gdg[i] * (dw_i * dw_i - h);
                x[i] += f[i] * h + g[i] * dw_i + correction;
            }

            t += h;
            step += 1;
            stats.n_accept += 1;

            // Save state
            if options.save_trajectory {
                t_out.push(t);
                y_out.extend_from_slice(&x);
            }
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Geometric Brownian Motion: dS = μS dt + σS dW
    /// For GBM, g(x) = σx, so g'(x) = σ, and g*g' = σ²x
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

        fn diffusion_derivative(&self, _t: f64, x: &[f64], gdg: &mut [f64]) {
            // g = σx, dg/dx = σ, so g * dg/dx = σ²x
            gdg[0] = self.sigma * self.sigma * x[0];
        }
    }

    #[test]
    fn test_milstein_gbm() {
        let gbm = GBM {
            mu: 0.05,
            sigma: 0.2,
        };
        let options = SdeOptions::default().dt(0.001).seed(42);

        let result =
            Milstein::solve(&gbm, 0.0, 1.0, &[100.0], &options, None).expect("Solve failed");

        assert!(result.success);
        assert!(!result.t.is_empty());
        let final_price = result.y_final().unwrap()[0];
        assert!(final_price > 0.0);
    }

    #[test]
    fn test_milstein_vs_euler_maruyama() {
        // For the same seed, Milstein should give different (more accurate) results
        let gbm = GBM {
            mu: 0.05,
            sigma: 0.3,
        };
        let options = SdeOptions::default().dt(0.01).seed(42);

        let em_result = crate::EulerMaruyama::solve(&gbm, 0.0, 1.0, &[100.0], &options, None)
            .expect("EM solve failed");
        let mil_result = Milstein::solve(&gbm, 0.0, 1.0, &[100.0], &options, None)
            .expect("Milstein solve failed");

        let em_final = em_result.y_final().unwrap()[0];
        let mil_final = mil_result.y_final().unwrap()[0];

        // They should be different due to the correction term
        // (same random numbers, different algorithm)
        assert!((em_final - mil_final).abs() > 0.01);
    }

    #[test]
    fn test_milstein_additive_noise() {
        // For additive noise (g constant), Milstein = Euler-Maruyama
        struct Additive;
        impl SdeSystem<f64> for Additive {
            fn dim(&self) -> usize {
                1
            }
            fn drift(&self, _t: f64, x: &[f64], f: &mut [f64]) {
                f[0] = -x[0];
            }
            fn diffusion(&self, _t: f64, _x: &[f64], g: &mut [f64]) {
                g[0] = 0.5; // Constant diffusion
            }
            fn diffusion_derivative(&self, _t: f64, _x: &[f64], gdg: &mut [f64]) {
                gdg[0] = 0.0; // dg/dx = 0, so g*g' = 0
            }
        }

        let options = SdeOptions::default().dt(0.01).seed(42);
        let em = crate::EulerMaruyama::solve(&Additive, 0.0, 1.0, &[1.0], &options, None)
            .expect("EM solve failed");
        let mil = Milstein::solve(&Additive, 0.0, 1.0, &[1.0], &options, None)
            .expect("Milstein solve failed");

        // For additive noise, results should be identical
        let em_final = em.y_final().unwrap()[0];
        let mil_final = mil.y_final().unwrap()[0];
        assert!((em_final - mil_final).abs() < 1e-10);
    }

    #[test]
    fn test_reproducibility() {
        let gbm = GBM {
            mu: 0.05,
            sigma: 0.2,
        };
        let options = SdeOptions::default().dt(0.01);

        let r1 =
            Milstein::solve(&gbm, 0.0, 1.0, &[100.0], &options, Some(42)).expect("Solve failed");
        let r2 =
            Milstein::solve(&gbm, 0.0, 1.0, &[100.0], &options, Some(42)).expect("Solve failed");

        let y1 = r1.y_final().unwrap()[0];
        let y2 = r2.y_final().unwrap()[0];
        assert!((y1 - y2).abs() < 1e-10);
    }
}

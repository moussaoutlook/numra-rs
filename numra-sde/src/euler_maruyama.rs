//! Euler-Maruyama method for SDEs.
//!
//! The simplest SDE solver with strong order 0.5.
//!
//! For an SDE: dX = f(t,X) dt + g(t,X) dW
//!
//! The update is:
//! X_{n+1} = X_n + f(t_n, X_n) * dt + g(t_n, X_n) * ΔW_n
//!
//! where ΔW_n ~ N(0, dt).
//!
//! Author: Moussa Leblouba
//! Date: 4 February 2026
//! Modified: 2 May 2026

use crate::system::{NoiseType, SdeOptions, SdeResult, SdeSolver, SdeStats, SdeSystem};
use crate::wiener::create_wiener;
use numra_core::Scalar;

/// Euler-Maruyama SDE solver.
///
/// Fixed time step solver with strong order 0.5 and weak order 1.0.
pub struct EulerMaruyama;

impl<S: Scalar> SdeSolver<S> for EulerMaruyama {
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

        let dt = options.dt;
        let n_wiener = system.n_wiener();
        let actual_seed = seed.or(options.seed);
        let mut wiener = create_wiener(n_wiener, actual_seed);

        // Allocate storage
        let mut t = t0;
        let mut x = x0.to_vec();
        let mut f = vec![S::ZERO; dim];
        let mut g = match system.noise_type() {
            NoiseType::Diagonal => vec![S::ZERO; dim],
            NoiseType::Scalar => vec![S::ZERO; dim],
            NoiseType::General { n_wiener } => vec![S::ZERO; dim * n_wiener],
        };

        let mut t_out = Vec::new();
        let mut y_out = Vec::new();
        let mut stats = SdeStats::default();

        // Save initial state
        if options.save_trajectory {
            t_out.push(t);
            y_out.extend_from_slice(&x);
        }

        let mut step = 0;
        while t < tf && step < options.max_steps {
            // Adjust final step
            let h = dt.min(tf - t);

            // Evaluate drift and diffusion
            system.drift(t, &x, &mut f);
            system.diffusion(t, &x, &mut g);
            stats.n_drift += 1;
            stats.n_diffusion += 1;

            // Generate Wiener increment
            let dw = wiener.increment(h);

            // Euler-Maruyama update
            match system.noise_type() {
                NoiseType::Diagonal => {
                    for i in 0..dim {
                        x[i] += f[i] * h + g[i] * dw.dw[i];
                    }
                }
                NoiseType::Scalar => {
                    for i in 0..dim {
                        x[i] += f[i] * h + g[i] * dw.dw[0];
                    }
                }
                NoiseType::General { n_wiener } => {
                    for i in 0..dim {
                        let mut noise_sum = S::ZERO;
                        for j in 0..n_wiener {
                            noise_sum += g[i * n_wiener + j] * dw.dw[j];
                        }
                        x[i] += f[i] * h + noise_sum;
                    }
                }
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
    fn test_euler_maruyama_gbm() {
        let gbm = GBM {
            mu: 0.05,
            sigma: 0.2,
        };
        let options = SdeOptions::default().dt(0.001).seed(42);

        let result =
            EulerMaruyama::solve(&gbm, 0.0, 1.0, &[100.0], &options, None).expect("Solve failed");

        assert!(result.success);
        assert!(!result.t.is_empty());
        let final_price = result.y_final().unwrap()[0];
        // Price should be positive
        assert!(final_price > 0.0);
        // With these parameters, expected value is S0 * exp(μT) ≈ 105.13
        // Actual path can vary, but should be in reasonable range
        assert!(final_price > 50.0 && final_price < 200.0);
    }

    /// Ornstein-Uhlenbeck: dX = θ(μ - X) dt + σ dW
    struct OrnsteinUhlenbeck {
        theta: f64,
        mu: f64,
        sigma: f64,
    }

    impl SdeSystem<f64> for OrnsteinUhlenbeck {
        fn dim(&self) -> usize {
            1
        }

        fn drift(&self, _t: f64, x: &[f64], f: &mut [f64]) {
            f[0] = self.theta * (self.mu - x[0]);
        }

        fn diffusion(&self, _t: f64, _x: &[f64], g: &mut [f64]) {
            g[0] = self.sigma;
        }
    }

    #[test]
    fn test_euler_maruyama_ou() {
        let ou = OrnsteinUhlenbeck {
            theta: 1.0,
            mu: 0.0,
            sigma: 0.5,
        };
        let options = SdeOptions::default().dt(0.01).seed(123);

        let result =
            EulerMaruyama::solve(&ou, 0.0, 10.0, &[1.0], &options, None).expect("Solve failed");

        assert!(result.success);
        // OU process should mean-revert towards μ = 0
        let final_x = result.y_final().unwrap()[0];
        // After long time, should be close to mean (with some noise)
        assert!(final_x.abs() < 3.0); // Within ~6 sigma
    }

    #[test]
    fn test_euler_maruyama_2d() {
        struct TwoD;
        impl SdeSystem<f64> for TwoD {
            fn dim(&self) -> usize {
                2
            }
            fn drift(&self, _t: f64, x: &[f64], f: &mut [f64]) {
                f[0] = -x[0];
                f[1] = -x[1];
            }
            fn diffusion(&self, _t: f64, _x: &[f64], g: &mut [f64]) {
                g[0] = 0.1;
                g[1] = 0.1;
            }
        }

        let options = SdeOptions::default().dt(0.01).seed(42);
        let result = EulerMaruyama::solve(&TwoD, 0.0, 1.0, &[1.0, 2.0], &options, None)
            .expect("Solve failed");

        assert!(result.success);
        assert_eq!(result.dim, 2);
    }

    #[test]
    fn test_reproducibility() {
        let gbm = GBM {
            mu: 0.05,
            sigma: 0.2,
        };
        let options = SdeOptions::default().dt(0.01);

        let r1 = EulerMaruyama::solve(&gbm, 0.0, 1.0, &[100.0], &options, Some(42))
            .expect("Solve failed");
        let r2 = EulerMaruyama::solve(&gbm, 0.0, 1.0, &[100.0], &options, Some(42))
            .expect("Solve failed");

        let y1 = r1.y_final().unwrap()[0];
        let y2 = r2.y_final().unwrap()[0];
        assert!((y1 - y2).abs() < 1e-10);
    }
}

//! Prony series (Sum of Exponentials) solver for IDEs.
//!
//! For kernels of the form K(τ) = Σᵢ aᵢ exp(-bᵢ τ), the memory integral
//! can be computed recursively without storing full history:
//!
//! ```text
//! Iᵢ(t) = ∫₀ᵗ aᵢ exp(-bᵢ(t-s)) y(s) ds
//! ```
//!
//! satisfies the ODE:
//! ```text
//! I'ᵢ = aᵢ y - bᵢ Iᵢ,  Iᵢ(0) = 0
//! ```
//!
//! This reduces memory from O(N) to O(M) where M is the number of exponential terms.
//!
//! Author: Moussa Leblouba
//! Date: 5 March 2026
//! Modified: 2 May 2026

use crate::kernels::PronyKernel;
use crate::system::{IdeOptions, IdeResult, IdeStats};
use numra_core::Scalar;

/// System for Prony solver: y' = f(t, y) + Σᵢ Iᵢ.
///
/// The kernel is implicitly defined by the Prony series.
pub trait PronySystem<S: Scalar> {
    /// Dimension of the state space.
    fn dim(&self) -> usize;

    /// Evaluate the local right-hand side f(t, y).
    fn rhs(&self, t: S, y: &[S], f: &mut [S]);

    /// Get the Prony kernel for computing memory effects.
    fn kernel(&self) -> &PronyKernel<S>;

    /// Optional coupling matrix: how each kernel term affects the state.
    ///
    /// By default, assumes kernel affects all components equally: I(t) * y(t).
    /// For more complex coupling, override this method.
    ///
    /// Returns: `coupling[i][k]` = coefficient for how I_k affects dy_i/dt
    fn coupling(&self) -> Option<Vec<Vec<S>>> {
        None
    }
}

/// Efficient solver for IDEs with sum-of-exponentials (Prony) kernels.
///
/// Uses the recursive property of exponential integrals to avoid
/// storing full solution history. Memory complexity is O(dim × n_terms)
/// instead of O(dim × n_steps).
pub struct PronySolver;

impl PronySolver {
    /// Solve an IDE with Prony kernel.
    pub fn solve<S: Scalar, Sys: PronySystem<S>>(
        system: &Sys,
        t0: S,
        tf: S,
        y0: &[S],
        options: &IdeOptions<S>,
    ) -> Result<IdeResult<S>, String> {
        let dim = system.dim();
        let kernel = system.kernel();
        let n_terms = kernel.num_terms();

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

        // State: y and auxiliary integrals I[k][i] for each term k and dimension i
        let mut y = y0.to_vec();
        let mut integrals: Vec<Vec<S>> = vec![vec![S::ZERO; dim]; n_terms];

        let mut t_out = vec![t0];
        let mut y_out = y0.to_vec();
        let mut stats = IdeStats::default();

        let mut t = t0;
        let mut f_buf = vec![S::ZERO; dim];

        // Get coupling matrix or use default
        let coupling = system.coupling();

        let half = S::from_f64(0.5);
        let sixth = S::ONE / S::from_f64(6.0);
        let two = S::from_f64(2.0);

        for _n in 1..=n_steps {
            let t_new = t + dt;

            // RK4 for the extended system [y, I_1, I_2, ...]

            // Stage 1: k1
            let (k1_y, k1_i) =
                compute_derivatives(system, t, &y, &integrals, &coupling, &mut f_buf, &mut stats);

            // Stage 2: k2 at t + dt/2
            let y_mid1: Vec<S> = y
                .iter()
                .zip(k1_y.iter())
                .map(|(&yi, &ki)| yi + half * dt * ki)
                .collect();
            let i_mid1: Vec<Vec<S>> = integrals
                .iter()
                .zip(k1_i.iter())
                .map(|(ii, ki)| {
                    ii.iter()
                        .zip(ki.iter())
                        .map(|(&ij, &kij)| ij + half * dt * kij)
                        .collect()
                })
                .collect();
            let (k2_y, k2_i) = compute_derivatives(
                system,
                t + half * dt,
                &y_mid1,
                &i_mid1,
                &coupling,
                &mut f_buf,
                &mut stats,
            );

            // Stage 3: k3 at t + dt/2
            let y_mid2: Vec<S> = y
                .iter()
                .zip(k2_y.iter())
                .map(|(&yi, &ki)| yi + half * dt * ki)
                .collect();
            let i_mid2: Vec<Vec<S>> = integrals
                .iter()
                .zip(k2_i.iter())
                .map(|(ii, ki)| {
                    ii.iter()
                        .zip(ki.iter())
                        .map(|(&ij, &kij)| ij + half * dt * kij)
                        .collect()
                })
                .collect();
            let (k3_y, k3_i) = compute_derivatives(
                system,
                t + half * dt,
                &y_mid2,
                &i_mid2,
                &coupling,
                &mut f_buf,
                &mut stats,
            );

            // Stage 4: k4 at t + dt
            let y_end: Vec<S> = y
                .iter()
                .zip(k3_y.iter())
                .map(|(&yi, &ki)| yi + dt * ki)
                .collect();
            let i_end: Vec<Vec<S>> = integrals
                .iter()
                .zip(k3_i.iter())
                .map(|(ii, ki)| {
                    ii.iter()
                        .zip(ki.iter())
                        .map(|(&ij, &kij)| ij + dt * kij)
                        .collect()
                })
                .collect();
            let (k4_y, k4_i) = compute_derivatives(
                system,
                t + dt,
                &y_end,
                &i_end,
                &coupling,
                &mut f_buf,
                &mut stats,
            );

            // RK4 combination for y
            for i in 0..dim {
                y[i] += sixth * dt * (k1_y[i] + two * k2_y[i] + two * k3_y[i] + k4_y[i]);
            }

            // RK4 combination for integrals
            for k in 0..n_terms {
                for i in 0..dim {
                    integrals[k][i] += sixth
                        * dt
                        * (k1_i[k][i] + two * k2_i[k][i] + two * k3_i[k][i] + k4_i[k][i]);
                }
            }

            // Store output
            t_out.push(t_new);
            y_out.extend_from_slice(&y);
            stats.n_steps += 1;

            t = t_new;
        }

        Ok(IdeResult::new(t_out, y_out, dim, stats))
    }
}

/// Compute derivatives for y and all integral terms.
fn compute_derivatives<S: Scalar, Sys: PronySystem<S>>(
    system: &Sys,
    t: S,
    y: &[S],
    integrals: &[Vec<S>],
    coupling: &Option<Vec<Vec<S>>>,
    f_buf: &mut [S],
    stats: &mut IdeStats,
) -> (Vec<S>, Vec<Vec<S>>) {
    let dim = y.len();
    let kernel = system.kernel();
    let n_terms = kernel.num_terms();

    // Compute local RHS
    system.rhs(t, y, f_buf);
    stats.n_rhs += 1;

    // Derivative of y: f(t,y) + sum of integrals
    let mut dy = f_buf.to_vec();

    if let Some(c) = coupling {
        // Custom coupling: dy[i] += sum_k c[i][k] * I_k[i]
        for i in 0..dim {
            for k in 0..n_terms {
                dy[i] += c[i][k] * integrals[k][i];
            }
        }
    } else {
        // Default: dy[i] += sum_k I_k[i]
        for i in 0..dim {
            for integral in integrals.iter().take(n_terms) {
                dy[i] += integral[i];
            }
        }
    }

    // Derivative of integrals: dI_k[i]/dt = a_k * y[i] - b_k * I_k[i]
    let mut di: Vec<Vec<S>> = Vec::with_capacity(n_terms);
    for (k, integral) in integrals.iter().enumerate().take(n_terms) {
        let a_k = kernel.amplitudes[k];
        let b_k = kernel.rates[k];
        let mut di_k = vec![S::ZERO; dim];
        for i in 0..dim {
            di_k[i] = a_k * y[i] - b_k * integral[i];
        }
        di.push(di_k);
        stats.n_kernel += dim;
    }

    (dy, di)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Viscoelastic material: y' = -k*y + ∫₀ᵗ a*exp(-b*(t-s)) * y(s) ds
    struct Viscoelastic {
        k: f64,
        kernel: PronyKernel<f64>,
    }

    impl Viscoelastic {
        fn new(k: f64, a: f64, b: f64) -> Self {
            Self {
                k,
                kernel: PronyKernel::single(a, b),
            }
        }
    }

    impl PronySystem<f64> for Viscoelastic {
        fn dim(&self) -> usize {
            1
        }

        fn rhs(&self, _t: f64, y: &[f64], f: &mut [f64]) {
            f[0] = -self.k * y[0];
        }

        fn kernel(&self) -> &PronyKernel<f64> {
            &self.kernel
        }
    }

    #[test]
    fn test_prony_viscoelastic() {
        let system = Viscoelastic::new(1.0, 0.5, 0.3);
        let options = IdeOptions::default().dt(0.01);

        let result = PronySolver::solve(&system, 0.0, 2.0, &[1.0], &options).expect("Solve failed");

        assert!(result.success);

        // Solution should be smooth and bounded
        let y_final = result.y_final().unwrap()[0];
        assert!(y_final > 0.0, "Solution should remain positive");
        assert!(y_final < 1.0, "Solution should decay");

        // Memory effect should slow down decay compared to pure exponential
        // Pure y' = -y gives y(2) = exp(-2) ≈ 0.135
        // With memory integral adding back, should be higher
        assert!(
            y_final > 0.135,
            "Memory should slow decay: y_final = {}",
            y_final
        );
    }

    #[test]
    fn test_prony_two_term() {
        // Two-term Prony series (generalized Maxwell model)
        struct TwoTermMaxwell {
            kernel: PronyKernel<f64>,
        }

        impl PronySystem<f64> for TwoTermMaxwell {
            fn dim(&self) -> usize {
                1
            }

            fn rhs(&self, _t: f64, y: &[f64], f: &mut [f64]) {
                f[0] = -2.0 * y[0]; // Strong local damping
            }

            fn kernel(&self) -> &PronyKernel<f64> {
                &self.kernel
            }
        }

        let system = TwoTermMaxwell {
            kernel: PronyKernel::two_term(0.8, 0.5, 0.4, 2.0),
        };
        let options = IdeOptions::default().dt(0.01);

        let result = PronySolver::solve(&system, 0.0, 3.0, &[1.0], &options).expect("Solve failed");

        assert!(result.success);

        // Check solution at various points
        for (i, &t) in result.t.iter().enumerate() {
            let y = result.y_at(i)[0];
            assert!(y.is_finite(), "Solution should be finite at t={}", t);
            assert!(
                y >= 0.0 || y.abs() < 0.1,
                "Solution should be non-negative or small negative at t={}",
                t
            );
        }
    }

    #[test]
    fn test_prony_2d_system() {
        struct TwoDProny {
            kernel: PronyKernel<f64>,
        }

        impl PronySystem<f64> for TwoDProny {
            fn dim(&self) -> usize {
                2
            }

            fn rhs(&self, _t: f64, y: &[f64], f: &mut [f64]) {
                f[0] = -y[0] + 0.1 * y[1];
                f[1] = -0.5 * y[1];
            }

            fn kernel(&self) -> &PronyKernel<f64> {
                &self.kernel
            }
        }

        let system = TwoDProny {
            kernel: PronyKernel::single(0.3, 0.5),
        };
        let options = IdeOptions::default().dt(0.01);

        let result =
            PronySolver::solve(&system, 0.0, 2.0, &[1.0, 1.0], &options).expect("Solve failed");

        assert!(result.success);
        let y_final = result.y_final().unwrap();
        assert_eq!(y_final.len(), 2);
    }

    #[test]
    fn test_prony_efficiency() {
        // The Prony solver should have constant memory per step
        let system = Viscoelastic::new(1.0, 0.5, 0.3);

        let options_short = IdeOptions::default().dt(0.01);
        let result_short = PronySolver::solve(&system, 0.0, 1.0, &[1.0], &options_short)
            .expect("Short solve failed");

        let options_long = IdeOptions::default().dt(0.01);
        let result_long = PronySolver::solve(&system, 0.0, 10.0, &[1.0], &options_long)
            .expect("Long solve failed");

        // Both should succeed - Prony doesn't accumulate memory errors
        assert!(result_short.success);
        assert!(result_long.success);

        // Kernel evaluations scale linearly with steps, not quadratically
        // (Unlike full quadrature which is O(n²))
        let ratio = result_long.stats.n_kernel as f64 / result_short.stats.n_kernel as f64;
        // Ratio should be close to 10 (time ratio), not 100 (quadratic)
        assert!(
            ratio < 15.0,
            "Kernel evals should scale linearly: ratio = {}",
            ratio
        );
    }

    #[test]
    fn test_prony_dimension_mismatch() {
        let system = Viscoelastic::new(1.0, 0.5, 0.3);
        let options = IdeOptions::default().dt(0.01);

        // y0 has 2 elements but system dim is 1
        let result = PronySolver::solve(&system, 0.0, 1.0, &[1.0, 2.0], &options);
        assert!(result.is_err());
        let msg = result.unwrap_err();
        assert!(msg.contains("dimension"), "Error message: {}", msg);
    }

    #[test]
    fn test_prony_max_steps_exceeded() {
        let system = Viscoelastic::new(1.0, 0.5, 0.3);
        // Very small dt with tiny max_steps to trigger error
        let options = IdeOptions::default().dt(0.001).max_steps(5);

        let result = PronySolver::solve(&system, 0.0, 1.0, &[1.0], &options);
        assert!(result.is_err());
        let msg = result.unwrap_err();
        assert!(msg.contains("exceeds maximum"), "Error message: {}", msg);
    }

    #[test]
    fn test_prony_zero_kernel() {
        // Kernel with amplitude 0 => no memory => pure ODE y' = -y
        struct PureDecay {
            kernel: PronyKernel<f64>,
        }

        impl PronySystem<f64> for PureDecay {
            fn dim(&self) -> usize {
                1
            }

            fn rhs(&self, _t: f64, y: &[f64], f: &mut [f64]) {
                f[0] = -y[0];
            }

            fn kernel(&self) -> &PronyKernel<f64> {
                &self.kernel
            }
        }

        let system = PureDecay {
            kernel: PronyKernel::single(0.0, 1.0),
        };
        let options = IdeOptions::default().dt(0.001);

        let result = PronySolver::solve(&system, 0.0, 1.0, &[1.0], &options).expect("Solve failed");

        let y_final = result.y_final().unwrap()[0];
        let expected = (-1.0_f64).exp(); // exp(-1) ≈ 0.3679
        assert!(
            (y_final - expected).abs() < 1e-4,
            "Zero kernel Prony should match pure ODE: got {}, expected {}",
            y_final,
            expected
        );
    }
}

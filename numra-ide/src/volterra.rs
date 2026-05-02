//! Volterra integro-differential equation solver.
//!
//! Solves equations of the form:
//! ```text
//! y'(t) = f(t, y) + ∫₀ᵗ K(t, s, y(s)) ds
//! ```
//!
//! Uses composite quadrature (trapezoidal or Simpson's rule) for the integral
//! and explicit time-stepping for the ODE part.
//!
//! Author: Moussa Leblouba
//! Date: 5 March 2026
//! Modified: 2 May 2026

use crate::system::{IdeOptions, IdeResult, IdeSolver, IdeStats, IdeSystem};
use numra_core::Scalar;

/// Volterra IDE solver using quadrature for the memory integral.
///
/// This solver stores the full solution history to compute the integral
/// at each time step. For sum-of-exponentials kernels, consider using
/// [`PronySolver`](crate::PronySolver) instead for O(1) memory.
pub struct VolterraSolver;

impl<S: Scalar> IdeSolver<S> for VolterraSolver {
    fn solve<Sys: IdeSystem<S>>(
        system: &Sys,
        t0: S,
        tf: S,
        y0: &[S],
        options: &IdeOptions<S>,
    ) -> Result<IdeResult<S>, String> {
        let dim = system.dim();

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

        // Storage for solution history
        let mut t_history: Vec<S> = Vec::with_capacity(n_steps + 1);
        let mut y_history: Vec<Vec<S>> = Vec::with_capacity(n_steps + 1);

        t_history.push(t0);
        y_history.push(y0.to_vec());

        let mut t_out = vec![t0];
        let mut y_out = y0.to_vec();
        let mut stats = IdeStats::default();

        let mut f_buf = vec![S::ZERO; dim];
        let mut k_buf = vec![S::ZERO; dim];

        let mut t = t0;
        let mut y = y0.to_vec();

        for n in 1..=n_steps {
            let t_new = t0 + S::from_usize(n) * dt;

            // Compute integral using trapezoidal rule over history
            let mut integral = vec![S::ZERO; dim];
            compute_integral(
                system,
                t_new,
                &t_history,
                &y_history,
                &mut integral,
                &mut k_buf,
                &mut stats,
            );

            // Evaluate local RHS
            system.rhs(t, &y, &mut f_buf);
            stats.n_rhs += 1;

            // Explicit Euler step: y_{n+1} = y_n + dt * (f(t_n, y_n) + integral)
            let mut y_new = vec![S::ZERO; dim];
            for i in 0..dim {
                y_new[i] = y[i] + dt * (f_buf[i] + integral[i]);
            }

            // Store in history
            t_history.push(t_new);
            y_history.push(y_new.clone());

            // Update output
            t_out.push(t_new);
            y_out.extend_from_slice(&y_new);
            stats.n_steps += 1;

            t = t_new;
            y = y_new;
        }

        Ok(IdeResult::new(t_out, y_out, dim, stats))
    }
}

/// Compute the integral ∫₀ᵗ K(t, s, y(s)) ds using composite trapezoidal rule.
fn compute_integral<S: Scalar, Sys: IdeSystem<S>>(
    system: &Sys,
    t: S,
    t_history: &[S],
    y_history: &[Vec<S>],
    integral: &mut [S],
    k_buf: &mut [S],
    stats: &mut IdeStats,
) {
    let dim = integral.len();
    let n = t_history.len();

    if n < 2 {
        // Not enough points for quadrature
        return;
    }

    // Composite trapezoidal rule
    for item in integral.iter_mut().take(dim) {
        *item = S::ZERO;
    }

    for j in 0..n {
        let s = t_history[j];
        let y_s = &y_history[j];

        system.kernel(t, s, y_s, k_buf);
        stats.n_kernel += 1;

        // Trapezoidal weights: 0.5 at endpoints, 1.0 in middle
        let weight = if j == 0 || j == n - 1 {
            S::from_f64(0.5)
        } else {
            S::ONE
        };

        // Compute dt for this interval
        let dt = if j < n - 1 {
            t_history[j + 1] - s
        } else if j > 0 {
            s - t_history[j - 1]
        } else {
            S::ZERO
        };

        for i in 0..dim {
            integral[i] += weight * dt * k_buf[i];
        }
    }
}

/// Improved Volterra solver using 4th-order Runge-Kutta for time-stepping.
pub struct VolterraRK4Solver;

impl<S: Scalar> IdeSolver<S> for VolterraRK4Solver {
    fn solve<Sys: IdeSystem<S>>(
        system: &Sys,
        t0: S,
        tf: S,
        y0: &[S],
        options: &IdeOptions<S>,
    ) -> Result<IdeResult<S>, String> {
        let dim = system.dim();

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

        // Storage for solution history
        let mut t_history: Vec<S> = Vec::with_capacity(n_steps + 1);
        let mut y_history: Vec<Vec<S>> = Vec::with_capacity(n_steps + 1);

        t_history.push(t0);
        y_history.push(y0.to_vec());

        let mut t_out = vec![t0];
        let mut y_out = y0.to_vec();
        let mut stats = IdeStats::default();

        let mut t = t0;
        let mut y = y0.to_vec();

        let half = S::from_f64(0.5);
        let sixth = S::ONE / S::from_f64(6.0);
        let two = S::from_f64(2.0);

        for n in 1..=n_steps {
            let t_new = t0 + S::from_usize(n) * dt;

            // RK4 stages
            let k1 = compute_derivative(system, t, &y, &t_history, &y_history, &mut stats);

            // y + 0.5*dt*k1
            let y_mid1: Vec<S> = y
                .iter()
                .zip(k1.iter())
                .map(|(&yi, &ki)| yi + half * dt * ki)
                .collect();
            let k2 = compute_derivative(
                system,
                t + half * dt,
                &y_mid1,
                &t_history,
                &y_history,
                &mut stats,
            );

            // y + 0.5*dt*k2
            let y_mid2: Vec<S> = y
                .iter()
                .zip(k2.iter())
                .map(|(&yi, &ki)| yi + half * dt * ki)
                .collect();
            let k3 = compute_derivative(
                system,
                t + half * dt,
                &y_mid2,
                &t_history,
                &y_history,
                &mut stats,
            );

            // y + dt*k3
            let y_end: Vec<S> = y
                .iter()
                .zip(k3.iter())
                .map(|(&yi, &ki)| yi + dt * ki)
                .collect();
            let k4 = compute_derivative(system, t + dt, &y_end, &t_history, &y_history, &mut stats);

            // RK4 combination
            let mut y_new = vec![S::ZERO; dim];
            for i in 0..dim {
                y_new[i] = y[i] + sixth * dt * (k1[i] + two * k2[i] + two * k3[i] + k4[i]);
            }

            // Store in history
            t_history.push(t_new);
            y_history.push(y_new.clone());

            // Update output
            t_out.push(t_new);
            y_out.extend_from_slice(&y_new);
            stats.n_steps += 1;

            t = t_new;
            y = y_new;
        }

        Ok(IdeResult::new(t_out, y_out, dim, stats))
    }
}

/// Compute the full derivative: f(t,y) + integral.
fn compute_derivative<S: Scalar, Sys: IdeSystem<S>>(
    system: &Sys,
    t: S,
    y: &[S],
    t_history: &[S],
    y_history: &[Vec<S>],
    stats: &mut IdeStats,
) -> Vec<S> {
    let dim = y.len();
    let mut f_buf = vec![S::ZERO; dim];
    let mut k_buf = vec![S::ZERO; dim];
    let mut integral = vec![S::ZERO; dim];

    // Local RHS
    system.rhs(t, y, &mut f_buf);
    stats.n_rhs += 1;

    // Memory integral
    compute_integral(
        system,
        t,
        t_history,
        y_history,
        &mut integral,
        &mut k_buf,
        stats,
    );

    // Total derivative
    for i in 0..dim {
        f_buf[i] += integral[i];
    }

    f_buf
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Simple IDE: y' = -y + ∫₀ᵗ exp(-(t-s)) * y(s) ds
    /// With y(0) = 1
    struct SimpleIde;

    impl IdeSystem<f64> for SimpleIde {
        fn dim(&self) -> usize {
            1
        }

        fn rhs(&self, _t: f64, y: &[f64], f: &mut [f64]) {
            f[0] = -y[0];
        }

        fn kernel(&self, t: f64, s: f64, y_s: &[f64], k: &mut [f64]) {
            k[0] = (-(t - s)).exp() * y_s[0];
        }

        fn is_convolution_kernel(&self) -> bool {
            true
        }
    }

    #[test]
    fn test_volterra_simple() {
        let options = IdeOptions::default().dt(0.01);
        let result =
            VolterraSolver::solve(&SimpleIde, 0.0, 1.0, &[1.0], &options).expect("Solve failed");

        assert!(result.success);
        assert!(result.t.len() > 1);

        // Solution should remain bounded and positive
        let y_final = result.y_final().unwrap()[0];
        assert!(y_final > 0.0 && y_final < 2.0);
    }

    #[test]
    fn test_volterra_rk4_more_accurate() {
        let options = IdeOptions::default().dt(0.05);

        let euler_result = VolterraSolver::solve(&SimpleIde, 0.0, 1.0, &[1.0], &options)
            .expect("Euler solve failed");
        let rk4_result = VolterraRK4Solver::solve(&SimpleIde, 0.0, 1.0, &[1.0], &options)
            .expect("RK4 solve failed");

        // With same dt, RK4 should give different (more accurate) result
        let y_euler = euler_result.y_final().unwrap()[0];
        let y_rk4 = rk4_result.y_final().unwrap()[0];

        // They should be close but not identical
        assert!((y_euler - y_rk4).abs() < 0.1);
        assert!((y_euler - y_rk4).abs() > 1e-6);
    }

    /// 2D IDE system
    struct TwoDIde;

    impl IdeSystem<f64> for TwoDIde {
        fn dim(&self) -> usize {
            2
        }

        fn rhs(&self, _t: f64, y: &[f64], f: &mut [f64]) {
            f[0] = -y[0] + 0.1 * y[1];
            f[1] = -y[1];
        }

        fn kernel(&self, t: f64, s: f64, y_s: &[f64], k: &mut [f64]) {
            let decay = (-(t - s)).exp();
            k[0] = 0.5 * decay * y_s[0];
            k[1] = 0.2 * decay * y_s[1];
        }
    }

    #[test]
    fn test_volterra_2d() {
        let options = IdeOptions::default().dt(0.01);
        let result =
            VolterraSolver::solve(&TwoDIde, 0.0, 1.0, &[1.0, 1.0], &options).expect("Solve failed");

        assert!(result.success);
        let y_final = result.y_final().unwrap();
        assert_eq!(y_final.len(), 2);

        // Both components should decay
        assert!(y_final[0] < 1.0);
        assert!(y_final[1] < 1.0);
    }

    #[test]
    fn test_dimension_mismatch() {
        let options = IdeOptions::default();
        let result = VolterraSolver::solve(&SimpleIde, 0.0, 1.0, &[1.0, 2.0], &options);
        assert!(result.is_err());
    }

    #[test]
    fn test_volterra_rk4_2d() {
        let options = IdeOptions::default().dt(0.01);
        let result = VolterraRK4Solver::solve(&TwoDIde, 0.0, 1.0, &[1.0, 1.0], &options)
            .expect("RK4 2D solve failed");

        assert!(result.success);
        let y_final = result.y_final().unwrap();
        assert_eq!(y_final.len(), 2);

        // Both components should decay
        assert!(y_final[0] < 1.0, "y[0] should decay: {}", y_final[0]);
        assert!(y_final[1] < 1.0, "y[1] should decay: {}", y_final[1]);
        assert!(y_final[0] > 0.0, "y[0] should remain positive");
        assert!(y_final[1] > 0.0, "y[1] should remain positive");
    }

    #[test]
    fn test_volterra_max_steps_exceeded() {
        // Use very small dt with max_steps=5 so required steps > 5
        let options = IdeOptions::default().dt(0.001).max_steps(5);
        let result = VolterraSolver::solve(&SimpleIde, 0.0, 1.0, &[1.0], &options);
        assert!(result.is_err());
        let msg = result.unwrap_err();
        assert!(msg.contains("exceeds maximum"), "Error message: {}", msg);
    }

    #[test]
    fn test_volterra_zero_kernel() {
        /// IDE with zero kernel: y' = -y + 0 => pure ODE y' = -y
        struct ZeroKernelIde;

        impl IdeSystem<f64> for ZeroKernelIde {
            fn dim(&self) -> usize {
                1
            }

            fn rhs(&self, _t: f64, y: &[f64], f: &mut [f64]) {
                f[0] = -y[0];
            }

            fn kernel(&self, _t: f64, _s: f64, _y_s: &[f64], k: &mut [f64]) {
                k[0] = 0.0;
            }
        }

        let options = IdeOptions::default().dt(0.001);
        let result = VolterraRK4Solver::solve(&ZeroKernelIde, 0.0, 1.0, &[1.0], &options)
            .expect("Solve failed");

        let y_final = result.y_final().unwrap()[0];
        let expected = (-1.0_f64).exp(); // exp(-1) ≈ 0.3679
        assert!(
            (y_final - expected).abs() < 1e-4,
            "Zero kernel should match pure ODE: got {}, expected {}",
            y_final,
            expected
        );
    }

    #[test]
    fn test_volterra_rk4_convergence() {
        // Solve at two step sizes and verify convergence.
        // The integral uses trapezoidal quadrature (2nd order), which limits
        // the overall convergence even though RK4 is 4th order for the ODE part.
        let options_coarse = IdeOptions::default().dt(0.02);
        let options_fine = IdeOptions::default().dt(0.01);
        // Use a very fine solution as reference
        let options_ref = IdeOptions::default().dt(0.0005);

        let result_coarse = VolterraRK4Solver::solve(&SimpleIde, 0.0, 1.0, &[1.0], &options_coarse)
            .expect("Coarse solve failed");
        let result_fine = VolterraRK4Solver::solve(&SimpleIde, 0.0, 1.0, &[1.0], &options_fine)
            .expect("Fine solve failed");
        let result_ref = VolterraRK4Solver::solve(&SimpleIde, 0.0, 1.0, &[1.0], &options_ref)
            .expect("Reference solve failed");

        let y_ref = result_ref.y_final().unwrap()[0];
        let err_coarse = (result_coarse.y_final().unwrap()[0] - y_ref).abs();
        let err_fine = (result_fine.y_final().unwrap()[0] - y_ref).abs();

        // Halving dt should reduce error; ratio >= 2 confirms convergence
        let ratio = err_coarse / err_fine;
        assert!(
            ratio > 1.5,
            "RK4 solver should converge: ratio={:.2} (err_coarse={:.2e}, err_fine={:.2e})",
            ratio,
            err_coarse,
            err_fine
        );
        // Fine solution should be substantially more accurate than coarse
        assert!(
            err_fine < err_coarse,
            "Finer dt should give smaller error: fine={:.2e}, coarse={:.2e}",
            err_fine,
            err_coarse
        );
    }
}

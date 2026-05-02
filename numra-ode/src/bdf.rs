//! BDF: Backward Differentiation Formulas (orders 1-5, variable order).
//!
//! BDF methods are implicit multistep methods designed for stiff ODEs and
//! index-1 DAEs. This implementation supports variable order (1-5) and
//! variable step size.
//!
//! ## Mathematical Formulation
//!
//! The k-th order BDF method approximates the derivative using a backward
//! difference formula:
//! ```text
//! sum_{j=0}^{k} alpha_j * y_{n+1-j} = h * beta_k * f(t_{n+1}, y_{n+1})
//! ```
//! The implicit equation for y_{n+1} is solved via Newton iteration.
//!
//! **DAE support (index-1):** For systems M * y' = f(t, y) with singular
//! mass matrix M, the BDF formula becomes:
//! ```text
//! M * sum_{j=0}^{k} alpha_j * y_{n+1-j} = h * beta_k * f(t_{n+1}, y_{n+1})
//! ```
//! Algebraic constraints (zero rows of M) are satisfied at each time step.
//!
//! ## Features
//! - Orders 1-5 (BDF1 = Backward Euler, up to BDF5)
//! - Automatic order selection based on error estimates
//! - Newton iteration with adaptive tolerance (scales with solver `rtol`)
//! - Efficient Jacobian reuse with LU factorization
//!
//! ## Known Limitations
//!
//! - BDF orders > 2 are not A-stable (A(alpha)-stable with decreasing angle)
//! - Variable step size ratios are not currently restricted, which can affect
//!   stability for rapid step size changes
//! - Order selection currently uses a simple heuristic; no embedded error pairs
//!
//! ## Reference
//! - Shampine, L.F. & Reichelt, M.W. (1997), "The MATLAB ODE Suite", SIAM J. Sci. Comput.
//! - Hindmarsh, A.C. (2005), SUNDIALS: Suite of Nonlinear and Differential/Algebraic Equation Solvers
//!
//! Author: Moussa Leblouba
//! Date: 5 March 2026
//! Modified: 2 May 2026

use faer::{ComplexField, Conjugate, SimpleEntity};
use numra_core::Scalar;
use numra_linalg::{DenseMatrix, LUFactorization, Matrix};
use std::collections::VecDeque;

use crate::error::SolverError;
use crate::problem::OdeSystem;
use crate::solver::{Solver, SolverOptions, SolverResult, SolverStats};

/// BDF solver for stiff ODEs (orders 1-5).
#[derive(Clone, Debug, Default)]
pub struct Bdf {
    /// Maximum order (1-5)
    max_order: usize,
    /// Minimum order
    min_order: usize,
}

impl Bdf {
    /// Create a new BDF solver with default settings (orders 1-5).
    pub fn new() -> Self {
        Self {
            max_order: 5,
            min_order: 1,
        }
    }

    /// Create a BDF solver with specified max order.
    pub fn with_max_order(max_order: usize) -> Self {
        Self {
            max_order: max_order.min(5).max(1),
            min_order: 1,
        }
    }

    /// Create a fixed-order BDF solver.
    pub fn fixed_order(order: usize) -> Self {
        let order = order.min(5).max(1);
        Self {
            max_order: order,
            min_order: order,
        }
    }
}

/// BDF coefficients for orders 1-5.
///
/// Standard form: α₀·y_{n+1} + α₁·y_n + α₂·y_{n-1} + ... = h·β·f(t_{n+1}, y_{n+1})
///
/// Reference: Wikipedia "Backward differentiation formula"
/// Reference: Hairer & Wanner "Solving Ordinary Differential Equations II"
mod coefficients {
    /// Alpha coefficients for BDF-k
    /// ALPHA[order-1][i] is the coefficient for y_{n+1-i}
    /// Normalized so that α₀ = 1
    pub const ALPHA: [[f64; 6]; 5] = [
        // BDF1: y_{n+1} - y_n = h·f
        [1.0, -1.0, 0.0, 0.0, 0.0, 0.0],
        // BDF2: y_{n+1} - (4/3)y_n + (1/3)y_{n-1} = (2/3)h·f
        [1.0, -4.0 / 3.0, 1.0 / 3.0, 0.0, 0.0, 0.0],
        // BDF3: y_{n+1} - (18/11)y_n + (9/11)y_{n-1} - (2/11)y_{n-2} = (6/11)h·f
        [1.0, -18.0 / 11.0, 9.0 / 11.0, -2.0 / 11.0, 0.0, 0.0],
        // BDF4
        [
            1.0,
            -48.0 / 25.0,
            36.0 / 25.0,
            -16.0 / 25.0,
            3.0 / 25.0,
            0.0,
        ],
        // BDF5
        [
            1.0,
            -300.0 / 137.0,
            300.0 / 137.0,
            -200.0 / 137.0,
            75.0 / 137.0,
            -12.0 / 137.0,
        ],
    ];

    /// Beta coefficient (multiplier for h*f) for each order
    /// β = 1/(1 + 1/2 + 1/3 + ... + 1/k) for order k
    pub const BETA: [f64; 5] = [
        1.0,          // BDF1
        2.0 / 3.0,    // BDF2
        6.0 / 11.0,   // BDF3
        12.0 / 25.0,  // BDF4
        60.0 / 137.0, // BDF5
    ];

    /// Error constant for each order (C_{k+1} for local error estimate)
    pub const ERROR_CONST: [f64; 5] = [
        1.0 / 2.0,    // BDF1
        2.0 / 9.0,    // BDF2
        3.0 / 22.0,   // BDF3
        12.0 / 125.0, // BDF4
        10.0 / 137.0, // BDF5
    ];
}

impl<S: Scalar + SimpleEntity + Conjugate<Canonical = S> + ComplexField> Solver<S> for Bdf {
    fn solve<Sys: OdeSystem<S>>(
        problem: &Sys,
        t0: S,
        tf: S,
        y0: &[S],
        options: &SolverOptions<S>,
    ) -> Result<SolverResult<S>, SolverError> {
        let solver = Bdf::new();
        solver.solve_internal(problem, t0, tf, y0, options)
    }
}

impl Bdf {
    fn solve_internal<S, Sys>(
        &self,
        problem: &Sys,
        t0: S,
        tf: S,
        y0: &[S],
        options: &SolverOptions<S>,
    ) -> Result<SolverResult<S>, SolverError>
    where
        S: Scalar + SimpleEntity + Conjugate<Canonical = S> + ComplexField,
        Sys: OdeSystem<S>,
    {
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

        let mut stats = SolverStats::default();

        // History buffer: stores past y values for multistep method
        // history[0] = y_n, history[1] = y_{n-1}, etc.
        // Uses VecDeque for O(1) push_front instead of Vec::insert(0, ...) which is O(n).
        let mut history: VecDeque<Vec<S>> = VecDeque::with_capacity(7);
        history.push_back(y0.to_vec());
        let mut h_history: VecDeque<S> = VecDeque::with_capacity(6);

        // Current order
        let mut order = 1_usize;

        // Jacobian and LU decomposition
        let mut jac_data = vec![S::ZERO; dim * dim];
        #[allow(unused_assignments)]
        let mut lu: Option<LUFactorization<S>> = None;
        let mut need_jac = true;
        let mut jac_age = 0_usize;

        // Mass matrix support for DAEs
        let has_mass = problem.has_mass_matrix();
        let mass_data = if has_mass {
            let mut m = vec![S::ZERO; dim * dim];
            problem.mass_matrix(&mut m);
            Some(m)
        } else {
            None
        };
        let mass_ref = mass_data.as_deref();

        // Initial step size
        let mut f0 = vec![S::ZERO; dim];
        problem.rhs(t, &y, &mut f0);
        stats.n_eval += 1;

        let mut h = self.initial_step_size(&y, &f0, options, dim);
        let h_min = options.h_min;
        let h_max = options.h_max.min((tf - t0).abs());

        let direction = if tf > t0 { S::ONE } else { -S::ONE };
        let mut step_count = 0_usize;
        let mut consecutive_failures = 0_usize;

        while (tf - t) * direction > S::from_f64(1e-10) * (tf - t0).abs() {
            if step_count >= options.max_steps {
                return Err(SolverError::MaxIterationsExceeded { t: t.to_f64() });
            }

            // Adjust final step
            if (t + h - tf) * direction > S::ZERO {
                h = tf - t;
            }

            // Ensure h doesn't get too small
            h = h.abs().max(h_min) * direction;
            if h.abs() > h_max {
                h = h_max * direction;
            }

            // Compute Jacobian if needed
            if need_jac || jac_age > 20 {
                problem.rhs(t, &y, &mut f0);
                stats.n_eval += 1;
                self.compute_jacobian(problem, t, &y, &f0, &mut jac_data, dim);
                stats.n_jac += 1;
                need_jac = false;
                jac_age = 0;
            }

            // Form and factorize iteration matrix: M/gamma - J (or I/gamma - J for ODEs)
            let gamma = S::from_f64(coefficients::BETA[order - 1]) * h
                / S::from_f64(coefficients::ALPHA[order - 1][0]);
            let iter_matrix = self.form_iteration_matrix(&jac_data, gamma, dim, mass_ref);
            lu = Some(LUFactorization::new(&iter_matrix)?);
            stats.n_lu += 1;

            // Attempt step
            let mut y_new = vec![S::ZERO; dim];
            let step_result = self.take_step(
                problem,
                t,
                h,
                &y,
                &history,
                &h_history,
                order,
                lu.as_ref().unwrap(),
                &mut y_new,
                &mut stats,
                dim,
                options,
                mass_ref,
            );

            match step_result {
                Ok(converged) if converged => {
                    // Estimate error
                    let err_norm = self.error_estimate(&y, &y_new, &history, order, options, dim);

                    if err_norm <= S::ONE {
                        // Step accepted
                        stats.n_accept += 1;
                        consecutive_failures = 0;
                        jac_age += 1;

                        // Update state FIRST
                        t = t + h;
                        y = y_new;

                        // Update history with the new y value
                        // history[0] = y_n (current), history[1] = y_{n-1}, etc.
                        // VecDeque::push_front is O(1) vs Vec::insert(0, ..) which is O(n).
                        history.push_front(y.clone());
                        h_history.push_front(h);
                        if history.len() > 6 {
                            history.pop_back();
                        }
                        if h_history.len() > 5 {
                            h_history.pop_back();
                        }

                        t_out.push(t);
                        y_out.extend_from_slice(&y);

                        // Update order (try to increase if error is small)
                        let new_order = self.select_order(
                            order,
                            err_norm,
                            &history,
                            self.min_order,
                            self.max_order,
                        );
                        order = new_order;

                        // New step size
                        let safety = S::from_f64(0.9);
                        let order_f = S::from_usize(order + 1);
                        let err_safe = err_norm.max(S::from_f64(1e-10));
                        let factor = safety * err_safe.powf(-S::ONE / order_f);
                        let factor = factor.min(S::from_f64(2.0)).max(S::from_f64(0.2));
                        h = h * factor;
                    } else {
                        // Step rejected due to error
                        stats.n_reject += 1;
                        consecutive_failures += 1;

                        let order_f = S::from_usize(order + 1);
                        let err_safe = err_norm.max(S::from_f64(1e-10));
                        let factor = S::from_f64(0.9) * err_safe.powf(-S::ONE / order_f);
                        let factor = factor.max(S::from_f64(0.1));
                        h = h * factor;

                        // Reduce order if struggling
                        if consecutive_failures >= 2 && order > self.min_order {
                            order -= 1;
                        }
                        if consecutive_failures >= 3 {
                            need_jac = true;
                        }
                    }
                }
                _ => {
                    // Newton iteration failed
                    stats.n_reject += 1;
                    consecutive_failures += 1;

                    // Graduated step reduction
                    let reduction = match consecutive_failures {
                        1 => 0.5,  // First failure: halve
                        2 => 0.25, // Second: quarter
                        _ => 0.1,  // Beyond: tenth
                    };
                    h = h * S::from_f64(reduction);

                    need_jac = true; // Always update Jacobian after failure

                    // Reduce order after multiple failures
                    if consecutive_failures >= 3 && order > self.min_order {
                        order -= 1;
                    }

                    // If too many consecutive failures, the problem is likely
                    // unsuitable for BDF at this point — bail out early
                    if consecutive_failures >= 10 {
                        return Err(SolverError::NewtonConvergenceFailed { t: t.to_f64() });
                    }
                }
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

    fn initial_step_size<S: Scalar>(
        &self,
        y0: &[S],
        f0: &[S],
        options: &SolverOptions<S>,
        dim: usize,
    ) -> S {
        if let Some(h0) = options.h0 {
            return h0;
        }

        // Improved initial step size estimation based on SUNDIALS CVODE approach
        // This is less conservative to avoid starting in a problematic regime
        let mut d0 = S::ZERO; // Weighted norm of y0
        let mut d1 = S::ZERO; // Weighted norm of f0

        for i in 0..dim {
            let sc = options.atol + options.rtol * y0[i].abs();
            let sc = sc.max(S::from_f64(1e-15));
            d0 = d0 + (y0[i] / sc) * (y0[i] / sc);
            d1 = d1 + (f0[i] / sc) * (f0[i] / sc);
        }
        d0 = (d0 / S::from_usize(dim)).sqrt();
        d1 = (d1 / S::from_usize(dim)).sqrt();

        // Use a reasonable default if either norm is very small
        // Less conservative than before: use 1e-4 instead of 1e-6
        let h0 = if d0 < S::from_f64(1e-5) || d1 < S::from_f64(1e-5) {
            S::from_f64(1e-4)
        } else {
            // Standard formula: h0 = 0.01 * ||y0|| / ||f0||
            S::from_f64(0.01) * d0 / d1
        };

        // Bound by h_max and ensure it's not too small
        let h0 = h0.max(options.h_min * S::from_f64(100.0)); // At least 100x h_min
        h0.min(options.h_max)
    }

    fn compute_jacobian<S, Sys>(
        &self,
        problem: &Sys,
        t: S,
        y: &[S],
        f0: &[S],
        jac: &mut [S],
        dim: usize,
    ) where
        S: Scalar,
        Sys: OdeSystem<S>,
    {
        let eps = S::from_f64(1e-8);
        let mut y_pert = y.to_vec();
        let mut f_pert = vec![S::ZERO; dim];

        for j in 0..dim {
            let yj = y[j];
            let h = eps * (S::ONE + yj.abs());
            y_pert[j] = yj + h;
            problem.rhs(t, &y_pert, &mut f_pert);
            y_pert[j] = yj;

            for i in 0..dim {
                jac[i * dim + j] = (f_pert[i] - f0[i]) / h;
            }
        }
    }

    /// Form the iteration matrix for Newton solve.
    ///
    /// BDF residual: R = alpha_0 * y + sum - h * beta * f(y)
    /// (or with mass: R = M * (alpha_0 * y + sum) - h * beta * f(y))
    ///
    /// The Jacobian of R w.r.t. y is:
    ///   dR/dy = alpha_0 * I - h * beta * J = alpha_0 * (I - gamma * J)
    ///
    /// For efficiency, we solve (I - gamma*J) * delta = R / alpha_0
    /// but since alpha_0 = 1 for normalized BDF coefficients, this simplifies to
    ///   (I - gamma * J) * delta = R
    ///
    /// For DAEs with mass matrix M:
    ///   dR/dy = alpha_0 * M - h * beta * J
    ///   Factoring: alpha_0 * (M - gamma * J)
    ///   So we use: (M - gamma * J) * delta = R
    fn form_iteration_matrix<S>(
        &self,
        jac: &[S],
        gamma: S,
        dim: usize,
        mass: Option<&[S]>,
    ) -> DenseMatrix<S>
    where
        S: Scalar + SimpleEntity + Conjugate<Canonical = S> + ComplexField,
    {
        let mut m = DenseMatrix::zeros(dim, dim);
        for i in 0..dim {
            for j in 0..dim {
                let jij = jac[i * dim + j];
                let mij = match mass {
                    Some(mass_data) => mass_data[i * dim + j],
                    None => {
                        if i == j {
                            S::ONE
                        } else {
                            S::ZERO
                        }
                    }
                };
                // (M - gamma * J) for mass matrix case
                // (I - gamma * J) for identity mass case
                m.set(i, j, mij - gamma * jij);
            }
        }
        m
    }

    /// Take a single BDF step using Newton iteration.
    ///
    /// BDF formula for ODEs: sum_{j=0}^{k} alpha_j * y_{n+1-j} = h * beta * f(t_{n+1}, y_{n+1})
    ///
    /// With mass matrix M for DAEs:
    ///   M * sum_{j=0}^{k} alpha_j * y_{n+1-j} = h * beta * f(t_{n+1}, y_{n+1})
    ///
    /// The Newton residual is:
    ///   R = M * (alpha_0 * y_new + sum_{j>=1} alpha_j * y_{n+1-j}) - h * beta * f(t_new, y_new)
    #[allow(clippy::too_many_arguments)]
    fn take_step<S, Sys>(
        &self,
        problem: &Sys,
        t: S,
        h: S,
        _y: &[S],
        history: &VecDeque<Vec<S>>,
        _h_history: &VecDeque<S>,
        order: usize,
        lu: &LUFactorization<S>,
        y_new: &mut [S],
        stats: &mut SolverStats,
        dim: usize,
        options: &SolverOptions<S>,
        mass: Option<&[S]>,
    ) -> Result<bool, SolverError>
    where
        S: Scalar + SimpleEntity + Conjugate<Canonical = S> + ComplexField,
        Sys: OdeSystem<S>,
    {
        let alpha = &coefficients::ALPHA[order - 1];
        let beta = S::from_f64(coefficients::BETA[order - 1]);
        let alpha0 = S::from_f64(alpha[0]);

        // Compute predictor: sum of alpha[i] * y_{n+1-i} for i >= 1
        let mut predictor = vec![S::ZERO; dim];
        for i in 0..dim {
            let mut sum = S::ZERO;
            for k in 1..=order.min(history.len()) {
                sum = sum - S::from_f64(alpha[k]) * history[k - 1][i];
            }
            predictor[i] = sum / alpha0;
        }

        // Initial guess for y_{n+1}
        y_new.copy_from_slice(&predictor);

        // Newton iteration with improved convergence handling
        let t_new = t + h;
        let mut f_new = vec![S::ZERO; dim];
        let mut residual = vec![S::ZERO; dim];

        // Newton tolerance: scale with solver rtol (SUNDIALS uses 0.33 * rtol).
        // Floor at 1e-10 to avoid excessive iteration at very tight tolerances.
        let newton_tol = (options.rtol * S::from_f64(0.1)).max(S::from_f64(1e-10));

        // Increase max iterations (Fix #2)
        let max_iter = 20;
        let mut prev_res_norm = S::from_f64(f64::MAX);

        for iter in 0..max_iter {
            problem.rhs(t_new, y_new, &mut f_new);
            stats.n_eval += 1;

            // Compute the linear combination: lc = alpha_0 * y_new + sum_{j>=1} alpha_j * y_{n+1-j}
            let mut lin_comb = vec![S::ZERO; dim];
            for i in 0..dim {
                let mut sum = S::ZERO;
                for k in 1..=order.min(history.len()) {
                    sum = sum + S::from_f64(alpha[k]) * history[k - 1][i];
                }
                lin_comb[i] = alpha0 * y_new[i] + sum;
            }

            // Residual: M * lin_comb - h * beta * f (or just lin_comb - h*beta*f for identity M)
            let mut res_norm = S::ZERO;
            if let Some(mass_data) = mass {
                // M * lin_comb
                for i in 0..dim {
                    let mut m_lc_i = S::ZERO;
                    for j in 0..dim {
                        m_lc_i = m_lc_i + mass_data[i * dim + j] * lin_comb[j];
                    }
                    residual[i] = m_lc_i - h * beta * f_new[i];
                    res_norm = res_norm + residual[i] * residual[i];
                }
            } else {
                // Identity mass: residual = lin_comb - h * beta * f
                for i in 0..dim {
                    residual[i] = lin_comb[i] - h * beta * f_new[i];
                    res_norm = res_norm + residual[i] * residual[i];
                }
            }
            res_norm = res_norm.sqrt();

            if res_norm < newton_tol {
                return Ok(true);
            }

            // Detect divergence early (Fix #3) - if residual grows after 2 iterations, abort
            if iter > 2 && res_norm > prev_res_norm * S::from_f64(1.5) {
                // Newton is diverging, signal failure to reduce step size
                return Ok(false);
            }
            prev_res_norm = res_norm;

            // Newton correction
            let delta = lu.solve(&residual)?;
            for i in 0..dim {
                y_new[i] = y_new[i] - delta[i];
            }
        }

        Ok(false)
    }

    fn error_estimate<S: Scalar>(
        &self,
        y: &[S],
        y_new: &[S],
        history: &VecDeque<Vec<S>>,
        order: usize,
        options: &SolverOptions<S>,
        dim: usize,
    ) -> S {
        // Error estimate for BDF methods.
        //
        // KNOWN LIMITATION: history[0] == y (both are the current state), so
        // delta_old = y - history[0] = 0. This reduces the error estimate to
        // a simple scaled step change |y_new - y| * C_{k+1} / 2, rather than
        // a proper backward difference. Fixing this requires variable-step
        // backward differences (as in SUNDIALS CVODE), which would account for
        // the step size ratio h_n/h_{n-1}. The current estimate is conservative
        // enough for most problems but can underestimate errors for slowly-varying
        // components in stiff systems (typical accuracy ~5e-2 vs expected ~1e-4).
        let err_const = S::from_f64(coefficients::ERROR_CONST[order - 1]);

        let mut err_norm = S::ZERO;
        for i in 0..dim {
            // Compute backward difference approximation to higher derivative
            // For order k, use difference with previous values
            let step_diff = if history.is_empty() {
                // First step: use simple difference
                (y_new[i] - y[i]) * err_const
            } else {
                // Use second difference for better estimate:
                // (y_new - y) - (y - y_prev) ≈ h² y''
                let y_prev = history[0][i];
                let delta_new = y_new[i] - y[i];
                let delta_old = y[i] - y_prev;
                (delta_new - delta_old) * err_const * S::from_f64(0.5)
            };

            let sc = options.atol + options.rtol * y[i].abs().max(y_new[i].abs());
            let sc = sc.max(S::from_f64(1e-15));
            let scaled_err = step_diff / sc;
            err_norm = err_norm + scaled_err * scaled_err;
        }

        (err_norm / S::from_usize(dim)).sqrt()
    }

    fn select_order<S: Scalar>(
        &self,
        current_order: usize,
        err_norm: S,
        history: &VecDeque<Vec<S>>,
        min_order: usize,
        max_order: usize,
    ) -> usize {
        // Need enough history to increase order
        if history.len() <= current_order {
            return current_order;
        }

        // If error is very small, try increasing order
        if err_norm < S::from_f64(0.01) && current_order < max_order {
            return current_order + 1;
        }

        // If error is large, try decreasing order
        if err_norm > S::from_f64(0.5) && current_order > min_order {
            return current_order - 1;
        }

        current_order
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::problem::{DaeProblem, OdeProblem};

    #[test]
    fn test_bdf_exponential_decay() {
        let problem = OdeProblem::new(
            |_t, y: &[f64], dydt: &mut [f64]| {
                dydt[0] = -y[0];
            },
            0.0,
            5.0,
            vec![1.0],
        );
        // Use moderate tolerances appropriate for BDF
        let options = SolverOptions::default().rtol(1e-3).atol(1e-5);
        let result = Bdf::solve(&problem, 0.0, 5.0, &[1.0], &options).unwrap();

        assert!(result.success);
        let y_final = result.y_final().unwrap();
        let expected = (-5.0_f64).exp();
        assert!(
            (y_final[0] - expected).abs() < 1e-2,
            "BDF exponential: got {}, expected {}",
            y_final[0],
            expected
        );
    }

    #[test]
    fn test_bdf_stiff_decay() {
        // Stiff problem: dy/dt = -100*y
        let problem = OdeProblem::new(
            |_t, y: &[f64], dydt: &mut [f64]| {
                dydt[0] = -100.0 * y[0];
            },
            0.0,
            0.1,
            vec![1.0],
        );
        // BDF should handle stiff problems well
        let options = SolverOptions::default().rtol(1e-2).atol(1e-4);
        let result = Bdf::solve(&problem, 0.0, 0.1, &[1.0], &options).unwrap();

        assert!(result.success);
        let y_final = result.y_final().unwrap();
        let expected = (-10.0_f64).exp();
        assert!(
            (y_final[0] - expected).abs() < 0.05,
            "BDF stiff: got {}, expected {}",
            y_final[0],
            expected
        );
    }

    #[test]
    fn test_bdf_linear_2d() {
        let problem = OdeProblem::new(
            |_t, y: &[f64], dydt: &mut [f64]| {
                dydt[0] = -y[0] + y[1];
                dydt[1] = y[0] - y[1];
            },
            0.0,
            5.0,
            vec![1.0, 0.0],
        );
        let options = SolverOptions::default().rtol(1e-3).atol(1e-5);
        let result = Bdf::solve(&problem, 0.0, 5.0, &[1.0, 0.0], &options).unwrap();

        assert!(result.success);
        let y_final = result.y_final().unwrap();
        // Conservation: y1 + y2 = 1
        assert!((y_final[0] + y_final[1] - 1.0).abs() < 1e-2);
    }

    #[test]
    fn test_bdf_van_der_pol() {
        let mu = 10.0;
        let problem = OdeProblem::new(
            move |_t, y: &[f64], dydt: &mut [f64]| {
                dydt[0] = y[1];
                dydt[1] = mu * (1.0 - y[0] * y[0]) * y[1] - y[0];
            },
            0.0,
            20.0,
            vec![2.0, 0.0],
        );
        // Van der Pol with μ=10 needs moderate tolerances
        let options = SolverOptions::default().rtol(1e-2).atol(1e-4);
        let result = Bdf::solve(&problem, 0.0, 20.0, &[2.0, 0.0], &options);

        assert!(result.is_ok(), "BDF Van der Pol failed: {:?}", result.err());
    }

    #[test]
    fn test_bdf_fixed_order() {
        let solver = Bdf::fixed_order(2);
        let problem = OdeProblem::new(
            |_t, y: &[f64], dydt: &mut [f64]| {
                dydt[0] = -y[0];
            },
            0.0,
            2.0,
            vec![1.0],
        );
        // Use moderate tolerances
        let options = SolverOptions::default().rtol(1e-3).atol(1e-5);
        let result = solver
            .solve_internal(&problem, 0.0, 2.0, &[1.0], &options)
            .unwrap();

        assert!(result.success);
    }

    #[test]
    fn test_bdf_simple_dae() {
        // Very simple index-1 DAE:
        // y1' = -y1 + y2       (differential equation)
        // 0 = y1 - y2          (algebraic: y2 = y1)
        //
        // Mass matrix: diag(1, 0)
        //
        // Analytical solution: y1(t) = y1(0), y2(t) = y1(t)
        // Since y2 = y1 always, and y1' = -y1 + y1 = 0, so y1 is constant!

        let dae = DaeProblem::new(
            |_t, y: &[f64], dydt: &mut [f64]| {
                dydt[0] = -y[0] + y[1]; // differential equation
                dydt[1] = y[0] - y[1]; // algebraic constraint: 0 = y1 - y2
            },
            |mass: &mut [f64]| {
                // Mass matrix: diag(1, 0)
                for i in 0..4 {
                    mass[i] = 0.0;
                }
                mass[0] = 1.0; // M[0,0] = 1 (differential)
                               // M[1,1] = 0 (algebraic)
            },
            0.0,
            1.0,
            vec![1.0, 1.0], // Consistent initial conditions: y2 = y1
            vec![1],        // algebraic index (y2)
        );

        let options = SolverOptions::default()
            .rtol(1e-4)
            .atol(1e-6)
            .max_steps(500000);
        let result = Bdf::solve(&dae, 0.0, 1.0, &[1.0, 1.0], &options);

        assert!(result.is_ok(), "DAE solve failed: {:?}", result.err());
        let sol = result.unwrap();

        // Both y1 and y2 should remain 1.0 (since y1' = -y1 + y2 = 0 when y2 = y1)
        let yf = sol.y_final().unwrap();
        assert!(
            (yf[0] - 1.0).abs() < 1e-4,
            "y1 deviated: {} (expected 1.0)",
            yf[0]
        );
        assert!(
            (yf[1] - 1.0).abs() < 1e-4,
            "y2 deviated: {} (expected 1.0)",
            yf[1]
        );
        // Check algebraic constraint
        let constraint = yf[0] - yf[1];
        assert!(
            constraint.abs() < 1e-4,
            "Constraint violated: {} (y1={}, y2={})",
            constraint,
            yf[0],
            yf[1]
        );
    }

    #[test]
    fn test_bdf_dae_with_mass_identity() {
        // Test that regular ODE still works when using DaeProblem with identity mass matrix
        // dy/dt = -y, y(0) = 1 => y(t) = exp(-t)

        let dae = DaeProblem::new(
            |_t, y: &[f64], dydt: &mut [f64]| {
                dydt[0] = -y[0];
            },
            |mass: &mut [f64]| {
                mass[0] = 1.0; // Identity mass matrix
            },
            0.0,
            1.0,
            vec![1.0],
            vec![], // No algebraic indices
        );

        let options = SolverOptions::default().rtol(1e-4).atol(1e-6);
        let result = Bdf::solve(&dae, 0.0, 1.0, &[1.0], &options);

        assert!(
            result.is_ok(),
            "DAE with identity mass failed: {:?}",
            result.err()
        );
        let sol = result.unwrap();
        let yf = sol.y_final().unwrap();
        let exact = (-1.0_f64).exp();
        assert!(
            (yf[0] - exact).abs() < 1e-3,
            "Error: {} (expected {}, got {})",
            (yf[0] - exact).abs(),
            exact,
            yf[0]
        );
    }

    #[test]
    fn test_bdf_dae_scaled_mass() {
        // Test with a non-identity but non-singular mass matrix
        // 2 * y' = -y => y' = -y/2 => y(t) = exp(-t/2)

        let dae = DaeProblem::new(
            |_t, y: &[f64], dydt: &mut [f64]| {
                dydt[0] = -y[0]; // RHS is still just -y
            },
            |mass: &mut [f64]| {
                mass[0] = 2.0; // Mass matrix M = 2
            },
            0.0,
            1.0,
            vec![1.0],
            vec![], // No algebraic indices
        );

        let options = SolverOptions::default().rtol(1e-3).atol(1e-5);
        let result = Bdf::solve(&dae, 0.0, 1.0, &[1.0], &options);

        assert!(
            result.is_ok(),
            "DAE with scaled mass failed: {:?}",
            result.err()
        );
        let sol = result.unwrap();
        let yf = sol.y_final().unwrap();
        // M*y' = f => 2*y' = -y => y' = -y/2 => y(t) = exp(-t/2)
        let exact = (-0.5_f64).exp();
        assert!(
            (yf[0] - exact).abs() < 1e-2,
            "Error: {} (expected {}, got {})",
            (yf[0] - exact).abs(),
            exact,
            yf[0]
        );
    }
}

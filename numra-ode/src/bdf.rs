//! BDF / NDF: variable-order, variable-step backward differentiation formulas.
//!
//! This is a Rust port of SciPy's `scipy.integrate.BDF` (the canonical
//! implementation of Shampine–Reichelt's NDF method from MATLAB's `ode15s`).
//! The previous revision of this file was algorithmically incorrect — see
//! "Why a rewrite" below.
//!
//! ## Mathematical formulation
//!
//! The k-th order BDF method approximates
//!   M · ∑ⱼ αⱼ · y_{n+1-j}  =  h · β · f(t_{n+1}, y_{n+1})
//! where M is an optional (possibly singular) mass matrix. For ODEs M = I.
//!
//! Rather than storing past `y` values and applying fixed-coefficient BDF
//! formulas (which is *not* order-k accurate when the step size varies — this
//! was the previous file's fundamental error), we work with **modified
//! divided differences** D, defined by:
//!   D⁰yₙ = yₙ
//!   Dʲyₙ = Dʲ⁻¹yₙ − Dʲ⁻¹y_{n-1}    (j ≥ 1)
//!
//! The differences are stored in an array `D[0..k+2]` whose rows are scaled
//! so that the BDF formula holds *as if* the step size were constant. When h
//! changes, the array is rescaled in-place via a small transformation matrix
//! (`change_d`); this preserves order accuracy. This is the
//! "fixed-leading-coefficient" / "quasi-constant step size" trick from
//! Shampine & Reichelt (1997).
//!
//! ## NDF coefficients (Klopfenstein–Shampine)
//!
//! For each order k we store a `KAPPA[k]` "boost" parameter; setting κ = 0
//! recovers pure BDF, while small negative κ gives the NDF method, which
//! reduces the error constant by ~25–30% at the cost of slight stability
//! tightening. Defaults follow Shampine–Reichelt.
//!
//! ## DAE support (index-1)
//!
//! For singular M, the iteration matrix becomes M − c·J (instead of I − c·J),
//! and the Newton residual carries M·ψ and M·d terms. Algebraic constraint
//! rows (zero rows of M) reduce to f_alg(t, y) = 0, enforced at every step.
//!
//! ## Why a rewrite
//!
//! The previous bdf.rs had three independent algorithmic bugs that
//! manifested catastrophically (silent wrong answers, runaway step counts):
//!
//! 1. **Newton convergence test on the residual norm** — when h·f was small
//!    in absolute terms, Newton declared convergence at iteration 0 with
//!    `y_new = predictor = y_old`, leaving the solution unchanged while time
//!    advanced. This is the source of the reported "y_final = y_initial at
//!    success=true" bug.
//! 2. **Error estimator identically ≈ 0** — the formula referenced
//!    `history[0]`, which was always equal to the current `y` (just pushed),
//!    so the estimate's leading term cancelled.
//! 3. **Fixed BDF coefficients applied to variable-h history** — without the
//!    modified-divided-differences rescaling, the method silently loses
//!    order whenever h changes. The textbook coefficients [1, -4/3, 1/3,
//!    2/3·h] only define BDF2 when y_n, y_{n-1} are spaced equally.
//!
//! ## References
//! - L. F. Shampine and M. W. Reichelt, "The MATLAB ODE Suite", SIAM J. Sci.
//!   Comput. 18(1), 1–22 (1997). [NDF coefficients, fixed-leading-coefficient
//!   BDF, error estimator.]
//! - G. D. Byrne and A. C. Hindmarsh, "A Polyalgorithm for the Numerical
//!   Solution of ODEs", ACM TOMS 1(1), 71–96 (1975).
//! - SciPy `scipy/integrate/_ivp/bdf.py` (BSD-3 / Apache-2.0). Reference
//!   implementation that this file ports.
//!
//! Author: Moussa Leblouba
//! Date: 5 March 2026
//! Modified: 6 May 2026 (full rewrite based on SciPy's NDF/MDD algorithm)

use faer::{ComplexField, Conjugate, SimpleEntity};
use numra_core::Scalar;
use numra_linalg::{DenseMatrix, LUFactorization, Matrix};

use crate::error::SolverError;
use crate::problem::OdeSystem;
use crate::solver::{Solver, SolverOptions, SolverResult, SolverStats};

/// BDF / NDF solver for stiff ODEs and index-1 DAEs (orders 1-5).
#[derive(Clone, Debug, Default)]
pub struct Bdf {
    max_order: usize,
    min_order: usize,
}

impl Bdf {
    pub fn new() -> Self {
        Self {
            max_order: MAX_ORDER,
            min_order: 1,
        }
    }

    /// Cap the maximum order. Useful to keep the method L-stable
    /// (max_order ≤ 2) for problems that need strict L-stability.
    pub fn with_max_order(max_order: usize) -> Self {
        Self {
            max_order: max_order.clamp(1, MAX_ORDER),
            min_order: 1,
        }
    }

    /// Pin the order. Note: BDF *always* starts at order 1 (single-step info
    /// available at startup); `fixed_order` only acts as a soft floor and a
    /// hard ceiling for adaptive selection.
    pub fn fixed_order(order: usize) -> Self {
        let order = order.clamp(1, MAX_ORDER);
        Self {
            max_order: order,
            min_order: order,
        }
    }
}

// ---- Algorithm constants ---------------------------------------------------

const MAX_ORDER: usize = 5;
const NEWTON_MAXITER: usize = 4;
const MIN_FACTOR: f64 = 0.2;
const MAX_FACTOR: f64 = 10.0;

/// Klopfenstein–Shampine NDF "boost" parameter for orders 0..=5.
/// Setting all to zero recovers pure BDF.
const KAPPA: [f64; MAX_ORDER + 1] = [0.0, -0.1850, -1.0 / 9.0, -0.0823, -0.0415, 0.0];

/// γ_k = ∑_{i=1}^{k} 1/i, with γ_0 = 0.
const GAMMA: [f64; MAX_ORDER + 1] = [
    0.0,
    1.0,
    1.0 + 1.0 / 2.0,
    1.0 + 1.0 / 2.0 + 1.0 / 3.0,
    1.0 + 1.0 / 2.0 + 1.0 / 3.0 + 1.0 / 4.0,
    1.0 + 1.0 / 2.0 + 1.0 / 3.0 + 1.0 / 4.0 + 1.0 / 5.0,
];

/// α_k = (1 − κ_k) γ_k. The leading coefficient in the BDF/NDF
/// fixed-leading-coefficient form.
const ALPHA: [f64; MAX_ORDER + 1] = [
    (1.0 - KAPPA[0]) * GAMMA[0],
    (1.0 - KAPPA[1]) * GAMMA[1],
    (1.0 - KAPPA[2]) * GAMMA[2],
    (1.0 - KAPPA[3]) * GAMMA[3],
    (1.0 - KAPPA[4]) * GAMMA[4],
    (1.0 - KAPPA[5]) * GAMMA[5],
];

/// Error constant C_{k+1} for order k: C = κ_k γ_k + 1/(k+1). Scales the
/// next-order divided difference into a local truncation error estimate.
const ERROR_CONST: [f64; MAX_ORDER + 1] = [
    KAPPA[0] * GAMMA[0] + 1.0,
    KAPPA[1] * GAMMA[1] + 1.0 / 2.0,
    KAPPA[2] * GAMMA[2] + 1.0 / 3.0,
    KAPPA[3] * GAMMA[3] + 1.0 / 4.0,
    KAPPA[4] * GAMMA[4] + 1.0 / 5.0,
    KAPPA[5] * GAMMA[5] + 1.0 / 6.0,
];

const RU_SIZE: usize = (MAX_ORDER + 1) * (MAX_ORDER + 1); // = 36

// ---- Solver trait impl -----------------------------------------------------

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

// ---- compute_R / change_D --------------------------------------------------
//
// `compute_r(order, factor)` builds the (order+1)×(order+1) transformation
// matrix R such that, when D is the modified-divided-differences array for
// step h, the rescaled differences for step (factor·h) are obtained by
//   D' = (R · U)^T · D
// where U = compute_r(order, 1). For numerical efficiency we use stack
// arrays of size (MAX_ORDER+1)² = 36 and only access the leading
// (order+1)×(order+1) submatrix.
//
// Reference: SciPy's `_ivp/bdf.py::compute_R`, derived from Shampine &
// Reichelt §3.

/// Compute R(order, factor) into the leading (order+1)×(order+1) block of `out`.
/// Layout: out[r * (MAX_ORDER+1) + c] = R[r, c].
fn compute_r<S: Scalar>(order: usize, factor: S, out: &mut [S; RU_SIZE]) {
    for slot in out.iter_mut() {
        *slot = S::ZERO;
    }
    let stride = MAX_ORDER + 1;

    // M[0, :] = 1
    for c in 0..=order {
        out[c] = S::ONE;
    }
    // M[r, c] = (r - 1 - factor * c) / r  for r, c in 1..=order
    for r in 1..=order {
        for c in 1..=order {
            let num = S::from_usize(r - 1) - factor * S::from_usize(c);
            out[r * stride + c] = num / S::from_usize(r);
        }
    }
    // Cumulative product down each column.
    for r in 1..=order {
        for c in 0..=order {
            let prev = out[(r - 1) * stride + c];
            out[r * stride + c] = out[r * stride + c] * prev;
        }
    }
}

/// In-place: replace D[0..=order] with (R·U)^T · D[0..=order],
/// where R = compute_r(order, factor) and U = compute_r(order, 1).
fn change_d<S: Scalar>(d: &mut [Vec<S>], order: usize, factor: S) {
    // factor = 1 is a no-op; bail early.
    if (factor - S::ONE).abs() < S::from_f64(1e-15) {
        return;
    }

    let mut r_mat = [S::ZERO; RU_SIZE];
    let mut u_mat = [S::ZERO; RU_SIZE];
    compute_r(order, factor, &mut r_mat);
    compute_r(order, S::ONE, &mut u_mat);

    let stride = MAX_ORDER + 1;

    // RU = R · U  (only the (order+1)×(order+1) leading block matters).
    let mut ru = [S::ZERO; RU_SIZE];
    for i in 0..=order {
        for j in 0..=order {
            let mut s = S::ZERO;
            for k in 0..=order {
                s = s + r_mat[i * stride + k] * u_mat[k * stride + j];
            }
            ru[i * stride + j] = s;
        }
    }

    // Apply (RU)^T to the rows of D:  D'_i = ∑_k RU[k, i] · D_k.
    let dim = d[0].len();
    let mut new_rows: Vec<Vec<S>> = (0..=order).map(|_| vec![S::ZERO; dim]).collect();
    for i in 0..=order {
        for k in 0..=order {
            let coef = ru[k * stride + i];
            if coef == S::ZERO {
                continue;
            }
            for col in 0..dim {
                new_rows[i][col] = new_rows[i][col] + coef * d[k][col];
            }
        }
    }
    for (i, row) in new_rows.into_iter().enumerate() {
        d[i] = row;
    }
}

// ---- Newton solver ---------------------------------------------------------

struct NewtonOutcome<S: Scalar> {
    converged: bool,
    n_iter: usize,
    y_new: Vec<S>,
    /// d = y_new − y_predict. Equals the (order+1)-th modified divided
    /// difference at the new step, and is the raw input to the local
    /// truncation error estimate.
    d: Vec<S>,
}

/// Bookkeeping for an accepted step, captured inside the inner retry loop
/// and consumed by the commit phase.
struct AcceptedStep<S: Scalar> {
    y_new: Vec<S>,
    d_corr: Vec<S>,
    err_norm: S,
    h_abs_used: S,
    safety: S,
}

/// Solve the BDF/NDF algebraic system via simplified Newton iteration.
///
/// At convergence:  M·d = c·f(t_new, y_predict + d) − M·ψ.
/// Iteration uses the pre-factored linear operator (M − c·J).
///
/// Convergence test (Hairer–Wanner ODE II §IV.8, also SciPy):
///   converged ⟺ rate / (1 − rate) · ‖dy‖ < tol
/// with `rate` = ratio of successive correction norms. This is a *correction*
/// norm test, not a residual norm test — using residual norms (as the
/// previous file did) leads to false convergence at iteration 0 when the
/// predictor is near the new point in absolute terms but Newton hasn't
/// actually moved.
#[allow(clippy::too_many_arguments)]
fn solve_bdf_system<S, Sys>(
    problem: &Sys,
    t_new: S,
    y_predict: &[S],
    c: S,
    psi: &[S],
    lu: &LUFactorization<S>,
    scale: &[S],
    tol: S,
    mass: Option<&[S]>,
    stats: &mut SolverStats,
    dim: usize,
) -> Result<NewtonOutcome<S>, SolverError>
where
    S: Scalar + SimpleEntity + Conjugate<Canonical = S> + ComplexField,
    Sys: OdeSystem<S>,
{
    let mut d = vec![S::ZERO; dim];
    let mut y = y_predict.to_vec();
    let mut f_buf = vec![S::ZERO; dim];
    let mut rhs_buf = vec![S::ZERO; dim];

    // Pre-compute M·ψ once (constant within the Newton loop).
    let mut m_psi = vec![S::ZERO; dim];
    if let Some(m) = mass {
        for i in 0..dim {
            let mut s = S::ZERO;
            for j in 0..dim {
                s = s + m[i * dim + j] * psi[j];
            }
            m_psi[i] = s;
        }
    } else {
        m_psi.copy_from_slice(psi);
    }

    let mut dy_norm_old: Option<S> = None;
    let mut converged = false;
    let mut n_iter_done = 0usize;

    for iter in 0..NEWTON_MAXITER {
        n_iter_done = iter + 1;

        problem.rhs(t_new, &y, &mut f_buf);
        stats.n_eval += 1;

        // Bail on NaN/Inf in f.
        if f_buf.iter().any(|v| !v.to_f64().is_finite()) {
            break;
        }

        // RHS of Newton update:  c·f − M·(ψ + d).
        // For ODEs (M = I), this simplifies to  c·f − ψ − d.
        if let Some(m) = mass {
            for i in 0..dim {
                let mut md = S::ZERO;
                for j in 0..dim {
                    md = md + m[i * dim + j] * d[j];
                }
                rhs_buf[i] = c * f_buf[i] - m_psi[i] - md;
            }
        } else {
            for i in 0..dim {
                rhs_buf[i] = c * f_buf[i] - psi[i] - d[i];
            }
        }

        let dy = lu.solve(&rhs_buf)?;

        // Weighted-RMS norm of the correction.
        let mut dy_norm_sq = S::ZERO;
        for i in 0..dim {
            let r = dy[i] / scale[i];
            dy_norm_sq = dy_norm_sq + r * r;
        }
        let dy_norm = (dy_norm_sq / S::from_usize(dim)).sqrt();

        let rate = dy_norm_old.and_then(|old| {
            if old > S::ZERO {
                Some(dy_norm / old)
            } else {
                None
            }
        });

        // Early divergence: rate ≥ 1, or projected error after the remaining
        // iterations would still exceed tol.
        if let Some(r) = rate {
            if r >= S::ONE {
                break;
            }
            let remaining = NEWTON_MAXITER - iter;
            let r_pow = r.powf(S::from_usize(remaining));
            if r_pow / (S::ONE - r) * dy_norm > tol {
                break;
            }
        }

        // Apply correction.
        for i in 0..dim {
            y[i] = y[i] + dy[i];
            d[i] = d[i] + dy[i];
        }

        // Convergence test on the *correction*.
        if dy_norm == S::ZERO {
            converged = true;
            break;
        }
        if let Some(r) = rate {
            if r / (S::ONE - r) * dy_norm < tol {
                converged = true;
                break;
            }
        }

        dy_norm_old = Some(dy_norm);
    }

    Ok(NewtonOutcome {
        converged,
        n_iter: n_iter_done,
        y_new: y,
        d,
    })
}

// ---- Main solve loop -------------------------------------------------------

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

        let mut stats = SolverStats::default();

        let mut t = t0;
        let mut t_out = vec![t0];
        let mut y_out = y0.to_vec();

        let direction = if tf > t0 { S::ONE } else { -S::ONE };

        // ---- Initial RHS, step size, and divided differences ----
        let mut f_eval = vec![S::ZERO; dim];
        problem.rhs(t, y0, &mut f_eval);
        stats.n_eval += 1;

        let mut h_abs = self.initial_step_size(y0, &f_eval, options, dim);

        // D[0] = y_0;  D[1] = h_0 · f_0 · direction;  D[2..] = 0.
        let mut d_arr: Vec<Vec<S>> = (0..MAX_ORDER + 3).map(|_| vec![S::ZERO; dim]).collect();
        d_arr[0].copy_from_slice(y0);
        for i in 0..dim {
            d_arr[1][i] = h_abs * f_eval[i] * direction;
        }

        // Mass matrix.
        let mass_data = if problem.has_mass_matrix() {
            let mut m = vec![S::ZERO; dim * dim];
            problem.mass_matrix(&mut m);
            Some(m)
        } else {
            None
        };
        let mass_ref = mass_data.as_deref();

        // Jacobian + LU.
        let mut jac = vec![S::ZERO; dim * dim];
        self.compute_jacobian(problem, t, y0, &f_eval, &mut jac, dim);
        stats.n_jac += 1;
        let mut current_jac = true;
        let mut lu: Option<LUFactorization<S>> = None;
        let mut last_c: Option<S> = None;

        // Order management.
        let mut order = 1usize;
        let mut n_equal_steps: usize = 0;

        // Newton tolerance (Hairer formula).
        let uround = S::from_f64(2.220446049250313e-16);
        let newton_tol = (S::from_f64(10.0) * uround / options.rtol)
            .max(S::from_f64(0.03).min(options.rtol.sqrt()));

        let h_min = options.h_min;
        let h_max = options.h_max.min((tf - t0).abs());

        let mut step_count = 0usize;

        while (tf - t) * direction > S::from_f64(1e-12) * (tf - t0).abs() {
            if step_count >= options.max_steps {
                return Err(SolverError::MaxIterationsExceeded { t: t.to_f64() });
            }

            // --- Step-size cap pre-checks (SciPy: top of _step_impl) ---
            let ulp_min = S::from_f64(10.0) * uround * (t.abs().max(tf.abs())).max(S::ONE);
            let min_step = h_min.max(ulp_min);

            if h_abs > h_max {
                let factor = h_max / h_abs;
                change_d(&mut d_arr, order, factor);
                n_equal_steps = 0;
                h_abs = h_max;
                lu = None;
                last_c = None;
            } else if h_abs < min_step {
                let factor = min_step / h_abs;
                change_d(&mut d_arr, order, factor);
                n_equal_steps = 0;
                h_abs = min_step;
                lu = None;
                last_c = None;
            }

            // --- Inner loop: try, retry, possibly reduce h ---
            let mut step_accepted = false;
            // (y_new, d_correction, error_norm, h_abs_used, safety_factor)
            let mut accepted: Option<AcceptedStep<S>> = None;

            while !step_accepted {
                if h_abs < min_step {
                    return Err(SolverError::StepSizeTooSmall {
                        t: t.to_f64(),
                        h: h_abs.to_f64(),
                        h_min: h_min.to_f64(),
                    });
                }

                let h = h_abs * direction;
                let mut t_new = t + h;
                let mut h_used = h;
                let mut h_abs_used = h_abs;

                // Clip to t_bound (rescale D to match the shorter step).
                if (t_new - tf) * direction > S::ZERO {
                    t_new = tf;
                    h_used = t_new - t;
                    h_abs_used = h_used.abs();
                    let factor = h_abs_used / h_abs;
                    change_d(&mut d_arr, order, factor);
                    n_equal_steps = 0;
                    h_abs = h_abs_used;
                    lu = None;
                    last_c = None;
                }

                // Predictor:  y_predict = ∑_{k=0..order} D[k]
                let mut y_predict = vec![S::ZERO; dim];
                for k in 0..=order {
                    for i in 0..dim {
                        y_predict[i] = y_predict[i] + d_arr[k][i];
                    }
                }

                // ψ = (∑_{j=1..order} γ_j · D[j]) / α_order
                let alpha_o = S::from_f64(ALPHA[order]);
                let mut psi = vec![S::ZERO; dim];
                for j in 1..=order {
                    let g = S::from_f64(GAMMA[j]);
                    for i in 0..dim {
                        psi[i] = psi[i] + g * d_arr[j][i];
                    }
                }
                for i in 0..dim {
                    psi[i] = psi[i] / alpha_o;
                }

                let c = h_used / alpha_o;

                // Pre-Newton scale based on |y_predict|.
                let mut scale = vec![S::ZERO; dim];
                for i in 0..dim {
                    scale[i] =
                        (options.atol + options.rtol * y_predict[i].abs()).max(S::from_f64(1e-300));
                }

                // --- Newton with retry-on-stale-Jacobian ---
                let outcome;
                loop {
                    if lu.is_none() || last_c != Some(c) {
                        let it = self.form_iteration_matrix(&jac, c, dim, mass_ref);
                        lu = Some(LUFactorization::new(&it)?);
                        stats.n_lu += 1;
                        last_c = Some(c);
                    }
                    let attempt = solve_bdf_system(
                        problem,
                        t_new,
                        &y_predict,
                        c,
                        &psi,
                        lu.as_ref().unwrap(),
                        &scale,
                        newton_tol,
                        mass_ref,
                        &mut stats,
                        dim,
                    )?;

                    if attempt.converged || current_jac {
                        outcome = attempt;
                        break;
                    }

                    // Stale Jacobian: refresh f baseline + J, retry
                    // *without* reducing h. Critical: previous BDF skipped
                    // this and reduced h on every Newton failure, which
                    // wasted enormous numbers of steps.
                    problem.rhs(t_new, &y_predict, &mut f_eval);
                    stats.n_eval += 1;
                    self.compute_jacobian(problem, t_new, &y_predict, &f_eval, &mut jac, dim);
                    stats.n_jac += 1;
                    current_jac = true;
                    lu = None;
                    last_c = None;
                }

                if !outcome.converged {
                    // Real Newton failure with fresh Jacobian. Halve h.
                    let factor = S::from_f64(0.5);
                    change_d(&mut d_arr, order, factor);
                    n_equal_steps = 0;
                    h_abs = h_abs * factor;
                    stats.n_reject += 1;
                    lu = None;
                    last_c = None;
                    continue;
                }

                // --- Newton converged; check error norm ---
                let safety = S::from_f64(0.9 * (2.0 * NEWTON_MAXITER as f64 + 1.0))
                    / S::from_f64(2.0 * NEWTON_MAXITER as f64 + outcome.n_iter as f64);

                let mut scale_new = vec![S::ZERO; dim];
                for i in 0..dim {
                    scale_new[i] = (options.atol + options.rtol * outcome.y_new[i].abs())
                        .max(S::from_f64(1e-300));
                }

                let err_const = S::from_f64(ERROR_CONST[order]);
                let mut err_sq = S::ZERO;
                for i in 0..dim {
                    let e = err_const * outcome.d[i] / scale_new[i];
                    err_sq = err_sq + e * e;
                }
                let err_norm = (err_sq / S::from_usize(dim)).sqrt();

                if err_norm > S::ONE {
                    // Reject by error: shrink h. Do NOT reset LU per SciPy
                    // (Newton was fine, only the error was too big), but DO
                    // reset last_c since c changes with h.
                    let order_p1 = S::from_usize(order + 1);
                    let factor =
                        S::from_f64(MIN_FACTOR).max(safety * err_norm.powf(-S::ONE / order_p1));
                    change_d(&mut d_arr, order, factor);
                    n_equal_steps = 0;
                    h_abs = h_abs * factor;
                    stats.n_reject += 1;
                    last_c = None;
                    continue;
                }

                accepted = Some(AcceptedStep {
                    y_new: outcome.y_new,
                    d_corr: outcome.d,
                    err_norm,
                    h_abs_used,
                    safety,
                });
                step_accepted = true;
            }

            let AcceptedStep {
                y_new,
                d_corr,
                err_norm,
                h_abs_used,
                safety,
            } = accepted.unwrap();

            // --- Commit accepted step ---
            stats.n_accept += 1;
            n_equal_steps += 1;
            t = t + h_abs_used * direction;
            h_abs = h_abs_used;

            // Update modified divided differences in-place. The principal
            // relation here is D^{j+1} y_n = D^j y_n − D^j y_{n-1}; combined
            // with d_corr = D^{order+1} y_n, this elegant cascade updates
            // the entire table:
            //   D[order+2] ← d − D[order+1]
            //   D[order+1] ← d
            //   D[i]      ← D[i] + D[i+1]    for i = order, …, 0
            for i in 0..dim {
                d_arr[order + 2][i] = d_corr[i] - d_arr[order + 1][i];
                d_arr[order + 1][i] = d_corr[i];
            }
            for k in (0..=order).rev() {
                for i in 0..dim {
                    d_arr[k][i] = d_arr[k][i] + d_arr[k + 1][i];
                }
            }

            // After update, D[0] should equal y_new (modulo Newton tolerance).
            debug_assert!({
                let mut ok = true;
                for i in 0..dim {
                    let diff = (d_arr[0][i] - y_new[i]).abs();
                    let scl = options.atol + options.rtol * y_new[i].abs().max(S::ONE);
                    if diff.to_f64() > scl.to_f64() * 1e3 {
                        ok = false;
                        break;
                    }
                }
                ok
            });

            t_out.push(t);
            y_out.extend_from_slice(&d_arr[0]);

            // Mark Jacobian as stale.
            current_jac = false;

            // --- Order / step adaptation (gated on quasi-constant steps) ---
            //
            // Until we have `order + 1` consecutive equal-size accepted
            // steps, the modified-divided-differences are not yet a faithful
            // proxy for "constant-step" differences, so the order-selection
            // formulas are not trustworthy. SciPy enforces the same gate.
            if n_equal_steps < order + 1 {
                step_count += 1;
                continue;
            }

            // Compute error norms for orders order−1 and order+1.
            let mut err_m_norm = S::from_f64(f64::INFINITY);
            if order > 1 {
                let ec = S::from_f64(ERROR_CONST[order - 1]);
                let mut s = S::ZERO;
                for i in 0..dim {
                    let scl =
                        (options.atol + options.rtol * y_new[i].abs()).max(S::from_f64(1e-300));
                    let e = ec * d_arr[order][i] / scl;
                    s = s + e * e;
                }
                err_m_norm = (s / S::from_usize(dim)).sqrt();
            }

            let mut err_p_norm = S::from_f64(f64::INFINITY);
            if order < self.max_order {
                let ec = S::from_f64(ERROR_CONST[order + 1]);
                let mut s = S::ZERO;
                for i in 0..dim {
                    let scl =
                        (options.atol + options.rtol * y_new[i].abs()).max(S::from_f64(1e-300));
                    let e = ec * d_arr[order + 2][i] / scl;
                    s = s + e * e;
                }
                err_p_norm = (s / S::from_usize(dim)).sqrt();
            }

            let safe_pow = |e: S, p: usize| -> S {
                let f = e.max(S::from_f64(1e-30));
                f.powf(-S::ONE / S::from_usize(p))
            };
            let factor_m = if order > self.min_order && err_m_norm.to_f64().is_finite() {
                safe_pow(err_m_norm, order)
            } else {
                S::ZERO
            };
            let factor_o = safe_pow(err_norm, order + 1);
            let factor_p = if order < self.max_order && err_p_norm.to_f64().is_finite() {
                safe_pow(err_p_norm, order + 2)
            } else {
                S::ZERO
            };

            let mut best = factor_o;
            let mut delta_order: i32 = 0;
            if factor_m > best {
                best = factor_m;
                delta_order = -1;
            }
            if factor_p > best {
                best = factor_p;
                delta_order = 1;
            }
            if !best.to_f64().is_finite() {
                best = S::ONE;
            }

            order = ((order as i32) + delta_order) as usize;

            let factor = (safety * best).min(S::from_f64(MAX_FACTOR));
            let factor = factor.max(S::from_f64(MIN_FACTOR));

            change_d(&mut d_arr, order, factor);
            n_equal_steps = 0;
            h_abs = h_abs * factor;
            if h_abs > h_max {
                let cap = h_max / h_abs;
                change_d(&mut d_arr, order, cap);
                h_abs = h_max;
            }
            lu = None;
            last_c = None;

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
        let mut d0 = S::ZERO;
        let mut d1 = S::ZERO;
        for i in 0..dim {
            let sc = (options.atol + options.rtol * y0[i].abs()).max(S::from_f64(1e-15));
            d0 = d0 + (y0[i] / sc) * (y0[i] / sc);
            d1 = d1 + (f0[i] / sc) * (f0[i] / sc);
        }
        d0 = (d0 / S::from_usize(dim)).sqrt();
        d1 = (d1 / S::from_usize(dim)).sqrt();
        let h0 = if d0 < S::from_f64(1e-5) || d1 < S::from_f64(1e-5) {
            S::from_f64(1e-6)
        } else {
            S::from_f64(0.01) * d0 / d1
        };
        h0.min(options.h_max).max(options.h_min)
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

    /// Build  M − c·J  (or  I − c·J  for ODEs).
    fn form_iteration_matrix<S>(
        &self,
        jac: &[S],
        c: S,
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
                m.set(i, j, mij - c * jij);
            }
        }
        m
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
        let problem = OdeProblem::new(
            |_t, y: &[f64], dydt: &mut [f64]| {
                dydt[0] = -100.0 * y[0];
            },
            0.0,
            0.1,
            vec![1.0],
        );
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
        let options = SolverOptions::default().rtol(1e-3).atol(1e-5);
        let result = solver
            .solve_internal(&problem, 0.0, 2.0, &[1.0], &options)
            .unwrap();
        assert!(result.success);
    }

    /// Regression: previously returned y_final = y_initial at success=true.
    /// The 2-state augmented system was the user's reported failure case.
    /// Analytical solution: y(t) = exp(-t/2), S(t) = -t·exp(-t/2).
    #[test]
    fn test_bdf_regression_silent_wrong_answer() {
        let problem = OdeProblem::new(
            |_t, y: &[f64], dydt: &mut [f64]| {
                dydt[0] = -0.5 * y[0];
                dydt[1] = -0.5 * y[1] - y[0];
            },
            0.0,
            2.0,
            vec![1.0, 0.0],
        );
        let options = SolverOptions::default().rtol(1e-4).atol(1e-6);
        let result = Bdf::solve(&problem, 0.0, 2.0, &[1.0, 0.0], &options).unwrap();
        assert!(result.success);
        let y_final = result.y_final().unwrap();
        // y(2) = exp(-1) ≈ 0.3679;  S(2) = -2·exp(-1) ≈ -0.7358
        let y_expected = (-1.0_f64).exp();
        let s_expected = -2.0 * (-1.0_f64).exp();
        assert!(
            (y_final[0] - y_expected).abs() < 1e-3,
            "y deviated: got {}, expected {}",
            y_final[0],
            y_expected
        );
        assert!(
            (y_final[1] - s_expected).abs() < 1e-3,
            "S deviated: got {}, expected {}",
            y_final[1],
            s_expected
        );
    }

    /// Regression: tight rtol on simple decay used to take 308k steps or
    /// fail outright. SciPy takes ~44 at rtol=1e-8.
    #[test]
    fn test_bdf_regression_step_efficiency() {
        let problem = OdeProblem::new(
            |_t, y: &[f64], dydt: &mut [f64]| {
                dydt[0] = -y[0];
            },
            0.0,
            1.0,
            vec![1.0],
        );
        let options = SolverOptions::default().rtol(1e-6).atol(1e-9);
        let result = Bdf::solve(&problem, 0.0, 1.0, &[1.0], &options).unwrap();
        assert!(result.success);
        let exact = (-1.0_f64).exp();
        assert!((result.y_final().unwrap()[0] - exact).abs() < 1e-5);
        // Loose upper bound. SciPy: ~27.
        assert!(
            result.stats.n_accept < 200,
            "Too many accepted steps: {} (expected ~30, regression bound 200)",
            result.stats.n_accept
        );
    }

    #[test]
    fn test_bdf_simple_dae() {
        let dae = DaeProblem::new(
            |_t, y: &[f64], dydt: &mut [f64]| {
                dydt[0] = -y[0] + y[1];
                dydt[1] = y[0] - y[1];
            },
            |mass: &mut [f64]| {
                for i in 0..4 {
                    mass[i] = 0.0;
                }
                mass[0] = 1.0;
            },
            0.0,
            1.0,
            vec![1.0, 1.0],
            vec![1],
        );
        let options = SolverOptions::default()
            .rtol(1e-4)
            .atol(1e-6)
            .max_steps(500_000);
        let result = Bdf::solve(&dae, 0.0, 1.0, &[1.0, 1.0], &options);
        assert!(result.is_ok(), "DAE solve failed: {:?}", result.err());
        let sol = result.unwrap();
        let yf = sol.y_final().unwrap();
        assert!((yf[0] - 1.0).abs() < 1e-4);
        assert!((yf[1] - 1.0).abs() < 1e-4);
        assert!((yf[0] - yf[1]).abs() < 1e-4);
    }

    #[test]
    fn test_bdf_dae_with_mass_identity() {
        let dae = DaeProblem::new(
            |_t, y: &[f64], dydt: &mut [f64]| {
                dydt[0] = -y[0];
            },
            |mass: &mut [f64]| {
                mass[0] = 1.0;
            },
            0.0,
            1.0,
            vec![1.0],
            vec![],
        );
        let options = SolverOptions::default().rtol(1e-4).atol(1e-6);
        let result = Bdf::solve(&dae, 0.0, 1.0, &[1.0], &options);
        assert!(result.is_ok());
        let yf = result.unwrap().y_final().unwrap();
        let exact = (-1.0_f64).exp();
        assert!((yf[0] - exact).abs() < 1e-3);
    }

    #[test]
    fn test_bdf_dae_scaled_mass() {
        let dae = DaeProblem::new(
            |_t, y: &[f64], dydt: &mut [f64]| {
                dydt[0] = -y[0];
            },
            |mass: &mut [f64]| {
                mass[0] = 2.0;
            },
            0.0,
            1.0,
            vec![1.0],
            vec![],
        );
        let options = SolverOptions::default().rtol(1e-3).atol(1e-5);
        let result = Bdf::solve(&dae, 0.0, 1.0, &[1.0], &options);
        assert!(result.is_ok());
        let yf = result.unwrap().y_final().unwrap();
        let exact = (-0.5_f64).exp();
        assert!((yf[0] - exact).abs() < 1e-2);
    }

    /// Cross-check accepted-step counts against SciPy's BDF reference on the
    /// six problems specified in the BDF rewrite handoff. Bounds are loose
    /// (3-5× the SciPy count) to avoid flake while still catching real
    /// regressions.
    #[test]
    fn test_bdf_step_counts_vs_scipy() {
        struct Probe {
            name: &'static str,
            n_accept: usize,
            scipy_ref: usize,
            bound: usize,
            ok: bool,
        }
        let mut probes: Vec<Probe> = Vec::new();

        // 1) y' = -y on [0, 1] at three tolerance levels.
        for &(rtol, atol, scipy_ref, bound) in &[
            (1e-4, 1e-6, 16usize, 60usize),
            (1e-6, 1e-9, 27, 100),
            (1e-8, 1e-11, 44, 200),
        ] {
            let p = OdeProblem::new(
                |_t, y: &[f64], dy: &mut [f64]| dy[0] = -y[0],
                0.0,
                1.0,
                vec![1.0],
            );
            let opts = SolverOptions::default().rtol(rtol).atol(atol);
            let r = Bdf::solve(&p, 0.0, 1.0, &[1.0], &opts).unwrap();
            let exact = (-1.0_f64).exp();
            let acc = (r.y_final().unwrap()[0] - exact).abs() < 10.0 * rtol;
            probes.push(Probe {
                name: Box::leak(format!("decay rtol={rtol:e}").into_boxed_str()),
                n_accept: r.stats.n_accept,
                scipy_ref,
                bound,
                ok: acc && r.success && r.stats.n_accept < bound,
            });
        }

        // 2) y' = -100 y on [0, 0.1] at rtol=1e-2 (stiff decay).
        {
            let p = OdeProblem::new(
                |_t, y: &[f64], dy: &mut [f64]| dy[0] = -100.0 * y[0],
                0.0,
                0.1,
                vec![1.0],
            );
            let opts = SolverOptions::default().rtol(1e-2).atol(1e-4);
            let r = Bdf::solve(&p, 0.0, 0.1, &[1.0], &opts).unwrap();
            probes.push(Probe {
                name: "stiff y'=-100y",
                n_accept: r.stats.n_accept,
                scipy_ref: 12,
                bound: 50,
                ok: r.success && r.stats.n_accept < 50,
            });
        }

        // 3) 2-state augmented system (the canary regression).
        {
            let p = OdeProblem::new(
                |_t, y: &[f64], dy: &mut [f64]| {
                    dy[0] = -0.5 * y[0];
                    dy[1] = -0.5 * y[1] - y[0];
                },
                0.0,
                2.0,
                vec![1.0, 0.0],
            );
            let opts = SolverOptions::default().rtol(1e-4).atol(1e-6);
            let r = Bdf::solve(&p, 0.0, 2.0, &[1.0, 0.0], &opts).unwrap();
            probes.push(Probe {
                name: "2-state augmented",
                n_accept: r.stats.n_accept,
                scipy_ref: 21,
                bound: 100,
                ok: r.success && r.stats.n_accept < 100,
            });
        }

        // 4) Van der Pol μ=10, t in [0, 20].
        {
            let mu = 10.0_f64;
            let p = OdeProblem::new(
                move |_t, y: &[f64], dy: &mut [f64]| {
                    dy[0] = y[1];
                    dy[1] = mu * (1.0 - y[0] * y[0]) * y[1] - y[0];
                },
                0.0,
                20.0,
                vec![2.0, 0.0],
            );
            let opts = SolverOptions::default().rtol(1e-2).atol(1e-4);
            let r = Bdf::solve(&p, 0.0, 20.0, &[2.0, 0.0], &opts).unwrap();
            probes.push(Probe {
                name: "VdP μ=10",
                n_accept: r.stats.n_accept,
                scipy_ref: 80,
                bound: 400,
                ok: r.success && r.stats.n_accept < 400,
            });
        }

        // Pretty-print the table and fail on any out-of-bound row.
        eprintln!("\n  problem                | n_accept |  scipy  |  bound  |  status");
        eprintln!("  -----------------------+----------+---------+---------+--------");
        for p in &probes {
            eprintln!(
                "  {:<22} | {:>8} | {:>7} | {:>7} | {}",
                p.name,
                p.n_accept,
                p.scipy_ref,
                p.bound,
                if p.ok { "ok" } else { "OVER" },
            );
        }
        assert!(
            probes.iter().all(|p| p.ok),
            "one or more BDF probes exceeded bound or failed accuracy"
        );
    }
}

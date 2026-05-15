//! Radau5: 3-stage Radau IIA implicit Runge-Kutta method (5th order, L-stable).
//!
//! This is a corrected implementation following Hairer & Wanner's algorithm
//! ("Solving Ordinary Differential Equations II", §IV.8) and aligned with the
//! reference implementations in radau5.f (Hairer, Geneva) and SciPy's port
//! (`scipy.integrate.Radau`).
//!
//! ## Mathematical Formulation
//!
//! Solves systems of the form:
//! ```text
//! M * y'(t) = f(t, y)
//! ```
//! where M is a (possibly singular) mass matrix. When M is the identity, this
//! reduces to the standard ODE form y' = f(t, y).
//!
//! **DAE support (index-1):** When M is singular, the system is a
//! differential-algebraic equation. Rows of M with zero diagonal correspond to
//! algebraic constraints. Radau5 handles index-1 DAEs natively because it is
//! L-stable and stiffly accurate (the last stage coincides with the step endpoint).
//!
//! ## Algorithm
//!
//! At each step, the 3-stage system is transformed via real-Schur decomposition
//! of the Radau IIA coefficient matrix A. This reduces the 3×3 block system to
//! one real and one complex linear solve per Newton iteration, cutting the cost
//! from O(3n)³ to O(n)³ per factorization.
//!
//! Step size control uses Hairer's ESTRAD error estimator with refinement on
//! the first step / rejected steps, combined with Gustafsson's predictive
//! controller (Hairer's `IWORK(8) = 1`, the default in radau5.f).
//!
//! ## Corrections vs. previous revision (5 May 2026)
//!
//! 1. (CRITICAL) `error_estimate`: forcing term is `f(t, y)` (the RHS), not
//!    `y` itself. Same bug existed in the mass-matrix branch (was `M*y`).
//!    Also: scale by max(|y|, |y_new|) per Hairer's ESTRAD.
//! 2. Newton initial guess: extrapolated collocation polynomial from the
//!    previous step (Hairer's `STARTN = 0`, the default).
//! 3. LU re-factorization is only triggered when the step ratio leaves
//!    [1.0, 1.2] (Hairer's `QUOT1`/`QUOT2` heuristic).
//! 4. Off-by-one in Newton convergence-rate check (`newt > 1` → `newt >= 1`).
//! 5. Step-controller `nit` is the max Newton iteration count (= 7), not 0.
//! 6. Gustafsson predictive step-size controller (uses prior step's err norm).
//! 7. `facl` is 8.0 (Hairer's default), not 5.0.
//! 8. `f0` is tracked across steps and updated after each accepted step
//!    (required for FIX 1; SciPy does this).
//!
//! ## Known Limitations
//!
//! - Only supports index-1 DAEs (algebraic variables appear linearly).
//! - The error estimator falls back to step rejection when the LU solve fails.
//!
//! ## Jacobian
//!
//! Radau5 calls `OdeSystem::jacobian` for each rebuild. Systems that
//! override the trait method get an analytical Jacobian for free; systems
//! that don't fall through to the canonical forward-FD default in
//! `crate::problem` (`h = sqrt(S::EPSILON) * (1 + |y_j|)`, row-major
//! dense output).
//!
//! ## References
//! - Hairer, E. & Wanner, G. (1996), "Solving Ordinary Differential Equations II:
//!   Stiff and Differential-Algebraic Problems", Springer (2nd ed.), §IV.8.
//! - radau5.f source (E. Hairer), available at <https://www.unige.ch/~hairer/>.
//! - SciPy `scipy/integrate/_ivp/radau.py` (Apache-2.0 / BSD-3 implementation).
//!
//! Author: Moussa Leblouba
//! Date: 10 February 2026
//! Modified: 5 May 2026 (corrections per Hairer-Wanner ODE II §IV.8 and SciPy port)

use faer::{ComplexField, Conjugate, SimpleEntity};
use numra_core::Scalar;
use numra_linalg::{DenseMatrix, LUFactorization, Matrix};

use crate::error::SolverError;
use crate::problem::OdeSystem;
use crate::solver::{Solver, SolverOptions, SolverResult, SolverStats};
use crate::t_eval::{validate_grid, TEvalEmitter};

/// Radau5 solver for stiff ODEs.
#[derive(Clone, Debug, Default)]
pub struct Radau5;

impl Radau5 {
    /// Create a new Radau5 solver.
    pub fn new() -> Self {
        Self
    }
}

/// Radau IIA coefficients and Hairer transformation matrices.
mod coefficients {
    pub const SQRT6: f64 = 2.449489742783178;

    // Stage points (c values)
    pub const C1: f64 = (4.0 - SQRT6) / 10.0; // ≈ 0.1551
    pub const C2: f64 = (4.0 + SQRT6) / 10.0; // ≈ 0.6449
    #[allow(dead_code)]
    pub const C3: f64 = 1.0; // Third Radau node (always 1 for Radau IIA)

    // Error estimation coefficients (DD values from Hairer's ESTRAD)
    pub const DD1: f64 = -(13.0 + 7.0 * SQRT6) / 3.0;
    pub const DD2: f64 = (-13.0 + 7.0 * SQRT6) / 3.0;
    pub const DD3: f64 = -1.0 / 3.0;

    // Eigenvalue-related constants from Hairer-Wanner.
    // 81^(1/3) ≈ 4.3267, 9^(1/3) ≈ 2.0801
    const CUBERT81: f64 = 4.3267487109222245;
    const CUBERT9: f64 = 2.080083823051904;

    // U1 = inverse of real eigenvalue
    const U1_RAW: f64 = (6.0 + CUBERT81 - CUBERT9) / 30.0;
    pub const U1: f64 = 1.0 / U1_RAW; // ≈ 3.6378342527444957

    // Complex eigenvalue: α ± iβ (after normalization)
    const ALPH_RAW: f64 = (12.0 - CUBERT81 + CUBERT9) / 60.0;
    const BETA_RAW: f64 = (CUBERT81 + CUBERT9) * 1.7320508075688772 / 60.0; // sqrt(3)
    const CNO: f64 = ALPH_RAW * ALPH_RAW + BETA_RAW * BETA_RAW;
    pub const ALPH: f64 = ALPH_RAW / CNO; // ≈ 2.6812
    pub const BETA: f64 = BETA_RAW / CNO; // ≈ 3.0504

    // Transformation matrix T (transforms from decoupled to original space).
    // Z = T * F where F is in transformed space. From radau5.f.
    pub const T11: f64 = 9.1232394870892942792e-02;
    pub const T12: f64 = -0.14125529502095420843;
    pub const T13: f64 = -3.0029194105147424492e-02;
    pub const T21: f64 = 0.24171793270710701896;
    pub const T22: f64 = 0.20412935229379993199;
    pub const T23: f64 = 0.38294211275726193779;
    pub const T31: f64 = 0.96604818261509293619;
    pub const T32: f64 = 1.0;
    #[allow(dead_code)]
    pub const T33: f64 = 0.0;

    // Inverse transformation matrix TI = T^{-1}. From radau5.f.
    pub const TI11: f64 = 4.3255798900631553510;
    pub const TI12: f64 = 0.33919925181580986954;
    pub const TI13: f64 = 0.54177053993587487119;
    pub const TI21: f64 = -4.1787185915519047273;
    pub const TI22: f64 = -0.32768282076106238708;
    pub const TI23: f64 = 0.47662355450055045196;
    pub const TI31: f64 = -0.50287263494578687595;
    pub const TI32: f64 = 2.5719269498556054292;
    pub const TI33: f64 = -0.59603920482822492497;

    // ---- Dense-output (continuous extension) coefficients ---------------
    //
    // The cubic collocation polynomial through {(0, y_old), (C1, y+Z1),
    // (C2, y+Z2), (1, y+Z3)} can be written as:
    //   y(t_old + θ*h) = y_old + Σ_k Q[i,k] * θ^(k+1)
    // where Q[i,k] = Σ_j P[j,k] * Z[j,i].
    //
    // We use this to extrapolate the previous step's stages to the current
    // step's abscissae as a Newton initial guess (Hairer's STARTN = 0).
    //
    // Reference: SciPy radau.py (which cites Hairer-Wanner §IV.8).
    pub const P11: f64 = 13.0 / 3.0 + 7.0 * SQRT6 / 3.0;
    pub const P12: f64 = -23.0 / 3.0 - 22.0 * SQRT6 / 3.0;
    pub const P13: f64 = 10.0 / 3.0 + 5.0 * SQRT6;
    pub const P21: f64 = 13.0 / 3.0 - 7.0 * SQRT6 / 3.0;
    pub const P22: f64 = -23.0 / 3.0 + 22.0 * SQRT6 / 3.0;
    pub const P23: f64 = 10.0 / 3.0 - 5.0 * SQRT6;
    pub const P31: f64 = 1.0 / 3.0;
    pub const P32: f64 = -8.0 / 3.0;
    pub const P33: f64 = 10.0 / 3.0;
}

/// Maximum Newton iterations per step (Hairer's NIT default).
const MAX_NEWTON_ITER: usize = 7;

impl<S: Scalar + SimpleEntity + Conjugate<Canonical = S> + ComplexField> Solver<S> for Radau5 {
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

        let direction_init = if tf > t0 { S::ONE } else { -S::ONE };
        if let Some(grid) = options.t_eval.as_deref() {
            validate_grid(grid, t0, tf)?;
        }
        let mut grid_emitter = options
            .t_eval
            .as_deref()
            .map(|g| TEvalEmitter::new(g, direction_init));
        let (mut t_out, mut y_out) = if grid_emitter.is_some() {
            (Vec::new(), Vec::new())
        } else {
            (vec![t0], y0.to_vec())
        };
        // Buffer holding f(t_old, y_old) right before output emission.
        let mut dy_old_buf = vec![S::ZERO; dim];

        // Working arrays
        let mut f0 = vec![S::ZERO; dim];
        let mut z1 = vec![S::ZERO; dim];
        let mut z2 = vec![S::ZERO; dim];
        let mut z3 = vec![S::ZERO; dim];
        let mut w1 = vec![S::ZERO; dim];
        let mut w2 = vec![S::ZERO; dim];
        let mut w3 = vec![S::ZERO; dim];
        let mut cont = vec![S::ZERO; dim];
        let mut scal = vec![S::ZERO; dim];
        let mut y_new = vec![S::ZERO; dim];
        let mut err = vec![S::ZERO; dim];
        let mut jac_data = vec![S::ZERO; dim * dim];

        // FIX 2 state: previous-step stages, used to extrapolate the
        // collocation polynomial as Newton's initial guess.
        let mut z1_prev = vec![S::ZERO; dim];
        let mut z2_prev = vec![S::ZERO; dim];
        let mut z3_prev = vec![S::ZERO; dim];
        let mut h_prev: S = S::ONE; // dummy until first accepted step
        let mut have_prev = false;

        // FIX 6 state: Gustafsson predictive controller history.
        let mut h_abs_old: Option<S> = None;
        let mut err_norm_old: Option<S> = None;

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

        let mut stats = SolverStats::default();

        // Initial scaling
        for i in 0..dim {
            scal[i] = options.atol + options.rtol * y[i].abs();
        }

        // FIX 8: f0 is tracked across steps; initialize once and refresh after
        // every accepted step so that the error estimator (FIX 1) always sees
        // f(t, y) at the start of the current step.
        problem.rhs(t, &y, &mut f0);
        stats.n_eval += 1;

        // Initial step size
        let mut h = Self::initial_step_size(&y, &f0, options, dim);
        let h_min = options.h_min;
        let h_max = (tf - t0).abs() * S::from_f64(0.5);

        // LU factorizations for the decoupled systems.
        let mut lu_real: Option<LUFactorization<S>> = None;
        let mut lu_complex: Option<LUFactorization<S>> = None;
        let mut need_jac = true;

        let mut first = true;
        let mut reject = false;
        let mut step_count = 0usize;
        let direction = if tf > t0 { S::ONE } else { -S::ONE };

        while (tf - t) * direction > S::ZERO {
            if step_count >= options.max_steps {
                return Err(SolverError::MaxIterationsExceeded { t: t.to_f64() });
            }

            // Don't overshoot tf
            if (t + h - tf) * direction > S::ZERO {
                h = tf - t;
            }

            // Recompute Jacobian if needed (set on Newton failure or first step).
            // The Jacobian depends on (t, y), not on h, so an h change alone does
            // NOT trigger a Jacobian recompute -- only an LU refactor.
            //
            // Delegates to OdeSystem::jacobian, which lets a system override
            // with an analytical Jacobian (e.g. MOLSystem*) and otherwise
            // falls through to the canonical FD default in problem.rs.
            // Costs one extra rhs eval per Jacobian rebuild and one allocation
            // for the trait default's internal scratch buffers, vs the previous
            // inlined zero-alloc path; benched within the ≤5% regression
            // bound on Van der Pol stiff workloads (see commit message).
            if need_jac {
                problem.jacobian(t, &y, &mut jac_data);
                stats.n_jac += 1;
                need_jac = false;
                // New Jacobian => existing LU is stale.
                lu_real = None;
                lu_complex = None;
            }

            // FIX 3: refactor LU only when explicitly invalidated. Hairer's
            // QUOT1 = 1.0 / QUOT2 = 1.2 heuristic skips refactorization when
            // h has changed by less than ~20% (we set this in the accept branch).
            if lu_real.is_none() {
                let (e1, e2) = Self::form_transformed_matrices(&jac_data, h, dim, mass_ref);
                lu_real = Some(LUFactorization::new(&e1)?);
                lu_complex = Some(LUFactorization::new(&e2)?);
                stats.n_lu += 2;
            }

            // Update scaling for this step
            for i in 0..dim {
                scal[i] = options.atol + options.rtol * y[i].abs();
            }

            // FIX 2: Newton initial guess.
            //   - On the first step or after a rejection, use zero.
            //   - Otherwise, extrapolate from the previous step's collocation
            //     polynomial. This typically reduces Newton iterations 5-7 -> 1-3.
            let use_extrapolation = !first && !reject && have_prev;
            if !use_extrapolation {
                for i in 0..dim {
                    z1[i] = S::ZERO;
                    z2[i] = S::ZERO;
                    z3[i] = S::ZERO;
                    w1[i] = S::ZERO;
                    w2[i] = S::ZERO;
                    w3[i] = S::ZERO;
                }
            } else {
                // Q[i,k] = Σ_j P[j,k] * Z_prev[j,i].
                // Then Z_init[k,i] = q0*r[k] + q1*r[k]^2 + q2*r[k]^3
                // where r[k] = h * C[k] / h_prev (relative position in
                // the previous step's parameterization).
                let p11 = S::from_f64(coefficients::P11);
                let p12 = S::from_f64(coefficients::P12);
                let p13 = S::from_f64(coefficients::P13);
                let p21 = S::from_f64(coefficients::P21);
                let p22 = S::from_f64(coefficients::P22);
                let p23 = S::from_f64(coefficients::P23);
                let p31 = S::from_f64(coefficients::P31);
                let p32 = S::from_f64(coefficients::P32);
                let p33 = S::from_f64(coefficients::P33);

                let c1 = S::from_f64(coefficients::C1);
                let c2 = S::from_f64(coefficients::C2);
                let c3 = S::ONE;

                let r1 = h * c1 / h_prev;
                let r2 = h * c2 / h_prev;
                let r3 = h * c3 / h_prev;

                for i in 0..dim {
                    let q0 = z1_prev[i] * p11 + z2_prev[i] * p21 + z3_prev[i] * p31;
                    let q1 = z1_prev[i] * p12 + z2_prev[i] * p22 + z3_prev[i] * p32;
                    let q2 = z1_prev[i] * p13 + z2_prev[i] * p23 + z3_prev[i] * p33;

                    z1[i] = q0 * r1 + q1 * r1 * r1 + q2 * r1 * r1 * r1;
                    z2[i] = q0 * r2 + q1 * r2 * r2 + q2 * r2 * r2 * r2;
                    z3[i] = q0 * r3 + q1 * r3 * r3 + q2 * r3 * r3 * r3;
                }

                // W = TI * Z (must be consistent with Z so the Newton inner
                // loop's back-transform Z = T*W stays stable).
                let ti11 = S::from_f64(coefficients::TI11);
                let ti12 = S::from_f64(coefficients::TI12);
                let ti13 = S::from_f64(coefficients::TI13);
                let ti21 = S::from_f64(coefficients::TI21);
                let ti22 = S::from_f64(coefficients::TI22);
                let ti23 = S::from_f64(coefficients::TI23);
                let ti31 = S::from_f64(coefficients::TI31);
                let ti32 = S::from_f64(coefficients::TI32);
                let ti33 = S::from_f64(coefficients::TI33);

                for i in 0..dim {
                    w1[i] = ti11 * z1[i] + ti12 * z2[i] + ti13 * z3[i];
                    w2[i] = ti21 * z1[i] + ti22 * z2[i] + ti23 * z3[i];
                    w3[i] = ti31 * z1[i] + ti32 * z2[i] + ti33 * z3[i];
                }
            }

            // Simplified Newton iteration (in transformed space).
            let newton_result = Self::newton_iteration(
                problem,
                t,
                h,
                &y,
                &scal,
                &mut z1,
                &mut z2,
                &mut z3,
                &mut w1,
                &mut w2,
                &mut w3,
                &mut cont,
                lu_real.as_ref().unwrap(),
                lu_complex.as_ref().unwrap(),
                mass_ref,
                &mut stats,
                dim,
                options,
            );

            let (newton_converged, newt_iter) = match newton_result {
                Ok((converged, iter)) => (converged, iter),
                Err(_) => (false, MAX_NEWTON_ITER),
            };

            if !newton_converged {
                // Newton failed -- reduce step size, force fresh Jacobian.
                h = h * S::from_f64(0.5);
                stats.n_reject += 1;
                reject = true;
                need_jac = true;

                if h.abs() < h_min {
                    return Err(SolverError::StepSizeTooSmall {
                        t: t.to_f64(),
                        h: h.to_f64(),
                        h_min: h_min.to_f64(),
                    });
                }
                continue;
            }

            // Compute new solution candidate: y_new = y + Z3 (since c3 = 1).
            for i in 0..dim {
                y_new[i] = y[i] + z3[i];
            }

            // FIX 1 + FIX 1b: error estimation now takes f0 explicitly and
            // scales by max(|y|, |y_new|).
            let err_norm = Self::error_estimate(
                problem,
                t,
                &f0,
                &z1,
                &z2,
                &z3,
                &y,
                &y_new,
                h,
                options,
                lu_real.as_ref().unwrap(),
                &mut err,
                dim,
                first,
                reject,
                &mut stats,
                mass_ref,
            );

            // FIX 5 + FIX 6: Gustafsson predictive controller with safety
            // factor 0.9 * (2*NIT+1) / (2*NIT + newt_iter), classical bound
            // [0.2, 8.0] (FIX 7).
            let safety = Self::safety_factor::<S>(newt_iter, MAX_NEWTON_ITER);
            let pred = Self::predict_factor(h.abs(), h_abs_old, err_norm, err_norm_old);
            let factor = (safety * pred).max(S::from_f64(0.2)).min(S::from_f64(8.0));

            if err_norm < S::ONE {
                // ----- Step accepted -----
                stats.n_accept += 1;

                // Save stages and h for next step's extrapolation (FIX 2).
                z1_prev.copy_from_slice(&z1);
                z2_prev.copy_from_slice(&z2);
                z3_prev.copy_from_slice(&z3);
                h_prev = h;
                have_prev = true;

                // Update Gustafsson state (FIX 6).
                h_abs_old = Some(h.abs());
                err_norm_old = Some(err_norm);

                let t_new = t + h;
                // Save the slope at the start of the step before we
                // overwrite f0 with the slope at the end (used both for the
                // next-step error estimator and Hermite interpolation).
                dy_old_buf.copy_from_slice(&f0);
                problem.rhs(t_new, &y_new, &mut f0);
                stats.n_eval += 1;

                if let Some(ref mut emitter) = grid_emitter {
                    emitter.emit_step(
                        t,
                        &y,
                        &dy_old_buf,
                        t_new,
                        &y_new,
                        &f0,
                        &mut t_out,
                        &mut y_out,
                    );
                } else {
                    t_out.push(t_new);
                    y_out.extend_from_slice(&y_new);
                }

                t = t_new;
                y.copy_from_slice(&y_new);

                first = false;
                reject = false;

                // FIX 3: only invalidate LU when factor >= 1.2 (Hairer's
                // QUOT2 = 1.2 heuristic). Small step changes don't justify
                // an LU refactor; keep h and LU in that case.
                if factor < S::from_f64(1.2) {
                    // Don't change h, don't refactor LU.
                } else {
                    let h_proposed = h * factor;
                    let h_capped = if h_proposed.abs() > h_max {
                        if h_proposed > S::ZERO {
                            h_max
                        } else {
                            -h_max
                        }
                    } else {
                        h_proposed
                    };
                    h = h_capped;
                    lu_real = None;
                    lu_complex = None;
                }
            } else {
                // ----- Step rejected -----
                stats.n_reject += 1;
                reject = true;

                // Always shrink h and refactor LU on rejection.
                h = h * factor;
                lu_real = None;
                lu_complex = None;

                if h.abs() < h_min {
                    return Err(SolverError::StepSizeTooSmall {
                        t: t.to_f64(),
                        h: h.to_f64(),
                        h_min: h_min.to_f64(),
                    });
                }
            }

            step_count += 1;
        }

        Ok(SolverResult::new(t_out, y_out, dim, stats))
    }
}

impl Radau5 {
    /// Initial step size estimation (simple heuristic, not Hairer's full HINIT).
    fn initial_step_size<S: Scalar>(y: &[S], f: &[S], options: &SolverOptions<S>, dim: usize) -> S {
        let mut d0 = S::ZERO;
        let mut d1 = S::ZERO;

        for i in 0..dim {
            let sc = options.atol + options.rtol * y[i].abs();
            d0 = d0 + (y[i] / sc) * (y[i] / sc);
            d1 = d1 + (f[i] / sc) * (f[i] / sc);
        }

        let d0 = (d0 / S::from_usize(dim)).sqrt();
        let d1 = (d1 / S::from_usize(dim)).sqrt();

        let h0 = if d0 < S::from_f64(1e-5) || d1 < S::from_f64(1e-5) {
            S::from_f64(1e-6)
        } else {
            S::from_f64(0.01) * d0 / d1
        };

        h0.min(options.h_max).max(options.h_min)
    }

    /// Compute Jacobian by finite differences.
    /// Form the transformed iteration matrices E1 (n×n real) and E2 (2n×2n
    /// real form of the complex system).
    ///
    /// For ODEs (identity mass):
    ///   E1 = (U1/h)*I - J
    ///   E2 = real form of ((α + iβ)/h)*I - J
    ///
    /// For DAEs (general mass M):
    ///   E1 = (U1/h)*M - J
    ///   E2 uses M instead of I in the diagonal blocks.
    fn form_transformed_matrices<S>(
        jac: &[S],
        h: S,
        dim: usize,
        mass: Option<&[S]>,
    ) -> (DenseMatrix<S>, DenseMatrix<S>)
    where
        S: Scalar + SimpleEntity + Conjugate<Canonical = S> + ComplexField,
    {
        let fac1 = S::from_f64(coefficients::U1) / h;
        let mut e1 = DenseMatrix::zeros(dim, dim);
        for i in 0..dim {
            for j in 0..dim {
                let jij = jac[i * dim + j];
                let mij = match mass {
                    Some(m) => m[i * dim + j],
                    None => {
                        if i == j {
                            S::ONE
                        } else {
                            S::ZERO
                        }
                    }
                };
                e1.set(i, j, fac1 * mij - jij);
            }
        }

        // E2 is the 2n×2n real matrix for the complex system:
        //   | alphn*M - J   -betan*M       |
        //   | betan*M        alphn*M - J   |
        // representing (alphn + i*betan)*M - J acting on (W2 + i*W3).
        let alphn = S::from_f64(coefficients::ALPH) / h;
        let betan = S::from_f64(coefficients::BETA) / h;
        let mut e2 = DenseMatrix::zeros(2 * dim, 2 * dim);

        for i in 0..dim {
            for j in 0..dim {
                let jij = jac[i * dim + j];
                let mij = match mass {
                    Some(m) => m[i * dim + j],
                    None => {
                        if i == j {
                            S::ONE
                        } else {
                            S::ZERO
                        }
                    }
                };
                e2.set(i, j, alphn * mij - jij);
                e2.set(i, dim + j, -betan * mij);
                e2.set(dim + i, j, betan * mij);
                e2.set(dim + i, dim + j, alphn * mij - jij);
            }
        }

        (e1, e2)
    }

    /// Safety factor for step-size selection (FIX 5).
    ///
    /// SciPy's formulation: `safety = 0.9 * (2*NIT + 1) / (2*NIT + n_iter)`.
    /// This is equivalent in spirit to Hairer's `min(SAFE, CFAC/(NEWT+2*NIT))`
    /// — the more Newton iterations a step needed, the more conservatively we
    /// scale h.
    fn safety_factor<S: Scalar>(n_iter: usize, max_iter: usize) -> S {
        let num = 0.9 * (2.0 * max_iter as f64 + 1.0);
        let den = 2.0 * max_iter as f64 + n_iter as f64;
        S::from_f64(num / den)
    }

    /// Gustafsson predictive step-size factor (FIX 6).
    ///
    /// Returns the multiplier such that `h_new = h * factor`. With history,
    /// the predictor multiplies the classical err^{-1/4} factor by
    /// `(h_old/h)^{-1} * (err_old/err)^{1/4}` (capped at 1) to brake when
    /// errors are trending up.
    ///
    /// References:
    /// - Gustafsson (1991), "Control theoretic techniques for stepsize selection
    ///   in explicit Runge-Kutta methods".
    /// - Hairer-Wanner ODE II, §IV.8 (PI controller variant).
    /// - SciPy `radau.py::predict_factor`.
    fn predict_factor<S: Scalar>(
        h_abs: S,
        h_abs_old: Option<S>,
        err_norm: S,
        err_norm_old: Option<S>,
    ) -> S {
        let multiplier = match (h_abs_old, err_norm_old) {
            (Some(h_old), Some(err_old)) if err_norm > S::ZERO && h_old > S::ZERO => {
                (h_abs / h_old) * (err_old / err_norm).powf(S::from_f64(0.25))
            }
            _ => S::ONE,
        };
        multiplier.min(S::ONE) * err_norm.powf(S::from_f64(-0.25))
    }

    /// Newton iteration in transformed space.
    ///
    /// Solves the implicit Radau IIA stage equations using a simplified Newton
    /// iteration on the transformed variables W = TI * Z. The 3×3 block system
    /// decouples into one real solve (E1) and one complex solve (E2).
    ///
    /// For DAEs with mass matrix M, the residual involves M*Z; the algorithm
    /// computes M*Z in the original space and then transforms.
    ///
    /// FIX 4: convergence-rate (theta) check now uses `newt >= 1` (matching
    /// Hairer's NEWT > 1 in 1-based Fortran), not the original `newt > 1`.
    #[allow(clippy::too_many_arguments)]
    fn newton_iteration<S, Sys>(
        problem: &Sys,
        t: S,
        h: S,
        y: &[S],
        scal: &[S],
        z1: &mut [S],
        z2: &mut [S],
        z3: &mut [S],
        w1: &mut [S],
        w2: &mut [S],
        w3: &mut [S],
        cont: &mut [S],
        lu_real: &LUFactorization<S>,
        lu_complex: &LUFactorization<S>,
        mass: Option<&[S]>,
        stats: &mut SolverStats,
        dim: usize,
        options: &SolverOptions<S>,
    ) -> Result<(bool, usize), SolverError>
    where
        S: Scalar + SimpleEntity + Conjugate<Canonical = S> + ComplexField,
        Sys: OdeSystem<S>,
    {
        let c1 = S::from_f64(coefficients::C1);
        let c2 = S::from_f64(coefficients::C2);

        // FNEWT: Newton tolerance, per Hairer's formula
        //   FNEWT = max(10*UROUND/RTOL, min(0.03, sqrt(RTOL))).
        let uround = S::from_f64(1e-16);
        let fnewt = (S::from_f64(10.0) * uround / options.rtol)
            .max(S::from_f64(0.03).min(options.rtol.sqrt()));

        // Eigenvalue scalings
        let fac1 = S::from_f64(coefficients::U1) / h;
        let alphn = S::from_f64(coefficients::ALPH) / h;
        let betan = S::from_f64(coefficients::BETA) / h;

        // T and TI (used inside the loop)
        let ti11 = S::from_f64(coefficients::TI11);
        let ti12 = S::from_f64(coefficients::TI12);
        let ti13 = S::from_f64(coefficients::TI13);
        let ti21 = S::from_f64(coefficients::TI21);
        let ti22 = S::from_f64(coefficients::TI22);
        let ti23 = S::from_f64(coefficients::TI23);
        let ti31 = S::from_f64(coefficients::TI31);
        let ti32 = S::from_f64(coefficients::TI32);
        let ti33 = S::from_f64(coefficients::TI33);

        let t11 = S::from_f64(coefficients::T11);
        let t12 = S::from_f64(coefficients::T12);
        let t13 = S::from_f64(coefficients::T13);
        let t21 = S::from_f64(coefficients::T21);
        let t22 = S::from_f64(coefficients::T22);
        let t23 = S::from_f64(coefficients::T23);
        let t31 = S::from_f64(coefficients::T31);
        let t32 = S::from_f64(coefficients::T32);
        // T33 = 0

        // State for convergence diagnostics
        let mut dynold: S = uround;
        let mut thqold: S = S::ONE;
        let mut faccon: S = S::ONE;

        let n3 = S::from_usize(3 * dim);

        // Pre-allocated buffers (avoid per-iteration allocation)
        let mut f2_temp = vec![S::ZERO; dim];
        let mut f3_temp = vec![S::ZERO; dim];
        let mut z1_orig = vec![S::ZERO; dim];
        let mut z2_orig = vec![S::ZERO; dim];
        let mut z3_orig = vec![S::ZERO; dim];
        let mut mz1_buf = vec![S::ZERO; dim];
        let mut mz2_buf = vec![S::ZERO; dim];
        let mut mz3_buf = vec![S::ZERO; dim];
        let mut rhs1 = vec![S::ZERO; dim];
        let mut rhs2 = vec![S::ZERO; dim];
        let mut rhs3 = vec![S::ZERO; dim];
        let mut rhs_complex = vec![S::ZERO; 2 * dim];

        for newt in 0..MAX_NEWTON_ITER {
            // Stage RHS evaluations: Y_i = y + Z_i, F_i = f(t + C_i*h, Y_i)
            for i in 0..dim {
                cont[i] = y[i] + z1[i];
            }
            problem.rhs(t + c1 * h, cont, z1); // store F1 in z1 temporarily

            for i in 0..dim {
                cont[i] = y[i] + z2[i];
            }
            problem.rhs(t + c2 * h, cont, &mut f2_temp);

            for i in 0..dim {
                cont[i] = y[i] + z3[i];
            }
            problem.rhs(t + h, cont, &mut f3_temp);
            stats.n_eval += 3;

            // Recompute Z = T*W from W (we just clobbered z1 with F1).
            for i in 0..dim {
                z1_orig[i] = t11 * w1[i] + t12 * w2[i] + t13 * w3[i];
                z2_orig[i] = t21 * w1[i] + t22 * w2[i] + t23 * w3[i];
                z3_orig[i] = t31 * w1[i] + t32 * w2[i]; // T33 = 0
            }

            // M*Z for each stage. For identity mass M*Z = Z.
            if let Some(m) = mass {
                for i in 0..dim {
                    mz1_buf[i] = S::ZERO;
                    mz2_buf[i] = S::ZERO;
                    mz3_buf[i] = S::ZERO;
                }
                for i in 0..dim {
                    for j in 0..dim {
                        let mij = m[i * dim + j];
                        mz1_buf[i] = mz1_buf[i] + mij * z1_orig[j];
                        mz2_buf[i] = mz2_buf[i] + mij * z2_orig[j];
                        mz3_buf[i] = mz3_buf[i] + mij * z3_orig[j];
                    }
                }
            } else {
                mz1_buf.copy_from_slice(&z1_orig);
                mz2_buf.copy_from_slice(&z2_orig);
                mz3_buf.copy_from_slice(&z3_orig);
            }

            // Build transformed RHS.
            //   RHS_real    = (TI*F)[0]  -  fac1  *  (TI*M*Z)[0]
            //   RHS_complex = (TI*F)[1]  +  i*(TI*F)[2]
            //               - (alphn+i*betan) * ((TI*M*Z)[1] + i*(TI*M*Z)[2])
            for i in 0..dim {
                let a1 = z1[i]; // F1 still here
                let a2 = f2_temp[i];
                let a3 = f3_temp[i];
                let tf1 = ti11 * a1 + ti12 * a2 + ti13 * a3;
                let tf2 = ti21 * a1 + ti22 * a2 + ti23 * a3;
                let tf3 = ti31 * a1 + ti32 * a2 + ti33 * a3;

                let tmz1 = ti11 * mz1_buf[i] + ti12 * mz2_buf[i] + ti13 * mz3_buf[i];
                let tmz2 = ti21 * mz1_buf[i] + ti22 * mz2_buf[i] + ti23 * mz3_buf[i];
                let tmz3 = ti31 * mz1_buf[i] + ti32 * mz2_buf[i] + ti33 * mz3_buf[i];

                rhs1[i] = tf1 - fac1 * tmz1;
                rhs2[i] = tf2 - alphn * tmz2 + betan * tmz3;
                rhs3[i] = tf3 - alphn * tmz3 - betan * tmz2;
            }

            // Decoupled linear solves
            let dw1 = lu_real.solve(&rhs1)?;

            for i in 0..dim {
                rhs_complex[i] = rhs2[i];
                rhs_complex[dim + i] = rhs3[i];
            }
            let dw_complex = lu_complex.solve(&rhs_complex)?;

            // DYNO: scaled RMS norm of correction (ΔW)
            let mut dyno = S::ZERO;
            for i in 0..dim {
                let denom = scal[i];
                dyno = dyno
                    + (dw1[i] / denom) * (dw1[i] / denom)
                    + (dw_complex[i] / denom) * (dw_complex[i] / denom)
                    + (dw_complex[dim + i] / denom) * (dw_complex[dim + i] / denom);
            }
            dyno = (dyno / n3).sqrt();

            // FIX 4: Convergence-rate check matches Hairer's NEWT > 1 in
            // 1-based Fortran; in our 0-based loop, `newt >= 1` means
            // "from the second iteration onwards".
            if (1..MAX_NEWTON_ITER - 1).contains(&newt) {
                let thq = dyno / dynold;
                let theta = if newt == 1 {
                    thq
                } else {
                    (thq * thqold).sqrt()
                };
                thqold = thq;

                if theta < S::from_f64(0.99) {
                    faccon = theta / (S::ONE - theta);
                    let dyth =
                        faccon * dyno * theta.powf(S::from_usize(MAX_NEWTON_ITER - 1 - newt))
                            / fnewt;
                    if dyth >= S::ONE {
                        return Ok((false, newt + 1));
                    }
                } else {
                    return Ok((false, newt + 1));
                }
            }
            dynold = dyno.max(uround);

            // Accumulate: W += dW (transformed space)
            for i in 0..dim {
                w1[i] = w1[i] + dw1[i];
                w2[i] = w2[i] + dw_complex[i];
                w3[i] = w3[i] + dw_complex[dim + i];
            }

            // Back-transform: Z = T * W
            for i in 0..dim {
                z1[i] = t11 * w1[i] + t12 * w2[i] + t13 * w3[i];
                z2[i] = t21 * w1[i] + t22 * w2[i] + t23 * w3[i];
                z3[i] = t31 * w1[i] + t32 * w2[i]; // T33 = 0
            }

            // Convergence test
            if faccon * dyno <= fnewt {
                return Ok((true, newt + 1));
            }
        }

        Ok((false, MAX_NEWTON_ITER))
    }

    /// Hairer's ESTRAD error estimate with refinement on first/rejected steps.
    ///
    /// The error estimator solves
    ///   (U1/h * M - J) * ê  =  f(t, y)  +  M (DD1*Z1 + DD2*Z2 + DD3*Z3) / h
    /// and returns the scaled RMS norm of ê.
    ///
    /// FIX 1 (CRITICAL): the forcing term is f(t, y), the RHS, not y itself.
    ///   The previous version used `y[i]` (and `M*y` in the mass-matrix branch),
    ///   which is dimensionally wrong and made the estimator return ~|y|
    ///   regardless of the actual local truncation error.
    ///
    /// FIX 1b: scale uses max(|y|, |y_new|), per Hairer's ESTRAD.
    #[allow(clippy::too_many_arguments)]
    fn error_estimate<S, Sys>(
        problem: &Sys,
        t: S,
        f0: &[S],
        z1: &[S],
        z2: &[S],
        z3: &[S],
        y: &[S],
        y_new: &[S],
        h: S,
        options: &SolverOptions<S>,
        lu_real: &LUFactorization<S>,
        err: &mut [S],
        dim: usize,
        first: bool,
        reject: bool,
        stats: &mut SolverStats,
        mass: Option<&[S]>,
    ) -> S
    where
        S: Scalar + SimpleEntity + Conjugate<Canonical = S> + ComplexField,
        Sys: OdeSystem<S>,
    {
        let dd1 = S::from_f64(coefficients::DD1);
        let dd2 = S::from_f64(coefficients::DD2);
        let dd3 = S::from_f64(coefficients::DD3);

        // f2 = DD1*Z1 + DD2*Z2 + DD3*Z3 (the integrated stage residual)
        let mut f2 = vec![S::ZERO; dim];
        for i in 0..dim {
            f2[i] = dd1 * z1[i] + dd2 * z2[i] + dd3 * z3[i];
        }

        let mut cont = vec![S::ZERO; dim];
        if let Some(m) = mass {
            // Mass-matrix case: cont = (M*f2)/h + f0
            let mut mf2 = vec![S::ZERO; dim];
            for i in 0..dim {
                for j in 0..dim {
                    mf2[i] = mf2[i] + m[i * dim + j] * f2[j];
                }
            }
            for i in 0..dim {
                cont[i] = mf2[i] / h + f0[i]; // FIX 1: f0, not M*y
            }
            // Save (M*f2)/h in f2 for the refinement branch below
            for i in 0..dim {
                f2[i] = mf2[i] / h;
            }
        } else {
            // Identity mass: cont = f2/h + f0
            for i in 0..dim {
                f2[i] = f2[i] / h;
                cont[i] = f2[i] + f0[i]; // FIX 1: f0, not y
            }
        }

        // Solve E1 * ê = cont (E1 = U1/h*M - J already factored)
        let solved = match lu_real.solve(&cont) {
            Ok(s) => s,
            Err(_) => return S::from_f64(1e6),
        };

        // FIX 1b: scale by max(|y|, |y_new|).
        let mut err_norm = S::ZERO;
        for i in 0..dim {
            err[i] = solved[i];
            let y_max = y[i].abs().max(y_new[i].abs());
            let scale = options.atol + options.rtol * y_max;
            let r = solved[i] / scale;
            err_norm = err_norm + r * r;
        }
        let err_norm = (err_norm / S::from_usize(dim)).sqrt();
        let err_norm = err_norm.max(S::from_f64(1e-10));

        // Refinement: on first/rejected steps, if err >= 1, redo the solve
        // using f(t, y + ê) on the right-hand side. This typically halves the
        // estimate when transients are dominating.
        if err_norm >= S::ONE && (first || reject) {
            for i in 0..dim {
                cont[i] = y[i] + solved[i];
            }
            let mut f1 = vec![S::ZERO; dim];
            problem.rhs(t, &cont, &mut f1);
            stats.n_eval += 1;

            for i in 0..dim {
                cont[i] = f1[i] + f2[i];
            }
            let solved2 = match lu_real.solve(&cont) {
                Ok(s) => s,
                Err(_) => return S::from_f64(1e6),
            };

            let mut err_norm2 = S::ZERO;
            for i in 0..dim {
                err[i] = solved2[i];
                let y_max = y[i].abs().max(y_new[i].abs());
                let scale = options.atol + options.rtol * y_max;
                let r = solved2[i] / scale;
                err_norm2 = err_norm2 + r * r;
            }
            let err_norm2 = (err_norm2 / S::from_usize(dim)).sqrt();
            return err_norm2.max(S::from_f64(1e-10));
        }

        err_norm
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::problem::{DaeProblem, OdeProblem};

    #[test]
    fn test_radau5_stiff_decay() {
        let problem = OdeProblem::new(
            |_t, y: &[f64], dydt: &mut [f64]| {
                dydt[0] = -100.0 * y[0];
            },
            0.0,
            0.1,
            vec![1.0],
        );
        let options = SolverOptions::default().rtol(1e-2).atol(1e-4);
        let result = Radau5::solve(&problem, 0.0, 0.1, &[1.0], &options).unwrap();
        assert!(result.success);
        let y_final = result.y_final().unwrap();
        let exact = (-10.0_f64).exp();
        assert!(
            (y_final[0] - exact).abs() < 1e-4,
            "Error: {}",
            (y_final[0] - exact).abs()
        );
    }

    #[test]
    fn test_radau5_exponential() {
        let problem = OdeProblem::new(
            |_t, y: &[f64], dydt: &mut [f64]| {
                dydt[0] = y[0];
            },
            0.0,
            1.0,
            vec![1.0],
        );
        let options = SolverOptions::default().rtol(1e-6).atol(1e-8);
        let result = Radau5::solve(&problem, 0.0, 1.0, &[1.0], &options).unwrap();
        assert!(result.success);
        let y_final = result.y_final().unwrap();
        let exact = 1.0_f64.exp();
        assert!((y_final[0] - exact).abs() < 1e-5);
    }

    #[test]
    fn test_radau5_linear_2d() {
        let problem = OdeProblem::new(
            |_t, y: &[f64], dydt: &mut [f64]| {
                dydt[0] = -y[0] + y[1];
                dydt[1] = -y[0] - y[1];
            },
            0.0,
            1.0,
            vec![1.0, 0.0],
        );
        let options = SolverOptions::default().rtol(1e-4).atol(1e-6);
        let result = Radau5::solve(&problem, 0.0, 1.0, &[1.0, 0.0], &options).unwrap();
        assert!(result.success);
    }

    #[test]
    fn test_radau5_van_der_pol_mild() {
        let mu = 10.0;
        let problem = OdeProblem::new(
            move |_t, y: &[f64], dydt: &mut [f64]| {
                dydt[0] = y[1];
                dydt[1] = mu * (1.0 - y[0] * y[0]) * y[1] - y[0];
            },
            0.0,
            2.0,
            vec![2.0, 0.0],
        );
        let options = SolverOptions::default().rtol(1e-4).atol(1e-6);
        let result = Radau5::solve(&problem, 0.0, 2.0, &[2.0, 0.0], &options);
        assert!(result.is_ok());
    }

    #[test]
    fn test_radau5_van_der_pol_stiff() {
        let mu = 100.0;
        let problem = OdeProblem::new(
            move |_t, y: &[f64], dydt: &mut [f64]| {
                dydt[0] = y[1];
                dydt[1] = mu * (1.0 - y[0] * y[0]) * y[1] - y[0];
            },
            0.0,
            20.0,
            vec![2.0, 0.0],
        );
        let options = SolverOptions::default().rtol(1e-3).atol(1e-5);
        let result = Radau5::solve(&problem, 0.0, 20.0, &[2.0, 0.0], &options);
        assert!(
            result.is_ok(),
            "Van der Pol μ=100 failed: {:?}",
            result.err()
        );
    }

    #[test]
    fn test_radau5_step_efficiency() {
        // After the corrections (Hairer's algorithm with Gustafsson controller,
        // extrapolated Newton initial guess, and the FIX 1 error-estimator
        // bugfix), this problem should accept ~15 steps — comparable to SciPy
        // (~11) and Hairer's reference. Before the fix, it took thousands.
        let mu = 100.0;
        let problem = OdeProblem::new(
            move |_t, y: &[f64], dydt: &mut [f64]| {
                dydt[0] = y[1];
                dydt[1] = mu * (1.0 - y[0] * y[0]) * y[1] - y[0];
            },
            0.0,
            20.0,
            vec![2.0, 0.0],
        );
        let options = SolverOptions::default().rtol(1e-3).atol(1e-5);
        let result = Radau5::solve(&problem, 0.0, 20.0, &[2.0, 0.0], &options).unwrap();

        // Loose upper bound (would fail dramatically if any FIX regressed).
        assert!(
            result.stats.n_accept < 200,
            "Too many accepted steps: {} (expected < 200, ~15 typical)",
            result.stats.n_accept
        );
        assert!(result.success);
    }

    #[test]
    fn test_radau5_simple_dae() {
        // y1' = -y1 + y2       (differential equation)
        // 0   =  y1 - y2       (algebraic constraint: y2 = y1)
        // Mass matrix: diag(1, 0). Analytical solution: y1 = y2 = 1 (constant).
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
        let result = Radau5::solve(&dae, 0.0, 1.0, &[1.0, 1.0], &options);

        assert!(result.is_ok(), "DAE solve failed: {:?}", result.err());
        let sol = result.unwrap();

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
    fn test_radau5_dae_with_mass_identity() {
        // dy/dt = -y, y(0)=1 ⇒ y(t)=exp(-t). Identity mass via DaeProblem.
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

        let options = SolverOptions::default().rtol(1e-6).atol(1e-8);
        let result = Radau5::solve(&dae, 0.0, 1.0, &[1.0], &options);

        assert!(
            result.is_ok(),
            "DAE with identity mass failed: {:?}",
            result.err()
        );
        let sol = result.unwrap();
        let yf = sol.y_final().unwrap();
        let exact = (-1.0_f64).exp();
        assert!(
            (yf[0] - exact).abs() < 1e-5,
            "Error: {} (expected {}, got {})",
            (yf[0] - exact).abs(),
            exact,
            yf[0]
        );
    }

    #[test]
    fn test_radau5_dae_scaled_mass() {
        // 2*y' = -y ⇒ y(t) = exp(-t/2).
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

        let options = SolverOptions::default().rtol(1e-4).atol(1e-6);
        let result = Radau5::solve(&dae, 0.0, 1.0, &[1.0], &options);

        assert!(
            result.is_ok(),
            "DAE with scaled mass failed: {:?}",
            result.err()
        );
        let sol = result.unwrap();
        let yf = sol.y_final().unwrap();
        let exact = (-0.5_f64).exp();
        assert!(
            (yf[0] - exact).abs() < 1e-3,
            "Error: {} (expected {}, got {})",
            (yf[0] - exact).abs(),
            exact,
            yf[0]
        );
    }
}

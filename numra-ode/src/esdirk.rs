//! ESDIRK: Explicit first Stage, Singly Diagonally Implicit Runge-Kutta methods.
//!
//! ESDIRK methods are efficient implicit methods for stiff ODEs.
//! They have the property that all implicit stages share the same diagonal
//! coefficient, allowing efficient Jacobian reuse.
//!
//! ## Available Methods
//! - `Esdirk32` - ESDIRK2(1)3L\[2\]SA: 3-stage, 2nd order (A-stable, L-stable)
//! - `Esdirk43` - ESDIRK3(2)4L\[2\]SA: 4-stage, 3rd order (A-stable, L-stable)
//! - `Esdirk54` - ESDIRK4(3)6L\[2\]SA: 6-stage, 4th order (L-stable, stiffly-accurate)
//!
//! ## Reference
//! - Kennedy, C.A. & Carpenter, M.H. (2016), "Diagonally Implicit Runge-Kutta
//!   Methods for Ordinary Differential Equations. A Review", NASA/TM-2016-219173.
//!
//! Author: Moussa Leblouba
//! Date: 5 March 2026
//! Modified: 2 May 2026

use faer::{ComplexField, Conjugate, SimpleEntity};
use numra_core::Scalar;
use numra_linalg::{DenseMatrix, LUFactorization, Matrix};

use crate::error::SolverError;
use crate::problem::OdeSystem;
use crate::solver::{Solver, SolverOptions, SolverResult, SolverStats};

// ============================================================================
// ESDIRK3(2) - 3 stages, 2nd order with embedded 1st order
// ============================================================================

/// ESDIRK3(2): 3-stage, 2nd order ESDIRK method.
#[derive(Clone, Debug, Default)]
pub struct Esdirk32;

impl Esdirk32 {
    pub fn new() -> Self {
        Self
    }
}

/// ESDIRK3(2) coefficients.
mod esdirk32_tableau {
    // Diagonal coefficient (gamma)
    pub const GAMMA: f64 = 0.2928932188134525; // (2 - sqrt(2)) / 2

    pub const C: [f64; 3] = [0.0, 2.0 * GAMMA, 1.0];

    pub const A: [[f64; 3]; 3] = [
        [0.0, 0.0, 0.0],
        [GAMMA, GAMMA, 0.0],
        [1.0 - 2.0 * GAMMA, GAMMA, GAMMA],
    ];

    pub const B: [f64; 3] = [1.0 - 2.0 * GAMMA, GAMMA, GAMMA];

    // Error estimation (embedded 1st order)
    pub const E: [f64; 3] = [1.0 - 2.0 * GAMMA - 0.5, GAMMA - 0.0, GAMMA - 0.5];
}

// ============================================================================
// ESDIRK4(3) - 4 stages, 3rd order with embedded 2nd order
// ============================================================================

/// ESDIRK4(3): 4-stage, 3rd order ESDIRK method.
#[derive(Clone, Debug, Default)]
pub struct Esdirk43;

impl Esdirk43 {
    pub fn new() -> Self {
        Self
    }
}

/// ESDIRK4(3) coefficients (Kvaerno3).
///
/// Reference: Kvaerno, "Singly diagonally implicit Runge-Kutta methods
/// with an explicit first stage", BIT Numerical Mathematics 44, 489-502 (2004).
///
/// gamma satisfies 6*gamma^3 - 18*gamma^2 + 9*gamma - 1 = 0 (L-stability).
mod esdirk43_tableau {
    pub const GAMMA: f64 = 0.4358665215084590;

    pub const C: [f64; 4] = [0.0, 2.0 * GAMMA, 1.0, 1.0];

    pub const A: [[f64; 4]; 4] = [
        [0.0, 0.0, 0.0, 0.0],
        [GAMMA, GAMMA, 0.0, 0.0],
        [0.4905633884217806, 0.0735700900697604, GAMMA, 0.0],
        [
            0.3088099699767466,
            1.4905633884217800,
            -1.2352398799069855,
            GAMMA,
        ],
    ];

    pub const B: [f64; 4] = [
        0.3088099699767466,
        1.4905633884217800,
        -1.2352398799069855,
        GAMMA,
    ];

    // Error coefficients: E = B - B_hat where B_hat is row 2 of A
    // (an embedded 2nd-order method).
    pub const E: [f64; 4] = [
        0.3088099699767466 - 0.4905633884217806, // -0.1817534184450340
        1.4905633884217800 - 0.0735700900697604, //  1.4169932983520196
        -1.2352398799069855 - GAMMA,             // -1.6711064014154445
        GAMMA,                                   //  0.4358665215084590
    ];
}

// ============================================================================
// ESDIRK5(4) - 6 stages, 4th order with embedded 3rd order
// ============================================================================

/// ESDIRK5(4): 6-stage, 4th order ESDIRK method (L-stable, stiffly-accurate).
///
/// Implements ESDIRK4(3)6L\[2\]SA from Kennedy & Carpenter (2016),
/// NASA/TM-2016-219173 (Table 7). Both the main (4th order) and embedded
/// (3rd order) methods are L-stable.
#[derive(Clone, Debug, Default)]
pub struct Esdirk54;

impl Esdirk54 {
    pub fn new() -> Self {
        Self
    }
}

/// ESDIRK4(3)6L[2]SA coefficients from Kennedy & Carpenter (2016).
///
/// 6-stage, 4th order main method with 3rd order embedded.
/// gamma = 1/4, stiffly-accurate (b = last row of A), L-stable.
///
/// Reference: C.A. Kennedy, M.H. Carpenter, "Diagonally Implicit Runge-Kutta
/// Methods for Ordinary Differential Equations. A Review", NASA/TM-2016-219173.
mod esdirk54_tableau {
    // Diagonal coefficient (gamma = 1/4)
    pub const GAMMA: f64 = 0.25;

    pub const C: [f64; 6] = [
        0.0,
        0.5,                 // 2 * gamma
        0.14644660940672624, // (2 - sqrt(2)) / 4
        0.625,               // 5/8
        1.04,                // 26/25
        1.0,
    ];

    // Coefficients from exact formulas involving sqrt(2):
    // A[2][0] = A[2][1] = (1 - sqrt(2)) / 8
    // A[3][0] = A[3][1] = (5 - 7*sqrt(2)) / 64
    // A[3][2] = 7*(1 + sqrt(2)) / 32
    // A[4][0] = A[4][1] = (-13796 - 54539*sqrt(2)) / 125000
    // A[4][2] = (506605 + 132109*sqrt(2)) / 437500
    // A[4][3] = 166*(-97 + 376*sqrt(2)) / 109375
    // A[5][0] = A[5][1] = (1181 - 987*sqrt(2)) / 13782
    // A[5][2] = 47*(-267 + 1783*sqrt(2)) / 273343
    // A[5][3] = -16*(-22922 + 3525*sqrt(2)) / 571953
    // A[5][4] = -15625*(97 + 376*sqrt(2)) / 90749876
    pub const A: [[f64; 6]; 6] = [
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [GAMMA, GAMMA, 0.0, 0.0, 0.0, 0.0],
        [
            -0.05177669529663689,
            -0.05177669529663689,
            GAMMA,
            0.0,
            0.0,
            0.0,
        ],
        [
            -0.07655460838455727,
            -0.07655460838455727,
            0.5281092167691145,
            GAMMA,
            0.0,
            0.0,
        ],
        [
            -0.7274063478261299,
            -0.7274063478261299,
            1.5849950617406794,
            0.6598176339115805,
            GAMMA,
            0.0,
        ],
        [
            -0.01558763503571651,
            -0.01558763503571651,
            0.3876576709132033,
            0.5017726195721631,
            -0.10825502041393352,
            GAMMA,
        ],
    ];

    // Stiffly-accurate: b = last row of A
    pub const B: [f64; 6] = [
        -0.01558763503571651,
        -0.01558763503571651,
        0.3876576709132033,
        0.5017726195721631,
        -0.10825502041393352,
        GAMMA,
    ];

    // Error estimation: E = Bhat - B, where Bhat is the 3rd order embedded method
    // Bhat = [-480923228411/4982971448372, -480923228411/4982971448372,
    //          6709447293961/12833189095359, 3513175791894/6748737351361,
    //         -498863281070/6042575550617, 2077005547802/8945017530137]
    pub const E: [f64; 6] = [
        -0.08092570713246382,
        -0.08092570713246382,
        0.13516228008303094,
        0.01879524505002539,
        0.0256969660063123,
        -0.01780307687444085,
    ];
}

// ============================================================================
// Generic ESDIRK solver
// ============================================================================

fn solve_esdirk<S, Sys, const STAGES: usize>(
    problem: &Sys,
    t0: S,
    tf: S,
    y0: &[S],
    options: &SolverOptions<S>,
    c: &[f64],
    a: &[[f64; STAGES]; STAGES],
    b: &[f64],
    e: &[f64],
    gamma: f64,
    order: usize,
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

    let mut k: Vec<Vec<S>> = (0..STAGES).map(|_| vec![S::ZERO; dim]).collect();
    let mut y_stage = vec![S::ZERO; dim];
    let mut y_new = vec![S::ZERO; dim];
    let mut err = vec![S::ZERO; dim];
    let mut jac_data = vec![S::ZERO; dim * dim];
    let mut f0 = vec![S::ZERO; dim];

    let mut stats = SolverStats::default();

    // Initial evaluation
    problem.rhs(t, &y, &mut k[0]);
    stats.n_eval += 1;
    f0.copy_from_slice(&k[0]);

    let mut h = initial_step_size(&y, &k[0], options, dim);
    let h_min = options.h_min;
    let h_max = options.h_max.min((tf - t0).abs());

    // Jacobian and LU
    let mut lu: Option<LUFactorization<S>> = None;
    let mut need_jac = true;
    let mut jac_h = h;

    let direction = if tf > t0 { S::ONE } else { -S::ONE };
    let mut step_count = 0_usize;
    let mut consecutive_failures = 0_usize;

    while (tf - t) * direction > S::from_f64(1e-10) * (tf - t0).abs() {
        if step_count >= options.max_steps {
            return Err(SolverError::MaxIterationsExceeded { t: t.to_f64() });
        }

        if (t + h - tf) * direction > S::ZERO {
            h = tf - t;
        }

        h = h.abs().max(h_min) * direction;
        if h.abs() > h_max {
            h = h_max * direction;
        }

        // Recompute Jacobian only when the state has changed (need_jac).
        // The Jacobian depends on (t, y), NOT on h, so h changes alone
        // should only trigger an LU refactorization.
        if need_jac {
            compute_jacobian(problem, t, &y, &f0, &mut jac_data, dim);
            stats.n_jac += 1;
            need_jac = false;
        }

        // Form and factorize iteration matrix: I - h*gamma*J
        if lu.is_none() || (h - jac_h).abs() > S::from_f64(1e-10) * h.abs() {
            let iter_matrix = form_iteration_matrix(&jac_data, h * S::from_f64(gamma), dim);
            lu = Some(LUFactorization::new(&iter_matrix)?);
            stats.n_lu += 1;
            jac_h = h;
        }

        // Compute stages
        let step_ok = compute_esdirk_stages::<S, Sys, STAGES>(
            problem,
            t,
            h,
            &y,
            c,
            a,
            gamma,
            lu.as_ref().unwrap(),
            &mut k,
            &mut y_stage,
            &mut stats,
            dim,
        )?;

        if !step_ok {
            stats.n_reject += 1;
            consecutive_failures += 1;
            h = h * S::from_f64(0.5);
            need_jac = true;

            if consecutive_failures >= 5 {
                return Err(SolverError::Other(format!(
                    "Too many consecutive failures at t = {}",
                    t.to_f64()
                )));
            }
            continue;
        }

        // Compute solution and error
        for i in 0..dim {
            let mut sum_b = S::ZERO;
            let mut sum_e = S::ZERO;
            for s in 0..STAGES {
                sum_b = sum_b + S::from_f64(b[s]) * k[s][i];
                sum_e = sum_e + S::from_f64(e[s]) * k[s][i];
            }
            y_new[i] = y[i] + h * sum_b;
            err[i] = h * sum_e;
        }

        let err_norm = error_norm(&err, &y, &y_new, options, dim);

        let safety = S::from_f64(0.9);
        let fac_max = S::from_f64(3.0);
        let fac_min = S::from_f64(0.2);
        let order_f = S::from_usize(order + 1);

        if err_norm <= S::ONE {
            stats.n_accept += 1;
            consecutive_failures = 0;

            t = t + h;
            y.copy_from_slice(&y_new);

            problem.rhs(t, &y, &mut f0);
            stats.n_eval += 1;
            k[0].copy_from_slice(&f0);

            t_out.push(t);
            y_out.extend_from_slice(&y);

            let err_safe = err_norm.max(S::EPSILON * S::from_f64(100.0));
            let fac = safety * err_safe.powf(-S::ONE / order_f);
            let fac = fac.min(fac_max).max(fac_min);
            h = h * fac;
        } else {
            stats.n_reject += 1;
            consecutive_failures += 1;

            let err_safe = err_norm.max(S::EPSILON * S::from_f64(100.0));
            let fac = safety * err_safe.powf(-S::ONE / order_f);
            let fac = fac.max(fac_min);
            h = h * fac;

            if consecutive_failures >= 3 {
                need_jac = true;
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

fn compute_esdirk_stages<S, Sys, const STAGES: usize>(
    problem: &Sys,
    t: S,
    h: S,
    y: &[S],
    c: &[f64],
    a: &[[f64; STAGES]; STAGES],
    gamma: f64,
    lu: &LUFactorization<S>,
    k: &mut [Vec<S>],
    y_stage: &mut [S],
    stats: &mut SolverStats,
    dim: usize,
) -> Result<bool, SolverError>
where
    S: Scalar + SimpleEntity + Conjugate<Canonical = S> + ComplexField,
    Sys: OdeSystem<S>,
{
    // Stage 0 is explicit (already computed as f(t, y))

    for s in 1..STAGES {
        // Compute initial guess from explicit part
        for i in 0..dim {
            let mut sum = S::ZERO;
            for j in 0..s {
                sum = sum + S::from_f64(a[s][j]) * k[j][i];
            }
            y_stage[i] = y[i] + h * sum;
        }

        // Newton iteration for implicit stage
        let t_stage = t + S::from_f64(c[s]) * h;
        let h_gamma = h * S::from_f64(gamma);

        let mut converged = false;
        for _iter in 0..10 {
            let mut f_stage = vec![S::ZERO; dim];
            problem.rhs(t_stage, y_stage, &mut f_stage);
            stats.n_eval += 1;

            // Residual: y_stage - y - h * sum(a[s][j] * k[j]) - h*gamma*f_stage
            let mut residual = vec![S::ZERO; dim];
            let mut res_norm = S::ZERO;
            for i in 0..dim {
                let mut sum = S::ZERO;
                for j in 0..s {
                    sum = sum + S::from_f64(a[s][j]) * k[j][i];
                }
                residual[i] = y_stage[i] - y[i] - h * sum - h_gamma * f_stage[i];
                res_norm = res_norm + residual[i] * residual[i];
            }
            res_norm = res_norm.sqrt();

            if res_norm < S::from_f64(1e-10) {
                k[s].copy_from_slice(&f_stage);
                converged = true;
                break;
            }

            // Newton correction
            let delta = lu.solve(&residual)?;
            for i in 0..dim {
                y_stage[i] = y_stage[i] - delta[i];
            }
        }

        if !converged {
            return Ok(false);
        }
    }

    Ok(true)
}

fn compute_jacobian<S, Sys>(problem: &Sys, t: S, y: &[S], f0: &[S], jac: &mut [S], dim: usize)
where
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

fn form_iteration_matrix<S>(jac: &[S], h_gamma: S, dim: usize) -> DenseMatrix<S>
where
    S: Scalar + SimpleEntity + Conjugate<Canonical = S> + ComplexField,
{
    let mut m = DenseMatrix::zeros(dim, dim);
    for i in 0..dim {
        for j in 0..dim {
            let jij = jac[i * dim + j];
            if i == j {
                m.set(i, j, S::ONE - h_gamma * jij);
            } else {
                m.set(i, j, -h_gamma * jij);
            }
        }
    }
    m
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

    if y_norm < S::EPSILON.sqrt() || f_norm < S::EPSILON.sqrt() {
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

// ============================================================================
// Solver trait implementations
// ============================================================================

impl<S: Scalar + SimpleEntity + Conjugate<Canonical = S> + ComplexField> Solver<S> for Esdirk32 {
    fn solve<Sys: OdeSystem<S>>(
        problem: &Sys,
        t0: S,
        tf: S,
        y0: &[S],
        options: &SolverOptions<S>,
    ) -> Result<SolverResult<S>, SolverError> {
        solve_esdirk::<S, Sys, 3>(
            problem,
            t0,
            tf,
            y0,
            options,
            &esdirk32_tableau::C,
            &esdirk32_tableau::A,
            &esdirk32_tableau::B,
            &esdirk32_tableau::E,
            esdirk32_tableau::GAMMA,
            2,
        )
    }
}

impl<S: Scalar + SimpleEntity + Conjugate<Canonical = S> + ComplexField> Solver<S> for Esdirk43 {
    fn solve<Sys: OdeSystem<S>>(
        problem: &Sys,
        t0: S,
        tf: S,
        y0: &[S],
        options: &SolverOptions<S>,
    ) -> Result<SolverResult<S>, SolverError> {
        solve_esdirk::<S, Sys, 4>(
            problem,
            t0,
            tf,
            y0,
            options,
            &esdirk43_tableau::C,
            &esdirk43_tableau::A,
            &esdirk43_tableau::B,
            &esdirk43_tableau::E,
            esdirk43_tableau::GAMMA,
            3,
        )
    }
}

impl<S: Scalar + SimpleEntity + Conjugate<Canonical = S> + ComplexField> Solver<S> for Esdirk54 {
    fn solve<Sys: OdeSystem<S>>(
        problem: &Sys,
        t0: S,
        tf: S,
        y0: &[S],
        options: &SolverOptions<S>,
    ) -> Result<SolverResult<S>, SolverError> {
        solve_esdirk::<S, Sys, 6>(
            problem,
            t0,
            tf,
            y0,
            options,
            &esdirk54_tableau::C,
            &esdirk54_tableau::A,
            &esdirk54_tableau::B,
            &esdirk54_tableau::E,
            esdirk54_tableau::GAMMA,
            4,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::problem::OdeProblem;

    #[test]
    fn test_esdirk32_exponential() {
        let problem = OdeProblem::new(
            |_t, y: &[f64], dydt: &mut [f64]| {
                dydt[0] = -y[0];
            },
            0.0,
            5.0,
            vec![1.0],
        );
        let options = SolverOptions::default().rtol(1e-4).atol(1e-6);
        let result = Esdirk32::solve(&problem, 0.0, 5.0, &[1.0], &options).unwrap();

        assert!(result.success);
        let y_final = result.y_final().unwrap();
        let expected = (-5.0_f64).exp();
        assert!((y_final[0] - expected).abs() < 1e-3);
    }

    #[test]
    fn test_esdirk43_stiff() {
        let problem = OdeProblem::new(
            |_t, y: &[f64], dydt: &mut [f64]| {
                dydt[0] = -50.0 * y[0];
            },
            0.0,
            0.5,
            vec![1.0],
        );
        let options = SolverOptions::default().rtol(1e-4).atol(1e-6);
        let result = Esdirk43::solve(&problem, 0.0, 0.5, &[1.0], &options).unwrap();

        assert!(result.success);
        let y_final = result.y_final().unwrap();
        let expected = (-25.0_f64).exp();
        assert!((y_final[0] - expected).abs() < 0.01);
    }

    #[test]
    fn test_esdirk54_linear_system() {
        let problem = OdeProblem::new(
            |_t, y: &[f64], dydt: &mut [f64]| {
                dydt[0] = -y[0] + y[1];
                dydt[1] = y[0] - y[1];
            },
            0.0,
            5.0,
            vec![1.0, 0.0],
        );
        let options = SolverOptions::default().rtol(1e-5).atol(1e-7);
        let result = Esdirk54::solve(&problem, 0.0, 5.0, &[1.0, 0.0], &options).unwrap();

        assert!(result.success);
        let y_final = result.y_final().unwrap();
        // Conservation: y1 + y2 = 1
        assert!((y_final[0] + y_final[1] - 1.0).abs() < 1e-4);
    }

    #[test]
    fn test_esdirk_van_der_pol() {
        let mu = 10.0;
        let problem = OdeProblem::new(
            move |_t, y: &[f64], dydt: &mut [f64]| {
                dydt[0] = y[1];
                dydt[1] = mu * (1.0 - y[0] * y[0]) * y[1] - y[0];
            },
            0.0,
            10.0,
            vec![2.0, 0.0],
        );
        let options = SolverOptions::default().rtol(1e-4).atol(1e-6);
        let result = Esdirk54::solve(&problem, 0.0, 10.0, &[2.0, 0.0], &options);

        assert!(result.is_ok());
    }

    #[test]
    fn test_esdirk_methods_agree() {
        let problem = OdeProblem::new(
            |_t, y: &[f64], dydt: &mut [f64]| {
                dydt[0] = -y[0];
            },
            0.0,
            2.0,
            vec![1.0],
        );
        let options = SolverOptions::default().rtol(1e-3).atol(1e-5);

        let r32 = Esdirk32::solve(&problem, 0.0, 2.0, &[1.0], &options).unwrap();
        let r43 = Esdirk43::solve(&problem, 0.0, 2.0, &[1.0], &options).unwrap();
        let r54 = Esdirk54::solve(&problem, 0.0, 2.0, &[1.0], &options).unwrap();

        let y32 = r32.y_final().unwrap()[0];
        let y43 = r43.y_final().unwrap()[0];
        let y54 = r54.y_final().unwrap()[0];
        let expected = (-2.0_f64).exp();

        // Use looser tolerance to account for different method accuracies
        assert!(
            (y32 - expected).abs() < 1e-2,
            "ESDIRK32: got {}, expected {}",
            y32,
            expected
        );
        assert!(
            (y43 - expected).abs() < 1e-2,
            "ESDIRK43: got {}, expected {}",
            y43,
            expected
        );
        assert!(
            (y54 - expected).abs() < 1e-2,
            "ESDIRK54: got {}, expected {}",
            y54,
            expected
        );
    }
}

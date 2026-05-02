//! Derivative-free optimization methods: Nelder-Mead and Powell.
//!
//! Author: Moussa Leblouba
//! Date: 9 February 2026
//! Modified: 2 May 2026

use numra_core::Scalar;

use crate::error::OptimError;
use crate::types::{IterationRecord, OptimResult, OptimStatus};

// ─── Nelder-Mead Simplex ─────────────────────────────────────────────────

/// Options for the Nelder-Mead simplex method.
#[derive(Clone, Debug)]
pub struct NelderMeadOptions<S: Scalar> {
    pub max_iter: usize,
    pub fatol: S,
    pub xatol: S,
    pub initial_simplex_scale: S,
    pub adaptive: bool,
    pub verbose: bool,
}

impl<S: Scalar> Default for NelderMeadOptions<S> {
    fn default() -> Self {
        Self {
            max_iter: 10_000,
            fatol: S::from_f64(1e-8),
            xatol: S::from_f64(1e-8),
            initial_simplex_scale: S::from_f64(0.05),
            adaptive: true,
            verbose: false,
        }
    }
}

#[allow(clippy::needless_range_loop)]
/// Nelder-Mead simplex method for derivative-free optimization.
///
/// Minimizes `f(x)` starting from `x0`. Uses the standard simplex operations:
/// reflection, expansion, contraction, and shrink.
///
/// # Arguments
///
/// * `f` - Objective function.
/// * `x0` - Initial guess.
/// * `opts` - Algorithm options.
pub fn nelder_mead<S, F>(
    f: F,
    x0: &[S],
    opts: &NelderMeadOptions<S>,
) -> Result<OptimResult<S>, OptimError>
where
    S: Scalar,
    F: Fn(&[S]) -> S,
{
    let start = std::time::Instant::now();
    let n = x0.len();
    if n == 0 {
        return Err(OptimError::DimensionMismatch {
            expected: 1,
            actual: 0,
        });
    }

    // Adaptive parameters (Gao & Han, 2012) for high-dimensional problems
    let (alpha, gamma, rho, sigma) = if opts.adaptive {
        let nf = n as f64;
        (
            S::ONE,
            S::ONE + S::TWO / S::from_f64(nf),
            S::from_f64(0.75 - 0.5 / nf),
            S::ONE - S::ONE / S::from_f64(nf),
        )
    } else {
        (S::ONE, S::TWO, S::HALF, S::HALF)
    };

    // Build initial simplex: x0 and n perturbations
    let mut simplex: Vec<Vec<S>> = Vec::with_capacity(n + 1);
    simplex.push(x0.to_vec());
    for i in 0..n {
        let mut v = x0.to_vec();
        let scale = if v[i].abs() > S::from_f64(1e-10) {
            v[i] * opts.initial_simplex_scale
        } else {
            opts.initial_simplex_scale
        };
        v[i] += scale;
        simplex.push(v);
    }

    let mut f_vals: Vec<S> = simplex.iter().map(|v| f(v)).collect();
    let mut n_feval = n + 1;

    let mut history: Vec<IterationRecord<S>> = Vec::new();
    let mut iterations = 0;
    let mut converged = false;

    for iter in 0..opts.max_iter {
        iterations = iter + 1;

        // Sort simplex by function value
        let mut indices: Vec<usize> = (0..=n).collect();
        indices.sort_by(|&a, &b| f_vals[a].to_f64().partial_cmp(&f_vals[b].to_f64()).unwrap());
        simplex = indices.iter().map(|&i| simplex[i].clone()).collect();
        f_vals = indices.iter().map(|&i| f_vals[i]).collect();

        let f_best = f_vals[0];
        let f_worst = f_vals[n];
        let f_second_worst = f_vals[n - 1];

        if opts.verbose && iter % 100 == 0 {
            eprintln!(
                "NM iter {}: f_best = {:.6e}, f_worst = {:.6e}",
                iter,
                f_best.to_f64(),
                f_worst.to_f64()
            );
        }

        history.push(IterationRecord {
            iteration: iter,
            objective: f_best,
            gradient_norm: S::ZERO,
            step_size: (f_worst - f_best).abs(),
            constraint_violation: S::ZERO,
        });

        // Check convergence: function values spread and simplex size
        let f_range = (f_worst - f_best).abs();
        let mut max_dx = S::ZERO;
        for i in 1..=n {
            for j in 0..n {
                let d = (simplex[i][j] - simplex[0][j]).abs();
                if d > max_dx {
                    max_dx = d;
                }
            }
        }

        if f_range < opts.fatol && max_dx < opts.xatol {
            converged = true;
            break;
        }

        // Centroid of the best n vertices (excluding worst)
        let mut centroid = vec![S::ZERO; n];
        for i in 0..n {
            for j in 0..n {
                centroid[j] += simplex[i][j];
            }
        }
        let n_s = S::from_usize(n);
        for c in centroid.iter_mut() {
            *c /= n_s;
        }

        // Reflection: x_r = centroid + alpha * (centroid - worst)
        let x_r: Vec<S> = (0..n)
            .map(|j| centroid[j] + alpha * (centroid[j] - simplex[n][j]))
            .collect();
        let f_r = f(&x_r);
        n_feval += 1;

        if f_r >= f_best && f_r < f_second_worst {
            // Accept reflection
            simplex[n] = x_r;
            f_vals[n] = f_r;
            continue;
        }

        if f_r < f_best {
            // Try expansion: x_e = centroid + gamma * (x_r - centroid)
            let x_e: Vec<S> = (0..n)
                .map(|j| centroid[j] + gamma * (x_r[j] - centroid[j]))
                .collect();
            let f_e = f(&x_e);
            n_feval += 1;

            if f_e < f_r {
                simplex[n] = x_e;
                f_vals[n] = f_e;
            } else {
                simplex[n] = x_r;
                f_vals[n] = f_r;
            }
            continue;
        }

        // f_r >= f_second_worst: contraction
        if f_r < f_worst {
            // Outside contraction: x_c = centroid + rho * (x_r - centroid)
            let x_c: Vec<S> = (0..n)
                .map(|j| centroid[j] + rho * (x_r[j] - centroid[j]))
                .collect();
            let f_c = f(&x_c);
            n_feval += 1;

            if f_c <= f_r {
                simplex[n] = x_c;
                f_vals[n] = f_c;
                continue;
            }
        } else {
            // Inside contraction: x_cc = centroid - rho * (centroid - worst)
            let x_cc: Vec<S> = (0..n)
                .map(|j| centroid[j] - rho * (centroid[j] - simplex[n][j]))
                .collect();
            let f_cc = f(&x_cc);
            n_feval += 1;

            if f_cc < f_worst {
                simplex[n] = x_cc;
                f_vals[n] = f_cc;
                continue;
            }
        }

        // Shrink: move all vertices toward the best
        for i in 1..=n {
            for j in 0..n {
                simplex[i][j] = simplex[0][j] + sigma * (simplex[i][j] - simplex[0][j]);
            }
            f_vals[i] = f(&simplex[i]);
            n_feval += 1;
        }
    }

    let (status, message) = if converged {
        (
            OptimStatus::FunctionConverged,
            format!("Nelder-Mead converged after {} iterations", iterations),
        )
    } else {
        (
            OptimStatus::MaxIterations,
            format!("Nelder-Mead: max iterations ({}) reached", opts.max_iter),
        )
    };

    Ok(OptimResult {
        x: simplex[0].clone(),
        f: f_vals[0],
        grad: Vec::new(),
        iterations,
        n_feval,
        n_geval: 0,
        converged,
        message,
        status,
        history,
        lambda_eq: Vec::new(),
        lambda_ineq: Vec::new(),
        active_bounds: Vec::new(),
        constraint_violation: S::ZERO,
        wall_time_secs: 0.0,
        pareto: None,
        sensitivity: None,
    }
    .with_wall_time(start))
}

// ─── Powell's Conjugate Direction Method ─────────────────────────────────

/// Options for Powell's conjugate direction method.
#[derive(Clone, Debug)]
pub struct PowellOptions<S: Scalar> {
    pub max_iter: usize,
    pub ftol: S,
    pub xtol: S,
    pub max_line_search_iter: usize,
    pub verbose: bool,
}

impl<S: Scalar> Default for PowellOptions<S> {
    fn default() -> Self {
        Self {
            max_iter: 10_000,
            ftol: S::from_f64(1e-8),
            xtol: S::from_f64(1e-8),
            max_line_search_iter: 100,
            verbose: false,
        }
    }
}

#[allow(clippy::needless_range_loop)]
/// Powell's conjugate direction method for derivative-free optimization.
///
/// Builds a set of conjugate directions by performing sequential line minimizations.
/// The direction set is updated each iteration to maintain conjugacy.
///
/// # Arguments
///
/// * `f` - Objective function.
/// * `x0` - Initial guess.
/// * `opts` - Algorithm options.
pub fn powell<S, F>(f: F, x0: &[S], opts: &PowellOptions<S>) -> Result<OptimResult<S>, OptimError>
where
    S: Scalar,
    F: Fn(&[S]) -> S,
{
    let start = std::time::Instant::now();
    let n = x0.len();
    if n == 0 {
        return Err(OptimError::DimensionMismatch {
            expected: 1,
            actual: 0,
        });
    }

    // Initialize direction set to identity (coordinate directions)
    let mut directions: Vec<Vec<S>> = (0..n)
        .map(|i| {
            let mut d = vec![S::ZERO; n];
            d[i] = S::ONE;
            d
        })
        .collect();

    let mut x = x0.to_vec();
    let mut f_x = f(&x);
    let mut n_feval = 1_usize;
    let mut history: Vec<IterationRecord<S>> = Vec::new();
    let mut converged = false;
    let mut iterations = 0;

    for iter in 0..opts.max_iter {
        iterations = iter + 1;
        let f_start = f_x;
        let x_start = x.clone();

        // Track which direction gave the biggest decrease
        let mut max_decrease = S::ZERO;
        let mut max_decrease_idx = 0;

        // Line search along each direction
        for i in 0..n {
            let f_before = f_x;
            let (x_new, f_new, evals) =
                line_minimize(&f, &x, &directions[i], f_x, opts.max_line_search_iter);
            n_feval += evals;
            let decrease = f_before - f_new;
            if decrease > max_decrease {
                max_decrease = decrease;
                max_decrease_idx = i;
            }
            x = x_new;
            f_x = f_new;
        }

        if opts.verbose && iter % 50 == 0 {
            eprintln!("Powell iter {}: f = {:.6e}", iter, f_x.to_f64());
        }

        history.push(IterationRecord {
            iteration: iter,
            objective: f_x,
            gradient_norm: S::ZERO,
            step_size: max_decrease,
            constraint_violation: S::ZERO,
        });

        // Check convergence
        let f_decrease = (f_start - f_x).abs();
        let mut x_change = S::ZERO;
        for j in 0..n {
            let d = (x[j] - x_start[j]).abs();
            if d > x_change {
                x_change = d;
            }
        }

        if f_decrease < opts.ftol && x_change < opts.xtol {
            converged = true;
            break;
        }

        // Update direction set: replace the direction with max decrease
        // with the overall displacement direction
        let new_dir: Vec<S> = (0..n).map(|j| x[j] - x_start[j]).collect();
        let dir_norm = new_dir.iter().map(|&d| d * d).sum::<S>().sqrt();
        if dir_norm > S::from_f64(1e-20) {
            let normalized: Vec<S> = new_dir.iter().map(|&d| d / dir_norm).collect();
            directions[max_decrease_idx] = normalized;
        }
    }

    let (status, message) = if converged {
        (
            OptimStatus::FunctionConverged,
            format!("Powell converged after {} iterations", iterations),
        )
    } else {
        (
            OptimStatus::MaxIterations,
            format!("Powell: max iterations ({}) reached", opts.max_iter),
        )
    };

    Ok(OptimResult {
        x,
        f: f_x,
        grad: Vec::new(),
        iterations,
        n_feval,
        n_geval: 0,
        converged,
        message,
        status,
        history,
        lambda_eq: Vec::new(),
        lambda_ineq: Vec::new(),
        active_bounds: Vec::new(),
        constraint_violation: S::ZERO,
        wall_time_secs: 0.0,
        pareto: None,
        sensitivity: None,
    }
    .with_wall_time(start))
}

/// Brent's method for 1-D line minimization along direction `d` from `x`.
///
/// Uses bracket-then-Brent approach for robust convergence.
/// Returns `(x_new, f_new, n_evaluations)`.
fn line_minimize<S, F>(f: &F, x: &[S], d: &[S], f_x: S, max_iter: usize) -> (Vec<S>, S, usize)
where
    S: Scalar,
    F: Fn(&[S]) -> S,
{
    let n = x.len();
    let mut evals = 0;

    // Helper: compute f at x + alpha * d
    let f_at_alpha = |alpha: S, evals: &mut usize| -> S {
        let xp: Vec<S> = (0..n).map(|j| x[j] + alpha * d[j]).collect();
        *evals += 1;
        f(&xp)
    };

    // Bracket the minimum using expansion (mnbrak-style)
    let mut ax = S::ZERO;
    let mut bx = S::ONE;
    let mut fa = f_x;
    let mut fb = f_at_alpha(bx, &mut evals);

    // If f(b) > f(a), swap to ensure we go downhill
    if fb > fa {
        core::mem::swap(&mut ax, &mut bx);
        core::mem::swap(&mut fa, &mut fb);
    }

    let gold = S::from_f64(1.618034); // golden ratio
    let mut cx = bx + gold * (bx - ax);
    let mut fc = f_at_alpha(cx, &mut evals);

    // Expand bracket until we have fa >= fb <= fc (i.e., fb is in the valley)
    let mut bracket_iters = 0;
    while fb > fc && bracket_iters < 50 {
        ax = bx;
        fa = fb;
        bx = cx;
        fb = fc;
        cx = bx + gold * (bx - ax);
        fc = f_at_alpha(cx, &mut evals);
        bracket_iters += 1;
    }

    // Ensure a < c
    if ax > cx {
        core::mem::swap(&mut ax, &mut cx);
        core::mem::swap(&mut fa, &mut fc);
    }

    // Brent's method on [ax, cx] with bx as initial best
    let tol = S::from_f64(1e-10);
    let cgold = S::from_f64(0.381966011250105);
    let mut a_br = ax;
    let mut b_br = cx;
    let mut w = bx; // second best
    let mut v = bx; // previous w
    let mut xmin = bx;
    let mut fw = fb;
    let mut fv = fb;
    let mut fx_br = fb;
    let mut e_br = S::ZERO; // distance moved two steps ago
    let mut delta = S::ZERO;

    for _ in 0..max_iter {
        let midpoint = S::HALF * (a_br + b_br);
        let tol1 = tol * xmin.abs() + S::from_f64(1e-20);
        let tol2 = S::TWO * tol1;

        if (xmin - midpoint).abs() <= tol2 - S::HALF * (b_br - a_br) {
            break;
        }

        // Try parabolic interpolation
        let mut use_golden = true;
        if e_br.abs() > tol1 {
            // Parabolic step
            let r = (xmin - w) * (fx_br - fv);
            let q = (xmin - v) * (fx_br - fw);
            let p = (xmin - v) * q - (xmin - w) * r;
            let q = S::TWO * (q - r);
            let (p, q) = if q > S::ZERO { (-p, q) } else { (p, -q) };
            let e_old = e_br;

            if p.abs() < (S::HALF * q * e_old).abs()
                && p > q * (a_br - xmin)
                && p < q * (b_br - xmin)
            {
                // Parabolic step
                delta = p / q;
                let u = xmin + delta;
                if (u - a_br) < tol2 || (b_br - u) < tol2 {
                    delta = if xmin < midpoint { tol1 } else { -tol1 };
                }
                use_golden = false;
                e_br = delta;
            }
        }

        if use_golden {
            // Golden section step
            e_br = if xmin >= midpoint {
                a_br - xmin
            } else {
                b_br - xmin
            };
            delta = cgold * e_br;
        }

        // Evaluate at new point
        let u = if delta.abs() >= tol1 {
            xmin + delta
        } else {
            xmin + if delta > S::ZERO { tol1 } else { -tol1 }
        };
        let fu = f_at_alpha(u, &mut evals);

        if fu <= fx_br {
            if u >= xmin {
                a_br = xmin;
            } else {
                b_br = xmin;
            }
            v = w;
            fv = fw;
            w = xmin;
            fw = fx_br;
            xmin = u;
            fx_br = fu;
        } else {
            if u < xmin {
                a_br = u;
            } else {
                b_br = u;
            }
            if fu <= fw || (w - xmin).abs() < S::from_f64(1e-20) {
                v = w;
                fv = fw;
                w = u;
                fw = fu;
            } else if fu <= fv
                || (v - xmin).abs() < S::from_f64(1e-20)
                || (v - w).abs() < S::from_f64(1e-20)
            {
                v = u;
                fv = fu;
            }
        }
    }

    // Compare with original point
    if f_x <= fx_br {
        (x.to_vec(), f_x, evals)
    } else {
        let x_best: Vec<S> = (0..n).map(|j| x[j] + xmin * d[j]).collect();
        (x_best, fx_br, evals)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ─── Nelder-Mead tests ───

    #[test]
    fn test_nelder_mead_quadratic() {
        // min x0^2 + x1^2, optimal at (0, 0)
        let result = nelder_mead(
            |x: &[f64]| x[0] * x[0] + x[1] * x[1],
            &[5.0, 3.0],
            &NelderMeadOptions::default(),
        )
        .unwrap();
        assert!(result.converged, "did not converge: {}", result.message);
        assert!(result.x[0].abs() < 1e-4, "x0={}", result.x[0]);
        assert!(result.x[1].abs() < 1e-4, "x1={}", result.x[1]);
        assert!(result.f < 1e-8, "f={}", result.f);
    }

    #[test]
    fn test_nelder_mead_rosenbrock() {
        // Rosenbrock: min (1-x)^2 + 100*(y - x^2)^2
        let result = nelder_mead(
            |x: &[f64]| (1.0 - x[0]).powi(2) + 100.0 * (x[1] - x[0] * x[0]).powi(2),
            &[-1.0, 1.0],
            &NelderMeadOptions {
                max_iter: 50_000,
                ..Default::default()
            },
        )
        .unwrap();
        assert!(result.f < 1e-4, "f={}, x={:?}", result.f, result.x);
        assert!(
            (result.x[0] - 1.0).abs() < 0.05,
            "x0={}, expected ~1.0",
            result.x[0]
        );
    }

    #[test]
    fn test_nelder_mead_1d() {
        let result = nelder_mead(
            |x: &[f64]| (x[0] - 3.0).powi(2),
            &[10.0],
            &NelderMeadOptions::default(),
        )
        .unwrap();
        assert!(result.converged);
        assert!((result.x[0] - 3.0).abs() < 1e-4, "x={}", result.x[0]);
    }

    #[test]
    fn test_nelder_mead_beale() {
        // Beale function, min at (3, 0.5)
        let result = nelder_mead(
            |x: &[f64]| {
                (1.5 - x[0] + x[0] * x[1]).powi(2)
                    + (2.25 - x[0] + x[0] * x[1] * x[1]).powi(2)
                    + (2.625 - x[0] + x[0] * x[1].powi(3)).powi(2)
            },
            &[1.0, 1.0],
            &NelderMeadOptions {
                max_iter: 50_000,
                ..Default::default()
            },
        )
        .unwrap();
        assert!(result.f < 1e-4, "f={}", result.f);
    }

    // ─── Powell tests ───

    #[test]
    fn test_powell_quadratic() {
        let result = powell(
            |x: &[f64]| x[0] * x[0] + x[1] * x[1],
            &[5.0, 3.0],
            &PowellOptions::default(),
        )
        .unwrap();
        assert!(result.converged, "did not converge: {}", result.message);
        assert!(result.x[0].abs() < 1e-4, "x0={}", result.x[0]);
        assert!(result.x[1].abs() < 1e-4, "x1={}", result.x[1]);
    }

    #[test]
    fn test_powell_quadratic_offset() {
        // min (x-2)^2 + (y-3)^2
        let result = powell(
            |x: &[f64]| (x[0] - 2.0).powi(2) + (x[1] - 3.0).powi(2),
            &[0.0, 0.0],
            &PowellOptions::default(),
        )
        .unwrap();
        assert!(result.converged);
        assert!((result.x[0] - 2.0).abs() < 1e-4, "x0={}", result.x[0]);
        assert!((result.x[1] - 3.0).abs() < 1e-4, "x1={}", result.x[1]);
    }

    #[test]
    fn test_powell_rosenbrock() {
        let result = powell(
            |x: &[f64]| (1.0 - x[0]).powi(2) + 100.0 * (x[1] - x[0] * x[0]).powi(2),
            &[-1.0, 1.0],
            &PowellOptions {
                max_iter: 50_000,
                ..Default::default()
            },
        )
        .unwrap();
        assert!(result.f < 0.01, "f={}", result.f);
    }

    #[test]
    fn test_powell_1d() {
        let result = powell(
            |x: &[f64]| (x[0] - 7.0).powi(2),
            &[0.0],
            &PowellOptions::default(),
        )
        .unwrap();
        assert!(result.converged);
        assert!((result.x[0] - 7.0).abs() < 1e-4, "x={}", result.x[0]);
    }
}

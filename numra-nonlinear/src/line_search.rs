//! Wolfe line search for globalized optimization.
//!
//! Implements Algorithm 3.6 from Nocedal & Wright, *Numerical Optimization*,
//! consisting of a bracket phase that finds an interval containing a step
//! satisfying the strong Wolfe conditions, followed by a zoom (bisection)
//! phase that narrows the interval.
//!
//! Author: Moussa Leblouba
//! Date: 5 March 2026
//! Modified: 2 May 2026

use numra_core::{NumraError, Scalar};

/// Errors that can occur during line search.
#[derive(Debug, Clone)]
pub enum LineSearchError {
    /// The search direction is not a descent direction.
    NotDescentDirection,
    /// Zoom bracket collapsed to zero width.
    BracketCollapsed,
    /// Maximum iterations reached without satisfying Wolfe conditions.
    MaxIterations,
}

impl std::fmt::Display for LineSearchError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotDescentDirection => write!(f, "search direction is not a descent direction"),
            Self::BracketCollapsed => write!(f, "zoom bracket collapsed"),
            Self::MaxIterations => write!(f, "max line search iterations reached"),
        }
    }
}

impl std::error::Error for LineSearchError {}

impl From<LineSearchError> for NumraError {
    fn from(e: LineSearchError) -> Self {
        NumraError::LineSearch(e.to_string())
    }
}

/// Options controlling the Wolfe line search.
#[derive(Debug, Clone)]
pub struct WolfeOptions<S: Scalar> {
    /// Sufficient-decrease (Armijo) parameter, typically 1e-4.
    pub c1: S,
    /// Curvature parameter, typically 0.9 for quasi-Newton, 0.1 for CG.
    pub c2: S,
    /// Maximum allowed step length.
    pub max_step: S,
    /// Maximum number of bracket-phase iterations.
    pub max_iter: usize,
}

impl<S: Scalar> Default for WolfeOptions<S> {
    fn default() -> Self {
        Self {
            c1: S::from_f64(1e-4),
            c2: S::from_f64(0.9),
            max_step: S::from_f64(1e20),
            max_iter: 40,
        }
    }
}

/// Result returned by [`wolfe_line_search`].
#[derive(Debug, Clone)]
pub struct LineSearchResult<S: Scalar> {
    /// Accepted step length.
    pub step: S,
    /// Function value at the accepted point.
    pub f_new: S,
    /// Total number of function evaluations performed.
    pub n_eval: usize,
}

/// Inner product of two slices.
fn dot<S: Scalar>(a: &[S], b: &[S]) -> S {
    a.iter()
        .zip(b.iter())
        .map(|(&ai, &bi)| ai * bi)
        .fold(S::ZERO, |acc, x| acc + x)
}

/// Bracket collapse tolerance for detecting degenerate intervals.
const BRACKET_COLLAPSE_TOL: f64 = 1e-16;

/// Zoom phase (Algorithm 3.6, lines 12-21 in Nocedal & Wright).
///
/// Bisects `[alpha_lo, alpha_hi]` until the strong Wolfe conditions are met.
#[allow(clippy::too_many_arguments)]
fn zoom<S, F, G>(
    f: &F,
    grad: &G,
    x: &[S],
    d: &[S],
    f0: S,
    dg0: S,
    mut alpha_lo: S,
    mut f_lo: S,
    mut alpha_hi: S,
    opts: &WolfeOptions<S>,
    n_eval: &mut usize,
) -> Result<LineSearchResult<S>, LineSearchError>
where
    S: Scalar,
    F: Fn(&[S]) -> S,
    G: Fn(&[S], &mut [S]),
{
    let n = x.len();
    let mut x_trial = vec![S::ZERO; n];
    let mut g_trial = vec![S::ZERO; n];

    for _ in 0..opts.max_iter {
        if (alpha_hi - alpha_lo).abs() < S::from_f64(BRACKET_COLLAPSE_TOL) {
            return Err(LineSearchError::BracketCollapsed);
        }

        let alpha_j = (alpha_lo + alpha_hi) / S::TWO;

        // Evaluate f at trial point
        for i in 0..n {
            x_trial[i] = x[i] + alpha_j * d[i];
        }
        let f_j = f(&x_trial);
        *n_eval += 1;

        if f_j > f0 + opts.c1 * alpha_j * dg0 || f_j >= f_lo {
            // Armijo fails or no improvement over lo — shrink from the right
            alpha_hi = alpha_j;
        } else {
            // Armijo holds — check curvature
            grad(&x_trial, &mut g_trial);
            let dg_j = dot(&g_trial, d);

            if dg_j.abs() <= -opts.c2 * dg0 {
                // Strong Wolfe satisfied
                return Ok(LineSearchResult {
                    step: alpha_j,
                    f_new: f_j,
                    n_eval: *n_eval,
                });
            }

            if dg_j * (alpha_hi - alpha_lo) >= S::ZERO {
                alpha_hi = alpha_lo;
            }

            alpha_lo = alpha_j;
            f_lo = f_j;
        }
    }

    // Exhausted iterations — return the best we found
    Err(LineSearchError::MaxIterations)
}

/// Wolfe line search (Nocedal & Wright Algorithm 3.6).
///
/// Given a current point `x`, a descent direction `d`, the current function
/// value `f0`, and the current gradient `g0`, find a step length `alpha`
/// such that the **strong Wolfe conditions** hold:
///
/// 1. Sufficient decrease (Armijo): `f(x + alpha*d) <= f0 + c1 * alpha * g0^T d`
/// 2. Curvature: `|grad f(x + alpha*d)^T d| <= c2 * |g0^T d|`
///
/// # Errors
///
/// Returns `Err` if `d` is not a descent direction (`g0^T d >= 0`) or if
/// the algorithm fails to find a satisfactory step within `max_iter`
/// iterations.
pub fn wolfe_line_search<S, F, G>(
    f: F,
    grad: G,
    x: &[S],
    d: &[S],
    f0: S,
    g0: &[S],
    opts: &WolfeOptions<S>,
) -> Result<LineSearchResult<S>, LineSearchError>
where
    S: Scalar,
    F: Fn(&[S]) -> S,
    G: Fn(&[S], &mut [S]),
{
    let dg0 = dot(g0, d);
    if dg0 >= S::ZERO {
        return Err(LineSearchError::NotDescentDirection);
    }

    let n = x.len();
    let mut x_trial = vec![S::ZERO; n];
    let mut g_trial = vec![S::ZERO; n];

    let mut alpha_prev = S::ZERO;
    let mut f_prev = f0;
    let mut alpha = S::ONE;
    let mut n_eval: usize = 0;

    for i in 1..=opts.max_iter {
        // Clamp to max_step
        if alpha > opts.max_step {
            alpha = opts.max_step;
        }

        // Evaluate f at trial point x + alpha * d
        for j in 0..n {
            x_trial[j] = x[j] + alpha * d[j];
        }
        let f_alpha = f(&x_trial);
        n_eval += 1;

        // Check Armijo condition or if function increased relative to previous step
        if f_alpha > f0 + opts.c1 * alpha * dg0 || (i > 1 && f_alpha >= f_prev) {
            return zoom(
                &f,
                &grad,
                x,
                d,
                f0,
                dg0,
                alpha_prev,
                f_prev,
                alpha,
                opts,
                &mut n_eval,
            );
        }

        // Armijo holds — check curvature (strong Wolfe)
        grad(&x_trial, &mut g_trial);
        let dg_alpha = dot(&g_trial, d);

        if dg_alpha.abs() <= -opts.c2 * dg0 {
            return Ok(LineSearchResult {
                step: alpha,
                f_new: f_alpha,
                n_eval,
            });
        }

        // If slope is positive, we overshot the minimum — zoom backwards
        if dg_alpha >= S::ZERO {
            return zoom(
                &f,
                &grad,
                x,
                d,
                f0,
                dg0,
                alpha,
                f_alpha,
                alpha_prev,
                opts,
                &mut n_eval,
            );
        }

        // Otherwise expand the step
        alpha_prev = alpha;
        f_prev = f_alpha;
        alpha *= S::TWO;
    }

    Err(LineSearchError::MaxIterations)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wolfe_quadratic() {
        // f(x) = x^2, grad = 2x
        let f = |x: &[f64]| x[0] * x[0];
        let grad = |x: &[f64], g: &mut [f64]| {
            g[0] = 2.0 * x[0];
        };

        let x = [2.0];
        let d = [-1.0];
        let f0 = f(&x);
        let g0 = [4.0]; // grad at x=2

        let opts = WolfeOptions::default();
        let res = wolfe_line_search(f, grad, &x, &d, f0, &g0, &opts).unwrap();

        assert!(res.step > 0.0, "step must be positive");
        assert!(
            res.f_new < f0,
            "function must decrease: f_new={} vs f0={}",
            res.f_new,
            f0
        );
    }

    #[test]
    fn test_wolfe_rosenbrock() {
        // Rosenbrock: f(x) = (1-x0)^2 + 100*(x1 - x0^2)^2
        let f = |x: &[f64]| {
            let a = 1.0 - x[0];
            let b = x[1] - x[0] * x[0];
            a * a + 100.0 * b * b
        };
        let grad = |x: &[f64], g: &mut [f64]| {
            g[0] = -2.0 * (1.0 - x[0]) - 400.0 * x[0] * (x[1] - x[0] * x[0]);
            g[1] = 200.0 * (x[1] - x[0] * x[0]);
        };

        let x = [-1.0, 1.0];
        let f0 = f(&x);
        let mut g0 = [0.0; 2];
        grad(&x, &mut g0);

        // Steepest descent direction
        let d = [-g0[0], -g0[1]];

        let opts = WolfeOptions::default();
        let res = wolfe_line_search(f, grad, &x, &d, f0, &g0, &opts).unwrap();

        assert!(res.step > 0.0, "step must be positive");
        assert!(
            res.f_new < f0,
            "function must decrease: f_new={} vs f0={}",
            res.f_new,
            f0
        );
    }

    #[test]
    fn test_wolfe_not_descent() {
        let f = |x: &[f64]| x[0] * x[0];
        let grad = |x: &[f64], g: &mut [f64]| {
            g[0] = 2.0 * x[0];
        };

        let x = [2.0];
        let d = [1.0]; // ascending direction
        let f0 = f(&x);
        let g0 = [4.0];

        let opts = WolfeOptions::default();
        let res = wolfe_line_search(f, grad, &x, &d, f0, &g0, &opts);

        assert!(res.is_err(), "must reject non-descent direction");
        assert!(
            matches!(res.unwrap_err(), LineSearchError::NotDescentDirection),
            "error should be NotDescentDirection"
        );
    }

    #[test]
    fn test_wolfe_f32() {
        // Test with f32 to verify generic Scalar works
        let f = |x: &[f32]| x[0] * x[0];
        let grad = |x: &[f32], g: &mut [f32]| {
            g[0] = 2.0 * x[0];
        };

        let x = [2.0f32];
        let d = [-1.0f32];
        let f0 = f(&x);
        let g0 = [4.0f32];

        let opts = WolfeOptions::default();
        let res = wolfe_line_search(f, grad, &x, &d, f0, &g0, &opts).unwrap();

        assert!(res.step > 0.0, "step must be positive");
        assert!(res.f_new < f0, "function must decrease");
    }
}

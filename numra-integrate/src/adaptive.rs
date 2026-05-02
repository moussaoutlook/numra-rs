//! Adaptive Gauss-Kronrod quadrature (G7K15).
//!
//! This is the workhorse integration routine, equivalent to SciPy's `quad`.
//! Uses a 15-point Kronrod rule with embedded 7-point Gauss rule for error
//! estimation, with adaptive bisection of subintervals ordered by error.
//!
//! Author: Moussa Leblouba
//! Date: 9 February 2026
//! Modified: 2 May 2026

use alloc::collections::BinaryHeap;
use core::cmp::Ordering;

use numra_core::Scalar;

use crate::error::IntegrationError;

extern crate alloc;

/// Options for adaptive quadrature.
#[derive(Clone, Debug)]
pub struct QuadOptions<S: Scalar> {
    /// Absolute tolerance.
    pub atol: S,
    /// Relative tolerance.
    pub rtol: S,
    /// Maximum number of subinterval subdivisions.
    pub max_subdivisions: usize,
    /// Known singularities or discontinuities where the interval should be pre-split.
    pub points: Vec<S>,
}

impl<S: Scalar> Default for QuadOptions<S> {
    fn default() -> Self {
        Self {
            atol: S::from_f64(1.49e-8),
            rtol: S::from_f64(1.49e-8),
            max_subdivisions: 50,
            points: Vec::new(),
        }
    }
}

impl<S: Scalar> QuadOptions<S> {
    /// Set absolute tolerance.
    pub fn atol(mut self, atol: S) -> Self {
        self.atol = atol;
        self
    }

    /// Set relative tolerance.
    pub fn rtol(mut self, rtol: S) -> Self {
        self.rtol = rtol;
        self
    }

    /// Set maximum subdivisions.
    pub fn max_subdivisions(mut self, max: usize) -> Self {
        self.max_subdivisions = max;
        self
    }

    /// Set known singularity/discontinuity points.
    pub fn points(mut self, pts: Vec<S>) -> Self {
        self.points = pts;
        self
    }
}

/// Result of numerical integration.
#[derive(Clone, Debug)]
pub struct QuadResult<S: Scalar> {
    /// Estimated value of the integral.
    pub value: S,
    /// Estimated absolute error.
    pub error_estimate: S,
    /// Number of function evaluations.
    pub n_evaluations: usize,
    /// Number of subinterval subdivisions performed.
    pub n_subdivisions: usize,
}

// ============================================================================
// Gauss-Kronrod G7K15 nodes and weights on [-1, 1]
//
// 15 Kronrod nodes (symmetric), with the 7 Gauss nodes being a subset.
// We store only the positive half (8 nodes for K15, 4 for G7 subset).
// Source: QUADPACK (Piessens et al.), 15-digit precision.
// ============================================================================

/// Kronrod abscissae (positive half, 8 values including 0)
const K15_NODES: [f64; 8] = [
    0.0,
    0.2077849550078985,
    0.4058451513773972,
    0.5860872354676911,
    0.7415311855993945,
    0.8648644233597691,
    0.9491079123427585,
    0.9914553711208126,
];

/// Kronrod weights (for the 8 positive nodes, including node 0)
const K15_WEIGHTS: [f64; 8] = [
    0.2094821410847278,
    0.2044329400752989,
    0.1903505780647854,
    0.1690047266392679,
    0.1406532597155259,
    0.1047900103222502,
    0.0630920926299786,
    0.0229353220105292,
];

/// Gauss weights for the G7 subset nodes (indices 0, 2, 4, 6 in K15_NODES)
/// corresponding to nodes: 0, 0.4058..., 0.7415..., 0.9491...
const G7_WEIGHTS: [f64; 4] = [
    0.4179591836734694,
    0.3818300505051189,
    0.2797053914892767,
    0.1294849661688697,
];

/// Apply G7K15 rule to a single interval [a, b].
/// Returns (kronrod_result, gauss_result, n_evals).
fn g7k15<S, F>(f: &mut F, a: S, b: S) -> (S, S, usize)
where
    S: Scalar,
    F: FnMut(S) -> S,
{
    let mid = (a + b) * S::HALF;
    let half_len = (b - a) * S::HALF;

    let mut k15 = S::ZERO;
    let mut g7 = S::ZERO;

    // Node 0 (center)
    let f_center = f(mid);
    k15 += S::from_f64(K15_WEIGHTS[0]) * f_center;
    g7 += S::from_f64(G7_WEIGHTS[0]) * f_center;

    // Nodes 1, 3, 5, 7 are Kronrod-only (odd indices in the full 15-point rule)
    for &i in &[1usize, 3, 5, 7] {
        let x = half_len * S::from_f64(K15_NODES[i]);
        let f_pos = f(mid + x);
        let f_neg = f(mid - x);
        k15 += S::from_f64(K15_WEIGHTS[i]) * (f_pos + f_neg);
    }

    // Nodes 2, 4, 6 are shared with G7 (even indices > 0)
    for (g_idx, &k_idx) in [2usize, 4, 6].iter().enumerate() {
        let x = half_len * S::from_f64(K15_NODES[k_idx]);
        let f_pos = f(mid + x);
        let f_neg = f(mid - x);
        let fsum = f_pos + f_neg;
        k15 += S::from_f64(K15_WEIGHTS[k_idx]) * fsum;
        g7 += S::from_f64(G7_WEIGHTS[g_idx + 1]) * fsum;
    }

    (k15 * half_len, g7 * half_len, 15)
}

/// A subinterval with its integral estimate and error, for the priority queue.
struct SubInterval<S: Scalar> {
    a: S,
    b: S,
    result: S,
    error: S,
}

impl<S: Scalar> PartialEq for SubInterval<S> {
    fn eq(&self, other: &Self) -> bool {
        self.error.to_f64() == other.error.to_f64()
    }
}

impl<S: Scalar> Eq for SubInterval<S> {}

impl<S: Scalar> PartialOrd for SubInterval<S> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<S: Scalar> Ord for SubInterval<S> {
    fn cmp(&self, other: &Self) -> Ordering {
        // Max-heap by error
        self.error
            .to_f64()
            .partial_cmp(&other.error.to_f64())
            .unwrap_or(Ordering::Equal)
    }
}

/// Adaptive Gauss-Kronrod quadrature (G7K15).
///
/// Integrates `f` over `[a, b]` using adaptive subdivision with a 15-point
/// Kronrod rule and 7-point embedded Gauss rule for error estimation.
///
/// # Example
///
/// ```
/// use numra_integrate::{quad, QuadOptions};
///
/// let result = quad(|x: f64| x.sin(), 0.0, std::f64::consts::PI, &QuadOptions::default()).unwrap();
/// assert!((result.value - 2.0).abs() < 1e-10);
/// ```
pub fn quad<S, F>(
    mut f: F,
    a: S,
    b: S,
    opts: &QuadOptions<S>,
) -> Result<QuadResult<S>, IntegrationError>
where
    S: Scalar,
    F: FnMut(S) -> S,
{
    // Build initial list of subintervals, splitting at known singularity points
    let mut breakpoints = Vec::new();
    breakpoints.push(a);
    for &p in &opts.points {
        if p > a && p < b {
            breakpoints.push(p);
        }
    }
    breakpoints.push(b);
    // Sort breakpoints
    breakpoints.sort_by(|x, y| {
        x.to_f64()
            .partial_cmp(&y.to_f64())
            .unwrap_or(Ordering::Equal)
    });
    // Remove duplicates
    breakpoints.dedup_by(|a, b| ((*a) - (*b)).abs() < S::EPSILON);

    let mut heap: BinaryHeap<SubInterval<S>> = BinaryHeap::new();
    let mut total_result = S::ZERO;
    let mut total_error = S::ZERO;
    let mut total_evals = 0usize;
    let mut n_subdivisions = 0usize;

    // Initial pass: apply G7K15 to each breakpoint segment
    for i in 0..breakpoints.len() - 1 {
        let seg_a = breakpoints[i];
        let seg_b = breakpoints[i + 1];
        let (k15, g7, ne) = g7k15(&mut f, seg_a, seg_b);
        let err = (k15 - g7).abs();
        total_result += k15;
        total_error += err;
        total_evals += ne;
        n_subdivisions += 1;

        // Check for invalid values
        if !k15.is_finite() {
            let mid = (seg_a + seg_b) * S::HALF;
            return Err(IntegrationError::InvalidValue { x: mid.to_f64() });
        }

        heap.push(SubInterval {
            a: seg_a,
            b: seg_b,
            result: k15,
            error: err,
        });
    }

    // Check if already converged
    let tol = opts.atol.max(opts.rtol * total_result.abs());
    if total_error <= tol {
        return Ok(QuadResult {
            value: total_result,
            error_estimate: total_error,
            n_evaluations: total_evals,
            n_subdivisions,
        });
    }

    // Adaptive refinement
    while n_subdivisions < opts.max_subdivisions {
        let worst = match heap.pop() {
            Some(w) => w,
            None => break,
        };

        // Bisect the worst interval
        let mid = (worst.a + worst.b) * S::HALF;

        let (k15_l, g7_l, ne_l) = g7k15(&mut f, worst.a, mid);
        let err_l = (k15_l - g7_l).abs();

        let (k15_r, g7_r, ne_r) = g7k15(&mut f, mid, worst.b);
        let err_r = (k15_r - g7_r).abs();

        total_evals += ne_l + ne_r;
        n_subdivisions += 1;

        // Update totals: remove old, add new
        total_result = total_result - worst.result + k15_l + k15_r;
        total_error = total_error - worst.error + err_l + err_r;

        if !k15_l.is_finite() || !k15_r.is_finite() {
            return Err(IntegrationError::InvalidValue { x: mid.to_f64() });
        }

        heap.push(SubInterval {
            a: worst.a,
            b: mid,
            result: k15_l,
            error: err_l,
        });
        heap.push(SubInterval {
            a: mid,
            b: worst.b,
            result: k15_r,
            error: err_r,
        });

        let tol = opts.atol.max(opts.rtol * total_result.abs());
        if total_error <= tol {
            return Ok(QuadResult {
                value: total_result,
                error_estimate: total_error,
                n_evaluations: total_evals,
                n_subdivisions,
            });
        }
    }

    // Didn't converge but return the best result with an error
    Err(IntegrationError::MaxSubdivisions {
        subdivisions: n_subdivisions,
        error_estimate: total_error.to_f64(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_quad_sin() {
        // integral of sin(x) from 0 to pi = 2
        let result = quad(
            |x: f64| x.sin(),
            0.0,
            core::f64::consts::PI,
            &QuadOptions::default(),
        )
        .unwrap();
        assert_relative_eq!(result.value, 2.0, epsilon = 1e-10);
        assert!(result.error_estimate < 1e-10);
    }

    #[test]
    fn test_quad_exp() {
        // integral of exp(x) from 0 to 1 = e - 1
        let result = quad(|x: f64| x.exp(), 0.0, 1.0, &QuadOptions::default()).unwrap();
        let expected = core::f64::consts::E - 1.0;
        assert_relative_eq!(result.value, expected, epsilon = 1e-12);
    }

    #[test]
    fn test_quad_polynomial() {
        // integral of x^4 from 0 to 1 = 1/5
        let result = quad(|x: f64| x.powi(4), 0.0, 1.0, &QuadOptions::default()).unwrap();
        assert_relative_eq!(result.value, 0.2, epsilon = 1e-14);
    }

    #[test]
    fn test_quad_singular_sqrt() {
        // integral of 1/sqrt(x) from 0 to 1 = 2 (singular at x=0)
        let opts = QuadOptions::default()
            .atol(1e-8)
            .rtol(1e-8)
            .max_subdivisions(100)
            .points(vec![0.0]);
        let result = quad(
            |x: f64| {
                if x.abs() < 1e-300 {
                    0.0
                } else {
                    1.0 / x.sqrt()
                }
            },
            0.0,
            1.0,
            &opts,
        )
        .unwrap();
        assert_relative_eq!(result.value, 2.0, epsilon = 1e-6);
    }

    #[test]
    fn test_quad_oscillatory() {
        // integral of sin(100x) from 0 to pi = (1 - cos(100*pi)) / 100 = 0
        let opts = QuadOptions::default().max_subdivisions(200);
        let result = quad(
            |x: f64| (100.0 * x).sin(),
            0.0,
            core::f64::consts::PI,
            &opts,
        )
        .unwrap();
        assert!(result.value.abs() < 1e-6);
    }

    #[test]
    fn test_quad_tight_tolerance() {
        // integral of cos(x) from 0 to pi/2 = 1
        let opts = QuadOptions::default().atol(1e-14).rtol(1e-14);
        let result = quad(|x: f64| x.cos(), 0.0, core::f64::consts::FRAC_PI_2, &opts).unwrap();
        assert_relative_eq!(result.value, 1.0, epsilon = 1e-13);
    }

    #[test]
    fn test_quad_f32() {
        // integral of sin(x) from 0 to pi = 2 in f32
        let opts = QuadOptions::<f32>::default().atol(1e-4).rtol(1e-4);
        let result = quad(|x: f32| x.sin(), 0.0f32, core::f32::consts::PI, &opts).unwrap();
        assert!((result.value - 2.0).abs() < 1e-4);
    }

    #[test]
    fn test_quad_gaussian() {
        // integral of exp(-x^2) from -5 to 5 ≈ sqrt(pi)
        let result = quad(|x: f64| (-x * x).exp(), -5.0, 5.0, &QuadOptions::default()).unwrap();
        assert_relative_eq!(result.value, core::f64::consts::PI.sqrt(), epsilon = 1e-10);
    }
}

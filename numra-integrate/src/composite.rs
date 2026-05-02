//! Composite quadrature rules for sampled data and function-based integration.
//!
//! Provides trapezoid, Simpson, cumulative trapezoid, and Romberg integration.
//!
//! Author: Moussa Leblouba
//! Date: 9 February 2026
//! Modified: 2 May 2026

use numra_core::Scalar;

use crate::adaptive::QuadResult;
use crate::error::IntegrationError;

/// Composite trapezoidal rule for uniformly-spaced data.
///
/// Computes `(dx/2) * (y[0] + 2*y[1] + ... + 2*y[n-2] + y[n-1])`.
///
/// # Panics
///
/// Panics if `y` has fewer than 2 elements.
pub fn trapezoid<S: Scalar>(y: &[S], dx: S) -> S {
    assert!(y.len() >= 2, "trapezoid requires at least 2 data points");
    let n = y.len();
    let mut sum = y[0] + y[n - 1];
    for yi in &y[1..n - 1] {
        sum += S::TWO * *yi;
    }
    sum * dx * S::HALF
}

/// Composite trapezoidal rule for non-uniformly-spaced data.
///
/// Computes `sum((x[i+1] - x[i]) * (y[i] + y[i+1]) / 2)`.
///
/// # Panics
///
/// Panics if `x` and `y` have different lengths or fewer than 2 elements.
pub fn trapezoid_nonuniform<S: Scalar>(x: &[S], y: &[S]) -> S {
    assert_eq!(x.len(), y.len(), "x and y must have the same length");
    assert!(x.len() >= 2, "trapezoid requires at least 2 data points");
    let mut sum = S::ZERO;
    for i in 0..x.len() - 1 {
        sum += (x[i + 1] - x[i]) * (y[i] + y[i + 1]) * S::HALF;
    }
    sum
}

/// Composite Simpson's 1/3 rule for uniformly-spaced data.
///
/// Requires an odd number of points (even number of panels).
/// Falls back to trapezoid if the number of points is even.
///
/// # Panics
///
/// Panics if `y` has fewer than 3 elements.
pub fn simpson<S: Scalar>(y: &[S], dx: S) -> S {
    assert!(y.len() >= 3, "simpson requires at least 3 data points");
    let n = y.len();
    // Need odd number of points (even panels) for Simpson's rule
    if n % 2 == 0 {
        // Even number of points: apply Simpson to first n-1 points, trapezoid for last panel
        let simp = simpson_odd(&y[..n - 1], dx);
        let trap = (y[n - 2] + y[n - 1]) * dx * S::HALF;
        return simp + trap;
    }
    simpson_odd(y, dx)
}

/// Simpson's rule for odd number of points.
fn simpson_odd<S: Scalar>(y: &[S], dx: S) -> S {
    let n = y.len();
    let three = S::from_f64(3.0);
    let four = S::from_f64(4.0);
    let mut sum = y[0] + y[n - 1];
    for i in (1..n - 1).step_by(2) {
        sum += four * y[i];
    }
    for i in (2..n - 1).step_by(2) {
        sum += S::TWO * y[i];
    }
    sum * dx / three
}

/// Cumulative trapezoidal integration for uniformly-spaced data.
///
/// Returns a vector of length `y.len() - 1` where `result[i]` is the integral
/// from `y[0]` to `y[i+1]`.
///
/// # Panics
///
/// Panics if `y` has fewer than 2 elements.
pub fn cumulative_trapezoid<S: Scalar>(y: &[S], dx: S) -> Vec<S> {
    assert!(
        y.len() >= 2,
        "cumulative_trapezoid requires at least 2 data points"
    );
    let mut result = Vec::with_capacity(y.len() - 1);
    let mut running = S::ZERO;
    for i in 0..y.len() - 1 {
        running += (y[i] + y[i + 1]) * dx * S::HALF;
        result.push(running);
    }
    result
}

/// Romberg integration using Richardson extrapolation of the trapezoidal rule.
///
/// Integrates `f` over `[a, b]` to the given tolerance using successive
/// refinements of the composite trapezoidal rule with Richardson extrapolation.
///
/// # Arguments
///
/// * `f` - Integrand function
/// * `a` - Lower integration bound
/// * `b` - Upper integration bound
/// * `tol` - Desired absolute tolerance
pub fn romberg<S, F>(f: F, a: S, b: S, tol: S) -> Result<QuadResult<S>, IntegrationError>
where
    S: Scalar,
    F: Fn(S) -> S,
{
    let max_order = 20;
    // R[i] holds the i-th row of the Romberg table
    let mut r_prev: Vec<S> = Vec::with_capacity(max_order);
    let mut r_curr: Vec<S> = Vec::with_capacity(max_order);

    let h = b - a;
    let mut n_eval = 2usize;

    // R[0][0] = (h/2)(f(a) + f(b))
    r_prev.push((f(a) + f(b)) * h * S::HALF);

    let mut h_k = h;

    for k in 1..max_order {
        h_k *= S::HALF;
        let n_new = 1usize << (k - 1); // number of new midpoints

        // Composite trapezoidal with 2^k panels
        let mut midpoint_sum = S::ZERO;
        for i in 0..n_new {
            let x = a + h_k * S::from_usize(2 * i + 1);
            midpoint_sum += f(x);
        }
        n_eval += n_new;

        // R[k][0] = 0.5 * R[k-1][0] + h_k * sum(midpoints)
        r_curr.clear();
        r_curr.push(r_prev[0] * S::HALF + h_k * midpoint_sum);

        // Richardson extrapolation: R[k][j] = (4^j * R[k][j-1] - R[k-1][j-1]) / (4^j - 1)
        let mut four_j = S::from_f64(4.0);
        for j in 1..=k {
            let val = (four_j * r_curr[j - 1] - r_prev[j - 1]) / (four_j - S::ONE);
            r_curr.push(val);
            four_j *= S::from_f64(4.0);
        }

        let error = (r_curr[k] - r_prev[k - 1]).abs();
        if error < tol {
            return Ok(QuadResult {
                value: r_curr[k],
                error_estimate: error,
                n_evaluations: n_eval,
                n_subdivisions: 1 << k,
            });
        }

        core::mem::swap(&mut r_prev, &mut r_curr);
    }

    Err(IntegrationError::MaxSubdivisions {
        subdivisions: 1 << (max_order - 1),
        error_estimate: (r_prev[max_order - 1] - r_prev[max_order - 2])
            .abs()
            .to_f64(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_trapezoid_linear() {
        // Integral of y=x from 0 to 2 with points [0, 1, 2] -> dx=1
        let y = vec![0.0, 1.0, 2.0];
        let result = trapezoid(&y, 1.0);
        assert_relative_eq!(result, 2.0, epsilon = 1e-12);
    }

    #[test]
    fn test_trapezoid_constant() {
        let y = vec![3.0, 3.0, 3.0, 3.0, 3.0];
        let result = trapezoid(&y, 0.5);
        assert_relative_eq!(result, 6.0, epsilon = 1e-12);
    }

    #[test]
    fn test_trapezoid_nonuniform() {
        // Integral of y=x from 0 to 3 with non-uniform spacing
        let x = vec![0.0, 1.0, 3.0];
        let y = vec![0.0, 1.0, 3.0];
        let result = trapezoid_nonuniform(&x, &y);
        assert_relative_eq!(result, 4.5, epsilon = 1e-12);
    }

    #[test]
    fn test_simpson_quadratic() {
        // Simpson's rule is exact for polynomials up to degree 3
        // Integral of x^2 from 0 to 4 with 5 points (dx=1)
        let y: Vec<f64> = (0..5).map(|i| (i as f64).powi(2)).collect();
        let result = simpson(&y, 1.0);
        // Exact: 64/3 ≈ 21.333...
        assert_relative_eq!(result, 64.0 / 3.0, epsilon = 1e-12);
    }

    #[test]
    fn test_simpson_cubic() {
        // Integral of x^3 from 0 to 2 with 5 points (dx=0.5)
        let y: Vec<f64> = (0..5).map(|i| (i as f64 * 0.5).powi(3)).collect();
        let result = simpson(&y, 0.5);
        // Exact: 4.0
        assert_relative_eq!(result, 4.0, epsilon = 1e-12);
    }

    #[test]
    fn test_simpson_even_points() {
        // 4 points (even) — Simpson on first 3 + trapezoid on last panel
        // y = [0, 1, 4, 9] (x^2 at x=0,1,2,3)
        // Simpson [0,1,4]: (1/3)(0 + 4*1 + 4) = 8/3
        // Trapezoid [4,9]: (1/2)(4 + 9) = 6.5
        // Total = 8/3 + 6.5 = 55/6
        let y: Vec<f64> = (0..4).map(|i| (i as f64).powi(2)).collect();
        let result = simpson(&y, 1.0);
        assert_relative_eq!(result, 55.0 / 6.0, epsilon = 1e-12);
    }

    #[test]
    fn test_cumulative_trapezoid() {
        // Integral of y=1 from 0 to 3 with dx=1
        let y = vec![1.0, 1.0, 1.0, 1.0];
        let result = cumulative_trapezoid(&y, 1.0);
        assert_eq!(result.len(), 3);
        assert_relative_eq!(result[0], 1.0, epsilon = 1e-12);
        assert_relative_eq!(result[1], 2.0, epsilon = 1e-12);
        assert_relative_eq!(result[2], 3.0, epsilon = 1e-12);
    }

    #[test]
    fn test_cumulative_trapezoid_quadratic() {
        // Integral of y=x from 0 to 4, dx=1
        let y = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let result = cumulative_trapezoid(&y, 1.0);
        assert_eq!(result.len(), 4);
        assert_relative_eq!(result[0], 0.5, epsilon = 1e-12);
        assert_relative_eq!(result[1], 2.0, epsilon = 1e-12);
        assert_relative_eq!(result[2], 4.5, epsilon = 1e-12);
        assert_relative_eq!(result[3], 8.0, epsilon = 1e-12);
    }

    #[test]
    fn test_romberg_x_squared() {
        // Integral of x^2 from 0 to 1 = 1/3
        let result = romberg(|x: f64| x * x, 0.0, 1.0, 1e-10).unwrap();
        assert_relative_eq!(result.value, 1.0 / 3.0, epsilon = 1e-10);
        assert!(result.error_estimate < 1e-10);
    }

    #[test]
    fn test_romberg_exp() {
        // Integral of e^x from 0 to 1 = e - 1
        let result = romberg(|x: f64| x.exp(), 0.0, 1.0, 1e-12).unwrap();
        let expected = core::f64::consts::E - 1.0;
        assert_relative_eq!(result.value, expected, epsilon = 1e-12);
    }

    #[test]
    fn test_romberg_sin() {
        // Integral of sin(x) from 0 to pi = 2
        let result = romberg(|x: f64| x.sin(), 0.0, core::f64::consts::PI, 1e-12).unwrap();
        assert_relative_eq!(result.value, 2.0, epsilon = 1e-12);
    }
}

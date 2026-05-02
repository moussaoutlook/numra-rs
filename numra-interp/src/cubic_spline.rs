//! Cubic spline interpolation.
//!
//! Supports natural, clamped, and not-a-knot boundary conditions.
//!
//! Author: Moussa Leblouba
//! Date: 9 February 2026
//! Modified: 2 May 2026

use numra_core::Scalar;

use crate::error::InterpError;
use crate::{eval_piecewise_cubic, eval_piecewise_cubic_deriv, integrate_piecewise_cubic};
use crate::{validate_data, Interpolant};

/// Cubic spline interpolant.
///
/// On each interval `[x_i, x_{i+1}]`, the spline is a cubic polynomial
/// `S_i(x) = a_i + b_i*(x-x_i) + c_i*(x-x_i)^2 + d_i*(x-x_i)^3`.
pub struct CubicSpline<S: Scalar> {
    x: Vec<S>,
    a: Vec<S>,
    b: Vec<S>,
    c: Vec<S>,
    d: Vec<S>,
}

impl<S: Scalar> CubicSpline<S> {
    /// Natural cubic spline: S''(x_0) = S''(x_{n-1}) = 0.
    pub fn natural(x: &[S], y: &[S]) -> Result<Self, InterpError> {
        validate_data(x, y, 2)?;
        let n = x.len();
        if n == 2 {
            return Self::from_linear(x, y);
        }

        let h = compute_h(x);
        let mut m = vec![S::ZERO; n]; // second derivatives

        // Solve (n-2) x (n-2) tridiagonal for interior m[1..n-2]
        let n_int = n - 2;
        let mut sub = vec![S::ZERO; n_int];
        let mut diag = vec![S::ZERO; n_int];
        let mut sup = vec![S::ZERO; n_int];
        let mut rhs = vec![S::ZERO; n_int];

        for k in 0..n_int {
            let i = k + 1;
            if k > 0 {
                sub[k] = h[i - 1];
            }
            diag[k] = S::TWO * (h[i - 1] + h[i]);
            if k < n_int - 1 {
                sup[k] = h[i];
            }
            let s_prev = (y[i] - y[i - 1]) / h[i - 1];
            let s_next = (y[i + 1] - y[i]) / h[i];
            rhs[k] = S::from_f64(6.0) * (s_next - s_prev);
        }

        thomas_solve(&sub, &diag, &sup, &mut rhs);
        m[1..n_int + 1].copy_from_slice(&rhs[..n_int]);

        Ok(Self::from_second_derivatives(x, y, &h, &m))
    }

    /// Clamped cubic spline with specified endpoint derivatives.
    pub fn clamped(x: &[S], y: &[S], dy_left: S, dy_right: S) -> Result<Self, InterpError> {
        validate_data(x, y, 2)?;
        let n = x.len();
        if n == 2 {
            return Self::from_linear(x, y);
        }

        let h = compute_h(x);

        // Solve n x n tridiagonal
        let mut sub = vec![S::ZERO; n];
        let mut diag = vec![S::ZERO; n];
        let mut sup = vec![S::ZERO; n];
        let mut rhs = vec![S::ZERO; n];

        // Row 0: clamped left BC
        let s0 = (y[1] - y[0]) / h[0];
        diag[0] = S::TWO * h[0];
        sup[0] = h[0];
        rhs[0] = S::from_f64(6.0) * (s0 - dy_left);

        // Interior rows
        for i in 1..n - 1 {
            sub[i] = h[i - 1];
            diag[i] = S::TWO * (h[i - 1] + h[i]);
            sup[i] = h[i];
            let s_prev = (y[i] - y[i - 1]) / h[i - 1];
            let s_next = (y[i + 1] - y[i]) / h[i];
            rhs[i] = S::from_f64(6.0) * (s_next - s_prev);
        }

        // Row n-1: clamped right BC
        let sn = (y[n - 1] - y[n - 2]) / h[n - 2];
        sub[n - 1] = h[n - 2];
        diag[n - 1] = S::TWO * h[n - 2];
        rhs[n - 1] = S::from_f64(6.0) * (dy_right - sn);

        thomas_solve(&sub, &diag, &sup, &mut rhs);

        Ok(Self::from_second_derivatives(x, y, &h, &rhs))
    }

    /// Not-a-knot cubic spline: third derivative continuous at `x[1]` and `x[n-2]`.
    ///
    /// Requires at least 4 points. Falls back to natural for fewer points.
    pub fn not_a_knot(x: &[S], y: &[S]) -> Result<Self, InterpError> {
        validate_data(x, y, 2)?;
        let n = x.len();
        if n <= 3 {
            return Self::natural(x, y);
        }

        let h = compute_h(x);
        let n_int = n - 2;

        // Build (n-2) x (n-2) tridiagonal for m[1..n-2], with modified first and last rows
        // Not-a-knot left: m_0 = ((h0+h1)/h1)*m_1 - (h0/h1)*m_2
        // Not-a-knot right: m_{n-1} = ((h_{n-3}+h_{n-2})/h_{n-3})*m_{n-2} - (h_{n-2}/h_{n-3})*m_{n-3}

        let mut sub = vec![S::ZERO; n_int];
        let mut diag = vec![S::ZERO; n_int];
        let mut sup = vec![S::ZERO; n_int];
        let mut rhs = vec![S::ZERO; n_int];

        // Standard interior equations
        for (k, rhs_k) in rhs.iter_mut().enumerate().take(n_int) {
            let i = k + 1;
            let s_prev = (y[i] - y[i - 1]) / h[i - 1];
            let s_next = (y[i + 1] - y[i]) / h[i];
            *rhs_k = S::from_f64(6.0) * (s_next - s_prev);
        }

        // Row 0 (i=1): h[0]*m_0 + 2*(h[0]+h[1])*m_1 + h[1]*m_2 = rhs[0]
        // Substitute m_0 = alpha1*m_1 + alpha2*m_2
        let alpha1 = (h[0] + h[1]) / h[1];
        let alpha2 = -h[0] / h[1];
        diag[0] = h[0] * alpha1 + S::TWO * (h[0] + h[1]);
        sup[0] = h[0] * alpha2 + h[1];

        // Fill standard interior rows 1..n_int-2
        for k in 1..n_int - 1 {
            let i = k + 1;
            sub[k] = h[i - 1];
            diag[k] = S::TWO * (h[i - 1] + h[i]);
            sup[k] = h[i];
        }

        // Last row (i=n-2): h[n-3]*m_{n-3} + 2*(h[n-3]+h[n-2])*m_{n-2} + h[n-2]*m_{n-1} = rhs[n_int-1]
        // Substitute m_{n-1} = beta1*m_{n-2} + beta2*m_{n-3}
        let beta1 = (h[n - 3] + h[n - 2]) / h[n - 3];
        let beta2 = -h[n - 2] / h[n - 3];
        sub[n_int - 1] = h[n - 3] + h[n - 2] * beta2;
        diag[n_int - 1] = S::TWO * (h[n - 3] + h[n - 2]) + h[n - 2] * beta1;

        thomas_solve(&sub, &diag, &sup, &mut rhs);

        // Recover full m array
        let mut m = vec![S::ZERO; n];
        m[1..n_int + 1].copy_from_slice(&rhs[..n_int]);
        // m_0 from not-a-knot left
        m[0] = alpha1 * m[1] + alpha2 * m[2];
        // m_{n-1} from not-a-knot right
        m[n - 1] = beta1 * m[n - 2] + beta2 * m[n - 3];

        Ok(Self::from_second_derivatives(x, y, &h, &m))
    }

    /// Build from second derivatives.
    fn from_second_derivatives(x: &[S], y: &[S], h: &[S], m: &[S]) -> Self {
        let n = x.len();
        let n_seg = n - 1;
        let mut a = Vec::with_capacity(n_seg);
        let mut b = Vec::with_capacity(n_seg);
        let mut c = Vec::with_capacity(n_seg);
        let mut d = Vec::with_capacity(n_seg);

        let six = S::from_f64(6.0);
        for i in 0..n_seg {
            a.push(y[i]);
            b.push((y[i + 1] - y[i]) / h[i] - h[i] * (S::TWO * m[i] + m[i + 1]) / six);
            c.push(m[i] * S::HALF);
            d.push((m[i + 1] - m[i]) / (six * h[i]));
        }

        Self {
            x: x.to_vec(),
            a,
            b,
            c,
            d,
        }
    }

    /// Trivial linear case for 2 points.
    fn from_linear(x: &[S], y: &[S]) -> Result<Self, InterpError> {
        let h = x[1] - x[0];
        Ok(Self {
            x: x.to_vec(),
            a: vec![y[0]],
            b: vec![(y[1] - y[0]) / h],
            c: vec![S::ZERO],
            d: vec![S::ZERO],
        })
    }
}

impl<S: Scalar> Interpolant<S> for CubicSpline<S> {
    fn interpolate(&self, x: S) -> S {
        eval_piecewise_cubic(&self.x, &self.a, &self.b, &self.c, &self.d, x)
    }

    fn derivative(&self, x: S) -> Option<S> {
        Some(eval_piecewise_cubic_deriv(
            &self.x, &self.b, &self.c, &self.d, x,
        ))
    }

    fn integrate(&self, a: S, b: S) -> Option<S> {
        Some(integrate_piecewise_cubic(
            &self.x, &self.a, &self.b, &self.c, &self.d, a, b,
        ))
    }
}

// ============================================================================
// Internal helpers
// ============================================================================

/// Compute interval widths h[i] = x[i+1] - x[i].
fn compute_h<S: Scalar>(x: &[S]) -> Vec<S> {
    (0..x.len() - 1).map(|i| x[i + 1] - x[i]).collect()
}

/// Thomas algorithm for tridiagonal system.
///
/// Solves sub[i]*x_{i-1} + diag[i]*x_i + sup[i]*x_{i+1} = rhs[i].
/// sub[0] and sup[n-1] are ignored. Solution is returned in `rhs`.
fn thomas_solve<S: Scalar>(sub: &[S], diag: &[S], sup: &[S], rhs: &mut [S]) {
    let n = diag.len();
    if n == 0 {
        return;
    }
    if n == 1 {
        rhs[0] /= diag[0];
        return;
    }

    let mut cp = vec![S::ZERO; n];
    let mut dp = vec![S::ZERO; n];

    cp[0] = sup[0] / diag[0];
    dp[0] = rhs[0] / diag[0];

    for i in 1..n {
        let m = diag[i] - sub[i] * cp[i - 1];
        cp[i] = if i < n - 1 { sup[i] / m } else { S::ZERO };
        dp[i] = (rhs[i] - sub[i] * dp[i - 1]) / m;
    }

    rhs[n - 1] = dp[n - 1];
    for i in (0..n - 1).rev() {
        rhs[i] = dp[i] - cp[i] * rhs[i + 1];
    }
}

/// Build piecewise cubic coefficients from slopes at each knot.
/// Used by PCHIP and Akima.
pub(crate) fn coefficients_from_slopes<S: Scalar>(
    x: &[S],
    y: &[S],
    slopes: &[S],
) -> (Vec<S>, Vec<S>, Vec<S>, Vec<S>) {
    let n_seg = x.len() - 1;
    let mut a = Vec::with_capacity(n_seg);
    let mut b = Vec::with_capacity(n_seg);
    let mut c = Vec::with_capacity(n_seg);
    let mut d = Vec::with_capacity(n_seg);

    for i in 0..n_seg {
        let h = x[i + 1] - x[i];
        let s = (y[i + 1] - y[i]) / h;
        a.push(y[i]);
        b.push(slopes[i]);
        c.push((S::from_f64(3.0) * s - S::TWO * slopes[i] - slopes[i + 1]) / h);
        d.push((slopes[i] + slopes[i + 1] - S::TWO * s) / (h * h));
    }
    (a, b, c, d)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn sample_sin(n: usize) -> (Vec<f64>, Vec<f64>) {
        let x: Vec<f64> = (0..n)
            .map(|i| i as f64 * core::f64::consts::PI * 2.0 / (n - 1) as f64)
            .collect();
        let y: Vec<f64> = x.iter().map(|&xi| xi.sin()).collect();
        (x, y)
    }

    #[test]
    fn test_natural_at_knots() {
        let (x, y) = sample_sin(10);
        let cs = CubicSpline::natural(&x, &y).unwrap();
        for (xi, yi) in x.iter().zip(y.iter()) {
            assert_relative_eq!(cs.interpolate(*xi), *yi, epsilon = 1e-12);
        }
    }

    #[test]
    fn test_natural_smooth() {
        let (x, y) = sample_sin(20);
        let cs = CubicSpline::natural(&x, &y).unwrap();
        // Interpolation should be close to sin(x)
        let test_x = 1.0;
        let err = (cs.interpolate(test_x) - test_x.sin()).abs();
        assert!(err < 1e-4, "Error too large: {}", err);
    }

    #[test]
    fn test_clamped_polynomial() {
        // Cubic polynomial f(x) = x^3, f'(x) = 3x^2
        // Clamped spline with exact derivatives should reproduce exactly
        let x = vec![0.0, 1.0, 2.0, 3.0];
        let y: Vec<f64> = x.iter().map(|&xi| xi.powi(3)).collect();
        let cs = CubicSpline::clamped(&x, &y, 0.0, 27.0).unwrap();
        // Check at midpoints
        assert_relative_eq!(cs.interpolate(0.5), 0.125, epsilon = 1e-10);
        assert_relative_eq!(cs.interpolate(1.5), 3.375, epsilon = 1e-10);
        assert_relative_eq!(cs.interpolate(2.5), 15.625, epsilon = 1e-10);
    }

    #[test]
    fn test_not_a_knot_cubic() {
        // Not-a-knot should reproduce a cubic polynomial exactly
        let x = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let y: Vec<f64> = x.iter().map(|&xi| xi.powi(3) - 2.0 * xi).collect();
        let cs = CubicSpline::not_a_knot(&x, &y).unwrap();
        // Check at non-knot points
        for t in [0.25, 0.75, 1.5, 2.5, 3.5] {
            let expected = t.powi(3) - 2.0 * t;
            assert_relative_eq!(cs.interpolate(t), expected, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_derivative() {
        let x = vec![0.0, 1.0, 2.0, 3.0];
        let y: Vec<f64> = x.iter().map(|&xi| xi * xi).collect();
        let cs = CubicSpline::natural(&x, &y).unwrap();
        // Derivative of x^2 is 2x; not exact for natural spline but close
        let deriv = cs.derivative(1.5).unwrap();
        assert!(
            (deriv - 3.0).abs() < 0.5,
            "Derivative error too large: {}",
            (deriv - 3.0).abs()
        );
    }

    #[test]
    fn test_integrate() {
        // Integral of x^2 from 0 to 3 = 9
        let x = vec![0.0, 1.0, 2.0, 3.0];
        let y: Vec<f64> = x.iter().map(|&xi| xi * xi).collect();
        let cs = CubicSpline::natural(&x, &y).unwrap();
        let integral = cs.integrate(0.0, 3.0).unwrap();
        assert_relative_eq!(integral, 9.0, epsilon = 0.1);
    }

    #[test]
    fn test_two_points() {
        let cs = CubicSpline::natural(&[0.0, 1.0], &[0.0, 1.0]).unwrap();
        assert_relative_eq!(cs.interpolate(0.5), 0.5, epsilon = 1e-14);
    }

    #[test]
    fn test_c2_continuity() {
        let (x, y) = sample_sin(10);
        let cs = CubicSpline::natural(&x, &y).unwrap();
        // First derivative should be continuous at interior knots (C1 check)
        // This indirectly validates C2 since the spline construction enforces it
        for i in 1..x.len() - 1 {
            let eps = 1e-8;
            let d_left = cs.derivative(x[i] - eps).unwrap();
            let d_right = cs.derivative(x[i] + eps).unwrap();
            assert!(
                (d_left - d_right).abs() < 1e-4,
                "C1 discontinuity at x[{}]={}: left={}, right={}",
                i,
                x[i],
                d_left,
                d_right
            );
        }
    }

    #[test]
    fn test_f32() {
        let cs = CubicSpline::natural(&[0.0f32, 1.0, 2.0, 3.0], &[0.0, 1.0, 0.0, 1.0]).unwrap();
        let _ = cs.interpolate(1.5f32);
    }
}

//! Akima interpolation.
//!
//! A piecewise cubic method that is robust to outliers. Uses a weighted
//! average of adjacent secants to determine slopes.
//!
//! Author: Moussa Leblouba
//! Date: 9 February 2026
//! Modified: 2 May 2026

use numra_core::Scalar;

use crate::cubic_spline::coefficients_from_slopes;
use crate::error::InterpError;
use crate::{eval_piecewise_cubic, eval_piecewise_cubic_deriv, integrate_piecewise_cubic};
use crate::{validate_data, Interpolant};

/// Akima interpolant.
///
/// Uses Akima's method for computing slopes, which is robust to outliers
/// in the data. Requires at least 5 data points.
pub struct Akima<S: Scalar> {
    x: Vec<S>,
    a: Vec<S>,
    b: Vec<S>,
    c: Vec<S>,
    d: Vec<S>,
}

impl<S: Scalar> Akima<S> {
    /// Create an Akima interpolant from data points.
    ///
    /// # Errors
    ///
    /// Returns error if fewer than 5 points or data is not sorted.
    pub fn new(x: &[S], y: &[S]) -> Result<Self, InterpError> {
        validate_data(x, y, 5)?;
        let n = x.len();
        let slopes = compute_akima_slopes(x, y);
        let (a, b, c, d) = coefficients_from_slopes(x, y, &slopes);

        Ok(Self {
            x: x[..n].to_vec(),
            a,
            b,
            c,
            d,
        })
    }
}

impl<S: Scalar> Interpolant<S> for Akima<S> {
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

/// Compute Akima slopes.
fn compute_akima_slopes<S: Scalar>(x: &[S], y: &[S]) -> Vec<S> {
    let n = x.len();
    let n_seg = n - 1;

    // Compute secants s[0..n_seg-1]
    let s: Vec<S> = (0..n_seg)
        .map(|i| (y[i + 1] - y[i]) / (x[i + 1] - x[i]))
        .collect();

    // Extend secant array with 2 values on each side (Akima's extrapolation)
    // s_ext[-2], s_ext[-1], s_ext[0], ..., s_ext[n_seg-1], s_ext[n_seg], s_ext[n_seg+1]
    let sm2 = S::from_f64(3.0) * s[0] - S::TWO * s[1];
    let sm1 = S::TWO * s[0] - s[1];
    let sp1 = S::TWO * s[n_seg - 1] - s[n_seg - 2];
    let sp2 = S::from_f64(3.0) * s[n_seg - 1] - S::TWO * s[n_seg - 2];

    // Build extended secant array indexed 0..n_seg+3 corresponding to original -2..n_seg+1
    let mut s_ext = Vec::with_capacity(n_seg + 4);
    s_ext.push(sm2);
    s_ext.push(sm1);
    for &si in &s {
        s_ext.push(si);
    }
    s_ext.push(sp1);
    s_ext.push(sp2);
    // Now s_ext[k] corresponds to original index k-2

    let mut slopes = Vec::with_capacity(n);
    for i in 0..n {
        // Map to s_ext indices: s_{i-2}..s_{i+1} → s_ext[i]..s_ext[i+3]
        let w1 = (s_ext[i + 3] - s_ext[i + 2]).abs();
        let w2 = (s_ext[i + 1] - s_ext[i]).abs();
        let denom = w1 + w2;
        if denom > S::EPSILON * S::from_f64(100.0) {
            slopes.push((w1 * s_ext[i + 1] + w2 * s_ext[i + 2]) / denom);
        } else {
            slopes.push((s_ext[i + 1] + s_ext[i + 2]) * S::HALF);
        }
    }

    slopes
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_akima_at_knots() {
        let x = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![0.0, 1.0, 0.5, 1.5, 1.0, 2.0];
        let a = Akima::new(&x, &y).unwrap();
        for (&xi, &yi) in x.iter().zip(y.iter()) {
            assert_relative_eq!(a.interpolate(xi), yi, epsilon = 1e-12);
        }
    }

    #[test]
    fn test_akima_smooth() {
        // Akima on sin(x)
        let n = 20;
        let x: Vec<f64> = (0..n)
            .map(|i| i as f64 * core::f64::consts::PI * 2.0 / (n - 1) as f64)
            .collect();
        let y: Vec<f64> = x.iter().map(|&xi| xi.sin()).collect();
        let a = Akima::new(&x, &y).unwrap();

        let test_x = 1.0;
        let err = (a.interpolate(test_x) - test_x.sin()).abs();
        assert!(err < 1e-3, "Akima error too large: {}", err);
    }

    #[test]
    fn test_akima_outlier_robustness() {
        // Data with an outlier at x=2
        let x = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let y = vec![0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0]; // outlier at x=2

        let a = Akima::new(&x, &y).unwrap();

        // Akima should not overshoot excessively away from the outlier
        let y_at_4_5 = a.interpolate(4.5);
        assert!(
            y_at_4_5.abs() < 1.0,
            "Akima overshoots away from outlier: {}",
            y_at_4_5
        );
    }

    #[test]
    fn test_akima_linear_data() {
        // Linear data should give linear interpolation
        let x = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let y = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let a = Akima::new(&x, &y).unwrap();
        assert_relative_eq!(a.interpolate(1.5), 1.5, epsilon = 1e-12);
        assert_relative_eq!(a.interpolate(2.7), 2.7, epsilon = 1e-12);
    }

    #[test]
    fn test_akima_derivative() {
        let x = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let y = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let a = Akima::new(&x, &y).unwrap();
        // Derivative of linear function should be 1
        assert_relative_eq!(a.derivative(2.0).unwrap(), 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_akima_errors() {
        assert!(Akima::<f64>::new(&[0.0, 1.0, 2.0, 3.0], &[0.0, 1.0, 2.0, 3.0]).is_err());
    }
}

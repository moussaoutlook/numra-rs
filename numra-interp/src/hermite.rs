//! Piecewise Cubic Hermite Interpolating Polynomial (PCHIP).
//!
//! Shape-preserving interpolation using the Fritsch-Carlson method.
//! Preserves monotonicity of the data.
//!
//! Author: Moussa Leblouba
//! Date: 9 February 2026
//! Modified: 2 May 2026

use numra_core::Scalar;

use crate::cubic_spline::coefficients_from_slopes;
use crate::error::InterpError;
use crate::{eval_piecewise_cubic, eval_piecewise_cubic_deriv, integrate_piecewise_cubic};
use crate::{validate_data, Interpolant};

/// Piecewise Cubic Hermite Interpolating Polynomial (PCHIP).
///
/// Uses the Fritsch-Carlson method to compute slopes that preserve
/// monotonicity of the data.
pub struct Pchip<S: Scalar> {
    x: Vec<S>,
    a: Vec<S>,
    b: Vec<S>,
    c: Vec<S>,
    d: Vec<S>,
}

impl<S: Scalar> Pchip<S> {
    /// Create a PCHIP interpolant from data points.
    ///
    /// # Errors
    ///
    /// Returns error if fewer than 2 points or data is not sorted.
    pub fn new(x: &[S], y: &[S]) -> Result<Self, InterpError> {
        validate_data(x, y, 2)?;
        let n = x.len();
        let slopes = compute_pchip_slopes(x, y);
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

impl<S: Scalar> Interpolant<S> for Pchip<S> {
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

/// Compute PCHIP slopes using Fritsch-Carlson method.
fn compute_pchip_slopes<S: Scalar>(x: &[S], y: &[S]) -> Vec<S> {
    let n = x.len();
    let mut slopes = vec![S::ZERO; n];

    if n == 2 {
        let s = (y[1] - y[0]) / (x[1] - x[0]);
        slopes[0] = s;
        slopes[1] = s;
        return slopes;
    }

    // Compute secants
    let n_seg = n - 1;
    let h: Vec<S> = (0..n_seg).map(|i| x[i + 1] - x[i]).collect();
    let s: Vec<S> = (0..n_seg).map(|i| (y[i + 1] - y[i]) / h[i]).collect();

    // Interior slopes
    for i in 1..n - 1 {
        if (s[i - 1] > S::ZERO && s[i] > S::ZERO) || (s[i - 1] < S::ZERO && s[i] < S::ZERO) {
            // Same sign: weighted harmonic mean
            let w1 = S::TWO * h[i] + h[i - 1];
            let w2 = h[i] + S::TWO * h[i - 1];
            slopes[i] = (w1 + w2) / (w1 / s[i - 1] + w2 / s[i]);
        }
        // else: slopes[i] = 0 (already initialized)
    }

    // Endpoint slopes: one-sided formula with limiting
    slopes[0] = endpoint_slope(h[0], h[1], s[0], s[1]);
    slopes[n - 1] = endpoint_slope(h[n_seg - 1], h[n_seg - 2], s[n_seg - 1], s[n_seg - 2]);

    slopes
}

/// One-sided endpoint slope with shape-preservation limiting.
fn endpoint_slope<S: Scalar>(h1: S, h2: S, s1: S, s2: S) -> S {
    let d = ((S::TWO * h1 + h2) * s1 - h1 * s2) / (h1 + h2);

    // Limit to preserve shape
    if (d > S::ZERO) != (s1 > S::ZERO) {
        return S::ZERO;
    }
    let three = S::from_f64(3.0);
    if (s1 > S::ZERO) != (s2 > S::ZERO) && d.abs() > three * s1.abs() {
        return three * s1;
    }
    d
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_pchip_at_knots() {
        let x = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let y = vec![0.0, 1.0, 0.5, 1.5, 1.0];
        let p = Pchip::new(&x, &y).unwrap();
        for (&xi, &yi) in x.iter().zip(y.iter()) {
            assert_relative_eq!(p.interpolate(xi), yi, epsilon = 1e-12);
        }
    }

    #[test]
    fn test_pchip_monotonicity() {
        // Monotone increasing data
        let x = vec![0.0, 1.0, 2.0, 5.0, 10.0];
        let y = vec![0.0, 1.0, 4.0, 5.0, 10.0];
        let p = Pchip::new(&x, &y).unwrap();

        // Check monotonicity between knots
        let mut prev = p.interpolate(0.0);
        for i in 1..100 {
            let xi = i as f64 * 10.0 / 100.0;
            let yi = p.interpolate(xi);
            assert!(
                yi >= prev - 1e-10,
                "Monotonicity violation at x={}: {} < {}",
                xi,
                yi,
                prev
            );
            prev = yi;
        }
    }

    #[test]
    fn test_pchip_step_function() {
        // Step-like data: PCHIP should not overshoot
        let x = vec![0.0, 1.0, 1.5, 2.0, 3.0];
        let y = vec![0.0, 0.0, 1.0, 1.0, 1.0];
        let p = Pchip::new(&x, &y).unwrap();

        // Should not go below 0 or above 1
        for i in 0..100 {
            let xi = i as f64 * 3.0 / 100.0;
            let yi = p.interpolate(xi);
            assert!(yi >= -0.01 && yi <= 1.01, "Overshoot at x={}: y={}", xi, yi);
        }
    }

    #[test]
    fn test_pchip_two_points() {
        let p = Pchip::new(&[0.0, 1.0], &[0.0, 1.0]).unwrap();
        assert_relative_eq!(p.interpolate(0.5), 0.5, epsilon = 1e-14);
    }

    #[test]
    fn test_pchip_derivative() {
        let x = vec![0.0, 1.0, 2.0, 3.0];
        let y = vec![0.0, 1.0, 4.0, 9.0]; // x^2
        let p = Pchip::new(&x, &y).unwrap();
        // Derivative at x=1 should be close to 2
        let d = p.derivative(1.0).unwrap();
        assert!((d - 2.0).abs() < 1.0, "Derivative at 1.0: {}", d);
    }

    #[test]
    fn test_pchip_integrate() {
        // Integral of x from 0 to 3 = 4.5
        let x = vec![0.0, 1.0, 2.0, 3.0];
        let y = vec![0.0, 1.0, 2.0, 3.0];
        let p = Pchip::new(&x, &y).unwrap();
        let integral = p.integrate(0.0, 3.0).unwrap();
        assert_relative_eq!(integral, 4.5, epsilon = 1e-10);
    }
}

//! Piecewise linear interpolation.
//!
//! Author: Moussa Leblouba
//! Date: 9 February 2026
//! Modified: 2 May 2026

use numra_core::Scalar;

use crate::error::InterpError;
use crate::{search_sorted, validate_data, Interpolant};

/// Piecewise linear interpolant.
pub struct Linear<S: Scalar> {
    x: Vec<S>,
    y: Vec<S>,
}

impl<S: Scalar> Linear<S> {
    /// Create a linear interpolant from data points.
    ///
    /// # Errors
    ///
    /// Returns error if fewer than 2 points or data is not sorted.
    pub fn new(x: &[S], y: &[S]) -> Result<Self, InterpError> {
        validate_data(x, y, 2)?;
        Ok(Self {
            x: x.to_vec(),
            y: y.to_vec(),
        })
    }
}

impl<S: Scalar> Interpolant<S> for Linear<S> {
    fn interpolate(&self, x: S) -> S {
        let i = search_sorted(&self.x, x);
        let h = self.x[i + 1] - self.x[i];
        let t = (x - self.x[i]) / h;
        self.y[i] * (S::ONE - t) + self.y[i + 1] * t
    }

    fn derivative(&self, x: S) -> Option<S> {
        let i = search_sorted(&self.x, x);
        let h = self.x[i + 1] - self.x[i];
        Some((self.y[i + 1] - self.y[i]) / h)
    }

    fn integrate(&self, a: S, b: S) -> Option<S> {
        if b.to_f64() <= a.to_f64() {
            return Some(S::ZERO);
        }
        let i_lo = search_sorted(&self.x, a);
        let i_hi = search_sorted(&self.x, b);

        let mut result = S::ZERO;
        for i in i_lo..=i_hi {
            let x_lo = if i == i_lo { a } else { self.x[i] };
            let x_hi = if i == i_hi { b } else { self.x[i + 1] };
            let y_lo = self.interpolate(x_lo);
            let y_hi = self.interpolate(x_hi);
            result += (y_lo + y_hi) * (x_hi - x_lo) * S::HALF;
        }
        Some(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_linear_at_knots() {
        let interp = Linear::new(&[0.0, 1.0, 2.0], &[0.0, 2.0, 1.0]).unwrap();
        assert_relative_eq!(interp.interpolate(0.0), 0.0, epsilon = 1e-14);
        assert_relative_eq!(interp.interpolate(1.0), 2.0, epsilon = 1e-14);
        assert_relative_eq!(interp.interpolate(2.0), 1.0, epsilon = 1e-14);
    }

    #[test]
    fn test_linear_midpoints() {
        let interp = Linear::new(&[0.0, 1.0, 2.0], &[0.0, 2.0, 1.0]).unwrap();
        assert_relative_eq!(interp.interpolate(0.5), 1.0, epsilon = 1e-14);
        assert_relative_eq!(interp.interpolate(1.5), 1.5, epsilon = 1e-14);
    }

    #[test]
    fn test_linear_derivative() {
        let interp = Linear::new(&[0.0, 1.0, 3.0], &[0.0, 2.0, 6.0]).unwrap();
        assert_relative_eq!(interp.derivative(0.5).unwrap(), 2.0, epsilon = 1e-14);
        assert_relative_eq!(interp.derivative(2.0).unwrap(), 2.0, epsilon = 1e-14);
    }

    #[test]
    fn test_linear_integrate() {
        // Integral of y=x from 0 to 2
        let interp = Linear::new(&[0.0, 1.0, 2.0], &[0.0, 1.0, 2.0]).unwrap();
        assert_relative_eq!(interp.integrate(0.0, 2.0).unwrap(), 2.0, epsilon = 1e-14);
    }

    #[test]
    fn test_linear_extrapolation() {
        let interp = Linear::new(&[0.0, 1.0], &[0.0, 1.0]).unwrap();
        // Extrapolates using first/last segment
        assert_relative_eq!(interp.interpolate(-0.5), -0.5, epsilon = 1e-14);
        assert_relative_eq!(interp.interpolate(1.5), 1.5, epsilon = 1e-14);
    }

    #[test]
    fn test_linear_errors() {
        assert!(Linear::<f64>::new(&[1.0], &[1.0]).is_err());
        assert!(Linear::new(&[1.0, 2.0], &[1.0]).is_err());
        assert!(Linear::new(&[2.0, 1.0], &[1.0, 2.0]).is_err());
    }

    #[test]
    fn test_linear_f32() {
        let interp = Linear::new(&[0.0f32, 1.0, 2.0], &[0.0, 1.0, 0.0]).unwrap();
        assert!((interp.interpolate(0.5f32) - 0.5).abs() < 1e-6);
    }
}

//! Polynomial interpolation via barycentric Lagrange formula.
//!
//! Author: Moussa Leblouba
//! Date: 9 February 2026
//! Modified: 2 May 2026

use numra_core::Scalar;

use crate::error::InterpError;
use crate::{validate_data, Interpolant};

/// Barycentric Lagrange interpolant.
///
/// Evaluates the unique polynomial of degree `n-1` passing through `n` data points
/// using the numerically stable barycentric formula. O(n) per evaluation after O(n^2) setup.
pub struct BarycentricLagrange<S: Scalar> {
    x: Vec<S>,
    y: Vec<S>,
    w: Vec<S>, // barycentric weights
}

impl<S: Scalar> BarycentricLagrange<S> {
    /// Create a barycentric Lagrange interpolant from data points.
    ///
    /// # Errors
    ///
    /// Returns error if fewer than 1 point or data is not sorted.
    pub fn new(x: &[S], y: &[S]) -> Result<Self, InterpError> {
        validate_data(x, y, 1)?;
        let w = compute_barycentric_weights(x);
        Ok(Self {
            x: x.to_vec(),
            y: y.to_vec(),
            w,
        })
    }

    /// Create from Chebyshev nodes on `[a, b]`.
    ///
    /// Uses `n` Chebyshev nodes of the first kind, which minimize the
    /// maximum interpolation error (near-optimal for polynomial interpolation).
    /// Uses the known closed-form barycentric weights for numerical stability.
    pub fn chebyshev<F: Fn(S) -> S>(f: F, a: S, b: S, n: usize) -> Self {
        let mut x = Vec::with_capacity(n);
        let mut y = Vec::with_capacity(n);
        let mut orig_weights = Vec::with_capacity(n);

        for k in 0..n {
            // Chebyshev nodes: x_k = cos((2k+1)/(2n) * pi), mapped to [a,b]
            let theta_f = (2 * k + 1) as f64 / (2 * n) as f64 * core::f64::consts::PI;
            let theta = S::from_f64(theta_f);
            let node = (a + b) * S::HALF + (b - a) * S::HALF * theta.cos();
            x.push(node);
            y.push(f(node));
            // Closed-form barycentric weight: (-1)^k * sin((2k+1)*pi/(2n))
            let sign = if k % 2 == 0 { 1.0 } else { -1.0 };
            orig_weights.push(sign * theta_f.sin());
        }

        // Sort by x (Chebyshev nodes come in decreasing order from cos)
        let mut indices: Vec<usize> = (0..n).collect();
        indices.sort_by(|&i, &j| x[i].to_f64().partial_cmp(&x[j].to_f64()).unwrap());
        let x_sorted: Vec<S> = indices.iter().map(|&i| x[i]).collect();
        let y_sorted: Vec<S> = indices.iter().map(|&i| y[i]).collect();
        let w: Vec<S> = indices
            .iter()
            .map(|&i| S::from_f64(orig_weights[i]))
            .collect();

        Self {
            x: x_sorted,
            y: y_sorted,
            w,
        }
    }
}

impl<S: Scalar> Interpolant<S> for BarycentricLagrange<S> {
    fn interpolate(&self, x: S) -> S {
        let n = self.x.len();

        // Check if x is exactly at a node
        for i in 0..n {
            let diff = x - self.x[i];
            if diff.abs() < S::EPSILON * S::from_f64(100.0) {
                return self.y[i];
            }
        }

        // Barycentric formula: p(x) = (sum w_i * y_i / (x - x_i)) / (sum w_i / (x - x_i))
        let mut numer = S::ZERO;
        let mut denom = S::ZERO;
        for i in 0..n {
            let t = self.w[i] / (x - self.x[i]);
            numer += t * self.y[i];
            denom += t;
        }
        numer / denom
    }
}

/// Compute barycentric weights.
/// Computes in f64 and normalizes to avoid cancellation in the barycentric formula.
fn compute_barycentric_weights<S: Scalar>(x: &[S]) -> Vec<S> {
    let n = x.len();
    let xf: Vec<f64> = x.iter().map(|xi| xi.to_f64()).collect();
    let mut wf = vec![1.0_f64; n];
    for i in 0..n {
        for j in 0..n {
            if i != j {
                wf[i] /= xf[i] - xf[j];
            }
        }
    }
    // Normalize to O(1) to reduce cancellation in the barycentric summation
    let max_abs = wf.iter().map(|w| w.abs()).fold(0.0_f64, f64::max);
    if max_abs > 0.0 {
        for w in &mut wf {
            *w /= max_abs;
        }
    }
    wf.iter().map(|&w| S::from_f64(w)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_lagrange_at_knots() {
        let x = vec![0.0, 1.0, 2.0, 3.0];
        let y = vec![1.0, 3.0, 2.0, 4.0];
        let p = BarycentricLagrange::new(&x, &y).unwrap();
        for (&xi, &yi) in x.iter().zip(y.iter()) {
            assert_relative_eq!(p.interpolate(xi), yi, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_lagrange_polynomial_exact() {
        // Polynomial of degree 2: y = x^2
        let x = vec![0.0, 1.0, 2.0];
        let y = vec![0.0, 1.0, 4.0];
        let p = BarycentricLagrange::new(&x, &y).unwrap();
        // Should reproduce x^2 exactly
        assert_relative_eq!(p.interpolate(0.5), 0.25, epsilon = 1e-10);
        assert_relative_eq!(p.interpolate(1.5), 2.25, epsilon = 1e-10);
    }

    #[test]
    fn test_lagrange_cubic_exact() {
        // Polynomial of degree 3: y = x^3 - x
        let x = vec![0.0, 1.0, 2.0, 3.0];
        let y: Vec<f64> = x.iter().map(|&xi| xi.powi(3) - xi).collect();
        let p = BarycentricLagrange::new(&x, &y).unwrap();
        assert_relative_eq!(p.interpolate(0.5), 0.5_f64.powi(3) - 0.5, epsilon = 1e-10);
        assert_relative_eq!(p.interpolate(2.5), 2.5_f64.powi(3) - 2.5, epsilon = 1e-10);
    }

    #[test]
    fn test_chebyshev_sin() {
        // Interpolate sin(x) on [0, pi] with 10 Chebyshev nodes
        let p = BarycentricLagrange::chebyshev(|x: f64| x.sin(), 0.0, core::f64::consts::PI, 10);
        // Check at several points
        for k in 1..10 {
            let x = k as f64 * core::f64::consts::PI / 10.0;
            let err = (p.interpolate(x) - x.sin()).abs();
            assert!(err < 1e-7, "Chebyshev error at x={}: {}", x, err);
        }
    }

    #[test]
    fn test_chebyshev_runge() {
        // Runge's function 1/(1+25x^2) on [-1, 1]
        // Chebyshev nodes should handle this much better than equidistant.
        // Need ~50 nodes for high accuracy due to poles at x = ±i/5 in C.
        let p = BarycentricLagrange::chebyshev(|x: f64| 1.0 / (1.0 + 25.0 * x * x), -1.0, 1.0, 50);
        // Check at x=0 (should be 1.0)
        assert_relative_eq!(p.interpolate(0.0), 1.0, epsilon = 1e-3);
        // Check near the edges (where equidistant nodes fail badly)
        let y_edge = p.interpolate(0.9);
        let exact = 1.0 / (1.0 + 25.0 * 0.81);
        assert!(
            (y_edge - exact).abs() < 1e-4,
            "Runge at 0.9: {} vs {}",
            y_edge,
            exact
        );
    }

    #[test]
    fn test_lagrange_single_point() {
        let p = BarycentricLagrange::new(&[1.0], &[5.0]).unwrap();
        assert_relative_eq!(p.interpolate(1.0), 5.0, epsilon = 1e-14);
        // Constant function
        assert_relative_eq!(p.interpolate(2.0), 5.0, epsilon = 1e-14);
    }

    #[test]
    fn test_lagrange_f32() {
        let p = BarycentricLagrange::new(&[0.0f32, 1.0, 2.0], &[0.0, 1.0, 4.0]).unwrap();
        assert!((p.interpolate(0.5f32) - 0.25).abs() < 1e-4);
    }
}

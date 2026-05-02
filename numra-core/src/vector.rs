//! Vector trait for numerical computation.
//!
//! This module defines the [`Vector`] trait which abstracts over different
//! vector implementations, allowing algorithms to work with:
//! - `Vec<S>` (standard library)
//! - faer vectors (when using numra-linalg)
//! - Fixed-size arrays `[S; N]`
//!
//! # Design Philosophy
//!
//! The trait is designed around BLAS-like operations that are fundamental
//! to numerical algorithms: axpy, dot product, norms, etc.
//!
//! Author: Moussa Leblouba
//! Date: 4 February 2026
//! Modified: 2 May 2026

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use crate::Scalar;

/// A vector type for numerical computation.
///
/// This trait provides the essential operations needed by ODE/SDE solvers
/// and other numerical algorithms.
///
/// # Example
///
/// ```rust
/// use numra_core::Vector;
///
/// fn compute_weighted_sum<V: Vector<f64>>(a: f64, x: &V, b: f64, y: &V) -> V {
///     let mut result = x.clone();
///     result.scale(a);
///     result.axpy(b, y);
///     result
/// }
/// ```
pub trait Vector<S: Scalar>: Clone + Sized {
    /// Create a zero vector of given length.
    fn zeros(len: usize) -> Self;

    /// Create a vector filled with a constant value.
    fn fill(len: usize, value: S) -> Self;

    /// Create from a slice.
    fn from_slice(data: &[S]) -> Self;

    /// Length of the vector.
    fn len(&self) -> usize;

    /// Check if empty.
    #[inline]
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get element at index (panics if out of bounds).
    fn get(&self, i: usize) -> S;

    /// Set element at index.
    fn set(&mut self, i: usize, value: S);

    /// Get mutable reference to element.
    fn get_mut(&mut self, i: usize) -> &mut S;

    /// Return as a slice.
    fn as_slice(&self) -> &[S];

    /// Return as mutable slice.
    fn as_mut_slice(&mut self) -> &mut [S];

    /// Copy elements from another vector.
    fn copy_from(&mut self, other: &Self);

    // ===== BLAS-like Operations =====

    /// AXPY: y = a*x + y (fundamental BLAS operation)
    ///
    /// This is the most important operation for numerical algorithms.
    fn axpy(&mut self, a: S, x: &Self);

    /// Scaled add: y = a*x + b*y
    fn axpby(&mut self, a: S, x: &Self, b: S);

    /// Dot product: x·y
    fn dot(&self, other: &Self) -> S;

    /// Scale in place: x *= a
    fn scale(&mut self, a: S);

    // ===== Norms =====

    /// Euclidean norm: ||x||_2 = sqrt(x·x)
    #[inline]
    fn norm2(&self) -> S {
        self.dot(self).sqrt()
    }

    /// Infinity norm: ||x||_∞ = max|x_i|
    fn norm_inf(&self) -> S;

    /// 1-norm: ||x||_1 = Σ|x_i|
    fn norm1(&self) -> S;

    /// Weighted RMS norm: sqrt(mean((x/w)²))
    /// Used for error control in ODE solvers.
    fn weighted_rms_norm(&self, weights: &Self) -> S {
        let n = S::from_usize(self.len());
        let mut sum = S::ZERO;
        for i in 0..self.len() {
            let xi = self.get(i) / weights.get(i);
            sum += xi * xi;
        }
        (sum / n).sqrt()
    }

    // ===== Element-wise Operations =====

    /// Element-wise absolute value in place.
    fn abs_inplace(&mut self);

    /// Element-wise maximum: `self[i] = max(self[i], other[i])`
    fn max_elementwise(&mut self, other: &Self);

    /// Element-wise minimum: `self[i] = min(self[i], other[i])`
    fn min_elementwise(&mut self, other: &Self);

    // ===== Reductions =====

    /// Sum of all elements.
    fn sum(&self) -> S;

    /// Maximum element.
    fn max_element(&self) -> S;

    /// Minimum element.
    fn min_element(&self) -> S;

    // ===== Utility =====

    /// Apply a function to each element.
    fn map_inplace<F: Fn(S) -> S>(&mut self, f: F);
}

// ============================================================================
// Implementation for Vec<S>
// ============================================================================

impl<S: Scalar> Vector<S> for Vec<S> {
    #[inline]
    fn zeros(len: usize) -> Self {
        vec![S::ZERO; len]
    }

    #[inline]
    fn fill(len: usize, value: S) -> Self {
        vec![value; len]
    }

    #[inline]
    fn from_slice(data: &[S]) -> Self {
        data.to_vec()
    }

    #[inline]
    fn len(&self) -> usize {
        Vec::len(self)
    }

    #[inline]
    fn get(&self, i: usize) -> S {
        self[i]
    }

    #[inline]
    fn set(&mut self, i: usize, value: S) {
        self[i] = value;
    }

    #[inline]
    fn get_mut(&mut self, i: usize) -> &mut S {
        &mut self[i]
    }

    #[inline]
    fn as_slice(&self) -> &[S] {
        self
    }

    #[inline]
    fn as_mut_slice(&mut self) -> &mut [S] {
        self
    }

    #[inline]
    fn copy_from(&mut self, other: &Self) {
        self.copy_from_slice(other);
    }

    fn axpy(&mut self, a: S, x: &Self) {
        debug_assert_eq!(self.len(), x.len());
        for (yi, xi) in self.iter_mut().zip(x.iter()) {
            *yi += a * *xi;
        }
    }

    fn axpby(&mut self, a: S, x: &Self, b: S) {
        debug_assert_eq!(self.len(), x.len());
        for (yi, xi) in self.iter_mut().zip(x.iter()) {
            *yi = a * *xi + b * *yi;
        }
    }

    fn dot(&self, other: &Self) -> S {
        debug_assert_eq!(self.len(), other.len());
        self.iter()
            .zip(other.iter())
            .fold(S::ZERO, |acc, (a, b)| acc + *a * *b)
    }

    fn scale(&mut self, a: S) {
        for x in self.iter_mut() {
            *x *= a;
        }
    }

    fn norm_inf(&self) -> S {
        self.iter().fold(S::ZERO, |acc, x| acc.max(x.abs()))
    }

    fn norm1(&self) -> S {
        self.iter().fold(S::ZERO, |acc, x| acc + x.abs())
    }

    fn abs_inplace(&mut self) {
        for x in self.iter_mut() {
            *x = x.abs();
        }
    }

    fn max_elementwise(&mut self, other: &Self) {
        debug_assert_eq!(self.len(), other.len());
        for (yi, xi) in self.iter_mut().zip(other.iter()) {
            *yi = yi.max(*xi);
        }
    }

    fn min_elementwise(&mut self, other: &Self) {
        debug_assert_eq!(self.len(), other.len());
        for (yi, xi) in self.iter_mut().zip(other.iter()) {
            *yi = yi.min(*xi);
        }
    }

    fn sum(&self) -> S {
        self.iter().fold(S::ZERO, |acc, x| acc + *x)
    }

    fn max_element(&self) -> S {
        self.iter().fold(S::NEG_INFINITY, |acc, x| acc.max(*x))
    }

    fn min_element(&self) -> S {
        self.iter().fold(S::INFINITY, |acc, x| acc.min(*x))
    }

    fn map_inplace<F: Fn(S) -> S>(&mut self, f: F) {
        for x in self.iter_mut() {
            *x = f(*x);
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zeros() {
        let v: Vec<f64> = Vector::zeros(5);
        assert_eq!(v.len(), 5);
        for x in &v {
            assert_eq!(*x, 0.0);
        }
    }

    #[test]
    fn test_fill() {
        let v: Vec<f64> = Vector::fill(3, 2.5);
        assert_eq!(v, vec![2.5, 2.5, 2.5]);
    }

    #[test]
    fn test_axpy() {
        let x: Vec<f64> = vec![1.0, 2.0, 3.0];
        let mut y: Vec<f64> = vec![4.0, 5.0, 6.0];
        y.axpy(2.0, &x);
        assert_eq!(y, vec![6.0, 9.0, 12.0]);
    }

    #[test]
    fn test_axpby() {
        let x: Vec<f64> = vec![1.0, 2.0, 3.0];
        let mut y: Vec<f64> = vec![4.0, 5.0, 6.0];
        // y = 2*x + 0.5*y = [2,4,6] + [2,2.5,3] = [4, 6.5, 9]
        y.axpby(2.0, &x, 0.5);
        assert!((y[0] - 4.0).abs() < 1e-10);
        assert!((y[1] - 6.5).abs() < 1e-10);
        assert!((y[2] - 9.0).abs() < 1e-10);
    }

    #[test]
    fn test_dot() {
        let x: Vec<f64> = vec![1.0, 2.0, 3.0];
        let y: Vec<f64> = vec![4.0, 5.0, 6.0];
        // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
        assert!((x.dot(&y) - 32.0).abs() < 1e-10);
    }

    #[test]
    fn test_norm2() {
        let v: Vec<f64> = vec![3.0, 4.0];
        assert!((v.norm2() - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_norm_inf() {
        let v: Vec<f64> = vec![-5.0, 3.0, -1.0];
        assert!((v.norm_inf() - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_norm1() {
        let v: Vec<f64> = vec![-1.0, 2.0, -3.0];
        assert!((v.norm1() - 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_scale() {
        let mut v: Vec<f64> = vec![1.0, 2.0, 3.0];
        v.scale(2.0);
        assert_eq!(v, vec![2.0, 4.0, 6.0]);
    }

    #[test]
    fn test_sum() {
        let v: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0];
        assert!((v.sum() - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_max_min_element() {
        let v: Vec<f64> = vec![3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0];
        assert!((v.max_element() - 9.0).abs() < 1e-10);
        assert!((v.min_element() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_weighted_rms_norm() {
        // y = [2, 4], w = [1, 2]
        // (y/w)^2 = [4, 4], mean = 4, sqrt = 2
        let y: Vec<f64> = vec![2.0, 4.0];
        let w: Vec<f64> = vec![1.0, 2.0];
        assert!((y.weighted_rms_norm(&w) - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_map_inplace() {
        let mut v: Vec<f64> = vec![1.0, 4.0, 9.0];
        v.map_inplace(|x| x.sqrt());
        assert!((v[0] - 1.0).abs() < 1e-10);
        assert!((v[1] - 2.0).abs() < 1e-10);
        assert!((v[2] - 3.0).abs() < 1e-10);
    }
}

#![allow(clippy::needless_range_loop)]
#![allow(clippy::manual_range_contains)]

//! Interpolation and splines for Numra.
//!
//! This crate provides 1D interpolation methods:
//!
//! - **Linear** ([`Linear`]): Piecewise linear interpolation
//! - **Cubic spline** ([`CubicSpline`]): Natural, clamped, and not-a-knot variants
//! - **PCHIP** ([`Pchip`]): Monotonicity-preserving piecewise cubic Hermite
//! - **Akima** ([`Akima`]): Robust piecewise cubic (tolerant of outliers)
//! - **Barycentric Lagrange** ([`BarycentricLagrange`]): Stable polynomial interpolation
//!
//! Author: Moussa Leblouba
//! Date: 9 February 2026
//! Modified: 2 May 2026

pub mod akima;
pub mod cubic_spline;
pub mod error;
pub mod hermite;
pub mod linear;
pub mod polynomial;

pub use akima::Akima;
pub use cubic_spline::CubicSpline;
pub use error::InterpError;
pub use hermite::Pchip;
pub use linear::Linear;
pub use polynomial::BarycentricLagrange;

use numra_core::Scalar;

/// Trait for 1D interpolants.
pub trait Interpolant<S: Scalar> {
    /// Evaluate the interpolant at `x`.
    /// Extrapolates using the first/last segment if `x` is outside the data range.
    fn interpolate(&self, x: S) -> S;

    /// Evaluate at multiple points.
    fn interpolate_vec(&self, xs: &[S]) -> Vec<S> {
        xs.iter().map(|&x| self.interpolate(x)).collect()
    }

    /// First derivative at `x`, if available.
    fn derivative(&self, x: S) -> Option<S> {
        let _ = x;
        None
    }

    /// Definite integral from `a` to `b`, if available.
    fn integrate(&self, a: S, b: S) -> Option<S> {
        let _ = (a, b);
        None
    }
}

/// Interpolation method selector for [`interp1d`].
pub enum Interp1dMethod {
    Linear,
    CubicSpline,
    Pchip,
    Akima,
}

/// Convenience: create an interpolant from data using the specified method.
pub fn interp1d<S: Scalar>(
    x: &[S],
    y: &[S],
    method: Interp1dMethod,
) -> Result<Box<dyn Interpolant<S>>, InterpError> {
    match method {
        Interp1dMethod::Linear => Ok(Box::new(Linear::new(x, y)?)),
        Interp1dMethod::CubicSpline => Ok(Box::new(CubicSpline::natural(x, y)?)),
        Interp1dMethod::Pchip => Ok(Box::new(Pchip::new(x, y)?)),
        Interp1dMethod::Akima => Ok(Box::new(Akima::new(x, y)?)),
    }
}

// ============================================================================
// Internal utilities
// ============================================================================

/// Binary search for the interval containing `x`.
/// Returns `i` such that `xs[i] <= x < xs[i+1]`, clamped to valid range.
pub(crate) fn search_sorted<S: Scalar>(xs: &[S], x: S) -> usize {
    let n = xs.len();
    if n < 2 {
        return 0;
    }
    if x.to_f64() <= xs[0].to_f64() {
        return 0;
    }
    if x.to_f64() >= xs[n - 1].to_f64() {
        return n - 2;
    }
    let mut lo = 0;
    let mut hi = n - 1;
    while lo < hi - 1 {
        let mid = (lo + hi) / 2;
        if x < xs[mid] {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    lo
}

/// Validate interpolation data: matching lengths, sufficient points, sorted.
pub(crate) fn validate_data<S: Scalar>(
    x: &[S],
    y: &[S],
    min_points: usize,
) -> Result<(), InterpError> {
    if x.len() != y.len() {
        return Err(InterpError::DimensionMismatch {
            x_len: x.len(),
            y_len: y.len(),
        });
    }
    if x.len() < min_points {
        return Err(InterpError::TooFewPoints {
            got: x.len(),
            min: min_points,
        });
    }
    for i in 1..x.len() {
        if x[i].to_f64() <= x[i - 1].to_f64() {
            return Err(InterpError::UnsortedData);
        }
    }
    Ok(())
}

/// Evaluate piecewise cubic: a[i] + b[i]*t + c[i]*t^2 + d[i]*t^3 where t = x - x_knots[i].
pub(crate) fn eval_piecewise_cubic<S: Scalar>(
    x_knots: &[S],
    a: &[S],
    b: &[S],
    c: &[S],
    d: &[S],
    x: S,
) -> S {
    let i = search_sorted(x_knots, x);
    let t = x - x_knots[i];
    a[i] + t * (b[i] + t * (c[i] + t * d[i]))
}

/// Derivative of piecewise cubic: b[i] + 2*c[i]*t + 3*d[i]*t^2.
pub(crate) fn eval_piecewise_cubic_deriv<S: Scalar>(
    x_knots: &[S],
    b: &[S],
    c: &[S],
    d: &[S],
    x: S,
) -> S {
    let i = search_sorted(x_knots, x);
    let t = x - x_knots[i];
    b[i] + t * (S::TWO * c[i] + S::from_f64(3.0) * t * d[i])
}

/// Integrate piecewise cubic from `lo` to `hi`.
pub(crate) fn integrate_piecewise_cubic<S: Scalar>(
    x_knots: &[S],
    a: &[S],
    b: &[S],
    c: &[S],
    d: &[S],
    lo: S,
    hi: S,
) -> S {
    if hi.to_f64() <= lo.to_f64() {
        return S::ZERO;
    }
    let i_lo = search_sorted(x_knots, lo);
    let i_hi = search_sorted(x_knots, hi);

    let antideriv = |idx: usize, t: S| -> S {
        let t2 = t * t;
        let t3 = t2 * t;
        let t4 = t3 * t;
        a[idx] * t
            + b[idx] * t2 * S::HALF
            + c[idx] * t3 / S::from_f64(3.0)
            + d[idx] * t4 / S::from_f64(4.0)
    };

    let mut result = S::ZERO;
    for i in i_lo..=i_hi {
        let t_lo = if i == i_lo { lo - x_knots[i] } else { S::ZERO };
        let t_hi = if i == i_hi {
            hi - x_knots[i]
        } else {
            x_knots[i + 1] - x_knots[i]
        };
        result += antideriv(i, t_hi) - antideriv(i, t_lo);
    }
    result
}

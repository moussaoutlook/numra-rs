//! Scalar type trait for numerical computation.
//!
//! This module defines the [`Scalar`] trait which captures all operations needed
//! for numerical algorithms without depending on external crates.
//!
//! # Supported Types
//!
//! - `f64` - 64-bit IEEE 754 floating point (recommended for most use)
//! - `f32` - 32-bit IEEE 754 floating point (for memory-constrained applications)
//!
//! # Design Notes
//!
//! The trait is designed to be:
//! - Self-contained (no external trait dependencies like num-traits or nalgebra)
//! - Comprehensive (all operations needed by numerical algorithms)
//! - Zero-cost (all methods inline to optimal assembly)

use core::fmt::{Debug, Display};
use core::iter::Sum;
use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

/// A real scalar type suitable for numerical computation.
///
/// This trait provides all mathematical operations needed by numerical methods,
/// including basic arithmetic, trigonometric functions, and special functions.
///
/// # Example
///
/// ```rust
/// use numra_core::Scalar;
///
/// fn quadratic_formula<S: Scalar>(a: S, b: S, c: S) -> Option<(S, S)> {
///     let discriminant = b * b - S::from_f64(4.0) * a * c;
///     if discriminant < S::ZERO {
///         return None;
///     }
///     let sqrt_d = discriminant.sqrt();
///     let two_a = S::TWO * a;
///     Some(((-b + sqrt_d) / two_a, (-b - sqrt_d) / two_a))
/// }
/// ```
//!
//! Author: Moussa Leblouba
//! Date: 9 February 2026
//! Modified: 2 May 2026

pub trait Scalar:
    Copy
    + Clone
    + Debug
    + Display
    + PartialOrd
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Neg<Output = Self>
    + AddAssign
    + SubAssign
    + MulAssign
    + DivAssign
    + Sum
    + Send
    + Sync
    + 'static
{
    // ===== Constants =====

    /// Additive identity: 0
    const ZERO: Self;
    /// Multiplicative identity: 1
    const ONE: Self;
    /// Two (commonly needed constant)
    const TWO: Self;
    /// One half
    const HALF: Self;
    /// Machine epsilon (smallest x such that 1 + x ≠ 1)
    const EPSILON: Self;
    /// Positive infinity
    const INFINITY: Self;
    /// Negative infinity
    const NEG_INFINITY: Self;
    /// Not a number
    const NAN: Self;
    /// Pi (π)
    const PI: Self;
    /// Euler's number (e)
    const E: Self;
    /// Square root of 2
    const SQRT_2: Self;
    /// Natural log of 2
    const LN_2: Self;

    // ===== Conversion =====

    /// Create from f64
    fn from_f64(x: f64) -> Self;

    /// Create from f32
    fn from_f32(x: f32) -> Self;

    /// Create from i32
    fn from_i32(x: i32) -> Self;

    /// Create from usize
    fn from_usize(n: usize) -> Self;

    /// Convert to f64
    fn to_f64(self) -> f64;

    /// Convert to f32
    fn to_f32(self) -> f32;

    // ===== Basic Operations =====

    /// Absolute value
    fn abs(self) -> Self;

    /// Square root
    fn sqrt(self) -> Self;

    /// Cube root
    fn cbrt(self) -> Self;

    /// Square (x²) - more efficient than x * x for some types
    #[inline]
    fn sq(self) -> Self {
        self * self
    }

    /// Integer power
    fn powi(self, n: i32) -> Self;

    /// Floating point power
    fn powf(self, n: Self) -> Self;

    /// Reciprocal (1/x)
    #[inline]
    fn recip(self) -> Self {
        Self::ONE / self
    }

    /// Hypotenuse: sqrt(x² + y²) computed without overflow
    fn hypot(self, other: Self) -> Self;

    // ===== Trigonometric Functions =====

    /// Sine
    fn sin(self) -> Self;

    /// Cosine
    fn cos(self) -> Self;

    /// Tangent
    fn tan(self) -> Self;

    /// Arcsine
    fn asin(self) -> Self;

    /// Arccosine
    fn acos(self) -> Self;

    /// Arctangent
    fn atan(self) -> Self;

    /// Two-argument arctangent (atan2)
    fn atan2(self, other: Self) -> Self;

    /// Simultaneous sine and cosine (more efficient than separate calls)
    #[inline]
    fn sincos(self) -> (Self, Self) {
        (self.sin(), self.cos())
    }

    // ===== Exponential and Logarithmic =====

    /// Natural exponential (e^x)
    fn exp(self) -> Self;

    /// Base-2 exponential (2^x)
    fn exp2(self) -> Self;

    /// exp(x) - 1, accurate for small x
    fn exp_m1(self) -> Self;

    /// Natural logarithm
    fn ln(self) -> Self;

    /// Base-2 logarithm
    fn log2(self) -> Self;

    /// Base-10 logarithm
    fn log10(self) -> Self;

    /// ln(1 + x), accurate for small x
    fn ln_1p(self) -> Self;

    // ===== Hyperbolic Functions =====

    /// Hyperbolic sine
    fn sinh(self) -> Self;

    /// Hyperbolic cosine
    fn cosh(self) -> Self;

    /// Hyperbolic tangent
    fn tanh(self) -> Self;

    /// Inverse hyperbolic sine
    fn asinh(self) -> Self;

    /// Inverse hyperbolic cosine
    fn acosh(self) -> Self;

    /// Inverse hyperbolic tangent
    fn atanh(self) -> Self;

    // ===== Comparison and Ordering =====

    /// Maximum of two values
    fn max(self, other: Self) -> Self;

    /// Minimum of two values
    fn min(self, other: Self) -> Self;

    /// Clamp value to range [min, max]
    fn clamp(self, min: Self, max: Self) -> Self;

    /// Copy sign from another value
    fn copysign(self, sign: Self) -> Self;

    // ===== Predicates =====

    /// Is the value finite (not NaN or infinity)?
    fn is_finite(self) -> bool;

    /// Is the value NaN?
    fn is_nan(self) -> bool;

    /// Is the value infinite?
    fn is_infinite(self) -> bool;

    /// Is the sign positive (including +0)?
    fn is_sign_positive(self) -> bool;

    /// Is the sign negative (including -0)?
    fn is_sign_negative(self) -> bool;

    // ===== Rounding =====

    /// Round toward negative infinity
    fn floor(self) -> Self;

    /// Round toward positive infinity
    fn ceil(self) -> Self;

    /// Round to nearest integer
    fn round(self) -> Self;

    /// Round toward zero
    fn trunc(self) -> Self;

    /// Fractional part
    fn fract(self) -> Self;

    // ===== Special Functions =====

    /// Gamma function Γ(x)
    fn gamma_fn(self) -> Self;

    /// Natural log of gamma function ln(Γ(x))
    fn ln_gamma(self) -> Self;

    /// Error function erf(x)
    fn erf_fn(self) -> Self;

    /// Complementary error function erfc(x) = 1 - erf(x)
    fn erfc_fn(self) -> Self;

    // ===== Utility Methods =====

    /// Fused multiply-add: (self * a) + b with single rounding
    fn mul_add(self, a: Self, b: Self) -> Self;

    /// Sign function: -1, 0, or 1
    #[inline]
    fn signum(self) -> Self {
        if self > Self::ZERO {
            Self::ONE
        } else if self < Self::ZERO {
            -Self::ONE
        } else {
            Self::ZERO
        }
    }
}

// ============================================================================
// Implementation for f64
// ============================================================================

impl Scalar for f64 {
    const ZERO: Self = 0.0;
    const ONE: Self = 1.0;
    const TWO: Self = 2.0;
    const HALF: Self = 0.5;
    const EPSILON: Self = f64::EPSILON;
    const INFINITY: Self = f64::INFINITY;
    const NEG_INFINITY: Self = f64::NEG_INFINITY;
    const NAN: Self = f64::NAN;
    const PI: Self = core::f64::consts::PI;
    const E: Self = core::f64::consts::E;
    const SQRT_2: Self = core::f64::consts::SQRT_2;
    const LN_2: Self = core::f64::consts::LN_2;

    #[inline]
    fn from_f64(x: f64) -> Self {
        x
    }
    #[inline]
    fn from_f32(x: f32) -> Self {
        x as f64
    }
    #[inline]
    fn from_i32(x: i32) -> Self {
        x as f64
    }
    #[inline]
    fn from_usize(n: usize) -> Self {
        n as f64
    }
    #[inline]
    fn to_f64(self) -> f64 {
        self
    }
    #[inline]
    fn to_f32(self) -> f32 {
        self as f32
    }

    #[inline]
    fn abs(self) -> Self {
        libm::fabs(self)
    }
    #[inline]
    fn sqrt(self) -> Self {
        libm::sqrt(self)
    }
    #[inline]
    fn cbrt(self) -> Self {
        libm::cbrt(self)
    }
    #[inline]
    fn powi(self, n: i32) -> Self {
        libm::pow(self, n as f64)
    }
    #[inline]
    fn powf(self, n: Self) -> Self {
        libm::pow(self, n)
    }
    #[inline]
    fn hypot(self, other: Self) -> Self {
        libm::hypot(self, other)
    }

    #[inline]
    fn sin(self) -> Self {
        libm::sin(self)
    }
    #[inline]
    fn cos(self) -> Self {
        libm::cos(self)
    }
    #[inline]
    fn tan(self) -> Self {
        libm::tan(self)
    }
    #[inline]
    fn asin(self) -> Self {
        libm::asin(self)
    }
    #[inline]
    fn acos(self) -> Self {
        libm::acos(self)
    }
    #[inline]
    fn atan(self) -> Self {
        libm::atan(self)
    }
    #[inline]
    fn atan2(self, other: Self) -> Self {
        libm::atan2(self, other)
    }
    #[inline]
    fn sincos(self) -> (Self, Self) {
        libm::sincos(self)
    }

    #[inline]
    fn exp(self) -> Self {
        libm::exp(self)
    }
    #[inline]
    fn exp2(self) -> Self {
        libm::exp2(self)
    }
    #[inline]
    fn exp_m1(self) -> Self {
        libm::expm1(self)
    }
    #[inline]
    fn ln(self) -> Self {
        libm::log(self)
    }
    #[inline]
    fn log2(self) -> Self {
        libm::log2(self)
    }
    #[inline]
    fn log10(self) -> Self {
        libm::log10(self)
    }
    #[inline]
    fn ln_1p(self) -> Self {
        libm::log1p(self)
    }

    #[inline]
    fn sinh(self) -> Self {
        libm::sinh(self)
    }
    #[inline]
    fn cosh(self) -> Self {
        libm::cosh(self)
    }
    #[inline]
    fn tanh(self) -> Self {
        libm::tanh(self)
    }
    #[inline]
    fn asinh(self) -> Self {
        libm::asinh(self)
    }
    #[inline]
    fn acosh(self) -> Self {
        libm::acosh(self)
    }
    #[inline]
    fn atanh(self) -> Self {
        libm::atanh(self)
    }

    #[inline]
    fn max(self, other: Self) -> Self {
        libm::fmax(self, other)
    }
    #[inline]
    fn min(self, other: Self) -> Self {
        libm::fmin(self, other)
    }
    #[inline]
    fn clamp(self, min: Self, max: Self) -> Self {
        libm::fmax(min, libm::fmin(self, max))
    }
    #[inline]
    fn copysign(self, sign: Self) -> Self {
        libm::copysign(self, sign)
    }

    #[inline]
    fn is_finite(self) -> bool {
        self.is_finite()
    }
    #[inline]
    fn is_nan(self) -> bool {
        self.is_nan()
    }
    #[inline]
    fn is_infinite(self) -> bool {
        self.is_infinite()
    }
    #[inline]
    fn is_sign_positive(self) -> bool {
        self.is_sign_positive()
    }
    #[inline]
    fn is_sign_negative(self) -> bool {
        self.is_sign_negative()
    }

    #[inline]
    fn floor(self) -> Self {
        libm::floor(self)
    }
    #[inline]
    fn ceil(self) -> Self {
        libm::ceil(self)
    }
    #[inline]
    fn round(self) -> Self {
        libm::round(self)
    }
    #[inline]
    fn trunc(self) -> Self {
        libm::trunc(self)
    }
    #[inline]
    fn fract(self) -> Self {
        self - libm::trunc(self)
    }

    #[inline]
    fn gamma_fn(self) -> Self {
        libm::tgamma(self)
    }
    #[inline]
    fn ln_gamma(self) -> Self {
        libm::lgamma(self)
    }
    #[inline]
    fn erf_fn(self) -> Self {
        libm::erf(self)
    }
    #[inline]
    fn erfc_fn(self) -> Self {
        libm::erfc(self)
    }

    #[inline]
    fn mul_add(self, a: Self, b: Self) -> Self {
        libm::fma(self, a, b)
    }
}

// ============================================================================
// Implementation for f32
// ============================================================================

impl Scalar for f32 {
    const ZERO: Self = 0.0;
    const ONE: Self = 1.0;
    const TWO: Self = 2.0;
    const HALF: Self = 0.5;
    const EPSILON: Self = f32::EPSILON;
    const INFINITY: Self = f32::INFINITY;
    const NEG_INFINITY: Self = f32::NEG_INFINITY;
    const NAN: Self = f32::NAN;
    const PI: Self = core::f32::consts::PI;
    const E: Self = core::f32::consts::E;
    const SQRT_2: Self = core::f32::consts::SQRT_2;
    const LN_2: Self = core::f32::consts::LN_2;

    #[inline]
    fn from_f64(x: f64) -> Self {
        x as f32
    }
    #[inline]
    fn from_f32(x: f32) -> Self {
        x
    }
    #[inline]
    fn from_i32(x: i32) -> Self {
        x as f32
    }
    #[inline]
    fn from_usize(n: usize) -> Self {
        n as f32
    }
    #[inline]
    fn to_f64(self) -> f64 {
        self as f64
    }
    #[inline]
    fn to_f32(self) -> f32 {
        self
    }

    #[inline]
    fn abs(self) -> Self {
        libm::fabsf(self)
    }
    #[inline]
    fn sqrt(self) -> Self {
        libm::sqrtf(self)
    }
    #[inline]
    fn cbrt(self) -> Self {
        libm::cbrtf(self)
    }
    #[inline]
    fn powi(self, n: i32) -> Self {
        libm::powf(self, n as f32)
    }
    #[inline]
    fn powf(self, n: Self) -> Self {
        libm::powf(self, n)
    }
    #[inline]
    fn hypot(self, other: Self) -> Self {
        libm::hypotf(self, other)
    }

    #[inline]
    fn sin(self) -> Self {
        libm::sinf(self)
    }
    #[inline]
    fn cos(self) -> Self {
        libm::cosf(self)
    }
    #[inline]
    fn tan(self) -> Self {
        libm::tanf(self)
    }
    #[inline]
    fn asin(self) -> Self {
        libm::asinf(self)
    }
    #[inline]
    fn acos(self) -> Self {
        libm::acosf(self)
    }
    #[inline]
    fn atan(self) -> Self {
        libm::atanf(self)
    }
    #[inline]
    fn atan2(self, other: Self) -> Self {
        libm::atan2f(self, other)
    }
    #[inline]
    fn sincos(self) -> (Self, Self) {
        libm::sincosf(self)
    }

    #[inline]
    fn exp(self) -> Self {
        libm::expf(self)
    }
    #[inline]
    fn exp2(self) -> Self {
        libm::exp2f(self)
    }
    #[inline]
    fn exp_m1(self) -> Self {
        libm::expm1f(self)
    }
    #[inline]
    fn ln(self) -> Self {
        libm::logf(self)
    }
    #[inline]
    fn log2(self) -> Self {
        libm::log2f(self)
    }
    #[inline]
    fn log10(self) -> Self {
        libm::log10f(self)
    }
    #[inline]
    fn ln_1p(self) -> Self {
        libm::log1pf(self)
    }

    #[inline]
    fn sinh(self) -> Self {
        libm::sinhf(self)
    }
    #[inline]
    fn cosh(self) -> Self {
        libm::coshf(self)
    }
    #[inline]
    fn tanh(self) -> Self {
        libm::tanhf(self)
    }
    #[inline]
    fn asinh(self) -> Self {
        libm::asinhf(self)
    }
    #[inline]
    fn acosh(self) -> Self {
        libm::acoshf(self)
    }
    #[inline]
    fn atanh(self) -> Self {
        libm::atanhf(self)
    }

    #[inline]
    fn max(self, other: Self) -> Self {
        libm::fmaxf(self, other)
    }
    #[inline]
    fn min(self, other: Self) -> Self {
        libm::fminf(self, other)
    }
    #[inline]
    fn clamp(self, min: Self, max: Self) -> Self {
        libm::fmaxf(min, libm::fminf(self, max))
    }
    #[inline]
    fn copysign(self, sign: Self) -> Self {
        libm::copysignf(self, sign)
    }

    #[inline]
    fn is_finite(self) -> bool {
        self.is_finite()
    }
    #[inline]
    fn is_nan(self) -> bool {
        self.is_nan()
    }
    #[inline]
    fn is_infinite(self) -> bool {
        self.is_infinite()
    }
    #[inline]
    fn is_sign_positive(self) -> bool {
        self.is_sign_positive()
    }
    #[inline]
    fn is_sign_negative(self) -> bool {
        self.is_sign_negative()
    }

    #[inline]
    fn floor(self) -> Self {
        libm::floorf(self)
    }
    #[inline]
    fn ceil(self) -> Self {
        libm::ceilf(self)
    }
    #[inline]
    fn round(self) -> Self {
        libm::roundf(self)
    }
    #[inline]
    fn trunc(self) -> Self {
        libm::truncf(self)
    }
    #[inline]
    fn fract(self) -> Self {
        self - libm::truncf(self)
    }

    #[inline]
    fn gamma_fn(self) -> Self {
        libm::tgammaf(self)
    }
    #[inline]
    fn ln_gamma(self) -> Self {
        libm::lgammaf(self)
    }
    #[inline]
    fn erf_fn(self) -> Self {
        libm::erff(self)
    }
    #[inline]
    fn erfc_fn(self) -> Self {
        libm::erfcf(self)
    }

    #[inline]
    fn mul_add(self, a: Self, b: Self) -> Self {
        libm::fmaf(self, a, b)
    }
}

// ============================================================================
// Batch conversion utilities
// ============================================================================

/// Convert a slice of any Scalar type to a `Vec<f64>`.
pub fn to_f64_vec<S: Scalar>(v: &[S]) -> Vec<f64> {
    v.iter().map(|x| x.to_f64()).collect()
}

/// Convert a slice of f64 to a Vec of any Scalar type.
pub fn from_f64_vec<S: Scalar>(v: &[f64]) -> Vec<S> {
    v.iter().map(|&x| S::from_f64(x)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[allow(clippy::assertions_on_constants)]
    fn test_constants_f64() {
        assert_eq!(f64::ZERO, 0.0);
        assert_eq!(f64::ONE, 1.0);
        assert_eq!(f64::TWO, 2.0);
        assert!(f64::EPSILON > 0.0);
        assert!(f64::EPSILON < 1e-10);
    }

    #[test]
    fn test_basic_ops_f64() {
        let x: f64 = 4.0;
        assert!((x.sqrt() - 2.0).abs() < 1e-10);
        assert!((x.sq() - 16.0).abs() < 1e-10);
    }

    #[test]
    fn test_trig_f64() {
        let x: f64 = f64::PI / 4.0;
        let (s, c) = x.sincos();
        assert!((s - c).abs() < 1e-10); // sin(π/4) = cos(π/4)
    }

    #[test]
    fn test_special_functions_f64() {
        // Gamma(5) = 4! = 24
        let g: f64 = 5.0_f64.gamma_fn();
        assert!((g - 24.0).abs() < 1e-10);

        // erf(0) = 0
        let e: f64 = 0.0_f64.erf_fn();
        assert!(e.abs() < 1e-10);
    }

    #[test]
    fn test_to_f64_vec_identity() {
        let v = vec![1.0_f64, 2.5, 3.7];
        let converted = to_f64_vec(&v);
        assert_eq!(v, converted);
    }

    #[test]
    fn test_from_f64_vec_identity() {
        let v = vec![1.0, 2.5, 3.7];
        let converted: Vec<f64> = from_f64_vec(&v);
        assert_eq!(v, converted);
    }

    #[test]
    fn test_f32_roundtrip() {
        let orig = vec![1.0_f32, 2.5, 3.7];
        let f64_vec = to_f64_vec(&orig);
        let back: Vec<f32> = from_f64_vec(&f64_vec);
        for (a, b) in orig.iter().zip(back.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }
}

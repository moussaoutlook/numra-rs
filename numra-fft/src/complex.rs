//! Minimal complex number type for FFT interfaces.
//!
//! Author: Moussa Leblouba
//! Date: 9 February 2026
//! Modified: 2 May 2026

use numra_core::Scalar;
use std::ops::{Add, Mul, Neg, Sub};

/// A complex number with real and imaginary parts, generic over scalar type.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Complex<S: Scalar> {
    pub re: S,
    pub im: S,
}

/// Type alias for backward compatibility: `Complex<f64>`.
pub type ComplexF64 = Complex<f64>;

impl<S: Scalar> Complex<S> {
    /// Create a new complex number.
    pub fn new(re: S, im: S) -> Self {
        Self { re, im }
    }

    /// Complex zero.
    pub fn zero() -> Self {
        Self {
            re: S::ZERO,
            im: S::ZERO,
        }
    }

    /// Squared magnitude |z|^2 = re^2 + im^2.
    pub fn norm_sqr(&self) -> S {
        self.re * self.re + self.im * self.im
    }

    /// Magnitude |z|.
    pub fn abs(&self) -> S {
        self.norm_sqr().sqrt()
    }

    /// Complex conjugate.
    pub fn conj(&self) -> Self {
        Self {
            re: self.re,
            im: -self.im,
        }
    }

    /// Phase angle (argument) in radians.
    pub fn arg(&self) -> S {
        self.im.atan2(self.re)
    }
}

impl<S: Scalar> Add for Complex<S> {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self {
            re: self.re + rhs.re,
            im: self.im + rhs.im,
        }
    }
}

impl<S: Scalar> Sub for Complex<S> {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self {
            re: self.re - rhs.re,
            im: self.im - rhs.im,
        }
    }
}

impl<S: Scalar> Mul for Complex<S> {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        Self {
            re: self.re * rhs.re - self.im * rhs.im,
            im: self.re * rhs.im + self.im * rhs.re,
        }
    }
}

impl<S: Scalar> Mul<S> for Complex<S> {
    type Output = Self;
    fn mul(self, rhs: S) -> Self {
        Self {
            re: self.re * rhs,
            im: self.im * rhs,
        }
    }
}

impl<S: Scalar> Neg for Complex<S> {
    type Output = Self;
    fn neg(self) -> Self {
        Self {
            re: -self.re,
            im: -self.im,
        }
    }
}

/// Convert from rustfft's Complex type.
impl From<rustfft::num_complex::Complex<f64>> for Complex<f64> {
    fn from(c: rustfft::num_complex::Complex<f64>) -> Self {
        Self { re: c.re, im: c.im }
    }
}

/// Convert to rustfft's Complex type.
impl From<Complex<f64>> for rustfft::num_complex::Complex<f64> {
    fn from(c: Complex<f64>) -> Self {
        Self::new(c.re, c.im)
    }
}

/// Convert from faer's c64 complex type.
impl From<faer::complex_native::c64> for Complex<f64> {
    fn from(c: faer::complex_native::c64) -> Self {
        Self { re: c.re, im: c.im }
    }
}

/// Convert to faer's c64 complex type.
impl From<Complex<f64>> for faer::complex_native::c64 {
    fn from(c: Complex<f64>) -> faer::complex_native::c64 {
        faer::complex_native::c64 { re: c.re, im: c.im }
    }
}

/// Convert from faer's c32 complex type.
impl From<faer::complex_native::c32> for Complex<f32> {
    fn from(c: faer::complex_native::c32) -> Self {
        Self { re: c.re, im: c.im }
    }
}

/// Convert to faer's c32 complex type.
impl From<Complex<f32>> for faer::complex_native::c32 {
    fn from(c: Complex<f32>) -> faer::complex_native::c32 {
        faer::complex_native::c32 { re: c.re, im: c.im }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_ops() {
        let a = Complex::new(1.0, 2.0);
        let b = Complex::new(3.0, 4.0);

        let sum = a + b;
        assert!((sum.re - 4.0).abs() < 1e-14);
        assert!((sum.im - 6.0).abs() < 1e-14);

        let prod = a * b;
        // (1+2i)(3+4i) = 3+4i+6i+8i^2 = -5+10i
        assert!((prod.re - (-5.0)).abs() < 1e-14);
        assert!((prod.im - 10.0).abs() < 1e-14);
    }

    #[test]
    fn test_magnitude() {
        let c = Complex::new(3.0, 4.0);
        assert!((c.abs() - 5.0).abs() < 1e-14);
        assert!((c.norm_sqr() - 25.0).abs() < 1e-14);
    }

    #[test]
    fn test_conjugate() {
        let c = Complex::new(1.0, 2.0);
        let cc = c.conj();
        assert!((cc.re - 1.0).abs() < 1e-14);
        assert!((cc.im - (-2.0)).abs() < 1e-14);
    }

    #[test]
    fn test_faer_c64_roundtrip() {
        let orig = Complex::new(3.14, -2.71);
        let c64_val: faer::complex_native::c64 = orig.into();
        let back: Complex<f64> = c64_val.into();
        assert!((back.re - orig.re).abs() < 1e-14);
        assert!((back.im - orig.im).abs() < 1e-14);
    }

    #[test]
    fn test_faer_c32_roundtrip() {
        let orig: Complex<f32> = Complex::new(1.5_f32, -0.5_f32);
        let c32_val: faer::complex_native::c32 = orig.into();
        let back: Complex<f32> = c32_val.into();
        assert!((back.re - 1.5_f32).abs() < 1e-6);
        assert!((back.im - (-0.5_f32)).abs() < 1e-6);
    }

    #[test]
    fn test_f32_complex_basic() {
        let a: Complex<f32> = Complex::new(1.0_f32, 2.0_f32);
        let b: Complex<f32> = Complex::new(3.0_f32, 4.0_f32);
        let sum = a + b;
        assert!((sum.re - 4.0_f32).abs() < 1e-6);
        assert!((sum.im - 6.0_f32).abs() < 1e-6);
    }
}

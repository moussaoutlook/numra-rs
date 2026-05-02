//! Dual number type for forward-mode automatic differentiation.
//!
//! A dual number `Dual<S>` carries a value and its derivative (the "epsilon"
//! or tangent part). All arithmetic and transcendental operations propagate
//! derivatives via the chain rule.
//!
//! Author: Moussa Leblouba
//! Date: 7 February 2026
//! Modified: 2 May 2026

use core::fmt;
use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};
use numra_core::Scalar;

/// A dual number for forward-mode automatic differentiation.
///
/// `Dual<S>` represents a value `val` together with its derivative `eps`
/// with respect to some variable of interest. When composed through
/// arithmetic and transcendental operations, the derivative is propagated
/// automatically via the chain rule.
///
/// # Examples
///
/// ```rust
/// use numra_autodiff::Dual;
///
/// let x = Dual::variable(2.0_f64); // x = 2, dx/dx = 1
/// let y = x * x;                    // y = 4, dy/dx = 2*x = 4
/// assert!((y.value() - 4.0).abs() < 1e-12);
/// assert!((y.deriv() - 4.0).abs() < 1e-12);
/// ```
#[derive(Copy, Clone)]
pub struct Dual<S: Scalar> {
    /// The primal value.
    val: S,
    /// The derivative (tangent/epsilon) part.
    eps: S,
}

impl<S: Scalar> Dual<S> {
    /// Create a new dual number with given value and derivative.
    #[inline]
    pub fn new(value: S, derivative: S) -> Self {
        Self {
            val: value,
            eps: derivative,
        }
    }

    /// Create a constant dual number (derivative is zero).
    #[inline]
    pub fn constant(value: S) -> Self {
        Self {
            val: value,
            eps: S::ZERO,
        }
    }

    /// Create a variable dual number (derivative is one).
    ///
    /// Use this when differentiating with respect to this variable.
    #[inline]
    pub fn variable(value: S) -> Self {
        Self {
            val: value,
            eps: S::ONE,
        }
    }

    /// Returns the primal value.
    #[inline]
    pub fn value(&self) -> S {
        self.val
    }

    /// Returns the derivative (tangent/epsilon) part.
    #[inline]
    pub fn deriv(&self) -> S {
        self.eps
    }

    // ===== Transcendental functions (chain rule applied) =====

    /// Sine: sin(x) -> (sin(val), eps * cos(val))
    #[inline]
    pub fn sin(self) -> Self {
        Self {
            val: self.val.sin(),
            eps: self.eps * self.val.cos(),
        }
    }

    /// Cosine: cos(x) -> (cos(val), -eps * sin(val))
    #[inline]
    pub fn cos(self) -> Self {
        Self {
            val: self.val.cos(),
            eps: -self.eps * self.val.sin(),
        }
    }

    /// Tangent: tan(x) -> (tan(val), eps / cos(val)^2)
    #[inline]
    pub fn tan(self) -> Self {
        let c = self.val.cos();
        Self {
            val: self.val.tan(),
            eps: self.eps / (c * c),
        }
    }

    /// Natural exponential: exp(x) -> (exp(val), eps * exp(val))
    #[inline]
    pub fn exp(self) -> Self {
        let e = self.val.exp();
        Self {
            val: e,
            eps: self.eps * e,
        }
    }

    /// Natural logarithm: ln(x) -> (ln(val), eps / val)
    #[inline]
    pub fn ln(self) -> Self {
        Self {
            val: self.val.ln(),
            eps: self.eps / self.val,
        }
    }

    /// Square root: sqrt(x) -> (sqrt(val), eps / (2 * sqrt(val)))
    #[inline]
    pub fn sqrt(self) -> Self {
        let s = self.val.sqrt();
        Self {
            val: s,
            eps: self.eps / (S::TWO * s),
        }
    }

    /// Absolute value: abs(x) -> (abs(val), eps * signum(val))
    #[inline]
    pub fn abs(self) -> Self {
        Self {
            val: self.val.abs(),
            eps: self.eps * self.val.signum(),
        }
    }

    /// Floating-point power: x^n -> (val^n, eps * n * val^(n-1))
    #[inline]
    pub fn powf(self, n: S) -> Self {
        Self {
            val: self.val.powf(n),
            eps: self.eps * n * self.val.powf(n - S::ONE),
        }
    }

    /// Power where both base and exponent are dual: x^y.
    ///
    /// d/dt(x^y) = x^y * (y_eps * ln(x_val) + y_val * x_eps / x_val)
    ///
    /// Requires x_val > 0 for ln(x_val) to be defined.
    #[inline]
    pub fn powf_dual(self, n: Self) -> Self {
        let val = self.val.powf(n.val);
        let eps = val * (n.eps * self.val.ln() + n.val * self.eps / self.val);
        Self { val, eps }
    }

    /// Integer power: x^n -> (val^n, eps * n * val^(n-1))
    #[inline]
    pub fn powi(self, n: i32) -> Self {
        let nf = S::from_i32(n);
        Self {
            val: self.val.powi(n),
            eps: self.eps * nf * self.val.powi(n - 1),
        }
    }

    /// Arcsine: asin(x) -> (asin(val), eps / sqrt(1 - val^2))
    #[inline]
    pub fn asin(self) -> Self {
        Self {
            val: self.val.asin(),
            eps: self.eps / (S::ONE - self.val * self.val).sqrt(),
        }
    }

    /// Arccosine: acos(x) -> (acos(val), -eps / sqrt(1 - val^2))
    #[inline]
    pub fn acos(self) -> Self {
        Self {
            val: self.val.acos(),
            eps: -self.eps / (S::ONE - self.val * self.val).sqrt(),
        }
    }

    /// Arctangent: atan(x) -> (atan(val), eps / (1 + val^2))
    #[inline]
    pub fn atan(self) -> Self {
        Self {
            val: self.val.atan(),
            eps: self.eps / (S::ONE + self.val * self.val),
        }
    }

    /// Hyperbolic sine: sinh(x) -> (sinh(val), eps * cosh(val))
    #[inline]
    pub fn sinh(self) -> Self {
        Self {
            val: self.val.sinh(),
            eps: self.eps * self.val.cosh(),
        }
    }

    /// Hyperbolic cosine: cosh(x) -> (cosh(val), eps * sinh(val))
    #[inline]
    pub fn cosh(self) -> Self {
        Self {
            val: self.val.cosh(),
            eps: self.eps * self.val.sinh(),
        }
    }

    /// Hyperbolic tangent: tanh(x) -> (tanh(val), eps * (1 - tanh(val)^2))
    #[inline]
    pub fn tanh(self) -> Self {
        let t = self.val.tanh();
        Self {
            val: t,
            eps: self.eps * (S::ONE - t * t),
        }
    }
}

// =============================================================================
// Arithmetic operators
// =============================================================================

impl<S: Scalar> Add for Dual<S> {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self {
            val: self.val + rhs.val,
            eps: self.eps + rhs.eps,
        }
    }
}

impl<S: Scalar> Sub for Dual<S> {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self {
            val: self.val - rhs.val,
            eps: self.eps - rhs.eps,
        }
    }
}

impl<S: Scalar> Mul for Dual<S> {
    type Output = Self;

    /// Product rule: (a + a'e)(b + b'e) = ab + (a'b + ab')e
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        Self {
            val: self.val * rhs.val,
            eps: self.eps * rhs.val + self.val * rhs.eps,
        }
    }
}

impl<S: Scalar> Div for Dual<S> {
    type Output = Self;

    /// Quotient rule: (a + a'e) / (b + b'e) = a/b + (a'b - ab')/(b^2) e
    #[inline]
    fn div(self, rhs: Self) -> Self {
        Self {
            val: self.val / rhs.val,
            eps: (self.eps * rhs.val - self.val * rhs.eps) / (rhs.val * rhs.val),
        }
    }
}

impl<S: Scalar> Neg for Dual<S> {
    type Output = Self;

    #[inline]
    fn neg(self) -> Self {
        Self {
            val: -self.val,
            eps: -self.eps,
        }
    }
}

// =============================================================================
// Compound assignment operators
// =============================================================================

impl<S: Scalar> AddAssign for Dual<S> {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl<S: Scalar> SubAssign for Dual<S> {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl<S: Scalar> MulAssign for Dual<S> {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl<S: Scalar> DivAssign for Dual<S> {
    #[inline]
    fn div_assign(&mut self, rhs: Self) {
        *self = *self / rhs;
    }
}

// =============================================================================
// Display and Debug
// =============================================================================

impl<S: Scalar> fmt::Debug for Dual<S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Dual")
            .field("val", &self.val)
            .field("eps", &self.eps)
            .finish()
    }
}

impl<S: Scalar> fmt::Display for Dual<S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} + {}e", self.val, self.eps)
    }
}

// =============================================================================
// Comparison (by value only)
// =============================================================================

impl<S: Scalar> PartialEq for Dual<S> {
    /// Two dual numbers are equal if their primal values are equal.
    /// The derivative part is not compared.
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.val == other.val
    }
}

impl<S: Scalar> PartialOrd for Dual<S> {
    /// Dual numbers are ordered by their primal values only.
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        self.val.partial_cmp(&other.val)
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1e-12;

    #[test]
    fn test_dual_arithmetic() {
        // Addition: (2 + 1e) + (3 + 4e) = (5 + 5e)
        let a = Dual::new(2.0_f64, 1.0);
        let b = Dual::new(3.0, 4.0);
        let sum = a + b;
        assert!((sum.value() - 5.0).abs() < TOL);
        assert!((sum.deriv() - 5.0).abs() < TOL);

        // Multiplication: (2 + 1e) * (3 + 4e) = (6 + (1*3 + 2*4)e) = (6 + 11e)
        let prod = a * b;
        assert!((prod.value() - 6.0).abs() < TOL);
        assert!((prod.deriv() - 11.0).abs() < TOL);

        // Division: (6 + 1e) / (3 + 0e) = (2 + 1/3 e)
        let c = Dual::new(6.0, 1.0);
        let d = Dual::constant(3.0);
        let quot = c / d;
        assert!((quot.value() - 2.0).abs() < TOL);
        assert!((quot.deriv() - 1.0 / 3.0).abs() < TOL);
    }

    #[test]
    fn test_dual_transcendental() {
        // sin(0) = 0, d/dx sin(x)|_{x=0} = cos(0) = 1
        let x = Dual::variable(0.0_f64);
        let s = x.sin();
        assert!((s.value() - 0.0).abs() < TOL);
        assert!((s.deriv() - 1.0).abs() < TOL);

        // exp(0) = 1, d/dx exp(x)|_{x=0} = exp(0) = 1
        let e = x.exp();
        assert!((e.value() - 1.0).abs() < TOL);
        assert!((e.deriv() - 1.0).abs() < TOL);
    }

    #[test]
    fn test_dual_chain_rule() {
        // f(x) = sin(x^2), f'(x) = 2x * cos(x^2)
        // At x = 1: f(1) = sin(1), f'(1) = 2 * cos(1)
        let x = Dual::variable(1.0_f64);
        let y = (x * x).sin();
        let expected_val = 1.0_f64.sin();
        let expected_deriv = 2.0 * 1.0_f64.cos();
        assert!((y.value() - expected_val).abs() < TOL);
        assert!((y.deriv() - expected_deriv).abs() < TOL);
    }

    #[test]
    fn test_dual_negation() {
        let x = Dual::new(3.0_f64, 2.0);
        let neg = -x;
        assert!((neg.value() - (-3.0)).abs() < TOL);
        assert!((neg.deriv() - (-2.0)).abs() < TOL);
    }

    #[test]
    fn test_dual_constant() {
        let c = Dual::<f64>::constant(5.0);
        assert!((c.value() - 5.0).abs() < TOL);
        assert!((c.deriv() - 0.0).abs() < TOL);

        // Operations with constants should not introduce spurious derivatives.
        let x = Dual::variable(3.0_f64);
        let y = x + c;
        assert!((y.value() - 8.0).abs() < TOL);
        assert!((y.deriv() - 1.0).abs() < TOL); // d/dx (x + 5) = 1
    }

    #[test]
    fn test_dual_powf() {
        // f(x) = x^2.5, f'(x) = 2.5 * x^1.5
        // At x = 4: f(4) = 4^2.5 = 32, f'(4) = 2.5 * 4^1.5 = 2.5 * 8 = 20
        let x = Dual::variable(4.0_f64);
        let y = x.powf(2.5);
        assert!((y.value() - 32.0).abs() < 1e-10);
        assert!((y.deriv() - 20.0).abs() < 1e-10);
    }

    #[test]
    fn test_dual_sqrt() {
        // f(x) = sqrt(x), f'(x) = 1 / (2 * sqrt(x))
        // At x = 4: f(4) = 2, f'(4) = 1/4 = 0.25
        let x = Dual::variable(4.0_f64);
        let y = x.sqrt();
        assert!((y.value() - 2.0).abs() < TOL);
        assert!((y.deriv() - 0.25).abs() < TOL);
    }

    #[test]
    fn test_dual_ln() {
        // f(x) = ln(x), f'(x) = 1/x
        // At x = 2: f(2) = ln(2), f'(2) = 0.5
        let x = Dual::variable(2.0_f64);
        let y = x.ln();
        assert!((y.value() - 2.0_f64.ln()).abs() < TOL);
        assert!((y.deriv() - 0.5).abs() < TOL);
    }

    #[test]
    fn test_dual_complex_expression() {
        // f(x) = exp(-x^2 / 2) (Gaussian kernel)
        // f'(x) = -x * exp(-x^2 / 2)
        // At x = 1: f(1) = exp(-0.5), f'(1) = -exp(-0.5)
        let x = Dual::variable(1.0_f64);
        let half = Dual::constant(0.5_f64);
        let y = (-(x * x) * half).exp();
        let expected_val = (-0.5_f64).exp();
        let expected_deriv = -(-0.5_f64).exp();
        assert!((y.value() - expected_val).abs() < TOL);
        assert!((y.deriv() - expected_deriv).abs() < TOL);
    }

    #[test]
    fn test_dual_powi() {
        // f(x) = x^3, f'(x) = 3x^2
        // At x = 2: f(2) = 8, f'(2) = 12
        let x = Dual::variable(2.0_f64);
        let y = x.powi(3);
        assert!((y.value() - 8.0).abs() < TOL);
        assert!((y.deriv() - 12.0).abs() < TOL);
    }

    #[test]
    fn test_dual_cos() {
        // f(x) = cos(x), f'(x) = -sin(x)
        // At x = pi/3: f = 0.5, f' = -sin(pi/3) = -sqrt(3)/2
        let x = Dual::variable(core::f64::consts::PI / 3.0);
        let y = x.cos();
        assert!((y.value() - 0.5).abs() < 1e-10);
        assert!((y.deriv() - (-(core::f64::consts::PI / 3.0).sin())).abs() < 1e-10);
    }

    #[test]
    fn test_dual_tan() {
        // f(x) = tan(x), f'(x) = 1/cos^2(x) = sec^2(x)
        // At x = pi/4: f = 1, f' = 2
        let x = Dual::variable(core::f64::consts::PI / 4.0);
        let y = x.tan();
        assert!((y.value() - 1.0).abs() < 1e-10);
        assert!((y.deriv() - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_dual_display_debug() {
        let x = Dual::new(1.0_f64, 2.0);
        let display = format!("{}", x);
        assert!(display.contains("1"));
        assert!(display.contains("2"));

        let debug = format!("{:?}", x);
        assert!(debug.contains("Dual"));
    }

    #[test]
    fn test_dual_partial_eq_ord() {
        let a = Dual::new(2.0_f64, 1.0);
        let b = Dual::new(2.0, 99.0); // same value, different derivative
        assert_eq!(a, b);

        let c = Dual::new(3.0_f64, 0.0);
        assert!(a < c);
        assert!(c > a);
    }

    #[test]
    fn test_dual_compound_assignment() {
        let mut x = Dual::new(2.0_f64, 1.0);
        let y = Dual::new(3.0, 4.0);

        x += y;
        assert!((x.value() - 5.0).abs() < TOL);
        assert!((x.deriv() - 5.0).abs() < TOL);

        x -= Dual::new(1.0, 1.0);
        assert!((x.value() - 4.0).abs() < TOL);
        assert!((x.deriv() - 4.0).abs() < TOL);

        x *= Dual::constant(2.0);
        assert!((x.value() - 8.0).abs() < TOL);
        assert!((x.deriv() - 8.0).abs() < TOL);

        x /= Dual::constant(4.0);
        assert!((x.value() - 2.0).abs() < TOL);
        assert!((x.deriv() - 2.0).abs() < TOL);
    }

    #[test]
    fn test_dual_asin_acos_atan() {
        // asin: d/dx asin(x) = 1/sqrt(1 - x^2)
        // At x = 0.5: f' = 1/sqrt(0.75) = 2/sqrt(3)
        let x = Dual::variable(0.5_f64);
        let y = x.asin();
        assert!((y.value() - 0.5_f64.asin()).abs() < TOL);
        assert!((y.deriv() - 1.0 / (0.75_f64).sqrt()).abs() < 1e-10);

        // acos: d/dx acos(x) = -1/sqrt(1 - x^2)
        let y2 = x.acos();
        assert!((y2.value() - 0.5_f64.acos()).abs() < TOL);
        assert!((y2.deriv() - (-1.0 / (0.75_f64).sqrt())).abs() < 1e-10);

        // atan: d/dx atan(x) = 1/(1 + x^2)
        // At x = 1: f' = 1/2 = 0.5
        let z = Dual::variable(1.0_f64);
        let w = z.atan();
        assert!((w.value() - 1.0_f64.atan()).abs() < TOL);
        assert!((w.deriv() - 0.5).abs() < TOL);
    }

    #[test]
    fn test_dual_sinh_cosh_tanh() {
        // sinh: d/dx sinh(x) = cosh(x)
        let x = Dual::variable(1.0_f64);
        let y = x.sinh();
        assert!((y.value() - 1.0_f64.sinh()).abs() < TOL);
        assert!((y.deriv() - 1.0_f64.cosh()).abs() < TOL);

        // cosh: d/dx cosh(x) = sinh(x)
        let y2 = x.cosh();
        assert!((y2.value() - 1.0_f64.cosh()).abs() < TOL);
        assert!((y2.deriv() - 1.0_f64.sinh()).abs() < TOL);

        // tanh: d/dx tanh(x) = 1 - tanh(x)^2
        let y3 = x.tanh();
        let t = 1.0_f64.tanh();
        assert!((y3.value() - t).abs() < TOL);
        assert!((y3.deriv() - (1.0 - t * t)).abs() < TOL);
    }

    #[test]
    fn test_dual_abs() {
        // abs(x) at x = -3: value = 3, deriv = -1
        let x = Dual::variable(-3.0_f64);
        let y = x.abs();
        assert!((y.value() - 3.0).abs() < TOL);
        assert!((y.deriv() - (-1.0)).abs() < TOL);

        // abs(x) at x = 5: value = 5, deriv = 1
        let x2 = Dual::variable(5.0_f64);
        let y2 = x2.abs();
        assert!((y2.value() - 5.0).abs() < TOL);
        assert!((y2.deriv() - 1.0).abs() < TOL);
    }

    #[test]
    fn test_dual_powf_dual() {
        // f(x, y) = x^y at (x, y) = (2, 3)
        // df/dx = y * x^(y-1) = 3 * 4 = 12
        // df/dy = x^y * ln(x) = 8 * ln(2)

        // Differentiate with respect to x (x is variable, y is constant)
        let x = Dual::variable(2.0_f64);
        let y = Dual::constant(3.0_f64);
        let z = x.powf_dual(y);
        assert!((z.value() - 8.0).abs() < 1e-10);
        assert!((z.deriv() - 12.0).abs() < 1e-10);

        // Differentiate with respect to y (x is constant, y is variable)
        let x = Dual::constant(2.0_f64);
        let y = Dual::variable(3.0_f64);
        let z = x.powf_dual(y);
        assert!((z.value() - 8.0).abs() < 1e-10);
        assert!((z.deriv() - 8.0 * 2.0_f64.ln()).abs() < 1e-10);
    }

    #[test]
    fn test_dual_powf_dual_both_variable() {
        // f(t) = (2t)^(t) at t=1 => f(1) = 2^1 = 2
        // f'(t) = d/dt[(2t)^t] = (2t)^t * [ln(2t) + 1]
        // At t=1: f'(1) = 2 * [ln(2) + 1] = 2*ln(2) + 2
        let t = Dual::variable(1.0_f64);
        let base = Dual::constant(2.0) * t; // 2t
        let z = base.powf_dual(t);
        assert!((z.value() - 2.0).abs() < 1e-10);
        let expected = 2.0 * (2.0_f64.ln() + 1.0);
        assert!((z.deriv() - expected).abs() < 1e-10);
    }
}

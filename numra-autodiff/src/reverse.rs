//! Reverse-mode automatic differentiation via tape-based computation graph.
//!
//! This module provides [`Var`], a tracked variable type that records operations
//! on a shared [`Tape`]. After building the computation graph in the forward pass,
//! call [`Tape::gradient`] to compute derivatives in a single backward pass.
//!
//! # Comparison with forward-mode
//!
//! - **Forward-mode** ([`Dual`](crate::Dual)): One directional derivative per pass.
//!   Cost: O(n) passes for n inputs. Best for few inputs, many outputs.
//! - **Reverse-mode** ([`Var`]): All gradients in one backward pass.
//!   Cost: O(m) passes for m outputs. Best for many inputs, few outputs (optimization).
//!
//! # Example
//!
//! ```rust
//! use numra_autodiff::reverse::{grad, hessian};
//!
//! // Gradient of Rosenbrock: f(x,y) = (1-x)^2 + 100*(y-x^2)^2
//! let g = grad(
//!     |x| {
//!         let a = x[0].cst(1.0) - x[0].clone();
//!         let b = x[1].clone() - x[0].clone() * x[0].clone();
//!         a.clone() * a + x[0].cst(100.0) * b.clone() * b
//!     },
//!     &[1.0, 1.0],
//! );
//! // At (1,1), gradient should be (0, 0)
//! assert!(g[0].abs() < 1e-10);
//! assert!(g[1].abs() < 1e-10);
//! ```
//!
//! Author: Moussa Leblouba
//! Date: 9 February 2026
//! Modified: 2 May 2026

use crate::tape::{Tape, TapeRef};
use std::ops::{Add, Div, Mul, Neg, Sub};
use std::rc::Rc;

/// A reverse-mode AD variable tracked on a computation tape.
///
/// Arithmetic operations on `Var` automatically record themselves on the
/// shared tape. After computation, use [`Tape::gradient`] to differentiate.
#[derive(Clone, Debug)]
pub struct Var {
    /// Index of this variable's node in the tape.
    pub(crate) index: usize,
    /// Primal (forward) value.
    pub value: f64,
    /// Reference to the shared tape.
    pub(crate) tape: TapeRef,
}

impl Var {
    /// Create a constant (not differentiated) on the same tape as `self`.
    pub fn cst(&self, value: f64) -> Var {
        let (index, value) = {
            let mut t = self.tape.borrow_mut();
            let idx = t.nodes.len();
            t.nodes.push(crate::tape::Node {
                value,
                parent1: None,
                parent2: None,
            });
            (idx, value)
        };
        Var {
            index,
            value,
            tape: Rc::clone(&self.tape),
        }
    }

    /// sin(self)
    pub fn sin(&self) -> Var {
        let val = self.value.sin();
        let deriv = self.value.cos(); // d sin(x)/dx = cos(x)
        let (index, value) = Tape::push_unary(&self.tape, val, self.index, deriv);
        Var {
            index,
            value,
            tape: Rc::clone(&self.tape),
        }
    }

    /// cos(self)
    pub fn cos(&self) -> Var {
        let val = self.value.cos();
        let deriv = -self.value.sin(); // d cos(x)/dx = -sin(x)
        let (index, value) = Tape::push_unary(&self.tape, val, self.index, deriv);
        Var {
            index,
            value,
            tape: Rc::clone(&self.tape),
        }
    }

    /// tan(self)
    pub fn tan(&self) -> Var {
        let val = self.value.tan();
        let c = self.value.cos();
        let deriv = 1.0 / (c * c); // d tan(x)/dx = sec^2(x)
        let (index, value) = Tape::push_unary(&self.tape, val, self.index, deriv);
        Var {
            index,
            value,
            tape: Rc::clone(&self.tape),
        }
    }

    /// exp(self)
    pub fn exp(&self) -> Var {
        let val = self.value.exp();
        let deriv = val; // d exp(x)/dx = exp(x)
        let (index, value) = Tape::push_unary(&self.tape, val, self.index, deriv);
        Var {
            index,
            value,
            tape: Rc::clone(&self.tape),
        }
    }

    /// ln(self) (natural logarithm)
    pub fn ln(&self) -> Var {
        let val = self.value.ln();
        let deriv = 1.0 / self.value; // d ln(x)/dx = 1/x
        let (index, value) = Tape::push_unary(&self.tape, val, self.index, deriv);
        Var {
            index,
            value,
            tape: Rc::clone(&self.tape),
        }
    }

    /// sqrt(self)
    pub fn sqrt(&self) -> Var {
        let val = self.value.sqrt();
        let deriv = 0.5 / val; // d sqrt(x)/dx = 1/(2*sqrt(x))
        let (index, value) = Tape::push_unary(&self.tape, val, self.index, deriv);
        Var {
            index,
            value,
            tape: Rc::clone(&self.tape),
        }
    }

    /// abs(self)
    pub fn abs(&self) -> Var {
        let val = self.value.abs();
        let deriv = if self.value >= 0.0 { 1.0 } else { -1.0 };
        let (index, value) = Tape::push_unary(&self.tape, val, self.index, deriv);
        Var {
            index,
            value,
            tape: Rc::clone(&self.tape),
        }
    }

    /// tanh(self)
    pub fn tanh(&self) -> Var {
        let val = self.value.tanh();
        let deriv = 1.0 - val * val; // d tanh(x)/dx = 1 - tanh^2(x)
        let (index, value) = Tape::push_unary(&self.tape, val, self.index, deriv);
        Var {
            index,
            value,
            tape: Rc::clone(&self.tape),
        }
    }

    /// sinh(self)
    pub fn sinh(&self) -> Var {
        let val = self.value.sinh();
        let deriv = self.value.cosh();
        let (index, value) = Tape::push_unary(&self.tape, val, self.index, deriv);
        Var {
            index,
            value,
            tape: Rc::clone(&self.tape),
        }
    }

    /// cosh(self)
    pub fn cosh(&self) -> Var {
        let val = self.value.cosh();
        let deriv = self.value.sinh();
        let (index, value) = Tape::push_unary(&self.tape, val, self.index, deriv);
        Var {
            index,
            value,
            tape: Rc::clone(&self.tape),
        }
    }

    /// self^n (power with another Var)
    pub fn pow(&self, n: &Var) -> Var {
        let val = self.value.powf(n.value);
        // d(x^y)/dx = y * x^(y-1)
        let d_self = n.value * self.value.powf(n.value - 1.0);
        // d(x^y)/dy = x^y * ln(x)
        let d_n = val * self.value.ln();
        let (index, value) = Tape::push_binary(&self.tape, val, self.index, d_self, n.index, d_n);
        Var {
            index,
            value,
            tape: Rc::clone(&self.tape),
        }
    }

    /// self^n (power with f64 constant)
    pub fn powf(&self, n: f64) -> Var {
        let val = self.value.powf(n);
        let deriv = n * self.value.powf(n - 1.0);
        let (index, value) = Tape::push_unary(&self.tape, val, self.index, deriv);
        Var {
            index,
            value,
            tape: Rc::clone(&self.tape),
        }
    }

    /// self^n (integer power)
    pub fn powi(&self, n: i32) -> Var {
        self.powf(n as f64)
    }

    /// asin(self)
    pub fn asin(&self) -> Var {
        let val = self.value.asin();
        let deriv = 1.0 / (1.0 - self.value * self.value).sqrt();
        let (index, value) = Tape::push_unary(&self.tape, val, self.index, deriv);
        Var {
            index,
            value,
            tape: Rc::clone(&self.tape),
        }
    }

    /// acos(self)
    pub fn acos(&self) -> Var {
        let val = self.value.acos();
        let deriv = -1.0 / (1.0 - self.value * self.value).sqrt();
        let (index, value) = Tape::push_unary(&self.tape, val, self.index, deriv);
        Var {
            index,
            value,
            tape: Rc::clone(&self.tape),
        }
    }

    /// atan(self)
    pub fn atan(&self) -> Var {
        let val = self.value.atan();
        let deriv = 1.0 / (1.0 + self.value * self.value);
        let (index, value) = Tape::push_unary(&self.tape, val, self.index, deriv);
        Var {
            index,
            value,
            tape: Rc::clone(&self.tape),
        }
    }
}

// ==================== Operator overloading ====================

// Var + Var
impl Add for Var {
    type Output = Var;
    fn add(self, rhs: Var) -> Var {
        let val = self.value + rhs.value;
        let (index, value) = Tape::push_binary(&self.tape, val, self.index, 1.0, rhs.index, 1.0);
        Var {
            index,
            value,
            tape: self.tape,
        }
    }
}

// &Var + &Var
impl Add for &Var {
    type Output = Var;
    fn add(self, rhs: &Var) -> Var {
        let val = self.value + rhs.value;
        let (index, value) = Tape::push_binary(&self.tape, val, self.index, 1.0, rhs.index, 1.0);
        Var {
            index,
            value,
            tape: Rc::clone(&self.tape),
        }
    }
}

// Var + f64
impl Add<f64> for Var {
    type Output = Var;
    fn add(self, rhs: f64) -> Var {
        let val = self.value + rhs;
        let (index, value) = Tape::push_unary(&self.tape, val, self.index, 1.0);
        Var {
            index,
            value,
            tape: self.tape,
        }
    }
}

// f64 + Var
impl Add<Var> for f64 {
    type Output = Var;
    fn add(self, rhs: Var) -> Var {
        let val = self + rhs.value;
        let (index, value) = Tape::push_unary(&rhs.tape, val, rhs.index, 1.0);
        Var {
            index,
            value,
            tape: rhs.tape,
        }
    }
}

// Var - Var
impl Sub for Var {
    type Output = Var;
    fn sub(self, rhs: Var) -> Var {
        let val = self.value - rhs.value;
        let (index, value) = Tape::push_binary(&self.tape, val, self.index, 1.0, rhs.index, -1.0);
        Var {
            index,
            value,
            tape: self.tape,
        }
    }
}

// &Var - &Var
impl Sub for &Var {
    type Output = Var;
    fn sub(self, rhs: &Var) -> Var {
        let val = self.value - rhs.value;
        let (index, value) = Tape::push_binary(&self.tape, val, self.index, 1.0, rhs.index, -1.0);
        Var {
            index,
            value,
            tape: Rc::clone(&self.tape),
        }
    }
}

// Var - f64
impl Sub<f64> for Var {
    type Output = Var;
    fn sub(self, rhs: f64) -> Var {
        let val = self.value - rhs;
        let (index, value) = Tape::push_unary(&self.tape, val, self.index, 1.0);
        Var {
            index,
            value,
            tape: self.tape,
        }
    }
}

// f64 - Var
impl Sub<Var> for f64 {
    type Output = Var;
    fn sub(self, rhs: Var) -> Var {
        let val = self - rhs.value;
        let (index, value) = Tape::push_unary(&rhs.tape, val, rhs.index, -1.0);
        Var {
            index,
            value,
            tape: rhs.tape,
        }
    }
}

// Var * Var
impl Mul for Var {
    type Output = Var;
    fn mul(self, rhs: Var) -> Var {
        let val = self.value * rhs.value;
        let (index, value) = Tape::push_binary(
            &self.tape, val, self.index, rhs.value, // d(x*y)/dx = y
            rhs.index, self.value, // d(x*y)/dy = x
        );
        Var {
            index,
            value,
            tape: self.tape,
        }
    }
}

// &Var * &Var
impl Mul for &Var {
    type Output = Var;
    fn mul(self, rhs: &Var) -> Var {
        let val = self.value * rhs.value;
        let (index, value) = Tape::push_binary(
            &self.tape, val, self.index, rhs.value, rhs.index, self.value,
        );
        Var {
            index,
            value,
            tape: Rc::clone(&self.tape),
        }
    }
}

// Var * f64
impl Mul<f64> for Var {
    type Output = Var;
    fn mul(self, rhs: f64) -> Var {
        let val = self.value * rhs;
        let (index, value) = Tape::push_unary(&self.tape, val, self.index, rhs);
        Var {
            index,
            value,
            tape: self.tape,
        }
    }
}

// f64 * Var
impl Mul<Var> for f64 {
    type Output = Var;
    fn mul(self, rhs: Var) -> Var {
        let val = self * rhs.value;
        let (index, value) = Tape::push_unary(&rhs.tape, val, rhs.index, self);
        Var {
            index,
            value,
            tape: rhs.tape,
        }
    }
}

// Var * &Var
impl Mul<&Var> for Var {
    type Output = Var;
    fn mul(self, rhs: &Var) -> Var {
        let val = self.value * rhs.value;
        let (index, value) = Tape::push_binary(
            &self.tape, val, self.index, rhs.value, rhs.index, self.value,
        );
        Var {
            index,
            value,
            tape: self.tape,
        }
    }
}

// &Var * Var
impl Mul<Var> for &Var {
    type Output = Var;
    fn mul(self, rhs: Var) -> Var {
        let val = self.value * rhs.value;
        let (index, value) = Tape::push_binary(
            &self.tape, val, self.index, rhs.value, rhs.index, self.value,
        );
        Var {
            index,
            value,
            tape: rhs.tape,
        }
    }
}

// Var / Var
impl Div for Var {
    type Output = Var;
    fn div(self, rhs: Var) -> Var {
        let val = self.value / rhs.value;
        let (index, value) = Tape::push_binary(
            &self.tape,
            val,
            self.index,
            1.0 / rhs.value, // d(x/y)/dx = 1/y
            rhs.index,
            -self.value / (rhs.value * rhs.value), // d(x/y)/dy = -x/y^2
        );
        Var {
            index,
            value,
            tape: self.tape,
        }
    }
}

// &Var / &Var
impl Div for &Var {
    type Output = Var;
    fn div(self, rhs: &Var) -> Var {
        let val = self.value / rhs.value;
        let (index, value) = Tape::push_binary(
            &self.tape,
            val,
            self.index,
            1.0 / rhs.value,
            rhs.index,
            -self.value / (rhs.value * rhs.value),
        );
        Var {
            index,
            value,
            tape: Rc::clone(&self.tape),
        }
    }
}

// Var / f64
impl Div<f64> for Var {
    type Output = Var;
    fn div(self, rhs: f64) -> Var {
        let val = self.value / rhs;
        let (index, value) = Tape::push_unary(&self.tape, val, self.index, 1.0 / rhs);
        Var {
            index,
            value,
            tape: self.tape,
        }
    }
}

// -Var
impl Neg for Var {
    type Output = Var;
    fn neg(self) -> Var {
        let val = -self.value;
        let (index, value) = Tape::push_unary(&self.tape, val, self.index, -1.0);
        Var {
            index,
            value,
            tape: self.tape,
        }
    }
}

// -&Var
impl Neg for &Var {
    type Output = Var;
    fn neg(self) -> Var {
        let val = -self.value;
        let (index, value) = Tape::push_unary(&self.tape, val, self.index, -1.0);
        Var {
            index,
            value,
            tape: Rc::clone(&self.tape),
        }
    }
}

// ==================== Convenience functions ====================

/// Compute the gradient of a scalar function f: R^n -> R using reverse-mode AD.
///
/// This is more efficient than forward-mode [`gradient`](crate::gradient()) when n is large,
/// as it requires only a single backward pass regardless of n.
///
/// # Example
///
/// ```rust
/// use numra_autodiff::reverse::grad;
///
/// let g = grad(|x| x[0].clone() * x[0].clone() + x[1].clone() * x[1].clone(), &[3.0, 4.0]);
/// assert!((g[0] - 6.0).abs() < 1e-12);
/// assert!((g[1] - 8.0).abs() < 1e-12);
/// ```
pub fn grad<F>(f: F, x: &[f64]) -> Vec<f64>
where
    F: Fn(&[Var]) -> Var,
{
    let tape = Tape::new();
    let vars: Vec<Var> = x.iter().map(|&xi| Tape::var(&tape, xi)).collect();
    let output = f(&vars);
    Tape::gradient(&tape, &output)
}

/// Compute the Jacobian of a vector function f: R^n -> R^m using reverse-mode AD.
///
/// Returns an m x n matrix (Vec of Vec), where row i is the gradient of output i.
///
/// # Example
///
/// ```rust
/// use numra_autodiff::reverse::jacobian_reverse;
///
/// // f(x,y) = (x+y, x*y)
/// let jac = jacobian_reverse(
///     |x| vec![&x[0] + &x[1], &x[0] * &x[1]],
///     &[2.0, 3.0],
/// );
/// assert!((jac[0][0] - 1.0).abs() < 1e-12); // df1/dx
/// assert!((jac[1][0] - 3.0).abs() < 1e-12); // df2/dx = y
/// ```
pub fn jacobian_reverse<F>(f: F, x: &[f64]) -> Vec<Vec<f64>>
where
    F: Fn(&[Var]) -> Vec<Var>,
{
    let tape = Tape::new();
    let vars: Vec<Var> = x.iter().map(|&xi| Tape::var(&tape, xi)).collect();
    let outputs = f(&vars);
    Tape::jacobian(&tape, &outputs)
}

/// Compute the Hessian of a scalar function f: R^n -> R.
///
/// Uses finite-difference of reverse-mode gradients. For each input variable,
/// perturbs slightly and recomputes the gradient to get the Hessian row.
///
/// Returns an n x n matrix as `Vec<Vec<f64>>`.
///
/// # Example
///
/// ```rust
/// use numra_autodiff::reverse::hessian;
///
/// // f(x,y) = x^2 + 2*x*y + 3*y^2
/// let h = hessian(|x| {
///     &x[0] * &x[0] + x[0].cst(2.0) * &x[0] * &x[1] + x[0].cst(3.0) * &x[1] * &x[1]
/// }, &[1.0, 1.0]);
/// assert!((h[0][0] - 2.0).abs() < 1e-6);  // d2f/dx2 = 2
/// assert!((h[0][1] - 2.0).abs() < 1e-6);  // d2f/dxdy = 2
/// assert!((h[1][0] - 2.0).abs() < 1e-6);  // d2f/dydx = 2
/// assert!((h[1][1] - 6.0).abs() < 1e-6);  // d2f/dy2 = 6
/// ```
pub fn hessian<F>(f: F, x: &[f64]) -> Vec<Vec<f64>>
where
    F: Fn(&[Var]) -> Var,
{
    let n = x.len();
    let eps = 1e-7;
    let mut h = vec![vec![0.0; n]; n];

    let g0 = grad(&f, x);

    for j in 0..n {
        let mut x_pert = x.to_vec();
        x_pert[j] += eps;
        let g_pert = grad(&f, &x_pert);

        for i in 0..n {
            h[i][j] = (g_pert[i] - g0[i]) / eps;
        }
    }

    h
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_arithmetic() {
        let tape = Tape::new();
        let x = Tape::var(&tape, 2.0);
        let y = Tape::var(&tape, 3.0);

        // z = x + y = 5
        let z = x.clone() + y.clone();
        assert!((z.value - 5.0).abs() < 1e-14);
        let g = Tape::gradient(&tape, &z);
        assert!((g[0] - 1.0).abs() < 1e-14);
        assert!((g[1] - 1.0).abs() < 1e-14);
    }

    #[test]
    fn test_multiplication() {
        let tape = Tape::new();
        let x = Tape::var(&tape, 2.0);
        let y = Tape::var(&tape, 3.0);

        // z = x * y = 6
        let z = x * y;
        let g = Tape::gradient(&tape, &z);
        assert!((g[0] - 3.0).abs() < 1e-14); // dz/dx = y
        assert!((g[1] - 2.0).abs() < 1e-14); // dz/dy = x
    }

    #[test]
    fn test_chain_rule() {
        let tape = Tape::new();
        let x = Tape::var(&tape, 2.0);

        // z = (x * x) * x = x^3, dz/dx = 3x^2 = 12
        let z = x.clone() * x.clone() * x;
        let g = Tape::gradient(&tape, &z);
        assert!((g[0] - 12.0).abs() < 1e-12);
    }

    #[test]
    fn test_subtraction() {
        let tape = Tape::new();
        let x = Tape::var(&tape, 5.0);
        let y = Tape::var(&tape, 3.0);

        let z = x - y;
        assert!((z.value - 2.0).abs() < 1e-14);
        let g = Tape::gradient(&tape, &z);
        assert!((g[0] - 1.0).abs() < 1e-14);
        assert!((g[1] - (-1.0)).abs() < 1e-14);
    }

    #[test]
    fn test_division() {
        let tape = Tape::new();
        let x = Tape::var(&tape, 6.0);
        let y = Tape::var(&tape, 3.0);

        let z = x / y; // z = 2
        let g = Tape::gradient(&tape, &z);
        assert!((g[0] - 1.0 / 3.0).abs() < 1e-14); // dz/dx = 1/y
        assert!((g[1] - (-6.0 / 9.0)).abs() < 1e-14); // dz/dy = -x/y^2
    }

    #[test]
    fn test_negation() {
        let tape = Tape::new();
        let x = Tape::var(&tape, 3.0);
        let z = -x;
        assert!((z.value - (-3.0)).abs() < 1e-14);
        let g = Tape::gradient(&tape, &z);
        assert!((g[0] - (-1.0)).abs() < 1e-14);
    }

    #[test]
    fn test_sin_cos() {
        let tape = Tape::new();
        let x = Tape::var(&tape, 1.0);

        let s = x.sin();
        let g = Tape::gradient(&tape, &s);
        assert!((g[0] - 1.0_f64.cos()).abs() < 1e-14);

        // New tape for cos
        let tape2 = Tape::new();
        let x2 = Tape::var(&tape2, 1.0);
        let c = x2.cos();
        let g2 = Tape::gradient(&tape2, &c);
        assert!((g2[0] - (-1.0_f64.sin())).abs() < 1e-14);
    }

    #[test]
    fn test_exp_ln() {
        let tape = Tape::new();
        let x = Tape::var(&tape, 2.0);
        let e = x.exp();
        let g = Tape::gradient(&tape, &e);
        assert!((g[0] - 2.0_f64.exp()).abs() < 1e-12);

        let tape2 = Tape::new();
        let x2 = Tape::var(&tape2, 3.0);
        let l = x2.ln();
        let g2 = Tape::gradient(&tape2, &l);
        assert!((g2[0] - 1.0 / 3.0).abs() < 1e-14);
    }

    #[test]
    fn test_sqrt() {
        let tape = Tape::new();
        let x = Tape::var(&tape, 4.0);
        let s = x.sqrt();
        assert!((s.value - 2.0).abs() < 1e-14);
        let g = Tape::gradient(&tape, &s);
        assert!((g[0] - 0.25).abs() < 1e-14); // 1/(2*sqrt(4)) = 0.25
    }

    #[test]
    fn test_tanh() {
        let tape = Tape::new();
        let x = Tape::var(&tape, 1.0);
        let t = x.tanh();
        let g = Tape::gradient(&tape, &t);
        let expected = 1.0 - 1.0_f64.tanh().powi(2);
        assert!((g[0] - expected).abs() < 1e-14);
    }

    #[test]
    fn test_powf() {
        let tape = Tape::new();
        let x = Tape::var(&tape, 3.0);
        let p = x.powf(2.0); // x^2
        assert!((p.value - 9.0).abs() < 1e-14);
        let g = Tape::gradient(&tape, &p);
        assert!((g[0] - 6.0).abs() < 1e-12); // d(x^2)/dx = 2x = 6
    }

    #[test]
    fn test_pow_var() {
        let tape = Tape::new();
        let x = Tape::var(&tape, 2.0);
        let y = Tape::var(&tape, 3.0);
        let p = x.pow(&y); // x^y = 8
        assert!((p.value - 8.0).abs() < 1e-12);
        let g = Tape::gradient(&tape, &p);
        // d(x^y)/dx = y * x^(y-1) = 3 * 4 = 12
        assert!((g[0] - 12.0).abs() < 1e-10);
        // d(x^y)/dy = x^y * ln(x) = 8 * ln(2)
        assert!((g[1] - 8.0 * 2.0_f64.ln()).abs() < 1e-10);
    }

    #[test]
    fn test_scalar_ops() {
        let tape = Tape::new();
        let x = Tape::var(&tape, 3.0);

        // x + 2.0
        let z1 = x.clone() + 2.0;
        assert!((z1.value - 5.0).abs() < 1e-14);

        // 2.0 + x
        let z2 = 2.0 + x.clone();
        assert!((z2.value - 5.0).abs() < 1e-14);

        // x * 3.0
        let z3 = x.clone() * 3.0;
        assert!((z3.value - 9.0).abs() < 1e-14);

        // 3.0 * x
        let z4 = 3.0 * x.clone();
        assert!((z4.value - 9.0).abs() < 1e-14);

        // x - 1.0
        let z5 = x.clone() - 1.0;
        assert!((z5.value - 2.0).abs() < 1e-14);

        // 10.0 - x
        let z6 = 10.0 - x.clone();
        assert!((z6.value - 7.0).abs() < 1e-14);

        // x / 2.0
        let z7 = x / 2.0;
        assert!((z7.value - 1.5).abs() < 1e-14);
    }

    #[test]
    fn test_reference_ops() {
        let tape = Tape::new();
        let x = Tape::var(&tape, 2.0);
        let y = Tape::var(&tape, 3.0);

        let z = &x + &y;
        assert!((z.value - 5.0).abs() < 1e-14);

        let z2 = &x * &y;
        assert!((z2.value - 6.0).abs() < 1e-14);

        let z3 = &x - &y;
        assert!((z3.value - (-1.0)).abs() < 1e-14);

        let z4 = &x / &y;
        assert!((z4.value - 2.0 / 3.0).abs() < 1e-14);
    }

    #[test]
    fn test_grad_rosenbrock() {
        // f(x,y) = (1-x)^2 + 100*(y-x^2)^2
        let g = grad(
            |x| {
                let a = x[0].cst(1.0) - x[0].clone(); // 1 - x
                let b = x[1].clone() - x[0].clone() * x[0].clone(); // y - x^2
                a.clone() * a + x[0].cst(100.0) * b.clone() * b
            },
            &[1.0, 1.0],
        );
        // At the minimum (1,1), gradient = (0, 0)
        assert!(g[0].abs() < 1e-10);
        assert!(g[1].abs() < 1e-10);
    }

    #[test]
    fn test_grad_rosenbrock_nonzero() {
        // At (0, 0): df/dx = -2(1-x) - 400x(y-x^2) = -2, df/dy = 200(y-x^2) = 0
        let g = grad(
            |x| {
                let a = x[0].cst(1.0) - x[0].clone();
                let b = x[1].clone() - x[0].clone() * x[0].clone();
                a.clone() * a + x[0].cst(100.0) * b.clone() * b
            },
            &[0.0, 0.0],
        );
        assert!((g[0] - (-2.0)).abs() < 1e-10);
        assert!(g[1].abs() < 1e-10);
    }

    #[test]
    fn test_jacobian_reverse_fn() {
        // f(x,y) = (x+y, x*y)
        let jac = jacobian_reverse(|x| vec![&x[0] + &x[1], &x[0] * &x[1]], &[2.0, 3.0]);
        assert_eq!(jac.len(), 2);
        assert!((jac[0][0] - 1.0).abs() < 1e-14);
        assert!((jac[0][1] - 1.0).abs() < 1e-14);
        assert!((jac[1][0] - 3.0).abs() < 1e-14);
        assert!((jac[1][1] - 2.0).abs() < 1e-14);
    }

    #[test]
    fn test_jacobian_rotation() {
        // Rotation by angle theta: (x*cos(theta) - y*sin(theta), x*sin(theta) + y*cos(theta))
        // Jacobian = [[cos, -sin], [sin, cos]]
        let theta: f64 = 0.5;
        let jac = jacobian_reverse(
            |v| {
                let x = &v[0];
                let y = &v[1];
                let ct = x.cst(theta.cos());
                let st = x.cst(theta.sin());
                vec![&(x * &ct) - &(y * &st), &(x * &st) + &(y * &ct)]
            },
            &[1.0, 0.0],
        );
        assert!((jac[0][0] - theta.cos()).abs() < 1e-12);
        assert!((jac[0][1] - (-theta.sin())).abs() < 1e-12);
        assert!((jac[1][0] - theta.sin()).abs() < 1e-12);
        assert!((jac[1][1] - theta.cos()).abs() < 1e-12);
    }

    #[test]
    fn test_hessian_quadratic() {
        // f(x,y) = x^2 + 2*x*y + 3*y^2
        // Hessian = [[2, 2], [2, 6]]
        let h = hessian(
            |x| &x[0] * &x[0] + x[0].cst(2.0) * &x[0] * &x[1] + x[0].cst(3.0) * &x[1] * &x[1],
            &[1.0, 1.0],
        );
        assert!((h[0][0] - 2.0).abs() < 1e-5);
        assert!((h[0][1] - 2.0).abs() < 1e-5);
        assert!((h[1][0] - 2.0).abs() < 1e-5);
        assert!((h[1][1] - 6.0).abs() < 1e-5);
    }

    #[test]
    fn test_hessian_rosenbrock() {
        // Rosenbrock at (1,1): Hessian = [[802, -400], [-400, 200]]
        let h = hessian(
            |x| {
                let a = x[0].cst(1.0) - x[0].clone();
                let b = x[1].clone() - x[0].clone() * x[0].clone();
                a.clone() * a + x[0].cst(100.0) * b.clone() * b
            },
            &[1.0, 1.0],
        );
        assert!((h[0][0] - 802.0).abs() < 1e-3);
        assert!((h[0][1] - (-400.0)).abs() < 1e-3);
        assert!((h[1][0] - (-400.0)).abs() < 1e-3);
        assert!((h[1][1] - 200.0).abs() < 1e-3);
    }

    #[test]
    fn test_inverse_trig() {
        // asin
        let tape = Tape::new();
        let x = Tape::var(&tape, 0.5);
        let a = x.asin();
        let g = Tape::gradient(&tape, &a);
        assert!((g[0] - 1.0 / (1.0 - 0.25_f64).sqrt()).abs() < 1e-12);

        // atan
        let tape2 = Tape::new();
        let x2 = Tape::var(&tape2, 1.0);
        let a2 = x2.atan();
        let g2 = Tape::gradient(&tape2, &a2);
        assert!((g2[0] - 0.5).abs() < 1e-14); // 1/(1+1^2) = 0.5
    }

    #[test]
    fn test_grad_matches_fd() {
        // Verify reverse-mode gradient matches finite differences
        let f = |x: &[Var]| x[0].sin() * x[1].exp() + x[2].clone() * x[2].clone();
        let x0 = [1.0, 2.0, 3.0];
        let g = grad(f, &x0);

        // Finite difference
        let eps = 1e-7;
        let f_val = |x: &[f64]| x[0].sin() * x[1].exp() + x[2] * x[2];
        let f0 = f_val(&x0);
        for i in 0..3 {
            let mut xp = x0;
            xp[i] += eps;
            let fd = (f_val(&xp) - f0) / eps;
            assert!(
                (g[i] - fd).abs() < 1e-5,
                "component {} mismatch: {} vs {}",
                i,
                g[i],
                fd
            );
        }
    }

    #[test]
    fn test_constant() {
        let tape = Tape::new();
        let x = Tape::var(&tape, 2.0);
        let c = x.cst(5.0);
        let z = x * c; // z = 2 * 5 = 10, dz/dx = 5
        let g = Tape::gradient(&tape, &z);
        assert!((g[0] - 5.0).abs() < 1e-14);
    }

    #[test]
    fn test_complex_composition() {
        // f(x) = sin(x^2 + exp(x))
        let g = grad(
            |x| {
                let x2 = &x[0] * &x[0];
                let ex = x[0].exp();
                let inner = x2 + ex;
                inner.sin()
            },
            &[1.0],
        );
        // df/dx = cos(x^2+exp(x)) * (2x + exp(x))
        let x = 1.0_f64;
        let inner = x * x + x.exp();
        let expected = inner.cos() * (2.0 * x + x.exp());
        assert!((g[0] - expected).abs() < 1e-10);
    }
}

//! Gradient and Jacobian computation via forward-mode AD.
//!
//! These helper functions use the [`Dual`] number type to compute gradients
//! of scalar-valued functions and Jacobians of vector-valued functions.
//!
//! Author: Moussa Leblouba
//! Date: 7 February 2026
//! Modified: 2 May 2026

use crate::Dual;
use numra_core::Scalar;

/// Compute the gradient of a scalar-valued function `f: R^n -> R`.
///
/// The gradient is computed by evaluating `f` once per input dimension,
/// each time seeding the derivative in direction `e_i`.
///
/// # Arguments
///
/// * `f` - A function taking a slice of `Dual<S>` and returning a `Dual<S>`.
/// * `x` - The point at which to evaluate the gradient.
///
/// # Returns
///
/// A `Vec<S>` of length `n` containing `[df/dx_0, df/dx_1, ..., df/dx_{n-1}]`.
///
/// # Example
///
/// ```rust
/// use numra_autodiff::{Dual, gradient};
///
/// // f(x) = x0^2 + 2*x1
/// let grad = gradient(
///     |x: &[Dual<f64>]| x[0] * x[0] + Dual::constant(2.0) * x[1],
///     &[3.0, 5.0],
/// );
/// assert!((grad[0] - 6.0).abs() < 1e-12); // df/dx0 = 2*x0 = 6
/// assert!((grad[1] - 2.0).abs() < 1e-12); // df/dx1 = 2
/// ```
pub fn gradient<S, F>(f: F, x: &[S]) -> Vec<S>
where
    S: Scalar,
    F: Fn(&[Dual<S>]) -> Dual<S>,
{
    let n = x.len();
    let mut grad = Vec::with_capacity(n);

    for i in 0..n {
        // Build dual input: seed direction e_i
        let dual_x: Vec<Dual<S>> = (0..n)
            .map(|j| {
                if j == i {
                    Dual::variable(x[j])
                } else {
                    Dual::constant(x[j])
                }
            })
            .collect();

        let result = f(&dual_x);
        grad.push(result.deriv());
    }

    grad
}

/// Compute the Jacobian of a vector-valued function `f: R^n -> R^m`.
///
/// The Jacobian is computed by evaluating `f` once per input dimension,
/// each time seeding the derivative in direction `e_i`. The result is
/// stored in row-major format: `J[i * n + j] = df_i / dx_j`.
///
/// # Arguments
///
/// * `f` - A function taking input `&[Dual<S>]` and writing output to `&mut [Dual<S>]`.
/// * `x` - The point at which to evaluate the Jacobian.
/// * `m` - The number of output components.
///
/// # Returns
///
/// A `Vec<S>` of length `m * n` containing the Jacobian in row-major format.
///
/// # Example
///
/// ```rust
/// use numra_autodiff::{Dual, jacobian};
///
/// // F(x) = [x0 + x1, x0 * x1]
/// let jac = jacobian(
///     |x: &[Dual<f64>], out: &mut [Dual<f64>]| {
///         out[0] = x[0] + x[1];
///         out[1] = x[0] * x[1];
///     },
///     &[2.0, 3.0],
///     2,
/// );
/// // Jacobian: [[1, 1], [3, 2]]
/// assert!((jac[0] - 1.0).abs() < 1e-12);
/// assert!((jac[1] - 1.0).abs() < 1e-12);
/// assert!((jac[2] - 3.0).abs() < 1e-12);
/// assert!((jac[3] - 2.0).abs() < 1e-12);
/// ```
pub fn jacobian<S, F>(f: F, x: &[S], m: usize) -> Vec<S>
where
    S: Scalar,
    F: Fn(&[Dual<S>], &mut [Dual<S>]),
{
    let n = x.len();
    // Row-major: J[i * n + j] = df_i / dx_j
    let mut jac = vec![S::ZERO; m * n];
    let mut out = vec![Dual::constant(S::ZERO); m];

    for j in 0..n {
        // Build dual input: seed direction e_j
        let dual_x: Vec<Dual<S>> = (0..n)
            .map(|k| {
                if k == j {
                    Dual::variable(x[k])
                } else {
                    Dual::constant(x[k])
                }
            })
            .collect();

        // Reset output
        for o in out.iter_mut() {
            *o = Dual::constant(S::ZERO);
        }

        f(&dual_x, &mut out);

        // Column j of the Jacobian
        for i in 0..m {
            jac[i * n + j] = out[i].deriv();
        }
    }

    jac
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1e-12;

    #[test]
    fn test_gradient_quadratic() {
        // f(x) = x0^2 + 2*x1^2 + x0*x1
        // df/dx0 = 2*x0 + x1
        // df/dx1 = 4*x1 + x0
        // At (1, 2): grad = [2*1 + 2, 4*2 + 1] = [4, 9]
        let grad = gradient(
            |x: &[Dual<f64>]| {
                let two = Dual::constant(2.0);
                x[0] * x[0] + two * x[1] * x[1] + x[0] * x[1]
            },
            &[1.0, 2.0],
        );
        assert!((grad[0] - 4.0).abs() < TOL);
        assert!((grad[1] - 9.0).abs() < TOL);
    }

    #[test]
    fn test_gradient_rosenbrock() {
        // f(x) = (1 - x0)^2 + 100*(x1 - x0^2)^2
        // At (1, 1): gradient = [0, 0] (minimum)
        let grad = gradient(
            |x: &[Dual<f64>]| {
                let one = Dual::constant(1.0);
                let hundred = Dual::constant(100.0);
                let a = one - x[0];
                let b = x[1] - x[0] * x[0];
                a * a + hundred * b * b
            },
            &[1.0, 1.0],
        );
        assert!((grad[0] - 0.0).abs() < 1e-10);
        assert!((grad[1] - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_jacobian() {
        // F(x) = [x0^2 + x1, x0 - x1^2]
        // J = [[2*x0, 1], [1, -2*x1]]
        // At (1, 2): J = [[2, 1], [1, -4]]
        let jac = jacobian(
            |x: &[Dual<f64>], out: &mut [Dual<f64>]| {
                out[0] = x[0] * x[0] + x[1];
                out[1] = x[0] - x[1] * x[1];
            },
            &[1.0, 2.0],
            2,
        );
        assert!((jac[0] - 2.0).abs() < TOL); // df0/dx0 = 2*x0 = 2
        assert!((jac[1] - 1.0).abs() < TOL); // df0/dx1 = 1
        assert!((jac[2] - 1.0).abs() < TOL); // df1/dx0 = 1
        assert!((jac[3] - (-4.0)).abs() < TOL); // df1/dx1 = -2*x1 = -4
    }

    #[test]
    fn test_jacobian_3x2() {
        // F: R^2 -> R^3
        // F(x) = [x0 + x1, x0 * x1, x0^2 - x1]
        // J = [[1, 1], [x1, x0], [2*x0, -1]]
        // At (3, 2): J = [[1, 1], [2, 3], [6, -1]]
        let jac = jacobian(
            |x: &[Dual<f64>], out: &mut [Dual<f64>]| {
                out[0] = x[0] + x[1];
                out[1] = x[0] * x[1];
                out[2] = x[0] * x[0] - x[1];
            },
            &[3.0, 2.0],
            3,
        );
        assert!((jac[0] - 1.0).abs() < TOL); // df0/dx0
        assert!((jac[1] - 1.0).abs() < TOL); // df0/dx1
        assert!((jac[2] - 2.0).abs() < TOL); // df1/dx0 = x1 = 2
        assert!((jac[3] - 3.0).abs() < TOL); // df1/dx1 = x0 = 3
        assert!((jac[4] - 6.0).abs() < TOL); // df2/dx0 = 2*x0 = 6
        assert!((jac[5] - (-1.0)).abs() < TOL); // df2/dx1 = -1
    }

    #[test]
    fn test_gradient_single_variable() {
        // f(x) = x^3
        // f'(2) = 12
        let grad = gradient(|x: &[Dual<f64>]| x[0] * x[0] * x[0], &[2.0]);
        assert!((grad[0] - 12.0).abs() < TOL);
    }
}

//! Bridge utilities for connecting autodiff with optimization.
//!
//! These helpers convert autodiff's function-based gradient API into the
//! closure signatures that `numra-optim` and `numra-fit` expect.
//!
//! Author: Moussa Leblouba
//! Date: 9 February 2026
//! Modified: 2 May 2026

use crate::dual::Dual;
use crate::gradient::gradient;
use numra_core::Scalar;

/// Convert an autodiff-compatible function into a gradient closure suitable
/// for `OptimProblem::gradient()`.
///
/// The returned closure has the signature `Fn(&[S], &mut [S])` where the first
/// argument is the point `x` and the second is the output gradient buffer.
///
/// # Example
///
/// ```rust
/// use numra_autodiff::{Dual, bridge::gradient_closure};
///
/// let grad_fn = gradient_closure(|x: &[Dual<f64>]| x[0] * x[0] + x[1] * x[1]);
/// let mut g = vec![0.0; 2];
/// grad_fn(&[3.0, 4.0], &mut g);
/// assert!((g[0] - 6.0).abs() < 1e-12); // d/dx0 = 2*x0
/// assert!((g[1] - 8.0).abs() < 1e-12); // d/dx1 = 2*x1
/// ```
pub fn gradient_closure<S, F>(f: F) -> impl Fn(&[S], &mut [S])
where
    S: Scalar,
    F: Fn(&[Dual<S>]) -> Dual<S>,
{
    move |x: &[S], g: &mut [S]| {
        let grad = gradient(&f, x);
        g[..grad.len()].copy_from_slice(&grad);
    }
}

/// Convert an autodiff-compatible function into a model Jacobian closure
/// suitable for `curve_fit_with_jacobian()`.
///
/// Given a model function `f(x, params)` written in terms of `Dual` numbers,
/// this returns a closure `Fn(S, &[S]) -> Vec<S>` that computes
/// `d(model)/d(params)` at a single data point.
///
/// # Example
///
/// ```rust
/// use numra_autodiff::{Dual, bridge::model_jacobian_closure};
///
/// // Model: y = a * exp(-b * x)
/// let jac_fn = model_jacobian_closure(|x: Dual<f64>, p: &[Dual<f64>]| {
///     p[0] * (-p[1] * x).exp()
/// });
/// let dm_dp = jac_fn(1.0_f64, &[5.0, 0.4]);
/// // dm/da = exp(-0.4) ≈ 0.6703
/// assert!((dm_dp[0] - (-0.4_f64).exp()).abs() < 1e-10);
/// ```
pub fn model_jacobian_closure<S, F>(f: F) -> impl Fn(S, &[S]) -> Vec<S>
where
    S: Scalar,
    F: Fn(Dual<S>, &[Dual<S>]) -> Dual<S>,
{
    move |x: S, params: &[S]| {
        let n = params.len();
        let x_dual = Dual::constant(x);
        let mut jac = Vec::with_capacity(n);

        for i in 0..n {
            let dual_p: Vec<Dual<S>> = (0..n)
                .map(|j| {
                    if j == i {
                        Dual::variable(params[j])
                    } else {
                        Dual::constant(params[j])
                    }
                })
                .collect();
            let result = f(x_dual, &dual_p);
            jac.push(result.deriv());
        }

        jac
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gradient_closure_quadratic() {
        // f(x) = x0^2 + 2*x1
        let grad_fn = gradient_closure(|x: &[Dual<f64>]| x[0] * x[0] + Dual::constant(2.0) * x[1]);

        let mut g = vec![0.0; 2];
        grad_fn(&[3.0, 5.0], &mut g);
        assert!((g[0] - 6.0).abs() < 1e-12);
        assert!((g[1] - 2.0).abs() < 1e-12);
    }

    #[test]
    fn test_gradient_closure_rosenbrock() {
        // f(x) = (1-x0)^2 + 100*(x1-x0^2)^2
        // At minimum (1,1), gradient = [0,0]
        let grad_fn = gradient_closure(|x: &[Dual<f64>]| {
            let one = Dual::constant(1.0);
            let hundred = Dual::constant(100.0);
            let a = one - x[0];
            let b = x[1] - x[0] * x[0];
            a * a + hundred * b * b
        });

        let mut g = vec![0.0; 2];
        grad_fn(&[1.0, 1.0], &mut g);
        assert!((g[0]).abs() < 1e-10);
        assert!((g[1]).abs() < 1e-10);
    }

    #[test]
    fn test_model_jacobian_closure_exp() {
        // Model: y = a * exp(-b * x)
        // dm/da = exp(-b*x), dm/db = -a*x*exp(-b*x)
        let jac_fn =
            model_jacobian_closure(|x: Dual<f64>, p: &[Dual<f64>]| p[0] * (-p[1] * x).exp());

        let dm_dp = jac_fn(1.0_f64, &[5.0, 0.4]);
        let expected_da = (-0.4_f64).exp();
        let expected_db = -5.0 * 1.0 * (-0.4_f64).exp();
        assert!((dm_dp[0] - expected_da).abs() < 1e-10);
        assert!((dm_dp[1] - expected_db).abs() < 1e-10);
    }
}

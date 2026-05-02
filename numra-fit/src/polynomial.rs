//! Polynomial fitting and evaluation.
//!
//! `polyfit` fits a polynomial of given degree to data via least squares.
//! `polyval` evaluates a polynomial at given points.
//!
//! Author: Moussa Leblouba
//! Date: 9 February 2026
//! Modified: 2 May 2026

use numra_core::Scalar;
use numra_linalg::{DenseMatrix, LUFactorization};

use crate::error::FitError;
use crate::types::FitResult;

/// Fit a polynomial of degree `deg` to data `(x, y)`.
///
/// Returns a `FitResult` where `params` are the polynomial coefficients
/// in descending order: `params[0]*x^deg + params[1]*x^(deg-1) + ... + params[deg]`.
///
/// Uses QR decomposition of the Vandermonde matrix for numerical stability.
///
/// # Example
/// ```
/// use numra_fit::polyfit;
///
/// let x = vec![0.0_f64, 1.0, 2.0, 3.0, 4.0];
/// let y = vec![1.0_f64, 3.0, 5.0, 7.0, 9.0]; // y = 2x + 1
///
/// let result = polyfit(&x, &y, 1).unwrap();
/// assert!((result.params[0] - 2.0_f64).abs() < 1e-10); // slope
/// assert!((result.params[1] - 1.0_f64).abs() < 1e-10); // intercept
/// ```
pub fn polyfit<S>(x_data: &[S], y_data: &[S], deg: usize) -> Result<FitResult<S>, FitError>
where
    S: Scalar + faer::SimpleEntity + faer::Conjugate<Canonical = S> + faer::ComplexField,
{
    let m = x_data.len();
    let n = deg + 1; // number of coefficients

    if m != y_data.len() {
        return Err(FitError::LengthMismatch {
            expected: m,
            got: y_data.len(),
        });
    }
    if m < n {
        return Err(FitError::Underdetermined {
            n_params: n,
            n_data: m,
        });
    }
    if m == 0 {
        return Err(FitError::InsufficientData { need: 1, got: 0 });
    }

    // Build Vandermonde matrix (m x n) in column order:
    // V[i, j] = x_data[i]^(deg - j) for descending power convention
    let mut v_data = vec![S::ZERO; m * n];
    for i in 0..m {
        let mut xi_pow = S::ONE;
        // Fill from lowest power (rightmost column) to highest (leftmost)
        for j in (0..n).rev() {
            v_data[i * n + j] = xi_pow;
            xi_pow = xi_pow * x_data[i];
        }
    }

    // Use DenseMatrix for QR-based least squares
    // DenseMatrix is column-major, so convert
    let vander = DenseMatrix::from_row_major(m, n, &v_data);

    // Solve via lstsq (SVD-based)
    let coeffs = vander
        .lstsq(y_data)
        .map_err(|e| FitError::OptimizationFailed(format!("lstsq failed: {e}")))?;

    // Compute residuals
    let residuals: Vec<S> = (0..m)
        .map(|i| {
            let y_pred = polyval_internal(&coeffs, x_data[i]);
            y_data[i] - y_pred
        })
        .collect();

    let chi_squared: S = residuals.iter().map(|&r| r * r).sum();
    let dof = m - n;
    let reduced_chi_sq = if dof > 0 {
        chi_squared / S::from_usize(dof)
    } else {
        S::ZERO
    };

    let y_mean: S = y_data.iter().copied().sum::<S>() / S::from_usize(m);
    let ss_tot: S = y_data.iter().map(|&y| (y - y_mean) * (y - y_mean)).sum();
    let r_squared = if ss_tot > S::ZERO {
        S::ONE - chi_squared / ss_tot
    } else {
        S::ONE
    };

    // Covariance from Vandermonde: Cov = sigma^2 * (V^T V)^{-1}
    let (covariance, std_errors) = if dof > 0 {
        compute_poly_covariance(&v_data, m, n, reduced_chi_sq)
    } else {
        (None, None)
    };

    Ok(FitResult {
        params: coeffs,
        covariance,
        std_errors,
        residuals,
        chi_squared,
        reduced_chi_squared: reduced_chi_sq,
        r_squared,
        dof,
        n_evaluations: 0,
        converged: true,
    })
}

/// Evaluate a polynomial at multiple points.
///
/// `coeffs` are in descending power order:
/// `coeffs[0]*x^n + coeffs[1]*x^(n-1) + ... + coeffs[n]`.
///
/// # Example
/// ```
/// use numra_fit::polyval;
///
/// let coeffs = vec![1.0_f64, -2.0, 3.0]; // x^2 - 2x + 3
/// let x = vec![0.0_f64, 1.0, 2.0, 3.0];
/// let y = polyval(&coeffs, &x);
///
/// assert!((y[0] - 3.0_f64).abs() < 1e-10);
/// assert!((y[1] - 2.0_f64).abs() < 1e-10);
/// assert!((y[2] - 3.0_f64).abs() < 1e-10);
/// assert!((y[3] - 6.0_f64).abs() < 1e-10);
/// ```
pub fn polyval<S: Scalar>(coeffs: &[S], x: &[S]) -> Vec<S> {
    x.iter().map(|&xi| polyval_internal(coeffs, xi)).collect()
}

/// Evaluate polynomial at a single point using Horner's method.
fn polyval_internal<S: Scalar>(coeffs: &[S], x: S) -> S {
    if coeffs.is_empty() {
        return S::ZERO;
    }
    let mut result = coeffs[0];
    for &c in &coeffs[1..] {
        result = result * x + c;
    }
    result
}

/// Compute covariance for polynomial fit: sigma^2 * (V^T V)^{-1}.
fn compute_poly_covariance<S>(
    v_data: &[S],
    m: usize,
    n: usize,
    sigma_sq: S,
) -> (Option<Vec<S>>, Option<Vec<S>>)
where
    S: Scalar + faer::SimpleEntity + faer::Conjugate<Canonical = S> + faer::ComplexField,
{
    // V^T V (n x n)
    let mut vtv = vec![S::ZERO; n * n];
    for i in 0..n {
        for j in 0..n {
            let mut sum = S::ZERO;
            for k in 0..m {
                sum = sum + v_data[k * n + i] * v_data[k * n + j];
            }
            vtv[i * n + j] = sum;
        }
    }

    let dense = DenseMatrix::from_row_major(n, n, &vtv);
    let lu = match LUFactorization::new(&dense) {
        Ok(l) => l,
        Err(_) => return (None, None),
    };

    let mut inv = vec![S::ZERO; n * n];
    for j in 0..n {
        let mut e = vec![S::ZERO; n];
        e[j] = S::ONE;
        let col = match lu.solve(&e) {
            Ok(c) => c,
            Err(_) => return (None, None),
        };
        for i in 0..n {
            inv[i * n + j] = col[i] * sigma_sq;
        }
    }

    let std_errors: Vec<S> = (0..n)
        .map(|i| {
            let v = inv[i * n + i];
            if v > S::ZERO {
                v.sqrt()
            } else {
                S::ZERO
            }
        })
        .collect();

    (Some(inv), Some(std_errors))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_polyfit_linear() {
        let x: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let y: Vec<f64> = x.iter().map(|&xi| 2.0 * xi + 1.0).collect();

        let result = polyfit(&x, &y, 1).unwrap();
        assert!((result.params[0] - 2.0).abs() < 1e-10); // slope
        assert!((result.params[1] - 1.0).abs() < 1e-10); // intercept
        assert!(result.r_squared > 1.0 - 1e-12);
        assert_eq!(result.dof, 8);
    }

    #[test]
    fn test_polyfit_quadratic() {
        let x: Vec<f64> = (0..20).map(|i| i as f64 * 0.5).collect();
        // y = 3x^2 - 2x + 1
        let y: Vec<f64> = x.iter().map(|&xi| 3.0 * xi * xi - 2.0 * xi + 1.0).collect();

        let result = polyfit(&x, &y, 2).unwrap();
        assert!((result.params[0] - 3.0).abs() < 1e-8);
        assert!((result.params[1] - (-2.0)).abs() < 1e-8);
        assert!((result.params[2] - 1.0).abs() < 1e-8);
        assert!(result.covariance.is_some());
    }

    #[test]
    fn test_polyfit_cubic() {
        let x: Vec<f64> = (0..30).map(|i| -3.0 + i as f64 * 0.2).collect();
        // y = x^3 - x
        let y: Vec<f64> = x.iter().map(|&xi| xi * xi * xi - xi).collect();

        let result = polyfit(&x, &y, 3).unwrap();
        assert!((result.params[0] - 1.0).abs() < 1e-6); // x^3 coeff
        assert!(result.params[1].abs() < 1e-6); // x^2 coeff
        assert!((result.params[2] - (-1.0)).abs() < 1e-6); // x^1 coeff
        assert!(result.params[3].abs() < 1e-6); // x^0 coeff
    }

    #[test]
    fn test_polyfit_overdetermined() {
        // Fit line to noisy quadratic: won't be perfect
        let x: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let y: Vec<f64> = x.iter().map(|&xi| xi * xi).collect();

        let result = polyfit(&x, &y, 1).unwrap();
        // Should still return a result, just with poor fit
        assert!(result.r_squared < 1.0);
        assert!(result.chi_squared > 0.0);
    }

    #[test]
    fn test_polyval() {
        // x^2 - 2x + 3
        let coeffs = vec![1.0_f64, -2.0, 3.0];
        let x = vec![0.0, 1.0, 2.0, 3.0];
        let y = polyval(&coeffs, &x);

        assert!((y[0] - 3.0).abs() < 1e-10);
        assert!((y[1] - 2.0).abs() < 1e-10);
        assert!((y[2] - 3.0).abs() < 1e-10);
        assert!((y[3] - 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_polyval_single_coeff() {
        let coeffs = vec![5.0_f64];
        let x = vec![1.0, 2.0, 100.0];
        let y = polyval(&coeffs, &x);
        for &yi in &y {
            assert!((yi - 5.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_polyval_empty() {
        let y = polyval::<f64>(&[], &[1.0, 2.0]);
        assert_eq!(y.len(), 2);
        assert!(y[0].abs() < 1e-10);
    }

    #[test]
    fn test_polyfit_length_mismatch() {
        let result = polyfit(&[1.0_f64, 2.0, 3.0], &[1.0, 2.0], 1);
        assert!(matches!(result, Err(FitError::LengthMismatch { .. })));
    }

    #[test]
    fn test_polyfit_underdetermined() {
        let result = polyfit(&[1.0_f64, 2.0], &[1.0, 2.0], 2);
        assert!(matches!(result, Err(FitError::Underdetermined { .. })));
    }

    #[test]
    fn test_polyfit_constant() {
        let x: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let y = vec![42.0_f64; 10];

        let result = polyfit(&x, &y, 0).unwrap();
        assert!((result.params[0] - 42.0).abs() < 1e-10);
    }
}

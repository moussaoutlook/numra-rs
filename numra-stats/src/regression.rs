//! Linear regression, multiple regression, and polynomial fitting.
//!
//! Author: Moussa Leblouba
//! Date: 9 February 2026
//! Modified: 2 May 2026

use numra_core::Scalar;

use crate::descriptive;
use crate::distributions::{student_t::StudentT, ContinuousDistribution};
use crate::error::StatsError;

/// Result of a regression analysis.
#[derive(Clone, Debug)]
pub struct RegressionResult<S: Scalar> {
    /// Coefficients: [intercept, slope1, slope2, ...].
    pub coefficients: Vec<S>,
    /// Coefficient of determination R^2.
    pub r_squared: S,
    /// Residuals (y - y_hat).
    pub residuals: Vec<S>,
    /// Standard errors of coefficients.
    pub std_errors: Vec<S>,
    /// p-values for each coefficient (two-tailed t-test).
    pub p_values: Vec<S>,
}

/// Simple linear regression: y = a + b*x.
pub fn linregress<S: Scalar>(x: &[S], y: &[S]) -> Result<RegressionResult<S>, StatsError> {
    if x.len() != y.len() {
        return Err(StatsError::LengthMismatch {
            expected: x.len(),
            got: y.len(),
        });
    }
    if x.len() < 3 {
        return Err(StatsError::EmptyData);
    }
    let n = x.len();
    let ns = S::from_usize(n);
    let mx = descriptive::mean(x)?;
    let my = descriptive::mean(y)?;

    // Compute slope and intercept
    let mut sxy = S::ZERO;
    let mut sxx = S::ZERO;
    for i in 0..n {
        let dx = x[i] - mx;
        sxy += dx * (y[i] - my);
        sxx += dx * dx;
    }
    let slope = sxy / sxx;
    let intercept = my - slope * mx;

    // Residuals
    let residuals: Vec<S> = (0..n).map(|i| y[i] - intercept - slope * x[i]).collect();

    // R-squared
    let ss_res: S = residuals.iter().fold(S::ZERO, |a, &r| a + r * r);
    let ss_tot: S = y.iter().fold(S::ZERO, |a, &yi| a + (yi - my) * (yi - my));
    let r_squared = S::ONE - ss_res / ss_tot;

    // Standard errors
    let mse = ss_res / S::from_usize(n - 2);
    let se_slope = (mse / sxx).sqrt();
    let se_intercept = (mse * (S::ONE / ns + mx * mx / sxx)).sqrt();

    // p-values via t-distribution
    let df = S::from_usize(n - 2);
    let t_dist = StudentT::new(df);
    let t_intercept = intercept / se_intercept;
    let t_slope = slope / se_slope;
    let p_intercept = S::TWO * (S::ONE - t_dist.cdf(t_intercept.abs()));
    let p_slope = S::TWO * (S::ONE - t_dist.cdf(t_slope.abs()));

    Ok(RegressionResult {
        coefficients: vec![intercept, slope],
        r_squared,
        residuals,
        std_errors: vec![se_intercept, se_slope],
        p_values: vec![p_intercept, p_slope],
    })
}

/// Multiple linear regression: y = X * beta (OLS via normal equations).
///
/// `x_vars` is a slice of predictor vectors, each of length n.
/// Returns coefficients [intercept, b1, b2, ...].
pub fn multiple_linregress<S: Scalar>(
    x_vars: &[Vec<S>],
    y: &[S],
) -> Result<RegressionResult<S>, StatsError> {
    if x_vars.is_empty() {
        return Err(StatsError::EmptyData);
    }
    let n = y.len();
    let p = x_vars.len() + 1; // +1 for intercept
    for v in x_vars {
        if v.len() != n {
            return Err(StatsError::LengthMismatch {
                expected: n,
                got: v.len(),
            });
        }
    }
    if n <= p {
        return Err(StatsError::InvalidParameter(
            "need more observations than predictors".into(),
        ));
    }

    // Build design matrix X (n x p) with intercept column
    // X^T X (p x p) and X^T y (p x 1)
    let mut xtx = vec![S::ZERO; p * p];
    let mut xty = vec![S::ZERO; p];

    for i in 0..n {
        let mut xi = vec![S::ONE]; // intercept
        for v in x_vars {
            xi.push(v[i]);
        }
        for r in 0..p {
            for c in 0..p {
                xtx[r * p + c] += xi[r] * xi[c];
            }
            xty[r] += xi[r] * y[i];
        }
    }

    // Solve X^T X * beta = X^T y using Gaussian elimination
    let beta = solve_linear_system::<S>(&xtx, &xty, p)?;

    // Residuals and R-squared
    let my = descriptive::mean(y)?;
    let mut residuals = Vec::with_capacity(n);
    let mut ss_res = S::ZERO;
    let mut ss_tot = S::ZERO;
    for i in 0..n {
        let mut y_hat = beta[0]; // intercept
        for (j, v) in x_vars.iter().enumerate() {
            y_hat += beta[j + 1] * v[i];
        }
        let r = y[i] - y_hat;
        residuals.push(r);
        ss_res += r * r;
        ss_tot += (y[i] - my) * (y[i] - my);
    }
    let r_squared = S::ONE - ss_res / ss_tot;

    // Standard errors: SE = sqrt(diag(MSE * (X^T X)^-1))
    let mse = ss_res / S::from_usize(n - p);
    let xtx_inv = invert_matrix::<S>(&xtx, p)?;
    let std_errors: Vec<S> = (0..p).map(|i| (mse * xtx_inv[i * p + i]).sqrt()).collect();

    // p-values
    let df = S::from_usize(n - p);
    let t_dist = StudentT::new(df);
    let p_values: Vec<S> = beta
        .iter()
        .zip(std_errors.iter())
        .map(|(&b, &se)| {
            let t = b / se;
            S::TWO * (S::ONE - t_dist.cdf(t.abs()))
        })
        .collect();

    Ok(RegressionResult {
        coefficients: beta,
        r_squared,
        residuals,
        std_errors,
        p_values,
    })
}

/// Polynomial regression: fit y = c0 + c1*x + c2*x^2 + ... + cn*x^n.
///
/// Returns coefficient vector [c0, c1, ..., cn].
pub fn polyfit<S: Scalar>(x: &[S], y: &[S], degree: usize) -> Result<Vec<S>, StatsError> {
    if x.len() != y.len() {
        return Err(StatsError::LengthMismatch {
            expected: x.len(),
            got: y.len(),
        });
    }
    // Build Vandermonde-style predictors: x^1, x^2, ..., x^degree
    let x_vars: Vec<Vec<S>> = (1..=degree)
        .map(|d| x.iter().map(|&xi| xi.powi(d as i32)).collect())
        .collect();
    let result = multiple_linregress(&x_vars, y)?;
    Ok(result.coefficients)
}

/// Solve a linear system A*x = b using Gaussian elimination with partial pivoting.
fn solve_linear_system<S: Scalar>(a: &[S], b: &[S], n: usize) -> Result<Vec<S>, StatsError> {
    // Augmented matrix [A | b]
    let mut aug = vec![S::ZERO; n * (n + 1)];
    for i in 0..n {
        for j in 0..n {
            aug[i * (n + 1) + j] = a[i * n + j];
        }
        aug[i * (n + 1) + n] = b[i];
    }

    // Forward elimination with partial pivoting
    for col in 0..n {
        // Find pivot
        let mut max_row = col;
        let mut max_val = aug[col * (n + 1) + col].to_f64().abs();
        for row in (col + 1)..n {
            let val = aug[row * (n + 1) + col].to_f64().abs();
            if val > max_val {
                max_val = val;
                max_row = row;
            }
        }
        if max_val < 1e-14 {
            return Err(StatsError::SingularMatrix);
        }
        // Swap rows
        if max_row != col {
            for j in 0..=n {
                aug.swap(col * (n + 1) + j, max_row * (n + 1) + j);
            }
        }
        // Eliminate below
        let pivot = aug[col * (n + 1) + col];
        for row in (col + 1)..n {
            let factor = aug[row * (n + 1) + col] / pivot;
            for j in col..=n {
                let val = aug[col * (n + 1) + j];
                aug[row * (n + 1) + j] -= factor * val;
            }
        }
    }

    // Back substitution
    let mut x = vec![S::ZERO; n];
    for i in (0..n).rev() {
        let mut sum = aug[i * (n + 1) + n];
        for j in (i + 1)..n {
            sum -= aug[i * (n + 1) + j] * x[j];
        }
        x[i] = sum / aug[i * (n + 1) + i];
    }
    Ok(x)
}

/// Invert a matrix using Gauss-Jordan elimination.
fn invert_matrix<S: Scalar>(a: &[S], n: usize) -> Result<Vec<S>, StatsError> {
    // Augmented [A | I]
    let mut aug = vec![S::ZERO; n * 2 * n];
    for i in 0..n {
        for j in 0..n {
            aug[i * 2 * n + j] = a[i * n + j];
        }
        aug[i * 2 * n + n + i] = S::ONE;
    }

    let w = 2 * n;
    for col in 0..n {
        // Partial pivot
        let mut max_row = col;
        let mut max_val = aug[col * w + col].to_f64().abs();
        for row in (col + 1)..n {
            let val = aug[row * w + col].to_f64().abs();
            if val > max_val {
                max_val = val;
                max_row = row;
            }
        }
        if max_val < 1e-14 {
            return Err(StatsError::SingularMatrix);
        }
        if max_row != col {
            for j in 0..w {
                aug.swap(col * w + j, max_row * w + j);
            }
        }
        // Scale pivot row
        let pivot = aug[col * w + col];
        for j in 0..w {
            aug[col * w + j] /= pivot;
        }
        // Eliminate all other rows
        for row in 0..n {
            if row == col {
                continue;
            }
            let factor = aug[row * w + col];
            for j in 0..w {
                let val = aug[col * w + j];
                aug[row * w + j] -= factor * val;
            }
        }
    }

    // Extract inverse
    let mut inv = vec![S::ZERO; n * n];
    for i in 0..n {
        for j in 0..n {
            inv[i * n + j] = aug[i * w + n + j];
        }
    }
    Ok(inv)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linregress_perfect_line() {
        // y = 2 + 3*x
        let x: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let y: Vec<f64> = x.iter().map(|&xi| 2.0 + 3.0 * xi).collect();
        let result = linregress(&x, &y).unwrap();
        assert!((result.coefficients[0] - 2.0).abs() < 1e-10);
        assert!((result.coefficients[1] - 3.0).abs() < 1e-10);
        assert!((result.r_squared - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_linregress_with_noise() {
        let x = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let y = vec![2.1, 3.9, 6.2, 7.8, 10.1, 12.0, 13.9, 16.1, 18.0, 20.2];
        let result = linregress(&x, &y).unwrap();
        assert!(result.r_squared > 0.99);
        // Slope should be close to 2
        assert!((result.coefficients[1] - 2.0).abs() < 0.2);
    }

    #[test]
    fn test_multiple_linregress() {
        // y = 1 + 2*x1 + 3*x2
        let x1: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let x2: Vec<f64> = vec![2.0, 1.0, 3.0, 2.0, 4.0, 3.0, 5.0, 4.0, 6.0, 5.0];
        let y: Vec<f64> = x1
            .iter()
            .zip(x2.iter())
            .map(|(&a, &b)| 1.0 + 2.0 * a + 3.0 * b)
            .collect();
        let result = multiple_linregress(&[x1, x2], &y).unwrap();
        assert!((result.coefficients[0] - 1.0).abs() < 1e-8);
        assert!((result.coefficients[1] - 2.0).abs() < 1e-8);
        assert!((result.coefficients[2] - 3.0).abs() < 1e-8);
        assert!((result.r_squared - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_polyfit_quadratic() {
        // y = 1 + 2*x + 3*x^2
        let x: Vec<f64> = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let y: Vec<f64> = x.iter().map(|&xi| 1.0 + 2.0 * xi + 3.0 * xi * xi).collect();
        let coeffs = polyfit(&x, &y, 2).unwrap();
        assert!((coeffs[0] - 1.0).abs() < 1e-8, "c0 = {}", coeffs[0]);
        assert!((coeffs[1] - 2.0).abs() < 1e-8, "c1 = {}", coeffs[1]);
        assert!((coeffs[2] - 3.0).abs() < 1e-8, "c2 = {}", coeffs[2]);
    }
}

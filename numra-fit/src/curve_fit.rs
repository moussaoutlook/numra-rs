//! Nonlinear curve fitting via Levenberg-Marquardt.
//!
//! Fits a parametric model `y = f(x, params)` to data `(x_data, y_data)` by
//! minimizing the sum of squared residuals.
//!
//! Author: Moussa Leblouba
//! Date: 9 February 2026
//! Modified: 2 May 2026

use std::sync::Arc;

use numra_core::Scalar;
use numra_linalg::{DenseMatrix, LUFactorization};
use numra_optim::levenberg_marquardt::{lm_minimize, LmOptions};

use crate::error::FitError;
use crate::types::{FitOptions, FitResult};

/// Fit a nonlinear model to data.
///
/// The model function `f(x, params) -> y` maps a single x-value and parameter
/// vector to a predicted y-value. The optimizer finds `params` that minimize
/// `sum_i (y_data[i] - f(x_data[i], params))^2`.
///
/// # Arguments
/// * `model` - Model function `f(x, params) -> y`
/// * `x_data` - Independent variable data
/// * `y_data` - Dependent variable data (same length as x_data)
/// * `p0` - Initial parameter guess
/// * `opts` - Fitting options (or `None` for defaults)
pub fn curve_fit<S, F>(
    model: F,
    x_data: &[S],
    y_data: &[S],
    p0: &[S],
    opts: Option<&FitOptions<S>>,
) -> Result<FitResult<S>, FitError>
where
    S: Scalar + faer::SimpleEntity + faer::Conjugate<Canonical = S> + faer::ComplexField,
    F: Fn(S, &[S]) -> S + 'static,
{
    let m = x_data.len();
    let n = p0.len();

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

    let defaults = FitOptions::default();
    let opts = opts.unwrap_or(&defaults);

    // Wrap model in Arc so both closures can share it
    let model_arc: Arc<dyn Fn(S, &[S]) -> S> = Arc::new(model);
    let model_res = Arc::clone(&model_arc);
    let model_jac = Arc::clone(&model_arc);

    let x_res = x_data.to_vec();
    let y_res = y_data.to_vec();
    let residual = move |p: &[S], r: &mut [S]| {
        for i in 0..m {
            r[i] = y_res[i] - model_res(x_res[i], p);
        }
    };

    let x_jac = x_data.to_vec();
    let y_jac = y_data.to_vec();
    let jacobian = move |p: &[S], jac: &mut [S]| {
        let h_factor = S::EPSILON.cbrt();
        let mut p_plus = p.to_vec();
        let mut p_minus = p.to_vec();
        for j in 0..n {
            let orig = p[j];
            let h = h_factor * (S::ONE + orig.abs());
            p_plus[j] = orig + h;
            p_minus[j] = orig - h;
            for i in 0..m {
                let r_plus = y_jac[i] - model_jac(x_jac[i], &p_plus);
                let r_minus = y_jac[i] - model_jac(x_jac[i], &p_minus);
                jac[i * n + j] = (r_plus - r_minus) / (S::TWO * h);
            }
            p_plus[j] = orig;
            p_minus[j] = orig;
        }
    };

    let lm_opts = LmOptions {
        max_iter: opts.max_iter,
        gtol: opts.gtol,
        xtol: opts.xtol,
        ftol: opts.ftol,
        ..LmOptions::default()
    };

    let optim_result = lm_minimize(&residual, &jacobian, p0, m, &lm_opts)
        .map_err(|e| FitError::OptimizationFailed(format!("{e}")))?;

    build_fit_result(
        &*model_arc,
        x_data,
        y_data,
        &optim_result.x,
        n,
        m,
        &optim_result,
    )
}

/// Fit a nonlinear model with per-point weights (weighted least squares).
///
/// Minimizes `sum_i weights[i] * (y_data[i] - f(x_data[i], params))^2`.
/// Weights should be positive; typically `weights[i] = 1/sigma_i^2`.
pub fn curve_fit_weighted<S, F>(
    model: F,
    x_data: &[S],
    y_data: &[S],
    weights: &[S],
    p0: &[S],
    opts: Option<&FitOptions<S>>,
) -> Result<FitResult<S>, FitError>
where
    S: Scalar + faer::SimpleEntity + faer::Conjugate<Canonical = S> + faer::ComplexField,
    F: Fn(S, &[S]) -> S + 'static,
{
    let m = x_data.len();
    let n = p0.len();

    if m != y_data.len() {
        return Err(FitError::LengthMismatch {
            expected: m,
            got: y_data.len(),
        });
    }
    if m != weights.len() {
        return Err(FitError::LengthMismatch {
            expected: m,
            got: weights.len(),
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

    let defaults = FitOptions::default();
    let opts = opts.unwrap_or(&defaults);

    let model_arc: Arc<dyn Fn(S, &[S]) -> S> = Arc::new(model);
    let model_res = Arc::clone(&model_arc);
    let model_jac = Arc::clone(&model_arc);

    let sqrt_w: Vec<S> = weights.iter().map(|&w| w.sqrt()).collect();

    let x_res = x_data.to_vec();
    let y_res = y_data.to_vec();
    let sw_res = sqrt_w.clone();
    let residual = move |p: &[S], r: &mut [S]| {
        for i in 0..m {
            r[i] = sw_res[i] * (y_res[i] - model_res(x_res[i], p));
        }
    };

    let x_jac = x_data.to_vec();
    let y_jac = y_data.to_vec();
    let sw_jac = sqrt_w;
    let jacobian = move |p: &[S], jac: &mut [S]| {
        let h_factor = S::EPSILON.cbrt();
        let mut p_plus = p.to_vec();
        let mut p_minus = p.to_vec();
        for j in 0..n {
            let orig = p[j];
            let h = h_factor * (S::ONE + orig.abs());
            p_plus[j] = orig + h;
            p_minus[j] = orig - h;
            for i in 0..m {
                let r_plus = sw_jac[i] * (y_jac[i] - model_jac(x_jac[i], &p_plus));
                let r_minus = sw_jac[i] * (y_jac[i] - model_jac(x_jac[i], &p_minus));
                jac[i * n + j] = (r_plus - r_minus) / (S::TWO * h);
            }
            p_plus[j] = orig;
            p_minus[j] = orig;
        }
    };

    let lm_opts = LmOptions {
        max_iter: opts.max_iter,
        gtol: opts.gtol,
        xtol: opts.xtol,
        ftol: opts.ftol,
        ..LmOptions::default()
    };

    let optim_result = lm_minimize(&residual, &jacobian, p0, m, &lm_opts)
        .map_err(|e| FitError::OptimizationFailed(format!("{e}")))?;

    let w_own = weights.to_vec();
    build_fit_result_weighted(
        &*model_arc,
        x_data,
        y_data,
        &w_own,
        &optim_result.x,
        n,
        m,
        &optim_result,
    )
}

/// Fit a nonlinear model to data with a user-provided analytical Jacobian.
///
/// Like [`curve_fit`], but instead of using finite-difference Jacobians, the user
/// provides a Jacobian closure that returns `d(model)/d(params)` at each data point.
/// This is useful when combined with automatic differentiation for exact derivatives.
///
/// # Arguments
/// * `model` - Model function `f(x, params) -> y`
/// * `model_jac` - Jacobian of the model w.r.t. parameters: `J(x, params) -> Vec<S>`
///   where `J[j] = d(model)/d(params[j])` at `(x, params)`
/// * `x_data` - Independent variable data
/// * `y_data` - Dependent variable data (same length as x_data)
/// * `p0` - Initial parameter guess
/// * `opts` - Fitting options (or `None` for defaults)
pub fn curve_fit_with_jacobian<S, F, J>(
    model: F,
    model_jac: J,
    x_data: &[S],
    y_data: &[S],
    p0: &[S],
    opts: Option<&FitOptions<S>>,
) -> Result<FitResult<S>, FitError>
where
    S: Scalar + faer::SimpleEntity + faer::Conjugate<Canonical = S> + faer::ComplexField,
    F: Fn(S, &[S]) -> S + 'static,
    J: Fn(S, &[S]) -> Vec<S> + 'static,
{
    let m = x_data.len();
    let n = p0.len();

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

    let defaults = FitOptions::default();
    let opts = opts.unwrap_or(&defaults);

    let model_arc: Arc<dyn Fn(S, &[S]) -> S> = Arc::new(model);
    let model_res = Arc::clone(&model_arc);
    let jac_arc: Arc<dyn Fn(S, &[S]) -> Vec<S>> = Arc::new(model_jac);

    let x_res = x_data.to_vec();
    let y_res = y_data.to_vec();
    let residual = move |p: &[S], r: &mut [S]| {
        for i in 0..m {
            r[i] = y_res[i] - model_res(x_res[i], p);
        }
    };

    let x_jac = x_data.to_vec();
    let jacobian = move |p: &[S], jac: &mut [S]| {
        // Build LM Jacobian (m x n, row-major) from per-point model Jacobians.
        // LM residual is r_i = y_i - model(x_i, p), so dr_i/dp_j = -dm_i/dp_j.
        for i in 0..m {
            let dm_dp = jac_arc(x_jac[i], p);
            for j in 0..n {
                jac[i * n + j] = -dm_dp[j];
            }
        }
    };

    let lm_opts = LmOptions {
        max_iter: opts.max_iter,
        gtol: opts.gtol,
        xtol: opts.xtol,
        ftol: opts.ftol,
        ..LmOptions::default()
    };

    let optim_result = lm_minimize(&residual, &jacobian, p0, m, &lm_opts)
        .map_err(|e| FitError::OptimizationFailed(format!("{e}")))?;

    build_fit_result(
        &*model_arc,
        x_data,
        y_data,
        &optim_result.x,
        n,
        m,
        &optim_result,
    )
}

/// Build FitResult from optimized parameters (unweighted case).
fn build_fit_result<S>(
    model: &dyn Fn(S, &[S]) -> S,
    x_data: &[S],
    y_data: &[S],
    params: &[S],
    n: usize,
    m: usize,
    optim: &numra_optim::OptimResult<S>,
) -> Result<FitResult<S>, FitError>
where
    S: Scalar + faer::SimpleEntity + faer::Conjugate<Canonical = S> + faer::ComplexField,
{
    let residuals: Vec<S> = (0..m)
        .map(|i| y_data[i] - model(x_data[i], params))
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

    let (covariance, std_errors) = compute_covariance(model, x_data, params, n, m, reduced_chi_sq);

    Ok(FitResult {
        params: params.to_vec(),
        covariance,
        std_errors,
        residuals,
        chi_squared,
        reduced_chi_squared: reduced_chi_sq,
        r_squared,
        dof,
        n_evaluations: optim.n_feval,
        converged: optim.converged,
    })
}

/// Build FitResult for weighted fit.
fn build_fit_result_weighted<S>(
    model: &dyn Fn(S, &[S]) -> S,
    x_data: &[S],
    y_data: &[S],
    weights: &[S],
    params: &[S],
    n: usize,
    m: usize,
    optim: &numra_optim::OptimResult<S>,
) -> Result<FitResult<S>, FitError>
where
    S: Scalar + faer::SimpleEntity + faer::Conjugate<Canonical = S> + faer::ComplexField,
{
    let residuals: Vec<S> = (0..m)
        .map(|i| y_data[i] - model(x_data[i], params))
        .collect();

    let chi_squared: S = residuals
        .iter()
        .zip(weights.iter())
        .map(|(&r, &w)| w * r * r)
        .sum();
    let dof = m - n;
    let reduced_chi_sq = if dof > 0 {
        chi_squared / S::from_usize(dof)
    } else {
        S::ZERO
    };

    let y_mean: S = y_data.iter().copied().sum::<S>() / S::from_usize(m);
    let ss_tot: S = y_data.iter().map(|&y| (y - y_mean) * (y - y_mean)).sum();
    let ss_res: S = residuals.iter().map(|&r| r * r).sum();
    let r_squared = if ss_tot > S::ZERO {
        S::ONE - ss_res / ss_tot
    } else {
        S::ONE
    };

    let (covariance, std_errors) =
        compute_covariance_weighted(model, x_data, weights, params, n, m);

    Ok(FitResult {
        params: params.to_vec(),
        covariance,
        std_errors,
        residuals,
        chi_squared,
        reduced_chi_squared: reduced_chi_sq,
        r_squared,
        dof,
        n_evaluations: optim.n_feval,
        converged: optim.converged,
    })
}

/// Compute covariance matrix via sigma^2 * (J^T J)^{-1}.
fn compute_covariance<S>(
    model: &dyn Fn(S, &[S]) -> S,
    x_data: &[S],
    params: &[S],
    n: usize,
    m: usize,
    sigma_sq: S,
) -> (Option<Vec<S>>, Option<Vec<S>>)
where
    S: Scalar + faer::SimpleEntity + faer::Conjugate<Canonical = S> + faer::ComplexField,
{
    let jac = model_jacobian(model, x_data, params, n, m);
    let jtj = jtj_product(&jac, n, m);
    invert_and_scale(jtj, n, sigma_sq)
}

/// Compute covariance for weighted fit: (J^T W J)^{-1}.
fn compute_covariance_weighted<S>(
    model: &dyn Fn(S, &[S]) -> S,
    x_data: &[S],
    weights: &[S],
    params: &[S],
    n: usize,
    m: usize,
) -> (Option<Vec<S>>, Option<Vec<S>>)
where
    S: Scalar + faer::SimpleEntity + faer::Conjugate<Canonical = S> + faer::ComplexField,
{
    let jac = model_jacobian(model, x_data, params, n, m);

    let mut jtwj = vec![S::ZERO; n * n];
    for i in 0..n {
        for j in 0..n {
            let mut sum = S::ZERO;
            for k in 0..m {
                sum = sum + jac[k * n + i] * weights[k] * jac[k * n + j];
            }
            jtwj[i * n + j] = sum;
        }
    }

    invert_and_scale(jtwj, n, S::ONE)
}

/// Compute the Jacobian of the model: dm_i/dp_j (m x n, row-major).
fn model_jacobian<S>(
    model: &dyn Fn(S, &[S]) -> S,
    x_data: &[S],
    params: &[S],
    n: usize,
    m: usize,
) -> Vec<S>
where
    S: Scalar,
{
    let h_factor = S::EPSILON.cbrt();
    let mut jac = vec![S::ZERO; m * n];
    let mut p_pert = params.to_vec();

    for j in 0..n {
        let orig = p_pert[j];
        let h = h_factor * (S::ONE + orig.abs());
        p_pert[j] = orig + h;
        let f_plus: Vec<S> = (0..m).map(|i| model(x_data[i], &p_pert)).collect();
        p_pert[j] = orig - h;
        let f_minus: Vec<S> = (0..m).map(|i| model(x_data[i], &p_pert)).collect();
        for i in 0..m {
            jac[i * n + j] = (f_plus[i] - f_minus[i]) / (S::TWO * h);
        }
        p_pert[j] = orig;
    }
    jac
}

/// Compute J^T J (n x n) from row-major J (m x n).
fn jtj_product<S: Scalar>(jac: &[S], n: usize, m: usize) -> Vec<S> {
    let mut jtj = vec![S::ZERO; n * n];
    for i in 0..n {
        for j in 0..n {
            let mut sum = S::ZERO;
            for k in 0..m {
                sum = sum + jac[k * n + i] * jac[k * n + j];
            }
            jtj[i * n + j] = sum;
        }
    }
    jtj
}

/// Invert a symmetric n x n matrix (row-major) and scale by sigma_sq.
fn invert_and_scale<S>(mat: Vec<S>, n: usize, sigma_sq: S) -> (Option<Vec<S>>, Option<Vec<S>>)
where
    S: Scalar + faer::SimpleEntity + faer::Conjugate<Canonical = S> + faer::ComplexField,
{
    let dense = DenseMatrix::from_row_major(n, n, &mat);

    let lu = match LUFactorization::new(&dense) {
        Ok(l) => l,
        Err(_) => return (None, None),
    };

    // Solve for identity to get inverse
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
    fn test_curve_fit_exponential_decay() {
        let a_true = 5.0_f64;
        let b_true = 0.4;
        let x_data: Vec<f64> = (0..20).map(|i| i as f64 * 0.5).collect();
        let y_data: Vec<f64> = x_data
            .iter()
            .map(|&x| a_true * (-b_true * x).exp())
            .collect();

        let result = curve_fit(
            |x: f64, p: &[f64]| p[0] * (-p[1] * x).exp(),
            &x_data,
            &y_data,
            &[4.0, 0.3],
            None,
        )
        .unwrap();

        assert!(result.converged);
        assert!((result.params[0] - a_true).abs() < 1e-6);
        assert!((result.params[1] - b_true).abs() < 1e-6);
        assert!(result.r_squared > 0.9999);
        assert!(result.covariance.is_some());
        assert!(result.std_errors.is_some());
    }

    #[test]
    fn test_curve_fit_linear() {
        let x_data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let y_data: Vec<f64> = x_data.iter().map(|&x| 2.0 + 3.0 * x).collect();

        let result = curve_fit(
            |x: f64, p: &[f64]| p[0] + p[1] * x,
            &x_data,
            &y_data,
            &[1.0, 1.0],
            None,
        )
        .unwrap();

        assert!(result.converged);
        assert!((result.params[0] - 2.0).abs() < 1e-6);
        assert!((result.params[1] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_curve_fit_gaussian() {
        let a = 3.0_f64;
        let mu = 2.0;
        let sigma = 0.5;
        let x_data: Vec<f64> = (0..40).map(|i| i as f64 * 0.1).collect();
        let y_data: Vec<f64> = x_data
            .iter()
            .map(|&x| a * (-((x - mu) / sigma).powi(2) / 2.0).exp())
            .collect();

        let result = curve_fit(
            |x: f64, p: &[f64]| p[0] * (-((x - p[1]) / p[2]).powi(2) / 2.0).exp(),
            &x_data,
            &y_data,
            &[2.5, 1.8, 0.6],
            None,
        )
        .unwrap();

        assert!(result.converged);
        assert!((result.params[0] - a).abs() < 1e-4);
        assert!((result.params[1] - mu).abs() < 1e-4);
        assert!((result.params[2] - sigma).abs() < 1e-3);
    }

    #[test]
    fn test_curve_fit_weighted() {
        let a_true = 5.0_f64;
        let b_true = 0.4;
        let x_data: Vec<f64> = (0..20).map(|i| i as f64 * 0.5).collect();
        let y_data: Vec<f64> = x_data
            .iter()
            .map(|&x| a_true * (-b_true * x).exp())
            .collect();
        let weights = vec![1.0_f64; 20];

        let result = curve_fit_weighted(
            |x: f64, p: &[f64]| p[0] * (-p[1] * x).exp(),
            &x_data,
            &y_data,
            &weights,
            &[4.0, 0.3],
            None,
        )
        .unwrap();

        assert!(result.converged);
        assert!((result.params[0] - a_true).abs() < 1e-6);
        assert!((result.params[1] - b_true).abs() < 1e-6);
    }

    #[test]
    fn test_curve_fit_length_mismatch() {
        let result = curve_fit(
            |x: f64, p: &[f64]| p[0] * x,
            &[1.0, 2.0, 3.0],
            &[1.0, 2.0],
            &[1.0],
            None,
        );
        assert!(matches!(result, Err(FitError::LengthMismatch { .. })));
    }

    #[test]
    fn test_curve_fit_underdetermined() {
        let result = curve_fit(
            |x: f64, p: &[f64]| p[0] + p[1] * x + p[2] * x * x,
            &[1.0, 2.0],
            &[1.0, 4.0],
            &[1.0, 1.0, 1.0],
            None,
        );
        assert!(matches!(result, Err(FitError::Underdetermined { .. })));
    }

    #[test]
    fn test_curve_fit_with_options() {
        let x_data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let y_data: Vec<f64> = x_data.iter().map(|&x| 2.0 + 3.0 * x).collect();

        let opts = FitOptions::default().max_iter(50).gtol(1e-8);
        let result = curve_fit(
            |x: f64, p: &[f64]| p[0] + p[1] * x,
            &x_data,
            &y_data,
            &[1.0, 1.0],
            Some(&opts),
        )
        .unwrap();

        assert!(result.converged);
    }

    #[test]
    fn test_fit_result_statistics() {
        let x_data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let y_data: Vec<f64> = x_data.iter().map(|&x| 2.0 + 3.0 * x).collect();

        let result = curve_fit(
            |x: f64, p: &[f64]| p[0] + p[1] * x,
            &x_data,
            &y_data,
            &[1.0, 1.0],
            None,
        )
        .unwrap();

        assert!(result.chi_squared < 1e-15);
        assert!(result.r_squared > 1.0 - 1e-12);
        assert_eq!(result.dof, 8);
        assert_eq!(result.residuals.len(), 10);
    }

    #[test]
    fn test_curve_fit_with_jacobian_exp_decay() {
        let a_true = 5.0_f64;
        let b_true = 0.4;
        let x_data: Vec<f64> = (0..20).map(|i| i as f64 * 0.5).collect();
        let y_data: Vec<f64> = x_data
            .iter()
            .map(|&x| a_true * (-b_true * x).exp())
            .collect();

        // Analytical Jacobian: dm/da = exp(-b*x), dm/db = -a*x*exp(-b*x)
        let result = curve_fit_with_jacobian(
            |x: f64, p: &[f64]| p[0] * (-p[1] * x).exp(),
            |x: f64, p: &[f64]| {
                let e = (-p[1] * x).exp();
                vec![e, -p[0] * x * e]
            },
            &x_data,
            &y_data,
            &[4.0, 0.3],
            None,
        )
        .unwrap();

        assert!(result.converged);
        assert!((result.params[0] - a_true).abs() < 1e-6);
        assert!((result.params[1] - b_true).abs() < 1e-6);
        assert!(result.r_squared > 0.9999);
    }

    #[test]
    fn test_curve_fit_with_jacobian_matches_fd() {
        // Compare analytical Jacobian fit vs FD fit — both should converge to same answer
        let x_data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let y_data: Vec<f64> = x_data.iter().map(|&x| 2.0 + 3.0 * x).collect();

        let result_fd = curve_fit(
            |x: f64, p: &[f64]| p[0] + p[1] * x,
            &x_data,
            &y_data,
            &[1.0, 1.0],
            None,
        )
        .unwrap();

        let result_jac = curve_fit_with_jacobian(
            |x: f64, p: &[f64]| p[0] + p[1] * x,
            |x: f64, _p: &[f64]| vec![1.0, x],
            &x_data,
            &y_data,
            &[1.0, 1.0],
            None,
        )
        .unwrap();

        assert!(result_jac.converged);
        assert!((result_jac.params[0] - result_fd.params[0]).abs() < 1e-8);
        assert!((result_jac.params[1] - result_fd.params[1]).abs() < 1e-8);
    }

    #[test]
    fn test_curve_fit_power_law() {
        // y = a * x^b
        let a_true = 2.5_f64;
        let b_true = 1.7;
        let x_data: Vec<f64> = (1..=15).map(|i| i as f64 * 0.5).collect();
        let y_data: Vec<f64> = x_data.iter().map(|&x| a_true * x.powf(b_true)).collect();

        let result = curve_fit(
            |x: f64, p: &[f64]| p[0] * x.powf(p[1]),
            &x_data,
            &y_data,
            &[2.0, 1.5],
            None,
        )
        .unwrap();

        assert!(result.converged);
        assert!((result.params[0] - a_true).abs() < 1e-5);
        assert!((result.params[1] - b_true).abs() < 1e-5);
    }
}

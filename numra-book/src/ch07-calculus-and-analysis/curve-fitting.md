# Curve Fitting

Numra provides nonlinear curve fitting, polynomial fitting, and polynomial
evaluation. The fitting engine is built on Levenberg-Marquardt optimization
(from `numra-optim`) for nonlinear models and SVD-based least squares
(from `numra-linalg`) for polynomial fits.

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
use numra::fit::{
    curve_fit,                // nonlinear least squares
    curve_fit_weighted,       // weighted nonlinear least squares
    curve_fit_with_jacobian,  // with user-provided analytical Jacobian
    polyfit,                  // polynomial least squares
    polyval,                  // polynomial evaluation
    FitResult,                // result type with params, statistics
    FitOptions,               // convergence control
};
```

## Nonlinear Curve Fitting -- `curve_fit`

`curve_fit` fits any parametric model \\(y = f(x, \mathbf{p})\\) to data by
minimizing the sum of squared residuals:

\\[
\min_{\mathbf{p}} \sum_{i=1}^{m} \left[y_i - f(x_i, \mathbf{p})\right]^2
\\]

The model function takes a single x-value and a parameter slice, returning
the predicted y-value.

### Basic Example: Exponential Decay

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
use numra::fit::{curve_fit, FitResult};

// Generate data: y = 5.0 * exp(-0.4 * x)
let x_data: Vec<f64> = (0..20).map(|i| i as f64 * 0.5).collect();
let y_data: Vec<f64> = x_data.iter()
    .map(|&x| 5.0 * (-0.4 * x).exp())
    .collect();

// Fit the model y = a * exp(-b * x)
let result = curve_fit(
    |x: f64, p: &[f64]| p[0] * (-p[1] * x).exp(),
    &x_data,
    &y_data,
    &[4.0, 0.3],  // initial guess for [a, b]
    None,          // default options
).unwrap();

println!("a = {:.4}, b = {:.4}", result.params[0], result.params[1]);
println!("R^2 = {:.8}", result.r_squared);
println!("Converged: {}", result.converged);
```

### Arguments

| Parameter | Type | Description |
|-----------|------|-------------|
| `model` | `Fn(S, &[S]) -> S` | Model function \\(f(x, \mathbf{p})\\) |
| `x_data` | `&[S]` | Independent variable data |
| `y_data` | `&[S]` | Dependent variable data (same length as `x_data`) |
| `p0` | `&[S]` | Initial parameter guess |
| `opts` | `Option<&FitOptions<S>>` | Convergence options (or `None` for defaults) |

### The `FitResult` Type

Every successful fit returns a `FitResult<S>` containing:

| Field | Type | Description |
|-------|------|-------------|
| `params` | `Vec<S>` | Optimized parameters |
| `covariance` | `Option<Vec<S>>` | Covariance matrix (row-major, \\(n \times n\\)) |
| `std_errors` | `Option<Vec<S>>` | Standard errors (\\(\sqrt{\text{diag}(\text{cov})}\\)) |
| `residuals` | `Vec<S>` | Residuals \\(y_i - f(x_i, \hat{\mathbf{p}})\\) |
| `chi_squared` | `S` | Sum of squared residuals |
| `reduced_chi_squared` | `S` | \\(\chi^2 / \text{dof}\\) |
| `r_squared` | `S` | Coefficient of determination \\(R^2\\) |
| `dof` | `usize` | Degrees of freedom (\\(m - n\\)) |
| `n_evaluations` | `usize` | Number of function evaluations |
| `converged` | `bool` | Whether the optimizer converged |

The covariance matrix is estimated as \\(\hat{\sigma}^2 (J^T J)^{-1}\\), where
\\(J\\) is the Jacobian of the model at the optimal parameters and
\\(\hat{\sigma}^2\\) is the reduced chi-squared.

## FitOptions

Control convergence behavior with the builder pattern:

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
use numra::fit::FitOptions;

let opts = FitOptions::default()
    .max_iter(500)    // maximum iterations (default: 200)
    .ftol(1e-14)      // function tolerance (default: 1e-12)
    .gtol(1e-12)      // gradient tolerance (default: 1e-10)
    .xtol(1e-12);     // step tolerance (default: 1e-10)
```

## Weighted Fitting -- `curve_fit_weighted`

When data points have different uncertainties, use `curve_fit_weighted` to
give more weight to more reliable measurements. The optimizer minimizes:

\\[
\min_{\mathbf{p}} \sum_{i=1}^{m} w_i \left[y_i - f(x_i, \mathbf{p})\right]^2
\\]

Weights are typically \\(w_i = 1/\sigma_i^2\\), where \\(\sigma_i\\) is the
measurement uncertainty at point \\(i\\).

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
use numra::fit::curve_fit_weighted;

let x_data: Vec<f64> = (0..20).map(|i| i as f64 * 0.5).collect();
let y_data: Vec<f64> = x_data.iter()
    .map(|&x| 5.0 * (-0.4 * x).exp())
    .collect();

// Higher weight on early points (more reliable)
let weights: Vec<f64> = x_data.iter()
    .map(|&x| 1.0 / (1.0 + x))  // decreasing confidence with x
    .collect();

let result = curve_fit_weighted(
    |x: f64, p: &[f64]| p[0] * (-p[1] * x).exp(),
    &x_data,
    &y_data,
    &weights,
    &[4.0, 0.3],
    None,
).unwrap();
```

The covariance for weighted fits uses \\((J^T W J)^{-1}\\), where \\(W\\) is
the diagonal weight matrix.

## Analytical Jacobians -- `curve_fit_with_jacobian`

By default, `curve_fit` computes Jacobians via central finite differences.
If you can provide the analytical Jacobian of the model with respect to its
parameters, use `curve_fit_with_jacobian` for faster convergence and exact
derivatives.

The Jacobian closure returns a `Vec<S>` of length \\(n\\) where element \\(j\\)
is \\(\partial f / \partial p_j\\) evaluated at \\((x, \mathbf{p})\\).

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
use numra::fit::curve_fit_with_jacobian;

// Model: y = a * exp(-b * x)
// dm/da = exp(-b*x)
// dm/db = -a * x * exp(-b*x)
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
).unwrap();
```

This is particularly useful when combined with automatic differentiation
(see the [Automatic Differentiation](automatic-differentiation.md) section).

## Polynomial Fitting -- `polyfit`

`polyfit` fits a polynomial of specified degree to data via SVD-based least
squares on the Vandermonde matrix:

\\[
y = c_0 x^n + c_1 x^{n-1} + \cdots + c_{n-1} x + c_n
\\]

Coefficients are returned in **descending power order**: `params[0]` is the
coefficient of \\(x^n\\), and `params[n]` is the constant term.

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
use numra::fit::polyfit;

// Linear fit: y = 2x + 1
let x = vec![0.0_f64, 1.0, 2.0, 3.0, 4.0];
let y = vec![1.0_f64, 3.0, 5.0, 7.0, 9.0];

let result = polyfit(&x, &y, 1).unwrap();
assert!((result.params[0] - 2.0).abs() < 1e-10);  // slope
assert!((result.params[1] - 1.0).abs() < 1e-10);  // intercept
println!("R^2 = {:.12}", result.r_squared);
```

### Quadratic and Higher Degree

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
use numra::fit::polyfit;

// Quadratic fit: y = 3x^2 - 2x + 1
let x: Vec<f64> = (0..20).map(|i| i as f64 * 0.5).collect();
let y: Vec<f64> = x.iter()
    .map(|&xi| 3.0 * xi * xi - 2.0 * xi + 1.0)
    .collect();

let result = polyfit(&x, &y, 2).unwrap();
assert!((result.params[0] - 3.0).abs() < 1e-8);   // x^2 coefficient
assert!((result.params[1] - (-2.0)).abs() < 1e-8); // x coefficient
assert!((result.params[2] - 1.0).abs() < 1e-8);   // constant

// Covariance and standard errors are available
if let Some(ref se) = result.std_errors {
    println!("Standard errors: {:?}", se);
}
```

### Constant Fit (Degree 0)

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
let x: Vec<f64> = (0..10).map(|i| i as f64).collect();
let y = vec![42.0_f64; 10];
let result = polyfit(&x, &y, 0).unwrap();
assert!((result.params[0] - 42.0).abs() < 1e-10);
```

## Polynomial Evaluation -- `polyval`

`polyval` evaluates a polynomial (given in descending power order) at multiple
points using Horner's method for numerical stability:

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
use numra::fit::polyval;

// p(x) = x^2 - 2x + 3
let coeffs = vec![1.0_f64, -2.0, 3.0];
let x = vec![0.0, 1.0, 2.0, 3.0];
let y = polyval(&coeffs, &x);

assert!((y[0] - 3.0).abs() < 1e-10);  // p(0) = 3
assert!((y[1] - 2.0).abs() < 1e-10);  // p(1) = 1-2+3 = 2
assert!((y[2] - 3.0).abs() < 1e-10);  // p(2) = 4-4+3 = 3
assert!((y[3] - 6.0).abs() < 1e-10);  // p(3) = 9-6+3 = 6
```

Combining `polyfit` and `polyval` for prediction:

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
use numra::fit::{polyfit, polyval};

let x_train = vec![0.0_f64, 1.0, 2.0, 3.0, 4.0];
let y_train = vec![1.0, 2.8, 5.1, 6.9, 9.2];

let result = polyfit(&x_train, &y_train, 1).unwrap();

// Predict at new points
let x_test = vec![5.0_f64, 6.0, 7.0];
let y_pred = polyval(&result.params, &x_test);
println!("Predictions: {:?}", y_pred);
```

## Common Fitting Models

### Gaussian Peak

\\[
y = a \cdot \exp\left(-\frac{(x - \mu)^2}{2\sigma^2}\right)
\\]

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
use numra::fit::curve_fit;

let a_true = 3.0_f64;
let mu_true = 2.0;
let sigma_true = 0.5;

let x_data: Vec<f64> = (0..40).map(|i| i as f64 * 0.1).collect();
let y_data: Vec<f64> = x_data.iter()
    .map(|&x| a_true * (-((x - mu_true) / sigma_true).powi(2) / 2.0).exp())
    .collect();

let result = curve_fit(
    |x: f64, p: &[f64]| p[0] * (-((x - p[1]) / p[2]).powi(2) / 2.0).exp(),
    &x_data, &y_data,
    &[2.5, 1.8, 0.6],  // initial guess
    None,
).unwrap();

println!("Amplitude: {:.4} (true: {})", result.params[0], a_true);
println!("Center:    {:.4} (true: {})", result.params[1], mu_true);
println!("Width:     {:.4} (true: {})", result.params[2], sigma_true);
```

### Power Law

\\[
y = a \cdot x^b
\\]

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
use numra::fit::curve_fit;

let x_data: Vec<f64> = (1..=15).map(|i| i as f64 * 0.5).collect();
let y_data: Vec<f64> = x_data.iter()
    .map(|&x| 2.5 * x.powf(1.7))
    .collect();

let result = curve_fit(
    |x: f64, p: &[f64]| p[0] * x.powf(p[1]),
    &x_data, &y_data,
    &[2.0, 1.5],
    None,
).unwrap();

println!("a = {:.4}, b = {:.4}", result.params[0], result.params[1]);
```

### Logistic / Sigmoid

\\[
y = \frac{L}{1 + e^{-k(x - x_0)}}
\\]

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
use numra::fit::curve_fit;

let result = curve_fit(
    |x: f64, p: &[f64]| p[0] / (1.0 + (-p[1] * (x - p[2])).exp()),
    &x_data, &y_data,
    &[1.0, 1.0, 0.0],  // [L, k, x0]
    None,
).unwrap();
```

## Inspecting Fit Quality

### R-Squared

The coefficient of determination \\(R^2\\) measures the fraction of variance
explained by the model:

\\[
R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}
\\]

Values close to 1.0 indicate an excellent fit.

### Residual Analysis

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
let result = curve_fit(/* ... */).unwrap();

// Residuals should be randomly scattered around zero
println!("Max residual: {:.2e}", result.residuals.iter()
    .map(|r| r.abs())
    .fold(0.0_f64, f64::max));

// Reduced chi-squared should be close to 1.0 for a good fit
// with correct uncertainty estimates
println!("Reduced chi^2: {:.4}", result.reduced_chi_squared);
```

### Parameter Uncertainties

When the covariance matrix is available, standard errors give 1-sigma
confidence intervals on the fitted parameters:

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
let result = curve_fit(/* ... */).unwrap();

if let (Some(ref cov), Some(ref se)) = (&result.covariance, &result.std_errors) {
    for (i, (&p, &e)) in result.params.iter().zip(se.iter()).enumerate() {
        println!("p[{}] = {:.4} +/- {:.4}", i, p, e);
    }

    // Correlation between parameters
    let n = result.params.len();
    let rho_01 = cov[0 * n + 1] / (se[0] * se[1]);
    println!("Correlation(p0, p1) = {:.4}", rho_01);
}
```

## Error Handling

All fitting functions return `Result<FitResult<S>, FitError>`. Possible errors:

| Error | Cause |
|-------|-------|
| `LengthMismatch` | `x_data` and `y_data` have different lengths |
| `Underdetermined` | More parameters than data points |
| `InsufficientData` | Zero data points |
| `OptimizationFailed` | Levenberg-Marquardt did not converge |

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
use numra::fit::{curve_fit, FitError};

let result = curve_fit(
    |x: f64, p: &[f64]| p[0] + p[1] * x + p[2] * x * x,
    &[1.0, 2.0],      // only 2 data points
    &[1.0, 4.0],
    &[1.0, 1.0, 1.0], // 3 parameters -- underdetermined!
    None,
);
assert!(matches!(result, Err(FitError::Underdetermined { .. })));
```

## Tips for Successful Fitting

- **Initial guess matters.** Nonlinear fitting can converge to local minima.
  A good initial guess (even approximate) dramatically improves success rates.

- **Scale your parameters.** If parameters differ by orders of magnitude
  (e.g., amplitude = 1000, rate = 0.001), the optimizer may struggle.
  Rescale the model or data so parameters are \\(O(1)\\).

- **Use `curve_fit_with_jacobian` for complex models.** Analytical Jacobians
  avoid the \\(2n\\) extra function evaluations per iteration that
  finite-difference Jacobians require.

- **Check `converged`.** If `result.converged` is `false`, the optimizer
  hit `max_iter` without satisfying the tolerance. Try increasing `max_iter`
  or improving the initial guess.

- **Watch for parameter correlation.** If the covariance matrix shows
  near-1.0 correlation between parameters, the model may be overparameterized.

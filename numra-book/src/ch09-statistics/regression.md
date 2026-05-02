# Regression

Numra's `numra-stats` crate provides linear regression (simple and multiple)
and polynomial fitting via ordinary least squares (OLS). All regression
functions return rich result structures with coefficients, R-squared,
residuals, standard errors, and p-values.

## RegressionResult

All regression functions return a `RegressionResult<S>` struct:

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
pub struct RegressionResult<S: Scalar> {
    pub coefficients: Vec<S>,  // [intercept, slope1, slope2, ...]
    pub r_squared: S,          // coefficient of determination R^2
    pub residuals: Vec<S>,     // y - y_hat
    pub std_errors: Vec<S>,    // standard errors of coefficients
    pub p_values: Vec<S>,      // two-tailed t-test p-values
}
```

| Field | Description |
|---|---|
| `coefficients` | The intercept followed by one slope per predictor |
| `r_squared` | Proportion of variance explained (0 to 1) |
| `residuals` | Observed minus predicted values |
| `std_errors` | Standard error for each coefficient |
| `p_values` | Statistical significance of each coefficient |

## Simple Linear Regression

Fits the model \\(y = a + bx\\) using ordinary least squares.

**Formulas:**

\\[
b = \frac{\sum_{i=1}^n (x_i - \bar{x})(y_i - \bar{y})}{\sum_{i=1}^n (x_i - \bar{x})^2},
\qquad
a = \bar{y} - b\bar{x}
\\]

**Coefficient of determination:**

\\[
R^2 = 1 - \frac{\text{SS}_{\text{res}}}{\text{SS}_{\text{tot}}}
= 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}
\\]

### Perfect Fit

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
use numra::stats::linregress;

// y = 2 + 3x (exact)
let x: Vec<f64> = (0..10).map(|i| i as f64).collect();
let y: Vec<f64> = x.iter().map(|&xi| 2.0 + 3.0 * xi).collect();

let result = linregress(&x, &y).unwrap();
assert!((result.coefficients[0] - 2.0).abs() < 1e-10); // intercept
assert!((result.coefficients[1] - 3.0).abs() < 1e-10); // slope
assert!((result.r_squared - 1.0).abs() < 1e-10);       // perfect fit
```

### Fit With Noise

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
use numra::stats::linregress;

let x = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
let y = vec![2.1, 3.9, 6.2, 7.8, 10.1, 12.0, 13.9, 16.1, 18.0, 20.2];

let result = linregress(&x, &y).unwrap();
println!("Intercept: {:.4}", result.coefficients[0]);
println!("Slope:     {:.4}", result.coefficients[1]);
println!("R-squared: {:.6}", result.r_squared);

// Slope should be close to 2.0, R^2 very high
assert!(result.r_squared > 0.99);
assert!((result.coefficients[1] - 2.0).abs() < 0.2);
```

### Examining Standard Errors and p-Values

Standard errors quantify the uncertainty in each coefficient estimate.
The p-values test whether each coefficient is significantly different from
zero using a two-tailed t-test with \\(n - 2\\) degrees of freedom.

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
use numra::stats::linregress;

let x = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
let y = vec![2.1, 3.9, 6.2, 7.8, 10.1, 12.0, 13.9, 16.1, 18.0, 20.2];

let result = linregress(&x, &y).unwrap();

println!("Coefficients:");
println!("  Intercept = {:.4} +/- {:.4}  (p = {:.6})",
    result.coefficients[0], result.std_errors[0], result.p_values[0]);
println!("  Slope     = {:.4} +/- {:.4}  (p = {:.2e})",
    result.coefficients[1], result.std_errors[1], result.p_values[1]);
println!("R-squared:  {:.6}", result.r_squared);
```

### Residual Analysis

Residuals should be randomly scattered around zero for a good fit:

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
use numra::stats::{linregress, mean};

let x = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
let y = vec![2.1, 3.9, 6.2, 7.8, 10.1, 12.0, 13.9, 16.1, 18.0, 20.2];

let result = linregress(&x, &y).unwrap();

// Residual mean should be approximately zero
let res_mean = mean(&result.residuals).unwrap();
assert!(res_mean.abs() < 1e-10);

// Print residuals
for (i, &r) in result.residuals.iter().enumerate() {
    println!("x={:.0}: residual = {:.4}", x[i], r);
}
```

## Multiple Linear Regression

Fits the model \\(y = \\beta_0 + \\beta_1 x_1 + \\beta_2 x_2 + \\ldots + \\beta_p x_p\\)
using OLS via the normal equations:

\\[
\hat{\boldsymbol{\beta}} = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y}
\\]

where \\(\mathbf{X}\\) is the design matrix with a column of ones for the intercept.

### Two Predictors

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
use numra::stats::multiple_linregress;

// True model: y = 1 + 2*x1 + 3*x2
let x1: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
let x2: Vec<f64> = vec![2.0, 1.0, 3.0, 2.0, 4.0, 3.0, 5.0, 4.0, 6.0, 5.0];
let y: Vec<f64> = x1.iter().zip(x2.iter())
    .map(|(&a, &b)| 1.0 + 2.0 * a + 3.0 * b)
    .collect();

let result = multiple_linregress(&[x1, x2], &y).unwrap();

assert!((result.coefficients[0] - 1.0).abs() < 1e-8); // intercept
assert!((result.coefficients[1] - 2.0).abs() < 1e-8); // beta_1
assert!((result.coefficients[2] - 3.0).abs() < 1e-8); // beta_2
assert!((result.r_squared - 1.0).abs() < 1e-10);
```

### Interpreting Multiple Regression Output

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
use numra::stats::multiple_linregress;

// House price model: price = f(sqft, bedrooms, age)
let sqft     = vec![1200.0, 1500.0, 1800.0, 2100.0, 2400.0, 1400.0, 1700.0, 2000.0];
let bedrooms = vec![2.0,    3.0,    3.0,    4.0,    4.0,    2.0,    3.0,    4.0];
let age      = vec![20.0,   15.0,   10.0,   5.0,    2.0,    25.0,   12.0,   8.0];
let price    = vec![200.0,  280.0,  340.0,  420.0,  500.0,  180.0,  310.0,  390.0];

let result = multiple_linregress(&[sqft, bedrooms, age], &price).unwrap();

let names = ["Intercept", "SqFt", "Bedrooms", "Age"];
println!("House Price Model (R^2 = {:.4})", result.r_squared);
println!("{:<12} {:>10} {:>10} {:>10}", "Variable", "Coeff", "SE", "p-value");
println!("{}", "-".repeat(44));
for i in 0..result.coefficients.len() {
    println!("{:<12} {:>10.4} {:>10.4} {:>10.4}",
        names[i], result.coefficients[i],
        result.std_errors[i], result.p_values[i]);
}
```

### Standard Errors in Multiple Regression

Standard errors are derived from the mean squared error (MSE) and the
inverse of the design matrix's cross-product:

\\[
\text{SE}(\hat{\beta}_j) = \sqrt{\text{MSE} \cdot (\mathbf{X}^T\mathbf{X})^{-1}_{jj}}
\\]

where \\(\text{MSE} = \text{SS}_{\text{res}} / (n - p)\\) and \\(p\\) is the number
of parameters (including the intercept).

## Polynomial Regression

Fits the model \\(y = c_0 + c_1 x + c_2 x^2 + \\ldots + c_d x^d\\) by
constructing Vandermonde-style predictors and using multiple regression
internally. The `polyfit` function returns coefficients \\([c_0, c_1, \\ldots, c_d]\\).

### Quadratic Fit

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
use numra::stats::polyfit;

// True model: y = 1 + 2*x + 3*x^2
let x: Vec<f64> = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
let y: Vec<f64> = x.iter().map(|&xi| 1.0 + 2.0 * xi + 3.0 * xi * xi).collect();

let coeffs = polyfit(&x, &y, 2).unwrap();
assert!((coeffs[0] - 1.0).abs() < 1e-8); // c0
assert!((coeffs[1] - 2.0).abs() < 1e-8); // c1
assert!((coeffs[2] - 3.0).abs() < 1e-8); // c2
```

### Evaluating the Polynomial

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
use numra::stats::polyfit;

let x = vec![0.0_f64, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
let y = vec![1.0, 2.5, 6.0, 12.0, 20.0, 31.0, 44.0, 59.5, 78.0, 99.0, 123.0];

let coeffs = polyfit(&x, &y, 2).unwrap();

// Evaluate the fitted polynomial at each point
fn polyval(coeffs: &[f64], x: f64) -> f64 {
    coeffs.iter().enumerate()
        .map(|(i, &c)| c * x.powi(i as i32))
        .sum()
}

for (&xi, &yi) in x.iter().zip(y.iter()) {
    let y_hat = polyval(&coeffs, xi);
    println!("x={:5.1}  y={:6.1}  y_hat={:6.2}  residual={:+.2}",
        xi, yi, y_hat, yi - y_hat);
}
```

### Higher-Order Polynomials

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
use numra::stats::polyfit;

// Fit a cubic: y = x^3 - 2x^2 + x + 5
let x: Vec<f64> = (-5..=5).map(|i| i as f64).collect();
let y: Vec<f64> = x.iter()
    .map(|&xi| xi * xi * xi - 2.0 * xi * xi + xi + 5.0)
    .collect();

let coeffs = polyfit(&x, &y, 3).unwrap();
println!("Cubic fit: {:.4} + {:.4}*x + {:.4}*x^2 + {:.4}*x^3",
    coeffs[0], coeffs[1], coeffs[2], coeffs[3]);
// Should be close to: 5 + 1*x - 2*x^2 + 1*x^3
```

## Comparing Regression Types

| Function | Model | Inputs | Returns |
|---|---|---|---|
| `linregress` | \\(y = a + bx\\) | One predictor `x` | Full `RegressionResult` |
| `multiple_linregress` | \\(y = \beta_0 + \sum \beta_j x_j\\) | Multiple predictors | Full `RegressionResult` |
| `polyfit` | \\(y = \sum c_i x^i\\) | One predictor + degree | Coefficient vector |

## Goodness of Fit: R-Squared

The coefficient of determination \\(R^2\\) ranges from 0 to 1:

| \\(R^2\\) | Interpretation |
|---|---|
| \\(> 0.95\\) | Excellent fit |
| \\(0.80 \\text{--} 0.95\\) | Good fit |
| \\(0.50 \\text{--} 0.80\\) | Moderate fit |
| \\(< 0.50\\) | Poor fit |

Note that \\(R^2\\) always increases (or stays the same) when more predictors
are added. For model comparison with different numbers of predictors, consider
using adjusted \\(R^2\\):

\\[
R^2_{\text{adj}} = 1 - \frac{(1 - R^2)(n - 1)}{n - p - 1}
\\]

## Error Handling

Regression functions return `Result<_, StatsError>` for proper error handling:

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
use numra::stats::{linregress, multiple_linregress, polyfit};

// Need at least 3 data points for simple regression
assert!(linregress::<f64>(&[1.0, 2.0], &[1.0, 2.0]).is_err());

// x and y must have the same length
assert!(linregress(&[1.0, 2.0, 3.0], &[1.0, 2.0]).is_err());

// Multiple regression: need more observations than predictors
let x1 = vec![1.0_f64, 2.0];
let x2 = vec![3.0, 4.0];
let y = vec![5.0, 6.0];
assert!(multiple_linregress(&[x1, x2], &y).is_err());
```

## Complete Example: Predicting Test Scores

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
use numra::stats::{linregress, multiple_linregress, polyfit, mean, std_dev};

fn main() {
    // Study hours vs. test scores
    let hours = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    let scores = vec![45.0, 50.0, 55.0, 62.0, 68.0, 73.0, 78.0, 82.0, 85.0, 88.0];

    // === Simple Linear Regression ===
    let linear = linregress(&hours, &scores).unwrap();
    println!("=== Simple Linear Regression ===");
    println!("  Score = {:.2} + {:.2} * Hours",
        linear.coefficients[0], linear.coefficients[1]);
    println!("  R^2 = {:.4}", linear.r_squared);
    println!("  Slope p-value = {:.2e}", linear.p_values[1]);

    // === Quadratic Fit ===
    let quad = polyfit(&hours, &scores, 2).unwrap();
    println!("\n=== Polynomial (degree 2) ===");
    println!("  Score = {:.2} + {:.2}*h + {:.4}*h^2",
        quad[0], quad[1], quad[2]);

    // === Multiple Regression with additional predictor ===
    let sleep = vec![7.0, 7.5, 6.0, 8.0, 7.0, 8.5, 6.5, 7.0, 8.0, 9.0];
    let multi = multiple_linregress(&[hours.clone(), sleep], &scores).unwrap();
    println!("\n=== Multiple Regression ===");
    println!("  Score = {:.2} + {:.2}*Hours + {:.2}*Sleep",
        multi.coefficients[0], multi.coefficients[1], multi.coefficients[2]);
    println!("  R^2 = {:.4}", multi.r_squared);

    // Compare all models
    println!("\n=== Model Comparison ===");
    println!("  Linear R^2:   {:.4}", linear.r_squared);
    println!("  Multiple R^2: {:.4}", multi.r_squared);
}
```

## Function Reference

| Function | Signature | Description |
|---|---|---|
| `linregress` | `fn linregress<S>(x, y) -> Result<RegressionResult<S>>` | Simple linear regression |
| `multiple_linregress` | `fn multiple_linregress<S>(x_vars, y) -> Result<RegressionResult<S>>` | Multiple linear regression |
| `polyfit` | `fn polyfit<S>(x, y, degree) -> Result<Vec<S>>` | Polynomial regression coefficients |

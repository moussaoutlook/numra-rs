# Correlation

Numra's `numra-stats` crate provides two correlation coefficients -- Pearson
and Spearman -- along with the covariance matrix for multivariate data. Both
correlation functions return the coefficient and its associated p-value for
testing statistical significance.

## Pearson Correlation Coefficient

The Pearson correlation coefficient \\(r\\) measures the strength and direction
of the **linear** relationship between two variables:

\\[
r = \frac{\text{Cov}(X, Y)}{s_X \, s_Y}
= \frac{\sum_{i=1}^n (x_i - \bar{x})(y_i - \bar{y})}
       {\sqrt{\sum (x_i - \bar{x})^2} \sqrt{\sum (y_i - \bar{y})^2}}
\\]

The value \\(r\\) lies in \\([-1, 1]\\):

| \\(r\\) | Interpretation |
|---|---|
| \\(+1\\) | Perfect positive linear relationship |
| \\(0\\) | No linear relationship |
| \\(-1\\) | Perfect negative linear relationship |

### Basic Usage

The `pearson_r` function returns a tuple `(r, p_value)`:

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
use numra::stats::pearson_r;

let x = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0];
let y = vec![2.0, 4.0, 6.0, 8.0, 10.0]; // y = 2x (perfect linear)

let (r, p) = pearson_r(&x, &y).unwrap();
assert!((r - 1.0).abs() < 1e-12);  // perfect positive correlation
assert!(p < 0.01);                   // highly significant
```

### Perfect Negative Correlation

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
use numra::stats::pearson_r;

let x = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0];
let y = vec![10.0, 8.0, 6.0, 4.0, 2.0]; // inversely related

let (r, p) = pearson_r(&x, &y).unwrap();
assert!((r + 1.0).abs() < 1e-12);  // perfect negative correlation
assert!(p < 0.01);
```

### p-Value Interpretation

The p-value tests \\(H_0: \\rho = 0\\) (no linear correlation) against
\\(H_1: \\rho \\neq 0\\) using the t-statistic:

\\[
t = r \sqrt{\frac{n - 2}{1 - r^2}}
\\]

with \\(n - 2\\) degrees of freedom.

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
use numra::stats::pearson_r;

let x = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
let y = vec![2.3, 4.1, 5.8, 8.2, 9.7, 12.1, 13.8, 16.0, 18.2, 19.9];

let (r, p) = pearson_r(&x, &y).unwrap();
println!("Pearson r = {:.6}", r);
println!("p-value   = {:.2e}", p);
println!("Significant at alpha=0.05? {}", if p < 0.05 { "Yes" } else { "No" });
```

### Strength Guidelines

| \\(\vert r\vert \\) | Strength |
|---|---|
| \\(0.00 \\text{--} 0.19\\) | Very weak |
| \\(0.20 \\text{--} 0.39\\) | Weak |
| \\(0.40 \\text{--} 0.59\\) | Moderate |
| \\(0.60 \\text{--} 0.79\\) | Strong |
| \\(0.80 \\text{--} 1.00\\) | Very strong |

## Spearman Rank Correlation

The Spearman correlation coefficient \\(\\rho\\) measures the strength of the
**monotonic** relationship between two variables. It is computed as the Pearson
correlation of the rank-transformed data:

\\[
\rho = r(\text{rank}(X), \text{rank}(Y))
\\]

Spearman correlation is more robust than Pearson because:

- It does not assume linearity -- any monotonic relationship gives \\(|\\rho| = 1\\)
- It is less sensitive to outliers
- It works with ordinal data

### Monotonic But Nonlinear

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
use numra::stats::spearman_r;

let x = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0];
let y = vec![1.0, 4.0, 9.0, 16.0, 25.0]; // y = x^2 (monotone, not linear)

let (rho, p) = spearman_r(&x, &y).unwrap();
assert!((rho - 1.0).abs() < 1e-12); // perfect monotonic relationship
println!("Spearman rho = {:.6}, p = {:.2e}", rho, p);
```

Note that \\(y = x^2\\) is perfectly monotonic for positive \\(x\\), so Spearman
gives \\(\\rho = 1\\), while Pearson would give \\(r < 1\\) because the
relationship is nonlinear.

### Handling Ties

When data contains tied values, Spearman uses average ranks:

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
use numra::stats::spearman_r;

let x = vec![1.0_f64, 2.0, 2.0, 3.0, 4.0];
let y = vec![1.0, 2.0, 2.0, 3.0, 4.0];

let (rho, _) = spearman_r(&x, &y).unwrap();
assert!((rho - 1.0).abs() < 1e-12); // still perfect agreement
```

The ranking algorithm assigns average ranks to tied values. For example,
if values at positions 2 and 3 are equal, both receive rank 2.5.

### Spearman vs Pearson

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
use numra::stats::{pearson_r, spearman_r};

// Exponential relationship: monotonic but not linear
let x: Vec<f64> = (1..=20).map(|i| i as f64).collect();
let y: Vec<f64> = x.iter().map(|&xi| (0.1 * xi).exp()).collect();

let (r, _) = pearson_r(&x, &y).unwrap();
let (rho, _) = spearman_r(&x, &y).unwrap();

println!("Pearson r  = {:.4}", r);
println!("Spearman rho = {:.4}", rho);
// rho will be 1.0 (perfect monotone), r < 1.0 (not linear)
```

## Pearson vs Spearman Comparison

| Property | Pearson \\(r\\) | Spearman \\(\\rho\\) |
|---|---|---|
| Measures | Linear association | Monotonic association |
| Data type | Continuous | Continuous or ordinal |
| Sensitivity to outliers | High | Low |
| Distribution assumption | Approximately normal | None |
| Perfect score when | \\(y = a + bx\\) | \\(y = f(x)\\), \\(f\\) monotonic |
| Function | `pearson_r` | `spearman_r` |

## Covariance Matrix

For multivariate analysis, the `covariance_matrix` function computes the
\\(p \\times p\\) sample covariance matrix for \\(p\\) variables:

\\[
\Sigma_{ij} = \text{Cov}(X_i, X_j) = \frac{1}{n-1}\sum_{k=1}^n (x_{ik} - \bar{x}_i)(x_{jk} - \bar{x}_j)
\\]

The diagonal elements are the variances, and the matrix is symmetric.

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
use numra::stats::{covariance_matrix, variance};

let x = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0];
let y = vec![5.0, 4.0, 3.0, 2.0, 1.0]; // inversely correlated

let cov = covariance_matrix(&[x.clone(), y.clone()]).unwrap();
assert_eq!(cov.len(), 4); // 2x2 matrix in row-major order

// Diagonal = variances
assert!((cov[0] - variance(&x).unwrap()).abs() < 1e-12); // Var(x)
assert!((cov[3] - variance(&y).unwrap()).abs() < 1e-12); // Var(y)

// Off-diagonal: negative covariance (inverse relationship)
assert!(cov[1] < 0.0);

// Symmetric: Cov(x,y) = Cov(y,x)
assert!((cov[1] - cov[2]).abs() < 1e-12);
```

### Three Variables

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
use numra::stats::covariance_matrix;

let a = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0];
let b = vec![2.0, 4.0, 6.0, 8.0, 10.0];  // b = 2a (perfectly correlated)
let c = vec![5.0, 3.0, 4.0, 2.0, 1.0];   // roughly inversely related to a

let cov = covariance_matrix(&[a, b, c]).unwrap();
assert_eq!(cov.len(), 9); // 3x3 matrix, row-major

// Print the covariance matrix
println!("Covariance matrix:");
for i in 0..3 {
    println!("  [{:8.4} {:8.4} {:8.4}]",
        cov[i * 3], cov[i * 3 + 1], cov[i * 3 + 2]);
}
```

### From Covariance to Correlation

The correlation matrix can be derived from the covariance matrix by
normalizing each entry by the product of the corresponding standard
deviations:

\\[
r_{ij} = \frac{\Sigma_{ij}}{\sqrt{\Sigma_{ii} \cdot \Sigma_{jj}}}
\\]

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
use numra::stats::covariance_matrix;

let x = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0];
let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];

let cov = covariance_matrix(&[x, y]).unwrap();

// Compute correlation from covariance
let r = cov[1] / (cov[0].sqrt() * cov[3].sqrt());
assert!((r - 1.0).abs() < 1e-12); // perfect correlation
println!("Correlation: {:.6}", r);
```

## When to Use Which

| Situation | Recommended Method |
|---|---|
| Two continuous variables, linear relationship | `pearson_r` |
| Nonlinear but monotonic relationship | `spearman_r` |
| Ordinal data (rankings, Likert scales) | `spearman_r` |
| Data contains outliers | `spearman_r` |
| Multivariate covariance structure | `covariance_matrix` |
| Testing if correlation is significant | Check p-value from `pearson_r` or `spearman_r` |

## Error Handling

Both correlation functions require at least 3 data points and equal-length
inputs:

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
use numra::stats::{pearson_r, spearman_r, covariance_matrix};

// Need at least 3 data points
assert!(pearson_r::<f64>(&[1.0, 2.0], &[1.0, 2.0]).is_err());
assert!(spearman_r::<f64>(&[1.0, 2.0], &[1.0, 2.0]).is_err());

// Inputs must have equal length
assert!(pearson_r(&[1.0, 2.0, 3.0], &[1.0, 2.0]).is_err());

// Covariance matrix needs at least 2 data points per variable
assert!(covariance_matrix::<f64>(&[vec![1.0]]).is_err());
```

## Complete Example: Analyzing Variable Relationships

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
use numra::stats::{pearson_r, spearman_r, covariance_matrix, mean, std_dev};

fn main() {
    // Dataset: temperature, ice cream sales, sunscreen sales
    let temp     = vec![60.0_f64, 65.0, 70.0, 75.0, 80.0, 85.0, 90.0, 95.0];
    let icecream = vec![100.0, 125.0, 160.0, 185.0, 220.0, 260.0, 300.0, 340.0];
    let sunscreen= vec![50.0, 60.0, 80.0, 90.0, 110.0, 130.0, 155.0, 180.0];

    // Pairwise Pearson correlations
    println!("=== Pearson Correlations ===");
    let (r1, p1) = pearson_r(&temp, &icecream).unwrap();
    let (r2, p2) = pearson_r(&temp, &sunscreen).unwrap();
    let (r3, p3) = pearson_r(&icecream, &sunscreen).unwrap();

    println!("  Temp vs Ice Cream:  r = {:.4}, p = {:.2e}", r1, p1);
    println!("  Temp vs Sunscreen:  r = {:.4}, p = {:.2e}", r2, p2);
    println!("  Ice Cream vs Sunscreen: r = {:.4}, p = {:.2e}", r3, p3);

    // Spearman correlations
    println!("\n=== Spearman Correlations ===");
    let (rho1, _) = spearman_r(&temp, &icecream).unwrap();
    let (rho2, _) = spearman_r(&temp, &sunscreen).unwrap();
    println!("  Temp vs Ice Cream:  rho = {:.4}", rho1);
    println!("  Temp vs Sunscreen:  rho = {:.4}", rho2);

    // Full covariance matrix
    println!("\n=== Covariance Matrix (3x3) ===");
    let cov = covariance_matrix(&[
        temp.clone(), icecream.clone(), sunscreen.clone()
    ]).unwrap();

    let names = ["Temp", "IceCream", "Sunscreen"];
    print!("{:>12}", "");
    for name in &names { print!("{:>12}", name); }
    println!();
    for i in 0..3 {
        print!("{:>12}", names[i]);
        for j in 0..3 {
            print!("{:>12.2}", cov[i * 3 + j]);
        }
        println!();
    }

    // Derive correlation matrix from covariance matrix
    println!("\n=== Correlation Matrix ===");
    print!("{:>12}", "");
    for name in &names { print!("{:>12}", name); }
    println!();
    for i in 0..3 {
        print!("{:>12}", names[i]);
        for j in 0..3 {
            let r = cov[i * 3 + j] / (cov[i * 3 + i].sqrt() * cov[j * 3 + j].sqrt());
            print!("{:>12.4}", r);
        }
        println!();
    }
}
```

## Function Reference

| Function | Signature | Description |
|---|---|---|
| `pearson_r` | `fn pearson_r<S>(x, y) -> Result<(S, S)>` | Pearson correlation \\(r\\) and p-value |
| `spearman_r` | `fn spearman_r<S>(x, y) -> Result<(S, S)>` | Spearman rank correlation \\(\\rho\\) and p-value |
| `covariance` | `fn covariance<S>(x, y) -> Result<S>` | Sample covariance between two variables |
| `covariance_matrix` | `fn covariance_matrix<S>(vars) -> Result<Vec<S>>` | \\(p \\times p\\) covariance matrix (row-major) |

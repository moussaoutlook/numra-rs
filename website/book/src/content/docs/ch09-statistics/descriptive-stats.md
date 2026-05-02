---
title: "Descriptive Statistics"
---

Numra's `numra-stats` crate provides a comprehensive set of descriptive
statistics functions for summarizing data. All functions are generic over
`S: Scalar` (f32 or f64) and return `Result` types for proper error handling.

## Central Tendency

### Mean

The arithmetic mean (average) of $n$ values:

$$
\bar{x} = \frac{1}{n} \sum_{i=1}^{n} x_i
$$

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust
use numra::stats::mean;

let data = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0];
let m = mean(&data).unwrap();
assert!((m - 3.0).abs() < 1e-14);
```

### Median

The middle value of the sorted data. For even-length data, it is the average
of the two central values.

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust
use numra::stats::median;

// Odd number of values
let data = vec![3.0_f64, 1.0, 2.0];
assert!((median(&data).unwrap() - 2.0).abs() < 1e-14);

// Even number of values: average of middle two
let data = vec![1.0_f64, 2.0, 3.0, 4.0];
assert!((median(&data).unwrap() - 2.5).abs() < 1e-14);
```

### Percentile

The $p$-th percentile is the value below which $p\%$ of the data falls.
Uses linear interpolation between adjacent data points.

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust
use numra::stats::percentile;

let data = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0];

// Boundary values
assert!((percentile(&data, 0.0).unwrap() - 1.0).abs() < 1e-14);    // minimum
assert!((percentile(&data, 100.0).unwrap() - 5.0).abs() < 1e-14);  // maximum

// 50th percentile = median
assert!((percentile(&data, 50.0).unwrap() - 3.0).abs() < 1e-14);

// Quartiles
let q1 = percentile(&data, 25.0).unwrap();
let q3 = percentile(&data, 75.0).unwrap();
let iqr = q3 - q1; // interquartile range
println!("IQR = {}", iqr);
```

### Computing the IQR

The interquartile range (IQR) measures statistical dispersion -- the spread
of the middle 50% of the data. It is robust to outliers unlike variance.

$$
\text{IQR} = Q_3 - Q_1 = P_{75} - P_{25}
$$

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust
use numra::stats::percentile;

let data = vec![2.0_f64, 7.0, 3.0, 12.0, 5.0, 8.0, 4.0, 6.0, 9.0, 1.0];
let q1 = percentile(&data, 25.0).unwrap();
let q3 = percentile(&data, 75.0).unwrap();
let iqr = q3 - q1;

// Outlier detection: values outside [Q1 - 1.5*IQR, Q3 + 1.5*IQR]
let lower = q1 - 1.5 * iqr;
let upper = q3 + 1.5 * iqr;
let outliers: Vec<f64> = data.iter()
    .filter(|&&x| x < lower || x > upper)
    .copied().collect();
println!("Outliers: {:?}", outliers);
```

## Dispersion

### Variance

Sample variance with Bessel's correction (divides by $n - 1$):

$$
s^2 = \frac{1}{n-1} \sum_{i=1}^{n} (x_i - \bar{x})^2
$$

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust
use numra::stats::variance;

let data = vec![2.0_f64, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
let v = variance(&data).unwrap();
assert!((v - 4.571428571428571).abs() < 1e-10);
```

### Standard Deviation

The square root of the sample variance:

$$
s = \sqrt{s^2}
$$

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust
use numra::stats::{std_dev, variance};

let data = vec![2.0_f64, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
let s = std_dev(&data).unwrap();
let v = variance(&data).unwrap();

// std_dev is the square root of variance
assert!((s * s - v).abs() < 1e-12);
```

## Shape

### Skewness

Fisher's adjusted skewness measures the asymmetry of the distribution:

$$
\tilde{\mu}_3 = \frac{n^2}{(n-1)(n-2)} \cdot \frac{1}{n} \sum_{i=1}^n \left(\frac{x_i - \bar{x}}{s}\right)^3
$$

| Skewness | Interpretation |
|---|---|
| $\approx 0$ | Symmetric distribution |
| $> 0$ | Right-skewed (long right tail) |
| $< 0$ | Left-skewed (long left tail) |

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust
use numra::stats::skewness;

// Symmetric data: skewness near 0
let symmetric = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
assert!(skewness(&symmetric).unwrap().abs() < 1e-10);

// Right-skewed data
let right_skew = vec![1.0_f64, 1.5, 2.0, 2.5, 3.0, 10.0, 20.0];
assert!(skewness(&right_skew).unwrap() > 0.0);
```

### Kurtosis

Fisher's excess kurtosis measures the "tailedness" of the distribution
relative to a normal distribution (which has excess kurtosis of 0):

$$
\text{Kurt} = \frac{n(n+1)}{(n-1)(n-2)(n-3)} \cdot \frac{\sum (x_i - \bar{x})^4}{s^4}
- \frac{3(n-1)^2}{(n-2)(n-3)}
$$

| Kurtosis | Interpretation |
|---|---|
| $\approx 0$ | Mesokurtic (normal-like tails) |
| $> 0$ | Leptokurtic (heavy tails, sharp peak) |
| $< 0$ | Platykurtic (light tails, flat peak) |

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust
use numra::stats::kurtosis;

// Uniform-like data has negative excess kurtosis
let data: Vec<f64> = (0..100).map(|i| i as f64).collect();
let k = kurtosis(&data).unwrap();
assert!(k < 0.0); // approximately -1.2 for uniform
println!("Excess kurtosis: {:.4}", k);
```

## Covariance

### Sample Covariance

Measures the joint variability of two data sets:

$$
\text{Cov}(X, Y) = \frac{1}{n-1} \sum_{i=1}^n (x_i - \bar{x})(y_i - \bar{y})
$$

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust
use numra::stats::{covariance, variance};

let x = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0];
let y = vec![2.0, 4.0, 6.0, 8.0, 10.0]; // y = 2x

let cov = covariance(&x, &y).unwrap();
// Cov(x, 2x) = 2 * Var(x)
let var_x = variance(&x).unwrap();
assert!((cov - 2.0 * var_x).abs() < 1e-12);

// Covariance of a variable with itself = variance
assert!((covariance(&x, &x).unwrap() - var_x).abs() < 1e-12);
```

### Covariance Matrix

For $p$ variables, the covariance matrix is a $p \times p$ symmetric matrix
where entry $(i, j)$ is $\text{Cov}(X_i, X_j)$. The diagonal contains
the variances.

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust
use numra::stats::{covariance_matrix, variance};

let x = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0];
let y = vec![5.0_f64, 4.0, 3.0, 2.0, 1.0]; // inversely correlated with x

let cov = covariance_matrix(&[x.clone(), y.clone()]).unwrap();
assert_eq!(cov.len(), 4); // 2x2 matrix, row-major

// Diagonal = variances
assert!((cov[0] - variance(&x).unwrap()).abs() < 1e-12); // Var(x)
assert!((cov[3] - variance(&y).unwrap()).abs() < 1e-12); // Var(y)

// Off-diagonal: negative (inversely correlated)
assert!(cov[1] < 0.0);

// Symmetric
assert!((cov[1] - cov[2]).abs() < 1e-12);
```

## Summary Statistics Table

| Function | Formula | Min Samples | Notes |
|---|---|---|---|
| `mean` | $\bar{x} = \frac{1}{n}\sum x_i$ | 1 | Arithmetic mean |
| `median` | Middle value | 1 | Sorts internally |
| `percentile(p)` | Linear interpolation | 1 | $p \in [0, 100]$ |
| `variance` | $\frac{\sum(x_i-\bar{x})^2}{n-1}$ | 2 | Bessel's correction |
| `std_dev` | $\sqrt{\text{variance}}$ | 2 | Sample std dev |
| `skewness` | Adjusted third moment | 3 | Fisher's definition |
| `kurtosis` | Adjusted fourth moment | 4 | Fisher's excess kurtosis |
| `covariance` | $\frac{\sum(x_i-\bar{x})(y_i-\bar{y})}{n-1}$ | 2 | Requires equal lengths |
| `covariance_matrix` | $p \times p$ matrix | 2 | Row-major storage |

## Error Handling

All functions return `Result<_, StatsError>`. Common error cases:

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust
use numra::stats::{mean, variance, median, percentile, covariance};

// Empty data
assert!(mean::<f64>(&[]).is_err());
assert!(variance::<f64>(&[]).is_err());
assert!(median::<f64>(&[]).is_err());

// Variance needs at least 2 data points
assert!(variance(&[1.0_f64]).is_err());

// Percentile must be in [0, 100]
let data = vec![1.0_f64, 2.0, 3.0];
assert!(percentile(&data, -1.0).is_err());
assert!(percentile(&data, 101.0).is_err());

// Covariance requires equal-length inputs
assert!(covariance(&[1.0, 2.0], &[1.0, 2.0, 3.0]).is_err());
```

## Complete Example: Data Summary

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust
use numra::stats::{mean, median, std_dev, variance, skewness, kurtosis, percentile};

fn main() {
    let data = vec![
        12.5, 14.2, 11.8, 13.1, 15.0, 12.9, 14.7,
        11.3, 13.6, 14.1, 12.2, 13.8, 15.5, 11.0,
        14.9, 13.3, 12.7, 14.4, 13.0, 12.1,
    ];

    println!("=== Data Summary (n = {}) ===", data.len());
    println!("Mean:      {:.4}", mean(&data).unwrap());
    println!("Median:    {:.4}", median(&data).unwrap());
    println!("Std Dev:   {:.4}", std_dev(&data).unwrap());
    println!("Variance:  {:.4}", variance(&data).unwrap());
    println!("Skewness:  {:.4}", skewness(&data).unwrap());
    println!("Kurtosis:  {:.4}", kurtosis(&data).unwrap());

    let q1 = percentile(&data, 25.0).unwrap();
    let q3 = percentile(&data, 75.0).unwrap();
    println!("Q1:        {:.4}", q1);
    println!("Q3:        {:.4}", q3);
    println!("IQR:       {:.4}", q3 - q1);
    println!("Min:       {:.4}", percentile(&data, 0.0).unwrap());
    println!("Max:       {:.4}", percentile(&data, 100.0).unwrap());
}
```

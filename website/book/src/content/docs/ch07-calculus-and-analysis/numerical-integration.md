---
title: "Numerical Integration"
---

Numra provides a complete suite of numerical integration (quadrature) methods,
from adaptive general-purpose routines to specialized Gaussian rules for
infinite intervals and composite rules for sampled data.

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust
use numra::integrate::{
    quad, QuadOptions, QuadResult,       // adaptive Gauss-Kronrod
    trapezoid, simpson, romberg,         // composite rules
    cumulative_trapezoid,                // running integral
    trapezoid_nonuniform,               // non-uniform spacing
    gauss_legendre,                     // finite intervals
    gauss_laguerre,                     // semi-infinite [0, inf)
    gauss_hermite,                      // infinite (-inf, inf)
    dblquad,                            // double integrals
};
```

## Adaptive Quadrature -- `quad`

The `quad` function is the workhorse integration routine, analogous to SciPy's
`scipy.integrate.quad`. It uses a 15-point Kronrod rule with an embedded 7-point
Gauss rule (G7K15) for error estimation, combined with adaptive bisection of
subintervals ordered by error magnitude.

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust
use numra::integrate::{quad, QuadOptions};

// Integrate sin(x) from 0 to pi -- exact answer is 2.0
let result = quad(
    |x: f64| x.sin(),
    0.0,
    std::f64::consts::PI,
    &QuadOptions::default(),
).unwrap();

assert!((result.value - 2.0).abs() < 1e-10);
println!("Value: {:.15}", result.value);
println!("Error estimate: {:.2e}", result.error_estimate);
println!("Function evaluations: {}", result.n_evaluations);
```

### QuadOptions

Control the accuracy and behavior of `quad` via `QuadOptions`:

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust
let opts = QuadOptions::default()
    .atol(1e-14)              // absolute tolerance
    .rtol(1e-14)              // relative tolerance
    .max_subdivisions(200)    // max adaptive bisections
    .points(vec![0.5]);       // pre-split at known trouble points
```

The convergence criterion is:

$$
|\text{error}| \le \max(\text{atol},\;\text{rtol} \cdot |I|)
$$

where $I$ is the current integral estimate.

### QuadResult

Every call to `quad` returns a `QuadResult` containing:

| Field | Type | Description |
|-------|------|-------------|
| `value` | `S` | Estimated integral value |
| `error_estimate` | `S` | Estimated absolute error |
| `n_evaluations` | `usize` | Total function evaluations |
| `n_subdivisions` | `usize` | Number of bisections performed |

### Handling Singularities

For integrands with known singularities or discontinuities, use the `points`
option to pre-split the interval. This prevents the adaptive algorithm from
wasting evaluations near the problematic location.

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust
// Integrate 1/sqrt(x) from 0 to 1 -- singular at x = 0, exact answer is 2.0
let opts = QuadOptions::default()
    .max_subdivisions(100)
    .points(vec![0.0]);

let result = quad(
    |x: f64| if x.abs() < 1e-300 { 0.0 } else { 1.0 / x.sqrt() },
    0.0, 1.0, &opts,
).unwrap();

assert!((result.value - 2.0).abs() < 1e-6);
```

## Composite Rules for Sampled Data

When you have tabulated data (arrays of y-values at uniform spacing), use
the composite rules directly.

### Trapezoid Rule

The composite trapezoidal rule approximates the integral using linear
interpolation between adjacent points:

$$
\int_a^b f(x)\,dx \approx \frac{dx}{2}\left[y_0 + 2y_1 + 2y_2 + \cdots + 2y_{n-2} + y_{n-1}\right]
$$

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust
use numra::integrate::trapezoid;

// Uniformly-spaced data: y = [0, 1, 4, 9] at x = 0, 1, 2, 3
let y = vec![0.0, 1.0, 4.0, 9.0];
let integral = trapezoid(&y, 1.0);  // dx = 1.0
println!("Trapezoid result: {}", integral);
```

For non-uniformly-spaced data, use `trapezoid_nonuniform`:

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust
use numra::integrate::trapezoid_nonuniform;

let x = vec![0.0, 0.5, 1.5, 3.0];
let y = vec![0.0, 0.25, 2.25, 9.0];
let integral = trapezoid_nonuniform(&x, &y);
```

### Simpson's Rule

Simpson's 1/3 rule uses quadratic interpolation through groups of three points.
It is exact for polynomials up to degree 3:

$$
\int_a^b f(x)\,dx \approx \frac{dx}{3}\left[y_0 + 4y_1 + 2y_2 + 4y_3 + \cdots + 4y_{n-2} + y_{n-1}\right]
$$

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust
use numra::integrate::simpson;

// Simpson's rule is exact for x^2 on uniform grids
let y: Vec<f64> = (0..5).map(|i| (i as f64).powi(2)).collect();
let integral = simpson(&y, 1.0);  // dx = 1.0
// Exact: integral of x^2 from 0 to 4 = 64/3
assert!((integral - 64.0 / 3.0).abs() < 1e-12);
```

If the number of points is even (odd number of panels), Simpson's rule is
applied to the first n-1 points, with a trapezoid correction for the last panel.

### Cumulative Trapezoid

`cumulative_trapezoid` returns the running integral at each point, useful
for computing antiderivatives from sampled data:

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust
use numra::integrate::cumulative_trapezoid;

let y = vec![0.0, 1.0, 2.0, 3.0, 4.0];
let running = cumulative_trapezoid(&y, 1.0);
// running[i] = integral from y[0] to y[i+1]
// [0.5, 2.0, 4.5, 8.0]
```

### Romberg Integration

Romberg integration applies Richardson extrapolation to successive refinements
of the trapezoidal rule, achieving high accuracy with smooth integrands:

$$
R_{k,j} = \frac{4^j R_{k,j-1} - R_{k-1,j-1}}{4^j - 1}
$$

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust
use numra::integrate::romberg;

// Integrate e^x from 0 to 1 (exact: e - 1)
let result = romberg(|x: f64| x.exp(), 0.0, 1.0, 1e-12).unwrap();
assert!((result.value - (std::f64::consts::E - 1.0)).abs() < 1e-12);
```

Romberg integration is particularly effective for smooth (infinitely
differentiable) integrands where the trapezoidal rule's error expansion has
only even powers of $h$.

## Gaussian Quadrature

Gaussian quadrature rules use optimally-chosen nodes and weights to achieve
exact integration of polynomials up to degree $2n - 1$ with $n$ points.

### Gauss-Legendre -- Finite Intervals

Integrates over a finite interval $[a, b]$. Numra provides precomputed
tables for $n = 1$ to $20$:

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust
use numra::integrate::gauss_legendre;

// 5-point rule is exact for polynomials up to degree 9
let result: f64 = gauss_legendre(|x: f64| x * x, 0.0, 1.0, 5).unwrap();
assert!((result - 1.0 / 3.0).abs() < 1e-14);

// High-order rule for transcendental functions
let result: f64 = gauss_legendre(|x: f64| x.exp(), -1.0, 1.0, 20).unwrap();
let exact = std::f64::consts::E - 1.0 / std::f64::consts::E;
assert!((result - exact).abs() < 1e-14);
```

### Gauss-Laguerre -- Semi-Infinite Intervals

Approximates integrals of the form $\int_0^{\infty} f(x)\, e^{-x}\, dx$.
The weight function $e^{-x}$ is built in; you provide only $f(x)$.
Supports $n = 1$ to $10$:

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust
use numra::integrate::gauss_laguerre;

// integral_0^inf x^2 * e^(-x) dx = Gamma(3) = 2! = 2
let result: f64 = gauss_laguerre(|x: f64| x * x, 5).unwrap();
assert!((result - 2.0).abs() < 1e-10);
```

### Gauss-Hermite -- Infinite Intervals

Approximates integrals of the form $\int_{-\infty}^{\infty} f(x)\, e^{-x^2}\, dx$.
The weight function $e^{-x^2}$ is built in. Supports $n = 1$ to $10$:

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust
use numra::integrate::gauss_hermite;

// integral_{-inf}^{inf} 1 * e^(-x^2) dx = sqrt(pi)
let result: f64 = gauss_hermite(|_x: f64| 1.0, 5).unwrap();
assert!((result - std::f64::consts::PI.sqrt()).abs() < 1e-12);

// integral_{-inf}^{inf} x^2 * e^(-x^2) dx = sqrt(pi)/2
let result: f64 = gauss_hermite(|x: f64| x * x, 5).unwrap();
assert!((result - std::f64::consts::PI.sqrt() / 2.0).abs() < 1e-12);
```

## Multi-Dimensional Integration -- `dblquad`

`dblquad` computes double integrals via iterated 1D adaptive quadrature:

$$
\int_{x_a}^{x_b} \int_{y_a(x)}^{y_b(x)} f(x, y)\, dy\, dx
$$

The y-limits can be functions of x, enabling integration over non-rectangular
regions.

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust
use numra::integrate::{dblquad, QuadOptions};

// Integral of x*y over the unit square [0,1] x [0,1] = 0.25
let result = dblquad(
    |x: f64, y: f64| x * y,
    0.0, 1.0,           // x limits
    |_x| 0.0, |_x| 1.0, // y limits (constant = rectangle)
    &QuadOptions::default(),
).unwrap();
assert!((result.value - 0.25).abs() < 1e-10);

// Integral over a triangle: 0 <= x <= 1, 0 <= y <= x
let result = dblquad(
    |_x: f64, _y: f64| 1.0,
    0.0, 1.0,
    |_x| 0.0, |x| x,    // y goes from 0 to x
    &QuadOptions::default(),
).unwrap();
assert!((result.value - 0.5).abs() < 1e-10);
```

## Method Comparison

| Method | Best For | Accuracy | Notes |
|--------|----------|----------|-------|
| `quad` | General-purpose adaptive | $10^{-14}$ | Default choice for function-based integration |
| `trapezoid` | Sampled data (uniform) | $O(h^2)$ | Simple, robust, works with tabulated data |
| `simpson` | Sampled data (uniform) | $O(h^4)$ | Exact for cubics; needs 3+ points |
| `romberg` | Smooth functions | $10^{-12}+$ | Richardson extrapolation of trapezoid rule |
| `gauss_legendre` | Smooth functions, finite interval | Exponential in $n$ | Best when you know the smoothness |
| `gauss_laguerre` | $\int_0^{\infty} f(x) e^{-x} dx$ | Exponential in $n$ | Weight function built in |
| `gauss_hermite` | $\int_{-\infty}^{\infty} f(x) e^{-x^2} dx$ | Exponential in $n$ | Weight function built in |
| `dblquad` | 2D integrals | Inherits from `quad` | Variable y-limits for non-rectangular regions |

### When to Use Each

- **"I have a function and want the integral"** -- Use `quad`. It handles
  singularities, oscillatory integrands, and automatically adapts to the
  required accuracy.

- **"I have tabulated data at uniform spacing"** -- Use `simpson` for best
  accuracy, `trapezoid` for simplicity, or `cumulative_trapezoid` if you need
  the running integral.

- **"I have non-uniformly-spaced data"** -- Use `trapezoid_nonuniform`.

- **"My integral is over a semi-infinite or infinite domain"** -- Use
  `gauss_laguerre` or `gauss_hermite` with the appropriate weight function.

- **"I need a 2D integral"** -- Use `dblquad` with constant or variable limits.

- **"I need maximum accuracy on a smooth function"** -- Use `gauss_legendre`
  with $n = 15$--$20$, or `romberg` with a tight tolerance.

## Generic Over Scalar Type

All integration routines are generic over `S: Scalar`, working with both
`f32` and `f64`:

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust
// f32 integration
let opts = QuadOptions::<f32>::default().atol(1e-4).rtol(1e-4);
let result = quad(|x: f32| x.sin(), 0.0f32, std::f32::consts::PI, &opts).unwrap();
assert!((result.value - 2.0).abs() < 1e-4);
```

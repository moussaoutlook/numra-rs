# Interpolation

Numra provides five 1D interpolation methods, ranging from simple piecewise
linear to numerically stable polynomial interpolation. All implement the
`Interpolant` trait, providing a uniform interface for evaluation, differentiation,
and integration through the interpolant.

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
use numra::interp::{
    Linear,               // piecewise linear
    CubicSpline,          // natural, clamped, not-a-knot
    Pchip,                // monotone Hermite (Fritsch-Carlson)
    Akima,                // outlier-robust piecewise cubic
    BarycentricLagrange,  // stable polynomial interpolation
    Interpolant,          // common trait
    Interp1dMethod,       // method selector enum
    interp1d,             // convenience constructor
};
```

## The `Interpolant` Trait

Every interpolation method implements the `Interpolant<S>` trait:

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
pub trait Interpolant<S: Scalar> {
    /// Evaluate the interpolant at x.
    fn interpolate(&self, x: S) -> S;

    /// Evaluate at multiple points.
    fn interpolate_vec(&self, xs: &[S]) -> Vec<S>;

    /// First derivative at x, if available.
    fn derivative(&self, x: S) -> Option<S>;

    /// Definite integral from a to b, if available.
    fn integrate(&self, a: S, b: S) -> Option<S>;
}
```

All piecewise cubic methods (CubicSpline, Pchip, Akima) support `derivative`
and `integrate`. Linear interpolation supports all three as well. Barycentric
Lagrange provides only `interpolate`.

Extrapolation beyond the data range uses the first or last segment automatically.

## Linear Interpolation

The simplest method: piecewise linear segments connecting adjacent data points.
Requires at least 2 sorted data points.

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
use numra::interp::Linear;

let x = vec![0.0, 1.0, 2.0, 3.0];
let y = vec![0.0, 2.0, 1.0, 3.0];

let interp = Linear::new(&x, &y).unwrap();
assert_eq!(interp.interpolate(0.5), 1.0);   // midpoint of [0,2]
assert_eq!(interp.interpolate(1.5), 1.5);   // midpoint of [2,1]

// Derivative: slope of each segment
assert_eq!(interp.derivative(0.5).unwrap(), 2.0);   // (2-0)/(1-0)
assert_eq!(interp.derivative(1.5).unwrap(), -1.0);  // (1-2)/(2-1)

// Integral from 0 to 2
let area = interp.integrate(0.0, 2.0).unwrap();
```

## Cubic Spline

Cubic spline interpolation constructs a piecewise cubic polynomial that passes
through every data point with continuous first and second derivatives (\\(C^2\\)
smoothness). On each interval \\([x_i, x_{i+1}]\\):

\\[
S_i(x) = a_i + b_i(x - x_i) + c_i(x - x_i)^2 + d_i(x - x_i)^3
\\]

Numra provides three boundary condition variants.

### Natural Spline

The second derivative is zero at both endpoints: \\(S''(x_0) = S''(x_{n-1}) = 0\\).
This is the most common default and produces no artificial curvature at the boundaries.

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
use numra::interp::CubicSpline;

let x = vec![0.0, 1.0, 2.0, 3.0, 4.0];
let y: Vec<f64> = x.iter().map(|&xi| xi.sin()).collect();

let cs = CubicSpline::natural(&x, &y).unwrap();

// Evaluate between knots
let val = cs.interpolate(0.5);
println!("sin(0.5) approx {:.6}, exact {:.6}", val, 0.5_f64.sin());

// First derivative
let deriv = cs.derivative(1.0).unwrap();
println!("cos(1.0) approx {:.6}, exact {:.6}", deriv, 1.0_f64.cos());

// Definite integral
let area = cs.integrate(0.0, 4.0).unwrap();
```

### Clamped Spline

Endpoint first derivatives are specified by the user:

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
// Clamped spline with exact endpoint derivatives
let x = vec![0.0, 1.0, 2.0, 3.0];
let y: Vec<f64> = x.iter().map(|&xi| xi.powi(3)).collect();

// f'(0) = 0, f'(3) = 27
let cs = CubicSpline::clamped(&x, &y, 0.0, 27.0).unwrap();

// Clamped with exact derivatives reproduces the cubic polynomial exactly
assert!((cs.interpolate(1.5) - 3.375).abs() < 1e-10);
```

### Not-a-Knot Spline

The third derivative is continuous at the second and second-to-last knots.
This eliminates the need for boundary conditions and reproduces cubic polynomials
exactly. Requires at least 4 data points; falls back to natural for fewer.

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
let x = vec![0.0, 1.0, 2.0, 3.0, 4.0];
let y: Vec<f64> = x.iter().map(|&xi| xi.powi(3) - 2.0 * xi).collect();

let cs = CubicSpline::not_a_knot(&x, &y).unwrap();

// Reproduces x^3 - 2x exactly at non-knot points
assert!((cs.interpolate(0.75) - (0.75_f64.powi(3) - 1.5)).abs() < 1e-10);
```

## Pchip -- Monotone Hermite Interpolation

The Piecewise Cubic Hermite Interpolating Polynomial (PCHIP) uses the
Fritsch-Carlson method to compute slopes that preserve the monotonicity of
the input data. Unlike cubic splines, PCHIP guarantees no overshoot or
undershoot: if the data is monotone increasing on an interval, the interpolant
will be too.

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
use numra::interp::Pchip;

// Monotone increasing data
let x = vec![0.0, 1.0, 2.0, 5.0, 10.0];
let y = vec![0.0, 1.0, 4.0, 5.0, 10.0];

let p = Pchip::new(&x, &y).unwrap();

// Guaranteed monotone between knots
let mut prev = p.interpolate(0.0);
for i in 1..100 {
    let xi = i as f64 * 10.0 / 100.0;
    let yi = p.interpolate(xi);
    assert!(yi >= prev - 1e-10);  // monotonicity preserved
    prev = yi;
}
```

PCHIP is the right choice when the shape of the data must be preserved.
For step-like data, it avoids the ringing artifacts that cubic splines produce:

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
// Step-like data: PCHIP will not overshoot past 0 or 1
let x = vec![0.0, 1.0, 1.5, 2.0, 3.0];
let y = vec![0.0, 0.0, 1.0, 1.0, 1.0];

let p = Pchip::new(&x, &y).unwrap();
// All interpolated values stay in [0, 1]
```

## Akima Interpolation

Akima interpolation uses a weighted average of adjacent secant slopes that is
robust to outliers in the data. While cubic splines use global information
(a tridiagonal solve), Akima slopes are computed locally, so a single outlier
affects only its immediate neighbors.

Requires at least 5 data points.

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
use numra::interp::Akima;

// Data with an outlier at x=2
let x = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
let y = vec![0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0];

let a = Akima::new(&x, &y).unwrap();

// Away from the outlier, interpolation is barely affected
let y_at_5 = a.interpolate(4.5);
assert!(y_at_5.abs() < 1.0);  // minimal overshoot far from outlier
```

Akima also reproduces linear data exactly:

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
let x = vec![0.0, 1.0, 2.0, 3.0, 4.0];
let y = vec![0.0, 1.0, 2.0, 3.0, 4.0];
let a = Akima::new(&x, &y).unwrap();
assert!((a.interpolate(2.7) - 2.7).abs() < 1e-12);
```

## Barycentric Lagrange Interpolation

`BarycentricLagrange` constructs the unique polynomial of degree \\(n-1\\)
passing through \\(n\\) data points, evaluated using the numerically stable
barycentric formula. Setup is \\(O(n^2)\\); each evaluation is \\(O(n)\\).

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
use numra::interp::BarycentricLagrange;

// Polynomial of degree 2: y = x^2
let x = vec![0.0, 1.0, 2.0];
let y = vec![0.0, 1.0, 4.0];

let p = BarycentricLagrange::new(&x, &y).unwrap();
assert!((p.interpolate(0.5) - 0.25).abs() < 1e-10);
assert!((p.interpolate(1.5) - 2.25).abs() < 1e-10);
```

### Chebyshev Node Construction

For interpolating a known function, Chebyshev nodes of the first kind minimize
the maximum interpolation error. The `chebyshev` constructor uses closed-form
barycentric weights for optimal numerical stability:

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
// Interpolate sin(x) on [0, pi] with 10 Chebyshev nodes
let p = BarycentricLagrange::chebyshev(
    |x: f64| x.sin(),
    0.0,
    std::f64::consts::PI,
    10,
);

// Sub-1e-7 accuracy across the interval
for k in 1..10 {
    let x = k as f64 * std::f64::consts::PI / 10.0;
    assert!((p.interpolate(x) - x.sin()).abs() < 1e-7);
}
```

Chebyshev nodes are essential for avoiding Runge's phenomenon -- the divergence
of polynomial interpolants on equidistant nodes for certain functions:

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
// Runge's function 1/(1+25x^2) on [-1, 1]
// Equidistant nodes would diverge; Chebyshev nodes converge
let p = BarycentricLagrange::chebyshev(
    |x: f64| 1.0 / (1.0 + 25.0 * x * x),
    -1.0, 1.0, 50,
);

assert!((p.interpolate(0.0) - 1.0).abs() < 1e-3);
assert!((p.interpolate(0.9) - 1.0 / (1.0 + 25.0 * 0.81)).abs() < 1e-4);
```

## Convenience Function -- `interp1d`

The `interp1d` function provides a quick way to create an interpolant using a
method selector enum. It returns a boxed `dyn Interpolant`:

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
use numra::interp::{interp1d, Interp1dMethod, Interpolant};

let x = vec![0.0, 1.0, 2.0, 3.0, 4.0];
let y = vec![0.0, 1.0, 0.5, 1.5, 1.0];

let interp = interp1d(&x, &y, Interp1dMethod::CubicSpline).unwrap();
println!("f(2.5) = {}", interp.interpolate(2.5));

// Switch methods without changing the rest of the code
let interp = interp1d(&x, &y, Interp1dMethod::Pchip).unwrap();
println!("f(2.5) = {}", interp.interpolate(2.5));
```

Available methods: `Interp1dMethod::Linear`, `CubicSpline` (natural),
`Pchip`, and `Akima`.

## Differentiation and Integration Through Interpolants

All piecewise cubic interpolants (CubicSpline, Pchip, Akima) and Linear
support analytical differentiation and integration:

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
use numra::interp::CubicSpline;

let x: Vec<f64> = (0..20).map(|i| i as f64 * 0.1).collect();
let y: Vec<f64> = x.iter().map(|&xi| xi.sin()).collect();
let cs = CubicSpline::natural(&x, &y).unwrap();

// Derivative: approximates cos(x)
let deriv = cs.derivative(0.5).unwrap();
println!("d/dx sin(0.5) approx {:.6}, exact {:.6}", deriv, 0.5_f64.cos());

// Integral: approximates -cos(b) + cos(a)
let area = cs.integrate(0.0, 1.9).unwrap();
let exact = -(1.9_f64.cos()) + 1.0;
println!("integral approx {:.6}, exact {:.6}", area, exact);
```

The integration is computed analytically from the piecewise cubic coefficients
using the antiderivative formula:

\\[
\int_{x_i}^{x_{i+1}} \left[a + bt + ct^2 + dt^3\right] dt = at + \frac{b}{2}t^2 + \frac{c}{3}t^3 + \frac{d}{4}t^4 \Bigg|_0^{h_i}
\\]

where \\(t = x - x_i\\).

## Method Comparison

| Method | Smoothness | Monotone | Min Points | Outlier Robust | Derivatives | Integration |
|--------|-----------|----------|------------|----------------|-------------|-------------|
| Linear | \\(C^0\\) | Yes | 2 | N/A | Yes | Yes |
| CubicSpline (natural) | \\(C^2\\) | No | 2 | No | Yes | Yes |
| CubicSpline (clamped) | \\(C^2\\) | No | 2 | No | Yes | Yes |
| CubicSpline (not-a-knot) | \\(C^2\\) | No | 4 | No | Yes | Yes |
| Pchip | \\(C^1\\) | Yes | 2 | Moderate | Yes | Yes |
| Akima | \\(C^1\\) | No | 5 | Yes | Yes | Yes |
| BarycentricLagrange | \\(C^{\infty}\\) | No | 1 | No | No | No |

### When to Use Each

- **Smooth data, no special requirements** -- Use `CubicSpline::natural` or
  `CubicSpline::not_a_knot`. The \\(C^2\\) smoothness gives the most visually
  pleasing curves.

- **Monotone data must stay monotone** -- Use `Pchip`. It is the only method
  that guarantees monotonicity preservation.

- **Data with outliers or noise** -- Use `Akima`. A single outlier affects
  only the two adjacent intervals, not the entire spline.

- **Known endpoint derivatives** -- Use `CubicSpline::clamped` with exact
  or estimated boundary slopes.

- **Interpolating a known function (not data)** -- Use
  `BarycentricLagrange::chebyshev` with Chebyshev nodes for near-optimal
  polynomial approximation.

- **Simple, fast, no surprises** -- Use `Linear`.

## Error Handling

All constructors return `Result<_, InterpError>`. Possible errors:

- `DimensionMismatch` -- x and y arrays have different lengths
- `TooFewPoints` -- fewer data points than the method requires
- `UnsortedData` -- x values are not strictly increasing

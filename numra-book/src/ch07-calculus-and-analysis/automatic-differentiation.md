# Automatic Differentiation

Automatic differentiation (AD) computes exact derivatives of computer programs
by systematically applying the chain rule. Unlike finite differences, AD
introduces no truncation error. Unlike symbolic differentiation, AD works on
arbitrary code without algebraic simplification.

Numra provides two AD modes:

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
use numra::autodiff::{
    Dual,                    // forward-mode dual numbers
    gradient,                // forward-mode gradient of f: R^n -> R
    jacobian,                // forward-mode Jacobian of f: R^n -> R^m
};
use numra::autodiff::reverse::{
    grad,                    // reverse-mode gradient of f: R^n -> R
    jacobian_reverse,        // reverse-mode Jacobian of f: R^n -> R^m
    hessian,                 // Hessian via reverse-mode + finite differences
    Var,                     // tape-tracked variable
};
```

## Forward Mode -- Dual Numbers

Forward-mode AD augments every value with a directional derivative. The `Dual<S>`
type carries a primal value and its derivative (the "epsilon" or tangent part):

\\[
\tilde{x} = x + x'\varepsilon, \quad \varepsilon^2 = 0
\\]

All arithmetic and transcendental operations propagate derivatives via the
chain rule automatically.

### Creating Dual Numbers

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
use numra::autodiff::Dual;

// A variable: value = 2.0, derivative = 1.0 (seeded for differentiation)
let x = Dual::variable(2.0_f64);

// A constant: value = 5.0, derivative = 0.0 (not being differentiated)
let c = Dual::constant(5.0_f64);

// Custom: value and derivative specified explicitly
let d = Dual::new(3.0, 0.5);  // value=3, deriv=0.5
```

### Differentiating a Single-Variable Function

To compute \\(f'(x)\\), create a dual variable and evaluate \\(f\\):

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
use numra::autodiff::Dual;

// f(x) = x^2 at x = 3
let x = Dual::variable(3.0_f64);
let y = x * x;
assert!((y.value() - 9.0).abs() < 1e-12);   // f(3) = 9
assert!((y.deriv() - 6.0).abs() < 1e-12);   // f'(3) = 2x = 6

// f(x) = sin(x^2) at x = 1
let x = Dual::variable(1.0_f64);
let y = (x * x).sin();
assert!((y.value() - 1.0_f64.sin()).abs() < 1e-12);
assert!((y.deriv() - 2.0 * 1.0_f64.cos()).abs() < 1e-12);  // 2x*cos(x^2)
```

### Supported Operations

`Dual<S>` supports all standard arithmetic (`+`, `-`, `*`, `/`, `+=`, etc.)
and a comprehensive set of transcendental functions:

| Function | Derivative Rule |
|----------|----------------|
| `sin(x)` | \\(\cos(x)\\) |
| `cos(x)` | \\(-\sin(x)\\) |
| `tan(x)` | \\(1/\cos^2(x)\\) |
| `exp(x)` | \\(e^x\\) |
| `ln(x)` | \\(1/x\\) |
| `sqrt(x)` | \\(1/(2\sqrt{x})\\) |
| `abs(x)` | \\(\text{sgn}(x)\\) |
| `powf(x, n)` | \\(n \cdot x^{n-1}\\) |
| `powi(x, n)` | \\(n \cdot x^{n-1}\\) |
| `asin(x)` | \\(1/\sqrt{1-x^2}\\) |
| `acos(x)` | \\(-1/\sqrt{1-x^2}\\) |
| `atan(x)` | \\(1/(1+x^2)\\) |
| `sinh(x)` | \\(\cosh(x)\\) |
| `cosh(x)` | \\(\sinh(x)\\) |
| `tanh(x)` | \\(1 - \tanh^2(x)\\) |
| `powf_dual(x, y)` | \\(x^y \cdot (y'\ln x + y \cdot x'/x)\\) |

The `powf_dual` method handles the case where both base and exponent are dual
numbers (i.e., both depend on the differentiation variable).

### Complex Expressions

The chain rule is applied automatically through operator composition:

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
use numra::autodiff::Dual;

// Gaussian kernel: f(x) = exp(-x^2 / 2)
// f'(x) = -x * exp(-x^2 / 2)
let x = Dual::variable(1.0_f64);
let half = Dual::constant(0.5);
let y = (-(x * x) * half).exp();

assert!((y.value() - (-0.5_f64).exp()).abs() < 1e-12);
assert!((y.deriv() - (-(-0.5_f64).exp())).abs() < 1e-12);
```

## Forward-Mode Gradient and Jacobian

### `gradient` -- Gradient of \\(f: \mathbb{R}^n \to \mathbb{R}\\)

The `gradient` function computes \\(\nabla f\\) by evaluating \\(f\\) once per
input dimension, each time seeding the derivative along a different coordinate
direction:

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
use numra::autodiff::{Dual, gradient};

// f(x) = x0^2 + 2*x1^2 + x0*x1
// grad f = [2*x0 + x1, 4*x1 + x0]
let grad = gradient(
    |x: &[Dual<f64>]| {
        let two = Dual::constant(2.0);
        x[0] * x[0] + two * x[1] * x[1] + x[0] * x[1]
    },
    &[1.0, 2.0],
);
assert!((grad[0] - 4.0).abs() < 1e-12);  // 2*1 + 2 = 4
assert!((grad[1] - 9.0).abs() < 1e-12);  // 4*2 + 1 = 9
```

**Cost:** \\(n\\) evaluations of \\(f\\) for \\(n\\) input variables.

### `jacobian` -- Jacobian of \\(F: \mathbb{R}^n \to \mathbb{R}^m\\)

The `jacobian` function computes the \\(m \times n\\) Jacobian matrix in
row-major format:

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
use numra::autodiff::{Dual, jacobian};

// F(x) = [x0 + x1, x0 * x1]
// J = [[1, 1], [x1, x0]]
let jac = jacobian(
    |x: &[Dual<f64>], out: &mut [Dual<f64>]| {
        out[0] = x[0] + x[1];
        out[1] = x[0] * x[1];
    },
    &[2.0, 3.0],
    2,  // number of outputs
);
// J[i*n + j] = df_i / dx_j
assert!((jac[0] - 1.0).abs() < 1e-12);  // df0/dx0 = 1
assert!((jac[1] - 1.0).abs() < 1e-12);  // df0/dx1 = 1
assert!((jac[2] - 3.0).abs() < 1e-12);  // df1/dx0 = x1 = 3
assert!((jac[3] - 2.0).abs() < 1e-12);  // df1/dx1 = x0 = 2
```

**Cost:** \\(n\\) evaluations of \\(F\\) for \\(n\\) input variables, regardless
of the number of outputs \\(m\\).

## Reverse Mode -- Tape-Based AD

Reverse-mode AD records the computation in a forward pass on a "tape" (directed
acyclic graph), then computes all partial derivatives in a single backward pass.
This is the same algorithm behind backpropagation in neural networks.

### Basic Usage with `Var`

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
use numra::autodiff::reverse::{Var, grad};
use numra::autodiff::tape::Tape;

let tape = Tape::new();
let x = Tape::var(&tape, 2.0);
let y = Tape::var(&tape, 3.0);

// z = x * y + x^2
let z = x.clone() * y.clone() + x.clone() * x.clone();
// dz/dx = y + 2x = 3 + 4 = 7
// dz/dy = x = 2

let g = Tape::gradient(&tape, &z);
assert!((g[0] - 7.0).abs() < 1e-12);
assert!((g[1] - 2.0).abs() < 1e-12);
```

### `reverse::grad` -- Convenience Gradient

The `grad` function handles tape creation automatically:

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
use numra::autodiff::reverse::grad;

// Rosenbrock function at (1, 1) -- gradient should be (0, 0)
let g = grad(
    |x| {
        let a = x[0].cst(1.0) - x[0].clone();         // 1 - x
        let b = x[1].clone() - x[0].clone() * x[0].clone();  // y - x^2
        a.clone() * a + x[0].cst(100.0) * b.clone() * b
    },
    &[1.0, 1.0],
);
assert!(g[0].abs() < 1e-10);
assert!(g[1].abs() < 1e-10);
```

Note the use of `.clone()` -- `Var` uses reference counting internally and
does not implement `Copy`, so values that are used more than once must be cloned.

Use `x.cst(value)` to create a constant on the same tape as `x`.

### `reverse::jacobian_reverse`

Computes the Jacobian of a vector-valued function by performing one backward
pass per output:

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
use numra::autodiff::reverse::jacobian_reverse;

// Rotation: F(x,y) = (x*cos(t) - y*sin(t), x*sin(t) + y*cos(t))
let theta = 0.5_f64;
let jac = jacobian_reverse(
    |v| {
        let ct = v[0].cst(theta.cos());
        let st = v[0].cst(theta.sin());
        vec![
            &(&v[0] * &ct) - &(&v[1] * &st),
            &(&v[0] * &st) + &(&v[1] * &ct),
        ]
    },
    &[1.0, 0.0],
);
// Jacobian is the rotation matrix [[cos, -sin], [sin, cos]]
assert!((jac[0][0] - theta.cos()).abs() < 1e-12);
assert!((jac[0][1] - (-theta.sin())).abs() < 1e-12);
```

### `reverse::hessian`

Computes the Hessian matrix by finite-differencing the reverse-mode gradient.
This is efficient because each gradient computation is \\(O(1)\\) backward
passes regardless of dimension:

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
use numra::autodiff::reverse::hessian;

// f(x,y) = x^2 + 2*x*y + 3*y^2
// Hessian = [[2, 2], [2, 6]]
let h = hessian(
    |x| {
        &x[0] * &x[0]
            + x[0].cst(2.0) * &x[0] * &x[1]
            + x[0].cst(3.0) * &x[1] * &x[1]
    },
    &[1.0, 1.0],
);
assert!((h[0][0] - 2.0).abs() < 1e-5);
assert!((h[0][1] - 2.0).abs() < 1e-5);
assert!((h[1][0] - 2.0).abs() < 1e-5);
assert!((h[1][1] - 6.0).abs() < 1e-5);
```

### Supported Operations on `Var`

`Var` supports all standard arithmetic (`+`, `-`, `*`, `/`, negation) for both
`Var + Var` and `Var + f64` combinations, plus these transcendentals:

`sin`, `cos`, `tan`, `exp`, `ln`, `sqrt`, `abs`, `tanh`, `sinh`, `cosh`,
`asin`, `acos`, `atan`, `powf`, `powi`, `pow` (Var exponent).

## Forward vs Reverse: When to Use Which

The key difference is computational cost:

| | Forward Mode | Reverse Mode |
|---|---|---|
| **Cost** | \\(O(n)\\) passes | \\(O(m)\\) passes |
| **Best when** | \\(m \gg n\\) (few inputs, many outputs) | \\(n \gg m\\) (many inputs, few outputs) |
| **Memory** | \\(O(1)\\) extra per variable | \\(O(T)\\) tape size (all operations stored) |
| **Type** | Generic over `S: Scalar` (f32/f64) | `f64` only |
| **API** | `Dual<S>` (implements `Copy`) | `Var` (requires `.clone()`) |

### Decision Guidelines

- **Gradient of a scalar loss function** (optimization, machine learning):
  Use `reverse::grad`. One backward pass gives all \\(n\\) partial derivatives.

- **Jacobian of a system with few inputs** (e.g., 2D or 3D transformations):
  Use forward-mode `jacobian`. With \\(n \le 5\\), the overhead of the tape
  is not justified.

- **Single derivative** \\(f'(x)\\): Use `Dual::variable(x)` directly.
  Simplest approach, no allocation.

- **Hessian**: Use `reverse::hessian`, which combines reverse-mode gradients
  with finite differences. For an \\(n \times n\\) Hessian this costs
  \\(O(n)\\) backward passes, much cheaper than forward-mode's \\(O(n^2)\\).

### Computational Cost Analysis

Consider computing the gradient of \\(f: \mathbb{R}^n \to \mathbb{R}\\):

**Forward mode:** Each evaluation propagates one directional derivative.
To get the full gradient, evaluate \\(f\\) once per input dimension. Total cost:
\\(n \times \text{cost}(f)\\).

**Reverse mode:** The forward pass records the computation graph. A single
backward pass computes all \\(n\\) partial derivatives. Total cost:
\\(\text{cost}(f) + \text{cost}(\text{backward}) \approx 3\text{-}5 \times \text{cost}(f)\\),
independent of \\(n\\).

For \\(n = 1000\\) parameters and a scalar output, reverse mode is roughly
200--300x faster than forward mode.

## Integration with Curve Fitting

The autodiff module integrates with `numra::fit` through bridge functions
that convert AD-computed Jacobians into the format expected by the
Levenberg-Marquardt optimizer:

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
use numra::autodiff::{Dual, gradient};
use numra::fit::{curve_fit_with_jacobian, FitResult};

// Exponential decay: y = a * exp(-b * x)
// Analytical Jacobian from AD knowledge:
// dm/da = exp(-b*x), dm/db = -a*x*exp(-b*x)
let result = curve_fit_with_jacobian(
    |x: f64, p: &[f64]| p[0] * (-p[1] * x).exp(),
    |x: f64, p: &[f64]| {
        let e = (-p[1] * x).exp();
        vec![e, -p[0] * x * e]  // [dm/da, dm/db]
    },
    &x_data, &y_data,
    &[4.0, 0.3],  // initial guess
    None,
).unwrap();
```

This avoids the finite-difference Jacobian approximation used by `curve_fit`,
giving faster convergence and exact derivatives.

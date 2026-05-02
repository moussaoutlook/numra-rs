# Error Propagation

When your inputs are uncertain, your outputs are uncertain too. Numra's
`Uncertain<S>` type tracks this automatically using first-order (linear)
uncertainty propagation.

## The Uncertain Type

An `Uncertain<S>` stores a mean value and its variance:

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
use numra::Uncertain;

// A measurement: 10.0 with standard deviation 0.5
let x = Uncertain::<f64>::from_std(10.0, 0.5);
println!("Mean: {}, Std: {}", x.mean, x.std());

// Or specify variance directly
let y = Uncertain::new(5.0, 0.25);  // mean=5, variance=0.25, std=0.5

// A certain (exact) value
let exact = Uncertain::certain(3.14);
```

## How Propagation Works

For a function \\( y = f(x) \\) where \\( x \\) has variance \\( \sigma_x^2 \\):

\\[
\sigma_y^2 \approx \left(\frac{\partial f}{\partial x}\right)^2 \sigma_x^2
\\]

This is a first-order Taylor expansion -- accurate when the uncertainty is small
relative to the scale over which \\( f \\) changes.

## Arithmetic Operations

All basic operations propagate uncertainty assuming independence:

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
use numra::Uncertain;

let a = Uncertain::<f64>::from_std(10.0, 1.0);
let b = Uncertain::<f64>::from_std(5.0, 0.5);

// Addition: Var(a+b) = Var(a) + Var(b)
let sum = a.add(&b);        // mean=15, var=1.25

// Subtraction: Var(a-b) = Var(a) + Var(b)
let diff = a.sub(&b);       // mean=5, var=1.25

// Multiplication: Var(ab) ≈ b²Var(a) + a²Var(b)
let prod = a.mul(&b);       // mean=50, var=125

// Division: Var(a/b) ≈ (1/b²)Var(a) + (a/b²)²Var(b)
let quot = a.div(&b);       // mean=2

// Scaling by constant: Var(ca) = c²Var(a)
let scaled = a.scale(2.0);  // mean=20, var=4.0

// Adding a constant: Var(a+c) = Var(a)
let shifted = a.add_const(100.0);  // mean=110, var=1.0
```

## Mathematical Functions

Common functions with built-in propagation:

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
let x = Uncertain::<f64>::from_std(1.0, 0.1);

let y_sq   = x.square();    // y = x², σ_y ≈ |2x| * σ_x
let y_sqrt = x.sqrt_unc();  // y = √x, σ_y ≈ σ_x / (2√x)
let y_exp  = x.exp();       // y = e^x, σ_y ≈ e^x * σ_x
let y_ln   = x.ln();        // y = ln(x), σ_y ≈ σ_x / x
let y_sin  = x.sin();       // y = sin(x), σ_y ≈ |cos(x)| * σ_x
let y_cos  = x.cos();       // y = cos(x), σ_y ≈ |sin(x)| * σ_x
```

## Custom Functions

Apply any function with a known derivative:

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
let x = Uncertain::<f64>::from_std(2.0, 0.1);

// y = x^3, dy/dx = 3x^2
let y = x.apply(|x| x * x * x, |x| 3.0 * x * x);
// mean = 8.0, σ_y ≈ 3*4 * 0.1 = 1.2
```

## Coefficient of Variation

The CV measures relative uncertainty:

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
let x = Uncertain::<f64>::from_std(100.0, 5.0);
println!("CV = {:.1}%", x.cv() * 100.0);  // 5%
```

## When Propagation Breaks Down

First-order propagation assumes:

| Assumption | Violated when |
|------------|--------------|
| Small uncertainty | \\( \sigma_x / x \\) is large |
| Linear approximation | \\( f \\) is highly nonlinear near \\( x \\) |
| Independence | Inputs are correlated |

For large uncertainties or strongly nonlinear functions, use Monte Carlo
simulation instead (see [Monte Carlo with ODEs](./monte-carlo-odes.md)).

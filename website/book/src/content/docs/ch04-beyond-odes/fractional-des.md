---
title: "Fractional Differential Equations"
---

Fractional differential equations (FDEs) replace integer-order derivatives with
fractional-order derivatives, enabling the modeling of systems with **memory effects**
and **anomalous dynamics**. They appear in viscoelastic materials, anomalous diffusion,
electrochemistry, and biological tissue modeling. Numra provides the `numra-fde` crate
with an L1 scheme solver for Caputo fractional derivatives.

## The Caputo Fractional Derivative

The Caputo fractional derivative of order $\alpha \in (0, 1]$ is defined as:

$$
{}^C D^\alpha y(t) = \frac{1}{\Gamma(1-\alpha)} \int_0^t (t-s)^{-\alpha} \, y'(s) \, ds
$$

where $\Gamma$ is the gamma function. Key properties:

| Property | Caputo Derivative |
|---|---|
| Constant rule | ${}^C D^\alpha c = 0$ (like ordinary derivatives) |
| Power rule | ${}^C D^\alpha t^n = \frac{\Gamma(n+1)}{\Gamma(n+1-\alpha)} t^{n-\alpha}$ for $n \geq 1$ |
| Integer limit | ${}^C D^1 y(t) = y'(t)$ (reduces to standard derivative) |
| Memory | Depends on entire history of $y$ from $0$ to $t$ |
| Initial conditions | Standard: $y(0) = y_0$ (same as ODEs) |

### Why Caputo over Riemann-Liouville?

The Caputo derivative is preferred for physical modeling because:
- It preserves the meaning of initial conditions ($y(0) = y_0$ rather than
  fractional integrals of $y$ at $t=0$).
- The Caputo derivative of a constant is zero, matching physical intuition.
- Standard initial value problem formulations carry over directly.

### The Memory Effect

The defining feature of fractional derivatives is **non-locality in time**. The integral
in the Caputo definition means the derivative at time $t$ depends on the entire
history $y(s)$ for $s \in [0, t]$. This "long memory" effect distinguishes FDEs
from both ODEs (no memory) and DDEs (finite memory at discrete delay points).

Physically, this models:
- **Viscoelastic materials**: stress depends on the entire strain history
- **Anomalous diffusion**: particles with power-law waiting times
- **Fractional relaxation**: decay slower than exponential

## The Mittag-Leffler Function

The Mittag-Leffler function $E_{\alpha}(z)$ generalizes the exponential and is the
natural solution to fractional relaxation equations:

$$
E_{\alpha}(z) = \sum_{k=0}^{\infty} \frac{z^k}{\Gamma(\alpha k + 1)}
$$

When $\alpha = 1$, this reduces to $E_1(z) = e^z$. For $\alpha < 1$, the
Mittag-Leffler function decays as a power law for large arguments:

$$
E_{\alpha}(-t^\alpha) \sim \frac{t^{-\alpha}}{\Gamma(1-\alpha)}, \quad t \to \infty
$$

This is slower than exponential decay, capturing the "long tail" behavior of fractional
systems. Numra provides `mittag_leffler_1(z, alpha)` for numerical evaluation.

## The FdeSystem Trait

Every FDE in Numra implements the `FdeSystem` trait:

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust
pub trait FdeSystem<S: Scalar> {
    /// Dimension of the state space.
    fn dim(&self) -> usize;

    /// Fractional order alpha in (0, 1].
    fn alpha(&self) -> S;

    /// Evaluate the right-hand side f(t, y).
    fn rhs(&self, t: S, y: &[S], f: &mut [S]);

    /// Check if the order is valid (0 < alpha <= 1).
    fn is_valid_order(&self) -> bool { ... }
}
```

The FDE solved is:

$$
{}^C D^\alpha y(t) = f(t, y), \quad y(0) = y_0
$$

## The L1 Scheme Solver

Numra uses the L1 scheme to discretize the Caputo derivative. The L1 approximation is:

$$
{}^C D^\alpha y(t_n) \approx \frac{\Delta t^{-\alpha}}{\Gamma(2-\alpha)}
\sum_{j=0}^{n-1} b_j \left( y(t_{n-j}) - y(t_{n-j-1}) \right)
$$

where the weights $b_j = (j+1)^{1-\alpha} - j^{1-\alpha}$ encode the memory kernel.

### Convergence

The L1 scheme has convergence order $O(\Delta t^{2-\alpha})$:

| $\alpha$ | Convergence Order | Effective for $\Delta t = 0.01$ |
|---|---|---|
| 0.3 | $O(\Delta t^{1.7})$ | Error ~ $10^{-3.4}$ |
| 0.5 | $O(\Delta t^{1.5})$ | Error ~ $10^{-3.0}$ |
| 0.8 | $O(\Delta t^{1.2})$ | Error ~ $10^{-2.4}$ |
| 1.0 | $O(\Delta t^{1.0})$ | Error ~ $10^{-2.0}$ (reduces to Euler) |

As $\alpha \to 0$, the scheme becomes more accurate; as $\alpha \to 1$, it
approaches first-order Euler accuracy.

### Computational Cost

Because of the memory effect, the L1 scheme at step $n$ must access all previous
solution values $y_0, y_1, \ldots, y_{n-1}$. This makes the total cost
$O(N^2)$ for $N$ time steps, compared to $O(N)$ for standard ODE methods.
The memory requirement is also $O(N)$ to store the full solution history.

## Solver Options

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust
use numra_fde::FdeOptions;

let opts = FdeOptions::default()
    .dt(0.01)           // Time step
    .max_steps(100_000) // Maximum number of steps
    .tol(1e-10);        // Tolerance for fixed-point iteration
```

| Option | Default | Description |
|---|---|---|
| `dt` | 0.01 | Time step size |
| `max_steps` | 100,000 | Maximum number of steps |
| `tol` | 1e-10 | Convergence tolerance for implicit iteration |
| `max_iter` | 100 | Maximum fixed-point iterations per step |

## Example: Fractional Relaxation

The fractional relaxation equation generalizes exponential decay:

$$
{}^C D^\alpha y = -\lambda y, \quad y(0) = 1
$$

The exact solution is $y(t) = E_\alpha(-\lambda t^\alpha)$, where $E_\alpha$ is
the Mittag-Leffler function.

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust
use numra_fde::{FdeSystem, L1Solver, FdeSolver, FdeOptions};

struct FractionalRelaxation {
    lambda: f64,
    alpha: f64,
}

impl FdeSystem<f64> for FractionalRelaxation {
    fn dim(&self) -> usize { 1 }
    fn alpha(&self) -> f64 { self.alpha }
    fn rhs(&self, _t: f64, y: &[f64], f: &mut [f64]) {
        f[0] = -self.lambda * y[0];
    }
}

// Half-order derivative: slower than exponential decay
let system = FractionalRelaxation { lambda: 1.0, alpha: 0.5 };
let opts = FdeOptions::default().dt(0.01);

let result = L1Solver::solve(&system, 0.0, 5.0, &[1.0], &opts)
    .expect("Solve failed");

// Compare numerical and exact solutions
use numra_fde::mittag_leffler_1;
for (i, &t) in result.t.iter().enumerate().step_by(50) {
    let y_numerical = result.y_at(i)[0];
    let y_exact = mittag_leffler_1(-t.powf(0.5), 0.5);
    println!("t = {:.2}: numerical = {:.6}, exact = {:.6}", t, y_numerical, y_exact);
}
```

### Comparing Different Fractional Orders

The fractional order controls how fast the system "forgets":

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust
use numra_fde::{FdeSystem, L1Solver, FdeSolver, FdeOptions};

struct Relaxation { alpha: f64 }

impl FdeSystem<f64> for Relaxation {
    fn dim(&self) -> usize { 1 }
    fn alpha(&self) -> f64 { self.alpha }
    fn rhs(&self, _t: f64, y: &[f64], f: &mut [f64]) {
        f[0] = -y[0];
    }
}

let opts = FdeOptions::default().dt(0.01);

// Compare alpha = 0.3, 0.5, 0.8, 1.0
for &alpha in &[0.3, 0.5, 0.8, 1.0] {
    let system = Relaxation { alpha };
    let result = L1Solver::solve(&system, 0.0, 5.0, &[1.0], &opts)
        .expect("Solve failed");

    let y_final = result.y_final().unwrap()[0];
    println!("alpha = {:.1}: y(5) = {:.6}", alpha, y_final);
}
// Output shows: smaller alpha => slower decay (longer memory)
```

## Example: Fractional Viscoelasticity

A fractional Maxwell model for viscoelastic stress relaxation:

$$
{}^C D^\alpha \sigma(t) = E \, {}^C D^\alpha \varepsilon(t) - \frac{E}{\eta} \sigma(t)
$$

For a step strain $\varepsilon(t) = \varepsilon_0 H(t)$, this simplifies to:

$$
{}^C D^\alpha \sigma = -\frac{E}{\eta} \sigma, \quad \sigma(0) = E \varepsilon_0
$$

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust
use numra_fde::{FdeSystem, L1Solver, FdeSolver, FdeOptions};

struct FractionalMaxwell {
    modulus: f64,       // E: elastic modulus
    viscosity: f64,     // eta: viscosity
    alpha: f64,         // fractional order
}

impl FdeSystem<f64> for FractionalMaxwell {
    fn dim(&self) -> usize { 1 }
    fn alpha(&self) -> f64 { self.alpha }
    fn rhs(&self, _t: f64, y: &[f64], f: &mut [f64]) {
        f[0] = -(self.modulus / self.viscosity) * y[0];
    }
}

let model = FractionalMaxwell {
    modulus: 1.0,
    viscosity: 10.0,
    alpha: 0.7,  // Sub-exponential relaxation
};

let sigma_0 = 1.0;  // Initial stress = E * epsilon_0
let opts = FdeOptions::default().dt(0.01);

let result = L1Solver::solve(&model, 0.0, 50.0, &[sigma_0], &opts)
    .expect("Solve failed");

// Stress relaxation follows Mittag-Leffler decay
for (t, y) in result.iter().step_by(100) {
    println!("t = {:.1}, stress = {:.6}", t, y[0]);
}
```

## Multi-Dimensional Systems

FDEs with multiple components sharing the same fractional order:

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust
use numra_fde::{FdeSystem, L1Solver, FdeSolver, FdeOptions};

struct TwoDFractional;

impl FdeSystem<f64> for TwoDFractional {
    fn dim(&self) -> usize { 2 }
    fn alpha(&self) -> f64 { 0.8 }
    fn rhs(&self, _t: f64, y: &[f64], f: &mut [f64]) {
        f[0] = -y[0] + 0.5 * y[1];
        f[1] = -2.0 * y[1];
    }
}

let opts = FdeOptions::default().dt(0.01);
let result = L1Solver::solve(&TwoDFractional, 0.0, 5.0, &[1.0, 1.0], &opts)
    .expect("Solve failed");

let y_final = result.y_final().unwrap();
// Component 2 decays faster (larger coefficient)
assert!(y_final[0] > y_final[1]);
```

## Working with Results

`FdeResult` uses the standard Numra result format:

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust
let result = L1Solver::solve(&system, t0, tf, &y0, &opts)?;

// Final state
let y_final = result.y_final().unwrap();

// Access at index
let y_i = result.y_at(50);

// Time points
let t = &result.t;

// Statistics
println!("RHS evaluations: {}", result.stats.n_rhs);
println!("Steps taken: {}", result.stats.n_steps);
```

## Practical Considerations

### Choosing the Time Step

Unlike adaptive ODE solvers, the L1 scheme uses a fixed time step. Guidelines:

- $\Delta t = 0.01$ is a good starting point for most problems
- Reduce $\Delta t$ for higher accuracy (convergence is $O(\Delta t^{2-\alpha})$)
- For $\alpha$ close to 0, even coarse steps can be accurate
- For $\alpha$ close to 1, you may need finer steps

### Long-Time Integration

Because of $O(N^2)$ cost, very long integrations can become slow. Strategies:
- Use the coarsest $\Delta t$ that meets accuracy requirements
- For problems where only the final state matters, reduce `max_steps` accordingly
- Future versions may support fast summation techniques ($O(N \log N)$) for the
  history weights

### Validating Fractional Order

The solver rejects orders outside $(0, 1]$:

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust
// This will return an error
let system = MySystem { alpha: 1.5 };  // Invalid: alpha > 1
let result = L1Solver::solve(&system, 0.0, 1.0, &[1.0], &opts);
assert!(result.is_err());
```

### Relationship to Other DE Types

- When $\alpha = 1$: FDE reduces to an ODE (use `numra-ode` for better performance)
- When $0 < \alpha < 1$: True fractional dynamics with power-law memory
- FDEs with power-law kernels are closely related to Volterra integro-differential
  equations (see the [IDE chapter](integro-des))

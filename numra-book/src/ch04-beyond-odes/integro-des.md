# Integro-Differential Equations

Integro-differential equations (IDEs) combine differential and integral operators,
modeling systems where the rate of change depends not only on the current state but on
a weighted accumulation of the entire past. They appear in viscoelasticity (hereditary
materials), population dynamics with age structure, heat conduction with memory, and
radiative transfer. Numra provides the `numra-ide` crate with quadrature-based Volterra
solvers and an efficient Prony series solver for sum-of-exponentials kernels.

## Mathematical Formulation

A Volterra integro-differential equation has the form:

\\[
y'(t) = f(t, y) + \int_0^t K(t, s, y(s)) \, ds, \quad y(0) = y_0
\\]

where:
- \\(f(t, y)\\) is the **local** (instantaneous) right-hand side,
- \\(K(t, s, y(s))\\) is the **memory kernel** encoding history dependence,
- the integral accumulates contributions from the entire past \\(s \in [0, t]\\).

### Convolution Kernels

A kernel is a **convolution kernel** if it depends only on the time difference:

\\[
K(t, s, y(s)) = \hat{K}(t-s) \cdot h(y(s))
\\]

Convolution kernels can be computed more efficiently and are the most common type in
physical applications.

### Comparison with Other Memory-Dependent DEs

| Equation Type | Memory Structure | Computational Cost |
|---|---|---|
| ODE | No memory | \\(O(N)\\) total |
| DDE | Discrete delay points | \\(O(N)\\) total |
| FDE | Power-law kernel (Caputo) | \\(O(N^2)\\) total |
| IDE | General kernel | \\(O(N^2)\\) total |
| IDE (Prony) | Sum of exponentials | \\(O(N)\\) total |

## The IdeSystem Trait

Every IDE in Numra implements the `IdeSystem` trait:

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
pub trait IdeSystem<S: Scalar> {
    /// Dimension of the state space.
    fn dim(&self) -> usize;

    /// Evaluate the local right-hand side f(t, y).
    fn rhs(&self, t: S, y: &[S], f: &mut [S]);

    /// Evaluate the memory kernel K(t, s, y(s)).
    fn kernel(&self, t: S, s: S, y_s: &[S], k: &mut [S]);

    /// Whether K depends only on (t-s), not t and s separately.
    fn is_convolution_kernel(&self) -> bool { false }
}
```

The `rhs` method evaluates the non-integral part, and `kernel` evaluates the integrand
at a specific past time \\(s\\) given the solution \\(y(s)\\) at that time.

## Memory Kernels

Numra provides several built-in kernel types via the `Kernel` trait:

### Exponential Kernel

\\[
K(\tau) = a \, e^{-b\tau}
\\]

Models the Maxwell element in viscoelasticity. Memory decays exponentially.

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
use numra_ide::kernels::ExponentialKernel;

let kernel = ExponentialKernel::new(2.0, 0.5);
// K(0) = 2.0, K(2) = 2 * exp(-1) ~ 0.736
```

### Power-Law Kernel

\\[
K(\tau) = a \, \tau^{-\alpha}
\\]

Models fractional memory effects. Singular at \\(\tau = 0\\) (regularized to zero in
Numra). For \\(0 < \alpha < 1\\), this is a weakly singular kernel.

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
use numra_ide::kernels::PowerLawKernel;

let kernel = PowerLawKernel::new(1.0, 0.5);
// K(1) = 1.0, K(4) = 0.5
```

### Prony Series (Sum of Exponentials)

\\[
K(\tau) = \sum_{i=1}^{m} a_i \, e^{-b_i \tau}
\\]

The most computationally efficient kernel type because the integral can be updated
recursively without storing the full history.

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
use numra_ide::kernels::PronyKernel;

// Two-term generalized Maxwell model
let kernel = PronyKernel::two_term(1.0, 0.5, 2.0, 1.0);
// K(0) = 1 + 2 = 3, K(2) = exp(-1) + 2*exp(-2) ~ 0.639
```

### Mittag-Leffler Kernel

\\[
K(\tau) = \tau^{\beta-1} \, E_{\alpha,\beta}(-\lambda \tau^\alpha)
\\]

Generalizes both exponential (\\(\alpha = \beta = 1\\)) and power-law kernels.

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
use numra_ide::kernels::MittagLefflerKernel;

let kernel = MittagLefflerKernel::relaxation(1.0, 0.5);
```

## Solvers

### VolterraSolver (Trapezoidal + Euler)

The basic solver uses composite trapezoidal quadrature for the integral and explicit
Euler for the time step:

\\[
y_{n+1} = y_n + \Delta t \left[ f(t_n, y_n) + \int_0^{t_{n+1}} K(t_{n+1}, s, y(s)) \, ds \right]
\\]

The integral is approximated with the trapezoidal rule over all stored history points.

- **Time stepping**: Explicit Euler
- **Quadrature**: Composite trapezoidal
- **Cost per step**: \\(O(n)\\) kernel evaluations
- **Total cost**: \\(O(N^2)\\)

### VolterraRK4Solver (Trapezoidal + RK4)

An improved solver that replaces explicit Euler with classical 4th-order Runge-Kutta
for the time-stepping:

- **Time stepping**: Classical RK4
- **Quadrature**: Composite trapezoidal
- **Accuracy**: Significantly better than Euler for the same step size
- **Cost per step**: \\(4 \times O(n)\\) kernel evaluations (four RK stages)

### PronySolver

For kernels that are sums of exponentials, the `PronySolver` achieves \\(O(1)\\) cost
per step by recursively updating auxiliary variables. The key insight is that for

\\[
I(t) = \int_0^t a \, e^{-b(t-s)} y(s) \, ds
\\]

the integral satisfies \\(I'(t) = a \, y(t) - b \, I(t)\\), converting the IDE into an
augmented ODE system.

## Solver Options

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
use numra_ide::IdeOptions;

let opts = IdeOptions::default()
    .dt(0.01)            // Time step
    .max_steps(100_000)  // Maximum number of steps
    .tol(1e-10)          // Tolerance for implicit iteration
    .quad_points(4);     // Quadrature points per step
```

| Option | Default | Description |
|---|---|---|
| `dt` | 0.01 | Time step size |
| `max_steps` | 100,000 | Maximum number of steps |
| `tol` | 1e-10 | Convergence tolerance |
| `max_iter` | 100 | Maximum iterations per step |
| `quad_points` | 4 | Gauss-Legendre quadrature points per subinterval |

## Example: Exponential Memory Decay

A system with exponential memory kernel modeling a viscoelastic material:

\\[
y'(t) = -k \, y(t) + \int_0^t e^{-(t-s)} y(s) \, ds, \quad y(0) = 1
\\]

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
use numra_ide::{IdeSystem, VolterraSolver, IdeSolver, IdeOptions};

struct ExponentialMemory { k: f64 }

impl IdeSystem<f64> for ExponentialMemory {
    fn dim(&self) -> usize { 1 }

    fn rhs(&self, _t: f64, y: &[f64], f: &mut [f64]) {
        f[0] = -self.k * y[0];
    }

    fn kernel(&self, t: f64, s: f64, y_s: &[f64], k: &mut [f64]) {
        k[0] = (-(t - s)).exp() * y_s[0];
    }

    fn is_convolution_kernel(&self) -> bool { true }
}

let system = ExponentialMemory { k: 1.0 };
let opts = IdeOptions::default().dt(0.01);

let result = VolterraSolver::solve(&system, 0.0, 5.0, &[1.0], &opts)
    .expect("Solve failed");

for (i, &t) in result.t.iter().enumerate().step_by(50) {
    println!("t = {:.2}, y = {:.6}", t, result.y_at(i)[0]);
}
```

### Higher Accuracy with RK4

For the same problem with improved time-stepping accuracy:

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
use numra_ide::{IdeSystem, VolterraRK4Solver, IdeSolver, IdeOptions};

// Same system definition as above...

let opts = IdeOptions::default().dt(0.05);  // Can use larger step with RK4

let result = VolterraRK4Solver::solve(&system, 0.0, 5.0, &[1.0], &opts)
    .expect("Solve failed");
```

## Example: Viscoelastic Material with Prony Kernel

For a generalized Maxwell model with two relaxation times:

\\[
y'(t) = -y(t) + \int_0^t \left[ a_1 e^{-b_1(t-s)} + a_2 e^{-b_2(t-s)} \right] y(s) \, ds
\\]

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
use numra_ide::{IdeSystem, VolterraSolver, IdeSolver, IdeOptions};
use numra_ide::kernels::PronyKernel;

struct ViscoelasticMaterial {
    kernel: PronyKernel<f64>,
}

impl IdeSystem<f64> for ViscoelasticMaterial {
    fn dim(&self) -> usize { 1 }

    fn rhs(&self, _t: f64, y: &[f64], f: &mut [f64]) {
        f[0] = -y[0];
    }

    fn kernel(&self, t: f64, s: f64, y_s: &[f64], k: &mut [f64]) {
        use numra_ide::Kernel;
        k[0] = self.kernel.evaluate(t - s) * y_s[0];
    }

    fn is_convolution_kernel(&self) -> bool { true }
}

let material = ViscoelasticMaterial {
    kernel: PronyKernel::two_term(1.0, 0.5, 0.5, 2.0),
};
let opts = IdeOptions::default().dt(0.01);

let result = VolterraSolver::solve(&material, 0.0, 10.0, &[1.0], &opts)
    .expect("Solve failed");
```

## Multi-Dimensional Systems

IDEs with coupled components:

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
use numra_ide::{IdeSystem, VolterraSolver, IdeSolver, IdeOptions};

struct CoupledIDE;

impl IdeSystem<f64> for CoupledIDE {
    fn dim(&self) -> usize { 2 }

    fn rhs(&self, _t: f64, y: &[f64], f: &mut [f64]) {
        f[0] = -y[0] + 0.1 * y[1];
        f[1] = -y[1];
    }

    fn kernel(&self, t: f64, s: f64, y_s: &[f64], k: &mut [f64]) {
        let decay = (-(t - s)).exp();
        k[0] = 0.5 * decay * y_s[0];
        k[1] = 0.2 * decay * y_s[1];
    }
}

let opts = IdeOptions::default().dt(0.01);
let result = VolterraSolver::solve(&CoupledIDE, 0.0, 5.0, &[1.0, 1.0], &opts)
    .expect("Solve failed");

let y_final = result.y_final().unwrap();
println!("y1(5) = {:.6}, y2(5) = {:.6}", y_final[0], y_final[1]);
```

## Working with Results

`IdeResult` provides the standard Numra result interface:

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
let result = VolterraSolver::solve(&system, t0, tf, &y0, &opts)?;

// Final state
let y_final = result.y_final().unwrap();

// Access at index
let y_i = result.y_at(50);

// Statistics
println!("RHS evaluations: {}", result.stats.n_rhs);
println!("Kernel evaluations: {}", result.stats.n_kernel);
println!("Steps taken: {}", result.stats.n_steps);
```

## Choosing a Solver

| Kernel Type | Recommended Solver | Cost | Notes |
|---|---|---|---|
| General \\(K(t,s,y)\\) | `VolterraRK4Solver` | \\(O(N^2)\\) | Most flexible |
| Simple problems | `VolterraSolver` | \\(O(N^2)\\) | Euler time-stepping |
| Sum of exponentials | `PronySolver` | \\(O(N)\\) | Fastest, limited to SOE kernels |
| Power-law | `VolterraRK4Solver` | \\(O(N^2)\\) | Consider FDE solver instead |

## Practical Considerations

### Quadrature Accuracy

The trapezoidal rule used for the integral is second-order accurate. For weakly singular
kernels (like power-law), the accuracy degrades near \\(s = t\\). Strategies:
- Use a smaller \\(\Delta t\\) to compensate
- For power-law kernels with \\(\alpha < 1\\), consider using the FDE solver instead

### Cost vs. Accuracy

The \\(O(N^2)\\) cost of general Volterra solvers becomes significant for long
integrations. To manage computational expense:
- Use `PronySolver` whenever the kernel can be approximated by a sum of exponentials
- Many physical kernels (even power-law) can be approximated by Prony series
- Use the coarsest \\(\Delta t\\) that meets accuracy requirements

### Connection to Fractional Calculus

When the kernel has the form \\(K(\tau) = \tau^{-\alpha} / \Gamma(1-\alpha)\\), the
Volterra integral becomes the Riemann-Liouville fractional integral, and the IDE reduces
to a fractional differential equation. For such problems, the dedicated FDE solver in
`numra-fde` is more efficient because it uses optimized Caputo weights.

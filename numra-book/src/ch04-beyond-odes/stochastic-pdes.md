# Stochastic Partial Differential Equations

Stochastic partial differential equations (SPDEs) extend PDEs by incorporating random
forcing terms, modeling systems where both spatial structure and noise play essential
roles. They appear in turbulence modeling, phase-field dynamics, population genetics,
surface growth, and financial mathematics. Numra provides the `numra-spde` crate which
combines the **Method of Lines** (MOL) spatial discretization from `numra-pde` with the
SDE solvers from `numra-sde`, yielding a unified framework for solving SPDEs numerically.

## Mathematical Background

An SPDE in Ito form is written as:

\\[
\frac{\partial u}{\partial t} = \mathcal{L}[u] + \sigma(u) \, \xi(x, t)
\\]

where:
- \\(\mathcal{L}[u]\\) is a **spatial differential operator** (the deterministic drift),
  such as the Laplacian \\(\alpha \nabla^2 u\\),
- \\(\sigma(u)\\) is the **diffusion coefficient** controlling noise intensity,
- \\(\xi(x, t)\\) is **space-time noise** -- a generalized random process.

### Additive vs. Multiplicative Noise

| Noise Type | Diffusion \\(\sigma\\) | Physical Meaning |
|---|---|---|
| **Additive** | \\(\sigma\\) independent of \\(u\\) | External random forcing (e.g., thermal fluctuations) |
| **Multiplicative** | \\(\sigma = \sigma(u)\\) | Noise intensity depends on state (e.g., concentration-dependent noise) |

Additive noise is simpler to analyze and compute. Many physical systems have additive
noise, and Numra marks it with the `is_additive()` method to enable optimizations.

### Noise Regularity

A fundamental issue with SPDEs is the regularity of the noise term. Space-time white
noise \\(\xi(x,t)\\) is so rough that solutions may not exist in the classical sense for
some equations. The noise regularity determines which equations are well-posed:

| Spatial Dimension | White Noise \\(\xi\\) | Colored/Trace-Class Noise |
|---|---|---|
| 1D | Solutions exist for heat equation | Solutions exist for broader class |
| 2D | Solutions may not exist (need renormalization) | Well-posed with sufficient regularity |
| 3D | Generally ill-posed with white noise | Requires strong trace-class condition |

In practice, spatial discretization on a grid with spacing \\(\Delta x\\) acts as a natural
regularization, cutting off spatial frequencies above \\(1/\Delta x\\). Numra's `numra-spde`
crate operates at this discretized level.

## The SpdeSystem Trait

Every SPDE in Numra implements the `SpdeSystem` trait:

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
pub trait SpdeSystem<S: Scalar> {
    /// Spatial dimension (1D, 2D, etc.)
    fn dim(&self) -> usize { 1 }

    /// Evaluate the deterministic spatial operator L[u].
    fn drift(&self, t: S, u: &[S], du: &mut [S], grid: &Grid1D<S>);

    /// Evaluate the noise coefficient sigma(u).
    fn diffusion(&self, t: S, u: &[S], sigma: &mut [S], grid: &Grid1D<S>);

    /// Noise correlation structure.
    fn noise_correlation(&self) -> NoiseCorrelation<S> {
        NoiseCorrelation::White
    }

    /// Whether the noise is additive (sigma independent of u).
    fn is_additive(&self) -> bool { true }
}
```

The `drift` method evaluates the spatial differential operator (e.g., the diffusion
Laplacian) at each interior grid point. The `diffusion` method returns the noise
intensity at each grid point. Both receive the spatial `Grid1D` for computing finite
difference stencils.

### Convenience Constructor: Spde1D

For simple cases, the `Spde1D` struct lets you define an SPDE from closures:

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
use numra_spde::system::Spde1D;

let spde = Spde1D::new(
    |t, u, du, grid| {
        // Drift: heat equation Laplacian
        let dx = grid.dx_uniform();
        let n = u.len();
        for i in 0..n {
            let u_left = if i == 0 { 0.0 } else { u[i - 1] };
            let u_right = if i == n - 1 { 0.0 } else { u[i + 1] };
            du[i] = 0.01 * (u_left - 2.0 * u[i] + u_right) / (dx * dx);
        }
    },
    |_t, _u, sigma, _grid| {
        for s in sigma.iter_mut() {
            *s = 0.1;  // Constant additive noise
        }
    },
);
```

## Noise Correlation Types

Numra supports three spatial noise correlation structures via `NoiseCorrelation`:

### White Noise (Default)

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
use numra_spde::NoiseCorrelation;

let correlation: NoiseCorrelation<f64> = NoiseCorrelation::White;
```

Spatially uncorrelated: each grid point receives an independent Wiener increment.
This is the simplest and cheapest option. The noise at grid point \\(i\\) is
independent of the noise at grid point \\(j\\) for \\(i \neq j\\).

Internally, this uses `NoiseType::Diagonal` from `numra-sde`, so the diffusion
buffer has size \\(n\\) (one entry per interior grid point).

### Colored Noise (Exponential Spatial Correlation)

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
use numra_spde::NoiseCorrelation;

let correlation: NoiseCorrelation<f64> = NoiseCorrelation::Colored {
    correlation_length: 0.2,
};
```

Spatially correlated with an exponential correlation function:

\\[
C(|x_i - x_j|) = \exp\!\left(-\frac{|x_i - x_j|}{\lambda}\right)
\\]

where \\(\lambda\\) is the `correlation_length`. Nearby grid points receive correlated
noise, producing smoother spatial noise patterns. The implementation pre-computes a
Cholesky factorization of the \\(n \times n\\) correlation matrix \\(C\\), then generates
correlated increments as \\(\Delta W = L \cdot Z\\) where \\(Z\\) is a vector of
independent standard normal samples and \\(C = L L^T\\).

**Guidelines for choosing `correlation_length`:**
- \\(\lambda \approx \Delta x\\): Nearly white noise; use `White` for efficiency.
- \\(\lambda \approx 5{-}20 \times \Delta x\\): Typical choice for smooth spatial patterns.
- \\(\lambda \approx L_{\text{domain}}\\): Nearly uniform noise across the domain.

### Trace-Class Noise (Spectral Representation)

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
use numra_spde::NoiseCorrelation;

let correlation: NoiseCorrelation<f64> = NoiseCorrelation::TraceClass {
    n_modes: 10,
    decay_rate: 2.0,
};
```

Expands the noise in eigenfunctions of the Laplacian (Q-Wiener process):

\\[
W(x, t) = \sum_{k=1}^{m} \sqrt{\lambda_k} \, e_k(x) \, W_k(t)
\\]

where \\(\lambda_k = k^{-\text{decay\_rate}}\\) are the eigenvalues and
\\(e_k(x) = \sqrt{2/L} \sin(k\pi x / L)\\) are the sinusoidal basis functions.
Each \\(W_k(t)\\) is an independent scalar Wiener process.

The `decay_rate` controls how quickly higher modes are suppressed:
- `decay_rate = 1.0`: Slow decay, rough noise
- `decay_rate = 2.0`: Moderate decay, smooth noise
- `decay_rate = 4.0`: Fast decay, very smooth noise

Trace-class noise is computationally efficient when `n_modes` is much smaller than the
number of grid points, because only `n_modes` independent Wiener processes are needed
(rather than one per grid point).

### Comparison of Noise Types

| Property | White | Colored | Trace-Class |
|---|---|---|---|
| Spatial correlation | None | Exponential | Spectral decay |
| Smoothness | Rough | Smooth | Smooth (tunable) |
| Wiener processes | \\(n\\) | \\(n\\) | \\(m \ll n\\) |
| Setup cost | \\(O(1)\\) | \\(O(n^2)\\) Cholesky | \\(O(nm)\\) basis |
| Per-step cost | \\(O(n)\\) | \\(O(n^2)\\) | \\(O(nm)\\) |
| Physical model | Thermal fluctuations | Correlated forcing | Q-Wiener process |

## The MolSdeSolver

The `MolSdeSolver` converts the SPDE into a system of SDEs using the Method of Lines,
then solves the resulting SDE system using solvers from `numra-sde`. The process is:

1. **Spatial discretization**: The user provides the drift function that computes
   \\(\mathcal{L}[u]\\) using finite differences on the grid.
2. **Noise construction**: The solver builds a `MolNoiseConfig` from the
   `NoiseCorrelation`, pre-computing Cholesky factors or spectral bases as needed.
3. **SDE wrapping**: The discretized SPDE is wrapped as an `SdeSystem` (via
   `MolSdeWrapper`), mapping the `SpdeSystem` trait methods to `SdeSystem::drift`
   and `SdeSystem::diffusion`.
4. **Time integration**: The wrapped system is solved by Euler-Maruyama or Milstein.

### SDE Methods

Numra provides two time-stepping methods via `SpdeMethod`:

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
use numra_spde::SpdeMethod;

SpdeMethod::EulerMaruyama  // Default: strong order 0.5
SpdeMethod::Milstein       // Strong order 1.0 (multiplicative noise)
```

For additive noise, Milstein automatically falls back to Euler-Maruyama since the
correction term vanishes.

## Solver Options

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
use numra_spde::SpdeOptions;

let opts = SpdeOptions::default()
    .dt(0.001)           // Time step
    .n_output(100)       // Number of output time points
    .method(SpdeMethod::EulerMaruyama)
    .seed(42)            // Random seed for reproducibility
    .adaptive(false);    // Fixed-step (default)
```

| Option | Default | Description |
|---|---|---|
| `dt` | 0.001 | Time step size |
| `n_output` | 100 | Number of output time points (subsampled from all steps) |
| `method` | `EulerMaruyama` | SDE time-stepping method |
| `seed` | None (entropy) | Random seed for reproducibility |
| `adaptive` | false | Enable adaptive time stepping |
| `rtol` | 1e-3 | Relative tolerance (adaptive only) |
| `atol` | 1e-6 | Absolute tolerance (adaptive only) |
| `dt_min` | 1e-10 | Minimum step size (adaptive only) |
| `dt_max` | 0.1 | Maximum step size (adaptive only) |

### Adaptive Time Stepping

When `adaptive(true)` is set, the solver uses **step-doubling** for error estimation:
each step is taken once with step size \\(h\\) and twice with \\(h/2\\) using independent
Wiener increments. The difference between the two results estimates the local error.

The step size controller uses a conservative exponent of \\(1/3\\) (appropriate for
Euler-Maruyama's weak order 1 with step-doubling error estimation):

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
let opts = SpdeOptions::default()
    .adaptive(true)
    .dt(0.001)
    .rtol(1e-3)
    .atol(1e-6)
    .dt_max(0.01)
    .seed(42);
```

Adaptive stepping adds overhead from the extra half-steps and noise generation, so
it is most beneficial when the dynamics have multiple time scales.

## Example: Stochastic Heat Equation

The stochastic heat equation models thermal diffusion with random forcing:

\\[
\frac{\partial T}{\partial t} = \alpha \frac{\partial^2 T}{\partial x^2} + \sigma \, \xi(x, t)
\\]

with Dirichlet boundary conditions \\(T(0,t) = T(1,t) = 0\\).

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
use numra_spde::{SpdeSystem, SpdeSolver, SpdeOptions, MolSdeSolver, NoiseCorrelation};
use numra_pde::Grid1D;

struct StochasticHeat {
    alpha: f64,
    sigma: f64,
}

impl SpdeSystem<f64> for StochasticHeat {
    fn drift(&self, _t: f64, u: &[f64], du: &mut [f64], grid: &Grid1D<f64>) {
        let dx = grid.dx_uniform();
        let n = u.len();
        for i in 0..n {
            let u_left = if i == 0 { 0.0 } else { u[i - 1] };
            let u_right = if i == n - 1 { 0.0 } else { u[i + 1] };
            du[i] = self.alpha * (u_left - 2.0 * u[i] + u_right) / (dx * dx);
        }
    }

    fn diffusion(&self, _t: f64, _u: &[f64], sigma: &mut [f64], _grid: &Grid1D<f64>) {
        for s in sigma.iter_mut() {
            *s = self.sigma;
        }
    }
}

let system = StochasticHeat { alpha: 0.01, sigma: 0.1 };
let grid = Grid1D::uniform(0.0_f64, 1.0, 21);

// Initial condition: sin(pi*x) at interior points
let u0: Vec<f64> = grid.interior_points().iter()
    .map(|&x| (std::f64::consts::PI * x).sin())
    .collect();

let options = SpdeOptions::default()
    .dt(0.0001)
    .n_output(50)
    .seed(42);

let result = MolSdeSolver::solve(&system, 0.0, 0.1, &u0, &grid, &options)
    .expect("Solve failed");

// Access solution
let u_final = result.y_final().unwrap();
println!("Solution at t = {:.3}:", result.t_final().unwrap());
for (i, &val) in u_final.iter().enumerate() {
    println!("  x_{} = {:.6}", i, val);
}
```

### Zero-Noise Limit

When \\(\sigma = 0\\), the SPDE reduces to the deterministic PDE, and the solver
produces identical results regardless of the random seed. This is a useful sanity check.

## Example: Multiplicative Noise

For multiplicative noise where the noise intensity depends on the field value:

\\[
\frac{\partial u}{\partial t} = \alpha \frac{\partial^2 u}{\partial x^2} + \sigma \, u \, \xi(x, t)
\\]

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
use numra_spde::{SpdeSystem, SpdeSolver, SpdeOptions, SpdeMethod, MolSdeSolver};
use numra_pde::Grid1D;

struct MultiplicativeNoise {
    alpha: f64,
    sigma: f64,
}

impl SpdeSystem<f64> for MultiplicativeNoise {
    fn drift(&self, _t: f64, u: &[f64], du: &mut [f64], grid: &Grid1D<f64>) {
        let dx = grid.dx_uniform();
        let n = u.len();
        for i in 0..n {
            let u_left = if i == 0 { 0.0 } else { u[i - 1] };
            let u_right = if i == n - 1 { 0.0 } else { u[i + 1] };
            du[i] = self.alpha * (u_left - 2.0 * u[i] + u_right) / (dx * dx);
        }
    }

    fn diffusion(&self, _t: f64, u: &[f64], sigma: &mut [f64], _grid: &Grid1D<f64>) {
        for i in 0..u.len() {
            sigma[i] = self.sigma * u[i];  // Noise scales with field value
        }
    }

    fn is_additive(&self) -> bool { false }
}

let system = MultiplicativeNoise { alpha: 0.01, sigma: 0.1 };
let grid = Grid1D::uniform(0.0_f64, 1.0, 51);
let u0: Vec<f64> = grid.interior_points().iter()
    .map(|&x| (std::f64::consts::PI * x).sin())
    .collect();

// Use Milstein for better accuracy with multiplicative noise
let options = SpdeOptions::default()
    .dt(0.0001)
    .method(SpdeMethod::Milstein)
    .n_output(50)
    .seed(42);

let result = MolSdeSolver::solve(&system, 0.0, 0.1, &u0, &grid, &options)
    .expect("Solve failed");
```

## Example: Colored Noise

Using spatially correlated noise to model smooth random forcing:

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
use numra_spde::{SpdeSystem, SpdeSolver, SpdeOptions, MolSdeSolver, NoiseCorrelation};
use numra_pde::Grid1D;

struct ColoredHeat {
    alpha: f64,
    sigma: f64,
    correlation_length: f64,
}

impl SpdeSystem<f64> for ColoredHeat {
    fn drift(&self, _t: f64, u: &[f64], du: &mut [f64], grid: &Grid1D<f64>) {
        let dx = grid.dx_uniform();
        let n = u.len();
        for i in 0..n {
            let u_left = if i == 0 { 0.0 } else { u[i - 1] };
            let u_right = if i == n - 1 { 0.0 } else { u[i + 1] };
            du[i] = self.alpha * (u_left - 2.0 * u[i] + u_right) / (dx * dx);
        }
    }

    fn diffusion(&self, _t: f64, _u: &[f64], sigma: &mut [f64], _grid: &Grid1D<f64>) {
        for s in sigma.iter_mut() {
            *s = self.sigma;
        }
    }

    fn noise_correlation(&self) -> NoiseCorrelation<f64> {
        NoiseCorrelation::Colored {
            correlation_length: self.correlation_length,
        }
    }
}

let system = ColoredHeat {
    alpha: 0.01,
    sigma: 0.1,
    correlation_length: 0.2,
};
let grid = Grid1D::uniform(0.0_f64, 1.0, 21);
let u0: Vec<f64> = grid.interior_points().iter()
    .map(|&x| (std::f64::consts::PI * x).sin())
    .collect();

let options = SpdeOptions::default()
    .dt(0.0001)
    .n_output(20)
    .seed(42);

let result = MolSdeSolver::solve(&system, 0.0, 0.01, &u0, &grid, &options)
    .expect("Solve failed");
```

## Example: Trace-Class Noise

Using a Q-Wiener process with spectral decay for mathematically well-posed noise:

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
use numra_spde::{SpdeSystem, SpdeSolver, SpdeOptions, MolSdeSolver, NoiseCorrelation};
use numra_pde::Grid1D;

struct TraceClassHeat {
    alpha: f64,
    sigma: f64,
    n_modes: usize,
    decay_rate: f64,
}

impl SpdeSystem<f64> for TraceClassHeat {
    fn drift(&self, _t: f64, u: &[f64], du: &mut [f64], grid: &Grid1D<f64>) {
        let dx = grid.dx_uniform();
        let n = u.len();
        for i in 0..n {
            let u_left = if i == 0 { 0.0 } else { u[i - 1] };
            let u_right = if i == n - 1 { 0.0 } else { u[i + 1] };
            du[i] = self.alpha * (u_left - 2.0 * u[i] + u_right) / (dx * dx);
        }
    }

    fn diffusion(&self, _t: f64, _u: &[f64], sigma: &mut [f64], _grid: &Grid1D<f64>) {
        for s in sigma.iter_mut() {
            *s = self.sigma;
        }
    }

    fn noise_correlation(&self) -> NoiseCorrelation<f64> {
        NoiseCorrelation::TraceClass {
            n_modes: self.n_modes,
            decay_rate: self.decay_rate,
        }
    }
}

let system = TraceClassHeat {
    alpha: 0.01,
    sigma: 0.1,
    n_modes: 5,      // Only 5 Wiener processes instead of n
    decay_rate: 2.0,  // Higher modes suppressed as k^(-2)
};
let grid = Grid1D::uniform(0.0_f64, 1.0, 51);
let u0: Vec<f64> = grid.interior_points().iter()
    .map(|&x| (std::f64::consts::PI * x).sin())
    .collect();

let options = SpdeOptions::default()
    .dt(0.0001)
    .n_output(20)
    .seed(42);

let result = MolSdeSolver::solve(&system, 0.0, 0.01, &u0, &grid, &options)
    .expect("Solve failed");
```

## Monte Carlo Ensemble Simulation

For computing statistics (expected fields, variance fields, confidence intervals),
`SpdeEnsemble` runs multiple trajectories with different noise realizations:

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
use numra_spde::{SpdeSystem, SpdeOptions, MolSdeSolver, SpdeEnsemble};
use numra_pde::Grid1D;

// ... define StochasticHeat as above ...

let system = StochasticHeat { alpha: 0.01, sigma: 0.05 };
let grid = Grid1D::uniform(0.0_f64, 1.0, 21);
let u0: Vec<f64> = grid.interior_points().iter()
    .map(|&x| (std::f64::consts::PI * x).sin())
    .collect();

let options = SpdeOptions::default()
    .dt(0.001)
    .n_output(10)
    .seed(100);

// Run 100 trajectories with different noise realizations
let results = SpdeEnsemble::solve(
    &system, 0.0, 0.01, &u0, &grid, &options, 100
).expect("Ensemble solve failed");

// Compute mean and standard deviation fields
let mean = SpdeEnsemble::mean(&results);
let std = SpdeEnsemble::std(&results, &mean);

// mean[i][j] = mean field at time i, grid point j
// std[i][j]  = standard deviation at time i, grid point j
let n_times = mean.len();
println!("Mean field at final time:");
for (j, &val) in mean[n_times - 1].iter().enumerate() {
    println!("  x_{}: mean = {:.6}, std = {:.6}", j, val, std[n_times - 1][j]);
}
```

Each trajectory receives a unique seed derived from the base seed (`seed + i`), ensuring
both reproducibility and statistical independence. Standard deviation uses Bessel's
correction (dividing by \\(n - 1\\)).

## Working with Results

`SpdeResult` provides access to the solution at all output time points:

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
let result = MolSdeSolver::solve(&system, t0, tf, &u0, &grid, &options)?;

// Check success
assert!(result.success);

// Final field
let u_final = result.y_final().unwrap();

// Final time
let t_final = result.t_final().unwrap();

// Field at a specific output index
let u_i = result.y_at(5);

// All time points
let times = &result.t;

// Spatial grid (stored with result)
let grid_ref = &result.grid;

// Statistics
println!("Steps taken: {}", result.stats.n_steps);
println!("Drift evaluations: {}", result.stats.n_drift);
println!("Diffusion evaluations: {}", result.stats.n_diffusion);
println!("Rejected steps: {}", result.stats.n_reject);
```

Note that the result contains subsampled output: the solver takes many internal steps
but only stores `n_output` time points (plus the final point). This controls memory
usage for fine-grid problems.

## Choosing a Solver Configuration

| Scenario | Method | Noise | Adaptive |
|---|---|---|---|
| Quick exploration | `EulerMaruyama` | White | No |
| Multiplicative noise, accuracy needed | `Milstein` | Any | No |
| Smooth random forcing | `EulerMaruyama` | Colored or TraceClass | No |
| Multi-scale dynamics | `EulerMaruyama` | Any | Yes |
| Ensemble statistics | `EulerMaruyama` | Any | No (fixed step for reproducibility) |

## Practical Considerations

### Step Size and Stability

The stochastic heat equation with explicit time stepping requires:

\\[
\Delta t \leq \frac{\Delta x^2}{2\alpha}
\\]

for the deterministic part to be stable. The noise term adds further constraints.
A good starting point is \\(\Delta t = \Delta x^2 / (4\alpha)\\), which provides a
safety margin. With adaptive stepping enabled, the solver handles this automatically.

### Grid Resolution and Noise Scaling

Space-time white noise on a discrete grid has variance proportional to \\(1/\Delta x\\).
This means that refining the spatial grid increases the noise intensity at each point.
The physical interpretation depends on the model:

- For **renormalized** models, the noise coefficient \\(\sigma\\) should scale with
  \\(\sqrt{\Delta x}\\) to keep the solution statistics independent of grid resolution.
- For **discrete** models (e.g., interacting particle systems), the grid-dependent
  scaling is the correct physical behavior.

### Computational Cost

The cost per time step depends on the noise type:

| Component | White Noise | Colored Noise | Trace-Class |
|---|---|---|---|
| Drift evaluation | \\(O(n)\\) | \\(O(n)\\) | \\(O(n)\\) |
| Noise generation | \\(O(n)\\) | \\(O(n^2)\\) | \\(O(nm)\\) |
| Memory | \\(O(n)\\) | \\(O(n^2)\\) | \\(O(nm)\\) |

For fine grids with colored noise, the \\(O(n^2)\\) Cholesky-based correlation step
dominates. If this becomes a bottleneck, consider using trace-class noise with
a moderate number of modes, which scales as \\(O(nm)\\) with \\(m \ll n\\).

### Reproducibility

With a fixed seed, the solver produces identical trajectories:

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
let opts = SpdeOptions::default().dt(0.001).seed(42);
let r1 = MolSdeSolver::solve(&system, 0.0, 0.01, &u0, &grid, &opts)?;
let r2 = MolSdeSolver::solve(&system, 0.0, 0.01, &u0, &grid, &opts)?;

// r1 and r2 produce identical results
let y1 = r1.y_final().unwrap();
let y2 = r2.y_final().unwrap();
for (a, b) in y1.iter().zip(y2.iter()) {
    assert!((a - b).abs() < 1e-10);
}
```

### Relationship to Other Numra Crates

The SPDE solver bridges two existing Numra crates:

| Crate | Role in SPDE Solving |
|---|---|
| `numra-pde` | Provides `Grid1D` for spatial discretization |
| `numra-sde` | Provides `EulerMaruyama`, `Milstein`, and `WienerProcess` for time stepping |
| `numra-spde` | Wraps `SpdeSystem` as `SdeSystem` via MOL, manages noise correlation |

When \\(\sigma = 0\\), the SPDE reduces to a deterministic PDE. For that case, using
`numra-pde` with `MOLSystem` and an adaptive ODE solver from `numra-ode` is more
efficient, since ODE solvers provide higher-order accuracy and adaptive error control.

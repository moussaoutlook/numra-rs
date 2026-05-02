---
title: "DE Type Comparison"
---

Numra supports seven types of differential equations, each designed for a different
class of dynamical system. This chapter provides a unified comparison to help you
choose the right equation type and solver for your problem.

## At a Glance

| Type | Crate | Equation Form | Memory | Randomness |
|---|---|---|---|---|
| **ODE** | `numra-ode` | $y'(t) = f(t, y)$ | None | None |
| **SDE** | `numra-sde` | $dX = f\,dt + g\,dW$ | None | Wiener process |
| **DDE** | `numra-dde` | $y'(t) = f(t, y(t), y(t-\tau))$ | Finite (discrete delays) | None |
| **FDE** | `numra-fde` | ${}^C D^\alpha y = f(t, y)$ | Infinite (power-law kernel) | None |
| **IDE** | `numra-ide` | $y' = f(t,y) + \int_0^t K(t,s,y)\,ds$ | Infinite (general kernel) | None |
| **PDE** | `numra-pde` | $\partial u/\partial t = \mathcal{L}[u]$ | None | None |
| **SPDE** | `numra-spde` | $\partial u/\partial t = \mathcal{L}[u] + \sigma\,\xi$ | None | Space-time noise |

## Mathematical Structure

### Independent Variables

| Type | Time | Space | Stochastic |
|---|---|---|---|
| ODE | $t$ | -- | -- |
| SDE | $t$ | -- | $W(t)$ |
| DDE | $t$ | -- | -- |
| FDE | $t$ | -- | -- |
| IDE | $t$ | -- | -- |
| PDE | $t$ | $x$ (1D, 2D, 3D) | -- |
| SPDE | $t$ | $x$ (1D) | $\xi(x,t)$ |

### Initial/Boundary Data

| Type | What You Must Provide |
|---|---|
| ODE | Point value $y(t_0)$ |
| SDE | Point value $X(t_0)$ + random seed |
| DDE | History function $\varphi(t)$ for $t \leq t_0$ |
| FDE | Point value $y(t_0)$ |
| IDE | Point value $y(t_0)$ |
| PDE | Initial field $u(x, t_0)$ + boundary conditions |
| SPDE | Initial field $u(x, t_0)$ + boundary conditions + random seed |

### Phase Space

| Type | State Space | Dimension |
|---|---|---|
| ODE | $\mathbb{R}^n$ | Finite |
| SDE | $\mathbb{R}^n$ | Finite |
| DDE | Function space $C([-\tau, 0], \mathbb{R}^n)$ | Infinite-dimensional |
| FDE | $\mathbb{R}^n$ (with full history stored internally) | Finite (but O(N) storage) |
| IDE | $\mathbb{R}^n$ (with full history stored internally) | Finite (but O(N) storage) |
| PDE | Function space (discretized to grid) | $N_{\text{grid}}$ |
| SPDE | Function space (discretized to grid) | $N_{\text{grid}}$ |

## Computational Complexity

### Cost Per Step

| Type | Cost Per Step | Total Cost ($N$ steps) | Memory |
|---|---|---|---|
| ODE | $O(n)$ | $O(Nn)$ | $O(n)$ |
| SDE | $O(n)$ | $O(Nn)$ | $O(n)$ |
| DDE | $O(n)$ per stage + interpolation | $O(Nn)$ | $O(Nn)$ for history |
| FDE | $O(n \cdot N)$ (history sum) | $O(N^2 n)$ | $O(Nn)$ |
| IDE (general) | $O(n \cdot N)$ (quadrature) | $O(N^2 n)$ | $O(Nn)$ |
| IDE (Prony) | $O(nm)$ ($m$ exponential terms) | $O(Nnm)$ | $O(nm)$ |
| PDE | $O(N_{\text{grid}})$ per ODE step | $O(N \cdot N_{\text{grid}})$ | $O(N_{\text{grid}})$ |
| SPDE | $O(N_{\text{grid}})$ (white) to $O(N_{\text{grid}}^2)$ (colored) | $O(N \cdot N_{\text{grid}})$ | $O(N_{\text{grid}})$ |

The $O(N^2)$ total cost of FDE and IDE solvers is the main computational bottleneck
for long integrations of memory-dependent equations. For IDEs with sum-of-exponential
kernels, the Prony solver reduces this to $O(N)$.

### Scalability Guidelines

| Problem Size | Recommended Approach |
|---|---|
| Small ($n \leq 20$, $N \leq 10^4$) | Any solver works well |
| Medium ($n \leq 100$, $N \leq 10^5$) | Consider implicit ODE solvers for stiff problems |
| Large ($n \leq 1000$, $N \leq 10^6$) | Avoid general IDE/FDE solvers; use Prony or ODE reduction |
| PDE ($N_{\text{grid}} \leq 200$) | MOL with explicit or implicit ODE solver |
| PDE ($N_{\text{grid}} > 200$) | MOL with implicit solver (Radau5, Esdirk, Bdf) |

## Solver Inventory

### ODE Solvers (`numra-ode`)

| Solver | Type | Order | Step Size | Best For |
|---|---|---|---|---|
| `DoPri5` | Explicit RK | 5(4) | Adaptive | General purpose |
| `Tsit5` | Explicit RK | 5(4) | Adaptive | Efficient (FSAL) |
| `Vern6` | Explicit RK | 6(5) | Adaptive | High accuracy |
| `Vern7` | Explicit RK | 7(6) | Adaptive | High accuracy |
| `Vern8` | Explicit RK | 8(7) | Adaptive | Very high accuracy |
| `Radau5` | Implicit RK | 5 | Adaptive | Stiff problems |
| `Esdirk32` | ESDIRK | 2(1) | Adaptive | Mildly stiff |
| `Esdirk43` | ESDIRK | 3(2) | Adaptive | Moderately stiff |
| `Esdirk54` | ESDIRK | 4(3) | Adaptive | Stiff, high accuracy |
| `Bdf` | Multistep | 1--5 | Adaptive | Stiff, variable order |
| `Auto` | Automatic | Varies | Adaptive | Stiffness detection |

### SDE Solvers (`numra-sde`)

| Solver | Strong Order | Weak Order | Step Size | Best For |
|---|---|---|---|---|
| `EulerMaruyama` | 0.5 | 1.0 | Fixed | Quick prototyping |
| `Milstein` | 1.0 | 1.0 | Fixed | Multiplicative noise |
| `Sra1` | 1.0--1.5 | -- | Adaptive | Additive noise, accuracy |
| `Sra2` | -- | 2.0 | Adaptive | Monte Carlo statistics |

### DDE Solver (`numra-dde`)

| Solver | Type | Order | Step Size | Features |
|---|---|---|---|---|
| `MethodOfSteps` | Explicit RK (DoPri5) | 5(4) | Adaptive | Hermite interpolation, discontinuity tracking, state-dependent delays |

### FDE Solver (`numra-fde`)

| Solver | Type | Convergence | Step Size | Features |
|---|---|---|---|---|
| `L1Solver` | L1 scheme | $O(\Delta t^{2-\alpha})$ | Fixed | Caputo derivative, fixed-point iteration |

### IDE Solvers (`numra-ide`)

| Solver | Time Stepping | Quadrature | Cost | Best For |
|---|---|---|---|---|
| `VolterraSolver` | Euler | Trapezoidal | $O(N^2)$ | Simple problems |
| `VolterraRK4Solver` | RK4 | Trapezoidal | $O(N^2)$ | General kernels |
| `PronySolver` | Augmented ODE | Recursive | $O(N)$ | Sum-of-exponential kernels |

### PDE Solver (`numra-pde`)

| Approach | Spatial Discretization | Time Integration | Best For |
|---|---|---|---|
| `MOLSystem` | Finite differences (2nd order) | Any ODE solver | Parabolic, reaction-diffusion |
| `MOLSystem2D` | Finite differences (2nd order) | Any ODE solver | 2D problems |

### SPDE Solver (`numra-spde`)

| Solver | Method | Noise Types | Step Size | Features |
|---|---|---|---|---|
| `MolSdeSolver` | MOL + EM/Milstein | White, Colored, TraceClass | Fixed or Adaptive | Ensemble support |

## System Trait Comparison

All equation types follow a common pattern: define a system struct implementing a
trait, configure options with a builder, call a solver, and inspect the result.

### Trait Signatures

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust
// ODE: y'(t) = f(t, y)
trait OdeSystem<S> {
    fn dim(&self) -> usize;
    fn rhs(&self, t: S, y: &[S], dydt: &mut [S]);
}

// SDE: dX = f dt + g dW
trait SdeSystem<S> {
    fn dim(&self) -> usize;
    fn drift(&self, t: S, x: &[S], f: &mut [S]);
    fn diffusion(&self, t: S, x: &[S], g: &mut [S]);
    fn noise_type(&self) -> NoiseType;
}

// DDE: y'(t) = f(t, y(t), y(t-tau))
trait DdeSystem<S> {
    fn dim(&self) -> usize;
    fn delays(&self) -> Vec<S>;
    fn rhs(&self, t: S, y: &[S], y_delayed: &[&[S]], dydt: &mut [S]);
}

// FDE: D^alpha y = f(t, y)
trait FdeSystem<S> {
    fn dim(&self) -> usize;
    fn alpha(&self) -> S;
    fn rhs(&self, t: S, y: &[S], f: &mut [S]);
}

// IDE: y' = f(t,y) + integral K(t,s,y(s)) ds
trait IdeSystem<S> {
    fn dim(&self) -> usize;
    fn rhs(&self, t: S, y: &[S], f: &mut [S]);
    fn kernel(&self, t: S, s: S, y_s: &[S], k: &mut [S]);
}

// PDE: du/dt = F(t, x, u, du/dx, d2u/dx2)
trait PdeSystem<S> {
    fn rhs(&self, t: S, x: S, u: S, du_dx: S, d2u_dx2: S) -> S;
}

// SPDE: du/dt = L[u] + sigma(u) * noise
trait SpdeSystem<S> {
    fn drift(&self, t: S, u: &[S], du: &mut [S], grid: &Grid1D<S>);
    fn diffusion(&self, t: S, u: &[S], sigma: &mut [S], grid: &Grid1D<S>);
    fn noise_correlation(&self) -> NoiseCorrelation<S>;
}
```

### Options Comparison

| Type | Options Struct | Key Parameters |
|---|---|---|
| ODE | `SolverOptions` | `rtol`, `atol`, `h0`, `h_max`, `dense` |
| SDE | `SdeOptions` | `dt`, `rtol`, `atol`, `seed`, `save_trajectory` |
| DDE | `DdeOptions` | `rtol`, `atol`, `h0`, `track_discontinuities`, `discontinuity_order` |
| FDE | `FdeOptions` | `dt`, `max_steps`, `tol`, `max_iter` |
| IDE | `IdeOptions` | `dt`, `max_steps`, `tol`, `quad_points` |
| PDE | `SolverOptions` (via ODE) | Same as ODE (MOL system is an ODE system) |
| SPDE | `SpdeOptions` | `dt`, `n_output`, `method`, `seed`, `adaptive` |

### Result Format

All result types provide a consistent interface:

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust
// Common across all result types:
result.y_final()      // Final state vector
result.y_at(i)        // State at index i
result.t              // Time points (Vec)
result.stats          // Solver statistics
```

| Type | Result Struct | Storage Format | Extra Fields |
|---|---|---|---|
| ODE | `SolverResult` | Row-major flat `Vec<S>` | `n_accept`, `n_reject`, `n_eval` |
| SDE | `SdeResult` | Row-major flat `Vec<S>` | `n_drift`, `n_diffusion`, `n_accept`, `n_reject` |
| DDE | `DdeResult` | Row-major flat `Vec<S>` | `n_eval`, `n_accept`, `n_reject`, `n_discontinuities` |
| FDE | `FdeResult` | Row-major flat `Vec<S>` | `n_rhs`, `n_steps` |
| IDE | `IdeResult` | Row-major flat `Vec<S>` | `n_rhs`, `n_kernel`, `n_steps` |
| PDE | `SolverResult` (via ODE) | Row-major flat `Vec<S>` | Same as ODE |
| SPDE | `SpdeResult` | `Vec<Vec<S>>` per time step | `n_steps`, `n_drift`, `n_diffusion`, `n_reject` |

## Decision Guide

### Which Equation Type Matches Your Problem?

**Start here: What does the derivative depend on?**

1. **Only the current state** $y(t)$: **ODE**
2. **Current state + random noise**: **SDE**
3. **Current state + state at past time(s)** $y(t - \tau)$: **DDE**
4. **Fractional-order derivative** (power-law memory): **FDE**
5. **Current state + integral over history** (general kernel): **IDE**
6. **Current state + spatial derivatives** $\partial^2 u / \partial x^2$: **PDE**
7. **Spatial derivatives + random forcing**: **SPDE**

### Stiffness Considerations

| Type | Non-Stiff Solver | Stiff Solver |
|---|---|---|
| ODE | `DoPri5`, `Tsit5`, `Vern6/7/8` | `Radau5`, `Esdirk`, `Bdf` |
| SDE | `EulerMaruyama` | -- (no implicit SDE solver currently) |
| DDE | `MethodOfSteps` | -- (explicit only currently) |
| FDE | `L1Solver` (implicit fixed-point) | `L1Solver` handles mild stiffness |
| IDE | `VolterraRK4Solver` | -- (explicit only currently) |
| PDE (via MOL) | `DoPri5` (coarse grid) | `Radau5`, `Bdf` (fine grid) |
| SPDE | `EulerMaruyama` | -- (explicit only currently) |

### Memory Effects

If your system has memory, the type of memory determines the equation:

| Memory Type | DE Type | Example |
|---|---|---|
| No memory | ODE | $y' = -ky$ |
| Discrete past states | DDE | $y'(t) = f(y(t), y(t-\tau))$ |
| Power-law (Caputo) kernel | FDE | ${}^C D^{0.5} y = f(y)$ |
| Exponential kernel | IDE (Prony) | $y' = f(y) + \int e^{-b(t-s)} y(s)\,ds$ |
| General kernel | IDE (Volterra) | $y' = f(y) + \int K(t,s,y(s))\,ds$ |

### Spatial Structure

If your system has spatial structure:

| Spatial Structure | DE Type | Approach |
|---|---|---|
| No spatial dependence | ODE, SDE, DDE, FDE, or IDE | Direct |
| Spatial derivatives (deterministic) | PDE | MOL -> ODE solver |
| Spatial derivatives + noise | SPDE | MOL -> SDE solver |

## Cross-References Between Types

Several equation types are closely related and can sometimes be converted:

### FDE as IDE

A fractional differential equation ${}^C D^\alpha y = f(t,y)$ is equivalent to a
Volterra IDE with a power-law kernel:

$$
y'(t) = f(t,y) + \frac{1}{\Gamma(1-\alpha)} \int_0^t (t-s)^{-\alpha} y'(s)\,ds
$$

The dedicated FDE solver (`L1Solver`) uses optimized Caputo weights that are more
efficient than the general Volterra quadrature for this specific kernel.

### PDE as ODE

Through the Method of Lines, any PDE becomes a large ODE system:

$$
\frac{du_i}{dt} = F(u_{i-1}, u_i, u_{i+1}), \quad i = 1, \ldots, N-1
$$

Numra's `MOLSystem` implements `OdeSystem`, so all ODE solvers work directly.

### SPDE as SDE

Through the Method of Lines, an SPDE becomes a system of SDEs (one per grid point).
Numra's `MolSdeWrapper` implements `SdeSystem`, bridging the two crates.

### IDE with Prony Kernel as ODE

When the IDE kernel is a sum of exponentials $K(\tau) = \sum a_i e^{-b_i \tau}$,
the integral satisfies an auxiliary ODE, so the IDE becomes an augmented ODE system
of dimension $n + nm$ where $m$ is the number of exponential terms.

### SDE with Zero Noise as ODE

When the diffusion coefficient $g = 0$, an SDE reduces to a deterministic ODE.
Use `numra-ode` for better accuracy and efficiency in this case.

### DDE with Zero Delay as ODE

When all delays $\tau_i = 0$, a DDE reduces to an ODE. Use `numra-ode` for
better performance.

## Example: Same Problem, Different Formulations

Consider modeling a system with exponential memory decay. This can be formulated as
three different equation types:

### As an IDE

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust
use numra_ide::{IdeSystem, VolterraSolver, IdeSolver, IdeOptions};

struct MemorySystem;
impl IdeSystem<f64> for MemorySystem {
    fn dim(&self) -> usize { 1 }
    fn rhs(&self, _t: f64, y: &[f64], f: &mut [f64]) {
        f[0] = -y[0];
    }
    fn kernel(&self, t: f64, s: f64, y_s: &[f64], k: &mut [f64]) {
        k[0] = (-(t - s)).exp() * y_s[0];
    }
}

let opts = IdeOptions::default().dt(0.01);
let result = VolterraSolver::solve(&MemorySystem, 0.0, 5.0, &[1.0], &opts)?;
// Cost: O(N^2)
```

### As an IDE with Prony Solver

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust
use numra_ide::{IdeSystem, PronySolver, IdeSolver, IdeOptions};

// Same system, but recognized as sum-of-exponentials
// PronySolver reduces the integral to an auxiliary ODE
let opts = IdeOptions::default().dt(0.01);
let result = PronySolver::solve(&MemorySystem, 0.0, 5.0, &[1.0], &opts)?;
// Cost: O(N)
```

### As an Augmented ODE

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust
use numra_ode::{DoPri5, Solver, SolverOptions};

// Manually augment: y' = -y + I, I' = y - I
// where I(t) = integral_0^t exp(-(t-s)) y(s) ds
struct AugmentedODE;
// ... (implement OdeSystem with dim = 2)
// Cost: O(N), and benefits from adaptive high-order ODE solvers
```

## Summary Table

| Feature | ODE | SDE | DDE | FDE | IDE | PDE | SPDE |
|---|---|---|---|---|---|---|---|
| Crate | `numra-ode` | `numra-sde` | `numra-dde` | `numra-fde` | `numra-ide` | `numra-pde` | `numra-spde` |
| Spatial | No | No | No | No | No | Yes | Yes |
| Stochastic | No | Yes | No | No | No | No | Yes |
| Memory | No | No | Discrete | Power-law | General | No | No |
| Adaptive dt | Yes | Yes (SRA) | Yes | No | No | Yes (via ODE) | Yes (step-doubling) |
| Implicit | Yes | No | No | Yes (FP) | No | Yes (via ODE) | No |
| Ensemble | No | Yes | No | No | No | No | Yes |
| Total cost | $O(N)$ | $O(N)$ | $O(N)$ | $O(N^2)$ | $O(N^2)$ | $O(N)$ | $O(N)$ |

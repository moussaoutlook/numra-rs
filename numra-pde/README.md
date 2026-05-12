# numra-pde

**Partial differential equation solvers for the [Numra](https://numra-rs.org/) workspace — Method of Lines for heat, advection-diffusion, and reaction-diffusion in 1D / 2D / 3D, plus Stefan-condition moving-boundary problems.**

[![Crates.io](https://img.shields.io/crates/v/numra-pde.svg)](https://crates.io/crates/numra-pde)
[![docs.rs](https://docs.rs/numra-pde/badge.svg)](https://docs.rs/numra-pde)

Method of Lines: discretize the spatial operator with finite differences, hand the resulting ODE system to a [`numra-ode`](https://docs.rs/numra-ode) stepper (often `Radau5` for stiff diffusion problems), and let stiffness and step control happen at the time-integration layer rather than in custom PDE code.

## Example

```rust
use numra_ode::{DoPri5, Solver, SolverOptions};
use numra_pde::boundary::DirichletBC;
use numra_pde::{Grid1D, HeatEquation1D, MOLSystem};

let grid = Grid1D::uniform(0.0_f64, 1.0, 51);
let pde = HeatEquation1D::new(0.01);
let mol = MOLSystem::new(pde, grid.clone(), DirichletBC::new(1.0), DirichletBC::new(0.0));

let u0: Vec<f64> = grid.points().iter()
    .map(|&x| (std::f64::consts::PI * x).sin())
    .collect();

let opts = SolverOptions::default().rtol(1e-6);
let _ = DoPri5::solve(&mol, 0.0, 0.5, &u0[1..u0.len()-1], &opts);
```

## What's in this crate

**Equations (built-in):**

- 1D: `HeatEquation1D`, `DiffusionReaction1D`
- 2D: `HeatEquation2D`, `AdvectionDiffusion2D`, `ReactionDiffusion2D`
- 3D: `HeatEquation3D`, `AdvectionDiffusion3D`, `ReactionDiffusion3D`

**Discretization:**

- `Grid1D`, `Grid2D`, `Grid3D` — uniform spatial grids
- `FDM`, `Stencil`, `DifferenceScheme` — finite-difference operators
- `MOLSystem`, `MOLSystem2D`, `MOLSystem3D` — Method-of-Lines wrappers
- `ParametricMOLSystem2D`, `ParametricMOLSystem3D` — parameterized variants for sensitivity studies

**Boundary conditions:**

- 1D: `DirichletBC`, `NeumannBC`, `RobinBC`, `PeriodicBC`
- 2D/3D: `BoundaryConditions2D`, `BoundaryConditions3D`

**Moving boundaries:** `MovingBound`, `StefanCondition`, `CoordinateTransform`, `Domain1D` — for melting/freezing and other phase-change problems.

**Sparse assembly:** `assemble_laplacian_2d`, `assemble_laplacian_3d`, `assemble_operator_2d`, `assemble_operator_3d`.

## Composes with

- [`numra-ode`](https://docs.rs/numra-ode) — time-stepping the spatially-discretized system (often `Radau5` for stiff diffusion)
- [`numra-spde`](https://docs.rs/numra-spde) — shares the same spatial discretizations under stochastic time stepping
- [`numra-stats`](https://docs.rs/numra-stats) — spatial-distribution statistics at final time (mean, variance, percentile)
- [`numra-linalg`](https://docs.rs/numra-linalg) — sparse Laplacian / Helmholtz operator assembly

See [interop workflows](https://github.com/moussaoutlook/numra-rs/blob/main/numra/tests/interop_workflows.rs) for the verified PDE → statistics workflow.

## Install

```toml
[dependencies]
numra-pde = "0.1"
```

Or via the umbrella crate:

```toml
[dependencies]
numra = "0.1"
```

## Documentation

- **API**: <https://docs.rs/numra-pde>
- **Book**: [Partial DEs](https://book.numra-rs.org/ch04-beyond-odes/partial-des/)
- **Source**: <https://github.com/moussaoutlook/numra-rs/tree/main/numra-pde>

## License

Numra Academic & Research License (Non-Commercial). Academic and research use is free; commercial use requires a separate license — contact `contact@spectralautomata.com`. See [LICENSE](https://github.com/moussaoutlook/numra-rs/blob/main/LICENSE).

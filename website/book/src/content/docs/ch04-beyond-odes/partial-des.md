---
title: "Partial Differential Equations"
---

Partial differential equations (PDEs) involve derivatives with respect to multiple
independent variables -- typically space and time. Numra provides the `numra-pde` crate
which uses the **Method of Lines** (MOL) approach: discretize spatial derivatives with
finite differences, then solve the resulting ODE system in time using the solvers from
`numra-ode`. This architecture lets you leverage Numra's adaptive, high-order ODE solvers
for PDE problems.

## The Method of Lines

The MOL approach converts a PDE into a system of ODEs through three steps:

1. **Spatial discretization**: Replace spatial derivatives with finite difference
   approximations on a grid.
2. **ODE formulation**: The PDE becomes a system of ODEs (one per interior grid point)
   with time as the independent variable.
3. **Time integration**: Solve the ODE system using any ODE solver (explicit or implicit).

For a parabolic PDE like the heat equation:

$$
\frac{\partial u}{\partial t} = \alpha \frac{\partial^2 u}{\partial x^2}
$$

Discretizing on a grid $x_0, x_1, \ldots, x_N$ with spacing $\Delta x$:

$$
\frac{du_i}{dt} = \alpha \frac{u_{i-1} - 2u_i + u_{i+1}}{\Delta x^2}, \quad i = 1, \ldots, N-1
$$

This is an $(N-1)$-dimensional ODE system that Numra solves with `DoPri5`, `Radau5`,
or any other solver from `numra-ode`.

## Spatial Grids

Numra provides 1D, 2D, and 3D grids:

### Grid1D

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust
use numra_pde::Grid1D;

// Uniform grid: 51 points from x=0 to x=1
let grid = Grid1D::uniform(0.0_f64, 1.0, 51);

// Grid with refinement near boundaries
let grid = Grid1D::clustered(0.0_f64, 1.0, 51, 2.0);

// Grid from explicit points
let points = vec![0.0, 0.1, 0.3, 0.6, 1.0];
let grid = Grid1D::from_points(points);

// Grid properties
println!("Total points: {}", grid.len());      // 51
println!("Interior points: {}", grid.n_interior()); // 49
println!("Spacing: {}", grid.dx_uniform());     // 0.02
```

### Grid2D and Grid3D

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust
use numra_pde::{Grid2D, Grid3D};

// 2D uniform grid
let grid2d = Grid2D::uniform(0.0, 1.0, 51, 0.0, 1.0, 51);
println!("Interior 2D points: {}", grid2d.n_interior()); // 49 * 49

// 3D uniform grid
let grid3d = Grid3D::uniform(0.0, 1.0, 11, 0.0, 1.0, 11, 0.0, 1.0, 11);
println!("Interior 3D points: {}", grid3d.n_interior()); // 9 * 9 * 9
```

## Boundary Conditions

Numra supports four types of boundary conditions:

### Dirichlet: $u = g$ on boundary

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust
use numra_pde::boundary::DirichletBC;

let bc = DirichletBC::new(100.0);  // u = 100 at boundary
```

### Neumann: $\partial u / \partial n = g$ on boundary

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust
use numra_pde::boundary::NeumannBC;

let bc = NeumannBC::new(0.0);       // Zero flux (insulated)
let bc = NeumannBC::zero_flux();     // Equivalent shorthand
```

### Robin: $\alpha u + \beta \partial u / \partial n = \gamma$

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust
use numra_pde::boundary::RobinBC;

// General Robin BC
let bc = RobinBC::new(1.0, 0.5, 2.0);  // u + 0.5 du/dn = 2

// Convective BC: h(u - u_ambient) = -k du/dn
let bc = RobinBC::convective(10.0, 25.0);  // h/k = 10, u_ambient = 25
```

### Periodic

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust
use numra_pde::boundary::PeriodicBC;

let bc = PeriodicBC;  // u(x_min) = u(x_max)
```

## The PdeSystem Trait

Custom PDEs implement the `PdeSystem` trait:

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust
pub trait PdeSystem<S: Scalar> {
    /// Compute the RHS of the PDE at a single spatial point.
    fn rhs(&self, t: S, x: S, u: S, du_dx: S, d2u_dx2: S) -> S;

    /// Number of solution components (for systems, default 1).
    fn n_components(&self) -> usize { 1 }
}
```

The `rhs` method receives the current time $t$, spatial coordinate $x$, solution
value $u$, first spatial derivative $\partial u / \partial x$, and second
spatial derivative $\partial^2 u / \partial x^2$. The MOL system computes these
derivatives automatically using central finite differences.

## Built-in Equation Types

### HeatEquation1D

$$
\frac{\partial u}{\partial t} = \alpha \frac{\partial^2 u}{\partial x^2}
$$

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust
use numra_pde::HeatEquation1D;

let heat = HeatEquation1D::new(0.01);  // Thermal diffusivity alpha = 0.01
```

### DiffusionReaction1D

$$
\frac{\partial u}{\partial t} = D \frac{\partial^2 u}{\partial x^2} + R(u)
$$

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust
use numra_pde::DiffusionReaction1D;

// Fisher equation: du/dt = D d2u/dx2 + r*u*(1-u)
let fisher = DiffusionReaction1D::fisher(0.01, 1.0);

// Allen-Cahn equation: du/dt = D d2u/dx2 + u - u^3
let allen_cahn = DiffusionReaction1D::allen_cahn(0.01);

// Custom reaction term
let custom = DiffusionReaction1D::new(0.01, |u: f64| -u * u);
```

### 2D Equations

Numra also provides 2D equation types:

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust
use numra_pde::{HeatEquation2D, AdvectionDiffusion2D, ReactionDiffusion2D};
```

## The MOLSystem

The `MOLSystem` wraps a PDE, grid, and boundary conditions into an `OdeSystem` that
can be solved by any Numra ODE solver:

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust
use numra_pde::{MOLSystem, PdeSystem};

let mol = MOLSystem::new(pde, grid, bc_left, bc_right);

// The MOL system implements OdeSystem, so it can be passed to any solver
// mol.dim() returns the number of interior points
```

## Example: 1D Heat Equation

Solve the heat equation with Dirichlet boundary conditions:

$$
\frac{\partial T}{\partial t} = \alpha \frac{\partial^2 T}{\partial x^2}, \quad
T(0,t) = 1, \quad T(1,t) = 0
$$

Initial condition: $T(x,0) = \sin(\pi x)$.

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust
use numra_pde::{Grid1D, HeatEquation1D, MOLSystem};
use numra_pde::boundary::DirichletBC;
use numra_ode::{DoPri5, Solver, SolverOptions};

// Spatial grid: 51 points on [0, 1]
let grid = Grid1D::uniform(0.0_f64, 1.0, 51);

// Heat equation with diffusivity alpha = 0.01
let pde = HeatEquation1D::new(0.01);

// Boundary conditions
let bc_left = DirichletBC::new(1.0);   // T(0,t) = 1
let bc_right = DirichletBC::new(0.0);  // T(1,t) = 0

// Create MOL system
let mol = MOLSystem::new(pde, grid.clone(), bc_left, bc_right);

// Initial condition (interior points only)
let u0: Vec<f64> = grid.interior_points()
    .iter()
    .map(|&x| (std::f64::consts::PI * x).sin())
    .collect();

// Solve with DoPri5
let options = SolverOptions::default().rtol(1e-6).atol(1e-9);
let result = DoPri5::solve(&mol, 0.0, 1.0, &u0, &options)
    .expect("Solve failed");

// Reconstruct full solution including boundaries
let y_final = result.y_final().unwrap();
let full_solution = mol.build_full_solution(1.0, &y_final);

for (i, &x) in grid.points().iter().enumerate() {
    println!("x = {:.2}, T = {:.6}", x, full_solution[i]);
}
```

## Example: Fisher Equation (Reaction-Diffusion)

The Fisher equation models population spread with logistic growth:

$$
\frac{\partial u}{\partial t} = D \frac{\partial^2 u}{\partial x^2} + r \, u(1 - u)
$$

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust
use numra_pde::{Grid1D, DiffusionReaction1D, MOLSystem};
use numra_pde::boundary::DirichletBC;
use numra_ode::{DoPri5, Solver, SolverOptions};

let grid = Grid1D::uniform(0.0_f64, 1.0, 101);

// Fisher equation: D = 0.01, growth rate r = 1.0
let pde = DiffusionReaction1D::fisher(0.01, 1.0);

let bc_left = DirichletBC::new(1.0);   // Populated region
let bc_right = DirichletBC::new(0.0);  // Unpopulated region

let mol = MOLSystem::new(pde, grid.clone(), bc_left, bc_right);

// Step initial condition: u = 1 for x < 0.3, u = 0 otherwise
let u0: Vec<f64> = grid.interior_points()
    .iter()
    .map(|&x| if x < 0.3 { 1.0 } else { 0.0 })
    .collect();

let options = SolverOptions::default().rtol(1e-6);
let result = DoPri5::solve(&mol, 0.0, 5.0, &u0, &options)
    .expect("Solve failed");

// The traveling wave front should have moved to the right
```

## Example: Using Implicit Solvers for Stiff PDEs

Fine spatial grids lead to stiff ODE systems. Use implicit solvers like `Radau5`:

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust
use numra_pde::{Grid1D, HeatEquation1D, MOLSystem};
use numra_pde::boundary::DirichletBC;
use numra_ode::{Radau5, Solver, SolverOptions};

// Fine grid -> stiff system
let grid = Grid1D::uniform(0.0_f64, 1.0, 201);
let pde = HeatEquation1D::new(1.0);  // Large diffusivity

let bc_left = DirichletBC::new(1.0);
let bc_right = DirichletBC::new(0.0);
let mol = MOLSystem::new(pde, grid.clone(), bc_left, bc_right);

let u0: Vec<f64> = grid.interior_points()
    .iter()
    .map(|&x| (std::f64::consts::PI * x).sin())
    .collect();

// Radau5 handles stiffness from fine spatial grid
let options = SolverOptions::default().rtol(1e-6).atol(1e-9);
let result = Radau5::solve(&mol, 0.0, 0.1, &u0, &options)
    .expect("Solve failed");
```

## Moving Boundary Problems

Numra supports moving boundary problems like the Stefan problem (phase change) through
the `moving` module:

$$
\frac{\partial T}{\partial t} = \alpha \frac{\partial^2 T}{\partial x^2}, \quad 0 < x < s(t)
$$

with the Stefan condition at the moving boundary:

$$
\frac{ds}{dt} = -k \left.\frac{\partial T}{\partial x}\right|_{x=s(t)}
$$

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust
use numra_pde::moving::{Domain1D, Bound, MovingBound, StefanCondition};
```

The moving boundary module uses a coordinate transform to map the moving domain
$[0, s(t)]$ to a fixed domain $[0, 1]$, then applies MOL on the transformed PDE.

## Stability Considerations

### CFL Condition

For explicit time integration of the heat equation, the CFL stability condition requires:

$$
\Delta t \leq \frac{\Delta x^2}{2\alpha}
$$

The adaptive step size control in Numra's ODE solvers automatically satisfies this, but
very fine grids make the system stiff. Guidelines:

| Grid Points | $\Delta x$ | Max Stable $\Delta t$ ($\alpha=1$) | Recommendation |
|---|---|---|---|
| 11 | 0.1 | 0.005 | `DoPri5` (explicit) |
| 51 | 0.02 | 0.0002 | `DoPri5` or `Radau5` |
| 201 | 0.005 | 1.25e-5 | `Radau5` (implicit) |
| 1001 | 0.001 | 5e-7 | `Radau5` or `Bdf` |

### When to Use Implicit Solvers

Use implicit solvers (`Radau5`, `Esdirk`, `Bdf`) when:
- The spatial grid is fine (many grid points)
- The diffusion coefficient is large
- The explicit solver requires extremely small time steps
- The problem has multiple time scales (reaction-diffusion with fast reactions)

## 2D Problems

For 2D PDEs, Numra provides `MOLSystem2D` and 2D equation types:

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust
use numra_pde::{Grid2D, HeatEquation2D, MOLSystem2D};
use numra_pde::boundary2d::BoundaryConditions2D;
```

The 2D MOL system uses sparse matrix assembly for the spatial operators, and the
resulting ODE system can be solved with the same solvers as the 1D case.

## Working with Results

The MOL system produces standard ODE results. To reconstruct the full spatial solution
including boundary values:

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust
let result = DoPri5::solve(&mol, 0.0, tf, &u0, &options)?;

// Get interior values at final time
let u_interior = result.y_final().unwrap();

// Reconstruct full solution with boundary values
let u_full = mol.build_full_solution(tf, &u_interior);

// u_full[0] = left boundary value
// u_full[1..n-1] = interior values
// u_full[n-1] = right boundary value
```

## Practical Considerations

### Grid Resolution

Start with a coarse grid and refine until the solution converges:
- **Heat equation**: 51 points is often sufficient for smooth solutions
- **Reaction-diffusion with sharp fronts**: 201+ points may be needed
- **2D problems**: Memory grows as $N^2$; start with 21x21 and increase

### Custom PDEs

Any PDE of the form $\partial u / \partial t = F(t, x, u, \partial u / \partial x, \partial^2 u / \partial x^2)$ can be solved by implementing `PdeSystem`:

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust
use numra_pde::mol::PdeSystem;
use numra_core::Scalar;

struct MyPDE;

impl PdeSystem<f64> for MyPDE {
    fn rhs(&self, t: f64, x: f64, u: f64, du_dx: f64, d2u_dx2: f64) -> f64 {
        // Burgers' equation: du/dt = nu * d2u/dx2 - u * du/dx
        0.01 * d2u_dx2 - u * du_dx
    }
}
```

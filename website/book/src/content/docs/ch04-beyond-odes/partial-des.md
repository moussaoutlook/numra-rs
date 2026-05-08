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

## 3D Problems

For 3D PDEs on box domains, Numra provides `MOLSystem3D` -- a structural twin
of `MOLSystem2D` extended with a third spatial axis. The same constructors,
the same `OdeSystem` integration, the same reaction hook -- everything that
works in 2D works in 3D. The spatial discretisation uses a 7-point stencil
(centre $-$ 6 face neighbours) assembled as a sparse matrix.

### API shape

`MOLSystem3D` provides four primary constructors mirroring their 2D
counterparts; for the most common equations there is also a higher-level
`equations3d` builder layer (`HeatEquation3D`, `AdvectionDiffusion3D`,
`ReactionDiffusion3D`) that wraps the constructors below with named
parameters.

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust
use numra_pde::{
    BoundaryConditions3D, Grid3D, MOLSystem3D, Operator3DCoefficients,
};

// 3D heat equation: u_t = alpha * Laplacian(u)
let mol = MOLSystem3D::heat(grid.clone(), alpha, &bc);

// Pure Laplacian: u_t = Laplacian(u)
let mol = MOLSystem3D::laplacian(grid.clone(), &bc);

// General linear operator: a*u_xx + b*u_yy + c*u_zz + d*u_x + e*u_y + f*u_z + g*u
let coeffs = Operator3DCoefficients::advection_diffusion(0.01, 1.0, 0.0, 0.0);
let mol = MOLSystem3D::with_operator(grid.clone(), &coeffs, &bc);

// Add a pointwise reaction term: u_t = L[u] + R(t, x, y, z, u)
let mol = MOLSystem3D::heat(grid.clone(), alpha, &bc)
    .with_reaction(|_t, _x, _y, _z, u| u * (1.0 - u));
```

For the named-equation builders:

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust
use numra_pde::{
    AdvectionDiffusion3D, BoundaryConditions3D, Grid3D,
    HeatEquation3D, ReactionDiffusion3D,
};

// Heat equation
let mol = HeatEquation3D::build(grid.clone(), alpha, &bc);

// Advection-diffusion with velocity (vx, vy, vz)
let mol = AdvectionDiffusion3D::build(grid.clone(), 0.01, 1.0, 0.0, 0.0, &bc);

// Custom reaction term
let mol = ReactionDiffusion3D::build(grid.clone(), 0.01, &bc,
    |_t, _x, _y, _z, u| -0.5 * u);

// Fisher-KPP shorthand: D = 0.01, growth rate r = 1.0
let mol = ReactionDiffusion3D::fisher(grid.clone(), 0.01, 1.0, &bc);
```

`MOLSystem3D` implements `OdeSystem<S>` directly, so it composes with every
solver in `numra-ode` -- `DoPri5`, `Radau5`, `Bdf`, `Tsit5`, `Verner` --
without adapter glue.

### Example: 3D heat equation with an analytic solution

For the cube $[0, 1]^3$ with zero Dirichlet boundaries and initial condition

$$
u(x, y, z, 0) = \sin(\pi x)\sin(\pi y)\sin(\pi z),
$$

the exact solution is $u(x, y, z, t) = u(x, y, z, 0) \exp(-3\pi^2 \alpha t)$.
This is the standard sanity check for any 3D parabolic discretisation:

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust
use numra_pde::{BoundaryConditions3D, Grid3D, MOLSystem3D};
use numra_ode::{DoPri5, Solver, SolverOptions};

let alpha = 0.01_f64;
let n = 13;
let grid = Grid3D::uniform(0.0, 1.0, n, 0.0, 1.0, n, 0.0, 1.0, n);
let bc = BoundaryConditions3D::all_zero_dirichlet();
let mol = MOLSystem3D::heat(grid.clone(), alpha, &bc);

// Initial condition on interior points (column-major: x varies fastest)
let nx_int = n - 2;
let ny_int = n - 2;
let nz_int = n - 2;
let n_int = nx_int * ny_int * nz_int;

let pi = std::f64::consts::PI;
let mut u0 = vec![0.0; n_int];
for kk in 0..nz_int {
    for jj in 0..ny_int {
        for ii in 0..nx_int {
            let x = grid.x_grid.points()[ii + 1];
            let y = grid.y_grid.points()[jj + 1];
            let z = grid.z_grid.points()[kk + 1];
            u0[kk * (nx_int * ny_int) + jj * nx_int + ii] =
                (pi * x).sin() * (pi * y).sin() * (pi * z).sin();
        }
    }
}

let options = SolverOptions::default().rtol(1e-6).atol(1e-9);
let result = DoPri5::solve(&mol, 0.0, 0.5, &u0, &options).unwrap();
let y_final = result.y_final().unwrap();

// Compare to the analytic decay at the cube centre
let decay = (-3.0 * pi * pi * alpha * 0.5).exp();
let mid = (nz_int / 2) * (nx_int * ny_int) + (ny_int / 2) * nx_int + (nx_int / 2);
println!("computed = {:.6}, exact = {:.6}", y_final[mid], decay);
```

A $13^3$ grid is coarse for visual quality but is enough for the centre-cell
relative error to come in below $5\%$, which is the regression bound used in
the test suite (`numra-pde/src/mol3d.rs::test_mol3d_heat_decay`). Refine to
$25^3$ or $33^3$ for production work; expect memory to scale as $N^3$ and
runtime as $N^3$ per RHS evaluation (sparse matvec with seven nonzeros per row).

### Example: 3D Fisher--KPP with a reaction term

The Fisher--Kolmogorov--Petrovsky--Piskunov equation in 3D models invasion
fronts with logistic growth:

$$
\frac{\partial u}{\partial t} = D \, \nabla^2 u + r \, u(1 - u).
$$

With a small initial bump in the centre of the cube, the solution spreads
outward as a roughly spherical wavefront:

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust
use numra_pde::{BoundaryConditions3D, Grid3D, MOLSystem3D};
use numra_ode::{DoPri5, Solver, SolverOptions};

let n = 21;
let grid = Grid3D::uniform(0.0, 1.0, n, 0.0, 1.0, n, 0.0, 1.0, n);
let bc = BoundaryConditions3D::all_zero_dirichlet();

// D = 0.01, growth rate r = 1.0
let mol = MOLSystem3D::heat(grid.clone(), 0.01_f64, &bc)
    .with_reaction(|_t, _x, _y, _z, u| u * (1.0 - u));

// IC: small spherical bump near the cube centre
let nx_int = n - 2;
let ny_int = n - 2;
let nz_int = n - 2;
let n_int = nx_int * ny_int * nz_int;

let mut u0 = vec![0.0_f64; n_int];
for kk in 0..nz_int {
    for jj in 0..ny_int {
        for ii in 0..nx_int {
            let x = grid.x_grid.points()[ii + 1];
            let y = grid.y_grid.points()[jj + 1];
            let z = grid.z_grid.points()[kk + 1];
            let r2 = (x - 0.5).powi(2) + (y - 0.5).powi(2) + (z - 0.5).powi(2);
            if r2 < 0.05 {
                u0[kk * (nx_int * ny_int) + jj * nx_int + ii] = 0.5;
            }
        }
    }
}

let options = SolverOptions::default().rtol(1e-4);
let result = DoPri5::solve(&mol, 0.0, 0.5, &u0, &options).unwrap();
```

The reaction closure signature is `Fn(t, x, y, z, u) -> S`, identical to the
2D version with a `z` axis added. It runs once per interior grid point per
RHS evaluation; keep it cheap.

### Example: 3D advection--diffusion via a custom operator

For problems that don't fit `heat` or `laplacian`, `Operator3DCoefficients`
exposes the full general operator
$a u_{xx} + b u_{yy} + c u_{zz} + d u_x + e u_y + f u_z + g u$:

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust
use numra_pde::{
    BoundaryConditions3D, Grid3D, MOLSystem3D, Operator3DCoefficients,
};

let grid = Grid3D::uniform(0.0, 1.0, 21, 0.0, 1.0, 21, 0.0, 1.0, 21);
let bc = BoundaryConditions3D::all_zero_dirichlet();

// Advection-diffusion: D=0.01, velocity field (1, 0, 0)
let coeffs = Operator3DCoefficients::advection_diffusion(0.01_f64, 1.0, 0.0, 0.0);
let mol = MOLSystem3D::with_operator(grid, &coeffs, &bc);
```

Two things to know about this constructor:

- **First-derivative terms use central differences.** This is the same choice
  as `assemble_operator_2d`. Central differencing is non-monotone on
  advection-dominated regimes; if your Péclet number is high you'll see
  oscillations and should expect to switch to upwind discretisation outside
  the built-in path. We don't ship upwind in v1.
- **The operator is asymmetric when any first-derivative coefficient is
  non-zero.** Symmetric solvers (e.g. CG) won't work directly; the standard
  Krylov path for non-symmetric operators is GMRES.

### Boundary conditions in 3D

`BoundaryConditions3D` carries one BC per face of the box:

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust
use numra_pde::boundary::BoxedBC;
use numra_pde::BoundaryConditions3D;

// All Dirichlet = 0
let bc = BoundaryConditions3D::<f64>::all_zero_dirichlet();

// Same Dirichlet value everywhere
let bc = BoundaryConditions3D::all_dirichlet(1.0_f64);

// Mixed: hot left face, cold right face, insulated top/bottom/front/back
let bc = BoundaryConditions3D {
    x_min: BoxedBC::dirichlet(1.0_f64),
    x_max: BoxedBC::dirichlet(0.0),
    y_min: BoxedBC::neumann(0.0),
    y_max: BoxedBC::neumann(0.0),
    z_min: BoxedBC::neumann(0.0),
    z_max: BoxedBC::neumann(0.0),
};
```

All four BC kinds (Dirichlet, Neumann, Robin via `BoxedBC`) work on every face;
each face is independent.

### Reconstructing the full 3D solution

The MOL system stores only interior values in column-major order
($x$ varies fastest, then $y$, then $z$). Use `build_full_solution` to put
boundary values back in for visualisation or post-processing:

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust
let result = DoPri5::solve(&mol, 0.0, t_final, &u0, &options).unwrap();
let u_interior = result.y_final().unwrap();

// nx*ny*nz entries; index with grid.linear_index(i, j, k)
let u_full = mol.build_full_solution(&u_interior);

// Read a value at full-grid index (i, j, k)
let (i, j, k) = (5, 5, 5);
let v = u_full[mol.grid().linear_index(i, j, k)];
```

`build_full_solution` fills boundary cells with zero in v1 (Dirichlet-zero
assumption); for non-zero boundary data, write the boundary slices yourself
using `BoundaryConditions3D` if you need them in the output array.

### Implicit solvers get an analytical Jacobian for free

`MOLSystem2D` and `MOLSystem3D` both override `OdeSystem::jacobian` --
the assembled sparse spatial operator already *is* $\partial L[u]/\partial u$, so
the override is a direct copy from CSC into the dense buffer the solver
expects. Pointwise reaction terms add a diagonal contribution
($\partial R_i/\partial u_j$ is non-zero only when $j = i$, so off-diagonal entries
cannot be populated by a pointwise reaction); the diagonal-FD fallback
costs one closure call per interior point instead of the $O(N)$ rhs
evaluations the trait-default FD path would need.

The practical consequence: when you solve an MOL system with `Radau5`,
`Bdf`, or any other implicit solver in `numra-ode`, the solver uses the
analytical Jacobian without you doing anything. There's no opt-in.
There's no flag.

The wall-clock impact is bounded because Radau5 caches the Jacobian
across steps when Newton converges quickly. On the canonical stiff 2D
heat-with-reaction workload (Allen-Cahn-like, $\alpha = 0.5$, $19^2 = 361$
interior points, $\text{rtol} = 10^{-6}$), the analytical path is **~2.8%
faster** than the FD-Jacobian fallback. Magnitudes will vary with
problem size, stiffness, grid resolution, and how aggressively the
solver rebuilds the Jacobian — workloads with frequent Jacobian rebuilds
(stiffer problems, looser tolerances triggering more rejected steps,
larger interior dimensions where the FD sweep cost grows) see larger
relative wins.

The full performance bench lives at
`numra-bench/benches/pde_mol.rs::bench_mol2d_radau5_jacobian_path` and
runs on every CI invocation of `cargo bench`.

### Forward sensitivity on PDE problems

`ParametricMOLSystem2D` and `ParametricMOLSystem3D` wrap a 2D / 3D
heat-equation MOL discretisation as a `ParametricOdeSystem`, letting you
compute $\partial u/\partial p$ for parameters $p = [\alpha, p_{R,0}, p_{R,1}, \ldots]$
through the same `solve_forward_sensitivity` entry point that ODE users
already know. No hand-rolled parametric system; no manual Jacobian
derivation.

The parameter layout is fixed: slot 0 is the diffusion coefficient
$\alpha$; remaining slots are reaction parameters supplied alongside the
reaction closure. The closure receives the *full* parameter slice so
it can reference $p[0]$ for the diffusion coefficient or $p[1..]$ for
its own reaction-specific parameters.

**Example: fitting a Fisher–KPP diffusion + growth-rate pair**

Suppose you have observed concentration data and want to know how the
solution depends on the diffusion coefficient $\alpha$ and the logistic
growth rate $r$ — the canonical setup for parameter-identification
workflows on reaction-diffusion problems.

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust
use numra_ode::sensitivity::solve_forward_sensitivity;
use numra_ode::{Radau5, SolverOptions};
use numra_pde::{BoundaryConditions2D, Grid2D, ParametricMOLSystem2D};

let n = 21;
let grid = Grid2D::uniform(0.0, 1.0, n, 0.0, 1.0, n);
let bc = BoundaryConditions2D::all_zero_dirichlet();

// Parameters: [alpha=0.01 (diffusion), r=1.0 (growth rate)]
let mol = ParametricMOLSystem2D::heat_with_reaction(
    grid.clone(),
    0.01_f64,           // alpha nominal
    &bc,
    vec![1.0],          // reaction params: just [r]
    |_t, _x, _y, u, p: &[f64]| {
        // p[0] is alpha; p[1] is the growth rate
        p[1] * u * (1.0 - u)
    },
);

// Initial condition: a small bump in the centre
let nx_int = n - 2;
let n_int = nx_int * nx_int;
let mut u0 = vec![0.0_f64; n_int];
for jj in 0..nx_int {
    for ii in 0..nx_int {
        let x = grid.x_grid.points()[ii + 1];
        let y = grid.y_grid.points()[jj + 1];
        let r2 = (x - 0.5).powi(2) + (y - 0.5).powi(2);
        if r2 < 0.04 {
            u0[jj * nx_int + ii] = 0.5;
        }
    }
}

let opts = SolverOptions::default().rtol(1e-6).atol(1e-9);
let result = solve_forward_sensitivity::<Radau5, f64, _>(
    &mol, 0.0, 0.5, &u0, &opts,
).unwrap();

// ∂u/∂α at the final time, at the cube centre
let mid = (nx_int / 2) * nx_int + (nx_int / 2);
let last_t = result.t.len() - 1;
let du_dalpha = result.dyi_dpj(last_t, mid, 0);
let du_dr     = result.dyi_dpj(last_t, mid, 1);
println!("∂u/∂α = {du_dalpha:.6}");
println!("∂u/∂r = {du_dr:.6}");
```

**What's analytical and what isn't**

The implementation exploits the linearity of the spatial operator in
$\alpha$: the assembled Laplacian scales exactly as $\alpha \cdot L_0$, and the
boundary RHS contribution scales the same way (including for Neumann
ghost-point flux terms — the ghost-point construction makes the flux
contribution $\alpha$-scaled automatically when the operator is a pure
Laplacian). So the parametric system pre-assembles $L_0$ and `bc_rhs_0`
once and scales by $\alpha$ at runtime. No per-call matrix re-assembly.

This means:

- $\partial(\text{rhs})/\partial y$ is fully analytical: $\alpha \cdot L_0$ plus a diagonal
  reaction contribution (FD on the closure, single call per interior
  point thanks to the pointwise-reaction property).
- $\partial(\text{rhs})/\partial \alpha$ is analytical: $L_0 \cdot y + \mathrm{bc\_rhs}_0$.
- $\partial(\text{rhs})/\partial p_{R,k}$ is FD-on-the-closure with the matching parameter
  slot perturbed (one closure call per interior point per reaction
  parameter).

All four `has_analytical_jacobian_*` flags are set correctly, so the
augmented-system hot path takes the analytical route.

**Scope and what's deferred**

`ParametricMOLSystem*` v1 ships heat-equation parametrisation: a single
parameter slot for $\alpha$. Full operator parametrisation — separate
slots for the diffusion coefficient $D$ and the velocity $(v_x, v_y)$ in
advection–diffusion, where the operator is *not* linear in any single
coefficient — would require per-parameter-change re-assembly and is
deferred. Tracked in `internal-followups.md` under §PDE.

For workflows where the parameters drive a custom operator that doesn't
fit the heat-equation shape, the escape hatch is to implement
`ParametricOdeSystem` directly — the full trait surface is in the
`ch11-uncertainty/sensitivity-analysis` chapter.

**Bench coverage**

`numra-bench/benches/pde_mol.rs::bench_mol2d_forward_sensitivity` runs
the parametric path on a stiff 2D heat-with-reaction workload at
$N_s \in \{1, 2, 3\}$ parameters. The current cost is dominated by dense
LU on the $N_{\text{int}}(1 + N_s)$-dimensional augmented system; future
block-diagonal-LU work will reduce this.

### Memory and runtime scaling

3D MOL is honest about its costs:

| Grid | Interior points | Sparse operator nnz | Recommendation |
|---|---|---|---|
| $9^3$ | $343$ | $\sim 2{,}400$ | `DoPri5`; runs in milliseconds |
| $17^3$ | $3{,}375$ | $\sim 23{,}600$ | `DoPri5` for explicit, `Radau5` for stiff |
| $33^3$ | $29{,}791$ | $\sim 209{,}000$ | `Radau5` if stiff; expect seconds per solve |
| $65^3$ | $250{,}047$ | $\sim 1{,}750{,}000$ | Stiff path mandatory; expect minutes |

Scaling rule of thumb: dense Jacobian factorisation in `Radau5` / `Bdf` costs
$O(N^3)$ in the interior dimension, so going from $33^3$ ($N \approx 30k$) to
$65^3$ ($N \approx 250k$) makes implicit solves $\sim 600\times$ more
expensive. Sparse-Jacobian-aware factorisation for MOL is on the follow-up
list; until then, keep grids modest for stiff 3D problems.

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

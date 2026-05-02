---
title: "Differential-Algebraic Equations"
---

Many physical systems involve not just differential equations, but also algebraic
constraints that must be satisfied at every point in time. The position of a pendulum
is constrained by its length. Currents in an electrical circuit obey Kirchhoff's
laws. Concentrations in a chemical reactor may be constrained by conservation.
These systems are called **differential-algebraic equations (DAEs)**.

## What Is a DAE?

A DAE is a system of the form:

$$
M \cdot \mathbf{y}'(t) = f(t, \mathbf{y})
$$

where $ M $ is the **mass matrix**, which may be singular. When $ M $ is the
identity matrix, this reduces to a standard ODE. When $ M $ has zero rows, those
rows correspond to *algebraic constraints* -- equations with no time derivatives
that the solution must satisfy at every instant.

**Example:** Consider two equations:

$$
\begin{aligned}
y_1' &= -y_1 + y_2 \quad &\text{(differential)} \\\\
0 &= y_1 - y_2 \quad &\text{(algebraic)}
\end{aligned}
$$

The mass matrix is:

$$
M = \begin{pmatrix} 1 & 0 \\\\ 0 & 0 \end{pmatrix}
$$

The algebraic constraint $ y_1 = y_2 $ must hold at all times. Substituting into
the differential equation gives $ y_1' = 0 $, so the solution is simply
$ y_1(t) = y_2(t) = y_1(0) $.

## The DAE Index

The **index** of a DAE measures how far it is from being an ODE. Roughly speaking,
the index is the number of times you must differentiate the algebraic constraints
before you can express the system as a pure ODE.

- **Index 0:** $ M $ is nonsingular -- this is just an ODE in disguise.
- **Index 1:** One differentiation of the constraints yields a standard ODE.
  These are the most common and well-supported type.
- **Index 2 and higher:** Multiple differentiations needed. These are much harder
  to solve numerically and often require special treatment
  (see [Index Reduction](index-reduction)).

Numra's stiff solvers (Radau5 and BDF) support **index-1 DAEs** natively.

## The DaeProblem API

Numra provides the `DaeProblem` struct for defining DAEs with mass matrices.

### Construction

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust
use numra::ode::{DaeProblem};

let dae = DaeProblem::new(
    f,              // RHS function: f(t, y, dydt)
    mass,           // Mass matrix function: mass(buf)
    t0,             // Initial time
    tf,             // Final time
    y0,             // Initial state (Vec<f64>, must be consistent!)
    alg_indices,    // Indices of algebraic variables (Vec<usize>)
);
```

The key components:

- **`f`** -- The right-hand side function, identical to the ODE case. It computes
  $ f(t, y) $ such that $ M \cdot y' = f(t, y) $.
- **`mass`** -- A closure that fills an $ n \times n $ row-major array with the
  mass matrix entries. Called once at the start of the solve.
- **`y0`** -- The initial state. **This must satisfy the algebraic constraints!**
  Inconsistent initial conditions will cause the solver to fail or produce
  incorrect results.
- **`alg_indices`** -- A vector listing which state variables are algebraic (i.e.,
  which rows of $ M $ are zero). This tells the solver which equations are
  constraints.

### Mass Matrix Format

The mass matrix is stored in row-major order in a flat array of length $ n^2 $:

```text
mass[i * n + j] = M[i, j]
```

For a diagonal mass matrix (the most common case), only the diagonal entries
$ M[i, i] $ are nonzero.

## Example: Simple Index-1 DAE

The simplest possible DAE -- one differential equation coupled with an algebraic
constraint:

$$
\begin{aligned}
y_1' &= -y_1 + y_2 \\\\
0 &= y_1 - y_2
\end{aligned}
$$

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust
use numra::ode::{DaeProblem, Radau5, Solver, SolverOptions};

let dae = DaeProblem::new(
    // RHS: f(t, y)
    |_t, y: &[f64], dydt: &mut [f64]| {
        dydt[0] = -y[0] + y[1];    // differential equation
        dydt[1] = y[0] - y[1];     // algebraic constraint: 0 = y1 - y2
    },
    // Mass matrix: diag(1, 0)
    |mass: &mut [f64]| {
        for i in 0..4 { mass[i] = 0.0; }
        mass[0] = 1.0;  // M[0,0] = 1 (differential)
        // M[1,1] = 0    (algebraic)
    },
    0.0,                // t0
    1.0,                // tf
    vec![1.0, 1.0],     // y0 (consistent: y1 = y2)
    vec![1],            // algebraic index: y2 (index 1)
);

let options = SolverOptions::default()
    .rtol(1e-4)
    .atol(1e-6);

let result = Radau5::solve(&dae, 0.0, 1.0, &[1.0, 1.0], &options)
    .expect("Simple DAE should solve");

let yf = result.y_final().unwrap();
// Both components remain at 1.0 (since y1' = -y1 + y2 = 0 when y1 = y2)
assert!((yf[0] - 1.0).abs() < 1e-3);
assert!((yf[1] - 1.0).abs() < 1e-3);
```

## Example: Pendulum as a DAE

The classic pendulum in Cartesian coordinates is a natural DAE. The position is
constrained to lie on a circle of radius $ L $:

$$
\begin{aligned}
x' &= v_x \\\\
y' &= v_y \\\\
v_x' &= -\lambda\,x \\\\
v_y' &= -\lambda\,y - g \\\\
0 &= x^2 + y^2 - L^2
\end{aligned}
$$

where $ \lambda $ is the Lagrange multiplier (tension per unit mass).

The mass matrix is:

$$
M = \begin{pmatrix}
1 & 0 & 0 & 0 & 0 \\\\
0 & 1 & 0 & 0 & 0 \\\\
0 & 0 & 1 & 0 & 0 \\\\
0 & 0 & 0 & 1 & 0 \\\\
0 & 0 & 0 & 0 & 0
\end{pmatrix}
$$

The last row is zero because the constraint $ x^2 + y^2 = L^2 $ has no derivative --
it is a purely algebraic equation.

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust
use numra::ode::{DaeProblem, Radau5, Solver, SolverOptions};

let g = 9.81;
let l = 1.0;  // pendulum length

// State: [x, y, vx, vy, lambda]
// Initial condition: pendulum at angle theta0 from vertical
let theta0: f64 = std::f64::consts::PI / 6.0;  // 30 degrees
let x0 = l * theta0.sin();
let y0 = -l * theta0.cos();  // y points downward

let dae = DaeProblem::new(
    move |_t, y: &[f64], dydt: &mut [f64]| {
        let x = y[0];
        let y_pos = y[1];
        let vx = y[2];
        let vy = y[3];
        let lam = y[4];

        dydt[0] = vx;                    // x' = vx
        dydt[1] = vy;                    // y' = vy
        dydt[2] = -lam * x;              // vx' = -lambda * x
        dydt[3] = -lam * y_pos - g;      // vy' = -lambda * y - g
        dydt[4] = x * x + y_pos * y_pos - l * l;  // constraint
    },
    |mass: &mut [f64]| {
        // 5x5 mass matrix, all zeros first
        for i in 0..25 { mass[i] = 0.0; }
        // Diagonal: [1, 1, 1, 1, 0]
        mass[0]  = 1.0;  // M[0,0]
        mass[6]  = 1.0;  // M[1,1]
        mass[12] = 1.0;  // M[2,2]
        mass[18] = 1.0;  // M[3,3]
        // M[4,4] = 0 (algebraic)
    },
    0.0,            // t0
    10.0,           // tf (10 seconds)
    vec![x0, y0, 0.0, 0.0, g * y0.cos() / l],  // consistent initial conditions
    vec![4],        // algebraic variable: lambda (index 4)
);

let options = SolverOptions::default()
    .rtol(1e-6)
    .atol(1e-8);

let result = Radau5::solve(
    &dae, 0.0, 10.0,
    &[x0, y0, 0.0, 0.0, g * y0.cos() / l],
    &options,
).expect("Pendulum DAE should solve");
```

> **Note:** The pendulum formulation above is actually an **index-3 DAE**. To solve
> it with Numra's index-1 solvers, you need to apply index reduction -- see
> [Index Reduction](index-reduction) for the technique. An alternative approach
> is to differentiate the constraint twice to get the *acceleration-level* form,
> or use the stabilized index-1 formulation shown in that chapter.

## Example: RC Circuit

A simple resistor-capacitor circuit illustrates DAEs arising from Kirchhoff's laws.
Consider a voltage source $ V(t) $, a resistor $ R $, and a capacitor $ C $
in series:

$$
\begin{aligned}
C \cdot V_C' &= i \\\\
0 &= V(t) - R\,i - V_C
\end{aligned}
$$

where $ V_C $ is the capacitor voltage and $ i $ is the current (algebraic variable).

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust
use numra::ode::{DaeProblem, Radau5, Solver, SolverOptions};

let r = 1000.0;   // 1 kOhm
let c = 1e-6;     // 1 microfarad
let v_source = |t: f64| -> f64 { 5.0 * (100.0 * t).sin() };  // 5V, 100 rad/s

// State: [V_C, i]
let dae = DaeProblem::new(
    move |t, y: &[f64], dydt: &mut [f64]| {
        let v_c = y[0];
        let i = y[1];
        dydt[0] = i / c;                         // C * V_C' = i
        dydt[1] = v_source(t) - r * i - v_c;     // KVL: 0 = V(t) - R*i - V_C
    },
    move |mass: &mut [f64]| {
        for k in 0..4 { mass[k] = 0.0; }
        mass[0] = 1.0;  // M[0,0] = 1 (differential for V_C)
        // M[1,1] = 0    (algebraic for i)
    },
    0.0, 0.1,
    vec![0.0, 0.0],  // initially uncharged
    vec![1],          // current is algebraic
);

let options = SolverOptions::default().rtol(1e-6).atol(1e-9);
let result = Radau5::solve(&dae, 0.0, 0.1, &[0.0, 0.0], &options)
    .expect("RC circuit DAE should solve");
```

## Non-Singular Mass Matrices

Not all mass matrices represent DAEs. A non-singular mass matrix simply rescales
the derivatives:

$$
M \cdot \mathbf{y}' = f(t, \mathbf{y}) \implies \mathbf{y}' = M^{-1} f(t, \mathbf{y})
$$

This arises in finite element discretizations of PDEs, where $ M $ is the
*consistent mass matrix*.

In Numra, you use `DaeProblem` with an empty `alg_indices` vector:

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust
use numra::ode::{DaeProblem, Radau5, Solver, SolverOptions};

// 2 * y' = -y  =>  y' = -y/2  =>  y(t) = exp(-t/2)
let dae = DaeProblem::new(
    |_t, y: &[f64], dydt: &mut [f64]| {
        dydt[0] = -y[0];     // RHS is -y
    },
    |mass: &mut [f64]| {
        mass[0] = 2.0;       // Mass matrix M = [2]
    },
    0.0, 1.0,
    vec![1.0],
    vec![],  // No algebraic indices -- mass is nonsingular
);

let options = SolverOptions::default().rtol(1e-4).atol(1e-6);
let result = Radau5::solve(&dae, 0.0, 1.0, &[1.0], &options)
    .expect("Non-singular mass matrix ODE should solve");

let yf = result.y_final().unwrap();
let exact = (-0.5_f64).exp();
assert!((yf[0] - exact).abs() < 1e-3);
```

## Solver Support for DAEs

Both Radau5 and BDF support DAEs via the mass matrix formulation:

| Solver | DAE support | Index supported | Mass matrix handling |
|--------|-------------|----------------|---------------------|
| Radau5 | Yes | Index-1 | Incorporated into LU factorization: $ (\gamma M - J) $ |
| BDF | Yes | Index-1 | Incorporated into iteration matrix: $ (M - \gamma J) $ |
| Esdirk | No (ODE only) | -- | Not applicable |
| DoPri5, Tsit5, Vern | No (ODE only) | -- | Not applicable |

### How the Solvers Use the Mass Matrix

For **Radau5**, the transformed iteration matrices become:

$$
E_1 = \frac{\mu_1}{h} M - J, \quad
E_2 = \begin{pmatrix}
\frac{\alpha}{h} M - J & -\frac{\beta}{h} M \\\\
\frac{\beta}{h} M & \frac{\alpha}{h} M - J
\end{pmatrix}
$$

where $ \mu_1, \alpha, \beta $ are the eigenvalue-related constants of the Radau IIA
method. For a standard ODE, $ M = I $ and these reduce to the usual forms.

For **BDF**, the iteration matrix becomes:

$$
M - \gamma J \quad \text{where } \gamma = \frac{h \beta_k}{\alpha_0}
$$

instead of the usual $ I - \gamma J $.

## Consistent Initial Conditions

The most common pitfall with DAEs is providing **inconsistent initial conditions**.
The algebraic constraints must be satisfied at $ t = t_0 $:

$$
\text{If } M[i, :] = 0, \text{ then } f_i(t_0, y_0) = 0
$$

If this condition is violated, the solver must project the initial conditions onto
the constraint manifold before it can begin integration. Numra does not currently
perform automatic initialization -- you must provide consistent initial conditions.

**How to check consistency:**

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust
// Evaluate f(t0, y0) for the algebraic components
let mut dydt = vec![0.0; dim];
problem.rhs(t0, &y0, &mut dydt);

// For each algebraic index, the residual should be zero
for &idx in &alg_indices {
    assert!(dydt[idx].abs() < 1e-10,
        "Inconsistent IC at index {}: residual = {}", idx, dydt[idx]);
}
```

## Summary

- DAEs arise naturally from constrained physical systems.
- Numra models them via the **mass matrix** formulation: $ M \cdot y' = f(t, y) $.
- Use `DaeProblem` with the mass matrix closure and algebraic index list.
- **Radau5** is the preferred solver for DAEs (L-stable, stiffly accurate).
- **BDF** also supports index-1 DAEs.
- **Always provide consistent initial conditions** -- the algebraic constraints must
  be satisfied at $ t_0 $.

---
title: "Index Reduction"
---

Not all DAEs are created equal. The **index** of a DAE determines how difficult it
is to solve numerically. Numra's solvers handle index-1 DAEs natively, but many
physical systems naturally produce higher-index DAEs. This section explains what
the index means, why it matters, and how to reduce a higher-index DAE to index-1
form so that Numra can solve it.

## What Is the DAE Index?

The **differentiation index** of a DAE is the minimum number of times you must
differentiate the system (or parts of it) with respect to time before you can
express it as an explicit ODE:

$$
\mathbf{y}' = g(t, \mathbf{y})
$$

### Index-0: Regular ODE

$$
\mathbf{y}' = f(t, \mathbf{y})
$$

The mass matrix is the identity (or any nonsingular matrix). No differentiation
needed. Every standard ODE solver works.

### Index-1: One Differentiation

$$
\begin{aligned}
\mathbf{x}' &= f(\mathbf{x}, \mathbf{z}) \\\\
0 &= g(\mathbf{x}, \mathbf{z})
\end{aligned}
$$

where $ \mathbf{x} $ are differential variables and $ \mathbf{z} $ are algebraic
variables. The key requirement is that $ \partial g / \partial \mathbf{z} $ is
nonsingular, meaning we can solve for $ \mathbf{z} $ from the constraint.

Differentiating the constraint once:

$$
0 = \frac{\partial g}{\partial \mathbf{x}} f(\mathbf{x}, \mathbf{z})
  + \frac{\partial g}{\partial \mathbf{z}} \mathbf{z}'
$$

gives an expression for $ \mathbf{z}' $, making the system an ODE. One
differentiation = index 1.

**Example:** The RC circuit DAE from the previous section is index-1. The algebraic
equation $ 0 = V(t) - R \cdot i - V_C $ can be solved directly for $ i $.

### Index-2: Two Differentiations

$$
\begin{aligned}
\mathbf{x}' &= f(\mathbf{x}, \mathbf{z}) \\\\
0 &= g(\mathbf{x})
\end{aligned}
$$

Notice that the constraint $ g $ depends only on $ \mathbf{x} $, not on
$ \mathbf{z} $. You must differentiate twice to get an ODE for $ \mathbf{z} $:

- First differentiation: $ 0 = \frac{\partial g}{\partial \mathbf{x}} \mathbf{x}' = \frac{\partial g}{\partial \mathbf{x}} f(\mathbf{x}, \mathbf{z}) $
  (this is a *hidden constraint* on $ \mathbf{z} $)
- Second differentiation: gives $ \mathbf{z}' $ explicitly

**Example:** A constrained mechanical system with the constraint expressed at the
position level (not velocity or acceleration).

### Index-3: Three Differentiations

The most famous example is the pendulum in Cartesian coordinates from the
[DAE Systems](dae-systems) chapter:

$$
\begin{aligned}
x' &= v_x \\\\
y' &= v_y \\\\
v_x' &= -\lambda\,x \\\\
v_y' &= -\lambda\,y - g \\\\
0 &= x^2 + y^2 - L^2
\end{aligned}
$$

The constraint involves only positions ($ x, y $), not velocities or the
multiplier $ \lambda $. You must differentiate three times:

1. $ 0 = 2x\,v_x + 2y\,v_y $ (velocity-level constraint)
2. $ 0 = 2(v_x^2 + v_y^2) + 2x(-\lambda x) + 2y(-\lambda y - g) $ (acceleration-level)
3. Differentiate again to find $ \lambda' $

This is an **index-3** DAE. Standard solvers (including Numra's) cannot handle it
directly.

## Why Does Index Matter?

Higher-index DAEs are progressively harder to solve numerically:

| Index | Difficulty | Solver requirements |
|-------|-----------|---------------------|
| 0 | Easy | Any ODE solver |
| 1 | Moderate | Implicit solver with mass matrix (Radau5, BDF) |
| 2 | Hard | Specialized index-2 solver or index reduction |
| 3 | Very hard | Almost always requires index reduction |
| > 3 | Extreme | Reformulate the problem |

The fundamental issue is that numerical errors accumulate differently at each index
level. For an index-1 DAE, the algebraic constraint is satisfied to the solver's
tolerance. For index-2, errors in the constraint can *grow linearly* with time.
For index-3, they can grow *quadratically*. This is called **drift**.

## Index Reduction Techniques

### Technique 1: Differentiate the Constraints

The most direct approach: differentiate the algebraic constraints until the system
becomes index-1 (or even index-0).

**Pendulum example:** Replace the position-level constraint with the acceleration-level
constraint:

$$
\begin{aligned}
x' &= v_x \\\\
y' &= v_y \\\\
v_x' &= -\lambda\,x \\\\
v_y' &= -\lambda\,y - g \\\\
0 &= v_x^2 + v_y^2 - \lambda(x^2 + y^2) - g\,y
\end{aligned}
$$

This is now index-1 (we can solve directly for $ \lambda $ from the last equation).

**Problem:** The position-level constraint $ x^2 + y^2 = L^2 $ is no longer
enforced. Numerical errors will cause the pendulum length to drift over time.

### Technique 2: Baumgarte Stabilization

Add damping terms to the differentiated constraint to prevent drift. Instead of
replacing the constraint, modify the acceleration-level equation:

$$
0 = \ddot{g} + 2\alpha\,\dot{g} + \beta^2\,g
$$

where $ g = x^2 + y^2 - L^2 $, $ \dot{g} = 2(x\,v_x + y\,v_y) $, and
$ \alpha, \beta > 0 $ are damping parameters.

This drives both the constraint violation $ g $ and its time derivative $ \dot{g} $
to zero exponentially. The resulting system is index-1.

For the pendulum:

$$
\begin{aligned}
x' &= v_x \\\\
y' &= v_y \\\\
v_x' &= -\lambda\,x \\\\
v_y' &= -\lambda\,y - g_{\text{grav}} \\\\
0 &= v_x^2 + v_y^2 - \lambda(x^2 + y^2) - g_{\text{grav}}\,y \\\\
  &\quad + 2\alpha(x\,v_x + y\,v_y) + \beta^2(x^2 + y^2 - L^2)
\end{aligned}
$$

**Choosing $ \alpha $ and $ \beta $:** Typical values are $ \alpha = \beta = 5{-}20 / T $
where $ T $ is the characteristic time of the system. Too large causes stiffness;
too small allows drift.

### Technique 3: GGL Formulation (Gear-Gupta-Leimkuhler)

This adds the original constraint as an additional equation alongside the
differentiated form. The system is augmented with a new multiplier $ \mu $:

$$
\begin{aligned}
x' &= v_x + \frac{\partial g}{\partial x}^T \mu \\\\
y' &= v_y + \frac{\partial g}{\partial y}^T \mu \\\\
v_x' &= -\lambda\,x \\\\
v_y' &= -\lambda\,y - g_{\text{grav}} \\\\
0 &= g(\mathbf{x}) = x^2 + y^2 - L^2 \\\\
0 &= \dot{g}(\mathbf{x}, \mathbf{v}) = x\,v_x + y\,v_y
\end{aligned}
$$

The extra multiplier $ \mu $ projects the velocity onto the constraint manifold.
This formulation is index-2 (which some specialized solvers can handle), and with
one more differentiation becomes a well-conditioned index-1 system.

### Technique 4: Reformulate as ODE

Sometimes the cleanest approach is to eliminate the algebraic variables entirely.
For the pendulum, use the angle $ \theta $ instead of Cartesian coordinates:

$$
\theta'' = -\frac{g}{L}\sin\theta
$$

This is a pure ODE -- no algebraic constraints at all!

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust
use numra::ode::{DoPri5, Solver, OdeProblem, SolverOptions};

let g = 9.81;
let l = 1.0;

// State: [theta, omega] where omega = theta'
let problem = OdeProblem::new(
    move |_t, y: &[f64], dydt: &mut [f64]| {
        dydt[0] = y[1];                    // theta' = omega
        dydt[1] = -(g / l) * y[0].sin();   // omega' = -(g/L) sin(theta)
    },
    0.0, 10.0,
    vec![std::f64::consts::PI / 6.0, 0.0],  // 30 degrees, zero velocity
);

let options = SolverOptions::default().rtol(1e-10).atol(1e-12);
let result = DoPri5::solve(
    &problem, 0.0, 10.0,
    &[std::f64::consts::PI / 6.0, 0.0],
    &options,
).expect("Pendulum ODE is non-stiff");
```

Of course, this requires insight into the problem structure and is not always
possible for complex multi-body systems.

## Practical Example: Stabilized Pendulum DAE

Here is a complete example of the pendulum using Baumgarte stabilization to reduce
the index-3 DAE to index-1, solvable by Numra:

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust
use numra::ode::{DaeProblem, Radau5, Solver, SolverOptions};

let grav = 9.81;
let l = 1.0;
let alpha = 10.0;  // Baumgarte stabilization parameter
let beta = 10.0;

let theta0: f64 = std::f64::consts::PI / 6.0;
let x0 = l * theta0.sin();
let y0 = -l * theta0.cos();

// Compute consistent initial lambda:
// From acceleration constraint at rest (vx=vy=0):
// 0 = 0 + 0 - lambda*(x0^2 + y0^2) - grav*y0 + 0 + beta^2*(0)
// => lambda = -grav*y0 / (x0^2 + y0^2) = -grav*y0 / L^2
let lam0 = -grav * y0 / (l * l);

let dae = DaeProblem::new(
    move |_t, y: &[f64], dydt: &mut [f64]| {
        let x = y[0];
        let yp = y[1];  // y-position
        let vx = y[2];
        let vy = y[3];
        let lam = y[4];

        dydt[0] = vx;
        dydt[1] = vy;
        dydt[2] = -lam * x;
        dydt[3] = -lam * yp - grav;

        // Baumgarte-stabilized acceleration-level constraint (index-1):
        let g_pos = x * x + yp * yp - l * l;         // position constraint
        let g_vel = 2.0 * (x * vx + yp * vy);        // velocity constraint
        let g_acc = 2.0 * (vx * vx + vy * vy)
                  - 2.0 * lam * (x * x + yp * yp)
                  - 2.0 * grav * yp;
        dydt[4] = g_acc + 2.0 * alpha * g_vel + beta * beta * g_pos;
    },
    |mass: &mut [f64]| {
        for i in 0..25 { mass[i] = 0.0; }
        mass[0]  = 1.0;
        mass[6]  = 1.0;
        mass[12] = 1.0;
        mass[18] = 1.0;
        // mass[24] = 0 (algebraic)
    },
    0.0, 10.0,
    vec![x0, y0, 0.0, 0.0, lam0],
    vec![4],  // lambda is algebraic
);

let options = SolverOptions::default()
    .rtol(1e-8)
    .atol(1e-10);

let result = Radau5::solve(
    &dae, 0.0, 10.0,
    &[x0, y0, 0.0, 0.0, lam0],
    &options,
).expect("Stabilized pendulum DAE should solve");

// Check that the length constraint is approximately maintained
let yf = result.y_final().unwrap();
let length_err = (yf[0] * yf[0] + yf[1] * yf[1]).sqrt() - l;
assert!(length_err.abs() < 1e-4,
    "Pendulum length drifted by {}", length_err);
```

## Which Solvers Handle Which Indices?

| DAE index | Numra solvers | Notes |
|-----------|--------------|-------|
| Index-0 (ODE) | All solvers | No special handling needed |
| Index-1 | Radau5, BDF | Native support via mass matrix |
| Index-2 | None directly | Apply index reduction first |
| Index-3+ | None directly | Apply index reduction first |

### Why Only Index-1?

Radau5 and BDF solve systems of the form $ M \cdot y' = f(t, y) $. The Newton
iteration for the implicit stages requires solving:

$$
(M - h\gamma J) \cdot \Delta y = \text{residual}
$$

For index-1 DAEs, the matrix $ (M - h\gamma J) $ is nonsingular (even though $ M $
itself is singular), because the Jacobian $ J $ has the right structure to "fill in"
the zero rows of $ M $. For index-2 and higher, this matrix can become singular or
very ill-conditioned, causing the Newton iteration to fail.

Additionally, Radau5's stiffly-accurate property (the last stage equals the step
endpoint) ensures that the algebraic constraints are satisfied at each time step
for index-1 problems. This property does not extend to higher-index DAEs.

## Index Reduction Checklist

When you encounter a higher-index DAE:

1. **Identify the index.** Count how many times you must differentiate the
   algebraic constraints to express all variables as ODEs.

2. **Try reformulation first.** Can you use generalized coordinates that eliminate
   constraints entirely? (e.g., angles instead of Cartesian coordinates)

3. **If reformulation is not possible, use Baumgarte stabilization:**
   - Differentiate constraints to the acceleration level.
   - Add $ 2\alpha\,\dot{g} + \beta^2\,g $ stabilization terms.
   - Choose $ \alpha, \beta $ based on the system timescale.

4. **Verify consistent initial conditions.** After index reduction, all algebraic
   constraints (including the differentiated ones) must be satisfied at $ t_0 $.

5. **Monitor constraint drift.** Even with stabilization, check the original
   constraint at the end of integration.

## Summary

- The **DAE index** counts differentiations needed to reach ODE form.
- Numra supports **index-1** DAEs natively with Radau5 and BDF.
- Higher-index DAEs require **index reduction** before solving.
- **Baumgarte stabilization** is the most practical technique: differentiate
  constraints and add damping to prevent drift.
- When possible, **reformulate** using minimal coordinates to avoid DAEs entirely.
- Always provide **consistent initial conditions** for the reduced system.

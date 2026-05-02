---
title: "Collocation"
---

Direct collocation transcribes an optimal control problem into a nonlinear
program (NLP) by discretizing both states and controls at mesh nodes and
enforcing dynamics through algebraic constraints.

## The Idea

Instead of integrating the ODE forward (as in shooting), collocation
approximates the state trajectory as a polynomial and requires that the
polynomial satisfies the dynamics at selected collocation points.

This converts the infinite-dimensional control problem into a finite-dimensional
NLP that standard optimizers can solve.

## Collocation Schemes

Numra supports two collocation schemes:

### Trapezoidal (2nd order)

The simplest scheme. The defect constraint at each interval is:

$$
x_{k+1} - x_k - \frac{h}{2}\left(f_k + f_{k+1}\right) = 0
$$

### Hermite-Simpson (3rd order)

The industry standard for direct collocation. Uses midpoint values and cubic
Hermite interpolation:

$$
x_{k+1} - x_k - \frac{h}{6}\left(f_k + 4f_{k+1/2} + f_{k+1}\right) = 0
$$

where $ f_{k+1/2} $ is evaluated at the midpoint state, which itself
satisfies a consistency condition.

## Basic Usage

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust
use numra::ocp::{CollocationProblem, CollocationScheme, CollocationResult};

let result = CollocationProblem::<f64>::new(2, 1)  // 2 states, 1 control
    .dynamics(|t, x, u, p, dxdt| {
        dxdt[0] = x[1];
        dxdt[1] = u[0];  // direct acceleration control
    })
    .initial_state(&[0.0, 0.0])
    .time_span(0.0, 2.0)
    .n_mesh(30)  // 30 mesh nodes
    .scheme(CollocationScheme::HermiteSimpson)
    .terminal_cost(|x, _tf| {
        let dx = x[0] - 1.0;
        let dv = x[1];
        100.0 * (dx * dx + dv * dv)
    })
    .running_cost(|_t, _x, u| u[0] * u[0])
    .state_bounds(0, (-2.0, 2.0))  // position bounds
    .control_bounds(0, (-10.0, 10.0))  // force bounds
    .solve()
    .unwrap();

println!("Objective: {:.6}", result.objective);
println!("Converged: {}", result.converged);
println!("Mesh nodes: {}", result.time.len());
```

## Collocation vs Shooting

| Aspect | Collocation | Shooting |
|--------|-------------|----------|
| **State trajectory** | Decision variable | Computed by ODE |
| **Dynamics** | Algebraic constraints | Forward integration |
| **NLP size** | $ (n_x + n_u) \times N $ | $ n_u \times N $ |
| **Path constraints** | Easy to add | Hard (require ODE checkpoints) |
| **Sparsity** | Banded structure | Dense gradients |
| **Accuracy** | Order 2 or 3 | Depends on ODE solver |
| **Robustness** | Better for complex constraints | Better for simple problems |

## When to Use Collocation

- Problems with **path constraints** (state or control limits along the trajectory)
- Problems with **free final time**
- Problems where shooting fails due to sensitivity
- Robotics, aerospace, and process control applications

## Scheme Comparison

| Scheme | Order | Defect evals | Interior points | Accuracy |
|--------|-------|-------------|-----------------|----------|
| Trapezoidal | 2 | 2 per interval | None | Low |
| Hermite-Simpson | 3 | 3 per interval | 1 midpoint | Medium |

Hermite-Simpson is recommended for most problems. Use trapezoidal only for
quick prototyping or when the extra accuracy is not needed.

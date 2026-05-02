---
title: "Shooting Methods"
---

Shooting methods convert an optimal control problem into a finite-dimensional
optimization problem by parameterizing the control and using ODE integration to
compute the resulting trajectory.

## The Optimal Control Problem

Given a controlled ODE:

$$
\frac{dy}{dt} = f(t, y, u), \quad y(t_0) = y_0
$$

find the control $ u(t) $ that minimizes:

$$
J = \phi(y(T)) + \int_{t_0}^{T} L(t, y, u) \, dt
$$

where $ \phi $ is the terminal cost and $ L $ is the running cost.

## Single Shooting

Single shooting discretizes the control into $ N $ piecewise-constant
segments and treats the segment values as decision variables:

$$
u(t) = u_k \quad \text{for} \quad t \in [t_k, t_{k+1})
$$

The trajectory is computed by forward integration, and the objective is
minimized by a nonlinear optimizer.

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust
use numra::ocp::{ShootingProblem, ShootingResult};

// Minimize terminal distance for a controlled oscillator
// dy1/dt = y2, dy2/dt = u (control is acceleration)
let result = ShootingProblem::<f64>::new(2, 1)  // 2 states, 1 control
    .dynamics(|t, y, dydt, u| {
        dydt[0] = y[1];
        dydt[1] = u[0];
    })
    .initial_state(&[0.0, 0.0])
    .time_span(0.0, 2.0)
    .n_segments(20)
    .terminal_cost(|y_tf| {
        // Minimize distance to target (1.0, 0.0)
        let dx = y_tf[0] - 1.0;
        let dv = y_tf[1];
        dx * dx + dv * dv
    })
    .running_cost(|_t, _y, u| {
        0.01 * u[0] * u[0]  // penalize control effort
    })
    .control_bounds(0, (-5.0, 5.0))  // bound control magnitude
    .solve()
    .unwrap();

println!("Objective: {:.6e}", result.objective);
println!("Converged: {}", result.converged);
println!("Iterations: {}", result.iterations);
println!("Final state: [{:.4}, {:.4}]", result.final_state[0], result.final_state[1]);
```

### How It Works

1. **Parameterize**: Represent $ u(t) $ as $ N $ scalar values
2. **Forward integrate**: Given a control vector, integrate the ODE from $ t_0 $ to $ T $
3. **Compute cost**: Sum the terminal cost and discretized running cost
4. **Optimize**: Use BFGS or L-BFGS to minimize the cost over control variables
5. **Iterate**: Repeat until convergence

### ShootingResult

The result contains the full solution:

| Field | Type | Description |
|-------|------|-------------|
| `controls` | `Vec<S>` | Optimal control values (flat) |
| `final_state` | `Vec<S>` | Terminal state $ y(T) $ |
| `objective` | `S` | Optimal cost |
| `converged` | `bool` | Optimizer convergence |
| `iterations` | `usize` | Number of optimizer iterations |
| `t_trajectory` | `Vec<S>` | Time grid |
| `y_trajectory` | `Vec<S>` | State trajectory (row-major) |

## Multiple Shooting

Multiple shooting breaks the time horizon into intervals and treats both
controls *and* intermediate states as decision variables. This improves
conditioning for problems where single shooting suffers from sensitivity to
control perturbations.

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust
use numra::ocp::{MultipleShootingProblem, MultipleShootingResult};

let result = MultipleShootingProblem::<f64>::new(2, 1)
    .dynamics(|t, y, dydt, u| {
        dydt[0] = y[1];
        dydt[1] = u[0] - y[0];
    })
    .initial_state(&[1.0, 0.0])
    .time_span(0.0, 5.0)
    .n_segments(10)
    .terminal_cost(|y| y[0] * y[0] + y[1] * y[1])
    .solve()
    .unwrap();
```

### Single vs Multiple Shooting

| Aspect | Single Shooting | Multiple Shooting |
|--------|----------------|-------------------|
| **Decision variables** | Controls only | Controls + states |
| **NLP size** | $ n_u \times N $ | $ (n_u + n_x) \times N $ |
| **Conditioning** | Poor for long horizons | Better conditioned |
| **Constraints** | Implicit (via ODE) | Explicit continuity |
| **Implementation** | Simpler | More complex |
| **Parallelism** | Sequential | Intervals independent |

## Tips

- **Start simple**: Use single shooting first. Switch to multiple shooting if
  convergence is poor or the time horizon is long.
- **Control bounds**: Always provide reasonable bounds. Unbounded controls can
  cause the optimizer to try extreme values that break the ODE integrator.
- **Number of segments**: More segments = finer control resolution but larger
  NLP. Start with 10-20 and refine.
- **ODE tolerances**: The shooting method is only as accurate as its ODE
  integration. Use tight tolerances (`rtol=1e-8`) for the inner integration.

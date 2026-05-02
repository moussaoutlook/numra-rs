---
title: "Adjoint Methods"
---

The adjoint method computes the gradient of an objective function with respect
to parameters in $ O(1) $ backward integrations, regardless of the number of
parameters. This makes it dramatically more efficient than forward sensitivity
when there are many parameters (as in machine learning or large-scale
parameter estimation).

## The Problem

Given a parameterized ODE:

$$
\frac{dx}{dt} = f(t, x, p), \quad x(t_0) = x_0
$$

and an objective:

$$
J = \phi(x(t_f)) + \int_{t_0}^{t_f} L(t, x, p) \, dt
$$

compute the gradient $ dJ/dp $.

## Forward vs Adjoint

**Forward sensitivity** propagates derivatives forward alongside the state:

$$
\frac{d}{dt}\frac{\partial x}{\partial p} = \frac{\partial f}{\partial x} \frac{\partial x}{\partial p} + \frac{\partial f}{\partial p}
$$

This requires solving $ n_x \times n_p $ additional equations -- expensive
when $ n_p $ is large.

**Adjoint method** introduces the costate $ \lambda(t) $ and integrates
backward:

$$
\frac{d\lambda}{dt} = -\left(\frac{\partial f}{\partial x}\right)^T \lambda - \left(\frac{\partial L}{\partial x}\right)^T
$$

with terminal condition $ \lambda(t_f) = \nabla_x \phi(x(t_f)) $.

The gradient is then:

$$
\frac{dJ}{dp} = \nabla_p \phi + \int_{t_0}^{t_f} \left[\left(\frac{\partial f}{\partial p}\right)^T \lambda + \frac{\partial L}{\partial p}\right] dt
$$

This requires only **one** backward integration regardless of $ n_p $.

## Usage

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust
use numra::ocp::adjoint_gradient;

// Model: dx/dt = -p[0]*x + p[1]*sin(t)
let gradient_result = adjoint_gradient(
    // model: f(t, x, dxdt, params)
    |t, x, dxdt, p| {
        dxdt[0] = -p[0] * x[0] + p[1] * t.sin();
    },
    1,     // n_states
    2,     // n_params
    0.0,   // t0
    10.0,  // tf
    &[1.0],         // x0
    &[0.5, 1.0],    // params
    // terminal cost: phi(x(tf))
    |x_tf| x_tf[0] * x_tf[0],
    // running cost (optional): L(t, x, p)
    Some(|_t: f64, _x: &[f64], _p: &[f64]| 0.0),
).unwrap();

println!("Objective: {:.6}", gradient_result.objective);
println!("dJ/dp = {:?}", gradient_result.gradient);
```

## Cost Comparison

| Method | Cost | Best when |
|--------|------|-----------|
| Forward sensitivity | $ O(n_x \times n_p) $ | $ n_p \leq n_x $ |
| Adjoint | $ O(n_x) $ backward | $ n_p \gg n_x $ |
| Finite differences | $ O(n_p) $ forward solves | Quick & dirty |

For a 3-state system with 100 parameters:
- Forward: 300 additional ODEs
- Adjoint: 3 backward ODEs + gradient quadrature
- Finite differences: 100 full ODE solves

## AdjointResult

| Field | Description |
|-------|-------------|
| `gradient` | $ dJ/dp $ vector |
| `objective` | Scalar objective $ J $ |
| `costate` | Costate trajectory $ \lambda(t) $ |
| `costate_time` | Time points for costate |

## Practical Notes

- Jacobians $ \partial f/\partial x $ and $ \partial f/\partial p $ are
  computed via finite differences internally.
- The forward state trajectory must be stored or recomputed during the backward
  pass. Numra stores the trajectory from the forward integration.
- For problems with discontinuous dynamics, the adjoint equations need jump
  conditions at discontinuity points.

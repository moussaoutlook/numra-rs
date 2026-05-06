---
title: "Parameter Importance"
---

When you have a model $ f(p_1, p_2, \ldots, p_n) $ that takes a parameter
vector and returns a scalar, "parameter importance" is the question
*which $ p_i $ matters most for the output*. Numra answers this with
[`compute_sensitivities`](https://docs.rs/numra/latest/numra/fn.compute_sensitivities.html),
a one-shot local sensitivity routine built on central finite differences.

:::tip[Two distinct sensitivity concepts]
This page covers **scalar-function parameter importance** — local
sensitivity coefficients of an arbitrary function $ f: \mathbb{R}^n \to \mathbb{R} $
with respect to its parameters. It returns a single number per
parameter at a single point.

For the *trajectory* sensitivity of an ODE/DAE system —
$ S(t) = \partial y(t)/\partial p $ — see the dedicated
[Sensitivity Analysis](./sensitivity-analysis) page. The two are
related but solve different problems.
:::

## Computing sensitivities

The basic form takes a closure, a nominal parameter vector, optional
parameter names, and an optional finite-difference step size:

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust
use numra::compute_sensitivities;

// A model: output = a^2 · sin(b) + c
let f = |p: &[f64]| p[0] * p[0] * p[1].sin() + p[2];
let params = [3.0, 1.5, 10.0];
let names = ["a", "b", "c"];

let result = compute_sensitivities(f, &params, &names, None);

println!("Output: {:.4}", result.output);
for s in &result.sensitivities {
    println!(
        "{}: coefficient = {:.4}, normalized = {:.4}",
        s.name, s.coefficient, s.normalized,
    );
}
```

Each parameter contributes one row to `result.sensitivities`.

## Sensitivity metrics

Each parameter gets two metrics:

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **Coefficient** | $ \partial f / \partial p_i $ | Absolute rate of change — has units. |
| **Normalized** | $ (p_i / f) \cdot \partial f / \partial p_i $ | Dimensionless elasticity. A value of $ 0.5 $ means a 1% change in $ p_i $ produces approximately a 0.5% change in the output. |

Use the **coefficient** when you care about the physical scale of
sensitivity (e.g. "how many degrees Celsius does the output shift per
unit change in this rate constant?"). Use the **normalized** sensitivity
when you want to rank parameters whose values span very different
magnitudes — say, a rate constant of $ 10^{-3} $ alongside a temperature
of $ 300 $.

## Finding the most important parameter

A common follow-up is "which is the most influential parameter?":

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust
let most = result.most_sensitive().unwrap();
println!(
    "Most influential: {} (normalized = {:.4})",
    most.name, most.normalized,
);
```

`most_sensitive` ranks by the absolute value of `normalized`. For
problems where parameters share a common physical unit, you may prefer
to rank by `coefficient.abs()` instead — iterate manually and sort.

## Uncertainty propagation from sensitivities

If you also know the variances of the input parameters, the local
linearisation lets you estimate the output variance directly:

$$
\sigma_f^2 \;\approx\; \sum_i \left(\frac{\partial f}{\partial p_i}\right)^{\!2} \sigma_{p_i}^2
$$

assuming the parameters are independent and the function is well
approximated as linear in the parameter range of interest. Numra
exposes this as a one-line helper:

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust
let param_variances = [0.1, 0.05, 1.0]; // Var(a), Var(b), Var(c)
let output_variance = result.propagate_uncertainty(&param_variances);
println!("Predicted output std: {:.4}", output_variance.sqrt());
```

For correlated parameters, larger uncertainties, or strongly nonlinear
$ f $, this first-order estimate can be very wrong — fall back to
[Monte Carlo](./monte-carlo-odes) or full uncertainty propagation via
[`Uncertain<S>`](./error-propagation).

## Finite-difference step size

The default step size is $ h = 10^{-7} \cdot (1 + |p_i|) $, a relative
step that adapts to each parameter's magnitude. Override the step
size when needed:

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust
let result = compute_sensitivities(f, &params, &names, Some(1e-5));
```

| Direction | Effect |
|-----------|--------|
| **Smaller $ h $** | Lower truncation error; more susceptible to floating-point cancellation. |
| **Larger $ h $** | More cancellation-resistant; truncation error grows quadratically (central FD). |

The default $ 10^{-7} \cdot (1 + \lvert p \rvert) $ is a good compromise
for most well-conditioned problems and gives roughly $ 10 $ correct
digits when the function is smooth and not pathologically scaled.

## When to use this vs forward sensitivity

`compute_sensitivities` is the right tool when:

- The model is a **scalar function of parameters**, not a trajectory.
- You want **a quick local picture** at one nominal parameter point.
- The function is reasonably smooth and you can afford $ 2n + 1 $
  evaluations (central FD plus baseline).

If your model is an ODE/DAE and you want $ \partial y(t) / \partial p $
at every output time *along the trajectory*, you want the forward
sensitivity primitive on the [Sensitivity Analysis](./sensitivity-analysis)
page instead. The augmented-system approach there integrates state
and sensitivity together — one solver call, single source of truth on
tolerances and step control, and analytical Jacobians available
where they matter.

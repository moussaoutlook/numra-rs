# Sensitivity Analysis

Sensitivity analysis answers: "Which parameters matter most?" Given a function
\\( f(p_1, p_2, \ldots, p_n) \\), it quantifies how much the output changes when
each parameter is perturbed.

## Computing Sensitivities

Numra computes sensitivities via central finite differences:

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
use numra::uncertainty::compute_sensitivities;

// A model: output = a^2 * sin(b) + c
let f = |p: &[f64]| p[0] * p[0] * p[1].sin() + p[2];
let params = [3.0, 1.5, 10.0];
let names = ["a", "b", "c"];

let result = compute_sensitivities(f, &params, &names, None);

println!("Output: {:.4}", result.output);
for s in &result.sensitivities {
    println!("{}: coeff={:.4}, normalized={:.4}", s.name, s.coefficient, s.normalized);
}
```

## Sensitivity Metrics

Each parameter gets two metrics:

| Metric | Formula | Meaning |
|--------|---------|---------|
| **Coefficient** | \\( \partial f / \partial p_i \\) | Absolute rate of change |
| **Normalized** | \\( (p_i / f) \cdot \partial f / \partial p_i \\) | Relative (elasticity) |

The normalized sensitivity tells you: "A 1% change in \\( p_i \\) causes
approximately a (normalized)% change in the output."

## Finding the Most Important Parameter

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
let most = result.most_sensitive().unwrap();
println!("Most sensitive: {} (normalized: {:.4})", most.name, most.normalized);
```

## Uncertainty Propagation from Sensitivities

Combine sensitivities with parameter uncertainties to predict output uncertainty:

\\[
\sigma_f^2 = \sum_i \left(\frac{\partial f}{\partial p_i}\right)^2 \sigma_{p_i}^2
\\]

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
let param_variances = [0.1, 0.05, 1.0];  // Var(a), Var(b), Var(c)
let output_variance = result.propagate_uncertainty(&param_variances);
println!("Output std: {:.4}", output_variance.sqrt());
```

## Sensitivity with ODE Models

For ODE models, combine sensitivity analysis with parameter estimation:

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
use numra::ocp::forward_sensitivity;

// Forward sensitivity: compute dx/dp alongside the state
let sens_result = forward_sensitivity(
    |t, x, dxdt, p| { dxdt[0] = -p[0] * x[0]; },
    1,           // n_states
    1,           // n_params
    0.0, 10.0,   // time span
    &[1.0],      // x0
    &[0.5],      // params
).unwrap();

// sens_result contains dx/dp at each time step
```

For many parameters, use the adjoint method instead (see
[Adjoint Methods](../ch10-optimal-control/adjoint-methods.md)).

## Finite Difference Step Size

The default step size is \\( h = 10^{-7} \times (1 + |p_i|) \\). Override for
specific problems:

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
let result = compute_sensitivities(f, &params, &names, Some(1e-5));
```

Smaller steps give more accurate derivatives but are more susceptible to
floating-point cancellation. The relative step \\( h \cdot (1 + |p|) \\) adapts
to the parameter scale.

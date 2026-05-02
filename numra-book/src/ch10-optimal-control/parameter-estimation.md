# Parameter Estimation

Parameter estimation fits an ODE model to experimental data by finding the
parameter values that minimize the discrepancy between model predictions and
observations.

## The Problem

Given a model:

\\[
\frac{dy}{dt} = f(t, y; p), \quad y(t_0) = y_0
\\]

and observed data \\( (t_i, \hat{y}_i) \\) for \\( i = 1, \ldots, M \\), find:

\\[
p^* = \arg\min_p \sum_{i=1}^{M} \|y(t_i; p) - \hat{y}_i\|^2
\\]

## Basic Usage

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
use numra::ocp::{ParamEstProblem, ParamEstResult};

// Exponential decay: dy/dt = -k*y, true k = 0.5
let t_data = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
let y_data = vec![1.0, 0.607, 0.368, 0.223, 0.135, 0.082];

let result = ParamEstProblem::<f64>::new(1, 1)  // 1 param, 1 state
    .model(|t, y, dydt, p| {
        dydt[0] = -p[0] * y[0];
    })
    .initial_state(&[1.0])
    .initial_params(&[1.0])  // initial guess for k
    .data(&t_data, &y_data)
    .param_bounds(0, (0.01, 10.0))
    .solve()
    .unwrap();

println!("Estimated k = {:.4}", result.params[0]);  // should be ~0.5
println!("Residual: {:.6e}", result.residual_norm);
println!("Converged: {}", result.converged);
println!("ODE integrations: {}", result.n_integrations);
```

## Multi-State Observations

When the ODE has multiple states but you only observe some of them:

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
let result = ParamEstProblem::<f64>::new(2, 3)  // 2 params, 3 states
    .model(|t, y, dydt, p| {
        dydt[0] = -p[0] * y[0] + p[1] * y[1];
        dydt[1] = p[0] * y[0] - p[1] * y[1];
        dydt[2] = p[1] * y[1];  // cumulative output
    })
    .initial_state(&[1.0, 0.0, 0.0])
    .initial_params(&[1.0, 0.5])
    .data(&t_data, &y_observed)
    .observed_indices(&[0, 2])  // only observe states 0 and 2
    .solve()
    .unwrap();
```

## ParamEstResult

| Field | Description |
|-------|-------------|
| `params` | Estimated parameters |
| `residual_norm` | L2 norm of residuals |
| `iterations` | Optimizer iterations |
| `converged` | Convergence flag |
| `predicted` | Model predictions at data times |
| `n_integrations` | Total ODE solves performed |
| `wall_time_secs` | Computation time |

## Tips for Good Results

1. **Initial guess**: Provide a reasonable initial guess. The optimizer may
   converge to a local minimum if started too far from the truth.

2. **Parameter bounds**: Always bound parameters to physically meaningful
   ranges. This prevents the optimizer from exploring regions where the ODE
   becomes stiff or unstable.

3. **Data quality**: More data points and less noise improve identifiability.
   If parameters are structurally unidentifiable (the model cannot distinguish
   between different parameter combinations), no amount of data will help.

4. **Scaling**: If parameters differ by orders of magnitude (e.g., one is 0.001
   and another is 1000), consider scaling them to be O(1) before estimation.

5. **Multi-start**: Run the estimation from several random initial guesses and
   pick the solution with the lowest residual to avoid local minima.

## Comparison with Curve Fitting

| Feature | Parameter Estimation | `curve_fit` |
|---------|---------------------|-------------|
| **Model type** | ODE-based | Algebraic function |
| **Requires** | ODE integration | Direct function call |
| **Cost per call** | Expensive (full ODE solve) | Cheap |
| **Best for** | Dynamic models | Static models |

Use `ParamEstProblem` when your model is defined by a differential equation.
Use `curve_fit` when you have a direct formula \\( y = f(x; p) \\).

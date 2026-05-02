---
title: "Multi-Objective Optimization"
---

Many engineering and scientific problems involve optimizing **multiple conflicting objectives** simultaneously. Instead of a single optimal point, the solution is a **Pareto front** -- the set of all non-dominated trade-off solutions.

$$
\min_{x \in \Omega} \bigl(f_1(x),\; f_2(x),\; \ldots,\; f_k(x)\bigr)
$$

A solution $ x^* $ is **Pareto-optimal** if no other feasible $ x $ improves one objective without worsening another:

$$
\nexists\, x \in \Omega : \forall\, i,\; f_i(x) \le f_i(x^*) \;\text{and}\; \exists\, j,\; f_j(x) < f_j(x^*).
$$

## NSGA-II Algorithm

Numra implements **NSGA-II** (Non-dominated Sorting Genetic Algorithm II), a widely-used evolutionary multi-objective optimizer. The algorithm uses:

1. **Non-dominated sorting:** The combined parent+offspring population is sorted into Pareto fronts $ F_0, F_1, F_2, \ldots $ where $ F_0 $ contains the non-dominated solutions.

2. **Crowding distance:** Within each front, individuals are ranked by crowding distance -- a measure of how isolated they are in objective space. Higher crowding distance means more diversity.

3. **Tournament selection:** Parents for the next generation are chosen by binary tournament comparing (a) front rank (lower is better), then (b) crowding distance (higher is better).

4. **SBX crossover:** Simulated Binary Crossover with distribution index $ \eta_c $ creates offspring near the parents with polynomial probability.

5. **Polynomial mutation:** Each variable is mutated with probability $ 1/n $, using a polynomial distribution controlled by $ \eta_m $.

### Basic Bi-Objective Example

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust
use numra::optim::{nsga2_optimize, NsgaIIOptions};

// f1 = x^2, f2 = (x-2)^2
// Pareto front: x in [0, 2] (any x in this range is Pareto-optimal)
let f1 = |x: &[f64]| x[0] * x[0];
let f2 = |x: &[f64]| (x[0] - 2.0).powi(2);
let objectives: Vec<&dyn Fn(&[f64]) -> f64> = vec![&f1, &f2];

let bounds = vec![(0.0, 4.0)];
let opts = NsgaIIOptions {
    pop_size: 50,
    max_generations: 100,
    ..Default::default()
};

let result = nsga2_optimize(&objectives, &bounds, &opts).unwrap();

// Extract the Pareto front
let pareto = result.pareto.as_ref().unwrap();
println!("Pareto front size: {}", pareto.points.len());

for p in &pareto.points {
    println!("  x={:.3}, f1={:.3}, f2={:.3}",
        p.x[0], p.objectives[0], p.objectives[1]);
}

// All Pareto points should have x in [0, 2]
for p in &pareto.points {
    assert!(p.x[0] >= -0.5 && p.x[0] <= 2.5);
}
```

### ZDT1 Benchmark

The ZDT1 test function is a standard benchmark with a convex Pareto front:

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust
use numra::optim::{nsga2_optimize, NsgaIIOptions};

let n = 3;
let bounds = vec![(0.0, 1.0); n];

let f1 = |x: &[f64]| x[0];
let f2 = |x: &[f64]| {
    let g = 1.0 + 9.0 * x[1..].iter().sum::<f64>() / (x.len() - 1) as f64;
    g * (1.0 - (x[0] / g).sqrt())
};

let objectives: Vec<&dyn Fn(&[f64]) -> f64> = vec![&f1, &f2];
let opts = NsgaIIOptions {
    pop_size: 50,
    max_generations: 100,
    ..Default::default()
};

let result = nsga2_optimize(&objectives, &bounds, &opts).unwrap();
let pareto = result.pareto.as_ref().unwrap();

assert!(pareto.points.len() >= 5);
println!("ZDT1 Pareto front: {} points", pareto.points.len());
```

### Three-Objective Example

NSGA-II handles any number of objectives:

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust
use numra::optim::{nsga2_optimize, NsgaIIOptions};

let bounds = vec![(0.0, 2.0), (0.0, 2.0)];
let f1 = |x: &[f64]| x[0] * x[0];
let f2 = |x: &[f64]| x[1] * x[1];
let f3 = |x: &[f64]| (x[0] - 1.0).powi(2) + (x[1] - 1.0).powi(2);

let objectives: Vec<&dyn Fn(&[f64]) -> f64> = vec![&f1, &f2, &f3];
let opts = NsgaIIOptions {
    pop_size: 50,
    max_generations: 100,
    ..Default::default()
};

let result = nsga2_optimize(&objectives, &bounds, &opts).unwrap();
let pareto = result.pareto.as_ref().unwrap();

for p in &pareto.points {
    assert_eq!(p.objectives.len(), 3);
}
println!("3-objective Pareto front: {} points", pareto.points.len());
```

## NSGA-II Options

| Option | Default | Description |
|--------|---------|-------------|
| `pop_size` | 100 | Population size (must be even) |
| `max_generations` | 200 | Maximum number of generations |
| `crossover_eta` | 20.0 | SBX crossover distribution index |
| `mutation_eta` | 20.0 | Polynomial mutation distribution index |
| `crossover_prob` | 0.9 | Crossover probability |
| `mutation_prob` | None (= 1/n) | Per-variable mutation probability |
| `seed` | 42 | Random seed for reproducibility |

**Tuning guidelines:**
- **Population size:** Use at least 50 for bi-objective, 100+ for three objectives.
- **Crossover eta:** Higher values (20--30) create offspring closer to parents. Lower values (2--5) allow more exploration.
- **Mutation eta:** Same interpretation as crossover eta.
- **Generations:** More generations improve Pareto front quality but increase cost.

## Result Structure

The `OptimResult` from NSGA-II contains:

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust
// result.x:      best point for the first objective
// result.f:      best first-objective value
// result.pareto: Some(ParetoResult { points: Vec<ParetoPoint> })

let pareto = result.pareto.unwrap();
for point in &pareto.points {
    let x = &point.x;            // decision variables
    let obj = &point.objectives; // objective values [f1, f2, ...]
}
```

Each `ParetoPoint` contains:
- `x: Vec<S>` -- the decision variable values.
- `objectives: Vec<S>` -- the objective function values.

The returned Pareto front is guaranteed to be **non-dominated**: no point in the set dominates any other.

## Declarative Builder API

The `OptimProblem` builder supports multi-objective optimization:

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust
use numra::optim::OptimProblem;

let result = OptimProblem::new(1)
    .multi_objective(vec![
        Box::new(|x: &[f64]| x[0] * x[0]),
        Box::new(|x: &[f64]| (x[0] - 2.0).powi(2)),
    ])
    .all_bounds(&[(0.0, 4.0)])
    .solve()
    .unwrap();

let pareto = result.pareto.as_ref().unwrap();
println!("Pareto front: {} points", pareto.points.len());
```

## Practical Considerations

1. **Scaling objectives:** If objectives have very different magnitudes, NSGA-II's crowding distance may favor one over another. Consider normalizing objectives to similar ranges.

2. **Constraint handling:** NSGA-II does not directly support constraints. For constrained multi-objective problems, use penalty terms in the objectives.

3. **Post-processing:** The Pareto front contains all trade-off solutions. Use domain knowledge to select a preferred operating point. Common selection criteria include:
   - Minimum distance to a utopia point.
   - Knee point detection (maximum curvature).
   - Weighted sum preference.

4. **Deterministic runs:** NSGA-II is deterministic given the same seed, population size, and bounds, enabling reproducible experiments.

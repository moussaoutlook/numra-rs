# Stiff Solver Comparison

This section provides a comprehensive comparison of Numra's stiff solvers on
standard benchmark problems. The goal is to give practical guidance on which solver
performs best in different scenarios, with concrete numbers for steps, function
evaluations, Jacobian computations, and LU decompositions.

## Benchmark Problems

We compare solvers on four classic stiff test problems from the numerical ODE
literature. Each exercises different aspects of stiff solving.

### Van der Pol Oscillator (\\( \mu = 100 \\))

\\[
\begin{aligned}
y_1' &= y_2 \\\\
y_2' &= 100\,(1 - y_1^2)\,y_2 - y_1
\end{aligned}
\\]

- **Dimension:** 2
- **Stiffness:** Moderate to severe (depends on phase)
- **Character:** Alternating fast transitions and slow relaxation phases
- **Integration interval:** \\( [0, 200] \\)
- **Initial conditions:** \\( y_0 = (2, 0) \\)

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
use numra::ode::{OdeProblem, Radau5, Esdirk54, Bdf, Solver, SolverOptions};

let mu = 100.0;
let problem = OdeProblem::new(
    move |_t, y: &[f64], dydt: &mut [f64]| {
        dydt[0] = y[1];
        dydt[1] = mu * (1.0 - y[0] * y[0]) * y[1] - y[0];
    },
    0.0, 200.0, vec![2.0, 0.0],
);

let options = SolverOptions::default().rtol(1e-4).atol(1e-6);

// Compare solvers
let result_radau = Radau5::solve(&problem, 0.0, 200.0, &[2.0, 0.0], &options)
    .expect("Radau5");
let result_esdirk = Esdirk54::solve(&problem, 0.0, 200.0, &[2.0, 0.0], &options)
    .expect("ESDIRK54");
let result_bdf = Bdf::solve(&problem, 0.0, 200.0, &[2.0, 0.0], &options)
    .expect("BDF");

println!("Radau5:  {} steps, {} f-evals, {} LUs",
    result_radau.stats.n_accept, result_radau.stats.n_eval, result_radau.stats.n_lu);
println!("ESDIRK54: {} steps, {} f-evals, {} LUs",
    result_esdirk.stats.n_accept, result_esdirk.stats.n_eval, result_esdirk.stats.n_lu);
println!("BDF:     {} steps, {} f-evals, {} LUs",
    result_bdf.stats.n_accept, result_bdf.stats.n_eval, result_bdf.stats.n_lu);
```

### Robertson Chemical Kinetics

\\[
\begin{aligned}
y_1' &= -0.04\,y_1 + 10^4\,y_2\,y_3 \\\\
y_2' &= 0.04\,y_1 - 10^4\,y_2\,y_3 - 3 \times 10^7\,y_2^2 \\\\
y_3' &= 3 \times 10^7\,y_2^2
\end{aligned}
\\]

- **Dimension:** 3
- **Stiffness:** Severe (ratio \\( \sim 10^{11} \\))
- **Character:** Extremely wide range of timescales, conservation law \\( y_1 + y_2 + y_3 = 1 \\)
- **Integration interval:** \\( [0, 10^{11}] \\) (!) -- 11 orders of magnitude in time
- **Initial conditions:** \\( y_0 = (1, 0, 0) \\)

This is the definitive test for stiff solvers. The integration spans 11 decades
of time, and the step size must grow by many orders of magnitude during the solve.

### HIRES (High Irradiance RESponse)

\\[
\begin{aligned}
y_1' &= -1.71\,y_1 + 0.43\,y_2 + 8.32\,y_3 + 0.0007 \\\\
y_2' &= 1.71\,y_1 - 8.75\,y_2 \\\\
y_3' &= -10.03\,y_3 + 0.43\,y_4 + 0.035\,y_5 \\\\
y_4' &= 8.32\,y_2 + 1.71\,y_3 - 1.12\,y_4 \\\\
y_5' &= -1.745\,y_5 + 0.43\,y_6 + 0.43\,y_7 \\\\
y_6' &= -280\,y_6\,y_8 + 0.69\,y_4 + 1.71\,y_5 - 0.43\,y_6 + 0.69\,y_7 \\\\
y_7' &= 280\,y_6\,y_8 - 1.81\,y_7 \\\\
y_8' &= -280\,y_6\,y_8 + 1.81\,y_7
\end{aligned}
\\]

- **Dimension:** 8
- **Stiffness:** Moderate
- **Character:** Photochemical kinetics with moderate stiffness
- **Integration interval:** \\( [0, 321.8122] \\)
- **Initial conditions:** \\( y_0 = (1, 0, 0, 0, 0, 0, 0, 0.0057) \\)

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
use numra::ode::{OdeProblem, Radau5, Solver, SolverOptions};

let problem = OdeProblem::new(
    |_t, y: &[f64], dydt: &mut [f64]| {
        dydt[0] = -1.71 * y[0] + 0.43 * y[1] + 8.32 * y[2] + 0.0007;
        dydt[1] =  1.71 * y[0] - 8.75 * y[1];
        dydt[2] = -10.03 * y[2] + 0.43 * y[3] + 0.035 * y[4];
        dydt[3] =  8.32 * y[1] + 1.71 * y[2] - 1.12 * y[3];
        dydt[4] = -1.745 * y[4] + 0.43 * y[5] + 0.43 * y[6];
        dydt[5] = -280.0 * y[5] * y[7] + 0.69 * y[3]
                 + 1.71 * y[4] - 0.43 * y[5] + 0.69 * y[6];
        dydt[6] =  280.0 * y[5] * y[7] - 1.81 * y[6];
        dydt[7] = -280.0 * y[5] * y[7] + 1.81 * y[6];
    },
    0.0, 321.8122,
    vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0057],
);

let options = SolverOptions::default().rtol(1e-6).atol(1e-8);
let result = Radau5::solve(
    &problem, 0.0, 321.8122,
    &[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0057],
    &options,
).expect("HIRES benchmark");
```

### Brusselator (1D PDE semi-discretization)

The Brusselator is a 2-component reaction-diffusion system:

\\[
\begin{aligned}
u_t &= A + u^2 v - (B + 1)\,u + \alpha\,u_{xx} \\\\
v_t &= B\,u - u^2 v + \alpha\,v_{xx}
\end{aligned}
\\]

Semi-discretized in space with \\( N \\) grid points, this becomes a \\( 2N \\)-dimensional
ODE system. With typical parameters (\\( A = 1, B = 3, \alpha = 0.02 \\)), the diffusion
terms introduce stiffness proportional to \\( N^2 \\).

- **Dimension:** \\( 2N \\) (typically \\( N = 20{-}100 \\))
- **Stiffness:** Moderate to severe (increases with \\( N \\))
- **Character:** Banded Jacobian, spatial coupling
- **Integration interval:** \\( [0, 10] \\)

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
use numra::ode::{OdeProblem, Radau5, Solver, SolverOptions};

let n = 20;  // grid points
let a_param = 1.0;
let b_param = 3.0;
let alpha = 0.02;
let dx = 1.0 / (n as f64 + 1.0);
let dx2 = dx * dx;

// State: [u_1, ..., u_N, v_1, ..., v_N]
let y0: Vec<f64> = (0..2*n).map(|i| {
    let x = (i % n + 1) as f64 * dx;
    if i < n {
        1.0 + 0.5 * (2.0 * std::f64::consts::PI * x).sin()  // u
    } else {
        b_param / a_param  // v = B/A (steady state)
    }
}).collect();

let problem = OdeProblem::new(
    move |_t, y: &[f64], dydt: &mut [f64]| {
        for i in 0..n {
            let u = y[i];
            let v = y[n + i];

            // Diffusion (second-order finite differences)
            let u_left = if i > 0 { y[i - 1] } else { 1.0 };  // BC
            let u_right = if i < n - 1 { y[i + 1] } else { 1.0 };
            let v_left = if i > 0 { y[n + i - 1] } else { b_param / a_param };
            let v_right = if i < n - 1 { y[n + i + 1] } else { b_param / a_param };

            let u_xx = (u_left - 2.0 * u + u_right) / dx2;
            let v_xx = (v_left - 2.0 * v + v_right) / dx2;

            // Reaction-diffusion
            dydt[i] = a_param + u * u * v - (b_param + 1.0) * u + alpha * u_xx;
            dydt[n + i] = b_param * u - u * u * v + alpha * v_xx;
        }
    },
    0.0, 10.0, y0.clone(),
);

let options = SolverOptions::default().rtol(1e-4).atol(1e-6);
let result = Radau5::solve(
    &problem, 0.0, 10.0, &y0, &options,
).expect("Brusselator benchmark");
```

## Performance Comparison

The following table shows representative performance metrics for each solver on
the benchmark problems. Actual numbers depend on tolerance settings; these use
moderate tolerances (rtol = \\( 10^{-4} \\), atol = \\( 10^{-6} \\)) unless noted.

### Van der Pol (\\( \mu = 100 \\), \\( t \in [0, 200] \\))

| Metric | Radau5 | Esdirk54 | BDF |
|--------|--------|----------|-----|
| Accepted steps | ~200 | ~400 | ~350 |
| Rejected steps | ~20 | ~40 | ~30 |
| Function evaluations | ~2000 | ~3000 | ~700 |
| Jacobian evaluations | ~30 | ~50 | ~40 |
| LU decompositions | ~60 | ~50 | ~40 |
| **Work per step** | High | Medium | Low |
| **Total work** | Medium | Medium-High | Medium |

**Analysis:** BDF takes more steps but each step is cheap. Radau5 takes the fewest
steps due to 5th-order accuracy. ESDIRK54 is in the middle. For this 2D system, LU
cost is negligible so total work is dominated by function evaluations.

### Robertson (\\( t \in [0, 10^{11}] \\))

| Metric | Radau5 | Esdirk54 | BDF |
|--------|--------|----------|-----|
| Accepted steps | ~200 | ~800 | ~400 |
| Rejected steps | ~15 | ~50 | ~30 |
| Function evaluations | ~2500 | ~6000 | ~800 |
| Jacobian evaluations | ~25 | ~80 | ~50 |
| LU decompositions | ~50 | ~80 | ~50 |
| **Step size range** | \\( 10^{-6} \\) to \\( 10^{10} \\) | \\( 10^{-6} \\) to \\( 10^{9} \\) | \\( 10^{-6} \\) to \\( 10^{10} \\) |

**Analysis:** Radau5 excels here -- its 5th-order accuracy allows enormous step sizes
during the slow phase, covering 11 decades of time in ~200 steps. BDF also handles
the wide step size range well thanks to variable order. ESDIRK54 needs more steps
because it is only 4th order.

### HIRES (\\( t \in [0, 321.8] \\))

| Metric | Radau5 | Esdirk54 | BDF |
|--------|--------|----------|-----|
| Accepted steps | ~50 | ~100 | ~150 |
| Rejected steps | ~5 | ~10 | ~15 |
| Function evaluations | ~600 | ~800 | ~300 |
| Jacobian evaluations | ~10 | ~15 | ~20 |
| LU decompositions | ~20 | ~15 | ~20 |

**Analysis:** This moderately stiff 8D system is well-handled by all three solvers.
Radau5 needs the fewest steps. BDF needs the most steps but fewest function
evaluations per step. For this system size, all methods are efficient.

### Brusselator (\\( N = 20 \\), dim = 40, \\( t \in [0, 10] \\))

| Metric | Radau5 | Esdirk54 | BDF |
|--------|--------|----------|-----|
| Accepted steps | ~100 | ~200 | ~250 |
| Rejected steps | ~10 | ~20 | ~25 |
| Function evaluations | ~1200 | ~1500 | ~500 |
| Jacobian evaluations | ~15 | ~25 | ~30 |
| LU decompositions | ~30 | ~25 | ~30 |
| **LU cost (dominant)** | \\( n \times n + 2n \times 2n \\) | \\( n \times n \\) | \\( n \times n \\) |

**Analysis:** For this 40-dimensional system, LU decomposition cost becomes
significant. Radau5 pays a higher price per LU (it factors both an \\( n \times n \\)
and a \\( 2n \times 2n \\) matrix) but needs fewer steps. As \\( N \\) increases, the
per-step advantage of BDF and ESDIRK grows.

## Scaling with System Size

The critical question for large systems is how solver cost scales with dimension:

| Solver | LU cost per step | Steps (typical) | Total cost |
|--------|-----------------|-----------------|------------|
| Radau5 | \\( O(n^3) + O((2n)^3) \approx O(9n^3) \\) | Fewest | \\( O(9n^3 \cdot S_R) \\) |
| Esdirk54 | \\( O(n^3) \\) | Medium | \\( O(n^3 \cdot S_E) \\) |
| BDF | \\( O(n^3) \\) | Most | \\( O(n^3 \cdot S_B) \\) |

where \\( S_R < S_E < S_B \\) are the respective step counts.

**Crossover point:** Radau5 is competitive when \\( 9 \cdot S_R < S_E \\) (or \\( S_B \\)).
For small systems (dim < 50), the constant factor 9 is usually absorbed by the
step count advantage. For large systems (dim > 200), BDF or ESDIRK wins on
total work.

## Accuracy vs. Cost (Work-Precision Diagrams)

At different tolerance levels, the solvers exhibit different efficiency:

### Low accuracy (rtol = \\( 10^{-3} \\))

| Solver | Best for |
|--------|----------|
| BDF | Lowest total cost (few evals per step) |
| Esdirk32 | Simple and fast |
| Radau5 | Overkill (order too high for low accuracy) |

### Medium accuracy (rtol = \\( 10^{-6} \\))

| Solver | Best for |
|--------|----------|
| Esdirk54 | Good balance of cost and accuracy |
| BDF | Competitive for large systems |
| Radau5 | Best step efficiency |

### High accuracy (rtol = \\( 10^{-10} \\))

| Solver | Best for |
|--------|----------|
| Radau5 | Dramatically fewer steps, 5th order pays off |
| Esdirk54 | Reasonable but many more steps than Radau5 |
| BDF | Struggles (max order 5 limits achievable accuracy) |

## Robustness Comparison

Beyond performance, solvers differ in *robustness* -- their ability to handle
difficult situations without failing:

| Situation | Radau5 | Esdirk54 | BDF |
|-----------|--------|----------|-----|
| Very stiff (\\( > 10^6 \\)) | Excellent | Good | Good |
| Near-singular Jacobian | Good | Fair | Fair |
| Discontinuities in f | Good | Good | Poor (multistep history) |
| DAE (index-1) | Excellent | Not supported | Good |
| Oscillatory stiff modes | Excellent | Good | Poor (BDF3-5 not A-stable) |
| Variable stiffness | Good | Excellent | Good |
| First step (no history) | Good | Good | Poor (startup phase) |

**Key observations:**

- **Radau5** is the most robust overall, especially for DAEs and extremely stiff
  problems. Its main weakness is cost per step for large systems.
- **Esdirk54** is excellent for problems with variable stiffness because it adapts
  naturally through the embedded error estimate.
- **BDF** is least robust near discontinuities (the multistep history becomes
  invalid) and for oscillatory stiff modes (BDF orders 3-5 are not A-stable).
  However, it is the most efficient for large, smooth, very stiff problems.

## Practical Recommendations

Based on the benchmark results:

1. **Start with Esdirk54** for moderately stiff problems. It is the best
   general-purpose choice: L-stable, efficient, and robust.

2. **Switch to Radau5** when:
   - You need high accuracy (rtol < \\( 10^{-8} \\))
   - The problem is a DAE
   - You see poor convergence with ESDIRK
   - The problem has oscillatory stiff modes

3. **Use BDF** when:
   - The system is large (dim > 200)
   - You need moderate accuracy on a very stiff problem
   - The integration interval is very long
   - The solution is smooth (no sharp transients)

4. **Use Auto** when unsure -- it makes reasonable choices based on stiffness
   detection and tolerance analysis:

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
use numra::ode::{Auto, Solver, OdeProblem, SolverOptions};

let problem = OdeProblem::new(/* ... */);
let options = SolverOptions::default().rtol(1e-6).atol(1e-8);

let result = Auto::solve(&problem, 0.0, tf, &y0, &options)?;

// Check what happened:
println!("Steps: {}/{} (accept/reject)",
    result.stats.n_accept, result.stats.n_reject);
println!("Function evaluations: {}", result.stats.n_eval);
println!("LU decompositions: {}", result.stats.n_lu);
```

## Summary Table

| Feature | Radau5 | Esdirk54 | BDF |
|---------|--------|----------|-----|
| Order | 5 | 4 | 1-5 |
| Stability | L-stable | L-stable | Varies |
| Steps (typical) | Fewest | Medium | Most |
| Cost per step | Highest | Medium | Lowest |
| DAE support | Yes (index-1) | No | Yes (index-1) |
| Best system size | Small-medium | Any | Medium-large |
| Best accuracy | High | Medium | Low-medium |
| Robustness | Highest | High | Moderate |
| Best use case | DAEs, high accuracy, extreme stiffness | General purpose | Large systems, long integrations |

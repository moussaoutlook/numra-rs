---
title: "Sensitivity Analysis"
---

For a parameterised initial-value problem

$$
\frac{dy}{dt} \;=\; f(t, y, p), \qquad y(t_0) \;=\; y_0(p),
$$

the **forward sensitivity** $ S(t) $ is the matrix of partial
derivatives of the trajectory with respect to the parameters,

$$
S_{j,k}(t) \;=\; \frac{\partial y_j(t)}{\partial p_k},
\qquad
S(t) \in \mathbb{R}^{N \times N_s}.
$$

Knowing $ S(t) $ — not just at one time but as a trajectory — is the
foundation of parameter estimation, optimal experimental design,
gradient-based calibration, and uncertainty quantification along an
ODE solution. Numra computes it via the simultaneous-corrector
augmented-system formulation introduced by CVODES; the rest of this
page documents the math, the API, the result layout, and the
performance characteristics.

:::tip[Two distinct sensitivity concepts]
This page covers **trajectory sensitivity for ODE/DAE systems** —
$ S(t) = \partial y(t) / \partial p $ at every output time, computed
alongside the state. For scalar local sensitivity coefficients of an
arbitrary $ f: \mathbb{R}^n \to \mathbb{R} $ at a single point, see
[Parameter Importance](./parameter-importance).

The two approaches answer different questions; nothing about choosing
one excludes the other.
:::

## Why solve for $ S(t) $ alongside the state?

The naive alternative is to take *finite differences of the solution*:
solve the ODE at $ p $ and at $ p + h \, e_k $, and approximate
$ \partial y / \partial p_k $ by their difference quotient. This is
sometimes called *external sensitivity* — and it works, sometimes.
But it has three persistent problems on real workloads:

1. **Cost scales as $ N_s + 1 $ full integrations** of the original
   system. Each one is independent and starts from scratch.
2. **Truncation error and round-off compete.** A central FD with step
   $ h $ has $ O(h^2) $ truncation but $ O(\text{eps}/h) $ round-off
   from cancellation; the optimum $ h $ depends on the parameter scale
   and is hard to pick a priori.
3. **Step-control coupling.** The two solves at $ p $ and $ p + h $
   make different adaptive step decisions. The difference quotient
   then conflates *parameter sensitivity* with *step-size noise*. At
   tight tolerances this can completely dominate.

The forward-sensitivity approach instead **augments the state**: it
solves $ y(t) $ and $ S(t) $ together as a single ODE of dimension
$ N \cdot (1 + N_s) $. One adaptive controller, one tolerance bag,
one set of step decisions — and analytical Jacobians can be supplied
where they materially change cost.

## The sensitivity equation

Differentiating the IVP with respect to $ p_k $ and exchanging the
order of differentiation gives the per-parameter sensitivity column

$$
\frac{d S_{:,k}}{dt} \;=\;
\underbrace{\frac{\partial f}{\partial y}}_{J_y}\, S_{:,k}
\;+\;
\underbrace{\frac{\partial f}{\partial p_k}}_{J_p\,e_k},
\qquad
S_{:,k}(t_0) \;=\; \frac{\partial y_0}{\partial p_k}.
$$

Stacking all $ N_s $ columns produces the augmented system

$$
\frac{d}{dt}
\begin{bmatrix} y \\ S \end{bmatrix}
\;=\;
\begin{bmatrix} f(t, y, p) \\ J_y \cdot S \;+\; J_p \end{bmatrix},
$$

with augmented Jacobian

$$
\frac{\partial}{\partial(y, S)}
\begin{bmatrix} f \\ J_y S + J_p \end{bmatrix}
\;=\;
\operatorname{block-diag}(J_y, J_y, \ldots, J_y)
\;+\; \text{higher-order terms in } S,
$$

dropping the second-derivative coupling. This is the
**simultaneous-corrector** form (CVODES `CV_SIMULTANEOUS`) — the
augmented Jacobian factorises as $ N_s + 1 $ identical $ N \times N $
blocks, and on stiff problems implicit solvers can in principle reuse
a single LU of $ M = (1/(\gamma h)) I_N - J_y $ across all blocks.
v1 of Numra's solvers does dense LU on the full augmented matrix;
block-aware factorisation is a tracked follow-up.

### Initial sensitivity

If $ y_0 $ does not depend on $ p $ — the most common case — then
$ S(t_0) = 0 $ and the augmented initial state is simply
$ [y_0;\ \mathbf{0}] $. This is what `AugmentedSystem::initial_augmented`
returns by default.

When $ y_0 $ does depend on $ p $ (for example, the initial value
*is* one of the parameters being calibrated), override
`ParametricOdeSystem::initial_sensitivity`:

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust
fn initial_sensitivity(&self, _y0: &[f64], s0: &mut [f64]) {
    // y_0 = p, so ∂y_0/∂p = 1 (column-major: s0[k * N + j]).
    s0[0] = 1.0;
}
```

The regression test `nontrivial_initial_sensitivity` in
`numra-ode/tests/sensitivity_regression.rs` pins this case for
`dy/dt = -k y, y(0) = p` and checks that the recovered
$ \partial y(t) / \partial p $ matches the analytical $ e^{-kt} $.

## Quick start: closure form

The fastest entry point is `solve_forward_sensitivity_with`, which
takes a closure of shape $ f(t, y, p, dy) $:

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust
use numra::ode::{DoPri5, SolverOptions};
use numra::sensitivity::solve_forward_sensitivity_with;

// dy/dt = -k y, y(0) = 1, k = 0.5.
// Analytical: y(t) = exp(-k t),  ∂y/∂k(t) = -t·exp(-k t).
let r = solve_forward_sensitivity_with::<DoPri5, f64, _>(
    |_t, y, p, dy| { dy[0] = -p[0] * y[0]; },
    &[1.0],     // y0
    &[0.5],     // params
    0.0, 5.0,   // [t0, tf]
    &SolverOptions::default().rtol(1e-10).atol(1e-12),
).unwrap();

let last = r.len() - 1;
let dy_dk = r.dyi_dpj(last, 0, 0);  // ∂y_0(t_final)/∂k_0
let analytical = -r.t[last] * (-0.5 * r.t[last]).exp();
assert!((dy_dk - analytical).abs() < 1e-7);
```

The closure form uses **forward finite differences** for both
Jacobians (the trait-default behaviour). For non-stiff problems and
small parameter counts, this is plenty. Move to the trait form when
you need analytical Jacobians, multi-call reuse, or tight stiff
performance.

## The `ParametricOdeSystem` trait

The full power lives in the trait. A minimal implementation supplies
four methods; everything else is defaulted:

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust
use numra::ode::ParametricOdeSystem;

struct Decay { k: f64 }

impl ParametricOdeSystem<f64> for Decay {
    fn n_states(&self) -> usize { 1 }
    fn n_params(&self) -> usize { 1 }
    fn params(&self)   -> &[f64] { std::slice::from_ref(&self.k) }
    fn rhs_with_params(&self, _t: f64, y: &[f64], p: &[f64], dy: &mut [f64]) {
        dy[0] = -p[0] * y[0];
    }
}
```

`rhs_with_params` is the primitive form: it accepts an arbitrary
parameter vector $ p $ rather than reading from `self`. This is what
lets the trait-default Jacobians perturb $ p $ for FD without mutating
`self` — and what lets `AugmentedSystem` reuse Jacobian scratch
buffers across calls without allocating.

### Default forward FD

When you don't override `jacobian_y` or `jacobian_p`, Numra falls back
to forward finite differences with step $ h = \sqrt{\varepsilon_{\text{mach}}} \cdot (1 + |y_j|) $
(or $ |p_k| $ respectively). On `f64` that's about $ 1.5 \times 10^{-8} $.
Forward FD costs $ N + N_s $ extra RHS evaluations per Jacobian call;
the augmented integrator caches scratch buffers across calls so the
per-call allocation is one-shot.

### Analytical Jacobians

For stiff problems where Jacobians appear inside Newton iteration,
analytical overrides are usually a 10–30% wall-clock win and a much
larger win on Jacobian-evaluation count. The trait expects two
specific layouts:

| Method | Shape | Layout |
|--------|-------|--------|
| `jacobian_y` | $ N \times N $ | **row-major**: `jy[i * N + j] = ∂f_i/∂y_j` |
| `jacobian_p` | $ N \times N_s $ | **column-major**: `jp[k * N + i] = ∂f_i/∂p_k` |

The asymmetry is deliberate: `jacobian_p`'s column-major layout lets
`AugmentedSystem` slice off $ J_p[:, k] $ as a contiguous vector when
assembling the per-parameter sensitivity column — no
gather-from-strided needed. `jacobian_y`'s row-major layout matches
the convention used elsewhere in `numra-ode` for explicit-solver
Jacobians.

### The `has_analytical_jacobian_*` flag contract

Overriding `jacobian_y` (or `jacobian_p`) is **not by itself enough**
to make `AugmentedSystem` use it. The trait also has two flag methods,
`has_analytical_jacobian_y` and `has_analytical_jacobian_p`, that
default to `false`. When the flag is `false`, the augmented hot path
inlines its own forward FD with reused scratch buffers — bypassing the
override entirely.

If you override the method, **also override the flag**:

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust
impl ParametricOdeSystem<f64> for Decay {
    // ... rhs_with_params, n_states, n_params, params ...
    fn jacobian_y(&self, _t: f64, _y: &[f64], jy: &mut [f64]) {
        jy[0] = -self.k;
    }
    fn jacobian_p(&self, _t: f64, y: &[f64], jp: &mut [f64]) {
        jp[0] = -y[0];
    }
    fn has_analytical_jacobian_y(&self) -> bool { true }
    fn has_analytical_jacobian_p(&self) -> bool { true }
}
```

This is the same pattern as CVODES's `set_jacobian_user_supplied` —
it lets the augmented hot path skip FD scratch allocation for the
common FD-default case while still respecting analytical overrides
when they're worth wiring up.

:::caution[Debug-build safety net]
Forgetting the flag is silent in release builds — your analytical
implementation is bypassed and FD is used instead. To catch this,
`AugmentedSystem` runs a **one-time consistency check** on the first
RHS call in debug builds: it evaluates `system.jacobian_*` and inline
FD at the current state, and panics if the Frobenius-relative
difference exceeds $ 10^{-3} $ with a message naming the flag you
forgot. The check runs at zero cost in release; it exists purely so
you encounter the contract in dev rather than ship slow code to
production.

The 1e-3 threshold is wide enough to absorb honest analytical-vs-FD
disagreement (FD is only accurate to $ \sqrt{\varepsilon_{\text{mach}}} \approx 1.5 \times 10^{-8} $)
without false positives, while still catching the "forgot the flag"
mode where the two disagree by $ O(1) $.
:::

## Solving and reading the result

Two entry points, depending on how your system is shaped:

| Function | Takes | Returns | Use when |
|----------|-------|---------|----------|
| [`solve_forward_sensitivity`](https://docs.rs/numra/latest/numra/fn.solve_forward_sensitivity.html) | `&Sys: ParametricOdeSystem` | `SensitivityResult<S>` | You have a struct with analytical Jacobians, multi-call reuse, or non-trivial state. |
| [`solve_forward_sensitivity_with`](https://docs.rs/numra/latest/numra/fn.solve_forward_sensitivity_with.html) | A closure $ f(t, y, p, dy) $ | `SensitivityResult<S>` | You want a quick one-shot call without a struct boilerplate. |

Both are generic over the choice of solver:

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust
use numra::ode::{Bdf, DoPri5, Radau5, SolverOptions, Tsit5};
use numra::solve_forward_sensitivity;

let opts = SolverOptions::default().rtol(1e-8).atol(1e-10);

// Pick the solver via turbofish:
let r = solve_forward_sensitivity::<Radau5, f64, _>(&system, 0.0, 40.0, &y0, &opts)?;
let r = solve_forward_sensitivity::<Bdf,    f64, _>(&system, 0.0, 40.0, &y0, &opts)?;
let r = solve_forward_sensitivity::<DoPri5, f64, _>(&system, 0.0, 40.0, &y0, &opts)?;
```

### Result layout

`SensitivityResult<S>` is a flat-vec result deliberately shaped for
zero-copy slicing. The state is row-major over time:
`y[i * N + j] = y_j(t_i)`. The sensitivity is row-major over time and
**column-major over parameters** within each per-time block:

$$
\texttt{sensitivity}[i \cdot N N_s \;+\; k \cdot N \;+\; j] \;=\; \frac{\partial y_j(t_i)}{\partial p_k}.
$$

The asymmetry is intentional: per-parameter columns
$ S_{:,k}(t_i) $ are contiguous slices, so `sensitivity_for_param`
returns a `&[f64]` of length $ N $ without any indexing dance. This
matches CVODES indexing conventions.

### Accessor surface

Reach for the typed accessors instead of indexing the flat `Vec`
directly. They're unit-tested, layout-agnostic from the caller's view,
and forward-compatible if the underlying flattening ever changes.

| Accessor | Returns | Meaning |
|----------|---------|---------|
| `r.len()` | `usize` | Number of output time points. |
| `r.t[i]` | `S` | Time $ t_i $. |
| `r.y_at(i)` | `&[S]` (length $ N $) | State vector at $ t_i $. |
| `r.sensitivity_at(i)` | `&[S]` (length $ N \cdot N_s $) | Full sensitivity block at $ t_i $, column-major. |
| `r.sensitivity_for_param(i, k)` | `&[S]` (length $ N $) | Contiguous $ \partial y / \partial p_k $ at $ t_i $. |
| `r.dyi_dpj(i, state, param)` | `S` | Single component $ \partial y_{\texttt{state}}(t_i) / \partial p_{\texttt{param}} $. |
| `r.final_state()` | `&[S]` | Convenience: `y_at(len() - 1)`. |
| `r.final_sensitivity()` | `&[S]` | Convenience: `sensitivity_at(len() - 1)`. |
| `r.normalized_sensitivity_at(i, p_nominal)` | `Vec<S>` | $ (p_k / y_j) \cdot \partial y_j / \partial p_k $ — dimensionless. |
| `r.success`, `r.message`, `r.stats` | — | Solver outcome and Criterion-style stats (`n_eval`, `n_jac`, `n_accept`, `n_reject`, `n_lu`). |

The **normalised sensitivity** is the version you usually want when
ranking parameters whose magnitudes span orders of magnitude — see
the Robertson example below for a worked case.

## Worked example: Robertson stiff kinetics

The Robertson chemical kinetics system is the canonical stiff
benchmark: three states, three rate constants spanning eight orders
of magnitude, mass conservation $ y_0 + y_1 + y_2 = 1 $.

$$
\begin{aligned}
dy_0/dt &= -k_1 y_0 + k_3 y_1 y_2 \\
dy_1/dt &=  k_1 y_0 - k_2 y_1^2 - k_3 y_1 y_2 \\
dy_2/dt &=  k_2 y_1^2
\end{aligned}
$$

The complete worked example lives at
[`numra/examples/robertson_sensitivity.rs`](https://github.com/moussaoutlook/numra-rs/blob/main/numra/examples/robertson_sensitivity.rs);
run it with

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```bash
cargo run --release --example robertson_sensitivity -p numra
```

The example walks through implementing `ParametricOdeSystem` with
analytical Jacobians, calling `solve_forward_sensitivity::<Radau5, _, _>`,
and reading the result via the full accessor surface. The most
interesting block is the **influence ranking**:

```text
Normalised sensitivity (k_k / y_j) · (∂y_j/∂k_k):
                    k1             k2             k3
  y0 normalised =     -2.3735e-1     -9.5904e-2      1.9182e-1
  y1 normalised =      1.9993e-1     -3.7169e-1     -2.5662e-1
  y2 normalised =      5.9790e-1      2.4160e-1     -4.8319e-1

Influence ranking (largest |normalised sensitivity| first):
  y0: k1 (2.37e-1) > k3 (1.92e-1) > k2 (9.59e-2)
  y1: k2 (3.72e-1) > k3 (2.57e-1) > k1 (2.00e-1)
  y2: k1 (5.98e-1) > k3 (4.83e-1) > k2 (2.42e-1)
```

Note how the ranking is *different per state*. The slow consumption
of A (state $ y_0 $) is dominated by $ k_1 $; the catalyst
concentration B ($ y_1 $) is dominated by the $ k_2 $ self-reaction;
the cumulative product C ($ y_2 $) is also $ k_1 $-dominated because
it inherits the slow A → B funnel. **A single global ranking would
miss all three of these.**

The output also passes two internal sanity checks:

- **State mass conservation:** $ y_0 + y_1 + y_2 = 1.000000000000 $
  to twelve digits.
- **Sensitivity-mass conservation:** $ \sum_j \partial y_j / \partial k_i = 0 $
  for each $ i $ — a direct consequence of state mass conservation.
  If the integrator drifted, this sum would too; that it stays at
  zero is good evidence the augmented integration is consistent.

## Choosing a solver

The same problem run through different solvers can have radically
different wall-clock costs once the augmented system is in play. The
Robertson bench plotted below makes the point:

![Wall-clock for Radau5 / BDF / DoPri5 / Tsit5 on Robertson with
forward sensitivity. Implicit solvers integrate to t=40; explicit
solvers held to t=1 because they cannot tolerate the stiffness at
the full horizon.](../../../assets/figures/perf_sensitivity_solver_choice.svg)

The takeaway: **for stiff problems, use an implicit solver** —
Radau5 or BDF. The bar chart's hatched bars (DoPri5, Tsit5) only
integrate to $ t = 1 $, *not* $ t = 40 $; comparing them to the
implicit bars at face value would be misleading, which is why the
chart annotates the asymmetry. On the same problem extended to
$ t = 40 $, the explicit solvers would either fail outright or take
orders of magnitude longer.

The general guidance:

| Problem class | First choice | Backup |
|---------------|--------------|--------|
| Non-stiff, smooth | **DoPri5** or **Tsit5** | **Vern6/7/8/9** at very tight tolerances |
| Mildly stiff | **ESDIRK54** | **Radau5** |
| Severely stiff (rate constants spanning $> 4 $ decades) | **Radau5** | **BDF** (NDF / `ode15s`-style) |
| DAE (index ≤ 1) | **Radau5** | **BDF** |

If you don't pick explicitly, [`auto_solve`](https://docs.rs/numra/latest/numra/ode/fn.auto_solve.html)
applies the same heuristics. For the augmented system specifically,
`auto_solve` is conservative and prefers an implicit solver whenever
the underlying state Jacobian shows stiffness — sensitivity equations
inherit the state's stiffness exactly, so this default is usually right.

## Performance

Two charts pin down the cost structure of the forward-sensitivity
integration. All measurements use the publication-grade 20s/3s
Criterion windows; the underlying bench harness lives at
[`numra-bench/benches/sensitivity.rs`](https://github.com/moussaoutlook/numra-rs/blob/main/numra-bench/benches/sensitivity.rs)
and the Criterion-flattened JSON is committed at
[`numra-bench/results/sensitivity_*.json`](https://github.com/moussaoutlook/numra-rs/tree/main/numra-bench/results)
so the SVGs can be regenerated on a different machine without
re-running the benches.

### Cost vs parameter count

The augmented system has dimension $ N \cdot (1 + N_s) $, so
naive theory says wall-clock should grow about linearly in $ N_s $.
What we actually see is **sub-linear** because the per-step overhead
(adaptive step decisions, tolerance checks, function-call dispatch)
amortises across the augmented state:

![Wall-clock per call vs number of parameters on a 2-state damped
oscillator under DoPri5. Going from 1 to 8 parameters costs about
4.7× — closer to linear than the 8× a worst-case scaling argument
would suggest.](../../../assets/figures/perf_sensitivity_param_scaling.svg)

The sub-linear scaling is one reason the simultaneous-corrector
formulation is the right baseline. With staggered correction (CVODES
`CV_STAGGERED` / `CV_STAGGERED1`) you'd recover linear scaling at the
cost of additional Newton iterations per step; both variants are
tracked as named follow-ups for stiff workloads where the trade-off
flips.

### Analytical vs FD Jacobians

For implicit solvers driving a stiff augmented system, Jacobian
quality shows up directly in the inner Newton loop:

![Wall-clock for the same 4-parameter Lotka–Volterra problem under
Radau5 across three Jacobian-supply modes: trait-default forward FD,
analytical state Jacobian only, and full analytical state +
parameter Jacobians.](../../../assets/figures/perf_sensitivity_jacobian_mode.svg)

Going from FD to fully analytical gives about a 19% wall-clock
speedup on this problem (3.0 ms → 2.5 ms). The bulk of the win
comes from $ J_y $ — the parameter Jacobian is only used in the
forcing term of the sensitivity equation, not inside Newton's
iteration matrix, so it has a smaller marginal effect.

The win scales with stiffness: on the more aggressive Robertson
problem (rate constants spanning eight decades), analytical
Jacobians can give 2–3× speedups — far more than the 19% above.
**If you're going to use a stiff solver for sensitivity, write the
analytical Jacobians.** The cost is a few extra trait methods and a
unit test; the payoff is real.

### Allocation policy

`AugmentedSystem` keeps reusable scratch buffers in `RefCell<Vec<S>>`
fields — one each for $ J_y $, $ J_p $, the FD perturbed state and
parameter vectors, and the FD baseline / perturbed RHS values. The
hot path borrows them mutably, never reallocates, and never reaches
back through the trait's default FD (which would allocate per call).
This is what made the
[`solve_trajectory` port land ~40% faster](../../changelog/) than
the previous private `AugmentedOdeSystem`: that version paid six
fresh `Vec` allocations on every RHS evaluation.

The scratch buffers are sized once at `AugmentedSystem::new` time
based on `n_states()` and `n_params()`. If you build a single
`AugmentedSystem` and call `solve_forward_sensitivity` on it
repeatedly with different parameter vectors, those calls share the
scratch — no per-call allocation pressure.

## Edge cases and gotchas

### Parameter scaling

If your parameters span very different magnitudes — say, a rate
constant of $ 10^{-3} $ alongside a temperature of $ 300 $ — the raw
sensitivity matrix will have entries spanning many decades and the
adaptive controller's `atol` may be biting one direction much harder
than another.

The fix is usually to **work in normalised coordinates**:
- Reformulate the model so all parameters are $ O(1) $ —
  e.g. $ k = k_{\text{nominal}} \cdot \tilde{k} $ with $ \tilde{k} $
  the new sensitivity variable.
- Or report results via `normalized_sensitivity_at`, which strips
  the scale by dividing by $ p_k / y_j $ and gives you a
  dimensionless number you can compare across parameters.

### Identifiability

Two parameters that always appear in a fixed combination
(e.g. only as the product $ k_1 k_2 $) will have parallel sensitivity
columns — $ S_{:,1}(t) $ proportional to $ S_{:,2}(t) $ at every
$ t $. The forward-sensitivity solver doesn't care; you'll still
get clean trajectories. But your downstream estimator will choke:
the Fisher information matrix becomes rank-deficient, and any
gradient-based fit will see infinitely many equivalent parameter
combinations.

A quick check: compute the sensitivity matrix at several time points,
stack them into a $ (k \cdot N) \times N_s $ matrix, and look at its
condition number or rank. If two columns are nearly parallel, your
parameters are not jointly identifiable — and no amount of better
data will fix it without a structural change to the model.

### Tolerance inheritance

The augmented system inherits `rtol` and `atol` from `SolverOptions`
uniformly across the state and the sensitivity. This is the SciML
default and works well for most problems. When the sensitivity
magnitudes are very different from the state magnitudes (e.g. you're
computing tiny sensitivities in absolute terms but the state is
$ O(1) $), you may want **separate sensitivity tolerances** —
CVODES exposes `CVodeSensSStolerances` for this. v1 of Numra ships
without that knob; it's tracked as a named follow-up.

### Non-trivial initial conditions

If your initial state itself depends on parameters (`y_0 = y_0(p)`),
the augmented IC for the sensitivity is **not zero**. Override
`initial_sensitivity` to return the correct $ \partial y_0 / \partial p $
in column-major form. Forgetting to do this gives you correct
$ y(t) $ but systematically wrong $ S(t) $ — a silent failure mode
that's particularly nasty in calibration workflows where the bias is
direction-consistent.

The regression test `nontrivial_initial_sensitivity` covers the
canonical case (`y_0 = p` for the simple decay system) and is a good
template for problem-specific overrides.

### Closure form perf

The closure form (`solve_forward_sensitivity_with`) wraps your
closure in a `ClosureSystem`, which reports
`has_analytical_jacobian_y/_p = false`. That means *every* closure-form
solve goes through inline FD with reused scratch — there's no way to
plug in analytical Jacobians via the closure interface. For one-shot
calls and scripts that's fine; for a production calibration loop,
move to the trait form. The performance difference can be 30%+ on
stiff problems.

## Forward vs adjoint sensitivity

Forward sensitivity is the right tool when:
- $ N_s $ (parameter count) is small — say, $ \lesssim 10 $.
- You need $ S(t) $ at many output times, not just one cost-functional gradient.
- You also want the trajectory $ y(t) $ as a side product (you do).

For very large $ N_s $ (think: PDE-constrained optimisation with a
discretised control field — hundreds of thousands of parameters), the
augmented dimension $ N \cdot (1 + N_s) $ blows up and forward
sensitivity becomes unaffordable. The **adjoint method** swaps the
linear-in-$ N_s $ scaling for linear-in-$ N_q $ where $ N_q $ is the
number of *cost-functional outputs* — usually one. Numra exposes
adjoint gradients through `numra-ocp::adjoint_gradient`; see
[Adjoint Methods](../ch10-optimal-control/adjoint-methods).

The rough crossover for ODE-constrained optimisation:

| $ N_s $ | First choice |
|---------|--------------|
| $ \le 10 $ | **Forward** — one solve, full $ S(t) $ trajectory. |
| $ 10 \,\text{–}\, 100 $ | Either; depends on whether you need $ S(t) $ at intermediate times. |
| $ > 100 $ | **Adjoint** — backward solve with state checkpointing. |

If you only need the gradient of a scalar cost
$ J(p) = \int_0^T g(y, p) \, dt $ — and not the full sensitivity
trajectory — adjoint wins almost regardless of $ N_s $.

## Compatibility with other Numra primitives

Forward sensitivity composes cleanly with the rest of the workspace:

- **`numra-ocp::ParamEstProblem`** uses forward sensitivity internally
  (when `OdeSolverChoice::Sensitivity` is set) to compute analytical
  gradients for parameter estimation; see
  [Parameter Estimation](../ch10-optimal-control/parameter-estimation).
- **`numra-ocp::forward_sensitivity`** is now a thin wrapper over
  `solve_forward_sensitivity_with` — same canonical type, same
  accessors, same allocation policy.
- **`numra-ode::uncertainty::solve_trajectory`** in
  `UncertaintyMode::Trajectory` likewise dispatches through the
  canonical primitive.

Anything that previously rolled its own augmented-system loop in the
workspace has been ported over. There is one source of truth.

## Reference

- Hindmarsh et al., *SUNDIALS: Suite of Nonlinear and
  Differential/Algebraic Equation Solvers*, ACM TOMS 31(3), 2005.
  Documents the simultaneous-corrector and staggered-corrector
  formulations.
- Serban & Hindmarsh, *CVODES: An ODE Solver with Sensitivity
  Analysis Capabilities*, Tech. Rep. UCRL-JP-200039, LLNL, 2003.
  Original CVODES design; Numra's column-major sensitivity layout
  follows this convention.
- Hairer, Nørsett & Wanner, *Solving Ordinary Differential
  Equations II: Stiff and Differential-Algebraic Problems*,
  Springer, 1996. Chapter VI.6 covers sensitivity analysis from
  first principles for both Runge–Kutta and BDF integrators.
- Cao, Li, Petzold & Serban, *Adjoint Sensitivity Analysis for
  Differential-Algebraic Equations: The Adjoint DAE System and Its
  Numerical Solution*, SIAM J. Sci. Comput. 24(3), 2003. The
  reference for the forward-vs-adjoint trade-off.

For implementation details and all eight regression tests pinning
the v1 invariants, see
[`numra-ode/tests/sensitivity_regression.rs`](https://github.com/moussaoutlook/numra-rs/blob/main/numra-ode/tests/sensitivity_regression.rs).
The full performance bench harness is at
[`numra-bench/benches/sensitivity.rs`](https://github.com/moussaoutlook/numra-rs/blob/main/numra-bench/benches/sensitivity.rs).

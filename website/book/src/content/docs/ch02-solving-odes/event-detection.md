---
title: "Event Detection"
---

Many physical problems require detecting when a condition is met during
integration: a ball hitting the ground, a chemical concentration reaching a
threshold, or a satellite crossing a plane. Numra's event detection system
finds these **zero-crossings** precisely using bisection on the dense output
interpolant.

## The EventFunction Trait

An event is defined by implementing the `EventFunction` trait:

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust
pub trait EventFunction<S: Scalar>: Send + Sync {
    /// The event function g(t, y). An event occurs when this crosses zero.
    fn evaluate(&self, t: S, y: &[S]) -> S;

    /// Which direction of zero-crossing to detect.
    fn direction(&self) -> EventDirection {
        EventDirection::Both
    }

    /// What to do when the event is detected.
    fn action(&self) -> EventAction {
        EventAction::Continue
    }
}
```

The solver monitors $ g(t, y) $ at each step. When a sign change is
detected between steps, bisection narrows down the crossing time to within
$ 10^{-13} $ in the time coordinate.

## Event Direction

`EventDirection` controls which zero-crossings trigger the event:

| Direction | Triggers when |
|-----------|---------------|
| `Rising` | g goes from negative to positive |
| `Falling` | g goes from positive to negative |
| `Both` | g crosses zero in either direction |

## Event Action

`EventAction` controls what happens after the event is located:

| Action | Behavior |
|--------|----------|
| `Stop` | Integration terminates at the event time |
| `Continue` | Event is recorded, integration continues |

## Registering Events

Add events to `SolverOptions` using the `.event()` builder method:

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust
use numra::ode::SolverOptions;
use numra::ode::events::{EventFunction, EventDirection, EventAction};

struct GroundContact;

impl EventFunction<f64> for GroundContact {
    fn evaluate(&self, _t: f64, y: &[f64]) -> f64 {
        y[0]  // height component
    }
    fn direction(&self) -> EventDirection {
        EventDirection::Falling  // only when falling to ground
    }
    fn action(&self) -> EventAction {
        EventAction::Stop  // stop at impact
    }
}

let options = SolverOptions::default()
    .event(Box::new(GroundContact));
```

Multiple events can be registered by chaining `.event()` calls. Each event
function is independently monitored.

## Example: Bouncing Ball

A ball launched upward under gravity, stopping when it returns to the ground.
The state is $ y = [h, v] $ where $ h $ is height and $ v $ is
vertical velocity:

$$
\frac{dh}{dt} = v, \qquad \frac{dv}{dt} = -g
$$

The event function is $ g(t, y) = h $, triggering when height falls through
zero.

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust
use numra::ode::{DoPri5, Solver, OdeProblem, SolverOptions};
use numra::ode::events::{EventFunction, EventDirection, EventAction};

struct GroundHit;

impl EventFunction<f64> for GroundHit {
    fn evaluate(&self, _t: f64, y: &[f64]) -> f64 {
        y[0]  // height
    }
    fn direction(&self) -> EventDirection {
        EventDirection::Falling
    }
    fn action(&self) -> EventAction {
        EventAction::Stop
    }
}

let g = 9.81;
let v0 = 20.0;  // launch velocity

let problem = OdeProblem::new(
    move |_t, y: &[f64], dydt: &mut [f64]| {
        dydt[0] = y[1];      // dh/dt = v
        dydt[1] = -g;        // dv/dt = -g
    },
    0.0, 10.0,
    vec![0.0, v0],  // start at ground with upward velocity
);

let options = SolverOptions::default()
    .rtol(1e-10)
    .atol(1e-12)
    .event(Box::new(GroundHit));

let result = DoPri5::solve(&problem, 0.0, 10.0, &[0.0, v0], &options).unwrap();

// Integration should have stopped at ground impact
assert!(result.terminated_by_event);

let t_impact = result.t_final().unwrap();
let y_impact = result.y_final().unwrap();

// Analytical: t_impact = 2 * v0 / g
let t_exact = 2.0 * v0 / g;
println!("Impact time:    {:.10} (exact: {:.10})", t_impact, t_exact);
println!("Height at impact: {:.2e}", y_impact[0]);  // should be near 0
println!("Events detected: {}", result.events.len());
```

## Inspecting Detected Events

The `SolverResult` stores all detected events:

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust
for event in &result.events {
    println!("Event {} at t = {:.6}", event.event_index, event.t);
    println!("  State: {:?}", event.y);
}

// Check if integration was stopped by an event
if result.terminated_by_event {
    println!("Integration terminated by event");
}
```

Each `Event` record contains:

| Field | Description |
|-------|-------------|
| `t` | Time of the zero-crossing |
| `y` | State vector at the event time |
| `event_index` | Index of the event function that triggered (0-based) |

## How Bisection Works

When the solver advances from $ (t_n, y_n) $ to $ (t_{n+1}, y_{n+1}) $
and detects a sign change in $ g $:

1. Set the bracket $ [t_a, t_b] = [t_n, t_{n+1}] $
2. Compute the midpoint $ t_m = (t_a + t_b) / 2 $
3. Interpolate the state $ y_m $ at $ t_m $ using dense output
4. Compute $ g(t_m, y_m) $
5. If $ |g_m| < 10^{-12} $, accept $ t_m $ as the event time
6. Otherwise, narrow the bracket: if $ g_a \cdot g_m < 0 $ set
   $ t_b = t_m $, else set $ t_a = t_m $
7. Repeat up to 50 iterations or until $ |t_b - t_a| < 10^{-13} $

The bisection uses the solver's dense output interpolant (not additional RHS
calls), so locating an event is cheap.

**Bisection parameters:**

| Parameter | Value | Purpose |
|-----------|-------|---------|
| Zero tolerance | $ 10^{-12} $ | Absolute g value considered zero |
| Interval tolerance | $ 10^{-13} $ | Stop when bracket is this narrow |
| Max iterations | 50 | Safety limit (gives $ h / 2^{50} \approx h \times 10^{-15} $ precision) |

## Multiple Events

Register multiple event functions by chaining `.event()` calls:

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust
use numra::ode::SolverOptions;

let options = SolverOptions::default()
    .event(Box::new(GroundHit))
    .event(Box::new(ApogeeDetect));
```

Each detected event records its `event_index`, so you can distinguish which
condition fired. If multiple events occur in the same step, the earliest one
(smallest t) is processed first. A `Stop` event terminates integration at that
time, even if other events would have fired later in the step.

## Practical Tips

**Use `Falling` or `Rising` when possible.** Detecting `Both` directions can
trigger spurious events when $ g $ merely touches zero without crossing.

**Combine events with tight tolerances.** The bisection accuracy is limited by
the integration accuracy. Use rtol of 1e-10 or tighter if you need precise
event times.

**Step size limits matter.** The solver can only detect events between adjacent
steps. If $ g $ crosses zero twice within one step, only one crossing (or
neither) will be detected. Use `.h_max()` to limit step sizes if needed.

**Events require dense output.** Registering any event function implicitly
enables dense output computation. DoPri5 provides the most accurate event
location due to its 4th-order interpolant.

**Smooth event functions work best.** The event function $ g(t, y) $ should
be continuous and differentiable. Discontinuous event functions may cause the
bisection to converge slowly or miss crossings entirely.

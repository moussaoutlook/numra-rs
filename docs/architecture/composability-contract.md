# Numra Composability Contract (v0)

**Status:** This document is the testable contract every Numra capability satisfies. It is the operational complement to the Foundation Specification: where the Foundation Specification describes the load-bearing architecture, this document describes the rules every capability follows so that the architecture's composability promise is preserved.

**Audience:** Anyone reviewing a new capability for inclusion in Numra. The author of the new capability is responsible for verifying their work satisfies the contract; the reviewer's job is to confirm.

**Scope:** Every capability in the workspace — every solver, optimizer, integrator, interpolator, statistical routine, signal-processing primitive, fitting routine, plotting backend (when added), ML primitive (when added), etc.

> **Status of the workspace at first publication.** At the time this contract was first published, ~13 of 21 shipped capabilities fail at least one contract item, predominantly item 3 (workspace error propagation) and item 7 (interop test). The gaps are tracked in `docs/internal-followups.md` as F-ERR and F-INTEROP-Q. These follow-ups are committed work — every foundation-affecting PR also closes at least one gap. The contract is the ratchet; the gaps will close over time.

---

## What "composable" means here

A capability is **composable** when:

- It can be used as part of a workflow alongside other Numra capabilities without the user writing adapter code.
- Its outputs flow into other capabilities' inputs through the foundation's type spine, not through capability-specific glue.
- It can be replaced by an alternative capability with the same role (e.g., one ODE solver swapped for another) without breaking the surrounding code.
- A user implementing a custom capability that satisfies the same trait can plug it into existing Numra workflows.

This is what "composability" means in Numra. Not "uses similar types," not "feels integrated," not "shares a workspace." Specifically the four properties above.

---

## The contract

A capability satisfies the contract when **all** of the following hold. Each rule includes how to verify it.

### 1. The problem-definition trait is publicly implementable

If the capability is a problem class (ODE, optimization problem, integration problem, etc.), users outside the workspace must be able to define their own problem instance by implementing the relevant trait.

**Verification:**

- The trait uses only `pub` items in its method signatures and default-method bodies.
- The trait has no `pub(crate)` types in its public surface.
- The trait is not sealed (no `private::Sealed` supertrait).
- A test exists demonstrating that a struct defined in a downstream test crate, implementing the trait using only `pub use` items from the Numra facade, integrates with the capability.

**Rationale:** Composability across the workspace boundary. If users can't extend the foundation, the workspace is closed; the composability story is internal-only and breaks the moment a user has a custom problem.

### 2. The primary result type is inspectable

If the capability produces a result, the result type exposes its underlying data through public fields or accessor methods.

**Verification:**

- All fields a downstream consumer might want are either `pub` or have a `pub fn` accessor.
- Accessors return references or owned data; they don't require the user to clone the entire result to read part of it.
- The result type is `Debug`. Helpful: `Clone` where the data shape allows.
- Optional but recommended: `Serialize`/`Deserialize` behind a `serde` feature flag.

**Rationale:** Composition by data flow requires that data is reachable. An opaque result is a dead-end.

### 3. Errors compose into the workspace error type

If the capability is fallible, its error type implements `From<MyError> for numra_core::NumraError` (the workspace-level error vehicle), so `?` propagates across crate boundaries.

**Verification:**

- The crate's error type has the required `From` impl.
- A test exists exercising `?`-propagation from the capability's API through at least one other capability's API into a single `Result` return.

**Rationale:** The unified `?` operator is one of the most user-visible composability properties. Manual `.map_err` chains are friction.

### 4. The capability accepts at least one foundation-typed input

The capability's primary entry point accepts an input expressible as a foundation type (e.g., `&[S]`, `OdeSystem`, `Vector`, `Matrix`, `SolverOptions`) without requiring the user to first construct a capability-specific wrapper that doesn't already exist as a foundation type.

**Verification:** Read the capability's primary entry point. Every input parameter is either:
- A foundation type, or
- A capability-specific type that the user constructs from foundation types using only `pub` items, with documentation showing how, or
- A primitive (e.g., `usize`, `bool`, `&str`) for a configuration parameter.

**Rationale:** "Drop in your data and go" is what composability looks like to the user. A capability that demands its own input wrapper before it can be used is composable in principle, frictional in practice.

### 5. The capability produces a foundation-flowable output

The capability's primary entry point returns an output that flows naturally into at least one other capability's input.

**Verification:** For at least one other capability in the workspace, demonstrate (in code, in a test) that the output of *this* capability can be passed as input to *that* capability without an adapter the user must write.

The chain of "X output → Y input" relationships across the workspace forms the composability graph. Every capability adds at least one edge.

**Rationale:** Composability is a graph property, not a node property. A capability that produces an output nothing else consumes is composable in isolation, useless in composition.

### 6. The capability is generic over `Scalar` where algorithmically possible

The capability's public API is generic over `S: Scalar`, not concrete `f64`, unless the underlying algorithm has a hard `f64` dependency (e.g., a wrapped C library that's `f64`-only, a faer routine that's `f64`-only).

**Verification:**

- Inspect the public function signatures. Are they generic, or concrete `f64`?
- If concrete, is the rationale documented in the rustdoc?
- Is there a test that exercises the capability with `S = f32` (or `Complex<f64>` once that's a `Scalar`)?

**Rationale:** Numerical workflows mix precisions. ODE in `f64`, optimization in `f64`, but a sensitivity computation that wants to drop to `f32` for memory, or a plotting layer that downcasts to `f32` for the GPU — composability requires the option to be exercised. Concrete-`f64` leaks become barriers to every future precision-related feature.

The Foundation Specification §2.1 audit (recorded 2026-05-10) found three remaining concrete-`f64` leaks in foundation-adjacent code: `numra-autodiff` reverse mode, `numra-linalg` general eigendecomposition (separate `f32`/`f64` impls rather than a generic blanket), and `numra-signal` filter design. Tracked as the F-LEAK follow-up in `internal-followups.md`.

### 7. The capability ships with at least one interop test

Every new capability lands with at least one test in `numra/tests/interop_workflows.rs` (or a sibling file) demonstrating composition with at least one existing capability.

**Verification:**

- The test exists and runs as part of the standard `cargo test --workspace` flow.
- The test does not include manual adapter code that should belong in a foundation-level conversion.
- The test failing means a real composability regression has happened.

**Rationale:** Tests are the ratchet. Without them, composability erodes silently as foundation-affecting changes land. With them, every change to the foundation is checked against real composition before it can break.

The interop test suite is the primary defense against composability rot.

---

## Worked examples

### Example: Forward sensitivity API (ParametricOdeSystem)

Walking the contract for the recently shipped sensitivity API. This is preserved deliberately as a worked example of *what surfacing contract violations looks like* — the API satisfies most of the contract but fails item 5, and the failure is informative.

1. ✓ **Publicly implementable trait.** `ParametricOdeSystem` (`numra-ode/src/sensitivity.rs:152`) is `pub`, no `pub(crate)` leakage, not sealed. External impl is demonstrated by the Robertson example (`numra/examples/robertson_sensitivity.rs:53`) and by `numra-ode/tests/sensitivity_regression.rs`.
2. ✓ **Inspectable result.** `SensitivityResult<S>` exposes `t, y, sensitivity, n_states, n_params, stats, success, message` as `pub` fields, plus the documented column-major layout and accessors `y_at, sensitivity_at, sensitivity_for_param, dyi_dpj, final_state, final_sensitivity, normalized_sensitivity_at`.
3. ◐ **Workspace error.** `solve_forward_sensitivity` returns `Result<SensitivityResult<S>, SolverError>`. `SolverError` does **not** currently implement `From<SolverError> for NumraError`, so `?`-propagation into a workspace-error pipeline requires manual `.map_err` today. Tracked as F-ERR.
4. ✓ **Foundation-typed input.** Takes `&Sys: ParametricOdeSystem`, `t0: S`, `tf: S`, `&[S]`, `&SolverOptions<S>` — all foundation types.
5. ✗ **Foundation-flowable output.** `SensitivityResult` ships with **no in-workspace downstream consumer**. The two consumers the original draft claimed (`LevenbergMarquardt::fit` in `numra-optim`; `numra-ode::uncertainty::solve_trajectory`) do not consume `SensitivityResult` at all — `lm_minimize` takes closure-shaped residual+jacobian, `solve_trajectory` takes a model closure and parameter list (it *produces* uncertainty by computing sensitivities internally; it does not *consume* a `SensitivityResult`). The only workspace site that mentions `SensitivityResult` outside the producing crate is `numra-ocp::forward_sensitivity` (`numra-ocp/src/sensitivity.rs:73`), which **returns** a re-exported `SensitivityResult` rather than consuming one. Composability gap tracked as F-SENS-DOWNSTREAM. The completion of the sensitivity capability requires building or naming downstream consumers.
6. ✓ **Generic over Scalar.** `solve_forward_sensitivity::<Sol, S, Sys>` is generic.
7. ◐ **Interop test.** The test `workflow_param_est_sensitivity_uncertainty` (`numra/tests/interop_workflows.rs:142`) exists, but it does not actually exercise `SensitivityResult` composition — it uses `ParamEstProblem` and `solve_with_uncertainty` (both of which consume sensitivities internally). The `SensitivityResult` type is never named in the test. Item 7's literal wording ("ships with at least one interop test") is technically met; the substantive composability the worked example claims is not. Tracked as F-INTEROP-Q.

The sensitivity API is **not currently** a complete contract-satisfier. The two ◐ items + the ✗ on item 5 are the visible costs of having shipped a capability before this contract was written. Item 5's gap is the most important — it surfaces that even a recently-celebrated, well-tested capability can lack a downstream composition story, and the contract is the document that makes that visible.

### Example: A capability that would fail the contract

Hypothetical: a "RealtimeTraceVisualizer" that takes an `OdeSystem`, runs it internally, and pushes results to a stdout-only formatted output.

1. ✗ **Inspectable result.** No structured result; output is stdout text.
2. ✗ **Foundation-flowable output.** Nothing downstream can consume stdout text without parsing.
3. ✓ Other rules pass.

This capability is *useful in isolation* but *not composable*. It would not pass review for inclusion in the workspace. The fix: produce a structured result type (`TraceResult` with `t`, `y`, `events`, etc.) and a separate formatting helper, so the capability composes by default and the formatting is a downstream consumer.

This is the kind of decision the contract makes early.

### Example: A capability that satisfies the contract trivially

A new ODE solver. Implements `Solver` trait. Existing problem types compose with it; existing result type is reused; no new error type needed; entry point takes `OdeSystem` and produces `SolverResult`. Generic-over-scalar inherited. Single interop test demonstrating it solves a Lorenz problem and the result feeds an interpolant.

The contract is easy to satisfy when foundation traits already exist for the role. This is the goal: capability authors mostly inherit composability for free.

---

## Process: how the contract is applied

### When a new capability is proposed

The author of the capability writes a short design note answering each of the seven contract items. The design note is reviewed before significant implementation work begins. If the capability proposes a foundation change to satisfy the contract — e.g., a new foundation-flowable type — the change is reviewed against the Foundation Specification.

### When a new capability is implemented

The author verifies each contract item is met before opening a PR. The PR description includes the contract verification (a checklist with seven items, each ticked with the line/file evidence).

### When a foundation change lands

The reviewer walks the existing capabilities and verifies each still satisfies the contract under the new foundation. Where a foundation change would break a capability's contract compliance, the foundation change is responsible for repairing it (not the capability).

### When the contract itself needs to evolve

This document is updated. The update is part of the change that requires it. The Foundation Specification's decision log records the change.

---

## Anti-patterns the contract forbids

These are patterns that violate the contract. Reviewers reject them.

### Anti-pattern: stringly-typed configuration

Configuration via string keys (`config.set("solver", "Radau5")`) instead of typed enums or generic parameters. Hides errors at the type level; breaks composition across crate boundaries.

### Anti-pattern: sentinel return values

Returning `f64::NAN` or `-1` to signal failure instead of `Result<T, E>`. Breaks `?` propagation; downstream consumers can't tell signal from data.

### Anti-pattern: hidden state via global mutable variables

`thread_local!` mutable state, `static mut`, lazy-initialized singletons. Breaks composition because the capability's behavior depends on history not visible in the call site.

### Anti-pattern: capability-specific clones of foundation types

A new "MyVector" type that's almost-but-not-quite `Vector`. Forces users to convert at every boundary. The new capability either uses the existing foundation type or proposes a foundation change that benefits everyone.

### Anti-pattern: print-or-panic instead of return-error

`println!` and `eprintln!` for error reporting from library code, or `panic!` on bad input. Breaks `?` propagation; makes capabilities unusable in non-interactive contexts (web servers, batch jobs, GUI applications).

### Anti-pattern: opaque result types

`pub struct Result(InternalState)` with no public accessors. Forces every downstream consumer to depend on framework-defined extraction methods.

### Anti-pattern: defensive `Send + Sync` bounds

Foundation traits and their direct consumers don't add `Send + Sync` defensively. Bounds added without a documented reason become breaking-to-remove. The recent `ParametricOdeSystem` decision is the precedent. Pre-existing defensive bounds in older foundation traits (`Scalar`, `Signal`, `EventFunction`, `NonlinearSystem`, `SdeSystem`, the PDE/OCP closure aliases) predate this anti-pattern's codification and are tracked as F-SENDSYNC; reviewers reject *new* defensive bounds.

### Anti-pattern: secret runtime requirements

A capability that only works if certain environment variables are set, or if certain files exist on disk, or if a specific tokio runtime is available — without these requirements being explicit in the API. Users hit the requirement at runtime, not at compile time.

---

## What the contract is NOT

This contract is about composability, not about correctness, performance, or completeness. A capability can satisfy the contract and still:

- Be slow.
- Have bugs.
- Cover only a subset of the problem class.
- Be missing tests beyond the single interop test.
- Have rough edges in its API ergonomics.

These are quality concerns, addressed through normal engineering review. The contract is the floor: below the floor, composability breaks. Above the floor, normal quality review applies.

---

## How this document is maintained

- **Updated when contract evolves.** New rules added when patterns emerge. Existing rules clarified when reviews surface ambiguity.
- **Referenced from PR templates.** New-capability PRs include the seven-item checklist as a verification step.
- **Reviewed annually.** Walk the workspace's capabilities; spot-check contract compliance; surface drift.

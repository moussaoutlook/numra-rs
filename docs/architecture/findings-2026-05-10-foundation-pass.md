# Foundation Specification & Composability Contract — Findings Report

**Date:** 2026-05-10.
**Scope:** Verification pass on the two draft architecture documents:

- `docs/foundation-specification.md` (v0 draft, 337 lines)
- `docs/composability-contract.md` (v0 draft, 229 lines)

**Method:** Workspace top-down walk; trait-signature extractions against
source; verification of stated design choices against current code; audit
sweeps for concrete-`f64` leaks, `pub(crate)` exposure in foundation
surfaces, sealed-trait machinery, interior mutability, defensive
`Send + Sync` bounds, cross-crate `?`-propagation, broken genericity;
decision-log extraction from CHANGELOG and `git log`; composability-graph
mapping from `numra/tests/interop_workflows.rs` plus cross-crate code.

**Findings format:** Grouped by the document section the finding affects.
Within each group, findings are numbered. Each finding records (a) the
draft callout or assertion under examination, (b) what the code shows
with file:line references, (c) the implication for the document, and (d)
any follow-up the finding implies.

**Scope discipline:** The verification pass surfaces issues. Source-code
changes implied by the findings are recorded as named follow-ups in
`docs/internal-followups.md`; they are not executed in this PR. Where a
finding only requires a draft revision (typo, wrong signature, factual
correction), the revision is applied to the local revised drafts at
`docs/foundation-specification.revised.md` /
`docs/composability-contract.revised.md` for review before they land at
the final architecture paths.

---

## Index

- A. Findings affecting Foundation Spec **§1 (Scope of "foundation")**
- B. Findings affecting Foundation Spec **§2 (Design principles)**
- C. Findings affecting Foundation Spec **§3 (Trait spine)**
- D. Findings affecting Foundation Spec **§4 (Composability primitives)**
- E. Findings affecting Foundation Spec **§5 (testable contract)**
- F. Findings affecting Foundation Spec **§6 (Decision log)**
- G. Findings affecting Foundation Spec **§7 (Open questions)**
- H. Findings affecting Composability Contract **(rules + worked
  examples)**
- I. Findings affecting **both documents** (cross-cutting)
- J. Meta-finding on contract-violation density across shipped
  capabilities

---

## A. Foundation Spec §1 — Scope of "foundation"

### A1. The §1 list omits `ParametricOdeSystem`

**Callout:** > "[CC: investigate] Verify this list against the actual
`numra/src/lib.rs` re-exports and the workspace dependency graph […] are
there other traits that meet the three-or-more-crate-consumer test?"

**Found:** `ParametricOdeSystem` is defined in
`numra-ode/src/sensitivity.rs:152` and is re-exported from the top-level
facade (`numra/src/lib.rs:62`). It is consumed by:

- `numra-ode` itself — `AugmentedSystem` impl
  (`numra-ode/src/sensitivity.rs:479`); `solve_forward_sensitivity`
  (line 831) and `solve_forward_sensitivity_with` (line 934);
  `solve_trajectory` (per CHANGELOG, reimplemented atop the canonical
  primitive at commit `6f26aaf`, 2026-05-06).
- `numra-ocp` — `forward_sensitivity` (per CHANGELOG, reimplemented as a
  thin wrapper at commit `34d81b6`, 2026-05-06; re-exports
  `SensitivityResult` from `numra-ode` at
  `numra-ocp/src/sensitivity.rs:26`).
- `numra-pde` — `ParametricMOLSystem2D` and `ParametricMOLSystem3D`
  (`numra-pde/src/mol2d_parametric.rs`, `mol3d_parametric.rs`; CHANGELOG
  entry "feat(numra-pde): ParametricMOLSystem2D and ParametricMOLSystem3D",
  commit `da0674b`, 2026-05-08).

That is three workspace crates implementing or consuming the trait, and a
fourth (`numra/src/lib.rs`) re-exporting it. It clears every clause of
the §1 criterion.

**Implication:** §1 should list `ParametricOdeSystem` alongside
`OdeSystem` as a foundational trait. Its absence is a documentation gap,
not a design defect — but the foundation can't be governed if it isn't
named.

**Document revision:** Add `ParametricOdeSystem` to the §1 enumeration,
between `OdeSystem` and `Signal`.

**Follow-up:** None — naming-only fix.

### A2. The §1 list miscites where `Matrix` lives

**Callout:** §1 says "Vector and Matrix traits (`numra-core` /
`numra-linalg`)".

**Found:** `Vector` is in `numra-core/src/vector.rs:40`. `Matrix` is in
`numra-linalg/src/matrix.rs:15`. There is no `Matrix` trait in
`numra-core`. `numra-linalg/src/lib.rs:55` exposes both as a crate-level
public API; `numra-core` does not.

**Implication:** The `/` formulation suggests both crates host pieces of
the trait or that core hosts the trait and linalg hosts impls. Neither
is true. Cleaner: split into "Vector trait (`numra-core`)" and "Matrix
trait (`numra-linalg`)" so the locations are unambiguous.

**Document revision:** Replace the single bullet with two bullets in §1.

**Follow-up:** None — citation fix.

### A3. The §1 criterion would also class `EventFunction` and `NonlinearSystem` as foundational, but the draft excludes them silently

**Found:**
- `EventFunction` (`numra-ode/src/events.rs:66`) is referenced in
  `SolverOptions::events` (`numra-ode/src/solver.rs:41`) so it lives in
  every solver options instance the workspace consumes. It satisfies §1
  clause 1 only loosely (its consumer count is one — the ODE crate),
  but clause 3 ("removing it would require redesigning at least one
  major workflow") plausibly applies to event-driven termination.
- `NonlinearSystem` (`numra-nonlinear/src/newton.rs:100`) is consumed by
  `numra-nonlinear` itself, `numra-optim` (Newton inside SQP), and
  implicitly by `numra-ode` solvers that call `numra-nonlinear`
  internally.

**Implication:** Both arguably qualify. The honest framing is that `§1`
is a guideline, not an oracle; the foundation list is curated for
load-bearingness, not derived mechanically from a metric. Suggest
softening §1's introduction to acknowledge this — "the foundation list
below is the current curated set; any trait that meets these criteria
is *eligible* for inclusion, but inclusion is a deliberate decision".

**Document revision:** Add a one-sentence qualifier after the §1
criterion list. Do not add the borderline traits to the foundation list
until a decision says they belong.

**Follow-up:** None — clarification only.

---

## B. Foundation Spec §2 — Design principles

### B1. §2.1 (concrete-`f64`) — full audit results

**Callout:** > "[CC: investigate] Audit the workspace for foundational
items that are concrete-`f64` when they could be generic. […] Surface
every concrete-f64 in foundational code as a candidate for generification
or deliberate annotation."

**Found:** Three classes of concrete-`f64` leak in code reachable from
the foundation:

1. **`numra-autodiff` reverse mode is entirely `f64`.** Public API
   surface:
   - `tape::Tape::var(tape, value: f64)` (`tape.rs:67`),
     `Tape::gradient` (`tape.rs:122` returns `Vec<f64>`),
     `Tape::jacobian` (`tape.rs:149` returns `Vec<Vec<f64>>`).
   - `reverse::Var::cst(value: f64)` (`reverse.rs:57`),
     `Var::powf(n: f64)` (`reverse.rs:212`).
   - `reverse::grad(f, x: &[f64]) -> Vec<f64>` (`reverse.rs:572`).
   - `reverse::jacobian_reverse(f, x: &[f64]) -> Vec<Vec<f64>>`
     (`reverse.rs:599`).
   - `reverse::hessian(f, x: &[f64]) -> Vec<Vec<f64>>` (`reverse.rs:630`).
   - The forward-mode `Dual<S>` (`dual.rs`) is generic; only reverse
     mode is concrete-`f64`. The audit's claim "reverse mode hard-coded
     `f64`" is confirmed and complete.

2. **`numra-linalg` general (non-symmetric) eigendecomposition is
   split into separate concrete impls, not a generic blanket.**
   - `EigenDecomposition<S: Scalar>` (`eigen.rs:57`) is generic in
     declaration; `impl EigenDecomposition<f64>` at `eigen.rs:63` and
     `impl EigenDecomposition<f32>` at `eigen.rs:90` are the only two
     constructors. There is no generic-over-`Scalar` impl.
   - The audit's earlier claim "general eigendecomposition hardcoded
     `f64`" is partially out of date — `f32` was added — but the
     underlying issue (no blanket impl for arbitrary `Scalar`) is real.
     External `Scalar` impls (e.g. a hypothetical `BigDecimal`) get
     nothing.
   - Cause: faer's `complex_native::c64`/`c32` for the underlying
     general Hessenberg routine. Symmetric eigendecomposition
     (`SymEigenDecomposition`, `eigen.rs:21`) is generic over the faer
     `Scalar + SimpleEntity + Conjugate<Canonical = S> + ComplexField`
     bound, so it covers any faer-friendly `Scalar`.

3. **`numra-signal` filter design is concrete-`f64`.**
   - `filter_design::butter(order: usize, cutoff: f64, fs: f64) -> Result<SosFilter<f64>, SignalError>`
     (`filter_design.rs:33`).
   - `hilbert::instantaneous_frequency<S: Scalar>(x: &[S], fs: f64)`
     (`hilbert.rs:104`) — partially generic but `fs` is concrete `f64`.
   - This isn't the foundation strictly, but `numra-signal`'s outputs
     feed downstream consumers (FFT, peaks) and the `f64` leak forces
     monomorphisation through any pipeline that touches it.

**Other concrete-`f64` items examined and judged acceptable:**
- `numra-core::scalar::to_f64_vec` / `from_f64_vec` (`scalar.rs:772, 777`)
  are by-design conversion utilities; concrete `f64` is the point.
- `numra-stats::distributions::Binomial::new(n: usize, p: f64)`
  (`binomial.rs:21`) — distribution probability parameter, narrow.
- `numra-sde::stats::percentile<S: Scalar>(data: &[S], p: f64)`
  (`stats.rs:365`) — percentile parameter, bounded `[0,1]`, narrow.
- `numra-optim::types::OptimResult.wall_time_secs: f64`
  (`types.rs:138`) — wall-clock seconds, never multiplied with
  numerical state.

**Implication:** §2.1's principle is correct and the audit confirms it
matters. Two concrete sites (autodiff reverse mode, signal filter
design) are foundational-adjacent leaks worth generifying. The eigen
case is acceptable as a documented faer constraint but worth annotating.

**Document revision:** §2.1 stays as written. The §3.1 forward-
compatibility risk paragraph already names these concretely.

**Follow-up:** Yes — see follow-up F-LEAK below ("Generify foundational
code over `Scalar`"). Per the granularity directive, this is one
substantial follow-up, with the three sites listed inside rather than
fragmented across three entries.

### B2. §2.2 (`pub(crate)` in foundation surfaces) — clean

**Callout:** > "[CC: investigate] Audit foundation traits for any of the
four anti-patterns above. […] does `OdeSystem` reference any
`pub(crate)` types in its method signatures or default bodies?"

**Found:** Walked every foundation trait listed in §1 plus
`ParametricOdeSystem`. Method signatures and default bodies use only
public items (`Scalar`, slices, `Vec<S>`, `usize`, `bool`,
`LinalgError`, `Option<S>`).

`pub(crate)` items in the workspace are confined to:
- Internal helpers in `numra-interp` (`search_sorted`, `validate_data`,
  `eval_piecewise_cubic`, `eval_piecewise_cubic_deriv`,
  `integrate_piecewise_cubic`, all in `lib.rs:86–164`).
- Internal builder fields in `numra-optim::OptimProblem`
  (`problem.rs:53–107`) — surfaced through the public builder methods.
- Internal tape state in `numra-autodiff::tape::Node`,
  `Tape.{nodes,n_inputs}` (`tape.rs:18, 49, 51, 84, 101`) and
  `reverse::Var.{index, tape}` (`reverse.rs:48, 52`).
- Internal scratch helpers (`numra-optim::lbfgs::two_loop_recursion`,
  `numra-spde::noise::cholesky_decompose`).

None are reachable from a foundation trait method signature or default
body.

**Implication:** §2.2 is satisfied.

**Document revision:** None.

**Follow-up:** None.

### B3. §2.2 (sealed-trait machinery in the foundation) — clean

**Found:** Workspace-wide `rg 'Sealed|private::|sealed::'` returns
nothing in foundation code. No `Sealed` supertrait pattern exists
anywhere. (`SparseScalar` in `numra-pde/src/sparse_assembly.rs:19` is a
public trait alias with a blanket impl — anyone can satisfy it
implicitly. Not sealing.)

**Implication:** §2.2's prohibition is satisfied.

**Document revision:** None.

**Follow-up:** None.

### B4. §2.3 (inspectable result types) — partial issues

**Callout:** > "[CC: investigate] Audit the public surface of
`SolverResult` (numra-ode), `SensitivityResult` (numra-ode),
`OptimResult` (numra-optim), and any equivalent in numra-sde, numra-dde,
numra-fde, numra-ide, numra-pde, numra-spde, numra-ocp."

**Found:**
- `SolverResult<S>` (`numra-ode/src/solver.rs:177`) — all fields `pub`
  (`t, y, dim, stats, success, message, events, terminated_by_event,
  dense_output`). Plus accessors `len`, `is_empty`, `t_final`, `y_final`,
  `y_at`, `n_steps`, `component`, `iter`. ✓
- `SensitivityResult<S>` (`numra-ode/src/sensitivity.rs:614`) — all
  fields `pub` (`t, y, sensitivity, n_states, n_params, stats, success,
  message`). Plus accessors `len, is_empty, y_at, sensitivity_at,
  sensitivity_for_param, dyi_dpj, final_state, final_sensitivity,
  normalized_sensitivity_at`. ✓
- `OptimResult<S>` (`numra-optim/src/types.rs:123`) — all fields `pub`
  (`x, f, grad, iterations, n_feval, n_geval, converged, message,
  status, history, lambda_eq, lambda_ineq, active_bounds,
  constraint_violation, wall_time_secs, pareto, sensitivity`). ✓
- `ParameterSensitivityResult<S>` (`numra-core/src/uncertainty.rs:226`)
  — accessors only; needs verification of field-vs-accessor coverage.

The §1 criterion's other crates (`numra-sde, numra-dde, numra-fde,
numra-ide, numra-pde, numra-spde, numra-ocp`) — none defines its own
"result" type; they all return `SolverResult<S>` or domain-specific
results that should be checked individually. Spot checks:
`numra-sde` returns `SolverResult` (per CHANGELOG and `solver.rs`);
`numra-pde::MOLSystem2D` impls `OdeSystem<S>` and downstream solvers
return `SolverResult<S>`.

**Implication:** §2.3 is satisfied for the named heavyweight results.
No "opaque result" anti-pattern detected.

**Document revision:** None.

**Follow-up:** None.

### B5. §2.4 (workspace error type) — substantial gap, needs surface

**Callout:** > "[CC: investigate] What's the current state? Is there a
workspace-wide `numra::Error` type that crate-specific errors convert
into?"

**Found:** A workspace error type **does** exist —
`numra_core::NumraError` (`numra-core/src/error.rs:16`) — and is
re-exported from the facade (`numra/src/lib.rs:43` exports
`NumraError, NumraResult`). But coverage of `From<…> for NumraError`
impls is patchy.

Crates **with** a `From` impl into `NumraError`:
- `numra-core` (sub-errors: `LinalgError`, `ConvergenceError`,
  `OptimizationError`).
- `numra-interp` (`InterpError`, `interp/src/error.rs:45`).
- `numra-integrate` (`IntegrationError`, `integrate/src/error.rs:51`).
- `numra-special` (`SpecialError`, `special/src/error.rs:38`).
- `numra-stats` (`StatsError`, `stats/src/error.rs:41`).

Crates **without** a `From` impl into `NumraError`:
- `numra-ode` (`SolverError`, `numra-ode/src/error.rs:12`) — the most
  consumed error type in the workspace; any cross-crate `?` from the
  ODE world fails to land in `NumraError` without manual conversion.
- `numra-optim` (`OptimError`, `numra-optim/src/error.rs:11`) — has
  inbound `From<numra_nonlinear::LineSearchError>` and
  `From<numra_nonlinear::LinalgError>`, but no outbound to `NumraError`.
- `numra-ocp` (`OcpError`, `numra-ocp/src/error.rs:11`).
- `numra-fit` (`FitError`, `numra-fit/src/error.rs:11`).
- `numra-signal` (`SignalError`, `numra-signal/src/error.rs:11`).

Crates with no error type at all (return concrete `Vec<S>`, etc., or
panic): `numra-linalg` (uses core's `LinalgError`),
`numra-nonlinear` (uses `LineSearchError` from itself; no `NumraError`
interop), `numra-sde, numra-dde, numra-fde, numra-ide, numra-pde,
numra-spde, numra-autodiff, numra-fft`.

**Implication:** §2.4's principle ("a user writing
`solve_ode(...)?.integrate(...)?.fit(...)?` must have `?` work") is
**not** satisfied today. The minimum surface needed is `From<SolverError>
for NumraError` and `From<OptimError> for NumraError` — those two
unblock most realistic pipelines. Adding the rest can be a follow-on.

**Document revision:** §2.4 should be reframed from "this requires a
workspace error-propagation discipline" (which reads as if one already
exists) to "the workspace-wide `NumraError` exists; the discipline that
every fallible crate's error converts into it is partially applied —
gaps are tracked as named follow-ups". The Open Question §7.6 ("[CC:
verify] If there isn't one, design and add. If there is one, is it
sufficient for cross-crate `?` propagation?") gets a verified answer
("exists; insufficient coverage today; coverage is the work item, not
the type") and can be closed.

**Follow-up:** Yes — see follow-up F-ERR below ("Complete the
`NumraError` From-impl coverage"). Recorded under the "API" section of
internal-followups.

### B6. §2.5 (`SolverOptions` shared across families) — partial truth

**Callout:** > "[CC: investigate] how widely is `SolverOptions` actually
used? Does `numra-sde` use it? `numra-dde`? `numra-pde`? `numra-spde`?"

**Found:** `numra-ode::SolverOptions<S>` is the canonical shared options
type. Cross-crate consumption:
- `numra-pde` MOL systems impl `OdeSystem<S>` and are solved via the
  same `Solver::solve(problem, t0, tf, y0, &SolverOptions<S>)` entry
  point — so PDE inherits `SolverOptions`. ✓ (Confirmed by the PDE
  example in `numra-pde/src/lib.rs` lib doc and by interop_workflow 5.)
- `numra-dde` uses `numra-ode` internally (Method-of-Steps over DoPri5)
  — inherits `SolverOptions` for the inner step.
- `numra-spde` uses `numra-pde` and `numra-sde`; PDE arm inherits
  `SolverOptions`, SDE arm has its own options struct (per audit Tier 9).
- `numra-sde` has its own options machinery (per audit) — does **not**
  share `SolverOptions`. Worth quoting the rationale.
- `numra-fde, numra-ide` — fixed-step solvers (per audit) with their
  own minimal options; do not consume `SolverOptions`.
- `numra-ocp` — wraps `numra-ode` (DoPri5-only per audit), inherits
  `SolverOptions` for the inner ODE solve.

**Implication:** §2.5's "one type used by every solver that consumes
ODEs" is correct as written — `SolverOptions` is used by every solver
that consumes ODEs (because it's tied to the `Solver` trait and the
ODE-derived problem class). But the principle's stronger framing ("New
solver families that define their own options struct must justify the
divergence") has not been retroactively applied to SDE/FDE/IDE — they
have separate options without rustdoc explaining why.

**Document revision:** §2.5 stays as a forward-looking principle.
Optional: add a note that the principle applies prospectively; the
existing SDE/FDE/IDE options divergences predate the principle and are
not retroactively justified in their rustdoc.

**Follow-up:** Yes — F-OPTS ("Document SDE/FDE/IDE options divergence
from `SolverOptions` in their rustdoc") — recorded under "API",
low-priority, scoped.

### B7. §2.6 (interior mutability — broader scope per Daisy directive)

**Callout:** > "[CC: investigate] Audit foundation traits for
interior-mutability requirements. The `AugmentedSystem` case is known.
Are there others? Specifically: does the recent Jacobian unification
work introduce `RefCell` anywhere in `OdeSystem` consumers?"
**Daisy directive (c):** broader reading — every `pub` trait whose
method takes `&self` and whose impls historically need scratch space.

**Found:**

Interior-mutability sites in workspace `src/` directories
(excluding tests):

1. `numra-autodiff::tape::TapeRef = Rc<RefCell<Tape>>`
   (`tape.rs:55, 60`) — the entire reverse-mode tape is built around
   shared interior mutability. `Var` (`reverse.rs`) holds a `TapeRef`
   so any `&self` operation on a `Var` mutates the shared tape. This
   is the algorithm; not a foundation-trait method requirement
   because `Var` doesn't implement a foundation trait.

2. `numra-ode::sensitivity::AugmentedSystem<S, Sys>`
   (`sensitivity.rs:288`) — six `RefCell<Vec<S>>` (jy_scratch,
   jp_scratch, fd_f0, fd_f1, fd_y_pert, fd_p_pert) plus one `Cell<bool>`
   (flag_check_done, debug-only). Documented in the AugmentedSystem
   rustdoc as a deliberate scratch-buffer design;
   `OdeSystem::rhs(&self, t, y, dydt)` is the consumer API and runs on
   `&self`, so the scratch must be interior-mutable.

3. No other workspace `src/` site uses `RefCell`, `Cell`, `Mutex`,
   `OnceCell`, or `OnceLock`.

**Implication:** §2.6's claim is essentially correct — the
`AugmentedSystem` case is the only foundation-relevant one; autodiff's
tape is internal and not foundation-routed. The Jacobian unification
work (Radau5/BDF route through `OdeSystem::jacobian`) **did not**
introduce any new interior mutability — verified by reading
`numra-ode/src/radau5.rs` and `numra-ode/src/bdf.rs` and finding no
`RefCell` use.

Per the broader-scope directive: no `pub` trait whose method takes
`&self` requires interior mutability beyond the documented case.

**Document revision:** §2.6 stays as written. The "Are there others?"
question gets a no.

**Follow-up:** None.

### B8. §2.7 (no defensive `Send + Sync`) — widely violated by older code

**Callout:** §2.7 references "the recent `ParametricOdeSystem` decision
(drop both bounds, document call-site escalation pattern) is the
precedent" but does not invite an audit of pre-existing traits.

**Found:** Several foundation and quasi-foundation traits carry
defensive `Send + Sync` bounds without documented parallel-consumer
justification:

- `Scalar: ... + Send + Sync + 'static` (`numra-core/src/scalar.rs:58–60`).
  Defensive on the trait; satisfied trivially by `f32`/`f64` so removing
  the bound is non-breaking for users. Not foundationally pinned to
  any actual parallel consumer.
- `Signal<S>: Send + Sync` (`numra-core/src/signal.rs:51`). Defensive
  on the trait; would propagate to every user-defined Signal.
- `EventFunction<S>: Send + Sync` (`numra-ode/src/events.rs:66`).
  Defensive; the `Arc<dyn EventFunction<S>>` field in `SolverOptions`
  (`solver.rs:41`) requires `Send + Sync` for `Arc` to be usable across
  threads, but `Arc` itself doesn't require its `T: Send + Sync` — only
  if the `Arc` is sent across thread boundaries. The bound is
  needlessly strict if no actual parallel consumer exists.
- `NonlinearSystem<S>: Send + Sync` (`numra-nonlinear/src/newton.rs:100`).
  Defensive.
- `SdeSystem<S>: Sync` (`numra-sde/src/system.rs:27`). Defensive
  (only one of the two bounds, asymmetric).

PDE / OCP closure type aliases require `Send + Sync + 'static` on
constructor closures (`numra-pde/src/mol2d.rs:19,84`,
`mol3d.rs:19,84`, `equations2d.rs:58`, `equations3d.rs:60`,
`mol2d_parametric.rs:44,96`, `mol3d_parametric.rs:17,57`;
`numra-ocp/src/param_est.rs:22,104`, `shooting.rs:29-38`,
`collocation.rs:32-41`). These are not trait bounds (the closures live
inside `Box<dyn Fn(...) + Send + Sync>` aliases), but they propagate
the same defensive virality up to user constructors.

**Implication:** §2.7 is correct as a forward principle. The honest
finding is that older code (Scalar, Signal, EventFunction,
NonlinearSystem, SdeSystem, the PDE/OCP closure aliases) does not
follow it. The principle is the new direction; backporting it is a
separate decision.

Per the directive on contract violations: surface, don't suppress.

**Document revision:** §2.7 stays as written. Add one paragraph
acknowledging that the principle applies prospectively; pre-existing
defensive bounds are tracked as follow-ups, not back-patched silently.

**Follow-up:** Yes — F-SENDSYNC ("Audit & remove defensive `Send + Sync`
on foundation traits") — recorded under "API". Per the granularity
directive, one consolidated follow-up listing the five trait sites and
the closure aliases inside.

---

## C. Foundation Spec §3 — Trait spine

### C1. §3.1 `Scalar` — extracted signature, design choices verified

**Callouts:** > "[CC: extract] Pull the actual `Scalar` trait
definition" / > "[CC: investigate] Verify these choices match the
code".

**Found:** `numra-core/src/scalar.rs:42–291`. The trait has supertraits:

```rust
pub trait Scalar:
    Copy + Clone + Debug + Display + PartialOrd
    + Add<Output = Self> + Sub<Output = Self> + Mul<Output = Self>
    + Div<Output = Self> + Neg<Output = Self>
    + AddAssign + SubAssign + MulAssign + DivAssign
    + Sum + Send + Sync + 'static
```

Required associated constants: `ZERO, ONE, TWO, HALF, EPSILON, INFINITY,
NEG_INFINITY, NAN, PI, E, SQRT_2, LN_2`.

Required methods: `from_f64, from_f32, from_i32, from_usize, to_f64,
to_f32, abs, sqrt, cbrt, powi, powf, hypot, sin, cos, tan, asin, acos,
atan, atan2, exp, exp2, exp_m1, ln, log2, log10, ln_1p, sinh, cosh,
tanh, asinh, acosh, atanh, max, min, clamp, copysign, is_finite, is_nan,
is_infinite, is_sign_positive, is_sign_negative, floor, ceil, round,
trunc, fract, gamma_fn, ln_gamma, erf_fn, erfc_fn, mul_add`.

Provided defaults: `sq, sincos, recip, signum`.

Blanket impls: `f64` (lines 297–528) and `f32` (lines 534–765), both
delegating to `libm`. No other impls.

**Verification of stated choices:**
- "Real-only for v1. […] `Complex<f64>` exists as a peer type in
  `numra-fft` but doesn't satisfy the trait." ✓ — confirmed; the trait
  requires `PartialOrd` (Complex doesn't satisfy it) and the default
  `signum` body uses `<` against `ZERO`, which is real-only.
- "`f32` and `f64` are the only blanket impls." ✓.
- "Math operations (`sin`, `exp`, `sqrt`, etc.) are required methods of
  the trait, not provided via `num-traits::Float`." ✓ — both impls
  delegate to `libm` directly. The doc comment at `scalar.rs:14`
  explicitly states "Self-contained (no external trait dependencies
  like num-traits or nalgebra)".

**Document revision:** Replace the [CC: extract] block in §3.1 with the
extracted signature (or a tightened summary thereof — the full trait is
~250 lines and would dominate the document; one option is to enumerate
only the supertraits, the constants, and the method-group headings,
deferring the full per-method list to the rustdoc).

**Follow-up:** None.

### C2. §3.1 `Scalar` — undocumented `Send + Sync + 'static` bounds

**Found:** The trait carries `Send + Sync + 'static` (lines 58–60) with
no rationale in the surrounding doc comment. Per the §2.7 principle,
this is defensive — both `f32` and `f64` are trivially `Send + Sync`
'static, so the bound costs nothing today but propagates virally to
every user type that wants to be `Scalar`.

**Implication:** Either (a) the bound has a rationale (e.g. the trait
is intended to be storeable in `Arc`s for parallel consumers in
`SolverOptions::events` or similar) and that rationale should be
documented, or (b) it's defensive and should be removed in alignment
with §2.7. Decision is out of scope for this verification pass; finding
recorded for follow-up.

**Document revision:** None to §3.1; covered by §2.7's principle and
the F-SENDSYNC follow-up.

**Follow-up:** Captured in F-SENDSYNC.

### C3. §3.2 `Vector` and `Matrix` — extracted; only one impl each in the workspace

**Callouts:** [CC: extract] / [CC: investigate].

**Found:**
- `Vector<S: Scalar>: Clone + Sized` (`numra-core/src/vector.rs:40`).
  Required methods: `zeros, fill, from_slice, len, get, set, get_mut,
  as_slice, as_mut_slice, copy_from, axpy, axpby, dot, scale,
  norm_inf, norm1, abs_inplace, max_elementwise, min_elementwise, sum,
  max_element, min_element, map_inplace`. Provided defaults: `is_empty,
  norm2, weighted_rms_norm`. Single workspace impl: `Vec<S>`
  (`vector.rs:151`). The doc comment at `vector.rs:6` claims "faer
  vectors (when using numra-linalg)" but there is no `Vector` impl on a
  faer type anywhere in the workspace. The doc comment is aspirational
  and overstates what's shipped.
- `Matrix<S: Scalar>: Clone + Sized` (`numra-linalg/src/matrix.rs:15`).
  Required methods: `zeros, identity, nrows, ncols, get, set,
  fill_zero, scale, mul_vec, add_scaled, solve`. Provided default:
  `is_square`. Single workspace impl: `DenseMatrix<S>`
  (`matrix.rs:154`) — but with the additional bound
  `S: Scalar + SimpleEntity + Conjugate<Canonical = S> + ComplexField`,
  which means even though `Matrix` looks pure-`Scalar`-implementable
  from outside, its workspace concrete type leaks faer's universe.
- The `numra-linalg::SparseMatrix<S>` (`sparse.rs`) is a separate
  concrete type that **does not** implement `Matrix<S>`. So the trait
  abstraction does not actually cover sparse, contradicting any
  reading where "the foundation accepts sparse uniformly".

**Verification of audit note:** Audit said "sparse direct solvers
convert to dense internally (sparse.rs:148–149)". Confirmed:
`SparseLU<S>` (`sparse.rs:147`) wraps a `LUFactorization<S>` and the
doc comment says "Converts to dense internally (Phase 1 pragmatic
approach). A future phase will use native faer sparse solvers for
O(nnz) performance."

**Implication:**
- The `Vector` trait is unused as an abstraction in practice — every
  workspace caller uses `Vec<S>` directly. The trait is a placeholder
  for a future faer-vector or fixed-size-array impl.
- The `Matrix` trait is similarly used only via `DenseMatrix<S>` today.
  Sparse is a separate concrete type, parallel to the trait.
- The §3.2 design claim "Foundation is dense + sparse (CSC). No dense
  column-major separately from dense row-major in the foundation;
  layout is a property of the concrete type." is **partially false**:
  sparse is not a `Matrix` impl; layout is a property of separate
  concrete types, but `Matrix` only covers dense.
- The §3.2 deferred decision on banded / GPU is forward-looking and
  unaffected.

**Document revision:** §3.2 needs revision to reflect:
1. The actual extracted trait signatures (separate Vector and Matrix
   bullets per A2).
2. A clear statement that today `Vec<S>` is the only `Vector` impl and
   `DenseMatrix<S>` is the only `Matrix` impl.
3. Correction of the "Foundation is dense + sparse" sentence: sparse is
   a separate concrete type (`SparseMatrix<S>`) outside the trait
   abstraction. This is a real foundation question — should sparse
   join the `Matrix` trait, or stay separate? — that belongs in §7
   (open questions), not as an asserted design choice.
4. Note the `DenseMatrix<S>` faer-bound leak (`SimpleEntity +
   Conjugate<Canonical = S> + ComplexField`) and that this leaks the
   faer universe into anyone who wants to instantiate the workspace's
   dense-matrix concrete type.

**Follow-up:** Optional — F-MATRIX-SHAPE ("Decide whether `SparseMatrix`
joins `Matrix` trait or stays separate"). Recorded under the "API"
section as a design question, status `scoped, not started`.

### C4. §3.3 `OdeSystem` — extracted; FD step formula draft is wrong

**Callouts:** [CC: extract] / [CC: investigate] including: > "what's
the FD step size formula in the trait default — is it `sqrt(eps_mach) *
(1 + |y|)` (as the recent Batch 2 work specified), or did it land as
something else?"

**Found:** `numra-ode/src/problem.rs:15–82`.

```text
pub trait OdeSystem<S: Scalar> {
    fn dim(&self) -> usize;
    fn rhs(&self, t: S, y: &[S], dydt: &mut [S]);
    fn jacobian(&self, t: S, y: &[S], jac: &mut [S]) { /* default FD */ }
    fn is_autonomous(&self) -> bool { false }
    fn has_mass_matrix(&self) -> bool { false }
    fn mass_matrix(&self, mass: &mut [S]) { /* default identity */ }
    fn is_singular_mass(&self) -> bool { false }
    fn algebraic_indices(&self) -> Vec<usize> { Vec::new() }
}
```

No `Send + Sync` bound on the trait itself. ✓
No `dense_output_required` method. (Draft hedged "if present"; not
present.)

**FD step formula in `jacobian` default body** (`problem.rs:24–44`):

```text
let eps = S::from_f64(1e-8);
let mut y_pert = y.to_vec();
let mut f0 = vec![S::ZERO; n];
let mut f1 = vec![S::ZERO; n];
self.rhs(t, y, &mut f0);
for j in 0..n {
    let yj_save = y_pert[j];
    let h = eps * (S::ONE + yj_save.abs());
    ...
}
```

So the formula is `eps = 1e-8` (concrete `f64` scaled into `S`),
`h = eps * (1 + |y_j|)`. That is **not** `sqrt(eps_mach) * (1 + |y|)`.
It is a hardcoded `1e-8` step size which:
- Approximates `sqrt(f64::EPSILON) ≈ 1.49e-8` in `f64` (off by ~50%).
- Falls **below** `f32::EPSILON ≈ 1.19e-7` in `f32`, making the FD
  step useless on `f32` (the perturbation gets quantised to zero or
  loses all signal).

The CHANGELOG entry under "Changed" makes this explicit:
> "Systems […] fall through to the canonical forward-FD default in
> numra-ode/src/problem.rs (eps = 1e-8, step = `eps * (1 + |y_j|)`,
> row-major dense output)."

The same CHANGELOG note explains the unification: prior to
`8c170d7`/`7bc3559` (2026-05-07), Radau5 inlined `eps * max(1, |y_j|)`
and BDF used the trait default — both have been unified onto the
trait default's `eps * (1 + |y_j|)` form.

By contrast, `ParametricOdeSystem::jacobian_y` and `_p` defaults
(`sensitivity.rs:182, 210`) **do** use `S::EPSILON.sqrt()`:

```text
let h_factor = S::EPSILON.sqrt();
let h = h_factor * (S::ONE + yj.abs());
```

So the FD step is **inconsistent across the two foundation traits**.
`OdeSystem` uses `1e-8`, `ParametricOdeSystem` uses `sqrt(EPSILON)`. The
draft §3.3 conflated them.

**`mass_matrix` shape:** `mass_matrix(&self, mass: &mut [S])` (in-place,
row-major; default = identity). NOT `Option<…>`. The DAE-ness flag is
`has_mass_matrix() -> bool`, not the option. Draft §3.3's hedged
phrasing "returns `Option<…>` or similar" is wrong about the shape — but
the design itself is reasonable: the in-place shape avoids allocation in
the hot path, consistent with the `rhs(&self, t, y, dydt)` style.

**Document revision:**
1. Replace the `[CC: extract]` block with the extracted signature.
2. Correct §3.3's design-choice bullet on FD step size: state that
   `OdeSystem::jacobian` default uses `eps = 1e-8` (concrete `f64`,
   scaled into `S`) and `h = eps * (1 + |y_j|)`, while
   `ParametricOdeSystem::jacobian_y/_p` defaults use
   `h = sqrt(S::EPSILON) * (1 + |y_j|)`. Both are documented as
   deliberate; they differ. Surface that the two formulas don't agree;
   surface that the `1e-8` `OdeSystem` formula is below `f32::EPSILON`
   and so makes FD on `f32` quietly useless.
3. Correct §3.3's design-choice bullet on `mass_matrix`: the actual
   shape is `mass_matrix(&self, mass: &mut [S])` writing row-major;
   DAE-ness is signalled via `has_mass_matrix() -> bool` and
   `is_singular_mass() -> bool` plus `algebraic_indices() -> Vec<usize>`.
   This is a deliberate flag-and-fill-buffer design, not an `Option`.
4. Drop the "if present" qualifier on `dense_output_required` — not
   present; dense output is opt-in via `SolverOptions.dense_output`,
   matching the existing draft claim.

**Follow-up:** Yes — F-FD-STEP ("Reconcile FD step formula between
`OdeSystem` and `ParametricOdeSystem` defaults"). Two design decisions
are needed: should both use `sqrt(EPSILON)`? Should the formula be made
generic-precision-aware? `f32` users currently get a silently-broken
`OdeSystem::jacobian` FD path. Recorded under "Solvers".

### C5. §3.4 `Solver` — extracted; "unit-type" claim is partly false

**Callout:** [CC: extract] / design-choice claim "Solver is a unit-type
trait — DoPri5, Radau5, etc. are zero-size types".

**Found:** `numra-ode/src/solver.rs:291–300`:

```text
pub trait Solver<S: Scalar> {
    fn solve<Sys: OdeSystem<S>>(
        problem: &Sys,
        t0: S,
        tf: S,
        y0: &[S],
        options: &SolverOptions<S>,
    ) -> Result<SolverResult<S>, SolverError>;
}
```

Static method (no `&self`). So solver state must come from
`SolverOptions` or be encoded in the impl, not the implementor's
fields.

Implementor types:
- `DoPri5;` (`dopri5.rs:53`) — unit struct ✓.
- `Tsit5;` (`tsit5.rs:27`) — unit struct ✓.
- `Vern6;`, `Vern7;`, `Vern8;` (`verner.rs:30, 430, 628`) — unit ✓.
- `Radau5;` (`radau5.rs:83`) — unit ✓.
- `Esdirk32, Esdirk43, Esdirk54;` (`esdirk.rs`) — unit ✓.
- `Bdf { max_order: usize, min_order: usize }` (`bdf.rs:83`) —
  **NOT unit; has fields**. Builder methods `Bdf::new()`,
  `Bdf::with_max_order(...)`, `Bdf::fixed_order(...)`. But because
  `Solver::solve` is static, the fields are unreachable from inside
  `solve()` — they're only meaningful if a caller instantiates a
  `Bdf` value and... can't pass it to anything (the trait method is
  static).
- `Auto { ... }` (`auto.rs:119`) — also not unit; has fields. Same
  unreachability story.

This is a **latent bug or design tension**: `Bdf` and `Auto` carry state
that has no path to `solve()` because the trait method is static. Either
(a) the fields are dead code, or (b) `Bdf::with_max_order` has been
sketched but not wired through the `Solver` trait, or (c) the builders
are for a future reshape of the trait. The §3.4 claim "State is in
`SolverOptions`, not in the solver type" is true at the trait level but
contradicted by Bdf/Auto carrying state that the trait can't see.

Solver-impl-bound asymmetry (separately worth noting): explicit RK
solvers are `impl<S: Scalar> Solver<S>`; implicit/stiff solvers
(`Radau5, Esdirk32/43/54, Bdf, Auto`) carry the additional faer bound
`S: Scalar + SimpleEntity + Conjugate<Canonical = S> + ComplexField`.
External `Scalar` impls (a hypothetical `BigDecimal` etc.) can use the
explicit solvers but not the implicit ones.

**Document revision:** §3.4's "State is in `SolverOptions`, not in the
solver type" claim should be qualified: it's true at the trait level
because `Solver::solve` is static; in implementor practice, `Bdf` and
`Auto` carry state that the trait method can't read, which is a known
quirk worth surfacing as either a follow-up to clean up or a rationale
to add. Add the implicit-solver faer-bound asymmetry as a note in §3.4
or in §3.2 (since the leak is architectural, not specific to one
solver).

**Follow-up:** Yes — F-SOLVER-FIELDS ("Clarify or remove `Bdf::max_order`
/ `Auto::*` fields that the static `Solver::solve` cannot read").
Recorded under "Solvers". Possibly trivial if the fields are wired
through some adapter I missed; either way, surface it.

### C6. §3.5 `Signal` — extracted; trait, not enum

**Callouts:** [CC: extract] / "[CC: investigate] Verify whether `Signal`
is a trait or an enum."

**Found:** `numra-core/src/signal.rs:51`. The trait is generic
over `S: Scalar` with bound `Send + Sync`. Required method: an
evaluator taking `&self, t: S` returning `S`. Provided default: a
derivative method using central finite differences with `h = 1e-8`.

Variants are concrete structs implementing the trait: `Harmonic<S>`,
`Step<S>`, `Ramp<S>`, `Pulse<S>`, `Chirp<S>`, `Tabulated<S>`, plus
composites (`Piecewise`, `Sum`, `Product`) and an opt-in `FromFile`
behind the `std` feature. Each variant is its own struct file-section;
adding a new variant is a non-breaking addition (just publish a new
struct that impls the trait). External users can also impl the trait
for their own types. So the composability story is "open trait
hierarchy" — both extension paths work.

The trait carries the defensive `Send + Sync` bound (per B8); the
default derivative uses central FD with hard-coded `h = 1e-8`, same
`f32`-vs-`f64` precision hazard as in C4 (F-FD-STEP).

**Document revision:** Replace the [CC: extract] block with the
extracted signature, drop the "[CC: investigate] trait or enum" callout
(answered: trait), document the open-hierarchy composability story in
the §3.5 design-choices bullet.

**Follow-up:** None unique — `Signal`'s `Send + Sync` is folded into
F-SENDSYNC; the FD step into F-FD-STEP.

### C7. §3.6 `Result types` — surfaced (see B4)

**Found:** Covered by B4. All foundational result types are inspectable
with `pub` fields plus accessors.

**Document revision:** §3.6 should inline a brief statement noting which
result types were audited (`SolverResult`, `SensitivityResult`,
`OptimResult`, `ParameterSensitivityResult`) and that all met the
`pub` fields + accessor bar.

### C8. §3.7 `SolverOptions` — extracted

**Callout:** [CC: extract].

**Found:** `numra-ode/src/solver.rs:20–152`. Fields (all `pub`):
`rtol, atol, h0, h_max, h_min, max_steps, t_eval, dense_output,
events`. Plus `events: Vec<Arc<dyn EventFunction<S>>>` (Arc enables
`Clone`). Builder methods: `rtol, atol, h0, h_max, t_eval, dense,
max_steps, h_min, event`. Defaults: `rtol = 1e-6, atol = 1e-9, h_max =
INFINITY, h_min = EPSILON * 100, max_steps = 100_000, dense_output =
false`. Generic `<S: Scalar>`.

The generic-over-`S` bound is meaningful because tolerance values, step
sizes, and `t_eval` are scaled into `S`. ✓

**Document revision:** Replace the [CC: extract] block with this
summary.

**Follow-up:** None.

---

## D. Foundation Spec §4 — Composability primitives

### D1. §4.1 Problem-class to ODE pipeline — uniformly satisfied

**Callout:** > "[CC: verify] Confirm this pattern is implemented
uniformly today. Surface deviations."

**Found:**
- `numra-pde::MOLSystem`, `MOLSystem2D`, `MOLSystem3D`,
  `ParametricMOLSystem2D/3D` all impl `OdeSystem<S>` (or
  `ParametricOdeSystem<S>`). ✓
- `numra-spde::SpdeSystem` reduces via MOL to an SDE problem; the SDE
  side is `numra-sde::SdeSystem`, separate trait. So SPDE composes onto
  SDE, not directly onto `OdeSystem`. ✓ as documented in audit.
- `numra-dde` Method-of-Steps wraps DoPri5 + Hermite history; the DDE
  problem itself is `numra-dde::DdeSystem` (`dde/system.rs:15`), a
  separate trait that the DDE solver consumes — it does not produce an
  `OdeSystem` impl. So DDE's composition with ODE is via `numra-dde`'s
  internal use of `numra-ode::DoPri5`, not via implementing
  `OdeSystem`.
- `numra-ide::IdeSystem` (Volterra type-2 only per audit) — separate
  trait; the IDE solver consumes it directly without going via
  `OdeSystem`.
- `numra-fde::FdeSystem` — separate trait; same pattern.
- `numra-ocp` — shooting/collocation transcribe to NLP; each transcribed
  segment uses `OdeSolverChoice::DoPri5` (private internal enum, not the
  generic `Solver` trait — per audit Tier 1.4).

**Implication:** §4.1 as written ("Every problem class that reduces to
an IVP […] provides an `OdeSystem` impl") is **partially true**:
- True for PDE, SPDE (via SDE), and the augmented sensitivity path.
- False (or weakly true via internal use rather than a public impl)
  for DDE, IDE, FDE, OCP. These domains have their own problem traits
  and own solvers; they consume `numra-ode` internally but do not
  expose their problem as an `OdeSystem`.

This is an honest finding about the workspace's actual shape. The
strong reading "composability via `OdeSystem` impl uniformly" is
aspirational; the actual reading is "PDE/SPDE/sensitivity expose
themselves as `OdeSystem`; DDE/IDE/FDE/OCP have their own problem
traits and reuse the ODE solver internally".

**Document revision:** §4.1 should be revised to state the actual
pattern: **two coexisting reductions** —
1. Problem classes that expose themselves as `OdeSystem` so any ODE
   solver consumes them: PDE (MOL), augmented sensitivity, parametric
   MOL.
2. Problem classes that have their own problem trait and reuse a
   specific ODE solver internally: DDE (Method-of-Steps + DoPri5), OCP
   (shooting/collocation + DoPri5).
3. Problem classes that have their own problem trait and do not reuse
   ODE solvers (their solvers are domain-specific): SDE, FDE, IDE,
   SPDE (the SDE arm).

This isn't a defect — different problem classes need different
abstractions — but the §4.1 framing should reflect it.

**Follow-up:** None — documentation refinement only.

### D2. §4.2 Solver to result-to-input flow — confirmed by interop tests

**Found:** Confirmed. See composability graph (D5).

### D3. §4.3 Sensitivity composition — confirmed; design verified

**Found:** Confirmed against `numra-ode/src/sensitivity.rs`. The
`ParametricOdeSystem` trait is foundational (per A1);
`AugmentedSystem<S, Sys: ParametricOdeSystem<S>>` implements
`OdeSystem<S>` (`sensitivity.rs:479`), so any `Solver` consumes it.
`SensitivityResult<S>` has the inspectable interface (B4). The
column-major flattening + asymmetric `J_y/J_p` convention + flag
pattern are documented in the trait rustdoc (`sensitivity.rs:24–117,
241–280`).

The §4.3 claim "Adjoint sensitivity, when added, follows the same
pattern" is a design intent; per audit Tier 1.4, adjoint exists today
only as `numra-ocp`-private and `DoPri5`-only. Promoting it would
follow the pattern; it isn't promoted yet. The §4.3 statement is fine
as a forward principle.

**Document revision:** None.

### D4. §4.4 Error composition — `?`-propagation untested in interop

**Callout:** > "[CC: verify] Audit the actual `?` propagation across
the six interop workflows in `numra/tests/interop_workflows.rs`. Are
there any places where manual `.map_err` is needed?"

**Found:** No manual `.map_err` exists anywhere in
`numra/tests/interop_workflows.rs`. **But** every test sidesteps the
question by using `.unwrap()` exclusively — the test functions return
`()`, not `NumraResult<()>`. So the cross-crate `?`-propagation story
is **not exercised by any interop test**.

This is the worst kind of evidence about contract item §3 ("Errors
compose into the workspace error type"): it's not that `?` works
without manual `.map_err` (which would be confirming evidence); it's
that the test author chose to `.unwrap()` everywhere, sidestepping the
question entirely. Combined with B5 (multiple crates lack
`From<...> for NumraError`), the honest conclusion is that
cross-crate `?` doesn't actually work for SolverError-to-anything,
OptimError-to-anything, etc.

**Document revision:** §4.4 should be revised from "every fallible
operation returns a workspace-compatible error type" (asserts the
property is true) to "the workspace-wide `NumraError` exists; the
discipline is partially applied — coverage gaps are tracked as named
follow-ups; no interop test exercises cross-crate `?` today (every
existing interop test uses `.unwrap()`)".

**Follow-up:** Folded into F-ERR (B5) plus a follow-on F-INTEROP-Q
("Add interop test that exercises `?`-propagation across at least two
crate boundaries"). Per the granularity directive, recorded as
one follow-up under "API" listing both pieces.

### D5. §4 Composability graph — mapped from interop tests + workspace inspection

The composability edges currently realised in the workspace, in the
`X output → Y input` shape of contract §5:

**From `numra/tests/interop_workflows.rs`:**

```text
ODE::SolverResult              → Interp::CubicSpline       → Integrate::quad
ODE::SolverResult.component()  → FFT::psd                  → Signal::butter+filtfilt → Signal::find_peaks
OCP::ParamEstProblem           → ODE::solve_with_uncertainty
Autodiff::Dual + gradient_closure → Optim::OptimProblem.gradient → Fit::curve_fit_with_jacobian
PDE::MOLSystem                 → ODE::DoPri5::solve        → Stats::{mean, variance, std_dev, median}
[manual sampling]              → ODE::OdeProblem ensemble  → Stats::{mean, std_dev}
```

**Cross-crate composition outside the interop suite (from `Cargo.toml`
deps and `pub use` walking):**

```text
ParametricOdeSystem            → AugmentedSystem (impls OdeSystem) → any Solver
ParametricMOLSystem2D/3D       → ParametricOdeSystem → AugmentedSystem → any Solver
DDE Method-of-Steps            → ODE::DoPri5 (internal)
SPDE                           → PDE (MOL spatial) + SDE (time)
OCP shooting / collocation     → ODE (DoPri5-only internal) + Optim
Stats distributions            → Special functions (gamma, erf, etc.)
Signal                         → FFT (filter design, hilbert, psd path)
Fit::curve_fit                 → Optim::lm_minimize
Linalg                         → faer (external)
Numra-core::compute_sensitivities  → (no downstream consumer; standalone)
```

**Edges that the contract §3.1 worked example claims but the code does
not realise** (see H2 below): `SensitivityResult → LM::fit` is **not**
a workspace edge; `SensitivityResult → solve_trajectory` is **not** a
workspace edge. The only workspace consumer of `SensitivityResult` (as
input or via re-export) is `numra-ocp::forward_sensitivity` (which
returns it).

**Document revision:** Replace the §4 narrative-only enumeration with
the explicit graph above, presented as evidence that the composability
primitives are real.

**Follow-up:** None for the graph itself. The
`SensitivityResult → downstream` gap surfaces a real composability
question worth tracking — see follow-up F-SENS-DOWNSTREAM under "API"
("Decide downstream consumers for `SensitivityResult` beyond
`numra-ocp` re-export").

### D6. §4.5 Generic-over-Scalar at every layer — broken by interop tests

**Callout:** > "[CC: verify] Audit the interop tests for places where
genericity is broken (a return type ends up `f64` even though the input
was generic). Surface findings."

**Found:** **Every interop test is monomorphised at `f64`.** Workflow 1
binds `OdeProblem::new(|_t, y: &[f64], dydt: &mut [f64]| ...)`.
Workflow 4 binds `Dual<f64>`. Workflow 5 binds `Vec<f64>`. Etc. The
tests demonstrate **functional** composition, not **generic** composition
preservation. So §4.5's principle is untested.

This isn't necessarily a defect of the production code — the
production code's APIs are largely generic over `S: Scalar` — but it is
a defect of the test coverage. A user trying to compose at `f32` or at
some external `Scalar` impl has no test telling them whether their
pipeline holds together.

**Document revision:** §4.5 stays as a forward principle. Add a
parenthetical noting that no interop test exercises the principle at a
non-`f64` type today.

**Follow-up:** Folded into F-INTEROP-Q (D4).

---

## E. Foundation Spec §5 — Composability Contract testable form — see §J

The §5 audit "[CC: verify against existing capabilities]" is treated
separately as the meta-finding J1, because the violation density is
high enough that a strategic call is needed.

---

## F. Foundation Spec §6 — Decision log

### F1. Decision log entries from CHANGELOG + git log

**Callout:** > "[CC: populate] Walk the CHANGELOG and recent PRs to
extract foundation-level decisions."

Per directive (b): commit dates when CHANGELOG only has release
versions, with parenthetical noting the source.

**Proposed §6 entries (chronological, append-only):**

1. **2026-05-05 (commit `9d93407`; release: `0.1.0` Unreleased)** —
   `SolverResult.dense_output: Option<DenseOutput<S>>` field added.
   Fixes a silent drop of the dense interpolant when
   `SolverOptions::dense()` was set. Foundation-affecting because the
   shape of the workspace's primary ODE result type changed. No
   follow-ups created.

2. **2026-05-05 (commit `7b6b816`; release: `0.1.0` Unreleased)** —
   Radau5 step controller rewritten against Hairer–Wanner ODE II §IV.8
   and SciPy's `Radau` reference. Internal algorithmic change; not
   strictly foundation-affecting at the trait level, but the controller
   change exposed a downstream behaviour shift in step counts that the
   regression suite absorbed. Recorded for completeness because the
   subsequent Jacobian unification work depended on the Radau5 rewrite
   being stable.

3. **2026-05-06 (commit `b420335`; release: `0.1.0` Unreleased)** —
   Sensitivity API consolidation: introduced `ParametricOdeSystem`
   trait (renamed from `SensitivityEquations`) with `params()` and
   `rhs_with_params(t, y, p, dydt)`; FD-default `jacobian_y` and
   `jacobian_p`; column-major sensitivity flattening matching CVODES
   indexing; asymmetric `J_y` (row-major) / `J_p` (column-major)
   layout; no `Send + Sync` defensive bounds — call-site escalation
   pattern documented; `has_analytical_jacobian_*` flag pattern with
   debug-build consistency-check safety net. Renamed
   `numra-core::Sensitivity → ParameterSensitivity` and
   `SensitivityResult → ParameterSensitivityResult` to disambiguate
   the two distinct sensitivity concepts (parameter-uncertainty vs ODE
   forward sensitivity). Follow-ups created: block-diagonal LU /
   JVP / AD `ParametricOdeSystem` impl / staggered correction /
   separate sensitivity tolerances / flag ergonomics.

4. **2026-05-06 (commit `8fa9729`)** — Public entry points
   `solve_forward_sensitivity` and `solve_forward_sensitivity_with`
   added to `numra-ode`. Consumer of (3).

5. **2026-05-06 (commit `ba09a03`; release: `0.1.0` Unreleased)** —
   BDF rewritten as a Rust port of SciPy's `BDF` (Shampine–Reichelt
   NDF / `ode15s` algorithm). Three independent algorithmic bugs fixed
   (Newton convergence-test target; LTE estimate using cumulative
   Newton correction; modified-divided-differences rescaling under
   variable steps). The previous "silent wrong answer" mode is pinned
   out by `test_bdf_regression_silent_wrong_answer`. Foundation-
   affecting because BDF is a foundational ODE solver and its
   trait-level Jacobian dispatch was settled in the same window.

6. **2026-05-06 (commits `6f26aaf` + `34d81b6`)** —
   `numra-ode::solve_trajectory` and `numra-ocp::forward_sensitivity`
   reimplemented atop the canonical `solve_forward_sensitivity_with`
   primitive. Eliminated parallel implementations; `numra-ocp`
   sensitivity layout changed from row-major-over-states to
   column-major-over-parameters; `numra_ocp::SensitivityResult` is now
   a re-export of `numra_ode::SensitivityResult`.

7. **2026-05-07 (commits `8c170d7` + `7bc3559`; release: `0.1.0`
   Unreleased)** — Jacobian unification: `Radau5` and `Bdf` now route
   their Jacobian computation through `OdeSystem::jacobian` instead of
   solver-inlined finite-difference paths. Systems that override the
   trait method (e.g. `MOLSystem2D`, `MOLSystem3D`) get an analytical
   Jacobian for free; systems that don't fall through to the trait
   default (`eps = 1e-8`, step `eps * (1 + |y_j|)`, row-major dense
   output). Resolves prior FD-step drift between Radau5's inlined
   `eps * max(1, |y_j|)` and the canonical `(1 + |y_j|)` form.
   Behavioural impact in last 2-3 digits at states with `|y_j| ≈ 1`;
   full test suite passes unchanged.

8. **2026-05-07 (commits `f7499ec` + `e592c6e` + `a91a5a6`;
   release: `0.1.0` Unreleased)** — 3D MOL parity work.
   `MOLSystem3D` wrapper (`mol3d.rs`) mirroring `MOLSystem2D`;
   `Operator3DCoefficients` + general 7-point operator assembly;
   convenience builders `HeatEquation3D::build`,
   `AdvectionDiffusion3D::build`, `ReactionDiffusion3D::{build,
   fisher}`. `MOLSystem2D` and `MOLSystem3D` both override
   `OdeSystem::jacobian` with CSC-to-row-major-dense copy of the
   spatial operator + diagonal-FD reaction term. Foundation-affecting
   because it pinned the analytical-Jacobian override pattern that
   downstream parametric MOL inherits.

9. **2026-05-08 (commit `da0674b`; release: `0.1.0` Unreleased)** —
   `ParametricMOLSystem2D` and `ParametricMOLSystem3D` ship in
   `numra-pde`. Wraps the heat-equation MOL discretisation as
   `ParametricOdeSystem`. Parameter layout `[α, reaction_p_0, ...]`;
   analytical state Jacobian (`α · L0` + diagonal reaction FD);
   analytical α-column of `J_p` (`L0·y + bc_rhs_0`); all four
   `has_analytical_jacobian_*` flag overrides set. v1 scope is
   alpha-on-Laplacian only; full operator parametrisation tracked as
   the named follow-up "Full operator parametrisation in MOL".

10. **2026-05-10 (commit `ec5f355`)** — `numra-ode`:
    `SolverOptions::t_eval` is now honored by every solver. Previously
    it was parsed, stored, cloned, and ignored. Now each solver emits
    `(t, y)` pairs at exactly the requested times via Hermite cubic
    interpolation between accepted step endpoints (closes GH#1).
    Pinned by `numra-ode/tests/t_eval_regression.rs`.
    Foundation-affecting because `SolverOptions` is the workspace's
    shared configuration vocabulary (§2.5) and `t_eval` was a public
    field whose contract was silently broken.

**Document revision:** Replace the [CC: populate] block in §6 with this
list (or a tightened version).

**Follow-up:** None.

---

## G. Foundation Spec §7 — Open questions

### G1. §7.6 (workspace error type) gets a verified answer

The verification pass resolves §7.6: a workspace error type exists; the
question is whether it has sufficient cross-crate `From` coverage. It
does not (per B5). The open question converts to the F-ERR follow-up
and §7.6 can be removed from the open-questions list.

**Document revision:** Remove §7.6 from open questions; replace with a
one-line note "Workspace error type exists (`numra_core::NumraError`);
coverage is the work item — tracked in `internal-followups.md`."

The remaining open questions (§7.1–§7.5, §7.7) are deferred design
decisions; the verification pass leaves them as-is per the
"don't decide open questions" directive.

**Follow-up:** None unique — F-ERR covers the implementation.

---

## H. Composability Contract — rule-by-rule and worked-example findings

### H1. Contract §6 (concrete-`f64`) cross-references — folded with B1

**Callout:** > "[CC: verify] Audit the workspace and surface every
concrete-`f64` in public capability surfaces."

**Found:** Same audit as B1; results recorded there. Contract §6 is
correct as written.

**Document revision:** Drop the `[CC: verify]` block; add a sentence
referencing the Foundation Spec §2.1 audit and the F-LEAK follow-up.

### H2. Worked example §3.1 (forward sensitivity API) — claim 5 is **factually wrong** in current code

**Directive (e):** verify each of the seven ✓ in the worked example.

**Verification of the seven:**

1. ✓ "Publicly implementable trait." `ParametricOdeSystem`
   (`sensitivity.rs:152`) is `pub`, no `pub(crate)` leakage in its
   method signatures or default bodies (per B2), not sealed (per B3).
   External impl is demonstrated by every test in
   `sensitivity_regression.rs` and by the Robertson example
   (`numra/examples/robertson_sensitivity.rs:53`). **Confirmed.**

2. ✓ "Inspectable result." `SensitivityResult<S>` exposes `t, y,
   sensitivity, n_states, n_params, stats, success, message` as `pub`
   fields, plus the documented column-major layout and the accessors
   `y_at, sensitivity_at, sensitivity_for_param, dyi_dpj, final_state,
   final_sensitivity, normalized_sensitivity_at`. **Confirmed.**

3. ✓ "Workspace error." The claim says "`SolverError` propagates to
   `numra::Error` via existing impl (verified by interop tests)". Per
   B5: `SolverError` does **not** have a `From<SolverError> for
   NumraError` impl. The interop tests use `.unwrap()` and so do not
   exercise propagation. **Claim 3 is false in current code.**

4. ✓ "Foundation-typed input." `solve_forward_sensitivity::<Sol, S, Sys>`
   takes `&Sys: ParametricOdeSystem`, `t0: S`, `tf: S`, `y0: &[S]`,
   `&SolverOptions<S>`. All foundation types. **Confirmed.**

5. ✗ **"Foundation-flowable output."** Claim says: "`SensitivityResult`
   feeds `LevenbergMarquardt::fit` (`numra-optim`) and
   `numra-ode::uncertainty::solve_trajectory` (the trajectory-uncertainty
   consumer). Two outgoing edges." **Both halves are false.**

   - `LevenbergMarquardt::fit` does not exist as an API. The closest
     workspace items are `numra-optim::lm_minimize(residual, jacobian,
     x0, opts)` (`numra-optim/src/levenberg_marquardt.rs:53`) which
     takes closure-shaped residual+jacobian, not a `SensitivityResult`,
     and `numra-fit::curve_fit{,_weighted,_with_jacobian}`
     (`numra-fit/src/curve_fit.rs:31, 128, 247`) which take data points
     and a model closure, not a `SensitivityResult`.
   - `numra-ode::uncertainty::solve_trajectory(model, y0, t0, tf,
     params, options)` (`numra-ode/src/uncertainty.rs:169`) takes a
     model closure and parameter list, not a `SensitivityResult`.
     `solve_trajectory` is itself **implemented** atop
     `solve_forward_sensitivity_with` (per CHANGELOG and the
     2026-05-06 commit `6f26aaf`) — that is the inverse direction:
     `solve_trajectory` *produces* uncertainty after computing
     sensitivities internally; it does not *consume* a
     `SensitivityResult`.

   The actual workspace consumer of `SensitivityResult` is
   `numra-ocp::forward_sensitivity`
   (`numra-ocp/src/sensitivity.rs:73, 100, 160`), which **returns** a
   `SensitivityResult` (re-exported from `numra-ode`). So
   `SensitivityResult` is produced by two crates (numra-ode and the
   numra-ocp re-export wrapper) and consumed by **zero** workspace
   crates. The worked example's claim of "two outgoing edges" is the
   reverse of the truth: there are zero outgoing edges.

   This is a real composability gap, not just a documentation error.
   `SensitivityResult` ships with no downstream consumer; the user
   must build any downstream pipeline themselves. The contract item §5
   ("at least one other capability can consume without an adapter the
   user must write") is **failed** for `SensitivityResult` as a result
   type.

6. ✓ "Generic over Scalar." `solve_forward_sensitivity::<Sol, S, Sys>`
   is generic over `S: Scalar` (`sensitivity.rs:840`). **Confirmed.**

7. ✓ "Interop test." The interop test
   `workflow_param_est_sensitivity_uncertainty`
   (`interop_workflows.rs:142`) exists and runs as part of the
   workspace test suite. But it does **not actually exercise
   `SensitivityResult` composition** — it uses `ParamEstProblem` (which
   internally uses sensitivity) and `solve_with_uncertainty` (which
   internally computes sensitivities). The `SensitivityResult` type
   itself is never named in the test. The test demonstrates that the
   sensitivity capability *exists* and that its products can be reached
   via higher-level APIs; it does not demonstrate that
   `SensitivityResult` itself composes with anything.

   So the interop test exists (✓ per the literal contract item 7 wording)
   but the composability it claims to verify — `SensitivityResult →
   downstream` — is not what the test exercises. **Half-confirmed.**

**Implication:** Worked example claim 5 is wrong as written; claim 3
is false in current code; claim 7 is technically present but
substantively misleading.

This is a **substantial composability finding**: the recently-shipped,
much-celebrated forward-sensitivity API ships with **no downstream
consumer of its primary result type**. The work to add downstream
consumers is the genuine completion of the sensitivity capability.

**Document revision:** Rewrite §3.1 of the contract worked example to
state the actual situation:
- ✓ for items 1, 2, 4, 6.
- ✗ for item 5 with the explanation: "`SensitivityResult` ships with
  no in-workspace downstream consumer today; this is a tracked
  composability gap (F-SENS-DOWNSTREAM) and the worked example is
  preserved here as the canonical case where a capability satisfies
  most of the contract but fails item 5".
- ◐ for item 3 (claim is false because `From<SolverError> for
  NumraError` doesn't exist; covered by F-ERR).
- ◐ for item 7 (test exists but the named composability isn't actually
  exercised; covered by F-INTEROP-Q).

This makes §3.1 a worked example of *what surfacing contract
violations looks like*, which is more pedagogically valuable than a
celebratory "look, all seven ticked".

**Follow-up:** F-SENS-DOWNSTREAM ("Decide downstream consumers for
`SensitivityResult` beyond `numra-ocp` re-export") — under "API",
status `scoped, not started`.

### H3. Worked example §3.2 (failing capability) — fine as written

**Found:** Hypothetical RealtimeTraceVisualizer example reads as
intended; no code to verify against.

**Document revision:** None.

### H4. Worked example §3.3 (trivially-satisfied capability) — fine

**Found:** "A new ODE solver" example reads as intended; foundationally
true given the existing `Solver` trait infrastructure (modulo the
Bdf/Auto static-method quirk surfaced in C5).

**Document revision:** None.

### H5. Anti-patterns list — accurate; no contradictions in the workspace

**Found:** Spot-checked each anti-pattern against the workspace:
- "Stringly-typed configuration" — not present (`SolverOptions` is
  typed; `OdeSolverChoice` per audit is an enum, not a string-keyed
  config).
- "Sentinel return values" — not present in foundational results
  (`SolverResult.success: bool` is paired with `message: String`;
  errors return `Result`).
- "Hidden state via global mutable variables" — `numra-autodiff`'s
  `TapeRef = Rc<RefCell<Tape>>` (`tape.rs:55`) is per-instance, not
  global; no `static mut` anywhere.
- "Capability-specific clones of foundation types" — `numra-pde`'s
  `MOLSystem*` impls go through `OdeSystem`, no clone; `numra-ocp`
  re-exports `numra-ode::SensitivityResult`. ✓
- "Print-or-panic instead of return-error" — most fallible APIs return
  `Result`; some `assert_eq!` exists in hot paths
  (`solve_forward_sensitivity` lines 845, 851 assert input dimensions),
  which is a panic-on-misuse but not on `Result`-able failure modes.
- "Opaque result types" — none found (per B4).
- "Defensive `Send + Sync`" — present (per B8); the contract anti-
  pattern is correct, the workspace doesn't yet uniformly follow it.
- "Secret runtime requirements" — none found (no `getenv` usage in
  `src/`; no implicit tokio runtime requirement).

**Document revision:** None.

---

## I. Cross-cutting findings (both documents)

### I1. Drafting voice / AI-attribution scrub

**Found:** Both drafts contain no `Co-Authored-By: Claude`, no
AI-named branches, no AI references in commit messages or doc bodies.
Drafting voice is consistent with the rest of the codebase
(authorial voice). No scrub needed.

**Document revision:** None.

### I2. The two documents reference each other consistently

**Found:** Foundation Spec §5 references the Composability Contract
inline; Composability Contract §6 references the Foundation Spec §2.1.
The cross-references are correct.

**Document revision:** None.

---

## J. Meta-finding on contract-violation density

**Directive (d):** > "if the violation count is genuinely massive
(more than half of existing capabilities fail at least one contract
item), surface that as a meta-finding."

**Found:** Running the seven-item contract against shipped capabilities:

| Capability | 1. Pub trait | 2. Inspectable | 3. Errors→Numra | 4. Foundation in | 5. Foundation out | 6. Generic-S | 7. Interop test |
|---|---|---|---|---|---|---|---|
| ODE solvers (DoPri5/Tsit5/Vern\*/Esdirk\*) | ✓ | ✓ | ✗ (SolverError no From) | ✓ | ✓ (SolverResult flows) | ✓ | ✓ |
| ODE solvers (Radau5/Bdf/Auto) | ✓ | ✓ | ✗ | ✓ | ✓ | ◐ (faer-bound leak) | ✓ |
| Forward sensitivity | ✓ | ✓ | ✗ | ✓ | **✗ (no downstream consumer of SensitivityResult)** | ✓ | ◐ (test exists; wrong target) |
| ODE uncertainty (`solve_trajectory`, `solve_with_uncertainty`) | n/a | ✓ | ✗ | ✓ | ✓ (used by interop test) | ✓ | ✓ |
| SDE solvers | ✓ | ✓ | ◐ (no SdeError surfaced; uses panics?) | ✓ | ✓ (SolverResult) | ✓ | ✗ (no interop test) |
| DDE solvers | ✓ | ✓ | ◐ | ✓ | ✓ | ✓ | ✗ |
| FDE solvers | ✓ | ✓ | ◐ | ✓ | ✓ | ✓ | ✗ |
| IDE solvers | ✓ | ✓ | ◐ | ✓ | ✓ | ✓ | ✗ |
| PDE MOL | ✓ | ✓ | ✗ | ✓ | ✓ | ✓ | ✓ |
| SPDE | ✓ | ✓ | ◐ | ✓ | ✓ | ✓ | ✗ |
| Optim (Bfgs/Lbfgs/LM/SQP/AugLag/CMAES/DE/NM/MILP) | ✓ | ✓ | ✗ (OptimError no From) | ✓ | ✓ (OptimResult; Pareto) | ✓ | ✓ (Workflow 4) |
| OCP (shooting/collocation/param-est) | ✓ | ✓ | ✗ (OcpError no From) | ✓ | ✓ | ✓ | ✓ (Workflow 3) |
| Linalg (dense+iterative) | ◐ (Matrix only one impl; sparse not in trait) | ✓ | ✓ (LinalgError→NumraError) | ✓ | ✓ | ◐ (eigen f32+f64 separate) | ✗ |
| Autodiff (forward Dual) | ✓ | ✓ | n/a | ✓ | ✓ (Workflow 4) | ✓ | ✓ |
| Autodiff (reverse Var) | ✓ | ✓ | n/a | ◐ (`Vec<f64>`-typed entry) | ✗ (concrete `f64` output blocks generic pipelines) | **✗ (concrete f64)** | ✗ |
| Integrate (quad/Romberg/dblquad) | n/a | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ (Workflow 1) |
| Interp (CubicSpline/PCHIP/Akima/Lagrange/Linear) | ✓ (Interpolant) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Stats (distributions + tests + OLS) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ (Workflows 5, 6) |
| Special functions | n/a | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ (no interop test; consumed indirectly via Stats) |
| FFT (rustfft + Welch + STFT) | n/a | ✓ | n/a | ✓ | ✓ | ◐ | ✓ (Workflow 2) |
| Signal (filters + Hilbert + peaks) | n/a | ✓ | ✗ (SignalError no From) | ✓ | ✓ | **✗ (butter is f64-only)** | ✓ (Workflow 2) |
| Fit (curve_fit + polyfit) | n/a | ✓ | ✗ (FitError no From) | ✓ | ✓ | ✓ | ✓ (Workflow 4) |

Legend: ✓ pass, ✗ fail, ◐ partial.

**Aggregate:** 21 capabilities tracked; ~13 fail at least one item
(item 3 alone — `From for NumraError` — fails for ODE solvers, SDE,
DDE, FDE, IDE, SPDE, Optim, OCP, Signal, Fit; item 7 fails for SDE,
DDE, FDE, IDE, SPDE, Linalg, Autodiff-reverse, Special). That is
**>50% of capabilities failing at least one contract item**.

**Implication (per the escalation rule):** This is a strategic
question, not absorbable into individual findings.

The pattern is two-fold:
- **Item 3 (`From for NumraError`) is the single biggest gap.**
  Adding the missing `From` impls is mostly mechanical work (10 crates
  × one impl each); see F-ERR.
- **Item 7 (interop test per capability) is the second biggest gap.**
  Several capabilities are exercised only in `numra/tests/` integration
  tests within their own crate (e.g. SDE, DDE) and not in
  `interop_workflows.rs`. The contract is strict ("ships with at least
  one interop test demonstrating composition"); the workspace was
  built before the contract was written. Backfilling tests is real
  work (~one test per capability not yet covered, ~6–8 capabilities).

**Strategic options to choose between:**
- (a) **Hold the line.** Contract stays as-is; the gaps surface as
  follow-ups; the next foundation-affecting PR is also responsible for
  closing one or two gaps. Eventually the table goes all-✓.
- (b) **Phase the contract.** Mark items 3 and 7 as "must-have for new
  capabilities; existing capabilities have a documented exemption window
  closing at version `0.X` with their backfill tracked".
- (c) **Soften the contract.** Reword item 3 from "implements `From<...>
  for numra::Error`" to "is reachable via `?` in some way (direct or
  via `.map_err`)". Reword item 7 from "ships with at least one interop
  test" to "is exercised by at least one cross-crate test in the
  workspace, even if the test is local to its own crate".

I do **not** recommend (c) — it dissolves the property the contract
was written to enforce. (a) is the principled choice; (b) is the
pragmatic one. This is the strategic question.

The contract as drafted commits to (a). My recommendation: keep (a) in
the document; the high violation count is information about the
codebase, not a sign the contract is wrong. Document this verification-
pass result as the baseline; the contract becomes the ratchet that
moves the table toward all-✓ over time.

**Document revision:** Add a brief paragraph in the contract (or in a
"Status" section near the top) acknowledging the baseline: "At the
time this contract was first published, ~13 of 21 shipped capabilities
fail at least one contract item; the gaps are tracked in
`internal-followups.md`. The contract enforces compliance going
forward; existing gaps are owned by named follow-ups, not silent
exemptions."

**Follow-up:** None unique — F-ERR and F-INTEROP-Q are the largest
gap-closing follow-ups, and they're already created above. The
strategic decision (a/b/c) is not a follow-up; it's a documentation
choice for review.

---

## Summary table — follow-ups created by this verification pass

Per the granularity directive, related issues are consolidated into one
substantial follow-up rather than many small ones.

| Tag | Title | Section in internal-followups.md | Status |
|---|---|---|---|
| F-LEAK | Generify foundational code over `Scalar` (autodiff reverse mode, signal filter design, eigen blanket) | Solvers | scoped, not started |
| F-ERR | Complete the `NumraError` `From`-impl coverage (10 crate-error types) | API | scoped, not started |
| F-FD-STEP | Reconcile FD step formula between `OdeSystem` and `ParametricOdeSystem` defaults; fix `f32` viability | Solvers | scoped, not started |
| F-SENDSYNC | Audit & remove defensive `Send + Sync` on foundation traits (Scalar, Signal, EventFunction, NonlinearSystem, SdeSystem; PDE/OCP closure aliases) | API | scoped, not started |
| F-OPTS | Document SDE/FDE/IDE options divergence from `SolverOptions` in their rustdoc | API | scoped, not started |
| F-SOLVER-FIELDS | Clarify or remove `Bdf::max_order` / `Auto::*` fields that the static `Solver::solve` cannot read | Solvers | scoped, not started |
| F-MATRIX-SHAPE | Decide whether `SparseMatrix` joins `Matrix` trait or stays separate | API | scoped, not started |
| F-INTEROP-Q | Backfill interop tests (a) exercising `?`-propagation across two crate boundaries, (b) exercising one composition at non-`f64` `Scalar`, (c) covering capabilities that lack an interop edge today (SDE, DDE, FDE, IDE, SPDE, Linalg, Autodiff-reverse, Special) | API | scoped, not started |
| F-SENS-DOWNSTREAM | Decide downstream consumers for `SensitivityResult` beyond `numra-ocp` re-export — surfaces from contract worked example | API | scoped, not started |

---

## Confirmation that the verification scope was respected

- No source code was changed by this pass.
- No revisions were silently applied to the drafts; revisions implied
  by the findings are recorded above and will be applied to the
  revised drafts after the Findings Report is approved.
- No interop tests were authored.
- No open questions in §7 were resolved (G1's §7.6 closure is a
  verification answer, not a design decision).
- No capabilities were refactored.
- No documents outside the two architecture documents and
  `docs/internal-followups.md` will be updated by this pass.

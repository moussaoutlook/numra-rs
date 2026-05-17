# Numra Foundation Specification (v0)

**Status:** This document captures the load-bearing architecture of the Numra workspace — the small set of traits, types, and conventions that everything else depends on — and the design decisions that govern them. It is intentionally minimal. The foundation is small so that hardening it is feasible.

**Audience:** Project author, future contributors, future Claude Code sessions making foundation-affecting changes.

**Process:** Foundation-affecting changes require explicit review against this document. The document is updated when changes land; the update is part of the change, not a separate task.

**First publication:** 2026-05-10. Verification pass on the v0 draft produced 35 findings against 21 capabilities; the resulting revisions are folded into this published version. The findings record itself is preserved at `docs/architecture/findings-2026-05-10-foundation-pass.md`.

---

## 0. What this document is and isn't

**Is:** A specification of the load-bearing abstractions in Numra — the traits and types that have many downstream consumers, where breaking changes cascade widely. Includes design decisions, open questions, deferred decisions.

**Is not:** A roadmap. A capability inventory. A list of features to build. Those are separate documents. This one is about architecture.

---

## 1. Scope of "foundation"

A trait, type, or convention is **foundational** if any of these is true:

1. It is implemented or consumed by code in three or more workspace crates.
2. Changes to its shape would force breaking changes in at least three places.
3. Removing it would require redesigning at least one major workflow (problem definition → solver → result → composition).

The foundation list below is the **curated set** — any trait that meets these criteria is *eligible* for inclusion, but inclusion is a deliberate decision. Borderline traits (e.g. `EventFunction`, `NonlinearSystem`) that mechanically satisfy the criteria but aren't load-bearing in the same way are tracked separately.

By this definition, the current foundation comprises:

- **`Scalar` trait** (`numra-core/src/scalar.rs:42`) — the abstraction over numeric types.
- **`Vector` trait** (`numra-core/src/vector.rs:40`) — the abstraction over vector types.
- **`Matrix` trait** (`numra-linalg/src/matrix.rs:15`) — the abstraction over matrix types.
- **`OdeSystem` trait** (`numra-ode/src/problem.rs:15`) — the abstraction over ODE problem definitions.
- **`ParametricOdeSystem` trait** (`numra-ode/src/sensitivity.rs:152`) — the abstraction over parameterised ODE systems for forward sensitivity. Implemented in `numra-ode` (`AugmentedSystem`), `numra-pde` (`ParametricMOLSystem2D/3D`); consumed by `numra-ocp::forward_sensitivity` and the `solve_forward_sensitivity{,_with}` entry points.
- **`Signal` trait** (`numra-core/src/signal.rs:51`) — the abstraction over time-domain inputs.
- **`Solver` trait** (`numra-ode/src/solver.rs:291`) — the abstraction over ODE integration algorithms.
- **Result types** (`SolverResult`, `SensitivityResult`, `OptimResult`, `ParameterSensitivityResult`) — the abstraction over computed outputs that flow into downstream consumers.
- **Error types** — the propagation discipline that makes `?` work across crate boundaries. Workspace error vehicle is `numra_core::NumraError` (`numra-core/src/error.rs:16`).
- **`SolverOptions`** (`numra-ode/src/solver.rs:20`) — the shared configuration object used by every ODE-derived solver.

Anything not in the list above is **not foundational** — it can change without rippling through the workspace, so its design is governed by the local crate's needs, not this document.

---

## 2. Design principles (foundational decisions that govern everything else)

These are the principles every foundation-affecting change must respect. They are deliberately few; principles that don't earn their place become noise.

### 2.1 Generic over Scalar wherever possible

Foundational traits are generic over `S: Scalar`, not concrete `f64`. Concrete-`f64` impls are acceptable only where the underlying algorithm has a `f64`-only dependency (a wrapped C library, a faer routine that's `f64`-only).

When a concrete type leak is unavoidable, document it in the trait's rustdoc with the reason. Don't hide it.

Audit baseline (recorded 2026-05-10): three foundational-adjacent concrete-`f64` leaks remain — `numra-autodiff` reverse mode, `numra-linalg` general (non-symmetric) eigendecomposition (split into separate `f32`/`f64` concrete impls rather than a generic blanket), and `numra-signal` filter design (`butter`, `filtfilt`, `instantaneous_frequency`'s `fs` parameter). Tracked in `internal-followups.md` as F-LEAK.

### 2.2 Foundation traits are implementable from outside the workspace

A user defining their own `OdeSystem`, `ParametricOdeSystem`, or `Solver` outside the workspace must be able to use it with Numra's machinery. This implies:

- Traits use only public items.
- Default-method bodies use only public items.
- No `pub(crate)` types appear in foundation trait signatures.
- Sealed traits are forbidden in the foundation. (Internal helpers may be sealed; foundational traits may not.)

Audit baseline: workspace-wide search returned no `pub(crate)` types reachable from foundation trait signatures or default bodies, and no sealed-trait machinery anywhere. ✓

### 2.3 Result types are inspectable, not opaque

Every foundational result type (`SolverResult`, `SensitivityResult`, `OptimResult`, `ParameterSensitivityResult`) exposes its underlying data through public fields or accessor methods. Users can read, slice, and convert without going through framework-defined operations.

This is what enables composition: ODE result → interpolant → integrand works because the result exposes `t`, `y`, and a way to construct an interpolant from them.

Audit baseline: every named result type meets the bar (all fields `pub` or covered by `pub fn` accessors). ✓

### 2.4 Error types compose across crate boundaries

A user writing `solve_ode(...)?.integrate(...)?.fit(...)?` must have `?` work without manual error conversion. The workspace error vehicle is `numra_core::NumraError`; every fallible crate's error type is expected to provide `From<MyError> for NumraError`.

Audit baseline (2026-05-10): the workspace error type exists and is re-exported from the facade, but `From`-impl coverage is incomplete. Crates with the impl: `numra-interp`, `numra-integrate`, `numra-special`, `numra-stats`, plus the core sub-errors. Crates without the impl (and therefore not yet covered by cross-crate `?`): `numra-ode (SolverError)`, `numra-optim (OptimError)`, `numra-ocp (OcpError)`, `numra-fit (FitError)`, `numra-signal (SignalError)`. Coverage gap tracked as F-ERR. The verification pass also surfaced that no interop test currently exercises cross-crate `?` (every test in `interop_workflows.rs` uses `.unwrap()`); tracked as part of F-INTEROP-Q.

### 2.5 Configuration objects are shared across solver families

`SolverOptions` (`rtol`, `atol`, `h_max`, `dense_output`, `t_eval`, etc.) is one type, used by every solver that consumes ODEs. New solver families that define their own options struct must justify the divergence — and the divergence must be visible in the rustdoc.

Audit baseline: `SolverOptions` is shared by every consumer of an `OdeSystem`-shaped problem (`numra-ode`, `numra-pde` via MOL, `numra-dde` via Method-of-Steps, `numra-ocp` via shooting/collocation, `numra-spde` for the PDE arm). `numra-sde`, `numra-fde`, `numra-ide` have their own options structs (predating this principle); their rustdoc divergence rationale is the F-OPTS follow-up.

### 2.6 No interior mutability in foundation traits without a documented reason

Foundation traits should not require `RefCell`, `Mutex`, or `OnceCell` in their method signatures. Implementors that need scratch buffers should use `&mut self` methods or design around per-call allocation. The exceptions documented today (`AugmentedSystem`'s `RefCell` scratch buffers for the augmented sensitivity Jacobian path) are acknowledged as performance trade-offs, not endorsed as patterns.

Audit baseline: only `AugmentedSystem` in `numra-ode/src/sensitivity.rs:288` (six `RefCell<Vec<S>>` plus one debug-only `Cell<bool>`) carries foundation-routed interior mutability. `numra-autodiff`'s `Rc<RefCell<Tape>>` is internal to the reverse-mode tape, not foundation-routed. The Jacobian unification work (Radau5/BDF routing through `OdeSystem::jacobian`, 2026-05-07) introduced no new interior mutability. ✓

### 2.7 No `Send + Sync` defensively

Foundation traits add `Send + Sync` only when callers actually require parallel access. Defensive bounds propagate as virality through every user-defined type and become breaking to remove. The recent `ParametricOdeSystem` decision (drop both bounds, document call-site escalation pattern) is the precedent.

This principle applies prospectively. Pre-existing defensive bounds — `Scalar: ... + Send + Sync + 'static`, `Signal<S>: Send + Sync`, `EventFunction<S>: Send + Sync`, `NonlinearSystem<S>: Send + Sync`, `SdeSystem<S>: Sync`, the `Send + Sync + 'static` closure aliases in `numra-pde` and `numra-ocp` — predate the principle and are not back-patched silently. They are tracked as F-SENDSYNC.

### 2.8 Foundation changes are visible, not silent

Every change to a foundational trait, type, or convention requires a CHANGELOG entry under "Changed" with rationale. Once 1.0 ships, foundational breaking changes also require a deprecation cycle of at least one minor version.

---

## 3. The trait spine

This section specifies the design of each foundational trait. For each: contract, current shape, design choices made, deferred decisions.

### 3.1 `Scalar`

**Purpose:** Abstract over numeric types so generic code can run on `f32`, `f64`, and (deferred) complex.

**Current signature (`numra-core/src/scalar.rs:42`):**

Supertraits: `Copy + Clone + Debug + Display + PartialOrd + Add + Sub + Mul + Div + Neg + AddAssign + SubAssign + MulAssign + DivAssign + Sum + Send + Sync + 'static`.

Required associated constants: `ZERO, ONE, TWO, HALF, EPSILON, INFINITY, NEG_INFINITY, NAN, PI, E, SQRT_2, LN_2`.

Required methods (grouped): conversion (`from_f64, from_f32, from_i32, from_usize, to_f64, to_f32`); basic ops (`abs, sqrt, cbrt, powi, powf, hypot`); trig (`sin, cos, tan, asin, acos, atan, atan2`); exp/log (`exp, exp2, exp_m1, ln, log2, log10, ln_1p`); hyperbolic (`sinh, cosh, tanh, asinh, acosh, atanh`); ordering (`max, min, clamp, copysign`); predicates (`is_finite, is_nan, is_infinite, is_sign_positive, is_sign_negative`); rounding (`floor, ceil, round, trunc, fract`); special (`gamma_fn, ln_gamma, erf_fn, erfc_fn`); utility (`mul_add`).

Provided defaults: `sq, sincos, recip, signum`.

Blanket impls: `f64` and `f32`, both delegating to `libm`. No other impls.

**Design choices:**

- Real-only for v1. Complex numbers are NOT a `Scalar`. `Complex<f64>` exists as a peer type in `numra-fft` but doesn't satisfy the trait. The trait requires `PartialOrd` and the default `signum` body uses ordering against `ZERO`, both real-only.
- `f32` and `f64` are the only blanket impls.
- Math operations are required methods of the trait, not provided via `num-traits::Float`. Deliberate: avoids an external dep with its own opinions. The doc comment at `scalar.rs:14` records this explicitly.
- The supertrait `Send + Sync + 'static` bound is defensive — both blanket impls satisfy it trivially, but it propagates virally to any user-defined `Scalar`. Tracked under F-SENDSYNC.

**Deferred decisions (open questions):**

- **Complex support.** Adding `impl Scalar for Complex<f64>` would require auditing every consumer of `Scalar` for real-only assumptions (signs, `<` comparisons, ordering). This is a significant project. Until a real workflow demands complex-domain ODE/optimization, the cost is not justified. Decision: defer; revisit when a complex-valued problem class is requested.
- **Arbitrary precision (`BigDecimal`, etc.).** Same logic. Defer.
- **Interval arithmetic types.** The `numra-uncertainty` work suggests `Uncertain<S>` could be a `Scalar` for some workflows. Not a current commitment; flagged as research direction.

**Forward-compatibility risks:**

- If complex support is added later, every concrete-`f64` leak in foundational code (audit item) becomes a barrier. This is one of the strongest reasons to audit and remove concrete-`f64` leaks now, even before complex is in scope.

### 3.2 `Vector` and `Matrix`

**Purpose:** Abstract over linear-algebra primitives so solvers don't depend on a specific layout.

**Current signatures:**

`Vector<S: Scalar>: Clone + Sized` (`numra-core/src/vector.rs:40`). Required methods: `zeros, fill, from_slice, len, get, set, get_mut, as_slice, as_mut_slice, copy_from, axpy, axpby, dot, scale, norm_inf, norm1, abs_inplace, max_elementwise, min_elementwise, sum, max_element, min_element, map_inplace`. Provided defaults: `is_empty, norm2, weighted_rms_norm`.

`Matrix<S: Scalar>: Clone + Sized` (`numra-linalg/src/matrix.rs:15`). Required methods: `zeros, identity, nrows, ncols, get, set, fill_zero, scale, mul_vec, add_scaled, solve`. Provided default: `is_square`.

**Workspace impls:**

- `Vector<S>`: a single blanket `impl Vector<S> for Vec<S>` (`vector.rs:151`). No faer-vector impl exists despite the doc comment claiming "faer vectors (when using numra-linalg)" — that line is aspirational.
- `Matrix<S>`: a single `impl Matrix<S> for DenseMatrix<S>` (`matrix.rs:154`) under the additional bound `S: Scalar + SimpleEntity + Conjugate<Canonical = S> + ComplexField`. The faer trait bounds leak into the Matrix concrete type's instantiation; external `Scalar` impls cannot use `DenseMatrix` without satisfying the faer bounds.
- `numra-linalg::SparseMatrix<S>` (`sparse.rs`) is a separate concrete type that **does not** implement `Matrix<S>`. The trait abstraction covers dense only.

**Design choices:**

- faer is the underlying engine for non-trivial linear algebra; `Vector`/`Matrix` traits are interfaces, not the implementation.
- Layout is a property of the concrete type, not the trait.
- Sparse is **not** in the `Matrix` trait today — `SparseMatrix<S>` is a parallel concrete type. This is an open question (see §7).

**Deferred decisions:**

- **Sparse joins `Matrix`?** Open question (§7). Decision when a sparse-aware solver path needs to dispatch across both dense and sparse via a single trait.
- **Banded / tridiagonal types.** Should be added when 1D PDEs demand it — but the foundation question is whether banded is a `Matrix` impl or a separate type with separate solver paths. Pin the decision when the work is done.
- **GPU / SIMD residency.** A "where does this live" abstraction (CPU vs GPU memory) would be needed if GPU is ever added. Deferred to when GPU work begins.

**Forward-compatibility risks:**

- If GPU residency is added later, every concrete `Vec<S>` use in solver code becomes a barrier (`Vec<S>` is CPU-only). Same logic as the `f64` leak: audit and abstract now to avoid a bigger refactor later.

### 3.3 `OdeSystem`

**Purpose:** The most-implemented foundation trait. Every ODE-derived problem (ODE, MOL-discretized PDE, augmented sensitivity, parametric MOL) implements it; every ODE solver consumes it.

**Current signature (`numra-ode/src/problem.rs:15`):**

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

**Design choices:**

- `rhs(&self, t, y, dydt)` is in-place: caller provides the output buffer. No allocation in the hot path.
- `jacobian(&self, t, y, jac)` has a forward-FD default with `eps = S::from_f64(1e-8)` and `h = eps * (1 + |y_j|)`, row-major dense output. Implementors override for analytical Jacobians. Documented in CHANGELOG under the Jacobian-unification work (Radau5/BDF route through this default since 2026-05-07).
  - **Known limitation.** The `eps = 1e-8` value approximates `sqrt(f64::EPSILON) ≈ 1.49e-8` for `f64` but is *below* `f32::EPSILON ≈ 1.19e-7`, which makes the default FD path silently useless on `f32` (the perturbation gets quantised to zero or loses all signal). Users implementing `OdeSystem` with `S = f32` should override `jacobian` with an analytical implementation, an FD path tuned for `f32`, or a generic-precision FD path. Tracked in F-FD-STEP — note that `ParametricOdeSystem::jacobian_y/_p` defaults already use `S::EPSILON.sqrt()`, which is the formula that should propagate to `OdeSystem` when F-FD-STEP lands.
- Mass-matrix DAE support: `mass_matrix(&self, mass: &mut [S])` writes a row-major mass matrix into the caller's buffer; default = identity. DAE-ness is signalled via the `has_mass_matrix() -> bool` flag plus `is_singular_mass() -> bool` and `algebraic_indices() -> Vec<usize>`. This is a deliberate flag-and-fill-buffer design (consistent with `rhs` style); not an `Option<...>`.
- Index-1 supported in solvers; index-2/3 via Pantelides reduction.
- No `Send + Sync` bound on the trait itself.
- Dense output is opt-in via `SolverOptions.dense_output`, not a trait method.

**Deferred decisions:**

- **Sparse Jacobian return.** The current `jacobian` writes into a dense buffer. For large stiff problems with sparse coupling (PDE MOL, big chemical kinetics networks), this is wasteful. Sparse Jacobian return is a follow-up; the question is whether to add a separate trait method (`jacobian_sparse`) or change the existing one to return `impl MatrixView`. Decision deferred until a workload demands it.
- **JVP / VJP variants.** `diffsol` (the Rust peer) uses Jacobian-vector products instead of materialized Jacobians. For Numra v1, the dense Jacobian path is correct; JVP is a follow-up.
- **Mass matrix as a first-class composable.** Currently mass-matrix DAE is supported in `Radau5` / `BDF` only; PDE/sensitivity coupling doesn't accept DAE form. Promoting mass-matrix to first-class in `OdeSystem` is a foundation change worth doing once a user requests it.

**Forward-compatibility risks:**

- The `Jacobian` enum the audit suggests (AutoDiff | Analytical | FD) would change the trait. Worth designing now so it's done once. Specifically: should Jacobian source be a constructor parameter (`OdeSystem::with_autodiff(...)`) or a runtime decision (the trait queries `jacobian_source()`)? The autodiff path needs to be a foundational integration, not a per-crate retrofit.

### 3.4 `Solver`

**Purpose:** Abstract over ODE integration algorithms so users can write `DoPri5::solve(...)` or `Radau5::solve(...)` without changing the surrounding code.

**Current signature (`numra-ode/src/solver.rs:291`):**

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

`solve` is a static method (no `&self`).

**Design choices:**

- `Solver` exposes a static method. Most implementors are zero-size unit types: `DoPri5`, `Tsit5`, `Vern6/7/8`, `Radau5`, `Esdirk32/43/54`. State at solve time comes from `SolverOptions`, not the implementor.
- BDF order control flows through `SolverOptions` (see §3.7). Resolved 2026-05-17 (decision-log §6 #12) per option (b) of the original design tension recorded in F-SOLVER-FIELDS: `Bdf::max_order` / `Bdf::min_order` previously lived on the `Bdf` struct as silently-non-functional builders (`Bdf::with_max_order`, `Bdf::fixed_order`) — `Bdf`'s static `Solver::solve` discarded `self` and they had no path through the trait. The (b) wiring promoted them to `SolverOptions::max_order` / `SolverOptions::min_order` (top-level `Option<usize>` fields, defaults preserving prior behavior); the vestigial `Bdf` builders, fields, `Bdf::new` constructor, and the parallel `Auto` struct (same silently-non-functional pattern) were removed. Option (a) — deleting the order-control capability — was rejected because the underlying numerical capability is legitimate. Option (c) — reshaping `Solver::solve` to `fn solve(&self, ...)` so implementors can carry instance state — remains open as §7 #8; adopting (c) is a deliberate foundation-design decision against §2.5's centralized-configuration principle and is not foreclosed by (b).
- Solver impls split into two bound classes:
  - Generic `<S: Scalar>` (no faer): `DoPri5`, `Tsit5`, `Vern6/7/8`.
  - Generic `<S: Scalar + SimpleEntity + Conjugate<Canonical = S> + ComplexField>` (faer-bound): `Radau5`, `Bdf`, `Esdirk32/43/54`, `Auto`.
  External `Scalar` impls (a hypothetical `BigDecimal` etc.) can use the explicit-RK solvers but not the implicit ones. The faer-universe leak into the implicit-solver bound is an architectural choice that mirrors §3.2's `DenseMatrix` situation.
- Adaptive step control is built into every implementor; fixed-step is achieved by setting `min_step == max_step`.

**Deferred decisions:**

- **Backward integration support.** Adjoint sensitivity needs backward integration. Currently `Solver::solve` integrates forward; adding backward requires either a flag in `SolverOptions` or a separate `solve_backward` method. Foundation decision when adjoint work happens.

### 3.5 `Signal`

**Purpose:** Abstract over time-domain inputs to controls and forcing functions.

**Current signature (`numra-core/src/signal.rs:51`):** A trait generic over `S: Scalar` with `Send + Sync` bound. Required: an evaluator method taking `&self, t: S` returning `S`. Provided default: a derivative method using central finite differences with `h = 1e-8`.

Variants ship as concrete structs implementing the trait: `Harmonic<S>`, `Step<S>`, `Ramp<S>`, `Pulse<S>`, `Chirp<S>`, `Tabulated<S>`, plus composites (`Piecewise`, `Sum`, `Product`) and an opt-in `FromFile` behind the `std` feature.

**Design choices:**

- **Open trait hierarchy.** `Signal` is a trait, not an enum. Both extension paths work: adding a new shipped variant (a new struct in `numra-core/src/signal.rs`) is a non-breaking addition, and external users can implement `Signal<S>` for their own types. Adding a new variant doesn't widen an enum (no exhaustiveness break) and doesn't seal the type set against external impls.
- The default derivative uses central FD with hard-coded `h = 1e-8`; same `f32`-vs-`f64` precision hazard as the `OdeSystem` default. Tracked in F-FD-STEP.
- The trait carries a defensive `Send + Sync` bound; tracked in F-SENDSYNC.

### 3.6 Result types

**Purpose:** Carry the outputs of solvers, optimizers, and analysis routines in a shape that downstream code can consume.

**Audited (2026-05-10):**
- `SolverResult<S>` (`numra-ode/src/solver.rs:177`) — fields all `pub`, with `len`, `is_empty`, `t_final`, `y_final`, `y_at`, `n_steps`, `component`, `iter` accessors. ✓
- `SensitivityResult<S>` (`numra-ode/src/sensitivity.rs:614`) — fields all `pub`, with `len, is_empty, y_at, sensitivity_at, sensitivity_for_param, dyi_dpj, final_state, final_sensitivity, normalized_sensitivity_at`. ✓
- `OptimResult<S>` (`numra-optim/src/types.rs:123`) — fields all `pub`. ✓
- `ParameterSensitivityResult<S>` (`numra-core/src/uncertainty.rs:226`) — accessors `column`, `row`, `get`, plus a `new` constructor. ✓

No "opaque result" anti-pattern detected in the audited types. Other crates (`numra-sde, numra-dde, numra-fde, numra-ide, numra-pde, numra-spde, numra-ocp`) return `SolverResult` or domain-specific results that are inspectable; spot-checked.

### 3.7 `SolverOptions` and configuration

**Purpose:** Shared configuration object for every ODE-derived solver family. Users learn one configuration vocabulary, not eighteen.

**Current shape (`numra-ode/src/solver.rs:20`):** struct generic over `S: Scalar`.

Public fields: `rtol, atol, h0, h_max, h_min, max_steps, t_eval, dense_output, max_order, min_order, events`. The `events` field is `Vec<Arc<dyn EventFunction<S>>>` (Arc enables `Clone` on the options). Builder methods: `rtol, atol, h0, h_max, t_eval, dense, max_steps, h_min, max_order, min_order, event`. Defaults: `rtol = 1e-6, atol = 1e-9, h_max = INFINITY, h_min = EPSILON * 100, max_steps = 100_000, dense_output = false, max_order = None, min_order = None, events = empty`.

The `max_order` / `min_order` `Option<usize>` fields are consumed by BDF (clamped to `[1, MAX_ORDER=5]`) and ignored by every other solver — the same "applicable to some solvers, ignored by the rest" pattern as `dense_output` (consumed only by solvers with dense interpolant support) and `events` (consumed only by event-aware solvers). Top-level placement was chosen over a per-family sub-struct as the spec-mandated reading of §2.5 ("one configuration vocabulary, not eighteen"); see decision-log §6 #12 and §3.4 for rationale.

**Deferred decisions:**

- **Sensitivity tolerances.** Currently sensitivity inherits state tolerances. CVODES exposes separate `sens_rtol` / `sens_atol`. Worth adding when parameter-ID workflows demand it.
- **Per-solver-family extension mechanism.** Some solvers still have their own knobs not yet promoted to `SolverOptions` (Radau5's Newton tolerance is the canonical example). The first such knob — BDF's `max_order` / `min_order` — was promoted to top-level `SolverOptions` fields on 2026-05-17 (decision-log §6 #12), closing a previously-documented spec-vs-reality drift. Whether subsequent per-family knobs continue to land as top-level fields (following the `dense_output` / `events` precedent) or whether the namespace eventually justifies a per-family sub-struct is a *foundation-design* decision triggered by the second or third per-family knob landing — not a default-anticipated pattern. The migration from "several top-level per-family fields" to "sub-struct organization" is a clean foundation pass if and when the namespace pressure actually surfaces.

---

## 4. Composability primitives

This section enumerates the patterns by which foundational types compose with each other. New capabilities respect these patterns; capabilities that don't fit must justify the divergence.

### 4.1 Problem-class to ODE pipeline — two coexisting reductions

Different problem classes reach the ODE solver world via different paths.

1. **Problem classes that expose themselves as `OdeSystem`** so any ODE solver consumes them: PDE (MOL via `MOLSystem`, `MOLSystem2D`, `MOLSystem3D`); augmented sensitivity (`AugmentedSystem` wraps a `ParametricOdeSystem` as an `OdeSystem`); parametric MOL (`ParametricMOLSystem2D/3D` impl `ParametricOdeSystem`, then composed via `AugmentedSystem`).
2. **Problem classes that have their own problem trait and reuse a specific ODE solver internally**: DDE (`DdeSystem` + Method-of-Steps wrapping `DoPri5` + Hermite history); OCP (`shooting`/`collocation` consume an `OdeSolverChoice::DoPri5` private enum, not the generic `Solver` trait — see audit Tier 1.4 for the promotion follow-up).
3. **Problem classes with their own problem trait and domain-specific solvers** (no ODE-solver reuse): SDE (`SdeSystem` + EM/Milstein/SRA*); FDE (`FdeSystem` + L1); IDE (`IdeSystem` + trapezoidal/RK4/Prony); SPDE's SDE arm.

This shape is not a defect; different problem classes need different abstractions. The composability story is correspondingly varied — pattern 1 gives free reuse of the ODE solver fleet; pattern 2 hardcodes one solver internally; pattern 3 stands alone.

### 4.2 Solver to result-to-input flow

Solver returns a result type whose fields and accessors enable construction of inputs to downstream solvers/analyses without the user building intermediate adapters. Concretely realised today:

```text
ODE::SolverResult              → Interp::CubicSpline       → Integrate::quad
ODE::SolverResult.component()  → FFT::psd                  → Signal::butter+filtfilt → Signal::find_peaks
PDE::MOLSystem                 → ODE::DoPri5::solve        → Stats::{mean, variance, std_dev, median}
[manual sampling]              → ODE::OdeProblem ensemble  → Stats::{mean, std_dev}
Autodiff::Dual + gradient_closure → Optim::OptimProblem.gradient → Fit::curve_fit_with_jacobian
OCP::ParamEstProblem           → ODE::solve_with_uncertainty
```

These six edges are exercised in `numra/tests/interop_workflows.rs`. Additional cross-crate composition (not covered by an interop test today, but realised in production code) is enumerated below.

### 4.3 Sensitivity composition

Forward sensitivity is built on `ParametricOdeSystem` (a peer trait to `OdeSystem`) plus an augmented-system construction. `AugmentedSystem<S, Sys: ParametricOdeSystem<S>>` implements `OdeSystem<S>`, so any `Solver` consumes it. The column-major sensitivity flattening, asymmetric `J_y` (row-major) / `J_p` (column-major) layout, and `has_analytical_jacobian_*` flag pattern with debug-build consistency-check safety net are documented in the trait rustdoc (`numra-ode/src/sensitivity.rs`).

Adjoint sensitivity, when added, follows the same pattern — currently it exists only as `numra-ocp`-private and `DoPri5`-only; promoting it is the named follow-up "Adjoint sensitivity" in `internal-followups.md`.

Sensitivity outputs are `SensitivityResult`. **Open composability gap:** as of 2026-05-10, `SensitivityResult` ships with no in-workspace downstream consumer (the `numra-ocp::forward_sensitivity` re-export is the only related edge, and it produces rather than consumes the type). Tracked as F-SENS-DOWNSTREAM.

### 4.4 Error composition

Every fallible operation is intended to return a workspace-compatible error type that converts (`From` impl) into the unified `numra_core::NumraError`, so that `?` propagates across crate boundaries.

**Current state (2026-05-10):** the workspace `NumraError` exists; the discipline is partially applied. Coverage gaps and the lack of an interop test exercising cross-crate `?` are tracked as F-ERR and F-INTEROP-Q. See §2.4.

### 4.5 Generic over Scalar at every layer

Composition preserves genericity. `solve_ode::<Sol, S, ...>(...)` returns `SolverResult<S>`, which feeds into `integrate::<S, ...>(...)` returning `S`, which feeds into `fit::<S, ...>(...)`. Generic propagation must not be broken by intermediate concrete-`f64` conversions.

**Test-coverage gap:** every interop workflow in `numra/tests/interop_workflows.rs` is monomorphised at `f64`. The principle is enforced by the production code's signatures (largely generic over `S: Scalar`) but no test exercises composition at `f32` or any external `Scalar` impl. Tracked as F-INTEROP-Q.

### 4.6 The composability graph

Cross-crate composition realised in production code (beyond the six interop edges in §4.2):

```text
ParametricOdeSystem            → AugmentedSystem (impls OdeSystem) → any Solver
ParametricMOLSystem2D/3D       → ParametricOdeSystem → AugmentedSystem → any Solver
DDE Method-of-Steps            → ODE::DoPri5 (internal)
SPDE                           → PDE (MOL spatial) + SDE (time)
OCP shooting / collocation     → ODE (DoPri5-only internal) + Optim
Stats distributions            → Special functions (gamma, erf, etc.)
Signal                         → FFT (filter design, Hilbert, psd path)
Fit::curve_fit                 → Optim::lm_minimize
Linalg                         → faer (external)
```

`numra-core::compute_sensitivities` produces `ParameterSensitivityResult` but has no in-workspace downstream consumer; standalone for now.

---

## 5. The Composability Contract (testable)

A capability satisfies the Composability Contract when all of these hold:

1. Its problem-definition trait is implementable from outside the workspace using only public types.
2. Its primary result type has public fields or accessors covering the data downstream consumers might want.
3. Its errors convert into the workspace error type via `From`.
4. It accepts at least one input from an existing capability without an adapter the user has to write themselves.
5. It produces an output that at least one existing capability can consume without an adapter.
6. Its public types are generic over `S: Scalar` where the underlying algorithm allows.
7. It ships with at least one interop test that exercises composition with at least one existing capability.

This is the contract every new capability is reviewed against. The detailed wording, verification methods, anti-patterns, and worked examples live in `docs/architecture/composability-contract.md`.

**Baseline state (2026-05-10):** verification pass against the 21 shipped capabilities found ~13 failing at least one item, predominantly items 3 and 7. The gaps are tracked as F-ERR and F-INTEROP-Q in `internal-followups.md`. The contract is the ratchet — every foundation-affecting PR also closes at least one gap; the gaps will close over time.

---

## 6. Decision log

Foundation-affecting decisions are logged here with rationale. Append-only.

1. **2026-05-05 (commit `9d93407`; release: `0.1.0` Unreleased)** — `SolverResult.dense_output: Option<DenseOutput<S>>` field added. Fixes a silent drop of the dense interpolant when `SolverOptions::dense()` was set. Foundation-affecting because the workspace's primary ODE result type changed shape.

2. **2026-05-05 (commit `7b6b816`)** — Radau5 step controller rewritten against Hairer–Wanner ODE II §IV.8 and SciPy's `Radau` reference. Internal algorithmic change; not strictly trait-affecting, but the controller stabilisation was a prerequisite for the subsequent Jacobian unification.

3. **2026-05-06 (commit `b420335`)** — Sensitivity API consolidation: introduced `ParametricOdeSystem` trait (renamed from `SensitivityEquations`) with `params()` and `rhs_with_params(t, y, p, dydt)`; FD-default `jacobian_y` and `jacobian_p`; column-major sensitivity flattening matching CVODES indexing; asymmetric `J_y` (row-major) / `J_p` (column-major) layout; no `Send + Sync` defensive bounds — call-site escalation pattern documented; `has_analytical_jacobian_*` flag pattern with debug-build consistency-check safety net. Renamed `numra-core::Sensitivity → ParameterSensitivity` and `SensitivityResult → ParameterSensitivityResult` to disambiguate the two distinct sensitivity concepts. Follow-ups created: block-diagonal LU, JVP, AD `ParametricOdeSystem` impl, staggered correction, separate sensitivity tolerances, flag ergonomics.

4. **2026-05-06 (commit `8fa9729`)** — Public entry points `solve_forward_sensitivity` and `solve_forward_sensitivity_with` added to `numra-ode`. Consumer of (3).

5. **2026-05-06 (commit `ba09a03`)** — BDF rewritten as a Rust port of SciPy's `BDF` (Shampine–Reichelt NDF / `ode15s`). Three independent algorithmic bugs fixed (Newton convergence-test target; LTE estimate using cumulative Newton correction; modified-divided-differences rescaling under variable steps). Pinned by `test_bdf_regression_silent_wrong_answer`. Foundation-affecting because BDF is a foundational ODE solver and its trait-level Jacobian dispatch was settled in the same window.

6. **2026-05-06 (commits `6f26aaf` + `34d81b6`)** — `numra-ode::solve_trajectory` and `numra-ocp::forward_sensitivity` reimplemented atop the canonical `solve_forward_sensitivity_with` primitive. Eliminated parallel implementations; `numra-ocp` sensitivity layout changed from row-major-over-states to column-major-over-parameters; `numra_ocp::SensitivityResult` is now a re-export of `numra_ode::SensitivityResult`.

7. **2026-05-07 (commits `8c170d7` + `7bc3559`)** — Jacobian unification: `Radau5` and `Bdf` now route their Jacobian computation through `OdeSystem::jacobian` instead of solver-inlined finite-difference paths. Systems that override the trait method (`MOLSystem2D`, `MOLSystem3D`) get an analytical Jacobian for free; systems that don't fall through to the trait default (`eps = 1e-8`, step `eps * (1 + |y_j|)`, row-major dense output). Resolves prior FD-step drift between Radau5's inlined `eps * max(1, |y_j|)` and the canonical `(1 + |y_j|)` form.

8. **2026-05-07 (commits `f7499ec` + `e592c6e` + `a91a5a6`)** — 3D MOL parity. `MOLSystem3D` wrapper mirroring `MOLSystem2D`; `Operator3DCoefficients` + general 7-point operator assembly; convenience builders. `MOLSystem2D` and `MOLSystem3D` both override `OdeSystem::jacobian` with CSC-to-row-major-dense copy of the spatial operator + diagonal-FD reaction term. Foundation-affecting because it pinned the analytical-Jacobian override pattern that downstream parametric MOL inherits.

9. **2026-05-08 (commit `da0674b`)** — `ParametricMOLSystem2D` and `ParametricMOLSystem3D` ship in `numra-pde`. v1 scope is alpha-on-Laplacian; full operator parametrisation tracked as the named follow-up "Full operator parametrisation in MOL".

10. **2026-05-10 (commit `ec5f355`)** — `numra-ode`: `SolverOptions::t_eval` is now honored by every solver. Previously parsed and ignored. Foundation-affecting because `SolverOptions` is the workspace's shared configuration vocabulary (§2.5) and `t_eval` was a public field whose contract was silently broken.

11. **2026-05-10** — Foundation Specification and Composability Contract published. Verification pass produced 35 findings against 21 capabilities; baseline state recorded as the §5 status note above. The two architecture documents are the project's first explicit commitment to the foundation/composability story; `docs/architecture/findings-2026-05-10-foundation-pass.md` preserves the pass record.

12. **2026-05-17 (commit `2ed799b`)** — F-SOLVER-FIELDS resolved per option (b). BDF order control (`max_order`, `min_order`) is now wired through `SolverOptions` as top-level `Option<usize>` fields with builders, replacing the silently-non-functional `Bdf::with_max_order` / `Bdf::fixed_order` instance builders whose fields were unreachable from the static `Solver::solve` trait method. Top-level shape was chosen over a per-family sub-struct as the spec-mandated reading of §2.5 (one configuration vocabulary); the `dense_output` / `events` precedent (top-level fields ignored by non-applicable solvers) was followed rather than establishing a novel sub-struct pattern speculatively. **Closes a documented §3.7 spec-vs-reality drift**: the spec previously claimed "BDF's max order" lives in `SolverOptions` while reality had it on the `Bdf` struct as a non-functional builder; the (b) wiring makes the claim become true and the spec's §3.7 paragraph has been rewritten to describe the now-real mechanism. The change is genuinely two distinct change-classes coupled in one PR for cohesion (per §2.8 visibility, each class is documented separately in the CHANGELOG): (i) *additive* — `SolverOptions::max_order` / `SolverOptions::min_order` fields and builders; `Bdf::solve` now reads them; (ii) *subtractive, technically breaking* — `Bdf::new`, `Bdf::with_max_order`, `Bdf::fixed_order`, `Bdf { max_order, min_order }` fields, `Auto` struct, `Auto::new`, `Auto::with_hints`, `impl Solver<S> for Auto`, `Auto::hints` field all deleted. All removed surface was silently non-functional under the pre-fix code path: every existing caller of the deleted Bdf builders got their order-control request silently ignored at solve time; `Auto::solve` discarded any user-supplied `SolverHints` (constructing `SolverHints::new()` internally regardless of what `Auto::with_hints(...)` was passed); the rustdoc example `solve_with_uncertainty::<Auto, f64>(...)` therefore used auto-detected dispatch regardless of user intent — which *can* produce inaccurate uncertainty-quantification results when the hints encoded problem structure that auto-detection's heuristics don't reconstruct (conditional, not unconditional: a user who never built hints was unaffected). No correct program depended on the deleted surface because no correct behavior was reachable through it. Migration: `Bdf::with_max_order(n)` → `SolverOptions::default().max_order(n)`; `Bdf::fixed_order(n)` → `SolverOptions::default().max_order(n).min_order(n)`; `Auto::solve(...)` / `Auto::with_hints(...)::solve(...)` → `auto_solve(...)` / `auto_solve_with_hints(...)`. The `auto_solve` / `auto_solve_with_hints` / `SolverHints` user-facing surface is unaffected. Option (c) (reshape `Solver::solve` to `&self`) was rejected as the resolution here but recorded as §7 #8 (a deliberate foundation-design alternative requiring explicit override of §2.5; not foreclosed by this change). Regression pinned by `numra-ode/tests/solver_options_order_regression.rs` (revert-confirm-restore protocol applied — temporarily reverting the wiring makes the test fail; restoring it makes the test pass).

13. **2026-05-17 (commit `98ee6ca`; release: `0.1.4` Unreleased)** — F-IC-SENS + F-AUTODIFF-JY landed in `numra-ode`. **F-IC-SENS (additive)**: first-class initial-condition sensitivity entry points `solve_initial_condition_sensitivity{,_with}` returning `StateTransitionResult<S>` (`Φ(t) = ∂y(t)/∂y₀` over any `Solver`). Implemented as a private wrapper that reinterprets a plain `OdeSystem<S>` as a `ParametricOdeSystem<S>` with `n_params := n_states`, `J_p ≡ 0`, `S(t₀) = I_N`, then delegates to the existing `solve_forward_sensitivity` / `AugmentedSystem` machinery — **no fork of the variational integrator**. The IC-NO-PARAM-READ invariant (the wrapped RHS provably does not read its dummy `params()` slice, so `∂f/∂p ≡ 0` is mathematically exact and the AugmentedSystem analytical-zero-J_p flag is truthful) is pinned by a NaN/∞-dummy-params test in `ic_sensitivity_regression.rs`. The variational-route choice is correctness given the actual codebase: the AD-through-y₀ alternative requires either making `Dual<f64>: Scalar` with dual-aware step-control norms across every solver (the existing `Dual<S>::PartialOrd` is primal-only, dropping the tangent — the STM — from norm-driven step adaptation), or a fork of the integrator; both are forbidden by §2.6 / §2.7 / the composability story for this scope. Foundation-affecting because §3.3's deferred "Jacobian enum (AutoDiff | Analytical | FD) would change the trait" question is **not** preempted by this entry: the AD adapter (see below) composes below the trait shape. On §3.3's adjacent call ("the autodiff path needs to be a foundational integration, not a per-crate retrofit"): the adapter is *not* a per-crate retrofit even on a broad reading of that sentence — it implements the foundation `OdeSystem<S>` trait uniformly, so every existing `Solver<S>` consumer of `OdeSystem` benefits from AD-derived `J_y` without per-crate plumbing. The per-crate retrofit risk §3.3 warns against concentrates on the gradient-source axis (multiple call sites in `numra-optim`, `numra-fit`, etc.), which §7 #1 is rewritten to keep explicitly open. **F-AUTODIFF-JY (additive, behind cargo feature `autodiff`)**: `AutodiffJacobianSystem<S, F>` adapter implementing `OdeSystem<S>` with a `Dual<S>`-typed RHS closure; per-step cost is **zero allocation** plus `N+1` Dual-RHS evaluations (same `O(N)` complexity class as the FD-default's `N+1` plain-RHS evals, ~2× per-eval constant for a round-off-exact Jacobian instead of `~sqrt(ε)`-truncated). Cargo-feature-gated so non-opting callers pay zero in compile time and binary size. **This entry resolves §7 #1** (autodiff integration shape) for the Jacobian-source axis: the answer committed is *per-system adapter (`OdeSystem<S>` impl bound to the system once at construction) made opt-in via cargo feature gate*. The three §7 #1 options collapse: adapter-as-`OdeSystem` is the API mechanism (a); per-system constructor is the entry-point shape (b); cargo feature is compile-time discoverability (c). §7 #1's narrower scope ("every Jacobian/gradient call site") is **not** fully closed: gradient-side wiring (numra-optim, numra-fit) remains open; only the `OdeSystem::jacobian` axis concretises. §7 #1 is rewritten in this commit to reflect the narrower remaining scope. The header SHA placeholder resolves via the §6 #12 two-commit pattern, per `feedback_spec_decision_log_anchor_pattern.md`'s fixed-point fix for the single-commit-with-amend chicken-and-egg: commit 1 (this commit) stamps the placeholder shown in the entry header above; commit 2 (a follow-up commit on the same PR) substitutes the merge SHA into that header's literal-token slot only. The substitution is contextual to the header line — the literal token appears exactly once in this entry, in the header, so identifying the substitution target is unambiguous (no global-replace footgun).

---

## 7. Open questions to resolve

These are foundation-relevant questions that don't have a current answer. Each needs a decision before the relevant capability is built.

1. **Autodiff integration shape — Jacobian-source axis resolved 2026-05-17, gradient-source axis open.** §6 #13 commits the *Jacobian-source* axis to *per-system adapter implementing `OdeSystem<S>`, opt-in via cargo feature gate* (`AutodiffJacobianSystem` behind `numra-ode`'s `autodiff` feature). The audit's Tier 1.1 narrower scope — gradient-side wiring across `numra-optim` (the existing `numra_autodiff::bridge::gradient_closure` is the bridge today; whether it is the durable shape, or whether optim's `OptimProblem::gradient` accepts a parallel `Dual<S>`-typed adapter, is the open question) and `numra-fit` (Jacobian-of-model-against-parameters is currently bridged by `numra_autodiff::bridge::model_jacobian_closure`; the same shape-decision applies) — remains open. The Jacobian-source axis's resolution gives the workspace a precedent for the gradient-side decision: adapter type + cargo feature is now the established shape. Re-evaluate whether the same shape carries to gradients, or whether gradients warrant a different shape (e.g., reverse-mode tape lifetime considerations).

2. **Top-level facade.** Audit item #38 — should there be a `numra::Problem` / `numra::Solve` umbrella? If yes, what's its shape?

3. **Mass-matrix DAE first-class promotion.** Currently DAE works in `Radau5`/`BDF`. Promoting it to first-class in the foundation requires `OdeSystem` to accept mass matrices as a non-default input or a `DaeSystem` peer trait. Which?

4. **Adjoint sensitivity API shape.** When adjoint is promoted from `numra-ocp`-private to public, what's the trait? Does it parallel `ParametricOdeSystem` or extend it?

5. **Sparse Jacobian return.** When sparse Jacobian is added, does `OdeSystem::jacobian` return `impl MatrixView` or does a new method appear? Related: should `SparseMatrix` join the `Matrix` trait (F-MATRIX-SHAPE)?

6. *(Closed by the verification pass.)* Workspace error type exists (`numra_core::NumraError`); the work item is `From`-impl coverage, tracked in `internal-followups.md` as F-ERR.

7. **GPU / parallelism strategy.** When parallelism is added, what foundation traits need a "where does this live" abstraction?

8. **`Solver::solve` reshape to `&self`.** Currently `Solver::solve` is a static trait method that cannot read implementor state. Per-solver state at solve time comes from `SolverOptions` (the centralized-configuration discipline of §2.5). An alternative shape — `fn solve(&self, problem, t0, tf, y0, options) -> Result<...>` — would let implementors carry instance state and let users build pre-configured solver values. This was option (c) of the F-SOLVER-FIELDS resolution (decision-log §6 #12); the (b) wiring resolved the immediate bug (silently non-functional `Bdf` order builders) without touching the trait shape, deliberately preserving the §2.5 centralized-configuration story. Adopting (c) is a *foundation-design* decision — it overrides §2.5 in favor of instance-shaped per-solver configuration as the organizing principle, which is a different design call from "add a knob." A future session that takes (c) on must justify the override against §2.5 explicitly. Until then, top-level `SolverOptions` fields remain the per-family extension mechanism (see §3.7).

Each open question becomes a focused design conversation when the relevant work is queued. None are blocked on each other; none are urgent.

---

## 8. What this document doesn't constrain

- **Capability surface.** Adding new solvers, new optimization methods, new statistical tests, new signal-processing primitives — these are not foundation changes if they implement existing foundational traits.
- **Internal implementation details.** A foundation trait's `impl` can be rewritten freely; the trait shape is what's stable.
- **Performance optimization.** Performance work that doesn't change trait shapes or public surfaces doesn't need foundation review.
- **Documentation, examples, tutorials.** Always welcome, never foundation-affecting.

---

## 9. How this document is maintained

- **Updated when foundation changes land.** Every PR that touches a foundational trait, type, or convention updates this document as part of the same PR. Reviewers verify the update is present.
- **Reviewed quarterly.** Once per quarter, walk the workspace and verify the document still reflects reality. Surface drift.
- **Open questions promoted to decisions.** When an open question is resolved, the decision moves to §6 and the question is removed from §7.

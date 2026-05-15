# Internal follow-ups

Local-only tracking of work that's been considered, scoped, or partially
built, but isn't appropriate to publish on the public roadmap. Not linked
from the website. Not for marketing. The point is to have a written record
that survives session-context churn so we can pick threads back up cleanly.

When you act on something here, prefer moving it into a CHANGELOG entry,
a closed GitHub issue, or the public roadmap — and remove it from this
file once it lands. Stale follow-ups files are how good intentions become
embarrassments.

Last updated: 2026-05-15 (F-CI-NODE20 retired; F-FD-STEP retired; F-FD-CROSSCRATE and F-FD-NOSCALE-BUG added).

## Recently retired

One-line entries for follow-ups that landed and were removed from the
file. Kept here so a future reader can find the closure record without
git-archaeology.

- **F-CI-NODE20: upgrade GitHub Actions runners to Node.js 24** — shipped 2026-05-15. Five action majors bumped to versions declaring `runs.using: node24`, clearing the 2026-06-02 deprecation deadline ahead of time: `actions/checkout@v4 → @v6` (11 usages), `actions/setup-node@v4 → @v6` (7), `actions/upload-artifact@v4 → @v7` (4), `pnpm/action-setup@v3 → @v6` (4; also dropped redundant `with: version: 9` and deferred to the `packageManager: pnpm@9.15.0` field in `package.json` as the single source of truth — required for the v4+ strict check), `cloudflare/wrangler-action@v3 → @v4` (3; default wrangler version implicitly upgrades v3 → v4 — `pages deploy` syntax is stable across the bump). Actions already on node24-runtime majors (`treosh/lighthouse-ci-action@v12`, `Swatinem/rust-cache@v2`) left at their current pins per scope discipline; composite actions (`taiki-e/install-action`, `dtolnay/rust-toolchain`, `rhysd/actionlint`) not affected by Node-runtime deprecation. Audit surfaced 4 actions missed from the original entry (`upload-artifact` needs bump; the three composites and lighthouse/rust-cache don't) — same audit-surfaces-more-than-named pattern as F-FD-STEP.
- **F-FD-STEP: foundation-trait FD-step reconciliation** — shipped 2026-05-15. `OdeSystem::jacobian` default switched from hardcoded `1e-8` to `sqrt(S::EPSILON) * (1 + |y_j|)`; `Signal::eval_derivative` default switched from hardcoded `1e-8` (no scaling) to `cbrt(S::EPSILON) * (1 + |t|)` (canonical central-FD step). `ParametricOdeSystem::jacobian_y/_p` defaults were already correct on `sqrt(S::EPSILON)`; no change. The six `MOLSystem{2,3}D::jacobian` and `ParametricMOLSystem{2,3}D::jacobian_y/_p` reaction-FD diagonals updated in lockstep (each referenced the trait default in code comments — preserving the consistency the comments claim required moving them together). Two `f32` regression tests added (`numra-ode/src/problem.rs::test_jacobian_finite_diff_f32`, `numra-core/src/signal.rs::test_signal_derivative_f32`) pinning out the silent-quantisation failure mode. The audit pass also surfaced two adjacent follow-ups that were explicitly out of scope for F-FD-STEP — see F-FD-CROSSCRATE and F-FD-NOSCALE-BUG below.
- **F-ERR: workspace error propagation across crate boundaries** — shipped 2026-05-14. `NumraError` (`numra-core/src/error.rs`) is now `#[non_exhaustive]` and gains 10 new tagged variants (`Ode`, `Optim`, `Ocp`, `Fit`, `Signal`, `LineSearch`, `Interp`, `Integrate`, `Special`, `Stats`); `Optimization` renamed to `NumericalOptim` for unambiguity against the new `Optim`. Six new `From<CrateError> for NumraError` impls in `numra-ode`, `numra-optim`, `numra-ocp`, `numra-fit`, `numra-signal`, `numra-nonlinear`; the four pre-existing impls (`InterpError`, `IntegrationError`, `SpecialError`, `StatsError`) migrated from collapsing-to-`InvalidInput` to the tagged-variant pattern, so the workspace error story is now uniform: every external-crate `?` lands in a programmatically-distinguishable variant. `workflow_ode_interp_integrate` rewritten to return `Result<(), NumraError>` and use `?` across three crate boundaries — structural CI signal that the property is real. Closes Foundation Specification §7 open-question 6 (workspace error type sufficient for cross-crate `?` propagation: yes). Also advances F-INTEROP-Q sub-item 1 (an interop test now exercises cross-crate `?`-propagation). Source-chain preservation deferred to F-ERR-CHAIN.
- **Parametric MOL systems for forward sensitivity** — shipped 2026-05-08. `ParametricMOLSystem2D` and `ParametricMOLSystem3D` (numra-pde) wrap the heat-equation MOL discretisation as `ParametricOdeSystem`. Parameter layout `[α, reaction_p_0, ...]`; analytical state Jacobian (`α · L0` + diagonal reaction FD), analytical α-column of `J_p` (`L0·y + bc_rhs_0`), all four flag overrides set. Linearity of the Laplacian operator means a single pre-assembled `L0` and `bc_rhs_0` cover both Dirichlet and Neumann BCs without splitting. v1 scope is alpha-on-Laplacian only; full operator parametrisation (D + velocity in advection-diffusion) is the remaining gap, narrowed below.
- **Jacobian unification & MOL analytical Jacobians** — shipped 2026-05-07. `Radau5` and `Bdf` now route through `OdeSystem::jacobian`; FD step formula unified to `eps * (1 + |y_j|)`; `MOLSystem2D` / `MOLSystem3D` override with CSC-to-dense copy of the spatial operator + diagonal-FD reaction term. ~2.8% measured win on stiff 2D heat-with-reaction; full neutrality on non-MOL systems. See CHANGELOG.
- **`equations3d` convenience builders** — shipped 2026-05-07. `HeatEquation3D::build`, `AdvectionDiffusion3D::build`, `ReactionDiffusion3D::{build, fisher}` mirroring `equations2d.rs`.
- **`MOLSystem3D` wrapper** — shipped 2026-05-07. `mol3d.rs` mirrors `mol2d.rs` exactly; backed by new `Operator3DCoefficients` and `assemble_operator_3d`. The audit-driven 3D-MOL gap is closed.
- **Forward-sensitivity API** — shipped 2026-05-06. `ParametricOdeSystem` trait, `solve_forward_sensitivity{,_with}`, `AugmentedSystem`, `SensitivityResult`, regression suite, Criterion harness, three perf figures, Robertson example, ch11-uncertainty/sensitivity-analysis chapter rewrite.

---

## Audit-driven scope corrections

These are not new work items — they're records of where earlier roadmap
drafts overstated or mis-scoped what we actually have. Kept here so the
next time someone drafts a public commitment based on memory rather than
the code, they catch the same correction without redoing the audit.

### "PDE: add 2D method-of-lines + finite element"

**The original draft claim**: ship 2D MOL plus FE in a Q1-2027-ish bucket.

**What the audit found** (2026-05-05; updated 2026-05-07):
- 2D MOL **already ships** — `numra-pde/src/mol2d.rs`, `equations2d.rs`
  with `HeatEquation2D`, `ReactionDiffusion2D`, Fisher, advection-diffusion;
  sparse 5-point Laplacian in `sparse_assembly.rs`; all four BC types
  tested.
- 3D MOL **also now ships** (added 2026-05-07) — `numra-pde/src/mol3d.rs`
  provides `MOLSystem3D::heat` / `::laplacian` / `::with_operator` /
  `::with_reaction`, backed by the new `Operator3DCoefficients` +
  `assemble_operator_3d` (general 7-point stencil with first-derivative
  central-difference terms). `assemble_laplacian_3d` now delegates to
  the general assembler, mirroring the 2D file's organisation. No
  `equations3d.rs` convenience layer yet — multi-component coupled 3D
  PDEs ("HeatEquation3D", "ReactionDiffusion3D" wrappers) remain on the
  list below.
- `Wave1D` is a documented stub (`equations.rs:129-151`) — a struct
  exists with a comment that it "would need the `PdeSystem` trait to
  support systems".
- **No FE infrastructure at all.** No spectral. No elliptic static solver.

**Correction**: "FE" was aspirational, not a gap to slot into a roadmap
without scoping the algorithm choice (CG-FEM? DG? hp-adaptive?). The
real, concrete PDE gaps to keep on the public roadmap as one bullet
("Expand PDE capabilities") are now: multi-component coupled PDEs
(hyperbolic / wave systems become first-class, including a real `Wave1D`),
3D `equations3d.rs` convenience constructors mirroring `equations2d.rs`,
and an elliptic static solver path. FE is either a separate multi-quarter
project or a deliberate "not now".

### "DAE: index-2+ in Q4"

**The original draft claim**: ship index-2+ DAE solving as a Q4 2026
deliverable.

**What the audit found** (2026-05-05):
- Index-1 ships in Radau5 / BDF (`numra-ode/src/radau5.rs:15-18`,
  `bdf.rs:16-21`).
- **Pantelides structural analysis and symbolic differentiation already
  ship** in `numra-ode/src/index_reduction.rs:118-768`. Tests live in
  `index_reduction_regression.rs` and exercise the algorithm itself.
- Consistent-IC helper ships in `numra-ode/src/dae_init.rs:51-159`,
  Newton-solving for algebraic variables on user-supplied initial guesses.
- What's **missing** is automatic invocation: solvers don't call
  `reduce_index()` or `compute_consistent_initial()` for the user, there's
  no automatic Baumgarte stabilisation, no sparse-Jacobian DAE path, no
  time-dependent mass matrices.

**Correction**: "Index-2+" was technically inaccurate — the math ships;
the ergonomics don't. The honest framing for "Expand DAE capabilities"
on the public roadmap is **plumbing-and-ergonomics work**, not new
algorithmic capability: solvers should detect higher-index input,
auto-reduce via the existing Pantelides pass, and pre-solve consistent
ICs without the user wiring it together by hand.

---

## Solvers

### Adjoint sensitivity

**Status**: 0%. Greenfield.

**What's there today**:
- Reverse-mode AD scaffolding in
  `numra-autodiff/src/reverse.rs` — works for scalar functions, no
  composition with ODE solvers.

**What needs doing**:
- Tape-based adjoint pass through the integration: forward solve, then
  back-solve the adjoint ODE with the right-hand side ∂L/∂y ·
  (∂f/∂y)ᵀ. Checkpointing for memory cost.
- Integration with `numra-autodiff` so users can write a loss function
  in normal Rust and get gradients w.r.t. ODE parameters.
- Stretch: continuous adjoint vs. discrete adjoint trade-off
  documentation.

**Why it's not on the public roadmap**: speculative work. Don't promise
adjoint until at least the forward-mode API is shipped and we know what
shape the adjoint API should take.

### Block-diagonal-aware factorisation in `AugmentedSystem`

**Status**: scoped, not started. Now unblocked — the
forward-sensitivity API has shipped.

The augmented Jacobian for forward sensitivity is
`block_diag(J_y, J_y, ..., J_y)` (CVODES *simultaneous-corrector* form,
`N_s + 1` diagonal blocks). With block-aware LU, an implicit step
needs a *single* factorisation of `M = (1/γh)·I_N - J_y` reused across
all `N_s + 1` sub-systems instead of one factorisation of the full
`N(N_s+1) × N(N_s+1)` matrix. Expected speedup is `O((N_s+1)²)` on the
LU step, which dominates Radau5 / BDF cost on stiff problems.

`numra-ode/src/sensitivity.rs::AugmentedSystem::jacobian` already
fills only the diagonal blocks, so the saving is sitting on the table
— it just requires Radau5 / BDF to either (a) detect block structure
in the Jacobian buffer, or (b) accept a `BlockDiagonalJacobian` hint
from the augmented-system path. Option (b) is cleaner: add a tagged
trait `OdeSystemBlockDiag` (or a method on `OdeSystem` returning
`Option<BlockSpec>`) and let solvers downcast.

Out of scope for v1 (the forward-sensitivity API itself has just
shipped); revisit once it has real users with measurable workloads.

### JVP-based variant of `ParametricOdeSystem`

**Status**: scoped, not started.

`diffsol`'s analogue (Robinson et al., JOSS 11(117), 2026) uses
Jacobian-vector products instead of materialised Jacobians, which
unlocks matrix-free Krylov solvers and sparse Jacobians without ever
forming `J_y`. For Numra's small-to-medium dense workloads, the
materialised path is the right v1 choice, but a `ParametricOdeSystemJvp`
companion trait (or a `jacobian_vec_product(t, y, v, out)` method on
the existing trait, with a default that materialises `J_y`) would be
the right shape for large sparse problems. Out of scope for v1.

### AD-based `ParametricOdeSystem` impl using Numra's autodiff primitives

**Status**: scoped, not started.

SciML's `ForwardDiffSensitivity` is the modern default — eliminates
manual Jacobian derivation entirely. We have reverse-mode AD scaffolding
in `numra-autodiff/src/reverse.rs` and forward-mode dual numbers
elsewhere in `numra-autodiff`; a thin adapter `AutoDiffSystem<F>`
implementing `ParametricOdeSystem` and computing `J_y` / `J_p` via
forward-mode AD on the closure would close most of the FD-noise
problems users hit. The v1 API has now stabilised and the Robertson
example has landed; this is unblocked but still out of v1 scope —
revisit once we have a workload that justifies the
`numra-autodiff` ↔ `numra-ode` integration cost.

### Staggered sensitivity correction (CVODES `CV_STAGGERED`)

**Status**: scoped, not started.

CVODES exposes two correction strategies: *simultaneous* (state and
sensitivities solved in one Newton iteration on the augmented system,
which is what v1 ships) and *staggered* (state Newton converges first,
then sensitivities re-use the converged factorisation). Staggered is
2-3× faster on stiff problems with `N_s ≥ 3` because the sensitivity
sub-systems become *linear* once the state converges. SciML defaults
to simultaneous, which is what v1 matches. Worth implementing for the
stiff path once block-aware factorisation lands; out of scope for v1.

### Per-parameter staggered correction (CVODES `CV_STAGGERED1`)

**Status**: scoped, not started.

A finer-grained variant of the above: each `S_{:,k}` Newton runs
independently, which lets sparsity-aware code skip parameters with
empty `J_p_{:,k}` columns. Useful for very large parameter sets with
sparse coupling (PDE-discretised models). Out of scope for v1.

### Separate sensitivity tolerances

**Status**: scoped, not started.

Parameter-ID workflows often want loose sensitivity tolerances (because
the gradient is consumed by an outer Gauss-Newton step that is itself
sloppy) while keeping the state tolerance tight. CVODES exposes
`CVodeSensSStolerances`. v1 inherits state tolerances throughout
(matches SciML's default and avoids polluting `SolverOptions`); add
`SolverOptions::sens_rtol` / `sens_atol` later if real workflows ask
for it.

### `ParametricOdeSystem` analytical-Jacobian flag ergonomics

**Status**: scoped, not started. Debug-build safety net shipped in the
forward-sensitivity v1 to mitigate the immediate footgun.

The current trait requires overriding both the Jacobian method *and* a
boolean flag (`has_analytical_jacobian_y` / `_p`) — and silently falls
back to FD when the flag is left at its `false` default. This is a
silent-misconfiguration class of bug: failing users see "Numra is slow
on my problem" with no diagnostic path. v1 ships with a debug-build
consistency check inside `AugmentedSystem::rhs` that compares the
user's `jacobian_*` output to inline FD on the first call and panics if
they diverge beyond a 1e-3 relative threshold (catches the
"forgot the flag entirely" case where analytical and FD differ by O(1)
without false-positive on legitimate FD-vs-analytical disagreement).
Release builds skip the check.

The check is a safety net, not an API fix. The contract is still
"override both," and the contract itself is what makes the bug
possible. Future options to consider when we revisit:

- **Procedural macro `analytical_jacobians!`** — wraps the impl block,
  emits both the method and the flag override from a single declaration
  site. Lowest friction; opaque to callers reading the trait.
- **Typestate phantom markers** — `ParametricOdeSystem<S, JyKind = FdJy>`
  with `AnalyticalJy` as a separate type. Catches the bug at compile
  time; doubles the trait's surface area; complicates `dyn`-friendly
  usage.
- **Trait redesign** returning `Result<(), FDFallback>` from
  `jacobian_y` so "no analytical override" is an explicit return value.
  Most rigorous; biggest breaking change; interacts awkwardly with
  default-impl FD.

All three are breaking changes. Out of scope for v1; revisit after the
public-flip release stabilizes the trait surface (post-1.0).

### Stiffness auto-detection (LSODA-equivalent)

**Status**: scoped, not started.

`auto_solve_with_hints` exists but the heuristic is mostly user-supplied
hints. SciPy's LSODA is the gold standard for "automatic stiffness
detection during the integration", and the §2.3 comparison page
acknowledges this is where SciPy beats Numra for first-time users picking
the wrong solver. Worth doing as a Numra-native algorithm rather than
porting LSODA.

**Increased value as of 2026-05-07**: the Jacobian-unification work
(MOL systems supplying analytical Jacobians, both stiff solvers
consuming them automatically) widened the explicit-vs-implicit cost
asymmetry on PDE workloads. Previously, picking `Radau5` over
`DoPri5` on a stiff PDE meant trading explicit-step efficiency for
the FD-Jacobian penalty; that penalty is now gone for MOL systems,
so the implicit path is uniformly preferable on stiff PDEs. A user
who picks the wrong solver loses more wall-clock today than they did
before — auto-detection has more to recover.

### F-LEAK: Generify foundational code over `Scalar`

**Status**: scoped, not started. Surfaced 2026-05-10 by the
foundation-pass verification (finding B1).

**What's there today**: three concrete-`f64` leaks in foundation-
adjacent code that block the `Scalar` story end-to-end:

1. `numra-autodiff` reverse mode — `tape::Tape::var(value: f64)`
   (`tape.rs:67`), `Tape::gradient` (`tape.rs:122`),
   `Tape::jacobian` (`tape.rs:149`); `reverse::Var::cst(f64)`
   (`reverse.rs:57`), `Var::powf(f64)` (`reverse.rs:212`); `grad`,
   `jacobian_reverse`, `hessian` free functions
   (`reverse.rs:572, 599, 630`). Forward-mode `Dual<S>` is already
   generic; only reverse mode is concrete.
2. `numra-linalg` general (non-symmetric) eigendecomposition — split
   into `impl EigenDecomposition<f64>` (`eigen.rs:63`) and
   `impl EigenDecomposition<f32>` (`eigen.rs:90`); no generic blanket.
   Cause: faer's `complex_native::c64`/`c32` for the underlying
   Hessenberg routine. External `Scalar` impls get nothing.
3. `numra-signal` filter design — `butter(order: usize, cutoff: f64,
   fs: f64) -> Result<SosFilter<f64>, _>` (`filter_design.rs:33`);
   `instantaneous_frequency<S: Scalar>(x: &[S], fs: f64)`
   (`hilbert.rs:104`) leaks `fs` through.

**What needs doing**: generify each site over `S: Scalar`, or, where
the underlying engine forces concretion (eigen/faer), document the
faer constraint in rustdoc and consider a `from_f64`/`to_f64` adapter
for the API surface so generic pipelines aren't broken at the boundary.

Recorded as one consolidated follow-up rather than three separate
entries because the three sites share the same generification approach
and benefit from being addressed as a single sweep.

### F-FD-CROSSCRATE: Reconcile FD step formula across non-foundation cross-crate sites

**Status**: scoped, not started. Surfaced 2026-05-15 by the F-FD-STEP
audit pass (Step 1, Decision B envelope).

**What's there today**: 13 cross-crate FD bodies that work correctly at
`f64` (each picks a reasonable step), but use a mix of three formulas:

- **Forward + additive `(1+|x|)` (foundation-shape)** at `1e-7` or `1e-8`:
  `numra-nonlinear/src/newton.rs:245`, `numra-ode/src/esdirk.rs:525`,
  `numra-ode/src/auto.rs:168`, `numra-ocp/src/adjoint.rs:82,109,128,147,163`,
  `numra-ode/src/dae_init.rs:105`,
  `numra-ode/src/index_reduction.rs:407,788`.
- **Central FD with multiplicative `orig*eps` scaling** (the
  `numra-fit::curve_fit` pattern) at `1e-7`:
  `numra-fit/src/curve_fit.rs:80,187,489`.
- **Central FD with additive `(1+|x|)` scaling**:
  `numra-core/src/uncertainty.rs:278` (`compute_sensitivities`,
  `1e-7`).

None of these are bug-class (no `f32` quantisation; large-`|x|`
behaviour acceptable because of the additive scaling on the foundation-
shape sites). The work is purely consistency: pick the canonical
forward / central formulas (`sqrt(EPSILON) * (1 + |x|)` /
`cbrt(EPSILON) * (1 + |x|)`) and apply them workspace-wide so a future
reader doesn't see four different FD-step recipes scattered through
the codebase.

**What needs doing**:
1. Audit each site against the chosen canonical formula (forward FD
   on the additive-scaling sites; central FD on the others).
2. Replace the hardcoded `1e-7` / `1e-8` constants with
   `S::EPSILON.sqrt()` or `S::EPSILON.cbrt()` per the FD direction.
3. Run the workspace test suite — these sites are all in algorithm
   internals, so any regression will surface as a test failure rather
   than a silent behavioral change.
4. Estimate: 1-2 focused days. The work is mechanical.

**Why not bundled with F-FD-STEP**: F-FD-STEP closes the foundation-
trait reconciliation (the only sites where divergence was a
bug-class issue). These cross-crate sites are correct today; touching
them is consistency hygiene. Different commitment, different risk
profile, different PR.

### F-FD-NOSCALE-BUG: Fix no-scaling correctness bug in public FD utilities

**Status**: scoped, not started. Surfaced 2026-05-15 by the F-FD-STEP
audit pass.

**What's there today**: five FD sites use `h = eps` with **no `(1 + |x|)`
scaling**, so the perturbation is dimensionally wrong for any caller
with large-magnitude state:

- `numra-optim/src/problem.rs:486` (`finite_diff_gradient`, **public API**)
- `numra-optim/src/problem.rs:503` (`finite_diff_jacobian`, **public API**)
- `numra-optim/src/robust.rs:409,446` (worst-case-param FD; private
  helpers in robust optimisation)
- `numra-dde/src/history.rs:188` (`History::evaluate_derivative` for
  initial-history regions; **public API**)
- `numra-sde/src/system.rs:68` (`SdeSystem::diffusion_derivative`
  default for Milstein; **public API trait default**)

For each, with `eps = 1e-8` and any state component `|x| > 1e6`, the
ratio `h/x = 1e-14` falls below `f64::EPSILON`, so the perturbation is
absorbed by the addition `x + h = x` and the FD comes back as
`(f(x) - f(x))/h = 0`. This is independent of the `f32` framing in
F-FD-STEP — `f64` callers with large state see the same silent zero.

**Why this is a bug, not a style issue**: a user's optimisation problem
with parameters in the `1e6` range silently gets an all-zeros gradient
back from `finite_diff_gradient`. No error, no diagnostic — just a
useless result that the optimiser then dutifully consumes. Same shape
on the other four sites.

**What needs doing**:
1. Replace each `h = eps` with `h = eps * (S::ONE + x.abs())` (forward-
   FD additive scaling) or `h = eps * (S::ONE + |x|)` adapted to
   central FD per site.
2. While there: also bring the eps onto the canonical
   `cbrt(S::EPSILON)` (central) or `sqrt(S::EPSILON)` (forward) per
   F-FD-CROSSCRATE — this work overlaps but is bounded narrowly here
   to the no-scaling sites where the consequence is correctness.
3. Add a regression test for each public-API site: run with `x = [1e6,
   1e6]` (or analogous), assert FD output is non-zero. The shape of
   the test mirrors F-FD-STEP's `f32` regression tests.
4. Estimate: 1-2 focused days plus a correctness-review pass.
   `numra-optim`'s public `finite_diff_*` deserve careful test
   coverage given downstream optimisers consume them.

**Why not bundled with F-FD-STEP or F-FD-CROSSCRATE**: F-FD-STEP was
foundation-trait reconciliation; F-FD-CROSSCRATE is "make existing
correct code consistent." This is "fix existing incorrect code." Each
is a different commitment with a different risk profile. Bundling
would dilute all three — the closure narratives and the review focus.

### F-SOLVER-FIELDS: Clarify or remove `Bdf::max_order` / `Auto::*` fields the static `Solver::solve` cannot read

**Status**: scoped, not started. Surfaced 2026-05-10 by the
foundation-pass verification (finding C5).

**What's there today**: `Solver::solve` (`numra-ode/src/solver.rs:291`)
is a static method (no `&self`). `DoPri5`, `Tsit5`, `Vern6/7/8`,
`Radau5`, `Esdirk32/43/54` are zero-size unit structs and don't carry
state — fine. But:

- `Bdf { max_order: usize, min_order: usize }` (`numra-ode/src/bdf.rs:83`)
  has fields, with builder methods `Bdf::new()`, `Bdf::with_max_order(...)`,
  `Bdf::fixed_order(...)`. The fields cannot be read from inside the
  static `Solver::solve` because no `&self` is passed.
- `Auto { ... }` (`numra-ode/src/auto.rs:119`) — same shape.

So either (a) the fields are dead code, (b) the builders are sketched
but the trait method needs a reshape to `fn solve(&self, ...)` to use
them, or (c) there's an internal adapter I missed.

**What needs doing**: investigate and either delete the fields/builders
(if dead), wire them through `SolverOptions` (if they encode caller
preferences), or reshape `Solver::solve` to take `&self` (if the fields
encode genuine per-solver state). Option (c) is the most foundation-
affecting; pin the design before changing the trait.

---

## PDE

These follow-ups were surfaced during the `MOLSystem3D` landing
(2026-05-07). They apply equally to `MOLSystem2D` — the 3D wrapper
mirrors the 2D wrapper exactly, so neither dimension regresses
relative to the other, but both share the same gaps relative to the
broader Numra solver surface. Listed here so that when we close them,
we close them for both wrappers in the same PR.

### Full operator parametrisation in MOL

**Status**: scoped, not started. v1 (heat-equation parametrisation,
single α slot) ships in `ParametricMOLSystem{2,3}D`; this entry tracks
the remaining gap.

`ParametricMOLSystem*` v1 supports parametrising the diffusion
coefficient α and any number of reaction parameters, but the operator
itself is restricted to a *scaled Laplacian* — all interior coefficients
share a single α factor. This is sufficient for the most common
workloads (heat, Fisher-KPP, Allen-Cahn). It is *not* sufficient for
problems where multiple operator coefficients are independently
parametric:

- **Advection-diffusion** with parametric diffusion `D` and parametric
  velocity `(v_x, v_y)`. The operator is `D · ∇² - v_x ∂_x - v_y ∂_y`,
  not linear in any single coefficient.
- **Anisotropic diffusion** with `D_x ≠ D_y`, where two independent
  diffusion parameters parameterise different stencil entries.
- **Cross-diffusion / reaction-diffusion-advection** with multiple
  independent operator parameters.

**What needs doing**: extend the parametric MOL surface to support a
user-supplied `Operator2DCoefficients = f(params)` mapping. Each rhs /
jacobian call re-assembles the operator from the current parameters.
Two implementation paths:

1. **Re-assembly per call**: simplest. Costs one full
   `assemble_operator_2d` per parameter change. Cheap on small grids
   (`n_int < 1000`) where assembly is microseconds; expensive on large
   grids. The hot path of `solve_forward_sensitivity` calls
   `jacobian_p` infrequently (once per Jacobian rebuild), so the cost
   may be tolerable in practice — needs benching.
2. **Coefficient-decomposition**: pre-assemble the operator components
   once (`L_xx`, `L_yy`, `L_x`, `L_y`, `M_0`) and combine at runtime as
   `α · L_xx + β · L_yy + ...`. Mirrors the linearity exploitation in
   the v1 alpha-on-Laplacian path. Requires `assemble_operator_2d` to
   support per-component disassembly — a non-trivial refactor of the
   stencil construction code.

Path 1 is the fast lift for the next user; path 2 is the
performant-at-scale answer.

**Composes with**: §Solvers / AD-based `ParametricOdeSystem` impl —
once that adapter ships, the user-supplied
`Operator2DCoefficients = f(params)` mapping could be AD-differentiated
to give `∂coeffs/∂params` automatically, removing the manual
parameter-Jacobian derivation. Combined with path 2 above, the result
would be analytical `J_p` for any user-specified linear operator.

**Out of scope for v1 of the parametric MOL work**; revisit when a
real workload (advection-diffusion identification, anisotropic
diffusion estimation) materialises.

### 1D `MOLSystem` analytical Jacobian

**Status**: scoped, not started. Affects `numra-pde/src/mol.rs`.

The 2D and 3D MOL wrappers store the assembled spatial operator as a
`SparseMatrix<S>` and override `OdeSystem::jacobian` with a direct
copy from CSC. The 1D `MOLSystem` is different — it uses the
`PdeSystem` trait + `FDM` discretisation and does not pre-assemble
an operator. When a stiff solver picks 1D MOL, it pays full FD-
Jacobian cost even though the underlying derivative structure (band-
3 tridiagonal + diagonal reaction) is implicit in the discretisation.

**Recommended approach when picked up**: cache the assembled
tridiagonal operator in a `OnceLock<SparseMatrix<S>>` field on
`MOLSystem`, populated on first `jacobian()` call. Subsequent calls
are zero-cost copies. The `OnceLock` choice over `RefCell`/`Mutex`
keeps the type `Send + Sync` without unsafe code and avoids the
mutex contention path that would otherwise show up in parallel
solver use; the trade-off is that the operator is built lazily once
per `MOLSystem` rather than upfront — measurable only on cold solves.

The reaction Jacobian inherits the same diagonal-FD pattern as 2D/3D
(see "Non-pointwise reaction Jacobians" below for when that
assumption breaks). Don't relitigate the cache-strategy decision when
the work is picked up.

### Non-pointwise reaction Jacobians

**Status**: scoped, not started. Triggered when (and only when) a
user wants a reaction term that *does* couple grid points.

The diagonal-FD reaction-Jacobian path that ships in the current
`MOLSystem{2,3}D::jacobian` is rigorously correct for *pointwise*
reactions: `R(t, x_i, ..., u_i)` depends only on local state, so
`∂R_i/∂u_m` is identically zero for `m ≠ i`. The diagonal property
holds by construction, not by approximation.

**The remaining gap is for non-pointwise reaction models**:
- **Nonlocal coupling** — reactions of the form `R(t, x, u(x), ∫K(x,x')u(x')dx')`
  appearing in chemotaxis, neural field equations, and some predator–prey
  spatial models.
- **Integro-PDE reactions** — `R` is itself a spatial integral of `u`.
- **Multi-component PDEs** — the reaction in component `i` depends on
  components `j ≠ i` at the same grid point (still local in space,
  but not pointwise in the `MOLSystem*` sense which is single-component).

For these, the current `Fn(t, x, y, z, u) -> S` closure signature is
insufficient and the diagonal-only Jacobian path is wrong. Two
sub-cases need different treatment:
- **Multi-component pointwise** is the easy lift: extend the closure
  to `Fn(t, x, y, z, &[S]) -> Vec<S>` and treat the local Jacobian as
  a `c × c` block at each grid point (where `c` is the number of
  components). The block is supplied either by sibling closures or
  by a trait. Diagonal-block analytical fall-through preserves the
  `O(1)` rebuild cost.
- **True nonlocal reactions** require the trait-default full FD
  fallback in `OdeSystem::jacobian`. Document the constraint in the
  MOL chapter; a user can always override `OdeSystem::jacobian` on
  a custom wrapper.

**API decision** to make when revisiting: closures vs. a trait.
Closures are non-breaking additions (sibling closures
`with_reaction_jacobian_u`, etc.); a `trait Reaction<S>` is more
rigorous, bigger surface area, and consistent with how
`ParametricOdeSystem` already works (defaulted FD impls for
non-overriders). Likely the right call is a trait once the
multi-component requirement is concrete.

**Why this is filed narrowly now**: the broader "reaction-term
composability" framing has been retired — the diagonal-FD path
makes the v1 pointwise case painless. The remaining specific gap is
the non-pointwise case, and there's no concrete user demand for it
yet, so revisit when one materialises.

---

## API

### First `cargo publish` of every crate (Step 7)

**Status**: blocked on the user's launch decision.

Already in the existing `binary-squishing-puffin.md` plan as Step 7 —
DOI mint, Zenodo enable, GH release, back-port DOI to `CITATION.cff` and
`cite.astro`. Tracked here only as a reminder that this is the gate
between "private repo" and "the website's `/api` page can stop linking
to local-only `cargo doc`".

### `0.2.0` minor release exercising the deprecation window

**Status**: not started.

The `/stability` page promises a deprecation cycle. Until we actually
*do* a deprecation cycle, that promise is unverified. Pick one
sub-optimal API choice (candidate: the `SolverOptions` builder, which
mixes f64 and S generic awkwardly) and run it through deprecation →
removal across two minor versions. Validates the policy in practice.

### F-ERR-CHAIN: Source-chain preservation across `NumraError` variants

**Status**: scoped, not started. Surfaced 2026-05-14 during the F-ERR
audit (Decision B). Out of F-ERR scope by design — F-ERR was bounded
to the `From`-impl coverage; source-chain preservation is greenfield
work and warrants its own design pass.

**What's there today**: `NumraError` (`numra-core/src/error.rs`) ships
ten new tagged variants (`Ode(String)`, `Optim(String)`, etc.) but
**every external-crate variant is stringified**. The `impl
std::error::Error for NumraError {}` block is empty — no `source()`
override; no `#[source]` annotations on any variant. A user calling
`err.source()` on a `NumraError` produced by `?` from a `SolverError`
gets `None`, even though the originating error is logically the
parent. The current pattern matches what shipped pre-F-ERR (the four
pre-existing external-crate impls also flattened); F-ERR continued the
pattern rather than inventing a new one mid-flight.

**The structural blocker**: `NumraError` derives `Clone + PartialEq`,
which prevents the standard `Box<dyn std::error::Error + Send + Sync>`
field idiom for source-chain preservation (`dyn Error` is neither
`Clone` nor `PartialEq`). Adding source preservation requires either
(a) dropping the derives (breaking, with downstream-impact unknowable
beyond what the F-ERR cliff already cost), or (b) per-variant Boxing
with manual `Display` / `PartialEq`-via-string delegation (workable
but cumbersome and asymmetric).

**What needs doing**:
1. Decide between (a) drop derives and adopt structural source
   preservation idiomatically, or (b) keep derives and use a
   per-variant Box-with-manual-delegation pattern, or (c) accept the
   stringified status quo and document the limitation in `NumraError`'s
   rustdoc as "source chains are not preserved; use the carried
   `String` for diagnostic detail".
2. If (a) or (b): add `#[source]` annotations / `source()` override;
   update each `From` impl to carry the originating error rather than
   stringify; pin with a regression test that `NumraError::Ode(...)`
   produced by `?` from a `SolverError` returns `Some(SolverError)`
   from `.source()` (downcast-checked).
3. If (a): paying the second breaking-change cliff in 0.1.x — likely
   defer to 0.2.0 and bundle with other breaking changes.

This composes with the broader "richer error context" question
(backtrace, location info) that was explicitly out of scope for F-ERR.
Worth doing as one design pass.

### F-SENDSYNC: Audit & remove defensive `Send + Sync` on foundation traits

**Status**: scoped, not started. Surfaced 2026-05-10 by the
foundation-pass verification (finding B8 / C2 / C6).

**What's there today**: defensive `Send + Sync` bounds on five
foundation/quasi-foundation traits, plus closure-type-alias virality
in PDE/OCP:

- `Scalar: ... + Send + Sync + 'static` (`numra-core/src/scalar.rs:58–60`).
- `Signal<S>: Send + Sync` (`numra-core/src/signal.rs:51`).
- `EventFunction<S>: Send + Sync` (`numra-ode/src/events.rs:66`). Note
  that `Arc<dyn EventFunction<S>>` is used in `SolverOptions::events`
  but `Arc<T>` doesn't require `T: Send + Sync` unless the `Arc` is
  sent across threads.
- `NonlinearSystem<S>: Send + Sync` (`numra-nonlinear/src/newton.rs:100`).
- `SdeSystem<S>: Sync` (`numra-sde/src/system.rs:27`) — asymmetric
  (only one of the two bounds).
- Closure aliases requiring `+ Send + Sync + 'static` in
  `numra-pde/src/{mol2d.rs:19,84, mol3d.rs:19,84, equations2d.rs:58,
  equations3d.rs:60, mol2d_parametric.rs:44,96,
  mol3d_parametric.rs:17,57}` and `numra-ocp/src/{param_est.rs:22,104,
  shooting.rs:29-38, collocation.rs:32-41}`.

The recent `ParametricOdeSystem` decision (no `Send + Sync` defensive
bounds, call-site escalation pattern) is the precedent for the
direction. These older bounds predate the principle.

**What needs doing**: for each site, decide whether (a) a real parallel
consumer requires the bound (document the consumer), or (b) the bound
is defensive (drop it). Loosening trait bounds is non-breaking, so the
removal is safe; the work is the audit, not the change. Bundle the
`Scalar` bound with §3.1's documented-rationale follow-up (either
keep with rustdoc explaining why, or drop).

### F-OPTS: Document SDE/FDE/IDE options divergence from `SolverOptions`

**Status**: scoped, not started. Surfaced 2026-05-10 by the
foundation-pass verification (finding B6). Low priority; documentation
only.

**What's there today**: `numra-sde`, `numra-fde`, `numra-ide` have
their own options structs (predating the §2.5 principle that new
solver families must justify divergence from `SolverOptions`). Their
rustdoc does not currently explain why they diverge — typically because
the principle didn't exist when they were written.

**What needs doing**: add a one-paragraph rustdoc note to each
divergent options struct explaining the rationale (fixed-step
algorithm, distinct adaptivity story, scalar-vs-Wiener noise time
control, etc.). Optionally consider whether the divergence is still
justified or whether one of these can be retro-fitted onto
`SolverOptions`.

### F-MATRIX-SHAPE: Decide whether `SparseMatrix` joins `Matrix` trait or stays separate

**Status**: scoped, not started. Surfaced 2026-05-10 by the
foundation-pass verification (finding C3). Design question, not
mechanical work.

**What's there today**: `numra-linalg::Matrix<S>`
(`matrix.rs:15`) has one workspace impl, `DenseMatrix<S>` (with the
faer-bound `S: Scalar + SimpleEntity + Conjugate<Canonical = S> +
ComplexField`). `SparseMatrix<S>` (`sparse.rs`) is a separate concrete
type that does **not** implement `Matrix<S>`. Sparse direct solvers
(`SparseLU<S>`) currently convert to dense internally
(`sparse.rs:148–149`).

The §3.2 design claim "Foundation is dense + sparse (CSC)" is
contradicted by the actual shape — sparse is parallel to the trait,
not within it. The Foundation Spec revision moves this question into
§7 (open questions, item 5).

**What needs doing**: decide one of:
- (a) Sparse joins the `Matrix` trait. Likely requires a `solve`
  method that can dispatch on storage layout, an iteration story,
  and probably a separator at the trait level for which operations
  make sense (matvec yes, dense indexing no).
- (b) Sparse stays separate; the trait stays dense-only and the
  rustdoc says so explicitly. Solvers that need to dispatch across
  both write their own enum or generic abstraction at the consumer
  layer.

This becomes urgent when a sparse-aware iterative solver path lands
that needs to call `solve` polymorphically over dense and sparse.
Until then it's a clarification, not a blocker.

### F-INTEROP-Q: Backfill interop tests covering `?`-propagation, non-`f64` Scalar, and capabilities currently missing an interop edge

**Status**: scoped, not started. Surfaced 2026-05-10 by the
foundation-pass verification (findings B5, D4, D6, J1).

**What's there today**: `numra/tests/interop_workflows.rs` contains
six workflow tests. Issues:

- **No test exercises cross-crate `?`-propagation.** Every test uses
  `.unwrap()` and returns `()`. The contract item 3 property is
  literally untested, even where the underlying impls do compose
  (e.g. `numra-interp::InterpError → NumraError`).
- **Every test is monomorphised at `f64`.** No interop test exercises
  the principle that composition preserves genericity. A user trying
  to compose at `f32` has no test telling them whether their pipeline
  holds together.
- **Capabilities lacking an interop edge entirely**: SDE, DDE, FDE,
  IDE, SPDE, Linalg-as-a-capability, Autodiff-reverse, Special. Each
  fails contract item 7.

**What needs doing**:
1. ~~Add at least one interop test that returns `NumraResult<()>` and
   uses `?` across at least two crate boundaries.~~ **Landed
   alongside F-ERR (2026-05-14)**: `workflow_ode_interp_integrate` now
   returns `Result<(), NumraError>` and uses `?` across three crate
   boundaries (numra-ode → numra-interp → numra-integrate).
2. Expand to cover the formerly-missing pairs that F-ERR unblocked
   (every other interop test in `interop_workflows.rs` still uses
   `.unwrap()` and could now exercise `?`-propagation through the new
   `From` impls).
3. Add at least one interop test with `S = f32` exercising a
   non-trivial pipeline (e.g. ODE → Interp → Quad). The
   silent-`f32`-FD-step issue is now pinned at the per-trait level
   by F-FD-STEP's two regression tests, but a pipeline-level `f32`
   interop test would catch any concrete-`f64` leak that monomorphises
   away in single-crate tests.
4. Add interop tests for the missing capabilities — one per. SDE
   into Stats; DDE into Interp; etc. The audit's roadmap §4 has the
   pairing logic.

This is one consolidated follow-up because the work shape is the
same for each (write a test in `interop_workflows.rs` or sibling
file). Remaining sequence: (2) any time; (3) before/alongside
F-FD-STEP; (4) over time as capabilities are touched.

### F-SENS-DOWNSTREAM: Decide downstream consumers for `SensitivityResult`

**Status**: scoped, not started. Surfaced 2026-05-10 by the
foundation-pass verification (finding H2 — composability contract
worked example, item 5).

**What's there today**: `numra-ode::SensitivityResult<S>` is
produced by `solve_forward_sensitivity{,_with}`
(`numra-ode/src/sensitivity.rs:831, 934`) and re-exported by
`numra-ocp::forward_sensitivity` (`numra-ocp/src/sensitivity.rs:73`).
**No workspace site consumes it as an input.** The composability
contract draft worked example claimed two outgoing edges
(`LevenbergMarquardt::fit`, `solve_trajectory`) but neither exists in
that direction in the code. The capability ships with no in-workspace
downstream story for its primary result type.

**What needs doing**: the sensitivity capability is incomplete until
at least one downstream consumer exists. Candidates worth scoping:

- **Parameter estimation** (likely highest value). A `param_est_with_sensitivity(...)
  -> ParamEstResult` API that takes a `SensitivityResult` produced
  externally and feeds it into a Levenberg-Marquardt loop without
  re-solving — useful for warm-starting and for users who want to
  bring their own sensitivity routine.
- **Uncertainty propagation**. A `propagate_uncertainty(SensitivityResult,
  ParamCovariance) -> StateUncertainty` API that turns sensitivity
  outputs into parameter-covariance-driven state uncertainty. Naturally
  pairs with `numra-core::uncertainty`.
- **Identifiability analysis**. A `identifiability(SensitivityResult)
  -> IdentifiabilityReport` that reports per-parameter rank deficiency,
  collinearity, and informativeness — the classical use case for forward
  sensitivity.

Each candidate is its own piece of work; this follow-up exists to name
the question and to make the worked-example ✗ visible until at least
one of the downstream consumers ships. Once one ships, the contract
worked example flips from ✗ to ✓ and this follow-up is retired.

---

## Documentation

### First blog post

**Status**: not started.

Plan §3.1 has the scaffold (Astro content collection at
`website/site/src/content/blog/`, RSS via `@astrojs/rss`, MDX with
frontmatter `{title, date, author, tags[], summary, hero_image}`).
Inaugural post candidates:

- "Designing a numerical methods workspace in Rust" — workspace
  philosophy, why 21 crates and not one.
- "Why Numra's `Radau5` is slow on Van der Pol (and what we're doing
  about it)" — turning the §2.3 comparison-chapter honesty into a
  dev-blog post.
- "Method of lines, but in Rust" — walkthrough of `numra-pde` on
  the heat equation.

### SUNDIALS comparison chapter (Phase 3 §3.2)

**Status**: scoped, blocked on DAE work above.

Honest SciML/SUNDIALS comparison is the right Phase-3 deliverable, but
running it before the Radau5 controller rewrite would just embarrass
us. Sequence: Radau5 fix → DAE ergonomics → SUNDIALS chapter.

### Broader perf comparisons (ndarray-linalg vs `numra-linalg`, rustfft vs `numra-fft`)

**Status**: scoped, blocked on linalg native work.

Same logic — `numra-linalg`'s dense path is a faer wrapper, so a
"Numra vs ndarray-linalg" comparison is really "faer vs
ndarray-linalg in disguise". Defer until there's a story to tell.

---

## Tooling

### CI: Renovate canary for Astro pre-releases

**Status**: not started.

Mentioned in `website/book/DEVIATIONS-LOCAL.md` (the §2.13
contingency). The recipe: open *non-blocking* PRs for Astro pre-releases
so we get early signal when the `starlight-katex` plugin's peer-deps
will break, with enough lead time to swap to the documented fallback.

### Bench freshness CI gate

**Status**: not started.

Plan §4 mentions: "`git log -1 --format=%cd numra-bench/results/` ≤ 90
days, CI weekly". Worth wiring as a soft warning on the website job —
not a build-failing gate, but a comment on PRs noting "bench data is
N days old; consider re-running before merging".

### WASM browser playground

**Status**: deferred per the plan, intentionally.

The plan explicitly drops this from Phase 3. Tracking it here only so
that "do we want WASM playground?" doesn't get re-litigated every six
months without context. The answer is no for now; revisit when reach
matters more than depth.

### Distributed / parallel solving (rayon-parallel RHS, MPI bindings)

**Status**: not started, no concrete user demand.

Belongs on this list rather than the public roadmap because there's no
specific use case requesting it. Add as a public direction *if* a user
opens an issue with a real workload that needs it.

---

## How to maintain this file

- When something here lands, delete its section. Don't leave checkmarked
  zombies.
- When something is rescoped onto the public roadmap, delete its section
  and add a one-line entry to that route instead.
- New follow-ups go under the right heading (Solvers / API / Documentation
  / Tooling).
- Cite file paths and line numbers when the follow-up is grounded in
  existing code, so a future reader can verify the premise still holds.

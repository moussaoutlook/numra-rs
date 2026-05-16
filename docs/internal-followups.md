# Internal follow-ups

Local-only tracking of work that's been considered, scoped, or partially
built, but isn't appropriate to publish on the public roadmap. Not linked
from the website. Not for marketing. The point is to have a written record
that survives session-context churn so we can pick threads back up cleanly.

When you act on something here, prefer moving it into a CHANGELOG entry,
a closed GitHub issue, or the public roadmap — and remove it from this
file once it lands. Stale follow-ups files are how good intentions become
embarrassments.

Last updated: 2026-05-16 (F-WEBSITE-AUDIT-GATES partially retired — config-staleness + book URL-list + Playwright wrong-expectation portions landed; four genuine-site-issue follow-ups opened as splits: F-WEBSITE-SEO, F-WEBSITE-MARKETING-A11Y, F-WEBSITE-MARKETING-PERF, F-WEBSITE-BOOK-LHC-FIXES. Two same-day amendments: (a) the URL-fix unmasked book-specific Lighthouse regressions, so F-WEBSITE-MARKETING-SEO was rescoped to F-WEBSITE-SEO covering both subdomains, and F-WEBSITE-BOOK-LHC-FIXES added for book-specific a11y + perf; (b) the Playwright marketing dark-mode failure was reinterpreted from "real regression" to "gate encoded a wrong expectation" after user-supplied design intent (marketing is deliberately light-only), so the three marketing dark-mode tests were removed in-scope here with a foreclosing docstring. F-WEBSITE-PR-FLOW framing tightened given the audit demonstrated the gates catch real shipping-blockers — and the same audit demonstrated that gates can also encode wrong expectations, which PR-FLOW's eventual scope must address).

## Recently retired

One-line entries for follow-ups that landed and were removed from the
file. Kept here so a future reader can find the closure record without
git-archaeology.

- **F-FD-NOSCALE-BUG: no-scaling correctness bug in public FD utilities** — landed in `Unreleased` (next 0.1.x release) 2026-05-15. Four FD utilities defaulting to hardcoded `h = 1e-8` without `(1 + |x|)` scaling silently degraded gradient/Jacobian outputs for callers with `|x| > ~5e7` (precision floor where `x + 1e-8` rounds back to `x` in `f64`). Fixed at all four named sites: `numra-optim/src/problem.rs:486` (`finite_diff_gradient`, central → `cbrt(EPSILON) * (1 + |x|)`), `numra-optim/src/problem.rs:503` (`finite_diff_jacobian`, central), `numra-dde/src/history.rs:188` (`History::evaluate_derivative` initial-history branch, central), `numra-sde/src/system.rs:68` (`SdeSystem::diffusion_derivative` trait default, **forward → `sqrt(EPSILON) * (1 + |x|)`** — direction-corrected by the audit; the entry had assumed central FD). Each pinned with a regression test at `|x| = 1e8` asserting analytical-truth proximity within `1e-3` relative; structural-correctness check verified on the forward-FD site (revert → fail → restore). Public-API rustdoc on the two `numra-optim` free functions documents the canonical step formula and the `~5e7` precision floor. Audit found exactly the four named sites — first follow-up where the audit confirmed the entry's scope rather than expanding it (different from F-FD-STEP / F-CI-NODE20 / F-FD-CROSSCRATE). The 0.1.2 CHANGELOG's note on this follow-up over-listed `numra-optim::robust` as no-scaling-bug-class; that was already corrected in F-FD-CROSSCRATE's audit pass and stands.
- **F-CI-NODE20: upgrade GitHub Actions runners to Node.js 24** — shipped 2026-05-15. Five action majors bumped to versions declaring `runs.using: node24`, clearing the 2026-06-02 deprecation deadline ahead of time: `actions/checkout@v4 → @v6` (11 usages), `actions/setup-node@v4 → @v6` (7), `actions/upload-artifact@v4 → @v7` (4), `pnpm/action-setup@v3 → @v6` (4; also dropped redundant `with: version: 9` and deferred to the `packageManager: pnpm@9.15.0` field in `package.json` as the single source of truth — required for the v4+ strict check), `cloudflare/wrangler-action@v3 → @v4` (3; default wrangler version implicitly upgrades v3 → v4 — `pages deploy` syntax is stable across the bump). Actions already on node24-runtime majors (`treosh/lighthouse-ci-action@v12`, `Swatinem/rust-cache@v2`) left at their current pins per scope discipline; composite actions (`taiki-e/install-action`, `dtolnay/rust-toolchain`, `rhysd/actionlint`) not affected by Node-runtime deprecation. Audit surfaced 4 actions missed from the original entry (`upload-artifact` needs bump; the three composites and lighthouse/rust-cache don't) — same audit-surfaces-more-than-named pattern as F-FD-STEP.
- **F-FD-STEP: foundation-trait FD-step reconciliation** — shipped 2026-05-15. `OdeSystem::jacobian` default switched from hardcoded `1e-8` to `sqrt(S::EPSILON) * (1 + |y_j|)`; `Signal::eval_derivative` default switched from hardcoded `1e-8` (no scaling) to `cbrt(S::EPSILON) * (1 + |t|)` (canonical central-FD step). `ParametricOdeSystem::jacobian_y/_p` defaults were already correct on `sqrt(S::EPSILON)`; no change. The six `MOLSystem{2,3}D::jacobian` and `ParametricMOLSystem{2,3}D::jacobian_y/_p` reaction-FD diagonals updated in lockstep (each referenced the trait default in code comments — preserving the consistency the comments claim required moving them together). Two `f32` regression tests added (`numra-ode/src/problem.rs::test_jacobian_finite_diff_f32`, `numra-core/src/signal.rs::test_signal_derivative_f32`) pinning out the silent-quantisation failure mode. The audit pass also surfaced two adjacent follow-ups that were explicitly out of scope for F-FD-STEP — F-FD-CROSSCRATE (now also retired below) and F-FD-NOSCALE-BUG.
- **F-FD-CROSSCRATE: cross-crate FD-step formula reconciliation** — shipped 2026-05-15. 18 cross-crate FD bodies aligned to the canonical precision-aware forms (forward → `sqrt(S::EPSILON) * (1 + |x_j|)`, central → `cbrt(S::EPSILON) * (1 + |x_j|)`). Forward sites: `numra-nonlinear/src/newton.rs:245`, `numra-ode/src/{auto.rs:168, dae_init.rs:105, esdirk.rs:525, index_reduction.rs:407,766,788,879}`, `numra-ocp/src/adjoint.rs:82,109,128,147,163`. Central sites: `numra-fit/src/curve_fit.rs:80,187,489` (also a structural recipe shift from multiplicative `orig*eps` to additive `(1+|orig|)*eps`), `numra-core/src/uncertainty.rs:278` (`compute_sensitivities` default), `numra-optim/src/optim_sensitivity.rs:44` (`compute_param_sensitivity` default). Audit pass surfaced 3 sites beyond the entry's named 15 — two `ReducedDaeSystem.fd_eps` field initializers and the `compute_param_sensitivity` default; same audit-surfaces-more-than-named pattern as F-FD-STEP and F-CI-NODE20. New regression test `numra-optim::optim_sensitivity::tests::test_sensitivity_canonical_default_eps_at_large_param` pins the canonical default at `|p| = 100` (asserts `dx*/dp ≈ 1` for `min (x - p)^2` to within `1e-2`). Behavioural deltas documented in CHANGELOG: adjoint/DAE-init sites now use a 6.6× smaller forward-FD step (was `1e-7`); curve_fit's structural shift produces ~120× larger steps at `|orig| ≈ 1` with better roundoff/discretization balance. F-FD-NOSCALE-BUG remains open as the only deferred FD-step item.
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

### F-WEBSITE-AUDIT-GATES: Fix orphan website audit gates surfaced by F-CI-NODE20

**Status**: partially retired 2026-05-16. Config-staleness, book
URL-list, and Playwright wrong-expectation portions landed in
`Unreleased` (next 0.1.x release); the four split-out follow-ups for
genuine deployed-site regressions are below. Full closure when those
land and the gates run clean on a PR-event trigger.

**Background**: surfaced 2026-05-15 by F-CI-NODE20's PR (#5) — first PR
ever to trigger `website.yml` on a `pull_request` event, which is when
these gates fire (`if: github.event_name == 'pull_request'`). Every
prior website-touching commit landed via direct push to main, so these
four jobs had never executed in the repo's history before. PR #5's
admin-merge was the documented one-time exception.

**Audit finding (2026-05-16)**: the entry's pre-diagnosis ("Lighthouse
stale config, others unknown") was substantially wrong. Reading the
actual PR #5 CI logs (run `25910476569`) revealed three distinct
failure modes, not one:

1. **Config staleness on Lighthouse configs** (in-scope, fixed):
   `_comment_pwa`, `_comment_third_party_summary`,
   `_comment_book_specific` keys sitting inside `assertions{}` (LHCI
   parses any key in that object as an audit ID and reports "is not a
   known audit"); three deprecated audit assertions
   (`no-unload-listeners`, `no-vulnerable-libraries`, `uses-https` —
   all removed in Lighthouse 12; `is-on-https` is the surviving
   HTTPS-validation audit); and `render-blocking-resources`'s
   `maxNumericValue` override (no-op in Lighthouse 12, which reshaped
   the audit to return a list — the preset's `maxLength` is what
   actually fires). Removed across both `lighthouserc.json` and
   `lighthouserc-book.json`. Top-level `_comment` fields expanded to
   document the audits that were removed upstream, as durable context
   for future readers.
2. **Workflow URL list mismatch** (in-scope, fixed): Lighthouse-book's
   workflow ran with `/ch01-fundamentals/numerical-stability/` and
   similar paths that don't exist in the deployed book — the actual
   directory is `ch01-introduction/`, with no `numerical-stability`
   page. Likely a holdover from an early outline. The 404 short-
   circuited Lighthouse-book's entire run before assertions could
   fire; fixing the URL list lets the gate exercise the config it
   was always supposed to.
3. **Genuine deployed-site regressions on three gates** (split out, see
   below): 78 WCAG2AA color-contrast violations across two marketing
   pages (one Astro component instanced many times); `is-crawlable: 0`
   across **both** the marketing site and the book (the entire Numra
   web presence is currently un-indexable by search engines); real
   CLS, render-blocking, image-delivery issues on the marketing site;
   book-side Lighthouse failures unmasked by this PR's URL fix
   (`label-content-name-mismatch`, `network-dependency-tree-insight`,
   `font-display-insight`, `lcp-discovery-insight`, `lcp-lazy-loaded`).
   Tracked as F-WEBSITE-SEO (highest priority overall — both
   subdomains), F-WEBSITE-MARKETING-A11Y, F-WEBSITE-MARKETING-PERF,
   and F-WEBSITE-BOOK-LHC-FIXES below.
4. **Playwright gate encoded a wrong expectation** (in-scope, fixed
   here). The four marketing dark-mode tests in
   `website/tests/specs/dark-mode.spec.ts` (three system-dark
   assertions on `/` and `/install`, plus the stored-preference
   `data-theme="dark"` assertion) asserted that the marketing site
   honors `prefers-color-scheme: dark` and applies `data-theme="dark"`
   from localStorage. **The marketing site is deliberately light-only
   by design** — that's a chosen product position, not an oversight.
   The Playwright gate was therefore not detecting a regression; it
   was asserting a non-requirement that had been silently failing
   since the gate first ran on a `pull_request` event. Removed in
   this PR, with a foreclosing docstring update on the spec file
   documenting the intentional asymmetry with the book (which DOES
   support dark mode via Starlight) and forbidding future
   contributors from re-introducing the wrong assertions. The book
   dark-mode tests, the marketing-light test, and the book-light
   test all remain — they assert real requirements that the deployed
   sites genuinely meet.

**What this PR landed**: the three in-scope items above
(config-staleness fixes across both Lighthouse configs; book chapter
URL-list correction; removal of the marketing dark-mode Playwright
assertions). The Lighthouse-book gate may actually pass on this PR's
own run — the book root loaded cleanly 3× in PR #5 and the only known
failure-blocker was the chapter-URL 404, which is now fixed (caveat:
the URL fix also unmasked book-specific assertion failures previously
hidden by the 404, tracked as F-WEBSITE-BOOK-LHC-FIXES — see same-day
amendment below). The Playwright gate should pass under the test
removal alone (the remaining tests already passed). The other two
gates (Lighthouse-marketing, pa11y) still fail because they're
catching real site regressions this PR explicitly does not address;
the PR is admin-merged with per-gate documentation citing each
split-out follow-up. Same documented-exception discipline as PR #5.

**Pattern callout**: the investigative-audit framing was load-bearing.
If this had been treated as enumerable config-fixing work per the
entry's pre-diagnosis, the resulting PR would have "fixed" the gates
by silencing them while leaving 78 a11y violations, a site-wide
search-indexing block, and a backlog of real site issues live in
production. Same lesson as the FD audits surfacing F-FD-NOSCALE-BUG /
F-FD-CROSSCRATE rather than absorbing everything into one PR — audits
discover real scope, they don't just confirm pre-stated scope. The
audit also has limits: it accurately diagnosed each gate's *failure*
from the CI logs but lacked design-intent context for the Playwright
marketing dark-mode case, and so misinterpreted a wrong-expectation
gate as a regression. The hard-stop verification discipline plus
user-supplied design intent caught the misinterpretation before it
entered permanent record. Both lessons (audits expand scope; audits
need design-intent inputs they can't derive from logs alone) belong
in any future audit playbook.

**Priority for full closure**: medium. The four split-outs have their
own per-entry priorities (F-WEBSITE-SEO is the highest-priority item in
the entire backlog — see its entry for cost-of-delay rationale). This
entry closes fully when those land and all four gates run green on a
PR-event trigger.

**Same-day amendments (2026-05-16)**:

- **Amendment 1 — URL fix unmasked book Lighthouse failures.** This
  PR's own gate run confirmed the audit's URL-fix hypothesis
  (`/ch01-introduction/installation/` now loads cleanly — no more
  404) but unmasked book-specific Lighthouse failures that the
  original 404 had hidden. Those are added to the split-out shape
  as F-WEBSITE-BOOK-LHC-FIXES, and the original
  F-WEBSITE-MARKETING-SEO was rescoped to F-WEBSITE-SEO because the
  `is-crawlable: 0` issue shows up on **both** subdomains (likely
  one shared root cause; the SEO follow-up's investigation will
  confirm or split).
- **Amendment 2 — Playwright wrong-expectation correction.** The
  Playwright marketing dark-mode failure was initially audited as
  a "real regression" and a fifth split-out follow-up
  (F-WEBSITE-DARKMODE-REGRESSION) was opened. User-supplied design
  intent during the pre-merge hard-stop review surfaced that the
  marketing site is deliberately light-only by design — the gate
  was asserting a non-requirement, not detecting a regression. The
  follow-up was retired before landing in PR history; the wrong
  assertions were removed from the spec file in-scope here as a
  fourth distinct failure mode (gate-correction-not-gate-silencing,
  same shape as removing stale Lighthouse audit IDs).

Four split-outs total after both amendments. Both amendments were
caught by the pre-merge hard-stop verification discipline before
falsifiable claims entered permanent record.

### F-WEBSITE-SEO: Entire Numra web presence is blocked from search indexing

**Status**: scoped, not started. Surfaced 2026-05-16 by
F-WEBSITE-AUDIT-GATES's audit pass; rescoped same-day from the
original F-WEBSITE-MARKETING-SEO after this PR's own gate run showed
`is-crawlable: 0` on the book as well as the marketing site.

**Finding**: Lighthouse reports `is-crawlable: 0` on **both subdomains**:

- **Marketing site**: every page audited (PR #5 + PR #8) — `/`,
  `/install`, `/license`, `/commercial`, `/cite`, `/community`,
  `/stability`, `/features`, `/privacy`. `categories:seo` sits at
  0.69 (target 0.95) dominated by this audit.
- **Book**: every page audited (PR #8, after the URL-fix unmasked
  the gate's assertion phase) — `/`, `/ch01-introduction/installation/`,
  `/ch02-solving-odes/your-first-ode/`, `/ch13-performance/`,
  `/ch13-performance/comparisons/`. `categories:seo` also fails.

`is-crawlable: 0` means the page is explicitly blocked from indexing
— either via `<meta name="robots" content="noindex">`, an
`X-Robots-Tag` header, a disallowing `robots.txt`, or a Cloudflare
Pages configuration that does the same.

**Cost-of-delay — highest-priority item in the entire backlog**.
The entire public Numra web presence has been organically
undiscoverable by search engines since launch. The marketing site
exists to introduce the project; the book exists as the long-form
reference. Both being un-indexable means anyone not given a direct
link cannot find either. Every day of delay is a day the project
can't be reached via search. This priority is not local-to-the-
website-track — it dominates the project-wide backlog.

**Expedite-if-trivial guidance** (this entry's primary load-bearing
note): the eventual audit should **assess fix size early**. Plausible
trivial causes:

- An accidental `X-Robots-Tag: noindex` (or `User-agent: * / Disallow: /`)
  in `website/site/public/_headers` and/or `website/book/public/_headers`
  — left over from pre-launch staging and never reverted.
- A disallowing `robots.txt` (only `website/site/public/robots.txt`
  exists; the book has no `robots.txt` of its own, but a project-wide
  Cloudflare config could be doing the same).
- A `<meta name="robots" content="noindex">` in the shared base
  layout of one or both sites.

If the audit confirms the cause is a trivial config fix, **surface
that finding immediately** so the fix can land as its own tiny PR
rather than waiting for normal scheduling. A 1-line `_headers` /
`robots.txt` change to restore indexing is qualitatively different
work from the multi-day a11y / perf remediation in the other
split-outs, and the cost-of-delay justifies the expedite. Don't
roll it into a larger SEO sweep; ship the small fix the moment the
root cause is confirmed.

**What needs doing**:
1. Two-property investigation. Check, in order of likelihood:
   - `website/site/public/_headers` (Cloudflare Pages headers config —
     the most likely culprit; will be `X-Robots-Tag: noindex` or similar).
   - `website/book/public/_headers` (same check for the book subdomain;
     could be a copy-paste of the marketing config).
   - `website/site/public/robots.txt` (file exists; check for a
     blanket Disallow).
   - The book has no `robots.txt`; verify it isn't being served one
     by a Cloudflare-level project default.
   - Both sites' base layouts for `<meta name="robots">` tags.
   - Astro config in both `astro.config.mjs` for any `build`-level
     SEO directives.
2. Confirm whether the cause is one shared source (project-wide) or
   two divergent sources (separate `_headers` files happen to both
   contain noindex). If one source: single fix. If two: still one
   follow-up (this one), but the fix has two touchpoints.
3. If the audit finds a trivial root cause (single config line on one
   or both sites), surface immediately and ship as an expedited tiny
   PR per the guidance above.
4. Verify with a follow-up Lighthouse run on the fix branch —
   `is-crawlable` should return to 1 on **both subdomains**;
   `categories:seo` should climb back above 0.95 on both.

**Priority**: highest in the entire backlog. Should be scheduled
ahead of every other follow-up — both F-WEBSITE-AUDIT-GATES split-outs
and any other currently-open follow-up — because the cost-of-delay
is project-wide visibility, not local to the website track.

### F-WEBSITE-MARKETING-A11Y: WCAG2AA violations on marketing root + examples gallery

**Status**: scoped, not started. Surfaced 2026-05-16 by
F-WEBSITE-AUDIT-GATES's audit pass.

**Finding**: pa11y on PR #5's preview reported 78 WCAG2AA violations
across two URLs — marketing root (49 errors) and examples gallery (29
errors). Lighthouse-marketing independently reports `color-contrast:
0`, `heading-order: 0`, `label-content-name-mismatch: 0`,
`link-in-text-block: 0` across all 9 marketing pages, and
`categories:accessibility` at 0.93–0.98 (target 1.00).

**Scope-bounding finding** (color-contrast is contained): all 78 pa11y
errors trace to **one Astro component** (`data-astro-cid-j7pv25f6`)
instanced many times across the two pages. Text spans with classes
`ch`, `num`, `src` have contrast ratios 2.16:1 or 2.25:1 against their
backgrounds (AA requires ≥ 4.5:1). pa11y's recommendation: change text
color to `#080d16` for the `ch`/`num` spans and `#727780` for the
`src` spans. **One CSS fix in that component clears all 78 errors.**

**The other a11y audits need separate per-issue investigation**:
- `heading-order`: marketing pages skip heading levels (e.g., `<h1>`
  directly to `<h3>`). Per-page audit needed to locate each break.
- `label-content-name-mismatch`: elements with visible text labels
  don't have matching accessible names. Likely buttons or links with
  icon + text where the accessible name is just the icon's
  `aria-label`. **Same-symptom, different-source caveat**:
  F-WEBSITE-BOOK-LHC-FIXES also includes a `label-content-name-mismatch`
  failure, but the book's instance is in Starlight templates (upstream
  theme, possibly fixed upstream rather than in our content); this
  marketing instance is in custom Astro components under
  `website/site/src/`. The two share an audit name but not a fix
  surface — do not conflate them when scheduling or reviewing.
- `link-in-text-block`: links inside prose are visually
  indistinguishable from surrounding text (rely only on color).
  Need underline or other non-color indicator.

**What needs doing**:
1. Fix the contained color-contrast component first (one CSS change
   clears 78 of the violations and is the most actionable diagnosis).
2. Per-issue investigation for `heading-order`,
   `label-content-name-mismatch`, `link-in-text-block`. Each may need
   its own component-level fix.
3. Verify with a follow-up pa11y + Lighthouse run on the fix branch —
   `categories:accessibility` should return to 1.00; pa11y should
   report 7/7 URLs passing.

**Priority**: medium. Below F-WEBSITE-SEO (which blocks discovery
entirely) but above F-WEBSITE-MARKETING-PERF and
F-WEBSITE-BOOK-LHC-FIXES.

### F-WEBSITE-MARKETING-PERF: Marketing site fails several Lighthouse performance audits

**Status**: scoped, not started. Surfaced 2026-05-16 by
F-WEBSITE-AUDIT-GATES's audit pass.

**Finding**: Lighthouse-marketing on PR #5 reports several performance
failures across the 9 marketing pages:

- `cumulative-layout-shift: 0.132803` on `/stability` (target ≤
  0.05), with `cls-culprits-insight: 0` flagging the culprits. CLS is
  page-specific; other pages are within budget but `/stability`
  regresses sharply.
- `render-blocking-resources` (and the newer companion
  `render-blocking-insight`): 2 render-blocking items on each page.
  The preset asserts `maxLength: 0`.
- `image-delivery-insight: 0.5` and `uses-responsive-images: 0.5`
  across pages — images aren't being sized correctly for the viewport.
- `network-dependency-tree-insight: 0` — the critical-request-chain
  depth exceeds the preset's threshold.

**What needs doing**:
1. Investigate `/stability`'s CLS specifically —
   `cls-culprits-insight` in the Lighthouse report identifies the
   offending elements; likely an unsized image, web font, or
   late-loaded component.
2. Identify the two render-blocking resources per page (the
   Lighthouse report names them; typically CSS in `<head>` or a
   synchronous script). Defer or inline as appropriate.
3. Audit images for responsive `srcset` / `sizes` attributes; convert
   to `modern-image-formats` (AVIF/WebP) where missing.
4. Trace the network dependency tree — what's the critical-request
   chain? Often a font or CSS dependency that's deeper than it needs
   to be.
5. Verify with a follow-up Lighthouse run on the fix branch —
   `categories:performance` should sit ≥ 0.95 across all 9 pages.

**Priority**: medium. May need sub-splits if individual issues turn
out to have separate root causes (the CLS-on-/stability fix is
probably distinct from the render-blocking fix).

### F-WEBSITE-BOOK-LHC-FIXES: Book Lighthouse failures unmasked by URL fix

**Status**: scoped, not started. Surfaced 2026-05-16 by this PR's own
gate run, after the F-WEBSITE-AUDIT-GATES URL fix replaced the
hardcoded chapter URL `/ch01-fundamentals/numerical-stability/` (404)
with `/ch01-introduction/installation/`. The book gate's assertion
phase had been short-circuited by that 404 on PR #5 and is now
exercising its config for the first time. The failures it reports
are genuine book-site Lighthouse regressions, not staleness in
`website/ci/lighthouserc-book.json`.

**Finding**: Lighthouse on PR #8's preview reports per-page failures
across all 5 book URLs (`/`, `/ch01-introduction/installation/`,
`/ch02-solving-odes/your-first-ode/`, `/ch13-performance/`,
`/ch13-performance/comparisons/`):

- `label-content-name-mismatch` — fires on every page. **Same-symptom,
  different-source caveat**: F-WEBSITE-MARKETING-A11Y also has a
  `label-content-name-mismatch` failure, but the marketing instance
  is in custom Astro components under `website/site/src/`; the book
  instance is in Starlight templates (`@astrojs/starlight` upstream).
  The two share an audit name but not a fix surface — the book's may
  resolve upstream (file/track a Starlight issue), or may need a
  Starlight-component override locally. Do not conflate with
  F-WEBSITE-MARKETING-A11Y's same-named item.
- `network-dependency-tree-insight` — fires on every page. Book ships
  Starlight UI runtime (~150 KB) plus Pagefind's WASM search index;
  the critical-request chain depth likely reflects the Pagefind
  init or Starlight's bundled UI components. Worth profiling whether
  Pagefind can be deferred past the critical path.
- `font-display-insight` — fires on `/` (book root). KaTeX serves
  several `.woff2` files for math glyphs, and Starlight has its own
  web fonts. Likely cause: one or more `@font-face` declarations
  missing `font-display: swap` (or `optional`). Worth checking
  whether the regression is in KaTeX's bundled CSS, Starlight's, or
  a project-side override.
- `lcp-discovery-insight` and `lcp-lazy-loaded` — fire only on
  `/ch13-performance/comparisons/`. That page has the most
  benchmark-result content; likely an above-the-fold image or
  embedded SVG with `loading="lazy"` set inappropriately, or an LCP
  element that isn't discoverable by Lighthouse's preload-detection
  heuristic.

`is-crawlable: 0` on the book is excluded from this entry's scope —
it's tracked in F-WEBSITE-SEO (covers both subdomains; same likely
root cause as marketing).

**What needs doing**:
1. Triage the four issues. `label-content-name-mismatch` is probably
   the biggest unknown — investigate whether the source is in
   `node_modules/@astrojs/starlight/...` or in a project-side
   component override, then decide between upstreaming a fix vs.
   local override.
2. `font-display-insight`: identify the missing-`font-display`
   declarations (probably 2–4 declarations across KaTeX and
   Starlight). The fix is typically a project-side `@font-face`
   override that reuses the same `src` but adds `font-display: swap`.
3. `network-dependency-tree-insight`: profile the critical-request
   chain in the Lighthouse report; assess whether Pagefind can be
   deferred or its WASM payload can be made non-critical.
4. `lcp-discovery-insight` / `lcp-lazy-loaded` on
   `/ch13-performance/comparisons/`: identify the LCP element from
   the Lighthouse report, fix its `loading=` attribute or add a
   `<link rel="preload">` hint as appropriate.
5. Verify with a follow-up Lighthouse-book run on the fix branch —
   all 5 book pages should clear the four audits (modulo the
   F-WEBSITE-SEO indexing issue, which is tracked separately).

**Priority**: medium. Lower than F-WEBSITE-SEO (project-wide
discoverability) but comparable to F-WEBSITE-MARKETING-PERF in scope
and impact (visitors who do find the book see slower-than-target
loads). May need a sub-split if the `label-content-name-mismatch`
investigation determines Starlight's upstream needs a patch and the
local workaround is materially different work from the other three
items.

### F-WEBSITE-PR-FLOW: Decide whether `website/` changes require PR-flow

**Status**: scoped, not started. Surfaced 2026-05-15 alongside
F-WEBSITE-AUDIT-GATES; framing tightened 2026-05-16 after that audit's
findings.

**The question**: should website-touching commits be required to land
via PR going forward, so the audit gates (Lighthouse, pa11y,
Playwright) actually run before the change lands? Currently the
recent website history shows direct-pushes to main:

- `9975e99 feat(site): derive release-tied content from CITATION.cff`
- `0a0efaa feat(site): add blog section with v0.1.0 release post`
- `0b6f7f9 fix(docs, site): unbreak repo README links and align homepage citation with cite.astro`

…and so on. None went through PR-flow, so none triggered the audit
gates. The gates have been silent — not ornamental.

**The "silent vs. ornamental" distinction matters here**: ornamental
means the gates aren't catching anything. The F-WEBSITE-AUDIT-GATES
audit pass on 2026-05-16 demonstrated the opposite. On their first-ever
PR-event run (PR #5), the gates correctly fired on 78 WCAG2AA violations
across two pages, a site-wide search-indexing block
(`is-crawlable: 0`), real CLS / render-blocking / image-delivery
regressions, and a broken dark-mode mechanism. The gates were silent
because direct-push prevented them from running, not because there
was nothing for them to catch. The "remove the gates" option below is
therefore not on the table — they demonstrably work.

**The remaining question is purely "when to enforce, not whether":**

- **Enforce now**: branch-protect `main` to require the four audit
  jobs to pass on `website/`-touching changes. Problem: until the
  four F-WEBSITE-AUDIT-GATES split-outs land (F-WEBSITE-SEO,
  F-WEBSITE-MARKETING-A11Y, F-WEBSITE-MARKETING-PERF,
  F-WEBSITE-BOOK-LHC-FIXES), 2 of 4 gates still fail on every PR
  (Lighthouse-marketing, pa11y; Lighthouse-book likely passes after
  this PR's URL fix; Playwright passes after this PR's wrong-
  expectation removal) — enforcement now would block all website
  work behind a still-broken signal.
- **Enforce after split-outs land**: wait until the four split-out
  follow-ups close and all four gates run green on a PR-event
  trigger; then branch-protect. Clean transition.

**Recommended sequencing**: F-WEBSITE-AUDIT-GATES (partial-retired
2026-05-16) → SEO (highest priority — likely expedited as a tiny PR
if root cause is trivial) → A11Y / PERF / BOOK-LHC-FIXES land → all
four gates green → branch-protect `main` on `website/**` paths gated
on the four audit jobs. The question becomes a one-line config change
once the gates are clean.

**Lesson F-WEBSITE-PR-FLOW must incorporate into its eventual scope**:
gates that *run* and gates that *assert the right thing* are
separate properties; the audit must cover both. F-WEBSITE-AUDIT-GATES
surfaced concrete worked examples of each failure mode:

- **Gate runs, asserts wrong thing**: the Playwright marketing
  dark-mode tests asserted that the marketing site honors
  `prefers-color-scheme: dark` and applies stored-preference
  `data-theme="dark"`. The marketing site is deliberately light-only
  by design — those assertions were a non-requirement that had been
  silently failing since the gate first ran on a `pull_request`
  event. The gate-correction was removing the wrong assertions, not
  fixing the deployed site.
- **Gate runs, asserts right thing, but the asserted thing is
  broken**: every other failure in F-WEBSITE-AUDIT-GATES — the 78
  WCAG2AA contrast violations, `is-crawlable: 0` on both subdomains,
  the book's `font-display` and LCP failures, etc. These are
  conventional "audit catches real bug" failures.

F-WEBSITE-PR-FLOW's enforce-vs-not decision must therefore include
**"each gate's expectations are correct against current design
intent"** as part of its eligibility check, not just **"each gate
runs and currently passes"**. A gate that runs, passes, and asserts
the wrong thing is worse than no gate at all — it manufactures
false confidence. The Playwright dark-mode case caught here is the
concrete instance proving why this check belongs in the scope.
Treat the design-intent-review pass as a one-time cost when scoping
PR-FLOW: walk each gate's expectations against current product
intent, document each as correct or remove/correct the wrong ones,
before enforcing.

**Decision is still a workflow-convention call, not implementation
work.** This entry exists so the question gets decided rather than
re-litigated each time someone wonders why the website audit gates
exist. The audit already discharged the "whether" question; the
"when" decision waits on the split-outs and the gate-expectations
review above.

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

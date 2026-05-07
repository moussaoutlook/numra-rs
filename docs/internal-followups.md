# Internal follow-ups

Local-only tracking of work that's been considered, scoped, or partially
built, but isn't appropriate to publish on the public roadmap. Not linked
from the website. Not for marketing. The point is to have a written record
that survives session-context churn so we can pick threads back up cleanly.

When you act on something here, prefer moving it into a CHANGELOG entry,
a closed GitHub issue, or the public roadmap — and remove it from this
file once it lands. Stale follow-ups files are how good intentions become
embarrassments.

Last updated: 2026-05-07 (`MOLSystem3D` wrapper landed: `numra-pde/src/mol3d.rs` mirrors `mol2d.rs` exactly — `heat`/`laplacian`/`with_operator`/`with_reaction` constructors, optional pointwise reaction term, `OdeSystem` impl, `build_full_solution` using `Grid3D::linear_index`. Backed by new `Operator3DCoefficients` + `assemble_operator_3d` in `sparse_assembly.rs`; `assemble_laplacian_3d` now delegates to the general assembler, mirroring the 2D pattern. 5 new tests pass plus existing 3D Laplacian regressions. The 3D-MOL bullet under §Audit-driven scope corrections has been removed; remaining PDE gaps acknowledged there are multi-component coupled PDEs and an elliptic static solver path. Earlier landing 2026-05-06: forward-sensitivity API.

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

---

## PDE

These follow-ups were surfaced during the `MOLSystem3D` landing
(2026-05-07). They apply equally to `MOLSystem2D` — the 3D wrapper
mirrors the 2D wrapper exactly, so neither dimension regresses
relative to the other, but both share the same gaps relative to the
broader Numra solver surface. Listed here so that when we close them,
we close them for both wrappers in the same PR.

### MOL systems don't implement `JacobianProvider`

**Status**: scoped, not started. Affects `MOLSystem2D`, `MOLSystem3D`,
and the 1D `MOLSystem` (`numra-pde/src/mol.rs`).

When a stiff solver (Radau5, BDF) integrates an MOL system, it falls
back to finite-difference Jacobian construction even though the
linear part of the Jacobian is *literally already in memory* — the
assembled sparse operator (`MOLSystem{2,3}D::operator`) is exactly
∂(L[u])/∂u. The reaction term contributes a diagonal block: for a
pointwise reaction `R(t, x, y, z, u)`, ∂R/∂u is itself a diagonal
matrix because R doesn't couple grid points.

**What needs doing**:
- Implement `JacobianProvider<S>` for `MOLSystem2D`, `MOLSystem3D`,
  and `MOLSystem`. Return the assembled operator plus, when a
  reaction is registered, the diagonal Jacobian contribution. The
  reaction Jacobian is the awkward part: the current closure
  signature is `Fn(t, x, y, z, u) -> S`, which gives the value but
  not ∂R/∂u. Two options:
  1. **FD on the reaction closure only** — exact for the linear
     operator (which dominates), FD-noise contained to the diagonal
     reaction contribution. Cheapest fix, biggest immediate win.
  2. **Add a `with_reaction_jacobian` constructor** that takes a
     second closure for ∂R/∂u. Optional; falls back to (1) when
     not supplied.
- Wire the operator into `JacobianProvider::sparsity_pattern()` so
  Radau5 / BDF can exploit the band structure (5 nnz/row in 2D,
  7 nnz/row in 3D plus diagonal for the reaction).

**Why it matters**: stiff PDE problems (Fisher-KPP near saturation,
Allen-Cahn, advection-diffusion-reaction with stiff chemistry) are
exactly where Numra's Radau5 / BDF should beat explicit DoPri5, but
right now the FD-Jacobian fallback eats the win. Closing this gap
makes the stiff path *useful* for PDEs rather than nominally
supported.

**Out of scope for v1**; revisit alongside the broader sparse-Jacobian
work in `numra-ode`. Cite this entry when sequencing.

### MOL systems don't implement `ParametricOdeSystem`

**Status**: scoped, not started. Affects all three MOL wrappers.

Forward sensitivity (just shipped, 2026-05-06) requires
`ParametricOdeSystem` — an ODE system parameterised by a `&[S]` of
parameters with `jacobian_y` and `jacobian_p` methods. None of the
MOL wrappers implement it, which means a user cannot ask "how does
my Fisher-KPP solution depend on the diffusion coefficient and the
reaction rate?" without writing the parametric system by hand.

**What needs doing**:
- A `ParametricMOLSystem{2,3}D` variant (or a generic-over-parameter
  extension of the existing wrappers) where:
  - `alpha` becomes a parameter slot rather than a stored constant.
  - The reaction closure takes `(t, x, y, z, u, &[S])` instead of
    `(t, x, y, z, u)`.
  - Re-assembly happens lazily inside `rhs` when parameters change,
    or — better — `jacobian_p` is computed analytically because for
    the heat-equation case it's just the assembled Laplacian times
    the state.
- Decide whether this is a separate type or a flag on the existing
  type. Separate type avoids polluting the v1 ergonomic surface;
  paying for parametricity should be opt-in.

**Composes with**: §Solvers / AD-based `ParametricOdeSystem` impl —
once that adapter ships, a parametric MOL wrapper could use AD on
the reaction closure to get `∂R/∂p` for free, removing the manual
Jacobian-derivation burden.

**Out of scope for v1**; revisit when forward sensitivity has its
first PDE-shaped user workload.

### Reaction-term composability with the analytical-Jacobian path

**Status**: noted, blocks the two items above.

The current `with_reaction(|t, x, y, z, u| ...)` closure signature
returns only the reaction *value*, not its Jacobian or its parameter
derivatives. That's fine for the explicit-RHS path but it's the
single piece of friction blocking *both* `JacobianProvider` and
`ParametricOdeSystem` impls from being painless. When we revisit
either, this signature is the API decision to make first.

**Options**:
- Add sibling closures (`with_reaction_jacobian_u`,
  `with_reaction_jacobian_p`) that default to FD on the value
  closure when not supplied.
- Switch to a trait-based reaction (`trait Reaction<S> { fn value(...);
  fn jac_u(...); fn jac_p(...); }`) with a default FD impl. More
  rigorous; bigger surface area; consistent with how
  `ParametricOdeSystem` already works.

Either is a non-breaking addition. Pick when the first of the two
gaps above is being closed.

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

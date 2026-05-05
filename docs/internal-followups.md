# Internal follow-ups

Local-only tracking of work that's been considered, scoped, or partially
built, but isn't appropriate to publish on the public roadmap. Not linked
from the website. Not for marketing. The point is to have a written record
that survives session-context churn so we can pick threads back up cleanly.

When you act on something here, prefer moving it into a CHANGELOG entry,
a closed GitHub issue, or the public roadmap — and remove it from this
file once it lands. Stale follow-ups files are how good intentions become
embarrassments.

Last updated: 2026-05-05.

---

## Audit-driven scope corrections

These are not new work items — they're records of where earlier roadmap
drafts overstated or mis-scoped what we actually have. Kept here so the
next time someone drafts a public commitment based on memory rather than
the code, they catch the same correction without redoing the audit.

### "PDE: add 2D method-of-lines + finite element"

**The original draft claim**: ship 2D MOL plus FE in a Q1-2027-ish bucket.

**What the audit found** (2026-05-05):
- 2D MOL **already ships** — `numra-pde/src/mol2d.rs`, `equations2d.rs`
  with `HeatEquation2D`, `ReactionDiffusion2D`, Fisher, advection-diffusion;
  sparse 5-point Laplacian in `sparse_assembly.rs`; all four BC types
  tested.
- 3D has `Grid3D` + `BoundaryConditions3D` + 3D Laplacian assembly in
  `sparse_assembly.rs:228+`, but **no `MOLSystem3D`** wrapper — the
  pieces are there, the time-stepping integration isn't wired up.
- `Wave1D` is a documented stub (`equations.rs:129-151`) — a struct
  exists with a comment that it "would need the `PdeSystem` trait to
  support systems".
- **No FE infrastructure at all.** No spectral. No elliptic static solver.

**Correction**: "FE" was aspirational, not a gap to slot into a roadmap
without scoping the algorithm choice (CG-FEM? DG? hp-adaptive?). The
real, concrete PDE gaps to keep on the public roadmap as one bullet
("Expand PDE capabilities") are: 3D MOL wrapper, multi-component coupled
PDEs (hyperbolic / wave systems become first-class), elliptic static
solver path. FE is either a separate multi-quarter project or a
deliberate "not now".

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

### Rewrite the Radau5 step controller

**Status**: scoped, not started. Real work, not just a cleanup.

**What's there today**: `numra-ode/src/radau5.rs` ships an L-stable
3-stage Radau IIA scheme. It runs, but the step controller is conservative
to the point of pathology — on Van der Pol μ=10 at rtol=1e-4 it takes
~310k accepted steps where Hairer's reference RADAU implementation takes
hundreds. The §2.3 comparison page (`ch13-performance/comparisons.md`)
calls this out honestly.

**What needs doing**: revisit the step-size heuristic against
Hairer & Wanner II §IV.8, especially the order-prediction and
step-rejection logic. The Newton-iteration cap at 7 (`radau5.rs:552`) and
the convergence-rate test at `radau5.rs:719-734` likely interact badly
with the controller and force unnecessary restarts.

**Why it's a follow-up, not a public roadmap item**: it's a *fix*, not a
new capability. The user-visible promise is already "Radau5 works"; the
work here makes it work *fast*, which belongs in a release note when it
lands, not a roadmap "we plan to" entry.

### Forward sensitivity analysis — expose in solver API

**Status**: ~70% built. The pieces ship; the solver API doesn't.

**What's there today**:
- `numra-ode/src/sensitivity.rs:139-303` — `SensitivityEquations` trait
  and `AugmentedSystem` wrapper. Implements
  `dS/dt = (∂f/∂y)·S + ∂f/∂p`. Unit-tested on Lotka-Volterra.
- `numra-ode/src/uncertainty.rs` — already uses `AugmentedSystem`
  internally for trajectory-mode uncertainty (GUM/first-order Taylor).
- `numra/tests/integration_tests.rs` — `test_radau5_with_sensitivity`
  (uses *finite-difference* sensitivity, not the augmented-system path).

**What needs doing**:
- Solver-side sugar: `solve_with_forward_sensitivity()` on every implicit
  solver (Radau5, BDF, ESDIRK54), so users don't have to manually wrap.
- Public examples: a worked sensitivity case in `numra/examples/`
  (parameter sensitivity of Lorenz on σ, e.g.).
- A short book section under ch13 or a new chapter explaining when to
  reach for forward sensitivity vs finite differences.

**Why it's not on the public roadmap**: shipping a refinement of a
machinery that already exists doesn't read well as a roadmap "direction".
This is a release-note item, not a direction-of-travel item.

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

### Stiffness auto-detection (LSODA-equivalent)

**Status**: scoped, not started.

`auto_solve_with_hints` exists but the heuristic is mostly user-supplied
hints. SciPy's LSODA is the gold standard for "automatic stiffness
detection during the integration", and the §2.3 comparison page
acknowledges this is where SciPy beats Numra for first-time users picking
the wrong solver. Worth doing as a Numra-native algorithm rather than
porting LSODA.

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

### Sensitivity-analysis book chapter

**Status**: blocked on the forward-sensitivity API work above.

---

## Tooling

### CI: actionlint workflow

**Status**: not started.

Two recent edits to `.github/workflows/website.yml` could have shipped
typos that only surface on the next PR run. Adding actionlint as a
pre-commit or CI step would catch those before they hit a real PR.
30-minute task.

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

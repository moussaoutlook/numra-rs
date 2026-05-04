# Book–code coverage matrix

**States:** `covered` | `missing` | `incorrect` (must be empty before release).  
**Book TOC:** the sidebar config in `[website/book/astro.config.mjs](../../website/book/astro.config.mjs)` (Starlight's equivalent of `SUMMARY.md`).  
**Inventory:** `[public-api-method-inventory.md](public-api-method-inventory.md)` and machine inventory `[inventory.yaml](inventory.yaml)`.

Use this table to ensure every **user-facing numerical entry point** in the inventory has documentation. Chapter paths are relative to `website/book/src/content/docs/`.

---

## Differential equations and control


| Inventory area                                    | Primary book location                                                                                                 | State   | Notes                                         |
| ------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------- | ------- | --------------------------------------------- |
| ODE explicit RK (`ode.dopri5`, `ode.tsit5`, `ode.verner`) | `ch02-solving-odes/explicit-methods.md`, appendix `solver-reference.md`, `butcher-tableaux.md`                        | covered |                                               |
| ODE implicit (`ode.esdirk`, `ode.radau5`, `ode.bdf`, `ode.auto`) | `ch02-solving-odes/implicit-methods.md`, `ch02-solving-odes/automatic-selection.md`, `ch03-stiff-systems/*`, appendix | covered |                                               |
| Dense output, events (`ode.events`)               | `ch02-solving-odes/dense-output.md`, `event-detection.md`                                                             | covered |                                               |
| DAE, index reduction (`ode.dae`)                  | `ch03-stiff-systems/dae-systems.md`, `index-reduction.md`                                                             | covered |                                               |
| SDE (`sde.euler_maruyama`, `sde.milstein`, `sde.sra`) | `ch04-beyond-odes/stochastic-des.md`, `de-type-comparison.md`, appendix                                               | covered | Appendix corrected to match `numra-sde` types |
| DDE (`dde.method_of_steps`)                       | `ch04-beyond-odes/delay-des.md`, `de-type-comparison.md`, appendix                                                    | covered | Appendix corrected; no `DdeDoPri5`            |
| FDE (`fde.l1_caputo`)                             | `ch04-beyond-odes/fractional-des.md`                                                                                  | covered |                                               |
| IDE (`ide.volterra_prony`)                        | `ch04-beyond-odes/integro-des.md`                                                                                     | covered |                                               |
| PDE (`pde.mol`)                                   | `ch04-beyond-odes/partial-des.md`                                                                                     | covered |                                               |
| SPDE (`spde.heat`)                                | `ch04-beyond-odes/stochastic-pdes.md`                                                                                 | covered |                                               |
| OCP (`ocp.overview`)                              | `ch10-optimal-control/*`                                                                                              | covered |                                               |


---

## Linear algebra and nonlinear


| Inventory area                         | Primary book location                                                | State   | Notes                                  |
| -------------------------------------- | -------------------------------------------------------------------- | ------- | -------------------------------------- |
| Dense / sparse / factors / eigen / SVD (`linalg.matrix`) | `ch06-linear-algebra/*`                                              | covered | High-level; rustdoc for full API       |
| Iterative solvers + preconditioners (`linalg.matrix`) | `ch06-linear-algebra/iterative-solvers.md`                           | covered |                                        |
| Newton + line search (`nonlinear.newton`) | `ch01-introduction/architecture-overview.md` (Layer 1 + facade list) | covered | Deep usage: rustdoc `numra::nonlinear` |


---

## Optimization, calculus, signal, stats


| Inventory area                                           | Primary book location                                     | State   | Notes                                                                |
| -------------------------------------------------------- | --------------------------------------------------------- | ------- | -------------------------------------------------------------------- |
| Unconstrained / constrained / LM / global / MOO / robust (`optim.overview`) | `ch05-optimization/*`                                     | covered | Large API; chapter is overview-level                                 |
| LP / MILP / QP (`optim.lp_qp_milp`)                      | `ch05-optimization/linear-programming.md`                 | covered |                                                                      |
| Quadrature + rules + `dblquad` (`integrate.quadrature`)  | `ch07-calculus-and-analysis/numerical-integration.md`     | covered |                                                                      |
| Interpolation (`interp.curves`)                          | `ch07-calculus-and-analysis/interpolation.md`             | covered |                                                                      |
| Autodiff (`autodiff.gradients`)                          | `ch07-calculus-and-analysis/automatic-differentiation.md` | covered |                                                                      |
| Special functions (`special.functions`)                  | `ch07-calculus-and-analysis/special-functions.md`         | covered | May not enumerate every function; cross-link rustdoc                 |
| Curve fitting (`fit.curves`)                             | `ch07-calculus-and-analysis/curve-fitting.md`             | covered |                                                                      |
| FFT / spectral / convolution (`fft.spectral`)            | `ch08-signal-processing/fft-and-spectral.md`              | covered |                                                                      |
| Filters / Hilbert / resample / peaks (`dsp.signal`)      | `ch08-signal-processing/*`                                | covered | Book uses `numra_signal`; facade is `numra::dsp`—call out explicitly |
| Stats / tests / regression / correlation (`stats.overview`) | `ch09-statistics/*`                                       | covered |                                                                      |


---

## Core cross-cutting


| Inventory area                       | Primary book location                                                      | State   | Notes |
| ------------------------------------ | -------------------------------------------------------------------------- | ------- | ----- |
| `Scalar`, `Vector`, `Signal`, errors (`core.traits`) | `ch01-introduction/quick-start.md`, `ch12-advanced-topics/no-std-usage.md` | covered |       |
| Uncertainty / sensitivity (`core.traits`)            | `ch11-uncertainty/*`                                                       | covered |       |


---

## Maintenance

Re-scan chapters when adding solvers or re-exports; keep this matrix, `[inventory.yaml](inventory.yaml)`, and `[public-api-method-inventory.md](public-api-method-inventory.md)` in sync.
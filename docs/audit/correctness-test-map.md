# Correctness test map

This map links numerical regression tests to `docs/audit/inventory.yaml` ids. It records invariants and reference values, not formal proof claims.

| Inventory id | Test path | Invariant / reference |
| --- | --- | --- |
| `ode.events` | `numra-ode/tests/events_regression.rs` | Harmonic oscillator zero crossings are recorded near analytic times; terminal events stop at the first falling crossing; near-initial rising events are localized with a small step. |
| `ode.dae` | `numra-ode/tests/index_reduction_regression.rs` | Structural index analysis distinguishes direct index-1 algebraic matching from hidden constraints that require differentiation. |
| `sde.euler_maruyama` | `numra-sde/tests/gbm_ou_bands.rs` | Fixed-seed GBM terminal mean/variance stay within loose analytic bands. |
| `sde.euler_maruyama` | `numra-sde/tests/gbm_ou_bands.rs` | Fixed-seed Ornstein-Uhlenbeck terminal mean/variance stay within loose analytic bands. |
| `optim.lp_qp_milp` | `numra-optim/tests/integration.rs::test_tiny_lp_has_unique_golden_vertex` | Tiny LP reaches the unique vertex `x = [2, 2]`, objective `-6`. |
| `optim.lp_qp_milp` | `numra-optim/tests/integration.rs::test_tiny_qp_has_unconstrained_golden_minimizer` | Convex diagonal QP reaches the analytic minimizer `x = [1, 2]`, objective `-9`. |
| `optim.lp_qp_milp` | `numra-optim/tests/integration.rs::test_tiny_milp_has_unique_binary_solution` | Binary knapsack MILP reaches the unique solution `x = [1, 0, 1]`, objective `-7`. |

## Existing supporting coverage

| Inventory id | Test path | Invariant / reference |
| --- | --- | --- |
| `ode.dopri5`, `ode.tsit5`, `ode.esdirk`, `ode.radau5` | `numra-ode/tests/convergence_order_tests.rs` | Explicit and implicit solver families preserve expected convergence behavior on analytic ODEs. |
| `ode.bdf` | `numra-ode/tests/bdf_algorithm_tests.rs` | BDF coefficients and algorithmic helper behavior remain stable. |
| `ode.dae` | `numra-ode/tests/dae_tests.rs` | Consistent initialization and small DAE examples solve through implicit methods. |
| `sde.euler_maruyama` | `numra/tests/composition_tests.rs::test_sde_reproducibility` | Fixed seed SDE solve is reproducible through the facade crate. |
| `linalg.matrix` | `numra-linalg/tests/dense_solve_smoke.rs` | Dense diagonal linear solve returns the known solution. |
| `integrate.quadrature` | `numra-integrate/tests/quadrature_smoke.rs` | Quadrature approximates analytic integrals within tolerance. |
| `interp.curves` | `numra-interp/tests/linear_endpoints.rs` | Linear interpolation hits knot endpoints exactly. |
| `fft.spectral` | `numra-fft/tests/fft_roundtrip.rs` | FFT/IFFT round trip reconstructs the input. |

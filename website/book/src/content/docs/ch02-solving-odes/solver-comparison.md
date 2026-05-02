---
title: "Solver Comparison"
---

This page provides comprehensive comparison tables to help you choose the right
solver for your problem.

## At a Glance

| Solver | Type | Order | Stiff? | DAE? | Best For |
|--------|------|-------|--------|------|----------|
| **DoPri5** | Explicit RK | 5(4) | No | No | General purpose non-stiff |
| **Tsit5** | Explicit RK | 5(4) | No | No | Slightly more efficient than DoPri5 |
| **Vern6** | Explicit RK | 6(5) | No | No | Medium-high accuracy |
| **Vern7** | Explicit RK | 7(6) | No | No | High accuracy |
| **Vern8** | Explicit RK | 8(7) | No | No | Very high accuracy |
| **Radau5** | Implicit RK | 5 | Yes | Yes | Stiff + high accuracy |
| **Esdirk32** | ESDIRK | 2(1) | Yes | No | Mildly stiff, cheap |
| **Esdirk43** | ESDIRK | 3(2) | Yes | No | Moderately stiff |
| **Esdirk54** | ESDIRK | 4(3) | Yes | No | General stiff |
| **Bdf** | Multistep | 1-5 | Yes | Yes | Very stiff, variable order |
| **Auto** | Adaptive | varies | Yes | Partial | Don't know / first try |

## Stability Properties

| Solver | A-stable | L-stable | A(alpha)-stable | Notes |
|--------|----------|----------|-----------------|-------|
| DoPri5 | No | No | No | Explicit -- no stability for stiff |
| Tsit5 | No | No | No | Same as DoPri5 |
| Vern6/7/8 | No | No | No | Explicit methods |
| Radau5 | Yes | **Yes** | Yes | Best for stiff: no oscillation damping issues |
| Esdirk32 | Yes | **Yes** | Yes | L-stable |
| Esdirk43 | Yes | **Yes** | Yes | L-stable |
| Esdirk54 | Yes | **Yes** | Yes | L-stable like Radau5 |
| Bdf-1 | Yes | **Yes** | Yes | Backward Euler |
| Bdf-2 | Yes | **Yes** | Yes | Most commonly used BDF order |
| Bdf-3,4,5 | No | No | Yes | A(alpha)-stable only |

**L-stability** means the method completely damps transients at infinity,
which is critical for problems where fast modes should die out quickly.
`Radau5`, all ESDIRK variants, and `Bdf-1/2` are L-stable.

## Cost Per Step

| Solver | RHS evals/step | Linear solves/step | Memory | Startup cost |
|--------|---------------|-------------------|--------|-------------|
| DoPri5 | 6 (FSAL) | 0 | O(dim) | None |
| Tsit5 | 6 (FSAL) | 0 | O(dim) | None |
| Vern6 | 9 (FSAL) | 0 | O(dim) | None |
| Vern7 | 10 | 0 | O(dim) | None |
| Vern8 | 13 | 0 | O(dim) | None |
| Radau5 | 3-5/Newton iter | 2 (real+complex) | O(dim^2) | Jacobian + LU |
| Esdirk32 | 2-3 | 2 | O(dim^2) | Jacobian + LU |
| Esdirk43 | 3-4 | 3 | O(dim^2) | Jacobian + LU |
| Esdirk54 | 4-5 | 5 | O(dim^2) | Jacobian + LU |
| Bdf | 1-3/Newton iter | 1 | O(dim^2) | Jacobian + LU |

Implicit methods require O(dim^2) memory for the Jacobian and O(dim^3) work
for LU factorization. For small systems (dim < 100), this overhead is
negligible. For large systems, consider sparse Jacobians or matrix-free methods.

## Decision Flowchart

```text
Is your problem stiff?
  |
  +-- No ---> Is high accuracy needed (rtol < 1e-8)?
  |             |
  |             +-- No ---> DoPri5 or Tsit5
  |             +-- Yes --> Vern6, Vern7, or Vern8
  |
  +-- Yes --> Is it a DAE (mass matrix)?
  |             |
  |             +-- Yes --> Radau5 or Bdf
  |             +-- No ---> How stiff?
  |                           |
  |                           +-- Mildly stiff --> Esdirk54
  |                           +-- Very stiff ----> Radau5 (high accuracy)
  |                           |                    or Bdf (moderate accuracy)
  |
  +-- Unknown -> Auto (tries non-stiff first, falls back to stiff)
```

## Performance on Test Problems

The following table shows function evaluations and final error for each solver
on standard benchmark problems (rtol = 1e-6, atol = 1e-9).

### Non-Stiff Problems

| Problem | DoPri5 | Tsit5 | Vern6 | Vern8 |
|---------|--------|-------|-------|-------|
| **Exponential decay** | | | | |
| y' = -y, t=[0,10] | ~60 evals | ~55 evals | ~50 evals | ~45 evals |
| Error | ~1e-7 | ~1e-7 | ~1e-8 | ~1e-9 |
| **Harmonic oscillator** | | | | |
| x'' = -x, t=[0,100] | ~350 evals | ~320 evals | ~280 evals | ~250 evals |
| Error | ~1e-6 | ~1e-6 | ~1e-7 | ~1e-8 |
| **Lorenz system** | | | | |
| sigma=10, rho=28, t=[0,20] | ~2100 evals | ~1900 evals | ~1700 evals | ~1500 evals |
| Error* | ~1e-5 | ~1e-5 | ~1e-6 | ~1e-7 |

*Lorenz error measured at short time (t=5) due to chaos.

### Stiff Problems

| Problem | Radau5 | Esdirk54 | Bdf | DoPri5 |
|---------|--------|----------|-----|--------|
| **Van der Pol (mu=1000)** | | | | |
| t=[0, 3000] | ~800 evals | ~1200 evals | ~600 evals | FAILS |
| LU factorizations | ~200 | ~300 | ~500 | -- |
| **Robertson** | | | | |
| t=[0, 1e11] | ~250 evals | ~400 evals | ~200 evals | FAILS |
| LU factorizations | ~80 | ~120 | ~180 | -- |

Explicit solvers (DoPri5, Tsit5, Verner) **fail** on stiff problems -- they
require impossibly small step sizes to maintain stability.

## Choosing Between Stiff Solvers

| Feature | Radau5 | Esdirk54 | Bdf |
|---------|--------|----------|-----|
| **Order** | 5 | 4 | 1-5 (variable) |
| **Stability** | L-stable | L-stable | A(alpha) for order > 2 |
| **Cost/step** | High | Medium | Low |
| **Step size** | Large | Medium | Large |
| **Best tolerance** | rtol < 1e-6 | rtol ~ 1e-4 to 1e-6 | rtol ~ 1e-3 to 1e-6 |
| **DAE support** | Yes (index-1) | No | Yes (index-1) |
| **Startup** | Slow (large LU) | Fast | Slow (ramp-up) |
| **Memory** | 3x Jacobian | 1x Jacobian | 1x Jacobian |

**Rules of thumb:**

- **Radau5**: Best for stiff problems requiring high accuracy. The higher order
  and L-stability make it ideal when you need 6+ correct digits.
- **Esdirk54**: Good middle ground. Cheaper per step than Radau5, L-stable,
  but lower order means more steps at tight tolerances.
- **Bdf**: Best for very stiff problems at moderate accuracy. Variable order
  adapts to the problem. Cheapest per step (single LU solve), but needs
  several steps to "ramp up" to high order.

## Auto Solver Strategy

The `Auto` solver uses a two-phase strategy:

1. **Classification**: Samples the Jacobian to estimate stiffness
   - Ratio of max/min Jacobian elements > 10,000: Very stiff
   - Ratio > 100: Moderately stiff
   - Otherwise: Non-stiff

2. **Solver selection**: Based on stiffness and requested accuracy
   - Non-stiff + standard accuracy: Tsit5
   - Non-stiff + high accuracy: Vern6 or Vern8
   - Moderately stiff: Esdirk54
   - Very stiff + standard accuracy: Bdf
   - Very stiff + high accuracy: Radau5
   - Unknown: Try Tsit5, fallback to Esdirk54 on failure

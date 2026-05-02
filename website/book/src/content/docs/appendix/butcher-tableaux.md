---
title: "Butcher Tableaux"
---

A Butcher tableau encodes the coefficients of a Runge-Kutta method. For an
$ s $-stage method, the tableau has the form:

```text
c_1 | a_11  a_12  ...  a_1s
c_2 | a_21  a_22  ...  a_2s
 :  |  :     :          :
c_s | a_s1  a_s2  ...  a_ss
----+----------------------
    | b_1   b_2   ...  b_s      (order p weights)
    | b*_1  b*_2  ...  b*_s     (order p-1 weights, for error estimate)
```

The coefficients define:
- **c**: Node points (when to sample within each step)
- **A**: Stage coupling (how stages depend on each other)
- **b**: Weights for the higher-order solution
- **b\***: Weights for the embedded lower-order solution (error estimate)

Row-sum condition: $ c_i = \sum_j a_{ij} $ for each $ i $.

## Explicit Methods

For explicit methods, $ A $ is strictly lower triangular ($ a_{ij} = 0 $ for $ j \geq i $).

### DoPri5 (Dormand-Prince 5(4))

7 stages, order 5(4). The most widely used adaptive RK method.

```text
0      |
1/5    | 1/5
3/10   | 3/40        9/40
4/5    | 44/45      -56/15       32/9
8/9    | 19372/6561 -25360/2187  64448/6561  -212/729
1      | 9017/3168  -355/33      46732/5247   49/176    -5103/18656
1      | 35/384      0           500/1113     125/192   -2187/6784    11/84
-------+-------------------------------------------------------------------
       | 35/384      0           500/1113     125/192   -2187/6784    11/84     0
       | 5179/57600  0           7571/16695   393/640   -92097/339200 187/2100  1/40
```

**Properties:** Not FSAL. The 5th-order solution and 4th-order error estimator
share the same stages.

**Reference:** Dormand & Prince (1980).

### Tsit5 (Tsitouras 5(4))

7 stages, order 5(4). FSAL (First Same As Last).

The Tsitouras method achieves the same order as DoPri5 but with the FSAL
property: the 7th stage of step $ n $ equals the 1st stage of step $ n+1 $, effectively
requiring only 6 RHS calls per accepted step.

**Properties:** FSAL. Optimized for minimal truncation error coefficients.

**Reference:** Tsitouras (2011).

### Vern6 (Verner 6(5))

9 stages, order 6(5).

Verner's "most efficient" 6th-order pair. Provides 6th-order accuracy with
a 5th-order embedded error estimator. The extra stages (compared to 5th-order
methods) buy an additional order of convergence, which pays off at tolerances
tighter than about $ 10^{-8} $.

**Properties:** Not FSAL. Optimized for minimum truncation error.

**Reference:** Verner (2010).

### Vern7 (Verner 7(6))

10 stages, order 7(6).

**Properties:** Not FSAL. Best efficiency in the $ \text{rtol} = 10^{-10} $ to $ 10^{-12} $ range.

### Vern8 (Verner 8(7))

13 stages, order 8(7).

The highest-order explicit method in Numra. Each step requires 13 RHS calls,
but at very tight tolerances the large steps compensate. Most efficient when
$ \text{rtol} < 10^{-12} $.

**Properties:** Not FSAL.

## Implicit Methods

For implicit methods, $ A $ has nonzero entries on or above the diagonal.

### ESDIRK Methods

ESDIRK (Explicit first stage, Singly Diagonally Implicit Runge-Kutta) methods
have the structure:

```text
0     | 0
c_2   | a_21   gamma
c_3   | a_31   a_32   gamma
 :    |  :      :       :     gamma
------+--------------------------------
      | b_1    b_2    ...     b_s
      | b*_1   b*_2   ...     b*_s
```

The key feature: all implicit stages share the same diagonal coefficient $ \gamma $.
This means the LU factorization $ (I - h \gamma J) $ can be reused across all stages
within a step.

**ESDIRK32**: 3 stages, order 2(1), $ \gamma = 0.2928932188\ldots $ (ESDIRK2(1)3L[2]SA)

**ESDIRK43**: 4 stages, order 3(2), $ \gamma = 0.4358665215\ldots $ (Kvaerno3)

**ESDIRK54**: 6 stages, order 4(3), $ \gamma = 0.25 $ (ESDIRK4(3)6L[2]SA)

All ESDIRK methods are L-stable and A-stable.

**References:** Kennedy & Carpenter (2016) for Esdirk32 and Esdirk54; Kvaerno (2004) for Esdirk43.

### Radau5 (Radau IIA)

3 implicit stages, order 5.

The Radau IIA method uses the collocation approach. The three stages correspond
to solving at the Gauss-Radau quadrature points:

```text
c_1 = (4 - sqrt(6)) / 10
c_2 = (4 + sqrt(6)) / 10
c_3 = 1
```

The method solves a coupled $ 3n \times 3n $ system at each step (where $ n $ is the ODE
dimension). This is expensive per step but allows very large steps through
stiff regions.

**Properties:**
- L-stable and A-stable
- Suitable for DAEs (index 1 and some index 2)
- Handles stiffness ratios exceeding $ 10^{11} $
- Uses simplified Newton iteration with the transformation to real Schur form

**Reference:** Hairer & Wanner (1996), Chapter IV.

## BDF Methods

BDF (Backward Differentiation Formulas) are multistep methods, not Runge-Kutta,
so they don't have Butcher tableaux. Instead, they use previous solution values:

### BDF order $ k $ formula

The $ k $-step BDF method for $ y' = f(t, y) $:

| Order | Formula | Stability |
|-------|---------|-----------|
| 1 | $ y_{n+1} = y_n + h f_{n+1} $ | A-stable |
| 2 | $ \tfrac{3}{2} y_{n+1} - 2 y_n + \tfrac{1}{2} y_{n-1} = h f_{n+1} $ | A-stable |
| 3 | $ \tfrac{11}{6} y_{n+1} - 3 y_n + \tfrac{3}{2} y_{n-1} - \tfrac{1}{3} y_{n-2} = h f_{n+1} $ | $ A(\alpha) $-stable |
| 4 | $ \tfrac{25}{12} y_{n+1} - 4 y_n + 3 y_{n-1} - \tfrac{4}{3} y_{n-2} + \tfrac{1}{4} y_{n-3} = h f_{n+1} $ | $ A(\alpha) $-stable |
| 5 | $ \tfrac{137}{60} y_{n+1} - 5 y_n + 5 y_{n-1} - \tfrac{10}{3} y_{n-2} + \tfrac{5}{4} y_{n-3} - \tfrac{1}{5} y_{n-4} = h f_{n+1} $ | $ A(\alpha) $-stable |
| 6 | Unstable -- not used | Zero-unstable |

Numra's BDF solver uses variable order (1 to 5), automatically selecting the
order that balances accuracy and stability for the current step.

## Verifying Coefficients

All Butcher tableaux in Numra have been verified against the original published
papers. The key checks are:

1. **Row-sum condition**: $ c_i = \sum_j a_{ij} $ (within floating-point precision)
2. **Order conditions**: $ B(p) $ order conditions for the stated order
3. **Embedded pair**: The error estimate $ b - b^* $ is of the correct order
4. **Stability**: A-stability / L-stability verified via the stability function

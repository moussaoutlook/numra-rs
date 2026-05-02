---
title: "What Is Stiffness?"
---

Stiffness is perhaps the single most important concept in computational ODE solving.
A stiff problem is one where standard explicit methods (like DoPri5 or Tsit5) are
forced to take absurdly small time steps -- not for accuracy, but for numerical
stability. Understanding stiffness lets you pick the right solver and avoid
wasting hours of compute time on what should be a trivial integration.

## A First Example

Consider two simple decay equations:

**Non-stiff:**
$$
y' = -y, \quad y(0) = 1
$$

**Stiff:**
$$
y' = -1000\,y, \quad y(0) = 1
$$

Both have the same form and the exact solution $ y(t) = e^{-\lambda t} $. The
second decays much faster, but its solution is equally smooth. So why is it harder
to solve numerically?

The answer lies in the *stability region* of the solver. An explicit method like
forward Euler requires the step size $ h $ to satisfy $ |1 + h\lambda| < 1 $.
For $ \lambda = -1000 $, this means $ h < 0.002 $. Even if you only care about
the solution at $ t = 10 $ (where $ y \approx 0 $ to machine precision), an
explicit method must still take thousands of tiny steps, each obeying the stability
constraint rather than the accuracy requirement.

## The Eigenvalue Interpretation

For a linear system $ \mathbf{y}' = A\,\mathbf{y} $, the eigenvalues
$ \lambda_1, \lambda_2, \ldots, \lambda_n $ of $ A $ determine the behavior of
each solution mode. Stiffness arises when these eigenvalues span a wide range:

$$
\text{Stiffness ratio} = \frac{\max_i |\operatorname{Re}(\lambda_i)|}{\min_i |\operatorname{Re}(\lambda_i)|}
$$

When this ratio is large (say, $ > 100 $), the system is stiff. The fast modes
(large $ |\lambda| $) decay quickly and become negligible, but they continue to
constrain the step size of explicit methods long after they contribute nothing to
the solution.

For nonlinear systems $ \mathbf{y}' = f(t, \mathbf{y}) $, the Jacobian
$ J = \partial f / \partial \mathbf{y} $ plays the role of $ A $. Its eigenvalues
change with time and state, so stiffness can be *localized* -- a problem might be
stiff in one region and non-stiff in another.

## Timescale Separation

Physically, stiffness corresponds to the presence of multiple timescales:

| Timescale | Eigenvalue | Role |
|-----------|-----------|------|
| Fast ($ \tau_{\text{fast}} \approx 1/\vert \lambda_{\max}\vert  $) | Large $ \vert \lambda\vert  $ | Transients that decay rapidly |
| Slow ($ \tau_{\text{slow}} \approx 1/\vert \lambda_{\min}\vert  $) | Small $ \vert \lambda\vert  $ | The dynamics you actually care about |

The ratio $ \tau_{\text{slow}} / \tau_{\text{fast}} $ is the stiffness ratio.
When it exceeds $ \sim 10^3 $, explicit methods become impractical.

**Example: Chemical kinetics.** A combustion reaction might involve radical species
with lifetimes of $ 10^{-9} $ seconds and bulk temperature changes over
$ 10^{-3} $ seconds -- a stiffness ratio of $ 10^6 $.

## A Precise Definition

There is no single universally accepted definition of stiffness, but the most
practical one comes from Lambert (1991):

> A system is **stiff** if the step size required for stability is much smaller
> than the step size required for accuracy.

Equivalently, a problem is stiff over the interval $ [t_0, t_f] $ if:

1. All eigenvalues of the Jacobian $ J(t, y) $ have negative real parts
   (the system is stable), and
2. The stiffness ratio $ \max |\operatorname{Re}(\lambda)| / \min |\operatorname{Re}(\lambda)| \gg 1 $.

Note that condition (1) is important: if an eigenvalue has a positive real part,
the solution is genuinely growing fast and small steps are needed for *accuracy*,
not just stability. That is not stiffness -- that is a genuinely fast phenomenon.

## Visual Intuition: The Phase Portrait

Consider the 2D system:

$$
\begin{aligned}
y_1' &= -1000\,y_1 + y_2 \\\\
y_2' &= -y_2
\end{aligned}
$$

The Jacobian is:

$$
J = \begin{pmatrix} -1000 & 1 \\\\ 0 & -1 \end{pmatrix}
$$

with eigenvalues $ \lambda_1 = -1000 $ and $ \lambda_2 = -1 $. In the phase
portrait:

- **Fast direction** ($ \lambda_1 = -1000 $): trajectories snap almost
  instantaneously onto the *slow manifold* defined by $ y_1 \approx y_2 / 1000 $.
- **Slow direction** ($ \lambda_2 = -1 $): trajectories then drift slowly along
  the manifold toward the origin.

An explicit solver must resolve the fast snap even during the slow drift, wasting
enormous computational effort.

## Classic Stiff Problems

### Van der Pol Oscillator

$$
\begin{aligned}
y_1' &= y_2 \\\\
y_2' &= \mu\,(1 - y_1^2)\,y_2 - y_1
\end{aligned}
$$

For small $ \mu $, this is non-stiff. For $ \mu \gg 1 $, sharp transitions
alternate with slow relaxation phases, making it severely stiff.

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust
use numra::ode::{Radau5, Solver, OdeProblem, SolverOptions};

let mu = 1000.0;
let problem = OdeProblem::new(
    move |_t, y: &[f64], dydt: &mut [f64]| {
        dydt[0] = y[1];
        dydt[1] = mu * (1.0 - y[0] * y[0]) * y[1] - y[0];
    },
    0.0, 2000.0, vec![2.0, 0.0],
);

let options = SolverOptions::default().rtol(1e-6).atol(1e-8);
let result = Radau5::solve(&problem, 0.0, 2000.0, &[2.0, 0.0], &options)
    .expect("Radau5 should handle stiff Van der Pol");
```

### Robertson Chemical Kinetics

$$
\begin{aligned}
y_1' &= -0.04\,y_1 + 10^4\,y_2\,y_3 \\\\
y_2' &= 0.04\,y_1 - 10^4\,y_2\,y_3 - 3 \times 10^7\,y_2^2 \\\\
y_3' &= 3 \times 10^7\,y_2^2
\end{aligned}
$$

This has rate constants spanning 11 orders of magnitude ($ 0.04 $ to
$ 3 \times 10^7 $), making it one of the classic test problems for stiff solvers.

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust
use numra::ode::{Radau5, Solver, OdeProblem, SolverOptions};

let problem = OdeProblem::new(
    |_t, y: &[f64], dydt: &mut [f64]| {
        dydt[0] = -0.04 * y[0] + 1e4 * y[1] * y[2];
        dydt[1] =  0.04 * y[0] - 1e4 * y[1] * y[2] - 3e7 * y[1] * y[1];
        dydt[2] =  3e7 * y[1] * y[1];
    },
    0.0, 1e11, vec![1.0, 0.0, 0.0],
);

let options = SolverOptions::default().rtol(1e-4).atol(1e-8);
let result = Radau5::solve(&problem, 0.0, 1e11, &[1.0, 0.0, 0.0], &options)
    .expect("Robertson system is severely stiff");
```

### Stiff Linear System

$$
\mathbf{y}' = \begin{pmatrix} -1000 & 1 \\\\ 0 & -1 \end{pmatrix} \mathbf{y}
$$

Stiffness ratio of 1000. The solution quickly approaches the slow manifold
and then decays exponentially with rate 1.

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust
use numra::ode::{Esdirk54, Solver, OdeProblem, SolverOptions};

let problem = OdeProblem::new(
    |_t, y: &[f64], dydt: &mut [f64]| {
        dydt[0] = -1000.0 * y[0] + y[1];
        dydt[1] = -y[1];
    },
    0.0, 10.0, vec![1.0, 1.0],
);

let options = SolverOptions::default().rtol(1e-6).atol(1e-9);
let result = Esdirk54::solve(&problem, 0.0, 10.0, &[1.0, 1.0], &options)
    .expect("ESDIRK handles moderate stiffness well");
```

## Why Explicit Methods Fail

An explicit Runge-Kutta method (like DoPri5 or Tsit5) has a *bounded* stability
region in the complex $ h\lambda $-plane. For a generic 4th-order method, the
stability region extends roughly to $ |h\lambda| \lesssim 2.8 $ along the
negative real axis.

This means the step size must satisfy:

$$
h \leq \frac{C}{|\lambda_{\max}|}
$$

where $ C \approx 2{-}3 $ depends on the method. When
$ |\lambda_{\max}| = 10^6 $, this forces $ h \leq 3 \times 10^{-6} $ regardless
of accuracy needs.

**Implicit methods** (Radau5, BDF, ESDIRK) solve a nonlinear system at each step,
which makes each step more expensive but gives them *unbounded* stability regions
(A-stability or L-stability). They can take large steps even when fast modes are
present, because the implicit formulation automatically damps them.

| Aspect | Explicit (DoPri5, Tsit5) | Implicit (Radau5, BDF) |
|--------|--------------------------|------------------------|
| Stability region | Bounded | Unbounded (A-stable) |
| Step size limit | $ h \leq C / \vert \lambda_{\max}\vert  $ | Limited only by accuracy |
| Cost per step | $ \mathcal{O}(s \cdot n) $ | $ \mathcal{O}(n^3) $ (LU solve) |
| Stiff problems | Catastrophically slow | Efficient |
| Non-stiff problems | Efficient | Overkill (wasted LU cost) |

## The Bottom Line

- **If your solver takes millions of tiny steps but the solution looks smooth**, your
  problem is probably stiff.
- **Stiffness is about eigenvalue spread**, not problem complexity.
- **Use implicit solvers** (Radau5, BDF, ESDIRK) for stiff problems.
- **Use explicit solvers** (DoPri5, Tsit5, Vern) for non-stiff problems.
- **Numra's `Auto` solver** detects stiffness automatically and picks the right method.

The next section shows how to detect stiffness before (or during) integration.

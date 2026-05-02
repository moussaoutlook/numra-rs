# Iterative Solvers

When matrices are too large for direct factorization, iterative (Krylov)
methods solve \\(Ax = b\\) using only matrix-vector products. Each iteration
costs \\(O(\text{nnz})\\) instead of the \\(O(n^3)\\) needed by LU or
Cholesky, making these methods practical for systems with millions of unknowns.

Numra provides five iterative solvers and three preconditioners, all operating
on `SparseMatrix`.

## IterativeOptions

All solvers share a common options struct configured with a builder pattern:

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
use numra::linalg::IterativeOptions;

let opts = IterativeOptions::default()
    .tol(1e-10)         // relative residual tolerance
    .max_iter(1000)     // maximum iterations
    .restart(30);       // GMRES restart parameter

// Defaults: tol = 1e-8, max_iter = 1000, restart = 30
```

The convergence criterion is:

\\[
\frac{\lVert b - Ax_k \rVert}{\lVert b \rVert} < \text{tol}
\\]

## IterativeResult

Every solver returns an `IterativeResult` containing:

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
pub struct IterativeResult<S: Scalar> {
    pub x: Vec<S>,            // solution vector
    pub residual_norm: S,     // final ||b - Ax||
    pub iterations: usize,    // iterations performed
    pub converged: bool,      // whether tolerance was met
}
```

Always check `converged` before trusting the result.

## Conjugate Gradient (CG)

The CG method is the gold standard for symmetric positive definite (SPD)
systems. It minimizes \\(\lVert x - x^* \rVert_A\\) over Krylov subspaces
and converges in at most \\(n\\) iterations in exact arithmetic.

### When to Use

- Matrix is symmetric (\\(A = A^T\\)) and positive definite
- Typical sources: Laplacians, mass matrices, FEM stiffness matrices, covariance matrices

### API

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
use numra::linalg::{SparseMatrix, cg, IterativeOptions};

// 1D Laplacian: tridiag(-1, 2, -1)
let n = 100;
let mut triplets = Vec::new();
for i in 0..n {
    triplets.push((i, i, 2.0));
    if i > 0     { triplets.push((i, i - 1, -1.0)); }
    if i < n - 1 { triplets.push((i, i + 1, -1.0)); }
}
let a = SparseMatrix::from_triplets(n, n, &triplets).unwrap();

let b: Vec<f64> = (0..n).map(|i| (i + 1) as f64).collect();
let opts = IterativeOptions::default().tol(1e-10);

let result = cg(&a, &b, None, &opts).unwrap();
assert!(result.converged);
println!("CG converged in {} iterations", result.iterations);
```

### Initial Guess

Providing a good initial guess can significantly reduce iterations:

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
use numra::linalg::{SparseMatrix, cg, IterativeOptions};

// ... (build matrix a and RHS b) ...
# let n = 20;
# let mut triplets = Vec::new();
# for i in 0..n {
#     triplets.push((i, i, 2.0));
#     if i > 0 { triplets.push((i, i - 1, -1.0)); }
#     if i < n - 1 { triplets.push((i, i + 1, -1.0)); }
# }
# let a = SparseMatrix::from_triplets(n, n, &triplets).unwrap();
# let b: Vec<f64> = (0..n).map(|i| (i + 1) as f64).collect();

// Use a previous solution as initial guess
let x0: Vec<f64> = (0..n).map(|i| i as f64 + 0.9).collect();
let opts = IterativeOptions::default().tol(1e-10);

let result = cg(&a, &b, Some(&x0), &opts).unwrap();
```

### Convergence Rate

CG convergence depends on the condition number \\(\kappa(A)\\):

\\[
\lVert x_k - x^* \rVert_A \leq 2 \left(\frac{\sqrt{\kappa} - 1}{\sqrt{\kappa} + 1}\right)^k \lVert x_0 - x^* \rVert_A
\\]

For the 1D Laplacian, \\(\kappa = O(n^2)\\), so CG needs \\(O(n)\\) iterations
without preconditioning. With a good preconditioner, this drops dramatically.

## Preconditioned Conjugate Gradient (PCG)

PCG solves the transformed system \\(M^{-1}Ax = M^{-1}b\\) where \\(M\\)
approximates \\(A\\). The effective condition number
\\(\kappa(M^{-1}A) \ll \kappa(A)\\) accelerates convergence.

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
use numra::linalg::{SparseMatrix, pcg, IterativeOptions, Jacobi};

// Build SPD system
let n = 100;
let mut triplets = Vec::new();
for i in 0..n {
    triplets.push((i, i, 2.0));
    if i > 0     { triplets.push((i, i - 1, -1.0)); }
    if i < n - 1 { triplets.push((i, i + 1, -1.0)); }
}
let a = SparseMatrix::from_triplets(n, n, &triplets).unwrap();
let b: Vec<f64> = (0..n).map(|i| (i + 1) as f64).collect();

// Apply Jacobi preconditioning
let precond = Jacobi::new(&a);
let opts = IterativeOptions::default().tol(1e-10);

let result = pcg(&a, &b, &precond, None, &opts).unwrap();
assert!(result.converged);
```

## GMRES (Generalized Minimal Residual)

GMRES is the standard method for general non-symmetric systems. It minimizes
\\(\lVert b - Ax_k \rVert\\) over the Krylov subspace
\\(\mathcal{K}_k(A, r_0)\\) using the Arnoldi process.

### When to Use

- Matrix is non-symmetric (convection-diffusion, Navier-Stokes linearizations)
- Matrix is well-conditioned or preconditioned

### Restarted GMRES

Memory grows linearly with iteration count (storing the Krylov basis), so
GMRES restarts after `opts.restart` iterations:

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
use numra::linalg::{SparseMatrix, gmres, IterativeOptions};

// Non-symmetric: convection-diffusion operator
let n = 50;
let eps = 0.1; // convection strength
let mut triplets = Vec::new();
for i in 0..n {
    triplets.push((i, i, 2.0));
    if i > 0     { triplets.push((i, i - 1, -1.0 - eps)); }
    if i < n - 1 { triplets.push((i, i + 1, -1.0 + eps)); }
}
let a = SparseMatrix::from_triplets(n, n, &triplets).unwrap();

let b: Vec<f64> = (0..n).map(|i| (i as f64).sin()).collect();
let opts = IterativeOptions::default()
    .tol(1e-10)
    .restart(30);  // restart every 30 iterations

let result = gmres(&a, &b, None, &opts).unwrap();
assert!(result.converged);
```

### Effect of Restart Parameter

| Restart | Memory per cycle | Convergence |
|---------|-----------------|-------------|
| Small (5-10) | Low | May stagnate on hard problems |
| Medium (30) | Moderate | Good default for most problems |
| Large (n) | Full GMRES | Guaranteed convergence, high memory |

## BiCGSTAB (Bi-Conjugate Gradient Stabilized)

BiCGSTAB is an alternative to GMRES for non-symmetric systems. Unlike GMRES,
it has fixed memory cost per iteration (no restart needed) but can exhibit
irregular convergence.

### When to Use

- Non-symmetric systems
- Memory is limited (fixed cost per iteration, unlike GMRES)
- GMRES with restart stagnates

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
use numra::linalg::{SparseMatrix, bicgstab, IterativeOptions};

let n = 50;
let eps = 0.2;
let mut triplets = Vec::new();
for i in 0..n {
    triplets.push((i, i, 2.0));
    if i > 0     { triplets.push((i, i - 1, -1.0 - eps)); }
    if i < n - 1 { triplets.push((i, i + 1, -1.0 + eps)); }
}
let a = SparseMatrix::from_triplets(n, n, &triplets).unwrap();

let b: Vec<f64> = (0..n).map(|i| (i as f64).cos()).collect();
let opts = IterativeOptions::default().tol(1e-10);

let result = bicgstab(&a, &b, None, &opts).unwrap();
assert!(result.converged);
```

## MINRES (Minimum Residual)

MINRES is designed for symmetric (possibly indefinite) systems. Unlike CG, it
does not require positive definiteness -- it works as long as \\(A\\) is
symmetric.

### When to Use

- Symmetric indefinite systems (saddle-point problems, augmented Lagrangian)
- KKT systems from constrained optimization
- Any symmetric system where CG would fail due to indefiniteness

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
use numra::linalg::{SparseMatrix, minres, IterativeOptions};

// Symmetric indefinite: alternating signs on diagonal
let n = 5;
let mut triplets = Vec::new();
for i in 0..n {
    let sign = if i % 2 == 0 { -1.0 } else { 2.0 };
    triplets.push((i, i, sign * 3.0));
    if i > 0     { triplets.push((i, i - 1, 1.0)); }
    if i < n - 1 { triplets.push((i, i + 1, 1.0)); }
}
let a = SparseMatrix::from_triplets(n, n, &triplets).unwrap();

let b = vec![1.0, 2.0, 3.0, 4.0, 5.0];
let opts = IterativeOptions::default().tol(1e-10);

let result = minres(&a, &b, None, &opts).unwrap();
assert!(result.converged);
```

## Preconditioners

Preconditioning transforms the system so that the effective condition number is
smaller, dramatically reducing the number of iterations needed. Numra provides
three preconditioners, all implementing the `Preconditioner` trait:

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
pub trait Preconditioner<S: Scalar> {
    /// Compute z = M^{-1} * r
    fn apply(&self, r: &[S], z: &mut [S]);
}
```

### Jacobi (Diagonal)

The simplest preconditioner: \\(M = \text{diag}(A)\\).

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
use numra::linalg::{SparseMatrix, Jacobi};

let triplets = vec![
    (0, 0, 4.0), (0, 1, 1.0),
    (1, 0, 1.0), (1, 1, 4.0),
];
let a = SparseMatrix::from_triplets(2, 2, &triplets).unwrap();
let jac = Jacobi::new(&a);

// Applying: z = D^{-1} * r
let r = [8.0, 12.0];
let mut z = [0.0; 2];
jac.apply(&r, &mut z);
// z = [8/4, 12/4] = [2.0, 3.0]
```

| Property | Value |
|----------|-------|
| Setup cost | \\(O(n)\\) |
| Apply cost | \\(O(n)\\) |
| Effectiveness | Good for diagonally dominant systems |
| Storage | \\(O(n)\\) |

### ILU(0) (Incomplete LU, Zero Fill-In)

An approximate LU factorization that drops fill-in outside the original
sparsity pattern. Much more effective than Jacobi at the cost of higher setup:

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
use numra::linalg::{SparseMatrix, Ilu0, pcg, IterativeOptions};

let n = 50;
let mut triplets = Vec::new();
for i in 0..n {
    triplets.push((i, i, 2.0));
    if i > 0     { triplets.push((i, i - 1, -1.0)); }
    if i < n - 1 { triplets.push((i, i + 1, -1.0)); }
}
let a = SparseMatrix::from_triplets(n, n, &triplets).unwrap();
let b: Vec<f64> = (0..n).map(|i| (i + 1) as f64).collect();

let ilu = Ilu0::new(&a).unwrap();
let opts = IterativeOptions::default().tol(1e-10);

// ILU(0) on a tridiagonal matrix is exact LU -- converges in 1-2 iterations!
let result = pcg(&a, &b, &ilu, None, &opts).unwrap();
assert!(result.converged);
assert!(result.iterations <= 2);
```

| Property | Value |
|----------|-------|
| Setup cost | \\(O(\text{nnz})\\) |
| Apply cost | \\(O(\text{nnz})\\) (forward + back substitution) |
| Effectiveness | Excellent for banded and well-structured matrices |
| Storage | \\(O(n^2)\\) in current implementation |

> **Note:** For tridiagonal matrices, ILU(0) equals exact LU, so CG converges
> in 1-2 iterations. For general sparse matrices, the quality depends on how
> much fill-in is dropped.

### SSOR (Symmetric Successive Over-Relaxation)

A symmetric preconditioner parameterized by relaxation factor \\(\omega \in (0, 2)\\):

\\[
M = \left(\frac{D}{\omega} + L\right) \left(\frac{D}{\omega}\right)^{-1} \left(\frac{D}{\omega} + U\right)
\\]

where \\(D\\) is the diagonal, \\(L\\) the strict lower triangle, and \\(U\\)
the strict upper triangle. Setting \\(\omega = 1\\) gives symmetric
Gauss-Seidel.

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
use numra::linalg::{SparseMatrix, Ssor, pcg, IterativeOptions};

let n = 20;
let mut triplets = Vec::new();
for i in 0..n {
    triplets.push((i, i, 2.0));
    if i > 0     { triplets.push((i, i - 1, -1.0)); }
    if i < n - 1 { triplets.push((i, i + 1, -1.0)); }
}
let a = SparseMatrix::from_triplets(n, n, &triplets).unwrap();
let b: Vec<f64> = (0..n).map(|i| (i + 1) as f64).collect();

// omega = 1.0 gives symmetric Gauss-Seidel
let ssor = Ssor::new(&a, 1.0);
let opts = IterativeOptions::default().tol(1e-8);

let result = pcg(&a, &b, &ssor, None, &opts).unwrap();
assert!(result.converged);
```

| Property | Value |
|----------|-------|
| Setup cost | \\(O(\text{nnz})\\) |
| Apply cost | \\(O(\text{nnz})\\) (forward + backward sweep) |
| Effectiveness | Good general-purpose; tune \\(\omega\\) for best results |
| Storage | \\(O(\text{nnz})\\) |

## Preconditioner Comparison

| Preconditioner | Setup | Apply | Quality | Best for |
|----------------|-------|-------|---------|----------|
| Jacobi | \\(O(n)\\) | \\(O(n)\\) | Low | Diagonally dominant systems |
| SSOR(\\(\omega\\)) | \\(O(\text{nnz})\\) | \\(O(\text{nnz})\\) | Medium | General SPD, tunable |
| ILU(0) | \\(O(\text{nnz})\\) | \\(O(\text{nnz})\\) | High | Well-structured sparsity |

## Solver Comparison

| Solver | Matrix type | Memory/iter | Convergence | Restart needed |
|--------|-------------|-------------|-------------|----------------|
| CG | SPD | \\(O(n)\\) | Monotone \\(\lVert \cdot \rVert_A\\) | No |
| PCG | SPD | \\(O(n)\\) | Accelerated CG | No |
| GMRES | Any | \\(O(kn)\\) | Monotone \\(\lVert r \rVert\\) | Yes |
| BiCGSTAB | Any | \\(O(n)\\) | Irregular | No |
| MINRES | Symmetric | \\(O(n)\\) | Monotone \\(\lVert r \rVert\\) | No |

### Decision Guide

```text
Is A symmetric?
  |
  +-- Yes: Is A positive definite?
  |         |
  |         +-- Yes --> CG (or PCG with preconditioner)
  |         |
  |         +-- No  --> MINRES
  |
  +-- No:  Is memory limited?
            |
            +-- Yes --> BiCGSTAB
            |
            +-- No  --> GMRES(restart=30)
```

## Comparing Iterative vs. Direct Solvers

For the same system, verify that iterative and direct methods agree:

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
use numra::linalg::{SparseMatrix, SparseLU, cg, IterativeOptions};

let n = 100;
let mut triplets = Vec::new();
for i in 0..n {
    triplets.push((i, i, 2.0));
    if i > 0     { triplets.push((i, i - 1, -1.0)); }
    if i < n - 1 { triplets.push((i, i + 1, -1.0)); }
}
let a = SparseMatrix::from_triplets(n, n, &triplets).unwrap();
let b: Vec<f64> = (0..n).map(|i| (i as f64 * 0.1).sin()).collect();

// Direct solve
let lu = SparseLU::new(&a).unwrap();
let x_direct = lu.solve(&b).unwrap();

// Iterative solve
let opts = IterativeOptions::default().tol(1e-12);
let result = cg(&a, &b, None, &opts).unwrap();

// Solutions should agree to high precision
for i in 0..n {
    assert!((result.x[i] - x_direct[i]).abs() < 1e-8);
}
```

## f32 Support

All iterative solvers work with `f32`:

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust,ignore
use numra::linalg::{SparseMatrix, cg, IterativeOptions};

let n = 10;
let mut triplets: Vec<(usize, usize, f32)> = Vec::new();
for i in 0..n {
    triplets.push((i, i, 2.0f32));
    if i > 0     { triplets.push((i, i - 1, -1.0f32)); }
    if i < n - 1 { triplets.push((i, i + 1, -1.0f32)); }
}
let a = SparseMatrix::from_triplets(n, n, &triplets).unwrap();
let b: Vec<f32> = (0..n).map(|i| (i + 1) as f32).collect();

let opts = IterativeOptions::default().tol(1e-5f32);
let result = cg(&a, &b, None, &opts).unwrap();
assert!(result.converged);
```

## Practical Tips

1. **Always precondition.** Unpreconditioned Krylov methods are rarely
   competitive. Even Jacobi (the cheapest preconditioner) helps.

2. **Start with ILU(0) for CG/PCG.** It provides the best cost-to-quality
   ratio for structured sparse matrices. On tridiagonal systems it is exact.

3. **Monitor convergence.** Check `result.iterations` and `result.converged`.
   If the solver hits `max_iter` without converging, try a stronger
   preconditioner or switch solvers.

4. **GMRES restart tuning.** Start with `restart=30`. If convergence stagnates,
   increase to 50 or 100. Setting `restart=n` gives full GMRES (guaranteed
   convergence in \\(n\\) iterations, but \\(O(n^2)\\) memory).

5. **Use CG when you can.** CG exploits symmetry and positive definiteness
   for optimal convergence. Never use GMRES on an SPD system -- CG will always
   be faster.

6. **Reuse preconditioners.** If solving multiple systems with the same matrix
   (e.g., time-stepping), construct the preconditioner once and reuse it.

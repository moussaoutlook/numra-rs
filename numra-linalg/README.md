# numra-linalg

**Linear algebra for the [Numra](https://numra-rs.org/) workspace — dense and sparse matrices, factorizations (LU / QR / Cholesky / SVD / eigen), and preconditioned Krylov solvers.**

[![Crates.io](https://img.shields.io/crates/v/numra-linalg.svg)](https://crates.io/crates/numra-linalg)
[![docs.rs](https://docs.rs/numra-linalg/badge.svg)](https://docs.rs/numra-linalg)

Backend-agnostic `Matrix` trait built on [faer](https://github.com/sarah-ek/faer-rs). Provides direct factorizations (dense and sparse), iterative Krylov solvers, and a small library of preconditioners.

## Example

```rust
use numra_linalg::{DenseMatrix, Matrix};

let mut a: DenseMatrix<f64> = DenseMatrix::zeros(3, 3);
a.set(0, 0, 2.0);
a.set(1, 1, 3.0);
a.set(2, 2, 4.0);

let b = vec![1.0, 2.0, 3.0];
let x = a.solve(&b).unwrap();

assert!((x[0] - 0.5).abs() < 1e-10);
assert!((x[1] - 2.0 / 3.0).abs() < 1e-10);
assert!((x[2] - 0.75).abs() < 1e-10);
```

## What's in this crate

| Area | Items |
|---|---|
| Dense matrix              | `DenseMatrix`, `Matrix` trait |
| Sparse matrix             | `SparseMatrix` |
| LU                        | `LUFactorization`, `LUSolver`, `SparseLU` |
| QR                        | `QRFactorization` |
| Cholesky                  | `CholeskyFactorization`, `SparseCholesky` |
| SVD                       | `SvdDecomposition`, `ThinSvdDecomposition` |
| Eigen                     | `EigenDecomposition`, `SymEigenDecomposition` |
| Iterative Krylov solvers  | `cg`, `pcg`, `gmres`, `bicgstab`, `minres` |
| Preconditioners           | `IdentityPreconditioner`, `Jacobi`, `Ssor`, `Ilu0` |

## Composes with

- [`numra-nonlinear`](https://docs.rs/numra-nonlinear) for Jacobian solves at each Newton step
- [`numra-ode`](https://docs.rs/numra-ode) for factorizations inside implicit Runge-Kutta and BDF stages
- [`numra-pde`](https://docs.rs/numra-pde) for sparse Laplacian / Helmholtz operator assembly
- [`numra-optim`](https://docs.rs/numra-optim) for Hessian solves and QP / SQP subproblems
- [`numra-fit`](https://docs.rs/numra-fit) for SVD-based polynomial regression

## Install

```toml
[dependencies]
numra-linalg = "0.1"
```

Or via the umbrella crate:

```toml
[dependencies]
numra = "0.1"
```

## Documentation

- **API**: <https://docs.rs/numra-linalg>
- **Book**: [Matrices and vectors](https://book.numra-rs.org/ch06-linear-algebra/matrices-and-vectors/) · [Factorizations](https://book.numra-rs.org/ch06-linear-algebra/factorizations/) · [Iterative solvers](https://book.numra-rs.org/ch06-linear-algebra/iterative-solvers/)
- **Source**: <https://github.com/moussaoutlook/numra-rs/tree/main/numra-linalg>

## License

Numra Academic & Research License (Non-Commercial). Academic and research use is free; commercial use requires a separate license — contact `contact@spectralautomata.com`. See [LICENSE](https://github.com/moussaoutlook/numra-rs/blob/main/LICENSE).

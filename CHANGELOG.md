# Changelog

All notable public changes to Numra are recorded here. The project follows semantic-versioning intent, with extra care around solver behavior, public re-exports, and documented book examples while the `0.1.x` API is still settling.

## 0.1.0 - Unreleased

### Added

- Workspace facade crate covering ODE, SDE, DDE, FDE, IDE, PDE, SPDE, optimization, optimal control, linear algebra, quadrature, interpolation, special functions, FFT, statistics, fitting, signal processing, and autodiff.
- Public release audit artifacts under `docs/audit/`, including API inventory, book coverage matrix, correctness test map, and release checklist.
- CI and local audit gates for formatting, clippy, tests, docs, mdBook, book snippets, inventory drift, MSRV, and supply-chain checks.

### Policy

- Root `Cargo.lock` is committed for reproducible public CI.
- Generated mdBook HTML under `numra-book/book/` is not tracked; book sources are the release artifact source of truth.
- Public book snippets must either compile or carry an explicit `book-ignore` reason.

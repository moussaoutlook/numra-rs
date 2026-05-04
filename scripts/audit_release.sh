#!/usr/bin/env bash
# Local full audit (mirrors CI). Run from repository root.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
MSRV="1.83"

echo "==> rustfmt"
cargo fmt --all -- --check

echo "==> clippy"
cargo clippy --workspace --all-targets -- -D warnings

echo "==> tests"
cargo test --workspace

echo "==> doctests"
cargo test --workspace --doc

echo "==> examples (facade crate)"
cargo build -p numra --examples

echo "==> rustdoc"
RUSTDOCFLAGS='-D warnings' cargo doc --workspace --no-deps

echo "==> cargo deny"
cargo deny check

if command -v rustup >/dev/null 2>&1; then
  if ! rustup run "${MSRV}" cargo --version >/dev/null 2>&1; then
    echo "==> install MSRV toolchain ${MSRV}"
    rustup toolchain install "${MSRV}" --profile minimal
  fi
  echo "==> MSRV check (${MSRV})"
  cargo "+${MSRV}" check --workspace
else
  echo "ERROR: rustup is required for MSRV check (${MSRV})." >&2
  exit 1
fi

echo "Audit script finished OK."

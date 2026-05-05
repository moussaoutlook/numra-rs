# Contributing to Numra

Thanks for your interest in contributing to Numra.

## Contributor License Agreement (CLA)

To keep Numra commercially licensable while remaining freely available for academic and
research use, all contributions require agreeing to the **Contributor License Agreement**:

- [`CLA.md`](CLA.md)

By submitting a pull request (or otherwise contributing code, docs, tests, or examples), you
agree to the CLA.

## Development workflow

- **Build**: `cargo build`
- **Test**: `cargo test --workspace`
- **Format**: `cargo fmt --all`
- **Lint**: `cargo clippy --workspace --all-targets -- -D warnings`

### Local pre-push gate

Activate the repo-tracked git hook once per clone so `cargo fmt --check` and `cargo clippy` run automatically before every `git push`:

```bash
git config core.hooksPath .githooks
```

The hook lives at `.githooks/pre-push`. Bypass it for genuine emergencies with `git push --no-verify`.

## Release-quality checks

The repo includes an audit script used for public-release quality:

```bash
bash scripts/audit_release.sh
```

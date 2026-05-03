# Security policy

## Supported versions

Numra is pre-1.0. Only the latest published release on the `main` branch is
supported with security fixes. Older `0.x` releases will not receive
backported patches.

| Version       | Supported          |
| ------------- | ------------------ |
| latest `0.x`  | :white_check_mark: |
| older `0.x`   | :x:                |

## Reporting a vulnerability

Please **do not** open a public GitHub issue for security-sensitive reports.

Instead, email **contact@spectralautomata.com** with:

- A description of the issue and its impact.
- A minimal reproduction, if possible.
- The Numra version (or commit SHA) and Rust toolchain you tested against.

You can expect:

- An acknowledgement within 5 business days.
- A coordinated-disclosure timeline if the report is confirmed.
- Public credit in the release notes once a fix is published, unless you
  prefer to remain anonymous.

## Scope

Numra is a numerical-methods library; the most realistic concerns are:

- Memory-safety issues in `unsafe` blocks (e.g. in linear algebra hot paths).
- Panics or unsoundness reachable from untrusted input to public APIs.
- Build- or supply-chain issues affecting the workspace's published crates.

Issues in dependencies should be reported upstream; we'll happily coordinate.

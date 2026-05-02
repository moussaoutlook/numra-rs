# Book snippet policy

`numra-book/src` is the source of truth for user-facing examples. Fenced code blocks must make their compile status explicit so readers can tell which snippets are copyable examples and which are illustrative fragments.

## Allowed fences

- Use `rust` for standalone examples that should compile against the workspace `numra` facade crate.
- Use `rust,ignore` only for examples that are intentionally not standalone, such as abbreviated setup, pseudocode-style solver sketches, long-running examples, or APIs explained across multiple fragments.
- Use `text`, `toml`, `bash`, or another non-Rust fence for output, configuration, commands, or prose-like notation.

## Required ignore reason

Every `rust,ignore` fence must be immediately preceded by an HTML comment with this prefix:

```text
<!-- book-ignore: reason here. -->
```

The reason should be short and specific. Acceptable examples:

- `<!-- book-ignore: illustrative excerpt; setup is shown in the surrounding section. -->`
- `<!-- book-ignore: pseudocode for solver selection policy, not a standalone Rust program. -->`
- `<!-- book-ignore: long-running example; covered by numra examples instead. -->`

## Enforcement

CI and `scripts/audit_release.sh` run both checks:

- `mdbook test` executes Rust doctest-style snippets that mdBook can test.
- `scripts/extract_book_snippets.py` rejects undocumented `rust,ignore` fences and compiles all plain `rust` fences in temporary crates under `target/book-snippets/`.

Prefer the public `numra::...` facade in book examples unless the section is specifically documenting a subcrate.

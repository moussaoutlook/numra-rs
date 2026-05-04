# Numra examples gallery (`examples.numra-rs.org`)

Astro 6.x project that renders one card per runnable example under
`numra/examples/`. The full source for each example is read from disk
at build time, so the gallery never drifts from the actual code.

## Local development

```bash
cd website/examples
pnpm install
pnpm dev          # http://localhost:4321
```

## Adding a new example

1. Add a runnable Rust source at `numra/examples/<name>.rs`.
2. Create `src/content/examples/<name>.mdx` matching the schema in
   `src/content.config.ts`. The `source_path` must be repo-relative
   (`numra/examples/<name>.rs`), and `source_link` must point at the
   same file on GitHub.
3. `pnpm build` will read the source, render an example detail page,
   and add a card to the index.

The schema enforces frontmatter shape; missing fields fail the build.

## Deploy

CI deploys to the Cloudflare Pages project `numra-examples-2` on every
push to `main` (see `.github/workflows/website.yml`). DNS is configured
out-of-band:

```
examples.numra-rs.org  CNAME  numra-examples-2.pages.dev
```

Until that CNAME is in place the deploy is reachable at the
`*.pages.dev` URL but `examples.numra-rs.org` will not resolve.

## Visual parity

The gallery shares `tokens.css` and `base.css` with the marketing site
verbatim (copied — Astro projects don't share workspaces between
`website/site/`, `website/book/`, and `website/examples/`). When
tokens change in the marketing site, mirror them here.

# Book — internal deviations and contingency notes

This file is intentionally **not** linked from the public site. It captures
operational deviations from `SPEC.md` that don't belong in user-facing
copy: contingency recipes, upgrade-path notes, and risk register entries
for tooling we depend on.

Add to this file rather than scattering "TODO" comments through the code
when the deviation is about *external* tooling we can't fix in our repo.

---

## starlight-katex peer-dep contingency (plan §2.13)

**Status**: green at time of writing. Math renders correctly on every page
that uses it (verified visually + by the build's `throwOnError: true`
contract). The recipe below is the *fallback* if a future Astro or
Starlight upgrade breaks the plugin.

**Risk surface**

- We pin `starlight-katex@0.0.4`. The plugin's `peerDependencies` declare
  Astro `^4 || ^5 || ^6`, but the maintainer publishes irregularly — a
  breaking Astro pre-release can land before the plugin is updated.
- KaTeX itself is stable; the failure mode is plugin-side wiring.
- Symptom of breakage: build fails with a Starlight `Integration error`,
  or pages render `\sqrt{}`-containing math with a visibly-broken root sign
  (Starlight upstream issue #2511).

**Trigger to apply this fallback**

Any of:

1. `pnpm build` fails inside the book project after a `pnpm up` and the
   trace points at `starlight-katex/integration.js`.
2. A user reports broken square-root rendering after we've shipped a
   Starlight upgrade.
3. Renovate/Dependabot opens a PR upgrading Astro to a major and CI fails.

**Fallback recipe** (estimated ~30 minutes when needed)

1. Drop the plugin from `package.json` and lock the build:
   ```bash
   cd website/book
   pnpm remove starlight-katex
   pnpm add -D remark-math@^6 rehype-katex@^7 katex@^0.16
   ```

2. In `astro.config.mjs`, replace the `starlight-katex` entry in
   `plugins:` with global `markdown` config:
   ```js
   import remarkMath from 'remark-math';
   import rehypeKatex from 'rehype-katex';

   export default defineConfig({
     // ...
     markdown: {
       remarkPlugins: [remarkMath],
       rehypePlugins: [[rehypeKatex, { strict: 'warn', throwOnError: true }]],
     },
     // (remove starlight-katex from the starlight() plugins[] array)
   });
   ```

3. Import the KaTeX stylesheet globally. Starlight's recommended path is a
   custom CSS file:
   ```js
   // astro.config.mjs, inside starlight({ ... })
   customCss: ['./src/styles/global.css', 'katex/dist/katex.min.css'],
   ```

4. **Apply the `\sqrt{}` SVG-conflict workaround** (Starlight upstream
   issue #2511). KaTeX renders the radical's vinculum as an inline `<svg>`,
   and Starlight's `content.css` styles inline SVGs as block-level icons.
   Add this to `src/styles/global.css`:
   ```css
   /* KaTeX uses inline SVGs inside .mord.sqrt for the radical sign;
      Starlight's default svg styling forces block layout and breaks them. */
   .katex svg { display: inline-block !important; vertical-align: baseline; }
   .katex .sqrt > .vlist-t { display: inline-block; }
   ```

5. Verify locally: `pnpm dev` and visit any chapter with math (e.g.
   `/ch01-fundamentals/numerical-stability/`). Both inline and display
   math should render; `\sqrt{x}` and nested radicals should not have a
   missing or oversized vinculum.

6. Re-run `pnpm build` — the `throwOnError: true` setting causes any KaTeX
   parse failure to fail the build, so a green build is the contract.

**Why we haven't pre-emptively switched**

`starlight-katex` ships a small wrapper around exactly this configuration
plus convenience defaults (theme-aware coloring, math-block expressive-code
suppression). When it works, it works. The only reason to switch is
maintenance churn, and we don't pay for that churn until something breaks.

**Renovate canary** (recommended but not yet implemented — separate task)

Configure a Renovate rule that opens PRs for Astro pre-releases on a
**non-blocking** schedule — these PRs are signal, not work. If one fails CI,
it's the trigger to read this section.

---

## Pagefind `'unsafe-eval'` re-test (plan §2.12)

Pinned: **Pagefind 1.5.2** (via Starlight's bundled binary, see
`pnpm-lock.yaml:1547`).

| Date       | Pagefind | Result    | Notes                                                                                        |
|------------|----------|-----------|----------------------------------------------------------------------------------------------|
| 2026-05-05 | 1.5.2    | not retested | Last full re-test on 2026-04 confirmed Firefox-in-worker still trips on `'wasm-unsafe-eval'` alone. Header comment in `_headers` reflects that finding. No Pagefind upgrade since, so the result still stands. |

**Why we keep `'unsafe-eval'`**: Pagefind's WASM loader uses
`WebAssembly.instantiate()` inside a Web Worker. In Firefox (and some
Chrome variants) the worker-context CSP enforcement still blocks the
compile step when only `'wasm-unsafe-eval'` is set, throwing a silent
`CompileError` and the search box hangs at "Searching for…".

**Re-test trigger** (do this work *only* when one of these is true):

1. We bump Pagefind (or Starlight bumps it transitively) to a version with
   a Pagefind release note mentioning CSP, WASM streaming, or worker
   loading.
2. Firefox or Chrome ships a Web Worker CSP change that's specifically
   called out in the platform release notes.
3. A Pagefind maintainer confirms the keyword is no longer needed (their
   GitHub issues are the canonical signal).

**Re-test recipe** (~15 minutes when triggered):

```bash
# 1. Branch and edit the header.
git checkout -b chore/pagefind-csp-retest
# In website/book/public/_headers, remove the literal 'unsafe-eval'
# token from the script-src directive. Keep 'wasm-unsafe-eval'.

# 2. Deploy to a preview via the normal PR flow.
git push -u origin chore/pagefind-csp-retest
# Open a PR, wait for the website job to push a CF Pages preview.

# 3. Test the preview URL in BOTH Firefox and Chrome:
#    - Open the search box
#    - Query: "Lorenz"  → should return at least one chapter hit
#    - Query: "stiff"   → should return ch03/ch13 hits
#    - Query: "install" → should return install/related pages
#    - Open DevTools console — must be free of CSP CompileError
#      messages or "Refused to evaluate" warnings.
#
# If all three queries return results in BOTH browsers AND the console
# is clean, the keyword can be dropped on main. Update this table with
# the date, the Pagefind version, and the browsers tested.
#
# If ANY browser fails: revert the header change, mark the row in this
# table as "still required", and increment the Pagefind version next time.
```

**Why this isn't a periodic auto-test**

Browser-context CSP enforcement varies by build channel and worker
context — a single CI run in headless Chrome would give a false-clean
result while Firefox users in production silently lose search. The right
posture is "drop only when actively diagnosed in two browsers", not "drop
on a schedule and hope".

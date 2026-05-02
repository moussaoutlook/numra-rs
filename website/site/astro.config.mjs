// @ts-check
import { defineConfig } from 'astro/config';
import sitemap from '@astrojs/sitemap';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';

/**
 * Numra marketing site — astro.config.mjs
 *
 * Static-first Astro 6 project. No SSR, no client framework, vanilla CSS
 * with design tokens (see src/styles/tokens.css).
 *
 * Math rendering: KaTeX server-rendered at build time via remark-math
 * + rehype-katex. The KaTeX stylesheet is imported globally in the base
 * layout; do not load it from a CDN.
 *
 * CSP: handled exclusively by Cloudflare Pages headers
 * (public/_headers). The experimental `security.csp` meta-tag
 * emitter is intentionally not used.
 *
 * @see https://docs.astro.build/en/reference/configuration-reference/
 */
export default defineConfig({
  site: 'https://numra-rs.org',
  trailingSlash: 'never',

  build: {
    // Inline small CSS files; let larger ones be linked. Keeps LCP low.
    inlineStylesheets: 'auto',
    // Use directory format for clean URLs without trailing slash.
    format: 'file',
  },

  markdown: {
    remarkPlugins: [remarkMath],
    rehypePlugins: [[rehypeKatex, {
      // Allow LaTeX commands KaTeX flags as "strict" warnings.
      // Numra's content uses standard LaTeX so this rarely matters.
      strict: 'warn',
      // Throw on parse errors so bad math fails the build, not the user.
      throwOnError: true,
    }]],
    shikiConfig: {
      // Built-in dual-theme support — emits CSS variables that switch
      // between themes based on data-theme attribute. Pair with
      // tokens.css.
      themes: {
        light: 'github-light',
        dark: 'github-dark-dimmed',
      },
      wrap: false,
    },
  },

  integrations: [
    sitemap({
      // Exclude non-canonical paths from the sitemap.
      filter: (page) => !page.includes('/404') && !page.includes('/_'),
    }),
  ],

  prefetch: {
    // Don't prefetch every link — that defeats the bandwidth budget.
    // Hover-prefetch is the right balance for a content site.
    prefetchAll: false,
    defaultStrategy: 'hover',
  },

  vite: {
    // Pin source map behavior. Disable in production for smaller output.
    build: { sourcemap: false },
    // Astro 6 + Vite 7: use modern target.
    optimizeDeps: { exclude: [] },
  },
});

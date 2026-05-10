// @ts-check
import { defineConfig } from 'astro/config';
import sitemap from '@astrojs/sitemap';
import mdx from '@astrojs/mdx';

/**
 * Numra examples gallery — astro.config.mjs (examples.numra-rs.org)
 *
 * Pure Astro 6.x (no Starlight). The marketing site's tokens.css is
 * imported directly to keep visual parity with numra-rs.org. Each
 * example is an MDX file under src/content/examples/ with frontmatter
 * declaring its filterable properties (equation class, stiffness,
 * events, dense output, complexity).
 *
 * Code samples are imported from numra/examples/*.rs at build time so
 * the gallery never drifts from the actual runnable code.
 *
 * @see https://docs.astro.build/en/guides/content-collections/
 */
export default defineConfig({
  site: 'https://examples.numra-rs.org',
  // Marketing site uses 'never' (`format: 'file'`); the book uses
  // 'always' (directory output). Pick one and stick to it. Examples
  // is a small flat structure, so file format is fine.
  trailingSlash: 'never',
  build: {
    format: 'file',
  },
  // Light-only site, so use a single light Shiki theme rather than the
  // dual-theme setup from the marketing site. Without this, Astro
  // defaults to `github-dark`, which paints code blocks with a dark
  // background that fights the cream palette.
  markdown: {
    shikiConfig: {
      theme: 'github-light',
      wrap: false,
    },
  },
  integrations: [
    mdx(),
    sitemap(),
  ],
});

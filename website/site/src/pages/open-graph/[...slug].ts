// Per-page Open Graph image generation for the marketing site.
//
// `astro-og-canvas` builds a 1200×630 PNG per route at build time and
// serves it at `/open-graph/<slug>.png`. The route map below lists every
// canonical page with the same title/description used in BaseLayout,
// keeping the OG card in sync with what's actually rendered on each page.
//
// New pages: add an entry below. The sitemap stays the source of truth
// for search; this map is the source of truth for social previews.

import { OGImageRoute } from 'astro-og-canvas';

interface PageMeta {
  title: string;
  description: string;
}

const pages: Record<string, PageMeta> = {
  index: {
    title: 'Numra',
    description: 'Composable numerical methods for Rust. ODE, SDE, DDE, FDE, IDE, PDE, and SPDE solvers in one library.',
  },
  features: {
    title: 'Features',
    description: 'What Numra covers — equation classes, solver families, and shared abstractions.',
  },
  install: {
    title: 'Install Numra',
    description: 'Install Numra in a Rust project and run your first ODE solve in under a minute.',
  },
  api: {
    title: 'API reference',
    description: 'How to read Numra’s Rust API documentation. Until the first crates.io publish, build the docs locally; afterwards, docs.rs hosts every published version.',
  },
  blog: {
    title: 'Numra blog',
    description: 'Release announcements, design notes, and longer-form posts about Numra.',
  },
  changelog: {
    title: 'Changelog',
    description: 'Release notes, rendered from CHANGELOG.md in the Numra repository at build time.',
  },
  cite: {
    title: 'Cite Numra',
    description: 'Citation information, BibTeX, and RIS for Numra. CITATION.cff drives Zenodo and GitHub’s Cite this repository button.',
  },
  commercial: {
    title: 'Commercial licensing',
    description: 'Production deployments, paid products, and hosted services using Numra require a separate license.',
  },
  community: {
    title: 'Community',
    description: 'How to contribute to Numra, where conversations happen, and what the contribution process looks like.',
  },
  license: {
    title: 'License',
    description: 'Numra is free for non-commercial academic and research use under a custom source-available license.',
  },
  privacy: {
    title: 'Privacy',
    description: 'What numra-rs.org collects and what it doesn’t. Spoiler: nothing personally identifying.',
  },
  roadmap: {
    title: 'Roadmap',
    description: 'Where Numra is headed: open issues, planned milestones, and stable commitments you can rely on today.',
  },
  stability: {
    title: 'Stability',
    description: 'MSRV, semver policy, deprecation windows, and feature-flag stability commitments.',
  },
};

// Brand palette — light-mode tokens from src/styles/tokens.css.
const ACCENT_TEAL: [number, number, number] = [10, 110, 107];   // #0A6E6B
const DEEP_TEAL: [number, number, number] = [6, 78, 76];        // darker for gradient end
const FG: [number, number, number] = [255, 255, 255];
const FG_MUTED: [number, number, number] = [200, 232, 230];

// Inter served by Google Fonts as TTF, which CanvasKit accepts. URLs
// resolved at the time of writing from
// https://fonts.googleapis.com/css2?family=Inter:wght@400;700 — Google
// Fonts revs these URLs when the font is updated; if a build fails with
// a 404, refresh by fetching that CSS endpoint and copying the new URLs.
const INTER_400 = 'https://fonts.gstatic.com/s/inter/v20/UcCO3FwrK3iLTeHuS_nVMrMxCp50SjIw2boKoduKmMEVuLyfMZg.ttf';
const INTER_700 = 'https://fonts.gstatic.com/s/inter/v20/UcCO3FwrK3iLTeHuS_nVMrMxCp50SjIw2boKoduKmMEVuFuYMZg.ttf';

export const { getStaticPaths, GET } = await OGImageRoute({
  param: 'slug',
  pages,
  getImageOptions: (_path, page: PageMeta) => ({
    title: page.title,
    description: page.description,
    bgGradient: [ACCENT_TEAL, DEEP_TEAL],
    border: { color: ACCENT_TEAL, width: 4, side: 'inline-start' },
    padding: 80,
    font: {
      title: { color: FG, families: ['Inter'], weight: 'Bold', size: 76, lineHeight: 1.1 },
      description: { color: FG_MUTED, families: ['Inter'], weight: 'Normal', size: 32, lineHeight: 1.4 },
    },
    fonts: [INTER_700, INTER_400],
    format: 'PNG',
  }),
});

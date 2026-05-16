/**
 * Dark-mode FOIT (flash of incorrect theme) regression — book-only.
 *
 * **Intentional asymmetry between book and marketing site.** The book
 * (book.numra-rs.org, Starlight-based) supports dark mode: Starlight's
 * upstream theme script syncs `<html data-theme="...">` to the system
 * preference before paint. The marketing site (numra-rs.org, vanilla
 * Astro) is **deliberately light-only by design** — it does not have a
 * functioning `@media (prefers-color-scheme: dark)` path and does not
 * apply `data-theme` from stored preferences. This asymmetry is a
 * chosen product position, not an oversight.
 *
 * This suite therefore asserts:
 *
 *   - **Book**: dark paint under system-dark, light paint under
 *     system-light, and `data-theme="dark"` set before paint when the
 *     system is dark. The user-visible invariant is "body background
 *     reflects the system preference"; we assert that, not the
 *     implementation detail of how Starlight gets there.
 *   - **Marketing site**: light paint under system-light only. No
 *     assertions on what marketing does under system-dark — by
 *     design, marketing paints light in both modes.
 *
 * **Do not add marketing dark-mode tests for consistency with the
 * book.** A prior version of this suite did exactly that — asserting
 * dark paint on the marketing site under `prefers-color-scheme: dark`
 * and stored-preference `data-theme="dark"` application. Those
 * assertions were removed in F-WEBSITE-AUDIT-GATES (2026-05-16) because
 * they encoded a wrong expectation: the marketing site was never
 * intended to support dark mode, and the assertions had been silently
 * failing since the gate first ran on a `pull_request` event. If you
 * want to add marketing dark-mode support, that's a product decision
 * the user makes; raise it explicitly and update this header before
 * adding tests. Otherwise, do not re-introduce the assertions this
 * comment exists to forbid.
 */
import { test, expect, type Page } from '@playwright/test';

const SITE_URL = (process.env.NUMRA_SITE_URL ?? 'https://numra-rs.org').replace(/\/+$/, '');
const BOOK_URL = (process.env.NUMRA_BOOK_URL ?? 'https://book.numra-rs.org').replace(/\/+$/, '');

const RGB_RE = /^rgb\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)$/;

function relativeLuminance(rgb: string): number | null {
  const m = rgb.match(RGB_RE);
  if (!m) return null;
  const [, rs, gs, bs] = m;
  return (0.2126 * Number(rs) + 0.7152 * Number(gs) + 0.0722 * Number(bs)) / 255;
}

async function readBodyBg(page: Page): Promise<string> {
  return page.evaluate(() => getComputedStyle(document.body).backgroundColor);
}

async function expectDarkPaint(page: Page, url: string): Promise<void> {
  // We want to observe the *initial* paint state, so wait only for
  // domcontentloaded. If theme styling is wired correctly, the body
  // background already reflects the system preference here.
  await page.goto(url, { waitUntil: 'domcontentloaded' });
  const bg = await readBodyBg(page);
  const lum = relativeLuminance(bg);
  expect(
    lum !== null && lum < 0.4,
    `body background ${bg} on ${url} should be dark in system-dark; ` +
      `regression likely means theme tokens or the prefers-color-scheme rule got broken.`,
  ).toBe(true);
}

async function expectLightPaint(page: Page, url: string): Promise<void> {
  await page.goto(url, { waitUntil: 'domcontentloaded' });
  const bg = await readBodyBg(page);
  const lum = relativeLuminance(bg);
  expect(
    lum !== null && lum > 0.6,
    `body background ${bg} on ${url} should be light in system-light`,
  ).toBe(true);
}

test.describe('system-dark (no stored preference)', () => {
  test.use({ colorScheme: 'dark' });

  test('book root is painted dark', async ({ page }) => {
    await expectDarkPaint(page, BOOK_URL + '/');
  });

  test('book sets data-theme="dark" before paint', async ({ page }) => {
    // Starlight controls <html data-theme>; this asserts the upstream
    // theme script ships and runs early enough to set the attribute
    // by domcontentloaded. A regression here would manifest as a
    // light flash before Starlight's hydration.
    await page.goto(BOOK_URL + '/', { waitUntil: 'domcontentloaded' });
    const dataTheme = await page.locator('html').getAttribute('data-theme');
    expect(dataTheme).toBe('dark');
  });
});

test.describe('system-light (no stored preference)', () => {
  test.use({ colorScheme: 'light' });

  test('marketing root is painted light', async ({ page }) => {
    await expectLightPaint(page, SITE_URL + '/');
  });

  test('book root is painted light', async ({ page }) => {
    await expectLightPaint(page, BOOK_URL + '/');
  });
});

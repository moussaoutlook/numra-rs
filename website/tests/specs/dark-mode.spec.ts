/**
 * Dark-mode FOIT (flash of incorrect theme) regression.
 *
 * Two different mechanisms ship dark mode on Numra:
 *
 *   - Marketing site: tokens.css carries `@media (prefers-color-scheme:
 *     dark)` blocks. theme-init.js *only* applies a stored choice
 *     (`localStorage["numra-theme"]`); when no preference is stored it
 *     intentionally leaves `<html>` alone and lets the media query do
 *     the work. So system-dark users get a dark paint with NO
 *     `data-theme` attribute — that's correct behavior.
 *
 *   - Book (Starlight): the upstream theme script syncs <html
 *     data-theme="..."> to the system preference *before* paint, so
 *     system-dark users get an explicit `data-theme="dark"`.
 *
 * The real user-visible invariant for both is "body background is dark
 * when the system is dark". We assert that — not the implementation
 * detail of how it gets there.
 *
 * Stored-preference round-trip is tested separately to lock in the
 * theme-init contract.
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

  test('marketing root is painted dark', async ({ page }) => {
    await expectDarkPaint(page, SITE_URL + '/');
  });

  test('marketing /install is painted dark', async ({ page }) => {
    await expectDarkPaint(page, SITE_URL + '/install');
  });

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

test.describe('stored preference', () => {
  test('marketing site honours stored "dark" before paint', async ({ page, context }) => {
    // Pretend the user previously chose dark via the toggle.
    await context.addInitScript(() => {
      window.localStorage.setItem('numra-theme', 'dark');
    });
    await page.goto(SITE_URL + '/', { waitUntil: 'domcontentloaded' });

    const dataTheme = await page.locator('html').getAttribute('data-theme');
    expect(
      dataTheme,
      'theme-init.js should set data-theme="dark" from localStorage at <head>',
    ).toBe('dark');

    const bg = await readBodyBg(page);
    const lum = relativeLuminance(bg);
    expect(lum !== null && lum < 0.4, `body background ${bg} should be dark`).toBe(true);
  });
});

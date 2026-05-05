/**
 * Playwright config for website regression tests.
 *
 * Targets are configured via env so the same suite can run against:
 *   - production (default; numra-rs.org + book.numra-rs.org)
 *   - per-PR preview URLs (set NUMRA_SITE_URL / NUMRA_BOOK_URL in CI)
 *
 * Most tests are theme/layout regressions. We don't snapshot full
 * screenshots because the marketing copy and the book TOC are
 * frequently churned — instead we assert on computed-style properties
 * (`data-theme`, background-color, text color) which are stable across
 * content edits.
 */
import { defineConfig, devices } from '@playwright/test';

export default defineConfig({
  testDir: './specs',
  timeout: 30_000,
  expect: { timeout: 5_000 },
  fullyParallel: true,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 1 : 0,
  workers: process.env.CI ? 2 : undefined,
  reporter: process.env.CI ? [['list'], ['html', { open: 'never' }]] : 'list',

  use: {
    actionTimeout: 5_000,
    navigationTimeout: 15_000,
    trace: 'on-first-retry',
    screenshot: 'only-on-failure',
    video: 'off',
  },

  projects: [
    {
      name: 'chromium-light',
      use: {
        ...devices['Desktop Chrome'],
        colorScheme: 'light',
      },
    },
    {
      name: 'chromium-dark',
      use: {
        ...devices['Desktop Chrome'],
        colorScheme: 'dark',
      },
    },
  ],
});

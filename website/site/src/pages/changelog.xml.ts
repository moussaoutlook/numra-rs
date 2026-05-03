import rss from '@astrojs/rss';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { createMarkdownProcessor } from '@astrojs/markdown-remark';
import type { APIContext } from 'astro';

/**
 * RSS feed for Numra release notes.
 *
 * Each `## <version> - <date>` heading in CHANGELOG.md becomes a feed item.
 * Item bodies contain the full HTML rendering of the section, so RSS readers
 * that prefer rich text get the same content as the rendered /changelog page.
 *
 * The CHANGELOG path is resolved against `process.cwd()` (the build CWD is
 * `website/site` both locally and in CI), matching the convention used by
 * `src/pages/changelog.astro`.
 */

interface ReleaseSection {
    /** "0.1.0", "0.1.1", "1.0.0", … */
    version: string;
    /** Whatever follows the version on the heading line, e.g. "Unreleased" or "2026-05-15". */
    rawDate: string;
    /** Markdown body for this release, without the heading itself. */
    body: string;
    /** Best-effort parsed publication date, or `undefined` for unreleased / unparseable. */
    pubDate: Date | undefined;
}

function splitReleases(markdown: string): ReleaseSection[] {
    // Drop everything before the first `## ` so the file's H1 + preamble aren't
    // mistaken for a release.
    const firstAt = markdown.search(/^## /m);
    const body = firstAt >= 0 ? markdown.slice(firstAt) : markdown;

    const lines = body.split(/\r?\n/);
    const releases: ReleaseSection[] = [];
    let current: { heading: string; lines: string[] } | null = null;

    for (const line of lines) {
        if (line.startsWith('## ')) {
            if (current) {
                releases.push(parseRelease(current));
            }
            current = { heading: line.slice(3).trim(), lines: [] };
        } else if (current) {
            current.lines.push(line);
        }
    }
    if (current) {
        releases.push(parseRelease(current));
    }
    return releases;
}

function parseRelease(raw: { heading: string; lines: string[] }): ReleaseSection {
    const m = raw.heading.match(/^([\w.\-+]+)\s*[-–—]\s*(.+)$/);
    let version = raw.heading;
    let rawDate = '';
    if (m) {
        version = m[1];
        rawDate = m[2].trim();
    }
    const pubDate = parsePubDate(rawDate);
    const body = raw.lines.join('\n').trim();
    return { version, rawDate, body, pubDate };
}

function parsePubDate(rawDate: string): Date | undefined {
    // Accept ISO-style "2026-05-15" or any string Date can parse.
    if (!rawDate || /unreleased/i.test(rawDate)) {
        return undefined;
    }
    const d = new Date(rawDate);
    return Number.isNaN(d.getTime()) ? undefined : d;
}

export async function GET(context: APIContext): Promise<Response> {
    const changelogPath = resolve(process.cwd(), '../../CHANGELOG.md');
    const markdown = readFileSync(changelogPath, 'utf-8');

    const processor = await createMarkdownProcessor();
    const releases = splitReleases(markdown);

    const items = await Promise.all(
        releases.map(async (rel) => {
            const { code: html } = await processor.render(rel.body);
            const status = rel.pubDate
                ? rel.pubDate.toISOString().slice(0, 10)
                : rel.rawDate || 'Unreleased';
            return {
                title: `Numra ${rel.version}`,
                link: `/changelog#${rel.version.toLowerCase().replace(/\./g, '')}---${status.toLowerCase().replace(/\s+/g, '-')}`,
                description: `Release notes for Numra ${rel.version} (${status}).`,
                content: html,
                ...(rel.pubDate ? { pubDate: rel.pubDate } : {}),
            };
        })
    );

    return rss({
        title: 'Numra changelog',
        description: 'Release notes for the Numra Rust numerical-methods workspace.',
        site: context.site!,
        items,
        customData: '<language>en-us</language>',
    });
}

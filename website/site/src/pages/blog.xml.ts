import rss from '@astrojs/rss';
import { getCollection } from 'astro:content';
import { createMarkdownProcessor } from '@astrojs/markdown-remark';
import type { APIContext } from 'astro';

/**
 * RSS feed for the Numra blog.
 *
 * Mirrors the structure of `changelog.xml.ts`: each blog post becomes an
 * item, with the full HTML body rendered at build time so readers that
 * prefer rich text get the same content as the site.
 */
export async function GET(context: APIContext): Promise<Response> {
  const posts = (await getCollection('blog', ({ data }) => !data.draft)).sort(
    (a, b) => b.data.pubDate.valueOf() - a.data.pubDate.valueOf(),
  );

  const processor = await createMarkdownProcessor();
  const items = await Promise.all(
    posts.map(async (post) => {
      const { code: html } = await processor.render(post.body ?? '');
      return {
        title: post.data.title,
        link: `/blog/${post.id}`,
        description: post.data.description,
        content: html,
        pubDate: post.data.pubDate,
        author: post.data.author,
        categories: post.data.tags,
      };
    }),
  );

  return rss({
    title: 'Numra blog',
    description:
      'Release announcements, design notes, and longer-form posts about Numra.',
    site: context.site!,
    items,
    customData: '<language>en-us</language>',
  });
}

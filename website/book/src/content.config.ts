// src/content.config.ts
//
// Starlight 0.38 requires the Content Layer API. The pre-0.33 implicit
// "docs" collection is gone; every project must declare it explicitly
// using docsLoader() and docsSchema() from Starlight. Do not remove
// without also migrating off Starlight.

import { defineCollection } from 'astro:content';
import { docsLoader } from '@astrojs/starlight/loaders';
import { docsSchema } from '@astrojs/starlight/schema';

export const collections = {
  docs: defineCollection({
    loader: docsLoader(),
    schema: docsSchema(),
  }),
};

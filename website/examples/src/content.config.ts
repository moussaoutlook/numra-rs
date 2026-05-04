// Astro 6 uses src/content.config.ts (replaces the legacy
// src/content/config.ts). Schema for the examples collection — every
// MDX under src/content/examples/ must validate against this.
//
// Filter facets are deliberately enumerated rather than free-text so
// the gallery filter UI can render checkboxes from the schema rather
// than from observed data.

import { defineCollection, z } from 'astro:content';
import { glob } from 'astro/loaders';

const EQUATION_CLASSES = [
  'ODE',
  'SDE',
  'DDE',
  'PDE',
  'SPDE',
  'IDE',
  'FDE',
  'DAE',
  'Hybrid',
] as const;

const COMPLEXITIES = ['beginner', 'intermediate', 'advanced'] as const;

const examples = defineCollection({
  loader: glob({ pattern: '**/*.{md,mdx}', base: './src/content/examples' }),
  schema: z.object({
    title: z.string(),
    summary: z.string(),
    equation_class: z.enum(EQUATION_CLASSES),
    stiff: z.boolean().default(false),
    has_events: z.boolean().default(false),
    dense_output: z.boolean().default(false),
    complexity: z.enum(COMPLEXITIES),
    /** Path to the runnable example, relative to repo root. */
    source_path: z.string(),
    /** Full URL of the source on GitHub (mirrors source_path). */
    source_link: z.string().url(),
    /** Deep link to the relevant book chapter. */
    book_link: z.string().url(),
    /**
     * Order for the gallery's default sort within a category. Lower
     * comes first. Beginner examples should generally have lower
     * `order` than advanced ones.
     */
    order: z.number().default(100),
  }),
});

export const collections = { examples };

export type EquationClass = (typeof EQUATION_CLASSES)[number];
export type Complexity = (typeof COMPLEXITIES)[number];

export const equationClasses = EQUATION_CLASSES;
export const complexities = COMPLEXITIES;

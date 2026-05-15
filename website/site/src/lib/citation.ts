// Single source of truth for release-tied citation metadata on the
// website. Reads CITATION.cff at the repository root at build time and
// exposes typed values that any Astro page or component can import.
//
// The intent is that future releases only need to update CITATION.cff
// (already part of the release sequence — Zenodo reads it directly) and
// every release-tied statement on numra-rs.org flows automatically on
// the next site build. No per-release website edits needed.
//
// Path resolution: the `?raw` Vite import resolves CITATION.cff relative
// to *this source file* at build time and bakes the contents into the
// bundle, so the result is independent of bundle output location or
// runtime cwd.

import cffRaw from '../../../../CITATION.cff?raw';
import { load as parseYaml } from 'js-yaml';

interface CffIdentifier {
  type: string;
  value: string;
  description: string;
}

interface Cff {
  version: string;
  'date-released': string;
  identifiers: CffIdentifier[];
}

const cff = parseYaml(cffRaw) as Cff;

const concept = cff.identifiers.find((i) => /concept/i.test(i.description));
const versioned = cff.identifiers.find((i) => /^Version /i.test(i.description));

if (!concept) throw new Error('CITATION.cff: missing concept-DOI identifier');
if (!versioned) throw new Error('CITATION.cff: missing version-DOI identifier');

export const version = cff.version;
export const dateReleased = cff['date-released'];
export const releaseYear = dateReleased.slice(0, 4);
export const conceptDoi = concept.value;
export const versionDoi = versioned.value;

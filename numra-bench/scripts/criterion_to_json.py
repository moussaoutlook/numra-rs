#!/usr/bin/env python3
"""Flatten Criterion's per-bench output into a single JSON file.

Criterion stores results as a tree:

    target/criterion/<group>/<bench>/[<param>/]new/{estimates.json,benchmark.json}

This script walks one or more groups, reads each `estimates.json`, and
emits a single flat JSON document at `numra-bench/results/<group>.json`
of the form:

    {
      "group": "tolerance_scaling",
      "samples": [
        {
          "bench": "dopri5",
          "param": "1e-3",
          "mean_ns": 12345.6,
          "stddev_ns": 78.9,
          "ci_low_ns": ...,
          "ci_high_ns": ...,
          "n_samples": 100
        },
        ...
      ]
    }

That format separates the slow measurement step from the fast
rendering step — matplotlib reads the JSON, never `target/criterion/`.

Usage:
    python3 numra-bench/scripts/criterion_to_json.py [<group>...]

If invoked with no arguments, every group under `target/criterion/`
is exported.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CRITERION_ROOT = REPO_ROOT / "target" / "criterion"
OUTPUT_ROOT = REPO_ROOT / "numra-bench" / "results"


def _read_estimates(estimates_path: Path) -> dict:
    """Pull the four fields we care about out of Criterion's JSON."""
    with estimates_path.open() as f:
        data = json.load(f)
    mean = data["mean"]
    stddev = data["std_dev"]
    return {
        "mean_ns": mean["point_estimate"],
        "stddev_ns": stddev["point_estimate"],
        "ci_low_ns": mean["confidence_interval"]["lower_bound"],
        "ci_high_ns": mean["confidence_interval"]["upper_bound"],
    }


def _read_n_samples(benchmark_path: Path) -> int:
    """Number of samples Criterion collected for this bench."""
    with benchmark_path.open() as f:
        data = json.load(f)
    return int(data.get("sample_count", 0) or 0)


def _walk_group(group_dir: Path) -> list[dict]:
    """Yield one sample dict per bench inside a group directory."""
    samples: list[dict] = []
    for estimates_path in sorted(group_dir.rglob("new/estimates.json")):
        # Path shape:  <group>/<bench>/[<param>/]new/estimates.json
        rel = estimates_path.relative_to(group_dir).parts
        # drop trailing  ['new', 'estimates.json']
        path_parts = rel[:-2]
        if not path_parts:
            continue
        bench = path_parts[0]
        param = "/".join(path_parts[1:]) if len(path_parts) > 1 else None

        sample = {"bench": bench}
        if param is not None:
            sample["param"] = param

        sample.update(_read_estimates(estimates_path))

        benchmark_path = estimates_path.with_name("benchmark.json")
        if benchmark_path.exists():
            sample["n_samples"] = _read_n_samples(benchmark_path)

        samples.append(sample)
    return samples


def export_group(group: str) -> Path:
    group_dir = CRITERION_ROOT / group
    if not group_dir.is_dir():
        raise SystemExit(f"criterion group not found: {group_dir}")

    samples = _walk_group(group_dir)
    if not samples:
        raise SystemExit(f"no samples found under {group_dir}")

    output = {"group": group, "samples": samples}
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_ROOT / f"{group}.json"
    with out_path.open("w") as f:
        json.dump(output, f, indent=2, sort_keys=True)
        f.write("\n")
    return out_path


def discover_groups() -> Iterable[str]:
    if not CRITERION_ROOT.is_dir():
        raise SystemExit(
            f"no criterion output at {CRITERION_ROOT} — run `cargo bench` first"
        )
    for child in sorted(CRITERION_ROOT.iterdir()):
        # Criterion drops a `report` directory at the root; skip it.
        if child.is_dir() and child.name != "report":
            yield child.name


def main(argv: list[str]) -> int:
    groups = argv or list(discover_groups())
    if not groups:
        print("no criterion groups to export", file=sys.stderr)
        return 1
    for group in groups:
        out_path = export_group(group)
        print(f"wrote {out_path.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

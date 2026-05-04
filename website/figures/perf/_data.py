"""Helpers shared by the perf-chapter rendering scripts.

The Criterion → JSON post-processor at
`numra-bench/scripts/criterion_to_json.py` writes one file per group
under `numra-bench/results/<group>.json`. This module loads that file
into a small DataFrame so the per-figure scripts only need to know
about the chart, not about the JSON shape.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
RESULTS_ROOT = REPO_ROOT / "numra-bench" / "results"


def load_group(group: str) -> pd.DataFrame:
    """Load `numra-bench/results/<group>.json` into a flat DataFrame.

    Columns: bench, param (if present), mean_ns, stddev_ns,
    ci_low_ns, ci_high_ns, n_samples.
    """
    path = RESULTS_ROOT / f"{group}.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"missing bench data: {path}\n"
            "Run `cargo bench -p numra-bench` then "
            "`python3 numra-bench/scripts/criterion_to_json.py`."
        )
    payload = json.loads(path.read_text())
    return pd.DataFrame(payload["samples"])

"""
Convert Criterion benchmark output into a flat DataFrame for plotting.

Criterion writes its results to `target/criterion/<group>/<bench>/<sample>/`.
The interesting per-bench JSON is `target/criterion/<group>/<bench>/new/estimates.json`,
which contains point-and-interval estimates of the mean, median, and slope.

We capture mean nanoseconds + the upper/lower CI bounds as our timing column.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def _read_estimate(path: Path) -> dict[str, float] | None:
    """Read the mean estimate from a Criterion estimates.json file."""
    try:
        raw = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    mean = raw.get("mean", {})
    if not mean:
        return None
    return {
        "mean_ns": mean.get("point_estimate", float("nan")),
        "ci_lower_ns": mean.get("confidence_interval", {}).get("lower_bound", float("nan")),
        "ci_upper_ns": mean.get("confidence_interval", {}).get("upper_bound", float("nan")),
    }


def load_criterion_group(criterion_root: Path, group: str) -> pd.DataFrame:
    """Return a DataFrame with one row per benchmark in `group`.

    Columns:
        bench:          benchmark id (whatever Criterion's `bench_with_input(id, ..)`
                        used as the parameter, e.g. "1e-6" for tolerance scaling)
        mean_ns         point-estimate mean wall time in nanoseconds
        ci_lower_ns     lower bound of the 95% CI on the mean
        ci_upper_ns     upper bound of the 95% CI on the mean
    """
    group_dir = Path(criterion_root) / group
    if not group_dir.is_dir():
        raise FileNotFoundError(f"Criterion group dir not found: {group_dir}")

    rows: list[dict[str, Any]] = []
    for bench_dir in sorted(group_dir.iterdir()):
        if not bench_dir.is_dir() or bench_dir.name == "report":
            continue
        est_path = bench_dir / "new" / "estimates.json"
        est = _read_estimate(est_path)
        if est is None:
            continue
        rows.append({"bench": bench_dir.name, **est})

    return pd.DataFrame(rows)


def parse_param(bench_id: str) -> str:
    """Criterion's bench_with_input id is sometimes URL-escaped (e.g. `1e-6` → `1e-6`,
    but values with slashes get encoded). For now this is identity; centralized so
    plot scripts have one place to add parsing rules later.
    """
    return bench_id

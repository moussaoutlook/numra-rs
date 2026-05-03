"""
Home-page correctness coverage heatmap.

Renders a categorical heatmap of which Numra equation classes have which
kinds of regression-test invariant. Source: `docs/audit/correctness-test-map.md`,
spot-checked cell-by-cell against the actual test code under each crate's
`tests/` directory before this script was written.

The matrix is hardcoded here rather than parsed from the markdown audit so
the rendering step has no dependency on document formatting. If the audit
document changes, update this script and re-render.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

THIS_DIR = Path(__file__).resolve().parent
FIGURES_DIR = THIS_DIR.parent
sys.path.insert(0, str(FIGURES_DIR))
from lib import style  # noqa: E402
from lib.captioning import Provenance, save_with_provenance  # noqa: E402

style.apply(square=True)

REPO_ROOT = FIGURES_DIR.parent.parent
OUT = REPO_ROOT / "website" / "site" / "public" / "figures" / "home_correctness_map.svg"

# Rows = equation class; columns = invariant category. 1 = covered, 0 = not.
# Each "1" corresponds to a test file under <crate>/tests/ verified to exist
# before commit; see docs/audit/correctness-test-map.md for the canonical map.
ROWS = [
    "ODE / DAE",
    "SDE",
    "Optimization (LP / QP / MILP)",
    "Linear algebra",
    "Quadrature",
    "Interpolation",
    "FFT",
]
COLS = [
    "Convergence\norder",
    "Analytic /\ngolden value",
    "Statistical\nbands",
    "Event\nlocalization",
]
COVERAGE = np.array(
    [
        # convergence, golden, statistical, events
        [1, 1, 0, 1],  # ODE / DAE: convergence_order_tests, dae_tests, events_regression
        [0, 0, 1, 0],  # SDE: gbm_ou_bands
        [0, 1, 0, 0],  # Optim: tiny_lp / tiny_qp / tiny_milp golden values
        [0, 1, 0, 0],  # Linalg: dense_solve_smoke
        [0, 1, 0, 0],  # Quadrature: quadrature_smoke
        [0, 1, 0, 0],  # Interp: linear_endpoints
        [0, 1, 0, 0],  # FFT: fft_roundtrip
    ],
    dtype=int,
)


def main() -> None:
    fig, ax = plt.subplots(figsize=(5.6, 4.0))

    # Color: covered cells use the brand accent, empty cells the neutral subtle.
    cmap = plt.matplotlib.colors.ListedColormap(["#F5F6F7", style.ACCENT])
    ax.imshow(COVERAGE, cmap=cmap, aspect="auto", vmin=0, vmax=1)

    # Tick labels.
    ax.set_xticks(np.arange(len(COLS)))
    ax.set_yticks(np.arange(len(ROWS)))
    ax.set_xticklabels(COLS, fontsize=8)
    ax.set_yticklabels(ROWS, fontsize=9)
    plt.setp(ax.get_xticklabels(), rotation=0, ha="center")

    # Mark covered cells with a checkmark glyph; empty cells stay blank.
    for i in range(len(ROWS)):
        for j in range(len(COLS)):
            if COVERAGE[i, j]:
                ax.text(j, i, "✓", ha="center", va="center",
                        color="white", fontsize=14, fontweight="bold",
                        family="DejaVu Sans")

    # Subtle grid between cells.
    ax.set_xticks(np.arange(-0.5, len(COLS), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(ROWS), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.tick_params(which="major", bottom=False, left=False)

    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_title("Regression-test coverage by invariant type", pad=12)

    prov = Provenance(
        problem="Numra regression-test inventory",
        solver="Multiple — see invariant per cell",
        tol="N/A",
        repro_script="docs/audit/correctness-test-map.md (audit) + this script",
        extra={
            "primary_tests": "7",
            "supporting_tests": "8",
            "audit_path": "docs/audit/correctness-test-map.md",
        },
    )

    save_with_provenance(fig, OUT, prov)
    print(f"wrote {OUT.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()

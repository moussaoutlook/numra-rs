"""
Open Graph default image — 1200×630 PNG.

Used by every page on numra-rs.org until per-page generation lands
(Phase 2, via astro-og-canvas). Today's default replaces the broken
file referenced by `BaseLayout.astro:39` so social shares (Twitter,
Slack, LinkedIn, etc.) get a real preview.

Generated as PNG (not SVG) because Open Graph spec mandates raster.
Output dimensions: 1200×630 at dpi=200, figsize=(6.0, 3.15).
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

style.apply()

REPO_ROOT = FIGURES_DIR.parent.parent
OUT = REPO_ROOT / "website" / "site" / "public" / "og" / "default.png"


def main() -> None:
    fig = plt.figure(figsize=(6.0, 3.15), dpi=200)

    # Background: subtle vertical gradient using accent → soft accent.
    ax_bg = fig.add_axes([0, 0, 1, 1])
    ax_bg.set_axis_off()
    grad = np.linspace(0.0, 1.0, 256).reshape(-1, 1)
    cmap = plt.matplotlib.colors.LinearSegmentedColormap.from_list(
        "numra-og-bg",
        ["#FFFFFF", "#E6F0EF"],  # color-bg → color-accent-soft
    )
    ax_bg.imshow(grad, aspect="auto", cmap=cmap, extent=[0, 1, 0, 1], alpha=1.0)

    # Accent rule top-left — a tiny brand mark.
    fig.add_artist(
        plt.matplotlib.patches.Rectangle(
            (0.05, 0.83),
            0.04,
            0.025,
            facecolor=style.ACCENT,
            edgecolor="none",
            transform=fig.transFigure,
        )
    )

    # Wordmark.
    fig.text(
        0.05,
        0.62,
        "Numra",
        family="serif",
        fontsize=72,
        fontweight="bold",
        color="#14171A",  # color-fg
    )

    # Tagline.
    fig.text(
        0.05,
        0.42,
        "Composable numerical methods for Rust.",
        family="serif",
        fontsize=22,
        color="#5C6370",  # color-fg-muted
    )

    # Equation classes — adds substance to the card without being noisy.
    fig.text(
        0.05,
        0.18,
        "ODE  ·  SDE  ·  DDE  ·  FDE  ·  IDE  ·  PDE  ·  SPDE  ·  Optimization",
        family="monospace",
        fontsize=12,
        color="#0A6E6B",  # accent
    )

    # Domain in the bottom-right corner — discrete attribution.
    fig.text(
        0.95,
        0.06,
        "numra-rs.org",
        family="serif",
        fontsize=14,
        color="#5C6370",
        ha="right",
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    # Override the global `savefig.bbox = 'tight'` from style.py: the OG
    # spec mandates exactly 1200×630, which we get from figsize=(6.0, 3.15)
    # at dpi=200 only when bbox is "standard" (no auto-tight cropping).
    with plt.rc_context({"savefig.bbox": "standard"}):
        fig.savefig(OUT, format="png", dpi=200, facecolor="white")
    print(f"wrote {OUT.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()

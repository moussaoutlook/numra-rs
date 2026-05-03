"""
Apple-touch-icon — 180×180 PNG.

iOS Safari requests `/apple-touch-icon.png` when adding the site to the
home screen and elsewhere. The official Apple guidance is 180×180 PNG.

Design: the brand glyph is the italic α from the wordmark — not the full
"Numra" wordmark, which would render unreadably small at 180px square.
The α sits centered on the brand accent color so the mark looks
intentional at every iOS multitasking scale.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt

THIS_DIR = Path(__file__).resolve().parent
FIGURES_DIR = THIS_DIR.parent
sys.path.insert(0, str(FIGURES_DIR))
from lib import style  # noqa: E402

style.apply()

REPO_ROOT = FIGURES_DIR.parent.parent
OUT = REPO_ROOT / "website" / "site" / "public" / "apple-touch-icon.png"


def main() -> None:
    # 180px at dpi=180 → figsize (1.0, 1.0). dpi=200 → 200, etc.
    # Use dpi=180 / size=1.0 for an exact 180×180 output.
    fig = plt.figure(figsize=(1.0, 1.0), dpi=180)

    # Solid accent background — readable at any iOS chrome size.
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    ax.set_facecolor(style.ACCENT)

    # Centered α glyph — italic serif so it matches the wordmark's tspan.
    fig.text(
        0.5,
        0.40,  # nudge slightly below center so the glyph reads optically-centered
        r"$\alpha$",
        family="serif",
        fontsize=120,
        fontweight="bold",
        color="#FFFFFF",
        ha="center",
        va="center",
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with plt.rc_context({"savefig.bbox": "standard"}):
        fig.savefig(OUT, format="png", dpi=180, facecolor=style.ACCENT)
    print(f"wrote {OUT.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()

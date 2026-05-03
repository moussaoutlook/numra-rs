"""
Numra figure style.

Single source of truth for matplotlib defaults across every plot shipped to
the website. Imported by every figure script via `from lib.style import apply`.

Math is rendered with matplotlib's built-in mathtext (Computer Modern), not a
TeX Live install. This keeps the pipeline portable. Escalate to
`text.usetex=True` only on a per-figure basis if a specific plot demands it.

The Wong color-blind-safe palette mirrors `--plot-1..8` in
`website/site/src/styles/tokens.css` so plots, the marketing site, and the
book share the same series colors.
"""

from __future__ import annotations

import matplotlib as mpl
from cycler import cycler

# Wong palette — matches tokens.css --plot-1..8 exactly.
WONG = [
    "#000000",  # plot-1
    "#E69F00",  # plot-2
    "#56B4E9",  # plot-3
    "#009E73",  # plot-4
    "#F0E442",  # plot-5
    "#0072B2",  # plot-6
    "#D55E00",  # plot-7
    "#CC79A7",  # plot-8
]

# Numra brand accent (deep teal, light mode). Used for emphasis lines.
ACCENT = "#0A6E6B"


def apply(*, square: bool = False) -> None:
    """Apply Numra figure defaults to the global matplotlib state.

    Call once at the top of every figure script before creating axes.

    Args:
        square: If True, default figure is 5.5×5.5 (thumbnails). Otherwise
            7.0×4.3 (golden-ratio landscape).
    """
    mpl.rcParams.update(
        {
            # Math: built-in Computer Modern, no TeX dependency.
            "text.usetex": False,
            "mathtext.fontset": "cm",
            "mathtext.rm": "serif",
            # Body fonts. DejaVu Serif is the matplotlib default fallback;
            # Source Serif Pro picks up if installed system-wide.
            "font.family": "serif",
            "font.serif": ["Source Serif Pro", "DejaVu Serif"],
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            # Figure shape and DPI.
            "figure.figsize": (5.5, 5.5) if square else (7.0, 4.3),
            "figure.dpi": 100,  # screen; SVG export is resolution-independent
            "savefig.dpi": 200,
            "savefig.bbox": "tight",
            "savefig.transparent": False,
            "savefig.facecolor": "white",
            # Grid: subtle, behind the data.
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.5,
            "axes.axisbelow": True,
            # Spines: just left and bottom.
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            # Series colors: Wong, in order.
            "axes.prop_cycle": cycler(color=WONG),
            # Lines.
            "lines.linewidth": 1.5,
            "lines.markersize": 4.5,
            # Legend chrome.
            "legend.frameon": False,
            "legend.handlelength": 1.5,
        }
    )

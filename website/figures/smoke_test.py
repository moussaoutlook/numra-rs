"""
Smoke test for the figure pipeline.

Renders a trivial decay plot to `../site/public/figures/_smoke.svg` with full
provenance. Used once after wiring the pipeline to confirm:
  1. matplotlib + mathtext renders correctly (no LaTeX install needed)
  2. SPEC §5.2 caption block lands at the bottom of the figure
  3. the sidecar `.provenance.txt` is written next to the SVG
  4. the output path is reachable from Astro's public/ dir
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from lib import style
from lib.captioning import Provenance, save_with_provenance

import matplotlib.pyplot as plt  # noqa: E402

style.apply()

OUT = Path(__file__).parent / "../site/public/figures/_smoke.svg"

# A trivial deterministic curve — no benchmark data needed for the smoke test.
t = np.linspace(0.0, 5.0, 200)
y = np.exp(-t)

fig, ax = plt.subplots()
ax.plot(t, y, label=r"$y(t) = e^{-t}$")
ax.set_xlabel(r"time $t$")
ax.set_ylabel(r"$y$")
ax.set_title("Pipeline smoke test")
ax.legend(loc="upper right")

prov = Provenance(
    problem="Smoke test (analytic decay)",
    solver="N/A — pipeline check",
    tol="N/A",
    repro_script="website/figures/smoke_test.py",
)

save_with_provenance(fig, OUT, prov)
print(f"wrote {OUT.resolve()}")

"""
Provenance captions for Numra figures.

SPEC §5.2 mandates that every plot on the website declare:
- Solver settings and tolerances
- Hardware (CPU, RAM)
- Compiler version and flags
- Commit SHA of the Numra version used
- A link to the script that reproduces the plot

This module collects those data points at figure-render time and writes them
into a single human-readable caption block plus a sidecar `.provenance.txt`
file shipped next to the SVG.
"""

from __future__ import annotations

import platform
import shutil
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import matplotlib as mpl


@dataclass(frozen=True)
class Provenance:
    problem: str
    solver: str
    tol: str
    repro_script: str
    extra: dict[str, str] = field(default_factory=dict)


def _run(cmd: list[str]) -> str:
    """Run a command and return stripped stdout. Empty string on any failure."""
    if shutil.which(cmd[0]) is None:
        return ""
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, text=True, timeout=5)
        return out.strip()
    except (subprocess.SubprocessError, OSError):
        return ""


def _cpu_brand() -> str:
    if platform.system() == "Darwin":
        return _run(["sysctl", "-n", "machdep.cpu.brand_string"]) or "Apple Silicon (unknown SKU)"
    if platform.system() == "Linux":
        try:
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
        except OSError:
            pass
    return platform.processor() or platform.machine()


def _ram_gb() -> str:
    if platform.system() == "Darwin":
        bytes_str = _run(["sysctl", "-n", "hw.memsize"])
        if bytes_str.isdigit():
            return f"{int(bytes_str) // (1024**3)} GB"
    if platform.system() == "Linux":
        try:
            for line in Path("/proc/meminfo").read_text().splitlines():
                if line.startswith("MemTotal:"):
                    kb = int(line.split()[1])
                    return f"{kb // (1024**2)} GB"
        except (OSError, ValueError, IndexError):
            pass
    return "unknown"


def _commit_sha(repo_root: Path) -> str:
    return _run(["git", "-C", str(repo_root), "rev-parse", "--short", "HEAD"]) or "unknown"


def _rustc_version() -> str:
    out = _run(["rustc", "--version"])
    # `rustc --version` → "rustc 1.83.0 (90b35a623 2026-01-15)" → "1.83.0"
    parts = out.split()
    return parts[1] if len(parts) >= 2 else out or "unknown"


def collect(repo_root: Path | None = None) -> dict[str, str]:
    """Gather machine/toolchain provenance. Cheap; safe to call per figure."""
    if repo_root is None:
        # Walk up from this file until we find a .git directory.
        here = Path(__file__).resolve()
        for ancestor in [here, *here.parents]:
            if (ancestor / ".git").exists():
                repo_root = ancestor
                break
        else:
            repo_root = Path.cwd()

    return {
        "commit": _commit_sha(repo_root),
        "cpu": _cpu_brand(),
        "ram": _ram_gb(),
        "os": f"{platform.system()} {platform.release()} ({platform.machine()})",
        "rustc": _rustc_version(),
        "python": platform.python_version(),
        "matplotlib": mpl.__version__,
        "rendered_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def caption_text(prov: Provenance, env: dict[str, str]) -> str:
    """One-line caption suitable for inclusion as a footnote on the figure."""
    parts = [
        f"Problem: {prov.problem}",
        f"Solver: {prov.solver}",
        f"tol: {prov.tol}",
        f"Numra @ {env['commit']}",
        f"rustc {env['rustc']}",
        env["cpu"],
        f"Repro: {prov.repro_script}",
    ]
    return " · ".join(parts)


def sidecar_text(prov: Provenance, env: dict[str, str]) -> str:
    """Verbose multi-line provenance, written next to the SVG as `.provenance.txt`."""
    lines = [
        f"Numra figure provenance — rendered {env['rendered_utc']}",
        "",
        f"Problem        : {prov.problem}",
        f"Solver         : {prov.solver}",
        f"Tolerance      : {prov.tol}",
        f"Repro script   : {prov.repro_script}",
        "",
        f"Numra commit   : {env['commit']}",
        f"rustc version  : {env['rustc']}",
        "",
        f"Hardware (CPU) : {env['cpu']}",
        f"Hardware (RAM) : {env['ram']}",
        f"OS             : {env['os']}",
        "",
        f"Python         : {env['python']}",
        f"matplotlib     : {env['matplotlib']}",
    ]
    if prov.extra:
        lines.append("")
        lines.append("Extra:")
        for k, v in sorted(prov.extra.items()):
            lines.append(f"  {k:14s}: {v}")
    lines.append("")
    return "\n".join(lines)


def annotate(fig, prov: Provenance, *, env: dict[str, str] | None = None) -> dict[str, str]:
    """Stamp a one-line caption onto the bottom of `fig`. Returns the env dict."""
    if env is None:
        env = collect()

    fig.text(
        0.5,
        0.005,
        caption_text(prov, env),
        ha="center",
        va="bottom",
        fontsize=6.5,
        color="#5C6370",  # tokens.css --color-fg-muted
        wrap=True,
    )
    # Reserve space at the bottom so the caption doesn't overlap axes labels.
    fig.subplots_adjust(bottom=0.18)
    return env


def save_with_provenance(fig, out_svg: Path, prov: Provenance) -> None:
    """Save SVG and write a sidecar `<out_svg>.provenance.txt`.

    The sidecar is checked into git so the figure's exact provenance is
    discoverable without re-running this script.
    """
    out_svg = Path(out_svg)
    env = annotate(fig, prov)
    out_svg.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_svg, format="svg")
    sidecar = out_svg.with_suffix(out_svg.suffix + ".provenance.txt")
    sidecar.write_text(sidecar_text(prov, env))

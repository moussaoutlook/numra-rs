#!/usr/bin/env python3
"""
Lightweight mdBook math linter for numra-book/src.

Goals:
- Make MathJax rendering predictable by standardizing on:
  - inline:  \\( ... \\)
  - display: \\[ ... \\]
- Catch common sources of broken rendering early in CI.

This is intentionally a pragmatic scanner (not a full Markdown parser).
It ignores fenced code blocks and tries to track whether we are inside
MathJax display math blocks.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BOOK_SRC = ROOT / "numra-book" / "src"


@dataclass(frozen=True)
class Issue:
    path: Path
    line: int
    kind: str
    message: str


FENCE_START_RE = re.compile(r"^```")

# Fast heuristics for "looks like TeX".
TEX_TOKENS_RE = re.compile(
    r"""\\(frac|sum|int|prod|sqrt|left|right|lVert|rVert|lvert|rvert|cdot|times|leq|geq|neq|approx|sim|to|infty|partial|nabla|mathcal|mathrm|mathbf|text)\b"""
)

BEGIN_END_RE = re.compile(r"\\(begin|end)\{([A-Za-z*]+)\}")


def strip_inline_math(text: str) -> str:
    """
    Remove inline MathJax segments so we can search for naked TeX outside math.
    This only strips \\( ... \\) segments on a single line.
    """
    out: list[str] = []
    i = 0
    while i < len(text):
        start = text.find(r"\(", i)
        if start == -1:
            out.append(text[i:])
            break
        end = text.find(r"\)", start + 2)
        if end == -1:
            out.append(text[i:])
            break
        out.append(text[i:start])
        i = end + 2
    return "".join(out)


def find_inline_dollar_math(line: str) -> bool:
    """
    Detect presence of $...$ on a line (excluding $$).

    We keep this intentionally simple: any odd number of single-dollar
    markers is suspicious, and any pair of single dollars suggests $...$ usage.
    """
    if "$" not in line:
        return False
    # Remove code spans to reduce false positives.
    scrubbed = re.sub(r"`[^`]*`", "", line)
    # Ignore $$ (display).
    scrubbed = scrubbed.replace("$$", "")
    return scrubbed.count("$") >= 2


def iter_issues(path: Path) -> list[Issue]:
    issues: list[Issue] = []
    lines = path.read_text(encoding="utf-8").splitlines()

    in_fence = False
    fence_info = ""

    in_display_math = False
    display_opener: tuple[str, int] | None = None  # (token, line_no)

    for idx, raw in enumerate(lines, start=1):
        line = raw.rstrip("\n")

        # Fence tracking (ignore content inside).
        if FENCE_START_RE.match(line):
            if not in_fence:
                in_fence = True
                fence_info = line[3:].strip()
            else:
                in_fence = False
                fence_info = ""
            continue
        if in_fence:
            continue

        # Ignore HTML comments entirely (they often contain escaped math examples).
        if line.strip().startswith("<!--"):
            continue

        # Enforce: no $$ ... $$ (we standardize on \\[...\\]).
        has_dollars_display = "$$" in line
        if has_dollars_display:
            issues.append(
                Issue(
                    path=path,
                    line=idx,
                    kind="dollars-display",
                    message="Found '$$' display math. Use '\\\\[ ... \\\\]' instead.",
                )
            )
            # Avoid noisy follow-on flags on the same line; we'll fix $$ mechanically.
            continue

        # Track \[ and \] display-math blocks.
        if r"\[" in line:
            if in_display_math:
                issues.append(
                    Issue(
                        path=path,
                        line=idx,
                        kind="display-nesting",
                        message="Nested '\\\\[' detected (already inside display math).",
                    )
                )
            else:
                in_display_math = True
                display_opener = (r"\[", idx)
        if r"\]" in line:
            if not in_display_math:
                issues.append(
                    Issue(
                        path=path,
                        line=idx,
                        kind="display-unopened",
                        message="'\\\\]' detected without a matching '\\\\['.",
                    )
                )
            else:
                in_display_math = False
                display_opener = None

        # Environment containment: \begin{...} / \end{...} must be inside display math.
        for m in BEGIN_END_RE.finditer(line):
            if not in_display_math:
                env = m.group(2)
                issues.append(
                    Issue(
                        path=path,
                        line=idx,
                        kind="env-outside-display",
                        message=f"TeX environment '{env}' appears outside '\\\\[...\\\\]'.",
                    )
                )

        # Inline dollar math is banned (standardize to \( ... \)).
        if find_inline_dollar_math(line):
            issues.append(
                Issue(
                    path=path,
                    line=idx,
                    kind="dollars-inline",
                    message="Found '$...$' inline math. Use '\\\\( ... \\\\)' instead.",
                )
            )

        # Naked TeX: TeX tokens outside any math delimiters on this line.
        if not in_display_math:
            candidate = strip_inline_math(line)
            if TEX_TOKENS_RE.search(candidate):
                issues.append(
                    Issue(
                        path=path,
                        line=idx,
                        kind="naked-tex",
                        message="TeX command appears outside math delimiters.",
                    )
                )

    if in_display_math and display_opener is not None:
        token, open_line = display_opener
        issues.append(
            Issue(
                path=path,
                line=open_line,
                kind="display-unclosed",
                message=f"Unclosed display math block opened with '{token}'.",
            )
        )

    return issues


def main() -> int:
    if not BOOK_SRC.is_dir():
        print(f"error: missing book source directory: {BOOK_SRC}", file=sys.stderr)
        return 2

    all_issues: list[Issue] = []
    for path in sorted(BOOK_SRC.rglob("*.md")):
        all_issues.extend(iter_issues(path))

    if all_issues:
        print("Book math lint FAILED:", file=sys.stderr)
        for issue in all_issues:
            rel = issue.path.relative_to(ROOT)
            print(f"  {rel}:{issue.line}: [{issue.kind}] {issue.message}", file=sys.stderr)
        return 1

    print("Book math lint OK (no issues found).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

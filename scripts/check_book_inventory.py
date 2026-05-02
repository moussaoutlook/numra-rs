#!/usr/bin/env python3
"""
Validate the mdBook against the machine-readable public API inventory.

Run from repo root:
  python3 scripts/check_book_inventory.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BOOK_SRC = ROOT / "numra-book" / "src"
INVENTORY = ROOT / "docs" / "audit" / "inventory.yaml"

# Strings that must not appear in book markdown (stale type names vs crates).
# Only flag strings that would compile incorrectly if copied from the book.
DENY_SUBSTRINGS = [
    "numra::sde::SriW1",
    "numra::dde::DdeDoPri5",
]


def parse_inventory(path: Path) -> list[dict[str, object]]:
    """Parse the deliberately small YAML subset used by docs/audit/inventory.yaml."""
    records: list[dict[str, object]] = []
    current: dict[str, object] | None = None
    current_list_key: str | None = None

    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.rstrip()
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        if line.startswith("- id: "):
            current = {"id": line.split(": ", 1)[1].strip()}
            records.append(current)
            current_list_key = None
            continue

        if current is None:
            raise ValueError(f"inventory entry data before first id: {line}")

        if line.startswith("  ") and not line.startswith("    - "):
            key, sep, value = stripped.partition(":")
            if not sep:
                raise ValueError(f"invalid inventory line: {line}")
            key = key.strip()
            value = value.strip()
            if value:
                if value in {"true", "false"}:
                    current[key] = value == "true"
                else:
                    current[key] = value
                current_list_key = None
            else:
                current[key] = []
                current_list_key = key
            continue

        if line.startswith("    - "):
            if current_list_key is None:
                raise ValueError(f"list item without list key: {line}")
            values = current[current_list_key]
            assert isinstance(values, list)
            values.append(stripped[2:].strip())
            continue

        raise ValueError(f"unsupported inventory line: {line}")

    return records


def inventory_paths(records: list[dict[str, object]]) -> set[str]:
    paths: set[str] = set()
    for record in records:
        facade_path = record.get("facade_path")
        if isinstance(facade_path, str):
            paths.add(facade_path)
        aliases = record.get("aliases")
        if isinstance(aliases, list):
            paths.update(str(alias) for alias in aliases)
    return paths


def validate_records(records: list[dict[str, object]]) -> list[str]:
    errors: list[str] = []
    seen: set[str] = set()
    for record in records:
        record_id = str(record.get("id", ""))
        if not record_id:
            errors.append("inventory row missing id")
            continue
        if record_id in seen:
            errors.append(f"duplicate inventory id: {record_id}")
        seen.add(record_id)

        anchors = record.get("book_anchors")
        theory_only = record.get("theory_only") is True
        if not theory_only and not anchors:
            errors.append(f"{record_id}: missing book_anchors")
        if isinstance(anchors, list):
            for anchor in anchors:
                anchor_path = ROOT / str(anchor)
                if not anchor_path.is_file():
                    errors.append(f"{record_id}: missing book anchor {anchor}")
    return errors


def main() -> int:
    if not BOOK_SRC.is_dir():
        print(f"error: missing {BOOK_SRC}", file=sys.stderr)
        return 2
    if not INVENTORY.is_file():
        print(f"error: missing {INVENTORY}", file=sys.stderr)
        return 2

    try:
        records = parse_inventory(INVENTORY)
    except ValueError as exc:
        print(f"error: invalid inventory: {exc}", file=sys.stderr)
        return 2

    record_errors = validate_records(records)
    allowed_paths = inventory_paths(records)

    bad: list[tuple[Path, str]] = []
    unknown_paths: list[tuple[Path, int, str]] = []
    for path in sorted(BOOK_SRC.rglob("*.md")):
        text = path.read_text(encoding="utf-8")
        for s in DENY_SUBSTRINGS:
            if s in text:
                bad.append((path, s))
        for line_no, line in enumerate(text.splitlines(), start=1):
            for match in re.finditer(r"`(numra::[A-Za-z0-9_:]+)`", line):
                book_path = match.group(1)
                if book_path not in allowed_paths:
                    unknown_paths.append((path, line_no, book_path))

    if record_errors:
        print("Book inventory check FAILED. Inventory errors:\n", file=sys.stderr)
        for error in record_errors:
            print(f"  {error}", file=sys.stderr)
        return 1

    if bad:
        print("Book inventory check FAILED. Stale identifiers found:\n", file=sys.stderr)
        for path, s in bad:
            rel = path.relative_to(ROOT)
            print(f"  {rel}: contains {s!r}", file=sys.stderr)
        print(
            "\nFix: align with numra-sde (Sra1, Sra2) and numra-dde (MethodOfSteps).",
            file=sys.stderr,
        )
        return 1

    if unknown_paths:
        print("Book inventory check FAILED. Unknown book API paths:\n", file=sys.stderr)
        for path, line_no, book_path in unknown_paths:
            rel = path.relative_to(ROOT)
            print(f"  {rel}:{line_no}: {book_path}", file=sys.stderr)
        print(
            "\nFix: add a docs/audit/inventory.yaml row or alias for each public path.",
            file=sys.stderr,
        )
        return 1

    print(
        f"Book inventory check OK ({len(records)} inventory rows, "
        "no stale or unknown book paths)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

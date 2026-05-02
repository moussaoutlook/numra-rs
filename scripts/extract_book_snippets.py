#!/usr/bin/env python3
"""Validate mdBook Rust snippets.

The script enforces documented `rust,ignore` fences and compiles every plain
`rust` fence in an isolated temporary crate under target/book-snippets.
"""

from __future__ import annotations

import hashlib
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BOOK_SRC = ROOT / "numra-book" / "src"
TARGET = ROOT / "target" / "book-snippets"
IGNORE_PREFIX = "<!-- book-ignore:"


@dataclass(frozen=True)
class Fence:
    path: Path
    start_line: int
    info: str
    code: str
    previous_nonblank: str


def iter_fences(path: Path) -> list[Fence]:
    lines = path.read_text(encoding="utf-8").splitlines()
    fences: list[Fence] = []
    in_fence = False
    info = ""
    start_line = 0
    body: list[str] = []
    previous_nonblank = ""

    for idx, line in enumerate(lines, start=1):
        if not in_fence:
            if line.startswith("```"):
                in_fence = True
                info = line[3:].strip()
                start_line = idx
                body = []
            elif line.strip():
                previous_nonblank = line.strip()
            continue

        if line.startswith("```"):
            fences.append(
                Fence(
                    path=path,
                    start_line=start_line,
                    info=info,
                    code="\n".join(body),
                    previous_nonblank=previous_nonblank,
                )
            )
            in_fence = False
            info = ""
            body = []
            previous_nonblank = line.strip()
            continue

        body.append(line)

    return fences


def is_rust(info: str) -> bool:
    parts = {part.strip() for part in info.split(",") if part.strip()}
    return "rust" in parts


def is_ignored(info: str) -> bool:
    parts = {part.strip() for part in info.split(",") if part.strip()}
    return "ignore" in parts


def visible_code(code: str) -> str:
    lines: list[str] = []
    for line in code.splitlines():
        if line.startswith("# "):
            lines.append(line[2:])
        elif line == "#":
            lines.append("")
        elif not line.startswith("#"):
            lines.append(line)
    return "\n".join(lines).strip()


def crate_source(code: str) -> str:
    cleaned = visible_code(code)
    if "fn main" in cleaned or "#![no_std]" in cleaned:
        return cleaned + "\n"
    return f"fn main() {{\n{indent(cleaned)}\n}}\n"


def indent(code: str) -> str:
    if not code:
        return ""
    return "\n".join(f"    {line}" if line else "" for line in code.splitlines())


def write_snippet_crate(fence: Fence, index: int) -> Path:
    digest = hashlib.sha256(
        f"{fence.path.relative_to(ROOT)}:{fence.start_line}".encode()
    ).hexdigest()[:12]
    crate_dir = TARGET / f"snippet_{index:04d}_{digest}"
    src_dir = crate_dir / "src"
    src_dir.mkdir(parents=True, exist_ok=True)
    (crate_dir / "Cargo.toml").write_text(
        "\n".join(
            [
                "[package]",
                f'name = "numra_book_snippet_{index:04d}_{digest}"',
                'version = "0.0.0"',
                'edition = "2021"',
                "",
                "[dependencies]",
                'numra = { path = "../../../numra" }',
                "",
            ]
        ),
        encoding="utf-8",
    )
    (src_dir / "main.rs").write_text(crate_source(fence.code), encoding="utf-8")
    return crate_dir


def main() -> int:
    if not BOOK_SRC.is_dir():
        print(f"error: missing book source directory: {BOOK_SRC}", file=sys.stderr)
        return 2

    rust_fences: list[Fence] = []
    ignored_fences = 0
    undocumented_ignores: list[Fence] = []

    for path in sorted(BOOK_SRC.rglob("*.md")):
        for fence in iter_fences(path):
            if not is_rust(fence.info):
                continue
            if is_ignored(fence.info):
                ignored_fences += 1
                if not fence.previous_nonblank.startswith(IGNORE_PREFIX):
                    undocumented_ignores.append(fence)
                continue
            rust_fences.append(fence)

    if undocumented_ignores:
        print("Book snippet check FAILED. Missing book-ignore reason:", file=sys.stderr)
        for fence in undocumented_ignores:
            rel = fence.path.relative_to(ROOT)
            print(f"  {rel}:{fence.start_line}", file=sys.stderr)
        return 1

    if TARGET.exists():
        shutil.rmtree(TARGET)
    TARGET.mkdir(parents=True, exist_ok=True)

    for index, fence in enumerate(rust_fences, start=1):
        crate_dir = write_snippet_crate(fence, index)
        rel = fence.path.relative_to(ROOT)
        result = subprocess.run(
            ["cargo", "check", "--quiet", "--manifest-path", str(crate_dir / "Cargo.toml")],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        if result.returncode != 0:
            print(
                f"Book snippet check FAILED for {rel}:{fence.start_line}",
                file=sys.stderr,
            )
            print(result.stdout, file=sys.stderr)
            print(result.stderr, file=sys.stderr)
            return result.returncode

    print(
        f"Book snippet check OK ({len(rust_fences)} compiled, "
        f"{ignored_fences} documented ignores)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

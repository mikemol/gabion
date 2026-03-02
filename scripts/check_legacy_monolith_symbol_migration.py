#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

_CANONICAL_LEGACY_MODULE = "src/gabion/analysis/legacy_dataflow_monolith.py"

_EXPLICIT_SYMBOL_RE = re.compile(
    r"(?:src/gabion/analysis/)?legacy_dataflow_monolith\.py::([A-Za-z_][A-Za-z0-9_]*)"
)
_BACKTICK_SYMBOL_RE = re.compile(r"`([A-Za-z_][A-Za-z0-9_]*)`")


def _normalize_legacy_symbol(raw_symbol: str) -> str:
    symbol = raw_symbol.strip().strip("`")
    if symbol == "legacy_dataflow_monolith.py":
        return _CANONICAL_LEGACY_MODULE
    if symbol.startswith(_CANONICAL_LEGACY_MODULE):
        return symbol
    if symbol.startswith("legacy_dataflow_monolith.py::"):
        return f"{_CANONICAL_LEGACY_MODULE}::{symbol.split('::', 1)[1]}"
    if symbol.startswith("legacy_dataflow_monolith.py"):
        return _CANONICAL_LEGACY_MODULE
    return symbol


def _collect_doc_anchors(path: Path) -> set[str]:
    text = path.read_text(encoding="utf-8")
    anchors: set[str] = set()

    if "legacy_dataflow_monolith.py" in text:
        anchors.add(_CANONICAL_LEGACY_MODULE)

    for match in _EXPLICIT_SYMBOL_RE.finditer(text):
        anchors.add(f"{_CANONICAL_LEGACY_MODULE}::{match.group(1)}")

    for line in text.splitlines():
        if "legacy_dataflow_monolith.py" not in line:
            continue
        for symbol_match in _BACKTICK_SYMBOL_RE.finditer(line):
            symbol = symbol_match.group(1)
            if symbol.startswith("_"):
                anchors.add(f"{_CANONICAL_LEGACY_MODULE}::{symbol}")

    return anchors


def _collect_index_rows(index_path: Path) -> set[str]:
    rows: set[str] = set()
    for raw_line in index_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line.startswith("|"):
            continue
        if line.startswith("| ---"):
            continue
        columns = [column.strip() for column in line.strip("|").split("|")]
        if not columns:
            continue
        legacy_raw = columns[0]
        legacy_symbol = _normalize_legacy_symbol(legacy_raw)
        if "legacy_dataflow_monolith.py" not in legacy_symbol:
            continue
        rows.add(legacy_symbol)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Verify docs/dataflow_legacy_monolith_symbol_migration_index.md covers "
            "every legacy_dataflow_monolith.py anchor in in/*.md"
        )
    )
    parser.add_argument("--root", default=".", help="Repository root")
    parser.add_argument(
        "--index",
        default="docs/dataflow_legacy_monolith_symbol_migration_index.md",
        help="Migration index markdown path",
    )
    parser.add_argument(
        "--in-glob",
        default="in/*.md",
        help="Glob used to collect in-doc anchors",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    index_path = (root / args.index).resolve()
    if not index_path.exists():
        print(f"error: missing migration index: {index_path}", file=sys.stderr)
        return 1

    anchors: set[str] = set()
    for path in sorted(root.glob(args.in_glob)):
        if path.is_file():
            anchors.update(_collect_doc_anchors(path))

    index_rows = _collect_index_rows(index_path)
    missing = sorted(anchor for anchor in anchors if anchor not in index_rows)

    if not anchors:
        print(
            "legacy monolith migration index coverage check failed: "
            "no legacy monolith anchors discovered in in/*.md",
            file=sys.stderr,
        )
        return 1

    if missing:
        print(
            "legacy monolith migration index coverage check failed:",
            file=sys.stderr,
        )
        for anchor in missing:
            print(f"  missing row: {anchor}", file=sys.stderr)
        return 1

    print(
        "legacy monolith migration index coverage check passed: "
        f"{len(anchors)} anchors covered"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

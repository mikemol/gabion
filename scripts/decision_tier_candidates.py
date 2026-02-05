#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


def _parse_lint_line(line: str) -> tuple[str, int, int, str] | None:
    parts = line.strip().split(": ", 1)
    if len(parts) != 2:
        return None
    location, remainder = parts
    loc_parts = location.split(":")
    if len(loc_parts) < 3:
        return None
    path = ":".join(loc_parts[:-2])
    try:
        line_no = int(loc_parts[-2])
        col_no = int(loc_parts[-1])
    except ValueError:
        return None
    code = remainder.split(" ", 1)[0]
    return path, line_no, col_no, code


def _latest_lint_path(root: Path) -> Path:
    marker = root / "artifacts" / "audit_snapshots" / "LATEST.txt"
    stamp = marker.read_text().strip()
    return root / "artifacts" / "audit_snapshots" / stamp / "lint.txt"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract decision-tier candidate keys from lint.txt."
    )
    parser.add_argument("--lint", type=Path, default=None, help="Path to lint.txt")
    parser.add_argument("--root", type=Path, default=Path("."), help="Repo root")
    parser.add_argument(
        "--tier",
        type=int,
        default=3,
        choices=(1, 2, 3),
        help="Tier to emit candidates for (default: 3).",
    )
    parser.add_argument(
        "--format",
        choices=("toml", "lines"),
        default="toml",
        help="Output format (default: toml).",
    )
    args = parser.parse_args()

    lint_path = args.lint or _latest_lint_path(args.root)
    codes = {"GABION_DECISION_SURFACE", "GABION_VALUE_DECISION_SURFACE"}
    keys: list[str] = []
    for line in lint_path.read_text().splitlines():
        parsed = _parse_lint_line(line)
        if not parsed:
            continue
        path, line_no, col_no, code = parsed
        if code not in codes:
            continue
        keys.append(f"{path}:{line_no}:{col_no}")

    keys = sorted(set(keys))
    if args.format == "lines":
        for key in keys:
            print(key)
        return 0

    tier_key = f"tier{args.tier}"
    print("[decision]")
    print(f"{tier_key} = [")
    for key in keys:
        print(f"  \"{key}\",")
    print("]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

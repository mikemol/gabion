#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable


def _parse_lint_line(line: str) -> tuple[str, int, int, str, str] | None:
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
    remainder_parts = remainder.split(" ", 1)
    if not remainder_parts:
        return None
    code = remainder_parts[0]
    message = remainder_parts[1] if len(remainder_parts) > 1 else ""
    return path, line_no, col_no, code, message


def _load_lines(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(path)
    return [line for line in path.read_text().splitlines() if line.strip()]


def _latest_lint_path(root: Path) -> Path:
    marker = root / "artifacts" / "audit_snapshots" / "LATEST.txt"
    if not marker.exists():
        raise FileNotFoundError(marker)
    stamp = marker.read_text().strip()
    if not stamp:
        raise ValueError("LATEST.txt is empty")
    return root / "artifacts" / "audit_snapshots" / stamp / "lint.txt"


def _summarize(lines: Iterable[str]) -> dict[str, object]:
    codes = Counter()
    files = Counter()
    total = 0
    by_code_file: dict[str, Counter[str]] = defaultdict(Counter)
    for line in lines:
        parsed = _parse_lint_line(line)
        if not parsed:
            continue
        path, _, _, code, _ = parsed
        total += 1
        codes[code] += 1
        files[path] += 1
        by_code_file[code][path] += 1
    return {
        "total": total,
        "codes": dict(codes.most_common()),
        "files": dict(files.most_common()),
        "by_code_file": {
            code: dict(counter.most_common()) for code, counter in by_code_file.items()
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize Gabion lint output.")
    parser.add_argument("--lint", type=Path, default=None, help="Path to lint.txt")
    parser.add_argument("--root", type=Path, default=Path("."), help="Repo root")
    parser.add_argument("--json", action="store_true", help="Emit JSON summary")
    parser.add_argument("--top", type=int, default=10, help="Show top N entries")
    args = parser.parse_args()

    lint_path = args.lint or _latest_lint_path(args.root)
    lines = _load_lines(lint_path)
    summary = _summarize(lines)

    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0

    total = summary["total"]
    print(f"Lint summary for {lint_path} ({total} findings)")
    print("\nTop codes:")
    for code, count in list(summary["codes"].items())[: args.top]:
        print(f"- {code}: {count}")
    print("\nTop files:")
    for path, count in list(summary["files"].items())[: args.top]:
        print(f"- {path}: {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

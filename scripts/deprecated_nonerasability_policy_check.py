#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping

from gabion.analysis.deprecated_substrate import (
    DeprecatedFiber,
    enforce_non_erasability_policy,
)


def _load_rows(path: Path) -> list[DeprecatedFiber]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        return []
    rows = raw.get("deprecated_fibers", [])
    if not isinstance(rows, list):
        return []
    fibers: list[DeprecatedFiber] = []
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        fibers.append(DeprecatedFiber.from_payload(row))
    return fibers


def main() -> int:
    parser = argparse.ArgumentParser(description="Enforce non-erasable deprecated fibers")
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--current", type=Path, required=True)
    args = parser.parse_args()

    baseline = _load_rows(args.baseline)
    current = _load_rows(args.current)
    result = enforce_non_erasability_policy(previous_fibers=baseline, current_fibers=current)
    if result.ok:
        return 0
    for error in result.errors:
        print(error)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

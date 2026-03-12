#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable, Mapping, Sequence

from gabion.tooling.runtime.policy_result_schema import make_policy_result, write_policy_result

from gabion.analysis.core.deprecated_substrate import (
    DeprecatedFiber, enforce_non_erasability_policy)


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


def _serialize_errors(errors: list[str]) -> list[dict[str, object]]:
    return [{"message": error, "render": error} for error in errors]


def _write_skip_result_if_requested(
    *,
    baseline: Path,
    current: Path,
    output: Path | None,
) -> None:
    if output is None:
        return
    write_policy_result(
        path=output.resolve(),
        result=make_policy_result(
            rule_id="deprecated_nonerasability",
            status="skip",
            violations=[
                {
                    "message": "baseline/current payload missing; rule skipped by child policy check",
                    "render": (
                        f"missing baseline={baseline.exists()} "
                        f"current={current.exists()}"
                    ),
                }
            ],
            baseline_mode="baseline_compare",
            source_tool="scripts/policy/deprecated_nonerasability_policy_check.py",
            input_scope={
                "baseline": str(baseline),
                "current": str(current),
            },
        ),
    )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Enforce non-erasable deprecated fibers")
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--current", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args(list(argv) if argv is not None else None)


def main(
    argv: Sequence[str] | None = None,
    *,
    print_fn: Callable[[str], None] = print,
) -> int:
    args = _parse_args(argv)

    if not args.baseline.exists() or not args.current.exists():
        _write_skip_result_if_requested(
            baseline=args.baseline,
            current=args.current,
            output=args.output,
        )
        return 0

    baseline = _load_rows(args.baseline)
    current = _load_rows(args.current)
    result = enforce_non_erasability_policy(previous_fibers=baseline, current_fibers=current)
    errors = list(result.errors)
    if args.output is not None:
        write_policy_result(
            path=args.output.resolve(),
            result=make_policy_result(
                rule_id="deprecated_nonerasability",
                status="pass" if result.ok else "fail",
                violations=_serialize_errors(errors),
                baseline_mode="baseline_compare",
                source_tool="scripts/policy/deprecated_nonerasability_policy_check.py",
                input_scope={
                    "baseline": str(args.baseline),
                    "current": str(args.current),
                },
            ),
        )
    if result.ok:
        return 0
    for error in errors:
        print_fn(error)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

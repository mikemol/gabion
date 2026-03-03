#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

DEFAULT_BASELINES = (
    "baselines/branchless_policy_baseline.json",
    "baselines/defensive_fallback_policy_baseline.json",
    "baselines/test_obsolescence_baseline.json",
)


def _iter_leaf_diffs(
    previous: Any,
    current: Any,
    prefix: str = "",
) -> Iterator[tuple[str, Any, Any]]:
    if type(previous) is not type(current):
        yield (prefix, previous, current)
        return
    if isinstance(previous, dict):
        keys = sorted(set(previous) | set(current))
        for key in keys:
            key_prefix = f"{prefix}.{key}" if prefix else key
            if key not in previous:
                yield (key_prefix, "<MISSING>", current[key])
                continue
            if key not in current:
                yield (key_prefix, previous[key], "<MISSING>")
                continue
            yield from _iter_leaf_diffs(previous[key], current[key], key_prefix)
        return
    if isinstance(previous, list):
        limit = max(len(previous), len(current))
        for idx in range(limit):
            idx_prefix = f"{prefix}[{idx}]"
            if idx >= len(previous):
                yield (idx_prefix, "<MISSING>", current[idx])
                continue
            if idx >= len(current):
                yield (idx_prefix, previous[idx], "<MISSING>")
                continue
            yield from _iter_leaf_diffs(previous[idx], current[idx], idx_prefix)
        return
    if previous != current:
        yield (prefix, previous, current)


def _is_path_leaf(pointer: str) -> bool:
    return pointer == "path" or pointer.endswith(".path")


def _load_head_json(path: str) -> Any:
    result = subprocess.run(
        ["git", "show", f"HEAD:{path}"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise ValueError(f"Unable to load HEAD:{path}: {result.stderr.strip()}")
    return json.loads(result.stdout)


def _load_working_json(path: str) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _validate_path_only_diff(path: str) -> tuple[int, int]:
    previous = _load_head_json(path)
    current = _load_working_json(path)
    diffs = list(_iter_leaf_diffs(previous, current))
    non_path = [entry for entry in diffs if not _is_path_leaf(entry[0])]
    return (len(diffs), len(non_path))


def _project_path_only(previous: Any, current: Any, prefix: str = "") -> Any:
    if _is_path_leaf(prefix):
        return current
    if type(previous) is not type(current):
        return previous
    if isinstance(previous, dict):
        result: dict[str, Any] = {}
        for key in previous:
            key_prefix = f"{prefix}.{key}" if prefix else key
            if key in current:
                result[key] = _project_path_only(previous[key], current[key], key_prefix)
            else:
                result[key] = previous[key]
        return result
    if isinstance(previous, list):
        result: list[Any] = []
        for idx, prev_item in enumerate(previous):
            idx_prefix = f"{prefix}[{idx}]"
            if idx < len(current):
                result.append(_project_path_only(prev_item, current[idx], idx_prefix))
            else:
                result.append(prev_item)
        return result
    return previous


def _rewrite_path_only(path: str) -> tuple[int, int]:
    previous = _load_head_json(path)
    current = _load_working_json(path)
    projected = _project_path_only(previous, current)
    before = list(_iter_leaf_diffs(previous, current))
    after = list(_iter_leaf_diffs(previous, projected))
    Path(path).write_text(json.dumps(projected, indent=2) + "\n", encoding="utf-8")
    return (len(before), len(after))


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fail when baseline JSON diffs include non-path leaf changes.",
    )
    parser.add_argument(
        "paths",
        nargs="*",
        default=list(DEFAULT_BASELINES),
        help="Baseline JSON files to validate.",
    )
    parser.add_argument(
        "--rewrite-path-only",
        action="store_true",
        help=(
            "Rewrite each baseline to keep only path-leaf edits relative to HEAD. "
            "All non-path edits are discarded."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.rewrite_path_only:
        for path in args.paths:
            try:
                before_count, after_count = _rewrite_path_only(path)
            except Exception as exc:
                print(f"[baseline-guard] {path}: error: {exc}", file=sys.stderr)
                return 2
            print(
                f"[baseline-guard] {path}: rewrite_path_only "
                f"before_leaf_diffs={before_count} after_leaf_diffs={after_count}"
            )
    has_violation = False
    for path in args.paths:
        try:
            total_diffs, non_path_diffs = _validate_path_only_diff(path)
        except Exception as exc:
            print(f"[baseline-guard] {path}: error: {exc}", file=sys.stderr)
            return 2
        if total_diffs == 0:
            print(f"[baseline-guard] {path}: no changes")
            continue
        print(
            f"[baseline-guard] {path}: total_leaf_diffs={total_diffs} "
            f"non_path_leaf_diffs={non_path_diffs}"
        )
        if non_path_diffs > 0:
            has_violation = True
    if has_violation:
        print("[baseline-guard] non-path baseline changes detected", file=sys.stderr)
        return 1
    print("[baseline-guard] all baseline diffs are path-only")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

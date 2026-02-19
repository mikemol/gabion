#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Mapping

_HUNK_RE = re.compile(r"@@ -\d+(?:,\d+)? \+(?P<start>\d+)(?:,(?P<count>\d+))? @@")


@dataclass(frozen=True)
class ChangedLine:
    path: str
    line: int


def _parse_changed_lines(diff_text: str) -> list[ChangedLine]:
    changed: list[ChangedLine] = []
    current_path: str | None = None
    for raw_line in diff_text.splitlines():
        if raw_line.startswith("+++ "):
            path_token = raw_line[4:].strip()
            if path_token == "/dev/null":
                current_path = None
                continue
            if path_token.startswith("b/"):
                path_token = path_token[2:]
            current_path = path_token
            continue
        if current_path is None:
            continue
        match = _HUNK_RE.match(raw_line)
        if match is None:
            continue
        start = int(match.group("start"))
        count = int(match.group("count") or "1")
        if count <= 0:
            continue
        changed.extend(ChangedLine(path=current_path, line=start + offset) for offset in range(count))
    return changed


def _git_diff_changed_lines(root: Path, *, base: str | None, head: str | None) -> list[ChangedLine]:
    if base and head:
        diff_range = f"{base}...{head}"
        cmd = ["git", "diff", "--unified=0", diff_range]
    elif base:
        cmd = ["git", "diff", "--unified=0", base]
    else:
        cmd = ["git", "diff", "--unified=0"]
    proc = subprocess.run(
        cmd,
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        message = proc.stderr.strip() or proc.stdout.strip() or "git diff failed"
        raise RuntimeError(message)
    return _parse_changed_lines(proc.stdout)


def _load_json(path: Path) -> Mapping[str, object] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, Mapping) else None


def _refresh_index(root: Path, index_path: Path, tests_root: str) -> bool:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/extract_test_evidence.py",
            "--root",
            str(root),
            "--tests",
            tests_root,
            "--out",
            str(index_path),
        ],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    return proc.returncode == 0


def _site_matches_changed_lines(site: Mapping[str, object], lines_by_path: dict[str, set[int]]) -> bool:
    site_path = str(site.get("path", "") or "").strip()
    if not site_path:
        return False
    changed_lines = lines_by_path.get(site_path)
    if not changed_lines:
        return False
    raw_span = site.get("span")
    if isinstance(raw_span, list) and len(raw_span) == 4:
        try:
            start = int(raw_span[0])
            end = int(raw_span[2])
        except (TypeError, ValueError):
            return True
        upper = max(start, end)
        lower = min(start, end)
        return any(lower <= line <= upper for line in changed_lines)
    return True


def _collect_changed_sets(changed_lines: Iterable[ChangedLine]) -> tuple[dict[str, set[int]], set[str], set[str]]:
    lines_by_path: dict[str, set[int]] = {}
    changed_paths: set[str] = set()
    changed_tests: set[str] = set()
    for item in changed_lines:
        changed_paths.add(item.path)
        lines_by_path.setdefault(item.path, set()).add(item.line)
        if item.path.startswith("tests/") and item.path.endswith(".py"):
            changed_tests.add(item.path)
    return lines_by_path, changed_paths, changed_tests


def _select_tests(
    payload: Mapping[str, object],
    *,
    changed_lines: list[ChangedLine],
    must_run_tests: set[str],
) -> tuple[list[str], list[str], list[str], float]:
    lines_by_path, changed_paths, changed_tests = _collect_changed_sets(changed_lines)
    tests = payload.get("tests")
    if not isinstance(tests, list):
        return [], sorted(changed_paths), sorted(changed_tests), 0.0

    impacted: set[str] = set()
    for entry in tests:
        if not isinstance(entry, Mapping):
            continue
        test_id = str(entry.get("test_id", "") or "").strip()
        if not test_id:
            continue
        test_file = str(entry.get("file", "") or "").strip()
        if test_file in changed_tests:
            impacted.add(test_id)
            continue
        evidence = entry.get("evidence")
        if not isinstance(evidence, list):
            continue
        for item in evidence:
            if not isinstance(item, Mapping):
                continue
            key = item.get("key")
            if not isinstance(key, Mapping):
                continue
            site = key.get("site")
            if isinstance(site, Mapping) and _site_matches_changed_lines(site, lines_by_path):
                impacted.add(test_id)
                break

    changed_code_paths = {
        path
        for path in changed_paths
        if path.endswith(".py") and not path.startswith("tests/")
    }
    mapped_paths = {
        path
        for path in changed_code_paths
        if path in {str((item.get("key") or {}).get("site", {}).get("path", "")) for entry in tests if isinstance(entry, Mapping) for item in (entry.get("evidence") if isinstance(entry.get("evidence"), list) else []) if isinstance(item, Mapping) and isinstance(item.get("key"), Mapping)}
    }
    path_coverage = 1.0 if not changed_code_paths else len(mapped_paths) / len(changed_code_paths)
    test_signal = 1.0 if impacted else 0.0
    confidence = round(0.7 * path_coverage + 0.3 * test_signal, 4)

    must_run_impacted = sorted(test for test in impacted if test in must_run_tests)
    return sorted(impacted), sorted(changed_paths), must_run_impacted, confidence


def _read_must_run_tests(path: Path | None, inline: Iterable[str]) -> set[str]:
    tests = {value.strip() for value in inline if value.strip()}
    if path and path.exists():
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if line and not line.startswith("#"):
                tests.add(line)
    return tests


def main(
    argv: list[str] | None = None,
    *,
    git_diff_changed_lines_fn: Callable[[Path, str | None, str | None], list[ChangedLine]] | None = None,
) -> int:
    parser = argparse.ArgumentParser(description="Select impacted tests from git diff and evidence index.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--diff-base", default=os.environ.get("GITHUB_BASE_SHA"))
    parser.add_argument("--diff-head", default=os.environ.get("GITHUB_HEAD_SHA"))
    parser.add_argument("--index", default="out/test_evidence.json")
    parser.add_argument("--tests-root", default="tests")
    parser.add_argument("--out", default="artifacts/audit_reports/impact_selection.json")
    parser.add_argument("--confidence-threshold", type=float, default=0.6)
    parser.add_argument("--stale-seconds", type=int, default=86_400)
    parser.add_argument("--must-run-file")
    parser.add_argument("--must-run-test", action="append", default=[])
    parser.add_argument("--no-refresh", action="store_true")
    args = parser.parse_args(argv)

    root = Path(args.root).resolve()
    index_path = root / args.index
    output_path = root / args.out

    if git_diff_changed_lines_fn is None:
        changed_lines = _git_diff_changed_lines(root, base=args.diff_base, head=args.diff_head)
    else:
        changed_lines = git_diff_changed_lines_fn(root, args.diff_base, args.diff_head)
    changed_count = len(changed_lines)
    changed_paths = sorted({item.path for item in changed_lines})

    index_payload = _load_json(index_path)
    stale = False
    refreshed = False
    if index_payload is None:
        stale = True
    elif args.stale_seconds >= 0:
        age_seconds = time.time() - index_path.stat().st_mtime
        stale = age_seconds > args.stale_seconds

    if (index_payload is None or stale) and not args.no_refresh:
        refreshed = _refresh_index(root, index_path, args.tests_root)
        index_payload = _load_json(index_path)

    must_run_tests = _read_must_run_tests(
        Path(args.must_run_file) if args.must_run_file else None,
        args.must_run_test,
    )

    reasons: list[str] = []
    confidence = 0.0
    impacted_tests: list[str] = []
    must_run_impacted: list[str] = []
    if index_payload is None:
        reasons.append("index_missing")
    else:
        impacted_tests, changed_paths, must_run_impacted, confidence = _select_tests(
            index_payload,
            changed_lines=changed_lines,
            must_run_tests=must_run_tests,
        )

    if stale and not refreshed:
        reasons.append("index_stale")
    if confidence < args.confidence_threshold:
        reasons.append("low_confidence")

    mode = "targeted"
    if reasons and ("index_missing" in reasons or "index_stale" in reasons or "low_confidence" in reasons):
        mode = "full"

    impacted_docs = sorted(path for path in changed_paths if path.startswith("docs/") and path.endswith(".md"))

    payload = {
        "schema_version": 1,
        "mode": mode,
        "fallback_reasons": sorted(set(reasons)),
        "confidence": confidence,
        "confidence_threshold": args.confidence_threshold,
        "diff": {
            "base": args.diff_base,
            "head": args.diff_head,
            "changed_line_count": changed_count,
            "changed_paths": changed_paths,
        },
        "index": {
            "path": str(index_path.relative_to(root)),
            "present": index_payload is not None,
            "stale": stale,
            "refreshed": refreshed,
        },
        "selection": {
            "impacted_tests": impacted_tests,
            "must_run_impacted_tests": must_run_impacted,
            "impacted_docs": impacted_docs,
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

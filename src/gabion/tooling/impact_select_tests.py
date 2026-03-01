#!/usr/bin/env python3
# gabion:boundary_normalization_module
# gabion:decision_protocol_module
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Callable, Iterable, Mapping

from gabion.order_contract import sort_once
from gabion.tooling import diff_evidence_index

ChangedLine = diff_evidence_index.ChangedLine


def _parse_changed_lines(diff_text: str) -> list[ChangedLine]:
    return diff_evidence_index.parse_changed_lines(diff_text)


def _git_diff_changed_lines(root: Path, *, base: str | None, head: str | None) -> list[ChangedLine]:
    return diff_evidence_index.git_diff_changed_lines(root, base=base, head=head)


def _load_json(path: Path) -> Mapping[str, object] | None:
    return diff_evidence_index.load_json(path)


def _refresh_index(root: Path, index_path: Path, tests_root: str) -> bool:
    return diff_evidence_index.refresh_test_evidence_index(
        root,
        index_path=index_path,
        tests_root=tests_root,
    )


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
        return (
            [],
            sort_once(
                changed_paths,
                source="_select_tests.changed_paths",
            ),
            sort_once(
                changed_tests,
                source="_select_tests.changed_tests",
            ),
            0.0,
        )

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

    must_run_impacted = sort_once(
        (test for test in impacted if test in must_run_tests),
        source="_select_tests.must_run_impacted",
    )
    return (
        sort_once(
            impacted,
            source="_select_tests.impacted",
        ),
        sort_once(
            changed_paths,
            source="_select_tests.changed_paths_result",
        ),
        must_run_impacted,
        confidence,
    )


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
    parser.add_argument(
        "--changed-lines-artifact",
        default="artifacts/out/changed_lines.json",
    )
    parser.add_argument(
        "--evidence-meta-artifact",
        default="artifacts/out/evidence_index_meta.json",
    )
    parser.add_argument("--confidence-threshold", type=float, default=0.6)
    parser.add_argument("--stale-seconds", type=int, default=86_400)
    parser.add_argument("--must-run-file")
    parser.add_argument("--must-run-test", action="append", default=[])
    parser.add_argument("--no-refresh", action="store_true")
    args = parser.parse_args(argv)

    root = Path(args.root).resolve()
    index_path = root / args.index
    output_path = root / args.out
    changed_lines_artifact_path = root / args.changed_lines_artifact
    evidence_meta_artifact_path = root / args.evidence_meta_artifact

    diff_error: str | None = None
    changed_lines: list[ChangedLine]
    changed_paths: list[str]
    index_payload: Mapping[str, object] | None
    stale: bool
    refreshed: bool
    index_key = diff_evidence_index.diff_evidence_key(
        root=root,
        base=args.diff_base,
        head=args.diff_head,
        index_path=index_path,
    )
    if git_diff_changed_lines_fn is None:
        try:
            diff_index_result = diff_evidence_index.build_diff_evidence_index(
                root=root,
                base=args.diff_base,
                head=args.diff_head,
                index_path=index_path,
                tests_root=args.tests_root,
                stale_seconds=args.stale_seconds,
                no_refresh=bool(args.no_refresh),
            )
        except RuntimeError as exc:
            diff_error = str(exc) or "git diff failed"
            changed_lines = []
            changed_paths = []
            index_payload = diff_evidence_index.load_json(index_path)
            stale = index_payload is None
            refreshed = False
        else:
            changed_lines = diff_index_result.changed_lines
            changed_paths = diff_index_result.changed_paths
            index_payload = diff_index_result.index_payload
            stale = diff_index_result.stale
            refreshed = diff_index_result.refreshed
            index_key = dict(diff_index_result.key)
    else:
        try:
            changed_lines = git_diff_changed_lines_fn(root, args.diff_base, args.diff_head)
        except RuntimeError as exc:
            diff_error = str(exc) or "git diff failed"
            changed_lines = []
        changed_paths = sort_once(
            {item.path for item in changed_lines},
            source="main.changed_paths",
        )
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

    diff_evidence_index.write_diff_evidence_artifacts(
        changed_lines_path=changed_lines_artifact_path,
        meta_path=evidence_meta_artifact_path,
        changed_lines=changed_lines,
        key=index_key,
        stale=stale,
        refreshed=refreshed,
        index_path=index_path,
    )

    changed_count = len(changed_lines)

    must_run_tests = _read_must_run_tests(
        Path(args.must_run_file) if args.must_run_file else None,
        args.must_run_test,
    )

    reasons: list[str] = []
    if diff_error is not None:
        reasons.append("diff_unavailable")
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
    if reasons and (
        "diff_unavailable" in reasons
        or "index_missing" in reasons
        or "index_stale" in reasons
        or "low_confidence" in reasons
    ):
        mode = "full"

    impacted_docs = sort_once(
        (
            path
            for path in changed_paths
            if path.startswith("docs/") and path.endswith(".md")
        ),
        source="main.impacted_docs",
    )

    payload = {
        "schema_version": 1,
        "mode": mode,
        "fallback_reasons": sort_once(
            set(reasons),
            source="main.fallback_reasons",
        ),
        "confidence": confidence,
        "confidence_threshold": args.confidence_threshold,
        "diff": {
            "base": args.diff_base,
            "head": args.diff_head,
            "error": diff_error,
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
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    print(f"wrote {output_path}")
    return 0

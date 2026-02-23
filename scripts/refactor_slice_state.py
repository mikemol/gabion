#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_STATE_PATH = Path("artifacts/audit_reports/refactor_slice_state.json")
DEFAULT_COMPLEXITY_EMIT_PATH = Path("artifacts/audit_reports/complexity_baseline.json")

_REQUIRED_GOVERNANCE_DOCS: tuple[Path, ...] = (
    Path("AGENTS.md"),
    Path("POLICY_SEED.md"),
    Path("glossary.md"),
    Path("README.md"),
    Path("CONTRIBUTING.md"),
)

_DOC_REVISION_PATTERN = re.compile(r"^doc_revision:\s*(\d+)\s*$", re.MULTILINE)


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise SystemExit(f"missing JSON file: {path}") from exc
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SystemExit(f"failed to read JSON file {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise SystemExit(f"expected JSON object at {path}")
    return payload


def _load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"schema_version": 1, "active_slice": None, "history": []}
    payload = _load_json(path)
    if "schema_version" not in payload:
        payload["schema_version"] = 1
    if "active_slice" not in payload:
        payload["active_slice"] = None
    if "history" not in payload or not isinstance(payload["history"], list):
        payload["history"] = []
    return payload


def _write_state(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _doc_revision(text: str) -> int | None:
    match = _DOC_REVISION_PATTERN.search(text)
    if match is None:
        return None
    return int(match.group(1))


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _scan_comprehension_docs(root: Path) -> list[dict[str, Any]]:
    seen: set[Path] = set()
    ordered_paths: list[Path] = []

    for path in _REQUIRED_GOVERNANCE_DOCS:
        candidate = root / path
        resolved = candidate.resolve()
        if resolved not in seen:
            ordered_paths.append(candidate)
            seen.add(resolved)

    for pattern in ("docs/*.md", "in/in-*.md"):
        for path in sorted((root / ".").glob(pattern)):
            resolved = path.resolve()
            if resolved not in seen:
                ordered_paths.append(path)
                seen.add(resolved)

    docs: list[dict[str, Any]] = []
    for path in ordered_paths:
        rel_path = path.relative_to(root)
        if not path.exists():
            raise SystemExit(f"required comprehension doc missing: {rel_path}")
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            raise SystemExit(f"failed to read required doc {rel_path}: {exc}") from exc
        lines = text.splitlines()
        docs.append(
            {
                "path": str(rel_path),
                "doc_revision": _doc_revision(text),
                "sha256": _sha256_text(text),
                "line_count": len(lines),
            },
        )
    return docs


def _doc_snapshot_map(docs: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    snapshot: dict[str, dict[str, Any]] = {}
    for doc in docs:
        path = doc["path"]
        snapshot[path] = {
            "doc_revision": doc["doc_revision"],
            "sha256": doc["sha256"],
        }
    return snapshot


def _capture_complexity_metrics(root: Path) -> dict[str, Any]:
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        cmd = [
            "mise",
            "exec",
            "--",
            "python",
            "scripts/complexity_audit.py",
            "--root",
            str(root),
            "--emit",
            str(tmp_path),
        ]
        proc = subprocess.run(cmd, cwd=root, capture_output=True, text=True, check=False)
        if proc.returncode != 0:
            stderr = proc.stderr.strip() or proc.stdout.strip() or "unknown error"
            raise SystemExit(f"complexity capture failed: {stderr}")
        payload = _load_json(tmp_path)
        return _extract_complexity_metrics(payload)
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass


def _extract_complexity_metrics(payload: dict[str, Any]) -> dict[str, Any]:
    summary = payload.get("summary")
    if not isinstance(summary, dict):
        raise SystemExit("complexity payload missing summary")

    top_lines = payload.get("top_functions_by_lines")
    top_branches = payload.get("top_functions_by_branch_count")
    top_line_entry = top_lines[0] if isinstance(top_lines, list) and top_lines else {}
    top_branch_entry = top_branches[0] if isinstance(top_branches, list) and top_branches else {}

    return {
        "captured_at_utc": _now_utc(),
        "max_function_line_count": summary.get("max_function_line_count"),
        "max_function_branch_count": summary.get("max_function_branch_count"),
        "top4_test_case_total": summary.get("top4_test_case_total"),
        "private_ref_counts": summary.get("private_ref_counts"),
        "top_line_hotspot": {
            "file": top_line_entry.get("file"),
            "function": top_line_entry.get("function"),
            "line_count": top_line_entry.get("line_count"),
            "branch_count": top_line_entry.get("branch_count"),
        },
        "top_branch_hotspot": {
            "file": top_branch_entry.get("file"),
            "function": top_branch_entry.get("function"),
            "line_count": top_branch_entry.get("line_count"),
            "branch_count": top_branch_entry.get("branch_count"),
        },
    }


def _load_complexity_metrics(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise SystemExit(f"complexity payload missing: {path}")
    payload = _load_json(path)
    return _extract_complexity_metrics(payload)


def _ensure_nonempty(label: str, values: list[str]) -> None:
    if not values:
        raise SystemExit(f"missing required --{label} value")


def _assert_brief_is_current(
    *,
    root: Path,
    active_slice: dict[str, Any],
) -> tuple[list[str], list[dict[str, Any]]]:
    brief = active_slice.get("brief")
    if not isinstance(brief, dict):
        raise SystemExit("active slice brief is missing")
    snapshot = brief.get("doc_snapshot")
    if not isinstance(snapshot, dict) or not snapshot:
        raise SystemExit("active slice brief doc snapshot is missing")

    current_docs = _scan_comprehension_docs(root)
    current_snapshot = _doc_snapshot_map(current_docs)

    stale_reasons: list[str] = []
    expected_paths = set(snapshot.keys())
    current_paths = set(current_snapshot.keys())
    missing = sorted(expected_paths - current_paths)
    added = sorted(current_paths - expected_paths)
    if missing:
        stale_reasons.append(f"docs removed since brief: {', '.join(missing)}")
    if added:
        stale_reasons.append(f"new docs not captured by brief: {', '.join(added)}")

    for path in sorted(expected_paths & current_paths):
        expected = snapshot[path]
        current = current_snapshot[path]
        expected_revision = expected.get("doc_revision")
        current_revision = current.get("doc_revision")
        if (
            expected_revision is not None
            and current_revision is not None
            and expected_revision != current_revision
        ):
            stale_reasons.append(
                f"{path} doc_revision changed {expected_revision} -> {current_revision}",
            )
            continue
        if expected.get("sha256") != current.get("sha256"):
            stale_reasons.append(f"{path} content changed without matching brief refresh")

    return stale_reasons, current_docs


def _cmd_start(args: argparse.Namespace) -> int:
    root = args.root.resolve()
    state_path = args.state_path
    state = _load_state(state_path)

    if state.get("active_slice") is not None:
        raise SystemExit("an active slice already exists; complete or clear it before starting a new one")

    _ensure_nonempty("invariant", args.invariant)
    _ensure_nonempty("impacted-module", args.impacted_module)
    _ensure_nonempty("doc-to-update", args.doc_to_update)
    _ensure_nonempty("hotspot", args.hotspot)

    docs = _scan_comprehension_docs(root)
    if args.capture_complexity:
        pre_metrics = _capture_complexity_metrics(root)
    else:
        pre_metrics = _load_complexity_metrics(args.pre_metrics_json)

    active_slice = {
        "slice_id": args.slice_id,
        "subsystem": args.subsystem,
        "status": "in_progress",
        "started_at_utc": _now_utc(),
        "hotspots": args.hotspot,
        "next_target": args.next_target,
        "brief": {
            "invariants": args.invariant,
            "impacted_modules": args.impacted_module,
            "docs_to_update": args.doc_to_update,
            "comprehension_docs": docs,
            "doc_snapshot": _doc_snapshot_map(docs),
        },
        "metrics": {
            "pre": pre_metrics,
            "post": None,
        },
        "verification": {
            "status": "not_run",
            "steps": [],
        },
    }

    state["active_slice"] = active_slice
    state["updated_at_utc"] = _now_utc()
    _write_state(state_path, state)
    print(f"started slice {args.slice_id} -> {state_path}")
    return 0


def _cmd_assert_brief_current(args: argparse.Namespace) -> int:
    state = _load_state(args.state_path)
    active_slice = state.get("active_slice")
    if not isinstance(active_slice, dict):
        raise SystemExit("no active slice in state file")

    stale_reasons, _ = _assert_brief_is_current(root=args.root.resolve(), active_slice=active_slice)
    if stale_reasons:
        details = "\n".join(f"- {reason}" for reason in stale_reasons)
        raise SystemExit(f"slice brief is stale:\n{details}")
    print("active slice brief is current")
    return 0


def _cmd_finish(args: argparse.Namespace) -> int:
    root = args.root.resolve()
    state_path = args.state_path
    state = _load_state(state_path)
    active_slice = state.get("active_slice")
    if not isinstance(active_slice, dict):
        raise SystemExit("no active slice to finish")

    stale_reasons, current_docs = _assert_brief_is_current(root=root, active_slice=active_slice)
    if stale_reasons:
        details = "\n".join(f"- {reason}" for reason in stale_reasons)
        raise SystemExit(f"cannot finish slice; brief is stale:\n{details}")

    if args.capture_complexity:
        post_metrics = _capture_complexity_metrics(root)
    else:
        post_metrics = _load_complexity_metrics(args.post_metrics_json)

    finished = copy.deepcopy(active_slice)
    finished["status"] = args.status
    finished["completed_at_utc"] = _now_utc()
    finished["next_target"] = args.next_target
    if args.hotspot:
        finished["hotspots"] = args.hotspot
    finished["brief"]["comprehension_docs"] = current_docs
    finished["brief"]["doc_snapshot"] = _doc_snapshot_map(current_docs)
    finished["metrics"]["post"] = post_metrics
    finished["verification"] = {
        "status": args.verification_status,
        "steps": args.verification_step,
    }

    history = state.get("history")
    if not isinstance(history, list):
        history = []
    history.append(finished)
    state["history"] = history
    state["active_slice"] = None
    state["updated_at_utc"] = _now_utc()
    _write_state(state_path, state)
    print(f"finished slice {finished.get('slice_id')} -> {state_path}")
    return 0


def _cmd_refresh_brief(args: argparse.Namespace) -> int:
    root = args.root.resolve()
    state_path = args.state_path
    state = _load_state(state_path)
    active_slice = state.get("active_slice")
    if not isinstance(active_slice, dict):
        raise SystemExit("no active slice to refresh")

    brief = active_slice.get("brief")
    if not isinstance(brief, dict):
        raise SystemExit("active slice brief is missing")

    docs = _scan_comprehension_docs(root)
    brief["comprehension_docs"] = docs
    brief["doc_snapshot"] = _doc_snapshot_map(docs)
    if args.invariant:
        brief["invariants"] = args.invariant
    if args.impacted_module:
        brief["impacted_modules"] = args.impacted_module
    if args.doc_to_update:
        brief["docs_to_update"] = args.doc_to_update

    active_slice["brief"] = brief
    state["active_slice"] = active_slice
    state["updated_at_utc"] = _now_utc()
    _write_state(state_path, state)
    print(f"refreshed brief for slice {active_slice.get('slice_id')} -> {state_path}")
    return 0


def _cmd_status(args: argparse.Namespace) -> int:
    state = _load_state(args.state_path)
    active_slice = state.get("active_slice")
    if active_slice is None:
        print("active_slice: none")
        print(f"history_entries: {len(state.get('history', []))}")
        return 0
    print(f"active_slice: {active_slice.get('slice_id')} [{active_slice.get('status')}]")
    print(f"subsystem: {active_slice.get('subsystem')}")
    print(f"next_target: {active_slice.get('next_target')}")
    print(f"history_entries: {len(state.get('history', []))}")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Track functional-core refactor slice progress with mandatory comprehension briefs "
            "and doc-revision staleness checks."
        ),
    )
    parser.add_argument(
        "--state-path",
        type=Path,
        default=DEFAULT_STATE_PATH,
        help=f"State artifact path (default: {DEFAULT_STATE_PATH})",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("."),
        help="Repository root path used for doc and complexity scanning.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    start = subparsers.add_parser("start", help="Start a new refactor slice and record the brief.")
    start.add_argument("--slice-id", required=True, help="Unique slice identifier.")
    start.add_argument("--subsystem", required=True, help="Subsystem name for this slice.")
    start.add_argument("--hotspot", action="append", default=[], help="Current hotspot target.")
    start.add_argument("--next-target", required=True, help="Next hotspot or milestone.")
    start.add_argument("--invariant", action="append", default=[], help="Invariant that must hold.")
    start.add_argument(
        "--impacted-module",
        action="append",
        default=[],
        help="Module impacted by this slice.",
    )
    start.add_argument(
        "--doc-to-update",
        action="append",
        default=[],
        help="Doc expected to be updated in this slice.",
    )
    start.add_argument(
        "--capture-complexity",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Capture pre-slice complexity metrics with scripts/complexity_audit.py.",
    )
    start.add_argument(
        "--pre-metrics-json",
        type=Path,
        default=DEFAULT_COMPLEXITY_EMIT_PATH,
        help=(
            "Use an existing complexity payload when --no-capture-complexity is set "
            f"(default: {DEFAULT_COMPLEXITY_EMIT_PATH})."
        ),
    )
    start.set_defaults(func=_cmd_start)

    verify = subparsers.add_parser(
        "assert-brief-current",
        help="Fail if the active slice brief is missing or stale against current doc revisions.",
    )
    verify.set_defaults(func=_cmd_assert_brief_current)

    finish = subparsers.add_parser("finish", help="Finish the active slice and persist post metrics.")
    finish.add_argument(
        "--status",
        choices=("completed", "blocked", "failed"),
        default="completed",
        help="Terminal status for the active slice.",
    )
    finish.add_argument(
        "--verification-status",
        required=True,
        help="Verification outcome summary (for example: passed, partial, failed).",
    )
    finish.add_argument(
        "--verification-step",
        action="append",
        default=[],
        help="Verification command/result note.",
    )
    finish.add_argument("--next-target", required=True, help="Next hotspot after this slice.")
    finish.add_argument("--hotspot", action="append", default=[], help="Updated hotspot list.")
    finish.add_argument(
        "--capture-complexity",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Capture post-slice complexity metrics with scripts/complexity_audit.py.",
    )
    finish.add_argument(
        "--post-metrics-json",
        type=Path,
        default=DEFAULT_COMPLEXITY_EMIT_PATH,
        help=(
            "Use an existing complexity payload when --no-capture-complexity is set "
            f"(default: {DEFAULT_COMPLEXITY_EMIT_PATH})."
        ),
    )
    finish.set_defaults(func=_cmd_finish)

    refresh = subparsers.add_parser(
        "refresh-brief",
        help="Refresh active slice brief snapshot when required docs changed.",
    )
    refresh.add_argument(
        "--invariant",
        action="append",
        default=[],
        help="Optional replacement invariant list.",
    )
    refresh.add_argument(
        "--impacted-module",
        action="append",
        default=[],
        help="Optional replacement impacted module list.",
    )
    refresh.add_argument(
        "--doc-to-update",
        action="append",
        default=[],
        help="Optional replacement docs-to-update list.",
    )
    refresh.set_defaults(func=_cmd_refresh_brief)

    status = subparsers.add_parser("status", help="Print active slice summary.")
    status.set_defaults(func=_cmd_status)

    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())

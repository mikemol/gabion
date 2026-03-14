#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from datetime import date, timezone, datetime
from pathlib import Path
from typing import Any

from gabion.tooling.docflow.compliance_identity import stable_docflow_compliance_row_id

try:
    import yaml
except ImportError:  # pragma: no cover - yaml is required in repo runtime.
    yaml = None


_ACTIVE_COMPLIANCE_STATUSES = {"contradicts", "excess", "proposed"}
_ACTIVE_SECTION_REVIEW_STATUSES = {"stale_dep", "missing_review"}
_ANCHOR_PATH_RE = re.compile(
    r"`(?P<path>(?:src|tests|scripts)/[A-Za-z0-9_./-]+\.py)(?:::(?P<symbol>[^`]+))?`"
)

_DEFAULT_PROVING_TESTS = (
    "tests/gabion/tooling/docflow/test_docflow_warning_failures.py::test_docflow_violations_fail_when_fail_on_violations_enabled",
)

_PROVING_TESTS_BY_SIGNAL: dict[str, tuple[str, ...]] = {
    "docflow:missing_explicit_reference": (
        "tests/gabion/tooling/docflow/test_docflow_class_fixture_rows.py::test_dfx_mer_001_missing_explicit_reference_minimal",
        "tests/gabion/tooling/docflow/test_docflow_class_fixture_rows.py::test_dfx_mer_002_missing_explicit_reference_implicit_only",
    ),
    "docflow:invalid_field_type": (
        "tests/gabion/tooling/docflow/test_docflow_class_fixture_rows.py::test_dfx_ift_001_invalid_field_type_doc_reviewed_as_of_null",
        "tests/gabion/tooling/docflow/test_docflow_class_fixture_rows.py::test_dfx_ift_002_invalid_field_type_doc_review_notes_scalar",
    ),
    "docflow:missing_governance_ref": (
        "tests/gabion/tooling/docflow/test_docflow_class_fixture_rows.py::test_dfx_mgr_001_missing_governance_ref_normative_no_roots",
        "tests/gabion/tooling/docflow/test_docflow_class_fixture_rows.py::test_dfx_mgr_002_missing_governance_ref_partial_roots",
    ),
    "docflow:review_pin_mismatch": (
        "tests/gabion/tooling/docflow/test_docflow_violation_formatter.py::test_format_docflow_violation_doc_review_pin_branches",
    ),
    "section_review_missing_review": (
        "tests/gabion/tooling/docflow/test_docflow_warning_failures.py::test_docflow_warnings_fail_when_fail_on_violations_enabled",
    ),
    "section_review_stale_dep": (
        "tests/gabion/tooling/docflow/test_docflow_compliance_rows.py::test_docflow_compliance_rows_dispatches_cover_never_require_active_and_proposed",
    ),
}


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit(f"invalid JSON payload at {path}: expected object")
    return payload


def _active_compliance_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = payload.get("rows")
    if not isinstance(rows, list):
        return []
    selected: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        if row.get("row_kind") != "docflow_compliance":
            continue
        if row.get("status") not in _ACTIVE_COMPLIANCE_STATUSES:
            continue
        path = row.get("path")
        if not isinstance(path, str) or not path:
            continue
        selected.append(row)
    return selected


def _active_section_review_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = payload.get("rows")
    if not isinstance(rows, list):
        return []
    selected: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        if row.get("row_kind") != "doc_section_review":
            continue
        if row.get("status") not in _ACTIVE_SECTION_REVIEW_STATUSES:
            continue
        path = row.get("path")
        if not isinstance(path, str) or not path:
            continue
        selected.append(row)
    return selected


def _frontmatter_owner(root: Path, rel_path: str) -> str | None:
    if yaml is None:
        return None
    path = root / rel_path
    if not path.exists():
        return None
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None
    if not text.startswith("---\n"):
        return None
    marker = "\n---\n"
    end_index = text.find(marker, 4)
    if end_index < 0:
        return None
    raw = text[4:end_index]
    loaded = yaml.safe_load(raw)
    if not isinstance(loaded, dict):
        return None
    owner = loaded.get("doc_owner")
    return str(owner).strip() if owner else None


def _anchor_suggestions(root: Path, missing_anchor_path: str) -> list[str]:
    basename = Path(missing_anchor_path).name
    candidates: list[str] = []
    for prefix in ("src", "tests", "scripts"):
        base = root / prefix
        if not base.exists():
            continue
        for path in sorted(base.rglob(basename)):
            candidates.append(path.relative_to(root).as_posix())
    return candidates[:5]


def _scan_stale_anchors(root: Path, rel_path: str) -> list[dict[str, Any]]:
    path = root / rel_path
    if not path.exists():
        return []
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return []

    stale: list[dict[str, Any]] = []
    seen: set[str] = set()
    for match in _ANCHOR_PATH_RE.finditer(text):
        anchor_path = match.group("path")
        if anchor_path in seen:
            continue
        seen.add(anchor_path)
        if (root / anchor_path).exists():
            continue
        stale.append(
            {
                "anchor_path": anchor_path,
                "symbol": match.group("symbol"),
                "suggestions": _anchor_suggestions(root, anchor_path),
            }
        )
    return stale


def _looks_version_domain_mismatch(row: dict[str, Any]) -> bool:
    dep_version = row.get("dep_version")
    expected = row.get("expected_dep_version")
    if not isinstance(dep_version, int) or not isinstance(expected, int):
        return False
    return dep_version > expected + 5 or (dep_version >= 10 and expected <= 4)


def _packet_classification(
    *,
    rows: list[dict[str, Any]],
    stale_anchors: list[dict[str, Any]],
) -> str:
    compliance_rows = [row for row in rows if row.get("row_kind") == "docflow_compliance"]
    section_rows = [row for row in rows if row.get("row_kind") == "doc_section_review"]
    section_statuses = {str(row.get("status")) for row in section_rows}
    invariants = {str(row.get("invariant")) for row in compliance_rows}

    if "missing_review" in section_statuses:
        return "needs_semantic_update"
    if stale_anchors:
        return "needs_semantic_update"
    if compliance_rows:
        if invariants == {"docflow:invalid_field_type"}:
            return "metadata_only"
        return "needs_semantic_update"
    if section_rows and section_statuses == {"stale_dep"}:
        if all(_looks_version_domain_mismatch(row) for row in section_rows):
            return "metadata_only"
        return "materially_still_true"
    return "materially_still_true"


def _packet_signals(rows: list[dict[str, Any]]) -> list[str]:
    signals: list[str] = []
    for row in rows:
        if row.get("row_kind") == "docflow_compliance":
            invariant = row.get("invariant")
            if isinstance(invariant, str) and invariant:
                signals.append(invariant)
            continue
        if row.get("row_kind") == "doc_section_review":
            status = row.get("status")
            if isinstance(status, str) and status:
                signals.append(f"section_review_{status}")
    return sorted(set(signals))


def _packet_proving_tests(rows: list[dict[str, Any]]) -> list[str]:
    selected: list[str] = list(_DEFAULT_PROVING_TESTS)
    for signal in _packet_signals(rows):
        selected.extend(_PROVING_TESTS_BY_SIGNAL.get(signal, ()))
    return sorted(set(selected))


def run(
    *,
    root: Path,
    compliance_path: Path,
    section_reviews_path: Path,
    out_path: Path,
    summary_out_path: Path,
) -> int:
    compliance_payload = _load_json(compliance_path)
    section_payload = _load_json(section_reviews_path)

    active_rows: list[dict[str, Any]] = []
    active_rows.extend(_active_compliance_rows(compliance_payload))
    active_rows.extend(_active_section_review_rows(section_payload))

    by_path: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in active_rows:
        path = row.get("path")
        if isinstance(path, str) and path:
            by_path[path].append(row)

    packets: list[dict[str, Any]] = []
    for rel_path in sorted(by_path):
        rows = sorted(
            by_path[rel_path],
            key=lambda row: (
                str(row.get("row_kind", "")),
                str(row.get("invariant", "")),
                str(row.get("status", "")),
                str(row.get("dep", "")),
                str(row.get("anchor", "")),
            ),
        )
        stale_anchors = _scan_stale_anchors(root, rel_path)
        classification = _packet_classification(rows=rows, stale_anchors=stale_anchors)
        packet_rows = []
        for row in rows:
            row_id = stable_docflow_compliance_row_id(row)
            packet_rows.append(
                {
                    "row_id": row_id,
                    "row_kind": row.get("row_kind"),
                    "status": row.get("status"),
                    "invariant": row.get("invariant"),
                    "source_row_kind": row.get("source_row_kind"),
                    "dep": row.get("dep"),
                    "anchor": row.get("anchor"),
                    "qual": row.get("qual"),
                    "dep_version": row.get("dep_version"),
                    "expected_dep_version": row.get("expected_dep_version"),
                }
            )
        packets.append(
            {
                "path": rel_path,
                "doc_owner": _frontmatter_owner(root, rel_path),
                "classification": classification,
                "touch_set": [rel_path],
                "signals": _packet_signals(rows),
                "proving_tests": _packet_proving_tests(rows),
                "rows": packet_rows,
                "stale_anchor_hints": stale_anchors,
                "auto_remediation": {
                    "eligible": classification == "metadata_only",
                    "mode": "frontmatter_section_pin_normalization_only"
                    if classification == "metadata_only"
                    else "human_semantic_review_required",
                },
            }
        )

    class_counts = Counter(packet["classification"] for packet in packets)
    report = {
        "scope": {
            "kind": "docflow_packetization",
            "as_of": datetime.now(timezone.utc).isoformat(),
            "source_artifacts": {
                "compliance": str(compliance_path),
                "section_reviews": str(section_reviews_path),
            },
        },
        "summary": {
            "active_rows": len(active_rows),
            "active_packets": len(packets),
            "classifications": dict(sorted(class_counts.items())),
        },
        "packets": packets,
    }

    summary = {
        "counts": {
            "active_rows": len(active_rows),
            "active_packets": len(packets),
            "classifications": dict(sorted(class_counts.items())),
        },
        "scope": {
            "as_of": date.today().isoformat(),
            "kind": "docflow_packetization",
            "source_artifacts": {
                "compliance": str(compliance_path),
                "section_reviews": str(section_reviews_path),
            },
        },
        "paths": [packet["path"] for packet in packets],
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary_out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary_out_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        "docflow-packetize: "
        f"rows={len(active_rows)} packets={len(packets)} out={out_path}"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--compliance", default="artifacts/out/docflow_compliance.json")
    parser.add_argument(
        "--section-reviews",
        default="artifacts/out/docflow_section_reviews.json",
    )
    parser.add_argument("--out", default="artifacts/out/docflow_warning_doc_packets.json")
    parser.add_argument(
        "--summary-out",
        default="artifacts/out/docflow_warning_doc_packet_summary.json",
    )
    args = parser.parse_args(argv)

    return run(
        root=Path(args.root).resolve(),
        compliance_path=Path(args.compliance).resolve(),
        section_reviews_path=Path(args.section_reviews).resolve(),
        out_path=Path(args.out).resolve(),
        summary_out_path=Path(args.summary_out).resolve(),
    )


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

from gabion.analysis import evidence_keys, test_evidence
from gabion.analysis.report_doc import ReportDoc
from gabion.analysis.timeout_context import check_deadline
from gabion.json_types import JSONValue

SEMANTIC_COVERAGE_MAP_VERSION = 1


@dataclass(frozen=True)
class SemanticCoverageEntry:
    obligation: str
    obligation_kind: str
    evidence_display: str

    @property
    def evidence_identity(self) -> str:
        key = evidence_keys.parse_display(self.evidence_display)
        if key is None:
            key = evidence_keys.make_opaque_key(self.evidence_display)
        return evidence_keys.key_identity(evidence_keys.normalize_key(key))


def build_semantic_coverage_payload(
    paths: Iterable[Path],
    *,
    root: Path,
    mapping_path: Path,
    evidence_path: Path,
    include: Iterable[str] | None = None,
    exclude: Iterable[str] | None = None,
) -> dict[str, JSONValue]:
    check_deadline()
    entries = load_mapping_entries(mapping_path)
    tags = test_evidence.collect_test_tags(
        paths,
        root=root,
        include=include,
        exclude=exclude,
    )
    annotation_index = _annotation_index(tags)
    artifact_index = _artifact_evidence_index(evidence_path)
    duplicate_entries = _duplicate_mapping_entries(entries)

    coverage_rows: list[dict[str, JSONValue]] = []
    by_obligation: dict[str, list[dict[str, JSONValue]]] = {}
    for entry in sorted(
        entries,
        key=lambda item: (
            item.obligation,
            item.obligation_kind,
            item.evidence_identity,
            item.evidence_display,
        ),
    ):
        check_deadline()
        evidence_id = entry.evidence_identity
        mapped_tests = sorted(annotation_index.get(evidence_id, []))
        present_in_artifact = evidence_id in artifact_index
        dead = not mapped_tests and not present_in_artifact
        row = {
            "obligation": entry.obligation,
            "obligation_kind": entry.obligation_kind,
            "evidence": entry.evidence_display,
            "evidence_identity": evidence_id,
            "mapped_tests": mapped_tests,
            "mapped": bool(mapped_tests),
            "dead": dead,
            "artifact_present": present_in_artifact,
        }
        coverage_rows.append(row)
        by_obligation.setdefault(entry.obligation, []).append(row)

    mapped_obligations: list[dict[str, JSONValue]] = []
    unmapped_obligations: list[dict[str, JSONValue]] = []
    for obligation in sorted(by_obligation):
        check_deadline()
        rows = by_obligation[obligation]
        kind = str(rows[0].get("obligation_kind", "invariant"))
        mapped = any(bool(row.get("mapped", False)) for row in rows)
        payload = {
            "obligation": obligation,
            "obligation_kind": kind,
            "evidence_count": len(rows),
            "mapped_test_count": sum(len(row.get("mapped_tests", [])) for row in rows),
        }
        if mapped:
            mapped_obligations.append(payload)
        else:
            unmapped_obligations.append(payload)

    dead_rows = [row for row in coverage_rows if bool(row.get("dead", False))]
    payload: dict[str, JSONValue] = {
        "version": SEMANTIC_COVERAGE_MAP_VERSION,
        "summary": {
            "mapping_entries": len(coverage_rows),
            "mapped_obligations": len(mapped_obligations),
            "unmapped_obligations": len(unmapped_obligations),
            "dead_mapping_entries": len(dead_rows),
            "duplicate_mapping_entries": len(duplicate_entries),
        },
        "mapped_obligations": mapped_obligations,
        "unmapped_obligations": unmapped_obligations,
        "dead_mapping_entries": dead_rows,
        "duplicate_mapping_entries": duplicate_entries,
        "mapping_entries": coverage_rows,
    }
    return payload


def render_markdown(payload: Mapping[str, JSONValue]) -> str:
    check_deadline()
    doc = ReportDoc("out_semantic_coverage_map")
    doc.section("Summary")
    doc.codeblock(payload.get("summary", {}))
    doc.line()
    mapped = payload.get("mapped_obligations", [])
    unmapped = payload.get("unmapped_obligations", [])
    dead = payload.get("dead_mapping_entries", [])
    duplicates = payload.get("duplicate_mapping_entries", [])
    if isinstance(mapped, list) and mapped:
        doc.line("Mapped obligations:")
        doc.codeblock(mapped)
        doc.line()
    if isinstance(unmapped, list) and unmapped:
        doc.line("Unmapped obligations:")
        doc.codeblock(unmapped)
        doc.line()
    if isinstance(dead, list) and dead:
        doc.line("Dead mapping entries:")
        doc.codeblock(dead)
        doc.line()
    if isinstance(duplicates, list) and duplicates:
        doc.line("Duplicate mapping entries:")
        doc.codeblock(duplicates)
    return doc.emit()


def write_semantic_coverage(
    payload: Mapping[str, JSONValue],
    *,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_mapping_entries(path: Path) -> list[SemanticCoverageEntry]:
    check_deadline()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(raw, Mapping):
        return []
    entries = raw.get("entries", [])
    if not isinstance(entries, list):
        return []
    parsed: list[SemanticCoverageEntry] = []
    for item in entries:
        check_deadline()
        if not isinstance(item, Mapping):
            continue
        obligation = str(item.get("obligation", "")).strip()
        evidence = str(item.get("evidence", "")).strip()
        if not obligation or not evidence:
            continue
        parsed.append(
            SemanticCoverageEntry(
                obligation=obligation,
                obligation_kind=str(item.get("obligation_kind", "invariant")).strip()
                or "invariant",
                evidence_display=evidence,
            )
        )
    return parsed


def _annotation_index(
    tags: Iterable[test_evidence.TestEvidenceTag],
) -> dict[str, set[str]]:
    check_deadline()
    index: dict[str, set[str]] = {}
    for entry in tags:
        check_deadline()
        for raw in entry.tags:
            display = str(raw).strip()
            if not display:
                continue
            key = evidence_keys.parse_display(display)
            if key is None:
                key = evidence_keys.make_opaque_key(display)
            identity = evidence_keys.key_identity(evidence_keys.normalize_key(key))
            index.setdefault(identity, set()).add(entry.test_id)
    return index


def _artifact_evidence_index(path: Path) -> set[str]:
    check_deadline()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return set()
    if not isinstance(raw, Mapping):
        return set()
    records = raw.get("evidence_index", [])
    if not isinstance(records, list):
        return set()
    identities: set[str] = set()
    for record in records:
        check_deadline()
        if not isinstance(record, Mapping):
            continue
        key = record.get("key")
        if isinstance(key, Mapping):
            identities.add(evidence_keys.key_identity(evidence_keys.normalize_key(key)))
            continue
        display = str(record.get("display", "")).strip()
        if not display:
            continue
        parsed = evidence_keys.parse_display(display)
        if parsed is None:
            parsed = evidence_keys.make_opaque_key(display)
        identities.add(evidence_keys.key_identity(evidence_keys.normalize_key(parsed)))
    return identities


def _duplicate_mapping_entries(
    entries: Iterable[SemanticCoverageEntry],
) -> list[dict[str, JSONValue]]:
    check_deadline()
    counts = Counter(
        (entry.obligation, entry.obligation_kind, entry.evidence_identity)
        for entry in entries
    )
    duplicates: list[dict[str, JSONValue]] = []
    for (obligation, kind, evidence_identity), count in sorted(counts.items()):
        check_deadline()
        if count < 2:
            continue
        duplicates.append(
            {
                "obligation": obligation,
                "obligation_kind": kind,
                "evidence_identity": evidence_identity,
                "count": count,
            }
        )
    return duplicates


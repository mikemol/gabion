# gabion:boundary_normalization_module
# gabion:decision_protocol_module
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

from gabion.analysis import evidence_keys, test_evidence, test_obsolescence
from gabion.analysis.baseline_io import write_json
from gabion.analysis.projection_exec import apply_spec
from gabion.analysis.projection_registry import (
    TEST_ANNOTATION_DRIFT_SPEC,
    spec_metadata_lines_from_payload,
    spec_metadata_payload,
)
from gabion.analysis.report_doc import ReportDoc
from gabion.json_types import JSONValue
from gabion.analysis.timeout_context import check_deadline

DRIFT_VERSION = 1


@dataclass(frozen=True)
class AnnotationDriftEntry:
    test_id: str
    tag: str
    status: str
    reason: str


def build_annotation_drift_payload(
    paths: Iterable[Path],
    *,
    root: Path,
    evidence_path: Path,
    include: Iterable[str] | None = None,
    exclude: Iterable[str] | None = None,
) -> dict[str, JSONValue]:
    check_deadline()
    # dataflow-bundle: evidence_path, exclude, include, root
    evidence_by_test, _ = test_obsolescence.load_test_evidence(str(evidence_path))
    universe = {ref.identity for refs in evidence_by_test.values() for ref in refs}
    display_index: dict[str, set[str]] = {}
    for refs in evidence_by_test.values():
        for ref in refs:
            display_index.setdefault(ref.display, set()).add(ref.identity)
    tag_entries = test_evidence.collect_test_tags(
        paths,
        root=root,
        include=include,
        exclude=exclude,
    )
    relation: list[dict[str, JSONValue]] = []
    for entry in tag_entries:
        for raw_tag in entry.tags:
            tag = str(raw_tag).strip()
            status, reason = _classify_tag(tag, universe, display_index)
            relation.append(
                {
                    "test_id": entry.test_id,
                    "tag": tag,
                    "status": status,
                    "reason": reason,
                }
            )

    projected = apply_spec(TEST_ANNOTATION_DRIFT_SPEC, relation)
    summary = _summarize(projected)
    payload: dict[str, JSONValue] = {
        "version": DRIFT_VERSION,
        "summary": summary,
        "entries": projected,
    }
    payload.update(spec_metadata_payload(TEST_ANNOTATION_DRIFT_SPEC))
    return payload


def render_markdown(payload: Mapping[str, JSONValue]) -> str:
    check_deadline()
    summary = payload.get("summary", {})
    entries = payload.get("entries", [])
    doc = ReportDoc("out_test_annotation_drift")
    doc.lines(spec_metadata_lines_from_payload(payload))
    doc.section("Summary")
    doc.codeblock(summary)
    doc.line()
    orphaned: list[Mapping[str, JSONValue]] = []
    legacy: list[Mapping[str, JSONValue]] = []
    legacy_ambiguous: list[Mapping[str, JSONValue]] = []
    if isinstance(entries, list):
        for entry in entries:
            if not isinstance(entry, Mapping):
                continue
            status = str(entry.get("status", ""))
            if status == "orphaned":
                orphaned.append(entry)
            elif status == "legacy_tag":
                legacy.append(entry)
            elif status == "legacy_ambiguous":
                legacy_ambiguous.append(entry)
    if orphaned:
        doc.line("Orphaned tags:")
        rows: list[str] = []
        for entry in orphaned:
            test_id = str(entry.get("test_id", "") or "")
            tag = str(entry.get("tag", "") or "")
            reason = str(entry.get("reason", "") or "")
            rows.append(f"{test_id} tag={tag} reason={reason}")
        doc.codeblock("\n".join(rows))
    if legacy:
        doc.line()
        doc.line("Legacy tags:")
        rows = []
        for entry in legacy:
            test_id = str(entry.get("test_id", "") or "")
            tag = str(entry.get("tag", "") or "")
            reason = str(entry.get("reason", "") or "")
            rows.append(f"{test_id} tag={tag} reason={reason}")
        doc.codeblock("\n".join(rows))
    if legacy_ambiguous:
        doc.line()
        doc.line("Ambiguous legacy tags:")
        rows = []
        for entry in legacy_ambiguous:
            test_id = str(entry.get("test_id", "") or "")
            tag = str(entry.get("tag", "") or "")
            reason = str(entry.get("reason", "") or "")
            rows.append(f"{test_id} tag={tag} reason={reason}")
        doc.codeblock("\n".join(rows))
    return doc.emit()


def write_annotation_drift(
    payload: Mapping[str, JSONValue],
    *,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(output_path, payload)


def _summarize(entries: Iterable[Mapping[str, JSONValue]]) -> dict[str, int]:
    check_deadline()
    summary = {
        "legacy_ambiguous": 0,
        "legacy_tag": 0,
        "ok": 0,
        "orphaned": 0,
    }
    for entry in entries:
        status = str(entry.get("status", ""))
        if status not in summary:
            summary[status] = 0
        summary[status] += 1
    return summary


def _classify_tag(
    tag: str,
    universe: set[str],
    display_index: Mapping[str, set[str]],
) -> tuple[str, str]:
    key = evidence_keys.parse_display(tag)
    if key is not None:
        normalized = evidence_keys.normalize_key(key)
        identity = evidence_keys.key_identity(normalized)
        if identity not in universe:
            return "orphaned", "missing_from_evidence_index"
        return "ok", ""
    if tag.startswith("E:"):
        identities = display_index.get(tag, set())
        if not identities:
            return "orphaned", "legacy_missing"
        if len(identities) > 1:
            return "legacy_ambiguous", "legacy_display_ambiguous"
        return "legacy_tag", "legacy_display"
    return "orphaned", "unparseable_tag"

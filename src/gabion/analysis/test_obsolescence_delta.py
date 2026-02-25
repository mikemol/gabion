# gabion:boundary_normalization_module
# gabion:decision_protocol_module
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Mapping

from gabion.analysis import evidence_keys, test_obsolescence
from gabion.analysis.baseline_io import (
    attach_spec_metadata,
    load_json,
    parse_spec_metadata,
    parse_version,
    write_json,
)
from gabion.analysis.delta_tools import TransitionPair
from gabion.analysis.delta_tools import coerce_int, format_delta
from gabion.analysis.delta_tools import format_transition
from gabion.analysis.projection_registry import (
    TEST_OBSOLESCENCE_BASELINE_SPEC,
    TEST_OBSOLESCENCE_DELTA_SPEC,
    spec_metadata_lines_from_payload,
)
from gabion.analysis.report_doc import ReportDoc
from gabion.json_types import JSONValue
from gabion.analysis.timeout_context import check_deadline
from gabion.order_contract import sort_once

BASELINE_VERSION = 2
DELTA_VERSION = 1
BASELINE_RELATIVE_PATH = Path("baselines/test_obsolescence_baseline.json")


@dataclass(frozen=True)
class EvidenceIndexEntry:
    key: dict[str, object]
    identity: str
    display: str
    witness_count: int


@dataclass(frozen=True)
class ObsolescenceBaseline:
    summary: dict[str, int]
    tests: dict[str, str]
    active: dict[str, JSONValue]
    evidence_index: dict[str, EvidenceIndexEntry]
    opaque_evidence_count: int
    generated_by_spec_id: str
    generated_by_spec: dict[str, JSONValue]


def resolve_baseline_path(root: Path) -> Path:
    return root / BASELINE_RELATIVE_PATH


# gabion:ambiguity_boundary
def build_baseline_payload(
    evidence_by_test: Mapping[str, Iterable[object]],
    status_by_test: Mapping[str, str],
    candidates: Iterable[Mapping[str, object]],
    summary_counts: Mapping[str, int],
    *,
    active_tests: Iterable[str] | None = None,
    active_summary: Mapping[str, int] | None = None,
) -> dict[str, JSONValue]:
    # dataflow-bundle: evidence_by_test, status_by_test, candidates, summary_counts
    summary = _normalize_summary_counts(summary_counts)
    tests = _tests_from_candidates(candidates)
    evidence_index_entries = _build_evidence_index(evidence_by_test, status_by_test)
    payload: dict[str, JSONValue] = {
        "version": BASELINE_VERSION,
        "summary": summary,
        "tests": tests,
        "evidence_index": evidence_index_entries,
        "opaque_evidence_count": _count_opaque_evidence(evidence_by_test),
    }
    active = _normalize_active_metadata(
        {"tests": list(active_tests or []), "summary": dict(active_summary or {})}
    )
    if active:
        payload["active"] = active
    return attach_spec_metadata(payload, spec=TEST_OBSOLESCENCE_BASELINE_SPEC)


# gabion:ambiguity_boundary
def parse_baseline_payload(payload: Mapping[str, JSONValue]) -> ObsolescenceBaseline:
    check_deadline()
    parse_version(
        payload,
        expected=(1, BASELINE_VERSION),
        error_context="test obsolescence baseline",
    )
    summary = _normalize_summary_counts(payload.get("summary", {}))
    tests: dict[str, str] = {}
    tests_payload = payload.get("tests", [])
    if isinstance(tests_payload, list):
        for entry in tests_payload:
            if not isinstance(entry, Mapping):
                continue
            test_id = str(entry.get("test_id", "") or "").strip()
            class_name = str(entry.get("class", "") or "").strip()
            if not test_id or not class_name:
                continue
            tests[test_id] = class_name
    active = _normalize_active_metadata(payload.get("active", {}))
    evidence_index = _parse_evidence_index(payload.get("evidence_index", []))
    opaque_count = coerce_int(payload.get("opaque_evidence_count"), 0)
    spec_id, spec = parse_spec_metadata(payload)
    return ObsolescenceBaseline(
        summary=summary,
        tests=tests,
        active=active,
        evidence_index=evidence_index,
        opaque_evidence_count=opaque_count,
        generated_by_spec_id=spec_id,
        generated_by_spec=spec,
    )


def load_baseline(path: str) -> ObsolescenceBaseline:
    return parse_baseline_payload(load_json(path))


def write_baseline(path: str, payload: Mapping[str, JSONValue]) -> None:
    write_json(path, payload)


# gabion:ambiguity_boundary
def build_delta_payload(
    baseline: ObsolescenceBaseline,
    current: ObsolescenceBaseline,
    *,
    baseline_path: str | None = None,
) -> dict[str, JSONValue]:
    # dataflow-bundle: baseline, current
    class_keys = _class_keys()
    baseline_summary = _normalize_summary_counts(baseline.summary)
    current_summary = _normalize_summary_counts(current.summary)
    summary_delta = {
        key: current_summary.get(key, 0) - baseline_summary.get(key, 0)
        for key in class_keys
    }

    baseline_tests = baseline.tests
    current_tests = current.tests
    added_tests = sort_once(
        set(current_tests) - set(baseline_tests),
        source="build_delta_payload.added_tests",
    )
    removed_tests = sort_once(
        set(baseline_tests) - set(current_tests),
        source="build_delta_payload.removed_tests",
    )
    changed_tests = sort_once(
        {
            test_id
            for test_id in set(baseline_tests) & set(current_tests)
            if baseline_tests[test_id] != current_tests[test_id]
        },
        source="build_delta_payload.changed_tests",
    )

    baseline_evidence = baseline.evidence_index
    current_evidence = current.evidence_index
    added_evidence_ids = sort_once(
        set(current_evidence) - set(baseline_evidence),
        source="build_delta_payload.added_evidence_ids",
    )
    removed_evidence_ids = sort_once(
        set(baseline_evidence) - set(current_evidence),
        source="build_delta_payload.removed_evidence_ids",
    )
    changed_evidence_ids = sort_once(
        {
            evidence_id
            for evidence_id in set(baseline_evidence) & set(current_evidence)
            if baseline_evidence[evidence_id].witness_count
            != current_evidence[evidence_id].witness_count
        },
        source="build_delta_payload.changed_evidence_ids",
    )

    tests_section = {
        "added": [
            {"test_id": test_id, "class": current_tests[test_id]}
            for test_id in added_tests
        ],
        "removed": [
            {"test_id": test_id, "class": baseline_tests[test_id]}
            for test_id in removed_tests
        ],
        "changed_class": [
            {
                "test_id": test_id,
                "before": baseline_tests[test_id],
                "after": current_tests[test_id],
            }
            for test_id in changed_tests
        ],
    }

    evidence_section = {
        "added": [
            _evidence_entry_payload(current_evidence[evidence_id])
            for evidence_id in added_evidence_ids
        ],
        "removed": [
            _evidence_entry_payload(baseline_evidence[evidence_id])
            for evidence_id in removed_evidence_ids
        ],
        "changed": [
            _evidence_change_payload(
                baseline_evidence[evidence_id],
                current_evidence[evidence_id],
            )
            for evidence_id in changed_evidence_ids
        ],
    }

    summary = {
        "counts": {
            "baseline": baseline_summary,
            "current": current_summary,
            "delta": summary_delta,
        },
        "tests": {
            "added": len(added_tests),
            "removed": len(removed_tests),
            "changed_class": len(changed_tests),
        },
        "evidence_keys": {
            "added": len(added_evidence_ids),
            "removed": len(removed_evidence_ids),
            "changed_witness_count": len(changed_evidence_ids),
        },
        "opaque_evidence": {
            "baseline": baseline.opaque_evidence_count,
            "current": current.opaque_evidence_count,
            "delta": current.opaque_evidence_count - baseline.opaque_evidence_count,
        },
    }

    payload: dict[str, JSONValue] = {
        "version": DELTA_VERSION,
        "baseline": _baseline_meta_payload(baseline, baseline_path),
        "current": _current_meta_payload(current),
        "summary": summary,
        "tests": tests_section,
        "evidence_keys": evidence_section,
    }
    return attach_spec_metadata(payload, spec=TEST_OBSOLESCENCE_DELTA_SPEC)


# gabion:ambiguity_boundary
def render_markdown(delta_payload: Mapping[str, JSONValue]) -> str:
    check_deadline()
    # dataflow-bundle: delta_payload
    summary = delta_payload.get("summary", {})
    counts = {}
    if isinstance(summary, Mapping):
        counts = summary.get("counts", {}) if isinstance(summary.get("counts"), Mapping) else {}
    baseline_counts = (
        counts.get("baseline", {}) if isinstance(counts.get("baseline"), Mapping) else {}
    )
    current_counts = (
        counts.get("current", {}) if isinstance(counts.get("current"), Mapping) else {}
    )
    delta_counts = (
        counts.get("delta", {}) if isinstance(counts.get("delta"), Mapping) else {}
    )
    opaque = {}
    if isinstance(summary, Mapping):
        opaque = summary.get("opaque_evidence", {}) if isinstance(summary.get("opaque_evidence"), Mapping) else {}
    tests_section = delta_payload.get("tests", {})
    evidence_section = delta_payload.get("evidence_keys", {})

    doc = ReportDoc("out_test_obsolescence_delta")
    doc.lines(spec_metadata_lines_from_payload(delta_payload))
    doc.section("Summary")

    baseline_meta = delta_payload.get("baseline", {})
    if isinstance(baseline_meta, Mapping):
        baseline_path = baseline_meta.get("path")
        if isinstance(baseline_path, str) and baseline_path:
            doc.line(f"- baseline: {baseline_path}")
        baseline_spec = baseline_meta.get("generated_by_spec_id")
        if isinstance(baseline_spec, str) and baseline_spec:
            doc.line(f"- baseline_spec_id: {baseline_spec}")
    current_meta = delta_payload.get("current", {})
    if isinstance(current_meta, Mapping):
        current_spec = current_meta.get("generated_by_spec_id")
        if isinstance(current_spec, str) and current_spec:
            doc.line(f"- current_spec_id: {current_spec}")

    for key in _class_keys():
        pair = TransitionPair(
            baseline=baseline_counts.get(key, 0),
            current=current_counts.get(key, 0),
        )
        doc.line(
            f"- {key}: {format_transition(pair, delta_counts.get(key, 0))}"
        )
    opaque_pair = TransitionPair(
        baseline=opaque.get("baseline", 0),
        current=opaque.get("current", 0),
    )
    opaque_line = format_transition(
        opaque_pair,
        opaque.get("delta", 0),
    )
    doc.line(f"- opaque_evidence_count: {opaque_line}")

    doc.line()
    doc.line("## Tests")
    _render_test_section(doc, "Added", _section_list(tests_section, "added"))
    _render_test_section(doc, "Removed", _section_list(tests_section, "removed"))
    _render_test_changes(doc, _section_list(tests_section, "changed_class"))

    doc.line()
    doc.line("## Evidence Keys")
    _render_evidence_section(doc, "Added", _section_list(evidence_section, "added"))
    _render_evidence_section(
        doc, "Removed", _section_list(evidence_section, "removed")
    )
    _render_evidence_changes(doc, _section_list(evidence_section, "changed"))

    return doc.emit()


# gabion:ambiguity_boundary
def build_baseline_payload_from_paths(
    evidence_path: str,
    risk_registry_path: str,
    *,
    load_test_evidence_fn: Callable[[str], tuple[Mapping[str, Iterable[object]], Mapping[str, str]]]
    | None = None,
    load_risk_registry_fn: Callable[[str], Mapping[str, object]] | None = None,
    classify_candidates_fn: Callable[
        [Mapping[str, Iterable[object]], Mapping[str, str], Mapping[str, object]],
        test_obsolescence.ClassificationResult,
    ]
    | None = None,
    build_baseline_payload_fn: Callable[..., dict[str, JSONValue]] | None = None,
) -> dict[str, JSONValue]:
    load_test_evidence = load_test_evidence_fn or test_obsolescence.load_test_evidence
    load_risk_registry = load_risk_registry_fn or test_obsolescence.load_risk_registry
    classify_candidates = classify_candidates_fn or test_obsolescence.classify_candidates
    build_baseline_payload_impl = build_baseline_payload_fn or build_baseline_payload

    evidence_by_test, status_by_test = load_test_evidence(evidence_path)
    risk_registry = load_risk_registry(risk_registry_path)
    classification = classify_candidates(
        evidence_by_test, status_by_test, risk_registry
    )
    return build_baseline_payload_impl(
        evidence_by_test,
        status_by_test,
        classification.stale_candidates,
        classification.stale_summary,
        active_tests=classification.active_tests,
        active_summary=classification.active_summary,
    )


# gabion:ambiguity_boundary
def _normalize_summary_counts(summary: Mapping[str, object] | object) -> dict[str, int]:
    check_deadline()
    result = {key: 0 for key in _class_keys()}
    if not isinstance(summary, Mapping):
        return result
    for key in result:
        result[key] = coerce_int(summary.get(key), 0)
    return result


# gabion:ambiguity_boundary
def _normalize_active_metadata(active: Mapping[str, object] | object) -> dict[str, JSONValue]:
    check_deadline()
    if not isinstance(active, Mapping):
        return {}
    tests_payload = active.get("tests", [])
    tests: list[str] = []
    if isinstance(tests_payload, list):
        seen: set[str] = set()
        for entry in tests_payload:
            if not isinstance(entry, str):
                continue
            test_id = entry.strip()
            if not test_id or test_id in seen:
                continue
            seen.add(test_id)
            tests.append(test_id)
    tests = sort_once(
        tests,
        source="_normalize_active_metadata.tests",
    )
    summary_payload = active.get("summary", {})
    summary: dict[str, int] = {}
    if isinstance(summary_payload, Mapping):
        for key, value in summary_payload.items():
            if not isinstance(key, str):
                continue
            summary[key] = coerce_int(value, 0)
    result: dict[str, JSONValue] = {}
    if tests:
        result["tests"] = tests
    if summary:
        result["summary"] = summary
    return result


# gabion:ambiguity_boundary
def _tests_from_candidates(
    candidates: Iterable[Mapping[str, object]],
) -> list[dict[str, JSONValue]]:
    check_deadline()
    tests: dict[str, str] = {}
    for entry in candidates:
        if not isinstance(entry, Mapping):
            continue
        test_id = str(entry.get("test_id", "") or "").strip()
        class_name = str(entry.get("class", "") or "").strip()
        if not test_id or not class_name:
            continue
        tests[test_id] = class_name
    return [
        {"test_id": test_id, "class": tests[test_id]}
        for test_id in sort_once(
            tests,
            source="_tests_from_candidates.tests",
        )
    ]


def _build_evidence_index(
    evidence_by_test: Mapping[str, Iterable[object]],
    status_by_test: Mapping[str, str],
) -> list[dict[str, JSONValue]]:
    check_deadline()
    entries: dict[str, EvidenceIndexEntry] = {}
    for test_id, evidence in evidence_by_test.items():
        if status_by_test.get(test_id) != "mapped":
            continue
        refs = test_obsolescence._normalize_evidence_refs(evidence)
        if not refs:
            continue
        for ref in refs:
            identity = ref.identity
            existing = entries.get(identity)
            if existing is None:
                entries[identity] = EvidenceIndexEntry(
                    key=evidence_keys.normalize_key(ref.key),
                    identity=identity,
                    display=ref.display,
                    witness_count=1,
                )
                continue
            witness_count = existing.witness_count + 1
            display = existing.display
            if ref.display and (not display or ref.display < display):
                display = ref.display
            entries[identity] = EvidenceIndexEntry(
                key=existing.key,
                identity=identity,
                display=display,
                witness_count=witness_count,
            )
    return [
        _evidence_entry_payload(entries[identity])
        for identity in sort_once(
            entries,
            source="_build_evidence_index.entries",
        )
    ]


# gabion:ambiguity_boundary
def _parse_evidence_index(value: object) -> dict[str, EvidenceIndexEntry]:
    check_deadline()
    entries: dict[str, EvidenceIndexEntry] = {}
    if not isinstance(value, list):
        return entries
    for entry in value:
        if not isinstance(entry, Mapping):
            continue
        raw_key = entry.get("key")
        if not isinstance(raw_key, Mapping):
            continue
        key = evidence_keys.normalize_key(raw_key)
        identity = evidence_keys.key_identity(key)
        display = entry.get("display")
        display_value = (
            str(display) if isinstance(display, str) else evidence_keys.render_display(key)
        )
        witness_count = coerce_int(entry.get("witness_count"), 0)
        existing = entries.get(identity)
        if existing is None:
            entries[identity] = EvidenceIndexEntry(
                key=key,
                identity=identity,
                display=display_value,
                witness_count=witness_count,
            )
            continue
        combined_count = max(existing.witness_count, witness_count)
        combined_display = existing.display
        if display_value and (not combined_display or display_value < combined_display):
            combined_display = display_value
        entries[identity] = EvidenceIndexEntry(
            key=existing.key,
            identity=identity,
            display=combined_display,
            witness_count=combined_count,
        )
    return entries


def _count_opaque_evidence(evidence_by_test: Mapping[str, Iterable[object]]) -> int:
    check_deadline()
    total = 0
    for evidence in evidence_by_test.values():
        refs = test_obsolescence._normalize_evidence_refs(evidence)
        if any(ref.opaque for ref in refs):
            total += 1
    return total


def _evidence_entry_payload(entry: EvidenceIndexEntry) -> dict[str, JSONValue]:
    return {
        "key": entry.key,
        "display": entry.display,
        "witness_count": entry.witness_count,
    }


def _evidence_change_payload(
    before: EvidenceIndexEntry, after: EvidenceIndexEntry
) -> dict[str, JSONValue]:
    delta = after.witness_count - before.witness_count
    return {
        "key": after.key,
        "display": after.display,
        "before": before.witness_count,
        "after": after.witness_count,
        "delta": delta,
    }


# gabion:ambiguity_boundary
def _baseline_meta_payload(
    baseline: ObsolescenceBaseline, baseline_path: str | None
) -> dict[str, JSONValue]:
    payload: dict[str, JSONValue] = {
        "generated_by_spec_id": baseline.generated_by_spec_id,
        "generated_by_spec": baseline.generated_by_spec,
    }
    if baseline_path:
        payload["path"] = baseline_path
    return payload


def _current_meta_payload(
    current: ObsolescenceBaseline,
) -> dict[str, JSONValue]:
    return {
        "generated_by_spec_id": current.generated_by_spec_id,
        "generated_by_spec": current.generated_by_spec,
    }


# gabion:ambiguity_boundary
def _section_list(container: Mapping[str, JSONValue] | object, key: str) -> list[dict[str, object]]:
    if not isinstance(container, Mapping):
        return []
    value = container.get(key, [])
    if not isinstance(value, list):
        return []
    return [entry for entry in value if isinstance(entry, Mapping)]


def _render_test_section(
    doc: ReportDoc,
    title: str,
    entries: list[Mapping[str, object]],
) -> None:
    check_deadline()
    doc.line(f"### {title}")
    if not entries:
        doc.line("- None")
        return
    for entry in entries:
        test_id = str(entry.get("test_id", "") or "")
        class_name = str(entry.get("class", "") or "")
        suffix = f" (class: {class_name})" if class_name else ""
        doc.line(f"- `{test_id}`{suffix}")


def _render_test_changes(
    doc: ReportDoc,
    entries: list[Mapping[str, object]],
) -> None:
    check_deadline()
    doc.line("### Class Changes")
    if not entries:
        doc.line("- None")
        return
    for entry in entries:
        test_id = str(entry.get("test_id", "") or "")
        before = str(entry.get("before", "") or "")
        after = str(entry.get("after", "") or "")
        doc.line(f"- `{test_id}`: {before} -> {after}")


def _render_evidence_section(
    doc: ReportDoc,
    title: str,
    entries: list[Mapping[str, object]],
) -> None:
    check_deadline()
    doc.line(f"### {title}")
    if not entries:
        doc.line("- None")
        return
    for entry in entries:
        display = str(entry.get("display", "") or "")
        witnesses = coerce_int(entry.get("witness_count"), 0)
        doc.line(f"- `{display}` (witnesses: {witnesses})")


def _render_evidence_changes(
    doc: ReportDoc,
    entries: list[Mapping[str, object]],
) -> None:
    check_deadline()
    doc.line("### Witness Count Changes")
    if not entries:
        doc.line("- None")
        return
    for entry in entries:
        display = str(entry.get("display", "") or "")
        before = coerce_int(entry.get("before"), 0)
        after = coerce_int(entry.get("after"), 0)
        delta = coerce_int(entry.get("delta"), after - before)
        doc.line(
            f"- `{display}`: {before} -> {after} ({format_delta(delta)})"
        )


def build_resolution_worklist(
    baseline: ObsolescenceBaseline,
    classification: test_obsolescence.ClassificationResult,
) -> dict[str, JSONValue]:
    check_deadline()
    stale_by_test = {
        str(entry.get("test_id", "")): str(entry.get("class", ""))
        for entry in classification.stale_candidates
        if str(entry.get("test_id", "")).strip()
    }
    rows: list[dict[str, JSONValue]] = []
    resolved_total = 0
    remaining_total = 0
    for test_id in sort_once(
        baseline.tests,
        source="build_resolution_worklist.baseline_test_ids",
    ):
        baseline_class = baseline.tests[test_id]
        current_class = stale_by_test.get(test_id, "")
        proposed_action = _proposed_action_for_class(baseline_class)
        if current_class:
            remaining_total += 1
            if current_class == "unmapped":
                disposition = "stale_unmapped"
            elif current_class == "obsolete_candidate":
                disposition = "stale_unresolved"
            else:
                disposition = "stale_overlap"
        elif test_id in classification.active_tests:
            resolved_total += 1
            disposition = "retained_active"
        else:
            resolved_total += 1
            disposition = "removed_or_missing"
        rows.append(
            {
                "test_id": test_id,
                "baseline_class": baseline_class,
                "current_class": current_class or None,
                "proposed_action": proposed_action,
                "final_disposition": disposition,
            }
        )
    return {
        "version": 1,
        "summary": {
            "baseline_total": len(baseline.tests),
            "resolved_total": resolved_total,
            "remaining_stale_total": remaining_total,
            "remaining_stale_by_class": _normalize_summary_counts(
                classification.stale_summary
            ),
        },
        "rows": rows,
    }


def build_resolution_worklist_from_paths(
    *,
    baseline_path: str,
    evidence_path: str,
    risk_registry_path: str,
) -> dict[str, JSONValue]:
    # dataflow-bundle: baseline_path, evidence_path, risk_registry_path
    baseline = load_baseline(baseline_path)
    evidence_by_test, status_by_test = test_obsolescence.load_test_evidence(evidence_path)
    risk_registry = test_obsolescence.load_risk_registry(risk_registry_path)
    classification = test_obsolescence.classify_candidates(
        evidence_by_test,
        status_by_test,
        risk_registry,
    )
    return build_resolution_worklist(baseline, classification)


def _proposed_action_for_class(class_name: str) -> str:
    if class_name == "unmapped":
        return "map_evidence"
    if class_name == "redundant_by_evidence":
        return "merge_or_prune"
    if class_name == "equivalent_witness":
        return "pareto_keep_one"
    if class_name == "obsolete_candidate":
        return "resolve_or_remove"
    return "review"


def _class_keys() -> list[str]:
    return [
        "redundant_by_evidence",
        "equivalent_witness",
        "obsolete_candidate",
        "unmapped",
    ]

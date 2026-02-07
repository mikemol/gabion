from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

from gabion.analysis import evidence_keys, test_obsolescence
from gabion.analysis.projection_registry import (
    TEST_OBSOLESCENCE_BASELINE_SPEC,
    TEST_OBSOLESCENCE_DELTA_SPEC,
    spec_metadata_lines,
    spec_metadata_payload,
)
from gabion.json_types import JSONValue

BASELINE_VERSION = 1
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
    evidence_index: dict[str, EvidenceIndexEntry]
    opaque_evidence_count: int
    generated_by_spec_id: str
    generated_by_spec: dict[str, JSONValue]


def resolve_baseline_path(root: Path) -> Path:
    return root / BASELINE_RELATIVE_PATH


def build_baseline_payload(
    evidence_by_test: Mapping[str, Iterable[object]],
    status_by_test: Mapping[str, str],
    candidates: Iterable[Mapping[str, object]],
    summary_counts: Mapping[str, int],
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
    payload.update(spec_metadata_payload(TEST_OBSOLESCENCE_BASELINE_SPEC))
    return payload


def parse_baseline_payload(payload: Mapping[str, JSONValue]) -> ObsolescenceBaseline:
    version = payload.get("version", BASELINE_VERSION)
    try:
        version_value = int(version) if version is not None else BASELINE_VERSION
    except (TypeError, ValueError):
        version_value = -1
    if version_value != BASELINE_VERSION:
        raise ValueError(
            f"Unsupported test obsolescence baseline version={version!r}; expected {BASELINE_VERSION}"
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
    evidence_index = _parse_evidence_index(payload.get("evidence_index", []))
    opaque_count = _coerce_int(payload.get("opaque_evidence_count"), 0)
    spec_id = str(payload.get("generated_by_spec_id", "") or "")
    spec_payload = payload.get("generated_by_spec", {})
    spec: dict[str, JSONValue] = {}
    if isinstance(spec_payload, Mapping):
        spec = {str(key): spec_payload[key] for key in spec_payload}
    return ObsolescenceBaseline(
        summary=summary,
        tests=tests,
        evidence_index=evidence_index,
        opaque_evidence_count=opaque_count,
        generated_by_spec_id=spec_id,
        generated_by_spec=spec,
    )


def load_baseline(path: str) -> ObsolescenceBaseline:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("Test obsolescence baseline must be a JSON object.")
    return parse_baseline_payload(payload)


def write_baseline(path: str, payload: Mapping[str, JSONValue]) -> None:
    Path(path).write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


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
    added_tests = sorted(set(current_tests) - set(baseline_tests))
    removed_tests = sorted(set(baseline_tests) - set(current_tests))
    changed_tests = sorted(
        test_id
        for test_id in set(baseline_tests) & set(current_tests)
        if baseline_tests[test_id] != current_tests[test_id]
    )

    baseline_evidence = baseline.evidence_index
    current_evidence = current.evidence_index
    added_evidence_ids = sorted(
        set(current_evidence) - set(baseline_evidence)
    )
    removed_evidence_ids = sorted(
        set(baseline_evidence) - set(current_evidence)
    )
    changed_evidence_ids = sorted(
        evidence_id
        for evidence_id in set(baseline_evidence) & set(current_evidence)
        if baseline_evidence[evidence_id].witness_count
        != current_evidence[evidence_id].witness_count
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
    payload.update(spec_metadata_payload(TEST_OBSOLESCENCE_DELTA_SPEC))
    return payload


def render_markdown(delta_payload: Mapping[str, JSONValue]) -> str:
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

    lines: list[str] = []
    lines.append("# Test Obsolescence Delta")
    lines.append("")
    lines.append("Summary:")
    lines.extend(spec_metadata_lines(TEST_OBSOLESCENCE_DELTA_SPEC))

    baseline_meta = delta_payload.get("baseline", {})
    if isinstance(baseline_meta, Mapping):
        baseline_path = baseline_meta.get("path")
        if isinstance(baseline_path, str) and baseline_path:
            lines.append(f"- baseline: {baseline_path}")
        baseline_spec = baseline_meta.get("generated_by_spec_id")
        if isinstance(baseline_spec, str) and baseline_spec:
            lines.append(f"- baseline_spec_id: {baseline_spec}")
    current_meta = delta_payload.get("current", {})
    if isinstance(current_meta, Mapping):
        current_spec = current_meta.get("generated_by_spec_id")
        if isinstance(current_spec, str) and current_spec:
            lines.append(f"- current_spec_id: {current_spec}")

    for key in _class_keys():
        lines.append(
            f"- {key}: {_format_delta(baseline_counts.get(key, 0), current_counts.get(key, 0), delta_counts.get(key, 0))}"
        )
    opaque_line = _format_delta(
        opaque.get("baseline", 0),
        opaque.get("current", 0),
        opaque.get("delta", 0),
    )
    lines.append(f"- opaque_evidence_count: {opaque_line}")

    lines.append("")
    lines.append("## Tests")
    _render_test_section(lines, "Added", _section_list(tests_section, "added"))
    _render_test_section(lines, "Removed", _section_list(tests_section, "removed"))
    _render_test_changes(lines, _section_list(tests_section, "changed_class"))

    lines.append("")
    lines.append("## Evidence Keys")
    _render_evidence_section(lines, "Added", _section_list(evidence_section, "added"))
    _render_evidence_section(
        lines, "Removed", _section_list(evidence_section, "removed")
    )
    _render_evidence_changes(lines, _section_list(evidence_section, "changed"))

    return "\n".join(lines).rstrip() + "\n"


def build_baseline_payload_from_paths(
    evidence_path: str, risk_registry_path: str
) -> dict[str, JSONValue]:
    evidence_by_test, status_by_test = test_obsolescence.load_test_evidence(
        evidence_path
    )
    risk_registry = test_obsolescence.load_risk_registry(risk_registry_path)
    candidates, summary_counts = test_obsolescence.classify_candidates(
        evidence_by_test, status_by_test, risk_registry
    )
    return build_baseline_payload(
        evidence_by_test, status_by_test, candidates, summary_counts
    )


def _normalize_summary_counts(summary: Mapping[str, object] | object) -> dict[str, int]:
    result = {key: 0 for key in _class_keys()}
    if not isinstance(summary, Mapping):
        return result
    for key in result:
        result[key] = _coerce_int(summary.get(key), 0)
    return result


def _tests_from_candidates(
    candidates: Iterable[Mapping[str, object]],
) -> list[dict[str, JSONValue]]:
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
        for test_id in sorted(tests)
    ]


def _build_evidence_index(
    evidence_by_test: Mapping[str, Iterable[object]],
    status_by_test: Mapping[str, str],
) -> list[dict[str, JSONValue]]:
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
        for identity in sorted(entries)
    ]


def _parse_evidence_index(value: object) -> dict[str, EvidenceIndexEntry]:
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
        witness_count = _coerce_int(entry.get("witness_count"), 0)
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


def _section_list(container: Mapping[str, JSONValue] | object, key: str) -> list[dict[str, object]]:
    if not isinstance(container, Mapping):
        return []
    value = container.get(key, [])
    if not isinstance(value, list):
        return []
    return [entry for entry in value if isinstance(entry, Mapping)]


def _render_test_section(
    lines: list[str], title: str, entries: list[Mapping[str, object]]
) -> None:
    lines.append(f"### {title}")
    if not entries:
        lines.append("- None")
        return
    for entry in entries:
        test_id = str(entry.get("test_id", "") or "")
        class_name = str(entry.get("class", "") or "")
        suffix = f" (class: {class_name})" if class_name else ""
        lines.append(f"- `{test_id}`{suffix}")


def _render_test_changes(
    lines: list[str], entries: list[Mapping[str, object]]
) -> None:
    lines.append("### Class Changes")
    if not entries:
        lines.append("- None")
        return
    for entry in entries:
        test_id = str(entry.get("test_id", "") or "")
        before = str(entry.get("before", "") or "")
        after = str(entry.get("after", "") or "")
        lines.append(f"- `{test_id}`: {before} -> {after}")


def _render_evidence_section(
    lines: list[str], title: str, entries: list[Mapping[str, object]]
) -> None:
    lines.append(f"### {title}")
    if not entries:
        lines.append("- None")
        return
    for entry in entries:
        display = str(entry.get("display", "") or "")
        witnesses = _coerce_int(entry.get("witness_count"), 0)
        lines.append(f"- `{display}` (witnesses: {witnesses})")


def _render_evidence_changes(
    lines: list[str], entries: list[Mapping[str, object]]
) -> None:
    lines.append("### Witness Count Changes")
    if not entries:
        lines.append("- None")
        return
    for entry in entries:
        display = str(entry.get("display", "") or "")
        before = _coerce_int(entry.get("before"), 0)
        after = _coerce_int(entry.get("after"), 0)
        delta = _coerce_int(entry.get("delta"), after - before)
        lines.append(
            f"- `{display}`: {before} -> {after} ({_format_delta_value(delta)})"
        )


def _class_keys() -> list[str]:
    return [
        "redundant_by_evidence",
        "equivalent_witness",
        "obsolete_candidate",
        "unmapped",
    ]


def _format_delta(baseline: object, current: object, delta: object) -> str:
    base = _coerce_int(baseline, 0)
    curr = _coerce_int(current, 0)
    if delta is None:
        delta_value = curr - base
    else:
        delta_value = _coerce_int(delta, curr - base)
    return f"{base} -> {curr} ({_format_delta_value(delta_value)})"


def _format_delta_value(delta: int) -> str:
    sign = "+" if delta > 0 else ""
    return f"{sign}{delta}"


def _coerce_int(value: object, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default

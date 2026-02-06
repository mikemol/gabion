from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping


@dataclass(frozen=True)
class TestEvidenceEntry:
    test_id: str
    file: str
    line: int
    evidence: tuple[str, ...]
    status: str


@dataclass(frozen=True)
class Suggestion:
    test_id: str
    file: str
    line: int
    suggested: tuple[str, ...]
    matches: tuple[str, ...]


@dataclass(frozen=True)
class SuggestionSummary:
    suggested: int
    skipped_mapped: int
    skipped_no_match: int


def load_test_evidence(path: str) -> list[TestEvidenceEntry]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    schema_version = payload.get("schema_version")
    if schema_version != 1:
        raise ValueError(
            f"Unsupported test evidence schema_version={schema_version!r}; expected 1"
        )
    tests = payload.get("tests", [])
    if not isinstance(tests, list):
        raise ValueError("test evidence payload is missing tests list")
    entries: list[TestEvidenceEntry] = []
    for entry in tests:
        if not isinstance(entry, Mapping):
            continue
        test_id = str(entry.get("test_id", "") or "").strip()
        if not test_id:
            continue
        file_path = str(entry.get("file", "") or "").strip()
        line = int(entry.get("line", 0) or 0)
        evidence = _normalize_evidence_list(entry.get("evidence", []))
        raw_status = entry.get("status")
        status = str(raw_status).strip() if raw_status is not None else ""
        if not status:
            status = "mapped" if evidence else "unmapped"
        entries.append(
            TestEvidenceEntry(
                test_id=test_id,
                file=file_path,
                line=line,
                evidence=tuple(evidence),
                status=status,
            )
        )
    return sorted(entries, key=lambda item: item.test_id)


def suggest_evidence(
    entries: Iterable[TestEvidenceEntry],
) -> tuple[list[Suggestion], SuggestionSummary]:
    # dataflow-bundle: entries
    suggestions: list[Suggestion] = []
    skipped_mapped = 0
    skipped_no_match = 0
    for entry in sorted(entries, key=lambda item: item.test_id):
        if entry.evidence:
            skipped_mapped += 1
            continue
        suggested, matches = _suggest_for_entry(entry)
        if not suggested:
            skipped_no_match += 1
            continue
        suggestions.append(
            Suggestion(
                test_id=entry.test_id,
                file=entry.file,
                line=entry.line,
                suggested=tuple(suggested),
                matches=tuple(matches),
            )
        )
    summary = SuggestionSummary(
        suggested=len(suggestions),
        skipped_mapped=skipped_mapped,
        skipped_no_match=skipped_no_match,
    )
    return suggestions, summary


def render_markdown(
    suggestions: list[Suggestion],
    summary: SuggestionSummary,
) -> str:
    # dataflow-bundle: suggestions, summary
    lines: list[str] = []
    lines.append("# Test Evidence Suggestions")
    lines.append("")
    lines.append("Summary:")
    lines.append(f"- suggested: {summary.suggested}")
    lines.append(f"- skipped_mapped: {summary.skipped_mapped}")
    lines.append(f"- skipped_no_match: {summary.skipped_no_match}")
    lines.append("")
    lines.append("## Suggestions")
    if not suggestions:
        lines.append("- None")
        return "\n".join(lines).rstrip() + "\n"

    for entry in sorted(suggestions, key=lambda item: item.test_id):
        evidence_list = ", ".join(entry.suggested)
        match_list = ", ".join(entry.matches)
        lines.append(
            f"- `{entry.test_id}` -> {evidence_list} (matched: {match_list})"
        )
    return "\n".join(lines).rstrip() + "\n"


def render_json_payload(
    suggestions: list[Suggestion],
    summary: SuggestionSummary,
) -> dict[str, object]:
    # dataflow-bundle: suggestions, summary
    return {
        "version": 1,
        "summary": {
            "suggested": summary.suggested,
            "skipped_mapped": summary.skipped_mapped,
            "skipped_no_match": summary.skipped_no_match,
        },
        "suggestions": [
            {
                "test_id": entry.test_id,
                "file": entry.file,
                "line": entry.line,
                "suggested": list(entry.suggested),
                "matched": list(entry.matches),
            }
            for entry in sorted(suggestions, key=lambda item: item.test_id)
        ],
    }


def _suggest_for_entry(entry: TestEvidenceEntry) -> tuple[list[str], list[str]]:
    haystack = _suggestion_haystack(entry)
    rules = _suggestion_rules()
    suggested: list[str] = []
    matches: list[str] = []
    for rule in rules:
        if rule.matches(haystack):
            suggested.extend(rule.evidence)
            matches.append(rule.rule_id)
    return sorted(set(suggested)), matches


def _suggestion_haystack(entry: TestEvidenceEntry) -> str:
    name = entry.test_id.split("::")[-1]
    parts = [entry.file, name]
    return " ".join(part.lower() for part in parts if part).strip()


@dataclass(frozen=True)
class _SuggestionRule:
    rule_id: str
    evidence: tuple[str, ...]
    needles: tuple[str, ...]

    def matches(self, haystack: str) -> bool:
        return any(needle in haystack for needle in self.needles)


def _suggestion_rules() -> list[_SuggestionRule]:
    return [
        _SuggestionRule(
            rule_id="alias_invariance",
            evidence=("E:bundle/alias_invariance",),
            needles=("alias_", "rename_invariance"),
        ),
        _SuggestionRule(
            rule_id="baseline_ratchet",
            evidence=("E:baseline/ratchet_monotonicity",),
            needles=("baseline",),
        ),
        _SuggestionRule(
            rule_id="aspf_forest",
            evidence=("E:forest/packed_reuse", "E:forest/canonical_paramset"),
            needles=("aspf",),
        ),
        _SuggestionRule(
            rule_id="never_invariants",
            evidence=("E:never/sink_classification",),
            needles=("never_invariants",),
        ),
        _SuggestionRule(
            rule_id="policy_check",
            evidence=("E:policy/workflow_safety",),
            needles=("policy_check",),
        ),
    ]


def _normalize_evidence_list(value: object) -> list[str]:
    if value is None:
        return []
    items: list[str] = []
    if isinstance(value, str):
        items = [value]
    elif isinstance(value, (list, tuple, set)):
        items = [item for item in value if isinstance(item, str)]
    else:
        return []
    cleaned = [item.strip() for item in items if item.strip()]
    return sorted(set(cleaned))

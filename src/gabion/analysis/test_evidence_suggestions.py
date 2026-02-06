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
    total: int
    suggested: int
    skipped_mapped: int
    skipped_no_match: int
    unmapped_modules: tuple[tuple[str, int], ...]
    unmapped_prefixes: tuple[tuple[str, int], ...]


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
    entry_list = sorted(entries, key=lambda item: item.test_id)
    total = len(entry_list)
    for entry in entry_list:
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
    unmapped_modules, unmapped_prefixes = _summarize_unmapped(entry_list)
    summary = SuggestionSummary(
        total=total,
        suggested=len(suggestions),
        skipped_mapped=skipped_mapped,
        skipped_no_match=skipped_no_match,
        unmapped_modules=unmapped_modules,
        unmapped_prefixes=unmapped_prefixes,
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
    lines.append(f"- total: {summary.total}")
    lines.append(f"- suggested: {summary.suggested}")
    lines.append(f"- skipped_mapped: {summary.skipped_mapped}")
    lines.append(f"- skipped_no_match: {summary.skipped_no_match}")
    lines.append("")
    lines.append("Top Unmapped Modules:")
    if summary.unmapped_modules:
        for module, count in summary.unmapped_modules:
            lines.append(f"- {module}: {count}")
    else:
        lines.append("- None")
    lines.append("")
    lines.append("Top Unmapped Test Prefixes:")
    if summary.unmapped_prefixes:
        for prefix, count in summary.unmapped_prefixes:
            lines.append(f"- {prefix}: {count}")
    else:
        lines.append("- None")
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
        "version": 2,
        "summary": {
            "total": summary.total,
            "suggested": summary.suggested,
            "skipped_mapped": summary.skipped_mapped,
            "skipped_no_match": summary.skipped_no_match,
        },
        "unmapped_modules": [
            {"module": module, "count": count}
            for module, count in summary.unmapped_modules
        ],
        "unmapped_prefixes": [
            {"prefix": prefix, "count": count}
            for prefix, count in summary.unmapped_prefixes
        ],
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
    file_haystack, name_haystack = _suggestion_haystack(entry)
    rules = _suggestion_rules()
    suggested: list[str] = []
    matches: list[str] = []
    for rule in rules:
        if rule.matches(file=file_haystack, name=name_haystack):
            suggested.extend(rule.evidence)
            matches.append(rule.rule_id)
    return sorted(set(suggested)), matches


def _suggestion_haystack(entry: TestEvidenceEntry) -> tuple[str, str]:
    name = entry.test_id.split("::")[-1]
    file_haystack = entry.file.lower()
    name_haystack = name.lower()
    return file_haystack, name_haystack


@dataclass(frozen=True)
class _SuggestionRule:
    rule_id: str
    evidence: tuple[str, ...]
    needles: tuple[str, ...]
    scope: str = "both"
    exclude: tuple[str, ...] = ()

    def matches(self, *, file: str, name: str) -> bool:
        if self.scope == "file":
            haystack = file
        elif self.scope == "name":
            haystack = name
        else:
            haystack = f"{file} {name}"
        if self.exclude and any(needle in haystack for needle in self.exclude):
            return False
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
            rule_id="fingerprint_registry",
            evidence=("E:fingerprint/registry_determinism",),
            needles=("type_fingerprints",),
            scope="file",
        ),
        _SuggestionRule(
            rule_id="fingerprint_provenance",
            evidence=("E:fingerprint/match_provenance",),
            needles=("provenance", "matches"),
            scope="name",
        ),
        _SuggestionRule(
            rule_id="fingerprint_rewrite_plan",
            evidence=("E:fingerprint/rewrite_plan_verification",),
            needles=("rewrite_plan", "rewrite_plans", "rewrite plan", "rewrite_plan_summary"),
            scope="name",
        ),
        _SuggestionRule(
            rule_id="schema_surfaces",
            evidence=("E:schema/anonymous_payload_surfaces",),
            needles=("schema_audit", "anonymous_schema", "schema_surfaces"),
            scope="both",
        ),
        _SuggestionRule(
            rule_id="cli_command_surface",
            evidence=("E:cli/command_surface_integrity",),
            needles=("test_cli_", "cli_helpers", "cli_commands", "cli_payloads"),
            scope="file",
        ),
        _SuggestionRule(
            rule_id="server_command_dispatch",
            evidence=("E:transport/server_command_dispatch",),
            needles=("server_execute", "execute_command"),
            scope="file",
        ),
        _SuggestionRule(
            rule_id="transport_payload_roundtrip",
            evidence=("E:transport/payload_roundtrip",),
            needles=("test_run",),
            scope="both",
            exclude=(
                "test_cli_",
                "cli_helpers",
                "cli_commands",
                "cli_payloads",
                "server_execute",
            ),
        ),
        _SuggestionRule(
            rule_id="decision_surface_direct",
            evidence=("E:decision_surface/direct",),
            needles=("decision_surface", "decision surfaces"),
            scope="name",
            exclude=("value_encoded", "value_decision", "value-encoded"),
        ),
        _SuggestionRule(
            rule_id="decision_surface_value_encoded",
            evidence=("E:decision_surface/value_encoded",),
            needles=("value_encoded", "value_decision", "value-encoded"),
            scope="name",
        ),
        _SuggestionRule(
            rule_id="decision_surface_rebranch",
            evidence=("E:decision_surface/rebranch_suggested",),
            needles=("rebranch", "value_rewrite", "value_rewrites", "decision_rewrite"),
            scope="name",
        ),
        _SuggestionRule(
            rule_id="contextvar_rewrite",
            evidence=("E:context/ambient_rewrite_suggested",),
            needles=("contextvar", "ambient"),
            scope="name",
        ),
        _SuggestionRule(
            rule_id="docflow_contract",
            evidence=("E:policy/docflow_contract",),
            needles=("docflow",),
            scope="name",
        ),
        _SuggestionRule(
            rule_id="policy_workflow_safety",
            evidence=("E:policy/workflow_safety",),
            needles=("policy", "policy_check", "workflow_safety"),
            scope="name",
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


def _summarize_unmapped(
    entries: Iterable[TestEvidenceEntry],
) -> tuple[tuple[tuple[str, int], ...], tuple[tuple[str, int], ...]]:
    # dataflow-bundle: entries
    module_counts: dict[str, int] = {}
    prefix_counts: dict[str, int] = {}
    for entry in entries:
        if entry.evidence:
            continue
        module_counts[entry.file] = module_counts.get(entry.file, 0) + 1
        prefix = _test_prefix(entry.test_id)
        if prefix:
            prefix_counts[prefix] = prefix_counts.get(prefix, 0) + 1
    modules = _top_counts(module_counts)
    prefixes = _top_counts(prefix_counts)
    return modules, prefixes


def _top_counts(source: dict[str, int], *, limit: int = 10) -> tuple[tuple[str, int], ...]:
    ordered = sorted(source.items(), key=lambda item: (-item[1], item[0]))
    return tuple(ordered[:limit])


def _test_prefix(test_id: str) -> str:
    name = test_id.split("::")[-1]
    if not name.startswith("test_"):
        return ""
    parts = name.split("_")
    if len(parts) < 2:
        return "test"
    return f"test_{parts[1]}"

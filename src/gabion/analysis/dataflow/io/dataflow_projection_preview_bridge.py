# gabion:boundary_normalization_module
# gabion:decision_protocol_module
from __future__ import annotations

"""Preview builders for report projection sections.

This bridge remains boundary-only but is runtime-import free.
"""

from collections import defaultdict
from collections.abc import Callable, Sequence
from pathlib import Path

from gabion.analysis.dataflow.engine.dataflow_contracts import ReportCarrier
from gabion.analysis.dataflow.io.dataflow_reporting_helpers import (
    parse_failure_violation_lines, runtime_obligation_violation_lines, summarize_deadline_obligations)
from gabion.analysis.foundation.json_types import JSONObject
from gabion.analysis.foundation.timeout_context import check_deadline
from gabion.invariants import never
from gabion.order_contract import sort_once

PreviewBuilder = Callable[[ReportCarrier, dict[Path, dict[str, list[set[str]]]]], list[str]]


def _known_violation_lines(report: ReportCarrier) -> list[str]:
    check_deadline()
    lines: list[str] = []
    lines.extend(runtime_obligation_violation_lines(report.resumability_obligations))
    lines.extend(runtime_obligation_violation_lines(report.incremental_report_obligations))
    lines.extend(parse_failure_violation_lines(report.parse_failure_witnesses))
    lines.extend(report.decision_warnings)
    seen: set[str] = set()
    unique: list[str] = []
    for line in lines:
        check_deadline()
        if line in seen:
            continue
        seen.add(line)
        unique.append(line)
    return unique


def _preview_components_section(
    report: ReportCarrier,
    groups_by_path: dict[Path, dict[str, list[set[str]]]],
) -> list[str]:
    check_deadline()
    path_count = len(groups_by_path)
    function_count = sum(len(groups) for groups in groups_by_path.values())
    bundle_alternatives = 0
    for groups in groups_by_path.values():
        check_deadline()
        for bundles in groups.values():
            check_deadline()
            bundle_alternatives += len(bundles)
    lines = [
        "Component preview (provisional).",
        f"- `paths_with_groups`: `{path_count}`",
        f"- `functions_with_groups`: `{function_count}`",
        f"- `bundle_alternatives`: `{bundle_alternatives}`",
    ]
    if report.forest.alts:
        lines.append(f"- `forest_alternatives`: `{len(report.forest.alts)}`")
        lines.append("- `component_count`: `deferred_until_post_phase`")
    return lines


def _preview_violations_section(
    report: ReportCarrier,
    _groups_by_path: dict[Path, dict[str, list[set[str]]]],
) -> list[str]:
    check_deadline()
    known = _known_violation_lines(report)
    lines = [
        "Violations preview (provisional).",
        f"- `known_violations`: `{len(known)}`",
    ]
    if not known:
        lines.append("- none observed yet")
        return lines
    lines.append("Top known violations:")
    for line in known[:10]:
        check_deadline()
        lines.append(f"- {line}")
    return lines


def _preview_type_flow_section(
    report: ReportCarrier,
    _groups_by_path: dict[Path, dict[str, list[set[str]]]],
) -> list[str]:
    check_deadline()
    lines = [
        "Type-flow audit preview (provisional).",
        f"- `type_suggestions`: `{len(report.type_suggestions)}`",
        f"- `type_ambiguities`: `{len(report.type_ambiguities)}`",
        f"- `type_callsite_evidence`: `{len(report.type_callsite_evidence)}`",
    ]
    if report.type_ambiguities:
        lines.append(f"- `sample_type_ambiguity`: `{report.type_ambiguities[0]}`")
    return lines


def _preview_deadline_summary_section(
    report: ReportCarrier,
    _groups_by_path: dict[Path, dict[str, list[set[str]]]],
) -> list[str]:
    check_deadline()
    if not report.deadline_obligations:
        return [
            "Deadline propagation preview (provisional).",
            "- no deadline obligations yet",
        ]
    summary = summarize_deadline_obligations(
        report.deadline_obligations,
        forest=report.forest,
    )
    lines = ["Deadline propagation preview (provisional)."]
    lines.extend(summary[:20])
    return lines


def _make_list_section_preview(
    *,
    title: str,
    count_label: str,
    values: Callable[[ReportCarrier], Sequence[str]],
    sample_label=None,
    extra_count_labels: tuple[tuple[str, Callable[[ReportCarrier], int]], ...] = (),
) -> Callable[[ReportCarrier, dict[Path, dict[str, list[set[str]]]]], list[str]]:
    def _preview(
        report: ReportCarrier,
        _groups_by_path: dict[Path, dict[str, list[set[str]]]],
    ) -> list[str]:
        check_deadline()
        series = values(report)
        lines = [
            f"{title} preview (provisional).",
            f"- `{count_label}`: `{len(series)}`",
        ]
        for label, getter in extra_count_labels:
            check_deadline()
            lines.append(f"- `{label}`: `{getter(report)}`")
        if sample_label and series:
            lines.append(f"- `{sample_label}`: `{series[0]}`")
        return lines

    return _preview


_preview_constant_smells_section = _make_list_section_preview(
    title="Constant-propagation smells",
    count_label="constant_smells",
    values=lambda report: report.constant_smells,
    sample_label="sample_constant_smell",
)

_preview_unused_arg_smells_section = _make_list_section_preview(
    title="Unused-argument smells",
    count_label="unused_arg_smells",
    values=lambda report: report.unused_arg_smells,
    sample_label="sample_unused_arg_smell",
)


def _preview_runtime_obligations_section(
    *,
    title: str,
    obligations: list[JSONObject],
) -> list[str]:
    check_deadline()
    violated = 0
    satisfied = 0
    pending = 0
    for entry in obligations:
        check_deadline()
        status = entry.get("status")
        if status == "VIOLATION":
            violated += 1
        elif status == "SATISFIED":
            satisfied += 1
        else:
            pending += 1
    lines = [
        f"{title} preview (provisional).",
        f"- `obligations`: `{len(obligations)}`",
        f"- `violations`: `{violated}`",
        f"- `satisfied`: `{satisfied}`",
        f"- `pending`: `{pending}`",
    ]
    for entry in obligations:
        check_deadline()
        if entry.get("status") != "VIOLATION":
            continue
        contract = str(entry.get("contract", "runtime_contract"))
        kind = str(entry.get("kind", "unknown"))
        detail = str(entry.get("detail", ""))
        lines.append(f"- `sample_violation`: `{contract}/{kind} {detail}`")
        break
    return lines


def _preview_resumability_obligations_section(
    report: ReportCarrier,
    _groups_by_path: dict[Path, dict[str, list[set[str]]]],
) -> list[str]:
    return _preview_runtime_obligations_section(
        title="Resumability obligations",
        obligations=report.resumability_obligations,
    )


def _preview_incremental_report_obligations_section(
    report: ReportCarrier,
    _groups_by_path: dict[Path, dict[str, list[set[str]]]],
) -> list[str]:
    return _preview_runtime_obligations_section(
        title="Incremental report obligations",
        obligations=report.incremental_report_obligations,
    )


def _preview_parse_failure_witnesses_section(
    report: ReportCarrier,
    _groups_by_path: dict[Path, dict[str, list[set[str]]]],
) -> list[str]:
    check_deadline()
    stage_counts: dict[str, int] = defaultdict(int)
    for witness in report.parse_failure_witnesses:
        check_deadline()
        stage = str(witness.get("stage", "") or "").strip()
        if not stage:
            stage_counts["unknown"] += 1
            continue
        stage_counts[stage] += 1
    lines = [
        "Parse failure witnesses preview (provisional).",
        f"- `parse_failure_witnesses`: `{len(report.parse_failure_witnesses)}`",
    ]
    for stage, count in sort_once(
        stage_counts.items(),
        source="preview_parse_failure_witnesses_section.stage_counts",
        key=lambda item: item[0],
    ):
        check_deadline()
        lines.append(f"- `stage[{stage}]`: `{count}`")
    return lines


def _preview_execution_pattern_suggestions_section(
    report: ReportCarrier,
    groups_by_path: dict[Path, dict[str, list[set[str]]]],
) -> list[str]:
    check_deadline()
    function_count = sum(len(groups) for groups in groups_by_path.values())
    return [
        "Execution pattern opportunities preview (provisional).",
        f"- `paths_with_groups`: `{len(groups_by_path)}`",
        f"- `functions_with_groups`: `{function_count}`",
        f"- `decision_surfaces_seen`: `{len(report.decision_surfaces)}`",
        "- `note`: `full execution-pattern synthesis is materialized in post-phase projection`",
    ]


def _preview_pattern_schema_residue_section(
    report: ReportCarrier,
    groups_by_path: dict[Path, dict[str, list[set[str]]]],
) -> list[str]:
    check_deadline()
    bundle_alternatives = 0
    for groups in groups_by_path.values():
        check_deadline()
        for bundles in groups.values():
            check_deadline()
            bundle_alternatives += len(bundles)
    return [
        "Pattern schema residue preview (provisional).",
        f"- `paths_with_groups`: `{len(groups_by_path)}`",
        f"- `bundle_alternatives`: `{bundle_alternatives}`",
        f"- `decision_surfaces_seen`: `{len(report.decision_surfaces)}`",
        f"- `value_decision_surfaces_seen`: `{len(report.value_decision_surfaces)}`",
    ]


_preview_decision_surfaces_section = _make_list_section_preview(
    title="Decision surfaces",
    count_label="decision_surfaces",
    values=lambda report: report.decision_surfaces,
    sample_label="sample_decision_surface",
    extra_count_labels=(("decision_warnings", lambda report: len(report.decision_warnings)),),
)

_preview_value_decision_surfaces_section = _make_list_section_preview(
    title="Value-encoded decision surfaces",
    count_label="value_decision_surfaces",
    values=lambda report: report.value_decision_surfaces,
    sample_label="sample_value_decision_surface",
    extra_count_labels=(("value_decision_rewrites", lambda report: len(report.value_decision_rewrites)),),
)

_preview_fingerprint_warnings_section = _make_list_section_preview(
    title="Fingerprint warnings",
    count_label="fingerprint_warnings",
    values=lambda report: report.fingerprint_warnings,
    sample_label="sample_fingerprint_warning",
)

_preview_fingerprint_matches_section = _make_list_section_preview(
    title="Fingerprint matches",
    count_label="fingerprint_matches",
    values=lambda report: report.fingerprint_matches,
    sample_label="sample_fingerprint_match",
)

_preview_fingerprint_synthesis_section = _make_list_section_preview(
    title="Fingerprint synthesis",
    count_label="fingerprint_synth",
    values=lambda report: report.fingerprint_synth,
    sample_label="sample_fingerprint_synth",
    extra_count_labels=(("fingerprint_provenance", lambda report: len(report.fingerprint_provenance)),),
)

_preview_context_suggestions_section = _make_list_section_preview(
    title="Context suggestions",
    count_label="context_suggestions",
    values=lambda report: report.context_suggestions,
    sample_label="sample_context_suggestion",
)


def _preview_schema_surfaces_section(
    _report: ReportCarrier,
    groups_by_path: dict[Path, dict[str, list[set[str]]]],
) -> list[str]:
    check_deadline()
    return [
        "Schema surfaces preview (provisional).",
        f"- `paths_with_groups`: `{len(groups_by_path)}`",
        "- `note`: `full schema-surface discovery is materialized in post-phase projection`",
    ]


def _preview_deprecated_substrate_section(
    report: ReportCarrier,
    _groups_by_path: dict[Path, dict[str, list[set[str]]]],
) -> list[str]:
    check_deadline()
    lines = [
        "Deprecated substrate preview (provisional).",
        f"- `informational_signals`: `{len(report.deprecated_signals)}`",
    ]
    for signal in report.deprecated_signals[:5]:
        check_deadline()
        lines.append(f"- {signal}")
    return lines


_PREVIEW_BUILDERS: dict[str, PreviewBuilder] = {
    "components": _preview_components_section,
    "violations": _preview_violations_section,
    "type_flow": _preview_type_flow_section,
    "constant_smells": _preview_constant_smells_section,
    "unused_arg_smells": _preview_unused_arg_smells_section,
    "deadline_summary": _preview_deadline_summary_section,
    "resumability_obligations": _preview_resumability_obligations_section,
    "incremental_report_obligations": _preview_incremental_report_obligations_section,
    "parse_failure_witnesses": _preview_parse_failure_witnesses_section,
    "execution_pattern_suggestions": _preview_execution_pattern_suggestions_section,
    "pattern_schema_residue": _preview_pattern_schema_residue_section,
    "decision_surfaces": _preview_decision_surfaces_section,
    "value_decision_surfaces": _preview_value_decision_surfaces_section,
    "fingerprint_warnings": _preview_fingerprint_warnings_section,
    "fingerprint_matches": _preview_fingerprint_matches_section,
    "fingerprint_synthesis": _preview_fingerprint_synthesis_section,
    "context_suggestions": _preview_context_suggestions_section,
    "schema_surfaces": _preview_schema_surfaces_section,
    "deprecated_substrate": _preview_deprecated_substrate_section,
}


def preview_section_lines(
    section_id: str,
    *,
    report: ReportCarrier,
    groups_by_path: dict[Path, dict[str, list[set[str]]]],
) -> list[str]:
    preview_builder = _PREVIEW_BUILDERS.get(section_id)
    if preview_builder is None:
        never(
            "preview section id missing from preview bridge map",
            section_id=section_id,
        )
    return preview_builder(report, groups_by_path)


__all__ = ["preview_section_lines"]

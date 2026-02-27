# gabion:decision_protocol_module
"""Rendering adapters extracted from ``dataflow_audit``."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable

from gabion.analysis.artifact_ordering import (
    canonical_count_summary_items,
    canonical_field_display_parts,
    canonical_protocol_specs,
    canonical_string_values,
)
from gabion.analysis.json_types import JSONObject


def render_synthesis_section(
    plan: JSONObject,
    *,
    check_deadline: Callable[[], None],
) -> str:
    check_deadline()
    protocols = plan.get("protocols", [])
    warnings = plan.get("warnings", [])
    errors = plan.get("errors", [])
    lines = ["", "## Synthesis plan (prototype)", ""]
    if not protocols:
        lines.append("No protocol candidates.")
    else:
        evidence_counts: Counter[str] = Counter()
        for spec in canonical_protocol_specs(protocols):
            check_deadline()
            name = spec.get("name", "Bundle")
            tier = spec.get("tier", "?")
            fields = spec.get("fields", [])
            parts = canonical_field_display_parts(fields)
            field_list = ", ".join(parts) if parts else "(no fields)"
            evidence = spec.get("evidence", [])
            if evidence:
                evidence_entries = canonical_string_values(evidence)
                evidence_str = ", ".join(evidence_entries)
                lines.append(f"- {name} (tier {tier}; evidence: {evidence_str}): {field_list}")
                evidence_counts.update(evidence_entries)
            else:
                lines.append(f"- {name} (tier {tier}): {field_list}")
        if evidence_counts:
            summary = ", ".join(
                f"{key}={count}"
                for key, count in canonical_count_summary_items(evidence_counts)
            )
            lines.append("")
            lines.append(f"Evidence summary: {summary}")
    if warnings:
        lines.append("")
        lines.append("Warnings:")
        lines.append("```")
        lines.extend(str(w) for w in warnings)
        lines.append("```")
    if errors:
        lines.append("")
        lines.append("Errors:")
        lines.append("```")
        lines.extend(str(e) for e in errors)
        lines.append("```")
    return "\n".join(lines)


def render_unsupported_by_adapter_section(
    diagnostics: list[JSONObject],
    *,
    check_deadline: Callable[[], None],
) -> list[str]:
    lines: list[str] = []
    for diagnostic in diagnostics:
        check_deadline()
        if type(diagnostic) is not dict:
            continue
        surface = str(diagnostic.get("surface", ""))
        adapter = str(diagnostic.get("adapter", "native"))
        required = bool(diagnostic.get("required_by_policy", False))
        line = f"{surface}: unsupported_by_adapter ({adapter})"
        if required:
            line = f"{line} [required]"
        lines.append(line)
    return lines

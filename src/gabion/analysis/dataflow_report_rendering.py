"""Rendering adapters extracted from ``dataflow_audit``."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable

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
        for spec in protocols:
            check_deadline()
            name = spec.get("name", "Bundle")
            tier = spec.get("tier", "?")
            fields = spec.get("fields", [])
            parts = []
            for field in fields:
                check_deadline()
                fname = field.get("name", "")
                type_hint = field.get("type_hint") or "Any"
                if fname:
                    parts.append(f"{fname}: {type_hint}")
            field_list = ", ".join(parts) if parts else "(no fields)"
            evidence = spec.get("evidence", [])
            if evidence:
                evidence_str = ", ".join(sorted(evidence))
                lines.append(f"- {name} (tier {tier}; evidence: {evidence_str}): {field_list}")
                evidence_counts.update(evidence)
            else:
                lines.append(f"- {name} (tier {tier}): {field_list}")
        if evidence_counts:
            summary = ", ".join(
                f"{key}={count}" for key, count in evidence_counts.most_common()
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

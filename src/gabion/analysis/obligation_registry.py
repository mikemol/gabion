# gabion:decision_protocol_module
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping

from gabion.analysis.timeout_context import check_deadline
from gabion.json_types import JSONValue


Context = Mapping[str, JSONValue]


@dataclass(frozen=True)
class ObligationRule:
    obligation_id: str
    operation: str
    context: str
    description: str
    enforcement: str
    when: Callable[[Context], bool]
    met: Callable[[Context], bool]


def _bool_flag(context: Context, key: str) -> bool:
    value = context.get(key)
    return value is True


OBLIGATION_REGISTRY: tuple[ObligationRule, ...] = (
    ObligationRule(
        obligation_id="sppf_gh_reference_validation",
        operation="docflow_plan",
        context="sppf_relevant_change",
        description=(
            "SPPF-relevant path changes require GH-reference validation."
        ),
        enforcement="fail",
        when=lambda c: _bool_flag(c, "sppf_relevant_paths_changed"),
        met=lambda c: _bool_flag(c, "gh_reference_validated"),
    ),
    ObligationRule(
        obligation_id="baseline_delta_guard",
        operation="docflow_plan",
        context="baseline_write",
        description="Baseline writes require a completed delta guard check.",
        enforcement="warn",
        when=lambda c: _bool_flag(c, "baseline_write_emitted"),
        met=lambda c: _bool_flag(c, "delta_guard_checked"),
    ),
    ObligationRule(
        obligation_id="doc_status_consistency",
        operation="docflow_plan",
        context="doc_status_change",
        description=(
            "Doc-status changes require checklist/influence consistency validation."
        ),
        enforcement="fail",
        when=lambda c: _bool_flag(c, "doc_status_changed"),
        met=lambda c: _bool_flag(c, "checklist_influence_consistent"),
    ),
)


def evaluate_obligations(*, operation: str, context: Context) -> list[dict[str, JSONValue]]:
    entries: list[dict[str, JSONValue]] = []
    for rule in OBLIGATION_REGISTRY:
        check_deadline()
        if rule.operation != operation:
            continue
        triggered = bool(rule.when(context))
        satisfied = bool(rule.met(context)) if triggered else True
        status = "met" if satisfied else "unmet"
        entries.append(
            {
                "obligation_id": rule.obligation_id,
                "operation": rule.operation,
                "context": rule.context,
                "description": rule.description,
                "enforcement": rule.enforcement,
                "triggered": triggered,
                "status": status,
            }
        )
    return entries


def summarize_obligations(entries: list[Mapping[str, JSONValue]]) -> dict[str, int]:
    summary = {
        "total": 0,
        "triggered": 0,
        "met": 0,
        "unmet_fail": 0,
        "unmet_warn": 0,
    }
    for entry in entries:
        check_deadline()
        summary["total"] += 1
        if entry.get("triggered") is not True:
            continue
        summary["triggered"] += 1
        if entry.get("status") == "met":
            summary["met"] += 1
            continue
        if entry.get("enforcement") == "fail":
            summary["unmet_fail"] += 1
        else:
            summary["unmet_warn"] += 1
    return summary


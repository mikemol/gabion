# gabion:decision_protocol_module
# gabion:boundary_normalization_module
# gabion:ambiguity_boundary_module
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence

from gabion.json_types import JSONObject
from gabion.order_contract import sort_once
from gabion.runtime.stable_encode import stable_compact_text

_ACTION_PLAN_FORMAT_VERSION = 1


def build_action_plan_payload(
    *,
    trace_id: str,
    command_profile: str,
    opportunities_payload: Mapping[str, object],
    semantic_surface_payloads: Mapping[str, object],
    root: Path | None = None,
    delta_ledger_payload: Mapping[str, object] | None = None,
    trace_payload: Mapping[str, object] | None = None,
) -> JSONObject:
    opportunities_raw = opportunities_payload.get("opportunities")
    opportunities = (
        [item for item in opportunities_raw if isinstance(item, Mapping)]
        if isinstance(opportunities_raw, list)
        else []
    )
    candidate_paths, candidate_scores, candidate_sources = _ranked_candidate_paths(
        semantic_surface_payloads=semantic_surface_payloads,
        delta_ledger_payload=delta_ledger_payload,
        trace_payload=trace_payload,
        root=root,
        limit=12,
    )
    actions: list[JSONObject] = []
    for raw in opportunities:
        kind = str(raw.get("kind", "")).strip()
        if not kind:
            continue
        confidence = _confidence(raw.get("confidence"))
        opportunity_id = str(raw.get("opportunity_id", "")).strip() or "opportunity:unknown"
        affected_surfaces = _string_list(raw.get("affected_surfaces"))
        witness_ids = _string_list(raw.get("witness_ids"))
        reason = str(raw.get("reason", "")).strip()
        action_material = stable_compact_text(
            {
                "kind": kind,
                "opportunity_id": opportunity_id,
                "affected_surfaces": affected_surfaces,
                "reason": reason,
                "confidence": confidence,
            }
        )
        action_digest = hashlib.sha256(action_material.encode("utf-8")).hexdigest()[:12]
        action_id = f"action:{kind}:{action_digest}"
        actions.append(
            {
                "action_id": action_id,
                "priority": _priority(confidence),
                "opportunity_kind": kind,
                "confidence": confidence,
                "targets": {
                    "paths": candidate_paths,
                    "path_scores": {
                        path: candidate_scores[path]
                        for path in candidate_paths
                        if path in candidate_scores
                    },
                    "command_profiles": [command_profile],
                },
                "affected_surfaces": affected_surfaces,
                "evidence_refs": {
                    "opportunity_id": opportunity_id,
                    "witness_ids": witness_ids,
                    "reason": reason,
                    "path_sources": {
                        path: candidate_sources[path]
                        for path in candidate_paths
                        if path in candidate_sources
                    },
                },
                "implementation_steps": _implementation_steps(kind),
                "validation_commands": _validation_commands(kind),
            }
        )
    actions = sorted(
        actions,
        key=lambda item: (
            _priority_rank(str(item.get("priority", ""))),
            -_confidence(item.get("confidence")),
            str(item.get("action_id", "")),
        ),
    )
    return {
        "format_version": _ACTION_PLAN_FORMAT_VERSION,
        "trace_id": str(trace_id),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "summary": {
            "action_count": len(actions),
            "high_priority": sum(
                1 for action in actions if str(action.get("priority")) == "high"
            ),
            "medium_priority": sum(
                1 for action in actions if str(action.get("priority")) == "medium"
            ),
            "low_priority": sum(
                1 for action in actions if str(action.get("priority")) == "low"
            ),
        },
        "actions": actions,
    }


def evaluate_action_plan_quality(
    *,
    action_plan_payload: Mapping[str, object],
    opportunities_payload: Mapping[str, object],
) -> JSONObject:
    action_rows = action_plan_payload.get("actions")
    actions = (
        [entry for entry in action_rows if isinstance(entry, Mapping)]
        if isinstance(action_rows, list)
        else []
    )
    opportunities_rows = opportunities_payload.get("opportunities")
    opportunities = (
        [entry for entry in opportunities_rows if isinstance(entry, Mapping)]
        if isinstance(opportunities_rows, list)
        else []
    )

    issues: list[JSONObject] = []
    opportunity_count = len(opportunities)
    action_count = len(actions)
    if opportunity_count > 0 and action_count == 0:
        issues.append(
            {
                "issue_id": "missing_actions_for_opportunities",
                "severity": "warning",
                "message": (
                    "Opportunities were detected but no actions were emitted."
                ),
                "action_id": None,
            }
        )
    if action_count < opportunity_count:
        issues.append(
            {
                "issue_id": "action_count_below_opportunity_count",
                "severity": "warning",
                "message": (
                    "Action count is lower than opportunity count; investigate dropped mappings."
                ),
                "action_id": None,
            }
        )
    for action in actions:
        action_id = str(action.get("action_id", "")).strip() or "action:unknown"
        targets = action.get("targets")
        paths = (
            _string_list(targets.get("paths"))
            if isinstance(targets, Mapping)
            else []
        )
        if not paths:
            issues.append(
                {
                    "issue_id": f"missing_target_paths:{action_id}",
                    "severity": "warning",
                    "message": "Action has no target paths.",
                    "action_id": action_id,
                }
            )
        commands = _string_list(action.get("validation_commands"))
        if not commands:
            issues.append(
                {
                    "issue_id": f"missing_validation_commands:{action_id}",
                    "severity": "warning",
                    "message": "Action has no validation commands.",
                    "action_id": action_id,
                }
            )
    issues = sorted(issues, key=lambda issue: str(issue.get("issue_id", "")))
    return {
        "status": "warning" if issues else "ok",
        "issues": issues,
        "summary": {
            "opportunity_count": opportunity_count,
            "action_count": action_count,
            "issue_count": len(issues),
        },
    }


def render_action_plan_markdown(payload: Mapping[str, object]) -> str:
    trace_id = str(payload.get("trace_id", "")).strip() or "unknown"
    actions_raw = payload.get("actions")
    actions = (
        [item for item in actions_raw if isinstance(item, Mapping)]
        if isinstance(actions_raw, list)
        else []
    )
    lines: list[str] = [
        "# ASPF Action Plan",
        "",
        f"- trace_id: `{trace_id}`",
        f"- action_count: `{len(actions)}`",
        "",
    ]
    if not actions:
        lines.append("No actionable opportunities found.")
        return "\n".join(lines)
    for index, action in enumerate(actions, start=1):
        action_id = str(action.get("action_id", "")).strip() or f"action-{index}"
        priority = str(action.get("priority", "")).strip() or "low"
        kind = str(action.get("opportunity_kind", "")).strip() or "unknown"
        confidence = _confidence(action.get("confidence"))
        lines.extend(
            [
                f"## {index}. {kind}",
                f"- action_id: `{action_id}`",
                f"- priority: `{priority}`",
                f"- confidence: `{confidence:.2f}`",
            ]
        )
        targets = action.get("targets")
        if isinstance(targets, Mapping):
            paths = _string_list(targets.get("paths"))
            if paths:
                lines.append(f"- target_paths: `{', '.join(paths)}`")
            command_profiles = _string_list(targets.get("command_profiles"))
            if command_profiles:
                lines.append(f"- command_profiles: `{', '.join(command_profiles)}`")
        affected_surfaces = _string_list(action.get("affected_surfaces"))
        if affected_surfaces:
            lines.append(f"- affected_surfaces: `{', '.join(affected_surfaces)}`")
        evidence = action.get("evidence_refs")
        if isinstance(evidence, Mapping):
            opp_id = str(evidence.get("opportunity_id", "")).strip()
            if opp_id:
                lines.append(f"- opportunity_id: `{opp_id}`")
            reason = str(evidence.get("reason", "")).strip()
            if reason:
                lines.append(f"- reason: {reason}")
            witness_ids = _string_list(evidence.get("witness_ids"))
            if witness_ids:
                lines.append(f"- witness_ids: `{', '.join(witness_ids)}`")
        steps = _string_list(action.get("implementation_steps"))
        if steps:
            lines.append("- implementation_steps:")
            lines.extend([f"  - {step}" for step in steps])
        commands = _string_list(action.get("validation_commands"))
        if commands:
            lines.append("- validation_commands:")
            lines.extend([f"  - `{command}`" for command in commands])
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _ranked_candidate_paths(
    *,
    semantic_surface_payloads: Mapping[str, object],
    delta_ledger_payload: Mapping[str, object] | None,
    trace_payload: Mapping[str, object] | None,
    root: Path | None,
    limit: int,
) -> tuple[list[str], dict[str, int], dict[str, list[str]]]:
    scores: dict[str, int] = {}
    sources: dict[str, set[str]] = {}

    def _register(raw_path: str, *, weight: int, source_label: str) -> None:
        normalized = _normalize_candidate_path(raw_path, root=root)
        if normalized is None:
            return
        scores[normalized] = int(scores.get(normalized, 0)) + int(weight)
        source_set = sources.setdefault(normalized, set())
        source_set.add(source_label)

    groups = semantic_surface_payloads.get("groups_by_path")
    if isinstance(groups, Mapping):
        for raw_path in groups:
            _register(
                str(raw_path),
                weight=10,
                source_label="semantic_surfaces.groups_by_path",
            )

    delta_payload = semantic_surface_payloads.get("delta_payload")
    if isinstance(delta_payload, Mapping):
        for raw_path in _delta_payload_file_site_paths(delta_payload):
            _register(
                raw_path,
                weight=3,
                source_label="semantic_surfaces.delta_payload.call_stack.site_table",
            )

    if isinstance(delta_ledger_payload, Mapping):
        raw_records = delta_ledger_payload.get("records")
        if isinstance(raw_records, list):
            for raw_record in raw_records:
                if not isinstance(raw_record, Mapping):
                    continue
                mutation_value = raw_record.get("mutation_value")
                if not isinstance(mutation_value, Mapping):
                    continue
                metadata = mutation_value.get("metadata")
                if isinstance(metadata, Mapping):
                    for key in ("path", "checkpoint_path"):
                        raw_path = metadata.get(key)
                        if isinstance(raw_path, str):
                            _register(
                                raw_path,
                                weight=2,
                                source_label=f"delta_ledger.records.metadata.{key}",
                            )
                for key in ("path", "checkpoint_path"):
                    raw_path = mutation_value.get(key)
                    if isinstance(raw_path, str):
                        _register(
                            raw_path,
                            weight=2,
                            source_label=f"delta_ledger.records.mutation_value.{key}",
                        )

    if isinstance(trace_payload, Mapping):
        one_cells = trace_payload.get("one_cells")
        if isinstance(one_cells, list):
            for row in one_cells:
                if not isinstance(row, Mapping):
                    continue
                metadata = row.get("metadata")
                if not isinstance(metadata, Mapping):
                    continue
                for key in ("path", "checkpoint_path"):
                    raw_path = metadata.get(key)
                    if isinstance(raw_path, str):
                        _register(
                            raw_path,
                            weight=2,
                            source_label=f"trace.one_cells.metadata.{key}",
                        )

    ranked = sorted(scores.items(), key=lambda item: (-int(item[1]), str(item[0])))
    if limit > 0:
        ranked = ranked[:limit]
    selected_paths = [path for path, _score in ranked]
    selected_scores = {path: int(score) for path, score in ranked}
    selected_sources = {
        path: list(
            sort_once(
                source_set,
                source=f"aspf_action_plan._ranked_candidate_paths.sources.{path}",
            )
        )
        for path, source_set in sources.items()
        if path in selected_scores
    }
    return selected_paths, selected_scores, selected_sources


def _confidence(raw: object) -> float:
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return 0.0
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


def _delta_payload_file_site_paths(delta_payload: Mapping[str, object]) -> list[str]:
    call_stack = delta_payload.get("call_stack")
    if not isinstance(call_stack, Mapping):
        return []
    site_table = call_stack.get("site_table")
    if not isinstance(site_table, list):
        return []
    discovered: list[str] = []
    for entry in site_table:
        for raw_path in _iter_file_site_paths(entry):
            if raw_path:
                discovered.append(raw_path)
    return list(
        sort_once(
            discovered,
            source="aspf_action_plan._delta_payload_file_site_paths",
        )
    )


def _iter_file_site_paths(value: object) -> list[str]:
    out: list[str] = []

    def _walk(node: object) -> None:
        if isinstance(node, Mapping):
            kind = str(node.get("kind", "")).strip()
            if kind == "FileSite":
                raw_key = node.get("key")
                normalized = _file_site_key_to_path(raw_key)
                if normalized:
                    out.append(normalized)
            for nested in node.values():
                _walk(nested)
            return
        if isinstance(node, Sequence) and not isinstance(node, (str, bytes, bytearray)):
            for nested in node:
                _walk(nested)

    _walk(value)
    return out


def _file_site_key_to_path(raw_key: object) -> str | None:
    if isinstance(raw_key, str):
        return raw_key.strip() or None
    if isinstance(raw_key, Sequence) and not isinstance(raw_key, (str, bytes, bytearray)):
        if not raw_key:
            return None
        head = raw_key[0]
        if isinstance(head, str):
            return head.strip() or None
    return None


def _normalize_candidate_path(
    raw_path: str,
    *,
    root: Path | None,
) -> str | None:
    text = raw_path.strip()
    if not text:
        return None
    if text.startswith("file://"):
        text = text[len("file://") :]
    text = text.replace("\\", "/")
    if "<" in text or ">" in text:
        return None
    path = Path(text)
    if root is not None:
        resolved_root = root.resolve()
        if path.is_absolute():
            try:
                path = path.resolve().relative_to(resolved_root)
            except (OSError, ValueError):
                return None
        elif text.startswith("./"):
            path = Path(text[2:])
        if str(path).startswith(".."):
            return None
    elif path.is_absolute():
        return None
    normalized = path.as_posix().strip()
    if not normalized or normalized == "." or normalized.startswith("../"):
        return None
    return normalized


def _priority(confidence: float) -> str:
    if confidence >= 0.80:
        return "high"
    if confidence >= 0.60:
        return "medium"
    return "low"


def _priority_rank(priority: str) -> int:
    if priority == "high":
        return 0
    if priority == "medium":
        return 1
    return 2


def _implementation_steps(kind: str) -> list[str]:
    steps_by_kind: dict[str, list[str]] = {
        "materialize_load_fusion": [
            "Consolidate duplicate checkpoint materialize/load boundaries into one reusable adapter.",
            "Use one canonical serialization shape for both write and read paths.",
            "Re-run check pipeline to confirm equivalent semantic surfaces.",
        ],
        "reusable_boundary_artifact": [
            "Promote shared representative payload into a reusable artifact boundary.",
            "Replace duplicate recomputation with import of the reusable artifact.",
            "Validate non-drift equivalence on affected semantic surfaces.",
        ],
        "fungible_execution_path_substitution": [
            "Substitute equivalent execution path using witness-backed representative mapping.",
            "Remove redundant path while preserving witness linkage.",
            "Confirm unchanged equivalence verdict and opportunity stability.",
        ],
    }
    return steps_by_kind.get(
        kind,
        [
            "Inspect witness and surface evidence.",
            "Apply smallest safe refactor that removes duplicated work.",
            "Re-run targeted checks and verify equivalence remains non-drift.",
        ],
    )


def _validation_commands(kind: str) -> list[str]:
    base = [
        "mise exec -- python -m gabion check run --aspf-state-json artifacts/out/aspf_state/session/step.snapshot.json --aspf-delta-jsonl artifacts/out/aspf_state/session/step.delta.jsonl",
        "mise exec -- python -m pytest tests/test_aspf_execution_fibration.py",
    ]
    if kind == "materialize_load_fusion":
        return base + ["mise exec -- python -m pytest tests/test_run_dataflow_stage.py"]
    if kind == "reusable_boundary_artifact":
        return base + ["mise exec -- python -m pytest tests/test_aspf_handoff.py"]
    if kind == "fungible_execution_path_substitution":
        return base + ["mise exec -- python -m pytest tests/test_aspf.py"]
    return base


def _string_list(value: object) -> list[str]:
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, Sequence):
        out: list[str] = []
        for item in value:
            text = str(item).strip()
            if text:
                out.append(text)
        return out
    return []

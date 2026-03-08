from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path

from gabion.analysis.dataflow.engine.dataflow_analysis_index import _build_analysis_index
from gabion.analysis.dataflow.engine.dataflow_contracts import AuditConfig, FunctionInfo
from gabion.analysis.dataflow.engine.dataflow_evidence_helpers import _resolve_callee
from gabion.analysis.dataflow.engine.dataflow_ingest_helpers import resolve_analysis_paths
from gabion.analysis.core.forest_signature import build_forest_signature_from_groups
from gabion.analysis.foundation.json_types import JSONObject
from gabion.analysis.foundation.timeout_context import check_deadline
from gabion.order_contract import sort_once
from gabion.synthesis.schedule import topological_schedule


def _function_key(scope: Iterable[str], name: str) -> str:
    parts = list(scope)
    parts.append(name)
    return ".".join(parts)


def _partial_forest_signature_metadata(
    groups_by_path: dict[Path, dict[str, list[set[str]]]],
    *,
    basis: str = "bundles_only",
) -> JSONObject:
    return {
        "forest_signature": build_forest_signature_from_groups(groups_by_path),
        "forest_signature_partial": True,
        "forest_signature_basis": basis,
    }


def build_refactor_plan(
    groups_by_path: dict[Path, dict[str, list[set[str]]]],
    paths: list[Path],
    *,
    config: AuditConfig,
) -> JSONObject:
    check_deadline()
    parse_failure_witnesses: list[JSONObject] = []
    signature_meta = _partial_forest_signature_metadata(groups_by_path)
    file_paths = resolve_analysis_paths(paths, config=config)
    if not file_paths:
        payload: JSONObject = {
            "bundles": [],
            "warnings": ["No files available for refactor plan."],
        }
        payload.update(signature_meta)
        return payload

    analysis_index = _build_analysis_index(
        file_paths,
        project_root=config.project_root,
        ignore_params=config.ignore_params,
        strictness=config.strictness,
        external_filter=config.external_filter,
        transparent_decorators=config.transparent_decorators,
        parse_failure_witnesses=parse_failure_witnesses,
    )
    by_name = analysis_index.by_name
    by_qual = analysis_index.by_qual
    symbol_table = analysis_index.symbol_table
    class_index = analysis_index.class_index

    info_by_path_name: dict[tuple[Path, str], FunctionInfo] = {}
    for infos in by_name.values():
        check_deadline()
        for info in infos:
            check_deadline()
            key = _function_key(info.scope, info.name)
            info_by_path_name[(info.path, key)] = info

    bundle_map: dict[tuple[str, ...], dict[str, FunctionInfo]] = defaultdict(dict)
    for path, groups in groups_by_path.items():
        check_deadline()
        for fn_key, bundles in groups.items():
            check_deadline()
            for bundle in bundles:
                check_deadline()
                normalized = tuple(
                    sort_once(
                        bundle,
                        source="dataflow_refactor_planning.build_refactor_plan.bundle",
                    )
                )
                info = info_by_path_name.get((path, fn_key))
                if info is not None:
                    bundle_map[normalized][info.qual] = info

    plans: list[JSONObject] = []
    for bundle, infos in sort_once(
        bundle_map.items(),
        key=lambda item: (len(item[0]), item[0]),
        source="dataflow_refactor_planning.build_refactor_plan.bundle_map",
    ):
        check_deadline()
        components = dict(infos)
        deps: dict[str, set[str]] = {qual: set() for qual in components}
        for info in infos.values():
            check_deadline()
            for call in info.calls:
                check_deadline()
                callee = _resolve_callee(
                    call.callee,
                    info,
                    by_name,
                    by_qual,
                    symbol_table,
                    config.project_root,
                    class_index,
                )
                if (
                    callee is not None
                    and callee.transparent
                    and callee.qual in components
                ):
                    deps[info.qual].add(callee.qual)

        schedule = topological_schedule(deps)
        plans.append(
            {
                "bundle": list(bundle),
                "functions": sort_once(
                    components.keys(),
                    source="dataflow_refactor_planning.build_refactor_plan.functions",
                ),
                "order": schedule.order,
                "cycles": [
                    sort_once(
                        list(cycle),
                        source="dataflow_refactor_planning.build_refactor_plan.cycles",
                    )
                    for cycle in schedule.cycles
                ],
            }
        )

    warnings: list[str] = []
    if not plans:
        warnings.append("No bundle components available for refactor plan.")
    payload = {"bundles": plans, "warnings": warnings}
    payload.update(signature_meta)
    return payload


def render_refactor_plan(plan: JSONObject) -> str:
    check_deadline()
    bundles = plan.get("bundles", [])
    warnings = plan.get("warnings", [])
    lines = ["", "## Refactoring plan (prototype)", ""]
    if not bundles:
        lines.append("No refactoring plan available.")
    else:
        for entry in bundles:
            check_deadline()
            bundle = entry.get("bundle", [])
            title = ", ".join(bundle) if bundle else "(unknown bundle)"
            lines.append(f"### Bundle: {title}")
            order = entry.get("order", [])
            if order:
                lines.append("Order (callee-first):")
                lines.append("```")
                for item in order:
                    check_deadline()
                    lines.append(f"- {item}")
                lines.append("```")
            cycles = entry.get("cycles", [])
            if cycles:
                lines.append("Cycles:")
                lines.append("```")
                for cycle in cycles:
                    check_deadline()
                    lines.append(", ".join(cycle))
                lines.append("```")
    if warnings:
        lines.append("")
        lines.append("Warnings:")
        lines.append("```")
        lines.extend(str(warning) for warning in warnings)
        lines.append("```")
    return "\n".join(lines)


def render_reuse_lemma_stubs(reuse: JSONObject) -> str:
    check_deadline()
    suggested = reuse.get("suggested_lemmas") or []
    lines = [
        "# Generated by gabion structure-reuse",
        "# Structured rewrite-plan artifacts for suggested reuse lemmas.",
        "",
    ]
    if not suggested:
        lines.append("# No lemma suggestions available.")
        lines.append("")
        return "\n".join(lines)

    suggested_entries: list[dict[object, object]] = []
    for entry in suggested:
        check_deadline()
        match entry:
            case dict() as suggested_entry:
                suggested_entries.append(suggested_entry)
            case _:
                continue
    plan_artifacts: list[JSONObject] = []
    for entry in sort_once(
        suggested_entries,
        key=lambda item: (
            str(item.get("kind", "")),
            str(item.get("suggested_name", "")),
        ),
        source="dataflow_refactor_planning.render_reuse_lemma_stubs.suggested",
    ):
        check_deadline()
        name = entry.get("suggested_name")
        match name:
            case str() as suggested_name if suggested_name:
                raw_plan = entry.get("rewrite_plan_artifact")
                match raw_plan:
                    case dict() as plan:
                        plan_artifacts.append({str(key): plan[key] for key in plan})

    payload: JSONObject = {
        "format_version": 1,
        "artifact_kind": "reuse_rewrite_plan_bundle",
        "plans": sort_once(
            plan_artifacts,
            key=lambda plan: str(plan.get("plan_id", "")),
            source="dataflow_refactor_planning.render_reuse_lemma_stubs.plans",
        ),
    }

    lines.append(json.dumps(payload, indent=2, sort_keys=False))
    lines.append("")
    return "\n".join(lines)

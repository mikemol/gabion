# gabion:decision_protocol_module
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from gabion.analysis.foundation.json_types import ParseFailureWitnesses


@dataclass(frozen=True)
class TypeFlowInferDeps:
    check_deadline_fn: Callable[[], None]
    analysis_pass_prerequisites_ctor: Callable[..., object]
    require_not_none_fn: Callable[..., object]
    analysis_index_resolved_call_edges_by_caller_fn: Callable[..., dict[str, tuple[object, ...]]]
    caller_param_bindings_for_call_fn: Callable[..., dict[str, set[str]]]
    function_key_fn: Callable[..., str]
    normalize_snapshot_path_fn: Callable[..., str]
    is_test_path_fn: Callable[..., bool]
    is_broad_type_fn: Callable[..., bool]
    sort_once_fn: Callable[..., list[object]]


def infer_type_flow(
    paths: list[Path],
    *,
    project_root,
    ignore_params: set[str],
    strictness: str,
    external_filter: bool,
    transparent_decorators=None,
    max_sites_per_param: int = 3,
    parse_failure_witnesses: ParseFailureWitnesses,
    analysis_index=None,
    deps: TypeFlowInferDeps,
):
    """Repo-wide fixed-point pass for downstream type tightening + evidence."""
    deps.check_deadline_fn()
    deps.analysis_pass_prerequisites_ctor(
        bundle_inference=True,
        call_propagation=True,
        decision_surfaces=True,
        type_flow=True,
        lint_evidence=True,
    ).validate(pass_id="type_flow")
    index = deps.require_not_none_fn(
        analysis_index,
        reason="_infer_type_flow requires prebuilt analysis_index",
        strict=True,
    )
    by_name = index.by_name
    resolved_edges_by_caller = deps.analysis_index_resolved_call_edges_by_caller_fn(
        index,
        project_root=project_root,
        require_transparent=True,
    )
    inferred: dict[str, dict[str, object]] = {}
    for infos in by_name.values():
        deps.check_deadline_fn()
        for info in infos:
            deps.check_deadline_fn()
            inferred[info.qual] = dict(info.annots)

    def _get_annot(info, param: str):
        value = inferred.get(info.qual, {}).get(param)
        if type(value) is str:
            return value
        return None

    def _downstream_for(info) -> tuple[dict[str, set[str]], dict[str, dict[str, set[str]]]]:
        deps.check_deadline_fn()
        downstream: dict[str, set[str]] = defaultdict(set)
        sites: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
        for edge in resolved_edges_by_caller.get(info.qual, ()):
            deps.check_deadline_fn()
            callee = edge.callee
            call = edge.call
            callee_to_caller = deps.caller_param_bindings_for_call_fn(
                call,
                callee,
                strictness=strictness,
            )
            for callee_param, callers in callee_to_caller.items():
                deps.check_deadline_fn()
                annot = _get_annot(callee, callee_param)
                if not annot:
                    continue
                for caller_param in callers:
                    deps.check_deadline_fn()
                    downstream[caller_param].add(annot)
                    caller_name = deps.function_key_fn(info.scope, info.name)
                    caller_path = deps.normalize_snapshot_path_fn(info.path, project_root)
                    if call.span is None:
                        loc = f"{caller_path}:{caller_name}"
                    else:
                        line, col, _, _ = call.span
                        loc = f"{caller_path}:{line + 1}:{col + 1}"
                    sites[caller_param][annot].add(
                        f"{loc}: {caller_name}.{caller_param} -> {callee.qual}.{callee_param} expects {annot}"
                    )
        return downstream, sites

    changed = True
    while changed:
        deps.check_deadline_fn()
        changed = False
        for infos in by_name.values():
            deps.check_deadline_fn()
            for info in infos:
                deps.check_deadline_fn()
                if deps.is_test_path_fn(info.path):
                    continue
                downstream, _ = _downstream_for(info)
                for param, annots in downstream.items():
                    deps.check_deadline_fn()
                    if len(annots) != 1:
                        continue
                    downstream_annot = next(iter(annots))
                    current = _get_annot(info, param)
                    if deps.is_broad_type_fn(current) and downstream_annot:
                        if inferred[info.qual].get(param) != downstream_annot:
                            inferred[info.qual][param] = downstream_annot
                            changed = True

    suggestions: set[str] = set()
    ambiguities: set[str] = set()
    evidence_lines: set[str] = set()
    for infos in by_name.values():
        deps.check_deadline_fn()
        for info in infos:
            deps.check_deadline_fn()
            if deps.is_test_path_fn(info.path):
                continue
            downstream, sites = _downstream_for(info)
            fn_key = deps.function_key_fn(info.scope, info.name)
            path_key = deps.normalize_snapshot_path_fn(info.path, project_root)
            for param, annots in downstream.items():
                deps.check_deadline_fn()
                if len(annots) > 1:
                    ambiguities.add(
                        f"{path_key}:{fn_key}.{param} downstream types conflict: {deps.sort_once_fn(annots, source = 'gabion.analysis.dataflow_indexed_file_scan._infer_type_flow.site_1')}"
                    )
                    for annot in deps.sort_once_fn(
                        annots,
                        source="gabion.analysis.dataflow_indexed_file_scan._infer_type_flow.site_2",
                    ):
                        deps.check_deadline_fn()
                        for site in deps.sort_once_fn(
                            sites.get(param, {}).get(annot, set()),
                            source="gabion.analysis.dataflow_indexed_file_scan._infer_type_flow.site_3",
                        )[:max_sites_per_param]:
                            deps.check_deadline_fn()
                            evidence_lines.add(site)
                    continue
                downstream_annot = next(iter(annots))
                original = info.annots.get(param)
                final = inferred.get(info.qual, {}).get(param)
                if deps.is_broad_type_fn(original) and final == downstream_annot and downstream_annot:
                    suggestions.add(
                        f"{path_key}:{fn_key}.{param} can tighten to {downstream_annot}"
                    )
                    for site in deps.sort_once_fn(
                        sites.get(param, {}).get(downstream_annot, set()),
                        source="gabion.analysis.dataflow_indexed_file_scan._infer_type_flow.site_4",
                    )[:max_sites_per_param]:
                        deps.check_deadline_fn()
                        evidence_lines.add(site)
    return (
        inferred,
        deps.sort_once_fn(
            suggestions,
            source="gabion.analysis.dataflow_indexed_file_scan._infer_type_flow.site_5",
        ),
        deps.sort_once_fn(
            ambiguities,
            source="gabion.analysis.dataflow_indexed_file_scan._infer_type_flow.site_6",
        ),
        deps.sort_once_fn(
            evidence_lines,
            source="gabion.analysis.dataflow_indexed_file_scan._infer_type_flow.site_7",
        ),
    )

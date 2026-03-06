# gabion:decision_protocol_module
from __future__ import annotations

import ast
from collections import defaultdict
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import cast

from gabion.analysis.aspf.aspf import Forest, Node, NodeId
from gabion.analysis.dataflow.engine.dataflow_contracts import CallArgs, FunctionInfo, OptionalString
from gabion.analysis.foundation.json_types import JSONObject
from gabion.analysis.foundation.timeout_context import check_deadline
from gabion.invariants import never
from gabion.order_contract import sort_once

from gabion.analysis.indexed_scan.deadline.deadline_fallback import fallback_deadline_arg_info as _fallback_deadline_arg_info

OptionalAstCall = ast.Call | None


@dataclass(frozen=True)
class FunctionSuiteKey:
    path: str
    qual: str


class FunctionSuiteLookupStatus(StrEnum):
    RESOLVED = "resolved"
    NODE_MISSING = "node_missing"
    SUITE_KIND_UNSUPPORTED = "suite_kind_unsupported"


@dataclass(frozen=True)
class FunctionSuiteLookupOutcome:
    status: FunctionSuiteLookupStatus
    suite_id: NodeId


@dataclass(frozen=True)
class DeadlineArgInfo:
    kind: str
    param: OptionalString = None
    const: OptionalString = None


def function_suite_key(path: str, qual: str) -> FunctionSuiteKey:
    return FunctionSuiteKey(path=path, qual=qual)


def function_suite_id(key: FunctionSuiteKey) -> NodeId:
    return NodeId("SuiteSite", (key.path, key.qual, "function"))


def node_to_function_suite_lookup_outcome(
    forest: Forest,
    node_id: NodeId,
) -> FunctionSuiteLookupOutcome:
    missing_suite_id = NodeId("MissingSuiteSite", ("", "", ""))
    node = forest.nodes.get(node_id)
    if node is None:
        return FunctionSuiteLookupOutcome(
            FunctionSuiteLookupStatus.NODE_MISSING,
            missing_suite_id,
        )
    if node.kind == "FunctionSite":
        path = str(node.meta.get("path", "") or "")
        qual = str(node.meta.get("qual", "") or "")
        if not path or not qual:
            never("function site missing identity", path=path, qual=qual)
        return FunctionSuiteLookupOutcome(
            FunctionSuiteLookupStatus.RESOLVED,
            function_suite_id(function_suite_key(path, qual)),
        )
    if node.kind == "SuiteSite":
        suite_kind = str(node.meta.get("suite_kind", "") or "")
        if suite_kind in {"function", "function_body"}:
            path = str(node.meta.get("path", "") or "")
            qual = str(node.meta.get("qual", "") or "")
            if not path or not qual:
                never("function suite missing identity", path=path, qual=qual)
            return FunctionSuiteLookupOutcome(
                FunctionSuiteLookupStatus.RESOLVED,
                function_suite_id(function_suite_key(path, qual)),
            )
    return FunctionSuiteLookupOutcome(
        FunctionSuiteLookupStatus.SUITE_KIND_UNSUPPORTED,
        missing_suite_id,
    )


def suite_caller_function_id(
    suite_node: Node,
) -> NodeId:
    path = str(suite_node.meta.get("path", "") or "")
    qual = str(suite_node.meta.get("qual", "") or "")
    if not path or not qual:
        never(
            "suite site missing caller identity",
            suite_kind=str(suite_node.meta.get("suite_kind", "") or ""),
            path=path,
            qual=qual,
        )
    return function_suite_id(function_suite_key(path, qual))


def node_to_function_suite_id(
    forest: Forest,
    node_id: NodeId,
):
    outcome = node_to_function_suite_lookup_outcome(forest, node_id)
    if outcome.status is not FunctionSuiteLookupStatus.RESOLVED:
        return None
    return outcome.suite_id


def _is_test_path(path: Path) -> bool:
    if "tests" in path.parts:
        return True
    return path.name.startswith("test_")


def _callee_key(name: str) -> str:
    if not name:
        return name
    return name.split(".")[-1]


def obligation_candidate_suite_ids(
    *,
    by_name: dict[str, list[FunctionInfo]],
    callee_key: str,
) -> set[NodeId]:
    key = _callee_key(callee_key)
    candidates: set[NodeId] = set()
    for info in by_name.get(key, []):
        check_deadline()
        if _is_test_path(info.path):
            continue
        candidates.add(function_suite_id(function_suite_key(info.path.name, info.qual)))
    return candidates


def collect_call_edges_from_forest(
    forest: Forest,
    *,
    by_name: dict[str, list[FunctionInfo]],
) -> dict[NodeId, set[NodeId]]:
    check_deadline()
    edges: dict[NodeId, set[NodeId]] = defaultdict(set)
    for alt in forest.alts:
        check_deadline()
        if alt.inputs:
            suite_id = alt.inputs[0]
            suite_node = forest.nodes.get(suite_id)
            if suite_node is not None and suite_node.kind == "SuiteSite":
                suite_kind = str(suite_node.meta.get("suite_kind", "") or "")
                if suite_kind == "call":
                    caller_id = suite_caller_function_id(suite_node)
                    if alt.kind == "CallCandidate":
                        if len(alt.inputs) >= 2:
                            candidate_id = node_to_function_suite_id(forest, alt.inputs[1])
                            if candidate_id is not None:
                                edges[caller_id].add(candidate_id)
                    elif alt.kind == "CallResolutionObligation":
                        callee_key = str(alt.evidence.get("callee", "") or "")
                        if callee_key:
                            for candidate_id in obligation_candidate_suite_ids(
                                by_name=by_name,
                                callee_key=callee_key,
                            ):
                                check_deadline()
                                edges[caller_id].add(candidate_id)
    return edges


def _obligation_span(
    *,
    suite_node: Node,
) -> tuple[int, int, int, int]:
    raw_span = suite_node.meta.get("span")
    if type(raw_span) is list and len(raw_span) == 4 and all(type(value) is int for value in raw_span):
        return (int(raw_span[0]), int(raw_span[1]), int(raw_span[2]), int(raw_span[3]))
    caller_path = str(suite_node.meta.get("path", "") or "")
    caller_qual = str(suite_node.meta.get("qual", "") or "")
    never(
        "call resolution obligation requires span",
        path=caller_path,
        qual=caller_qual,
    )


def collect_call_resolution_obligations_from_forest(
    forest: Forest,
) -> list[tuple[NodeId, NodeId, tuple[int, int, int, int], str]]:
    check_deadline()
    obligations: list[tuple[NodeId, NodeId, tuple[int, int, int, int], str]] = []
    seen: set[tuple[NodeId, NodeId, tuple[int, int, int, int], str]] = set()
    for alt in forest.alts:
        check_deadline()
        if alt.kind != "CallResolutionObligation":
            continue
        if not alt.inputs:
            continue
        suite_id = alt.inputs[0]
        suite_node = forest.nodes.get(suite_id)
        suite_kind = suite_node.kind if suite_node is not None else ""
        if suite_kind != "SuiteSite":
            continue
        node_suite_kind = str(suite_node.meta.get("suite_kind", "") or "")
        if node_suite_kind != "call":
            continue
        caller_id = suite_caller_function_id(suite_node)
        span = _obligation_span(suite_node=suite_node)
        callee_key = str(alt.evidence.get("callee", "") or "")
        if not callee_key:
            continue
        record = (caller_id, suite_id, span, callee_key)
        if record in seen:
            continue
        seen.add(record)
        obligations.append(record)
    return obligations


def collect_call_resolution_obligation_details_from_forest(
    forest: Forest,
) -> list[tuple[NodeId, NodeId, tuple[int, int, int, int], str, str]]:
    evidence_by_key: dict[tuple[NodeId, str], JSONObject] = {}
    for alt in forest.alts:
        check_deadline()
        if alt.kind != "CallResolutionObligation" or not alt.inputs:
            continue
        suite_id = alt.inputs[0]
        callee_value = alt.evidence.get("callee")
        callee_key = str(callee_value or "")
        if not callee_key:
            continue
        key = (suite_id, callee_key)
        if key not in evidence_by_key:
            evidence_by_key[key] = alt.evidence

    records: list[tuple[NodeId, NodeId, tuple[int, int, int, int], str, str]] = []
    for caller_id, suite_id, span, callee_key in collect_call_resolution_obligations_from_forest(
        forest
    ):
        check_deadline()
        evidence = evidence_by_key.get((suite_id, callee_key), {})
        obligation_kind = str(evidence.get("kind", "") or "")
        if not obligation_kind:
            obligation_kind = "unresolved_internal_callee"
        records.append((caller_id, suite_id, span, callee_key, obligation_kind))
    return records


def call_candidate_target_site(
    *,
    forest: Forest,
    candidate: FunctionInfo,
) -> NodeId:
    check_deadline()
    if candidate.function_span is None:
        never(
            "call candidate target requires function span",
            path=candidate.path.name,
            qual=candidate.qual,
        )
    return forest.add_suite_site(
        candidate.path.name,
        candidate.qual,
        "function",
        span=candidate.function_span,
    )


def materialize_call_candidates(
    *,
    forest: Forest,
    by_name: dict[str, list[FunctionInfo]],
    by_qual: dict[str, FunctionInfo],
    symbol_table,
    project_root,
    class_index: dict[str, object],
    resolve_callee_outcome_fn: Callable[..., object],
    normalize_snapshot_path_fn: Callable[[Path, object], str],
) -> None:
    check_deadline()
    seen: set[tuple[NodeId, NodeId]] = set()
    obligation_seen: set[NodeId] = set()
    for alt in forest.alts:
        check_deadline()
        if alt.kind != "CallCandidate" or len(alt.inputs) < 2:
            if alt.kind == "CallResolutionObligation" and alt.inputs:
                obligation_seen.add(alt.inputs[0])
            continue
        seen.add((alt.inputs[0], alt.inputs[1]))
    for infos in by_name.values():
        check_deadline()
        for info in infos:
            check_deadline()
            if _is_test_path(info.path):
                continue
            for call in info.calls:
                check_deadline()
                if call.is_test:
                    continue
                resolution = resolve_callee_outcome_fn(
                    call.callee,
                    info,
                    by_name,
                    by_qual,
                    symbol_table=symbol_table,
                    project_root=project_root,
                    class_index=class_index,
                    call=call,
                )
                if call.span is None:
                    if resolution.status != "unresolved_external":
                        never(
                            "call candidate requires span",
                            path=normalize_snapshot_path_fn(info.path, project_root),
                            qual=info.qual,
                            callee=call.callee,
                        )
                    continue
                suite_id = forest.add_suite_site(
                    info.path.name,
                    info.qual,
                    "call",
                    span=call.span,
                )
                if resolution.status in {"resolved", "ambiguous"}:
                    for candidate in resolution.candidates:
                        check_deadline()
                        candidate_id = call_candidate_target_site(
                            forest=forest,
                            candidate=candidate,
                        )
                        key = (suite_id, candidate_id)
                        if key in seen:
                            continue
                        seen.add(key)
                        forest.add_alt(
                            "CallCandidate",
                            (suite_id, candidate_id),
                            evidence={
                                "resolution": resolution.status,
                                "phase": resolution.phase,
                                "callee": resolution.callee_key,
                            },
                        )
                    continue
                obligation_kind_by_status = {
                    "unresolved_internal": "unresolved_internal_callee",
                    "unresolved_dynamic": "unresolved_dynamic_callee",
                }
                obligation_kind = obligation_kind_by_status.get(resolution.status)
                if obligation_kind is not None and suite_id not in obligation_seen:
                    obligation_seen.add(suite_id)
                    forest.add_alt(
                        "CallResolutionObligation",
                        (suite_id,),
                        evidence={
                            "phase": resolution.phase,
                            "callee": call.callee,
                            "kind": obligation_kind,
                        },
                    )


def bind_call_args(
    call_node: ast.Call,
    callee: FunctionInfo,
    *,
    strictness: str,
) -> dict[str, ast.AST]:
    check_deadline()
    pos_params = (
        list(callee.positional_params)
        if callee.positional_params
        else list(callee.params)
    )
    kwonly_params = set(callee.kwonly_params or ())
    named_params = set(pos_params) | kwonly_params
    mapping: dict[str, ast.AST] = {}
    star_args: list[ast.AST] = []
    star_kwargs: list[ast.AST] = []
    for idx, arg in enumerate(call_node.args):
        check_deadline()
        if type(arg) is ast.Starred:
            star_args.append(cast(ast.Starred, arg).value)
            continue
        if idx < len(pos_params):
            mapping[pos_params[idx]] = arg
        elif callee.vararg is not None:
            mapping.setdefault(callee.vararg, arg)
    for kw in call_node.keywords:
        check_deadline()
        if kw.arg is None:
            star_kwargs.append(kw.value)
            continue
        if kw.arg in named_params:
            mapping[kw.arg] = kw.value
        elif callee.kwarg is not None:
            mapping.setdefault(callee.kwarg, kw.value)
    if strictness == "low":
        remaining = [
            param
            for param in sort_once(
                named_params,
                source="indexed_scan.deadline_runtime.bind_call_args.remaining",
            )
            if param not in mapping
        ]
        if len(star_args) == 1 and type(star_args[0]) is ast.Name:
            for param in remaining:
                check_deadline()
                mapping.setdefault(param, star_args[0])
        if len(star_kwargs) == 1 and type(star_kwargs[0]) is ast.Name:
            for param in remaining:
                check_deadline()
                mapping.setdefault(param, star_kwargs[0])
    return mapping


def caller_param_bindings_for_call(
    call: CallArgs,
    callee: FunctionInfo,
    *,
    strictness: str,
) -> dict[str, set[str]]:
    check_deadline()
    pos_params = (
        list(callee.positional_params)
        if callee.positional_params
        else list(callee.params)
    )
    kwonly_params = set(callee.kwonly_params or ())
    named_params = set(pos_params) | kwonly_params
    mapping: dict[str, set[str]] = defaultdict(set)
    mapped_params: set[str] = set()
    call_mapping = call.argument_mapping()
    for pos_idx, caller_param in call_mapping.positional.items():
        check_deadline()
        if pos_idx < len(pos_params):
            callee_param = pos_params[pos_idx]
        elif callee.vararg is not None:
            callee_param = callee.vararg
        else:
            continue
        mapped_params.add(callee_param)
        mapping[callee_param].add(caller_param.value)
    for kw_name, caller_param in call_mapping.keywords.items():
        check_deadline()
        if kw_name in named_params:
            mapped_params.add(kw_name)
            mapping[kw_name].add(caller_param.value)
        elif callee.kwarg is not None:
            mapped_params.add(callee.kwarg)
            mapping[callee.kwarg].add(caller_param.value)
    if strictness == "low":
        remaining = [
            param
            for param in sort_once(
                named_params,
                source="indexed_scan.deadline_runtime.caller_param_bindings_for_call.remaining",
            )
            if param not in mapped_params
        ]
        if callee.vararg is not None and callee.vararg not in mapped_params:
            remaining.append(callee.vararg)
        if callee.kwarg is not None and callee.kwarg not in mapped_params:
            remaining.append(callee.kwarg)
        if len(call_mapping.star_positional) == 1:
            _, star_param = call_mapping.star_positional[0]
            for param in remaining:
                check_deadline()
                mapping[param].add(star_param.value)
        if len(call_mapping.star_keywords) == 1:
            star_param = call_mapping.star_keywords[0]
            for param in remaining:
                check_deadline()
                mapping[param].add(star_param.value)
    return mapping


def is_deadline_origin_call(expr: ast.AST) -> bool:
    if type(expr) is not ast.Call:
        return False
    callee = cast(ast.Call, expr).func
    if type(callee) is ast.Name:
        return cast(ast.Name, callee).id == "Deadline"
    if type(callee) is ast.Attribute:
        attr = cast(ast.Attribute, callee)
        if attr.attr in {"from_timeout", "from_timeout_ms", "from_timeout_ticks"}:
            value = attr.value
            if type(value) is ast.Name and cast(ast.Name, value).id == "Deadline":
                return True
            if type(value) is ast.Attribute and cast(ast.Attribute, value).attr == "Deadline":
                return True
    return False


def classify_deadline_expr(
    expr: ast.AST,
    *,
    alias_to_param: Mapping[str, str],
    origin_vars: set[str],
) -> DeadlineArgInfo:
    expr_type = type(expr)
    if expr_type is ast.Name:
        name = cast(ast.Name, expr).id
        if name in alias_to_param:
            return DeadlineArgInfo(kind="param", param=alias_to_param[name])
        if name in origin_vars:
            return DeadlineArgInfo(kind="origin", param=name)
    if is_deadline_origin_call(expr):
        return DeadlineArgInfo(kind="origin")
    if expr_type is ast.Constant:
        constant_value = cast(ast.Constant, expr).value
        if constant_value is None:
            return DeadlineArgInfo(kind="none")
        return DeadlineArgInfo(kind="const", const=repr(constant_value))
    return DeadlineArgInfo(kind="unknown")


def fallback_deadline_arg_info(
    call: CallArgs,
    callee: FunctionInfo,
    *,
    strictness: str,
) -> dict[str, DeadlineArgInfo]:
    return cast(
        dict[str, DeadlineArgInfo],
        _fallback_deadline_arg_info(
            call,
            callee,
            strictness=strictness,
            deadline_arg_info_factory=DeadlineArgInfo,
            check_deadline_fn=check_deadline,
            sort_once_fn=sort_once,
        ),
    )


def deadline_arg_info_map(
    call: CallArgs,
    callee: FunctionInfo,
    *,
    call_node: OptionalAstCall,
    alias_to_param: Mapping[str, str],
    origin_vars: set[str],
    strictness: str,
) -> dict[str, DeadlineArgInfo]:
    check_deadline()
    if call_node is None:
        return fallback_deadline_arg_info(call, callee, strictness=strictness)
    expr_map = bind_call_args(call_node, callee, strictness=strictness)
    info_map: dict[str, DeadlineArgInfo] = {}
    for param, expr in expr_map.items():
        check_deadline()
        info_map[param] = classify_deadline_expr(
            expr,
            alias_to_param=alias_to_param,
            origin_vars=origin_vars,
        )
    return info_map


def deadline_loop_forwarded_params(
    *,
    qual: str,
    loop_fact: object,
    deadline_params: Mapping[str, set[str]],
    call_infos: Mapping[str, list[tuple[CallArgs, FunctionInfo, dict[str, DeadlineArgInfo]]]],
) -> set[str]:
    forwarded: set[str] = set()
    caller_params = deadline_params.get(qual, set())
    if not caller_params:
        return forwarded
    loop_spans = set(getattr(loop_fact, "call_spans", set()))
    for call, callee, arg_info in call_infos.get(qual, []):
        check_deadline()
        if call.span is not None and call.span in loop_spans:
            for callee_param in deadline_params.get(callee.qual, set()):
                check_deadline()
                info = arg_info.get(callee_param)
                if info is not None and info.kind == "param" and info.param in caller_params:
                    forwarded.add(info.param)
    return forwarded


__all__ = [
    "DeadlineArgInfo",
    "FunctionSuiteKey",
    "FunctionSuiteLookupOutcome",
    "FunctionSuiteLookupStatus",
    "bind_call_args",
    "call_candidate_target_site",
    "caller_param_bindings_for_call",
    "classify_deadline_expr",
    "collect_call_edges_from_forest",
    "collect_call_resolution_obligation_details_from_forest",
    "collect_call_resolution_obligations_from_forest",
    "deadline_arg_info_map",
    "deadline_loop_forwarded_params",
    "fallback_deadline_arg_info",
    "function_suite_id",
    "function_suite_key",
    "is_deadline_origin_call",
    "materialize_call_candidates",
    "node_to_function_suite_id",
    "node_to_function_suite_lookup_outcome",
    "obligation_candidate_suite_ids",
    "suite_caller_function_id",
]

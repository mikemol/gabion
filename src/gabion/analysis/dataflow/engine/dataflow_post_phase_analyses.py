# gabion:boundary_normalization_module
# gabion:decision_protocol_module
from __future__ import annotations

"""Post-phase analysis owner module for WS-5 decomposition."""

import ast
from pathlib import Path
from typing import Sequence, cast

from gabion.analysis.dataflow.engine.dataflow_contracts import (
    CallArgs,
    FunctionInfo,
    InvariantProposition,
)
from gabion.analysis.dataflow.engine.dataflow_lint_helpers import (
    _expand_type_hint as _expand_type_hint_impl,
    _split_top_level as _split_top_level_impl,
)
from gabion.analysis.dataflow.engine.dataflow_resume_paths import (
    normalize_snapshot_path as _normalize_snapshot_path_impl,
)
from gabion.analysis.dataflow.engine.dataflow_resume_serialization import (
    _deserialize_invariants_for_resume,
    _invariant_confidence,
    _invariant_digest,
    _normalize_invariant_proposition,
)
from gabion.analysis.foundation.json_types import JSONObject, JSONValue
from gabion.analysis.foundation.timeout_context import check_deadline
from gabion.analysis.indexed_scan.calls.callsite_evidence import (
    CallsiteEvidenceDeps as _CallsiteEvidenceDeps,
    callsite_evidence_for_bundle as _callsite_evidence_for_bundle_impl,
)
from gabion.analysis.indexed_scan.scanners.materialization.property_hook_manifest import (
    PropertyHookCallableIndexDeps as _PropertyHookCallableIndexDeps,
    PropertyHookManifestDeps as _PropertyHookManifestDeps,
    build_property_hook_callable_index as _build_property_hook_callable_index_impl,
    generate_property_hook_manifest as _generate_property_hook_manifest_impl,
)
from gabion.analysis.semantics.semantic_primitives import SpanIdentity
from gabion.invariants import require_not_none
from gabion.order_contract import sort_once

# Temporary boundary adapters for unmoved post-phase owners.
_BOUNDARY_ADAPTER_LIFECYCLE: dict[str, object] = {
    "actor": "codex",
    "rationale": "WS-5 staged post-phase hard-cut; keep import surface stable while migrating in slices",
    "scope": "dataflow_post_phase_analyses.runtime_delegates",
    "start": "2026-03-04",
    "expiry": "WS-5-D completion",
    "rollback_condition": "post-phase owner extraction complete in canonical module",
    "evidence_links": ["docs/ws5_decomposition_ledger.md"],
}

_LITERAL_EVAL_ERROR_TYPES = (
    SyntaxError,
    ValueError,
    TypeError,
    MemoryError,
    RecursionError,
)

_NONE_TYPES = {"None", "NoneType", "type(None)"}


def _runtime_callable(name: str):
    from gabion.analysis.dataflow.engine import dataflow_indexed_file_scan as _runtime

    return getattr(_runtime, name)


def _function_key(scope, name: str) -> str:
    parts = list(scope)
    parts.append(name)
    return ".".join(parts)


def _normalize_snapshot_path(path: Path, root) -> str:
    return _normalize_snapshot_path_impl(path, root)


def _type_from_const_repr(value: str):
    try:
        literal = ast.literal_eval(value)
    except _LITERAL_EVAL_ERROR_TYPES:
        return None
    literal_type = type(literal)
    if literal is None:
        return "None"
    if literal_type is bool:
        return "bool"
    if literal_type is int:
        return "int"
    if literal_type is float:
        return "float"
    if literal_type is complex:
        return "complex"
    if literal_type is str:
        return "str"
    if literal_type is bytes:
        return "bytes"
    if literal_type is list:
        return "list"
    if literal_type is tuple:
        return "tuple"
    if literal_type is set:
        return "set"
    if literal_type is dict:
        return "dict"
    return None


def _split_top_level(value: str, sep: str) -> list[str]:
    return _split_top_level_impl(value, sep)


def _expand_type_hint(hint: str) -> set[str]:
    return _expand_type_hint_impl(hint)


def _combine_type_hints(types: set[str]) -> tuple[str, bool]:
    check_deadline()
    normalized_sets = []
    for hint in types:
        check_deadline()
        expanded = _expand_type_hint(hint)
        normalized_sets.append(
            tuple(
                sort_once(
                    (t for t in expanded if t not in _NONE_TYPES),
                    source="gabion.analysis.dataflow_post_phase_analyses._combine_type_hints.site_1",
                )
            )
        )
    unique_normalized = {norm for norm in normalized_sets if norm}
    expanded: set[str] = set()
    for hint in types:
        check_deadline()
        expanded.update(_expand_type_hint(hint))
    none_types = {t for t in expanded if t in _NONE_TYPES}
    expanded -= none_types
    if not expanded:
        return "Any", bool(types)
    sorted_types = sort_once(
        expanded,
        source="gabion.analysis.dataflow_post_phase_analyses._combine_type_hints.site_2",
    )
    if len(sorted_types) == 1:
        base = sorted_types[0]
        if none_types:
            conflicted = len(unique_normalized) > 1
            return f"Optional[{base}]", conflicted
        return base, len(unique_normalized) > 1
    union = f"Union[{', '.join(sorted_types)}]"
    if none_types:
        return f"Optional[{union}]", len(unique_normalized) > 1
    return union, len(unique_normalized) > 1


def _format_call_site(caller: FunctionInfo, call: CallArgs) -> str:
    """Render a stable, human-friendly call site identifier."""
    caller_name = _function_key(caller.scope, caller.name)
    span = call.span
    if span is None:
        return f"{caller.path.name}:{caller_name}"
    line, col, _, _ = span
    return f"{caller.path.name}:{line + 1}:{col + 1}:{caller_name}"


def _format_type_flow_site(
    *,
    caller: FunctionInfo,
    call: CallArgs,
    callee: FunctionInfo,
    caller_param: str,
    callee_param: str,
    annot: str,
    project_root,
) -> str:
    """Format a stable, machine-actionable callsite for type-flow evidence."""
    caller_name = _function_key(caller.scope, caller.name)
    caller_path = _normalize_snapshot_path(caller.path, project_root)
    if call.span is None:
        loc = f"{caller_path}:{caller_name}"
    else:
        line, col, _, _ = call.span
        loc = f"{caller_path}:{line + 1}:{col + 1}"
    return (
        f"{loc}: {caller_name}.{caller_param} -> {callee.qual}.{callee_param} expects {annot}"
    )


def _callsite_evidence_for_bundle(
    calls: list[CallArgs],
    bundle: set[str],
    *,
    limit: int = 12,
) -> list[JSONObject]:
    return cast(
        list[JSONObject],
        _callsite_evidence_for_bundle_impl(
            calls,
            bundle,
            limit=limit,
            deps=_CallsiteEvidenceDeps(
                check_deadline_fn=check_deadline,
                sort_once_fn=sort_once,
                require_not_none_fn=require_not_none,
                span_identity_from_tuple_fn=SpanIdentity.from_tuple,
            ),
        ),
    )


def generate_property_hook_manifest(
    invariants: Sequence[InvariantProposition],
    *,
    min_confidence: float = 0.7,
    emit_hypothesis_templates: bool = False,
) -> JSONObject:
    return _generate_property_hook_manifest_impl(
        invariants,
        min_confidence=min_confidence,
        emit_hypothesis_templates=emit_hypothesis_templates,
        deps=_PropertyHookManifestDeps(
            check_deadline_fn=check_deadline,
            sort_once_fn=sort_once,
            invariant_confidence_fn=_invariant_confidence,
            normalize_invariant_proposition_fn=_normalize_invariant_proposition,
            invariant_digest_fn=_invariant_digest,
        ),
    )


def _build_property_hook_callable_index(hooks: Sequence[JSONValue]) -> list[JSONObject]:
    return _build_property_hook_callable_index_impl(
        hooks,
        deps=_PropertyHookCallableIndexDeps(
            check_deadline_fn=check_deadline,
            sort_once_fn=sort_once,
        ),
    )


def analyze_type_flow_repo_with_map(*args, **kwargs):
    return _runtime_callable("analyze_type_flow_repo_with_map")(*args, **kwargs)


def analyze_type_flow_repo_with_evidence(*args, **kwargs):
    return _runtime_callable("analyze_type_flow_repo_with_evidence")(*args, **kwargs)


def analyze_constant_flow_repo(*args, **kwargs):
    return _runtime_callable("analyze_constant_flow_repo")(*args, **kwargs)


def analyze_deadness_flow_repo(*args, **kwargs):
    return _runtime_callable("analyze_deadness_flow_repo")(*args, **kwargs)


def analyze_unused_arg_flow_repo(*args, **kwargs):
    return _runtime_callable("analyze_unused_arg_flow_repo")(*args, **kwargs)


def _collect_constant_flow_details(*args, **kwargs):
    return _runtime_callable("_collect_constant_flow_details")(*args, **kwargs)


def _collect_exception_obligations(*args, **kwargs):
    return _runtime_callable("_collect_exception_obligations")(*args, **kwargs)


def _collect_handledness_witnesses(*args, **kwargs):
    return _runtime_callable("_collect_handledness_witnesses")(*args, **kwargs)


def _collect_never_invariants(*args, **kwargs):
    return _runtime_callable("_collect_never_invariants")(*args, **kwargs)


def _collect_invariant_propositions(*args, **kwargs):
    return _runtime_callable("_collect_invariant_propositions")(*args, **kwargs)


def _param_annotations_by_path(*args, **kwargs):
    return _runtime_callable("_param_annotations_by_path")(*args, **kwargs)


def analyze_decision_surfaces_repo(*args, **kwargs):
    return _runtime_callable("analyze_decision_surfaces_repo")(*args, **kwargs)


def analyze_value_encoded_decisions_repo(*args, **kwargs):
    return _runtime_callable("analyze_value_encoded_decisions_repo")(*args, **kwargs)


def _compute_knob_param_names(*args, **kwargs):
    return _runtime_callable("_compute_knob_param_names")(*args, **kwargs)


def _collect_config_bundles(*args, **kwargs):
    return _runtime_callable("_collect_config_bundles")(*args, **kwargs)


def _iter_config_fields(*args, **kwargs):
    return _runtime_callable("_iter_config_fields")(*args, **kwargs)


def _collect_dataclass_registry(*args, **kwargs):
    return _runtime_callable("_collect_dataclass_registry")(*args, **kwargs)


def _iter_dataclass_call_bundles(*args, **kwargs):
    return _runtime_callable("_iter_dataclass_call_bundles")(*args, **kwargs)


__all__ = [
    "_build_property_hook_callable_index",
    "_callsite_evidence_for_bundle",
    "_collect_config_bundles",
    "_collect_constant_flow_details",
    "_collect_dataclass_registry",
    "_collect_exception_obligations",
    "_collect_handledness_witnesses",
    "_collect_invariant_propositions",
    "_collect_never_invariants",
    "_combine_type_hints",
    "_compute_knob_param_names",
    "_deserialize_invariants_for_resume",
    "_expand_type_hint",
    "_format_call_site",
    "_format_type_flow_site",
    "_iter_config_fields",
    "_iter_dataclass_call_bundles",
    "_param_annotations_by_path",
    "_split_top_level",
    "_type_from_const_repr",
    "analyze_constant_flow_repo",
    "analyze_deadness_flow_repo",
    "analyze_decision_surfaces_repo",
    "analyze_type_flow_repo_with_evidence",
    "analyze_type_flow_repo_with_map",
    "analyze_unused_arg_flow_repo",
    "analyze_value_encoded_decisions_repo",
    "generate_property_hook_manifest",
]

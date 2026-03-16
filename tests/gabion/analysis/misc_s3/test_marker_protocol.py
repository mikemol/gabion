from __future__ import annotations

import ast
from dataclasses import dataclass

import pytest

from gabion.analysis.foundation.marker_protocol import (
    MarkerKind,
    MarkerKindProfile,
    MarkerLifecycleState,
    MarkerReasoning,
    marker_identity,
    marker_kind_mapping_config,
    normalize_marker_payload,
    normalize_marker_reasoning,
    normalize_semantic_links,
    runtime_marker_kind_mapping_scope,
    resolve_marker_kind_for_profile,
)
from gabion.analysis.indexed_scan.scanners import marker_metadata
from gabion.exceptions import NeverThrown
from gabion.invariants import (
    InvariantMarkerWarning,
    InvariantProfile,
    InvariantRuntimeBehaviorConfig,
    deprecated,
    invariant_factory,
    invariant_runtime_behavior_scope,
    never,
    todo,
)


# gabion:evidence E:function_site::marker_protocol.py::gabion.analysis.marker_protocol.marker_identity
# gabion:behavior primary=desired
def test_marker_identity_is_deterministic() -> None:
    payload = normalize_marker_payload(
        reason="boom",
        owner="platform",
        links=[
            {"kind": "doc_id", "value": "in-46"},
            {"kind": "policy_id", "value": "NCI-LSP-FIRST"},
        ],
    )
    assert marker_identity(payload) == marker_identity(payload)


# gabion:evidence E:function_site::marker_protocol.py::gabion.analysis.marker_protocol.marker_identity
# gabion:behavior primary=desired
def test_marker_identity_changes_when_reasoning_fields_change() -> None:
    base_payload = normalize_marker_payload(
        reason="boom",
        reasoning={
            "summary": "boom",
            "control": "branch-a",
            "blocking_dependencies": ["dep-a", "dep-b"],
        },
    )
    changed_summary = normalize_marker_payload(
        reason="boom",
        reasoning={
            "summary": "boom-2",
            "control": "branch-a",
            "blocking_dependencies": ["dep-a", "dep-b"],
        },
    )
    changed_control = normalize_marker_payload(
        reason="boom",
        reasoning={
            "summary": "boom",
            "control": "branch-b",
            "blocking_dependencies": ["dep-a", "dep-b"],
        },
    )
    changed_dependencies = normalize_marker_payload(
        reason="boom",
        reasoning={
            "summary": "boom",
            "control": "branch-a",
            "blocking_dependencies": ["dep-a", "dep-c"],
        },
    )

    base_identity = marker_identity(base_payload)
    assert marker_identity(changed_summary) != base_identity
    assert marker_identity(changed_control) != base_identity
    assert marker_identity(changed_dependencies) != base_identity


# gabion:evidence E:function_site::marker_protocol.py::gabion.analysis.marker_protocol.normalize_marker_reasoning
@dataclass(frozen=True)
class _ReasoningInput:
    summary: str
    control: str
    blocking_dependencies: tuple[str, ...]


# gabion:evidence E:function_site::marker_protocol.py::gabion.analysis.marker_protocol.normalize_marker_reasoning
# gabion:behavior primary=desired
def test_normalize_marker_reasoning_supports_dataclass_mapping_and_scalar() -> None:
    dataclass_reasoning = normalize_marker_reasoning(
        _ReasoningInput(
            summary="  summary  ",
            control="  guard  ",
            blocking_dependencies=("dep-b", " dep-a ", "dep-a"),
        )
    )
    assert dataclass_reasoning == MarkerReasoning(
        summary="summary",
        control="guard",
        blocking_dependencies=("dep-a", "dep-b"),
    )

    mapping_reasoning = normalize_marker_reasoning(
        {
            "summary": "  mapped ",
            "control": "  flow  ",
            "blocking_dependencies": ["dep-c", " dep-a ", "dep-c"],
        }
    )
    assert mapping_reasoning.blocking_dependencies == ("dep-a", "dep-c")

    scalar_dependency_reasoning = normalize_marker_reasoning(
        {
            "summary": "single dependency",
            "blocking_dependencies": " dep-z ",
        }
    )
    assert scalar_dependency_reasoning.blocking_dependencies == ("dep-z",)

    scalar_reasoning = normalize_marker_reasoning("  scalar reason  ")
    assert scalar_reasoning == MarkerReasoning(
        summary="scalar reason",
        control="",
        blocking_dependencies=(),
    )


def test_normalize_marker_payload_preserves_landed_lifecycle_state() -> None:
    payload = normalize_marker_payload(
        reason="closed item",
        marker_kind=MarkerKind.TODO,
        lifecycle_state=MarkerLifecycleState.LANDED,
        reasoning={
            "summary": "closed item",
            "control": "closure.integrity",
            "blocking_dependencies": [],
        },
    )
    assert payload.lifecycle_state is MarkerLifecycleState.LANDED
    assert payload.reasoning.blocking_dependencies == ()


# gabion:evidence E:function_site::marker_protocol.py::gabion.analysis.marker_protocol.never_marker_payload
# gabion:behavior primary=verboten facets=never
def test_never_carries_marker_payload() -> None:
    with pytest.raises(NeverThrown) as exc_info:
        never(
            "broken",
            links=[{"kind": "doc_id", "value": "in-46"}],
            owner="core",
        )
    assert exc_info.value.marker_kind == "never"
    payload = exc_info.value.marker_payload_dict
    assert payload["reason"] == "broken"
    assert payload["owner"] == "core"


# gabion:evidence E:function_site::marker_protocol.py::gabion.analysis.marker_protocol.normalize_semantic_links
# gabion:behavior primary=desired
def test_normalize_semantic_links_filters_unknown_kinds() -> None:
    links = normalize_semantic_links(
        (
            {"kind": "doc_id", "value": "in-46"},
            {"kind": "unknown", "value": "x"},
            {"kind": "policy_id", "value": ""},
        )
    )
    assert tuple((link.kind.value, link.value) for link in links) == (
        ("doc_id", "in-46"),
    )


# gabion:evidence E:function_site::marker_protocol.py::gabion.analysis.marker_protocol.normalize_marker_payload


# gabion:evidence E:function_site::invariants.py::gabion.invariants.invariant_factory
# gabion:behavior primary=desired
def test_invariant_factory_applies_marker_specific_payload_defaults() -> None:
    with pytest.raises(NeverThrown) as never_exc:
        invariant_factory("never", reasoning={"summary": "structured never"})
    assert never_exc.value.marker_kind == "never"
    assert never_exc.value.marker_payload.reason == "structured never"

    todo_payload = invariant_factory("todo")
    assert todo_payload.marker_kind is MarkerKind.TODO
    assert todo_payload.reason == "todo() marker reached"

    with pytest.raises(NeverThrown) as deprecated_exc:
        invariant_factory("deprecated")
    assert deprecated_exc.value.marker_kind == "deprecated"
    assert deprecated_exc.value.marker_payload.reason == "deprecated() marker reached"


# gabion:behavior primary=allowed_unwanted facets=deprecated
def test_todo_and_deprecated_markers_carry_kind() -> None:
    todo_payload = todo("later", links=[{"kind": "doc_id", "value": "in-50"}])
    assert todo_payload.marker_kind is MarkerKind.TODO

    with pytest.raises(NeverThrown) as deprecated_exc:
        deprecated("legacy", links=[{"kind": "policy_id", "value": "NCI-LSP-FIRST"}])
    assert deprecated_exc.value.marker_kind == "deprecated"


def _call(source: str) -> ast.Call:
    node = ast.parse(source, mode="eval").body
    assert isinstance(node, ast.Call)
    return node


def _check_deadline() -> None:
    return None


def _sort_once(values, *, key=None, **_kwargs):
    return sorted(values, key=key)


# gabion:evidence E:function_site::marker_protocol.py::gabion.analysis.marker_protocol.resolve_marker_kind_for_profile
# gabion:behavior primary=desired
def test_marker_kind_profile_native_preserves_extracted_kind() -> None:
    profile = marker_kind_mapping_config(MarkerKindProfile.NATIVE)
    assert resolve_marker_kind_for_profile(MarkerKind.TODO, mapping_config=profile) is MarkerKind.TODO
    assert resolve_marker_kind_for_profile(MarkerKind.DEPRECATED, mapping_config=profile) is MarkerKind.DEPRECATED


# gabion:evidence E:function_site::marker_protocol.py::gabion.analysis.marker_protocol.resolve_marker_kind_for_profile
# gabion:behavior primary=verboten facets=never
def test_marker_kind_profile_collapse_to_never_remaps_non_never_kinds() -> None:
    profile = marker_kind_mapping_config(MarkerKindProfile.COLLAPSE_TO_NEVER)
    assert resolve_marker_kind_for_profile(MarkerKind.NEVER, mapping_config=profile) is MarkerKind.NEVER
    assert resolve_marker_kind_for_profile(MarkerKind.TODO, mapping_config=profile) is MarkerKind.NEVER
    assert resolve_marker_kind_for_profile(MarkerKind.DEPRECATED, mapping_config=profile) is MarkerKind.NEVER


# gabion:behavior primary=desired
def test_runtime_marker_behavior_is_independent_from_marker_kind_mapping_profile() -> None:
    collapse_profile = marker_kind_mapping_config(MarkerKindProfile.COLLAPSE_TO_NEVER)

    with runtime_marker_kind_mapping_scope(collapse_profile):
        with invariant_runtime_behavior_scope(
            InvariantRuntimeBehaviorConfig(profile=InvariantProfile.DEBT_GATE)
        ):
            with pytest.warns(InvariantMarkerWarning):
                payload = todo("debt gate marker")
    assert payload.marker_kind is MarkerKind.TODO

    with runtime_marker_kind_mapping_scope(collapse_profile):
        with invariant_runtime_behavior_scope(
            InvariantRuntimeBehaviorConfig(profile=InvariantProfile.DIAGNOSTIC)
        ):
            with pytest.warns(InvariantMarkerWarning):
                payload = todo("diagnostic marker")
    assert payload.marker_kind is MarkerKind.TODO


# gabion:evidence E:function_site::indexed_scan/marker_metadata.py::gabion.analysis.indexed_scan.marker_metadata.never_marker_metadata
# gabion:behavior primary=desired
def test_marker_metadata_site_identity_fields_stable_across_kind_remaps() -> None:
    call = _call("todo(reason='defer', owner='team')")
    native = marker_metadata.never_marker_metadata(
        call,
        "never:mod.py:f:1:1",
        "defer",
        marker_kind=MarkerKind.TODO,
        marker_kind_mapping=marker_kind_mapping_config(MarkerKindProfile.NATIVE),
        check_deadline_fn=_check_deadline,
        sort_once_fn=_sort_once,
    )
    collapsed = marker_metadata.never_marker_metadata(
        call,
        "never:mod.py:f:1:1",
        "defer",
        marker_kind=MarkerKind.TODO,
        marker_kind_mapping=marker_kind_mapping_config(MarkerKindProfile.COLLAPSE_TO_NEVER),
        check_deadline_fn=_check_deadline,
        sort_once_fn=_sort_once,
    )

    assert native["marker_kind"] == MarkerKind.TODO.value
    assert collapsed["marker_kind"] == MarkerKind.NEVER.value
    assert native["marker_site_id"] == collapsed["marker_site_id"]
    assert native["owner"] == collapsed["owner"]
    assert native["expiry"] == collapsed["expiry"]
    assert native["links"] == collapsed["links"]

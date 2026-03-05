from __future__ import annotations

from dataclasses import dataclass

import pytest

from gabion.analysis.foundation.marker_protocol import (
    MarkerReasoning,
    marker_identity,
    normalize_marker_payload,
    normalize_marker_reasoning,
    normalize_semantic_links,
)
from gabion.exceptions import NeverThrown
from gabion.invariants import deprecated, invariant_factory, never, todo


# gabion:evidence E:function_site::marker_protocol.py::gabion.analysis.marker_protocol.marker_identity
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

    scalar_reasoning = normalize_marker_reasoning("  scalar reason  ")
    assert scalar_reasoning == MarkerReasoning(
        summary="scalar reason",
        control="",
        blocking_dependencies=(),
    )


# gabion:evidence E:function_site::marker_protocol.py::gabion.analysis.marker_protocol.never_marker_payload
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
def test_invariant_factory_applies_marker_specific_payload_defaults() -> None:
    with pytest.raises(NeverThrown) as never_exc:
        invariant_factory("never", reasoning={"summary": "structured never"})
    assert never_exc.value.marker_kind == "never"
    assert never_exc.value.marker_payload.reason == "structured never"

    with pytest.raises(NeverThrown) as todo_exc:
        invariant_factory("todo")
    assert todo_exc.value.marker_kind == "todo"
    assert todo_exc.value.marker_payload.reason == "todo() marker reached"

    with pytest.raises(NeverThrown) as deprecated_exc:
        invariant_factory("deprecated")
    assert deprecated_exc.value.marker_kind == "deprecated"
    assert deprecated_exc.value.marker_payload.reason == "deprecated() marker reached"


def test_todo_and_deprecated_markers_carry_kind() -> None:
    with pytest.raises(NeverThrown) as todo_exc:
        todo("later", links=[{"kind": "doc_id", "value": "in-50"}])
    assert todo_exc.value.marker_kind == "todo"

    with pytest.raises(NeverThrown) as deprecated_exc:
        deprecated("legacy", links=[{"kind": "policy_id", "value": "NCI-LSP-FIRST"}])
    assert deprecated_exc.value.marker_kind == "deprecated"

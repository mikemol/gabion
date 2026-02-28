from __future__ import annotations

import pytest

from gabion.analysis.marker_protocol import marker_identity, normalize_marker_payload, normalize_semantic_links
from gabion.invariants import deprecated, never, todo
from gabion.exceptions import NeverThrown


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
    assert tuple((link.kind.value, link.value) for link in links) == (("doc_id", "in-46"),)


# gabion:evidence E:function_site::marker_protocol.py::gabion.analysis.marker_protocol.normalize_marker_payload
def test_todo_and_deprecated_markers_carry_kind() -> None:
    with pytest.raises(NeverThrown) as todo_exc:
        todo("later", links=[{"kind": "doc_id", "value": "in-50"}])
    assert todo_exc.value.marker_kind == "todo"

    with pytest.raises(NeverThrown) as deprecated_exc:
        deprecated("legacy", links=[{"kind": "policy_id", "value": "NCI-LSP-FIRST"}])
    assert deprecated_exc.value.marker_kind == "deprecated"

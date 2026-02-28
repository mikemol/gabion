from __future__ import annotations

import pytest

from gabion.analysis.marker_protocol import marker_identity, normalize_marker_payload
from gabion.invariants import never
from gabion.exceptions import NeverThrown


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

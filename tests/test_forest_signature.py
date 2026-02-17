from __future__ import annotations

import pytest

from gabion.analysis.aspf import Forest
from gabion.analysis.forest_signature import (
    build_forest_signature,
    build_forest_signature_from_groups,
    build_forest_signature_payload,
    _path_name,
    _normalize_key,
)
from gabion.exceptions import NeverThrown


# gabion:evidence E:function_site::forest_signature.py::gabion.analysis.forest_signature.build_forest_signature
def test_forest_signature_deterministic() -> None:
    forest = Forest()
    site_id = forest.add_site("a.py", "f")
    paramset_id = forest.add_paramset(["x", "y"])
    forest.add_alt("SignatureBundle", (site_id, paramset_id))
    sig1 = build_forest_signature(forest)
    sig2 = build_forest_signature(forest)
    assert sig1 == sig2


# gabion:evidence E:function_site::forest_signature.py::gabion.analysis.forest_signature.build_forest_signature_from_groups
def test_forest_signature_from_groups() -> None:
    groups_by_path = {
        "a.py": {"f": [set(["a", "b"])]},
        "b.py": {"g": [set(["c"])]},
    }
    signature = build_forest_signature_from_groups(groups_by_path)
    assert signature["nodes"]["count"] > 0
    assert signature["alts"]["count"] > 0


def test_forest_signature_can_emit_legacy_and_fingerprint_intern_payloads() -> None:
    forest = Forest()
    forest.add_site("a.py", "f")

    signature = build_forest_signature_payload(
        forest,
        include_legacy_intern=True,
        include_fingerprint_intern=True,
    )

    assert "intern" in signature["nodes"]
    assert "intern_fingerprint" in signature["nodes"]
    assert signature["nodes"]["count"] == len(signature["nodes"]["intern"])
    assert signature["nodes"]["count"] == len(signature["nodes"]["intern_fingerprint"])


def test_forest_signature_can_emit_fingerprint_intern_without_legacy_intern() -> None:
    forest = Forest()
    forest.add_site("a.py", "f")

    signature = build_forest_signature_payload(
        forest,
        include_legacy_intern=False,
        include_fingerprint_intern=True,
    )

    assert "intern" not in signature["nodes"]
    assert "intern_fingerprint" in signature["nodes"]
    assert signature["nodes"]["count"] == len(signature["nodes"]["intern_fingerprint"])


def test_forest_signature_from_groups_rejects_path_order_regression() -> None:
    with pytest.raises(NeverThrown):
        build_forest_signature_from_groups(
            {
                "b.py": {"g": [set(["c"])]},
                "a.py": {"f": [set(["a", "b"])]},
            }
        )


# gabion:evidence E:function_site::forest_signature.py::gabion.analysis.forest_signature._normalize_key
def test_normalize_key_handles_objects() -> None:
    class Dummy:
        def __str__(self) -> str:
            return "dummy"

    assert _normalize_key([Dummy()]) == ["dummy"]


# gabion:evidence E:function_site::forest_signature.py::gabion.analysis.forest_signature._path_name
def test_path_name_falls_back_to_str() -> None:
    class Dummy:
        name = 123

        def __str__(self) -> str:
            return "fallback"

    assert _path_name(Dummy()) == "fallback"

from __future__ import annotations

from gabion.analysis.aspf import Forest
from gabion.analysis.forest_signature import (
    build_forest_signature,
    build_forest_signature_from_groups,
)


def test_forest_signature_deterministic() -> None:
    forest = Forest()
    site_id = forest.add_site("a.py", "f")
    paramset_id = forest.add_paramset(["x", "y"])
    forest.add_alt("SignatureBundle", (site_id, paramset_id))
    sig1 = build_forest_signature(forest)
    sig2 = build_forest_signature(forest)
    assert sig1 == sig2


def test_forest_signature_from_groups() -> None:
    groups_by_path = {
        "a.py": {"f": [set(["a", "b"])]},
        "b.py": {"g": [set(["c"])]},
    }
    signature = build_forest_signature_from_groups(groups_by_path)
    assert signature["nodes"]["count"] > 0
    assert signature["alts"]["count"] > 0

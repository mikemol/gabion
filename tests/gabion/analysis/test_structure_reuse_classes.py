from __future__ import annotations

from gabion.analysis.aspf import NodeId
from gabion.analysis.structure_reuse_classes import AspfStructureClass


def test_structure_class_key_payload_matches_canonical_identity_payload() -> None:
    structure = AspfStructureClass(
        kind="bundle",
        node_id=NodeId(kind="Reuse:bundle", key=("alpha",)),
        child_fingerprints=("left", "right"),
    )
    assert structure.key_payload() == structure.canonical_identity_payload()

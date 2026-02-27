# gabion:boundary_normalization_module
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Iterable

from gabion.analysis.aspf import NodeId, structural_key_atom, structural_key_json
from gabion.analysis.json_types import JSONObject


@dataclass(frozen=True)
class AspfStructureClass:
    """Canonical ASPF-aligned structure class for subtree reuse grouping."""

    kind: str
    node_id: NodeId
    child_fingerprints: tuple[str, ...]

    def canonical_identity_payload(self) -> JSONObject:
        return {
            "kind": self.kind,
            "node_id": self.node_id.as_dict(),
            "node_fingerprint": [self.node_id.fingerprint()[0], list(self.node_id.fingerprint()[1])],
            "child_fingerprints": list(self.child_fingerprints),
        }

    def key_payload(self) -> JSONObject:
        return self.canonical_identity_payload()

    def digest(self) -> str:
        payload = self.canonical_identity_payload()
        return hashlib.sha1(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()


def build_structure_class(
    *,
    kind: str,
    value: object,
    child_hashes: Iterable[str],
) -> AspfStructureClass:
    atom = structural_key_atom(value, source="structure_reuse_classes.value")
    node_id = NodeId(kind=f"Reuse:{kind}", key=(atom,))
    return AspfStructureClass(
        kind=kind,
        node_id=node_id,
        child_fingerprints=tuple(child_hashes),
    )


def structure_class_payload(structure_class: AspfStructureClass) -> JSONObject:
    canonical_identity = structure_class.canonical_identity_payload()
    canonical_identity["node_id"] = {
        "kind": structure_class.node_id.kind,
        "key": structural_key_json(structure_class.node_id.key),
    }
    payload: JSONObject = {
        "kind": structure_class.kind,
        "canonical_identity": canonical_identity,
        # Compatibility surface for older consumers that still select by digest.
        "digest": structure_class.digest(),
    }
    payload["node_id"] = canonical_identity["node_id"]
    payload["node_fingerprint"] = canonical_identity["node_fingerprint"]
    payload["child_fingerprints"] = canonical_identity["child_fingerprints"]
    return payload

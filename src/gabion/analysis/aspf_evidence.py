# gabion:boundary_normalization_module
from __future__ import annotations

from typing import cast

from gabion.runtime import stable_encode


def normalize_alt_evidence_payload(evidence: object) -> dict[str, object]:
    """Canonicalize Alt evidence shape and ordering for ASPF alternatives."""

    payload = evidence if evidence is not None else {}
    canonical = stable_encode.stable_json_value(
        payload,
        source="aspf_evidence.normalize_alt_evidence_payload",
    )
    match canonical:
        case dict() as payload_map:
            return cast(dict[str, object], payload_map)
        case _:
            return {}

from __future__ import annotations

from gabion.analysis.foundation.json_types import JSONObject


def apply_forest_signature_metadata(
    payload: JSONObject,
    snapshot: JSONObject,
    *,
    prefix: str = "",
) -> None:
    signature = snapshot.get("forest_signature")
    if signature is not None:
        payload[f"{prefix}forest_signature"] = signature
    partial = snapshot.get("forest_signature_partial")
    if partial is not None:
        payload[f"{prefix}forest_signature_partial"] = partial
    basis = snapshot.get("forest_signature_basis")
    if basis is not None:
        payload[f"{prefix}forest_signature_basis"] = basis
    if signature is None:
        payload[f"{prefix}forest_signature_partial"] = True
        if basis is None:
            payload[f"{prefix}forest_signature_basis"] = "missing"

# gabion:boundary_normalization_module
from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
import hashlib
import json
from typing import Mapping

from gabion.analysis.json_types import JSONObject, JSONValue
from gabion.analysis.resume_codec import mapping_or_none, sequence_or_none
from gabion.analysis.timeout_context import check_deadline
from gabion.order_contract import sort_once


class IdentityAxis(StrEnum):
    SCHEMA = "schema"


IDENTITY_CONTRACT_VERSION = "identity_contract.v1"


@dataclass(frozen=True)
class IdentityContract:
    axis: IdentityAxis
    kind: str
    version: str
    canonical_payload: str
    digest: str


def _is_sortable_signature_scalar(value: JSONValue) -> bool:
    return value is None or type(value) in {str, int, float, bool}


def _normalize_identity_value(value: JSONValue) -> JSONValue:
    check_deadline()
    mapping_value = mapping_or_none(value)
    if mapping_value is not None:
        normalized: JSONObject = {}
        for key in sort_once(
            mapping_value,
            source="identity_contract._normalize_identity_value.dict_keys",
        ):
            check_deadline()
            normalized[str(key)] = _normalize_identity_value(mapping_value[key])
        return normalized
    sequence_value = sequence_or_none(value, allow_str=False)
    if sequence_value is not None:
        normalized_items = [_normalize_identity_value(item) for item in sequence_value]
        sortable = True
        for item in normalized_items:
            check_deadline()
            if not _is_sortable_signature_scalar(item):
                sortable = False
                break
        if sortable:
            return sort_once(
                normalized_items,
                key=lambda item: json.dumps(item, sort_keys=False, separators=(",", ":")),
                source="identity_contract._normalize_identity_value.list_items",
            )
        return normalized_items
    return value


def normalize_identity_mapping(payload: Mapping[str, JSONValue]) -> JSONObject:
    check_deadline()
    normalized: JSONObject = {}
    for key in sort_once(
        payload,
        source="identity_contract.normalize_identity_mapping.payload_keys",
    ):
        check_deadline()
        normalized[str(key)] = _normalize_identity_value(payload[key])
    return normalized


def build_identity_contract(
    *,
    axis: IdentityAxis,
    kind: str,
    payload: Mapping[str, JSONValue],
    version: str = IDENTITY_CONTRACT_VERSION,
) -> IdentityContract:
    check_deadline()
    canonical_payload = json.dumps(
        {
            "identity_contract": version,
            "axis": axis.value,
            "kind": kind,
            "payload": normalize_identity_mapping(payload),
        },
        sort_keys=False,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(canonical_payload.encode("utf-8")).hexdigest()[:16]
    return IdentityContract(
        axis=axis,
        kind=kind,
        version=version,
        canonical_payload=canonical_payload,
        digest=digest,
    )

# gabion:decision_protocol_module
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from enum import StrEnum
from typing import Mapping, Sequence

from gabion.analysis.json_types import JSONObject, JSONValue
from gabion.analysis.resume_codec import mapping_or_none, sequence_or_none
from gabion.analysis.timeout_context import check_deadline
from gabion.order_contract import sort_once


class PatternAxis(StrEnum):
    DATAFLOW = "dataflow"
    EXECUTION = "execution"
    DUAL = "dual"


PATTERN_SCHEMA_CONTRACT_VERSION = "pattern_schema.v2"


def _normalize_signature_value(value: JSONValue) -> JSONValue:
    check_deadline()
    mapping_value = mapping_or_none(value)
    if mapping_value is not None:
        normalized: JSONObject = {}
        for key in sort_once(
            mapping_value,
            source="pattern_schema._normalize_signature_value.dict_keys",
        ):
            check_deadline()
            normalized[str(key)] = _normalize_signature_value(mapping_value[key])
        return normalized
    sequence_value = sequence_or_none(value, allow_str=False)
    if sequence_value is not None:
        normalized_items = [_normalize_signature_value(item) for item in sequence_value]
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
                source="pattern_schema._normalize_signature_value.list_items",
            )
        return normalized_items
    return value


def _is_sortable_signature_scalar(value: JSONValue) -> bool:
    return value is None or type(value) in {str, int, float, bool}


def normalize_signature(signature: Mapping[str, JSONValue]) -> JSONObject:
    check_deadline()
    normalized: JSONObject = {}
    for key in sort_once(
        signature,
        source="pattern_schema.normalize_signature.signature_keys",
    ):
        check_deadline()
        normalized[str(key)] = _normalize_signature_value(signature[key])
    return normalized


def _canonical_schema_identity_payload(
    *,
    kind: str,
    signature: Mapping[str, JSONValue],
    schema_contract: str,
) -> str:
    # dataflow-bundle: kind, schema_contract
    normalized = normalize_signature(signature)
    return json.dumps(
        {
            "schema_contract": schema_contract,
            "kind": kind,
            "signature": normalized,
        },
        sort_keys=False,
        separators=(",", ":"),
    )


def pattern_schema_id(*, kind: str, signature: Mapping[str, JSONValue]) -> str:
    check_deadline()
    canonical = _canonical_schema_identity_payload(
        kind=kind,
        signature=signature,
        schema_contract=PATTERN_SCHEMA_CONTRACT_VERSION,
    )
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
    return f"schema:{digest}"


def legacy_pattern_schema_id(*, axis: PatternAxis, kind: str, signature: Mapping[str, JSONValue]) -> str:
    check_deadline()
    normalized = normalize_signature(signature)
    canonical = json.dumps(
        {
            "axis": axis.value,
            "kind": kind,
            "signature": normalized,
        },
        sort_keys=False,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
    return f"{axis.value}:{kind}:{digest}"


@dataclass(frozen=True)
class PatternSchema:
    schema_id: str
    legacy_schema_id: str
    schema_contract: str
    axis: PatternAxis
    kind: str
    normalized_signature: JSONObject
    normalization: JSONObject

    @classmethod
    def build(
        cls,
        *,
        axis: PatternAxis,
        kind: str,
        signature: Mapping[str, JSONValue],
        normalization: Mapping[str, JSONValue],
    ) -> PatternSchema:
        # dataflow-bundle: normalization, signature
        check_deadline()
        normalized_signature = normalize_signature(signature)
        return cls(
            schema_id=pattern_schema_id(kind=kind, signature=normalized_signature),
            legacy_schema_id=legacy_pattern_schema_id(
                axis=axis,
                kind=kind,
                signature=normalized_signature,
            ),
            schema_contract=PATTERN_SCHEMA_CONTRACT_VERSION,
            axis=axis,
            kind=kind,
            normalized_signature=normalized_signature,
            normalization=normalize_signature(normalization),
        )


@dataclass(frozen=True)
class PatternResidue:
    schema_id: str
    reason: str
    payload: JSONObject


def mismatch_residue_payload(
    *,
    axis: PatternAxis,
    kind: str,
    expected: Mapping[str, JSONValue],
    observed: Mapping[str, JSONValue],
) -> JSONObject:
    # dataflow-bundle: expected, observed
    return {
        "schema_contract": PATTERN_SCHEMA_CONTRACT_VERSION,
        "axis": axis.value,
        "kind": kind,
        "expected": normalize_signature(expected),
        "observed": normalize_signature(observed),
    }


@dataclass(frozen=True)
class PatternInstance:
    schema: PatternSchema
    members: tuple[str, ...]
    suggestion: str
    residue: tuple[PatternResidue, ...] = ()

    @classmethod
    def build(
        cls,
        *,
        schema: PatternSchema,
        members: Sequence[str],
        suggestion: str,
        residue: Sequence[PatternResidue] = (),
    ) -> PatternInstance:
        # dataflow-bundle: members, residue
        check_deadline()
        return cls(
            schema=schema,
            members=tuple(members),
            suggestion=suggestion,
            residue=tuple(residue),
        )

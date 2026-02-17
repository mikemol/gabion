from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from enum import StrEnum
from typing import Mapping, Sequence

from gabion.analysis.json_types import JSONObject, JSONValue
from gabion.analysis.timeout_context import check_deadline
from gabion.order_contract import ordered_or_sorted


class PatternAxis(StrEnum):
    DATAFLOW = "dataflow"
    EXECUTION = "execution"
    DUAL = "dual"


def _normalize_signature_value(value: JSONValue) -> JSONValue:
    check_deadline()
    if isinstance(value, dict):
        normalized: JSONObject = {}
        for key in ordered_or_sorted(
            value,
            source="pattern_schema._normalize_signature_value.dict_keys",
        ):
            check_deadline()
            normalized[str(key)] = _normalize_signature_value(value[key])
        return normalized
    if isinstance(value, list):
        normalized_items = [_normalize_signature_value(item) for item in value]
        sortable = True
        for item in normalized_items:
            check_deadline()
            if not (isinstance(item, (str, int, float, bool)) or item is None):
                sortable = False
                break
        if sortable:
            return ordered_or_sorted(
                normalized_items,
                key=lambda item: json.dumps(item, sort_keys=True, separators=(",", ":")),
                source="pattern_schema._normalize_signature_value.list_items",
            )
        return normalized_items
    return value


def normalize_signature(signature: Mapping[str, JSONValue]) -> JSONObject:
    check_deadline()
    normalized: JSONObject = {}
    for key in ordered_or_sorted(
        signature,
        source="pattern_schema.normalize_signature.signature_keys",
    ):
        check_deadline()
        normalized[str(key)] = _normalize_signature_value(signature[key])
    return normalized


def pattern_schema_id(*, axis: PatternAxis, kind: str, signature: Mapping[str, JSONValue]) -> str:
    check_deadline()
    normalized = normalize_signature(signature)
    canonical = json.dumps(
        {
            "axis": axis.value,
            "kind": kind,
            "signature": normalized,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
    return f"{axis.value}:{kind}:{digest}"


@dataclass(frozen=True)
class PatternSchema:
    schema_id: str
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
            schema_id=pattern_schema_id(axis=axis, kind=kind, signature=normalized_signature),
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

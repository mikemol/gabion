# gabion:decision_protocol_module
from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Mapping, Sequence

from gabion.analysis.identity_contract import (
    IdentityAxis,
    build_identity_contract,
    normalize_identity_mapping,
)
from gabion.analysis.json_types import JSONObject, JSONValue
from gabion.analysis.timeout_context import check_deadline
from gabion.order_contract import sort_once


class PatternAxis(StrEnum):
    DATAFLOW = "dataflow"
    EXECUTION = "execution"
    DUAL = "dual"


PATTERN_SCHEMA_CONTRACT_VERSION = "pattern_schema.v2"


def normalize_signature(signature: Mapping[str, JSONValue]) -> JSONObject:
    check_deadline()
    return normalize_identity_mapping(signature)


def pattern_schema_id(*, kind: str, signature: Mapping[str, JSONValue]) -> str:
    check_deadline()
    normalized_signature = normalize_signature(signature)
    identity_contract = build_identity_contract(
        axis=IdentityAxis.SCHEMA,
        kind=kind,
        payload={
            "schema_contract": PATTERN_SCHEMA_CONTRACT_VERSION,
            "signature": normalized_signature,
        },
    )
    return f"schema:{identity_contract.digest}"


def execution_signature(
    *,
    family: str,
    members: Sequence[str],
) -> JSONObject:
    check_deadline()
    canonical_members = sort_once(
        {str(member) for member in members},
        source="pattern_schema.execution_signature.members",
    )
    return normalize_signature(
        {
            "family": family,
            "members": list(canonical_members),
            "member_count": len(canonical_members),
        }
    )



@dataclass(frozen=True)
class PatternSchema:
    schema_id: str
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

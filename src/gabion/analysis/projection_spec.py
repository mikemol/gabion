# gabion:boundary_normalization_module
# gabion:decision_protocol_module
from __future__ import annotations

from dataclasses import dataclass, field
from collections.abc import Mapping

from gabion.json_types import JSONValue
from gabion.analysis.resume_codec import mapping_or_none, sequence_or_none
from gabion.analysis.timeout_context import check_deadline


@dataclass(frozen=True)
class ProjectionOp:
    op: str
    params: dict[str, JSONValue] = field(default_factory=dict)


@dataclass(frozen=True)
class ProjectionSpec:
    spec_version: int
    name: str
    domain: str
    pipeline: tuple[ProjectionOp, ...] = ()
    params: dict[str, JSONValue] = field(default_factory=dict)


@dataclass(frozen=True)
class ProjectionOpPayload:
    op: str
    params: dict[str, JSONValue] = field(default_factory=dict)


@dataclass(frozen=True)
class ProjectionOpParseResult:
    is_valid: bool
    payload: ProjectionOpPayload


_EMPTY_PROJECTION_OP_PAYLOAD = ProjectionOpPayload(op="", params={})


def spec_to_dict(spec: ProjectionSpec) -> dict[str, JSONValue]:
    return {
        "spec_version": spec.spec_version,
        "name": spec.name,
        "domain": spec.domain,
        "params": dict(spec.params),
        "pipeline": [
            {
                "op": op.op,
                "params": dict(op.params),
            }
            for op in spec.pipeline
        ],
    }


def spec_from_dict(payload: Mapping[str, JSONValue]) -> ProjectionSpec:
    check_deadline()
    spec_version = payload.get("spec_version", 1)
    try:
        version = int(spec_version) if spec_version is not None else 1
    except (TypeError, ValueError):
        version = 1
    name = str(payload.get("name", "") or "")
    domain = str(payload.get("domain", "") or "")
    params = mapping_or_none(payload.get("params"))
    params_map: dict[str, JSONValue] = {}
    if params is not None:
        params_map = {str(k): v for k, v in params.items()}
    ops = tuple(
        ProjectionOp(op=op_payload.op, params=dict(op_payload.params))
        for op_payload in _op_payloads_from_pipeline(payload.get("pipeline"))
    )
    return ProjectionSpec(
        spec_version=version,
        name=name,
        domain=domain,
        pipeline=ops,
        params=params_map,
    )


def _op_payloads_from_pipeline(
    pipeline_payload: object,
) -> tuple[ProjectionOpPayload, ...]:
    check_deadline()
    pipeline = sequence_or_none(pipeline_payload)
    if pipeline is None:
        return ()
    payloads: list[ProjectionOpPayload] = []
    for entry in pipeline:
        check_deadline()
        parse_result = _op_payload_from_entry(entry)
        if parse_result.is_valid:
            payloads.append(parse_result.payload)
    return tuple(payloads)


def _op_payload_from_entry(entry: object) -> ProjectionOpParseResult:
    check_deadline()
    entry_map = mapping_or_none(entry)
    if entry_map is None:
        return ProjectionOpParseResult(
            is_valid=False,
            payload=_EMPTY_PROJECTION_OP_PAYLOAD,
        )
    op_name = str(entry_map.get("op", "") or "").strip()
    if not op_name:
        return ProjectionOpParseResult(
            is_valid=False,
            payload=_EMPTY_PROJECTION_OP_PAYLOAD,
        )
    op_params = mapping_or_none(entry_map.get("params"))
    params: dict[str, JSONValue] = {}
    if op_params is not None:
        params = {str(key): value for key, value in op_params.items()}
    return ProjectionOpParseResult(
        is_valid=True,
        payload=ProjectionOpPayload(op=op_name, params=params),
    )

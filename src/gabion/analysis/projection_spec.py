from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from gabion.json_types import JSONValue
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
    params = payload.get("params")
    params_map: dict[str, JSONValue] = {}
    if isinstance(params, Mapping):
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
    pipeline_payload: JSONValue | None,
) -> tuple[ProjectionOpPayload, ...]:
    check_deadline()
    if not isinstance(pipeline_payload, list):
        return ()
    payloads: list[ProjectionOpPayload] = []
    for entry in pipeline_payload:
        check_deadline()
        payload = _op_payload_from_entry(entry)
        if payload is not None:
            payloads.append(payload)
    return tuple(payloads)


def _op_payload_from_entry(entry: JSONValue) -> ProjectionOpPayload | None:
    check_deadline()
    if not isinstance(entry, Mapping):
        return None
    op_name = str(entry.get("op", "") or "").strip()
    if not op_name:
        return None
    op_params = entry.get("params")
    params: dict[str, JSONValue] = {}
    if isinstance(op_params, Mapping):
        params = {str(key): value for key, value in op_params.items()}
    return ProjectionOpPayload(op=op_name, params=params)

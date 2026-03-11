# gabion:decision_protocol_module
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Mapping

from gabion.analysis.projection.projection_registry import spec_metadata_payload
from gabion.analysis.projection.projection_spec import ProjectionSpec
from gabion.analysis.foundation.resume_codec import mapping_optional, sequence_optional
from gabion.json_types import JSONValue
from gabion.invariants import never


@dataclass(frozen=True)
class ParsedSpecMetadata:
    spec_id: str
    spec: dict[str, JSONValue]

    def __iter__(self):
        yield self.spec_id
        yield self.spec


def load_json(path: object) -> Mapping[str, JSONValue]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    payload_map = mapping_optional(payload)
    if payload_map is None:
        raise ValueError("Baseline payload must be a JSON object.")
    return payload_map


def write_json(path: object, payload: Mapping[str, JSONValue]) -> None:
    Path(path).write_text(
        json.dumps(payload, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )


def parse_version(
    payload: Mapping[str, JSONValue],
    *,
    expected: object,
    field: str = "version",
    error_context: str = "baseline",
) -> int:
    expected_values = sequence_optional(expected)
    match expected:
        case int() as expected_int:
            allowed = (int(expected_int),)
        case expected_other:
            _ = expected_other
            allowed = tuple(map(int, expected_values or ()))
    if not allowed:
        raise ValueError(
            "parse_version expected requires at least one allowed value"
        )
    default_value = allowed[0]
    raw = payload.get(field, default_value)
    try:
        value = int(raw) if raw is not None else default_value
    except (TypeError, ValueError):
        value = -1
    if value not in allowed:
        expected_display = (
            str(allowed[0])
            if len(allowed) == 1
            else ", ".join(map(str, allowed))
        )
        raise ValueError(
            f"Unsupported {error_context} {field}={raw!r}; expected {expected_display}"
        )
    return value


def parse_spec_metadata(
    payload: Mapping[str, JSONValue],
) -> ParsedSpecMetadata:
    spec_id = str(payload.get("generated_by_spec_id", "") or "")
    spec_payload = mapping_optional(payload.get("generated_by_spec", {}))
    spec: dict[str, JSONValue] = {}
    if spec_payload is not None:
        spec = dict(map(lambda key: (str(key), spec_payload[key]), spec_payload))
    return ParsedSpecMetadata(spec_id=spec_id, spec=spec)


def attach_spec_metadata(
    payload: dict[str, JSONValue],
    *,
    spec: ProjectionSpec,
) -> dict[str, JSONValue]:
    payload.update(spec_metadata_payload(spec))
    return payload

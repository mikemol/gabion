# gabion:decision_protocol_module
from __future__ import annotations

import json
from pathlib import Path
from collections.abc import Mapping

from gabion.analysis.projection_registry import spec_metadata_payload
from gabion.analysis.projection_spec import ProjectionSpec
from gabion.analysis.resume_codec import mapping_or_none, sequence_or_none
from gabion.json_types import JSONValue


def load_json(path: object) -> Mapping[str, JSONValue]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    payload_map = mapping_or_none(payload)
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
    expected_values = sequence_or_none(expected)
    allowed = (
        (int(expected),)
        if type(expected) is int
        else tuple(int(value) for value in (expected_values or ()))
    )
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
            else ", ".join(str(entry) for entry in allowed)
        )
        raise ValueError(
            f"Unsupported {error_context} {field}={raw!r}; expected {expected_display}"
        )
    return value


def parse_spec_metadata(
    payload: Mapping[str, JSONValue],
) -> tuple[str, dict[str, JSONValue]]:
    spec_id = str(payload.get("generated_by_spec_id", "") or "")
    spec_payload = mapping_or_none(payload.get("generated_by_spec", {}))
    spec: dict[str, JSONValue] = {}
    if spec_payload is not None:
        spec = {str(key): spec_payload[key] for key in spec_payload}
    return spec_id, spec


def attach_spec_metadata(
    payload: dict[str, JSONValue],
    *,
    spec: ProjectionSpec,
) -> dict[str, JSONValue]:
    payload.update(spec_metadata_payload(spec))
    return payload

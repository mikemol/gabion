from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

from gabion.analysis.projection_registry import spec_metadata_payload
from gabion.analysis.projection_spec import ProjectionSpec
from gabion.json_types import JSONValue


def load_json(path: str | Path) -> Mapping[str, JSONValue]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("Baseline payload must be a JSON object.")
    return payload


def write_json(path: str | Path, payload: Mapping[str, JSONValue]) -> None:
    Path(path).write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def parse_version(
    payload: Mapping[str, JSONValue],
    *,
    expected: int,
    field: str = "version",
    error_context: str = "baseline",
) -> int:
    raw = payload.get(field, expected)
    try:
        value = int(raw) if raw is not None else expected
    except (TypeError, ValueError):
        value = -1
    if value != expected:
        raise ValueError(
            f"Unsupported {error_context} version={raw!r}; expected {expected}"
        )
    return value


def parse_spec_metadata(
    payload: Mapping[str, JSONValue],
) -> tuple[str, dict[str, JSONValue]]:
    spec_id = str(payload.get("generated_by_spec_id", "") or "")
    spec_payload = payload.get("generated_by_spec", {})
    spec: dict[str, JSONValue] = {}
    if isinstance(spec_payload, Mapping):
        spec = {str(key): spec_payload[key] for key in spec_payload}
    return spec_id, spec


def attach_spec_metadata(
    payload: dict[str, JSONValue],
    *,
    spec: ProjectionSpec,
) -> dict[str, JSONValue]:
    payload.update(spec_metadata_payload(spec))
    return payload

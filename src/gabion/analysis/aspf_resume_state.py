# gabion:decision_protocol_module
# gabion:boundary_normalization_module
# gabion:ambiguity_boundary_module
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Mapping, Sequence, cast

from gabion.json_types import JSONObject, JSONValue

_DELTA_FORMAT_VERSION = 1


def append_delta_record(
    *,
    records: list[JSONObject],
    event_kind: str,
    phase: str,
    analysis_state: str | None,
    mutation_target: str,
    mutation_value: object,
    one_cell_ref: str | None = None,
) -> JSONObject:
    seq = len(records) + 1
    record: JSONObject = {
        "seq": seq,
        "ts_utc": datetime.now(timezone.utc).isoformat(),
        "event_kind": str(event_kind),
        "phase": str(phase),
        "analysis_state": str(analysis_state) if analysis_state is not None else None,
        "mutation_target": str(mutation_target),
        "mutation_value": _as_json_value(mutation_value),
        "one_cell_ref": str(one_cell_ref) if one_cell_ref is not None else None,
    }
    records.append(record)
    return record


def build_delta_ledger_payload(
    *,
    trace_id: str,
    records: Sequence[Mapping[str, object]],
) -> JSONObject:
    normalized: list[JSONObject] = []
    for raw in records:
        normalized.append({str(key): _as_json_value(raw[key]) for key in raw})
    return {
        "format_version": _DELTA_FORMAT_VERSION,
        "trace_id": str(trace_id),
        "records": normalized,
    }


def write_delta_jsonl(
    *,
    path: Path,
    records: Sequence[Mapping[str, object]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for raw in records:
        payload = {str(key): _as_json_value(raw[key]) for key in raw}
        lines.append(json.dumps(payload, sort_keys=False))
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def replay_resume_projection(
    *,
    snapshot: Mapping[str, object],
    delta_records: Sequence[Mapping[str, object]],
) -> JSONObject:
    projection: JSONObject = {str(key): _as_json_value(snapshot[key]) for key in snapshot}
    for record in delta_records:
        target = str(record.get("mutation_target", "")).strip()
        assert target
        _assign_by_path(
            projection,
            target.split("."),
            _as_json_value(record.get("mutation_value")),
        )
    return projection


def load_resume_projection_from_state_files(
    *,
    state_paths: Sequence[Path],
) -> tuple[JSONObject | None, tuple[JSONObject, ...]]:
    latest_projection: JSONObject | None = None
    all_records: list[JSONObject] = []
    for path in state_paths:
        payload = _load_json(path)
        resume = payload.get("resume_projection")
        assert isinstance(resume, Mapping)
        latest_projection = {str(key): _as_json_value(resume[key]) for key in resume}
        ledger = payload.get("delta_ledger")
        assert isinstance(ledger, Mapping)
        raw_records = ledger.get("records")
        assert isinstance(raw_records, list)
        for raw_record in raw_records:
            assert isinstance(raw_record, Mapping)
            all_records.append(
                {str(key): _as_json_value(raw_record[key]) for key in raw_record}
            )
    return latest_projection, tuple(all_records)


def _load_json(path: Path) -> JSONObject:
    raw = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(raw, Mapping)
    return {str(key): _as_json_value(raw[key]) for key in raw}


def _assign_by_path(
    payload: JSONObject,
    path_tokens: Sequence[str],
    value: JSONValue,
) -> None:
    assert path_tokens
    cursor: JSONObject = payload
    for raw_token in path_tokens[:-1]:
        token = str(raw_token).strip()
        cursor = cast(JSONObject, cursor.setdefault(token, {}))
    leaf = str(path_tokens[-1]).strip()
    assert leaf
    cursor[leaf] = value


def _as_json_value(value: object) -> JSONValue:
    if isinstance(value, Mapping):
        return {str(key): _as_json_value(value[key]) for key in value}
    if isinstance(value, (list, tuple, set)):
        return [_as_json_value(item) for item in value]
    return cast(JSONValue, value)

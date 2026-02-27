# gabion:decision_protocol_module
# gabion:boundary_normalization_module
# gabion:ambiguity_boundary_module
from __future__ import annotations

from collections import deque
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Iterable, Iterator, Mapping, Sequence, cast

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
    records: Iterable[Mapping[str, object]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for raw in records:
            payload = {str(key): _as_json_value(raw[key]) for key in raw}
            handle.write(json.dumps(payload, sort_keys=False))
            handle.write("\n")


def append_delta_jsonl_record(
    *,
    path: Path,
    record: Mapping[str, object],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {str(key): _as_json_value(record[key]) for key in record}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=False))
        handle.write("\n")


def replay_resume_projection(
    *,
    snapshot: Mapping[str, object],
    delta_records: Iterable[Mapping[str, object]],
) -> JSONObject:
    return apply_resume_mutations(snapshot=snapshot, mutations=delta_records)


def apply_resume_mutations(
    *,
    snapshot: Mapping[str, object],
    mutations: Iterable[Mapping[str, object]],
) -> JSONObject:
    projection: JSONObject = {str(key): _as_json_value(snapshot[key]) for key in snapshot}
    for record in mutations:
        target = str(record.get("mutation_target", "")).strip()
        assert target
        _assign_by_path(
            projection,
            target.split("."),
            _as_json_value(record.get("mutation_value")),
        )
    return projection


def fold_resume_mutations(
    *,
    snapshot: Mapping[str, object],
    mutations: Iterable[Mapping[str, object]],
    tail_limit: int = 0,
) -> tuple[JSONObject, int, tuple[JSONObject, ...]]:
    projection: JSONObject = {str(key): _as_json_value(snapshot[key]) for key in snapshot}
    tail = deque(maxlen=max(int(tail_limit), 0))
    mutation_count = 0
    for mutation in mutations:
        normalized_mutation = {
            str(key): _as_json_value(mutation[key]) for key in mutation
        }
        mutation_count += 1
        target = str(normalized_mutation.get("mutation_target", "")).strip()
        assert target
        _assign_by_path(
            projection,
            target.split("."),
            _as_json_value(normalized_mutation.get("mutation_value")),
        )
        if tail.maxlen:
            tail.append(normalized_mutation)
    return projection, mutation_count, tuple(tail)


def load_resume_projection_from_state_files(
    *,
    state_paths: Sequence[Path],
) -> tuple[JSONObject | None, tuple[JSONObject, ...]]:
    latest_projection = load_latest_resume_projection_from_state_files(state_paths=state_paths)
    return latest_projection, tuple(iter_delta_records_from_state_files(state_paths=state_paths))


def iter_delta_records_from_state_files(
    *,
    state_paths: Sequence[Path],
) -> Iterator[JSONObject]:
    for path in state_paths:
        delta_path = _delta_jsonl_path_for_state_path(path)
        if delta_path.exists():
            yield from iter_delta_records_from_jsonl_paths(jsonl_paths=(delta_path,))
            continue
        payload = _load_json(path)
        yield from _iter_delta_records_from_state_payload(payload=payload)


def iter_delta_records_from_jsonl_paths(
    *,
    jsonl_paths: Sequence[Path],
) -> Iterator[JSONObject]:
    for path in jsonl_paths:
        with path.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line:
                    continue
                loaded = json.loads(line)
                assert isinstance(loaded, Mapping)
                yield {str(key): _as_json_value(loaded[key]) for key in loaded}


def iter_delta_records(
    *,
    state_paths: Sequence[Path] = (),
    jsonl_paths: Sequence[Path] = (),
) -> Iterator[JSONObject]:
    yield from iter_resume_mutations(state_paths=state_paths, jsonl_paths=jsonl_paths)


def iter_resume_mutations(
    *,
    state_paths: Sequence[Path] = (),
    jsonl_paths: Sequence[Path] = (),
) -> Iterator[JSONObject]:
    yield from iter_delta_records_from_state_files(state_paths=state_paths)
    yield from iter_delta_records_from_jsonl_paths(jsonl_paths=jsonl_paths)


def load_latest_resume_projection_from_state_files(
    *,
    state_paths: Sequence[Path],
) -> JSONObject | None:
    latest_projection: JSONObject | None = None
    for path in state_paths:
        payload = _load_json(path)
        resume = payload.get("resume_projection")
        assert isinstance(resume, Mapping)
        latest_projection = {str(key): _as_json_value(resume[key]) for key in resume}
    return latest_projection


def _iter_delta_records_from_state_payload(*, payload: Mapping[str, object]) -> Iterator[JSONObject]:
    ledger = payload.get("delta_ledger")
    assert isinstance(ledger, Mapping)
    raw_records = ledger.get("records")
    assert isinstance(raw_records, list)
    for raw_record in raw_records:
        assert isinstance(raw_record, Mapping)
        yield {str(key): _as_json_value(raw_record[key]) for key in raw_record}


def _delta_jsonl_path_for_state_path(path: Path) -> Path:
    if path.name.endswith(".snapshot.json"):
        return path.with_name(path.name[: -len(".snapshot.json")] + ".delta.jsonl")
    return path.with_suffix(".delta.jsonl")


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

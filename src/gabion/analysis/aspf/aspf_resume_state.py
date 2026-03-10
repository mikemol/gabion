# gabion:ambiguity_boundary_module
from __future__ import annotations

from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator, Mapping, Sequence, cast

from gabion.analysis.foundation import aspf_io_boundary, wire_text_codec
from gabion.analysis.foundation.wire_types import WireObject, WireValue

_DELTA_FORMAT_VERSION = 1


def append_delta_record(
    *,
    records: list[WireObject],
    event_kind: str,
    phase: str,
    analysis_state: str | None,
    mutation_target: str,
    mutation_value: object,
    one_cell_ref: str | None = None,
) -> WireObject:
    seq = len(records) + 1
    record: WireObject = {
        "seq": seq,
        "ts_utc": datetime.now(timezone.utc).isoformat(),
        "event_kind": str(event_kind),
        "phase": str(phase),
        "analysis_state": str(analysis_state) if analysis_state is not None else None,
        "mutation_target": str(mutation_target),
        "mutation_value": _as_wire_value(mutation_value),
        "one_cell_ref": str(one_cell_ref) if one_cell_ref is not None else None,
    }
    records.append(record)
    return record


def build_delta_ledger_payload(
    *,
    trace_id: str,
    records: Sequence[Mapping[str, object]],
) -> WireObject:
    normalized: list[WireObject] = []
    for raw in records:
        normalized.append({str(key): _as_wire_value(raw[key]) for key in raw})
    return {
        "format_version": _DELTA_FORMAT_VERSION,
        "trace_id": str(trace_id),
        "records": normalized,
    }


def write_delta_stream(
    *,
    path: Path,
    records: Iterable[Mapping[str, object]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")
    for raw in records:
        payload = {str(key): _as_wire_value(raw[key]) for key in raw}
        wire_text_codec.append_line(path, payload)


def append_delta_stream_record(
    *,
    path: Path,
    record: Mapping[str, object],
) -> None:
    payload = {str(key): _as_wire_value(record[key]) for key in record}
    wire_text_codec.append_line(path, payload)


def replay_resume_projection(
    *,
    snapshot: Mapping[str, object],
    delta_records: Iterable[Mapping[str, object]],
) -> WireObject:
    return apply_resume_mutations(snapshot=snapshot, mutations=delta_records)


def apply_resume_mutations(
    *,
    snapshot: Mapping[str, object],
    mutations: Iterable[Mapping[str, object]],
) -> WireObject:
    projection: WireObject = {str(key): _as_wire_value(snapshot[key]) for key in snapshot}
    for record in mutations:
        target = str(record.get("mutation_target", "")).strip()
        assert target
        _assign_by_path(
            projection,
            target.split("."),
            _as_wire_value(record.get("mutation_value")),
        )
    return projection


def fold_resume_mutations(
    *,
    snapshot: Mapping[str, object],
    mutations: Iterable[Mapping[str, object]],
    tail_limit: int = 0,
) -> tuple[WireObject, int, tuple[WireObject, ...]]:
    projection: WireObject = {str(key): _as_wire_value(snapshot[key]) for key in snapshot}
    tail = deque(maxlen=max(int(tail_limit), 0))
    mutation_count = 0
    for mutation in mutations:
        normalized_mutation = {
            str(key): _as_wire_value(mutation[key]) for key in mutation
        }
        mutation_count += 1
        target = str(normalized_mutation.get("mutation_target", "")).strip()
        assert target
        _assign_by_path(
            projection,
            target.split("."),
            _as_wire_value(normalized_mutation.get("mutation_value")),
        )
        if tail.maxlen:
            tail.append(normalized_mutation)
    return projection, mutation_count, tuple(tail)


def load_resume_projection_from_state_files(
    *,
    state_paths: Sequence[Path],
) -> tuple[WireObject | None, tuple[WireObject, ...]]:
    latest_projection = load_latest_resume_projection_from_state_files(state_paths=state_paths)
    return latest_projection, tuple(iter_delta_records_from_state_files(state_paths=state_paths))


def iter_delta_records_from_state_files(
    *,
    state_paths: Sequence[Path],
) -> Iterator[WireObject]:
    for path in state_paths:
        delta_path = _delta_stream_path_for_state_path(path)
        if delta_path.exists():
            yield from iter_delta_records_from_stream_paths(stream_paths=(delta_path,))
            continue
        payload = _load_wire_object(path)
        yield from _iter_delta_records_from_state_payload(payload=payload)


def iter_delta_records_from_stream_paths(
    *,
    stream_paths: Sequence[Path],
) -> Iterator[WireObject]:
    for path in stream_paths:
        for line in wire_text_codec.iter_nonempty_lines(path):
            loaded = wire_text_codec.decode_text(line)
            match loaded:
                case dict() as loaded_mapping:
                    yield {
                        str(key): _as_wire_value(loaded_mapping[key])
                        for key in loaded_mapping
                    }
                case _:
                    raise AssertionError("delta stream line must decode to a mapping")


def iter_delta_records(
    *,
    state_paths: Sequence[Path] = (),
    stream_paths: Sequence[Path] = (),
) -> Iterator[WireObject]:
    yield from iter_resume_mutations(state_paths=state_paths, stream_paths=stream_paths)


def iter_resume_mutations(
    *,
    state_paths: Sequence[Path] = (),
    stream_paths: Sequence[Path] = (),
) -> Iterator[WireObject]:
    yield from iter_delta_records_from_state_files(state_paths=state_paths)
    yield from iter_delta_records_from_stream_paths(stream_paths=stream_paths)


def load_latest_resume_projection_from_state_files(
    *,
    state_paths: Sequence[Path],
) -> WireObject | None:
    latest_projection: WireObject | None = None
    for path in state_paths:
        payload = _load_wire_object(path)
        resume = payload.get("resume_projection")
        match resume:
            case dict() as resume_mapping:
                latest_projection = {
                    str(key): _as_wire_value(resume_mapping[key])
                    for key in resume_mapping
                }
            case _:
                raise AssertionError("resume_projection must be a mapping")
    return latest_projection


def _iter_delta_records_from_state_payload(*, payload: Mapping[str, object]) -> Iterator[WireObject]:
    ledger = payload.get("delta_ledger")
    match ledger:
        case dict() as ledger_mapping:
            raw_records = ledger_mapping.get("records")
        case _:
            raise AssertionError("delta_ledger must be a mapping")
    match raw_records:
        case list() as raw_record_list:
            pass
        case _:
            raise AssertionError("delta_ledger.records must be a list")
    for raw_record in raw_record_list:
        match raw_record:
            case dict() as raw_record_mapping:
                yield {
                    str(key): _as_wire_value(raw_record_mapping[key])
                    for key in raw_record_mapping
                }
            case _:
                raise AssertionError("delta_ledger.records entries must be mappings")


def _delta_stream_path_for_state_path(path: Path) -> Path:
    if path.name.endswith(aspf_io_boundary.STATE_SNAPSHOT_SUFFIX):
        return path.with_name(
            path.name[: -len(aspf_io_boundary.STATE_SNAPSHOT_SUFFIX)]
            + aspf_io_boundary.DELTA_STREAM_SUFFIX
        )
    return path.with_suffix(aspf_io_boundary.DELTA_STREAM_SUFFIX)


def _load_wire_object(path: Path) -> WireObject:
    mapping = wire_text_codec.decode_mapping_text(path.read_text(encoding="utf-8"))
    return {str(key): _as_wire_value(mapping[key]) for key in mapping}


def _assign_by_path(
    payload: WireObject,
    path_tokens: Sequence[str],
    value: WireValue,
) -> None:
    assert path_tokens
    cursor: WireObject = payload
    for raw_token in path_tokens[:-1]:
        token = str(raw_token).strip()
        cursor = cast(WireObject, cursor.setdefault(token, {}))
    leaf = str(path_tokens[-1]).strip()
    assert leaf
    cursor[leaf] = value


def _as_wire_value(value: object) -> WireValue:
    match value:
        case dict() as mapping:
            return {str(key): _as_wire_value(mapping[key]) for key in mapping}
        case list() | tuple() | set() as sequence:
            return [_as_wire_value(item) for item in sequence]
        case str() | int() | float() | bool() | None:
            return value
        case _:
            raise AssertionError("resume state payload values must be wire-compatible")

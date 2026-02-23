# gabion:boundary_normalization_module
# gabion:decision_protocol_module
from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterator, Mapping, Sequence, TypeVar, cast

from gabion.analysis.json_types import JSONValue
from gabion.analysis.timeout_context import check_deadline


def mapping_or_none(value: JSONValue | object) -> Mapping[str, JSONValue] | None:
    if isinstance(value, Mapping):
        return cast(Mapping[str, JSONValue], value)
    return None


def mapping_payload(
    payload: JSONValue | object,
) -> Mapping[str, JSONValue] | None:
    return mapping_or_none(payload)


def payload_with_phase(
    payload: Mapping[str, JSONValue] | None,
    *,
    phase: str,
) -> Mapping[str, JSONValue] | None:
    mapping = mapping_payload(payload)
    if mapping is None:
        return None
    if mapping.get("phase") != phase:
        return None
    return mapping


def payload_with_format(
    payload: Mapping[str, JSONValue] | None,
    *,
    format_version: int,
) -> Mapping[str, JSONValue] | None:
    mapping = mapping_payload(payload)
    if mapping is None:
        return None
    if mapping.get("format_version") != format_version:
        return None
    return mapping


def mapping_sections(
    payload: Mapping[str, JSONValue],
    *,
    section_keys: Sequence[str],
) -> tuple[Mapping[str, JSONValue], ...] | None:
    check_deadline()
    sections: list[Mapping[str, JSONValue]] = []
    for key in section_keys:
        check_deadline()
        section = mapping_or_none(payload.get(key))
        if section is None:
            return None
        sections.append(section)
    return tuple(sections)


def mapping_or_empty(value: JSONValue | object) -> Mapping[str, JSONValue]:
    mapping = mapping_or_none(value)
    if mapping is None:
        return {}
    return mapping


def sequence_or_none(
    value: JSONValue | object,
    *,
    allow_str: bool = False,
) -> Sequence[JSONValue] | None:
    if not isinstance(value, Sequence):
        return None
    if not allow_str and isinstance(value, (str, bytes, bytearray)):
        return None
    return cast(Sequence[JSONValue], value)


def str_list_from_sequence(value: JSONValue | object) -> list[str]:
    sequence = sequence_or_none(value)
    if sequence is None:
        return []
    out: list[str] = []
    for entry in sequence:
        check_deadline()
        if isinstance(entry, str):
            out.append(entry)
    return out


def str_tuple_from_sequence(value: JSONValue | object) -> tuple[str, ...]:
    return tuple(str_list_from_sequence(value))


def str_set_from_sequence(value: JSONValue | object) -> set[str]:
    return set(str_list_from_sequence(value))


def str_map_from_mapping(value: JSONValue | object) -> dict[str, str]:
    mapping = mapping_or_none(value)
    if mapping is None:
        return {}
    out: dict[str, str] = {}
    for key, entry in mapping.items():
        check_deadline()
        if isinstance(key, str) and isinstance(entry, str):
            out[key] = entry
    return out


def int_tuple4_or_none(value: JSONValue | object) -> tuple[int, int, int, int] | None:
    sequence = sequence_or_none(value)
    if sequence is None or len(sequence) != 4:
        return None
    try:
        parsed = tuple(int(part) for part in sequence)
    except (TypeError, ValueError):
        return None
    return cast(tuple[int, int, int, int], parsed)


def int_str_pairs_from_sequence(value: JSONValue | object) -> list[tuple[int, str]]:
    sequence = sequence_or_none(value)
    if sequence is None:
        return []
    out: list[tuple[int, str]] = []
    for entry in sequence:
        check_deadline()
        pair = sequence_or_none(entry)
        if pair is None or len(pair) != 2:
            continue
        idx, name = pair
        if not isinstance(name, str):
            continue
        try:
            idx_value = int(idx)
        except (TypeError, ValueError):
            continue
        out.append((idx_value, name))
    return out


def str_pair_set_from_sequence(value: JSONValue | object) -> set[tuple[str, str]]:
    sequence = sequence_or_none(value)
    if sequence is None:
        return set()
    out: set[tuple[str, str]] = set()
    for entry in sequence:
        check_deadline()
        pair = sequence_or_none(entry)
        if pair is None or len(pair) != 2:
            continue
        left, right = pair
        if isinstance(left, str) and isinstance(right, str):
            out.add((left, right))
    return out


def iter_valid_key_entries(
    *,
    payload: Mapping[str, JSONValue],
    valid_keys: set[str],
) -> Iterator[tuple[str, JSONValue]]:
    for key, raw_value in payload.items():
        check_deadline()
        if not isinstance(key, str) or key not in valid_keys:
            continue
        yield key, raw_value


_ParsedValue = TypeVar("_ParsedValue")


def allowed_path_lookup(
    paths: Sequence[Path],
    *,
    key_fn: Callable[[Path], str],
) -> dict[str, Path]:
    return {key_fn(path): path for path in paths}


def load_allowed_paths_from_sequence(
    value: JSONValue | object,
    *,
    allowed_paths: Mapping[str, Path],
) -> list[Path]:
    sequence = sequence_or_none(value)
    if sequence is None:
        return []
    out: list[Path] = []
    seen: set[str] = set()
    for raw_path in sequence:
        check_deadline()
        if not isinstance(raw_path, str):
            continue
        if raw_path in seen:
            continue
        path = allowed_paths.get(raw_path)
        if path is not None:
            seen.add(raw_path)
            out.append(path)
    return out


def load_resume_map(
    *,
    payload: Mapping[str, JSONValue],
    valid_keys: set[str],
    parser: Callable[[JSONValue], _ParsedValue | None],
) -> dict[str, _ParsedValue]:
    check_deadline()
    out: dict[str, _ParsedValue] = {}
    for key, raw_value in iter_valid_key_entries(
        payload=payload,
        valid_keys=valid_keys,
    ):
        check_deadline()
        parsed = parser(raw_value)
        if parsed is None:
            continue
        out[key] = parsed
    return out

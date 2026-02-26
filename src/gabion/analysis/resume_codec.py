# gabion:boundary_normalization_module
# gabion:decision_protocol_module
from __future__ import annotations

from pathlib import Path
from collections.abc import Callable, Iterator, Mapping, Sequence
from typing import TypeVar, cast

from gabion.analysis.json_types import JSONValue
from gabion.analysis.timeout_context import check_deadline


_NO_VALUE = None


def mapping_or_none(value: object):
    match value:
        case Mapping() as value_map:
            return cast(Mapping[str, JSONValue], value_map)
        case _:
            return _NO_VALUE


def mapping_payload(
    payload: object,
):
    return mapping_or_none(payload)


def payload_with_phase(
    payload: object,
    *,
    phase: str,
) -> object:
    mapping = mapping_payload(payload)
    if mapping is None:
        return _NO_VALUE
    if mapping.get("phase") != phase:
        return _NO_VALUE
    return mapping


def payload_with_format(
    payload: object,
    *,
    format_version: int,
) -> object:
    mapping = mapping_payload(payload)
    if mapping is None:
        return _NO_VALUE
    if mapping.get("format_version") != format_version:
        return _NO_VALUE
    return mapping


def mapping_sections(
    payload: Mapping[str, JSONValue],
    *,
    section_keys: Sequence[str],
) -> object:
    check_deadline()
    sections: list[Mapping[str, JSONValue]] = []
    for key in section_keys:
        check_deadline()
        section = mapping_or_none(payload.get(key))
        if section is None:
            return _NO_VALUE
        sections.append(section)
    return tuple(sections)


def mapping_or_empty(value: object) -> Mapping[str, JSONValue]:
    mapping = mapping_or_none(value)
    if mapping is None:
        return {}
    return mapping


def sequence_or_none(value: object, *, allow_str: bool = False):
    match value:
        case str() | bytes() | bytearray():
            if allow_str:
                return cast(Sequence[JSONValue], value)
            return _NO_VALUE
        case Sequence() as sequence_value:
            return cast(Sequence[JSONValue], sequence_value)
        case _:
            return _NO_VALUE


def str_list_from_sequence(value: object) -> list[str]:
    sequence = sequence_or_none(value)
    if sequence is None:
        return list()
    out: list[str] = []
    for entry in sequence:
        check_deadline()
        match entry:
            case str() as entry_text:
                out.append(entry_text)
            case _:
                pass
    return out


def str_tuple_from_sequence(value: object) -> tuple[str, ...]:
    return tuple(str_list_from_sequence(value))


def str_set_from_sequence(value: object) -> set[str]:
    return set(str_list_from_sequence(value))


def str_map_from_mapping(value: object) -> dict[str, str]:
    mapping = mapping_or_none(value)
    if mapping is None:
        return {}
    out: dict[str, str] = {}
    for key, entry in mapping.items():
        check_deadline()
        match (key, entry):
            case (str() as key_text, str() as entry_text):
                out[key_text] = entry_text
            case _:
                pass
    return out


def int_tuple4_or_none(value: object):
    sequence = sequence_or_none(value)
    if sequence is None or len(sequence) != 4:
        return _NO_VALUE
    try:
        parsed = tuple(int(part) for part in sequence)
    except (TypeError, ValueError):
        return _NO_VALUE
    return cast(tuple[int, int, int, int], parsed)


def int_str_pairs_from_sequence(value: object) -> list[tuple[int, str]]:
    sequence = sequence_or_none(value)
    if sequence is None:
        return list()
    out: list[tuple[int, str]] = []
    for entry in sequence:
        check_deadline()
        pair = sequence_or_none(entry)
        if pair is not None and len(pair) == 2:
            idx, name = pair
            match name:
                case str() as name_text:
                    try:
                        idx_value = int(idx)
                    except (TypeError, ValueError):
                        pass
                    else:
                        out.append((idx_value, name_text))
                case _:
                    pass
    return out


def str_pair_set_from_sequence(value: object) -> set[tuple[str, str]]:
    sequence = sequence_or_none(value)
    if sequence is None:
        return set()
    out: set[tuple[str, str]] = set()
    for entry in sequence:
        check_deadline()
        pair = sequence_or_none(entry)
        if pair is not None and len(pair) == 2:
            left, right = pair
            match (left, right):
                case (str() as left_text, str() as right_text):
                    out.add((left_text, right_text))
                case _:
                    pass
    return out


def iter_valid_key_entries(
    *,
    payload: Mapping[str, JSONValue],
    valid_keys: set[str],
) -> Iterator[tuple[str, JSONValue]]:
    for key, raw_value in payload.items():
        check_deadline()
        match key:
            case str() as key_text if key_text in valid_keys:
                yield key_text, raw_value
            case _:
                pass


_ParsedValue = TypeVar("_ParsedValue")


def allowed_path_lookup(
    paths: Sequence[Path],
    *,
    key_fn: Callable[[Path], str],
) -> dict[str, Path]:
    return {key_fn(path): path for path in paths}


def load_allowed_paths_from_sequence(
    value: object,
    *,
    allowed_paths: Mapping[str, Path],
) -> list[Path]:
    sequence = sequence_or_none(value)
    if sequence is None:
        return list()
    out: list[Path] = []
    seen: set[str] = set()
    for raw_path in sequence:
        check_deadline()
        match raw_path:
            case str() as raw_path_text:
                if raw_path_text not in seen:
                    path = allowed_paths.get(raw_path_text)
                    if path is not None:
                        seen.add(raw_path_text)
                        out.append(path)
            case _:
                pass
    return out


def load_resume_map(
    *,
    payload: Mapping[str, JSONValue],
    valid_keys: set[str],
    parser: Callable[[JSONValue], object],
) -> dict[str, _ParsedValue]:
    check_deadline()
    out: dict[str, _ParsedValue] = {}
    for key, raw_value in iter_valid_key_entries(
        payload=payload,
        valid_keys=valid_keys,
    ):
        check_deadline()
        parsed = parser(raw_value)
        if parsed is not None:
            out[key] = cast(_ParsedValue, parsed)
    return out

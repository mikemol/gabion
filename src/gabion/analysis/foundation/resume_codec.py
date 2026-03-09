from __future__ import annotations

from pathlib import Path
from collections.abc import Callable, Iterator, Mapping, Sequence
from typing import TypeVar

from gabion.analysis.foundation.json_types import JSONValue
from gabion.analysis.foundation.timeout_context import check_deadline
from gabion.runtime_shape_dispatch import str_optional


_TParsed = TypeVar("_TParsed")
_NO_VALUE = None


def iter_mapping_items(value):
    match value:
        case Mapping() as value_map:
            for key in value_map:
                yield key, value_map[key]
        case _:
            return


def iter_mapping_values(value) -> Iterator[Mapping[str, JSONValue]]:
    for entry in iter_sequence_items(value):
        mapped = mapping_optional(entry)
        if mapped is not None:
            yield mapped


def iter_str_key_mappings(value) -> Iterator[tuple[str, Mapping[str, JSONValue]]]:
    for key, entry in iter_mapping_items(value):
        key_text = str_optional(key)
        if key_text is not None:
            mapped = mapping_optional(entry)
            if mapped is not None:
                yield key_text, mapped


def mapping_optional(value):
    match value:
        case Mapping():
            return dict(iter_mapping_items(value))
        case _:
            return None


def mapping_payload(
    payload,
):
    return mapping_optional(payload)


def payload_with_phase(
    payload,
    *,
    phase: str,
):
    mapping = mapping_payload(payload)
    if mapping is None:
        return _NO_VALUE
    if mapping.get("phase") != phase:
        return _NO_VALUE
    return mapping


def payload_with_format(
    payload,
    *,
    format_version: int,
):
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
):
    check_deadline()
    sections = tuple(
        _iter_required_mapping_sections(
            payload=payload,
            section_keys=section_keys,
        )
    )
    if len(sections) != len(section_keys):
        return _NO_VALUE
    return sections


def _iter_required_mapping_sections(
    *,
    payload: Mapping[str, JSONValue],
    section_keys: Sequence[str],
) -> Iterator[Mapping[str, JSONValue]]:
    for key in section_keys:
        check_deadline()
        section = mapping_optional(payload.get(key))
        if section is not None:
            yield section


def mapping_default_empty(value) -> Mapping[str, JSONValue]:
    return dict(iter_mapping_items(value))


def sequence_optional(value, *, allow_str: bool = False):
    match value:
        case str() | bytes() | bytearray():
            if allow_str:
                return value
            return None
        case Sequence() as sequence_value:
            return sequence_value
        case _:
            return None


def iter_sequence_items(value, *, allow_str: bool = False):
    for entry in sequence_optional(value, allow_str=allow_str) or ():
        check_deadline()
        yield entry


def iter_str_from_sequence(value) -> Iterator[str]:
    for entry in iter_sequence_items(value):
        match entry:
            case str() as entry_text:
                yield entry_text
            case _:
                pass


def str_list_from_sequence(value) -> list[str]:
    return list(iter_str_from_sequence(value))


def str_tuple_from_sequence(value) -> tuple[str, ...]:
    return tuple(iter_str_from_sequence(value))


def str_set_from_sequence(value) -> set[str]:
    return set(iter_str_from_sequence(value))


def str_map_from_mapping(value) -> dict[str, str]:
    return dict(iter_str_mapping_items(value))


def iter_str_mapping_items(value) -> Iterator[tuple[str, str]]:
    for key, entry in iter_mapping_items(value):
        check_deadline()
        match (key, entry):
            case (str() as key_text, str() as entry_text):
                yield key_text, entry_text
            case _:
                pass


def iter_int_tuple4_from_sequence(value) -> Iterator[tuple[int, int, int, int]]:
    for entry in iter_sequence_items(value):
        parts = tuple(iter_sequence_items(entry))
        if len(parts) != 4:
            continue
        try:
            line, col, end_line, end_col = map(int, parts)
        except (TypeError, ValueError):
            continue
        yield line, col, end_line, end_col


def iter_int_str_pairs_from_sequence(value) -> Iterator[tuple[int, str]]:
    for entry in iter_sequence_items(value):
        pair = sequence_optional(entry)
        if pair is not None and len(pair) == 2:
            idx, name = pair
            match name:
                case str() as name_text:
                    try:
                        idx_value = int(idx)
                    except (TypeError, ValueError):
                        pass
                    else:
                        yield idx_value, name_text
                case _:
                    pass


def int_str_pairs_from_sequence(value) -> list[tuple[int, str]]:
    return list(iter_int_str_pairs_from_sequence(value))


def iter_str_pairs_from_sequence(value) -> Iterator[tuple[str, str]]:
    for entry in iter_sequence_items(value):
        pair = sequence_optional(entry)
        if pair is not None and len(pair) == 2:
            left, right = pair
            match (left, right):
                case (str() as left_text, str() as right_text):
                    yield left_text, right_text
                case _:
                    pass


def str_pair_set_from_sequence(value) -> set[tuple[str, str]]:
    return set(iter_str_pairs_from_sequence(value))


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


def allowed_path_lookup(
    paths: Sequence[Path],
    *,
    key_fn: Callable[[Path], str],
) -> dict[str, Path]:
    return dict(iter_allowed_path_entries(paths=paths, key_fn=key_fn))


def iter_allowed_path_entries(
    *,
    paths: Sequence[Path],
    key_fn: Callable[[Path], str],
) -> Iterator[tuple[str, Path]]:
    for path in paths:
        yield key_fn(path), path


def load_allowed_paths_from_sequence(
    value,
    *,
    allowed_paths: Mapping[str, Path],
) -> list[Path]:
    return list(iter_allowed_paths_from_sequence(value, allowed_paths=allowed_paths))


def iter_allowed_paths_from_sequence(
    value,
    *,
    allowed_paths: Mapping[str, Path],
) -> Iterator[Path]:
    seen: set[str] = set()
    for raw_path in iter_sequence_items(value):
        match raw_path:
            case str() as raw_path_text:
                if raw_path_text not in seen:
                    path = allowed_paths.get(raw_path_text)
                    if path is not None:
                        seen.add(raw_path_text)
                        yield path
            case _:
                pass


def load_resume_map(
    *,
    payload: Mapping[str, JSONValue],
    valid_keys: set[str],
    parser: Callable[[JSONValue], Iterator[_TParsed]],
) -> dict[str, _TParsed]:
    return dict(
        iter_resume_map_entries(
            payload=payload,
            valid_keys=valid_keys,
            parser=parser,
        )
    )


def iter_resume_map_entries(
    *,
    payload: Mapping[str, JSONValue],
    valid_keys: set[str],
    parser: Callable[[JSONValue], Iterator[_TParsed]],
) -> Iterator[tuple[str, _TParsed]]:
    check_deadline()
    for key, raw_value in iter_valid_key_entries(
        payload=payload,
        valid_keys=valid_keys,
    ):
        check_deadline()
        yield from _iter_parsed_key_entries(
            key=key,
            parsed_values=parser(raw_value),
        )


def _iter_parsed_key_entries(
    *,
    key: str,
    parsed_values: Iterator[_TParsed],
) -> Iterator[tuple[str, _TParsed]]:
    for parsed in parsed_values:
        yield key, parsed

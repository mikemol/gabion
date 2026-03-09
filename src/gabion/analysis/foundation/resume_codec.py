# gabion:decision_protocol_module
from __future__ import annotations

from pathlib import Path
from itertools import chain, repeat
from collections.abc import Callable, Iterator, Mapping, Sequence
from typing import TypeVar

from gabion.analysis.foundation.json_types import JSONValue
from gabion.analysis.foundation.timeout_context import check_deadline
from gabion.runtime_shape_dispatch import str_optional


_TParsed = TypeVar("_TParsed")
_NO_VALUE = None


def _is_not_none(value) -> bool:
    return value is not None


def _path_text_is_selectable(*, allowed_paths: Mapping[str, Path], seen: set[str]):
    def _predicate(raw_path_text: str | None) -> bool:
        match raw_path_text:
            case str() as path_text:
                if path_text in seen:
                    return False
                if path_text not in allowed_paths:
                    return False
                seen.add(path_text)
                return True
            case _:
                return False

    return _predicate


def iter_mapping_items(value):
    match value:
        case Mapping() as value_map:
            for key in value_map:
                yield key, value_map[key]
        case _:
            return


def iter_mapping_values(value) -> Iterator[Mapping[str, JSONValue]]:
    return filter(
        _is_not_none,
        map(mapping_optional, iter_sequence_items(value)),
    )


def iter_str_key_mappings(value) -> Iterator[tuple[str, Mapping[str, JSONValue]]]:
    return filter(
        lambda mapped: mapped[0] is not None and mapped[1] is not None,
        map(
            lambda key_entry: (
                str_optional(key_entry[0]),
                mapping_optional(key_entry[1]),
            ),
            iter_mapping_items(value),
        ),
    )


def mapping_optional(value):
    match value:
        case Mapping() as value_map:
            return value_map
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
    return filter(
        _is_not_none,
        map(
            lambda key: mapping_optional(payload.get(key)),
            map(
                lambda key: (check_deadline(), key)[1],
                section_keys,
            ),
        ),
    )


def mapping_default_empty(value) -> Mapping[str, JSONValue]:
    match value:
        case Mapping() as value_map:
            return value_map
        case _:
            return {}


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
    return filter(
        _is_not_none,
        map(str_optional, iter_sequence_items(value)),
    )


def str_list_from_sequence(value) -> Iterator[str]:
    return iter_str_from_sequence(value)


def str_tuple_from_sequence(value) -> Iterator[str]:
    return iter_str_from_sequence(value)


def str_set_from_sequence(value) -> Iterator[str]:
    return iter_str_from_sequence(value)


def str_map_from_mapping(value) -> Iterator[tuple[str, str]]:
    return iter_str_mapping_items(value)


def iter_str_mapping_items(value) -> Iterator[tuple[str, str]]:
    return filter(
        lambda pair: pair[0] is not None and pair[1] is not None,
        map(
            lambda key_entry: (
                str_optional(key_entry[0]),
                str_optional(key_entry[1]),
            ),
            map(
                lambda key_entry: (check_deadline(), key_entry)[1],
                iter_mapping_items(value),
            ),
        ),
    )


def iter_int_tuple4_from_sequence(value) -> Iterator[tuple[int, int, int, int]]:
    return chain.from_iterable(map(_iter_int_tuple4_entry, iter_sequence_items(value)))


def _iter_int_tuple4_entry(entry) -> Iterator[tuple[int, int, int, int]]:
    parts = tuple(iter_sequence_items(entry))
    if len(parts) != 4:
        return chain()
    try:
        line, col, end_line, end_col = map(int, parts)
    except (TypeError, ValueError):
        return chain()
    return repeat((line, col, end_line, end_col), 1)


def iter_int_str_pairs_from_sequence(value) -> Iterator[tuple[int, str]]:
    return chain.from_iterable(map(_iter_int_str_pair_entry, iter_sequence_items(value)))


def _iter_int_str_pair_entry(entry) -> Iterator[tuple[int, str]]:
    pair = tuple(iter_sequence_items(entry))
    if len(pair) != 2:
        return chain()
    idx, name = pair
    match name:
        case str() as name_text:
            try:
                idx_value = int(idx)
            except (TypeError, ValueError):
                return chain()
            return repeat((idx_value, name_text), 1)
        case _:
            return chain()


def int_str_pairs_from_sequence(value) -> Iterator[tuple[int, str]]:
    return iter_int_str_pairs_from_sequence(value)


def iter_str_pairs_from_sequence(value) -> Iterator[tuple[str, str]]:
    return chain.from_iterable(map(_iter_str_pair_entry, iter_sequence_items(value)))


def _iter_str_pair_entry(entry) -> Iterator[tuple[str, str]]:
    pair = tuple(iter_sequence_items(entry))
    if len(pair) != 2:
        return chain()
    left, right = pair
    match (left, right):
        case (str() as left_text, str() as right_text):
            return repeat((left_text, right_text), 1)
        case _:
            return chain()


def str_pair_set_from_sequence(value) -> Iterator[tuple[str, str]]:
    return iter_str_pairs_from_sequence(value)


def iter_valid_key_entries(
    *,
    payload: Mapping[str, JSONValue],
    valid_keys: set[str],
) -> Iterator[tuple[str, JSONValue]]:
    return filter(
        lambda key_entry: key_entry[0] is not None and key_entry[0] in valid_keys,
        map(
            lambda key_entry: (
                str_optional(key_entry[0]),
                key_entry[1],
            ),
            map(
                lambda key_entry: (check_deadline(), key_entry)[1],
                payload.items(),
            ),
        ),
    )


def allowed_path_lookup(
    paths: Sequence[Path],
    *,
    key_fn: Callable[[Path], str],
) -> Iterator[tuple[str, Path]]:
    return iter_allowed_path_entries(paths=paths, key_fn=key_fn)


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
) -> Iterator[Path]:
    return iter_allowed_paths_from_sequence(value, allowed_paths=allowed_paths)


def iter_allowed_paths_from_sequence(
    value,
    *,
    allowed_paths: Mapping[str, Path],
) -> Iterator[Path]:
    seen: set[str] = set()
    raw_path_texts = map(str_optional, iter_sequence_items(value))
    selectable_path_texts = filter(
        _path_text_is_selectable(allowed_paths=allowed_paths, seen=seen),
        raw_path_texts,
    )
    return map(allowed_paths.__getitem__, selectable_path_texts)


def load_resume_map(
    *,
    payload: Mapping[str, JSONValue],
    valid_keys: set[str],
    parser: Callable[[JSONValue], Iterator[_TParsed]],
) -> Iterator[tuple[str, _TParsed]]:
    return iter_resume_map_entries(
        payload=payload,
        valid_keys=valid_keys,
        parser=parser,
    )


def iter_resume_map_entries(
    *,
    payload: Mapping[str, JSONValue],
    valid_keys: set[str],
    parser: Callable[[JSONValue], Iterator[_TParsed]],
) -> Iterator[tuple[str, _TParsed]]:
    check_deadline()
    return chain.from_iterable(
        map(
            lambda key_value: _iter_parsed_key_entries(
                key=key_value[0],
                parsed_values=parser(key_value[1]),
            ),
            iter_valid_key_entries(
                payload=payload,
                valid_keys=valid_keys,
            ),
        )
    )


def _iter_parsed_key_entries(
    *,
    key: str,
    parsed_values: Iterator[_TParsed],
) -> Iterator[tuple[str, _TParsed]]:
    for parsed in parsed_values:
        yield key, parsed

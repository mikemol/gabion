from __future__ import annotations

from pathlib import Path
from typing import Sequence, cast

from gabion.json_types import JSONValue
from gabion.runtime.coercion_contract import (
    CORE_STR_OPTIONAL_POLICY,
    FLOAT_ONLY_OPTIONAL_POLICY,
    MAPPING_OPTIONAL_POLICY,
    NON_BOOL_INT_OPTIONAL_POLICY,
    RuntimeCoercionCase,
    RuntimeOptionalCoercion,
)

_NONE_TYPE = type(None)


def _mapping_or_none(value: object) -> dict[str, object] | None:
    return MAPPING_OPTIONAL_POLICY(value)


def _json_mapping_or_none(value: object) -> dict[str, JSONValue] | None:
    return cast(dict[str, JSONValue] | None, MAPPING_OPTIONAL_POLICY(value))


def _non_negative_float_from_number(value: object) -> float:
    return max(float(cast(int | float | bool, value)), 0.0)


def _int_from_int_like(value: object) -> int:
    return int(cast(int | bool, value))


def _zero(_: object) -> int:
    return 0


def _list_identity(value: object) -> Sequence[object]:
    return cast(list[object], value)


def _tuple_identity(value: object) -> Sequence[object]:
    return cast(tuple[object, ...], value)


def _set_identity(value: object) -> Sequence[object]:
    return cast(set[object], value)


def _none(value: object) -> None:
    _ = value
    return None


OBJECT_MAPPING_OPTIONAL_POLICY = RuntimeOptionalCoercion[dict[str, object]](
    policy_name="server_object_mapping_optional",
    cases=(
        RuntimeCoercionCase((dict,), cast(object, _mapping_or_none)),
        RuntimeCoercionCase((list, tuple, set, str, int, float, bool, _NONE_TYPE), _none),
    ),
)

JSON_MAPPING_OPTIONAL_POLICY = RuntimeOptionalCoercion[dict[str, JSONValue]](
    policy_name="server_json_mapping_optional",
    cases=(
        RuntimeCoercionCase((dict,), cast(object, _json_mapping_or_none)),
        RuntimeCoercionCase((list, tuple, set, str, int, float, bool, _NONE_TYPE), _none),
    ),
)

BOOL_OPTIONAL_POLICY = RuntimeOptionalCoercion[bool](
    policy_name="server_bool_optional",
    cases=(
        RuntimeCoercionCase((bool,), cast(object, lambda value: cast(bool, value))),
        RuntimeCoercionCase((int, float, str, list, tuple, set, dict, _NONE_TYPE), _none),
    ),
)

NON_NEGATIVE_FLOAT_OPTIONAL_POLICY = RuntimeOptionalCoercion[float](
    policy_name="server_non_negative_float_optional",
    cases=(
        RuntimeCoercionCase((bool, int, float), _non_negative_float_from_number),
        RuntimeCoercionCase((str, list, tuple, set, dict, _NONE_TYPE), _none),
    ),
)

INT_OR_ZERO_POLICY = RuntimeOptionalCoercion[int](
    policy_name="server_int_or_zero",
    cases=(
        RuntimeCoercionCase((bool, int), _int_from_int_like),
        RuntimeCoercionCase((float, str, list, tuple, set, dict, _NONE_TYPE), _zero),
    ),
)

NON_STRING_SEQUENCE_OPTIONAL_POLICY = RuntimeOptionalCoercion[Sequence[object]](
    policy_name="server_non_string_sequence_optional",
    cases=(
        RuntimeCoercionCase((list,), _list_identity),
        RuntimeCoercionCase((tuple,), _tuple_identity),
        RuntimeCoercionCase((set,), _set_identity),
        RuntimeCoercionCase((str, dict, int, float, bool, _NONE_TYPE), _none),
    ),
)


def _object_mapping_optional(value: object) -> dict[str, object] | None:
    return OBJECT_MAPPING_OPTIONAL_POLICY(value)


def _string_optional(value: object) -> str | None:
    return CORE_STR_OPTIONAL_POLICY(value)


def _non_empty_string_optional(value: object) -> str | None:
    text = _string_optional(value)
    if text:
        return text
    return None


def _config_path_optional(value: object) -> Path | None:
    config_text = _string_optional(value)
    if config_text is None:
        return None
    normalized = config_text.strip()
    if not normalized:
        return None
    return Path(normalized)


def _bool_optional(value: object) -> bool | None:
    return BOOL_OPTIONAL_POLICY(value)


def _non_negative_float_optional(value: object) -> float | None:
    return NON_NEGATIVE_FLOAT_OPTIONAL_POLICY(value)


def _int_or_zero(value: object) -> int:
    return INT_OR_ZERO_POLICY(value)


def _json_mapping_optional(value: object) -> dict[str, JSONValue] | None:
    return JSON_MAPPING_OPTIONAL_POLICY(value)


def _json_mapping_default_empty(value: object) -> dict[str, JSONValue]:
    mapping = _json_mapping_optional(value)
    if mapping is not None:
        return mapping
    return {}


def _int_optional(value: object) -> int | None:
    return NON_BOOL_INT_OPTIONAL_POLICY(value)


def _non_negative_int_optional(value: object) -> int | None:
    int_value = _int_optional(value)
    if int_value is None:
        return None
    return max(int_value, 0)


def _non_string_sequence_optional(value: object) -> Sequence[object] | None:
    return NON_STRING_SEQUENCE_OPTIONAL_POLICY(value)


def _str_optional(value: object) -> str | None:
    return CORE_STR_OPTIONAL_POLICY(value)


def _float_optional(value: object) -> float | None:
    return FLOAT_ONLY_OPTIONAL_POLICY(value)


__all__ = [
    "BOOL_OPTIONAL_POLICY",
    "INT_OR_ZERO_POLICY",
    "JSON_MAPPING_OPTIONAL_POLICY",
    "NON_NEGATIVE_FLOAT_OPTIONAL_POLICY",
    "NON_STRING_SEQUENCE_OPTIONAL_POLICY",
    "OBJECT_MAPPING_OPTIONAL_POLICY",
    "_bool_optional",
    "_config_path_optional",
    "_float_optional",
    "_int_optional",
    "_int_or_zero",
    "_json_mapping_default_empty",
    "_json_mapping_optional",
    "_non_empty_string_optional",
    "_non_negative_float_optional",
    "_non_negative_int_optional",
    "_non_string_sequence_optional",
    "_object_mapping_optional",
    "_str_optional",
    "_string_optional",
]

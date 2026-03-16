from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Generic, TypeVar, cast

from gabion.invariants import never
from gabion.json_types import JSONValue

T = TypeVar("T")
_NONE_TYPE = type(None)


@dataclass(frozen=True)
class RuntimeCoercionCase(Generic[T]):
    runtime_types: tuple[type[object], ...]
    coerce: Callable[[object], T | None]


@dataclass(frozen=True)
class RuntimeOptionalCoercion(Generic[T]):
    policy_name: str
    cases: tuple[RuntimeCoercionCase[T], ...]

    def __call__(self, value: object) -> T | None:
        for case in self.cases:
            if isinstance(value, case.runtime_types):
                return case.coerce(value)
        never("unregistered runtime type", policy=self.policy_name, value_type=type(value).__name__)


def _none(value: object) -> None:
    _ = value
    return None


def _json_mapping_identity(value: object) -> dict[str, JSONValue]:
    return cast(dict[str, JSONValue], value)


def _json_list_identity(value: object) -> list[JSONValue]:
    return cast(list[JSONValue], value)


def _list_identity(value: object) -> list[object]:
    return cast(list[object], value)


def _tuple_to_list(value: object) -> list[object]:
    return list(cast(tuple[object, ...], value))


def _string_identity(value: object) -> str:
    return cast(str, value)


def _int_identity(value: object) -> int:
    return cast(int, value)


def _int_like_identity(value: object) -> int | bool:
    return cast(int | bool, value)


def _float_identity(value: object) -> float:
    return cast(float, value)


def _float_from_number(value: object) -> float:
    return float(cast(int | float | bool, value))


def _mapping_identity(value: object) -> dict[str, object]:
    return cast(dict[str, object], value)


def _string_key_mapping(value: object) -> dict[str, object]:
    mapping = cast(dict[object, object], value)
    return {str(key): mapping[key] for key in mapping}


JSON_MAPPING_OPTIONAL_POLICY = RuntimeOptionalCoercion[dict[str, JSONValue]](
    policy_name="json_mapping_optional",
    cases=(
        RuntimeCoercionCase((dict,), _json_mapping_identity),
        RuntimeCoercionCase(
            (list, tuple, set, str, int, float, bool, bytes, bytearray, _NONE_TYPE),
            _none,
        ),
    ),
)

JSON_LIST_OPTIONAL_POLICY = RuntimeOptionalCoercion[list[JSONValue]](
    policy_name="json_list_optional",
    cases=(
        RuntimeCoercionCase((list,), _json_list_identity),
        RuntimeCoercionCase(
            (tuple, set, dict, str, int, float, bool, bytes, bytearray, _NONE_TYPE),
            _none,
        ),
    ),
)

STR_OPTIONAL_POLICY = RuntimeOptionalCoercion[str](
    policy_name="str_optional",
    cases=(
        RuntimeCoercionCase((str,), _string_identity),
        RuntimeCoercionCase(
            (int, float, bool, dict, list, tuple, set, bytes, bytearray, _NONE_TYPE),
            _none,
        ),
    ),
)

CORE_STR_OPTIONAL_POLICY = RuntimeOptionalCoercion[str](
    policy_name="core_str_optional",
    cases=(
        RuntimeCoercionCase((str,), _string_identity),
        RuntimeCoercionCase((int, float, bool, dict, list, tuple, set, _NONE_TYPE), _none),
    ),
)

MAPPING_OPTIONAL_POLICY = RuntimeOptionalCoercion[dict[str, object]](
    policy_name="mapping_optional",
    cases=(
        RuntimeCoercionCase((dict,), _mapping_identity),
        RuntimeCoercionCase((list, tuple, set, str, int, float, bool, _NONE_TYPE), _none),
    ),
)

LIST_OPTIONAL_POLICY = RuntimeOptionalCoercion[list[object]](
    policy_name="list_optional",
    cases=(
        RuntimeCoercionCase((list,), _list_identity),
        RuntimeCoercionCase((dict, tuple, set, str, int, float, bool, _NONE_TYPE), _none),
    ),
)

LIST_OR_TUPLE_TO_LIST_OPTIONAL_POLICY = RuntimeOptionalCoercion[list[object]](
    policy_name="list_or_tuple_to_list_optional",
    cases=(
        RuntimeCoercionCase((list,), _list_identity),
        RuntimeCoercionCase((tuple,), _tuple_to_list),
        RuntimeCoercionCase((dict, set, str, int, float, bool, _NONE_TYPE), _none),
    ),
)

NON_BOOL_INT_OPTIONAL_POLICY = RuntimeOptionalCoercion[int](
    policy_name="int_non_bool_optional",
    cases=(
        RuntimeCoercionCase((bool,), _none),
        RuntimeCoercionCase((int,), _int_identity),
        RuntimeCoercionCase((float, str, dict, list, tuple, set, _NONE_TYPE), _none),
    ),
)

INT_LIKE_OPTIONAL_POLICY = RuntimeOptionalCoercion[int | bool](
    policy_name="int_like_optional",
    cases=(
        RuntimeCoercionCase((bool,), _int_like_identity),
        RuntimeCoercionCase((int,), _int_like_identity),
        RuntimeCoercionCase((float, str, dict, list, tuple, set, _NONE_TYPE), _none),
    ),
)

FLOAT_OPTIONAL_POLICY = RuntimeOptionalCoercion[float](
    policy_name="float_optional",
    cases=(
        RuntimeCoercionCase((bool,), _float_from_number),
        RuntimeCoercionCase((int,), _float_from_number),
        RuntimeCoercionCase((float,), _float_identity),
        RuntimeCoercionCase((str, dict, list, tuple, set, bytes, bytearray, _NONE_TYPE), _none),
    ),
)

FLOAT_ONLY_OPTIONAL_POLICY = RuntimeOptionalCoercion[float](
    policy_name="float_only_optional",
    cases=(
        RuntimeCoercionCase((float,), _float_identity),
        RuntimeCoercionCase((int, bool, str, dict, list, tuple, set, _NONE_TYPE), _none),
    ),
)

NON_BOOL_FLOAT_OPTIONAL_POLICY = RuntimeOptionalCoercion[float](
    policy_name="float_non_bool_optional",
    cases=(
        RuntimeCoercionCase((bool,), _none),
        RuntimeCoercionCase((int,), _float_from_number),
        RuntimeCoercionCase((float,), _float_identity),
        RuntimeCoercionCase((str, dict, list, tuple, set, _NONE_TYPE), _none),
    ),
)

ROW_FLOAT_OPTIONAL_POLICY = RuntimeOptionalCoercion[float](
    policy_name="float_row_optional",
    cases=(
        RuntimeCoercionCase((bool,), _float_from_number),
        RuntimeCoercionCase((int,), _float_from_number),
        RuntimeCoercionCase((float,), _float_identity),
        RuntimeCoercionCase((str, dict, list, tuple, set, _NONE_TYPE), _none),
    ),
)

STRING_KEY_DICT_OPTIONAL_POLICY = RuntimeOptionalCoercion[dict[str, object]](
    policy_name="str_key_dict_optional",
    cases=(
        RuntimeCoercionCase((dict,), _string_key_mapping),
        RuntimeCoercionCase((list, tuple, set, str, int, float, bool, _NONE_TYPE), _none),
    ),
)

INT_OPTIONAL_POLICY = NON_BOOL_INT_OPTIONAL_POLICY

__all__ = [
    "CORE_STR_OPTIONAL_POLICY",
    "FLOAT_ONLY_OPTIONAL_POLICY",
    "FLOAT_OPTIONAL_POLICY",
    "INT_LIKE_OPTIONAL_POLICY",
    "INT_OPTIONAL_POLICY",
    "JSON_LIST_OPTIONAL_POLICY",
    "JSON_MAPPING_OPTIONAL_POLICY",
    "LIST_OPTIONAL_POLICY",
    "LIST_OR_TUPLE_TO_LIST_OPTIONAL_POLICY",
    "MAPPING_OPTIONAL_POLICY",
    "NON_BOOL_FLOAT_OPTIONAL_POLICY",
    "NON_BOOL_INT_OPTIONAL_POLICY",
    "ROW_FLOAT_OPTIONAL_POLICY",
    "RuntimeCoercionCase",
    "RuntimeOptionalCoercion",
    "STRING_KEY_DICT_OPTIONAL_POLICY",
    "STR_OPTIONAL_POLICY",
]

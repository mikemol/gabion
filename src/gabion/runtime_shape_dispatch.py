from __future__ import annotations

from gabion.json_types import JSONValue
from gabion.runtime.coercion_contract import (
    FLOAT_OPTIONAL_POLICY,
    INT_OPTIONAL_POLICY,
    JSON_LIST_OPTIONAL_POLICY,
    JSON_MAPPING_OPTIONAL_POLICY,
    STR_OPTIONAL_POLICY,
)


def json_mapping_optional(value: object) -> dict[str, JSONValue] | None:
    return JSON_MAPPING_OPTIONAL_POLICY(value)


def json_list_optional(value: object) -> list[JSONValue] | None:
    return JSON_LIST_OPTIONAL_POLICY(value)


def str_optional(value: object) -> str | None:
    return STR_OPTIONAL_POLICY(value)


def int_optional(value: object) -> int | None:
    return INT_OPTIONAL_POLICY(value)


def float_optional(value: object) -> float | None:
    return FLOAT_OPTIONAL_POLICY(value)

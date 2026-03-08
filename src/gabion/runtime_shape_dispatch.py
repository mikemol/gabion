from __future__ import annotations

from functools import singledispatch

from gabion.invariants import never
from gabion.json_types import JSONValue

_NONE_TYPE = type(None)


def _none_mapping(value: object):
    _ = value
    return None


@singledispatch
def json_mapping_optional(value: object):
    never("unregistered runtime type", value_type=type(value).__name__)


@json_mapping_optional.register(dict)
def _(value: dict[str, JSONValue]):
    return value


for _runtime_type in (list, tuple, set, str, int, float, bool, bytes, bytearray, _NONE_TYPE):
    json_mapping_optional.register(_runtime_type)(_none_mapping)


def _none_list(value: object):
    _ = value
    return None


@singledispatch
def json_list_optional(value: object):
    never("unregistered runtime type", value_type=type(value).__name__)


@json_list_optional.register(list)
def _(value: list[JSONValue]):
    return value


for _runtime_type in (tuple, set, dict, str, int, float, bool, bytes, bytearray, _NONE_TYPE):
    json_list_optional.register(_runtime_type)(_none_list)


def _none_str(value: object):
    _ = value
    return None


@singledispatch
def str_optional(value: object):
    never("unregistered runtime type", value_type=type(value).__name__)


@str_optional.register(str)
def _(value: str):
    return value


for _runtime_type in (int, float, bool, dict, list, tuple, set, bytes, bytearray, _NONE_TYPE):
    str_optional.register(_runtime_type)(_none_str)


def _none_int(value: object):
    _ = value
    return None


@singledispatch
def int_optional(value: object):
    never("unregistered runtime type", value_type=type(value).__name__)


@int_optional.register(int)
def _(value: int):
    return value


@int_optional.register(bool)
def _(value: bool):
    _ = value
    return None


for _runtime_type in (float, str, dict, list, tuple, set, bytes, bytearray, _NONE_TYPE):
    int_optional.register(_runtime_type)(_none_int)


def _none_float(value: object):
    _ = value
    return None


@singledispatch
def float_optional(value: object):
    never("unregistered runtime type", value_type=type(value).__name__)


@float_optional.register(float)
def _(value: float):
    return value


@float_optional.register(int)
def _(value: int):
    return float(value)


@float_optional.register(bool)
def _(value: bool):
    return float(value)


for _runtime_type in (str, dict, list, tuple, set, bytes, bytearray, _NONE_TYPE):
    float_optional.register(_runtime_type)(_none_float)

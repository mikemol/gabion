from __future__ import annotations

from functools import singledispatch

from gabion.invariants import never
from gabion.json_types import JSONValue

_NONE_TYPE = type(None)


def _none_mapping(value: object):
    _ = value
    return None


@singledispatch
def json_mapping_or_none(value: object):
    never("unregistered runtime type", value_type=type(value).__name__)


@json_mapping_or_none.register(dict)
def _(value: dict[str, JSONValue]):
    return value


for _runtime_type in (list, tuple, set, str, int, float, bool, bytes, bytearray, _NONE_TYPE):
    json_mapping_or_none.register(_runtime_type)(_none_mapping)


def _none_list(value: object):
    _ = value
    return None


@singledispatch
def json_list_or_none(value: object):
    never("unregistered runtime type", value_type=type(value).__name__)


@json_list_or_none.register(list)
def _(value: list[JSONValue]):
    return value


for _runtime_type in (tuple, set, dict, str, int, float, bool, bytes, bytearray, _NONE_TYPE):
    json_list_or_none.register(_runtime_type)(_none_list)


def _none_str(value: object):
    _ = value
    return None


@singledispatch
def str_or_none(value: object):
    never("unregistered runtime type", value_type=type(value).__name__)


@str_or_none.register(str)
def _(value: str):
    return value


for _runtime_type in (int, float, bool, dict, list, tuple, set, bytes, bytearray, _NONE_TYPE):
    str_or_none.register(_runtime_type)(_none_str)


def _none_int(value: object):
    _ = value
    return None


@singledispatch
def int_or_none(value: object):
    never("unregistered runtime type", value_type=type(value).__name__)


@int_or_none.register(int)
def _(value: int):
    return value


@int_or_none.register(bool)
def _(value: bool):
    _ = value
    return None


for _runtime_type in (float, str, dict, list, tuple, set, bytes, bytearray, _NONE_TYPE):
    int_or_none.register(_runtime_type)(_none_int)


def _none_float(value: object):
    _ = value
    return None


@singledispatch
def float_or_none(value: object):
    never("unregistered runtime type", value_type=type(value).__name__)


@float_or_none.register(float)
def _(value: float):
    return value


@float_or_none.register(int)
def _(value: int):
    return float(value)


@float_or_none.register(bool)
def _(value: bool):
    return float(value)


for _runtime_type in (str, dict, list, tuple, set, bytes, bytearray, _NONE_TYPE):
    float_or_none.register(_runtime_type)(_none_float)

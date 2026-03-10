from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(frozen=True)
class ObjectEntry:
    key: str
    value: object


class FrozenObjectMap(dict[str, object]):
    def __init__(self, entries: Iterable[ObjectEntry]) -> None:
        super().__init__(map(lambda entry: (entry.key, entry.value), entries))

    def _immutable(self, *args: object, **kwargs: object) -> None:
        _ = args
        _ = kwargs
        raise TypeError("FrozenObjectMap is immutable")

    __setitem__ = _immutable
    __delitem__ = _immutable
    clear = _immutable
    pop = _immutable
    popitem = _immutable
    setdefault = _immutable
    update = _immutable


def make_object_map(entries: Iterable[ObjectEntry]) -> FrozenObjectMap:
    return FrozenObjectMap(entries)


__all__ = [
    "FrozenObjectMap",
    "ObjectEntry",
    "make_object_map",
]

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from itertools import chain, tee
from typing import Generic, TypeVar

_StreamItem = TypeVar("_StreamItem")
_SourceItem = TypeVar("_SourceItem")
_MappedItem = TypeVar("_MappedItem")


@dataclass(frozen=True)
class ReplayableStream(Generic[_StreamItem]):
    factory: Callable[[], Iterator[_StreamItem]]

    def __iter__(self) -> Iterator[_StreamItem]:
        return self.factory()


def empty_stream[T]() -> ReplayableStream[T]:
    return ReplayableStream(factory=lambda: iter(()))


def stream_from_single[T](value: T) -> ReplayableStream[T]:
    return ReplayableStream(factory=lambda: iter((value,)))


def stream_from_iterable[T](values: Iterable[T]) -> ReplayableStream[T]:
    return ReplayableStream(factory=lambda: iter(values))


def stream_from_iterator[T](values: Iterator[T]) -> ReplayableStream[T]:
    source = values

    def iter_items() -> Iterator[T]:
        nonlocal source
        source, clone = tee(source)
        return clone

    return ReplayableStream(factory=iter_items)


def map_stream[S, T](
    source: ReplayableStream[S],
    map_fn: Callable[[S], T],
) -> ReplayableStream[T]:
    return ReplayableStream(factory=lambda: (map_fn(item) for item in source))


def chain_streams[T](*sources: ReplayableStream[T]) -> ReplayableStream[T]:
    return ReplayableStream(
        factory=lambda: chain.from_iterable(iter(source) for source in sources)
    )


__all__ = [
    "ReplayableStream",
    "chain_streams",
    "empty_stream",
    "map_stream",
    "stream_from_iterable",
    "stream_from_iterator",
    "stream_from_single",
]

from __future__ import annotations

from gabion.foundation.replayable_stream import (
    chain_streams,
    map_stream,
    stream_from_factory,
    stream_from_iterable,
    stream_from_iterator,
    stream_from_single,
)


def test_stream_from_iterator_is_replayable() -> None:
    stream = stream_from_iterator(iter(["a", "b", "c"]))

    assert list(stream) == ["a", "b", "c"]
    assert list(stream) == ["a", "b", "c"]


def test_stream_from_single_is_stable() -> None:
    stream = stream_from_single("alpha")

    assert list(stream) == ["alpha"]
    assert list(stream) == ["alpha"]


def test_stream_from_factory_is_replayable() -> None:
    stream = stream_from_factory(lambda: iter(("x", "y")))

    assert list(stream) == ["x", "y"]
    assert list(stream) == ["x", "y"]


def test_chain_streams_preserves_composition_order() -> None:
    stream = chain_streams(
        stream_from_iterable(("a", "b")),
        stream_from_single("c"),
        stream_from_iterable(("d",)),
    )

    assert list(stream) == ["a", "b", "c", "d"]
    assert list(stream) == ["a", "b", "c", "d"]


def test_map_stream_replays_mapped_values() -> None:
    stream = map_stream(
        stream_from_iterator(iter([1, 2, 3])),
        lambda value: value * 10,
    )

    assert list(stream) == [10, 20, 30]
    assert list(stream) == [10, 20, 30]

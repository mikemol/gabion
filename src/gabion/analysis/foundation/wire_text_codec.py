from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator, Mapping

from gabion.analysis.foundation.wire_types import WireObject, WireValue


def encode_text(
    value: object,
    *,
    sort_keys: bool = False,
) -> str:
    return json.dumps(value, sort_keys=sort_keys, separators=None)


def encode_pretty_text(
    value: object,
    *,
    sort_keys: bool = False,
    indent: int = 2,
) -> str:
    return json.dumps(value, sort_keys=sort_keys, indent=indent, separators=None)


def encode_compact_text(
    value: object,
    *,
    sort_keys: bool = False,
) -> str:
    return json.dumps(value, sort_keys=sort_keys, separators=(",", ":"))


def decode_text(text: str) -> object:
    return json.loads(text)


def decode_mapping_text(text: str) -> WireObject:
    decoded = decode_text(text)
    match decoded:
        case dict() as mapping:
            return {str(key): mapping[key] for key in mapping}
        case _:
            raise AssertionError("wire payload must decode to an object mapping")


def decode_mapping_bytes(payload: bytes) -> WireObject:
    try:
        return decode_mapping_text(payload.decode("utf-8"))
    except UnicodeDecodeError as exc:
        raise AssertionError("wire payload must be valid UTF-8 text") from exc


def encode_mapping_bytes(
    payload: Mapping[str, WireValue],
    *,
    sort_keys: bool = True,
) -> bytes:
    return encode_compact_text(payload, sort_keys=sort_keys).encode("utf-8")


def write_pretty_mapping(path: Path, payload: Mapping[str, object]) -> None:
    path.write_text(encode_pretty_text(payload, sort_keys=False), encoding="utf-8")


def append_trailing_newline(path: Path) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write("\n")


def append_line(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(encode_compact_text(payload, sort_keys=False))
        handle.write("\n")


def iter_nonempty_lines(path: Path) -> Iterator[str]:
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if line:
                yield line


def iter_mapping_lines(path: Path) -> Iterator[WireObject]:
    for line in iter_nonempty_lines(path):
        yield decode_mapping_text(line)


def canonical_compact_text(value: object) -> str:
    return encode_compact_text(value, sort_keys=True)


def equivalent_canonical(left: object, right: object) -> bool:
    return canonical_compact_text(left) == canonical_compact_text(right)

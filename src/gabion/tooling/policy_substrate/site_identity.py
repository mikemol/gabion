# gabion:decision_protocol_module
from __future__ import annotations

import hashlib
from functools import reduce


def canonical_site_identity(
    *,
    rel_path: str,
    qualname: str,
    line: int,
    column: int,
    node_kind: str,
    surface: str,
) -> str:
    return stable_hash(
        "site_identity",
        rel_path,
        qualname,
        line,
        column,
        node_kind,
        surface,
    )


def stable_hash(*parts: object) -> str:
    return reduce(_digest_update, parts, hashlib.sha256()).hexdigest()


def _digest_update(digest: object, part: object):
    digest.update(_hash_part_bytes(part))
    digest.update(b"\x00")
    return digest


def _hash_part_bytes(value: object) -> bytes:
    match value:
        case bool() as flag:
            return b"1" if flag else b"0"
        case int() as integer:
            return _int_bytes(integer)
        case str() as text:
            return text.encode("utf-8")
        case bytes() as raw:
            return raw
        case _:
            return b"<unsupported>"


def _int_bytes(value: int) -> bytes:
    magnitude = abs(value)
    width = max(1, (magnitude.bit_length() + 7) // 8)
    sign = b"-" if value < 0 else b"+"
    return sign + magnitude.to_bytes(width, byteorder="big", signed=False)


__all__ = ["canonical_site_identity", "stable_hash"]

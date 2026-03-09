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
        str(int(line)),
        str(int(column)),
        node_kind,
        surface,
    )


def stable_hash(*parts: str) -> str:
    return reduce(_digest_update, parts, hashlib.sha256()).hexdigest()


def _digest_update(digest: object, part: str):
    digest.update(part.encode("utf-8"))
    digest.update(b"\x00")
    return digest


__all__ = ["canonical_site_identity", "stable_hash"]

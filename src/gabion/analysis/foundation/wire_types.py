from __future__ import annotations

"""Typed wire-carrier aliases for boundary-serializable values.

Core analysis modules should use this naming so storage format details remain
outside semantic neighborhoods.
"""

from typing import TypeAlias

from gabion.json_types import (
    JSONArray as WireArray,
    JSONObject as WireObject,
    JSONScalar as WireScalar,
    JSONValue as WireValue,
)

ParseFailureWitnesses: TypeAlias = list[WireObject]

__all__ = [
    "WireScalar",
    "WireValue",
    "WireObject",
    "WireArray",
    "ParseFailureWitnesses",
]


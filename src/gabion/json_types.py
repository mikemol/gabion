from __future__ import annotations

"""JSON-like value types used at semantic transport / artifact boundaries.

These aliases intentionally avoid `object`/`Any` so "anonymous schema" surfaces
remain auditable: if an artifact is meant to be JSON, its value space should be
explicitly declared as JSON-compatible.
"""

from typing import TypeAlias


JSONScalar: TypeAlias = str | int | float | bool | None
JSONValue: TypeAlias = JSONScalar | list["JSONValue"] | dict[str, "JSONValue"]
JSONObject: TypeAlias = dict[str, JSONValue]
JSONArray: TypeAlias = list[JSONValue]

from __future__ import annotations

"""Compatibility re-export for JSON value aliases.

Keep JSON-ish boundary types outside the analysis core so thin clients (CLI,
helpers) can depend on them without importing `gabion.analysis`.
"""

from gabion.json_types import JSONArray, JSONObject, JSONScalar, JSONValue

__all__ = ["JSONScalar", "JSONValue", "JSONObject", "JSONArray"]


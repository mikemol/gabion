from __future__ import annotations

from importlib import import_module
from typing import Mapping

from gabion.json_types import JSONValue


def _yaml_module():
    return import_module("yaml")


def parse_strict_yaml_frontmatter(
    text: str,
    *,
    require_parser: bool = False,
) -> tuple[dict[str, JSONValue], str]:
    if not text.startswith("---\n"):
        return {}, text
    lines = text.splitlines()
    end = None
    for index in range(1, len(lines)):
        if lines[index].strip() == "---":
            end = index
            break
    if end is None:
        return {}, text
    raw = lines[1:end]
    body = "\n".join(lines[end + 1 :])
    try:
        yaml = _yaml_module()
    except ImportError:
        if require_parser:
            raise
        return {}, body
    try:
        parsed = yaml.safe_load("\n".join(raw))
    except Exception:
        parsed = None
    if not isinstance(parsed, Mapping):
        return {}, body
    normalized: dict[str, JSONValue] = {}
    for key, value in parsed.items():
        normalized[str(key)] = value
    return normalized, body

from __future__ import annotations

from importlib import import_module
from typing import Mapping

from gabion.json_types import JSONValue


class FrontmatterParseError(ValueError):
    pass


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
        if require_parser:
            raise FrontmatterParseError("unterminated YAML frontmatter")
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
    except Exception as exc:
        if require_parser:
            raise FrontmatterParseError("invalid YAML frontmatter") from exc
        parsed = None
    if parsed is None:
        return {}, body
    if not isinstance(parsed, Mapping):
        if require_parser:
            raise FrontmatterParseError("frontmatter root must be a mapping")
        return {}, body
    normalized: dict[str, JSONValue] = {}
    for key, value in parsed.items():
        normalized[str(key)] = value
    return normalized, body

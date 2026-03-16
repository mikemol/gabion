from __future__ import annotations

# gabion:grade_boundary kind=semantic_carrier_adapter name=yaml_frontmatter_boundary
from gabion.frontmatter_ingress import FrontmatterParseMode, parse_frontmatter_document
from gabion.json_types import JSONValue


class FrontmatterParseError(ValueError):
    pass


def parse_lenient_yaml_frontmatter(
    text: str,
) -> tuple[dict[str, JSONValue], str]:
    carrier = parse_frontmatter_document(text)
    if carrier.mode is FrontmatterParseMode.YAML:
        return carrier.payload_mapping(), carrier.body
    return {}, carrier.body


def parse_strict_yaml_frontmatter(
    text: str,
    *,
    require_parser: bool = False,
) -> tuple[dict[str, JSONValue], str]:
    carrier = parse_frontmatter_document(text)
    if carrier.mode is FrontmatterParseMode.YAML:
        return carrier.payload_mapping(), carrier.body
    if require_parser:
        if not carrier.parser_available:
            raise ImportError("No module named 'yaml'")
        if carrier.is_unterminated:
            raise FrontmatterParseError("unterminated YAML frontmatter")
        if carrier.detail and carrier.detail.startswith(
            "frontmatter root must be a mapping"
        ):
            raise FrontmatterParseError("frontmatter root must be a mapping")
        if carrier.mode is FrontmatterParseMode.YAML_PARSE_FAILED:
            raise FrontmatterParseError("invalid YAML frontmatter")
    return {}, carrier.body

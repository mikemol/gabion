from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

from gabion.analysis.foundation.timeout_context import check_deadline

FrontmatterScalar: TypeAlias = str | int
FrontmatterValue: TypeAlias = FrontmatterScalar | list["FrontmatterValue"] | dict[str, "FrontmatterValue"]
FrontmatterMap: TypeAlias = dict[str, FrontmatterValue]


@dataclass(frozen=True)
class ParsedFrontmatter:
    mapping: FrontmatterMap
    body: str


def parse_frontmatter(text: str) -> ParsedFrontmatter:
    if not text.startswith("---\n"):
        return ParsedFrontmatter(mapping={}, body=text)
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return ParsedFrontmatter(mapping={}, body=text)
    frontmatter_lines: list[str] = []
    index = 1
    while index < len(lines):
        check_deadline()
        line = lines[index]
        if line.strip() == "---":
            index += 1
            body = "\n".join(lines[index:])
            return ParsedFrontmatter(mapping=_parse_yaml_like(frontmatter_lines), body=body)
        frontmatter_lines.append(line)
        index += 1
    return ParsedFrontmatter(mapping={}, body=text)


def _parse_yaml_like(lines: list[str]) -> FrontmatterMap:
    def _parse_scalar(raw: str) -> FrontmatterScalar:
        value = raw.strip()
        if (value.startswith('"') and value.endswith('"')) or (
            value.startswith("'") and value.endswith("'")
        ):
            value = value[1:-1]
        return int(value) if value.isdigit() else value

    def _next_significant(start: int) -> tuple[int, str] | None:
        idx = start
        while idx < len(lines):
            check_deadline()
            candidate = lines[idx].rstrip()
            stripped = candidate.strip()
            if not stripped or stripped.startswith("#"):
                idx += 1
                continue
            return idx, candidate
        return None

    data: FrontmatterMap = {}
    stack: list[tuple[int, object]] = [(0, data)]
    idx = 0
    while idx < len(lines):
        check_deadline()
        raw = lines[idx].rstrip()
        idx += 1
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue

        indent = len(raw) - len(raw.lstrip(" "))
        while stack and indent < stack[-1][0]:
            check_deadline()
            stack.pop()
        if not stack:
            stack = [(0, data)]
        container = stack[-1][1]

        if stripped.startswith("- "):
            if not isinstance(container, list):
                continue
            item = stripped[2:].strip()
            if ":" not in item:
                container.append(_parse_scalar(item))
                continue
            key, value = item.split(":", 1)
            entry: dict[str, FrontmatterValue] = {}
            key = key.strip()
            value = value.strip()
            if value:
                entry[key] = _parse_scalar(value)
                container.append(entry)
                lookahead = _next_significant(idx)
                if lookahead and (len(lookahead[1]) - len(lookahead[1].lstrip(" "))) > indent:
                    stack.append((indent + 2, entry))
                continue
            lookahead = _next_significant(idx)
            nested: FrontmatterValue = (
                [] if lookahead and lookahead[1].lstrip().startswith("- ") else {}
            )
            entry[key] = nested
            container.append(entry)
            stack.append((indent + 2, nested))
            continue

        if ":" not in stripped or not isinstance(container, dict):
            continue
        key, value = stripped.split(":", 1)
        key = key.strip()
        value = value.strip()
        if value in ("[]", "[ ]"):
            container[key] = []
            continue
        if value in ("{}", "{ }"):
            container[key] = {}
            continue
        if value:
            container[key] = _parse_scalar(value)
            continue
        lookahead = _next_significant(idx)
        nested = [] if lookahead and lookahead[1].lstrip().startswith("- ") else {}
        container[key] = nested
        stack.append((indent + 2, nested))

    return data


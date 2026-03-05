from __future__ import annotations

import re
from collections.abc import Iterable

from gabion_governance.compliance_render.decision_contracts import DecisionSurface, LintEntry

from .contracts import SurfaceParseResult

PARAM_RE = re.compile(r"param '([^']+)'\s*\(")
LINT_ENTRY_RE = re.compile(r"^(?P<path>.+?):(?P<line>\d+):(?P<col>\d+):\s*(?P<rest>.*)$")
DECISION_RE = re.compile(
    r"^(?P<path>[^:]+):(?P<qual>\S+) decision surface params: (?P<params>.+) \((?P<meta>[^)]+)\)$"
)
VALUE_DECISION_RE = re.compile(
    r"^(?P<path>[^:]+):(?P<qual>\S+) value-encoded decision params: (?P<params>.+) \((?P<meta>[^)]+)\)$"
)


def parse_lint_entry(line: str) -> LintEntry | None:
    match = LINT_ENTRY_RE.match(line.strip())
    if not match:
        return None
    try:
        line_no = int(match.group("line"))
        col_no = int(match.group("col"))
    except ValueError:
        return None
    remainder = match.group("rest")
    remainder_parts = remainder.split(" ", 1) if remainder else []
    if not remainder_parts:
        return None
    code = remainder_parts[0]
    message = remainder_parts[1] if len(remainder_parts) > 1 else ""
    param_match = PARAM_RE.search(message)
    return LintEntry(
        path=match.group("path"),
        line=line_no,
        col=col_no,
        code=code,
        message=message,
        param=param_match.group(1) if param_match else None,
    )


def parse_surface_line(line: str, *, value_encoded: bool) -> SurfaceParseResult | None:
    pattern = VALUE_DECISION_RE if value_encoded else DECISION_RE
    match = pattern.match(line.strip())
    if not match:
        return None
    params = tuple(p.strip() for p in match.group("params").split(",") if p.strip())
    return SurfaceParseResult(
        path=match.group("path"),
        qual=match.group("qual"),
        params=params,
        meta=match.group("meta"),
    )


def parse_surfaces(lines: Iterable[str], *, value_encoded: bool) -> list[DecisionSurface]:
    parsed: list[DecisionSurface] = []
    for line in lines:
        outcome = parse_surface_line(line, value_encoded=value_encoded)
        if outcome is None:
            continue
        parsed.append(
            DecisionSurface(
                path=outcome.path,
                qual=outcome.qual,
                params=outcome.params,
                meta=outcome.meta,
            )
        )
    return parsed

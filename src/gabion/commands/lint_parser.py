from __future__ import annotations

import re

from gabion.schema import LintEntryDTO

_LINT_RE = re.compile(r"^(?P<path>.+?):(?P<line>\d+):(?P<col>\d+):\s*(?P<rest>.*)$")


def parse_lint_line(line: str) -> LintEntryDTO | None:
    match = _LINT_RE.match(line.strip())
    if not match:
        return None
    rest = match.group("rest").strip()
    if not rest:
        return None
    code, _, message = rest.partition(" ")
    return LintEntryDTO(
        path=match.group("path"),
        line=int(match.group("line")),
        col=int(match.group("col")),
        code=code,
        message=message,
    )

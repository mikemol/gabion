# gabion:decision_protocol_module
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Iterable, Sequence

from gabion.invariants import never
from gabion.analysis.report_markdown import render_report_markdown
from gabion.analysis.timeout_context import check_deadline


@dataclass
class ReportDoc:
    doc_id: str
    doc_scope: tuple[str, ...] = ("repo", "artifacts")
    _lines: list[str] = field(default_factory=list)

    def lines(self, items: Iterable[str]) -> None:
        self._lines.extend(items)

    def line(self, value: str = "") -> None:
        self._lines.append(value)

    def section(self, title: str) -> None:
        self._lines.append(f"{title}:")

    def header(self, level: int, title: str) -> None:
        if level < 1 or level > 6:
            never(
                "report header level out of range",
                level=level,
            )
        self._lines.append(f"{'#' * level} {title}")

    def bullets(self, items: Iterable[str]) -> None:
        check_deadline()
        for item in items:
            check_deadline()
            self._lines.append(f"- {item}")

    def codeblock(self, content: str | object, *, language: str = "") -> None:
        if isinstance(content, str):
            rendered = content
        else:
            rendered = json.dumps(content, indent=2, sort_keys=False)
        fence = f"```{language}" if language else "```"
        self._lines.append(fence)
        self._lines.extend(rendered.splitlines() or [""])
        self._lines.append("```")

    def table(self, headers: Sequence[str], rows: Iterable[Sequence[object]]) -> None:
        check_deadline()
        header_cells = [str(entry) for entry in headers]
        if not header_cells:
            never("report table requires at least one header")
        self._lines.append("| " + " | ".join(header_cells) + " |")
        self._lines.append("| " + " | ".join("---" for _ in header_cells) + " |")
        for row in rows:
            check_deadline()
            row_cells = [str(entry) for entry in row]
            if len(row_cells) != len(header_cells):
                never(
                    "report table row length mismatch",
                    expected=len(header_cells),
                    actual=len(row_cells),
                )
            self._lines.append("| " + " | ".join(row_cells) + " |")

    def emit(self) -> str:
        return render_report_markdown(
            self.doc_id,
            self._lines,
            doc_scope=self.doc_scope,
        )

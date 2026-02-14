from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Iterable

from gabion.analysis.report_markdown import render_report_markdown


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

    def bullets(self, items: Iterable[str]) -> None:
        for item in items:
            self._lines.append(f"- {item}")

    def codeblock(self, content: str | object, *, language: str = "") -> None:
        if isinstance(content, str):
            rendered = content
        else:
            rendered = json.dumps(content, sort_keys=True)
        fence = f"```{language}" if language else "```"
        self._lines.append(fence)
        self._lines.extend(rendered.splitlines() or [""])
        self._lines.append("```")

    def emit(self) -> str:
        return render_report_markdown(
            self.doc_id,
            self._lines,
            doc_scope=self.doc_scope,
        )

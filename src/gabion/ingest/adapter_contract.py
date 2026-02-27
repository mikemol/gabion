from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from gabion.analysis.dataflow_audit import AuditConfig


@dataclass(frozen=True)
class ParsedFileUnit:
    path: Path
    tree: ast.AST
    function_count: int


@dataclass(frozen=True)
class NormalizedIngestBundle:
    language_id: str
    file_paths: tuple[Path, ...]
    parsed_units: tuple[ParsedFileUnit, ...]


@runtime_checkable
class LanguageAdapter(Protocol):
    language_id: str
    file_extensions: tuple[str, ...]

    def discover_files(
        self,
        paths: list[Path],
        *,
        config: AuditConfig,
    ) -> list[Path]: ...

    def parse_files(self, paths: list[Path]) -> list[ParsedFileUnit]: ...

    def normalize(
        self,
        paths: list[Path],
        *,
        config: AuditConfig,
    ) -> NormalizedIngestBundle: ...

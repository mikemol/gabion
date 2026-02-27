from __future__ import annotations

import ast
from pathlib import Path

from gabion.analysis.dataflow_audit import AuditConfig, _collect_functions, resolve_analysis_paths
from gabion.ingest.adapter_contract import (
    LanguageAdapter,
    NormalizedIngestBundle,
    ParsedFileUnit,
)


class PythonAdapter(LanguageAdapter):
    language_id = "python"
    file_extensions = (".py",)

    def discover_files(self, paths: list[Path], *, config: AuditConfig) -> list[Path]:
        return resolve_analysis_paths(paths, config=config)

    def parse_files(self, paths: list[Path]) -> list[ParsedFileUnit]:
        parsed_units: list[ParsedFileUnit] = []
        for path in paths:
            source = path.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(path))
            parsed_units.append(
                ParsedFileUnit(
                    path=path,
                    tree=tree,
                    function_count=len(_collect_functions(tree)),
                )
            )
        return parsed_units

    def normalize(self, paths: list[Path], *, config: AuditConfig) -> NormalizedIngestBundle:
        discovered_paths = self.discover_files(paths, config=config)
        parsed_units = self.parse_files(discovered_paths)
        return NormalizedIngestBundle(
            language_id=self.language_id,
            file_paths=tuple(discovered_paths),
            parsed_units=tuple(parsed_units),
        )

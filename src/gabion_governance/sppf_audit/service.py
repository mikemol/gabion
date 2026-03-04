from __future__ import annotations

from pathlib import Path
from typing import Callable

from gabion_governance.docflow_audit.contracts import Doc

from .contracts import SppfGraphResult, SppfStatusConsistencyResult


def build_sppf_graph(*, root: Path, issues_json: Path | None, build_graph: Callable[..., dict[str, object]]) -> SppfGraphResult:
    return SppfGraphResult(graph=build_graph(root, issues_json=issues_json))


def run_status_consistency(
    *,
    root: Path,
    extra_paths: list[str] | None,
    load_docs: Callable[..., dict[str, Doc]],
    axis_audit: Callable[..., tuple[list[str], list[str]]],
    sync_check: Callable[..., tuple[list[str], list[str]]],
) -> SppfStatusConsistencyResult:
    docs = load_docs(root=root, extra_paths=extra_paths)
    violations, warnings = axis_audit(root, docs)
    sync_violations, sync_warnings = sync_check(root, mode="required")
    return SppfStatusConsistencyResult(
        violations=[*violations, *sync_violations],
        warnings=[*warnings, *sync_warnings],
    )

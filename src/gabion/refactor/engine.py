from __future__ import annotations

from pathlib import Path

import libcst as cst

from gabion.refactor.model import RefactorPlan, RefactorRequest


class RefactorEngine:
    def __init__(self, project_root: Path | None = None) -> None:
        self.project_root = project_root

    def plan_protocol_extraction(self, request: RefactorRequest) -> RefactorPlan:
        path = Path(request.target_path)
        if self.project_root and not path.is_absolute():
            path = self.project_root / path
        try:
            source = path.read_text()
        except Exception as exc:
            return RefactorPlan(errors=[f"Failed to read {path}: {exc}"])
        try:
            cst.parse_module(source)
        except Exception as exc:
            return RefactorPlan(errors=[f"LibCST parse failed for {path}: {exc}"])
        protocol = request.protocol_name or "(unnamed)"
        warning = (
            "Protocol refactor engine scaffold only; no edits generated for "
            f"{protocol}."
        )
        return RefactorPlan(warnings=[warning])

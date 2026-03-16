from __future__ import annotations

from gabion.server_core.primitive_contract_registry import build_contract_class


AnalysisPrimitives = build_contract_class("AnalysisPrimitives", module_name=__name__)


def default_analysis_primitives() -> AnalysisPrimitives:
    return AnalysisPrimitives()


__all__ = ["AnalysisPrimitives", "default_analysis_primitives"]

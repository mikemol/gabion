from __future__ import annotations

from gabion.server_core.primitive_contract_registry import build_contract_class


ProgressStageContract = build_contract_class("ProgressStageContract", module_name=__name__)


__all__ = ["ProgressStageContract"]

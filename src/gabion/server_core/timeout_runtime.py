from __future__ import annotations

from gabion.server_core.primitive_contract_registry import build_contract_class


TimeoutStageRuntime = build_contract_class("TimeoutStageRuntime", module_name=__name__)


__all__ = ["TimeoutStageRuntime"]

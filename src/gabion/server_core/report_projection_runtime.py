from __future__ import annotations

from gabion.server_core.primitive_contract_registry import build_contract_class


ReportProjectionRuntime = build_contract_class("ReportProjectionRuntime", module_name=__name__)


__all__ = ["ReportProjectionRuntime"]

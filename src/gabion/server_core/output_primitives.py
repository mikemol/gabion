from __future__ import annotations

from gabion.server_core.primitive_contract_registry import build_contract_class


OutputPrimitives = build_contract_class("OutputPrimitives", module_name=__name__)


def default_output_primitives() -> OutputPrimitives:
    return OutputPrimitives()


__all__ = ["OutputPrimitives", "default_output_primitives"]

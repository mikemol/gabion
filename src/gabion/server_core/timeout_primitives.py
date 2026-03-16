from __future__ import annotations

from gabion.server_core.primitive_contract_registry import build_contract_class


TimeoutPrimitives = build_contract_class("TimeoutPrimitives", module_name=__name__)


def default_timeout_primitives() -> TimeoutPrimitives:
    return TimeoutPrimitives()


__all__ = ["TimeoutPrimitives", "default_timeout_primitives"]

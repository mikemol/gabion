from __future__ import annotations

from gabion.server_core.primitive_contract_registry import build_contract_class


ProgressPrimitives = build_contract_class("ProgressPrimitives", module_name=__name__)


def default_progress_primitives() -> ProgressPrimitives:
    return ProgressPrimitives()


__all__ = ["ProgressPrimitives", "default_progress_primitives"]

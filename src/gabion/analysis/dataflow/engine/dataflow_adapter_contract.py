# gabion:boundary_normalization_module
# gabion:decision_protocol_module
from __future__ import annotations

"""Adapter contract normalization owner surface."""

from dataclasses import dataclass
from typing import cast

from gabion.analysis.foundation.json_types import JSONObject


@dataclass(frozen=True)
class AdapterCapabilities:
    bundle_inference: bool = True
    decision_surfaces: bool = True
    type_flow: bool = True
    exception_obligations: bool = True
    rewrite_plan_support: bool = True


def parse_adapter_capabilities(payload: object) -> AdapterCapabilities:
    if type(payload) is not dict:
        return AdapterCapabilities()
    raw = cast(dict[object, object], payload)

    def _read(name: str, default: bool = True) -> bool:
        value = raw.get(name)
        if type(value) is bool:
            return bool(value)
        return default

    return AdapterCapabilities(
        bundle_inference=_read("bundle_inference"),
        decision_surfaces=_read("decision_surfaces"),
        type_flow=_read("type_flow"),
        exception_obligations=_read("exception_obligations"),
        rewrite_plan_support=_read("rewrite_plan_support"),
    )


def normalize_adapter_contract(payload: object) -> JSONObject:
    if type(payload) is not dict:
        return {"name": "native", "capabilities": AdapterCapabilities().__dict__}
    raw = cast(dict[object, object], payload)
    name = str(raw.get("name", "native") or "native")
    capabilities = parse_adapter_capabilities(raw.get("capabilities")).__dict__
    return {
        "name": name,
        "capabilities": {str(key): bool(capabilities[key]) for key in capabilities},
    }


__all__ = [
    "AdapterCapabilities",
    "normalize_adapter_contract",
    "parse_adapter_capabilities",
]

from __future__ import annotations
from gabion.invariants import never

"""Adapter contract normalization owner surface."""

from dataclasses import dataclass

from gabion.analysis.foundation.json_types import JSONObject


@dataclass(frozen=True)
class AdapterCapabilities:
    bundle_inference: bool = True
    decision_surfaces: bool = True
    type_flow: bool = True
    exception_obligations: bool = True
    rewrite_plan_support: bool = True


def parse_adapter_capabilities(payload: object) -> AdapterCapabilities:
    match payload:
        case dict() as raw:
            def _read(name: str, default: bool = True) -> bool:
                match raw.get(name):
                    case bool() as bool_value:
                        return bool_value
                    case _:
                        return default

                        never("unreachable wildcard match fall-through")
            return AdapterCapabilities(
                bundle_inference=_read("bundle_inference"),
                decision_surfaces=_read("decision_surfaces"),
                type_flow=_read("type_flow"),
                exception_obligations=_read("exception_obligations"),
                rewrite_plan_support=_read("rewrite_plan_support"),
            )
        case _:
            return AdapterCapabilities()


            never("unreachable wildcard match fall-through")
def normalize_adapter_contract(payload: object) -> JSONObject:
    match payload:
        case dict() as raw:
            name = str(raw.get("name", "native") or "native")
            capabilities = parse_adapter_capabilities(raw.get("capabilities")).__dict__
            return {
                "name": name,
                "capabilities": {str(key): bool(capabilities[key]) for key in capabilities},
            }
        case _:
            return {"name": "native", "capabilities": AdapterCapabilities().__dict__}


            never("unreachable wildcard match fall-through")
__all__ = [
    "AdapterCapabilities",
    "normalize_adapter_contract",
    "parse_adapter_capabilities",
]

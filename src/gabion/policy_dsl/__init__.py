from typing import Mapping

from .eval import first_match
from .registry import build_registry
from .schema import PolicyDecision, PolicyDomain


def evaluate_policy(*, domain: PolicyDomain, data: Mapping[str, object]) -> PolicyDecision:
    registry = build_registry()
    return first_match(registry.program, domain=domain, data=dict(data))


__all__ = ["PolicyDecision", "PolicyDomain", "evaluate_policy"]

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .schema import EvidenceContract, PolicyDomain, PolicyOutcomeKind, PolicySeverity


@dataclass(frozen=True)
class IRRule:
    rule_id: str
    domain: PolicyDomain
    severity: PolicySeverity
    predicate: Mapping[str, Any]
    outcome_kind: PolicyOutcomeKind
    outcome_message: str
    outcome_details: Mapping[str, Any]
    evidence_contract: EvidenceContract
    priority: int


@dataclass(frozen=True)
class IRTransform:
    transform_id: str
    domain: PolicyDomain | None
    intro_from: str
    erase_when: str
    priority: int


@dataclass(frozen=True)
class IRProgram:
    rules: tuple[IRRule, ...]
    transforms: tuple[IRTransform, ...] = ()

    def by_domain(self, domain: PolicyDomain) -> tuple[IRRule, ...]:
        return tuple(rule for rule in self.rules if rule.domain is domain)

    def transforms_by_domain(self, domain: PolicyDomain) -> tuple[IRTransform, ...]:
        return tuple(
            transform
            for transform in self.transforms
            if transform.domain in (None, domain)
        )

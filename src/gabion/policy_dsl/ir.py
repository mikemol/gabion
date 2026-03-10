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
    evidence_contract: EvidenceContract
    priority: int


@dataclass(frozen=True)
class IRProgram:
    rules: tuple[IRRule, ...]

    def by_domain(self, domain: PolicyDomain) -> tuple[IRRule, ...]:
        return tuple(rule for rule in self.rules if rule.domain is domain)

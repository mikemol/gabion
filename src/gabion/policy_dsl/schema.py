from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping


class PolicyDomain(str, Enum):
    GOVERNANCE_GATE = "governance_gate"
    AMBIGUITY_CONTRACT = "ambiguity_contract"
    AMBIGUITY_CONTRACT_AST = "ambiguity_contract_ast"
    GRADE_MONOTONICITY = "grade_monotonicity"
    POLICY_SCANNER = "policy_scanner"
    ASPF_OPPORTUNITY = "aspf_opportunity"
    PROJECTION_FIBER = "projection_fiber"


class PolicySeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    BLOCKING = "blocking"


class PolicyOutcomeKind(str, Enum):
    PASS = "pass"
    WARN = "warn"
    BLOCK = "block"
    SKIP = "skip"


class EvidenceContract(str, Enum):
    NONE = "none"
    REPRESENTATIVE_PAIR = "representative_pair"
    TWO_CELL_WITNESS = "two_cell_witness"
    COFIBRATION_WITNESS = "cofibration_witness"


@dataclass(frozen=True)
class RuleIdentity:
    rule_id: str
    domain: PolicyDomain
    severity: PolicySeverity


@dataclass(frozen=True)
class RuleSchema:
    identity: RuleIdentity
    predicate: Mapping[str, Any]
    outcome: Mapping[str, Any]
    evidence_contract: EvidenceContract


@dataclass(frozen=True)
class CompileIssue:
    code: str
    message: str
    rule_id: str | None = None


@dataclass(frozen=True)
class TypecheckIssue:
    code: str
    message: str
    rule_id: str | None = None


@dataclass(frozen=True)
class PolicyDecision:
    rule_id: str
    domain: PolicyDomain
    severity: PolicySeverity
    outcome: PolicyOutcomeKind
    message: str
    evidence_contract: EvidenceContract
    matched: bool
    details: Mapping[str, object]

from __future__ import annotations

from .ir import IRProgram
from .schema import EvidenceContract, PolicyDomain, TypecheckIssue

_ALLOWED_OPS = {"always", "int_gte", "bool_true", "str_eq", "all", "any", "not"}
_ASPF_ALLOWED_EVIDENCE = {
    EvidenceContract.REPRESENTATIVE_PAIR,
    EvidenceContract.TWO_CELL_WITNESS,
    EvidenceContract.COFIBRATION_WITNESS,
}


def typecheck(program: IRProgram) -> list[TypecheckIssue]:
    issues: list[TypecheckIssue] = []
    seen: set[str] = set()
    domain_fallback: set[PolicyDomain] = set()
    for rule in program.rules:
        if rule.rule_id in seen:
            issues.append(TypecheckIssue(code="duplicate_rule_id", message="rule_id must be globally unique", rule_id=rule.rule_id))
        seen.add(rule.rule_id)
        op = str(rule.predicate.get("op", ""))
        if op not in _ALLOWED_OPS:
            issues.append(TypecheckIssue(code="unsupported_predicate_op", message=f"unsupported predicate op: {op}", rule_id=rule.rule_id))
        if op == "always" or rule.outcome_kind.value == "pass":
            domain_fallback.add(rule.domain)
        if (
            rule.domain is PolicyDomain.ASPF_OPPORTUNITY
            and rule.evidence_contract not in _ASPF_ALLOWED_EVIDENCE
        ):
            issues.append(
                TypecheckIssue(
                    code="aspf_evidence_contract_incompatible",
                    message=(
                        "ASPF opportunity rules must use representative_pair, "
                        "two_cell_witness, or cofibration_witness evidence contracts"
                    ),
                    rule_id=rule.rule_id,
                )
            )
    for domain in {item.domain for item in program.rules}:
        if domain not in domain_fallback:
            issues.append(TypecheckIssue(code="missing_totality_fallback", message=f"domain {domain.value} requires at least one always predicate", rule_id=None))
    return sorted(issues, key=lambda item: (item.rule_id or "", item.code, item.message))

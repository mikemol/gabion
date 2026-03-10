from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

from gabion.tooling.governance_rules import gate_policy_to_dsl_sources, load_governance_rules

from .compile import compile_document, compile_rules
from .ir import IRProgram
from .schema import CompileIssue, TypecheckIssue
from .typecheck import typecheck


@dataclass(frozen=True)
class RegistrySnapshot:
    program: IRProgram


def _governance_gate_rules() -> list[Mapping[str, Any]]:
    rules = load_governance_rules()
    compiled: list[Mapping[str, Any]] = []
    for gate_id, gate in sorted(rules.gates.items()):
        gate_rules = [dict(item) for item in gate_policy_to_dsl_sources(gate)]
        for rule in gate_rules:
            rule["predicate"] = {
                "op": "all",
                "predicates": [
                    {"op": "str_eq", "path": ["gate_id"], "value": gate_id},
                    dict(rule["predicate"]),
                ],
            }
        compiled.extend(gate_rules)
    return compiled


def _rules_from_document(path: Path) -> tuple[list[Mapping[str, Any]], list[CompileIssue]]:
    if not path.exists():
        return [], []
    program, issues = compile_document(path)
    if issues:
        return [], issues
    if program is None:
        return [], []
    return [
        {
            "rule_id": rule.rule_id,
            "domain": rule.domain.value,
            "severity": rule.severity.value,
            "predicate": dict(rule.predicate),
            "outcome": {
                "kind": rule.outcome_kind.value,
                "message": rule.outcome_message,
            },
            "evidence_contract": rule.evidence_contract.value,
        }
        for rule in program.rules
    ], []


@lru_cache(maxsize=1)
def build_registry() -> RegistrySnapshot:
    repo_root = Path(__file__).resolve().parents[3]
    aspf_rules, aspf_compile_issues = _rules_from_document(repo_root / "docs" / "aspf_opportunity_rules.yaml")
    policy_rules, policy_compile_issues = _rules_from_document(repo_root / "docs" / "policy_rules.yaml")
    all_rules: list[Mapping[str, Any]] = []
    all_rules.extend(_governance_gate_rules())
    all_rules.extend(policy_rules)
    all_rules.extend(aspf_rules)

    program, compile_issues = compile_rules([dict(item) for item in all_rules])
    merged_compile = [*compile_issues, *policy_compile_issues, *aspf_compile_issues]
    if merged_compile:
        raise ValueError("policy compile failed", merged_compile)
    if program is None:
        raise ValueError("policy compile produced empty program")
    type_issues: list[TypecheckIssue] = typecheck(program)
    if type_issues:
        raise ValueError("policy typecheck failed", type_issues)
    return RegistrySnapshot(program=program)

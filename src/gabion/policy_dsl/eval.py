from __future__ import annotations

from typing import Any, Mapping

from .ir import IRProgram, IRRule
from .schema import PolicyDecision, PolicyDomain, PolicyOutcomeKind


def _get_path(data: Mapping[str, Any], path: tuple[str, ...]) -> Any:
    node: Any = data
    for key in path:
        if not isinstance(node, Mapping):
            return None
        node = node.get(key)
    return node


def _eval_predicate(predicate: Mapping[str, Any], data: Mapping[str, Any]) -> bool:
    op = str(predicate.get("op", ""))
    if op == "always":
        return True
    if op == "bool_true":
        path = tuple(str(p) for p in predicate.get("path", ()))
        return bool(_get_path(data, path))
    if op == "int_gte":
        path = tuple(str(p) for p in predicate.get("path", ()))
        threshold = int(predicate.get("value", 0))
        raw = _get_path(data, path)
        try:
            numeric = int(raw if raw is not None else 0)
        except (TypeError, ValueError):
            numeric = 0
        return numeric >= threshold
    if op == "str_eq":
        path = tuple(str(p) for p in predicate.get("path", ()))
        expected = str(predicate.get("value", ""))
        return str(_get_path(data, path) or "") == expected
    if op == "all":
        children = predicate.get("predicates", [])
        return isinstance(children, list) and all(_eval_predicate(item, data) for item in children if isinstance(item, Mapping))
    if op == "any":
        children = predicate.get("predicates", [])
        return isinstance(children, list) and any(_eval_predicate(item, data) for item in children if isinstance(item, Mapping))
    if op == "not":
        child = predicate.get("predicate")
        return isinstance(child, Mapping) and not _eval_predicate(child, data)
    if op == "rows_any":
        path = tuple(str(p) for p in predicate.get("path", ()))
        row_predicate = predicate.get("predicate")
        rows = _get_path(data, path)
        if not isinstance(rows, list) or not isinstance(row_predicate, Mapping):
            return False
        return any(
            _eval_predicate(row_predicate, row)
            for row in rows
            if isinstance(row, Mapping)
        )
    return False


def evaluate(program: IRProgram, *, domain: PolicyDomain, data: Mapping[str, Any]) -> tuple[PolicyDecision, ...]:
    decisions: list[PolicyDecision] = []
    for rule in sorted(program.by_domain(domain), key=lambda item: (item.priority, item.rule_id)):
        matched = _eval_predicate(rule.predicate, data)
        decisions.append(
            PolicyDecision(
                rule_id=rule.rule_id,
                domain=rule.domain,
                severity=rule.severity,
                outcome=rule.outcome_kind if matched else PolicyOutcomeKind.PASS,
                message=rule.outcome_message if matched else "",
                evidence_contract=rule.evidence_contract,
                matched=matched,
                details={},
            )
        )
        if matched:
            break
    return tuple(decisions)


def first_match(program: IRProgram, *, domain: PolicyDomain, data: Mapping[str, Any]) -> PolicyDecision:
    decisions = evaluate(program, domain=domain, data=data)
    for decision in decisions:
        if decision.matched:
            return decision
    raise ValueError(f"no matching decision for domain {domain.value}")

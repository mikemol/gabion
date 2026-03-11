from __future__ import annotations

from importlib import import_module
from pathlib import Path
from typing import Any, Mapping

from .ir import IRProgram, IRRule, IRTransform
from .schema import CompileIssue, EvidenceContract, PolicyDomain, PolicyOutcomeKind, PolicySeverity, RuleIdentity, RuleSchema


def _yaml_module():
    return import_module("yaml")


def _load_document(path: Path) -> object:
    suffix = path.suffix.lower()
    text = path.read_text(encoding="utf-8")
    if suffix == ".json":
        import json

        return json.loads(text)
    if suffix in {".yaml", ".yml"}:
        yaml = _yaml_module()
        return yaml.safe_load(text)
    raise ValueError(f"unsupported policy document suffix: {suffix}")


def _compile_rule(raw: Mapping[str, Any], *, index: int) -> tuple[IRRule | None, list[CompileIssue]]:
    issues: list[CompileIssue] = []
    rule_id = str(raw.get("rule_id", "")).strip()
    if not rule_id:
        issues.append(CompileIssue(code="missing_rule_id", message="rule_id is required"))
        return None, issues

    def _enum(enum_type: type, value: object, *, field: str):
        try:
            return enum_type(str(value))
        except ValueError:
            issues.append(CompileIssue(code=f"invalid_{field}", message=f"{field} is invalid", rule_id=rule_id))
            return None

    domain = _enum(PolicyDomain, raw.get("domain", ""), field="domain")
    severity = _enum(PolicySeverity, raw.get("severity", ""), field="severity")
    evidence_contract = _enum(EvidenceContract, raw.get("evidence_contract", "none"), field="evidence_contract")
    predicate = raw.get("predicate")
    outcome = raw.get("outcome")
    if not isinstance(predicate, Mapping):
        issues.append(CompileIssue(code="invalid_predicate", message="predicate must be an object", rule_id=rule_id))
    if not isinstance(outcome, Mapping):
        issues.append(CompileIssue(code="invalid_outcome", message="outcome must be an object", rule_id=rule_id))
    if issues:
        return None, issues
    assert isinstance(predicate, Mapping)
    assert isinstance(outcome, Mapping)
    outcome_kind = _enum(PolicyOutcomeKind, outcome.get("kind", ""), field="outcome_kind")
    outcome_message = str(outcome.get("message", "")).strip()
    if not outcome_message:
        issues.append(CompileIssue(code="missing_outcome_message", message="outcome.message is required", rule_id=rule_id))
    if issues:
        return None, issues
    assert domain is not None and severity is not None and outcome_kind is not None and evidence_contract is not None
    schema = RuleSchema(
        identity=RuleIdentity(rule_id=rule_id, domain=domain, severity=severity),
        predicate=predicate,
        outcome=outcome,
        evidence_contract=evidence_contract,
    )
    return IRRule(
        rule_id=schema.identity.rule_id,
        domain=schema.identity.domain,
        severity=schema.identity.severity,
        predicate=schema.predicate,
        outcome_kind=outcome_kind,
        outcome_message=outcome_message,
        evidence_contract=schema.evidence_contract,
        priority=index,
    ), []


def _compile_transform(
    raw: Mapping[str, Any],
    *,
    index: int,
    default_domain: PolicyDomain | None,
) -> tuple[IRTransform | None, list[CompileIssue]]:
    issues: list[CompileIssue] = []
    transform_id = str(raw.get("transform_id", "")).strip()
    if not transform_id:
        issues.append(
            CompileIssue(
                code="missing_transform_id",
                message="transform_id is required",
            )
        )
        return None, issues
    intro_from = str(raw.get("intro_from", "")).strip()
    if not intro_from:
        issues.append(
            CompileIssue(
                code="missing_intro_from",
                message="intro_from is required",
                rule_id=transform_id,
            )
        )
    erase_when = str(raw.get("erase_when", "")).strip()
    if not erase_when:
        issues.append(
            CompileIssue(
                code="missing_erase_when",
                message="erase_when is required",
                rule_id=transform_id,
            )
        )
    raw_domain = raw.get("domain")
    domain_value = str(raw_domain).strip() if raw_domain not in (None,) else ""
    domain: PolicyDomain | None = default_domain
    if domain_value not in ("",):
        try:
            domain = PolicyDomain(domain_value)
        except ValueError:
            issues.append(
                CompileIssue(
                    code="invalid_transform_domain",
                    message="transform domain is invalid",
                    rule_id=transform_id,
                )
            )
    if issues:
        return None, issues
    return IRTransform(
        transform_id=transform_id,
        domain=domain,
        intro_from=intro_from,
        erase_when=erase_when,
        priority=index,
    ), []


def compile_rules(raw_rules: list[Mapping[str, Any]]) -> tuple[IRProgram | None, list[CompileIssue]]:
    issues: list[CompileIssue] = []
    compiled: list[IRRule] = []
    for index, raw in enumerate(raw_rules):
        rule, rule_issues = _compile_rule(raw, index=index)
        issues.extend(rule_issues)
        if rule is not None:
            compiled.append(rule)
    if issues:
        return None, sorted(issues, key=lambda item: (item.rule_id or "", item.code, item.message))
    return IRProgram(
        rules=tuple(sorted(compiled, key=lambda item: (item.priority, item.rule_id))),
    ), []


def compile_document(path: Path) -> tuple[IRProgram | None, list[CompileIssue]]:
    payload = _load_document(path)
    if not isinstance(payload, Mapping):
        return None, [CompileIssue(code="invalid_root", message="policy document root must be an object")]
    rules = payload.get("rules")
    if not isinstance(rules, list):
        return None, [CompileIssue(code="invalid_rules", message="policy document must define rules: []")]
    normalized = [dict(item) for item in rules if isinstance(item, Mapping)]
    program, issues = compile_rules(normalized)
    if issues:
        return None, issues
    assert program is not None
    raw_transforms = payload.get("transforms", [])
    if raw_transforms in (None,):
        raw_transforms = []
    if not isinstance(raw_transforms, list):
        return None, [
            CompileIssue(
                code="invalid_transforms",
                message="policy document transforms must be a list",
            )
        ]
    domains = sorted({rule.domain for rule in program.rules}, key=lambda item: item.value)
    default_domain = domains[0] if len(domains) == 1 else None
    compiled_transforms: list[IRTransform] = []
    transform_issues: list[CompileIssue] = []
    for index, item in enumerate(raw_transforms):
        if not isinstance(item, Mapping):
            transform_issues.append(
                CompileIssue(
                    code="invalid_transform",
                    message="transform entry must be an object",
                )
            )
            continue
        transform, item_issues = _compile_transform(
            item,
            index=index,
            default_domain=default_domain,
        )
        transform_issues.extend(item_issues)
        if transform is not None:
            compiled_transforms.append(transform)
    if transform_issues:
        return None, sorted(
            transform_issues,
            key=lambda item: (item.rule_id or "", item.code, item.message),
        )
    return IRProgram(
        rules=program.rules,
        transforms=tuple(
            sorted(compiled_transforms, key=lambda item: (item.priority, item.transform_id))
        ),
    ), []

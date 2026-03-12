from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

from gabion.tooling.governance.governance_rules import (
    gate_policy_to_dsl_sources,
    load_governance_rules,
)

from .compile import compile_document, compile_rules
from .ir import IRProgram, IRTransform
from .schema import CompileIssue
from .typecheck import typecheck


@dataclass(frozen=True)
class RegistrySnapshot:
    program: IRProgram


def _repo_relative_path(path: Path, *, repo_root: Path) -> str:
    return path.relative_to(repo_root).as_posix()


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


def _program_from_document(
    path: Path,
) -> tuple[IRProgram | None, list[CompileIssue]]:
    if not path.exists():
        return None, []
    return compile_document(path)


def _rule_mapping_from_ir(rule: object) -> Mapping[str, Any]:
    if not hasattr(rule, "rule_id"):
        raise TypeError("IR rule missing rule_id")
    if not hasattr(rule, "domain"):
        raise TypeError("IR rule missing domain")
    if not hasattr(rule, "severity"):
        raise TypeError("IR rule missing severity")
    if not hasattr(rule, "predicate"):
        raise TypeError("IR rule missing predicate")
    if not hasattr(rule, "outcome_kind"):
        raise TypeError("IR rule missing outcome_kind")
    if not hasattr(rule, "outcome_message"):
        raise TypeError("IR rule missing outcome_message")
    if not hasattr(rule, "outcome_details"):
        raise TypeError("IR rule missing outcome_details")
    if not hasattr(rule, "evidence_contract"):
        raise TypeError("IR rule missing evidence_contract")
    return {
        "rule_id": str(getattr(rule, "rule_id")),
        "domain": getattr(rule, "domain").value,
        "severity": getattr(rule, "severity").value,
        "predicate": dict(getattr(rule, "predicate")),
        "outcome": {
            "kind": getattr(getattr(rule, "outcome_kind"), "value"),
            "message": str(getattr(rule, "outcome_message")),
            **dict(getattr(rule, "outcome_details")),
        },
        "evidence_contract": getattr(getattr(rule, "evidence_contract"), "value"),
    }


def _iter_policy_source_documents(repo_root: Path) -> tuple[Path, ...]:
    markdown_rule_root = repo_root / "docs" / "policy_rules"
    markdown_docs = (
        tuple(sorted(markdown_rule_root.glob("*.md"), key=lambda item: item.name))
        if markdown_rule_root.exists()
        else tuple()
    )
    return (
        repo_root / "docs" / "policy_rules.yaml",
        *markdown_docs,
        repo_root / "docs" / "aspf_opportunity_rules.yaml",
        repo_root / "docs" / "projection_fiber_rules.yaml",
    )


def _duplicate_source_issues(
    *,
    kind: str,
    source_map: Mapping[str, set[str]],
) -> list[CompileIssue]:
    issues: list[CompileIssue] = []
    for item_id, sources in sorted(source_map.items()):
        if len(sources) < 2:
            continue
        issues.append(
            CompileIssue(
                code=f"duplicate_{kind}_source",
                message=(
                    f"{kind} {item_id} defined in multiple policy sources: "
                    f"{', '.join(sorted(sources))}"
                ),
                rule_id=item_id,
            )
        )
    return issues


def _build_registry_for_root(repo_root: Path) -> RegistrySnapshot:
    compiled_rule_mappings: list[Mapping[str, Any]] = []
    compiled_transforms: list[IRTransform] = []
    compile_issues: list[CompileIssue] = []
    rule_sources: dict[str, set[str]] = {}
    transform_sources: dict[str, set[str]] = {}

    governance_source = "<governance_gate_rules>"
    governance_rules = _governance_gate_rules()
    compiled_rule_mappings.extend(governance_rules)
    for item in governance_rules:
        rule_id = str(item.get("rule_id", "")).strip()
        if rule_id:
            rule_sources.setdefault(rule_id, set()).add(governance_source)

    for path in _iter_policy_source_documents(repo_root):
        program, issues = _program_from_document(path)
        compile_issues.extend(issues)
        if program is None:
            continue
        source_id = _repo_relative_path(path, repo_root=repo_root)
        for rule in program.rules:
            compiled_rule_mappings.append(_rule_mapping_from_ir(rule))
            rule_sources.setdefault(rule.rule_id, set()).add(source_id)
        for transform in program.transforms:
            compiled_transforms.append(transform)
            transform_sources.setdefault(transform.transform_id, set()).add(source_id)

    compile_issues.extend(_duplicate_source_issues(kind="rule_id", source_map=rule_sources))
    compile_issues.extend(
        _duplicate_source_issues(kind="transform_id", source_map=transform_sources)
    )
    if compile_issues:
        raise ValueError("policy compile failed", compile_issues)

    program, merged_compile_issues = compile_rules([dict(item) for item in compiled_rule_mappings])
    if merged_compile_issues:
        raise ValueError("policy compile failed", merged_compile_issues)
    if program is None:
        raise ValueError("policy compile produced empty program")
    program = IRProgram(
        rules=program.rules,
        transforms=tuple(
            IRTransform(
                transform_id=item.transform_id,
                domain=item.domain,
                intro_from=item.intro_from,
                erase_when=item.erase_when,
                priority=index,
            )
            for index, item in enumerate(compiled_transforms)
        ),
    )
    type_issues = typecheck(program)
    if type_issues:
        raise ValueError("policy typecheck failed", type_issues)
    return RegistrySnapshot(program=program)


@lru_cache(maxsize=1)
def build_registry() -> RegistrySnapshot:
    repo_root = Path(__file__).resolve().parents[3]
    return _build_registry_for_root(repo_root)

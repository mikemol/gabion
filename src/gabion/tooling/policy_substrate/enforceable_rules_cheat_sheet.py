from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse

import yaml

from gabion.tooling.policy_substrate import governance_loop_docs

_RULE_MATRIX_BEGIN = "<!-- BEGIN:generated_rule_matrix -->"
_RULE_MATRIX_END = "<!-- END:generated_rule_matrix -->"
_GUARDRAILS_BEGIN = "<!-- BEGIN:generated_implementation_guardrails -->"
_GUARDRAILS_END = "<!-- END:generated_implementation_guardrails -->"
_QUICK_VALIDATION_BEGIN = "<!-- BEGIN:generated_quick_validation_commands -->"
_QUICK_VALIDATION_END = "<!-- END:generated_quick_validation_commands -->"


@dataclass(frozen=True)
class CheatSheetSourceClause:
    label: str
    href: str


@dataclass(frozen=True)
class CheatSheetRuleRow:
    rule_id: str
    enforceable_rule: str
    source_clauses: tuple[CheatSheetSourceClause, ...]
    operational_check: str
    failure_signal: str


@dataclass(frozen=True)
class CheatSheetGuardrailRow:
    change_type: str
    loop_domains: tuple[str, ...]
    mandatory_checks: tuple[str, ...]
    prohibited_shortcuts: tuple[str, ...]
    required_evidence_artifacts: tuple[str, ...]
    source_clauses: tuple[CheatSheetSourceClause, ...]


@dataclass(frozen=True)
class CheatSheetCommandEntry:
    command: str
    loop_domains: tuple[str, ...]


@dataclass(frozen=True)
class CheatSheetQuickValidation:
    required_commands: tuple[CheatSheetCommandEntry, ...]
    optional_commands: tuple[CheatSheetCommandEntry, ...]


@dataclass(frozen=True)
class EnforceableRulesCatalog:
    version: int
    rule_matrix_rows: tuple[CheatSheetRuleRow, ...]
    implementation_guardrails: tuple[CheatSheetGuardrailRow, ...]
    quick_validation: CheatSheetQuickValidation


def _require_mapping(raw: object, *, context: str) -> dict[str, object]:
    if not isinstance(raw, dict):
        raise ValueError(f"{context} must be a mapping")
    return dict(raw)


def _require_list(raw: object, *, context: str) -> list[object]:
    if not isinstance(raw, list):
        raise ValueError(f"{context} must be a list")
    return list(raw)


def _require_string(raw: object, *, context: str) -> str:
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError(f"{context} must be a non-empty string")
    return raw.strip()


def _load_source_clauses(raw: object, *, context: str) -> tuple[CheatSheetSourceClause, ...]:
    raw_sources = _require_list(raw, context=context)
    if not raw_sources:
        raise ValueError(f"{context} must be a non-empty list")
    return tuple(
        CheatSheetSourceClause(
            label=_require_string(
                _require_mapping(raw_source, context=f"{context}[{index}]").get("label"),
                context=f"{context}[{index}].label",
            ),
            href=_require_string(
                _require_mapping(raw_source, context=f"{context}[{index}]").get("href"),
                context=f"{context}[{index}].href",
            ),
        )
        for index, raw_source in enumerate(raw_sources, start=1)
    )


def _load_string_list(raw: object, *, context: str) -> tuple[str, ...]:
    return tuple(
        _require_string(item, context=f"{context}[{index}]")
        for index, item in enumerate(_require_list(raw, context=context), start=1)
    )


def _load_command_entries(raw: object, *, context: str) -> tuple[CheatSheetCommandEntry, ...]:
    return tuple(
        CheatSheetCommandEntry(
            command=_require_string(
                _require_mapping(raw_entry, context=f"{context}[{index}]").get("command"),
                context=f"{context}[{index}].command",
            ),
            loop_domains=_load_string_list(
                _require_mapping(raw_entry, context=f"{context}[{index}]").get(
                    "loop_domains",
                    [],
                ),
                context=f"{context}[{index}].loop_domains",
            ),
        )
        for index, raw_entry in enumerate(_require_list(raw, context=context), start=1)
    )


def load_catalog(path: Path) -> EnforceableRulesCatalog:
    raw_payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    payload = _require_mapping(raw_payload, context="catalog")
    version = payload.get("version")
    if not isinstance(version, int):
        raise ValueError("catalog.version must be an integer")

    rule_matrix = _require_mapping(payload.get("rule_matrix"), context="catalog.rule_matrix")
    rule_matrix_rows = tuple(
        CheatSheetRuleRow(
            rule_id=_require_string(
                _require_mapping(raw_row, context=f"catalog.rule_matrix.rows[{index}]").get(
                    "rule_id"
                ),
                context=f"catalog.rule_matrix.rows[{index}].rule_id",
            ),
            enforceable_rule=_require_string(
                _require_mapping(raw_row, context=f"catalog.rule_matrix.rows[{index}]").get(
                    "enforceable_rule"
                ),
                context=f"catalog.rule_matrix.rows[{index}].enforceable_rule",
            ),
            source_clauses=_load_source_clauses(
                _require_mapping(raw_row, context=f"catalog.rule_matrix.rows[{index}]").get(
                    "source_clauses"
                ),
                context=f"catalog.rule_matrix.rows[{index}].source_clauses",
            ),
            operational_check=_require_string(
                _require_mapping(raw_row, context=f"catalog.rule_matrix.rows[{index}]").get(
                    "operational_check"
                ),
                context=f"catalog.rule_matrix.rows[{index}].operational_check",
            ),
            failure_signal=_require_string(
                _require_mapping(raw_row, context=f"catalog.rule_matrix.rows[{index}]").get(
                    "failure_signal"
                ),
                context=f"catalog.rule_matrix.rows[{index}].failure_signal",
            ),
        )
        for index, raw_row in enumerate(
            _require_list(rule_matrix.get("rows"), context="catalog.rule_matrix.rows"),
            start=1,
        )
    )

    guardrail_section = _require_mapping(
        payload.get("implementation_guardrails"),
        context="catalog.implementation_guardrails",
    )
    guardrail_rows = tuple(
        CheatSheetGuardrailRow(
            change_type=_require_string(
                _require_mapping(
                    raw_row,
                    context=f"catalog.implementation_guardrails.rows[{index}]",
                ).get("change_type"),
                context=f"catalog.implementation_guardrails.rows[{index}].change_type",
            ),
            loop_domains=_load_string_list(
                _require_mapping(
                    raw_row,
                    context=f"catalog.implementation_guardrails.rows[{index}]",
                ).get("loop_domains", []),
                context=f"catalog.implementation_guardrails.rows[{index}].loop_domains",
            ),
            mandatory_checks=_load_string_list(
                _require_mapping(
                    raw_row,
                    context=f"catalog.implementation_guardrails.rows[{index}]",
                ).get("mandatory_checks"),
                context=f"catalog.implementation_guardrails.rows[{index}].mandatory_checks",
            ),
            prohibited_shortcuts=_load_string_list(
                _require_mapping(
                    raw_row,
                    context=f"catalog.implementation_guardrails.rows[{index}]",
                ).get("prohibited_shortcuts"),
                context=(
                    f"catalog.implementation_guardrails.rows[{index}].prohibited_shortcuts"
                ),
            ),
            required_evidence_artifacts=_load_string_list(
                _require_mapping(
                    raw_row,
                    context=f"catalog.implementation_guardrails.rows[{index}]",
                ).get("required_evidence_artifacts"),
                context=(
                    "catalog.implementation_guardrails.rows"
                    f"[{index}].required_evidence_artifacts"
                ),
            ),
            source_clauses=_load_source_clauses(
                _require_mapping(
                    raw_row,
                    context=f"catalog.implementation_guardrails.rows[{index}]",
                ).get("source_clauses"),
                context=f"catalog.implementation_guardrails.rows[{index}].source_clauses",
            ),
        )
        for index, raw_row in enumerate(
            _require_list(
                guardrail_section.get("rows"),
                context="catalog.implementation_guardrails.rows",
            ),
            start=1,
        )
    )

    quick_validation = _require_mapping(
        payload.get("quick_validation"),
        context="catalog.quick_validation",
    )
    quick_validation_section = CheatSheetQuickValidation(
        required_commands=_load_command_entries(
            quick_validation.get("required_commands"),
            context="catalog.quick_validation.required_commands",
        ),
        optional_commands=_load_command_entries(
            quick_validation.get("optional_commands", []),
            context="catalog.quick_validation.optional_commands",
        ),
    )

    return EnforceableRulesCatalog(
        version=version,
        rule_matrix_rows=rule_matrix_rows,
        implementation_guardrails=guardrail_rows,
        quick_validation=quick_validation_section,
    )


def _escape_cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ").strip()


def _format_source_clauses(source_clauses: tuple[CheatSheetSourceClause, ...]) -> str:
    return ", ".join(f"[{source.label}]({source.href})" for source in source_clauses)


def render_rule_matrix(catalog: EnforceableRulesCatalog) -> str:
    lines = [
        "| Rule ID | Enforceable Rule | Source Clause(s) | Operational Check | Failure Signal |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in catalog.rule_matrix_rows:
        lines.append(
            "| {rule_id} | {rule} | {sources} | {check} | {failure} |".format(
                rule_id=f"`{_escape_cell(row.rule_id)}`",
                rule=_escape_cell(row.enforceable_rule),
                sources=_escape_cell(_format_source_clauses(row.source_clauses)),
                check=_escape_cell(row.operational_check),
                failure=_escape_cell(row.failure_signal),
            )
        )
    return "\n".join(lines)


def render_rule_matrix_block(catalog: EnforceableRulesCatalog) -> str:
    return "\n".join(
        (
            _RULE_MATRIX_BEGIN,
            "_This Rule Matrix is generated from `docs/enforceable_rules_catalog.yaml` "
            "via `mise exec -- python -m scripts.policy.render_enforceable_rules_cheat_sheet`._",
            "",
            render_rule_matrix(catalog),
            _RULE_MATRIX_END,
        )
    )


def _is_command_like(text: str) -> bool:
    prefixes = (
        "mise exec",
        "python ",
        "python -m",
        "git diff",
        "gabion ",
        "scripts/",
    )
    return text.startswith(prefixes)


def _format_inline_items(items: tuple[str, ...]) -> str:
    rendered: list[str] = []
    for item in items:
        if item.startswith("`") and item.endswith("`"):
            rendered.append(item)
            continue
        if _is_command_like(item):
            rendered.append(f"`{item}`")
            continue
        rendered.append(item)
    return "; ".join(rendered)


def render_implementation_guardrails(catalog: EnforceableRulesCatalog) -> str:
    lines = [
        "| Change Type | Mandatory Checks | Prohibited Shortcuts | Required Evidence Artifacts | Source Clause(s) |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in catalog.implementation_guardrails:
        lines.append(
            "| {change_type} | {mandatory} | {prohibited} | {artifacts} | {sources} |".format(
                change_type=_escape_cell(row.change_type),
                mandatory=_escape_cell(_format_inline_items(row.mandatory_checks)),
                prohibited=_escape_cell(_format_inline_items(row.prohibited_shortcuts)),
                artifacts=_escape_cell(_format_inline_items(row.required_evidence_artifacts)),
                sources=_escape_cell(_format_source_clauses(row.source_clauses)),
            )
        )
    return "\n".join(lines)


def render_implementation_guardrails_block(catalog: EnforceableRulesCatalog) -> str:
    return "\n".join(
        (
            _GUARDRAILS_BEGIN,
            "_This guardrail table is generated from `docs/enforceable_rules_catalog.yaml` "
            "with loop-domain validation against `docs/governance_control_loops.yaml` "
            "via `mise exec -- python -m scripts.policy.render_enforceable_rules_cheat_sheet`._",
            "",
            render_implementation_guardrails(catalog),
            _GUARDRAILS_END,
        )
    )


def render_quick_validation_commands_block(catalog: EnforceableRulesCatalog) -> str:
    required_lines = [entry.command for entry in catalog.quick_validation.required_commands]
    optional_lines = [entry.command for entry in catalog.quick_validation.optional_commands]
    parts = [
        _QUICK_VALIDATION_BEGIN,
        "_This validation bundle is generated from `docs/enforceable_rules_catalog.yaml` "
        "with loop-domain validation against `docs/governance_control_loops.yaml` "
        "via `mise exec -- python -m scripts.policy.render_enforceable_rules_cheat_sheet`._",
        "",
        "```bash",
        *required_lines,
        "```",
    ]
    if optional_lines:
        parts.extend(
            (
                "",
                "Optional governance sanity:",
                "",
                "```bash",
                *optional_lines,
                "```",
            )
        )
    parts.append(_QUICK_VALIDATION_END)
    return "\n".join(parts)


def _replace_generated_block(
    document_text: str,
    *,
    begin_marker: str,
    end_marker: str,
    replacement: str,
    label: str,
) -> str:
    start = document_text.find(begin_marker)
    end = document_text.find(end_marker, start if start >= 0 else 0)
    if start < 0 or end < 0 or end < start:
        raise ValueError(f"cheat sheet is missing generated {label} markers")
    end += len(end_marker)
    return document_text[:start] + replacement + document_text[end:]


def _known_loop_domains(
    governance_catalog: governance_loop_docs.GovernanceLoopCatalog,
) -> frozenset[str]:
    return frozenset(loop.domain for loop in governance_catalog.first_order_loops)


def _validate_loop_domains(
    *,
    catalog: EnforceableRulesCatalog,
    governance_catalog: governance_loop_docs.GovernanceLoopCatalog,
) -> None:
    known = _known_loop_domains(governance_catalog)
    for row in catalog.implementation_guardrails:
        for domain in row.loop_domains:
            if domain not in known:
                raise ValueError(
                    f"unknown loop domain in implementation guardrails: {domain}"
                )
    for entry in (
        *catalog.quick_validation.required_commands,
        *catalog.quick_validation.optional_commands,
    ):
        for domain in entry.loop_domains:
            if domain not in known:
                raise ValueError(f"unknown loop domain in quick validation: {domain}")


def run(
    *,
    catalog_path: Path,
    cheat_sheet_path: Path,
    governance_loop_catalog_path: Path,
    check: bool = False,
) -> int:
    catalog = load_catalog(catalog_path)
    governance_catalog = governance_loop_docs.load_governance_loop_catalog(
        governance_loop_catalog_path
    )
    _validate_loop_domains(catalog=catalog, governance_catalog=governance_catalog)
    current = cheat_sheet_path.read_text(encoding="utf-8")
    rendered = current
    rendered = _replace_generated_block(
        rendered,
        begin_marker=_RULE_MATRIX_BEGIN,
        end_marker=_RULE_MATRIX_END,
        replacement=render_rule_matrix_block(catalog),
        label="rule-matrix",
    )
    rendered = _replace_generated_block(
        rendered,
        begin_marker=_GUARDRAILS_BEGIN,
        end_marker=_GUARDRAILS_END,
        replacement=render_implementation_guardrails_block(catalog),
        label="implementation-guardrails",
    )
    rendered = _replace_generated_block(
        rendered,
        begin_marker=_QUICK_VALIDATION_BEGIN,
        end_marker=_QUICK_VALIDATION_END,
        replacement=render_quick_validation_commands_block(catalog),
        label="quick-validation",
    )
    if check:
        return 0 if rendered == current else 1
    cheat_sheet_path.write_text(rendered, encoding="utf-8")
    return 0


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render the generated blocks for the enforceable-rules cheat sheet.",
    )
    parser.add_argument(
        "--catalog",
        default="docs/enforceable_rules_catalog.yaml",
        help="Structured rule catalog that owns the generated cheat-sheet sections.",
    )
    parser.add_argument(
        "--governance-loop-catalog",
        default="docs/governance_control_loops.yaml",
        help="Governance loop catalog used to validate loop-domain references.",
    )
    parser.add_argument(
        "--cheat-sheet",
        default="docs/enforceable_rules_cheat_sheet.md",
        help="Cheat sheet markdown document containing the generated block markers.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail if the cheat sheet is out of sync with the catalog.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    return run(
        catalog_path=Path(args.catalog),
        governance_loop_catalog_path=Path(args.governance_loop_catalog),
        cheat_sheet_path=Path(args.cheat_sheet),
        check=args.check,
    )


__all__ = [
    "CheatSheetCommandEntry",
    "CheatSheetGuardrailRow",
    "CheatSheetQuickValidation",
    "CheatSheetRuleRow",
    "CheatSheetSourceClause",
    "EnforceableRulesCatalog",
    "load_catalog",
    "main",
    "render_implementation_guardrails",
    "render_implementation_guardrails_block",
    "render_quick_validation_commands_block",
    "render_rule_matrix",
    "render_rule_matrix_block",
    "run",
]

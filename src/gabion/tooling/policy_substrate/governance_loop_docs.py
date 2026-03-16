from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse

import yaml

_CONTROL_BEGIN = "<!-- BEGIN:generated_governance_loop_registry -->"
_CONTROL_END = "<!-- END:generated_governance_loop_registry -->"
_MATRIX_BEGIN = "<!-- BEGIN:generated_governance_loop_matrix -->"
_MATRIX_END = "<!-- END:generated_governance_loop_matrix -->"


@dataclass(frozen=True)
class LoopClause:
    label: str
    href: str


@dataclass(frozen=True)
class CorrectionMode:
    mode: str
    description: str


@dataclass(frozen=True)
class BoundedStepRule:
    statement: str
    required_values: tuple[str, ...] = ()


@dataclass(frozen=True)
class LoopEntry:
    domain: str
    clauses: tuple[LoopClause, ...]
    sensor: str
    state_artifact: str
    target_predicate: str
    error_signal: str
    actuator: str
    max_correction_step: str
    verification_command: str
    escalation_threshold: str


@dataclass(frozen=True)
class SecondOrderLoopEntry:
    title: str
    clauses: tuple[LoopClause, ...]
    preamble: tuple[str, ...]
    sensor: str
    state_artifact: str
    target_predicate: str
    error_signal: str
    actuator: str
    max_correction_step: str
    verification_command: str
    escalation_threshold: str


@dataclass(frozen=True)
class MatrixRow:
    gate_id: str
    loop_domain: str
    sensor_command: str
    state_artifact_paths: tuple[str, ...]
    override_note: str


@dataclass(frozen=True)
class GovernanceLoopCatalog:
    version: int
    correction_modes: tuple[CorrectionMode, ...]
    transition_criteria: tuple[str, ...]
    bounded_step_rules: tuple[BoundedStepRule, ...]
    normalized_loop_schema: tuple[str, ...]
    first_order_loops: tuple[LoopEntry, ...]
    second_order_loops: tuple[SecondOrderLoopEntry, ...]
    matrix_rows: tuple[MatrixRow, ...]


@dataclass(frozen=True)
class GovernanceGateRule:
    gate_id: str
    env_flag: str
    enabled_mode: str
    correction_mode: str
    warning_threshold: int
    blocking_threshold: int


@dataclass(frozen=True)
class GovernanceRulesCatalog:
    gates: dict[str, GovernanceGateRule]


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


def _require_int(raw: object, *, context: str) -> int:
    if not isinstance(raw, int):
        raise ValueError(f"{context} must be an integer")
    return raw


def _load_clause(raw: object, *, context: str) -> LoopClause:
    payload = _require_mapping(raw, context=context)
    return LoopClause(
        label=_require_string(payload.get("label"), context=f"{context}.label"),
        href=_require_string(payload.get("href"), context=f"{context}.href"),
    )


def load_governance_loop_catalog(path: Path) -> GovernanceLoopCatalog:
    raw_payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    payload = _require_mapping(raw_payload, context="governance_loop_catalog")
    version = _require_int(payload.get("version"), context="governance_loop_catalog.version")
    correction_modes = tuple(
        CorrectionMode(
            mode=_require_string(
                _require_mapping(item, context=f"correction_modes[{index}]").get("mode"),
                context=f"correction_modes[{index}].mode",
            ),
            description=_require_string(
                _require_mapping(item, context=f"correction_modes[{index}]").get("description"),
                context=f"correction_modes[{index}].description",
            ),
        )
        for index, item in enumerate(
            _require_list(payload.get("correction_modes"), context="correction_modes"),
            start=1,
        )
    )
    transition_criteria = tuple(
        _require_string(item, context=f"transition_criteria[{index}]")
        for index, item in enumerate(
            _require_list(payload.get("transition_criteria"), context="transition_criteria"),
            start=1,
        )
    )
    bounded_step_rules = tuple(
        BoundedStepRule(
            statement=_require_string(
                _require_mapping(item, context=f"bounded_step_correction_rules[{index}]").get("statement"),
                context=f"bounded_step_correction_rules[{index}].statement",
            ),
            required_values=tuple(
                _require_string(value, context=f"bounded_step_correction_rules[{index}].required_values[{value_index}]")
                for value_index, value in enumerate(
                    _require_list(
                        _require_mapping(item, context=f"bounded_step_correction_rules[{index}]").get(
                            "required_values", []
                        ),
                        context=f"bounded_step_correction_rules[{index}].required_values",
                    ),
                    start=1,
                )
            ),
        )
        for index, item in enumerate(
            _require_list(
                payload.get("bounded_step_correction_rules"),
                context="bounded_step_correction_rules",
            ),
            start=1,
        )
    )
    normalized_loop_schema = tuple(
        _require_string(item, context=f"normalized_loop_schema[{index}]")
        for index, item in enumerate(
            _require_list(payload.get("normalized_loop_schema"), context="normalized_loop_schema"),
            start=1,
        )
    )
    first_order_loops = tuple(
        LoopEntry(
            domain=_require_string(
                _require_mapping(item, context=f"first_order_loops[{index}]").get("domain"),
                context=f"first_order_loops[{index}].domain",
            ),
            clauses=tuple(
                _load_clause(raw_clause, context=f"first_order_loops[{index}].clauses[{clause_index}]")
                for clause_index, raw_clause in enumerate(
                    _require_list(
                        _require_mapping(item, context=f"first_order_loops[{index}]").get("clauses", []),
                        context=f"first_order_loops[{index}].clauses",
                    ),
                    start=1,
                )
            ),
            sensor=_require_string(
                _require_mapping(item, context=f"first_order_loops[{index}]").get("sensor"),
                context=f"first_order_loops[{index}].sensor",
            ),
            state_artifact=_require_string(
                _require_mapping(item, context=f"first_order_loops[{index}]").get("state_artifact"),
                context=f"first_order_loops[{index}].state_artifact",
            ),
            target_predicate=_require_string(
                _require_mapping(item, context=f"first_order_loops[{index}]").get("target_predicate"),
                context=f"first_order_loops[{index}].target_predicate",
            ),
            error_signal=_require_string(
                _require_mapping(item, context=f"first_order_loops[{index}]").get("error_signal"),
                context=f"first_order_loops[{index}].error_signal",
            ),
            actuator=_require_string(
                _require_mapping(item, context=f"first_order_loops[{index}]").get("actuator"),
                context=f"first_order_loops[{index}].actuator",
            ),
            max_correction_step=_require_string(
                _require_mapping(item, context=f"first_order_loops[{index}]").get("max_correction_step"),
                context=f"first_order_loops[{index}].max_correction_step",
            ),
            verification_command=_require_string(
                _require_mapping(item, context=f"first_order_loops[{index}]").get("verification_command"),
                context=f"first_order_loops[{index}].verification_command",
            ),
            escalation_threshold=_require_string(
                _require_mapping(item, context=f"first_order_loops[{index}]").get("escalation_threshold"),
                context=f"first_order_loops[{index}].escalation_threshold",
            ),
        )
        for index, item in enumerate(
            _require_list(payload.get("first_order_loops"), context="first_order_loops"),
            start=1,
        )
    )
    second_order_loops = tuple(
        SecondOrderLoopEntry(
            title=_require_string(
                _require_mapping(item, context=f"second_order_loops[{index}]").get("title"),
                context=f"second_order_loops[{index}].title",
            ),
            clauses=tuple(
                _load_clause(raw_clause, context=f"second_order_loops[{index}].clauses[{clause_index}]")
                for clause_index, raw_clause in enumerate(
                    _require_list(
                        _require_mapping(item, context=f"second_order_loops[{index}]").get("clauses", []),
                        context=f"second_order_loops[{index}].clauses",
                    ),
                    start=1,
                )
            ),
            preamble=tuple(
                _require_string(
                    sentence,
                    context=f"second_order_loops[{index}].preamble[{sentence_index}]",
                )
                for sentence_index, sentence in enumerate(
                    _require_list(
                        _require_mapping(item, context=f"second_order_loops[{index}]").get("preamble"),
                        context=f"second_order_loops[{index}].preamble",
                    ),
                    start=1,
                )
            ),
            sensor=_require_string(
                _require_mapping(item, context=f"second_order_loops[{index}]").get("sensor"),
                context=f"second_order_loops[{index}].sensor",
            ),
            state_artifact=_require_string(
                _require_mapping(item, context=f"second_order_loops[{index}]").get("state_artifact"),
                context=f"second_order_loops[{index}].state_artifact",
            ),
            target_predicate=_require_string(
                _require_mapping(item, context=f"second_order_loops[{index}]").get("target_predicate"),
                context=f"second_order_loops[{index}].target_predicate",
            ),
            error_signal=_require_string(
                _require_mapping(item, context=f"second_order_loops[{index}]").get("error_signal"),
                context=f"second_order_loops[{index}].error_signal",
            ),
            actuator=_require_string(
                _require_mapping(item, context=f"second_order_loops[{index}]").get("actuator"),
                context=f"second_order_loops[{index}].actuator",
            ),
            max_correction_step=_require_string(
                _require_mapping(item, context=f"second_order_loops[{index}]").get("max_correction_step"),
                context=f"second_order_loops[{index}].max_correction_step",
            ),
            verification_command=_require_string(
                _require_mapping(item, context=f"second_order_loops[{index}]").get("verification_command"),
                context=f"second_order_loops[{index}].verification_command",
            ),
            escalation_threshold=_require_string(
                _require_mapping(item, context=f"second_order_loops[{index}]").get("escalation_threshold"),
                context=f"second_order_loops[{index}].escalation_threshold",
            ),
        )
        for index, item in enumerate(
            _require_list(payload.get("second_order_loops"), context="second_order_loops"),
            start=1,
        )
    )
    matrix_rows = tuple(
        MatrixRow(
            gate_id=_require_string(
                _require_mapping(item, context=f"matrix_rows[{index}]").get("gate_id"),
                context=f"matrix_rows[{index}].gate_id",
            ),
            loop_domain=_require_string(
                _require_mapping(item, context=f"matrix_rows[{index}]").get("loop_domain"),
                context=f"matrix_rows[{index}].loop_domain",
            ),
            sensor_command=_require_string(
                _require_mapping(item, context=f"matrix_rows[{index}]").get("sensor_command"),
                context=f"matrix_rows[{index}].sensor_command",
            ),
            state_artifact_paths=tuple(
                _require_string(path_value, context=f"matrix_rows[{index}].state_artifact_paths[{path_index}]")
                for path_index, path_value in enumerate(
                    _require_list(
                        _require_mapping(item, context=f"matrix_rows[{index}]").get("state_artifact_paths"),
                        context=f"matrix_rows[{index}].state_artifact_paths",
                    ),
                    start=1,
                )
            ),
            override_note=_require_string(
                _require_mapping(item, context=f"matrix_rows[{index}]").get("override_note"),
                context=f"matrix_rows[{index}].override_note",
            ),
        )
        for index, item in enumerate(
            _require_list(payload.get("matrix_rows"), context="matrix_rows"),
            start=1,
        )
    )
    return GovernanceLoopCatalog(
        version=version,
        correction_modes=correction_modes,
        transition_criteria=transition_criteria,
        bounded_step_rules=bounded_step_rules,
        normalized_loop_schema=normalized_loop_schema,
        first_order_loops=first_order_loops,
        second_order_loops=second_order_loops,
        matrix_rows=matrix_rows,
    )


def load_governance_rules(path: Path) -> GovernanceRulesCatalog:
    raw_payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    payload = _require_mapping(raw_payload, context="governance_rules")
    raw_gates = _require_mapping(payload.get("gates"), context="governance_rules.gates")
    gates: dict[str, GovernanceGateRule] = {}
    for gate_id, raw_gate in raw_gates.items():
        gate_payload = _require_mapping(raw_gate, context=f"governance_rules.gates[{gate_id}]")
        severity = _require_mapping(gate_payload.get("severity"), context=f"governance_rules.gates[{gate_id}].severity")
        correction = _require_mapping(gate_payload.get("correction"), context=f"governance_rules.gates[{gate_id}].correction")
        gates[str(gate_id)] = GovernanceGateRule(
            gate_id=str(gate_id),
            env_flag=_require_string(gate_payload.get("env_flag"), context=f"governance_rules.gates[{gate_id}].env_flag"),
            enabled_mode=_require_string(
                gate_payload.get("enabled_mode"),
                context=f"governance_rules.gates[{gate_id}].enabled_mode",
            ),
            correction_mode=_require_string(
                correction.get("mode"),
                context=f"governance_rules.gates[{gate_id}].correction.mode",
            ),
            warning_threshold=_require_int(
                severity.get("warning_threshold"),
                context=f"governance_rules.gates[{gate_id}].severity.warning_threshold",
            ),
            blocking_threshold=_require_int(
                severity.get("blocking_threshold"),
                context=f"governance_rules.gates[{gate_id}].severity.blocking_threshold",
            ),
        )
    return GovernanceRulesCatalog(gates=gates)


def _escape_cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ").strip()


def _render_clause_links(clauses: tuple[LoopClause, ...]) -> str:
    return ", ".join(f"[{clause.label}]({clause.href})" for clause in clauses)


def render_governance_control_loops_block(catalog: GovernanceLoopCatalog) -> str:
    lines = [
        _CONTROL_BEGIN,
        "_The registry sections below are generated from `docs/governance_control_loops.yaml` "
        "via `mise exec -- python -m scripts.policy.render_governance_loop_docs`._",
        "",
        "## Correction modes",
        "",
    ]
    for mode in catalog.correction_modes:
        lines.append(f"- `{mode.mode}`: {mode.description}")
    lines.extend(("", "## Transition criteria", ""))
    for index, criterion in enumerate(catalog.transition_criteria, start=1):
        lines.append(f"{index}. {criterion}")
    lines.extend(("", "## Bounded-step correction rules", ""))
    for index, rule in enumerate(catalog.bounded_step_rules, start=1):
        lines.append(f"{index}. {rule.statement}")
        for required_value in rule.required_values:
            lines.append(f"- `{required_value}`")
    lines.extend(("", "## Normalized loop schema", "", "Each loop entry must define:", ""))
    for field_name in catalog.normalized_loop_schema:
        lines.append(f"- `{field_name}`")
    lines.extend(("", "## First-order loop registry", ""))
    for index, loop in enumerate(catalog.first_order_loops, start=1):
        lines.append(f"### {index}) {loop.domain}")
        lines.append("")
        if loop.clauses:
            lines.append(f"Clause links: {_render_clause_links(loop.clauses)}.")
            lines.append("")
        lines.extend(
            (
                f"- **sensor:** {loop.sensor}",
                f"- **state artifact:** {loop.state_artifact}",
                f"- **target predicate:** {loop.target_predicate}",
                f"- **error signal:** {loop.error_signal}",
                f"- **actuator:** {loop.actuator}",
                f"- **max correction step:** {loop.max_correction_step}",
                f"- **verification command:** `{loop.verification_command}`.",
                f"- **escalation threshold:** {loop.escalation_threshold}",
                "",
            )
        )
    for loop in catalog.second_order_loops:
        lines.append(f"## {loop.title}")
        lines.append("")
        if loop.clauses:
            lines.append(f"Clause links: {_render_clause_links(loop.clauses)}.")
            lines.append("")
        for sentence in loop.preamble:
            lines.append(sentence)
        lines.append("")
        lines.extend(
            (
                f"- **sensor:** {loop.sensor}",
                f"- **state artifact:** {loop.state_artifact}",
                f"- **target predicate:** {loop.target_predicate}",
                f"- **error signal:** {loop.error_signal}",
                f"- **actuator:** {loop.actuator}",
                f"- **max correction step:** {loop.max_correction_step}",
                f"- **verification command:** `{loop.verification_command}`.",
                f"- **escalation threshold:** {loop.escalation_threshold}",
                "",
            )
        )
    lines.append(_CONTROL_END)
    return "\n".join(lines)


def render_governance_loop_matrix_block(
    catalog: GovernanceLoopCatalog,
    *,
    rules: GovernanceRulesCatalog,
) -> str:
    lines = [
        _MATRIX_BEGIN,
        "_This matrix is generated from `docs/governance_control_loops.yaml` and "
        "`docs/governance_rules.yaml` via `mise exec -- python -m scripts.policy.render_governance_loop_docs`._",
        "",
        "| loop domain | gate ID | sensor command | state artifact path | correction mode | warning/blocking thresholds | override mechanism |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in catalog.matrix_rows:
        gate = rules.gates.get(row.gate_id)
        if gate is None:
            raise ValueError(f"matrix row references unknown gate id {row.gate_id}")
        thresholds = f"warning={gate.warning_threshold}, block={gate.blocking_threshold}"
        override_mechanism = (
            f"Gate toggle: `{gate.env_flag}` (`{gate.enabled_mode}`); {row.override_note}"
        )
        lines.append(
            "| {domain} | {gate_id} | {sensor} | {artifact_paths} | {mode} | {thresholds} | {override} |".format(
                domain=_escape_cell(row.loop_domain),
                gate_id=f"`{_escape_cell(row.gate_id)}`",
                sensor=f"`{_escape_cell(row.sensor_command)}`",
                artifact_paths=_escape_cell(", ".join(row.state_artifact_paths)),
                mode=f"`{_escape_cell(gate.correction_mode)}`",
                thresholds=_escape_cell(thresholds),
                override=_escape_cell(override_mechanism),
            )
        )
    lines.append(_MATRIX_END)
    return "\n".join(lines)


def _replace_block(document_text: str, *, begin: str, end: str, replacement: str) -> str:
    start = document_text.find(begin)
    finish = document_text.find(end)
    if start < 0 or finish < 0 or finish < start:
        raise ValueError(f"document is missing generated block markers {begin} / {end}")
    finish += len(end)
    return document_text[:start] + replacement + document_text[finish:]


def run(
    *,
    catalog_path: Path,
    governance_rules_path: Path,
    control_loops_doc_path: Path,
    loop_matrix_doc_path: Path,
    check: bool = False,
) -> int:
    catalog = load_governance_loop_catalog(catalog_path)
    rules = load_governance_rules(governance_rules_path)
    current_control = control_loops_doc_path.read_text(encoding="utf-8")
    current_matrix = loop_matrix_doc_path.read_text(encoding="utf-8")
    rendered_control = _replace_block(
        current_control,
        begin=_CONTROL_BEGIN,
        end=_CONTROL_END,
        replacement=render_governance_control_loops_block(catalog),
    )
    rendered_matrix = _replace_block(
        current_matrix,
        begin=_MATRIX_BEGIN,
        end=_MATRIX_END,
        replacement=render_governance_loop_matrix_block(catalog, rules=rules),
    )
    if check:
        return 0 if rendered_control == current_control and rendered_matrix == current_matrix else 1
    control_loops_doc_path.write_text(rendered_control, encoding="utf-8")
    loop_matrix_doc_path.write_text(rendered_matrix, encoding="utf-8")
    return 0


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render governance control-loop docs from the shared structured registry.",
    )
    parser.add_argument(
        "--catalog",
        default="docs/governance_control_loops.yaml",
        help="Structured governance loop catalog.",
    )
    parser.add_argument(
        "--governance-rules",
        default="docs/governance_rules.yaml",
        help="Governance rules catalog used to derive gate thresholds and correction modes.",
    )
    parser.add_argument(
        "--control-loops-doc",
        default="docs/governance_control_loops.md",
        help="Governance control-loops markdown document containing generated block markers.",
    )
    parser.add_argument(
        "--loop-matrix-doc",
        default="docs/governance_loop_matrix.md",
        help="Governance loop-matrix markdown document containing generated block markers.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail if the rendered documents are out of sync with the structured registry.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    return run(
        catalog_path=Path(args.catalog),
        governance_rules_path=Path(args.governance_rules),
        control_loops_doc_path=Path(args.control_loops_doc),
        loop_matrix_doc_path=Path(args.loop_matrix_doc),
        check=args.check,
    )


__all__ = [
    "GovernanceLoopCatalog",
    "GovernanceRulesCatalog",
    "LoopClause",
    "LoopEntry",
    "MatrixRow",
    "SecondOrderLoopEntry",
    "load_governance_loop_catalog",
    "load_governance_rules",
    "main",
    "render_governance_control_loops_block",
    "render_governance_loop_matrix_block",
    "run",
]

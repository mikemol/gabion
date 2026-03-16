from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse

import yaml

_RULE_MATRIX_BEGIN = "<!-- BEGIN:generated_rule_matrix -->"
_RULE_MATRIX_END = "<!-- END:generated_rule_matrix -->"


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
class EnforceableRulesCatalog:
    version: int
    rule_matrix_rows: tuple[CheatSheetRuleRow, ...]


def _require_mapping(raw: object, *, context: str) -> dict[str, object]:
    if not isinstance(raw, dict):
        raise ValueError(f"{context} must be a mapping")
    return dict(raw)


def _require_string(raw: object, *, context: str) -> str:
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError(f"{context} must be a non-empty string")
    return raw.strip()


def load_catalog(path: Path) -> EnforceableRulesCatalog:
    raw_payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    payload = _require_mapping(raw_payload, context="catalog")
    version = payload.get("version")
    if not isinstance(version, int):
        raise ValueError("catalog.version must be an integer")
    rule_matrix = _require_mapping(
        payload.get("rule_matrix"),
        context="catalog.rule_matrix",
    )
    raw_rows = rule_matrix.get("rows")
    if not isinstance(raw_rows, list):
        raise ValueError("catalog.rule_matrix.rows must be a list")
    rows: list[CheatSheetRuleRow] = []
    for index, raw_row in enumerate(raw_rows, start=1):
        row = _require_mapping(raw_row, context=f"catalog.rule_matrix.rows[{index}]")
        raw_sources = row.get("source_clauses")
        if not isinstance(raw_sources, list) or not raw_sources:
            raise ValueError(
                f"catalog.rule_matrix.rows[{index}].source_clauses must be a non-empty list"
            )
        sources = tuple(
            CheatSheetSourceClause(
                label=_require_string(
                    _require_mapping(
                        raw_source,
                        context=(
                            "catalog.rule_matrix.rows"
                            f"[{index}].source_clauses[{source_index}]"
                        ),
                    ).get("label"),
                    context=(
                        "catalog.rule_matrix.rows"
                        f"[{index}].source_clauses[{source_index}].label"
                    ),
                ),
                href=_require_string(
                    _require_mapping(
                        raw_source,
                        context=(
                            "catalog.rule_matrix.rows"
                            f"[{index}].source_clauses[{source_index}]"
                        ),
                    ).get("href"),
                    context=(
                        "catalog.rule_matrix.rows"
                        f"[{index}].source_clauses[{source_index}].href"
                    ),
                ),
            )
            for source_index, raw_source in enumerate(raw_sources, start=1)
        )
        rows.append(
            CheatSheetRuleRow(
                rule_id=_require_string(
                    row.get("rule_id"),
                    context=f"catalog.rule_matrix.rows[{index}].rule_id",
                ),
                enforceable_rule=_require_string(
                    row.get("enforceable_rule"),
                    context=f"catalog.rule_matrix.rows[{index}].enforceable_rule",
                ),
                source_clauses=sources,
                operational_check=_require_string(
                    row.get("operational_check"),
                    context=f"catalog.rule_matrix.rows[{index}].operational_check",
                ),
                failure_signal=_require_string(
                    row.get("failure_signal"),
                    context=f"catalog.rule_matrix.rows[{index}].failure_signal",
                ),
            )
        )
    return EnforceableRulesCatalog(version=version, rule_matrix_rows=tuple(rows))


def _escape_cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ").strip()


def render_rule_matrix(catalog: EnforceableRulesCatalog) -> str:
    lines = [
        "| Rule ID | Enforceable Rule | Source Clause(s) | Operational Check | Failure Signal |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in catalog.rule_matrix_rows:
        source_clauses = ", ".join(
            f"[{source.label}]({source.href})" for source in row.source_clauses
        )
        lines.append(
            "| {rule_id} | {rule} | {sources} | {check} | {failure} |".format(
                rule_id=f"`{_escape_cell(row.rule_id)}`",
                rule=_escape_cell(row.enforceable_rule),
                sources=_escape_cell(source_clauses),
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


def replace_generated_rule_matrix(
    document_text: str,
    *,
    catalog: EnforceableRulesCatalog,
) -> str:
    start = document_text.find(_RULE_MATRIX_BEGIN)
    end = document_text.find(_RULE_MATRIX_END)
    if start < 0 or end < 0 or end < start:
        raise ValueError("cheat sheet is missing generated rule-matrix markers")
    end += len(_RULE_MATRIX_END)
    return (
        document_text[:start]
        + render_rule_matrix_block(catalog)
        + document_text[end:]
    )


def run(*, catalog_path: Path, cheat_sheet_path: Path, check: bool = False) -> int:
    catalog = load_catalog(catalog_path)
    current = cheat_sheet_path.read_text(encoding="utf-8")
    rendered = replace_generated_rule_matrix(current, catalog=catalog)
    if check:
        return 0 if rendered == current else 1
    cheat_sheet_path.write_text(rendered, encoding="utf-8")
    return 0


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render the generated Rule Matrix block for the enforceable-rules cheat sheet.",
    )
    parser.add_argument(
        "--catalog",
        default="docs/enforceable_rules_catalog.yaml",
        help="Structured rule catalog that owns the generated Rule Matrix.",
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
        cheat_sheet_path=Path(args.cheat_sheet),
        check=args.check,
    )


__all__ = [
    "CheatSheetRuleRow",
    "CheatSheetSourceClause",
    "EnforceableRulesCatalog",
    "load_catalog",
    "main",
    "render_rule_matrix",
    "render_rule_matrix_block",
    "replace_generated_rule_matrix",
    "run",
]

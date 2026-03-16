from __future__ import annotations

from pathlib import Path

from gabion.tooling.policy_substrate import enforceable_rules_cheat_sheet


def test_render_rule_matrix_block_uses_catalog_runtime_rows() -> None:
    repo_root = Path(__file__).resolve().parents[4]
    catalog = enforceable_rules_cheat_sheet.load_catalog(
        repo_root / "docs" / "enforceable_rules_catalog.yaml"
    )

    rendered = enforceable_rules_cheat_sheet.render_rule_matrix_block(catalog)

    assert "<!-- BEGIN:generated_rule_matrix -->" in rendered
    assert "<!-- END:generated_rule_matrix -->" in rendered
    assert "`SEM-001`" in rendered
    assert "NCI-RUNTIME-DISTINCTION-LADDER" in rendered
    assert "docs/enforceable_rules_catalog.yaml" in rendered


def test_run_rewrites_generated_rule_matrix_and_check_detects_drift(
    tmp_path: Path,
) -> None:
    catalog_path = tmp_path / "catalog.yaml"
    cheat_sheet_path = tmp_path / "cheat_sheet.md"
    catalog_path.write_text(
        "\n".join(
            (
                "version: 1",
                "rule_matrix:",
                "  rows:",
                "    - rule_id: TEST-001",
                "      enforceable_rule: Structured row owned by catalog.",
                "      source_clauses:",
                "        - label: Clause-A",
                "          href: ./clause-a",
                "      operational_check: mise exec -- python -m pytest tests/test_a.py",
                "      failure_signal: Drift is detected.",
                "",
            )
        ),
        encoding="utf-8",
    )
    cheat_sheet_path.write_text(
        "\n".join(
            (
                "# Cheat Sheet",
                "",
                "<!-- BEGIN:generated_rule_matrix -->",
                "stale",
                "<!-- END:generated_rule_matrix -->",
                "",
            )
        ),
        encoding="utf-8",
    )

    assert (
        enforceable_rules_cheat_sheet.run(
            catalog_path=catalog_path,
            cheat_sheet_path=cheat_sheet_path,
        )
        == 0
    )
    assert (
        enforceable_rules_cheat_sheet.run(
            catalog_path=catalog_path,
            cheat_sheet_path=cheat_sheet_path,
            check=True,
        )
        == 0
    )

    cheat_sheet_path.write_text(
        cheat_sheet_path.read_text(encoding="utf-8").replace("TEST-001", "stale-row"),
        encoding="utf-8",
    )

    assert (
        enforceable_rules_cheat_sheet.run(
            catalog_path=catalog_path,
            cheat_sheet_path=cheat_sheet_path,
            check=True,
        )
        == 1
    )

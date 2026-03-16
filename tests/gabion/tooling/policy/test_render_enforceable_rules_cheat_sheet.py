from __future__ import annotations

from pathlib import Path

from gabion.tooling.policy_substrate import enforceable_rules_cheat_sheet


def _minimal_governance_loop_catalog_yaml() -> str:
    return "\n".join(
        (
            "version: 1",
            "correction_modes:",
            "  - mode: advisory",
            "    description: emit diagnostics only.",
            "transition_criteria:",
            "  - advisory -> ratchet when warnings recur.",
            "bounded_step_correction_rules:",
            "  - statement: Baseline writes require explicit flags.",
            "normalized_loop_schema:",
            "  - sensor",
            "first_order_loops:",
            "  - domain: security/workflows",
            "    sensor: workflows",
            "    state_artifact: workflows.json",
            "    target_predicate: workflows clean",
            "    error_signal: workflow failures",
            "    actuator: patch workflows",
            "    max_correction_step: one patch",
            "    verification_command: mise exec -- python -m scripts.policy_check --workflows",
            "    escalation_threshold: one recurrence",
            "  - domain: docs/docflow",
            "    sensor: docflow",
            "    state_artifact: docflow.json",
            "    target_predicate: docflow clean",
            "    error_signal: docflow failures",
            "    actuator: patch docs",
            "    max_correction_step: one patch",
            "    verification_command: mise exec -- python -m gabion docflow",
            "    escalation_threshold: one recurrence",
            "  - domain: dataflow grammar",
            "    sensor: check",
            "    state_artifact: check.json",
            "    target_predicate: check clean",
            "    error_signal: check failures",
            "    actuator: patch semantics",
            "    max_correction_step: one patch",
            "    verification_command: mise exec -- python -m gabion check",
            "    escalation_threshold: one recurrence",
            "  - domain: execution coverage",
            "    sensor: coverage",
            "    state_artifact: coverage.xml",
            "    target_predicate: coverage stays 100",
            "    error_signal: coverage failures",
            "    actuator: patch tests",
            "    max_correction_step: one patch",
            "    verification_command: mise exec -- python -m pytest --cov=src/gabion",
            "    escalation_threshold: one recurrence",
            "  - domain: LSP architecture",
            "    sensor: architecture",
            "    state_artifact: architecture.json",
            "    target_predicate: thin client",
            "    error_signal: architecture drift",
            "    actuator: patch cli/server",
            "    max_correction_step: one patch",
            "    verification_command: mise exec -- python scripts/checks.sh --no-docflow",
            "    escalation_threshold: one recurrence",
            "  - domain: baseline ratchets",
            "    sensor: baselines",
            "    state_artifact: baseline.json",
            "    target_predicate: ratchet clean",
            "    error_signal: ratchet failures",
            "    actuator: refresh baseline",
            "    max_correction_step: one patch",
            "    verification_command: mise exec -- python -m scripts.refresh_baselines --docflow",
            "    escalation_threshold: one recurrence",
            "second_order_loops:",
            "  - title: Controller loop",
            "    clauses: []",
            "    preamble:",
            "      - second-order preamble",
            "    sensor: drift",
            "    state_artifact: drift.json",
            "    target_predicate: drift zero",
            "    error_signal: drift failures",
            "    actuator: patch governance",
            "    max_correction_step: one patch",
            "    verification_command: mise exec -- python scripts/governance_controller_audit.py",
            "    escalation_threshold: one recurrence",
            "matrix_rows:",
            "  - gate_id: docflow",
            "    loop_domain: docs/docflow",
            "    sensor_command: mise exec -- python -m gabion.tooling.docflow_delta_gate",
            "    state_artifact_paths:",
            "      - artifacts/out/docflow_compliance_delta.json",
            "    override_note: strictness reductions require override token.",
            "",
        )
    )


def test_render_cheat_sheet_generated_blocks_use_catalogs() -> None:
    repo_root = Path(__file__).resolve().parents[4]
    catalog = enforceable_rules_cheat_sheet.load_catalog(
        repo_root / "docs" / "enforceable_rules_catalog.yaml"
    )

    rendered_matrix = enforceable_rules_cheat_sheet.render_rule_matrix_block(catalog)
    rendered_guardrails = enforceable_rules_cheat_sheet.render_implementation_guardrails_block(
        catalog
    )
    rendered_validation = enforceable_rules_cheat_sheet.render_quick_validation_commands_block(
        catalog
    )

    assert "<!-- BEGIN:generated_rule_matrix -->" in rendered_matrix
    assert "docs/enforceable_rules_catalog.yaml" in rendered_matrix
    assert "`SEM-001`" in rendered_matrix
    assert "<!-- BEGIN:generated_implementation_guardrails -->" in rendered_guardrails
    assert "docs/governance_control_loops.yaml" in rendered_guardrails
    assert "Workflow/CI YAML" in rendered_guardrails
    assert "<!-- BEGIN:generated_quick_validation_commands -->" in rendered_validation
    assert "Optional governance sanity:" in rendered_validation
    assert "scripts/no_monkeypatch_policy_check.py" in rendered_validation


def test_run_rewrites_generated_cheat_sheet_blocks_and_check_detects_drift(
    tmp_path: Path,
) -> None:
    catalog_path = tmp_path / "catalog.yaml"
    governance_catalog_path = tmp_path / "governance_control_loops.yaml"
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
                "implementation_guardrails:",
                "  rows:",
                "    - change_type: Test surface",
                "      loop_domains:",
                "        - docs/docflow",
                "      mandatory_checks:",
                "        - mise exec -- python -m gabion docflow",
                "      prohibited_shortcuts:",
                "        - drift",
                "      required_evidence_artifacts:",
                "        - docflow report",
                "      source_clauses:",
                "        - label: Clause-B",
                "          href: ./clause-b",
                "quick_validation:",
                "  required_commands:",
                "    - command: mise exec -- python -m gabion docflow",
                "      loop_domains:",
                "        - docs/docflow",
                "  optional_commands:",
                "    - command: mise exec -- python -m gabion status-consistency --fail-on-violations",
                "      loop_domains: []",
                "",
            )
        ),
        encoding="utf-8",
    )
    governance_catalog_path.write_text(
        _minimal_governance_loop_catalog_yaml(),
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
                "<!-- BEGIN:generated_implementation_guardrails -->",
                "stale",
                "<!-- END:generated_implementation_guardrails -->",
                "",
                "<!-- BEGIN:generated_quick_validation_commands -->",
                "stale",
                "<!-- END:generated_quick_validation_commands -->",
                "",
            )
        ),
        encoding="utf-8",
    )

    assert (
        enforceable_rules_cheat_sheet.run(
            catalog_path=catalog_path,
            governance_loop_catalog_path=governance_catalog_path,
            cheat_sheet_path=cheat_sheet_path,
        )
        == 0
    )
    assert (
        enforceable_rules_cheat_sheet.run(
            catalog_path=catalog_path,
            governance_loop_catalog_path=governance_catalog_path,
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
            governance_loop_catalog_path=governance_catalog_path,
            cheat_sheet_path=cheat_sheet_path,
            check=True,
        )
        == 1
    )

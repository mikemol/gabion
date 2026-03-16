from __future__ import annotations

from pathlib import Path

from gabion.tooling.policy_substrate import governance_loop_docs


def test_render_governance_loop_blocks_use_shared_catalog() -> None:
    repo_root = Path(__file__).resolve().parents[4]
    catalog = governance_loop_docs.load_governance_loop_catalog(
        repo_root / "docs" / "governance_control_loops.yaml"
    )
    rules = governance_loop_docs.load_governance_rules(
        repo_root / "docs" / "governance_rules.yaml"
    )

    rendered_control = governance_loop_docs.render_governance_control_loops_block(catalog)
    rendered_matrix = governance_loop_docs.render_governance_loop_matrix_block(
        catalog,
        rules=rules,
    )

    assert "<!-- BEGIN:generated_governance_loop_registry -->" in rendered_control
    assert "docs/governance_control_loops.yaml" in rendered_control
    assert "## First-order loop registry" in rendered_control
    assert "### 7) GitHub status API process" in rendered_control
    assert "<!-- BEGIN:generated_governance_loop_matrix -->" in rendered_matrix
    assert "docs/governance_rules.yaml" in rendered_matrix
    assert "`docflow_packet_loop`" in rendered_matrix
    assert "`hard-fail`" in rendered_matrix


def test_run_rewrites_generated_governance_blocks_and_check_detects_drift(
    tmp_path: Path,
) -> None:
    catalog_path = tmp_path / "governance_control_loops.yaml"
    rules_path = tmp_path / "governance_rules.yaml"
    control_doc_path = tmp_path / "governance_control_loops.md"
    matrix_doc_path = tmp_path / "governance_loop_matrix.md"
    catalog_path.write_text(
        "\n".join(
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
                "  - domain: docs/docflow",
                "    sensor: docflow sensor",
                "    state_artifact: artifacts/out/docflow.json",
                "    target_predicate: docflow stays clean",
                "    error_signal: docflow violations",
                "    actuator: patch docs",
                "    max_correction_step: one doc cycle",
                "    verification_command: mise exec -- python -m gabion docflow",
                "    escalation_threshold: repeat failure after one cycle",
                "second_order_loops:",
                "  - title: Second-order loop",
                "    clauses: []",
                "    preamble:",
                "      - second-order preamble",
                "    sensor: drift sensor",
                "    state_artifact: artifacts/out/drift.json",
                "    target_predicate: drift stays zero",
                "    error_signal: drift findings",
                "    actuator: patch governance",
                "    max_correction_step: one governance patch",
                "    verification_command: mise exec -- python audit.py",
                "    escalation_threshold: repeat drift",
                "matrix_rows:",
                "  - gate_id: docflow",
                "    loop_domain: docs/docflow",
                "    sensor_command: mise exec -- python -m gabion.tooling.docflow_delta_gate",
                "    state_artifact_paths:",
                "      - artifacts/out/docflow_delta.json",
                "    override_note: strictness reductions require token + rationale.",
                "",
            )
        ),
        encoding="utf-8",
    )
    rules_path.write_text(
        "\n".join(
            (
                "version: 1",
                "gates:",
                "  docflow:",
                "    env_flag: GABION_GATE_DOCFLOW_DELTA",
                "    enabled_mode: truthy_only",
                "    severity:",
                "      warning_threshold: 0",
                "      blocking_threshold: 1",
                "    correction:",
                "      mode: advisory",
                "",
            )
        ),
        encoding="utf-8",
    )
    control_doc_path.write_text(
        "\n".join(
            (
                "# Governance control loops",
                "",
                "<!-- BEGIN:generated_governance_loop_registry -->",
                "stale",
                "<!-- END:generated_governance_loop_registry -->",
                "",
            )
        ),
        encoding="utf-8",
    )
    matrix_doc_path.write_text(
        "\n".join(
            (
                "# Governance loop matrix",
                "",
                "<!-- BEGIN:generated_governance_loop_matrix -->",
                "stale",
                "<!-- END:generated_governance_loop_matrix -->",
                "",
            )
        ),
        encoding="utf-8",
    )

    assert (
        governance_loop_docs.run(
            catalog_path=catalog_path,
            governance_rules_path=rules_path,
            control_loops_doc_path=control_doc_path,
            loop_matrix_doc_path=matrix_doc_path,
        )
        == 0
    )
    assert (
        governance_loop_docs.run(
            catalog_path=catalog_path,
            governance_rules_path=rules_path,
            control_loops_doc_path=control_doc_path,
            loop_matrix_doc_path=matrix_doc_path,
            check=True,
        )
        == 0
    )

    control_doc_path.write_text(
        control_doc_path.read_text(encoding="utf-8").replace("docs/docflow", "stale-domain"),
        encoding="utf-8",
    )

    assert (
        governance_loop_docs.run(
            catalog_path=catalog_path,
            governance_rules_path=rules_path,
            control_loops_doc_path=control_doc_path,
            loop_matrix_doc_path=matrix_doc_path,
            check=True,
        )
        == 1
    )

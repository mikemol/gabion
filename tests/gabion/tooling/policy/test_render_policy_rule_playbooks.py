from __future__ import annotations

from pathlib import Path

from gabion.tooling.policy_substrate import policy_rule_playbook_docs


def test_load_playbook_sections_uses_frontmatter_guidance_and_references() -> None:
    repo_root = Path(__file__).resolve().parents[4]
    ambiguity_sections = policy_rule_playbook_docs.load_playbook_sections(
        repo_root / "docs" / "policy_rules" / "ambiguity_contract.md"
    )
    grade_sections = policy_rule_playbook_docs.load_playbook_sections(
        repo_root / "docs" / "policy_rules" / "grade_monotonicity.md"
    )

    assert tuple(section.rule_id for section in ambiguity_sections) == (
        "ambiguity.new_violations",
        "ACP-003",
        "ACP-004",
        "ACP-002",
        "ACP-007",
        "ACP-005",
        "ACP-006",
    )
    assert ambiguity_sections[-1].references == (
        policy_rule_playbook_docs.PlaybookReference(
            label="Shift-Ambiguity-Left Protocol",
            href="../shift_ambiguity_left_protocol.md#shift_ambiguity_left_protocol",
        ),
    )
    assert tuple(section.rule_id for section in grade_sections) == (
        "grade_monotonicity.new_violations",
    )
    assert grade_sections[0].references == (
        policy_rule_playbook_docs.PlaybookReference(
            label="Shift-Ambiguity-Left Protocol",
            href="../shift_ambiguity_left_protocol.md#shift_ambiguity_left_protocol",
        ),
    )


def test_render_playbook_blocks_rewrite_generated_regions_without_touching_manual_tail(
    tmp_path: Path,
) -> None:
    ambiguity_doc = tmp_path / "ambiguity_contract.md"
    grade_doc = tmp_path / "grade_monotonicity.md"
    ambiguity_doc.write_text(
        "\n".join(
            (
                "---",
                "rules:",
                "  - rule_id: ambiguity.new_violations",
                "    outcome:",
                "      guidance:",
                "        why: summary meaning",
                "        prefer:",
                "          - move the seam",
                "        avoid:",
                "          - do not drift",
                "    playbook_anchor: ambiguity-new-violations",
                "playbook_rendering:",
                "  references:",
                "    ambiguity.new_violations:",
                "      - label: Protocol",
                "        href: ../protocol.md#protocol",
                "---",
                "",
                "# Ambiguity",
                "",
                "<!-- BEGIN:generated_policy_rule_playbooks -->",
                "stale",
                "<!-- END:generated_policy_rule_playbooks -->",
                "",
            )
        ),
        encoding="utf-8",
    )
    grade_doc.write_text(
        "\n".join(
            (
                "---",
                "rules:",
                "  - rule_id: grade_monotonicity.new_violations",
                "    outcome:",
                "      guidance:",
                "        why: grade meaning",
                "        prefer: keep downstream edges strict",
                "        avoid:",
                "          - do not widen",
                "    playbook_anchor: grade-monotonicity-new-violations",
                "---",
                "",
                "# Grade",
                "",
                "<!-- BEGIN:generated_policy_rule_playbooks -->",
                "stale",
                "<!-- END:generated_policy_rule_playbooks -->",
                "",
                "<a id=\"gmp-001\"></a>",
                "## `GMP-001`",
                "",
                "Meaning: manual violation playbook",
                "",
            )
        ),
        encoding="utf-8",
    )

    assert (
        policy_rule_playbook_docs.run(
            ambiguity_contract_doc_path=ambiguity_doc,
            grade_monotonicity_doc_path=grade_doc,
        )
        == 0
    )

    ambiguity_text = ambiguity_doc.read_text(encoding="utf-8")
    grade_text = grade_doc.read_text(encoding="utf-8")
    assert "summary meaning" in ambiguity_text
    assert "Reference: [Protocol](../protocol.md#protocol)." in ambiguity_text
    assert "grade meaning" in grade_text
    assert "## `GMP-001`" in grade_text

    assert (
        policy_rule_playbook_docs.run(
            ambiguity_contract_doc_path=ambiguity_doc,
            grade_monotonicity_doc_path=grade_doc,
            check=True,
        )
        == 0
    )

    ambiguity_doc.write_text(
        ambiguity_text.replace(
            "Reference: [Protocol](../protocol.md#protocol).",
            "Reference: [Drifted](../protocol.md#protocol).",
        ),
        encoding="utf-8",
    )

    assert (
        policy_rule_playbook_docs.run(
            ambiguity_contract_doc_path=ambiguity_doc,
            grade_monotonicity_doc_path=grade_doc,
            check=True,
        )
        == 1
    )

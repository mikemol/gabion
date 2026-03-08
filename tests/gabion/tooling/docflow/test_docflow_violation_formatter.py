from __future__ import annotations

from gabion.tooling.governance import governance_audit as audit_impl


# gabion:behavior primary=desired
def test_format_docflow_violation_known_kinds() -> None:
    assert (
        audit_impl._format_docflow_violation(
            {
                "row_kind": "doc_required_field",
                "path": "README.md",
                "field": "doc_id",
            }
        )
        == "README.md: missing frontmatter field 'doc_id'"
    )
    assert (
        audit_impl._format_docflow_violation(
            {
                "row_kind": "doc_loop_matrix_gate",
                "path": "docs/governance_loop_matrix.md",
                "gate_id": "coverage_gate",
            }
        )
        == "docs/governance_loop_matrix.md: governance loop matrix drift; missing gate row for: coverage_gate"
    )

    assert (
        audit_impl._format_docflow_violation(
            {
                "row_kind": "doc_implication_matrix_conflict",
                "path": "docs/counterexample.md",
                "subject": "keep matrix parity",
            }
        )
        == "docs/counterexample.md: implication matrix conflict for notion subject: keep matrix parity"
    )


# gabion:behavior primary=desired
def test_format_docflow_violation_doc_review_pin_branches() -> None:
    assert (
        audit_impl._format_docflow_violation(
            {
                "row_kind": "doc_review_pin",
                "path": "AGENTS.md",
                "req": "README.md#repo_contract",
                "resolved": False,
            }
        )
        == "AGENTS.md: doc_reviewed_as_of cannot resolve README.md#repo_contract"
    )
    assert (
        audit_impl._format_docflow_violation(
            {
                "row_kind": "doc_review_pin",
                "path": "AGENTS.md",
                "req": "README.md#repo_contract",
                "resolved": True,
                "seen": "2",
            }
        )
        == "AGENTS.md: doc_reviewed_as_of[README.md#repo_contract] must be an integer"
    )


# gabion:behavior primary=desired
def test_format_docflow_violation_unknown_kind_falls_back_to_generic_message() -> None:
    assert (
        audit_impl._format_docflow_violation({"row_kind": "unknown_row_kind", "path": "docs/foo.md"})
        == "docs/foo.md: docflow invariant violation"
    )

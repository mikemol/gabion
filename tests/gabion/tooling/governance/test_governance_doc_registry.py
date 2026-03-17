from __future__ import annotations

from gabion_governance import governance_audit_impl
from gabion_governance.governance_doc_registry import (
    load_governance_docflow_registry,
)


def test_governance_docflow_registry_exposes_inventory_and_schema_catalog() -> None:
    registry = load_governance_docflow_registry()

    assert registry.core_governance_docs == (
        "POLICY_SEED.md",
        "glossary.md",
        "README.md",
        "CONTRIBUTING.md",
        "AGENTS.md",
    )
    assert registry.governance_docs[:5] == registry.core_governance_docs
    assert "docs/normative_clause_index.md" in registry.governance_docs
    assert "docs/planning_chart_architecture.md" in registry.governance_docs
    assert "docs/planning_substrate.md" in registry.governance_docs
    assert registry.review_note_revision_lint_docs == frozenset(
        {
            "AGENTS.md",
            "README.md",
            "CONTRIBUTING.md",
            "POLICY_SEED.md",
            "glossary.md",
            "docs/normative_clause_index.md",
        }
    )
    assert registry.required_frontmatter_fields == (
        "doc_id",
        "doc_role",
        "doc_scope",
        "doc_authority",
        "doc_requires",
        "doc_reviewed_as_of",
        "doc_review_notes",
        "doc_change_protocol",
    )
    assert registry.list_frontmatter_fields == (
        "doc_scope",
        "doc_requires",
        "doc_commutes_with",
        "doc_invariants",
        "doc_erasure",
    )
    assert registry.map_frontmatter_fields == (
        "doc_reviewed_as_of",
        "doc_review_notes",
        "doc_sections",
        "doc_section_requires",
        "doc_section_reviews",
    )


def test_governance_audit_impl_uses_shared_governance_docflow_registry() -> None:
    assert (
        governance_audit_impl._GOVERNANCE_DOCFLOW_REGISTRY
        == load_governance_docflow_registry()
    )

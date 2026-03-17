from __future__ import annotations

from dataclasses import dataclass
from functools import cache


@dataclass(frozen=True)
class GovernanceDocflowRegistry:
    core_governance_docs: tuple[str, ...]
    governance_docs: tuple[str, ...]
    review_note_revision_lint_docs: frozenset[str]
    governance_control_loops_doc: str
    normative_loop_domains: tuple[str, ...]
    required_frontmatter_fields: tuple[str, ...]
    list_frontmatter_fields: tuple[str, ...]
    map_frontmatter_fields: tuple[str, ...]


@cache
def load_governance_docflow_registry() -> GovernanceDocflowRegistry:
    core_governance_docs = (
        "POLICY_SEED.md",
        "glossary.md",
        "README.md",
        "CONTRIBUTING.md",
        "AGENTS.md",
    )
    governance_docs = core_governance_docs + (
        "docs/governance_control_loops.md",
        "docs/governance_loop_matrix.md",
        "docs/publishing_practices.md",
        "docs/influence_index.md",
        "docs/coverage_semantics.md",
        "docs/normative_clause_index.md",
        "docs/matrix_acceptance.md",
        "docs/planning_chart_architecture.md",
        "docs/planning_substrate.md",
        "docs/sppf_checklist.md",
    )
    return GovernanceDocflowRegistry(
        core_governance_docs=core_governance_docs,
        governance_docs=governance_docs,
        review_note_revision_lint_docs=frozenset(
            {
                "AGENTS.md",
                "README.md",
                "CONTRIBUTING.md",
                "POLICY_SEED.md",
                "glossary.md",
                "docs/normative_clause_index.md",
            }
        ),
        governance_control_loops_doc="docs/governance_control_loops.md",
        normative_loop_domains=(
            "security/workflows",
            "docs/docflow",
            "LSP architecture",
            "dataflow grammar",
            "baseline ratchets",
        ),
        required_frontmatter_fields=(
            "doc_id",
            "doc_role",
            "doc_scope",
            "doc_authority",
            "doc_requires",
            "doc_reviewed_as_of",
            "doc_review_notes",
            "doc_change_protocol",
        ),
        list_frontmatter_fields=(
            "doc_scope",
            "doc_requires",
            "doc_commutes_with",
            "doc_invariants",
            "doc_erasure",
        ),
        map_frontmatter_fields=(
            "doc_reviewed_as_of",
            "doc_review_notes",
            "doc_sections",
            "doc_section_requires",
            "doc_section_reviews",
        ),
    )

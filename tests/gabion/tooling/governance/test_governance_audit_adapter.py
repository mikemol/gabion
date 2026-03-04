from __future__ import annotations

from gabion.tooling.governance import governance_audit
from gabion_governance import governance_audit_impl


# gabion:evidence E:call_footprint::tests/test_governance_audit_adapter.py::test_adapter_reexports_impl_symbols_and_lifecycle_metadata::governance_audit.py::gabion.tooling.governance.governance_audit.BOUNDARY_ADAPTER_METADATA::governance_audit.py::gabion.tooling.governance.governance_audit._parse_frontmatter::governance_audit.py::gabion.tooling.governance.governance_audit.run_decision_tiers_cli
def test_adapter_reexports_impl_symbols_and_lifecycle_metadata() -> None:
    metadata = governance_audit.BOUNDARY_ADAPTER_METADATA
    assert set(metadata) == {
        "actor",
        "rationale",
        "scope",
        "start",
        "expiry",
        "rollback_condition",
        "evidence_links",
    }
    assert metadata["actor"] == "codex"

    assert governance_audit._parse_frontmatter is governance_audit_impl._parse_frontmatter
    assert governance_audit.run_decision_tiers_cli is governance_audit_impl.run_decision_tiers_cli
    assert governance_audit.run_lint_summary_cli is governance_audit_impl.run_lint_summary_cli

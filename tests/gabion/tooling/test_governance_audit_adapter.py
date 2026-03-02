from __future__ import annotations

from gabion.tooling import governance_audit


# gabion:evidence E:call_footprint::tests/test_governance_audit_adapter.py::test_adapter_wrappers_cover_parse_and_runner_paths::governance_audit.py::gabion.tooling.governance_audit._parse_frontmatter::governance_audit.py::gabion.tooling.governance_audit.run_decision_tiers_cli::governance_audit.py::gabion.tooling.governance_audit.run_lint_summary_cli
def test_adapter_wrappers_cover_parse_and_runner_paths() -> None:
    original_parse = governance_audit._impl._parse_frontmatter
    original_tiers = governance_audit._impl.run_decision_tiers_cli
    original_lint = governance_audit._impl.run_lint_summary_cli
    try:
        governance_audit._impl._parse_frontmatter = lambda text: ({"x": "y"}, text)
        governance_audit._impl.run_decision_tiers_cli = lambda _argv=None: 11
        governance_audit._impl.run_lint_summary_cli = lambda _argv=None: 17

        parsed, body = governance_audit._parse_frontmatter("---\n---\nbody")
        assert parsed == {"x": "y"}
        assert body == "---\n---\nbody"
        assert governance_audit.run_decision_tiers_cli(["--help"]) == 11
        assert governance_audit.run_lint_summary_cli(["--help"]) == 17
    finally:
        governance_audit._impl._parse_frontmatter = original_parse
        governance_audit._impl.run_decision_tiers_cli = original_tiers
        governance_audit._impl.run_lint_summary_cli = original_lint

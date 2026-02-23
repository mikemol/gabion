from __future__ import annotations


def _load_audit_tools():
    from gabion.tooling import governance_audit as audit_tools

    return audit_tools


# gabion:evidence E:call_footprint::tests/test_docflow_governance_control_loops.py::test_docflow_loop_registry_violation_when_domain_missing::governance_audit.py::gabion.tooling.governance_audit._docflow_invariant_rows::test_docflow_governance_control_loops.py::tests.test_docflow_governance_control_loops._load_audit_tools
def test_docflow_loop_registry_violation_when_domain_missing() -> None:
    audit_tools = _load_audit_tools()
    docs = {
        "docs/governance_control_loops.md": audit_tools.Doc(
            frontmatter={
                "doc_id": "governance_control_loops",
                "loop_domains": ["security/workflows"],
            },
            body="",
        )
    }

    rows, _warnings = audit_tools._docflow_invariant_rows(
        docs=docs,
        revisions={},
        core_set=set(audit_tools.CORE_GOVERNANCE_DOCS),
        missing_frontmatter=set(),
    )
    violations = audit_tools._evaluate_docflow_invariants(
        rows,
        invariants=audit_tools.DOCFLOW_AUDIT_INVARIANTS,
    )

    assert any("missing governance control-loop declaration" in item for item in violations)


# gabion:evidence E:call_footprint::tests/test_docflow_governance_control_loops.py::test_docflow_loop_registry_satisfied_when_all_domains_declared::governance_audit.py::gabion.tooling.governance_audit._docflow_invariant_rows::test_docflow_governance_control_loops.py::tests.test_docflow_governance_control_loops._load_audit_tools
def test_docflow_loop_registry_satisfied_when_all_domains_declared() -> None:
    audit_tools = _load_audit_tools()
    docs = {
        "docs/governance_control_loops.md": audit_tools.Doc(
            frontmatter={
                "doc_id": "governance_control_loops",
                "loop_domains": list(audit_tools.NORMATIVE_LOOP_DOMAINS),
            },
            body="",
        )
    }

    rows, _warnings = audit_tools._docflow_invariant_rows(
        docs=docs,
        revisions={},
        core_set=set(audit_tools.CORE_GOVERNANCE_DOCS),
        missing_frontmatter=set(),
    )
    violations = audit_tools._evaluate_docflow_invariants(
        rows,
        invariants=audit_tools.DOCFLOW_AUDIT_INVARIANTS,
    )

    assert not any("missing governance control-loop declaration" in item for item in violations)

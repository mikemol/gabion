from __future__ import annotations

from gabion.tooling.governance_rules import load_governance_rules


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


# gabion:evidence E:call_footprint::tests/test_docflow_governance_control_loops.py::test_docflow_governance_loop_matrix_drift_detected_when_gate_missing::governance_audit.py::gabion.tooling.governance_audit._docflow_invariant_rows
def test_docflow_governance_loop_matrix_drift_detected_when_gate_missing() -> None:
    audit_tools = _load_audit_tools()
    docs = {
        "docs/governance_control_loops.md": audit_tools.Doc(
            frontmatter={
                "doc_id": "governance_control_loops",
                "loop_domains": list(audit_tools.NORMATIVE_LOOP_DOMAINS),
            },
            body="",
        ),
        "docs/governance_loop_matrix.md": audit_tools.Doc(
            frontmatter={"doc_id": "governance_loop_matrix"},
            body=(
                "| loop domain | gate ID | sensor command | state artifact path | correction mode | warning/blocking thresholds | override mechanism |\n"
                "| --- | --- | --- | --- | --- | --- | --- |\n"
                "| baseline ratchets | `obsolescence_opaque` | `mise exec -- python -m gabion.tooling.obsolescence_delta_gate` | `artifacts/out/test_obsolescence_delta.json` | `hard-fail` | `warning=0, block=1` | `GABION_GATE_OPAQUE_DELTA` |\n"
            ),
        ),
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

    assert any("governance loop matrix drift" in item for item in violations)


# gabion:evidence E:call_footprint::tests/test_docflow_governance_control_loops.py::test_docflow_governance_loop_matrix_registry_satisfied_for_all_gate_rows::governance_audit.py::gabion.tooling.governance_audit._docflow_invariant_rows
def test_docflow_governance_loop_matrix_registry_satisfied_for_all_gate_rows() -> None:
    audit_tools = _load_audit_tools()
    gate_rows = "\n".join(
        f"| baseline ratchets | `{gate_id}` | `sensor` | `artifact` | `mode` | `warning=0, block=1` | `override` |"
        for gate_id in sorted(load_governance_rules().gates)
    )
    docs = {
        "docs/governance_control_loops.md": audit_tools.Doc(
            frontmatter={
                "doc_id": "governance_control_loops",
                "loop_domains": list(audit_tools.NORMATIVE_LOOP_DOMAINS),
            },
            body="",
        ),
        "docs/governance_loop_matrix.md": audit_tools.Doc(
            frontmatter={"doc_id": "governance_loop_matrix"},
            body=(
                "| loop domain | gate ID | sensor command | state artifact path | correction mode | warning/blocking thresholds | override mechanism |\n"
                "| --- | --- | --- | --- | --- | --- | --- |\n"
                + gate_rows
                + "\n"
            ),
        ),
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

    assert not any("governance loop matrix drift" in item for item in violations)

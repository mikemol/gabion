from __future__ import annotations

import json
from pathlib import Path

from gabion.tooling import governance_audit as audit_tools


def _doc(*, revision: int, reviewed: dict[str, int], body: str) -> audit_tools.Doc:
    return audit_tools.Doc(
        frontmatter={
            "doc_revision": revision,
            "doc_reviewed_as_of": reviewed,
        },
        body=body,
    )


# gabion:evidence E:function_site::governance_audit.py::gabion.tooling.governance_audit._agent_instruction_graph E:decision_surface/direct::governance_audit.py::gabion.tooling.governance_audit._agent_instruction_graph::stale_eaa890508e34
def test_agent_instruction_graph_reports_drift_categories(tmp_path: Path) -> None:
    docs = {
        "AGENTS.md": _doc(
            revision=2,
            reviewed={"POLICY_SEED.md#policy_seed": 1},
            body=(
                "## Required behavior\n"
                "- Do not weaken runner protections.\n"
                "- Use `mise exec -- python` for tooling.\n"
            ),
        ),
        "POLICY_SEED.md": _doc(
            revision=2,
            reviewed={},
            body=(
                "## Policy\n"
                "- Use `--policy-override` only for emergency workflows.\n"
            ),
        ),
        "CONTRIBUTING.md": _doc(
            revision=1,
            reviewed={},
            body=(
                "## Contributing\n"
                "- Do not weaken runner protections.\n"
            ),
        ),
        "in/AGENTS.md": _doc(
            revision=1,
            reviewed={"POLICY_SEED.md#policy_seed": 1},
            body=(
                "## Required behavior\n"
                "- Weaken runner protections.\n"
            ),
        ),
    }

    warnings, violations = audit_tools._agent_instruction_graph(
        root=tmp_path,
        docs=docs,
        json_output=tmp_path / "out.json",
        md_output=tmp_path / "out.md",
    )

    assert warnings
    assert violations

    payload = json.loads((tmp_path / "out.json").read_text(encoding="utf-8"))
    assert payload["summary"]["duplicate_mandatory"] == 1
    assert payload["summary"]["precedence_conflicts"] == 1
    assert payload["summary"]["stale_dependency_revisions"] == 2
    assert payload["summary"]["hidden_operational_toggles"] == 1
    assert payload["summary"]["scoped_delta_violations"] == 1


# gabion:evidence E:function_site::governance_audit.py::gabion.tooling.governance_audit._agent_instruction_graph E:decision_surface/direct::governance_audit.py::gabion.tooling.governance_audit._agent_instruction_graph::stale_e67de1f8ca1f_457c58da
def test_agent_instruction_graph_allows_explicit_scoped_delta(tmp_path: Path) -> None:
    docs = {
        "AGENTS.md": _doc(
            revision=1,
            reviewed={"POLICY_SEED.md#policy_seed": 1},
            body="## Required behavior\n- Keep workflows pinned.\n",
        ),
        "POLICY_SEED.md": _doc(revision=1, reviewed={}, body="# Policy\n"),
        "CONTRIBUTING.md": _doc(revision=1, reviewed={}, body="# Contributing\n"),
        "in/AGENTS.md": _doc(
            revision=1,
            reviewed={"POLICY_SEED.md#policy_seed": 1},
            body="## Required behavior\n- [delta] Run additional local checks.\n",
        ),
    }

    warnings, violations = audit_tools._agent_instruction_graph(
        root=tmp_path,
        docs=docs,
        json_output=tmp_path / "out.json",
        md_output=tmp_path / "out.md",
    )

    assert warnings == []
    assert "scoped AGENTS directives must be canonical or explicit deltas" not in "\n".join(violations)


# gabion:evidence E:function_site::governance_audit.py::gabion.tooling.governance_audit._agent_instruction_graph E:decision_surface/direct::governance_audit.py::gabion.tooling.governance_audit._agent_instruction_graph::stale_82d5855095e4
def test_agent_instruction_graph_uses_anchor_revision_when_available(tmp_path: Path) -> None:
    docs = {
        "AGENTS.md": audit_tools.Doc(
            frontmatter={
                "doc_revision": 1,
                "doc_reviewed_as_of": {"POLICY_SEED.md#policy_seed": 1},
            },
            body="## Required behavior\n- Keep workflows pinned.\n",
        ),
        "POLICY_SEED.md": audit_tools.Doc(
            frontmatter={
                "doc_revision": 42,
                "doc_reviewed_as_of": {},
                "doc_sections": {"policy_seed": 1},
            },
            body="# Policy\n",
        ),
        "CONTRIBUTING.md": _doc(revision=1, reviewed={}, body="# Contributing\n"),
    }

    _warnings, violations = audit_tools._agent_instruction_graph(
        root=tmp_path,
        docs=docs,
        json_output=tmp_path / "out.json",
        md_output=tmp_path / "out.md",
    )

    assert "stale dependency revisions detected" not in "\n".join(violations)

from __future__ import annotations

import json
from pathlib import Path

from gabion.analysis import test_evidence, test_evidence_suggestions
from gabion.analysis.aspf import Forest
from gabion.analysis.dataflow_audit import AuditConfig


def _entries_from_payload(payload: dict[str, object]) -> list[test_evidence_suggestions.TestEvidenceEntry]:
    tests = payload.get("tests", [])
    entries: list[test_evidence_suggestions.TestEvidenceEntry] = []
    for entry in tests:
        evidence = tuple(item.get("display") for item in entry.get("evidence", []) if isinstance(item, dict))
        entries.append(
            test_evidence_suggestions.TestEvidenceEntry(
                test_id=entry["test_id"],
                file=entry["file"],
                line=entry["line"],
                evidence=tuple(item for item in evidence if item),
                status=entry["status"],
            )
        )
    return entries


def test_graph_decision_surface_suggestion(tmp_path: Path) -> None:
    root = tmp_path
    (root / "src" / "pkg").mkdir(parents=True)
    (root / "tests").mkdir(parents=True)
    (root / "src" / "pkg" / "__init__.py").write_text("")
    (root / "src" / "pkg" / "core.py").write_text(
        "def decide(flag):\n"
        "    if flag:\n"
        "        return 1\n"
        "    return 0\n"
    )
    (root / "tests" / "test_core.py").write_text(
        "from pkg.core import decide\n\n"
        "def test_decide():\n"
        "    decide(True)\n"
    )
    payload = test_evidence.build_test_evidence_payload([root / "tests"], root=root)
    entries = _entries_from_payload(payload)
    forest = Forest()
    site_id = forest.add_site("core.py", "pkg.core.decide")
    paramset_id = forest.add_paramset(["flag"])
    forest.add_alt("DecisionSurface", (site_id, paramset_id), evidence={"boundary": "boundary"})

    suggestions, summary = test_evidence_suggestions.suggest_evidence(
        entries,
        root=root,
        paths=[root],
        forest=forest,
        config=AuditConfig(project_root=root),
    )

    assert summary.suggested_graph == 1
    assert summary.suggested_heuristic == 0
    assert suggestions[0].source == "graph"
    assert suggestions[0].suggested[0].display == (
        "E:decision_surface/direct::core.py::pkg.core.decide::flag"
    )


def test_heuristic_fallback_when_graph_unavailable() -> None:
    entry = test_evidence_suggestions.TestEvidenceEntry(
        test_id="tests/test_alias_attribute.py::test_alias_attribute_forwarding",
        file="tests/test_alias_attribute.py",
        line=10,
        evidence=(),
        status="unmapped",
    )
    suggestions, summary = test_evidence_suggestions.suggest_evidence(
        [entry],
        root=Path("."),
        forest=None,
    )
    assert summary.suggested_heuristic == 1
    assert suggestions[0].source == "heuristic"
    assert suggestions[0].matches == ("alias_invariance",)


def test_graph_resolution_blocks_heuristics(tmp_path: Path) -> None:
    root = tmp_path
    (root / "src" / "pkg").mkdir(parents=True)
    (root / "tests").mkdir(parents=True)
    (root / "src" / "pkg" / "__init__.py").write_text("")
    (root / "src" / "pkg" / "core.py").write_text("def helper():\n    return 1\n")
    (root / "tests" / "test_alias_attribute.py").write_text(
        "from pkg.core import helper\n\n"
        "def test_alias_attribute_forwarding():\n"
        "    helper()\n"
    )
    payload = test_evidence.build_test_evidence_payload([root / "tests"], root=root)
    entries = _entries_from_payload(payload)

    suggestions, summary = test_evidence_suggestions.suggest_evidence(
        entries,
        root=root,
        paths=[root],
        forest=Forest(),
        config=AuditConfig(project_root=root),
    )

    assert suggestions == []
    assert summary.suggested_heuristic == 0
    assert summary.skipped_no_match == 1
    assert summary.graph_unresolved == 0


def test_skips_mapped_entries() -> None:
    entry = test_evidence_suggestions.TestEvidenceEntry(
        test_id="tests/test_baseline_ratchet.py::test_baseline_write_and_apply",
        file="tests/test_baseline_ratchet.py",
        line=2,
        evidence=("E:baseline/ratchet_monotonicity",),
        status="mapped",
    )
    suggestions, summary = test_evidence_suggestions.suggest_evidence(
        [entry],
        root=Path("."),
        forest=None,
    )
    assert suggestions == []
    assert summary.total == 1
    assert summary.skipped_mapped == 1


def test_load_test_evidence_payload(tmp_path: Path) -> None:
    payload = {
        "schema_version": 2,
        "scope": {"root": ".", "include": [], "exclude": []},
        "tests": [
            {
                "test_id": "tests/test_policy_check.py::test_policy_check_runs",
                "file": "tests/test_policy_check.py",
                "line": 5,
                "evidence": [],
                "status": "unmapped",
            }
        ],
        "evidence_index": [],
    }
    path = tmp_path / "test_evidence.json"
    path.write_text(json.dumps(payload))
    entries = test_evidence_suggestions.load_test_evidence(str(path))
    assert entries[0].test_id.endswith("test_policy_check.py::test_policy_check_runs")

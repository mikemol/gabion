from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from gabion import cli
from gabion.tooling import normative_symdiff


def _write_doc(path: Path, *, authority: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            (
                "---",
                "doc_revision: 1",
                "doc_id: sample",
                "doc_role: sample",
                "doc_scope: [repo]",
                f"doc_authority: {authority}",
                "---",
                "# Sample",
                "",
            )
        ),
        encoding="utf-8",
    )


def _seed_minimal_symdiff_root(root: Path) -> None:
    _write_doc(root / "AGENTS.md", authority="normative")
    _write_doc(root / "docs" / "extra_normative.md", authority="normative")
    _write_doc(root / "docs" / "info.md", authority="informative")
    clause_index = root / "docs" / "normative_clause_index.md"
    clause_index.parent.mkdir(parents=True, exist_ok=True)
    clause_index.write_text(
        "\n".join(
            (
                "<a id=\"clause-a\"></a>",
                "### `NCI-A` — Clause A",
                "<a id=\"clause-b\"></a>",
                "### `NCI-B` — Clause B",
                "",
            )
        ),
        encoding="utf-8",
    )
    enforcement_map = root / "docs" / "normative_enforcement_map.yaml"
    enforcement_map.write_text(
        "\n".join(
            (
                "version: 1",
                "clauses:",
                "  NCI-A:",
                "    status: partial",
                "    enforcing_modules: [src/missing.py]",
                "    ci_anchors:",
                "      - workflow: .github/workflows/ci.yml",
                "        job: checks",
                "        step: Missing Step",
                "    expected_artifacts: []",
                "  NCI-C:",
                "    status: enforced",
                "    enforcing_modules: []",
                "    ci_anchors: []",
                "    expected_artifacts: []",
                "",
            )
        ),
        encoding="utf-8",
    )
    workflow = root / ".github" / "workflows" / "ci.yml"
    workflow.parent.mkdir(parents=True, exist_ok=True)
    workflow.write_text(
        "\n".join(
            (
                "jobs:",
                "  checks:",
                "    runs-on: ubuntu-latest",
                "    steps:",
                "      - name: Existing Step",
                "        run: echo ok",
                "",
            )
        ),
        encoding="utf-8",
    )


def test_collect_scope_inventory_two_layer_classification(tmp_path: Path) -> None:
    _seed_minimal_symdiff_root(tmp_path)
    inventory = normative_symdiff.collect_scope_inventory(tmp_path)
    assert "AGENTS.md" in inventory.normative_docs
    assert "docs/extra_normative.md" in inventory.normative_docs
    assert "docs/info.md" not in inventory.normative_docs
    assert "docs/extra_normative.md" in inventory.extended_layer_docs
    assert "AGENTS.md" not in inventory.extended_layer_docs
    assert "docs/extra_normative.md" in inventory.outside_default_strict_docs


def test_analyze_clause_enforcement_detects_clause_map_drift(tmp_path: Path) -> None:
    _seed_minimal_symdiff_root(tmp_path)
    clause_ids = normative_symdiff._parse_clause_ids(
        tmp_path / "docs" / "normative_clause_index.md"
    )
    analysis = normative_symdiff.analyze_clause_enforcement(
        root=tmp_path,
        clause_ids=clause_ids,
        enforcement_map_path=tmp_path / "docs" / "normative_enforcement_map.yaml",
    )
    assert analysis["missing_from_map"] == ["NCI-B"]
    assert analysis["unknown_in_map"] == ["NCI-C"]
    assert analysis["partial_clauses"] == ["NCI-A"]
    assert any("missing enforcing module" in item for item in analysis["missing_modules"])
    assert any("missing workflow step anchor" in item for item in analysis["ci_anchor_errors"])


def test_score_gaps_dual_view_behaviors() -> None:
    absolute_only = {
        "doc_to_code_gaps": [
            {
                "gap_id": "ABS-1",
                "layer": "core",
                "direction": "doc_to_code",
                "model": "absolute",
                "severity": "high",
                "count": 1,
                "message": "absolute gap",
                "evidence": [],
            }
        ],
        "code_to_doc_gaps": [],
    }
    absolute_scores = normative_symdiff.score_gaps(absolute_only)
    assert absolute_scores["ratchet"]["overall"]["score"] == 100
    assert absolute_scores["absolute"]["overall"]["score"] < 100

    ratchet_and_absolute = {
        "doc_to_code_gaps": [
            {
                "gap_id": "RATCHET-1",
                "layer": "core",
                "direction": "doc_to_code",
                "model": "ratchet",
                "severity": "high",
                "count": 1,
                "message": "ratchet gap",
                "evidence": [],
            }
        ],
        "code_to_doc_gaps": [],
    }
    ratchet_scores = normative_symdiff.score_gaps(ratchet_and_absolute)
    assert ratchet_scores["ratchet"]["overall"]["score"] < 100
    assert ratchet_scores["absolute"]["overall"]["score"] == 100

    empty_scores = normative_symdiff.score_gaps(
        {"doc_to_code_gaps": [], "code_to_doc_gaps": []}
    )
    assert empty_scores["ratchet"]["overall"]["score"] == 100
    assert empty_scores["absolute"]["overall"]["score"] == 100


def test_ordered_gap_items_is_stable() -> None:
    gaps = [
        normative_symdiff.GapItem(
            gap_id="B",
            layer="core",
            direction="doc_to_code",
            model="absolute",
            severity="medium",
            count=1,
            message="m",
            evidence=(),
        ),
        normative_symdiff.GapItem(
            gap_id="A",
            layer="core",
            direction="doc_to_code",
            model="absolute",
            severity="high",
            count=1,
            message="h",
            evidence=(),
        ),
        normative_symdiff.GapItem(
            gap_id="C",
            layer="extended",
            direction="code_to_doc",
            model="absolute",
            severity="low",
            count=1,
            message="l",
            evidence=(),
        ),
    ]
    ordered_once = normative_symdiff._ordered_gap_items(
        gaps, source="tests.normative_symdiff.ordered_once"
    )
    ordered_twice = normative_symdiff._ordered_gap_items(
        list(reversed(gaps)),
        source="tests.normative_symdiff.ordered_twice",
    )
    assert [item.gap_id for item in ordered_once] == ["A", "B", "C"]
    assert [item.gap_id for item in ordered_once] == [
        item.gap_id for item in ordered_twice
    ]


def test_run_emits_json_and_markdown_shapes(tmp_path: Path) -> None:
    _seed_minimal_symdiff_root(tmp_path)
    json_out = tmp_path / "out" / "normative_symdiff.json"
    md_out = tmp_path / "out" / "normative_symdiff.md"
    exit_code = normative_symdiff.run(
        root=tmp_path,
        json_out=json_out,
        md_out=md_out,
        probe_mode="skip",
    )
    assert exit_code == 0
    payload = json.loads(json_out.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert "inventory" in payload
    assert "clauses" in payload
    assert "probes" in payload
    assert "gaps" in payload
    assert "scoring" in payload
    markdown = md_out.read_text(encoding="utf-8")
    assert "## Executive Summary" in markdown
    assert "## Core Layer Matrix" in markdown
    assert "## Extended Layer Matrix" in markdown
    assert "## How Close/Far" in markdown


def test_cli_normative_symdiff_smoke(tmp_path: Path) -> None:
    _seed_minimal_symdiff_root(tmp_path)
    json_out = tmp_path / "artifacts" / "out" / "normative_symdiff.json"
    md_out = tmp_path / "artifacts" / "audit_reports" / "normative_symdiff.md"
    runner = CliRunner()
    result = runner.invoke(
        cli.app,
        [
            "normative-symdiff",
            "--root",
            str(tmp_path),
            "--probe-mode",
            "skip",
            "--json-out",
            str(json_out),
            "--md-out",
            str(md_out),
        ],
        env={
            "GABION_DIRECT_RUN": "1",
            "GABION_LSP_TIMEOUT_TICKS": "50000",
            "GABION_LSP_TIMEOUT_TICK_NS": "1000000",
        },
    )
    assert result.exit_code == 0
    assert json_out.exists()
    assert md_out.exists()

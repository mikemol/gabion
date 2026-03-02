from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
import runpy
import sys
from types import SimpleNamespace

import pytest

from gabion.tooling import normative_symdiff


@contextmanager
def _swap_attr(obj: object, name: str, value: object):
    original = getattr(obj, name)
    setattr(obj, name, value)
    try:
        yield
    finally:
        setattr(obj, name, original)


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


def _seed_minimal_root(root: Path) -> None:
    _write_doc(root / "AGENTS.md", authority="normative")
    _write_doc(root / "docs" / "extra_normative.md", authority="normative")
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
                "    status: enforced",
                "    enforcing_modules: []",
                "    ci_anchors: []",
                "    expected_artifacts: []",
                "  NCI-B:",
                "    status: enforced",
                "    enforcing_modules: []",
                "    ci_anchors: []",
                "    expected_artifacts: []",
                "",
            )
        ),
        encoding="utf-8",
    )


# gabion:evidence E:function_site::tests/test_normative_symdiff_edges.py::test_scalar_helpers_and_distance_bands_cover_edges
def test_scalar_helpers_and_distance_bands_cover_edges() -> None:
    assert normative_symdiff._coerce_int(True) == 1
    assert normative_symdiff._coerce_int(7) == 7
    assert normative_symdiff._coerce_int(7.9) == 7
    assert normative_symdiff._coerce_int(" 8 ") == 8
    assert normative_symdiff._coerce_int("bad") == 0
    assert normative_symdiff._coerce_int(object()) == 0

    assert normative_symdiff._message_path_prefix("docs/a.md: x") == "docs/a.md"
    assert normative_symdiff._message_path_prefix("no-colon") is None
    assert normative_symdiff._message_path_prefix(":missing-prefix") is None

    assert normative_symdiff._ordered_strings(
        ["x", "x", " ", "y"],
        source="tests.normative_symdiff_edges.ordered_strings",
    ) == ["x", "y"]

    assert normative_symdiff._distance_band(95) == "very_close"
    assert normative_symdiff._distance_band(80) == "close"
    assert normative_symdiff._distance_band(60) == "moderate_gap"
    assert normative_symdiff._distance_band(59) == "far"

    assert normative_symdiff._gap_penalty_points(
        [
            {"severity": "high", "count": 0},
            {"severity": "medium", "count": 15},
        ]
    ) > 0


# gabion:evidence E:function_site::tests/test_normative_symdiff_edges.py::test_workflow_anchor_and_clause_analysis_cover_shape_errors
def test_workflow_anchor_and_clause_analysis_cover_shape_errors(tmp_path: Path) -> None:
    workflow = tmp_path / ".github" / "workflows" / "ci.yml"
    workflow.parent.mkdir(parents=True, exist_ok=True)
    workflow.write_text(
        "\n".join(
            (
                "jobs:",
                "  checks:",
                "    runs-on: ubuntu-latest",
                "    steps:",
                "      - bad",
                "      - name: Existing Step",
                "        run: echo ok",
                "",
            )
        ),
        encoding="utf-8",
    )
    workflow_nonlist_steps = tmp_path / ".github" / "workflows" / "nonlist_steps.yml"
    workflow_nonlist_steps.write_text(
        "\n".join(
            (
                "jobs:",
                "  checks:",
                "    runs-on: ubuntu-latest",
                "    steps: bad",
                "",
            )
        ),
        encoding="utf-8",
    )

    errors = normative_symdiff._workflow_anchor_errors(
        root=tmp_path,
        clauses_payload={
            "NCI-BAD-LIST": {"ci_anchors": "bad"},
            "NCI-BAD-ITEM": {"ci_anchors": ["bad"]},
            "NCI-MISSING-WORKFLOW": {
                "ci_anchors": [
                    {
                        "workflow": ".github/workflows/missing.yml",
                        "job": "checks",
                        "step": "Existing Step",
                    }
                ]
            },
            "NCI-MISSING-JOB": {
                "ci_anchors": [
                    {
                        "workflow": ".github/workflows/ci.yml",
                        "job": "missing-job",
                        "step": "",
                    }
                ]
            },
            "NCI-EMPTY-STEP": {
                "ci_anchors": [
                    {
                        "workflow": ".github/workflows/ci.yml",
                        "job": "checks",
                        "step": "",
                    }
                ]
            },
            "NCI-EXISTING-STEP": {
                "ci_anchors": [
                    {
                        "workflow": ".github/workflows/ci.yml",
                        "job": "checks",
                        "step": "Existing Step",
                    }
                ]
            },
            "NCI-NONLIST-STEPS": {
                "ci_anchors": [
                    {
                        "workflow": ".github/workflows/nonlist_steps.yml",
                        "job": "checks",
                        "step": "Any Step",
                    }
                ]
            },
            "NCI-MISSING-STEP": {
                "ci_anchors": [
                    {
                        "workflow": ".github/workflows/ci.yml",
                        "job": "checks",
                        "step": "Missing Step",
                    }
                ]
            },
        },
    )
    rendered = "\n".join(errors)
    assert "ci_anchors must be a list" in rendered
    assert "ci anchor must be mapping" in rendered
    assert "missing workflow .github/workflows/missing.yml" in rendered
    assert "missing workflow job anchor .github/workflows/ci.yml:missing-job" in rendered
    assert "missing workflow step anchor .github/workflows/ci.yml:checks:Missing Step" in rendered

    enforcement_map = tmp_path / "docs" / "normative_enforcement_map.yaml"
    enforcement_map.parent.mkdir(parents=True, exist_ok=True)
    existing_module = tmp_path / "src" / "existing.py"
    existing_module.parent.mkdir(parents=True, exist_ok=True)
    existing_module.write_text("# ok\n", encoding="utf-8")
    enforcement_map.write_text(
        "\n".join(
            (
                "version: 1",
                "clauses:",
                "  NCI-A:",
                "    status: partial",
                "    enforcing_modules: bad",
                "    ci_anchors: []",
                "    expected_artifacts: []",
                "  NCI-B:",
                "    status: enforced",
                "    enforcing_modules: [src/existing.py, src/missing.py]",
                "    ci_anchors:",
                "      - workflow: .github/workflows/ci.yml",
                "        job: missing-job",
                "        step: Existing Step",
                "    expected_artifacts: []",
                "  NCI-C: bad",
                "",
            )
        ),
        encoding="utf-8",
    )
    analysis = normative_symdiff.analyze_clause_enforcement(
        root=tmp_path,
        clause_ids=["NCI-A", "NCI-D"],
        enforcement_map_path=enforcement_map,
    )
    assert analysis["missing_from_map"] == ["NCI-D"]
    assert analysis["unknown_in_map"] == ["NCI-B", "NCI-C"]
    assert analysis["partial_clauses"] == ["NCI-A"]
    assert any("missing enforcing module" in msg for msg in analysis["missing_modules"])
    assert any("missing workflow job anchor" in msg for msg in analysis["ci_anchor_errors"])


# gabion:evidence E:function_site::tests/test_normative_symdiff_edges.py::test_collect_scope_inventory_skips_non_file_markdown_matches
def test_collect_scope_inventory_skips_non_file_markdown_matches(tmp_path: Path) -> None:
    _seed_minimal_root(tmp_path)
    (tmp_path / "docs" / "not_a_file.md").mkdir(parents=True, exist_ok=True)
    inventory = normative_symdiff.collect_scope_inventory(tmp_path)
    assert "docs/not_a_file.md" not in inventory.normative_docs


# gabion:evidence E:function_site::tests/test_normative_symdiff_edges.py::test_policy_and_controller_scope_contexts_restore_globals
def test_policy_and_controller_scope_contexts_restore_globals(tmp_path: Path) -> None:
    before_repo_root = normative_symdiff.policy_check.REPO_ROOT
    before_workflow_dir = normative_symdiff.policy_check.WORKFLOW_DIR
    before_allowed_actions = normative_symdiff.policy_check.ALLOWED_ACTIONS_FILE
    before_map = normative_symdiff.policy_check.NORMATIVE_ENFORCEMENT_MAP
    with normative_symdiff._policy_check_repo_scope(tmp_path):
        assert normative_symdiff.policy_check.REPO_ROOT == tmp_path
        assert normative_symdiff.policy_check.WORKFLOW_DIR == tmp_path / ".github" / "workflows"
        assert normative_symdiff.policy_check.ALLOWED_ACTIONS_FILE == tmp_path / "docs" / "allowed_actions.txt"
        assert (
            normative_symdiff.policy_check.NORMATIVE_ENFORCEMENT_MAP
            == tmp_path / "docs" / "normative_enforcement_map.yaml"
        )
    assert normative_symdiff.policy_check.REPO_ROOT == before_repo_root
    assert normative_symdiff.policy_check.WORKFLOW_DIR == before_workflow_dir
    assert normative_symdiff.policy_check.ALLOWED_ACTIONS_FILE == before_allowed_actions
    assert normative_symdiff.policy_check.NORMATIVE_ENFORCEMENT_MAP == before_map

    before_controller_root = normative_symdiff.governance_controller_audit.REPO_ROOT
    before_policy_path = normative_symdiff.governance_controller_audit.POLICY_PATH
    before_default_out = normative_symdiff.governance_controller_audit.DEFAULT_OUT
    with normative_symdiff._controller_audit_repo_scope(tmp_path):
        assert normative_symdiff.governance_controller_audit.REPO_ROOT == tmp_path
        assert normative_symdiff.governance_controller_audit.POLICY_PATH == tmp_path / "POLICY_SEED.md"
        assert (
            normative_symdiff.governance_controller_audit.DEFAULT_OUT
            == tmp_path / "artifacts" / "out" / "controller_drift.json"
        )
    assert normative_symdiff.governance_controller_audit.REPO_ROOT == before_controller_root
    assert normative_symdiff.governance_controller_audit.POLICY_PATH == before_policy_path
    assert normative_symdiff.governance_controller_audit.DEFAULT_OUT == before_default_out


# gabion:evidence E:function_site::tests/test_normative_symdiff_edges.py::test_capture_policy_check_and_lsp_parity_probe_edges
def test_capture_policy_check_and_lsp_parity_probe_edges(tmp_path: Path) -> None:
    ok_probe = normative_symdiff._capture_policy_check("ok", lambda: None)
    assert ok_probe["ok"] is True
    assert ok_probe["exit_code"] == 0

    fail_probe = normative_symdiff._capture_policy_check(
        "fail",
        lambda: (_ for _ in ()).throw(SystemExit("bad")),
    )
    assert fail_probe["ok"] is False
    assert fail_probe["exit_code"] == 1

    def _fake_lsp_parity(_ls: object, _payload: dict[str, object]) -> dict[str, object]:
        return {
            "checked_commands": "bad",
            "errors": "bad",
            "exit_code": "2",
        }

    with _swap_attr(normative_symdiff.server, "_execute_lsp_parity_gate_total", _fake_lsp_parity):
        probe = normative_symdiff._collect_lsp_parity(tmp_path)
    assert probe["ok"] is False
    assert probe["checked_command_count"] == 0
    assert probe["error_count"] == 0
    assert probe["errors"] == []


# gabion:evidence E:function_site::tests/test_normative_symdiff_edges.py::test_policy_probe_collectors_cover_new_and_total_branches
def test_policy_probe_collectors_cover_new_and_total_branches(tmp_path: Path) -> None:
    ambiguity_items = [
        SimpleNamespace(key="k0", rule_id="ACP-003", path="a.py"),
        SimpleNamespace(key="k1", rule_id="ACP-004", path="a.py"),
    ]
    with _swap_attr(
        normative_symdiff.ambiguity_contract_policy_check,
        "collect_violations",
        lambda _root: ambiguity_items,
    ):
        ambiguity = normative_symdiff._collect_ambiguity_probe(tmp_path)
    assert ambiguity["total"] == 2
    assert ambiguity["by_rule"] == {"ACP-003": 1, "ACP-004": 1}
    assert ambiguity["by_path"] == {"a.py": 2}

    with (
        _swap_attr(normative_symdiff.branchless_policy_check, "collect_violations", lambda **_kwargs: [SimpleNamespace(key="b0"), SimpleNamespace(key="b1")]),
        _swap_attr(normative_symdiff.branchless_policy_check, "_load_baseline", lambda _path: {"b0"}),
    ):
        branchless = normative_symdiff._collect_branchless_probe(tmp_path)
    assert branchless["new"] == 1
    assert branchless["total"] == 2

    with (
        _swap_attr(normative_symdiff.defensive_fallback_policy_check, "collect_violations", lambda **_kwargs: [SimpleNamespace(key="d0")]),
        _swap_attr(normative_symdiff.defensive_fallback_policy_check, "_load_baseline", lambda _path: set()),
    ):
        defensive = normative_symdiff._collect_defensive_probe(tmp_path)
    assert defensive["new"] == 1
    assert defensive["total"] == 1

    with _swap_attr(
        normative_symdiff.no_monkeypatch_policy_check,
        "collect_violations",
        lambda **_kwargs: [SimpleNamespace(path="tests/a.py"), SimpleNamespace(path="tests/a.py")],
    ):
        monkey = normative_symdiff._collect_no_monkeypatch_probe(tmp_path)
    assert monkey["total"] == 2
    assert monkey["by_path"] == {"tests/a.py": 2}

    with _swap_attr(
        normative_symdiff.order_lifetime_check,
        "collect_violations",
        lambda **_kwargs: [SimpleNamespace(path="src/a.py"), SimpleNamespace(path="src/b.py")],
    ):
        order_lifetime = normative_symdiff._collect_order_lifetime_probe(tmp_path)
    assert order_lifetime["total"] == 2
    assert order_lifetime["by_path"] == {"src/a.py": 1, "src/b.py": 1}

    with _swap_attr(
        normative_symdiff.structural_hash_policy_check,
        "collect_violations",
        lambda **_kwargs: [SimpleNamespace(path="src/hash.py")],
    ):
        structural_hash = normative_symdiff._collect_structural_hash_probe(tmp_path)
    assert structural_hash["total"] == 1
    assert structural_hash["by_path"] == {"src/hash.py": 1}


# gabion:evidence E:function_site::tests/test_normative_symdiff_edges.py::test_docflow_agent_and_default_probe_collectors_cover_filtering
def test_docflow_agent_and_default_probe_collectors_cover_filtering(tmp_path: Path) -> None:
    def _fake_docflow_context(**_kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(
            violations=["docs/extra_normative.md: missing anchor"],
            warnings=["docs/other.md: warning"],
        )

    def _fake_agent_instruction_graph(
        *,
        root: Path,
        docs: object,
        json_output: Path,
        md_output: Path,
    ) -> tuple[list[str], list[str]]:
        _ = root, docs
        json_output.write_text(
            json.dumps(
                {
                    "summary": {"hidden_toggle_count": 1},
                    "hidden_operational_toggles": [
                        {"source": "docs/governance.md", "token": "ENABLE_X"}
                    ],
                }
            ),
            encoding="utf-8",
        )
        md_output.write_text("# graph\n", encoding="utf-8")
        return (["warn"], ["violation"])

    with (
        _swap_attr(normative_symdiff.governance_audit, "_docflow_audit_context", _fake_docflow_context),
        _swap_attr(normative_symdiff.governance_audit, "_load_docflow_docs", lambda **_kwargs: []),
        _swap_attr(normative_symdiff.governance_audit, "_agent_instruction_graph", _fake_agent_instruction_graph),
    ):
        docflow = normative_symdiff._collect_docflow_context(
            root=tmp_path,
            extra_paths=["docs/extra_normative.md"],
            extra_strict=True,
        )
        assert docflow["violation_count"] == 1
        assert docflow["warning_count"] == 1

        agent_probe = normative_symdiff._collect_agent_instruction_probe(tmp_path)
        assert agent_probe["warnings"] == ["warn"]
        assert agent_probe["violations"] == ["violation"]
        assert agent_probe["hidden_operational_toggles"] == [
            {"source": "docs/governance.md", "token": "ENABLE_X"}
        ]

    @contextmanager
    def _null_scope():
        yield

    with (
        _swap_attr(normative_symdiff.governance_audit, "_audit_deadline_scope", _null_scope),
        _swap_attr(normative_symdiff, "_capture_policy_check", lambda _name, _fn: {"ok": False, "stderr": "x", "exit_code": 1}),
        _swap_attr(normative_symdiff, "_collect_controller_drift", lambda _root: {"summary": {"high_severity_findings": 1}, "findings": []}),
        _swap_attr(normative_symdiff, "_collect_lsp_parity", lambda _root: {"error_count": 1, "errors": ["e"]}),
        _swap_attr(normative_symdiff, "_collect_ambiguity_probe", lambda _root: {"total": 2, "by_rule": {}, "by_path": {}}),
        _swap_attr(normative_symdiff, "_collect_branchless_probe", lambda _root: {"new": 0, "total": 1, "baseline_keys": 2}),
        _swap_attr(normative_symdiff, "_collect_defensive_probe", lambda _root: {"new": 0, "total": 1, "baseline_keys": 3}),
        _swap_attr(normative_symdiff, "_collect_no_monkeypatch_probe", lambda _root: {"total": 1, "by_path": {"a.py": 1}}),
        _swap_attr(normative_symdiff, "_collect_order_lifetime_probe", lambda _root: {"total": 1, "by_path": {"b.py": 1}}),
        _swap_attr(normative_symdiff, "_collect_structural_hash_probe", lambda _root: {"total": 1, "by_path": {"c.py": 1}}),
        _swap_attr(
            normative_symdiff,
            "_collect_docflow_context",
            lambda *, root, extra_paths, extra_strict: {
                "violations": [
                    "docs/extra_normative.md: v1",
                    "docs/other.md: v2",
                ],
                "warnings": [
                    "docs/extra_normative.md: w1",
                    "docs/other.md: w2",
                ],
                "violation_count": 2,
                "warning_count": 2,
            },
        ),
        _swap_attr(
            normative_symdiff,
            "_collect_agent_instruction_probe",
            lambda _root: {
                "warnings": [],
                "violations": [],
                "summary": {},
                "hidden_operational_toggles": [],
            },
        ),
    ):
        probes = normative_symdiff._collect_default_probes(
            root=tmp_path,
            extended_layer_docs=["docs/extra_normative.md"],
        )
    extended = probes["docflow_extended_strict"]
    assert extended["extended_violation_count"] == 1
    assert extended["extended_warning_count"] == 1
    assert probes["workflow_policy"]["ok"] is False
    assert probes["normative_map_policy"]["ok"] is False


# gabion:evidence E:function_site::tests/test_normative_symdiff_edges.py::test_gap_synthesis_scoring_and_markdown_edge_paths
def test_gap_synthesis_scoring_and_markdown_edge_paths() -> None:
    scope = normative_symdiff.ScopeInventory(
        normative_docs=("AGENTS.md",),
        core_layer_docs=("AGENTS.md",),
        extended_layer_docs=("docs/extra_normative.md",),
        outside_default_strict_docs=("docs/extra_normative.md",),
    )
    clause_analysis = {
        "missing_from_map": ["NCI-A"],
        "unknown_in_map": ["NCI-Z"],
        "partial_clauses": ["NCI-B"],
        "missing_modules": ["NCI-A: missing module"],
        "ci_anchor_errors": ["NCI-A: missing ci anchor"],
    }
    probes = {
        "workflow_policy": {"ok": False, "stderr": "workflow err"},
        "lsp_parity_gate": {"error_count": 1, "errors": ["parity err"]},
        "controller_drift": {
            "summary": {"high_severity_findings": 2},
            "findings": [
                {"sensor": "checks_without_normative_anchor", "detail": "missing anchor"},
                {"sensor": "unindexed_enforcement_surfaces", "detail": "missing clause id"},
                {"sensor": "x", "detail": "other"},
            ],
        },
        "ambiguity_contract": {"total": 9, "by_rule": {"ACP-003": 3}, "by_path": {"a.py": 9}},
        "branchless_policy": {"new": 1, "total": 2, "baseline_keys": 5},
        "defensive_fallback_policy": {"new": 1, "total": 2, "baseline_keys": 6},
        "no_monkeypatch_policy": {"total": 1, "by_path": {"tests/a.py": 1}},
        "order_lifetime_policy": {"total": 1, "by_path": {"src/a.py": 1}},
        "structural_hash_policy": {"total": 1, "by_path": {"src/b.py": 1}},
        "docflow_core": {
            "violation_count": 1,
            "warning_count": 1,
            "violations": ["AGENTS.md: missing"],
            "warnings": ["AGENTS.md: warning"],
        },
        "docflow_extended_strict": {
            "extended_violation_count": 1,
            "extended_warning_count": 1,
            "extended_violations": ["docs/extra_normative.md: missing"],
            "extended_warnings": ["docs/extra_normative.md: warning"],
        },
        "agent_instruction_graph": {
            "hidden_operational_toggles": [{"source": "docs/x.md", "token": "ENABLE_X"}]
        },
    }
    gap_payload = normative_symdiff.synthesize_gaps(
        scope_inventory=scope,
        clause_analysis=clause_analysis,
        probes=probes,
    )
    gap_ids = {
        str(item["gap_id"])
        for item in gap_payload["doc_to_code_gaps"] + gap_payload["code_to_doc_gaps"]
    }
    assert "DOC-CODE-AMBIGUITY-TOTAL" in gap_ids
    assert "CODE-DOC-HIDDEN-TOGGLES" in gap_ids
    assert "CODE-DOC-OUTSIDE-DEFAULT-STRICT" in gap_ids

    score_matrix = normative_symdiff.score_gaps(gap_payload)
    assert score_matrix["ratchet"]["overall"]["score"] < 100
    assert score_matrix["absolute"]["overall"]["score"] < 100
    score_with_non_list = normative_symdiff.score_gaps(
        {"doc_to_code_gaps": "bad", "code_to_doc_gaps": []}
    )
    assert score_with_non_list["ratchet"]["overall"]["score"] == 100

    markdown = normative_symdiff.render_markdown(
        {
            "generated_at_utc": "2026-01-01T00:00:00Z",
            "scope_model": "two-layer",
            "debt_model": "dual",
            "code_state": "worktree",
            "summary": {
                "doc_to_code_gap_count": len(gap_payload["doc_to_code_gaps"]),
                "code_to_doc_gap_count": len(gap_payload["code_to_doc_gaps"]),
                "ratchet_overall_score": score_matrix["ratchet"]["overall"]["score"],
                "ratchet_overall_band": score_matrix["ratchet"]["overall"]["distance_band"],
                "absolute_overall_score": score_matrix["absolute"]["overall"]["score"],
                "absolute_overall_band": score_matrix["absolute"]["overall"]["distance_band"],
            },
            "inventory": {"counts": {"core_layer_docs": 1, "extended_layer_docs": 1, "outside_default_strict_docs": 1}},
            "clauses": {"clause_ids": ["NCI-A"], "partial_clauses": ["NCI-B"]},
            "gaps": gap_payload,
            "scoring": {"ratchet": [], "absolute": {"core": {}, "extended": [], "overall": {}}},
        }
    )
    assert "## How Close/Far" in markdown
    markdown_without_scoring = normative_symdiff.render_markdown(
        {
            "generated_at_utc": "2026-01-01T00:00:00Z",
            "scope_model": "two-layer",
            "debt_model": "dual",
            "code_state": "worktree",
            "summary": {},
            "inventory": {},
            "clauses": {},
            "gaps": {"doc_to_code_gaps": [], "code_to_doc_gaps": []},
            "scoring": [],
        }
    )
    assert "## How Close/Far" not in markdown_without_scoring
    assert "- (none)" in normative_symdiff._render_gap_lines([])
    assert normative_symdiff._render_gap_lines(gap_payload["doc_to_code_gaps"])

    non_mapping_branches = normative_symdiff.synthesize_gaps(
        scope_inventory=scope,
        clause_analysis=clause_analysis,
        probes={
            "workflow_policy": [],
            "lsp_parity_gate": [],
            "controller_drift": [],
            "ambiguity_contract": [],
            "branchless_policy": [],
            "defensive_fallback_policy": [],
            "no_monkeypatch_policy": [],
            "order_lifetime_policy": [],
            "structural_hash_policy": [],
            "docflow_core": [],
            "docflow_extended_strict": [],
            "agent_instruction_graph": [],
        },
    )
    assert non_mapping_branches["doc_to_code_gaps"] or non_mapping_branches["code_to_doc_gaps"]


# gabion:evidence E:function_site::tests/test_normative_symdiff_edges.py::test_run_and_main_cover_probe_mode_and_module_entrypoint
def test_run_and_main_cover_probe_mode_and_module_entrypoint(tmp_path: Path) -> None:
    _seed_minimal_root(tmp_path)
    json_out = tmp_path / "out" / "symdiff.json"
    md_out = tmp_path / "out" / "symdiff.md"
    exit_code = normative_symdiff.run(
        root=tmp_path,
        json_out=json_out,
        md_out=md_out,
        probe_mode="full",
        probe_collector=lambda _root, _extended: {},
    )
    assert exit_code == 0
    assert json_out.exists()
    assert md_out.exists()

    argv_before = list(sys.argv)
    try:
        sys.argv = [
            "normative_symdiff.py",
            "--root",
            str(tmp_path),
            "--probe-mode",
            "skip",
            "--json-out",
            str(tmp_path / "out" / "module_symdiff.json"),
            "--md-out",
            str(tmp_path / "out" / "module_symdiff.md"),
        ]
        with pytest.raises(SystemExit) as exc:
            runpy.run_module("gabion.tooling.normative_symdiff", run_name="__main__")
        assert exc.value.code == 0
    finally:
        sys.argv = argv_before


# gabion:evidence E:function_site::tests/test_normative_symdiff_edges.py::test_collect_controller_drift_handles_missing_output_file
def test_collect_controller_drift_handles_missing_output_file(tmp_path: Path) -> None:
    def _fake_run(*, policy_path: Path, out_path: Path, fail_on_severity: object) -> int:
        _ = policy_path, out_path, fail_on_severity
        return 0

    with _swap_attr(normative_symdiff.governance_controller_audit, "run", _fake_run):
        payload = normative_symdiff._collect_controller_drift(tmp_path)
    assert payload["ok"] is True
    assert payload["findings"] == []
    assert payload["findings_by_sensor"] == {}


# gabion:evidence E:function_site::tests/test_normative_symdiff_edges.py::test_collect_controller_drift_counts_sensor_findings_when_output_written
def test_collect_controller_drift_counts_sensor_findings_when_output_written(
    tmp_path: Path,
) -> None:
    def _fake_run(*, policy_path: Path, out_path: Path, fail_on_severity: object) -> int:
        _ = policy_path, fail_on_severity
        out_path.write_text(
            json.dumps(
                {
                    "summary": {"high_severity_findings": 1},
                    "findings": [
                        {"sensor": "sensor-a", "detail": "x"},
                        {"sensor": "", "detail": "skip"},
                        {"sensor": "sensor-a", "detail": "y"},
                    ],
                }
            ),
            encoding="utf-8",
        )
        return 1

    with _swap_attr(normative_symdiff.governance_controller_audit, "run", _fake_run):
        payload = normative_symdiff._collect_controller_drift(tmp_path)
    assert payload["ok"] is False
    assert payload["findings_by_sensor"] == {"sensor-a": 2}

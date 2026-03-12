from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from scripts.policy import build_projection_spec_history as history_builder


def _inventory() -> dict[str, object]:
    return {
        "generated_at_utc": "2026-03-10T22:31:10.611285+00:00",
        "summary": {
            "file_count": 4,
            "hit_count": 10,
            "term_hit_count": 20,
        },
        "files": [
            {
                "path": "src/gabion/analysis/projection/projection_spec.py",
                "term_hit_count": 9,
                "category": "src",
                "area_tags": ["projection_core"],
            },
            {
                "path": "src/gabion/analysis/dataflow/io/dataflow_projection_helpers.py",
                "term_hit_count": 7,
                "category": "src",
                "area_tags": ["dataflow_projection"],
            },
            {
                "path": "docs/ws5_decomposition_ledger.md",
                "term_hit_count": 3,
                "category": "docs",
                "area_tags": ["general"],
            },
            {
                "path": "tests/gabion/analysis/projection/test_projection_spec.py",
                "term_hit_count": 1,
                "category": "tests",
                "area_tags": ["projection_core"],
            },
        ],
    }


@dataclass(frozen=True)
class _FakeCommit:
    sha: str
    date: str
    subject: str


class _FakeGit(history_builder.GitInterface):
    def __init__(self) -> None:
        self._default_commits = (
            _FakeCommit("aaa1111", "2026-02-06", "projection core start"),
            _FakeCommit("bbb2222", "2026-03-08", "projection core update"),
            _FakeCommit("ccc3333", "2026-03-10", "dsl convergence"),
        )
        self._all_shas = {
            commit.sha
            for commit in self._default_commits
        }

    def log(self, *, paths: tuple[str, ...]) -> tuple[history_builder.CommitRecord, ...]:
        # deterministic static evidence regardless of path selection.
        return tuple(
            history_builder.CommitRecord(
                sha=record.sha,
                date=record.date,
                subject=record.subject,
            )
            for record in self._default_commits
        )

    def status(self) -> tuple[history_builder.WorkspaceChange, ...]:
        return (
            history_builder.WorkspaceChange(
                status_code=" M",
                path="src/gabion/analysis/projection/projection_exec.py",
            ),
            history_builder.WorkspaceChange(
                status_code="??",
                path="docs/projection_fiber_rules.yaml",
            ),
        )

    def commit_exists(self, sha: str) -> bool:
        return sha in self._all_shas

    def tracked(self, path: str) -> bool:
        return path not in {
            "docs/projection_fiber_rules.yaml",
            "src/gabion/analysis/aspf/aspf_lattice_algebra.py",
        }


def _write_statement_source_docs(root: Path) -> None:
    docs = {
        "in/in-30.md": "ProjectionSpec remains unchanged — only the base carrier improves.\nProjection Idempotence\n",
        "in/in-31.md": "ProjectionSpec is a quotient morphism that erases evidence.\nInternment is the gauge-fixing inverse.\n",
        "in/in-32.md": "After Phase 2: Decoration model is backward compatible with ProjectionSpec\n",
        "docs/ws5_decomposition_ledger.md": "_materialize_projection_spec_rows\n_topologically_order_report_projection_specs\nReportProjectionSpec\n",
        "docs/audits/dataflow_runtime_debt_ledger.md": "DFD-037\nDFD-038\n",
        "docs/governance_control_loops.md": "policy_dsl/compile.py policy_dsl/eval.py\n",
        "docs/enforceable_rules_cheat_sheet.md": "typed policy DSL with aspf_opportunity_rules.yaml\n",
        "docs/aspf_execution_fibration.md": "Policy DSL ownership\nLattice algebra ownership\n",
        "docs/policy_dsl_migration_notes.md": "temporary boundary adapters\nremoval_condition\n",
        "scripts/policy/policy_check.py": (
            "iter_semantic_lattice_convergence\n"
            "materialize_semantic_lattice_convergence\n"
        ),
    }
    for rel_path, content in docs.items():
        path = root / rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")


# gabion:evidence E:function_site::test_build_projection_spec_history.py::tests.gabion.tooling.policy.test_build_projection_spec_history.test_build_history_is_deterministic
# gabion:behavior primary=desired
def test_build_history_is_deterministic(tmp_path: Path) -> None:
    _write_statement_source_docs(tmp_path)
    (tmp_path / "artifacts/out").mkdir(parents=True, exist_ok=True)
    (tmp_path / "artifacts/out/projection_spec_inventory.json").write_text(
        "{}\n", encoding="utf-8"
    )

    git = _FakeGit()
    first = history_builder.build_history(
        repo_root=tmp_path,
        inventory=_inventory(),
        git=git,
    )
    second = history_builder.build_history(
        repo_root=tmp_path,
        inventory=_inventory(),
        git=git,
    )
    assert first == second


# gabion:evidence E:function_site::test_build_projection_spec_history.py::tests.gabion.tooling.policy.test_build_projection_spec_history.test_validate_history_checks_schema_and_commit_integrity
# gabion:behavior primary=desired
def test_validate_history_checks_schema_and_commit_integrity(tmp_path: Path) -> None:
    _write_statement_source_docs(tmp_path)
    (tmp_path / "artifacts/out").mkdir(parents=True, exist_ok=True)
    (tmp_path / "artifacts/out/projection_spec_inventory.json").write_text(
        "{}\n", encoding="utf-8"
    )
    git = _FakeGit()
    history = history_builder.build_history(
        repo_root=tmp_path,
        inventory=_inventory(),
        git=git,
    )
    assert history["eras"]
    first_era = history["eras"][0]
    first_era["evidence_commits"] = [
        {
            "sha": "missing123",
            "date": "2026-03-10",
            "subject": "missing",
        }
    ]
    errors = history_builder.validate_history(
        history=history,
        git=git,
        inventory=_inventory(),
    )
    assert any("references missing commit missing123" in error for error in errors)


# gabion:evidence E:function_site::test_build_projection_spec_history.py::tests.gabion.tooling.policy.test_build_projection_spec_history.test_workspace_delta_entries_are_marked_provisional
# gabion:behavior primary=desired
def test_workspace_delta_entries_are_marked_provisional(tmp_path: Path) -> None:
    _write_statement_source_docs(tmp_path)
    (tmp_path / "artifacts/out").mkdir(parents=True, exist_ok=True)
    (tmp_path / "artifacts/out/projection_spec_inventory.json").write_text(
        "{}\n", encoding="utf-8"
    )
    git = _FakeGit()
    history = history_builder.build_history(
        repo_root=tmp_path,
        inventory=_inventory(),
        git=git,
    )
    eras = history["eras"]
    assert isinstance(eras, list)
    projection_era = next(item for item in eras if item["era_id"] == "PS-ERA-01")
    workspace_delta = projection_era["workspace_delta"]
    assert workspace_delta
    assert all(item["provisional"] is True for item in workspace_delta)


# gabion:evidence E:function_site::test_build_projection_spec_history.py::tests.gabion.tooling.policy.test_build_projection_spec_history.test_summary_and_hotspot_rankings_match_inventory
# gabion:behavior primary=desired
def test_summary_and_hotspot_rankings_match_inventory(tmp_path: Path) -> None:
    _write_statement_source_docs(tmp_path)
    (tmp_path / "artifacts/out").mkdir(parents=True, exist_ok=True)
    (tmp_path / "artifacts/out/projection_spec_inventory.json").write_text(
        "{}\n", encoding="utf-8"
    )
    git = _FakeGit()
    inventory = _inventory()
    history = history_builder.build_history(
        repo_root=tmp_path,
        inventory=inventory,
        git=git,
    )
    errors = history_builder.validate_history(
        history=history,
        git=git,
        inventory=inventory,
    )
    assert errors == []
    assert history["summary"]["file_count"] == inventory["summary"]["file_count"]
    assert history["summary"]["term_hit_count"] == inventory["summary"]["term_hit_count"]
    top_hotspots = history["top_hotspots"]
    assert top_hotspots[0]["path"] == "src/gabion/analysis/projection/projection_spec.py"
    assert top_hotspots[0]["term_hit_count"] == 9
    lowering_focus = history["semantic_lowering_focus"]
    assert lowering_focus["summary"]["registered_spec_count"] > 0
    rows = lowering_focus["rows"]
    frontier_row = next(
        item for item in rows if item["spec_name"] == "projection_fiber_frontier"
    )
    assert frontier_row["lowering_status"] == "mixed"
    assert frontier_row["semantic_op_count"] == 1
    assert frontier_row["presentation_op_count"] == 1
    assert frontier_row["bridge_op_count"] == 0
    assert frontier_row["quotient_faces"] == ["projection_fiber.frontier"]
    reflective_boundary_row = next(
        item
        for item in rows
        if item["spec_name"] == "projection_fiber_reflective_boundary"
    )
    assert reflective_boundary_row["lowering_status"] == "mixed"
    assert reflective_boundary_row["semantic_op_count"] == 1
    assert reflective_boundary_row["presentation_op_count"] == 1
    assert reflective_boundary_row["bridge_op_count"] == 0
    assert reflective_boundary_row["quotient_faces"] == [
        "projection_fiber.reflective_boundary"
    ]
    reflection_row = next(
        item for item in rows if item["spec_name"] == "projection_fiber_reflection"
    )
    assert reflection_row["lowering_status"] == "semantic_promoted"
    assert reflection_row["semantic_op_count"] == 1
    assert reflection_row["presentation_op_count"] == 0
    assert reflection_row["bridge_op_count"] == 0
    assert reflection_row["quotient_faces"] == []
    support_reflection_row = next(
        item for item in rows if item["spec_name"] == "projection_fiber_support_reflection"
    )
    assert support_reflection_row["lowering_status"] == "semantic_promoted"
    assert support_reflection_row["semantic_op_count"] == 1
    assert support_reflection_row["presentation_op_count"] == 0
    assert support_reflection_row["bridge_op_count"] == 0
    assert support_reflection_row["quotient_faces"] == []
    context_wedge_row = next(
        item for item in rows if item["spec_name"] == "projection_fiber_context_wedge"
    )
    assert context_wedge_row["lowering_status"] == "semantic_promoted"
    assert context_wedge_row["semantic_op_count"] == 1
    assert context_wedge_row["presentation_op_count"] == 0
    assert context_wedge_row["bridge_op_count"] == 0
    assert context_wedge_row["quotient_faces"] == []
    reindex_row = next(
        item for item in rows if item["spec_name"] == "projection_fiber_reindex"
    )
    assert reindex_row["lowering_status"] == "semantic_promoted"
    assert reindex_row["semantic_op_count"] == 1
    assert reindex_row["presentation_op_count"] == 0
    assert reindex_row["bridge_op_count"] == 0
    assert reindex_row["quotient_faces"] == []
    witness_synthesis_row = next(
        item for item in rows if item["spec_name"] == "projection_fiber_witness_synthesis"
    )
    assert witness_synthesis_row["lowering_status"] == "semantic_promoted"
    assert witness_synthesis_row["semantic_op_count"] == 1
    assert witness_synthesis_row["presentation_op_count"] == 0
    assert witness_synthesis_row["bridge_op_count"] == 0
    assert witness_synthesis_row["quotient_faces"] == []

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from scripts.policy import policy_check
from tests.gabion.tooling.runtime_policy.invariant_graph_test_support import (
    connectivity_synergy_with_psf_stub_workstream_registries,
    synthetic_connectivity_workstream_registries,
    write_minimal_invariant_repo,
)


_CONNECTIVITY_SYNERGY_WITH_PSF_STUB_DECLARED_REGISTRIES = (
    connectivity_synergy_with_psf_stub_workstream_registries()
)
_SYNTHETIC_CONNECTIVITY_DECLARED_REGISTRIES = (
    synthetic_connectivity_workstream_registries()
)


def _load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"failed to load module spec: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_policy_check_perf_helpers(repo_root: Path):
    helper_path = repo_root / "src" / "gabion" / "policy_check_perf_helpers.py"
    helper_path.write_text(
        "\n".join(
            [
                "from pathlib import Path",
                "from gabion.invariants import todo_decorator",
                "",
                "@todo_decorator(",
                "    reason='policy_check perf helper',",
                "    owner='tests.gabion.tooling.runtime_policy',",
                "    expiry='2099-01-01',",
                "    reasoning={",
                "        'summary': 'Synthetic policy_check perf helper',",
                "        'control': 'tests.runtime_policy.policy_check_perf_helper',",
                "        'blocking_dependencies': ['TEST-PERF-HELPER'],",
                "    },",
                "    links=[{'kind': 'object_id', 'value': 'TEST-PERF-HELPER'}],",
                ")",
                "def write_git_state_artifact(",
                "    *,",
                "    output_path: Path,",
                "    repo_root: Path | None = None,",
                ") -> None:",
                "    total = 0",
                "    for index in range(50_000):",
                "        total += index",
                "    (output_path.parent / 'fake_git_state.json').write_text(",
                "        str(total),",
                "        encoding='utf-8',",
                "    )",
                "",
                "@todo_decorator(",
                "    reason='policy_check perf helper',",
                "    owner='tests.gabion.tooling.runtime_policy',",
                "    expiry='2099-01-01',",
                "    reasoning={",
                "        'summary': 'Synthetic policy_check perf helper',",
                "        'control': 'tests.runtime_policy.policy_check_perf_helper',",
                "        'blocking_dependencies': ['TEST-PERF-HELPER'],",
                "    },",
                "    links=[{'kind': 'object_id', 'value': 'TEST-PERF-HELPER'}],",
                ")",
                "def write_cross_origin_witness_contract_artifact(",
                "    *,",
                "    output_path: Path,",
                "    repo_root: Path | None = None,",
                ") -> None:",
                "    (output_path.parent / 'fake_cross_origin_witness_contract.json').write_text(",
                "        'ok',",
                "        encoding='utf-8',",
                "    )",
                "",
                "@todo_decorator(",
                "    reason='policy_check perf helper',",
                "    owner='tests.gabion.tooling.runtime_policy',",
                "    expiry='2099-01-01',",
                "    reasoning={",
                "        'summary': 'Synthetic policy_check perf helper',",
                "        'control': 'tests.runtime_policy.policy_check_perf_helper',",
                "        'blocking_dependencies': ['TEST-PERF-HELPER'],",
                "    },",
                "    links=[{'kind': 'object_id', 'value': 'TEST-PERF-HELPER'}],",
                ")",
                "def write_kernel_vm_alignment_artifact(",
                "    *,",
                "    output_path: Path,",
                "    repo_root: Path | None = None,",
                ") -> None:",
                "    (output_path.parent / 'fake_kernel_vm_alignment.json').write_text(",
                "        'ok',",
                "        encoding='utf-8',",
                "    )",
                "",
                "@todo_decorator(",
                "    reason='policy_check perf helper',",
                "    owner='tests.gabion.tooling.runtime_policy',",
                "    expiry='2099-01-01',",
                "    reasoning={",
                "        'summary': 'Synthetic policy_check perf helper',",
                "        'control': 'tests.runtime_policy.policy_check_perf_helper',",
                "        'blocking_dependencies': ['TEST-PERF-HELPER'],",
                "    },",
                "    links=[{'kind': 'object_id', 'value': 'TEST-PERF-HELPER'}],",
                ")",
                "def write_identity_grammar_completion_artifact(",
                "    *,",
                "    output_path: Path,",
                "    repo_root: Path | None = None,",
                ") -> None:",
                "    (output_path.parent / 'fake_identity_grammar_completion.json').write_text(",
                "        'ok',",
                "        encoding='utf-8',",
                "    )",
                "",
                "@todo_decorator(",
                "    reason='policy_check perf helper',",
                "    owner='tests.gabion.tooling.runtime_policy',",
                "    expiry='2099-01-01',",
                "    reasoning={",
                "        'summary': 'Synthetic policy_check perf helper',",
                "        'control': 'tests.runtime_policy.policy_check_perf_helper',",
                "        'blocking_dependencies': ['TEST-PERF-HELPER'],",
                "    },",
                "    links=[{'kind': 'object_id', 'value': 'TEST-PERF-HELPER'}],",
                ")",
                "def write_ingress_merge_parity_artifact(",
                "    *,",
                "    output_path: Path,",
                "    repo_root: Path | None = None,",
                ") -> None:",
                "    (output_path.parent / 'fake_ingress_merge_parity.json').write_text(",
                "        'ok',",
                "        encoding='utf-8',",
                "    )",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return _load_module("gabion_policy_check_perf_helpers", helper_path)


# gabion:evidence E:function_site::tests/gabion/tooling/runtime_policy/test_policy_check_output.py::tests.gabion.tooling.runtime_policy.test_policy_check_output.test_policy_check_output_carries_projection_fiber_semantics_on_pass::policy_check.py::scripts.policy.policy_check.main
# gabion:behavior primary=desired
def test_policy_check_output_carries_projection_fiber_semantics_on_pass(
    tmp_path: Path,
    monkeypatch: object,
) -> None:
    repo_root = write_minimal_invariant_repo(tmp_path)
    output = tmp_path / "policy_check_result.json"
    monkeypatch.setattr(policy_check, "check_policy_dsl", lambda: None)
    monkeypatch.setattr(
        policy_check,
        "collect_aspf_lattice_convergence_result",
        lambda: policy_check.ProjectionFiberLatticeConvergenceResult(
            decision_rule_id="projection_fiber.convergence.ok",
            decision_outcome="pass",
            decision_severity="info",
            decision_message="projection fiber clean",
            report_payload={
                "semantic_rows": [
                    {
                        "structural_identity": "row-1",
                        "surface": "projection_fiber",
                    }
                ],
                "compiled_projection_semantic_bundles": [
                    {
                        "spec_name": "projection_fiber_frontier",
                        "bindings": [{"quotient_face": "projection_fiber.frontier"}],
                        "compiled_shacl_plans": [],
                        "compiled_sparql_plans": [],
                    }
                ],
            },
            error_messages=(),
        ),
    )

    result = policy_check.main(
        [
            "--policy-dsl",
            "--output",
            str(output),
        ],
        repo_root=repo_root,
        invariant_declared_registries=_CONNECTIVITY_SYNERGY_WITH_PSF_STUB_DECLARED_REGISTRIES,
    )

    assert result == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["rule_id"] == "policy_check"
    assert payload["status"] == "pass"
    semantics = payload["projection_fiber_semantics"]
    assert semantics["decision"]["rule_id"] == "projection_fiber.convergence.ok"
    assert semantics["report"]["compiled_projection_semantic_bundles"][0]["spec_name"] == (
        "projection_fiber_frontier"
    )
    queue_payload = json.loads(
        (tmp_path / "projection_semantic_fragment_queue.json").read_text(
            encoding="utf-8"
        )
    )
    invariant_graph_payload = json.loads(
        (tmp_path / "invariant_graph.json").read_text(encoding="utf-8")
    )
    invariant_workstreams_payload = json.loads(
        (tmp_path / "invariant_workstreams.json").read_text(encoding="utf-8")
    )
    invariant_ledger_payload = json.loads(
        (tmp_path / "invariant_ledger_projections.json").read_text(encoding="utf-8")
    )
    ingress_merge_parity_payload = json.loads(
        (repo_root / "artifacts" / "out" / "ingress_merge_parity.json").read_text(
            encoding="utf-8"
        )
    )
    git_state_payload = json.loads(
        (tmp_path / "git_state.json").read_text(encoding="utf-8")
    )
    cross_origin_witness_payload = json.loads(
        (tmp_path / "cross_origin_witness_contract.json").read_text(encoding="utf-8")
    )
    kernel_vm_alignment_payload = json.loads(
        (tmp_path / "kernel_vm_alignment.json").read_text(encoding="utf-8")
    )
    identity_grammar_completion_payload = json.loads(
        (tmp_path / "identity_grammar_completion.json").read_text(encoding="utf-8")
    )
    assert queue_payload["source_artifact"] == str(output)
    assert queue_payload["current_state"]["decision"]["rule_id"] == (
        "projection_fiber.convergence.ok"
    )
    phase5_structure = queue_payload["phase5_structure"]
    phase5_item = next(
        item for item in queue_payload["items"] if item["queue_id"] == "PSF-007"
    )
    if phase5_item["status"] == "in_progress":
        current_frontier = phase5_structure["current_frontier"]
        assert current_frontier is not None
        assert phase5_item["planning_chain"] is not None
        assert current_frontier["object_id"] in phase5_item["planning_chain"]["observed_state"]
    assert invariant_graph_payload["format_version"] == 1
    assert invariant_workstreams_payload["format_version"] == 1
    assert invariant_ledger_payload["format_version"] == 1
    assert ingress_merge_parity_payload["artifact_kind"] == "ingress_merge_parity"
    assert "cases" in ingress_merge_parity_payload
    assert any(
        item["case_key"] == "frontmatter_adapter_projection_parity"
        for item in ingress_merge_parity_payload["cases"]
        if isinstance(item, dict)
    )
    assert git_state_payload["artifact_kind"] == "git_state"
    assert "summary" in git_state_payload
    assert "entries" in git_state_payload
    assert cross_origin_witness_payload["artifact_kind"] == "cross_origin_witness_contract"
    assert "cases" in cross_origin_witness_payload
    assert "witness_rows" in cross_origin_witness_payload
    assert any(
        item["case_key"] == "analysis_union_path_remap"
        for item in cross_origin_witness_payload["cases"]
        if isinstance(item, dict)
    )
    assert kernel_vm_alignment_payload["artifact_kind"] == "kernel_vm_alignment"
    assert "bindings" in kernel_vm_alignment_payload
    assert "residues" in kernel_vm_alignment_payload
    assert (
        identity_grammar_completion_payload["artifact_kind"]
        == "identity_grammar_completion"
    )
    assert "surfaces" in identity_grammar_completion_payload
    assert "residues" in identity_grammar_completion_payload
    assert invariant_workstreams_payload["counts"]["workstream_count"] >= 1
    assert "diagnostic_summary" in invariant_workstreams_payload
    assert "planning_chart_summary" in invariant_workstreams_payload
    assert "repo_next_actions" in invariant_workstreams_payload
    assert "diagnostic_count" in invariant_workstreams_payload["counts"]
    assert "workspace_preservation_count" in invariant_workstreams_payload["diagnostic_summary"]
    assert "orphaned_workspace_change_count" in invariant_workstreams_payload["diagnostic_summary"]
    assert "phases" in invariant_workstreams_payload["planning_chart_summary"]
    assert "dominant_followup_class" in invariant_workstreams_payload["repo_next_actions"]
    assert "next_human_followup_family" in invariant_workstreams_payload["repo_next_actions"]
    assert "recommended_followup" in invariant_workstreams_payload["repo_next_actions"]
    assert "recommended_code_followup" in invariant_workstreams_payload["repo_next_actions"]
    assert "recommended_human_followup" in invariant_workstreams_payload["repo_next_actions"]
    assert "recommended_queue" in invariant_workstreams_payload["repo_next_actions"]
    assert "recommended_code_queue" in invariant_workstreams_payload["repo_next_actions"]
    assert "recommended_human_queue" in invariant_workstreams_payload["repo_next_actions"]
    assert "ranked_followups" in invariant_workstreams_payload["repo_next_actions"]
    assert "queues" in invariant_workstreams_payload["repo_next_actions"]
    assert "followup_lanes" in invariant_workstreams_payload["repo_next_actions"]
    assert "diagnostic_lanes" in invariant_workstreams_payload["repo_next_actions"]
    assert (
        "recommended_workspace_commit_unit"
        in invariant_workstreams_payload["repo_next_actions"]
    )
    assert "workspace_commit_units" in invariant_workstreams_payload["repo_next_actions"]
    assert invariant_ledger_payload["counts"]["ledger_count"] >= 1
    for workstream in invariant_workstreams_payload["workstreams"]:
        assert "next_actions" in workstream
        assert "health_summary" in workstream
        assert "doc_alignment_summary" in workstream
        assert "doc_ids" in workstream
        assert "policy_ids" in workstream
        assert "failing_test_case_count" in workstream
        assert "test_failure_count" in workstream
        assert "dominant_blocker_class" in workstream["next_actions"]
        assert "recommended_remediation_family" in workstream["next_actions"]
        assert "dominant_doc_alignment_status" in workstream["next_actions"]
        assert "recommended_doc_alignment_action" in workstream["next_actions"]
        assert "next_human_followup_family" in workstream["next_actions"]
        assert "recommended_doc_followup_target_doc_id" in workstream["next_actions"]
        assert "misaligned_target_doc_ids" in workstream["next_actions"]
        assert "recommended_followup" in workstream["next_actions"]
        assert "documentation_followup_lanes" in workstream["next_actions"]
        assert "ranked_followups" in workstream["next_actions"]
        assert "remediation_lanes" in workstream["next_actions"]
    for ledger in invariant_ledger_payload["ledgers"]:
        assert "target_doc_ids" in ledger
        assert "recommended_ledger_action" in ledger
        assert "summary" in ledger
        assert "current_snapshot" in ledger
        assert "target_doc_alignments" in ledger
        assert "alignment_summary" in ledger
    assert (tmp_path / "projection_semantic_fragment_queue.md").exists()


# gabion:evidence E:function_site::tests/gabion/tooling/runtime_policy/test_policy_check_output.py::tests.gabion.tooling.runtime_policy.test_policy_check_output.test_policy_check_output_carries_projection_fiber_semantics_on_block::policy_check.py::scripts.policy.policy_check.main
# gabion:behavior primary=desired
def test_policy_check_output_carries_projection_fiber_semantics_on_block(
    tmp_path: Path,
    monkeypatch: object,
) -> None:
    repo_root = write_minimal_invariant_repo(tmp_path)
    output = tmp_path / "policy_check_result.json"
    monkeypatch.setattr(policy_check, "check_policy_dsl", lambda: None)
    monkeypatch.setattr(
        policy_check,
        "collect_aspf_lattice_convergence_result",
        lambda: policy_check.ProjectionFiberLatticeConvergenceResult(
            decision_rule_id="projection_fiber.convergence.blocking",
            decision_outcome="block",
            decision_severity="error",
            decision_message="projection fiber blocked",
            report_payload={
                "semantic_rows": [],
                "compiled_projection_semantic_bundles": [],
            },
            error_messages=("frontier witness incomplete",),
        ),
    )

    result = policy_check.main(
        [
            "--policy-dsl",
            "--output",
            str(output),
        ],
        repo_root=repo_root,
        invariant_declared_registries=_CONNECTIVITY_SYNERGY_WITH_PSF_STUB_DECLARED_REGISTRIES,
    )

    assert result == 2
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["rule_id"] == "policy_check"
    assert payload["status"] == "fail"
    assert payload["violations"][0]["message"] == "projection fiber blocked"
    semantics = payload["projection_fiber_semantics"]
    assert semantics["decision"]["outcome"] == "block"
    assert semantics["error_messages"] == ["frontier witness incomplete"]


# gabion:behavior primary=desired
def test_policy_check_workflows_output_emits_invariant_graph_artifact(
    tmp_path: Path,
    monkeypatch: object,
) -> None:
    output = tmp_path / "policy_check_result.json"
    repo_root = write_minimal_invariant_repo(tmp_path)
    monkeypatch.setattr(
        policy_check,
        "_write_projection_semantic_fragment_queue_artifacts",
        lambda *, output_path, phase5_workstreams_projection=None: None,
    )
    monkeypatch.setattr(policy_check, "check_workflows", lambda: None)
    monkeypatch.setattr(policy_check, "check_aspf_taint_crosswalk_ack", lambda: None)
    monkeypatch.setattr(policy_check, "check_policy_dsl", lambda: None)
    monkeypatch.setattr(
        policy_check,
        "collect_aspf_lattice_convergence_result",
        lambda: policy_check.ProjectionFiberLatticeConvergenceResult(
            decision_rule_id="projection_fiber.convergence.ok",
            decision_outcome="pass",
            decision_severity="info",
            decision_message="projection fiber clean",
            report_payload={
                "semantic_rows": [],
                "compiled_projection_semantic_bundles": [],
            },
            error_messages=(),
        ),
    )

    result = policy_check.main(
        [
            "--workflows",
            "--output",
            str(output),
        ],
        repo_root=repo_root,
        invariant_declared_registries=_SYNTHETIC_CONNECTIVITY_DECLARED_REGISTRIES,
    )

    assert result == 0
    assert (tmp_path / "invariant_graph.json").exists()
    assert (tmp_path / "invariant_workstreams.json").exists()
    assert (tmp_path / "invariant_ledger_projections.json").exists()
    invariant_graph_payload = json.loads(
        (tmp_path / "invariant_graph.json").read_text(encoding="utf-8")
    )
    invariant_workstreams_payload = json.loads(
        (tmp_path / "invariant_workstreams.json").read_text(encoding="utf-8")
    )
    assert invariant_graph_payload["workstream_root_ids"] == [
        "CSA-IDR",
        "CSA-IGM",
        "CSA-IVL",
        "CSA-RGC",
        "SCC",
    ]
    recommended_followup = invariant_workstreams_payload["repo_next_actions"][
        "recommended_followup"
    ]
    assert recommended_followup["followup_family"] == "coverage_gap"
    assert recommended_followup["queue_id"].startswith(
        "planner_queue|followup_family=coverage_gap|"
    )
    assert recommended_followup["owner_root_object_id"] in {
        "CSA-IDR",
        "CSA-IGM",
        "CSA-IVL",
        "CSA-RGC",
    }
    recommended_queue = invariant_workstreams_payload["repo_next_actions"][
        "recommended_queue"
    ]
    assert recommended_queue["queue_id"] == recommended_followup["queue_id"]
    assert recommended_queue["selection_scope_kind"] == "mixed_root_followup_family"
    assert {
        item["owner_root_object_id"]
        for item in recommended_followup["cofrontier_followup_cohort"]
    } == {"CSA-IDR", "CSA-IGM", "CSA-IVL", "CSA-RGC", "SCC"}
    assert invariant_workstreams_payload["repo_next_actions"][
        "recommended_followup_lane"
    ]["root_object_ids"] == ["CSA-IDR", "CSA-IGM", "CSA-IVL", "CSA-RGC", "SCC"]


# gabion:evidence E:function_site::tests/gabion/tooling/runtime_policy/test_policy_check_output.py::tests.gabion.tooling.runtime_policy.test_policy_check_output.test_policy_check_workflows_requires_output_to_emit_invariant_artifacts::policy_check.py::scripts.policy.policy_check.main
# gabion:behavior primary=desired
def test_policy_check_workflows_requires_output_to_emit_invariant_artifacts(
    tmp_path: Path,
    monkeypatch: object,
) -> None:
    repo_root = write_minimal_invariant_repo(tmp_path)
    monkeypatch.setattr(
        policy_check,
        "_write_projection_semantic_fragment_queue_artifacts",
        lambda *, output_path, phase5_workstreams_projection=None: None,
    )
    monkeypatch.setattr(policy_check, "check_workflows", lambda: None)
    monkeypatch.setattr(policy_check, "check_aspf_taint_crosswalk_ack", lambda: None)
    monkeypatch.setattr(policy_check, "check_policy_dsl", lambda: None)
    monkeypatch.setattr(
        policy_check,
        "collect_aspf_lattice_convergence_result",
        lambda: policy_check.ProjectionFiberLatticeConvergenceResult(
            decision_rule_id="projection_fiber.convergence.ok",
            decision_outcome="pass",
            decision_severity="info",
            decision_message="projection fiber clean",
            report_payload={
                "semantic_rows": [],
                "compiled_projection_semantic_bundles": [],
            },
            error_messages=(),
        ),
    )

    result = policy_check.main(
        ["--workflows"],
        repo_root=repo_root,
        invariant_declared_registries=_SYNTHETIC_CONNECTIVITY_DECLARED_REGISTRIES,
    )

    assert result == 0
    assert not (tmp_path / "policy_check_result.json").exists()
    assert not (tmp_path / "invariant_graph.json").exists()
    assert not (tmp_path / "invariant_workstreams.json").exists()
    assert not (tmp_path / "invariant_ledger_projections.json").exists()


# gabion:behavior primary=desired
def test_policy_check_workflows_emits_perf_artifact_when_requested(
    tmp_path: Path,
    monkeypatch: object,
) -> None:
    output = tmp_path / "policy_check_result.json"
    perf_artifact = tmp_path / "policy_check_workflows_perf.json"
    repo_root = write_minimal_invariant_repo(tmp_path)
    perf_helpers = _write_policy_check_perf_helpers(repo_root)
    monkeypatch.setattr(
        policy_check,
        "_write_projection_semantic_fragment_queue_artifacts",
        lambda *, output_path, phase5_workstreams_projection=None: None,
    )
    monkeypatch.setattr(policy_check, "check_workflows", lambda: None)
    monkeypatch.setattr(policy_check, "check_aspf_taint_crosswalk_ack", lambda: None)
    monkeypatch.setattr(policy_check, "check_policy_dsl", lambda: None)
    monkeypatch.setattr(
        policy_check,
        "collect_aspf_lattice_convergence_result",
        lambda: policy_check.ProjectionFiberLatticeConvergenceResult(
            decision_rule_id="projection_fiber.convergence.ok",
            decision_outcome="pass",
            decision_severity="info",
            decision_message="projection fiber clean",
            report_payload={
                "semantic_rows": [],
                "compiled_projection_semantic_bundles": [],
            },
            error_messages=(),
        ),
    )
    monkeypatch.setattr(
        policy_check,
        "_write_git_state_artifact",
        perf_helpers.write_git_state_artifact,
    )
    monkeypatch.setattr(
        policy_check,
        "_write_cross_origin_witness_contract_artifact",
        perf_helpers.write_cross_origin_witness_contract_artifact,
    )
    monkeypatch.setattr(
        policy_check,
        "_write_kernel_vm_alignment_artifact",
        perf_helpers.write_kernel_vm_alignment_artifact,
    )
    monkeypatch.setattr(
        policy_check,
        "_write_identity_grammar_completion_artifact",
        perf_helpers.write_identity_grammar_completion_artifact,
    )
    monkeypatch.setattr(
        policy_check,
        "_write_ingress_merge_parity_artifact",
        perf_helpers.write_ingress_merge_parity_artifact,
    )

    result = policy_check.main(
        [
            "--workflows",
            "--output",
            str(output),
            "--perf-artifact",
            str(perf_artifact),
        ],
        repo_root=repo_root,
        invariant_declared_registries=_SYNTHETIC_CONNECTIVITY_DECLARED_REGISTRIES,
    )

    assert result == 0
    perf_payload = json.loads(perf_artifact.read_text(encoding="utf-8"))
    assert perf_payload["profiles"][0]["profiler"] == "cProfile"
    assert perf_payload["requested_checks"] == ["workflows"]
    assert perf_payload["command"][0] == "scripts/policy/policy_check.py"
    assert perf_payload["command"][-2:] == ["--perf-artifact", str(perf_artifact)]
    assert perf_payload["observation_count"] >= 1
    sample = next(
        item
        for item in perf_payload["profiles"][0]["samples"]
        if "artifact_node" in item
    )
    assert sample["artifact_node"]["wire"]
    assert sample["artifact_node"]["site_identity"]
    assert sample["artifact_node"]["structural_identity"]


# gabion:behavior primary=desired
def test_policy_check_perf_artifact_includes_output_phase_writers(
    tmp_path: Path,
    monkeypatch: object,
) -> None:
    output = tmp_path / "policy_check_result.json"
    perf_artifact = tmp_path / "policy_check_workflows_perf.json"
    repo_root = write_minimal_invariant_repo(tmp_path)
    perf_helpers = _write_policy_check_perf_helpers(repo_root)

    monkeypatch.setattr(policy_check, "check_workflows", lambda: None)
    monkeypatch.setattr(policy_check, "check_aspf_taint_crosswalk_ack", lambda: None)
    monkeypatch.setattr(policy_check, "check_policy_dsl", lambda: None)
    monkeypatch.setattr(
        policy_check,
        "collect_aspf_lattice_convergence_result",
        lambda: policy_check.ProjectionFiberLatticeConvergenceResult(
            decision_rule_id="projection_fiber.convergence.ok",
            decision_outcome="pass",
            decision_severity="info",
            decision_message="projection fiber clean",
            report_payload={
                "semantic_rows": [],
                "compiled_projection_semantic_bundles": [],
            },
            error_messages=(),
        ),
    )
    monkeypatch.setattr(
        policy_check,
        "_write_projection_semantic_fragment_queue_artifacts",
        lambda *, output_path, phase5_workstreams_projection=None: None,
    )
    monkeypatch.setattr(
        policy_check,
        "_write_git_state_artifact",
        perf_helpers.write_git_state_artifact,
    )
    monkeypatch.setattr(
        policy_check,
        "_write_cross_origin_witness_contract_artifact",
        perf_helpers.write_cross_origin_witness_contract_artifact,
    )
    monkeypatch.setattr(
        policy_check,
        "_write_kernel_vm_alignment_artifact",
        perf_helpers.write_kernel_vm_alignment_artifact,
    )
    monkeypatch.setattr(
        policy_check,
        "_write_identity_grammar_completion_artifact",
        perf_helpers.write_identity_grammar_completion_artifact,
    )
    monkeypatch.setattr(
        policy_check,
        "_write_ingress_merge_parity_artifact",
        perf_helpers.write_ingress_merge_parity_artifact,
    )

    result = policy_check.main(
        [
            "--workflows",
            "--output",
            str(output),
            "--perf-artifact",
            str(perf_artifact),
        ],
        repo_root=repo_root,
        invariant_declared_registries=_SYNTHETIC_CONNECTIVITY_DECLARED_REGISTRIES,
    )

    assert result == 0
    perf_payload = json.loads(perf_artifact.read_text(encoding="utf-8"))
    profiled_output_writer = next(
        item
        for item in perf_payload["profiles"][0]["samples"]
        if item.get("artifact_node", {}).get("rel_path")
        == "src/gabion/policy_check_perf_helpers.py"
        and item.get("artifact_node", {}).get("qualname")
        == "write_git_state_artifact"
    )
    assert profiled_output_writer["inclusive_value"] > 0


# gabion:evidence E:function_site::tests/gabion/tooling/runtime_policy/test_policy_check_output.py::tests.gabion.tooling.runtime_policy.test_policy_check_output.test_policy_check_passes_in_memory_workstreams_projection_to_queue_writer::policy_check.py::scripts.policy.policy_check.main
# gabion:behavior primary=desired
def test_policy_check_passes_in_memory_workstreams_projection_to_queue_writer(
    tmp_path: Path,
    monkeypatch: object,
) -> None:
    repo_root = write_minimal_invariant_repo(tmp_path)
    output = tmp_path / "policy_check_result.json"
    phase5_workstreams_projection = {
        "format_version": 1,
        "generated_at_utc": "2026-03-13T00:00:00+00:00",
        "root": str(tmp_path),
        "workstreams": [{"object_id": "PSF-007", "title": "Synthetic Phase 5"}],
        "counts": {"workstream_count": 1},
        "repo_next_actions": {},
        "diagnostic_summary": {},
    }
    captured: dict[str, object] = {}
    monkeypatch.setattr(policy_check, "check_workflows", lambda: None)
    monkeypatch.setattr(policy_check, "check_aspf_taint_crosswalk_ack", lambda: None)
    monkeypatch.setattr(policy_check, "check_policy_dsl", lambda: None)
    monkeypatch.setattr(
        policy_check,
        "collect_aspf_lattice_convergence_result",
        lambda: policy_check.ProjectionFiberLatticeConvergenceResult(
            decision_rule_id="projection_fiber.convergence.ok",
            decision_outcome="pass",
            decision_severity="info",
            decision_message="projection fiber clean",
            report_payload={
                "semantic_rows": [],
                "compiled_projection_semantic_bundles": [],
            },
            error_messages=(),
        ),
    )
    monkeypatch.setattr(
        policy_check,
        "_write_invariant_graph_artifact",
        lambda *, output_path, repo_root=None, invariant_declared_registries=None: (
            phase5_workstreams_projection
        ),
    )
    monkeypatch.setattr(
        policy_check,
        "_write_projection_semantic_fragment_queue_artifacts",
        lambda *, output_path, phase5_workstreams_projection=None: captured.update(
            {
                "output_path": output_path,
                "phase5_workstreams_projection": phase5_workstreams_projection,
            }
        ),
    )

    result = policy_check.main(
        [
            "--workflows",
            "--output",
            str(output),
        ],
        repo_root=repo_root,
        invariant_declared_registries=_SYNTHETIC_CONNECTIVITY_DECLARED_REGISTRIES,
    )

    assert result == 0
    assert captured["output_path"] == output
    assert captured["phase5_workstreams_projection"] is phase5_workstreams_projection

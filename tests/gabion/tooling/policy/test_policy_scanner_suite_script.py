from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from scripts.policy import policy_scanner_suite


# gabion:evidence E:function_site::test_policy_scanner_suite_script.py::tests.gabion.tooling.policy.test_policy_scanner_suite_script.test_main_requires_explicit_out
# gabion:behavior primary=desired
def test_main_requires_explicit_out() -> None:
    with pytest.raises(SystemExit) as excinfo:
        policy_scanner_suite.main([])
    assert excinfo.value.code == 2


def test_policy_result_status_and_projection_fiber_semantics_normalize_boundary_payload() -> None:
    policy_check_payload = {
        "rule_id": "policy_check",
        "status": "pass",
        "projection_fiber_semantics": {
            "decision": {"rule_id": "projection_fiber.convergence.ok"},
            "report": {
                "semantic_rows": [],
                "compiled_projection_semantic_bundles": [],
            },
        },
    }
    assert policy_scanner_suite._policy_result_status(policy_check_payload) == "pass"
    assert policy_scanner_suite._policy_result_status(
        {"rule_id": "custom_rule", "status": "skip"}
    ) == "skip"
    assert policy_scanner_suite._policy_result_status({"rule_id": "broken_rule"}) is None
    assert policy_scanner_suite._policy_check_projection_fiber_semantics(
        policy_check_payload
    ) == {
        "decision": {"rule_id": "projection_fiber.convergence.ok"},
        "report": {
            "semantic_rows": [],
            "compiled_projection_semantic_bundles": [],
        },
    }
    assert (
        policy_scanner_suite._policy_check_projection_fiber_semantics(
            {"rule_id": "custom_rule", "projection_fiber_semantics": {"ignored": True}}
        )
        is None
    )


# gabion:evidence E:function_site::test_policy_scanner_suite_script.py::tests.gabion.tooling.policy.test_policy_scanner_suite_script.test_run_skips_semantic_queue_backfill_without_policy_check_owned_artifact
# gabion:behavior primary=desired
def test_run_skips_semantic_queue_backfill_without_policy_check_owned_artifact(
    tmp_path: Path,
    monkeypatch: object,
) -> None:
    root = tmp_path
    out = root / "artifacts/out/policy_suite_results.json"
    source = root / "src/gabion/branch_sample.py"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text(
        "def branchy(flag: bool) -> int:\n    if flag:\n        return 1\n    return 0\n",
        encoding="utf-8",
    )

    policy_check_payload = policy_scanner_suite.policy_result_schema.make_policy_result(
        rule_id="policy_check",
        status="pass",
        violations=[],
        baseline_mode="current_only",
        source_tool="tests.gabion.tooling.policy.test_policy_scanner_suite_script",
        input_scope={"root": str(root)},
    )
    policy_check_payload["projection_fiber_semantics"] = {
        "decision": {
            "rule_id": "projection_fiber.convergence.ok",
            "outcome": "pass",
            "severity": "info",
        },
        "report": {
            "semantic_rows": [
                {
                    "structural_identity": "row-1",
                    "obligation_state": "discharged",
                    "payload": {
                        "path": "src/gabion/branch_sample.py",
                        "qualname": "branch_sample.branchy",
                        "structural_path": "branch_sample.branchy::branch[0]",
                        "complete": True,
                    },
                }
            ],
            "compiled_projection_semantic_bundles": [
                {
                    "spec_name": "projection_fiber_frontier",
                    "bindings": [
                        {
                            "quotient_face": "projection_fiber.frontier",
                            "source_structural_identity": "row-1",
                        }
                    ],
                }
            ],
        },
    }

    def _fake_external_child_inputs(
        *, root: Path, out: Path
    ) -> policy_scanner_suite.ExternalChildInputs:
        return policy_scanner_suite.ExternalChildInputs(
            child_statuses={"policy_check": "pass"},
            runtime_child_inputs=policy_scanner_suite.runtime_policy_scanner_suite.PolicySuiteChildInputs(
                projection_fiber_semantics=dict(
                    policy_check_payload["projection_fiber_semantics"]
                ),
            ),
        )

    monkeypatch.setattr(
        policy_scanner_suite,
        "_resolve_external_child_inputs",
        _fake_external_child_inputs,
    )

    rc = policy_scanner_suite.run(root=root, out=out)

    assert rc == 1
    assert out.exists()
    assert (out.parent / "hotspot_neighborhood_queue.json").exists()
    assert (out.parent / "hotspot_neighborhood_queue.md").exists()
    assert not (out.parent / "projection_semantic_fragment_queue.json").exists()
    assert not (out.parent / "projection_semantic_fragment_queue.md").exists()
    hotspot_payload = json.loads(
        (out.parent / "hotspot_neighborhood_queue.json").read_text(encoding="utf-8")
    )
    assert "projection_fiber_decision" in hotspot_payload["source"]
    assert "projection_fiber_semantic_previews" in hotspot_payload["source"]


# gabion:evidence E:function_site::test_policy_scanner_suite_script.py::tests.gabion.tooling.policy.test_policy_scanner_suite_script.test_run_preserves_policy_check_owned_semantic_queue
# gabion:behavior primary=desired
def test_run_preserves_policy_check_owned_semantic_queue(
    tmp_path: Path,
    monkeypatch: object,
) -> None:
    root = tmp_path
    out = root / "artifacts/out/policy_suite_results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    queue_json = out.parent / "projection_semantic_fragment_queue.json"
    queue_md = out.parent / "projection_semantic_fragment_queue.md"
    policy_check_result = out.parent / "policy_check_result.json"

    policy_check_payload = {
        "rule_id": "policy_check",
        "status": "pass",
        "violations": [],
        "projection_fiber_semantics": {
            "decision": {
                "rule_id": "projection_fiber.convergence.ok",
                "outcome": "pass",
                "severity": "info",
            },
            "report": {
                "semantic_rows": [
                    {
                        "structural_identity": "row-1",
                        "obligation_state": "discharged",
                        "payload": {
                            "path": "src/gabion/example.py",
                            "qualname": "example.frontier",
                            "structural_path": "example.frontier::branch[0]",
                            "complete": True,
                        },
                    }
                ],
                "compiled_projection_semantic_bundles": [
                    {
                        "spec_name": "projection_fiber_frontier",
                        "bindings": [
                            {
                                "quotient_face": "projection_fiber.frontier",
                                "source_structural_identity": "row-1",
                            }
                        ],
                    }
                ],
            },
        },
    }

    def _fake_external_child_inputs(
        *, root: Path, out: Path
    ) -> policy_scanner_suite.ExternalChildInputs:
        policy_check_result.write_text(
            json.dumps(policy_check_payload, indent=2) + "\n",
            encoding="utf-8",
        )
        queue_json.write_text(
            json.dumps(
                {
                    "format_version": 1,
                    "source_artifact": str(policy_check_result),
                    "current_state": {
                        "decision": {
                            "rule_id": "projection_fiber.convergence.ok",
                        },
                        "semantic_row_count": 1,
                        "compiled_projection_semantic_bundle_count": 1,
                        "compiled_projection_semantic_spec_names": [
                            "projection_fiber_frontier"
                        ],
                        "semantic_preview_count": 1,
                        "semantic_previews": [
                            {
                                "spec_name": "projection_fiber_frontier",
                                "quotient_face": "projection_fiber.frontier",
                                "path": "src/gabion/example.py",
                                "qualname": "example.frontier",
                                "structural_path": "example.frontier::branch[0]",
                            }
                        ],
                    },
                    "next_queue_ids": ["PSF-004", "PSF-005", "PSF-006", "PSF-007"],
                    "items": [],
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        queue_md.write_text("# Projection Semantic Fragment Queue\n", encoding="utf-8")
        return policy_scanner_suite.ExternalChildInputs(
            child_statuses={"policy_check": "pass"},
            runtime_child_inputs=policy_scanner_suite.runtime_policy_scanner_suite.PolicySuiteChildInputs(
                projection_fiber_semantics=dict(
                    policy_check_payload["projection_fiber_semantics"]
                ),
            ),
        )

    monkeypatch.setattr(
        policy_scanner_suite,
        "_resolve_external_child_inputs",
        _fake_external_child_inputs,
    )

    rc = policy_scanner_suite.run(root=root, out=out)

    assert rc == 0
    queue_payload = json.loads(queue_json.read_text(encoding="utf-8"))
    assert queue_payload["source_artifact"] == str(policy_check_result)
    assert (out.parent / "hotspot_neighborhood_queue.json").exists()


# gabion:evidence E:function_site::test_policy_scanner_suite_script.py::tests.gabion.tooling.policy.test_policy_scanner_suite_script.test_run_does_not_regenerate_missing_policy_check_owned_semantic_queue
# gabion:behavior primary=desired
def test_run_does_not_regenerate_missing_policy_check_owned_semantic_queue(
    tmp_path: Path,
    monkeypatch: object,
) -> None:
    root = tmp_path
    out = root / "artifacts/out/policy_suite_results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    policy_check_result = out.parent / "policy_check_result.json"

    policy_check_payload = {
        "rule_id": "policy_check",
        "status": "pass",
        "violations": [],
        "projection_fiber_semantics": {
            "decision": {
                "rule_id": "projection_fiber.convergence.ok",
                "outcome": "pass",
                "severity": "info",
            },
            "report": {
                "semantic_rows": [],
                "compiled_projection_semantic_bundles": [],
            },
        },
    }

    def _fake_external_child_inputs(
        *, root: Path, out: Path
    ) -> policy_scanner_suite.ExternalChildInputs:
        policy_check_result.write_text(
            json.dumps(policy_check_payload, indent=2) + "\n",
            encoding="utf-8",
        )
        return policy_scanner_suite.ExternalChildInputs(
            child_statuses={"policy_check": "pass"},
            runtime_child_inputs=policy_scanner_suite.runtime_policy_scanner_suite.PolicySuiteChildInputs(
                projection_fiber_semantics=dict(
                    policy_check_payload["projection_fiber_semantics"]
                ),
            ),
        )

    monkeypatch.setattr(
        policy_scanner_suite,
        "_resolve_external_child_inputs",
        _fake_external_child_inputs,
    )

    rc = policy_scanner_suite.run(root=root, out=out)

    assert rc == 0
    assert not (out.parent / "projection_semantic_fragment_queue.json").exists()
    assert not (out.parent / "projection_semantic_fragment_queue.md").exists()


def test_run_passes_in_memory_payload_to_hotspot_queue(
    tmp_path: Path,
    monkeypatch: object,
) -> None:
    root = tmp_path
    out = root / "artifacts/out/policy_suite_results.json"
    captured: dict[str, object] = {}

    policy_check_payload = policy_scanner_suite.policy_result_schema.make_policy_result(
        rule_id="policy_check",
        status="pass",
        violations=[],
        baseline_mode="current_only",
        source_tool="tests.gabion.tooling.policy.test_policy_scanner_suite_script",
        input_scope={"root": str(root)},
    )

    def _fake_external_child_inputs(
        *, root: Path, out: Path
    ) -> policy_scanner_suite.ExternalChildInputs:
        return policy_scanner_suite.ExternalChildInputs(
            child_statuses={"policy_check": "pass"},
            runtime_child_inputs=policy_scanner_suite.runtime_policy_scanner_suite.PolicySuiteChildInputs(
                projection_fiber_semantics=None,
            ),
        )

    def _fake_run_from_payload(
        *,
        payload: dict[str, object],
        out_path: Path,
        markdown_out: Path | None = None,
        config: object | None = None,
    ) -> int:
        captured["payload"] = payload
        captured["out_path"] = out_path
        captured["markdown_out"] = markdown_out
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(
                {
                    "format_version": 1,
                    "generated_at_utc": "2026-03-11T00:00:00Z",
                    "source": {
                        "projection_fiber_decision": {},
                        "projection_fiber_semantic_bundle_count": 0,
                        "projection_fiber_semantic_preview_count": 0,
                        "projection_fiber_semantic_previews": [],
                    },
                    "counts": {"neighborhood_count": 0, "large_zone_backlog_count": 0},
                    "neighborhoods": [],
                    "large_zone_backlog": [],
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        if markdown_out is not None:
            markdown_out.write_text("# Hotspot Neighborhood Queue\n", encoding="utf-8")
        return 0

    monkeypatch.setattr(
        policy_scanner_suite,
        "_resolve_external_child_inputs",
        _fake_external_child_inputs,
    )
    monkeypatch.setattr(
        policy_scanner_suite.hotspot_neighborhood_queue,
        "run_from_payload",
        _fake_run_from_payload,
    )

    rc = policy_scanner_suite.run(root=root, out=out)

    assert rc == 0
    payload = captured["payload"]
    assert isinstance(payload, dict)
    assert "counts" not in payload
    assert "policy_results" not in payload
    assert "projection_fiber_semantics" not in payload
    assert captured["out_path"] == out.parent / "hotspot_neighborhood_queue.json"
    assert captured["markdown_out"] == out.parent / "hotspot_neighborhood_queue.md"


# gabion:evidence E:function_site::test_policy_scanner_suite_script.py::tests.gabion.tooling.policy.test_policy_scanner_suite_script.test_resolve_external_child_inputs_preserve_preexisting_child_artifacts
# gabion:behavior primary=desired
def test_resolve_external_child_inputs_preserve_preexisting_child_artifacts(
    tmp_path: Path,
    monkeypatch: object,
) -> None:
    root = tmp_path
    out = root / "artifacts/out/policy_suite_results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    policy_check_result = out.parent / "policy_check_result.json"
    structural_hash_result = out.parent / "structural_hash_result.json"
    deprecated_result = out.parent / "deprecated_nonerasability_result.json"

    policy_scanner_suite.policy_result_schema.write_policy_result(
        path=policy_check_result,
        result=policy_scanner_suite.policy_result_schema.make_policy_result(
            rule_id="policy_check",
            status="pass",
            violations=[],
            baseline_mode="current_only",
            source_tool="tests.gabion.tooling.policy.test_policy_scanner_suite_script",
            input_scope={"root": str(root), "mode": "preserved"},
        ),
    )
    policy_scanner_suite.policy_result_schema.write_policy_result(
        path=structural_hash_result,
        result=policy_scanner_suite.policy_result_schema.make_policy_result(
            rule_id="structural_hash",
            status="pass",
            violations=[],
            baseline_mode="current_only",
            source_tool="tests.gabion.tooling.policy.test_policy_scanner_suite_script",
            input_scope={"root": str(root), "mode": "preserved"},
        ),
    )
    policy_scanner_suite.policy_result_schema.write_policy_result(
        path=deprecated_result,
        result=policy_scanner_suite.policy_result_schema.make_policy_result(
            rule_id="deprecated_nonerasability",
            status="skip",
            violations=[
                {
                    "message": "baseline/current payload missing; rule skipped by child policy check",
                    "render": "missing baseline=False current=False",
                }
            ],
            baseline_mode="baseline_compare",
            source_tool="scripts/policy/deprecated_nonerasability_policy_check.py",
            input_scope={
                "baseline": str(root / "out" / "deprecated_fibers_baseline.json"),
                "current": str(root / "out" / "deprecated_fibers_current.json"),
                "mode": "preserved",
            },
        ),
    )

    def _fake_run(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        raise AssertionError("wrapper must preserve preexisting child-owned artifacts")

    monkeypatch.setattr(policy_scanner_suite.subprocess, "run", _fake_run)

    child_inputs = policy_scanner_suite._resolve_external_child_inputs(root=root, out=out)

    assert child_inputs.child_statuses == {
        "policy_check": "pass",
        "structural_hash": "pass",
        "deprecated_nonerasability": "skip",
    }
    assert child_inputs.runtime_child_inputs.projection_fiber_semantics is None


# gabion:evidence E:function_site::test_policy_scanner_suite_script.py::tests.gabion.tooling.policy.test_policy_scanner_suite_script.test_resolve_external_child_inputs_fail_closed_when_child_artifact_missing
# gabion:behavior primary=desired
def test_resolve_external_child_inputs_fail_closed_when_child_artifact_missing(
    tmp_path: Path,
    monkeypatch: object,
) -> None:
    root = tmp_path
    out = root / "artifacts/out/policy_suite_results.json"

    monkeypatch.setattr(
        policy_scanner_suite.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args[0], returncode=1),
    )
    monkeypatch.setattr(
        policy_scanner_suite.policy_result_schema,
        "load_policy_result",
        lambda path: None,
    )

    try:
        policy_scanner_suite._resolve_external_child_inputs(root=root, out=out)
    except RuntimeError as exc:
        message = str(exc)
    else:  # pragma: no cover
        raise AssertionError("wrapper must fail closed when a child artifact is missing")

    assert "external policy result artifact missing after wrapper invocation" in message
    assert "rule_id=policy_check" in message


# gabion:evidence E:function_site::test_policy_scanner_suite_script.py::tests.gabion.tooling.policy.test_policy_scanner_suite_script.test_resolve_external_child_inputs_load_child_emitted_deprecated_skip
# gabion:behavior primary=desired
def test_resolve_external_child_inputs_load_child_emitted_deprecated_skip(
    tmp_path: Path,
    monkeypatch: object,
) -> None:
    root = tmp_path
    out = root / "artifacts/out/policy_suite_results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    policy_scanner_suite.policy_result_schema.write_policy_result(
        path=out.parent / "policy_check_result.json",
        result=policy_scanner_suite.policy_result_schema.make_policy_result(
            rule_id="policy_check",
            status="pass",
            violations=[],
            baseline_mode="current_only",
            source_tool="tests.gabion.tooling.policy.test_policy_scanner_suite_script",
            input_scope={"root": str(root)},
        ),
    )
    policy_scanner_suite.policy_result_schema.write_policy_result(
        path=out.parent / "structural_hash_result.json",
        result=policy_scanner_suite.policy_result_schema.make_policy_result(
            rule_id="structural_hash",
            status="pass",
            violations=[],
            baseline_mode="current_only",
            source_tool="tests.gabion.tooling.policy.test_policy_scanner_suite_script",
            input_scope={"root": str(root)},
        ),
    )

    def _fake_run(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        if "deprecated_nonerasability_policy_check.py" in str(args[1]):
            policy_scanner_suite.policy_result_schema.write_policy_result(
                path=out.parent / "deprecated_nonerasability_result.json",
                result=policy_scanner_suite.policy_result_schema.make_policy_result(
                    rule_id="deprecated_nonerasability",
                    status="skip",
                    violations=[
                        {
                            "message": "baseline/current payload missing; rule skipped by child policy check",
                            "render": "missing baseline=False current=False",
                        }
                    ],
                    baseline_mode="baseline_compare",
                    source_tool=(
                        "scripts/policy/deprecated_nonerasability_policy_check.py"
                    ),
                    input_scope={
                        "baseline": str(root / "out" / "deprecated_fibers_baseline.json"),
                        "current": str(root / "out" / "deprecated_fibers_current.json"),
                    },
                ),
            )
        return subprocess.CompletedProcess(args, returncode=0)

    monkeypatch.setattr(policy_scanner_suite.subprocess, "run", _fake_run)

    child_inputs = policy_scanner_suite._resolve_external_child_inputs(root=root, out=out)

    assert child_inputs.child_statuses == {
        "policy_check": "pass",
        "structural_hash": "pass",
        "deprecated_nonerasability": "skip",
    }
    assert child_inputs.runtime_child_inputs.projection_fiber_semantics is None

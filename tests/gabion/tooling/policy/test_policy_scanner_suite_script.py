from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.policy import policy_scanner_suite


# gabion:evidence E:function_site::test_policy_scanner_suite_script.py::tests.gabion.tooling.policy.test_policy_scanner_suite_script.test_main_requires_explicit_out
# gabion:behavior primary=desired
def test_main_requires_explicit_out() -> None:
    with pytest.raises(SystemExit) as excinfo:
        policy_scanner_suite.main([])
    assert excinfo.value.code == 2


# gabion:evidence E:function_site::test_policy_scanner_suite_script.py::tests.gabion.tooling.policy.test_policy_scanner_suite_script.test_run_emits_hotspot_queue_without_projection_semantic_fragment_artifacts
# gabion:behavior primary=desired
def test_run_emits_hotspot_queue_without_projection_semantic_fragment_artifacts(
    tmp_path: Path,
    monkeypatch: object,
) -> None:
    root = tmp_path
    out_dir = root / "artifacts/out"
    source = root / "src/gabion/branch_sample.py"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text(
        "def branchy(flag: bool) -> int:\n    if flag:\n        return 1\n    return 0\n",
        encoding="utf-8",
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    policy_check_payload = (
        policy_scanner_suite.hotspot_neighborhood_queue.policy_result_schema.make_policy_result(
            rule_id="policy_check",
            status="pass",
            violations=[],
            baseline_mode="current_only",
            source_tool="tests.gabion.tooling.policy.test_policy_scanner_suite_script",
            input_scope={"root": str(root)},
        )
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
    policy_scanner_suite.hotspot_neighborhood_queue.policy_result_schema.write_policy_result(
        path=out_dir / "policy_check_result.json",
        result=policy_check_payload,
    )

    rc = policy_scanner_suite.run(root=root, out_dir=out_dir)

    assert rc == 1
    assert not (out_dir / "policy_suite_results.json").exists()
    assert (out_dir / "hotspot_neighborhood_queue.json").exists()
    assert (out_dir / "hotspot_neighborhood_queue.md").exists()
    assert not (out_dir / "projection_semantic_fragment_queue.json").exists()
    assert not (out_dir / "projection_semantic_fragment_queue.md").exists()
    hotspot_payload = json.loads(
        (out_dir / "hotspot_neighborhood_queue.json").read_text(encoding="utf-8")
    )
    assert "projection_fiber_decision" in hotspot_payload["source"]
    assert "projection_fiber_semantic_previews" in hotspot_payload["source"]


# gabion:behavior primary=desired
def test_run_passes_canonical_inputs_to_hotspot_queue(
    tmp_path: Path,
    monkeypatch: object,
) -> None:
    root = tmp_path
    out_dir = root / "artifacts/out"
    captured: dict[str, object] = {}

    def _fake_run_from_inputs(
        *,
        violations_by_rule: dict[str, list[dict[str, object]]],
        policy_check_result_path: Path | None = None,
        out_path: Path,
        markdown_out: Path | None = None,
        config: object | None = None,
    ) -> int:
        captured["violations_by_rule"] = violations_by_rule
        captured["policy_check_result_path"] = policy_check_result_path
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
        policy_scanner_suite.hotspot_neighborhood_queue,
        "run_from_inputs",
        _fake_run_from_inputs,
    )

    rc = policy_scanner_suite.run(root=root, out_dir=out_dir)

    assert rc == 0
    assert isinstance(captured["violations_by_rule"], dict)
    assert all(
        isinstance(items, list) and not items
        for items in captured["violations_by_rule"].values()
    )
    assert captured["policy_check_result_path"] == (out_dir / "policy_check_result.json")
    assert captured["out_path"] == out_dir / "hotspot_neighborhood_queue.json"
    assert captured["markdown_out"] == out_dir / "hotspot_neighborhood_queue.md"
    assert not (out_dir / "policy_suite_results.json").exists()


# gabion:behavior primary=desired
def test_run_resolves_changed_paths_at_wrapper_boundary(
    tmp_path: Path,
    monkeypatch: object,
) -> None:
    root = tmp_path
    out_dir = root / "artifacts/out"
    captured: dict[str, object] = {}

    def _fake_changed_paths_from_git(
        *,
        root: Path,
        base_sha: str | None,
        head_sha: str | None,
    ) -> set[str] | None:
        captured["root"] = root
        captured["base_sha"] = base_sha
        captured["head_sha"] = head_sha
        return {"src/gabion/example_boundary.py"}

    def _fake_scan_policy_suite(
        *,
        root: Path,
        changed_paths: set[str] | None = None,
    ) -> dict[str, list[dict[str, object]]]:
        captured["scan_root"] = root
        captured["changed_paths"] = changed_paths
        return {}

    def _fake_run_from_inputs(
        *,
        violations_by_rule: dict[str, list[dict[str, object]]],
        policy_check_result_path: Path | None = None,
        out_path: Path,
        markdown_out: Path | None = None,
        config: object | None = None,
    ) -> int:
        _ = violations_by_rule, policy_check_result_path, config
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("{}\n", encoding="utf-8")
        if markdown_out is not None:
            markdown_out.write_text("# Hotspot Neighborhood Queue\n", encoding="utf-8")
        return 0

    monkeypatch.setattr(
        policy_scanner_suite,
        "_changed_paths_from_git",
        _fake_changed_paths_from_git,
    )
    monkeypatch.setattr(
        policy_scanner_suite.runtime_policy_scanner_suite,
        "scan_policy_suite",
        _fake_scan_policy_suite,
    )
    monkeypatch.setattr(
        policy_scanner_suite.hotspot_neighborhood_queue,
        "run_from_inputs",
        _fake_run_from_inputs,
    )

    rc = policy_scanner_suite.run(
        root=root,
        out_dir=out_dir,
        base_sha="base123",
        head_sha="head456",
    )

    assert rc == 0
    assert captured["root"] == root
    assert captured["base_sha"] == "base123"
    assert captured["head_sha"] == "head456"
    assert captured["scan_root"] == root
    assert captured["changed_paths"] == {"src/gabion/example_boundary.py"}


# gabion:behavior primary=desired
def test_run_passes_minimal_boundary_shape_with_policy_check_result_path(
    tmp_path: Path,
    monkeypatch: object,
) -> None:
    root = tmp_path
    out_dir = root / "artifacts/out"
    captured: dict[str, object] = {}

    def _fake_scan_policy_suite(**_: object) -> object:
        return {"branchless": [{"path": "src/gabion/example.py"}]}

    def _fake_run_from_inputs(
        *,
        violations_by_rule: dict[str, list[dict[str, object]]],
        policy_check_result_path: Path | None = None,
        out_path: Path,
        markdown_out: Path | None = None,
        config: object | None = None,
    ) -> int:
        _ = config
        captured["violations_by_rule"] = violations_by_rule
        captured["policy_check_result_path"] = policy_check_result_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("{}\n", encoding="utf-8")
        if markdown_out is not None:
            markdown_out.write_text("# Hotspot Neighborhood Queue\n", encoding="utf-8")
        return 0

    monkeypatch.setattr(
        policy_scanner_suite.runtime_policy_scanner_suite,
        "scan_policy_suite",
        _fake_scan_policy_suite,
    )
    monkeypatch.setattr(
        policy_scanner_suite.hotspot_neighborhood_queue,
        "run_from_inputs",
        _fake_run_from_inputs,
    )

    rc = policy_scanner_suite.run(root=root, out_dir=out_dir)

    assert rc == 1
    assert captured["violations_by_rule"] == {
        "branchless": [{"path": "src/gabion/example.py"}]
    }
    assert captured["policy_check_result_path"] == (out_dir / "policy_check_result.json")
    assert not (out_dir / "policy_suite_results.json").exists()


# gabion:behavior primary=desired
def test_run_prints_nonempty_violation_families_from_runtime_result(
    tmp_path: Path,
    monkeypatch: object,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = tmp_path
    out_dir = root / "artifacts/out"

    def _fake_scan_policy_suite(**_: object) -> object:
        return {
            "branchless": [{"render": "branchless render"}],
            "future_rule_family": [{"render": "future render"}],
        }

    def _fake_run_from_inputs(
        *,
        violations_by_rule: dict[str, list[dict[str, object]]],
        policy_check_result_path: Path | None = None,
        out_path: Path,
        markdown_out: Path | None = None,
        config: object | None = None,
    ) -> int:
        _ = violations_by_rule, policy_check_result_path, config
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("{}\n", encoding="utf-8")
        if markdown_out is not None:
            markdown_out.write_text("# Hotspot Neighborhood Queue\n", encoding="utf-8")
        return 0

    monkeypatch.setattr(
        policy_scanner_suite.runtime_policy_scanner_suite,
        "scan_policy_suite",
        _fake_scan_policy_suite,
    )
    monkeypatch.setattr(
        policy_scanner_suite.hotspot_neighborhood_queue,
        "run_from_inputs",
        _fake_run_from_inputs,
    )

    rc = policy_scanner_suite.run(root=root, out_dir=out_dir)

    assert rc == 1
    captured = capsys.readouterr()
    assert "projection-semantic-fragment queue:" not in captured.out
    assert "branchless violations:" in captured.out
    assert "future_rule_family violations:" in captured.out
    assert "future render" in captured.out

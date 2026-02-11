from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from gabion import server
from gabion.analysis.timeout_context import Deadline, deadline_scope
from gabion.exceptions import NeverThrown


class _DummyWorkspace:
    def __init__(self, root_path: str) -> None:
        self.root_path = root_path


class _DummyServer:
    def __init__(self, root_path: str) -> None:
        self.workspace = _DummyWorkspace(root_path)


@dataclass
class _CommandResult:
    exit_code: int
    violations: int


_TIMEOUT_PAYLOAD = {
    "analysis_timeout_ticks": 1000,
    "analysis_timeout_tick_ns": 1_000_000,
}


def _with_timeout(payload: dict) -> dict:
    return {**_TIMEOUT_PAYLOAD, **payload}


@pytest.mark.parametrize(
    "payload",
    [
        {"analysis_timeout_ticks": 0, "analysis_timeout_tick_ns": 1},
        {"analysis_timeout_ticks": 1},
        {"analysis_timeout_ticks": 1, "analysis_timeout_tick_ns": 0},
        {"analysis_timeout_ms": 0},
        {"analysis_timeout_seconds": 0},
        {"analysis_timeout_seconds": "0.0001"},
        {},
    ],
)
def test_deadline_from_payload_rejects_invalid(payload: dict) -> None:
    with pytest.raises(NeverThrown):
        server._deadline_from_payload(payload)


def test_deadline_scope_requires_payload() -> None:
    with pytest.raises(NeverThrown):
        with server._deadline_scope_from_payload(None):
            pass


# gabion:evidence E:decision_surface/direct::server.py::gabion.server._normalize_transparent_decorators::value
def test_normalize_transparent_decorators() -> None:
    with deadline_scope(Deadline.from_timeout_ms(100)):
        assert server._normalize_transparent_decorators(None) is None
        assert server._normalize_transparent_decorators("a, b") == {"a", "b"}
        assert server._normalize_transparent_decorators(["a", "b,c"]) == {"a", "b", "c"}
        assert server._normalize_transparent_decorators([]) is None


# gabion:evidence E:function_site::server.py::gabion.server._uri_to_path
def test_uri_to_path_file_scheme() -> None:
    path = server._uri_to_path("file:///tmp/example.py")
    assert str(path).endswith("/tmp/example.py")


# gabion:evidence E:function_site::server.py::gabion.server._uri_to_path
def test_uri_to_path_plain_path() -> None:
    path = server._uri_to_path("/tmp/example.py")
    assert str(path).endswith("/tmp/example.py")


def _write_minimal_module(path: Path) -> None:
    path.write_text(
        "def alpha():\n"
        "    return 1\n"
    )


# gabion:evidence E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.analyze_paths::config,include_bundle_forest,include_coherence_witnesses,include_constant_smells,include_deadness_witnesses,include_decision_surfaces,include_exception_obligations,include_handledness_witnesses,include_invariant_propositions,include_lint_lines,include_never_invariants,include_rewrite_plans,include_unused_arg_smells,include_value_decision_surfaces,type_audit,type_audit_report E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.compute_structure_metrics::forest E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.render_structure_snapshot::forest,invariant_propositions E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.render_decision_snapshot::forest,project_root E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.render_protocol_stubs::kind E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.build_synthesis_plan::merge_overlap_threshold E:decision_surface/direct::server.py::gabion.server.execute_command::payload E:decision_surface/direct::config.py::gabion.config.decision_ignore_list::section E:decision_surface/direct::config.py::gabion.config.decision_require_tiers::section E:decision_surface/direct::config.py::gabion.config.decision_tier_map::section E:decision_surface/direct::config.py::gabion.config.exception_never_list::section E:decision_surface/direct::server.py::gabion.server._normalize_transparent_decorators::value
def test_execute_command_no_violations(tmp_path: Path) -> None:
    module_path = tmp_path / "sample.py"
    _write_minimal_module(module_path)
    ls = _DummyServer(str(tmp_path))
    result = server.execute_command(
        ls,
        _with_timeout(
            {
            "root": str(tmp_path),
            "paths": [str(module_path)],
            "fail_on_violations": True,
            "type_audit": False,
            }
        ),
    )
    assert _CommandResult(
        exit_code=result.get("exit_code", -1),
        violations=result.get("violations", -1),
    ) == _CommandResult(exit_code=0, violations=0)


def test_execute_command_defaults_paths(tmp_path: Path) -> None:
    module_path = tmp_path / "sample.py"
    _write_minimal_module(module_path)
    ls = _DummyServer(str(tmp_path))
    result = server.execute_command(
        ls,
        _with_timeout(
            {
                "root": str(tmp_path),
            }
        ),
    )
    assert "violations" in result


# gabion:evidence E:decision_surface/direct::ambiguity_delta.py::gabion.analysis.ambiguity_delta.build_delta_payload::baseline_path E:decision_surface/direct::config.py::gabion.config.decision_ignore_list::section E:decision_surface/direct::config.py::gabion.config.decision_require_tiers::section E:decision_surface/direct::config.py::gabion.config.decision_tier_map::section E:decision_surface/direct::config.py::gabion.config.exception_never_list::section E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.analyze_paths::config,include_ambiguities,include_bundle_forest,include_coherence_witnesses,include_constant_smells,include_deadness_witnesses,include_decision_surfaces,include_exception_obligations,include_handledness_witnesses,include_invariant_propositions,include_lint_lines,include_never_invariants,include_rewrite_plans,include_unused_arg_smells,include_value_decision_surfaces,type_audit,type_audit_report E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.build_synthesis_plan::merge_overlap_threshold E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.compute_structure_metrics::forest E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.render_decision_snapshot::forest,project_root E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.render_protocol_stubs::kind E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.render_structure_snapshot::forest,invariant_propositions E:decision_surface/direct::server.py::gabion.server._normalize_transparent_decorators::value E:decision_surface/direct::server.py::gabion.server.execute_command::payload E:decision_surface/value_encoded::ambiguity_delta.py::gabion.analysis.ambiguity_delta.build_delta_payload::baseline,current
def test_execute_command_baseline_write(tmp_path: Path) -> None:
    module_path = tmp_path / "sample.py"
    _write_minimal_module(module_path)
    baseline_path = tmp_path / "baseline.txt"
    ls = _DummyServer(str(tmp_path))
    result = server.execute_command(
        ls,
        _with_timeout(
            {
            "root": str(tmp_path),
            "paths": [str(module_path)],
            "baseline": str(baseline_path),
            "baseline_write": True,
            }
        ),
    )
    assert result.get("baseline_written") is True
    assert baseline_path.exists()
    baseline_text = baseline_path.read_text()
    assert baseline_text.startswith("# gabion baseline")


# gabion:evidence E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.analyze_paths::config,include_bundle_forest,include_coherence_witnesses,include_constant_smells,include_deadness_witnesses,include_decision_surfaces,include_exception_obligations,include_handledness_witnesses,include_invariant_propositions,include_lint_lines,include_never_invariants,include_rewrite_plans,include_unused_arg_smells,include_value_decision_surfaces,type_audit,type_audit_report E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.compute_structure_metrics::forest E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.render_structure_snapshot::forest,invariant_propositions E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.render_decision_snapshot::forest,project_root E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.render_protocol_stubs::kind E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.build_synthesis_plan::merge_overlap_threshold E:decision_surface/direct::server.py::gabion.server.execute_command::payload E:decision_surface/direct::config.py::gabion.config.decision_ignore_list::section E:decision_surface/direct::config.py::gabion.config.decision_require_tiers::section E:decision_surface/direct::config.py::gabion.config.decision_tier_map::section E:decision_surface/direct::config.py::gabion.config.exception_never_list::section E:decision_surface/direct::server.py::gabion.server._normalize_transparent_decorators::value
def test_execute_command_report_and_dot(tmp_path: Path) -> None:
    module_path = tmp_path / "sample.py"
    _write_minimal_module(module_path)
    report_path = tmp_path / "report.md"
    dot_path = tmp_path / "graph.dot"
    ls = _DummyServer(str(tmp_path))
    result = server.execute_command(
        ls,
        _with_timeout(
            {
            "root": str(tmp_path),
            "paths": [str(module_path)],
            "report": str(report_path),
            "dot": str(dot_path),
            }
        ),
    )
    assert report_path.exists()
    assert dot_path.exists()
    assert "violations" in result


# gabion:evidence E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.analyze_paths::config,include_bundle_forest,include_coherence_witnesses,include_constant_smells,include_deadness_witnesses,include_decision_surfaces,include_exception_obligations,include_handledness_witnesses,include_invariant_propositions,include_lint_lines,include_never_invariants,include_rewrite_plans,include_unused_arg_smells,include_value_decision_surfaces,type_audit,type_audit_report E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.compute_structure_metrics::forest E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.render_structure_snapshot::forest,invariant_propositions E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.render_decision_snapshot::forest,project_root E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.render_protocol_stubs::kind E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.build_synthesis_plan::merge_overlap_threshold E:decision_surface/direct::server.py::gabion.server.execute_command::payload E:decision_surface/direct::config.py::gabion.config.decision_ignore_list::section E:decision_surface/direct::config.py::gabion.config.decision_require_tiers::section E:decision_surface/direct::config.py::gabion.config.decision_tier_map::section E:decision_surface/direct::config.py::gabion.config.exception_never_list::section E:decision_surface/direct::server.py::gabion.server._normalize_transparent_decorators::value
def test_execute_command_structure_tree(tmp_path: Path) -> None:
    module_path = tmp_path / "sample.py"
    _write_minimal_module(module_path)
    snapshot_path = tmp_path / "snapshot.json"
    ls = _DummyServer(str(tmp_path))
    result = server.execute_command(
        ls,
        _with_timeout(
            {
            "root": str(tmp_path),
            "paths": [str(module_path)],
            "structure_tree": str(snapshot_path),
            }
        ),
    )
    assert snapshot_path.exists()
    assert "violations" in result


# gabion:evidence E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.analyze_paths::config,include_bundle_forest,include_coherence_witnesses,include_constant_smells,include_deadness_witnesses,include_decision_surfaces,include_exception_obligations,include_handledness_witnesses,include_invariant_propositions,include_lint_lines,include_never_invariants,include_rewrite_plans,include_unused_arg_smells,include_value_decision_surfaces,type_audit,type_audit_report E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.compute_structure_metrics::forest E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.render_structure_snapshot::forest,invariant_propositions E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.render_decision_snapshot::forest,project_root E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.render_protocol_stubs::kind E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.build_synthesis_plan::merge_overlap_threshold E:decision_surface/direct::server.py::gabion.server.execute_command::payload E:decision_surface/direct::config.py::gabion.config.decision_ignore_list::section E:decision_surface/direct::config.py::gabion.config.decision_require_tiers::section E:decision_surface/direct::config.py::gabion.config.decision_tier_map::section E:decision_surface/direct::config.py::gabion.config.exception_never_list::section E:decision_surface/direct::server.py::gabion.server._normalize_transparent_decorators::value
def test_execute_command_structure_tree_stdout(tmp_path: Path) -> None:
    module_path = tmp_path / "sample.py"
    _write_minimal_module(module_path)
    ls = _DummyServer(str(tmp_path))
    result = server.execute_command(
        ls,
        _with_timeout(
            {
            "root": str(tmp_path),
            "paths": [str(module_path)],
            "structure_tree": "-",
            }
        ),
    )
    assert "structure_tree" in result
    assert result["structure_tree"]["files"]


def test_execute_command_defaults_tick_ns(tmp_path: Path) -> None:
    module_path = tmp_path / "sample.py"
    _write_minimal_module(module_path)
    ls = _DummyServer(str(tmp_path))
    result = server.execute_command(
        ls,
        {
            "root": str(tmp_path),
            "paths": [str(module_path)],
            "analysis_timeout_ticks": 1000,
            "analysis_timeout_tick_ns": 1_000_000,
        },
    )
    assert result.get("exit_code") == 0


# gabion:evidence E:decision_surface/direct::server.py::gabion.server.execute_structure_diff::payload
def test_execute_structure_diff(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.json"
    current = tmp_path / "current.json"
    baseline.write_text("{\"files\": [{\"path\": \"a.py\", \"functions\": []}]}")
    current.write_text("{\"files\": [{\"path\": \"a.py\", \"functions\": [{\"name\": \"f\", \"bundles\": [[\"a\"]]}]}]}")
    result = server.execute_structure_diff(
        None,
        _with_timeout({"baseline": str(baseline), "current": str(current)}),
    )
    assert result["exit_code"] == 0
    assert result["diff"]["added"][0]["bundle"] == ["a"]


# gabion:evidence E:decision_surface/direct::server.py::gabion.server.execute_structure_diff::payload
def test_execute_structure_diff_missing_payload() -> None:
    result = server.execute_structure_diff(None, _with_timeout({}))
    assert result["exit_code"] == 2


# gabion:evidence E:decision_surface/direct::server.py::gabion.server.execute_structure_diff::payload
def test_execute_structure_diff_missing_paths() -> None:
    result = server.execute_structure_diff(None, _with_timeout({}))
    assert result["exit_code"] == 2
    assert "required" in result["errors"][0]


# gabion:evidence E:decision_surface/direct::server.py::gabion.server.execute_structure_diff::payload
def test_execute_structure_diff_invalid_snapshot(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.json"
    current = tmp_path / "current.json"
    baseline.write_text("{bad-json")
    current.write_text("{\"files\": []}")
    result = server.execute_structure_diff(
        None,
        _with_timeout({"baseline": str(baseline), "current": str(current)}),
    )
    assert result["exit_code"] == 2
    assert "Invalid snapshot JSON" in result["errors"][0]


# gabion:evidence E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.analyze_paths::config,include_bundle_forest,include_coherence_witnesses,include_constant_smells,include_deadness_witnesses,include_decision_surfaces,include_exception_obligations,include_handledness_witnesses,include_invariant_propositions,include_lint_lines,include_never_invariants,include_rewrite_plans,include_unused_arg_smells,include_value_decision_surfaces,type_audit,type_audit_report E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.compute_structure_metrics::forest E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.render_structure_snapshot::forest,invariant_propositions E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.render_decision_snapshot::forest,project_root E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.render_protocol_stubs::kind E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.build_synthesis_plan::merge_overlap_threshold E:decision_surface/direct::server.py::gabion.server.execute_command::payload E:decision_surface/direct::config.py::gabion.config.decision_ignore_list::section E:decision_surface/direct::config.py::gabion.config.decision_require_tiers::section E:decision_surface/direct::config.py::gabion.config.decision_tier_map::section E:decision_surface/direct::config.py::gabion.config.exception_never_list::section E:decision_surface/direct::server.py::gabion.server._normalize_transparent_decorators::value
def test_execute_command_structure_metrics(tmp_path: Path) -> None:
    module_path = tmp_path / "sample.py"
    _write_minimal_module(module_path)
    metrics_path = tmp_path / "metrics.json"
    ls = _DummyServer(str(tmp_path))
    result = server.execute_command(
        ls,
        _with_timeout(
            {
            "root": str(tmp_path),
            "paths": [str(module_path)],
            "structure_metrics": str(metrics_path),
            }
        ),
    )
    assert metrics_path.exists()
    assert "violations" in result


# gabion:evidence E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.analyze_paths::config,include_bundle_forest,include_coherence_witnesses,include_constant_smells,include_deadness_witnesses,include_decision_surfaces,include_exception_obligations,include_handledness_witnesses,include_invariant_propositions,include_lint_lines,include_never_invariants,include_rewrite_plans,include_unused_arg_smells,include_value_decision_surfaces,type_audit,type_audit_report E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.compute_structure_metrics::forest E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.render_structure_snapshot::forest,invariant_propositions E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.render_decision_snapshot::forest,project_root E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.render_protocol_stubs::kind E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.build_synthesis_plan::merge_overlap_threshold E:decision_surface/direct::server.py::gabion.server.execute_command::payload E:decision_surface/direct::config.py::gabion.config.decision_ignore_list::section E:decision_surface/direct::config.py::gabion.config.decision_require_tiers::section E:decision_surface/direct::config.py::gabion.config.decision_tier_map::section E:decision_surface/direct::config.py::gabion.config.exception_never_list::section E:decision_surface/direct::server.py::gabion.server._normalize_transparent_decorators::value
def test_execute_command_structure_metrics_stdout(tmp_path: Path) -> None:
    module_path = tmp_path / "sample.py"
    _write_minimal_module(module_path)
    ls = _DummyServer(str(tmp_path))
    result = server.execute_command(
        ls,
        _with_timeout(
            {
                "root": str(tmp_path),
                "paths": [str(module_path)],
                "structure_metrics": "-",
            }
        ),
    )
    assert "structure_metrics" in result


# gabion:evidence E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.analyze_paths::config,include_bundle_forest,include_coherence_witnesses,include_constant_smells,include_deadness_witnesses,include_decision_surfaces,include_exception_obligations,include_handledness_witnesses,include_invariant_propositions,include_lint_lines,include_never_invariants,include_rewrite_plans,include_unused_arg_smells,include_value_decision_surfaces,type_audit,type_audit_report E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.compute_structure_metrics::forest E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.render_structure_snapshot::forest,invariant_propositions E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.render_decision_snapshot::forest,project_root E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.render_protocol_stubs::kind E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.build_synthesis_plan::merge_overlap_threshold E:decision_surface/direct::server.py::gabion.server.execute_command::payload E:decision_surface/direct::config.py::gabion.config.decision_ignore_list::section E:decision_surface/direct::config.py::gabion.config.decision_require_tiers::section E:decision_surface/direct::config.py::gabion.config.decision_tier_map::section E:decision_surface/direct::config.py::gabion.config.exception_never_list::section E:decision_surface/direct::server.py::gabion.server._normalize_transparent_decorators::value
def test_execute_command_synthesis_outputs(tmp_path: Path) -> None:
    module_path = tmp_path / "sample.py"
    _write_minimal_module(module_path)
    plan_path = tmp_path / "plan.json"
    protocol_path = tmp_path / "protocols.py"
    ls = _DummyServer(str(tmp_path))
    result = server.execute_command(
        ls,
        _with_timeout(
            {
                "root": str(tmp_path),
                "paths": [str(module_path)],
                "synthesis_plan": str(plan_path),
                "synthesis_protocols": str(protocol_path),
            }
        ),
    )
    assert plan_path.exists()
    assert protocol_path.exists()
    assert "violations" in result


# gabion:evidence E:decision_surface/direct::server.py::gabion.server.execute_synthesis::payload
def test_execute_synthesis_minimal_payload() -> None:
    result = server.execute_synthesis(None, _with_timeout({"bundles": []}))
    assert result["protocols"] == []
    assert result["errors"] == []


# gabion:evidence E:decision_surface/direct::server.py::gabion.server.execute_refactor::ls,payload
def test_execute_refactor_invalid_payload() -> None:
    result = server.execute_refactor(None, _with_timeout({}))
    assert result["errors"]

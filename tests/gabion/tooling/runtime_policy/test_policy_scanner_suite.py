from __future__ import annotations

import json
from pathlib import Path

from gabion.tooling.runtime import policy_scanner_suite
from gabion.tooling.runtime.projection_fiber_semantics_summary import (
    projection_fiber_semantics_summary_from_payload,
)


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


# gabion:evidence E:call_footprint::tests/test_policy_scanner_suite.py::test_policy_scanner_suite_scan_and_cache::policy_scanner_suite.py::gabion.tooling.policy_scanner_suite.scan_policy_suite
# gabion:behavior primary=desired
def test_policy_scanner_suite_scan_and_cache(tmp_path: Path) -> None:
    root = tmp_path
    _write(
        root / "tests/test_no_monkeypatch_sample.py",
        "def test_example(monkeypatch):\n    assert monkeypatch is not None\n",
    )
    _write(
        root / "src/gabion/branch_sample.py",
        "def branchy(flag):\n    if flag:\n        return 1\n    return 0\n",
    )
    _write(
        root / "src/gabion/fallback_sample.py",
        "def normalize(value):\n    if value is None:\n        return None\n    return value\n",
    )
    _write(
        root / "tests/test_legacy_import.py",
        "from gabion.analysis import legacy_dataflow_monolith\n",
    )
    _write(
        root / "src/gabion/shape_sample.py",
        "\n".join(
            [
                "def shape(value):",
                "    pair = (value, value + 1)",
                "    payload = {'value': value}",
                "    text = str(value)",
                "    return pair, payload, text",
            ]
        )
        + "\n",
    )
    _write(root / "src/gabion/bad_syntax.py", "def broken(:\n")
    _write(root / "src/gabion/__pycache__/ignored.py", "def ignored():\n    return 1\n")

    artifact_path = root / "artifacts/out/policy_suite_results.json"
    first = policy_scanner_suite.load_or_scan_policy_suite(
        root=root,
        artifact_path=artifact_path,
    )
    assert first.cached is False
    assert first.result.total_violations() > 0
    decision = first.result.decision()
    assert decision.outcome.value in {"block", "warn", "pass", "skip"}
    assert policy_scanner_suite.violations_for_rule(first.result, rule="branchless")
    branchless_violation = policy_scanner_suite.violations_for_rule(first.result, rule="branchless")[0]
    assert "lattice_witness" in branchless_violation
    assert "recombination_frontier" not in branchless_violation
    assert branchless_violation["lattice_witness"]["complete"] in {True, False}
    assert "obligations" in branchless_violation["lattice_witness"]
    assert "boundary_crossings" in branchless_violation["lattice_witness"]
    assert policy_scanner_suite.violations_for_rule(first.result, rule="defensive_fallback")
    assert policy_scanner_suite.violations_for_rule(first.result, rule="fiber_loop_structure_contract")
    assert policy_scanner_suite.violations_for_rule(first.result, rule="fiber_filter_processor_contract")
    assert policy_scanner_suite.violations_for_rule(first.result, rule="fiber_return_shape_contract")
    assert policy_scanner_suite.violations_for_rule(first.result, rule="fiber_scalar_sentinel_contract")
    assert policy_scanner_suite.violations_for_rule(first.result, rule="fiber_type_dispatch_contract")
    assert policy_scanner_suite.violations_for_rule(first.result, rule="no_anonymous_tuple")
    assert policy_scanner_suite.violations_for_rule(first.result, rule="no_mutable_dict")
    assert policy_scanner_suite.violations_for_rule(first.result, rule="no_scalar_conversion_boundary")
    assert policy_scanner_suite.violations_for_rule(first.result, rule="no_monkeypatch")
    assert policy_scanner_suite.violations_for_rule(first.result, rule="no_legacy_monolith_import")
    assert policy_scanner_suite.violations_for_rule(first.result, rule="orchestrator_primitive_barrel") == []
    assert policy_scanner_suite.violations_for_rule(first.result, rule="typing_surface")
    assert policy_scanner_suite.violations_for_rule(first.result, rule="runtime_narrowing_boundary")
    assert policy_scanner_suite.violations_for_rule(first.result, rule="aspf_normalization_idempotence") == []
    assert policy_scanner_suite.violations_for_rule(first.result, rule="boundary_core_contract") == []
    assert policy_scanner_suite.violations_for_rule(first.result, rule="fiber_normalization_contract") == []
    assert policy_scanner_suite.violations_for_rule(first.result, rule="test_subprocess_hygiene") == []
    assert policy_scanner_suite.violations_for_rule(first.result, rule="test_sleep_hygiene") == []
    first_payload = {
        "format_version": 1,
        "violations": first.result.violations_by_rule,
    }
    assert "decision" not in first_payload
    assert "generated_at_utc" not in first_payload
    assert "root" not in first_payload
    assert "counts" not in first_payload
    assert "inventory_hash" not in first_payload
    assert "rule_set_hash" not in first_payload

    second = policy_scanner_suite.load_or_scan_policy_suite(
        root=root,
        artifact_path=artifact_path,
    )
    assert second.cached is True
    cached_payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert "counts" not in cached_payload
    assert "decision" not in cached_payload
    assert "generated_at_utc" not in cached_payload
    assert "policy_results" not in cached_payload
    assert isinstance(cached_payload.get("inventory_hash"), str)
    assert isinstance(cached_payload.get("rule_set_hash"), str)


# gabion:evidence E:call_footprint::tests/test_policy_scanner_suite.py::test_policy_scanner_suite_cache_invalidation_and_payload_normalization::policy_scanner_suite.py::gabion.tooling.policy_scanner_suite.scan_policy_suite
# gabion:behavior primary=desired
def test_policy_scanner_suite_cache_invalidation_and_payload_normalization(
    tmp_path: Path,
) -> None:
    root = tmp_path
    _write(root / "src/gabion/a.py", "def f(x):\n    if x:\n        return 1\n    return 0\n")
    _write(root / "tests/test_a.py", "def test_a(monkeypatch):\n    assert monkeypatch is not None\n")
    artifact_path = root / "artifacts/out/policy_suite_results.json"

    baseline = policy_scanner_suite.load_or_scan_policy_suite(
        root=root,
        artifact_path=artifact_path,
    )
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    payload["violations"] = "bad-shape"
    artifact_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    normalized = policy_scanner_suite.load_or_scan_policy_suite(
        root=root,
        artifact_path=artifact_path,
    )
    assert normalized.cached is True
    assert normalized.result.violations_by_rule["branchless"] == []
    assert normalized.result.violations_by_rule["defensive_fallback"] == []
    assert normalized.result.violations_by_rule["fiber_loop_structure_contract"] == []
    assert normalized.result.violations_by_rule["fiber_filter_processor_contract"] == []
    assert normalized.result.violations_by_rule["fiber_return_shape_contract"] == []
    assert normalized.result.violations_by_rule["fiber_scalar_sentinel_contract"] == []
    assert normalized.result.violations_by_rule["fiber_type_dispatch_contract"] == []
    assert normalized.result.violations_by_rule["no_anonymous_tuple"] == []
    assert normalized.result.violations_by_rule["no_mutable_dict"] == []
    assert normalized.result.violations_by_rule["no_scalar_conversion_boundary"] == []
    assert normalized.result.violations_by_rule["no_monkeypatch"] == []
    assert normalized.result.violations_by_rule["no_legacy_monolith_import"] == []
    assert normalized.result.violations_by_rule["orchestrator_primitive_barrel"] == []
    assert normalized.result.violations_by_rule["typing_surface"] == []
    assert normalized.result.violations_by_rule["runtime_narrowing_boundary"] == []
    assert normalized.result.violations_by_rule["aspf_normalization_idempotence"] == []
    assert normalized.result.violations_by_rule["boundary_core_contract"] == []
    assert normalized.result.violations_by_rule["fiber_normalization_contract"] == []
    assert normalized.result.violations_by_rule["test_subprocess_hygiene"] == []
    assert normalized.result.violations_by_rule["test_sleep_hygiene"] == []

    _write(
        root / "src/gabion/new_file.py",
        "def f():\n    if True:\n        return 1\n    return 0\n",
    )
    invalidated = policy_scanner_suite.load_or_scan_policy_suite(
        root=root,
        artifact_path=artifact_path,
    )
    assert invalidated.cached is False
    invalidated_payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert invalidated_payload["inventory_hash"] != payload["inventory_hash"]


# gabion:evidence E:call_footprint::tests/test_policy_scanner_suite.py::test_policy_scanner_suite_private_cache_and_payload_branches::policy_scanner_suite.py::gabion.tooling.policy_scanner_suite.scan_policy_suite
# gabion:behavior primary=desired
def test_policy_scanner_suite_private_cache_and_payload_branches(
    tmp_path: Path,
) -> None:
    broken_path = tmp_path / "broken.json"
    broken_path.write_text("{bad", encoding="utf-8")
    assert policy_scanner_suite._load_cached_payload(broken_path) is None

    wrong_format_path = tmp_path / "wrong_format.json"
    wrong_format_path.write_text(
        json.dumps({"format_version": 999, "violations": {}}),
        encoding="utf-8",
    )
    assert policy_scanner_suite._load_cached_payload(wrong_format_path) is None

    normalized = policy_scanner_suite._violations_from_payload(
        {
            "violations": {
                "no_monkeypatch": {"bad": "shape"},
                "branchless": [],
                "defensive_fallback": [],
                "fiber_loop_structure_contract": [],
                "fiber_filter_processor_contract": [],
                "fiber_return_shape_contract": [],
                "fiber_scalar_sentinel_contract": [],
                "no_anonymous_tuple": [],
                "no_mutable_dict": [],
                "no_scalar_conversion_boundary": [],
            }
        }
    )
    assert normalized["no_monkeypatch"] == []
    assert normalized["orchestrator_primitive_barrel"] == []
    assert normalized["fiber_loop_structure_contract"] == []

    child_inputs = policy_scanner_suite.PolicySuiteChildInputs(
        projection_fiber_semantics=None,
    )
    assert child_inputs.projection_fiber_semantics is None
    assert normalized["fiber_filter_processor_contract"] == []
    assert normalized["fiber_return_shape_contract"] == []
    assert normalized["fiber_scalar_sentinel_contract"] == []
    assert normalized["fiber_type_dispatch_contract"] == []
    assert normalized["no_anonymous_tuple"] == []
    assert normalized["no_mutable_dict"] == []
    assert normalized["no_scalar_conversion_boundary"] == []
    assert normalized["typing_surface"] == []
    assert normalized["runtime_narrowing_boundary"] == []
    assert normalized["aspf_normalization_idempotence"] == []
    assert normalized["boundary_core_contract"] == []
    assert normalized["fiber_normalization_contract"] == []
    assert normalized["test_subprocess_hygiene"] == []
    assert normalized["test_sleep_hygiene"] == []


# gabion:evidence E:call_footprint::tests/test_policy_scanner_suite.py::test_policy_scanner_suite_scan_with_explicit_nonstandard_files::policy_scanner_suite.py::gabion.tooling.policy_scanner_suite.scan_policy_suite
# gabion:behavior primary=desired
def test_policy_scanner_suite_scan_with_explicit_nonstandard_files(tmp_path: Path) -> None:
    root = tmp_path
    external_file = root / "external.py"
    _write(external_file, "def utility():\n    return 1\n")

    result = policy_scanner_suite.scan_policy_suite(
        root=root,
        files=(external_file.resolve(),),
    )
    assert result.total_violations() == 0


# gabion:behavior primary=desired
def test_policy_scanner_suite_flags_fiber_type_dispatch_contract(
    tmp_path: Path,
) -> None:
    root = tmp_path
    _write(
        root / "src/gabion/type_route_sample.py",
        "\n".join(
            [
                "def route(value: object) -> object:",
                "    if isinstance(value, str):",
                "        return value",
                "    return value",
            ]
        )
        + "\n",
    )
    result = policy_scanner_suite.scan_policy_suite(root=root)
    violations = policy_scanner_suite.violations_for_rule(
        result,
        rule="fiber_type_dispatch_contract",
    )
    assert violations
    assert any(item.get("kind") == "manual_type_guard" for item in violations)


# gabion:behavior primary=desired
def test_policy_scanner_suite_flags_fiber_loop_structure_contract(
    tmp_path: Path,
) -> None:
    root = tmp_path
    _write(
        root / "src/gabion/loop_structure_sample.py",
        "\n".join(
            [
                "def nested(values):",
                "    for outer in values:",
                "        for inner in outer:",
                "            yield inner",
                "",
                "def single(values):",
                "    for value in values:",
                "        print(value)",
            ]
        )
        + "\n",
    )
    result = policy_scanner_suite.scan_policy_suite(root=root)
    violations = policy_scanner_suite.violations_for_rule(
        result,
        rule="fiber_loop_structure_contract",
    )
    assert violations
    kinds = {str(item.get("kind", "")) for item in violations}
    assert "nested_loop" in kinds
    assert "single_loop_non_generator" in kinds


# gabion:behavior primary=desired
def test_policy_scanner_suite_flags_fiber_filter_processor_contract(
    tmp_path: Path,
) -> None:
    root = tmp_path
    _write(
        root / "src/gabion/filter_processor_sample.py",
        "\n".join(
            [
                "def process(values):",
                "    for value in values:",
                "        if value:",
                "            yield value",
                "",
                "def comp(values):",
                "    return [value for value in values if value]",
            ]
        )
        + "\n",
    )
    result = policy_scanner_suite.scan_policy_suite(root=root)
    violations = policy_scanner_suite.violations_for_rule(
        result,
        rule="fiber_filter_processor_contract",
    )
    assert violations
    kinds = {str(item.get("kind", "")) for item in violations}
    assert "branch_in_loop_processor" in kinds
    assert "comprehension_filter_branch" in kinds


# gabion:behavior primary=desired
def test_policy_scanner_suite_flags_fiber_return_shape_contract(
    tmp_path: Path,
) -> None:
    root = tmp_path
    _write(
        root / "src/gabion/return_shape_sample.py",
        "\n".join(
            [
                "def eager(values):",
                "    return [value for value in values]",
                "",
                "def singleton_iter():",
                "    return iter((1,))",
                "",
            ]
        )
        + "\n",
    )
    result = policy_scanner_suite.scan_policy_suite(root=root)
    violations = policy_scanner_suite.violations_for_rule(
        result,
        rule="fiber_return_shape_contract",
    )
    assert violations
    kinds = {str(item.get("kind", "")) for item in violations}
    assert "container_return_prefer_iterator" in kinds
    assert "iterator_return_prefer_item" in kinds


# gabion:evidence E:call_footprint::tests/test_policy_scanner_suite.py::test_policy_scanner_suite_flags_retired_monolith_module_file::policy_scanner_suite.py::gabion.tooling.policy_scanner_suite.scan_policy_suite
# gabion:behavior primary=desired
def test_policy_scanner_suite_flags_retired_monolith_module_file(tmp_path: Path) -> None:
    root = tmp_path
    _write(
        root / "src/gabion/analysis/legacy_dataflow_monolith.py",
        "def retired():\n    return 1\n",
    )
    result = policy_scanner_suite.scan_policy_suite(root=root)
    violations = policy_scanner_suite.violations_for_rule(result, rule="no_legacy_monolith_import")
    assert violations
    assert any(item["kind"] == "module_present" for item in violations)


# gabion:evidence E:call_footprint::tests/test_policy_scanner_suite.py::test_policy_scanner_suite_flags_all_legacy_monolith_import_forms::policy_scanner_suite.py::gabion.tooling.policy_scanner_suite.scan_policy_suite
# gabion:behavior primary=allowed_unwanted facets=legacy
def test_policy_scanner_suite_flags_all_legacy_monolith_import_forms(tmp_path: Path) -> None:
    root = tmp_path
    _write(
        root / "src/gabion/analysis/imports.py",
        "\n".join(
            [
                "import gabion.analysis.legacy_dataflow_monolith as lm, os",
                "from gabion.analysis.legacy_dataflow_monolith import run",
                "from gabion.analysis import other_name, legacy_dataflow_monolith",
                "from .legacy_dataflow_monolith import run as rel_run",
                "from . import other_name",
                "from . import other_name, legacy_dataflow_monolith",
                "",
            ]
        ),
    )
    result = policy_scanner_suite.scan_policy_suite(root=root)
    violations = policy_scanner_suite.violations_for_rule(result, rule="no_legacy_monolith_import")
    assert len(violations) >= 4
    kinds = {str(item.get("kind", "")) for item in violations}
    assert "import" in kinds
    assert "import_from" in kinds


# gabion:evidence E:call_footprint::tests/test_policy_scanner_suite.py::test_policy_scanner_suite_private_baseline_loader_and_filter_branches::policy_scanner_suite.py::gabion.tooling.policy_scanner_suite._load_rule_baseline_keys
# gabion:behavior primary=desired
def test_policy_scanner_suite_private_baseline_loader_and_filter_branches(tmp_path: Path) -> None:
    class _ModuleNoLoader:
        pass

    class _ModuleBadLoader:
        @staticmethod
        def _load_baseline(_path: Path):
            raise ValueError("bad")

    class _ModuleNonSetLoader:
        @staticmethod
        def _load_baseline(_path: Path):
            return ["not", "a", "set"]

    class _ModuleSetLoader:
        @staticmethod
        def _load_baseline(_path: Path):
            return {"ok", 1}

    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text("{}", encoding="utf-8")
    assert policy_scanner_suite._load_rule_baseline_keys(
        module=_ModuleNoLoader(),
        baseline_path=baseline_path,
    ) == set()
    assert policy_scanner_suite._load_rule_baseline_keys(
        module=_ModuleBadLoader(),
        baseline_path=baseline_path,
    ) == set()
    assert policy_scanner_suite._load_rule_baseline_keys(
        module=_ModuleNonSetLoader(),
        baseline_path=baseline_path,
    ) == set()
    assert policy_scanner_suite._load_rule_baseline_keys(
        module=_ModuleSetLoader(),
        baseline_path=baseline_path,
    ) == {"ok"}

    class _Violation:
        def __init__(self, key: str) -> None:
            self.key = key

    violations = [_Violation("allow"), _Violation("deny"), object()]
    filtered = policy_scanner_suite._filter_baseline_violations(
        violations,
        allowed_keys={"allow"},
    )
    assert len(filtered) == 2
    assert isinstance(filtered[0], _Violation)
    assert getattr(filtered[0], "key", "") == "deny"


# gabion:evidence E:call_footprint::tests/test_policy_scanner_suite.py::test_policy_scanner_suite_respects_branch_and_fallback_baselines::policy_scanner_suite.py::gabion.tooling.policy_scanner_suite.scan_policy_suite
# gabion:behavior primary=allowed_unwanted facets=fallback
def test_policy_scanner_suite_respects_branch_and_fallback_baselines(tmp_path: Path) -> None:
    root = tmp_path
    _write(
        root / "src/gabion/branch_sample.py",
        "def branchy(flag):\n    if flag:\n        return 1\n    return 0\n",
    )
    _write(
        root / "src/gabion/fallback_sample.py",
        "def normalize(value):\n    if value:\n        return None\n    return value\n",
    )
    _write(
        root / "baselines/branchless_policy_baseline.json",
        json.dumps(
            {
                "version": 1,
                "violations": [
                    {
                        "path": "src/gabion/branch_sample.py",
                        "line": 2,
                        "column": 5,
                        "qualname": "branchy",
                        "kind": "if",
                        "message": "branch construct outside decision protocol",
                    },
                    {
                        "path": "src/gabion/fallback_sample.py",
                        "line": 2,
                        "column": 5,
                        "qualname": "normalize",
                        "kind": "if",
                        "message": "branch construct outside decision protocol",
                    }
                ],
            },
            indent=2,
        )
        + "\n",
    )
    _write(
        root / "baselines/defensive_fallback_policy_baseline.json",
        json.dumps(
            {
                "version": 1,
                "violations": [
                    {
                        "path": "src/gabion/fallback_sample.py",
                        "line": 2,
                        "column": 5,
                        "qualname": "normalize",
                        "kind": "sentinel_return",
                        "message": "returns sentinel fallback",
                    }
                ],
            },
            indent=2,
        )
        + "\n",
    )
    _write(
        root / "baselines/runtime_narrowing_boundary_policy_baseline.json",
        json.dumps(
            {
                "version": 1,
                "violations": [
                    {
                        "path": "src/gabion/fallback_sample.py",
                        "line": 2,
                        "column": 8,
                        "qualname": "normalize",
                        "kind": "isinstance_call",
                        "call": "isinstance(value, str)",
                    }
                ],
            },
            indent=2,
        )
        + "\n",
    )

    result = policy_scanner_suite.scan_policy_suite(root=root)
    assert policy_scanner_suite.violations_for_rule(result, rule="branchless") == []
    assert policy_scanner_suite.violations_for_rule(result, rule="defensive_fallback") == []
    assert policy_scanner_suite.violations_for_rule(result, rule="fiber_loop_structure_contract") == []
    assert policy_scanner_suite.violations_for_rule(result, rule="fiber_filter_processor_contract") == []
    assert policy_scanner_suite.violations_for_rule(result, rule="fiber_return_shape_contract") == []
    assert policy_scanner_suite.violations_for_rule(result, rule="fiber_scalar_sentinel_contract") == []
    assert policy_scanner_suite.violations_for_rule(result, rule="fiber_type_dispatch_contract") == []
    assert policy_scanner_suite.violations_for_rule(result, rule="no_anonymous_tuple") == []
    assert policy_scanner_suite.violations_for_rule(result, rule="no_mutable_dict") == []
    assert policy_scanner_suite.violations_for_rule(result, rule="no_scalar_conversion_boundary") == []
    assert policy_scanner_suite.violations_for_rule(result, rule="no_monkeypatch") == []
    assert policy_scanner_suite.violations_for_rule(result, rule="no_legacy_monolith_import") == []
    assert policy_scanner_suite.violations_for_rule(result, rule="orchestrator_primitive_barrel") == []
    assert policy_scanner_suite.violations_for_rule(result, rule="typing_surface") == []
    assert policy_scanner_suite.violations_for_rule(result, rule="runtime_narrowing_boundary") == []
    assert policy_scanner_suite.violations_for_rule(result, rule="aspf_normalization_idempotence") == []
    assert policy_scanner_suite.violations_for_rule(result, rule="boundary_core_contract") == []
    assert policy_scanner_suite.violations_for_rule(result, rule="fiber_normalization_contract") == []
    assert policy_scanner_suite.violations_for_rule(result, rule="test_subprocess_hygiene") == []
    assert policy_scanner_suite.violations_for_rule(result, rule="test_sleep_hygiene") == []


# gabion:behavior primary=desired
def test_policy_scanner_suite_flags_wide_orchestrator_primitive_barrel(tmp_path: Path) -> None:
    root = tmp_path
    _write(
        root / "src/gabion/server_core/command_orchestrator_primitives.py",
        "\n".join(["x = 1"] * 2401),
    )
    result = policy_scanner_suite.scan_policy_suite(root=root)
    violations = policy_scanner_suite.violations_for_rule(result, rule="orchestrator_primitive_barrel")
    assert violations
    assert any(item.get("kind") == "line_threshold" for item in violations)


# gabion:evidence E:call_footprint::tests/test_policy_scanner_suite.py::test_policy_scanner_suite_flags_typing_surface_and_respects_baseline_and_waivers::policy_scanner_suite.py::gabion.tooling.policy_scanner_suite.scan_policy_suite
# gabion:behavior primary=desired
def test_policy_scanner_suite_flags_typing_surface_and_respects_baseline_and_waivers(tmp_path: Path) -> None:
    root = tmp_path
    _write(
        root / "src/gabion/analysis/core/sample.py",
        "from typing import Any\n\ndef normalize(payload: dict[str, object], raw: Any, marker: object) -> None:\n    return None\n",
    )
    result = policy_scanner_suite.scan_policy_suite(root=root)
    violations = policy_scanner_suite.violations_for_rule(result, rule="typing_surface")
    assert len(violations) == 3

    baseline_path = root / "baselines/typing_surface_policy_baseline.json"
    baseline_path.parent.mkdir(parents=True, exist_ok=True)
    baseline_path.write_text(
        json.dumps({"version": 1, "violations": violations}, indent=2) + "\n",
        encoding="utf-8",
    )
    with_baseline = policy_scanner_suite.scan_policy_suite(root=root)
    assert policy_scanner_suite.violations_for_rule(with_baseline, rule="typing_surface") == []

    baseline_path.write_text(json.dumps({"version": 1, "violations": []}, indent=2) + "\n", encoding="utf-8")
    waivers_path = root / "baselines/typing_surface_policy_waivers.json"
    waivers_path.write_text(
        json.dumps(
            {
                "version": 1,
                "waivers": [
                    {
                        "path": "src/gabion/analysis/core/sample.py",
                        "qualname": "normalize",
                        "line": 3,
                        "kind": "dict_str_object_annotation",
                        "rationale": "legacy inbound payload shape",
                        "scope": "semantic_core",
                        "expiry": "2027-01-01",
                        "owner": "@gabion-core",
                    }
                ],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    with_waiver = policy_scanner_suite.scan_policy_suite(root=root)
    waiver_violations = policy_scanner_suite.violations_for_rule(with_waiver, rule="typing_surface")
    assert len(waiver_violations) == 2



# gabion:evidence E:call_footprint::tests/test_policy_scanner_suite.py::test_policy_scanner_suite_flags_invalid_typing_surface_waiver_metadata::policy_scanner_suite.py::gabion.tooling.policy_scanner_suite.scan_policy_suite
# gabion:behavior primary=verboten facets=invalid
def test_policy_scanner_suite_flags_invalid_typing_surface_waiver_metadata(tmp_path: Path) -> None:
    root = tmp_path
    _write(root / "src/gabion/analysis/core/sample.py", "from typing import Any\n\nvalue: Any = 'x'\n")
    waivers_path = root / "baselines/typing_surface_policy_waivers.json"
    waivers_path.parent.mkdir(parents=True, exist_ok=True)
    waivers_path.write_text(
        json.dumps({"version": 1, "waivers": [{"path": "src/gabion/analysis/core/sample.py"}]}, indent=2) + "\n",
        encoding="utf-8",
    )

    result = policy_scanner_suite.scan_policy_suite(root=root)
    violations = policy_scanner_suite.violations_for_rule(result, rule="typing_surface")
    assert any(item.get("kind") == "invalid_waiver" for item in violations)


# gabion:behavior primary=desired
def test_policy_scanner_suite_flags_runtime_narrowing_boundary_and_respects_baseline_and_waivers(tmp_path: Path) -> None:
    root = tmp_path
    _write(
        root / "src/gabion/analysis/core/sample.py",
        "from typing import cast\n\ndef normalize(payload: object) -> str:\n    if isinstance(payload, str):\n        return payload\n    return cast(str, payload)\n",
    )
    result = policy_scanner_suite.scan_policy_suite(root=root)
    violations = policy_scanner_suite.violations_for_rule(result, rule="runtime_narrowing_boundary")
    assert len(violations) == 2

    baseline_path = root / "baselines/runtime_narrowing_boundary_policy_baseline.json"
    baseline_path.parent.mkdir(parents=True, exist_ok=True)
    baseline_path.write_text(
        json.dumps({"version": 1, "violations": violations}, indent=2) + "\n",
        encoding="utf-8",
    )
    with_baseline = policy_scanner_suite.scan_policy_suite(root=root)
    assert policy_scanner_suite.violations_for_rule(with_baseline, rule="runtime_narrowing_boundary") == []

    baseline_path.write_text(json.dumps({"version": 1, "violations": []}, indent=2) + "\n", encoding="utf-8")
    waivers_path = root / "baselines/runtime_narrowing_boundary_policy_waivers.json"
    waivers_path.write_text(
        json.dumps(
            {
                "version": 1,
                "waivers": [
                    {
                        "path": "src/gabion/analysis/core/sample.py",
                        "qualname": "normalize",
                        "line": 4,
                        "kind": "isinstance_call",
                        "rationale": "legacy ingress shape normalization",
                        "scope": "semantic_core",
                        "expiry": "2027-01-01",
                        "owner": "@gabion-core",
                    }
                ],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    with_waiver = policy_scanner_suite.scan_policy_suite(root=root)
    waiver_violations = policy_scanner_suite.violations_for_rule(with_waiver, rule="runtime_narrowing_boundary")
    assert len(waiver_violations) == 1


# gabion:behavior primary=verboten facets=invalid
def test_policy_scanner_suite_flags_invalid_runtime_narrowing_boundary_waiver_metadata(tmp_path: Path) -> None:
    root = tmp_path
    _write(root / "src/gabion/analysis/core/sample.py", "def normalize(value):\n    return isinstance(value, str)\n")
    waivers_path = root / "baselines/runtime_narrowing_boundary_policy_waivers.json"
    waivers_path.parent.mkdir(parents=True, exist_ok=True)
    waivers_path.write_text(
        json.dumps({"version": 1, "waivers": [{"path": "src/gabion/analysis/core/sample.py"}]}, indent=2) + "\n",
        encoding="utf-8",
    )

    result = policy_scanner_suite.scan_policy_suite(root=root)
    violations = policy_scanner_suite.violations_for_rule(result, rule="runtime_narrowing_boundary")
    assert any(item.get("kind") == "invalid_waiver" for item in violations)


# gabion:behavior primary=desired
def test_policy_scanner_suite_flags_duplicate_pre_core_normalization_on_same_canonical_flow(
    tmp_path: Path,
) -> None:
    root = tmp_path
    trace_payload = {
        "trace_id": "aspf-trace:test",
        "one_cells": [
            {
                "source": "ingress:payload",
                "target": "boundary:carrier",
                "representative": "parse:start",
                "basis_path": ["ingress", "parse"],
                "kind": "boundary_parse",
                "surface": "",
                "metadata": {},
            },
            {
                "source": "ingress:payload",
                "target": "boundary:carrier",
                "representative": "parse:repeat",
                "basis_path": ["ingress", "parse"],
                "kind": "boundary_parse",
                "surface": "",
                "metadata": {},
            },
            {
                "source": "runtime:inputs",
                "target": "analysis:engine",
                "representative": "analyze_paths.start",
                "basis_path": ["analysis", "call", "start"],
                "kind": "analysis_call_start",
                "surface": "",
                "metadata": {},
            },
        ],
    }
    _write(
        root / "artifacts/out/aspf_trace.json",
        json.dumps(trace_payload, indent=2) + "\n",
    )

    result = policy_scanner_suite.scan_policy_suite(root=root)
    violations = policy_scanner_suite.violations_for_rule(
        result, rule="aspf_normalization_idempotence"
    )
    assert len(violations) == 1
    assert violations[0]["kind"] == "duplicate_normalization_class_pre_core"
    assert violations[0]["normalization_class"] == "parse"
    assert violations[0]["fiber_trace"]
    assert violations[0]["applicability_bounds"] is not None
    assert violations[0]["counterfactual_boundary"] is not None


# gabion:behavior primary=verboten facets=invalid
def test_policy_scanner_suite_flags_invalid_aspf_baseline_payload(
    tmp_path: Path,
) -> None:
    root = tmp_path
    _write(
        root / "baselines/aspf_normalization_idempotence_policy_baseline.json",
        json.dumps(
            {
                "version": 1,
                "violations": [
                    {
                        "path": "artifacts/out/aspf_trace.json",
                        "qualname": "flow:path:1.2.3",
                        "kind": "duplicate_normalization_class_pre_core",
                    },
                ],
            },
            indent=2,
        )
        + "\n",
    )

    result = policy_scanner_suite.scan_policy_suite(root=root)
    violations = policy_scanner_suite.violations_for_rule(
        result,
        rule="aspf_normalization_idempotence",
    )
    assert len(violations) == 1
    assert violations[0]["kind"] == "invalid_baseline_payload"


# gabion:behavior primary=desired
def test_policy_scanner_suite_serializes_fiber_normalization_diagnostics(
    tmp_path: Path,
) -> None:
    root = tmp_path
    _write(
        root / "src/gabion/example_boundary.py",
        "\n".join(
            [
                "# gabion:boundary_normalization_module",
                "import gabion.example_core as example_core",
                "",
                "def run_boundary(value: object) -> str:",
                "    if isinstance(value, str):",
                "        pass",
                "    if isinstance(value, str):",
                "        pass",
                "    return example_core.run_core(value)",
            ]
        )
        + "\n",
    )

    result = policy_scanner_suite.scan_policy_suite(root=root)
    violations = policy_scanner_suite.violations_for_rule(
        result, rule="fiber_normalization_contract"
    )
    assert len(violations) == 1
    assert violations[0]["kind"] == "duplicate_normalization_before_core"
    assert violations[0]["fiber_trace"]
    assert violations[0]["applicability_bounds"] is not None
    assert violations[0]["counterfactual_boundary"] is not None


# gabion:behavior primary=desired
def test_policy_scanner_suite_scopes_boundary_core_rule_to_changed_paths(
    tmp_path: Path,
) -> None:
    root = tmp_path
    _write(
        root / "src/gabion/example_boundary.py",
        "\n".join(
            [
                "# gabion:boundary_normalization_module",
                "import gabion.example_core as example_core",
                "",
                "def run_boundary(value: str) -> str:",
                "    return value",
            ]
        )
        + "\n",
    )
    _write(
        root / "src/gabion/example_core.py",
        "def run_core(value: str) -> str:\n    return value\n",
    )
    _write(root / "src/gabion/unrelated.py", "def ok():\n    return 1\n")
    unscoped_result = policy_scanner_suite.scan_policy_suite(root=root)
    assert policy_scanner_suite.violations_for_rule(
        unscoped_result,
        rule="boundary_core_contract",
    )
    scoped_result = policy_scanner_suite.scan_policy_suite(
        root=root,
        changed_paths={"src/gabion/unrelated.py"},
    )
    assert (
        policy_scanner_suite.violations_for_rule(
            scoped_result,
            rule="boundary_core_contract",
        )
        == []
    )


# gabion:evidence E:call_footprint::tests/test_policy_scanner_suite.py::test_policy_scanner_suite_carries_external_policy_results::policy_scanner_suite.py::gabion.tooling.policy_scanner_suite.scan_policy_suite
# gabion:behavior primary=desired
def test_policy_scanner_suite_carries_external_policy_results(tmp_path: Path) -> None:
    root = tmp_path
    artifact_path = root / "artifacts/out/policy_suite_results.json"
    policy_results = {
        "policy_check": {
            "rule_id": "policy_check",
            "status": "pass",
            "violations": [],
            "projection_fiber_semantics": {
                "decision": {"rule_id": "projection_fiber.convergence.ok"},
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
                    ]
                },
            },
        },
    }
    result = policy_scanner_suite.scan_policy_suite(
        root=root,
        child_inputs=policy_scanner_suite.PolicySuiteChildInputs(
            projection_fiber_semantics=policy_results["policy_check"][
                "projection_fiber_semantics"
            ],
        ),
    )
    semantics = result.projection_fiber_semantics
    assert semantics is not None
    assert semantics["report"]["compiled_projection_semantic_bundles"][0]["spec_name"] == (
        "projection_fiber_frontier"
    )
    payload = {
        "format_version": 1,
        "violations": result.violations_by_rule,
        "projection_fiber_semantics": semantics,
    }
    assert "policy_results" not in payload
    assert "cached" not in payload
    assert "generated_at_utc" not in payload
    assert "root" not in payload
    assert payload["projection_fiber_semantics"] == semantics
    assert "projection_fiber_semantics_summary" not in payload
    summary = projection_fiber_semantics_summary_from_payload(payload)
    assert summary is not None
    assert summary.decision["rule_id"] == "projection_fiber.convergence.ok"
    assert summary.semantic_row_count == 1
    assert summary.compiled_projection_semantic_bundle_count == 1
    assert list(summary.compiled_projection_semantic_spec_names) == [
        "projection_fiber_frontier"
    ]
    assert [item.as_payload() for item in summary.semantic_previews] == [
        {
            "spec_name": "projection_fiber_frontier",
            "quotient_face": "projection_fiber.frontier",
            "source_structural_identity": "row-1",
            "path": "src/gabion/example.py",
            "qualname": "example.frontier",
            "structural_path": "example.frontier::branch[0]",
            "obligation_state": "discharged",
            "complete": True,
        }
    ]

    cached = policy_scanner_suite.load_or_scan_policy_suite(
        root=root,
        artifact_path=artifact_path,
        child_inputs=policy_scanner_suite.PolicySuiteChildInputs(
            projection_fiber_semantics=policy_results["policy_check"][
                "projection_fiber_semantics"
            ],
        ),
    )
    assert cached.cached is False
    persisted_payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert "policy_results" not in persisted_payload
    assert "cached" not in persisted_payload
    assert "root" not in persisted_payload
    assert "decision" not in persisted_payload
    assert "generated_at_utc" not in persisted_payload
    assert "projection_fiber_semantics" not in persisted_payload

    cached_again = policy_scanner_suite.load_or_scan_policy_suite(
        root=root,
        artifact_path=artifact_path,
        child_inputs=policy_scanner_suite.PolicySuiteChildInputs(
            projection_fiber_semantics=policy_results["policy_check"][
                "projection_fiber_semantics"
            ],
        ),
    )
    assert cached_again.cached is True
    cached_summary = projection_fiber_semantics_summary_from_payload(
        {
            "format_version": 1,
            "violations": cached_again.result.violations_by_rule,
            "projection_fiber_semantics": (
                cached_again.result.projection_fiber_semantics
            ),
        }
    )
    assert cached_summary is not None
    assert cached_summary.decision["rule_id"] == "projection_fiber.convergence.ok"


def test_projection_fiber_semantics_summary_requires_canonical_payload_shape() -> None:
    assert projection_fiber_semantics_summary_from_payload(
        {
            "projection_fiber_semantics_summary": {
                "decision": {"rule_id": "projection_fiber.convergence.ok"},
                "semantic_row_count": 1,
                "compiled_projection_semantic_bundle_count": 1,
                "compiled_projection_semantic_spec_names": [
                    "projection_fiber_frontier"
                ],
                "semantic_previews": [
                    {
                        "spec_name": "projection_fiber_frontier",
                        "quotient_face": "projection_fiber.frontier",
                        "source_structural_identity": "row-1",
                        "path": "src/gabion/example.py",
                        "qualname": "example.frontier",
                        "structural_path": "example.frontier::branch[0]",
                        "obligation_state": "discharged",
                        "complete": True,
                    }
                ],
            }
        }
    ) is None
    assert projection_fiber_semantics_summary_from_payload(
        {
            "policy_results": {
                "policy_check": {
                    "projection_fiber_semantics": {
                        "decision": {"rule_id": "projection_fiber.convergence.ok"},
                        "report": {
                            "semantic_rows": [],
                            "compiled_projection_semantic_bundles": [],
                        },
                    }
                }
            }
        }
    ) is None
    assert projection_fiber_semantics_summary_from_payload(
        {
            "policy_check": {
                "projection_fiber_semantics": {
                    "decision": {"rule_id": "projection_fiber.convergence.ok"},
                    "report": {
                        "semantic_rows": [],
                        "compiled_projection_semantic_bundles": [],
                    },
                }
            }
        }
    ) is None

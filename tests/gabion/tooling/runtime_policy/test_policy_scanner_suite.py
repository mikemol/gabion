from __future__ import annotations

import json
from pathlib import Path

from gabion.tooling.runtime import policy_scanner_suite


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


# gabion:evidence E:call_footprint::tests/test_policy_scanner_suite.py::test_policy_scanner_suite_scan_and_cache::policy_scanner_suite.py::gabion.tooling.policy_scanner_suite.scan_policy_suite
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
    _write(root / "src/gabion/bad_syntax.py", "def broken(:\n")
    _write(root / "src/gabion/__pycache__/ignored.py", "def ignored():\n    return 1\n")

    artifact_path = root / "artifacts/out/policy_suite_results.json"
    first = policy_scanner_suite.load_or_scan_policy_suite(
        root=root,
        artifact_path=artifact_path,
    )
    assert first.cached is False
    assert first.total_violations() > 0
    assert policy_scanner_suite.violations_for_rule(first, rule="branchless")
    assert policy_scanner_suite.violations_for_rule(first, rule="defensive_fallback")
    assert policy_scanner_suite.violations_for_rule(first, rule="no_monkeypatch")
    assert policy_scanner_suite.violations_for_rule(first, rule="no_legacy_monolith_import")
    assert policy_scanner_suite.violations_for_rule(first, rule="orchestrator_primitive_barrel") == []
    assert policy_scanner_suite.violations_for_rule(first, rule="server_orchestrator_helper_duplicates") == []

    second = policy_scanner_suite.load_or_scan_policy_suite(
        root=root,
        artifact_path=artifact_path,
    )
    assert second.cached is True
    assert second.inventory_hash == first.inventory_hash
    assert second.rule_set_hash == first.rule_set_hash


# gabion:evidence E:call_footprint::tests/test_policy_scanner_suite.py::test_policy_scanner_suite_cache_invalidation_and_payload_normalization::policy_scanner_suite.py::gabion.tooling.policy_scanner_suite.scan_policy_suite
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
    assert normalized.violations_by_rule["branchless"] == []
    assert normalized.violations_by_rule["defensive_fallback"] == []
    assert normalized.violations_by_rule["no_monkeypatch"] == []
    assert normalized.violations_by_rule["no_legacy_monolith_import"] == []

    _write(
        root / "src/gabion/new_file.py",
        "def f():\n    if True:\n        return 1\n    return 0\n",
    )
    invalidated = policy_scanner_suite.load_or_scan_policy_suite(
        root=root,
        artifact_path=artifact_path,
    )
    assert invalidated.cached is False
    assert invalidated.inventory_hash != baseline.inventory_hash


# gabion:evidence E:call_footprint::tests/test_policy_scanner_suite.py::test_policy_scanner_suite_private_cache_and_payload_branches::policy_scanner_suite.py::gabion.tooling.policy_scanner_suite.scan_policy_suite
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
            }
        }
    )
    assert normalized["no_monkeypatch"] == []
    assert normalized["orchestrator_primitive_barrel"] == []
    assert normalized["server_orchestrator_helper_duplicates"] == []


# gabion:evidence E:call_footprint::tests/test_policy_scanner_suite.py::test_policy_scanner_suite_scan_with_explicit_nonstandard_files::policy_scanner_suite.py::gabion.tooling.policy_scanner_suite.scan_policy_suite
def test_policy_scanner_suite_scan_with_explicit_nonstandard_files(tmp_path: Path) -> None:
    root = tmp_path
    external_file = root / "external.py"
    _write(external_file, "def utility():\n    return 1\n")

    result = policy_scanner_suite.scan_policy_suite(
        root=root,
        files=(external_file.resolve(),),
    )
    assert result.total_violations() == 0


# gabion:evidence E:call_footprint::tests/test_policy_scanner_suite.py::test_policy_scanner_suite_flags_retired_monolith_module_file::policy_scanner_suite.py::gabion.tooling.policy_scanner_suite.scan_policy_suite
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
def test_policy_scanner_suite_respects_branch_and_fallback_baselines(tmp_path: Path) -> None:
    root = tmp_path
    _write(
        root / "src/gabion/branch_sample.py",
        "def branchy(flag):\n    if flag:\n        return 1\n    return 0\n",
    )
    _write(
        root / "src/gabion/fallback_sample.py",
        "def normalize(value):\n    if value is None:\n        return None\n    return value\n",
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

    result = policy_scanner_suite.scan_policy_suite(root=root)
    assert policy_scanner_suite.violations_for_rule(result, rule="branchless") == []
    assert policy_scanner_suite.violations_for_rule(result, rule="defensive_fallback") == []
    assert policy_scanner_suite.violations_for_rule(result, rule="no_monkeypatch") == []
    assert policy_scanner_suite.violations_for_rule(result, rule="no_legacy_monolith_import") == []
    assert policy_scanner_suite.violations_for_rule(result, rule="orchestrator_primitive_barrel") == []
    assert policy_scanner_suite.violations_for_rule(result, rule="server_orchestrator_helper_duplicates") == []


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



def test_policy_scanner_suite_flags_duplicate_server_orchestrator_helpers(tmp_path: Path) -> None:
    root = tmp_path
    _write(
        root / "src/gabion/server.py",
        "def _normalize_dataflow_response(response):\n    return response\n",
    )
    _write(
        root / "src/gabion/server_core/command_orchestrator_primitives.py",
        "def _normalize_dataflow_response(response):\n    return response\n",
    )

    result = policy_scanner_suite.scan_policy_suite(root=root)
    violations = policy_scanner_suite.violations_for_rule(
        result,
        rule="server_orchestrator_helper_duplicates",
    )
    assert len(violations) == 1
    assert violations[0]["kind"] == "duplicate_helper"
    assert violations[0]["qualname"] == "_normalize_dataflow_response"

from __future__ import annotations

import json
from pathlib import Path

from gabion.tooling import policy_scanner_suite


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

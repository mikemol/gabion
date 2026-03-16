from __future__ import annotations

from pathlib import Path

from gabion.tooling.policy_rules import defensive_fallback_rule as policy
from gabion.tooling.runtime.policy_scan_batch import build_policy_scan_batch


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


# gabion:evidence E:call_footprint::tests/test_defensive_fallback_policy_check.py::test_defensive_fallback_policy_flags_sentinel_return_guard::defensive_fallback_policy_check.py::scripts.defensive_fallback_policy_check.collect_violations

# gabion:behavior primary=allowed_unwanted facets=fallback
def test_defensive_fallback_policy_flags_sentinel_return_guard(tmp_path: Path) -> None:
    _write(
        tmp_path / "src" / "gabion" / "sample.py",
        "def fn(value):\n"
        "    if value is None:\n"
        "        return None\n"
        "    return value\n",
    )
    batch = build_policy_scan_batch(root=tmp_path, target_globs=(policy.TARGET_GLOB,))
    violations = policy.collect_violations(batch=batch)
    assert violations
    assert any(v.kind == "sentinel_return" for v in violations)


# gabion:evidence E:call_footprint::tests/test_defensive_fallback_policy_check.py::test_defensive_fallback_policy_allows_boundary_marker::defensive_fallback_policy_check.py::scripts.defensive_fallback_policy_check.run

# gabion:behavior primary=allowed_unwanted facets=fallback
def test_defensive_fallback_policy_allows_boundary_marker(tmp_path: Path) -> None:
    _write(
        tmp_path / "src" / "gabion" / "sample.py",
        "# gabion:boundary_normalization\n"
        "def _normalize_sample(value):\n"
        "    if value is None:\n"
        "        return None\n"
        "    return value\n",
    )
    assert policy.run(root=tmp_path) == 0


# gabion:evidence E:call_footprint::tests/test_defensive_fallback_policy_check.py::test_defensive_fallback_policy_baseline_write_and_ratchet::defensive_fallback_policy_check.py::scripts.defensive_fallback_policy_check.run

# gabion:behavior primary=allowed_unwanted facets=fallback
def test_defensive_fallback_policy_baseline_write_and_ratchet(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.json"
    _write(
        tmp_path / "src" / "gabion" / "sample.py",
        "def fn(value):\n"
        "    if value is None:\n"
        "        return None\n"
        "    return value\n",
    )

    assert policy.run(root=tmp_path, baseline=baseline, baseline_write=True) == 0
    assert baseline.exists()
    assert policy.run(root=tmp_path, baseline=baseline) == 0

    _write(
        tmp_path / "src" / "gabion" / "extra.py",
        "def f(value):\n"
        "    if isinstance(value, str):\n"
        "        return ''\n"
        "    return value\n",
    )
    assert policy.run(root=tmp_path, baseline=baseline) == 1

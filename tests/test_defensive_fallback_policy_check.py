from __future__ import annotations

from pathlib import Path

from scripts import defensive_fallback_policy_check as policy


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


# gabion:evidence E:call_footprint::tests/test_defensive_fallback_policy_check.py::test_defensive_fallback_policy_flags_sentinel_return_guard::defensive_fallback_policy_check.py::scripts.defensive_fallback_policy_check.collect_violations

def test_defensive_fallback_policy_flags_sentinel_return_guard(tmp_path: Path) -> None:
    _write(
        tmp_path / "src" / "gabion" / "sample.py",
        "def fn(value):\n"
        "    if value is None:\n"
        "        return None\n"
        "    return value\n",
    )
    violations = policy.collect_violations(root=tmp_path)
    assert violations
    assert any(v.kind == "sentinel_return" for v in violations)


# gabion:evidence E:call_footprint::tests/test_defensive_fallback_policy_check.py::test_defensive_fallback_policy_allows_boundary_marker::defensive_fallback_policy_check.py::scripts.defensive_fallback_policy_check.run

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

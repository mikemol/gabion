from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _write_profile(path: Path, *, total_elapsed_ns: int, site_elapsed_ns: int) -> None:
    payload = {
        "checks_total": 10,
        "total_elapsed_ns": total_elapsed_ns,
        "unattributed_elapsed_ns": 0,
        "sites": [
            {
                "path": "src/gabion/server.py",
                "qual": "gabion.server._execute_command_total",
                "check_count": 5,
                "elapsed_between_checks_ns": site_elapsed_ns,
                "max_gap_ns": site_elapsed_ns,
            }
        ],
        "edges": [],
        "io": [
            {
                "name": "analysis_resume.checkpoint_write",
                "event_count": 2,
                "elapsed_ns": 40_000_000,
                "max_event_ns": 25_000_000,
                "bytes_total": 8192,
            }
        ],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


# gabion:evidence E:call_footprint::tests/test_deadline_profile_ci_summary.py::test_deadline_profile_ci_summary_allows_missing_local::test_deadline_profile_ci_summary.py::tests.test_deadline_profile_ci_summary._write_profile
def test_deadline_profile_ci_summary_allows_missing_local(tmp_path: Path) -> None:
    ci_profile = tmp_path / "deadline_profile_ci.json"
    _write_profile(ci_profile, total_elapsed_ns=2_000_000_000, site_elapsed_ns=200_000_000)
    summary_json = tmp_path / "summary.json"
    summary_md = tmp_path / "summary.md"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/deadline_profile_ci_summary.py",
            "--ci-profile",
            str(ci_profile),
            "--local-profile",
            str(tmp_path / "missing_local.json"),
            "--allow-missing-local",
            "--json-out",
            str(summary_json),
            "--md-out",
            str(summary_md),
        ],
        check=False,
        cwd=Path(__file__).resolve().parents[1],
    )
    assert result.returncode == 0
    assert summary_json.exists()
    assert summary_md.exists()
    assert "Local profile not provided" in summary_md.read_text()
    assert "Top CI I/O" in summary_md.read_text()


# gabion:evidence E:call_footprint::tests/test_deadline_profile_ci_summary.py::test_deadline_profile_ci_summary_compares_local_profile::test_deadline_profile_ci_summary.py::tests.test_deadline_profile_ci_summary._write_profile
def test_deadline_profile_ci_summary_compares_local_profile(tmp_path: Path) -> None:
    ci_profile = tmp_path / "deadline_profile_ci.json"
    local_profile = tmp_path / "deadline_profile_local.json"
    _write_profile(ci_profile, total_elapsed_ns=2_000_000_000, site_elapsed_ns=200_000_000)
    _write_profile(local_profile, total_elapsed_ns=1_000_000_000, site_elapsed_ns=100_000_000)
    summary_json = tmp_path / "summary.json"
    summary_md = tmp_path / "summary.md"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/deadline_profile_ci_summary.py",
            "--ci-profile",
            str(ci_profile),
            "--local-profile",
            str(local_profile),
            "--json-out",
            str(summary_json),
            "--md-out",
            str(summary_md),
        ],
        check=False,
        cwd=Path(__file__).resolve().parents[1],
    )
    assert result.returncode == 0
    summary = json.loads(summary_json.read_text())
    comparison = summary.get("comparison")
    assert isinstance(comparison, dict)
    ratio = comparison.get("total_elapsed_ratio")
    assert isinstance(ratio, float)
    assert ratio > 1.0
    assert "CI vs Local" in summary_md.read_text()
    assert "Top CI I/O Regressions vs Local" in summary_md.read_text()

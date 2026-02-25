from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _write_payload(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_nonerasability_policy_check_blocks_silent_deletion(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.json"
    current = tmp_path / "current.json"
    _write_payload(
        baseline,
        {
            "deprecated_fibers": [
                {
                    "fiber_id": "aspf:a/b",
                    "canonical_aspf_path": ["a", "b"],
                    "lifecycle": "active",
                    "blocker_payload": [
                        {"blocker_id": "B1", "kind": "owner", "summary": "needs owner"}
                    ],
                }
            ]
        },
    )
    _write_payload(current, {"deprecated_fibers": []})

    result = subprocess.run(
        [
            sys.executable,
            "scripts/deprecated_nonerasability_policy_check.py",
            "--baseline",
            str(baseline),
            "--current",
            str(current),
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert "erased without explicit resolution metadata" in result.stdout


def test_nonerasability_policy_check_allows_resolved_lifecycle(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.json"
    current = tmp_path / "current.json"
    _write_payload(
        baseline,
        {
            "deprecated_fibers": [
                {
                    "fiber_id": "aspf:a/b",
                    "canonical_aspf_path": ["a", "b"],
                    "lifecycle": "resolved",
                    "resolution_metadata": {"ticket": "ABC-123"},
                }
            ]
        },
    )
    _write_payload(current, {"deprecated_fibers": []})

    result = subprocess.run(
        [
            sys.executable,
            "scripts/deprecated_nonerasability_policy_check.py",
            "--baseline",
            str(baseline),
            "--current",
            str(current),
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0

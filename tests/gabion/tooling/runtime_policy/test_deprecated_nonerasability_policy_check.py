from __future__ import annotations

import json
from pathlib import Path

from scripts.policy import deprecated_nonerasability_policy_check


def _write_payload(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


# gabion:evidence E:function_site::tests/test_deprecated_nonerasability_policy_check.py::tests.test_deprecated_nonerasability_policy_check.test_nonerasability_policy_check_blocks_silent_deletion
# gabion:behavior primary=allowed_unwanted facets=deprecated
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

    observed: list[str] = []
    result = deprecated_nonerasability_policy_check.main(
        [
            "--baseline",
            str(baseline),
            "--current",
            str(current),
        ],
        print_fn=observed.append,
    )
    assert result == 1
    assert any("erased without explicit resolution metadata" in line for line in observed)


# gabion:evidence E:function_site::tests/test_deprecated_nonerasability_policy_check.py::tests.test_deprecated_nonerasability_policy_check.test_nonerasability_policy_check_allows_resolved_lifecycle
# gabion:behavior primary=allowed_unwanted facets=deprecated
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

    result = deprecated_nonerasability_policy_check.main(
        [
            "--baseline",
            str(baseline),
            "--current",
            str(current),
        ]
    )
    assert result == 0


# gabion:evidence E:function_site::tests/test_deprecated_nonerasability_policy_check.py::tests.test_deprecated_nonerasability_policy_check.test_nonerasability_policy_check_writes_policy_result_output
# gabion:behavior primary=allowed_unwanted facets=deprecated
def test_nonerasability_policy_check_writes_policy_result_output(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.json"
    current = tmp_path / "current.json"
    output = tmp_path / "out/nonerasability.json"
    _write_payload(baseline, {"deprecated_fibers": []})
    _write_payload(current, {"deprecated_fibers": []})
    result = deprecated_nonerasability_policy_check.main(
        [
            "--baseline",
            str(baseline),
            "--current",
            str(current),
            "--output",
            str(output),
        ]
    )
    assert result == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["rule_id"] == "deprecated_nonerasability"
    assert payload["status"] == "pass"

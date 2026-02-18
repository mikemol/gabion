from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import (
    ambiguity_delta_gate,
    annotation_drift_orphaned_gate,
    obsolescence_delta_gate,
    obsolescence_delta_unmapped_gate,
)


@pytest.mark.parametrize(
    ("gate", "name"),
    [
        (ambiguity_delta_gate.check_gate, "Ambiguity delta artifact missing"),
        (
            annotation_drift_orphaned_gate.check_gate,
            "Annotation drift delta artifact missing",
        ),
        (
            obsolescence_delta_gate.check_gate,
            "Test obsolescence delta artifact missing",
        ),
        (
            obsolescence_delta_unmapped_gate.check_gate,
            "Test obsolescence delta artifact missing",
        ),
    ],
)
def test_delta_gates_fail_for_missing_artifact(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], gate, name: str
) -> None:
    missing = tmp_path / "missing.json"
    assert gate(missing, enabled=True) == 1
    assert name in capsys.readouterr().out


@pytest.mark.parametrize(
    "gate",
    [
        ambiguity_delta_gate.check_gate,
        annotation_drift_orphaned_gate.check_gate,
        obsolescence_delta_gate.check_gate,
        obsolescence_delta_unmapped_gate.check_gate,
    ],
)
def test_delta_gates_fail_for_malformed_artifact(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], gate
) -> None:
    payload_path = tmp_path / "delta.json"
    payload_path.write_text("not-json")
    assert gate(payload_path, enabled=True) == 1
    assert "artifact malformed" in capsys.readouterr().out


def test_ambiguity_gate_failure_output_has_before_current_delta(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    payload_path = tmp_path / "ambiguity.json"
    payload_path.write_text(
        json.dumps(
            {
                "summary": {
                    "total": {
                        "baseline": 3,
                        "current": 5,
                        "delta": 2,
                    }
                }
            }
        )
    )
    assert ambiguity_delta_gate.check_gate(payload_path, enabled=True) == 1
    out = capsys.readouterr().out
    assert "before=3" in out
    assert "current=5" in out
    assert "delta=+2" in out


def test_annotation_drift_gate_failure_output_has_before_current_delta(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    payload_path = tmp_path / "annotation.json"
    payload_path.write_text(
        json.dumps(
            {
                "summary": {
                    "baseline": {"orphaned": 1},
                    "current": {"orphaned": 2},
                    "delta": {"orphaned": 1},
                }
            }
        )
    )
    assert annotation_drift_orphaned_gate.check_gate(payload_path, enabled=True) == 1
    out = capsys.readouterr().out
    assert "before=1" in out
    assert "current=2" in out
    assert "delta=+1" in out


def test_obsolescence_gates_failure_output_has_before_current_delta(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    payload_path = tmp_path / "obsolescence.json"
    payload = {
        "summary": {
            "opaque_evidence": {"baseline": 2, "current": 4, "delta": 2},
            "counts": {
                "baseline": {"unmapped": 1},
                "current": {"unmapped": 2},
                "delta": {"unmapped": 1},
            },
        }
    }
    payload_path.write_text(json.dumps(payload))

    assert obsolescence_delta_gate.check_gate(payload_path, enabled=True) == 1
    out = capsys.readouterr().out
    assert "before=2" in out
    assert "current=4" in out
    assert "delta=+2" in out

    assert obsolescence_delta_unmapped_gate.check_gate(payload_path, enabled=True) == 1
    out = capsys.readouterr().out
    assert "before=1" in out
    assert "current=2" in out
    assert "delta=+1" in out


@pytest.mark.parametrize(
    ("enabled_fn", "false_value"),
    [
        (ambiguity_delta_gate._enabled, "0"),
        (annotation_drift_orphaned_gate._enabled, "false"),
        (obsolescence_delta_gate._enabled, "no"),
        (obsolescence_delta_unmapped_gate._enabled, "off"),
    ],
)
def test_gate_enabled_defaults_on_and_can_be_disabled(enabled_fn, false_value: str) -> None:
    assert enabled_fn(None) is True
    assert enabled_fn("") is True
    assert enabled_fn(false_value) is False

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import pytest

from scripts import ambiguity_delta_gate
from scripts import annotation_drift_orphaned_gate
from scripts import obsolescence_delta_gate
from scripts import obsolescence_delta_unmapped_gate


@pytest.mark.parametrize(
    ("gate_fn", "payload"),
    [
        (ambiguity_delta_gate.check_gate, {"summary": {"total": {"delta": 1}}}),
        (
            annotation_drift_orphaned_gate.check_gate,
            {"summary": {"delta": {"orphaned": 1}}},
        ),
        (obsolescence_delta_gate.check_gate, {"summary": {"opaque_evidence": {"delta": 1}}}),
        (
            obsolescence_delta_unmapped_gate.check_gate,
            {"summary": {"counts": {"delta": {"unmapped": 1}}}},
        ),
    ],
)
def test_gate_detects_positive_delta(
    tmp_path: Path,
    gate_fn: Callable[..., int],
    payload: dict[str, object],
) -> None:
    delta_path = tmp_path / "delta.json"
    delta_path.write_text(json.dumps(payload), encoding="utf-8")
    assert gate_fn(delta_path, enabled=True) == 1


@pytest.mark.parametrize(
    "gate_fn",
    [
        ambiguity_delta_gate.check_gate,
        annotation_drift_orphaned_gate.check_gate,
        obsolescence_delta_gate.check_gate,
        obsolescence_delta_unmapped_gate.check_gate,
    ],
)
def test_gate_fails_when_payload_missing(tmp_path: Path, gate_fn: Callable[..., int]) -> None:
    delta_path = tmp_path / "missing.json"
    assert gate_fn(delta_path, enabled=True) == 2


@pytest.mark.parametrize(
    "gate_fn",
    [
        ambiguity_delta_gate.check_gate,
        annotation_drift_orphaned_gate.check_gate,
        obsolescence_delta_gate.check_gate,
        obsolescence_delta_unmapped_gate.check_gate,
    ],
)
def test_gate_fails_when_payload_malformed(tmp_path: Path, gate_fn: Callable[..., int]) -> None:
    delta_path = tmp_path / "delta.json"
    delta_path.write_text("{not-json", encoding="utf-8")
    assert gate_fn(delta_path, enabled=True) == 2


@pytest.mark.parametrize(
    "enabled_fn",
    [
        ambiguity_delta_gate._enabled,
        annotation_drift_orphaned_gate._enabled,
        obsolescence_delta_gate._enabled,
        obsolescence_delta_unmapped_gate._enabled,
    ],
)
def test_gate_default_is_enabled(enabled_fn: Callable[[str | None], bool]) -> None:
    assert enabled_fn(None) is True
    assert enabled_fn("0") is False

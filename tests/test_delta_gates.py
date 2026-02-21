from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import pytest

from gabion.tooling import ambiguity_delta_gate
from gabion.tooling import annotation_drift_orphaned_gate
from gabion.tooling import obsolescence_delta_gate
from gabion.tooling import obsolescence_delta_unmapped_gate
from tests.env_helpers import env_scope


# gabion:evidence E:function_site::tests/test_delta_gates.py::test_gate_detects_positive_delta
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


# gabion:evidence E:function_site::tests/test_delta_gates.py::test_gate_fails_when_payload_missing
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


# gabion:evidence E:function_site::tests/test_delta_gates.py::test_gate_fails_when_payload_malformed
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


# gabion:evidence E:function_site::tests/test_delta_gates.py::test_gate_default_is_enabled
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


# gabion:evidence E:call_footprint::tests/test_delta_gates.py::test_gate_enabled_reads_environment_and_disabled_paths::ambiguity_delta_gate.py::gabion.tooling.ambiguity_delta_gate._enabled::obsolescence_delta_gate.py::gabion.tooling.obsolescence_delta_gate._enabled
@pytest.mark.parametrize(
    ("module", "env_flag"),
    [
        (ambiguity_delta_gate, ambiguity_delta_gate.ENV_FLAG),
        (annotation_drift_orphaned_gate, annotation_drift_orphaned_gate.ENV_FLAG),
        (obsolescence_delta_gate, obsolescence_delta_gate.ENV_FLAG),
        (obsolescence_delta_unmapped_gate, obsolescence_delta_unmapped_gate.ENV_FLAG),
    ],
)
def test_gate_enabled_reads_environment_and_disabled_paths(
    tmp_path: Path,
    module: object,
    env_flag: str,
) -> None:
    gate_fn = getattr(module, "check_gate")
    enabled_fn = getattr(module, "_enabled")
    with env_scope({env_flag: "off"}):
        assert enabled_fn() is False
        assert gate_fn(tmp_path / "missing.json") == 0
    with env_scope({env_flag: "true"}):
        assert enabled_fn() is True


# gabion:evidence E:call_footprint::tests/test_delta_gates.py::test_gate_delta_value_handles_invalid_shapes::ambiguity_delta_gate.py::gabion.tooling.ambiguity_delta_gate._delta_value::obsolescence_delta_unmapped_gate.py::gabion.tooling.obsolescence_delta_unmapped_gate._delta_value
@pytest.mark.parametrize(
    ("module", "key_path"),
    [
        (ambiguity_delta_gate, ("summary", "total", "delta")),
        (annotation_drift_orphaned_gate, ("summary", "delta", "orphaned")),
        (obsolescence_delta_gate, ("summary", "opaque_evidence", "delta")),
        (obsolescence_delta_unmapped_gate, ("summary", "counts", "delta", "unmapped")),
    ],
)
def test_gate_delta_value_handles_invalid_shapes(
    module: object,
    key_path: tuple[str, ...],
) -> None:
    delta_value = getattr(module, "_delta_value")
    assert delta_value({}) == 0
    assert delta_value({"summary": []}) == 0
    if key_path[1] == "counts":
        assert delta_value({"summary": {"counts": []}}) == 0
        assert delta_value({"summary": {"counts": {"delta": []}}}) == 0
        assert delta_value({"summary": {"counts": {"delta": {key_path[-1]: "bad"}}}}) == 0
    elif key_path[1] == "total":
        assert delta_value({"summary": {"total": []}}) == 0
        assert delta_value({"summary": {"total": {"delta": "bad"}}}) == 0
    elif key_path[1] == "opaque_evidence":
        assert delta_value({"summary": {"opaque_evidence": []}}) == 0
        assert delta_value({"summary": {"opaque_evidence": {"delta": "bad"}}}) == 0
    else:
        assert delta_value({"summary": {"delta": []}}) == 0
        assert delta_value({"summary": {"delta": {key_path[-1]: "bad"}}}) == 0

from __future__ import annotations

from gabion.analysis.aspf_mutation_log import (
    AspfMutationRecord,
    shadow_replay_equivalence,
    snapshot_state,
)


def test_shadow_replay_equivalence_passes_for_matching_live_state() -> None:
    snapshot = snapshot_state({"a": 1}, seq=1)
    tail = [AspfMutationRecord(op_id="2", op_kind="set", payload={"key": "b", "value": 2})]
    result = shadow_replay_equivalence(live_state={"a": 1, "b": 2}, snapshot=snapshot, tail=tail)
    assert result.equivalent is True
    assert result.tail_length == 1


def test_shadow_replay_equivalence_detects_divergence() -> None:
    snapshot = snapshot_state({"a": 1}, seq=1)
    tail = [AspfMutationRecord(op_id="2", op_kind="delete", payload={"key": "a"})]
    result = shadow_replay_equivalence(live_state={"a": 1}, snapshot=snapshot, tail=tail)
    assert result.equivalent is False

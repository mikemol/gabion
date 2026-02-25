from __future__ import annotations

from gabion.analysis.aspf_mutation_log import (
    AspfMutationRecord,
    apply_mutation,
    shadow_replay_equivalence,
    snapshot_state,
)


# gabion:evidence E:function_site::tests/test_aspf_mutation_log.py::tests.test_aspf_mutation_log.test_shadow_replay_equivalence_passes_for_matching_live_state
def test_shadow_replay_equivalence_passes_for_matching_live_state() -> None:
    snapshot = snapshot_state({"a": 1}, seq=1)
    tail = [AspfMutationRecord(op_id="2", op_kind="set", payload={"key": "b", "value": 2})]
    result = shadow_replay_equivalence(live_state={"a": 1, "b": 2}, snapshot=snapshot, tail=tail)
    assert result.equivalent is True
    assert result.tail_length == 1


# gabion:evidence E:function_site::tests/test_aspf_mutation_log.py::tests.test_aspf_mutation_log.test_shadow_replay_equivalence_detects_divergence
def test_shadow_replay_equivalence_detects_divergence() -> None:
    snapshot = snapshot_state({"a": 1}, seq=1)
    tail = [AspfMutationRecord(op_id="2", op_kind="delete", payload={"key": "a"})]
    result = shadow_replay_equivalence(live_state={"a": 1}, snapshot=snapshot, tail=tail)
    assert result.equivalent is False


# gabion:evidence E:function_site::tests/test_aspf_mutation_log.py::tests.test_aspf_mutation_log.test_apply_mutation_unknown_ops_and_empty_keys
def test_apply_mutation_unknown_ops_and_empty_keys() -> None:
    state = {"a": 1, "_unknown_ops": "legacy"}

    no_set = apply_mutation(
        state,
        AspfMutationRecord(op_id="1", op_kind="set", payload={"key": "", "value": 2}),
    )
    assert no_set["a"] == 1

    no_delete = apply_mutation(
        no_set,
        AspfMutationRecord(op_id="2", op_kind="delete", payload={"key": ""}),
    )
    assert no_delete["a"] == 1

    unknown = apply_mutation(
        no_delete,
        AspfMutationRecord(op_id="3", op_kind="noop", payload={}),
    )
    assert unknown["_unknown_ops"] == ["noop"]

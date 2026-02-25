# gabion:decision_protocol_module
# gabion:boundary_normalization_module
from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Mapping

from gabion.analysis.json_types import JSONObject, JSONValue


@dataclass(frozen=True)
class AspfMutationRecord:
    op_id: str
    op_kind: str
    payload: JSONObject


@dataclass(frozen=True)
class AspfMutationSnapshot:
    seq: int
    state: JSONObject


@dataclass(frozen=True)
class AspfShadowReplayResult:
    equivalent: bool
    expected: JSONObject
    replayed: JSONObject
    tail_length: int


def apply_mutation(state: JSONObject, record: AspfMutationRecord) -> JSONObject:
    next_state: JSONObject = dict(state)
    if record.op_kind == "set":
        key = str(record.payload.get("key", ""))
        if key:
            next_state[key] = record.payload.get("value")
    elif record.op_kind == "delete":
        key = str(record.payload.get("key", ""))
        if key:
            next_state.pop(key, None)
    else:
        ops = list(next_state.get("_unknown_ops") or []) if isinstance(next_state.get("_unknown_ops"), list) else []
        ops.append(record.op_kind)
        next_state["_unknown_ops"] = ops
    return next_state


def snapshot_state(state: Mapping[str, JSONValue], *, seq: int) -> AspfMutationSnapshot:
    return AspfMutationSnapshot(seq=seq, state=dict(state))


def replay_tail(snapshot: AspfMutationSnapshot, tail: list[AspfMutationRecord]) -> JSONObject:
    state = dict(snapshot.state)
    for record in tail:
        state = apply_mutation(state, record)
    return state


def shadow_replay_equivalence(
    *,
    live_state: Mapping[str, JSONValue],
    snapshot: AspfMutationSnapshot,
    tail: list[AspfMutationRecord],
) -> AspfShadowReplayResult:
    replayed = replay_tail(snapshot, tail)
    expected = dict(live_state)
    equivalent = json.dumps(expected, sort_keys=True) == json.dumps(replayed, sort_keys=True)
    return AspfShadowReplayResult(
        equivalent=equivalent,
        expected=expected,
        replayed=replayed,
        tail_length=len(tail),
    )

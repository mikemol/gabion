# gabion:decision_protocol_module
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

from gabion.analysis.foundation.json_types import JSONObject


@dataclass(frozen=True)
class CallAmbiguitiesEmitDeps:
    check_deadline_fn: Callable[[], None]
    normalize_snapshot_path_fn: Callable[..., str]
    normalize_targets_fn: Callable[[list[dict[str, str]]], list[dict[str, str]]]
    never_fn: Callable[..., None]
    call_candidate_target_site_fn: Callable[..., object]
    add_interned_alt_fn: Callable[..., None]
    make_ambiguity_set_key_fn: Callable[..., JSONObject]
    normalize_key_fn: Callable[[JSONObject], JSONObject]
    make_partition_witness_key_fn: Callable[..., JSONObject]
    key_identity_fn: Callable[[JSONObject], str]


def emit_call_ambiguities(
    ambiguities: Iterable[object],
    *,
    project_root,
    forest: object,
    deps: CallAmbiguitiesEmitDeps,
) -> list[JSONObject]:
    deps.check_deadline_fn()
    entries: list[JSONObject] = []
    for entry in ambiguities:
        deps.check_deadline_fn()
        call_span = entry.call.span if entry.call is not None else None
        site_path = deps.normalize_snapshot_path_fn(entry.caller.path, project_root)
        site_payload: JSONObject = {
            "path": site_path,
            "function": entry.caller.qual,
        }
        if call_span is not None:
            site_payload["span"] = list(call_span)
        candidate_targets: list[dict[str, str]] = []
        for candidate in entry.candidates:
            deps.check_deadline_fn()
            candidate_targets.append(
                {
                    "path": deps.normalize_snapshot_path_fn(candidate.path, project_root),
                    "qual": candidate.qual,
                }
            )
        candidate_targets = deps.normalize_targets_fn(candidate_targets)
        payload: JSONObject = {
            "kind": entry.kind,
            "site": site_payload,
            "candidates": candidate_targets,
            "candidate_count": len(candidate_targets),
            "phase": entry.phase,
        }
        entries.append(payload)
        if call_span is None:
            deps.never_fn(
                "call ambiguity requires span",
                path=site_path,
                qual=entry.caller.qual,
                kind=entry.kind,
                phase=entry.phase,
            )
        suite_id = forest.add_suite_site(
            entry.caller.path.name,
            entry.caller.qual,
            "call",
            span=call_span,
        )
        ambiguity_key = deps.make_ambiguity_set_key_fn(
            path=site_path,
            qual=entry.caller.qual,
            span=call_span,
            candidates=candidate_targets,
        )
        ambiguity_key = deps.normalize_key_fn(ambiguity_key)
        for candidate in entry.candidates:
            deps.check_deadline_fn()
            candidate_id = deps.call_candidate_target_site_fn(
                forest=forest,
                candidate=candidate,
            )
            deps.add_interned_alt_fn(
                forest=forest,
                kind="CallCandidate",
                inputs=(suite_id, candidate_id),
                evidence={
                    "kind": entry.kind,
                    "phase": entry.phase,
                    "ambiguity_key": ambiguity_key,
                },
            )
        witness_key = deps.make_partition_witness_key_fn(
            kind=entry.kind,
            site=ambiguity_key.get("site", {}),
            ambiguity=ambiguity_key,
            support={
                "phase": entry.phase,
                "reason": "multiple local candidates",
            },
            collapse={
                "hint": "add explicit qualifier or disambiguating annotation",
            },
        )
        witness_key = deps.normalize_key_fn(witness_key)
        witness_identity = deps.key_identity_fn(witness_key)
        witness_node = forest.add_node(
            "PartitionWitness",
            (witness_identity,),
            meta={"evidence_key": witness_key},
        )
        deps.add_interned_alt_fn(
            forest=forest,
            kind="PartitionWitness",
            inputs=(suite_id, witness_node),
            evidence={
                "kind": entry.kind,
                "phase": entry.phase,
            },
        )
    return entries

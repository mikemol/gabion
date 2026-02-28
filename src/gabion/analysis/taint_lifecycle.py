# gabion:decision_protocol_module
from __future__ import annotations

from collections.abc import Mapping, Sequence

from gabion.analysis.baseline_io import attach_spec_metadata
from gabion.analysis.projection_registry import (
    QUOTIENT_DEMOTION_INCIDENTS_SPEC,
    QUOTIENT_PROMOTION_DECISION_SPEC,
    QUOTIENT_PROTOCOL_READINESS_SPEC,
)
from gabion.analysis.resume_codec import mapping_or_none, sequence_or_none
from gabion.analysis.taint_projection import (
    TaintProfile,
    TaintStatus,
    normalize_taint_profile,
)
from gabion.json_types import JSONObject, JSONValue
from gabion.order_contract import sort_once


_READINESS_GATES: dict[TaintProfile, tuple[str, ...]] = {
    TaintProfile.OBSERVE: ("state_present",),
    TaintProfile.CONTAIN: ("state_present", "no_illegal_locus"),
    TaintProfile.ENFORCE: (
        "state_present",
        "no_illegal_locus",
        "no_missing_witness",
        "no_strict_unresolved",
        "no_expired_exemption",
    ),
}

_NEXT_PROFILE = {
    TaintProfile.OBSERVE: TaintProfile.CONTAIN,
    TaintProfile.CONTAIN: TaintProfile.ENFORCE,
    TaintProfile.ENFORCE: TaintProfile.ENFORCE,
}


def build_readiness_payload(
    *,
    taint_state_payload: Mapping[str, JSONValue],
    current_profile: object = TaintProfile.OBSERVE,
) -> JSONObject:
    state_summary = mapping_or_none(taint_state_payload.get("summary")) or {}
    by_status = mapping_or_none(state_summary.get("by_status")) or {}
    current = normalize_taint_profile(current_profile)
    readiness_rows = [
        _profile_readiness_row(
            profile=profile,
            by_status=by_status,
            state_payload=taint_state_payload,
        )
        for profile in TaintProfile
    ]
    active_row = next(
        row for row in readiness_rows if row["profile"] == current.value
    )
    payload: JSONObject = {
        "profile": current.value,
        "profiles": readiness_rows,
        "active_profile_ready": bool(active_row.get("ready", False)),
    }
    return attach_spec_metadata(payload, spec=QUOTIENT_PROTOCOL_READINESS_SPEC)


def build_promotion_decision_payload(
    *,
    readiness_payload: Mapping[str, JSONValue],
    current_profile: object = TaintProfile.OBSERVE,
) -> JSONObject:
    current = normalize_taint_profile(current_profile)
    target = _NEXT_PROFILE[current]
    readiness_rows = sequence_or_none(readiness_payload.get("profiles")) or ()
    readiness_by_profile: dict[str, Mapping[str, JSONValue]] = {}
    for row in readiness_rows:
        payload = mapping_or_none(row)
        if payload is not None:
            readiness_by_profile[str(payload.get("profile", ""))] = payload
    target_row = readiness_by_profile.get(target.value, {})
    current_row = readiness_by_profile.get(current.value, {})
    target_ready = bool(target_row.get("ready", False))
    current_ready = bool(current_row.get("ready", False))
    if target_ready and current is not target:
        decision = "promote"
        reason_codes = ["target_profile_ready", f"from_{current.value}", f"to_{target.value}"]
    elif not current_ready and current is not TaintProfile.OBSERVE:
        decision = "demote"
        reason_codes = ["current_profile_not_ready", f"from_{current.value}"]
    else:
        decision = "hold"
        reason_codes = ["target_not_ready"]
    payload: JSONObject = {
        "decision": decision,
        "current_profile": current.value,
        "target_profile": target.value,
        "reason_codes": sort_once(
            reason_codes,
            source="taint_lifecycle.build_promotion_decision_payload.reason_codes",
        ),
    }
    return attach_spec_metadata(payload, spec=QUOTIENT_PROMOTION_DECISION_SPEC)


def build_demotion_incidents_payload(
    *,
    taint_state_payload: Mapping[str, JSONValue],
    current_profile: object = TaintProfile.ENFORCE,
) -> JSONObject:
    summary = mapping_or_none(taint_state_payload.get("summary")) or {}
    by_status = mapping_or_none(summary.get("by_status")) or {}
    profile = normalize_taint_profile(current_profile)
    incidents: list[JSONObject] = []
    for status in (
        TaintStatus.ILLEGAL_LOCUS.value,
        TaintStatus.MISSING_WITNESS.value,
        TaintStatus.UNRESOLVED.value,
        TaintStatus.EXPIRED_EXEMPTION.value,
    ):
        count = int(by_status.get(status, 0) or 0)
        if count <= 0:
            continue
        incidents.append(
            {
                "trigger_id": f"taint_status:{status}",
                "impacted_profile": profile.value,
                "count": count,
                "recovery_requirements": _recovery_requirements(status),
                "closure_status": "open",
            }
        )
    payload: JSONObject = {
        "current_profile": profile.value,
        "incidents": sort_once(
            incidents,
            source="taint_lifecycle.build_demotion_incidents_payload.incidents",
            key=lambda row: str(row.get("trigger_id", "")),
        ),
    }
    return attach_spec_metadata(payload, spec=QUOTIENT_DEMOTION_INCIDENTS_SPEC)


def _profile_readiness_row(
    *,
    profile: TaintProfile,
    by_status: Mapping[str, object],
    state_payload: Mapping[str, JSONValue],
) -> JSONObject:
    gates = _READINESS_GATES[profile]
    gate_rows: list[JSONObject] = []
    for gate_id in gates:
        gate_rows.append(
            {
                "gate_id": gate_id,
                "passed": _evaluate_gate(gate_id=gate_id, by_status=by_status, state_payload=state_payload),
            }
        )
    ready = all(bool(row.get("passed")) for row in gate_rows)
    return {
        "profile": profile.value,
        "ready": ready,
        "gates": gate_rows,
    }


def _evaluate_gate(
    *,
    gate_id: str,
    by_status: Mapping[str, object],
    state_payload: Mapping[str, JSONValue],
) -> bool:
    if gate_id == "state_present":
        return bool(state_payload.get("taint_records") is not None)
    if gate_id == "no_illegal_locus":
        return int(by_status.get(TaintStatus.ILLEGAL_LOCUS.value, 0) or 0) == 0
    if gate_id == "no_missing_witness":
        return int(by_status.get(TaintStatus.MISSING_WITNESS.value, 0) or 0) == 0
    if gate_id == "no_strict_unresolved":
        return int(by_status.get(TaintStatus.UNRESOLVED.value, 0) or 0) == 0
    if gate_id == "no_expired_exemption":
        return int(by_status.get(TaintStatus.EXPIRED_EXEMPTION.value, 0) or 0) == 0
    return False


def _recovery_requirements(status: str) -> list[str]:
    if status == TaintStatus.ILLEGAL_LOCUS.value:
        return ["register_legal_boundary_locus", "revalidate_boundary_owner_expiry"]
    if status == TaintStatus.MISSING_WITNESS.value:
        return ["emit_erasure_witness", "attach_policy_basis_and_justification_code"]
    if status == TaintStatus.EXPIRED_EXEMPTION.value:
        return ["renew_or_remove_exemption", "revalidate_expiry_window"]
    return ["resolve_strict_ingress_taint"]

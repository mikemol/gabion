# gabion:decision_protocol_module
from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from enum import StrEnum
from hashlib import sha1
import json
from typing import cast

from gabion.analysis.resume_codec import mapping_or_none, sequence_or_none
from gabion.analysis.timeout_context import check_deadline
from gabion.json_types import JSONObject, JSONValue
from gabion.order_contract import sort_once
from gabion.runtime import stable_encode


class TaintColor(StrEnum):
    PERMITTED = "taint_permitted"
    FORBIDDEN = "taint_forbidden"


class TaintKind(StrEnum):
    BOOLEAN_AMBIGUITY = "boolean_ambiguity"
    TYPE_AMBIGUITY = "type_ambiguity"
    CONTROL_AMBIGUITY = "control_ambiguity"


class TaintStatus(StrEnum):
    RESOLVED = "resolved"
    UNRESOLVED = "unresolved"
    MISSING_WITNESS = "missing_witness"
    ILLEGAL_LOCUS = "illegal_locus"
    EXPIRED_EXEMPTION = "expired_exemption"
    EXEMPT = "exempt"


class TaintProfile(StrEnum):
    OBSERVE = "observe"
    CONTAIN = "contain"
    ENFORCE = "enforce"


_TAINT_KIND_ALIASES = {
    "boolean": TaintKind.BOOLEAN_AMBIGUITY,
    "boolean_ambiguity": TaintKind.BOOLEAN_AMBIGUITY,
    "type": TaintKind.TYPE_AMBIGUITY,
    "type_ambiguity": TaintKind.TYPE_AMBIGUITY,
    "control": TaintKind.CONTROL_AMBIGUITY,
    "control_ambiguity": TaintKind.CONTROL_AMBIGUITY,
}

_PROFILE_ALIASES = {
    "observe": TaintProfile.OBSERVE,
    "experimental": TaintProfile.OBSERVE,
    "contain": TaintProfile.CONTAIN,
    "boundary": TaintProfile.CONTAIN,
    "enforce": TaintProfile.ENFORCE,
    "strict-core": TaintProfile.ENFORCE,
    "strict_core": TaintProfile.ENFORCE,
}

_MARKER_TO_TAINT_KIND = {
    "todo": TaintKind.CONTROL_AMBIGUITY,
    "deprecated": TaintKind.TYPE_AMBIGUITY,
    "never": TaintKind.CONTROL_AMBIGUITY,
}

_STRICT_BLOCKING_STATUSES = {
    TaintStatus.UNRESOLVED.value,
    TaintStatus.MISSING_WITNESS.value,
    TaintStatus.ILLEGAL_LOCUS.value,
    TaintStatus.EXPIRED_EXEMPTION.value,
}

_DIAG_UNKNOWN_TAINT_KIND_TAG = "unknown_taint_kind_tag"
_DIAG_MISSING_WITNESS_FIELD_PREFIX = "missing_witness_field"
_DIAG_BOUNDARY_LOCUS_MISSING = "boundary_locus_missing"
_DIAG_BOUNDARY_KIND_FORBIDDEN = "boundary_taint_kind_forbidden"
_DIAG_EXPIRY_VIOLATION = "expiry_violation"


@dataclass(frozen=True)
class TaintBoundaryLocus:
    boundary_id: str
    suite_id: str
    allowed_taint_kinds: tuple[TaintKind, ...]
    owner: str = ""
    reason: str = ""
    expiry: str = ""

    def is_expired(self, *, today: object = None) -> bool:
        if not self.expiry.strip():
            return False
        return _date_from_iso(self.expiry) < _normalized_today(today)


@dataclass(frozen=True)
class TaintErasureWitness:
    taint_kind: TaintKind
    source_suite_id: str
    target_suite_id: str
    eraser_id: str
    input_shape: str
    output_shape: str
    policy_basis: str
    justification_code: str
    owner: str = ""
    expiry: str = ""
    notes: str = ""

    def identity_payload(self) -> JSONObject:
        return {
            "taint_kind": self.taint_kind.value,
            "source_suite_id": self.source_suite_id,
            "target_suite_id": self.target_suite_id,
            "eraser_id": self.eraser_id,
            "input_shape": self.input_shape,
            "output_shape": self.output_shape,
            "policy_basis": self.policy_basis,
            "justification_code": self.justification_code,
        }

    @property
    def witness_id(self) -> str:
        canonical = stable_encode.stable_compact_text(self.identity_payload())
        digest = sha1(canonical.encode("utf-8")).hexdigest()[:16]
        return f"taint-witness:{digest}"


@dataclass(frozen=True)
class TaintKindResolutionOutcome:
    taint_kind: TaintKind
    diagnostic_codes: tuple[str, ...]


@dataclass(frozen=True)
class TaintWitnessBuildOutcome:
    witness: object
    diagnostic_codes: tuple[str, ...]


def normalize_taint_profile(value: object) -> TaintProfile:
    key = str(value or TaintProfile.OBSERVE.value).strip().lower()
    return _PROFILE_ALIASES.get(key, TaintProfile.OBSERVE)


def normalize_taint_kind(value: object, *, default: TaintKind) -> TaintKind:
    return resolve_taint_kind(value, default=default).taint_kind


def resolve_taint_kind(
    value: object,
    *,
    default: TaintKind,
) -> TaintKindResolutionOutcome:
    key = str(value or "").strip().lower()
    if not key:
        return TaintKindResolutionOutcome(
            taint_kind=default,
            diagnostic_codes=(),
        )
    resolved = _TAINT_KIND_ALIASES.get(key)
    if resolved is None:
        return TaintKindResolutionOutcome(
            taint_kind=default,
            diagnostic_codes=(_DIAG_UNKNOWN_TAINT_KIND_TAG,),
        )
    return TaintKindResolutionOutcome(
        taint_kind=resolved,
        diagnostic_codes=(),
    )


def parse_taint_boundary_registry(payload: object) -> tuple[TaintBoundaryLocus, ...]:
    entries = _boundary_entries_payload(payload)
    loci: list[TaintBoundaryLocus] = []
    for entry in entries:
        check_deadline()
        boundary_id = str(entry.get("boundary_id", "") or "").strip()
        suite_id = str(entry.get("suite_id", "") or "").strip()
        if not boundary_id or not suite_id:
            continue
        allowed_raw = _sequence_payload(entry.get("allowed_taint_kinds"))
        allowed = tuple(
            normalize_taint_kind(raw_kind, default=TaintKind.CONTROL_AMBIGUITY)
            for raw_kind in allowed_raw
            if str(raw_kind).strip()
        )
        dedup_allowed = tuple(
            dict.fromkeys(allowed)
        )
        loci.append(
            TaintBoundaryLocus(
                boundary_id=boundary_id,
                suite_id=suite_id,
                allowed_taint_kinds=dedup_allowed,
                owner=str(entry.get("owner", "") or "").strip(),
                reason=str(entry.get("reason", "") or "").strip(),
                expiry=str(entry.get("expiry", "") or "").strip(),
            )
        )
    return tuple(
        sort_once(
            loci,
            source="taint_projection.parse_taint_boundary_registry.loci",
            key=lambda locus: (
                locus.suite_id,
                locus.boundary_id,
            ),
        )
    )


def project_taint_ledgers(
    *,
    marker_rows: Iterable[Mapping[str, object]],
    boundary_registry: Sequence[TaintBoundaryLocus] = (),
    profile: object = TaintProfile.OBSERVE,
    today: object = None,
) -> tuple[list[JSONObject], list[JSONObject]]:
    normalized_profile = normalize_taint_profile(profile)
    boundary_by_id = {entry.boundary_id: entry for entry in boundary_registry}
    boundary_by_suite = {entry.suite_id: entry for entry in boundary_registry}
    records: list[JSONObject] = []
    witnesses: dict[str, JSONObject] = {}
    for raw_entry in marker_rows:
        check_deadline()
        entry = {str(key): raw_entry[key] for key in raw_entry}
        site_payload = _mapping_payload(entry.get("site"))
        site_path = str(site_payload.get("path", "") or "")
        site_function = str(site_payload.get("function", "") or "")
        source_suite_id = str(site_payload.get("suite_id", "") or "")
        if not source_suite_id:
            source_suite_id = _derived_suite_id(
                path=site_path,
                function=site_function,
            )
        links = _normalize_links(entry.get("links"))
        link_tags = _semantic_link_tags(links)
        marker_kind = str(entry.get("marker_kind", "never") or "never").strip().lower()
        kind_outcome = resolve_taint_kind(
            link_tags.get("taint_kind", ""),
            default=_MARKER_TO_TAINT_KIND.get(marker_kind, TaintKind.CONTROL_AMBIGUITY),
        )
        taint_kind = kind_outcome.taint_kind
        target_suite_id = (
            link_tags.get("target_suite_id", "").strip() or source_suite_id
        )
        boundary_id = link_tags.get("boundary_id", "").strip()
        boundary = boundary_by_id.get(boundary_id) if boundary_id else None
        if boundary is None:
            boundary = boundary_by_suite.get(source_suite_id)
            if boundary is not None:
                boundary_id = boundary.boundary_id
        expiry = str(entry.get("expiry", "") or "").strip()
        if boundary is not None and not expiry:
            expiry = boundary.expiry
        witness_outcome = _build_witness_outcome(
            taint_kind=taint_kind,
            source_suite_id=source_suite_id,
            target_suite_id=target_suite_id,
            marker_entry=entry,
            link_tags=link_tags,
            expiry=expiry,
        )
        witness = witness_outcome.witness
        diagnostic_codes = _sorted_diagnostic_codes(
            (
                *kind_outcome.diagnostic_codes,
                *witness_outcome.diagnostic_codes,
                *_boundary_diagnostic_codes(
                    profile=normalized_profile,
                    taint_kind=taint_kind,
                    boundary=boundary,
                ),
                *_expiry_diagnostic_codes(expiry=expiry, today=today),
            )
        )
        status = _status_for_entry(
            profile=normalized_profile,
            taint_kind=taint_kind,
            witness=witness,
            boundary=boundary,
            expiry=expiry,
            today=today,
            diagnostic_codes=diagnostic_codes,
        )
        witness_id = ""
        if witness is not None:
            witness_payload = _witness_payload(witness)
            witnesses[witness.witness_id] = witness_payload
            witness_id = witness.witness_id
        record_payload = _record_payload(
            marker_entry=entry,
            taint_kind=taint_kind,
            source_suite_id=source_suite_id,
            target_suite_id=target_suite_id,
            boundary_id=boundary_id,
            status=status,
            witness_id=witness_id,
            expiry=expiry,
            diagnostic_codes=diagnostic_codes,
        )
        records.append(record_payload)
    ordered_records = sort_once(
        records,
        source="taint_projection.project_taint_ledgers.records",
        key=lambda row: (
            str(row.get("status", "")),
            str(row.get("taint_kind", "")),
            str(row.get("source_suite_id", "")),
            str(row.get("target_suite_id", "")),
            str(row.get("record_id", "")),
        ),
    )
    ordered_witnesses = sort_once(
        list(witnesses.values()),
        source="taint_projection.project_taint_ledgers.witnesses",
        key=lambda row: str(row.get("witness_id", "")),
    )
    return ordered_records, ordered_witnesses


def build_taint_summary(
    records: Iterable[Mapping[str, object]],
) -> JSONObject:
    by_status: dict[str, int] = {}
    by_kind: dict[str, int] = {}
    by_diagnostic_code: dict[str, int] = {}
    records_with_diagnostics = 0
    strict_unresolved = 0
    total = 0
    for row in records:
        check_deadline()
        total += 1
        status = str(row.get("status", "") or "")
        taint_kind = str(row.get("taint_kind", "") or "")
        by_status[status] = by_status.get(status, 0) + 1
        by_kind[taint_kind] = by_kind.get(taint_kind, 0) + 1
        diagnostic_codes_payload = sequence_or_none(
            row.get("diagnostic_codes"),
            allow_str=False,
        ) or ()
        normalized_codes = _sorted_diagnostic_codes(
            str(code) for code in diagnostic_codes_payload
        )
        if normalized_codes:
            records_with_diagnostics += 1
        for code in normalized_codes:
            by_diagnostic_code[code] = by_diagnostic_code.get(code, 0) + 1
        if status in _STRICT_BLOCKING_STATUSES:
            strict_unresolved += 1
    summary: JSONObject = {
        "total": total,
        "strict_unresolved": strict_unresolved,
        "by_status": {
            key: by_status[key]
            for key in sort_once(
                by_status,
                source="taint_projection.build_taint_summary.by_status",
            )
        },
        "by_kind": {
            key: by_kind[key]
            for key in sort_once(
                by_kind,
                source="taint_projection.build_taint_summary.by_kind",
            )
        },
    }
    if by_diagnostic_code:
        summary["diagnostics"] = {
            "records_with_diagnostics": records_with_diagnostics,
            "by_code": {
                key: by_diagnostic_code[key]
                for key in sort_once(
                    by_diagnostic_code,
                    source="taint_projection.build_taint_summary.by_diagnostic_code",
                )
            },
        }
    return summary


def boundary_payloads(
    boundary_registry: Sequence[TaintBoundaryLocus],
) -> list[JSONObject]:
    rows: list[JSONObject] = []
    for locus in boundary_registry:
        check_deadline()
        rows.append(
            {
                "boundary_id": locus.boundary_id,
                "suite_id": locus.suite_id,
                "allowed_taint_kinds": [kind.value for kind in locus.allowed_taint_kinds],
                "owner": locus.owner,
                "reason": locus.reason,
                "expiry": locus.expiry,
            }
        )
    return sort_once(
        rows,
        source="taint_projection.boundary_payloads.rows",
        key=lambda row: (
            str(row.get("suite_id", "")),
            str(row.get("boundary_id", "")),
        ),
    )


def _boundary_entries_payload(payload: object) -> list[Mapping[str, object]]:
    payload_mapping = mapping_or_none(payload)
    if payload_mapping is not None:
        entries = sequence_or_none(payload_mapping.get("boundaries"), allow_str=False) or ()
        return _mapping_entries(entries)
    entries = sequence_or_none(payload, allow_str=False) or ()
    return _mapping_entries(entries)


def _date_from_iso(value: str) -> date:
    text = str(value or "").strip()
    try:
        return date.fromisoformat(text)
    except ValueError:
        return date.max


def _sequence_payload(value: object) -> tuple[object, ...]:
    sequence = sequence_or_none(value, allow_str=False) or ()
    return tuple(sequence)


def _mapping_payload(value: object) -> Mapping[str, object]:
    mapping = mapping_or_none(value)
    if mapping is None:
        return {}
    return {str(key): cast(object, mapping[key]) for key in mapping}


def _normalize_links(value: object) -> tuple[JSONObject, ...]:
    entries = _sequence_payload(value)
    links: list[JSONObject] = []
    for item in entries:
        check_deadline()
        payload = _mapping_payload(item)
        kind = str(payload.get("kind", "") or "").strip().lower()
        raw_value = str(payload.get("value", "") or "").strip()
        if kind and raw_value:
            links.append({"kind": kind, "value": raw_value})
    return tuple(
        sort_once(
            links,
            source="taint_projection._normalize_links.links",
            key=lambda row: (
                str(row.get("kind", "")),
                str(row.get("value", "")),
            ),
        )
    )


def _semantic_link_tags(links: Sequence[Mapping[str, object]]) -> dict[str, str]:
    tags: dict[str, str] = {}
    policy_ids: list[str] = []
    for link in links:
        check_deadline()
        kind = str(link.get("kind", "") or "").strip().lower()
        value = str(link.get("value", "") or "").strip()
        if not kind or not value:
            continue
        if kind == "policy_id":
            policy_ids.append(value)
        if ":" not in value:
            continue
        tag_key, _, tag_value = value.partition(":")
        key = tag_key.strip().lower()
        normalized_value = tag_value.strip()
        if key and normalized_value:
            tags.setdefault(key, normalized_value)
    if policy_ids:
        tags.setdefault("policy_basis", policy_ids[0])
    return tags


def _derived_suite_id(*, path: str, function: str) -> str:
    payload = {"path": path, "function": function}
    digest = sha1(
        stable_encode.stable_compact_text(payload).encode("utf-8")
    ).hexdigest()[:16]
    return f"suite:derived:{digest}"


def _build_witness_outcome(
    *,
    taint_kind: TaintKind,
    source_suite_id: str,
    target_suite_id: str,
    marker_entry: Mapping[str, object],
    link_tags: Mapping[str, str],
    expiry: str,
) -> TaintWitnessBuildOutcome:
    policy_basis = str(link_tags.get("policy_basis", "") or "").strip()
    justification_code = str(
        link_tags.get("justification_code", "")
        or link_tags.get("justification", "")
        or ""
    ).strip()
    missing_fields: list[str] = []
    if not policy_basis:
        missing_fields.append("policy_basis")
    if not justification_code:
        missing_fields.append("justification_code")
    if missing_fields:
        return TaintWitnessBuildOutcome(
            witness=None,
            diagnostic_codes=_sorted_diagnostic_codes(
                _missing_witness_field_code(field_name)
                for field_name in missing_fields
            ),
        )
    eraser_id = str(
        link_tags.get("eraser_id", "")
        or marker_entry.get("marker_id", "")
        or marker_entry.get("marker_site_id", "")
        or ""
    ).strip()
    input_shape = str(
        link_tags.get("input_shape", "") or marker_entry.get("reason", "") or "unknown_input"
    ).strip()
    output_shape = str(link_tags.get("output_shape", "") or "strict_contract").strip()
    return TaintWitnessBuildOutcome(
        witness=TaintErasureWitness(
            taint_kind=taint_kind,
            source_suite_id=source_suite_id,
            target_suite_id=target_suite_id,
            eraser_id=eraser_id,
            input_shape=input_shape,
            output_shape=output_shape,
            policy_basis=policy_basis,
            justification_code=justification_code,
            owner=str(marker_entry.get("owner", "") or "").strip(),
            expiry=expiry,
            notes=str(link_tags.get("notes", "") or "").strip(),
        ),
        diagnostic_codes=(),
    )


def _status_for_entry(
    *,
    profile: TaintProfile,
    taint_kind: TaintKind,
    witness: object,
    boundary: object,
    expiry: str,
    today: object,
    diagnostic_codes: Sequence[str],
) -> TaintStatus:
    witness_payload = _witness_from_payload(witness)
    boundary_payload = _boundary_from_payload(boundary)
    diagnostics = set(diagnostic_codes)
    if _DIAG_EXPIRY_VIOLATION in diagnostics:
        return TaintStatus.EXPIRED_EXEMPTION
    if profile is TaintProfile.OBSERVE:
        if _DIAG_UNKNOWN_TAINT_KIND_TAG in diagnostics:
            return TaintStatus.UNRESOLVED
        return TaintStatus.RESOLVED if witness_payload is not None else TaintStatus.UNRESOLVED
    if (
        _DIAG_BOUNDARY_LOCUS_MISSING in diagnostics
        or _DIAG_BOUNDARY_KIND_FORBIDDEN in diagnostics
    ):
        return TaintStatus.ILLEGAL_LOCUS
    if boundary_payload is None:
        return TaintStatus.ILLEGAL_LOCUS
    if witness_payload is None or _has_missing_witness_diagnostics(diagnostic_codes):
        return TaintStatus.MISSING_WITNESS
    if _DIAG_UNKNOWN_TAINT_KIND_TAG in diagnostics:
        return TaintStatus.UNRESOLVED
    return TaintStatus.RESOLVED


def _witness_payload(witness: TaintErasureWitness) -> JSONObject:
    payload: JSONObject = dict(witness.identity_payload())
    payload.update(
        {
            "witness_id": witness.witness_id,
            "owner": witness.owner,
            "expiry": witness.expiry,
            "notes": witness.notes,
        }
    )
    return payload


def _record_payload(
    *,
    marker_entry: Mapping[str, object],
    taint_kind: TaintKind,
    source_suite_id: str,
    target_suite_id: str,
    boundary_id: str,
    status: TaintStatus,
    witness_id: str,
    expiry: str,
    diagnostic_codes: Sequence[str],
) -> JSONObject:
    marker_id = str(marker_entry.get("marker_id", "") or "").strip()
    marker_kind = str(marker_entry.get("marker_kind", "") or "").strip()
    normalized_codes = _sorted_diagnostic_codes(diagnostic_codes)
    identity_payload = {
        "taint_kind": taint_kind.value,
        "source_suite_id": source_suite_id,
        "target_suite_id": target_suite_id,
        "boundary_id": boundary_id,
        "status": status.value,
        "witness_id": witness_id,
        "marker_id": marker_id,
        "diagnostic_codes": list(normalized_codes),
    }
    encoded = json.dumps(identity_payload, sort_keys=True, separators=(",", ":"))
    record_id = f"taint-record:{sha1(encoded.encode('utf-8')).hexdigest()[:16]}"
    return {
        "record_id": record_id,
        "taint_kind": taint_kind.value,
        "source_color": TaintColor.PERMITTED.value,
        "target_color": TaintColor.FORBIDDEN.value,
        "source_suite_id": source_suite_id,
        "target_suite_id": target_suite_id,
        "boundary_id": boundary_id,
        "status": status.value,
        "witness_id": witness_id,
        "marker_id": marker_id,
        "marker_kind": marker_kind,
        "owner": str(marker_entry.get("owner", "") or "").strip(),
        "expiry": expiry,
        "diagnostic_codes": list(normalized_codes),
        "diagnostic_count": len(normalized_codes),
    }


def _mapping_entries(payload: Sequence[object]) -> list[Mapping[str, object]]:
    entries: list[Mapping[str, object]] = []
    for item in payload:
        check_deadline()
        mapping = mapping_or_none(item)
        if mapping is not None:
            entries.append({str(key): cast(object, mapping[key]) for key in mapping})
    return entries


def _witness_from_payload(payload: object) -> object:
    match payload:
        case TaintErasureWitness() as witness:
            return witness
        case _:
            return None


def _boundary_from_payload(payload: object) -> object:
    match payload:
        case TaintBoundaryLocus() as boundary:
            return boundary
        case _:
            return None


def _boundary_diagnostic_codes(
    *,
    profile: TaintProfile,
    taint_kind: TaintKind,
    boundary: object,
) -> tuple[str, ...]:
    if profile is TaintProfile.OBSERVE:
        return ()
    boundary_payload = _boundary_from_payload(boundary)
    if boundary_payload is None:
        return (_DIAG_BOUNDARY_LOCUS_MISSING,)
    if (
        boundary_payload.allowed_taint_kinds
        and taint_kind not in boundary_payload.allowed_taint_kinds
    ):
        return (_DIAG_BOUNDARY_KIND_FORBIDDEN,)
    return ()


def _expiry_diagnostic_codes(
    *,
    expiry: str,
    today: object,
) -> tuple[str, ...]:
    expiry_date = _date_from_iso(expiry)
    if expiry.strip() and expiry_date < _normalized_today(today):
        return (_DIAG_EXPIRY_VIOLATION,)
    return ()


def _missing_witness_field_code(field_name: str) -> str:
    return f"{_DIAG_MISSING_WITNESS_FIELD_PREFIX}:{field_name}"


def _has_missing_witness_diagnostics(diagnostic_codes: Sequence[str]) -> bool:
    return any(
        str(code).startswith(f"{_DIAG_MISSING_WITNESS_FIELD_PREFIX}:")
        for code in diagnostic_codes
    )


def _sorted_diagnostic_codes(codes: Iterable[object]) -> tuple[str, ...]:
    normalized_codes = [
        str(code).strip()
        for code in codes
        if str(code).strip()
    ]
    if not normalized_codes:
        return ()
    deduped = dict.fromkeys(normalized_codes)
    return tuple(
        sort_once(
            deduped,
            source="taint_projection._sorted_diagnostic_codes.codes",
        )
    )


def _normalized_today(payload: object) -> date:
    match payload:
        case date() as day:
            return day
        case _:
            return date.today()

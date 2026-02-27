# gabion:boundary_normalization_module
# gabion:decision_protocol_module
# gabion:ambiguity_boundary_module
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Mapping, Sequence, cast

from gabion.json_types import JSONObject, JSONValue
from gabion.order_contract import sort_once
from gabion.runtime.stable_encode import stable_compact_text

from . import aspf_resume_state
from .aspf_core import (
    AspfOneCell,
    AspfTwoCellWitness,
    BasisZeroCell,
    parse_2cell_witness,
    validate_2cell_compatibility,
)
from .aspf_decision_surface import (
    RepresentativeSelectionMode,
    RepresentativeSelectionOptions,
    classify_drift_by_homotopy,
    select_representative,
)
from .aspf_morphisms import (
    AspfPrimeBasis,
    CofibrationWitnessCarrier,
    DomainPrimeBasis,
    DomainToAspfCofibration,
    DomainToAspfCofibrationEntry,
)

DEFAULT_PHASE1_SEMANTIC_SURFACES: tuple[str, ...] = (
    "groups_by_path",
    "decision_surfaces",
    "rewrite_plans",
    "synthesis_plan",
    "delta_state",
    "delta_payload",
    "violation_summary",
)

_TRACE_FORMAT_VERSION = 1
_EQUIVALENCE_FORMAT_VERSION = 1
_OPPORTUNITY_FORMAT_VERSION = 1
_STATE_FORMAT_VERSION = 1


@dataclass(frozen=True)
class AspfTraceControls:
    aspf_trace_json: Path | None
    aspf_import_trace: tuple[Path, ...]
    aspf_equivalence_against: tuple[Path, ...]
    aspf_opportunities_json: Path | None
    aspf_state_json: Path | None
    aspf_import_state: tuple[Path, ...]
    aspf_delta_jsonl: Path | None
    aspf_semantic_surface: tuple[str, ...]

    def enabled(self) -> bool:
        return bool(
            self.aspf_trace_json is not None
            or self.aspf_import_trace
            or self.aspf_equivalence_against
            or self.aspf_opportunities_json is not None
            or self.aspf_state_json is not None
            or self.aspf_import_state
            or self.aspf_delta_jsonl is not None
            or self.aspf_semantic_surface != DEFAULT_PHASE1_SEMANTIC_SURFACES
        )


@dataclass
class AspfExecutionTraceState:
    trace_id: str
    controls: AspfTraceControls
    started_at_utc: str
    command_profile: str
    one_cells: list[AspfOneCell] = field(default_factory=list)
    one_cell_metadata: list[JSONObject] = field(default_factory=list)
    two_cell_witnesses: list[AspfTwoCellWitness] = field(default_factory=list)
    cofibrations: list[CofibrationWitnessCarrier] = field(default_factory=list)
    surface_representatives: dict[str, str] = field(default_factory=dict)
    imported_trace_payloads: list[JSONObject] = field(default_factory=list)
    delta_records: list[JSONObject] = field(default_factory=list)


@dataclass(frozen=True)
class AspfFinalizationArtifacts:
    trace_payload: JSONObject
    equivalence_payload: JSONObject
    opportunities_payload: JSONObject
    delta_ledger_payload: JSONObject
    trace_path: Path
    equivalence_path: Path
    opportunities_path: Path
    delta_jsonl_path: Path
    state_payload: JSONObject | None = None
    state_path: Path | None = None


def controls_from_payload(payload: Mapping[str, object]) -> AspfTraceControls:
    return AspfTraceControls(
        aspf_trace_json=_optional_path(payload.get("aspf_trace_json")),
        aspf_import_trace=_path_sequence(payload.get("aspf_import_trace")),
        aspf_equivalence_against=_path_sequence(payload.get("aspf_equivalence_against")),
        aspf_opportunities_json=_optional_path(payload.get("aspf_opportunities_json")),
        aspf_state_json=_optional_path(payload.get("aspf_state_json")),
        aspf_import_state=_path_sequence(payload.get("aspf_import_state")),
        aspf_delta_jsonl=_optional_path(payload.get("aspf_delta_jsonl")),
        aspf_semantic_surface=_semantic_surface_sequence(
            payload.get("aspf_semantic_surface")
        ),
    )


def start_execution_trace(
    *,
    root: Path,
    payload: Mapping[str, object],
) -> AspfExecutionTraceState | None:
    controls = controls_from_payload(payload)
    if not controls.enabled():
        return None
    trace_id_payload = stable_compact_text(
        {
            "root": str(root),
            "pid": int(os.getpid()),
            "controls": {
                "aspf_trace_json": (
                    str(controls.aspf_trace_json)
                    if controls.aspf_trace_json is not None
                    else None
                ),
                "aspf_import_trace": [str(path) for path in controls.aspf_import_trace],
                "aspf_equivalence_against": [
                    str(path) for path in controls.aspf_equivalence_against
                ],
                "aspf_opportunities_json": (
                    str(controls.aspf_opportunities_json)
                    if controls.aspf_opportunities_json is not None
                    else None
                ),
                "aspf_state_json": (
                    str(controls.aspf_state_json)
                    if controls.aspf_state_json is not None
                    else None
                ),
                "aspf_import_state": [str(path) for path in controls.aspf_import_state],
                "aspf_delta_jsonl": (
                    str(controls.aspf_delta_jsonl)
                    if controls.aspf_delta_jsonl is not None
                    else None
                ),
                "aspf_semantic_surface": list(controls.aspf_semantic_surface),
            },
        }
    )
    trace_id = hashlib.sha256(trace_id_payload.encode("utf-8")).hexdigest()[:16]
    state = AspfExecutionTraceState(
        trace_id=f"aspf-trace:{trace_id}",
        controls=controls,
        started_at_utc=datetime.now(timezone.utc).isoformat(),
        command_profile=_command_profile_from_payload(payload),
    )
    entries = tuple(
        DomainToAspfCofibrationEntry(
            domain=DomainPrimeBasis(domain_key=f"domain:{surface}", prime=prime),
            aspf=AspfPrimeBasis(aspf_key=f"aspf:{surface}", prime=prime),
        )
        for surface, prime in zip(
            controls.aspf_semantic_surface,
            _first_primes(len(controls.aspf_semantic_surface)),
            strict=False,
        )
    )
    record_cofibration(
        state=state,
        canonical_identity_kind="canonical_aspf_execution_surface",
        cofibration=DomainToAspfCofibration(entries=entries),
    )
    return state


def record_1cell(
    state: AspfExecutionTraceState,
    *,
    kind: str,
    source_label: str,
    target_label: str,
    representative: str,
    basis_path: Sequence[str],
    surface: str | None = None,
    metadata: Mapping[str, object] | None = None,
) -> AspfOneCell:
    normalized_basis = tuple(str(step) for step in basis_path)
    cell = AspfOneCell(
        source=BasisZeroCell(str(source_label)),
        target=BasisZeroCell(str(target_label)),
        representative=str(representative),
        basis_path=normalized_basis,
    )
    state.one_cells.append(cell)
    metadata_payload = (
        _as_json_value(
            {str(key): metadata[key] for key in metadata}
        )
        if isinstance(metadata, Mapping)
        else {}
    )
    raw_metadata: JSONObject = {
        "kind": str(kind),
        "surface": str(surface) if surface is not None else "",
        "metadata": metadata_payload,
    }
    state.one_cell_metadata.append(raw_metadata)
    mutation_target = f"one_cells.{len(state.one_cells)}"
    phase = normalized_basis[0] if normalized_basis else "runtime"
    aspf_resume_state.append_delta_record(
        records=state.delta_records,
        event_kind=str(kind),
        phase=str(phase),
        analysis_state=(
            str(metadata.get("analysis_state"))
            if isinstance(metadata, Mapping)
            and isinstance(metadata.get("analysis_state"), str)
            else None
        ),
        mutation_target=mutation_target,
        mutation_value={
            "source": str(source_label),
            "target": str(target_label),
            "representative": str(representative),
            "surface": str(surface) if surface is not None else None,
            "metadata": metadata_payload,
        },
        one_cell_ref=mutation_target,
    )
    return cell


def record_2cell_witness(
    state: AspfExecutionTraceState,
    *,
    left: AspfOneCell,
    right: AspfOneCell,
    witness_id: str,
    reason: str,
) -> AspfTwoCellWitness:
    witness = AspfTwoCellWitness(
        left=left,
        right=right,
        witness_id=str(witness_id),
        reason=str(reason),
    )
    validate_2cell_compatibility(witness)
    state.two_cell_witnesses.append(witness)
    return witness


def record_cofibration(
    *,
    state: AspfExecutionTraceState,
    canonical_identity_kind: str,
    cofibration: DomainToAspfCofibration,
) -> None:
    cofibration.validate()
    state.cofibrations.append(
        CofibrationWitnessCarrier(
            canonical_identity_kind=str(canonical_identity_kind),
            cofibration=cofibration,
        )
    )


def merge_imported_trace(
    *,
    state: AspfExecutionTraceState,
    trace_payload: Mapping[str, object],
) -> None:
    payload = {str(key): trace_payload[key] for key in trace_payload}
    state.imported_trace_payloads.append(payload)
    _merge_surface_representatives(state=state, trace_payload=payload)
    _merge_one_cells(state=state, trace_payload=payload)
    _merge_two_cells(state=state, trace_payload=payload)
    _merge_cofibrations(state=state, trace_payload=payload)


def merge_imported_trace_paths(
    *,
    state: AspfExecutionTraceState,
    paths: Sequence[Path],
) -> None:
    for path in paths:
        payload = load_trace_payload(path)
        merge_imported_trace(state=state, trace_payload=payload)


def register_semantic_surface(
    *,
    state: AspfExecutionTraceState,
    surface: str,
    value: object,
    phase: str = "post",
) -> AspfOneCell:
    normalized_value = _as_json_value(value)
    representative = stable_compact_text(normalized_value)
    state.surface_representatives[str(surface)] = representative
    aspf_resume_state.append_delta_record(
        records=state.delta_records,
        event_kind="semantic_surface_projection",
        phase=str(phase),
        analysis_state=None,
        mutation_target=f"semantic_surfaces.{surface}",
        mutation_value=normalized_value,
        one_cell_ref=None,
    )
    return record_1cell(
        state,
        kind="semantic_surface_projection",
        source_label=f"surface:{surface}:domain",
        target_label=f"surface:{surface}:carrier",
        representative=representative,
        basis_path=(str(surface), str(phase), "projection"),
        surface=surface,
    )


def build_trace_payload(state: AspfExecutionTraceState) -> JSONObject:
    one_cells_payload: list[JSONValue] = []
    for index, cell in enumerate(state.one_cells):
        one_cell_payload = cell.as_dict()
        metadata = (
            state.one_cell_metadata[index]
            if index < len(state.one_cell_metadata)
            else {"kind": "", "surface": "", "metadata": {}}
        )
        one_cell_payload["kind"] = str(metadata.get("kind", ""))
        one_cell_payload["surface"] = str(metadata.get("surface", ""))
        one_cell_payload["metadata"] = _as_json_value(metadata.get("metadata", {}))
        one_cells_payload.append(one_cell_payload)
    return {
        "format_version": _TRACE_FORMAT_VERSION,
        "trace_id": state.trace_id,
        "started_at_utc": state.started_at_utc,
        "controls": {
            "aspf_trace_json": (
                str(state.controls.aspf_trace_json)
                if state.controls.aspf_trace_json is not None
                else None
            ),
            "aspf_import_trace": [str(path) for path in state.controls.aspf_import_trace],
            "aspf_equivalence_against": [
                str(path) for path in state.controls.aspf_equivalence_against
            ],
            "aspf_opportunities_json": (
                str(state.controls.aspf_opportunities_json)
                if state.controls.aspf_opportunities_json is not None
                else None
            ),
            "aspf_state_json": (
                str(state.controls.aspf_state_json)
                if state.controls.aspf_state_json is not None
                else None
            ),
            "aspf_import_state": [str(path) for path in state.controls.aspf_import_state],
            "aspf_delta_jsonl": (
                str(state.controls.aspf_delta_jsonl)
                if state.controls.aspf_delta_jsonl is not None
                else None
            ),
            "aspf_semantic_surface": list(state.controls.aspf_semantic_surface),
        },
        "one_cells": one_cells_payload,
        "two_cell_witnesses": [witness.as_dict() for witness in state.two_cell_witnesses],
        "cofibration_witnesses": [carrier.as_dict() for carrier in state.cofibrations],
        "surface_representatives": {
            surface: state.surface_representatives[surface]
            for surface in sort_once(
                state.surface_representatives,
                source="aspf_execution_fibration.build_trace_payload.surface_representatives",
            )
        },
        "imported_trace_count": len(state.imported_trace_payloads),
        "delta_record_count": len(state.delta_records),
    }


def build_equivalence_payload(
    *,
    state: AspfExecutionTraceState,
    baseline_traces: Sequence[Mapping[str, object]],
) -> JSONObject:
    baseline_candidates: dict[str, list[str]] = {}
    for payload in baseline_traces:
        _merge_candidates(
            destination=baseline_candidates,
            candidates=_surface_candidates_from_trace_payload(payload),
        )
    table: list[JSONObject] = []
    known_witnesses = _all_known_witnesses(state=state, baseline_traces=baseline_traces)
    for surface in state.controls.aspf_semantic_surface:
        current_representative = state.surface_representatives.get(surface)
        candidates = tuple(
            sort_once(
                set(baseline_candidates.get(surface, [])),
                source=f"aspf_execution_fibration.build_equivalence_payload.{surface}.candidates",
            )
        )
        selection_witness: JSONObject | None = None
        if candidates:
            selection = select_representative(
                RepresentativeSelectionOptions(
                    mode=RepresentativeSelectionMode.SHORTEST_PATH_THEN_LEXICOGRAPHIC,
                    candidates=candidates,
                )
            )
            baseline_representative = selection.selected
            selection_witness = selection.as_dict()
        else:
            baseline_representative = current_representative or ""
        witness = _find_witness(
            witnesses=known_witnesses,
            baseline_representative=baseline_representative,
            current_representative=current_representative or "",
        )
        classification = classify_drift_by_homotopy(
            baseline_representative=baseline_representative,
            current_representative=current_representative or "",
            equivalence_witness=witness,
            has_equivalence_witness=witness is not None,
        )
        table.append(
            {
                "surface": surface,
                "classification": classification,
                "baseline_representative": baseline_representative,
                "current_representative": current_representative,
                "witness_id": witness.witness_id if witness is not None else None,
                "representative_selection": selection_witness,
            }
        )
    verdict = "non_drift"
    if any(entry.get("classification") == "drift" for entry in table):
        verdict = "drift"
    return {
        "format_version": _EQUIVALENCE_FORMAT_VERSION,
        "trace_id": state.trace_id,
        "verdict": verdict,
        "surface_table": table,
    }


def build_opportunities_payload(
    *,
    state: AspfExecutionTraceState,
    equivalence_payload: Mapping[str, object],
) -> JSONObject:
    opportunities: list[JSONObject] = []
    opportunities.extend(_materialize_load_opportunities(state))
    opportunities.extend(_reusable_artifact_opportunities(state))
    opportunities.extend(_fungible_execution_opportunities(equivalence_payload))
    opportunities = sorted(
        opportunities,
        key=lambda item: str(item.get("opportunity_id", "")),
    )
    return {
        "format_version": _OPPORTUNITY_FORMAT_VERSION,
        "trace_id": state.trace_id,
        "opportunities": opportunities,
    }


def finalize_execution_trace(
    *,
    state: AspfExecutionTraceState | None,
    root: Path,
    semantic_surface_payloads: Mapping[str, object],
    exit_code: int | None = None,
    analysis_state: str | None = None,
) -> AspfFinalizationArtifacts | None:
    if state is None:
        return None
    merge_imported_trace_paths(state=state, paths=state.controls.aspf_import_trace)
    merge_imported_trace_paths(state=state, paths=state.controls.aspf_import_state)
    semantic_surface_keys = set(state.controls.aspf_semantic_surface)
    for surface, value in semantic_surface_payloads.items():
        if surface in semantic_surface_keys:
            register_semantic_surface(
                state=state,
                surface=surface,
                value=value,
            )
    baseline_paths = (
        state.controls.aspf_equivalence_against
        if state.controls.aspf_equivalence_against
        else (state.controls.aspf_import_trace + state.controls.aspf_import_state)
    )
    baseline_traces = [load_trace_payload(path) for path in baseline_paths]
    trace_payload = build_trace_payload(state)
    equivalence_payload = build_equivalence_payload(
        state=state,
        baseline_traces=baseline_traces,
    )
    opportunities_payload = build_opportunities_payload(
        state=state,
        equivalence_payload=equivalence_payload,
    )
    resume_snapshot: JSONObject = {
        "analysis_state": str(analysis_state) if analysis_state is not None else None,
        "collection_resume": (
            _as_json_value(semantic_surface_payloads.get("_resume_collection"))
            if isinstance(semantic_surface_payloads.get("_resume_collection"), Mapping)
            else {}
        ),
        "latest_collection_progress": (
            _as_json_value(semantic_surface_payloads.get("_latest_collection_progress"))
            if isinstance(semantic_surface_payloads.get("_latest_collection_progress"), Mapping)
            else {}
        ),
        "semantic_progress": (
            _as_json_value(semantic_surface_payloads.get("_semantic_progress"))
            if isinstance(semantic_surface_payloads.get("_semantic_progress"), Mapping)
            else {}
        ),
        "semantic_surfaces": {
            str(key): _as_json_value(semantic_surface_payloads[key])
            for key in semantic_surface_payloads
            if key in state.controls.aspf_semantic_surface
        },
        "exit_code": int(exit_code) if exit_code is not None else None,
    }
    resume_projection = aspf_resume_state.replay_resume_projection(
        snapshot=resume_snapshot,
        delta_records=iter(state.delta_records),
    )
    delta_ledger_payload = aspf_resume_state.build_delta_ledger_payload(
        trace_id=state.trace_id,
        records=iter(state.delta_records),
    )
    trace_path = state.controls.aspf_trace_json or (root / "artifacts/out/aspf_trace.json")
    equivalence_path = root / "artifacts/out/aspf_equivalence.json"
    opportunities_path = state.controls.aspf_opportunities_json or (
        root / "artifacts/out/aspf_opportunities.json"
    )
    delta_jsonl_path = state.controls.aspf_delta_jsonl or (
        root / "artifacts/out/aspf_delta.jsonl"
    )
    _write_json(trace_path, trace_payload)
    _write_json(equivalence_path, equivalence_payload)
    _write_json(opportunities_path, opportunities_payload)
    aspf_resume_state.write_delta_jsonl(
        path=delta_jsonl_path,
        records=iter(state.delta_records),
    )
    state_path = state.controls.aspf_state_json or (
        root / "artifacts/out/aspf_state/default/0001_aspf.snapshot.json"
    )
    state_payload = _build_state_payload(
        state=state,
        state_path=state_path,
        trace_payload=trace_payload,
        equivalence_payload=equivalence_payload,
        opportunities_payload=opportunities_payload,
        semantic_surface_payloads=semantic_surface_payloads,
        exit_code=exit_code,
        analysis_state=analysis_state,
        resume_projection=resume_projection,
        delta_ledger_payload=delta_ledger_payload,
    )
    _write_json(state_path, state_payload)
    return AspfFinalizationArtifacts(
        trace_payload=trace_payload,
        equivalence_payload=equivalence_payload,
        opportunities_payload=opportunities_payload,
        delta_ledger_payload=delta_ledger_payload,
        trace_path=trace_path,
        equivalence_path=equivalence_path,
        opportunities_path=opportunities_path,
        delta_jsonl_path=delta_jsonl_path,
        state_payload=state_payload,
        state_path=state_path,
    )


def load_trace_payload(path: Path) -> JSONObject:
    payload = cast(Mapping[str, object], json.loads(path.read_text(encoding="utf-8")))
    normalized_payload = {str(key): _as_json_value(payload[key]) for key in payload}
    trace_payload = normalized_payload.get("trace")
    if trace_payload is None:
        return normalized_payload
    trace_payload = cast(Mapping[str, object], trace_payload)
    normalized_trace: JSONObject = {
        str(key): _as_json_value(trace_payload[key]) for key in trace_payload
    }
    equivalence_payload = cast(
        Mapping[str, object], normalized_payload.get("equivalence", {})
    )
    opportunities_payload = cast(
        Mapping[str, object], normalized_payload.get("opportunities", {})
    )
    normalized_trace["equivalence"] = {
        str(key): _as_json_value(equivalence_payload[key]) for key in equivalence_payload
    }
    normalized_trace["opportunities"] = {
        str(key): _as_json_value(opportunities_payload[key]) for key in opportunities_payload
    }
    return normalized_trace


def _command_profile_from_payload(payload: Mapping[str, object]) -> str:
    synth_requested = (payload.get("synthesis_plan") is not None) | bool(
        payload.get("synthesis_report")
    )
    return ("check.run", "synth")[int(synth_requested)]


def _build_state_payload(
    *,
    state: AspfExecutionTraceState,
    state_path: Path,
    trace_payload: Mapping[str, object],
    equivalence_payload: Mapping[str, object],
    opportunities_payload: Mapping[str, object],
    semantic_surface_payloads: Mapping[str, object],
    exit_code: int | None,
    analysis_state: str | None,
    resume_projection: Mapping[str, object],
    delta_ledger_payload: Mapping[str, object],
) -> JSONObject:
    session_id, step_id = _session_and_step_from_path(state_path)
    state_material = stable_compact_text(
        {
            "trace_id": state.trace_id,
            "session_id": session_id,
            "step_id": step_id,
            "trace": trace_payload,
            "equivalence": equivalence_payload,
            "opportunities": opportunities_payload,
        }
    )
    state_id = f"aspf-state:{hashlib.sha256(state_material.encode('utf-8')).hexdigest()[:16]}"
    analysis_manifest_digest = semantic_surface_payloads.get("_analysis_manifest_digest")
    resume_source = semantic_surface_payloads.get("_resume_source")
    resume_compatibility_status = semantic_surface_payloads.get(
        "_resume_compatibility_status"
    )
    return {
        "format_version": _STATE_FORMAT_VERSION,
        "state_id": state_id,
        "session_id": session_id,
        "step_id": step_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "command_profile": state.command_profile,
        "analysis_manifest_digest": (
            str(analysis_manifest_digest) if analysis_manifest_digest is not None else None
        ),
        "resume_source": str(resume_source) if resume_source is not None else None,
        "resume_compatibility_status": (
            str(resume_compatibility_status)
            if resume_compatibility_status is not None
            else None
        ),
        "trace": {str(key): _as_json_value(trace_payload[key]) for key in trace_payload},
        "equivalence": {
            str(key): _as_json_value(equivalence_payload[key])
            for key in equivalence_payload
        },
        "opportunities": {
            str(key): _as_json_value(opportunities_payload[key])
            for key in opportunities_payload
        },
        "semantic_surfaces": {
            str(key): _as_json_value(semantic_surface_payloads[key])
            for key in semantic_surface_payloads
            if key in state.controls.aspf_semantic_surface
        },
        "resume_projection": {
            str(key): _as_json_value(resume_projection[key]) for key in resume_projection
        },
        "delta_ledger": {
            str(key): _as_json_value(delta_ledger_payload[key])
            for key in delta_ledger_payload
        },
        "exit_code": int(exit_code) if exit_code is not None else None,
        "analysis_state": str(analysis_state) if analysis_state is not None else None,
    }


def _session_and_step_from_path(path: Path) -> tuple[str, str]:
    session_id = path.parent.name.strip() if path.parent.name else "default"
    step_id = path.stem.strip() if path.stem else "step"
    return session_id, step_id


def _optional_path(value: object) -> Path | None:
    if value is None:
        return None
    return Path(str(value).strip())


def _path_sequence(value: object) -> tuple[Path, ...]:
    if value is None:
        return ()
    return tuple(Path(str(item).strip()) for item in value)


def _semantic_surface_sequence(value: object) -> tuple[str, ...]:
    if value is None:
        return DEFAULT_PHASE1_SEMANTIC_SURFACES
    surfaces = [str(item).strip() for item in value if str(item).strip()]
    if not surfaces:
        return DEFAULT_PHASE1_SEMANTIC_SURFACES
    return tuple(
        sort_once(
            set(surfaces),
            source="aspf_execution_fibration._semantic_surface_sequence.surfaces",
        )
    )


def _first_primes(count: int) -> tuple[int, ...]:
    primes: list[int] = []
    candidate = 2
    while len(primes) < count:
        if _is_prime(candidate):
            primes.append(candidate)
        candidate += 1
    return tuple(primes)


def _is_prime(value: int) -> bool:
    if value == 2:
        return True
    if value % 2 == 0:
        return False
    divisor = 3
    while divisor * divisor <= value:
        if value % divisor == 0:
            return False
        divisor += 2
    return True


def _as_json_value(value: object) -> JSONValue:
    if isinstance(value, Mapping):
        key_pairs = tuple(
            sort_once(
                [(str(key), key) for key in value],
                source="aspf_execution_fibration._as_json_value.mapping_keys",
                key=lambda pair: pair[0],
            )
        )
        return {
            text_key: _as_json_value(value[raw_key])
            for text_key, raw_key in key_pairs
        }
    if isinstance(value, (list, tuple)):
        return [_as_json_value(item) for item in value]
    if isinstance(value, set):
        normalized = [_as_json_value(item) for item in value]
        return sorted(
            normalized,
            key=lambda item: stable_compact_text(item),
        )
    return cast(JSONValue, value)


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")


def _merge_surface_representatives(
    *,
    state: AspfExecutionTraceState,
    trace_payload: Mapping[str, object],
) -> None:
    raw = trace_payload["surface_representatives"]
    for key in raw:
        state.surface_representatives.setdefault(str(key), str(raw[key]))


def _merge_one_cells(
    *,
    state: AspfExecutionTraceState,
    trace_payload: Mapping[str, object],
) -> None:
    raw_cells = trace_payload["one_cells"]
    for raw in raw_cells:
        source = str(raw["source"])
        target = str(raw["target"])
        representative = str(raw["representative"])
        basis_path = tuple(str(item) for item in raw["basis_path"])
        record_1cell(
            state,
            kind=str(raw.get("kind", "imported_1cell")),
            source_label=source,
            target_label=target,
            representative=representative,
            basis_path=basis_path,
            surface=str(raw.get("surface", "")) or None,
            metadata=raw.get("metadata"),
        )


def _merge_two_cells(
    *,
    state: AspfExecutionTraceState,
    trace_payload: Mapping[str, object],
) -> None:
    raw_witnesses = trace_payload["two_cell_witnesses"]
    state.two_cell_witnesses.extend(
        cast(AspfTwoCellWitness, parse_2cell_witness(raw)) for raw in raw_witnesses
    )


def _merge_cofibrations(
    *,
    state: AspfExecutionTraceState,
    trace_payload: Mapping[str, object],
) -> None:
    raw_cofibrations = trace_payload["cofibration_witnesses"]
    for raw in raw_cofibrations:
        cofibration = _parse_cofibration(raw["cofibration"])
        record_cofibration(
            state=state,
            canonical_identity_kind=str(raw["canonical_identity_kind"]),
            cofibration=cofibration,
        )


def _parse_cofibration(raw: Mapping[str, object]) -> DomainToAspfCofibration:
    entries_raw = raw["entries"]
    entries: list[DomainToAspfCofibrationEntry] = []
    for raw_entry in entries_raw:
        domain_raw = raw_entry["domain"]
        aspf_raw = raw_entry["aspf"]
        entries.append(
            DomainToAspfCofibrationEntry(
                domain=DomainPrimeBasis(
                    domain_key=str(domain_raw["key"]),
                    prime=int(domain_raw["prime"]),
                ),
                aspf=AspfPrimeBasis(
                    aspf_key=str(aspf_raw["key"]),
                    prime=int(aspf_raw["prime"]),
                ),
            )
        )
    return DomainToAspfCofibration(entries=tuple(entries))


def _surface_candidates_from_trace_payload(
    payload: Mapping[str, object],
) -> dict[str, list[str]]:
    candidates: dict[str, list[str]] = {}
    raw = payload.get("surface_representatives", {})
    for key in raw:
        candidates.setdefault(str(key), []).append(str(raw[key]))
    equivalence = payload.get("equivalence", {})
    table = equivalence.get("surface_table", [])
    for row in table:
        surface = str(row.get("surface", "")).strip() or "unknown_surface"
        baseline_rep = (
            str(row.get("baseline_representative", "")).strip()
            or "unknown_representative"
        )
        candidates.setdefault(surface, []).append(baseline_rep)
    return candidates


def _merge_candidates(
    *,
    destination: dict[str, list[str]],
    candidates: Mapping[str, list[str]],
) -> None:
    for surface in candidates:
        existing = destination.setdefault(surface, [])
        existing.extend(candidates[surface])


def _all_known_witnesses(
    *,
    state: AspfExecutionTraceState,
    baseline_traces: Sequence[Mapping[str, object]],
) -> list[AspfTwoCellWitness]:
    baseline_witnesses = [
        cast(AspfTwoCellWitness, parse_2cell_witness(entry))
        for payload in baseline_traces
        for entry in payload.get("two_cell_witnesses", [])
    ]
    return [*state.two_cell_witnesses, *baseline_witnesses]


def _find_witness(
    *,
    witnesses: Sequence[AspfTwoCellWitness],
    baseline_representative: str,
    current_representative: str,
) -> AspfTwoCellWitness | None:
    return next(
        (
            witness
            for witness in witnesses
            if witness.links(
                baseline_representative=baseline_representative,
                current_representative=current_representative,
            )
        ),
        None,
    )


def _materialize_load_opportunities(state: AspfExecutionTraceState) -> list[JSONObject]:
    by_resume_ref: dict[str, set[str]] = {}
    for metadata in state.one_cell_metadata:
        kind = str(metadata.get("kind", ""))
        raw_payload = metadata.get("metadata", {})
        payload = raw_payload if isinstance(raw_payload, Mapping) else {}
        resume_ref = ""
        for candidate_key in ("state_path", "import_state_path"):
            candidate = str(payload.get(candidate_key, "")).strip()
            if candidate:
                resume_ref = candidate
                break
        if not resume_ref:
            continue
        by_resume_ref.setdefault(resume_ref, set()).add(kind)
    opportunities: list[JSONObject] = []
    for resume_ref in sort_once(
        by_resume_ref,
        source="aspf_execution_fibration._materialize_load_opportunities.resume_ref",
    ):
        kinds = by_resume_ref[resume_ref]
        fused = int("resume_load" in kinds and "resume_write" in kinds)
        opportunities.append(
            {
                "opportunity_id": f"opp:materialize-load-fusion:{resume_ref}",
                "kind": ("materialize_load_observed", "materialize_load_fusion")[fused],
                "confidence": (0.51, 0.74)[fused],
                "affected_surfaces": [],
                "witness_ids": [],
                "reason": (
                    "resume boundary observed state reference",
                    "resume load and write boundaries share state reference",
                )[fused],
            }
        )
    return opportunities


def _reusable_artifact_opportunities(state: AspfExecutionTraceState) -> list[JSONObject]:
    representative_to_surfaces: dict[str, list[str]] = {}
    for surface in state.surface_representatives:
        representative = state.surface_representatives[surface]
        representative_to_surfaces.setdefault(representative, []).append(surface)
    opportunities: list[JSONObject] = []
    for representative in sort_once(
        representative_to_surfaces,
        source="aspf_execution_fibration._reusable_artifact_opportunities.representative",
    ):
        surfaces = tuple(
            sort_once(
                representative_to_surfaces[representative],
                source="aspf_execution_fibration._reusable_artifact_opportunities.surfaces",
            )
        )
        digest = hashlib.sha256(representative.encode("utf-8")).hexdigest()[:12]
        opportunities.append(
            {
                "opportunity_id": f"opp:reusable-boundary:{digest}",
                "kind": "reusable_boundary_artifact",
                "confidence": 0.67,
                "affected_surfaces": list(surfaces),
                "witness_ids": [],
                "reason": "multiple semantic surfaces share deterministic representative",
            }
        )
    return opportunities


def _fungible_execution_opportunities(
    equivalence_payload: Mapping[str, object],
) -> list[JSONObject]:
    table = equivalence_payload.get("surface_table", [])
    opportunities: list[JSONObject] = []
    for row in table:
        if str(row.get("classification", "")) != "non_drift":
            continue
        witness_id = row.get("witness_id")
        if not isinstance(witness_id, str) or not witness_id:
            continue
        surface = str(row.get("surface", "")).strip() or "unknown_surface"
        opportunities.append(
            {
                "opportunity_id": f"opp:fungible-substitution:{surface}",
                "kind": "fungible_execution_path_substitution",
                "confidence": 0.82,
                "affected_surfaces": [surface],
                "witness_ids": [witness_id],
                "reason": "2-cell witness links baseline/current representatives",
            }
        )
    return opportunities

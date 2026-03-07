#!/usr/bin/env python3
# gabion:boundary_normalization_module gabion:decision_protocol_module
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from gabion.analysis.core.prime_identity_adapter import PrimeIdentityAdapter
from gabion.analysis.core.type_fingerprints import PrimeRegistry
from gabion.analysis.foundation.event_algebra import (
    CanonicalRunContext,
    GlobalEventSequencer,
    derive_identity_projection_from_tokens,
)
from gabion.analysis.foundation.identity_space import GlobalIdentitySpace


BASELINE_VERSION = 1
_DEFAULT_INCLUDE_SNAPSHOT_ARCHIVE = False


_NORMALIZATION_CLASS_HINTS: dict[str, tuple[str, ...]] = {
    "parse": ("parse", "decode"),
    "validate": ("validate", "check"),
    "narrow": ("narrow", "coerce", "cast"),
    "normalize": ("normalize", "canonicalize"),
}
_CORE_ENTRY_KINDS = {
    "analysis_call_start",
    "analysis_call_end",
    "analysis_call_skipped",
    "core_start",
    "core_call_start",
}
_PRE_CORE_PHASE_HINTS = {
    "adapter",
    "boundary",
    "command",
    "decode",
    "ingress",
    "narrow",
    "normalize",
    "parse",
    "pre",
    "runtime",
    "validate",
}


@dataclass(frozen=True)
class Violation:
    path: str
    line: int
    column: int
    qualname: str
    kind: str
    message: str
    normalization_class: str
    flow_identity: str
    event_kind: str
    structured_hash: str

    @property
    def key(self) -> str:
        return f"{self.path}:{self.qualname}:{self.kind}:{self.structured_hash}"

    @property
    def legacy_key(self) -> str:
        return f"{self.path}:{self.qualname}:{self.line}:{self.kind}"

    def render(self) -> str:
        return f"{self.path}:{self.line}:{self.column}: [{self.qualname}] {self.message}"


@dataclass(frozen=True)
class _TraceSource:
    path: Path
    one_cells: tuple[_OneCellDTO, ...]
    controls: _TraceControlsDTO | None
    delta_records: tuple[_DeltaRecordDTO, ...]


@dataclass(frozen=True)
class _ObservedNormalization:
    event_index: int
    event_kind: str
    phase_hint: str
    normalization_class: str


class _TraceControlsDTO(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)

    aspf_delta_jsonl: str | None = None


class _OneCellMetadataDTO(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)

    normalization_class: str | None = None
    normalization_kind: str | None = None
    normalization_step: str | None = None


class _OneCellDTO(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)

    source: str | None = None
    target: str | None = None
    surface: str | None = None
    basis_path: list[str] = Field(default_factory=list)
    kind: str | None = None
    metadata: _OneCellMetadataDTO | None = None


class _DeltaRecordDTO(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)

    one_cell_ref: str
    phase: str | None = None


class _DeltaLedgerDTO(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)

    records: list[_DeltaRecordDTO] = Field(default_factory=list)


class _TracePayloadDTO(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)

    one_cells: list[_OneCellDTO]
    controls: _TraceControlsDTO | None = None


class _TraceDocumentDTO(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)

    trace: _TracePayloadDTO | None = None
    one_cells: list[_OneCellDTO] | None = None
    controls: _TraceControlsDTO | None = None
    delta_ledger: _DeltaLedgerDTO | None = None


class _BaselineViolationDTO(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)

    path: str
    qualname: str
    kind: str
    structured_hash: str


class _BaselinePayloadDTO(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)

    violations: list[_BaselineViolationDTO] = Field(default_factory=list)


def collect_violations(
    *,
    root: Path,
    include_snapshot_archive: bool = _DEFAULT_INCLUDE_SNAPSHOT_ARCHIVE,
) -> list[Violation]:
    violations: list[Violation] = []
    for source in _discover_trace_sources(
        root=root,
        include_snapshot_archive=include_snapshot_archive,
    ):
        violations.extend(_collect_source_violations(root=root, source=source))
    return violations


def collect_ingress_violations(
    *,
    root: Path,
    baseline_path: Path | None = None,
    include_snapshot_archive: bool = _DEFAULT_INCLUDE_SNAPSHOT_ARCHIVE,
) -> list[Violation]:
    violations: list[Violation] = []
    for path in _candidate_trace_paths(
        root,
        include_snapshot_archive=include_snapshot_archive,
    ):
        payload, load_error, load_line = _load_json_object_with_diagnostic(path)
        if payload is None:
            violations.append(
                _ingress_violation(
                    path=path,
                    line=load_line or 1,
                    kind="invalid_trace_document_json",
                    message=load_error or "unable to parse ASPF trace payload JSON",
                )
            )
            continue

        trace_document = _extract_trace_document(payload)
        if trace_document is None:
            violations.append(
                _ingress_violation(
                    path=path,
                    line=1,
                    kind="invalid_trace_document_shape",
                    message="ASPF trace payload does not match strict trace document schema",
                )
            )
            continue

        trace_payload = _trace_payload_from_document(trace_document)
        if trace_payload is None:
            violations.append(
                _ingress_violation(
                    path=path,
                    line=1,
                    kind="invalid_trace_payload_shape",
                    message="ASPF trace payload must provide either trace.one_cells or top-level one_cells",
                )
            )
            continue

        _load_delta_records(
            path=path,
            controls=trace_payload.controls,
            root=root,
            source_document=trace_document,
            diagnostics=violations,
        )

    if baseline_path is not None:
        violations.extend(_collect_baseline_ingress_violations(path=baseline_path))
    return violations


def _candidate_trace_paths(
    root: Path,
    *,
    include_snapshot_archive: bool = _DEFAULT_INCLUDE_SNAPSHOT_ARCHIVE,
) -> tuple[Path, ...]:
    candidates: list[Path] = []
    default_trace_path = root / "artifacts/out/aspf_trace.json"
    if default_trace_path.exists():
        candidates.append(default_trace_path)
    if include_snapshot_archive:
        for snapshot_path in sorted((root / "artifacts/out").glob("**/*.snapshot.json")):
            candidates.append(snapshot_path)
    return tuple(candidates)


def _discover_trace_sources(
    *,
    root: Path,
    include_snapshot_archive: bool = _DEFAULT_INCLUDE_SNAPSHOT_ARCHIVE,
) -> tuple[_TraceSource, ...]:
    seen_paths: set[Path] = set()
    sources: list[_TraceSource] = []
    for path in _candidate_trace_paths(
        root,
        include_snapshot_archive=include_snapshot_archive,
    ):
        resolved = path.resolve()
        if resolved in seen_paths:
            continue
        seen_paths.add(resolved)

        loaded_payload = _load_json_object(path)
        if loaded_payload is None:
            continue
        trace_document = _extract_trace_document(loaded_payload)
        if trace_document is None:
            continue
        trace_payload = _trace_payload_from_document(trace_document)
        if trace_payload is None:
            continue

        delta_records = tuple(
            _load_delta_records(
                path=path,
                controls=trace_payload.controls,
                root=root,
                source_document=trace_document,
                diagnostics=None,
            )
        )
        sources.append(
            _TraceSource(
                path=path,
                one_cells=tuple(trace_payload.one_cells),
                controls=trace_payload.controls,
                delta_records=delta_records,
            )
        )

    return tuple(sources)


def _extract_trace_document(payload: object) -> _TraceDocumentDTO | None:
    try:
        return _TraceDocumentDTO.model_validate(payload)
    except ValidationError:
        return None


def _trace_payload_from_document(
    document: _TraceDocumentDTO,
) -> _TracePayloadDTO | None:
    if document.trace is not None:
        return document.trace
    if document.one_cells is None:
        return None
    return _TracePayloadDTO(
        one_cells=document.one_cells,
        controls=document.controls,
    )


def _load_delta_records(
    *,
    path: Path,
    controls: _TraceControlsDTO | None,
    root: Path,
    source_document: _TraceDocumentDTO,
    diagnostics: list[Violation] | None,
) -> list[_DeltaRecordDTO]:
    records: list[_DeltaRecordDTO] = []

    if source_document.delta_ledger is not None:
        records.extend(source_document.delta_ledger.records)

    if controls is not None:
        delta_jsonl_path = _strict_delta_jsonl_path_from_controls(
            controls=controls,
            root=root,
        )
        if delta_jsonl_path is not None and _is_repo_local_path(
            path=delta_jsonl_path,
            root=root,
        ):
            if diagnostics is not None and not delta_jsonl_path.exists():
                diagnostics.append(
                    _ingress_violation(
                        path=delta_jsonl_path,
                        line=1,
                        kind="missing_delta_jsonl_path",
                        message="ASPF delta JSONL path declared in controls does not exist",
                    )
                )
            records.extend(
                _records_from_jsonl_path(
                    delta_jsonl_path,
                    diagnostics=diagnostics,
                )
            )

    return records


def _is_repo_local_path(*, path: Path, root: Path) -> bool:
    resolved_path = path.resolve()
    resolved_root = root.resolve()
    try:
        resolved_path.relative_to(resolved_root)
    except ValueError:
        return False
    return True


def _records_from_jsonl_path(
    path: Path,
    *,
    diagnostics: list[Violation] | None,
) -> list[_DeltaRecordDTO]:
    if not path.exists() or not path.is_file():
        return []
    records: list[_DeltaRecordDTO] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                loaded = json.loads(line)
            except json.JSONDecodeError:
                if diagnostics is not None:
                    diagnostics.append(
                        _ingress_violation(
                            path=path,
                            line=line_number,
                            kind="invalid_delta_jsonl_line_json",
                            message="ASPF delta JSONL line is not valid JSON",
                        )
                    )
                continue
            try:
                parsed = _DeltaRecordDTO.model_validate(loaded)
            except ValidationError:
                if diagnostics is not None:
                    diagnostics.append(
                        _ingress_violation(
                            path=path,
                            line=line_number,
                            kind="invalid_delta_jsonl_line_shape",
                            message="ASPF delta JSONL line does not match strict delta record schema",
                        )
                    )
                continue
            records.append(parsed)
    return records


def _collect_source_violations(*, root: Path, source: _TraceSource) -> list[Violation]:
    run_context = CanonicalRunContext(
        run_id="policy:aspf_normalization_idempotence",
        sequencer=GlobalEventSequencer(),
        identity_space=GlobalIdentitySpace(
            allocator=PrimeIdentityAdapter(registry=PrimeRegistry())
        ),
    )

    one_cells = source.one_cells
    if not one_cells:
        return []

    phase_by_one_cell_ref = _phase_hints_by_one_cell_ref(source.delta_records)
    core_entry_index = _first_core_entry_index(one_cells=one_cells)
    observed: dict[tuple[str, str], list[_ObservedNormalization]] = {}

    for index, one_cell in enumerate(one_cells, start=1):
        normalization_class = _normalization_class(one_cell)
        if normalization_class is None:
            continue

        one_cell_ref = f"one_cells.{index}"
        phase_hint = phase_by_one_cell_ref.get(one_cell_ref, "")
        if not _is_pre_core_event(
            event_index=index,
            core_entry_index=core_entry_index,
            phase_hint=phase_hint,
        ):
            continue

        flow_identity = _derive_canonical_flow_identity(
            run_context=run_context,
            one_cell=one_cell,
        )
        event_kind = (one_cell.kind or "").strip()
        observed.setdefault((flow_identity, normalization_class), []).append(
            _ObservedNormalization(
                event_index=index,
                event_kind=event_kind,
                phase_hint=phase_hint,
                normalization_class=normalization_class,
            )
        )

    rel_path = source.path.resolve()
    try:
        display_path = rel_path.relative_to(root.resolve()).as_posix()
    except ValueError:
        display_path = rel_path.as_posix()

    violations: list[Violation] = []
    for (flow_identity, normalization_class), events in observed.items():
        if len(events) <= 1:
            continue
        ordered_events = sorted(events, key=lambda item: item.event_index)
        event_kinds = ",".join(
            event.event_kind or "<unknown>" for event in ordered_events
        )
        phase_hints = ",".join(
            event.phase_hint or "<none>" for event in ordered_events
        )
        structured_hash = _structured_hash(
            flow_identity,
            normalization_class,
            event_kinds,
            phase_hints,
        )
        first_duplicate_index = ordered_events[1].event_index
        kind = "duplicate_normalization_class_pre_core"
        qualname = f"flow:{flow_identity}"
        message = (
            f"normalization class '{normalization_class}' was applied "
            f"{len(ordered_events)} times before core on the same canonical flow"
        )
        violations.append(
            Violation(
                path=display_path,
                line=first_duplicate_index,
                column=1,
                qualname=qualname,
                kind=kind,
                message=message,
                normalization_class=normalization_class,
                flow_identity=flow_identity,
                event_kind=ordered_events[1].event_kind or "<unknown>",
                structured_hash=structured_hash,
            )
        )

    return violations


def _phase_hints_by_one_cell_ref(
    delta_records: Sequence[_DeltaRecordDTO],
) -> dict[str, str]:
    hints: dict[str, str] = {}
    for record in delta_records:
        one_cell_ref = record.one_cell_ref.strip()
        if not one_cell_ref:
            continue
        phase_hint = (record.phase or "").strip()
        hints[one_cell_ref] = phase_hint
    return hints


def _first_core_entry_index(*, one_cells: Sequence[_OneCellDTO]) -> int | None:
    for index, one_cell in enumerate(one_cells, start=1):
        kind = (one_cell.kind or "").strip().lower()
        if kind in _CORE_ENTRY_KINDS:
            return index
    return None


def _is_pre_core_event(
    *,
    event_index: int,
    core_entry_index: int | None,
    phase_hint: str,
) -> bool:
    if core_entry_index is not None:
        return event_index < core_entry_index

    normalized_phase_hint = str(phase_hint).strip().lower()
    if not normalized_phase_hint:
        return True
    if normalized_phase_hint in _PRE_CORE_PHASE_HINTS:
        return True
    if normalized_phase_hint in {"analysis", "core", "post"}:
        return False
    return True


def _normalization_class(one_cell: _OneCellDTO) -> str | None:
    metadata = one_cell.metadata
    if metadata is not None:
        for value in (
            metadata.normalization_class,
            metadata.normalization_kind,
            metadata.normalization_step,
        ):
            normalized = _normalize_class_label(value)
            if normalized is not None:
                return normalized

    kind = (one_cell.kind or "").strip().lower()
    if not kind:
        return None
    fragments = tuple(
        fragment for fragment in re.split(r"[^a-z0-9]+", kind) if fragment
    )
    for class_name, hints in _NORMALIZATION_CLASS_HINTS.items():
        if any(hint in fragments for hint in hints):
            return class_name
    return None


def _normalize_class_label(value: str | None) -> str | None:
    if value is None:
        return None
    lowered = value.strip().lower()
    if not lowered:
        return None
    if lowered in {"parse", "decode"}:
        return "parse"
    if lowered in {"validate", "check"}:
        return "validate"
    if lowered in {"narrow", "cast", "coerce"}:
        return "narrow"
    if lowered in {"normalize", "canonicalize"}:
        return "normalize"
    return None


def _derive_canonical_flow_identity(
    *,
    run_context: CanonicalRunContext,
    one_cell: _OneCellDTO,
) -> str:
    source = (one_cell.source or "").strip()
    target = (one_cell.target or "").strip()
    surface = (one_cell.surface or "").strip() or "<none>"
    basis_path = tuple(item.strip() for item in one_cell.basis_path if item.strip())
    basis_text = "/".join(basis_path) if basis_path else "<none>"

    projection = derive_identity_projection_from_tokens(
        run_context=run_context,
        tokens=(
            "aspf.normalization_flow",
            f"source:{source or '<none>'}",
            f"target:{target or '<none>'}",
            f"surface:{surface}",
            f"basis:{basis_text}",
        ),
    )
    atoms = ".".join(str(atom) for atom in projection.basis_path.atoms)
    return f"{projection.basis_path.namespace}:{atoms}"


def _structured_hash(*parts: str) -> str:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(part.encode("utf-8"))
        digest.update(b"\x00")
    return digest.hexdigest()


def _load_json_object(path: Path) -> object | None:
    loaded, _, _ = _load_json_object_with_diagnostic(path)
    return loaded


def _load_json_object_with_diagnostic(
    path: Path,
) -> tuple[object | None, str | None, int | None]:
    try:
        return json.loads(path.read_text(encoding="utf-8")), None, None
    except OSError as exc:
        return None, f"unable to read JSON payload: {exc}", 1
    except json.JSONDecodeError as exc:
        return None, f"invalid JSON payload: {exc.msg}", int(exc.lineno or 1)


def _load_baseline(path: Path) -> set[str]:
    if not path.exists():
        return set()
    payload = _load_json_object(path)
    if payload is None:
        return set()
    try:
        baseline_payload = _BaselinePayloadDTO.model_validate(payload)
    except ValidationError:
        return set()
    keys: set[str] = set()
    for baseline_violation in baseline_payload.violations:
        keys.add(
            f"{baseline_violation.path}:{baseline_violation.qualname}:"
            f"{baseline_violation.kind}:{baseline_violation.structured_hash}"
        )
    return keys


def _collect_baseline_ingress_violations(*, path: Path) -> list[Violation]:
    if not path.exists():
        return []
    loaded, load_error, load_line = _load_json_object_with_diagnostic(path)
    if loaded is None:
        return [
            _ingress_violation(
                path=path,
                line=load_line or 1,
                kind="invalid_baseline_json",
                message=load_error or "unable to parse ASPF baseline JSON",
            )
        ]
    try:
        _BaselinePayloadDTO.model_validate(loaded)
    except ValidationError as exc:
        return [
            _ingress_violation(
                path=path,
                line=1,
                kind="invalid_baseline_payload",
                message=_summarize_validation_error(exc),
            )
        ]
    return []


def _summarize_validation_error(exc: ValidationError) -> str:
    first_error = exc.errors()[0] if exc.errors() else {}
    loc = ".".join(str(item) for item in first_error.get("loc", ()))
    msg = str(first_error.get("msg", "validation error"))
    if loc:
        return f"strict schema validation failed at {loc}: {msg}"
    return f"strict schema validation failed: {msg}"


def _ingress_violation(
    *,
    path: Path,
    line: int,
    kind: str,
    message: str,
) -> Violation:
    path_text = path.as_posix()
    structured_hash = _structured_hash(
        path_text,
        str(line),
        kind,
        message,
    )
    return Violation(
        path=path_text,
        line=line,
        column=1,
        qualname="<ingress>",
        kind=kind,
        message=message,
        normalization_class="<ingress>",
        flow_identity="<ingress>",
        event_kind="<ingress>",
        structured_hash=structured_hash,
    )


def _strict_delta_jsonl_path_from_controls(
    *,
    controls: _TraceControlsDTO,
    root: Path,
) -> Path | None:
    if controls.aspf_delta_jsonl is None:
        return None
    text = controls.aspf_delta_jsonl.strip()
    if not text:
        return None
    path = Path(text)
    if path.is_absolute():
        return path
    return (root / path).resolve()


__all__ = [
    "BASELINE_VERSION",
    "Violation",
    "collect_violations",
    "collect_ingress_violations",
]

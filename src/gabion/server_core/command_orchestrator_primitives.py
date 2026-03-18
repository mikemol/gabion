from __future__ import annotations

"""Static primitives extracted from server for command_orchestrator DAG decoupling."""

import hashlib
import json
import threading
import time
from itertools import chain, repeat
from datetime import (datetime, timezone)
from dataclasses import dataclass
from decimal import (Decimal, InvalidOperation)
from pathlib import Path
from typing import (Callable, Literal, Mapping, Sequence, cast)
from gabion.json_types import (JSONObject, JSONValue)
from gabion.commands import (boundary_order, command_ids, payload_codec, progress_contract as progress_timeline)
from gabion.commands.lint_parser import parse_lint_line
from gabion.commands.check_contract import LintEntriesDecision
from gabion.plan import (ExecutionPlan, ExecutionPlanObligations, ExecutionPlanPolicyMetadata, write_execution_plan_artifact)
from gabion.analysis import (AnalysisResult, AuditConfig, ReportCarrier, analyze_paths, apply_baseline, build_analysis_collection_resume_seed, compute_structure_metrics, compute_violations, build_refactor_plan, build_synthesis_plan, load_baseline, extract_report_sections, project_report_sections, report_projection_phase_rank, report_projection_spec_rows, render_dot, render_structure_snapshot, render_decision_snapshot, DecisionSnapshotSurfaces, render_protocol_stubs, render_refactor_plan, render_report, render_synthesis_section, resolve_baseline_path, write_baseline)
from gabion.analysis.aspf import (aspf_execution_fibration, aspf_resume_state)
from gabion.analysis.aspf.aspf import Forest
from gabion.analysis.core import ambiguity_delta
from gabion.analysis.core import ambiguity_state
from gabion.analysis.call_cluster import call_cluster_consolidation
from gabion.analysis.call_cluster import call_clusters
from gabion.analysis.semantics import semantic_coverage_map
from gabion.analysis.taint import taint_delta
from gabion.analysis.taint import taint_lifecycle
from gabion.analysis.taint import taint_state
from gabion.analysis.surfaces import test_annotation_drift
from gabion.analysis.surfaces import test_annotation_drift_delta
from gabion.analysis.surfaces import test_obsolescence
from gabion.analysis.surfaces import test_obsolescence_delta
from gabion.analysis.surfaces import test_obsolescence_state
from gabion.analysis.surfaces import test_evidence_suggestions
from gabion.analysis.foundation.timeout_context import (Deadline, GasMeter, TimeoutExceeded, check_deadline, get_deadline, get_deadline_clock, record_deadline_io, reset_deadline_clock, reset_forest, set_forest, reset_deadline_profile, reset_deadline, set_deadline_profile, set_deadline, set_deadline_clock)
from gabion.invariants import never
from gabion.order_contract import sort_once
from gabion.config import (dataflow_defaults, dataflow_deadline_roots, decision_defaults, decision_ignore_list, decision_require_tiers, decision_tier_map, exception_defaults, exception_never_list, fingerprint_defaults, taint_boundary_registry, taint_defaults, taint_profile, merge_payload)
from gabion.analysis.core.type_fingerprints import (Fingerprint, PrimeRegistry, TypeConstructorRegistry, build_fingerprint_registry)
from gabion.refactor.rewrite_plan import (normalize_rewrite_plan_order, validate_rewrite_plan_payload)
from gabion.schema import (DataflowCanonicalResponseDTO, DataflowResponseEnvelopeDTO, LintEntryDTO)
from gabion.server_core.ingress_primitives import (
    AnalysisDeps,
    ExecuteCommandDeps,
    OutputDeps,
    ProgressDeps,
    RuntimeDeps,
)
from gabion.server_core.coercion_contract import (
    _bool_optional,
    _float_optional,
    _int_optional,
    _json_mapping_default_empty,
    _json_mapping_optional,
    _non_negative_int_optional,
    _non_string_sequence_optional,
    _str_optional,
)
from gabion.server_core import dataflow_runtime_contract as runtime_contract
from gabion.server_core.command_orchestrator_progress import (
    _analysis_index_resume_hydrated_count,
    _analysis_index_resume_signature,
    _analysis_resume_progress,
    _build_phase_progress_v2,
    _collection_semantic_progress,
    _in_progress_scan_states,
    _normalize_progress_work,
    _report_projection_phase_rank_optional,
)

DATAFLOW_COMMAND = command_ids.DATAFLOW_COMMAND

_SERVER_DEADLINE_OVERHEAD_MIN_NS = 10_000_000

_SERVER_DEADLINE_OVERHEAD_MAX_NS = 200_000_000

_SERVER_DEADLINE_OVERHEAD_DIVISOR = 20

_ANALYSIS_TIMEOUT_GRACE_RATIO_NUMERATOR = 1

_ANALYSIS_TIMEOUT_GRACE_RATIO_DENOMINATOR = 5

_ANALYSIS_INPUT_MANIFEST_FORMAT_VERSION = 1

_DEFAULT_PHASE_TIMELINE_MD = runtime_contract.DEFAULT_PHASE_TIMELINE_MD

_DEFAULT_PHASE_TIMELINE_JSONL = runtime_contract.DEFAULT_PHASE_TIMELINE_JSONL

_REPORT_SECTION_JOURNAL_FORMAT_VERSION = 1

_DEFAULT_REPORT_SECTION_JOURNAL = runtime_contract.DEFAULT_REPORT_SECTION_JOURNAL

_COLLECTION_CHECKPOINT_FLUSH_INTERVAL_NS = runtime_contract.COLLECTION_CHECKPOINT_FLUSH_INTERVAL_NS

_COLLECTION_CHECKPOINT_MEANINGFUL_MIN_INTERVAL_NS = runtime_contract.COLLECTION_CHECKPOINT_MEANINGFUL_MIN_INTERVAL_NS

_COLLECTION_REPORT_FLUSH_INTERVAL_NS = runtime_contract.COLLECTION_REPORT_FLUSH_INTERVAL_NS

_COLLECTION_REPORT_FLUSH_COMPLETED_STRIDE = runtime_contract.COLLECTION_REPORT_FLUSH_COMPLETED_STRIDE

_DEFAULT_DEADLINE_PROFILE_SAMPLE_INTERVAL = 16

_DEFAULT_PROGRESS_HEARTBEAT_SECONDS = runtime_contract.DEFAULT_PROGRESS_HEARTBEAT_SECONDS

_MIN_PROGRESS_HEARTBEAT_SECONDS = runtime_contract.MIN_PROGRESS_HEARTBEAT_SECONDS

_PROGRESS_DEADLINE_FLUSH_SECONDS = runtime_contract.PROGRESS_DEADLINE_FLUSH_SECONDS

_PROGRESS_DEADLINE_WATCHDOG_SECONDS = runtime_contract.PROGRESS_DEADLINE_WATCHDOG_SECONDS

_PROGRESS_HEARTBEAT_POLL_SECONDS = runtime_contract.PROGRESS_HEARTBEAT_POLL_SECONDS

_PROGRESS_DEADLINE_FLUSH_MARGIN_SECONDS = runtime_contract.PROGRESS_DEADLINE_FLUSH_MARGIN_SECONDS


_LSP_PROGRESS_NOTIFICATION_METHOD = runtime_contract.LSP_PROGRESS_NOTIFICATION_METHOD

_LSP_PROGRESS_TOKEN_V2 = runtime_contract.LSP_PROGRESS_TOKEN_V2

_LSP_PROGRESS_TOKEN = _LSP_PROGRESS_TOKEN_V2

_CANONICAL_PROGRESS_EVENT_SCHEMA_V2 = "gabion/canonical_progress_event_v2"

_STDOUT_ALIAS = runtime_contract.STDOUT_ALIAS

_STDOUT_PATH = runtime_contract.STDOUT_PATH

def _is_stdout_target(target: object) -> bool:
    return runtime_contract.is_stdout_target(target)

def _analysis_resume_cache_verdict(
    *,
    status: str | None,
    reused_files: int,
    compatibility_status: str | None,
) -> Literal["hit", "miss", "invalidated", "seeded"]:
    invalidation_statuses = {
        "checkpoint_manifest_mismatch",
        "checkpoint_unreadable",
        "checkpoint_invalid_payload",
        "checkpoint_manifest_missing",
        "checkpoint_missing_collection_resume",
        "aspf_state_manifest_mismatch",
        "aspf_state_missing_manifest_digest",
        "aspf_state_missing_current_manifest_digest",
        "aspf_state_missing_collection_resume",
    }
    if status in {"checkpoint_loaded", "aspf_state_loaded"}:
        if reused_files > 0:
            return "hit"
        return "miss"
    if status == "checkpoint_seeded":
        if compatibility_status in invalidation_statuses:
            return "invalidated"
        return "seeded"
    return "invalidated" if compatibility_status in invalidation_statuses else "miss"

def _deadline_tick_budget_allows_check(clock: object) -> bool:
    return runtime_contract.deadline_tick_budget_allows_check(clock)

def _collection_checkpoint_flush_due(
    *,
    intro_changed: bool,
    remaining_files: int,
    semantic_substantive_progress: bool = False,
    now_ns: int,
    last_flush_ns: int,
) -> bool:
    return runtime_contract.collection_checkpoint_flush_due(
        intro_changed=intro_changed,
        remaining_files=remaining_files,
        semantic_substantive_progress=semantic_substantive_progress,
        now_ns=now_ns,
        last_flush_ns=last_flush_ns,
    )

def _collection_report_flush_due(
    *,
    completed_files: int,
    remaining_files: int,
    now_ns: int,
    last_flush_ns: int,
    last_flush_completed: int,
) -> bool:
    return runtime_contract.collection_report_flush_due(
        completed_files=completed_files,
        remaining_files=remaining_files,
        now_ns=now_ns,
        last_flush_ns=last_flush_ns,
        last_flush_completed=last_flush_completed,
    )

def _projection_phase_flush_due(
    *,
    phase: Literal["collection", "forest", "edge", "post"],
    now_ns: int,
    last_flush_ns: int,
) -> bool:
    return runtime_contract.projection_phase_flush_due(
        phase=phase,
        now_ns=now_ns,
        last_flush_ns=last_flush_ns,
    )

def _read_text_profiled(
    path: Path,
    *,
    io_name: str,
    encoding: str | None = None,
) -> str:
    started_ns = time.monotonic_ns()
    text = path.read_text() if encoding is None else path.read_text(encoding=encoding)
    record_deadline_io(
        name=io_name,
        elapsed_ns=time.monotonic_ns() - started_ns,
        bytes_count=len(text.encode("utf-8")),
    )
    return text

def _write_text_profiled(
    path: Path,
    text: str,
    *,
    io_name: str,
    encoding: str | None = None,
) -> None:
    started_ns = time.monotonic_ns()
    if encoding is None:
        path.write_text(text)
    else:
        path.write_text(text, encoding=encoding)
    record_deadline_io(
        name=io_name,
        elapsed_ns=time.monotonic_ns() - started_ns,
        bytes_count=len(text.encode("utf-8")),
    )

def _analysis_witness_config_payload(config: AuditConfig) -> JSONObject:
    return {
        "exclude_dirs": sort_once(
            config.exclude_dirs,
            source="_analysis_witness_config_payload.exclude_dirs",
        ),
        "ignore_params": sort_once(
            config.ignore_params,
            source="_analysis_witness_config_payload.ignore_params",
        ),
        "strictness": config.strictness,
        "external_filter": config.external_filter,
        "transparent_decorators": sort_once(
            config.transparent_decorators or [],
            source="_analysis_witness_config_payload.transparent_decorators",
        ),
    }

def _analysis_input_manifest(
    *,
    root: Path,
    file_paths: list[Path],
    recursive: bool,
    include_invariant_propositions: bool,
    include_wl_refinement: bool,
    config: AuditConfig,
) -> JSONObject:
    files = list(map(_analysis_input_manifest_file_entry, file_paths))
    return {
        "format_version": _ANALYSIS_INPUT_MANIFEST_FORMAT_VERSION,
        "root": str(root),
        "recursive": recursive,
        "include_invariant_propositions": include_invariant_propositions,
        "include_wl_refinement": include_wl_refinement,
        "config": _analysis_witness_config_payload(config),
        "files": files,
    }


def _analysis_input_manifest_file_entry(path: Path) -> JSONObject:
    check_deadline()
    entry: JSONObject = {"path": str(path)}
    try:
        stat = path.stat()
    except OSError:
        entry["missing"] = True
    else:
        entry["size"] = int(stat.st_size)
    return entry

def _analysis_input_manifest_digest(manifest: JSONObject) -> str:
    return hashlib.sha1(_canonical_json_text(manifest).encode("utf-8")).hexdigest()

def _canonical_json_text(payload: object) -> str:
    return json.dumps(payload, sort_keys=False, separators=(",", ":"), ensure_ascii=True)

def _load_aspf_resume_state(
    *,
    import_state_paths: Sequence[Path],
    include_delta_records: bool = False,
    diagnostic_tail_limit: int = 32,
) -> JSONObject | None:
    latest_manifest_digest: str | None = None
    latest_resume_source: str | None = None
    latest_has_resume_projection = False
    for path in import_state_paths:
        raw_payload = cast(
            Mapping[str, object],
            json.loads(path.read_text(encoding="utf-8")),
        )
        manifest_digest = str(raw_payload.get("analysis_manifest_digest", "") or "")
        latest_manifest_digest = manifest_digest or latest_manifest_digest
        resume_source = str(raw_payload.get("resume_source", "") or "")
        latest_resume_source = resume_source or latest_resume_source
        latest_has_resume_projection = "resume_projection" in raw_payload
    if latest_has_resume_projection:
        projection = cast(
            JSONObject,
            aspf_resume_state.load_latest_resume_projection_from_state_files(
                state_paths=import_state_paths,
            )
            or {},
        )
    else:
        projection = {}
    mutation_count = 0
    mutation_tail: list[JSONObject] = []
    for record in aspf_resume_state.iter_resume_mutations(state_paths=import_state_paths):
        mutation_count += 1
        if not latest_has_resume_projection:
            projection, _, _ = aspf_resume_state.fold_resume_mutations(
                snapshot=projection,
                mutations=(record,),
                tail_limit=0,
            )
        if diagnostic_tail_limit <= 0:
            continue
        mutation_tail.append(dict(record))
        if len(mutation_tail) > diagnostic_tail_limit:
            mutation_tail.pop(0)
    payload: JSONObject = {
        "resume_projection": projection,
        "delta_record_count": mutation_count,
        "delta_records_tail": mutation_tail,
        "analysis_manifest_digest": latest_manifest_digest,
        "resume_source": latest_resume_source,
    }
    if include_delta_records:
        payload["delta_records"] = [
            dict(record)
            for record in aspf_resume_state.iter_resume_mutations(
                state_paths=import_state_paths,
            )
        ]
    return payload

def _resolve_report_output_path(*, root: Path, report_path: str | None) -> Path | None:
    if report_path not in (None, "") and not _is_stdout_target(report_path):
        candidate = Path(str(report_path))
        if candidate.is_absolute():
            return candidate
        return root / candidate
    return None

def _resolve_report_section_journal_path(
    *,
    root: Path,
    report_path: str | None,
) -> Path | None:
    resolved_report = _resolve_report_output_path(root=root, report_path=report_path)
    if resolved_report is not None:
        default_journal = root / _DEFAULT_REPORT_SECTION_JOURNAL
        if resolved_report.name == "dataflow_report.md":
            return default_journal
        return resolved_report.with_name(f"{resolved_report.stem}_sections.json")
    return None

def _report_witness_digest(
    *,
    input_witness: Mapping[str, JSONValue] | None,
    manifest_digest: str | None,
) -> str | None:
    digest = input_witness.get("witness_digest") if input_witness is not None else None
    digest_text = _str_optional(digest)
    if digest_text == "":
        digest_text = None
    manifest_text = _str_optional(manifest_digest)
    if manifest_text == "":
        manifest_text = None
    return digest_text or manifest_text


def _is_not_none_text(value: str | None) -> bool:
    return value is not None


def _normalize_section_entry(
    item: tuple[object, object],
) -> tuple[str | None, Mapping[str, JSONValue] | None]:
    key, entry = item
    return _str_optional(key), _json_mapping_optional(entry)


def _has_section_key_mapping(
    item: tuple[str | None, Mapping[str, JSONValue] | None],
) -> bool:
    key_text, entry_mapping = item
    return key_text is not None and entry_mapping is not None


def _section_lines_item(
    item: tuple[str | None, Mapping[str, JSONValue] | None],
) -> tuple[str, list[str]]:
    key_text, entry_mapping = item
    if key_text is None or entry_mapping is None:
        never("invalid section entry shape")
    return key_text, _coerce_section_lines(entry_mapping.get("lines"))


def _has_section_lines(item: tuple[str, list[str]]) -> bool:
    _key, lines = item
    return bool(lines)


def _projection_row_section_id(row: Mapping[str, JSONValue]) -> str:
    return str(row.get("section_id", "") or "")


def _projection_row_has_section_id(row: Mapping[str, JSONValue]) -> bool:
    return bool(_projection_row_section_id(row))


def _projection_row_deps(row: Mapping[str, JSONValue]) -> list[str]:
    deps_raw = row.get("deps")
    normalized_deps = _non_string_sequence_optional(deps_raw) or ()
    return list(
        filter(
            _is_not_none_text,
            map(_str_optional, normalized_deps),
        )
    )


def _journal_section_status(
    *,
    section_id: str,
    sections: Mapping[str, list[str]],
) -> str:
    return "resolved" if section_id in sections else "pending"


def _journal_projection_row_payload(
    *,
    row: Mapping[str, JSONValue],
    sections: Mapping[str, list[str]],
    pending_reasons: Mapping[str, str],
) -> tuple[str, JSONObject, JSONObject]:
    section_id = _projection_row_section_id(row)
    phase = str(row.get("phase", "") or "")
    deps = _projection_row_deps(row)
    status = _journal_section_status(section_id=section_id, sections=sections)
    section_entry: JSONObject = {
        "phase": phase,
        "deps": deps,
        "status": status,
        "lines": sections.get(section_id, []),
        "reason": pending_reasons.get(section_id),
    }
    row_entry: JSONObject = {
        "section_id": section_id,
        "phase": phase,
        "deps": deps,
        "status": status,
    }
    return section_id, section_entry, row_entry


def _coerce_section_lines(value: object) -> list[str]:
    entries = _non_string_sequence_optional(value) or ()
    return list(
        filter(
            _is_not_none_text,
            map(_str_optional, entries),
        )
    )

def _load_report_section_journal(
    *,
    path: Path | None,
    witness_digest: str | None,
) -> tuple[dict[str, list[str]], str | None]:
    if path is None or not path.exists():
        return {}, None
    try:
        payload = json.loads(
            _read_text_profiled(path, io_name="report_section_journal.read")
        )
    except (OSError, UnicodeError, json.JSONDecodeError):
        return {}, "policy"
    payload_mapping = _json_mapping_optional(payload)
    if payload_mapping is None:
        return {}, "policy"
    if payload_mapping.get("format_version") != _REPORT_SECTION_JOURNAL_FORMAT_VERSION:
        return {}, "policy"
    expected_digest_text = _str_optional(payload_mapping.get("witness_digest"))
    witness_digest_text = _str_optional(witness_digest)
    if expected_digest_text is not None:
        if witness_digest_text is None or expected_digest_text != witness_digest_text:
            return {}, "stale_input"
    sections_payload = _json_mapping_optional(payload_mapping.get("sections"))
    if sections_payload is None:
        return {}, "policy"
    sections = dict(
        filter(
            _has_section_lines,
            map(
                _section_lines_item,
                filter(
                    _has_section_key_mapping,
                    map(_normalize_section_entry, sections_payload.items()),
                ),
            ),
        )
    )
    return sections, None

def _write_report_section_journal(
    *,
    path: Path | None,
    witness_digest: str | None,
    projection_rows: Sequence[Mapping[str, JSONValue]],
    sections: Mapping[str, list[str]],
    pending_reasons: Mapping[str, str] | None = None,
) -> None:
    if path is not None:
        pending_reasons = pending_reasons or {}
        row_payloads = tuple(
            map(
                lambda row: _journal_projection_row_payload(
                    row=row,
                    sections=sections,
                    pending_reasons=pending_reasons,
                ),
                filter(_projection_row_has_section_id, projection_rows),
            )
        )
        sections_payload: JSONObject = dict(
            map(lambda item: (item[0], item[1]), row_payloads)
        )
        rows_payload: list[JSONObject] = list(map(lambda item: item[2], row_payloads))
        payload: JSONObject = {
            "format_version": _REPORT_SECTION_JOURNAL_FORMAT_VERSION,
            "witness_digest": witness_digest,
            "sections": sections_payload,
            "projection_rows": rows_payload,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        _write_text_profiled(
            path,
            json.dumps(payload, indent=2, sort_keys=False) + "\n",
            io_name="report_section_journal.write",
        )


@dataclass(frozen=True)
class _BootstrapPendingRow:
    section_id: str
    phase: str
    deps: list[str]
    reason: str


def _deadline_tick() -> None:
    if _deadline_tick_budget_allows_check(get_deadline_clock()):
        check_deadline(allow_frame_fallback=False)
    else:
        get_deadline()


def _bootstrap_existing_reason(
    *,
    report_section_journal_path: Path | None,
    witness_digest: str | None,
) -> str | None:
    if report_section_journal_path is None or not report_section_journal_path.exists():
        return None
    try:
        existing_payload = json.loads(
            _read_text_profiled(
                report_section_journal_path,
                io_name="report_section_journal.read",
            )
        )
    except (OSError, UnicodeError, json.JSONDecodeError):
        return "policy"
    existing_payload_mapping = _json_mapping_optional(existing_payload)
    if existing_payload_mapping is None:
        return "policy"
    if (
        existing_payload_mapping.get("format_version")
        != _REPORT_SECTION_JOURNAL_FORMAT_VERSION
    ):
        return "policy"
    expected_digest = _str_optional(existing_payload_mapping.get("witness_digest"))
    witness_digest_text = _str_optional(witness_digest)
    if expected_digest and (witness_digest_text is None or expected_digest != witness_digest_text):
        return "stale_input"
    return None


def _is_pending_bootstrap_section(section_id: str) -> bool:
    return bool(section_id) and section_id != "intro"


def _pending_reason_from_deps(
    *,
    deps: Sequence[str],
    sections: Mapping[str, list[str]],
) -> str:
    return "missing_dep" if not set(deps).issubset(set(sections)) else "policy"


def _bootstrap_pending_row(
    *,
    row: Mapping[str, JSONValue],
    sections: Mapping[str, list[str]],
    existing_reason: str | None,
) -> _BootstrapPendingRow:
    _deadline_tick()
    section_id = _projection_row_section_id(row)
    phase = str(row.get("phase", "") or "")
    deps = _projection_row_deps(row)
    reason = existing_reason or _pending_reason_from_deps(deps=deps, sections=sections)
    return _BootstrapPendingRow(
        section_id=section_id,
        phase=phase,
        deps=deps,
        reason=reason,
    )


def _bootstrap_report_block(row: _BootstrapPendingRow) -> tuple[str, str, str]:
    dep_text = ", ".join(row.deps) or "none"
    return (
        f"## Section `{row.section_id}`",
        f"PENDING (phase: {row.phase}; deps: {dep_text})",
        "",
    )


def _bootstrap_row_payload(row: _BootstrapPendingRow) -> JSONObject:
    return {
        "section_id": row.section_id,
        "phase": row.phase,
        "deps": row.deps,
        "status": "pending",
    }


def _bootstrap_section_payload_item(row: _BootstrapPendingRow) -> tuple[str, JSONObject]:
    return row.section_id, {
        "phase": row.phase,
        "deps": row.deps,
        "status": "pending",
        "lines": [],
        "reason": row.reason,
    }

def _write_bootstrap_incremental_artifacts(
    *,
    report_output_path: Path | None,
    report_section_journal_path: Path | None,
    report_phase_checkpoint_path: Path | None,
    witness_digest: str | None,
    root: Path,
    paths_requested: int,
    projection_rows: Sequence[Mapping[str, JSONValue]],
    phase_checkpoint_state: JSONObject,
) -> None:
    _ = report_phase_checkpoint_path
    get_deadline()
    _deadline_tick()
    bootstrap_artifacts_ready = report_output_path is not None and bool(projection_rows)
    if bootstrap_artifacts_ready:
        existing_reason = _bootstrap_existing_reason(
            report_section_journal_path=report_section_journal_path,
            witness_digest=witness_digest,
        )
        intro_lines = [
            "Collection bootstrap checkpoint (provisional).",
            f"- `root`: `{root}`",
            f"- `paths_requested`: `{paths_requested}`",
        ]
        sections: dict[str, list[str]] = {"intro": intro_lines}
        pending_rows = tuple(
            filter(
                lambda row: _is_pending_bootstrap_section(row.section_id),
                map(
                    lambda row: _bootstrap_pending_row(
                        row=row,
                        sections=sections,
                        existing_reason=existing_reason,
                    ),
                    projection_rows,
                ),
            )
        )
        report_lines = [
            "<!-- dataflow-grammar -->",
            "Dataflow grammar audit (observed forwarding bundles).",
            "",
            "## Incremental Status",
            "",
            "- `analysis_state`: `analysis_bootstrap_in_progress`",
            "",
            "## Section `intro`",
            *intro_lines,
            "",
            *chain.from_iterable(map(_bootstrap_report_block, pending_rows)),
        ]
        rows_payload: list[JSONObject] = [
            {
                "section_id": "intro",
                "phase": "collection",
                "deps": [],
                "status": "resolved",
            },
            *list(map(_bootstrap_row_payload, pending_rows)),
        ]
        sections_payload: JSONObject = {
            "intro": {
                "phase": "collection",
                "deps": [],
                "status": "resolved",
                "lines": intro_lines,
            }
        }
        sections_payload.update(dict(map(_bootstrap_section_payload_item, pending_rows)))
        report_output_path.parent.mkdir(parents=True, exist_ok=True)
        _write_text_profiled(
            report_output_path,
            "\n".join(report_lines).rstrip() + "\n",
            io_name="report_markdown.write",
        )
        if report_section_journal_path is not None:
            report_section_journal_path.parent.mkdir(parents=True, exist_ok=True)
            _write_text_profiled(
                report_section_journal_path,
                json.dumps(
                    {
                        "format_version": _REPORT_SECTION_JOURNAL_FORMAT_VERSION,
                        "witness_digest": witness_digest,
                        "sections": sections_payload,
                        "projection_rows": rows_payload,
                    },
                    indent=2,
                    sort_keys=False,
                )
                + "\n",
                io_name="report_section_journal.write",
            )
        phase_checkpoint_state["collection"] = {
            "status": "bootstrap",
            "work_done": 0,
            "work_total": 0,
            "completed_files": 0,
            "in_progress_files": 0,
            "remaining_files": 0,
            "section_ids": sort_once(sections, source = 'src/gabion/server.py:2075'),
        }


@dataclass(frozen=True)
class _IncrementalProjectionRow:
    section_id: str
    phase: str
    deps: list[str]


def _incremental_projection_row(row: Mapping[str, JSONValue]) -> _IncrementalProjectionRow:
    check_deadline()
    return _IncrementalProjectionRow(
        section_id=_projection_row_section_id(row),
        phase=str(row.get("phase", "") or ""),
        deps=_projection_row_deps(row),
    )


def _has_incremental_projection_section(row: _IncrementalProjectionRow) -> bool:
    return bool(row.section_id)


def _resolved_section_block(
    *,
    section_id: str,
    section_lines: Sequence[str],
) -> tuple[str, ...]:
    return (
        f"## Section `{section_id}`",
        *tuple(section_lines),
        "",
    )


def _pending_section_reason(
    *,
    deps: Sequence[str],
    sections: Mapping[str, list[str]],
) -> str:
    return "missing_dep" if not set(deps).issubset(set(sections)) else "policy"


def _pending_section_block(row: _IncrementalProjectionRow) -> tuple[str, str, str]:
    dep_text = ", ".join(row.deps) or "none"
    return (
        f"## Section `{row.section_id}`",
        f"PENDING (phase: {row.phase}; deps: {dep_text})",
        "",
    )


def _render_incremental_report(
    *,
    analysis_state: str,
    progress_payload: Mapping[str, JSONValue] | None,
    projection_rows: Sequence[Mapping[str, JSONValue]],
    sections: Mapping[str, list[str]],
) -> tuple[str, dict[str, str]]:
    lines = [
        "<!-- dataflow-grammar -->",
        "Dataflow grammar audit (observed forwarding bundles).",
        "",
        "## Incremental Status",
        "",
        f"- `analysis_state`: `{analysis_state}`",
    ]
    progress_payload_mapping = _json_mapping_optional(progress_payload)
    if progress_payload_mapping is not None:
        phase = _str_optional(progress_payload_mapping.get("phase"))
        if phase:
            lines.append(f"- `phase`: `{phase}`")
        event_kind = _str_optional(progress_payload_mapping.get("event_kind"))
        if event_kind:
            lines.append(f"- `event_kind`: `{event_kind}`")
        work_done_raw = _int_optional(progress_payload_mapping.get("work_done"))
        work_total_raw = _int_optional(progress_payload_mapping.get("work_total"))
        if work_done_raw is not None and work_total_raw is not None:
            work_done = max(work_done_raw, 0)
            work_total = max(work_total_raw, 0)
            if work_total > 0:
                work_done = min(work_done, work_total)
            lines.append(f"- `work_done`: `{work_done}`")
            lines.append(f"- `work_total`: `{work_total}`")
            if work_total > 0:
                lines.append(
                    f"- `work_percent`: `{(100.0 * work_done / work_total):.2f}`"
                )
        phase_progress_v2 = _json_mapping_optional(
            progress_payload_mapping.get("phase_progress_v2")
        )
        if phase_progress_v2 is not None:
            primary_unit = str(phase_progress_v2.get("primary_unit", "") or "")
            raw_primary_done = phase_progress_v2.get("primary_done")
            raw_primary_total = phase_progress_v2.get("primary_total")
            primary_done = _non_negative_int_optional(raw_primary_done)
            primary_total = _non_negative_int_optional(raw_primary_total)
            if primary_done is not None and primary_total is not None:
                if primary_total:
                    primary_done = min(primary_done, primary_total)
                lines.append(
                    f"- `primary_progress`: `{primary_done}/{primary_total}`"
                )
            if primary_unit:
                lines.append(f"- `primary_unit`: `{primary_unit}`")
            dimensions_summary = _phase_progress_dimensions_summary(phase_progress_v2)
            if dimensions_summary:
                lines.append(f"- `dimensions`: `{dimensions_summary}`")
        stale_for_s_raw = progress_payload_mapping.get("stale_for_s")
        stale_for_s_float = _float_optional(stale_for_s_raw)
        stale_for_s_int = _int_optional(stale_for_s_raw)
        if stale_for_s_float is not None:
            lines.append(f"- `stale_for_s`: `{float(stale_for_s_float):.1f}`")
        elif stale_for_s_int is not None:
            lines.append(f"- `stale_for_s`: `{float(stale_for_s_int):.1f}`")
        classification = _str_optional(progress_payload_mapping.get("classification"))
        if classification is not None:
            lines.append(f"- `classification`: `{classification}`")
        retry_recommended = _bool_optional(
            progress_payload_mapping.get("retry_recommended")
        )
        if retry_recommended is not None:
            lines.append(f"- `retry_recommended`: `{retry_recommended}`")
        resume_supported = _bool_optional(
            progress_payload_mapping.get("resume_supported")
        )
        if resume_supported is not None:
            lines.append(f"- `resume_supported`: `{resume_supported}`")
    lines.append("")

    projection_state_rows = tuple(
        filter(
            _has_incremental_projection_section,
            map(_incremental_projection_row, projection_rows),
        )
    )
    resolved_rows = tuple(
        filter(
            lambda row: bool(sections.get(row.section_id)),
            projection_state_rows,
        )
    )
    pending_rows = tuple(
        filter(
            lambda row: not sections.get(row.section_id),
            projection_state_rows,
        )
    )
    pending_reasons = dict(
        map(
            lambda row: (
                row.section_id,
                _pending_section_reason(deps=row.deps, sections=sections),
            ),
            pending_rows,
        )
    )
    lines.extend(
        chain.from_iterable(
            map(
                lambda row: _resolved_section_block(
                    section_id=row.section_id,
                    section_lines=sections.get(row.section_id, []),
                ),
                resolved_rows,
            )
        )
    )
    lines.extend(chain.from_iterable(map(_pending_section_block, pending_rows)))
    return "\n".join(lines).rstrip() + "\n", pending_reasons

def _phase_timeline_md_path(*, root: Path) -> Path:
    return root / _DEFAULT_PHASE_TIMELINE_MD

def _phase_timeline_jsonl_path(*, root: Path) -> Path:
    return root / _DEFAULT_PHASE_TIMELINE_JSONL

def _progress_heartbeat_seconds(payload: Mapping[str, JSONValue]) -> float:
    return runtime_contract.progress_heartbeat_seconds(payload)

def _markdown_table_cell(value: object) -> str:
    return ("" if value is None else str(value)).replace("\n", " ").replace("|", "\\|")


def _phase_dimension_fragment(
    *,
    raw_dimensions: Mapping[str, JSONValue],
    dim_name: str,
) -> str:
    payload_mapping = _json_mapping_optional(raw_dimensions.get(dim_name))
    if payload_mapping is None:
        return ""
    done = _non_negative_int_optional(payload_mapping.get("done"))
    total = _non_negative_int_optional(payload_mapping.get("total"))
    if done is None or total is None:
        return ""
    normalized_done = min(done, total) if total else done
    return f"{dim_name}={normalized_done}/{total}"


def _phase_progress_dimensions_summary(
    phase_progress_v2: Mapping[str, JSONValue] | None,
) -> str:
    phase_progress_v2_mapping = _json_mapping_optional(phase_progress_v2)
    if phase_progress_v2_mapping is None:
        return ""
    raw_dimensions = _json_mapping_optional(phase_progress_v2_mapping.get("dimensions"))
    if raw_dimensions is None:
        return ""
    dim_names = sort_once(
        filter(_is_not_none_text, map(_str_optional, raw_dimensions)),
        source="src/gabion/server.py:2253",
    )
    fragments = tuple(
        filter(
            bool,
            map(
                lambda dim_name: _phase_dimension_fragment(
                    raw_dimensions=raw_dimensions,
                    dim_name=dim_name,
                ),
                dim_names,
            ),
        )
    )
    return "; ".join(fragments)

def _append_phase_timeline_event(
    *,
    markdown_path: Path,
    jsonl_path: Path,
    progress_value: Mapping[str, JSONValue],
) -> tuple[str | None, str]:
    header = _phase_timeline_header_columns()
    header_line = "| " + " | ".join(_markdown_table_cell(cell) for cell in header) + " |"
    separator_line = "| " + " | ".join(["---"] * len(header)) + " |"
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    row_line = progress_timeline.phase_timeline_row_from_phase_progress(progress_value)
    header_block: str | None = None
    if not markdown_path.exists():
        with markdown_path.open("w", encoding="utf-8") as handle:
            handle.write("# Dataflow Phase Timeline\n\n")
            handle.write(header_line + "\n")
            handle.write(separator_line + "\n")
        header_block = header_line + "\n" + separator_line
    with markdown_path.open("a", encoding="utf-8") as handle:
        handle.write(row_line + "\n")
    with jsonl_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(progress_value, sort_keys=False) + "\n")
    return header_block, row_line

def _phase_timeline_header_columns() -> list[str]:
    return progress_timeline.phase_timeline_header_columns()

def _phase_timeline_header_block() -> str:
    return progress_timeline.phase_timeline_header_block()


def _in_progress_state_processed_count(state_mapping: Mapping[str, JSONValue]) -> int:
    processed_count_value = _int_optional(state_mapping.get("processed_functions_count"))
    if processed_count_value is not None:
        return processed_count_value
    raw_processed = state_mapping.get("processed_functions")
    processed_entries = _non_string_sequence_optional(raw_processed) or ()
    return sum(
        map(
            int,
            map(
                _is_not_none_text,
                map(_str_optional, processed_entries),
            ),
        )
    )


def _in_progress_state_function_count(state_mapping: Mapping[str, JSONValue]) -> int:
    function_count_value = _int_optional(state_mapping.get("function_count"))
    if function_count_value is not None:
        return function_count_value
    raw_fn_names = state_mapping.get("fn_names")
    fn_name_mapping = _json_mapping_optional(raw_fn_names) or {}
    return sum(
        map(
            int,
            map(
                _is_not_none_text,
                map(_str_optional, fn_name_mapping.keys()),
            ),
        )
    )


def _in_progress_detail_entry(item: tuple[str, Mapping[str, JSONValue]]) -> str:
    raw_path, state_mapping = item
    check_deadline()
    phase_text = _str_optional(state_mapping.get("phase")) or "unknown"
    processed_count = _in_progress_state_processed_count(state_mapping)
    function_count = _in_progress_state_function_count(state_mapping)
    return (
        f"{raw_path} "
        f"(phase={phase_text}, processed_functions={processed_count}, "
        f"function_count={function_count})"
    )


def _in_progress_detail_line(detail: str) -> str:
    return f"- `in_progress_detail`: `{detail}`"


def _collection_progress_intro_lines(
    *,
    collection_resume: Mapping[str, JSONValue],
    total_files: int,
    resume_state_intro: Mapping[str, JSONValue] | None = None,
) -> list[str]:
    check_deadline()
    progress = _analysis_resume_progress(
        collection_resume=collection_resume,
        total_files=total_files,
    )
    lines = [
        "Collection progress checkpoint (provisional).",
        f"- `completed_files`: `{progress['completed_files']}`",
        f"- `in_progress_files`: `{progress['in_progress_files']}`",
        f"- `remaining_files`: `{progress['remaining_files']}`",
        f"- `total_files`: `{progress['total_files']}`",
    ]
    resume_state = _json_mapping_default_empty(resume_state_intro)
    state_path = str(resume_state.get("state_path", "") or "")
    status = str(resume_state.get("status", "") or "")
    reused_files = int(resume_state.get("reused_files", 0) or 0)
    total_resume_files = int(
        resume_state.get("total_files", progress["total_files"]) or 0
    )
    lines.append(
        "- `resume_state`: "
        f"`path={state_path or '<none>'} "
        f"status={status or 'unknown'} "
        f"reused_files={reused_files}/{total_resume_files}`"
    )
    semantic_progress = _json_mapping_optional(collection_resume.get("semantic_progress"))
    if semantic_progress is not None:
        semantic_witness_digest = _str_optional(
            semantic_progress.get("current_witness_digest")
        )
        if semantic_witness_digest:
            lines.append(f"- `semantic_witness_digest`: `{semantic_witness_digest}`")
        new_processed_functions = _int_optional(
            semantic_progress.get("new_processed_functions_count")
        )
        if new_processed_functions is not None:
            lines.append(f"- `new_processed_functions`: `{new_processed_functions}`")
        regressed_processed_functions = _int_optional(
            semantic_progress.get(
            "regressed_processed_functions_count"
            )
        )
        if regressed_processed_functions is not None:
            lines.append(
                f"- `regressed_processed_functions`: `{regressed_processed_functions}`"
            )
        completed_delta = _int_optional(semantic_progress.get("completed_files_delta"))
        if completed_delta is not None:
            lines.append(f"- `completed_files_delta`: `{completed_delta}`")
        substantive_progress = _bool_optional(
            semantic_progress.get("substantive_progress")
        )
        if substantive_progress is not None:
            lines.append(f"- `substantive_progress`: `{substantive_progress}`")
    in_progress_states = _in_progress_scan_states(collection_resume)
    if in_progress_states:
        in_progress_paths = tuple(map(str, in_progress_states.keys()))
        sample = ", ".join(in_progress_paths[:3])
        lines.append(f"- `in_progress_path_sample`: `{sample}`")
        detail_entries = tuple(
            map(
                _in_progress_detail_entry,
                tuple(in_progress_states.items())[:3],
            )
        )
        lines.extend(map(_in_progress_detail_line, detail_entries))
    raw_analysis_index_resume = _json_mapping_optional(
        collection_resume.get("analysis_index_resume")
    )
    if raw_analysis_index_resume is not None:
        hydrated_paths_count = _int_optional(
            raw_analysis_index_resume.get("hydrated_paths_count")
        )
        if hydrated_paths_count is None:
            hydrated_paths_count = _analysis_index_resume_hydrated_count(collection_resume)
        lines.append(f"- `hydrated_paths_count`: `{hydrated_paths_count}`")
        function_count = _non_negative_int_optional(
            raw_analysis_index_resume.get("function_count")
        )
        if function_count is not None:
            lines.append(f"- `hydrated_function_count`: `{function_count}`")
        class_count = _non_negative_int_optional(
            raw_analysis_index_resume.get("class_count")
        )
        if class_count is not None:
            lines.append(f"- `hydrated_class_count`: `{class_count}`")
    return lines

def _collection_components_preview_lines(
    *,
    collection_resume: Mapping[str, JSONValue],
) -> list[str]:
    check_deadline()
    raw_groups = _json_mapping_optional(collection_resume.get("groups_by_path"))
    if raw_groups is None:
        return [
            "Component preview (provisional).",
            "- `paths_with_groups`: `0`",
            "- `functions_with_groups`: `0`",
            "- `bundle_alternatives`: `0`",
        ]
    path_items = tuple(
        filter(
            lambda item: item[0] is not None and item[1] is not None,
            map(
                lambda item: (
                    _str_optional(item[0]),
                    _json_mapping_optional(item[1]),
                ),
                raw_groups.items(),
            ),
        )
    )
    function_bundle_items = tuple(
        chain.from_iterable(
            map(
                lambda path_item: filter(
                    lambda item: item[0] is not None and item[1] is not None,
                    map(
                        lambda item: (
                            _str_optional(item[0]),
                            _non_string_sequence_optional(item[1]),
                        ),
                        path_item[1].items(),
                    ),
                ),
                path_items,
            )
        )
    )
    path_count = len(path_items)
    function_count = len(function_bundle_items)
    bundle_alternatives = sum(
        map(
            lambda item: sum(
                map(
                    int,
                    map(
                        lambda bundle: _non_string_sequence_optional(bundle) is not None,
                        item[1],
                    ),
                )
            ),
            function_bundle_items,
        )
    )
    return [
        "Component preview (provisional).",
        f"- `paths_with_groups`: `{path_count}`",
        f"- `functions_with_groups`: `{function_count}`",
        f"- `bundle_alternatives`: `{bundle_alternatives}`",
    ]

def _groups_by_path_from_collection_resume(
    collection_resume: Mapping[str, JSONValue],
) -> dict[Path, dict[str, list[set[str]]]]:
    check_deadline()
    raw_groups = _json_mapping_optional(collection_resume.get("groups_by_path")) or {}
    path_items = tuple(
        filter(
            lambda item: item[0] is not None and item[1] is not None,
            map(
                lambda item: (
                    _str_optional(item[0]),
                    _json_mapping_optional(item[1]),
                ),
                raw_groups.items(),
            ),
        )
    )
    return dict(
        map(
            lambda path_item: (
                Path(path_item[0]),
                dict(
                    map(
                        lambda item: (
                            item[0],
                            list(
                                filter(
                                    lambda bundle: bundle is not None,
                                    map(
                                        lambda raw_bundle: (
                                            set(
                                                filter(
                                                    _is_not_none_text,
                                                    map(
                                                        _str_optional,
                                                        _non_string_sequence_optional(raw_bundle)
                                                        or (),
                                                    ),
                                                )
                                            )
                                            if _non_string_sequence_optional(raw_bundle) is not None
                                            else None
                                        ),
                                        item[1],
                                    ),
                                )
                            ),
                        ),
                        filter(
                            lambda item: item[0] is not None and item[1] is not None,
                            map(
                                lambda item: (
                                    _str_optional(item[0]),
                                    _non_string_sequence_optional(item[1]),
                                ),
                                path_item[1].items(),
                            ),
                        ),
                    )
                ),
            ),
            path_items,
        )
    )


def _section_projection_obligation(
    *,
    row: Mapping[str, JSONValue],
    sections: Mapping[str, list[str]],
    pending_reasons: Mapping[str, str],
    stale_input_detected: bool,
) -> JSONObject:
    section_id = _projection_row_section_id(row)
    if section_id in sections:
        status = "SATISFIED"
        detail = "section reused from witness-matched journal"
    else:
        status = "OBLIGATION"
        pending_detail = pending_reasons.get(section_id, "section pending")
        detail = "stale_input" if stale_input_detected and pending_detail == "policy" else pending_detail
    return {
        "status": status,
        "contract": "incremental_projection_contract",
        "kind": "section_projection_state",
        "section_id": section_id,
        "phase": str(row.get("phase", "") or ""),
        "detail": detail,
    }


def _incremental_progress_obligations(
    *,
    analysis_state: str,
    progress_payload: Mapping[str, JSONValue] | None,
    resume_payload_available: bool,
    partial_report_written: bool,
    report_requested: bool,
    projection_rows: Sequence[Mapping[str, JSONValue]],
    sections: Mapping[str, list[str]],
    pending_reasons: Mapping[str, str],
) -> list[JSONObject]:
    check_deadline()
    obligations: list[JSONObject] = []
    normalized_progress_payload = _json_mapping_optional(progress_payload)
    if normalized_progress_payload is None:
        classification = ""
        resume_supported = False
        semantic_progress = None
    else:
        classification = str(normalized_progress_payload.get("classification", "") or "")
        resume_supported = bool(normalized_progress_payload.get("resume_supported"))
        semantic_progress = _json_mapping_optional(
            normalized_progress_payload.get("semantic_progress")
        )
    semantic_monotonic_progress: bool | None = None
    semantic_substantive_progress: bool | None = None
    if semantic_progress is not None:
        raw_monotonic = semantic_progress.get("monotonic_progress")
        raw_substantive = semantic_progress.get("substantive_progress")
        monotonic_value = _bool_optional(raw_monotonic)
        substantive_value = _bool_optional(raw_substantive)
        if monotonic_value is not None:
            semantic_monotonic_progress = monotonic_value
        if substantive_value is not None:
            semantic_substantive_progress = substantive_value
    is_timeout_state = analysis_state.startswith("timed_out_")
    if is_timeout_state:
        expected_progress = analysis_state == "timed_out_progress_resume"
        classification_ok = (expected_progress and resume_supported) or (
            not expected_progress and not resume_supported
        )
    else:
        classification_ok = True
    obligations.append(
        {
            "status": "SATISFIED" if classification_ok else "VIOLATION",
            "contract": "resume_contract",
            "kind": "classification_matches_resume_support",
            "detail": (
                f"analysis_state={analysis_state} classification={classification} "
                f"resume_supported={resume_supported}"
            ),
        }
    )
    if semantic_monotonic_progress is not None:
        obligations.append(
            {
                "status": "SATISFIED" if semantic_monotonic_progress else "VIOLATION",
                "contract": "resume_contract",
                "kind": "progress_monotonicity",
                "detail": (
                    "semantic progress is monotonic"
                    if semantic_monotonic_progress
                    else "semantic progress regression detected"
                ),
            }
        )
    if is_timeout_state and semantic_substantive_progress is not None:
        expected_substantive_progress = analysis_state == "timed_out_progress_resume"
        obligations.append(
            {
                "status": (
                    "SATISFIED"
                    if semantic_substantive_progress == expected_substantive_progress
                    else "VIOLATION"
                ),
                "contract": "resume_contract",
                "kind": "substantive_progress_required",
                "detail": (
                    f"analysis_state={analysis_state} "
                    f"substantive_progress={semantic_substantive_progress}"
                ),
            }
        )

    resume_payload_ok = True
    if resume_supported and is_timeout_state:
        resume_payload_ok = resume_payload_available
    obligations.append(
        {
            "status": "SATISFIED" if resume_payload_ok else "VIOLATION",
            "contract": "resume_contract",
            "kind": "resume_payload_present_when_resumable",
            "detail": "resume payload available"
            if resume_payload_ok
            else "resume payload missing",
        }
    )
    stale_input_detected = any(
        reason == "stale_input" for reason in pending_reasons.values()
    )
    if stale_input_detected and sections:
        restart_status = "VIOLATION"
        restart_detail = "witness_mismatch_with_reused_sections"
    elif stale_input_detected:
        restart_status = "SATISFIED"
        restart_detail = "restart_required"
    else:
        restart_status = "OBLIGATION"
        restart_detail = "no_witness_mismatch"
    obligations.append(
        {
            "status": restart_status,
            "contract": "resume_contract",
            "kind": "restart_required_on_witness_mismatch",
            "detail": restart_detail,
        }
    )
    if report_requested and is_timeout_state:
        projection_count = sum(
            map(
                int,
                map(_projection_row_has_section_id, projection_rows),
            )
        )
        resolved_count = len(sections)
        obligations.append(
            {
                "status": "SATISFIED" if resolved_count > 0 else "VIOLATION",
                "contract": "resume_contract",
                "kind": "no_projection_progress",
                "detail": (
                    f"resolved_sections={resolved_count} "
                    f"projected_sections={projection_count}"
                ),
            }
        )

    if report_requested and is_timeout_state:
        obligations.append(
            {
                "status": "SATISFIED" if partial_report_written else "VIOLATION",
                "contract": "progress_report_contract",
                "kind": "partial_report_emitted",
                "detail": "partial report emission on timeout",
            }
        )
    elif report_requested:
        obligations.append(
            {
                "status": "SATISFIED",
                "contract": "progress_report_contract",
                "kind": "partial_report_emitted",
                "detail": "not_applicable_without_timeout",
            }
        )
    obligations.extend(
        map(
            lambda row: _section_projection_obligation(
                row=row,
                sections=sections,
                pending_reasons=pending_reasons,
                stale_input_detected=stale_input_detected,
            ),
            filter(_projection_row_has_section_id, projection_rows),
        )
    )
    return obligations


def _is_not_none_mapping_entry(
    entry: Mapping[str, JSONValue] | None,
) -> bool:
    return entry is not None


def _json_object_entry(entry_mapping: Mapping[str, JSONValue]) -> JSONObject:
    return dict(map(lambda item: (str(item[0]), item[1]), entry_mapping.items()))


def _contract_name(entry: Mapping[str, JSONValue]) -> str:
    return str(entry.get("contract", "") or "")


def _is_resume_contract_entry(entry: Mapping[str, JSONValue]) -> bool:
    return _contract_name(entry) == "resume_contract"


def _is_incremental_contract_entry(entry: Mapping[str, JSONValue]) -> bool:
    return _contract_name(entry) in {"progress_report_contract", "incremental_projection_contract"}


def _split_incremental_obligations(
    obligations: Sequence[Mapping[str, JSONValue]],
) -> tuple[list[JSONObject], list[JSONObject]]:
    check_deadline()
    normalized_entries = tuple(
        map(
            _json_object_entry,
            filter(
                _is_not_none_mapping_entry,
                map(_json_mapping_optional, obligations),
            ),
        )
    )
    resumability = list(filter(_is_resume_contract_entry, normalized_entries))
    incremental = list(filter(_is_incremental_contract_entry, normalized_entries))
    return resumability, incremental

def _apply_journal_pending_reason(
    *,
    projection_rows: Sequence[Mapping[str, JSONValue]],
    sections: Mapping[str, object],
    pending_reasons: dict[str, str],
    journal_reason: str | None,
) -> None:
    if journal_reason in {"stale_input", "policy"}:
        section_ids = map(_projection_row_section_id, projection_rows)
        pending_section_ids = filter(
            lambda section_id: section_id and section_id not in sections,
            section_ids,
        )
        pending_reasons.update(dict(zip(pending_section_ids, repeat(journal_reason))))

def _latest_report_phase(phases: Mapping[str, JSONValue] | None) -> str | None:
    check_deadline()
    phase_mapping = _json_mapping_optional(phases)
    phase_names: Mapping[str, JSONValue] = phase_mapping if phase_mapping is not None else {}
    phase_rank_pairs = tuple(
        filter(
            lambda item: item[1] is not None,
            map(
                lambda phase_name: (
                    phase_name,
                    _report_projection_phase_rank_optional(phase_name),
                ),
                phase_names,
            ),
        )
    )
    if not phase_rank_pairs:
        return None
    return max(phase_rank_pairs, key=lambda item: item[1])[0]

def _parse_lint_line(line: str) -> LintEntryDTO | None:
    return parse_lint_line(line)

def _parse_lint_line_as_payload(line: str) -> dict[str, object] | None:
    entry = _parse_lint_line(line)
    return entry.model_dump() if entry is not None else None


def _rewrite_plan_schema_error_optional(
    entry: Mapping[str, JSONValue],
) -> dict[str, object] | None:
    issues = validate_rewrite_plan_payload(entry)
    if not issues:
        return None
    return {
        "plan_id": str(entry.get("plan_id", "")),
        "issues": issues,
    }


def _normalize_dataflow_response(
    response: Mapping[str, JSONValue],
) -> DataflowResponseEnvelopeDTO:
    lint_decision = LintEntriesDecision.from_response(response)
    lint_lines = list(lint_decision.lint_lines)
    lint_entries = lint_decision.normalize_entries(
        parse_lint_entry_fn=_parse_lint_line_as_payload,
    )
    aspf_trace_raw = response.get("aspf_trace")
    aspf_equivalence_raw = response.get("aspf_equivalence")
    aspf_opportunities_raw = response.get("aspf_opportunities")
    aspf_delta_ledger_raw = response.get("aspf_delta_ledger")
    aspf_state_raw = response.get("aspf_state")
    supported_analysis_surfaces_raw = response.get("supported_analysis_surfaces")
    disabled_surface_reasons_raw = response.get("disabled_surface_reasons")
    supported_surfaces_entries = _non_string_sequence_optional(
        supported_analysis_surfaces_raw
    )
    supported_analysis_surfaces = (
        sort_once(
            [str(item) for item in supported_surfaces_entries]
            if supported_surfaces_entries is not None
            else [],
            source="server._normalize_dataflow_response.supported_analysis_surfaces",
        )
    )
    disabled_surface_reasons_mapping = _json_mapping_optional(
        disabled_surface_reasons_raw
    )
    disabled_surface_reasons = (
        {
            str(key): str(disabled_surface_reasons_mapping[key])
            for key in disabled_surface_reasons_mapping
        }
        if disabled_surface_reasons_mapping is not None
        else {}
    )
    error_entries = _non_string_sequence_optional(response.get("errors"))
    canonical = DataflowCanonicalResponseDTO(
        exit_code=int(response.get("exit_code", 0) or 0),
        timeout=bool(response.get("timeout", False)),
        analysis_state=(str(response.get("analysis_state")) if response.get("analysis_state") is not None else None),
        errors=[str(err) for err in error_entries] if error_entries is not None else [],
        lint_lines=lint_lines,
        lint_entries=[LintEntryDTO.model_validate(entry) for entry in lint_entries],
        selected_adapter=(
            str(response.get("selected_adapter"))
            if response.get("selected_adapter") is not None
            else None
        ),
        supported_analysis_surfaces=supported_analysis_surfaces,
        disabled_surface_reasons=disabled_surface_reasons,
        aspf_trace=_json_mapping_optional(aspf_trace_raw),
        aspf_equivalence=(
            _json_mapping_optional(aspf_equivalence_raw)
        ),
        aspf_opportunities=(
            _json_mapping_optional(aspf_opportunities_raw)
        ),
        aspf_delta_ledger=(
            _json_mapping_optional(aspf_delta_ledger_raw)
        ),
        aspf_state=_json_mapping_optional(aspf_state_raw),
    )
    normalized = {str(key): response[key] for key in response}
    rewrite_plans = normalized.get("fingerprint_rewrite_plans")
    rewrite_plan_entries = _non_string_sequence_optional(rewrite_plans)
    if rewrite_plan_entries is not None:
        rewrite_plan_mappings = tuple(
            filter(
                _is_not_none_mapping_entry,
                map(_json_mapping_optional, rewrite_plan_entries),
            )
        )
        ordered_plans = normalize_rewrite_plan_order(
            list(rewrite_plan_mappings)
        )
        normalized["fingerprint_rewrite_plans"] = ordered_plans
        rewrite_plan_schema_errors = tuple(
            filter(
                _is_not_none_mapping_entry,
                map(_rewrite_plan_schema_error_optional, ordered_plans),
            )
        )
        if rewrite_plan_schema_errors:
            normalized["rewrite_plan_schema_errors"] = list(rewrite_plan_schema_errors)

    normalized["exit_code"] = canonical.exit_code
    normalized["timeout"] = canonical.timeout
    normalized["analysis_state"] = canonical.analysis_state
    normalized["errors"] = canonical.errors
    normalized["lint_lines"] = canonical.lint_lines
    normalized["selected_adapter"] = canonical.selected_adapter
    normalized["supported_analysis_surfaces"] = list(
        canonical.supported_analysis_surfaces
    )
    normalized["disabled_surface_reasons"] = dict(canonical.disabled_surface_reasons)
    normalized["lint_entries"] = [entry.model_dump() for entry in canonical.lint_entries]
    payload = normalized.get("payload")
    payload_mapping = _json_mapping_optional(payload)
    if payload_mapping is not None:
        payload_updates: dict[str, object] = {
            "selected_adapter": canonical.selected_adapter,
            "supported_analysis_surfaces": list(canonical.supported_analysis_surfaces),
            "disabled_surface_reasons": dict(canonical.disabled_surface_reasons),
        }
        normalized["payload"] = boundary_order.apply_boundary_updates_once(
            {str(key): payload_mapping[key] for key in payload_mapping},
            payload_updates,
            source="server._normalize_dataflow_response.payload_capabilities",
        )
    return DataflowResponseEnvelopeDTO(canonical=canonical, payload=normalized)


def _serialize_dataflow_response(
    response: DataflowResponseEnvelopeDTO,
) -> dict[str, object]:
    normalized = dict(response.payload)
    canonical = response.canonical
    normalized["exit_code"] = canonical.exit_code
    normalized["timeout"] = canonical.timeout
    normalized["analysis_state"] = canonical.analysis_state
    normalized["errors"] = list(canonical.errors)
    normalized["lint_lines"] = list(canonical.lint_lines)
    normalized["selected_adapter"] = canonical.selected_adapter
    normalized["supported_analysis_surfaces"] = list(
        canonical.supported_analysis_surfaces
    )
    normalized["disabled_surface_reasons"] = dict(canonical.disabled_surface_reasons)
    normalized["lint_entries"] = [entry.model_dump() for entry in canonical.lint_entries]
    if canonical.aspf_trace is not None:
        normalized["aspf_trace"] = canonical.aspf_trace.model_dump()
    if canonical.aspf_equivalence is not None:
        normalized["aspf_equivalence"] = canonical.aspf_equivalence.model_dump()
    if canonical.aspf_opportunities is not None:
        normalized["aspf_opportunities"] = canonical.aspf_opportunities.model_dump()
    if canonical.aspf_delta_ledger is not None:
        normalized["aspf_delta_ledger"] = canonical.aspf_delta_ledger.model_dump()
    if canonical.aspf_state is not None:
        normalized["aspf_state"] = canonical.aspf_state.model_dump()
    return boundary_order.canonicalize_boundary_mapping(
        normalized,
        source="server._normalize_dataflow_response",
    )

def _truthy_flag(value: object) -> bool:
    bool_value = _bool_optional(value)
    int_value = _int_optional(value)
    float_value = _float_optional(value)
    text = str(value).strip().lower()
    return (
        bool_value
        if bool_value is not None
        else (
            int_value != 0
            if int_value is not None
            else (
                float_value != 0
                if float_value is not None
                else text in {"1", "true", "yes", "on"}
            )
        )
    )

def _server_deadline_overhead_ns(
    total_ns: int,
    *,
    divisor: int | None = None,
) -> int:
    if total_ns <= 0:
        return 0
    divisor_value = (
        _SERVER_DEADLINE_OVERHEAD_DIVISOR
        if divisor is None
        else int(divisor)
    )
    if divisor_value <= 0:
        never("invalid server deadline overhead divisor", divisor=divisor_value)
    overhead = total_ns // divisor_value
    if overhead < _SERVER_DEADLINE_OVERHEAD_MIN_NS:
        overhead = _SERVER_DEADLINE_OVERHEAD_MIN_NS
    if overhead > _SERVER_DEADLINE_OVERHEAD_MAX_NS:
        overhead = _SERVER_DEADLINE_OVERHEAD_MAX_NS
    if overhead >= total_ns:
        overhead = max(0, total_ns - 1)
    return overhead

def _analysis_timeout_total_ns(payload: dict[str, JSONValue]) -> int:
    return payload_codec.analysis_timeout_total_ns(
        payload,
        source="server._analysis_timeout_total_ns.payload_keys",
        reject_sub_millisecond_seconds=True,
    )

def _analysis_timeout_total_ticks(payload: dict[str, JSONValue]) -> int:
    return payload_codec.analysis_timeout_total_ticks(
        payload,
        source="server._analysis_timeout_total_ticks.payload_keys",
    )

def _analysis_timeout_grace_ns(
    payload: dict[str, JSONValue],
    *,
    total_ns: int,
) -> int:
    if total_ns <= 1:
        return 0
    grace_cap_ns = max(
        1,
        (total_ns * _ANALYSIS_TIMEOUT_GRACE_RATIO_NUMERATOR)
        // _ANALYSIS_TIMEOUT_GRACE_RATIO_DENOMINATOR,
    )
    provided_grace_ns: int | None = None
    grace_ticks = payload.get("analysis_timeout_grace_ticks")
    grace_tick_ns = payload.get("analysis_timeout_grace_tick_ns")
    grace_ms = payload.get("analysis_timeout_grace_ms")
    grace_seconds = payload.get("analysis_timeout_grace_seconds")
    if grace_ticks not in (None, ""):
        try:
            grace_ticks_value = int(grace_ticks)
        except (TypeError, ValueError):
            never("invalid analysis timeout grace ticks", ticks=grace_ticks)
        if grace_ticks_value <= 0:
            never("invalid analysis timeout grace ticks", ticks=grace_ticks)
        if grace_tick_ns in (None, ""):
            never(
                "missing analysis timeout grace tick_ns",
                analysis_timeout_grace_ticks=grace_ticks_value,
            )
        try:
            grace_tick_ns_value = int(grace_tick_ns)
        except (TypeError, ValueError):
            never("invalid analysis timeout grace tick_ns", tick_ns=grace_tick_ns)
        if grace_tick_ns_value <= 0:
            never("invalid analysis timeout grace tick_ns", tick_ns=grace_tick_ns)
        provided_grace_ns = grace_ticks_value * grace_tick_ns_value
    elif grace_ms not in (None, ""):
        try:
            grace_ms_value = int(grace_ms)
        except (TypeError, ValueError):
            never("invalid analysis timeout grace ms", ms=grace_ms)
        if grace_ms_value <= 0:
            never("invalid analysis timeout grace ms", ms=grace_ms)
        provided_grace_ns = grace_ms_value * 1_000_000
    elif grace_seconds not in (None, ""):
        try:
            grace_seconds_value = Decimal(str(grace_seconds))
        except (InvalidOperation, ValueError):
            never("invalid analysis timeout grace seconds", seconds=grace_seconds)
        if grace_seconds_value <= 0:
            never("invalid analysis timeout grace seconds", seconds=grace_seconds)
        provided_grace_ns = int(grace_seconds_value * Decimal(1_000_000_000))
    if provided_grace_ns is None:
        return min(total_ns - 1, grace_cap_ns)
    return max(1, min(total_ns - 1, grace_cap_ns, provided_grace_ns))

def _analysis_timeout_budget_ns(
    payload: dict[str, JSONValue],
) -> tuple[int, int, int]:
    total_ns = _analysis_timeout_total_ns(payload)
    cleanup_grace_ns = _analysis_timeout_grace_ns(payload, total_ns=total_ns)
    analysis_ns = max(1, total_ns - cleanup_grace_ns)
    cleanup_ns = max(0, total_ns - analysis_ns)
    return total_ns, analysis_ns, cleanup_ns

def _deadline_profile_sample_interval(
    payload: dict[str, JSONValue],
    *,
    default_interval: int = _DEFAULT_DEADLINE_PROFILE_SAMPLE_INTERVAL,
) -> int:
    raw_value = payload.get("deadline_profile_sample_interval")
    if raw_value in (None, ""):
        return max(1, int(default_interval))
    try:
        interval = int(raw_value)
    except (TypeError, ValueError):
        never(
            "invalid deadline profile sample interval",
            deadline_profile_sample_interval=raw_value,
        )
    if interval <= 0:
        never(
            "invalid deadline profile sample interval",
            deadline_profile_sample_interval=raw_value,
        )
    return interval

def _output_dirs(report_root: Path) -> tuple[Path, Path]:
    out_dir = report_root / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir = report_root / "artifacts" / "out"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    return out_dir, artifact_dir


def _csv_parts(value_text: str) -> tuple[str, ...]:
    return tuple(filter(bool, map(str.strip, value_text.split(","))))


def _csv_parts_for_item(*, item: object, strict: bool) -> tuple[str, ...]:
    check_deadline()
    item_text = _str_optional(item)
    if item_text is None:
        if strict:
            never("name set contains non-string entry", value_type=type(item).__name__)
        return ()
    return _csv_parts(item_text)


def _normalize_csv_or_iterable_names(value: object, *, strict: bool) -> list[str]:
    check_deadline()
    value_text = _str_optional(value)
    if value_text is not None:
        return list(_csv_parts(value_text))
    entries = _non_string_sequence_optional(value)
    if entries is None:
        if strict and value is not None:
            never("invalid name set payload", value_type=type(value).__name__)
    entries_stream = entries or ()
    return list(
        chain.from_iterable(
            map(
                lambda item: _csv_parts_for_item(item=item, strict=strict),
                entries_stream,
            )
        )
    )

def _normalize_transparent_decorators(value: object) -> set[str] | None:
    items = _normalize_csv_or_iterable_names(value, strict=False)
    if not items:
        return None
    return set(items)

def _normalize_name_set(value: object) -> set[str] | None:
    if value is None:
        return None
    return set(_normalize_csv_or_iterable_names(value, strict=True))

@dataclass(frozen=True)
class DataflowNameFilterBundle:
    exclude_dirs: set[str]
    ignore_params: set[str]
    decision_ignore_params: set[str]
    transparent_decorators: set[str] | None

    @classmethod
    def from_payload(
        cls,
        *,
        payload: Mapping[str, object],
        defaults: Mapping[str, object],
        decision_section: Mapping[str, object],
    ) -> "DataflowNameFilterBundle":
        exclude_dirs = (
            _normalize_name_set(payload.get("exclude"))
            or _normalize_name_set(defaults.get("exclude"))
            or set()
        )
        ignore_params = (
            _normalize_name_set(payload.get("ignore_params"))
            or _normalize_name_set(defaults.get("ignore_params"))
            or set()
        )

        decision_ignore_params = set(ignore_params)
        decision_ignore_params.update(decision_ignore_list(decision_section))

        transparent_payload = payload.get(
            "transparent_decorators",
            defaults.get("transparent_decorators"),
        )
        transparent_decorators = _normalize_transparent_decorators(transparent_payload)

        return cls(
            exclude_dirs=exclude_dirs,
            ignore_params=ignore_params,
            decision_ignore_params=decision_ignore_params,
            transparent_decorators=transparent_decorators,
        )

def _timeout_context_payload(exc: TimeoutExceeded) -> JSONObject:
    context_obj = getattr(exc, "context", None)
    if hasattr(context_obj, "as_payload"):
        payload = context_obj.as_payload()
        payload_mapping = _json_mapping_optional(payload)
        if payload_mapping is not None:
            return {str(key): payload_mapping[key] for key in payload_mapping}
    return {
        "summary": "Analysis timed out.",
        "progress": {"classification": "timed_out_no_progress"},
    }


def _key_value_to_str_key(item: tuple[object, JSONValue]) -> tuple[str, JSONValue]:
    key, value = item
    return str(key), value


def _key_int_optional_item(item: tuple[object, JSONValue]) -> tuple[str, int | None]:
    key, value = item
    return str(key), _int_optional(value)


def _has_optional_int_value(item: tuple[str, int | None]) -> bool:
    _key, value = item
    return value is not None


def _required_int_item(item: tuple[str, int | None]) -> tuple[str, int]:
    key, value = item
    if value is None:
        never("missing required int value")
    return key, value


def _materialize_execution_plan(payload: Mapping[str, JSONValue]) -> ExecutionPlan:
    request_value = payload.get("execution_plan_request")
    request_mapping = _json_mapping_optional(request_value)
    if request_mapping is not None:
        req_ops = _non_string_sequence_optional(
            request_mapping.get("requested_operations")
        )
        requested_operations = (
            [str(op) for op in req_ops] if req_ops is not None else [DATAFLOW_COMMAND]
        )
        inputs_value = _json_mapping_optional(request_mapping.get("inputs"))
        if inputs_value is not None:
            inputs = dict(map(_key_value_to_str_key, inputs_value.items()))
        else:
            inputs = dict(
                map(
                    _key_value_to_str_key,
                    filter(
                        lambda item: item[0] != "execution_plan_request",
                        payload.items(),
                    ),
                )
            )
        artifacts_value = _non_string_sequence_optional(
            request_mapping.get("derived_artifacts")
        )
        derived_artifacts = (
            [str(path) for path in artifacts_value]
            if artifacts_value is not None
            else ["artifacts/out/execution_plan.json"]
        )
        obligations_value = _json_mapping_optional(request_mapping.get("obligations"))
        preconditions: list[str] = []
        postconditions: list[str] = []
        if obligations_value is not None:
            pre_raw = _non_string_sequence_optional(obligations_value.get("preconditions"))
            post_raw = _non_string_sequence_optional(
                obligations_value.get("postconditions")
            )
            if pre_raw is not None:
                preconditions = [str(item) for item in pre_raw]
            if post_raw is not None:
                postconditions = [str(item) for item in post_raw]
        policy_value = _json_mapping_optional(request_mapping.get("policy_metadata"))
        policy_deadline: dict[str, int] = {}
        policy_baseline_mode = "none"
        policy_docflow_mode = "disabled"
        if policy_value is not None:
            deadline_value = policy_value.get("deadline")
            deadline_mapping = _json_mapping_optional(deadline_value)
            if deadline_mapping is not None:
                policy_deadline = dict(
                    map(
                        _required_int_item,
                        filter(
                            _has_optional_int_value,
                            map(_key_int_optional_item, deadline_mapping.items()),
                        ),
                    )
                )
            baseline_mode = policy_value.get("baseline_mode")
            baseline_mode_text = _str_optional(baseline_mode)
            if baseline_mode_text is not None:
                policy_baseline_mode = baseline_mode_text
            docflow_mode = policy_value.get("docflow_mode")
            docflow_mode_text = _str_optional(docflow_mode)
            if docflow_mode_text is not None:
                policy_docflow_mode = docflow_mode_text
        return ExecutionPlan(
            requested_operations=requested_operations,
            inputs=inputs,
            derived_artifacts=derived_artifacts,
            obligations=ExecutionPlanObligations(
                preconditions=preconditions,
                postconditions=postconditions,
            ),
            policy_metadata=ExecutionPlanPolicyMetadata(
                deadline=policy_deadline,
                baseline_mode=policy_baseline_mode,
                docflow_mode=policy_docflow_mode,
            ),
        )
    inputs = dict(map(_key_value_to_str_key, payload.items()))
    return ExecutionPlan(
        requested_operations=[DATAFLOW_COMMAND],
        inputs=inputs,
        derived_artifacts=["artifacts/out/execution_plan.json"],
        obligations=ExecutionPlanObligations(
            preconditions=["payload accepted by server"],
            postconditions=["command response emitted"],
        ),
        policy_metadata=ExecutionPlanPolicyMetadata(deadline={}, baseline_mode="none", docflow_mode="disabled"),
    )

def _default_execute_command_deps() -> ExecuteCommandDeps:
    return ExecuteCommandDeps(
        analysis=AnalysisDeps(
            analyze_paths_fn=analyze_paths,
            load_aspf_resume_state_fn=_load_aspf_resume_state,
            analysis_input_manifest_fn=_analysis_input_manifest,
            analysis_input_manifest_digest_fn=_analysis_input_manifest_digest,
            build_analysis_collection_resume_seed_fn=build_analysis_collection_resume_seed,
            collection_semantic_progress_fn=_collection_semantic_progress,
            project_report_sections_fn=project_report_sections,
            report_projection_spec_rows_fn=report_projection_spec_rows,
        ),
        output=OutputDeps(
            collection_checkpoint_flush_due_fn=_collection_checkpoint_flush_due,
            write_bootstrap_incremental_artifacts_fn=_write_bootstrap_incremental_artifacts,
            load_report_section_journal_fn=_load_report_section_journal,
        ),
        progress=ProgressDeps(
            start_trace_fn=aspf_execution_fibration.start_execution_trace,
            record_1cell_fn=aspf_execution_fibration.record_1cell,
            record_2cell_witness_fn=aspf_execution_fibration.record_2cell_witness,
            record_cofibration_fn=aspf_execution_fibration.record_cofibration,
            merge_imported_trace_fn=aspf_execution_fibration.merge_imported_trace,
            finalize_trace_fn=aspf_execution_fibration.finalize_execution_trace,
        ),
        runtime=RuntimeDeps(
            monotonic_ns_fn=time.monotonic_ns,
            heartbeat_wait_fn=lambda stop_event, timeout_seconds: stop_event.wait(
                timeout_seconds
            ),
        ),
    )

__all__ = [
    'AnalysisResult',
    'AuditConfig',
    'Callable',
    'DataflowNameFilterBundle',
    'Deadline',
    'DecisionSnapshotSurfaces',
    'ExecutionPlan',
    'Fingerprint',
    'Forest',
    'GasMeter',
    'JSONObject',
    'JSONValue',
    'Path',
    'PrimeRegistry',
    'ReportCarrier',
    'TimeoutExceeded',
    'TypeConstructorRegistry',
    '_LSP_PROGRESS_NOTIFICATION_METHOD',
    '_LSP_PROGRESS_TOKEN_V2',
    '_CANONICAL_PROGRESS_EVENT_SCHEMA_V2',
    '_PROGRESS_DEADLINE_FLUSH_MARGIN_SECONDS',
    '_PROGRESS_DEADLINE_FLUSH_SECONDS',
    '_PROGRESS_DEADLINE_WATCHDOG_SECONDS',
    '_PROGRESS_HEARTBEAT_POLL_SECONDS',
    '_analysis_index_resume_hydrated_count',
    '_analysis_index_resume_signature',
    '_analysis_resume_cache_verdict',
    '_analysis_resume_progress',
    '_analysis_timeout_budget_ns',
    '_analysis_timeout_total_ticks',
    '_append_phase_timeline_event',
    '_apply_journal_pending_reason',
    '_build_phase_progress_v2',
    '_collection_components_preview_lines',
    '_collection_progress_intro_lines',
    '_collection_report_flush_due',
    '_deadline_profile_sample_interval',
    '_default_execute_command_deps',
    '_groups_by_path_from_collection_resume',
    '_incremental_progress_obligations',
    '_is_stdout_target',
    '_latest_report_phase',
    '_materialize_execution_plan',
    '_normalize_dataflow_response',
    '_output_dirs',
    '_phase_timeline_header_block',
    '_phase_timeline_jsonl_path',
    '_phase_timeline_md_path',
    '_progress_heartbeat_seconds',
    '_projection_phase_flush_due',
    '_render_incremental_report',
    '_report_witness_digest',
    '_resolve_report_output_path',
    '_resolve_report_section_journal_path',
    '_split_incremental_obligations',
    '_timeout_context_payload',
    '_truthy_flag',
    '_write_report_section_journal',
    '_write_text_profiled',
    'ambiguity_delta',
    'ambiguity_state',
    'apply_baseline',
    'boundary_order',
    'build_fingerprint_registry',
    'build_refactor_plan',
    'build_synthesis_plan',
    'call_cluster_consolidation',
    'call_clusters',
    'check_deadline',
    'compute_structure_metrics',
    'compute_violations',
    'dataflow_deadline_roots',
    'dataflow_defaults',
    'datetime',
    'decision_defaults',
    'decision_require_tiers',
    'decision_tier_map',
    'exception_defaults',
    'exception_never_list',
    'extract_report_sections',
    'fingerprint_defaults',
    'load_baseline',
    'merge_payload',
    'never',
    'render_decision_snapshot',
    'render_dot',
    'render_protocol_stubs',
    'render_refactor_plan',
    'render_report',
    'render_structure_snapshot',
    'render_synthesis_section',
    'report_projection_phase_rank',
    'reset_deadline',
    'reset_deadline_clock',
    'reset_deadline_profile',
    'reset_forest',
    'resolve_baseline_path',
    'semantic_coverage_map',
    'set_deadline',
    'set_deadline_clock',
    'set_deadline_profile',
    'set_forest',
    'sort_once',
    'taint_boundary_registry',
    'taint_defaults',
    'taint_delta',
    'taint_lifecycle',
    'taint_profile',
    'taint_state',
    'test_annotation_drift',
    'test_annotation_drift_delta',
    'test_evidence_suggestions',
    'test_obsolescence',
    'test_obsolescence_delta',
    'test_obsolescence_state',
    'threading',
    'time',
    'timezone',
    'write_baseline',
    'write_execution_plan_artifact',
]

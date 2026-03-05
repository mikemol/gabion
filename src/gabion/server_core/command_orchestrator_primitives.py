# gabion:boundary_normalization_module
from __future__ import annotations

"""Static primitives extracted from server for command_orchestrator DAG decoupling."""

import hashlib
import json
import threading
import time
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
from gabion.schema import (LegacyDataflowMonolithResponseDTO, LintEntryDTO)
from gabion.server_core.ingress_primitives import AnalysisDeps, ExecuteCommandDeps, OutputDeps, ProgressDeps
from gabion.server_core import dataflow_runtime_contract as runtime_contract

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

_PHASE_PRIMARY_UNITS: Mapping[str, str] = runtime_contract.PHASE_PRIMARY_UNITS

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
    files: list[JSONObject] = []
    for path in file_paths:
        check_deadline()
        entry: JSONObject = {"path": str(path)}
        try:
            stat = path.stat()
        except OSError:
            entry["missing"] = True
        else:
            entry["size"] = int(stat.st_size)
        files.append(entry)
    return {
        "format_version": _ANALYSIS_INPUT_MANIFEST_FORMAT_VERSION,
        "root": str(root),
        "recursive": recursive,
        "include_invariant_propositions": include_invariant_propositions,
        "include_wl_refinement": include_wl_refinement,
        "config": _analysis_witness_config_payload(config),
        "files": files,
    }

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
    for path in import_state_paths:
        raw_payload = cast(
            Mapping[str, object],
            json.loads(path.read_text(encoding="utf-8")),
        )
        manifest_digest = str(raw_payload.get("analysis_manifest_digest", "") or "")
        latest_manifest_digest = manifest_digest or latest_manifest_digest
        resume_source = str(raw_payload.get("resume_source", "") or "")
        latest_resume_source = resume_source or latest_resume_source
    projection, mutation_count, mutation_tail = aspf_resume_state.fold_resume_mutations(
        snapshot={},
        mutations=aspf_resume_state.iter_resume_mutations(
            state_paths=import_state_paths,
        ),
        tail_limit=diagnostic_tail_limit,
    )
    payload: JSONObject = {
        "resume_projection": projection,
        "delta_record_count": mutation_count,
        "delta_records_tail": list(mutation_tail),
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

def _analysis_resume_progress(
    *,
    collection_resume: Mapping[str, JSONValue] | None,
    total_files: int,
) -> dict[str, int]:
    if not isinstance(collection_resume, Mapping):
        normalized_total_files = max(total_files, 0)
        return {
            "completed_files": 0,
            "in_progress_files": 0,
            "remaining_files": normalized_total_files,
            "total_files": normalized_total_files,
        }
    completed_paths = collection_resume.get("completed_paths")
    completed = 0
    if isinstance(completed_paths, list):
        completed = sum(1 for path in completed_paths if isinstance(path, str))
    in_progress_scan = collection_resume.get("in_progress_scan_by_path")
    in_progress = 0
    if isinstance(in_progress_scan, Mapping):
        in_progress = sum(
            1
            for path, state in in_progress_scan.items()
            if isinstance(path, str) and isinstance(state, Mapping)
        )
    if total_files >= 0:
        # A timeout can occur before path discovery populates total_files.
        # Preserve observed checkpoint progress instead of clamping it away.
        observed_files = completed + in_progress
        if observed_files > total_files:
            total_files = observed_files
        completed = min(completed, total_files)
        in_progress = min(in_progress, max(total_files - completed, 0))
    remaining = max(total_files - completed, 0)
    return {
        "completed_files": completed,
        "in_progress_files": in_progress,
        "remaining_files": remaining,
        "total_files": total_files,
    }

def _normalize_progress_work(
    *,
    work_done: object | None,
    work_total: object | None,
) -> tuple[int | None, int | None]:
    normalized_done: int | None = None
    normalized_total: int | None = None
    if isinstance(work_done, int) and not isinstance(work_done, bool):
        normalized_done = max(work_done, 0)
    if isinstance(work_total, int) and not isinstance(work_total, bool):
        normalized_total = max(work_total, 0)
    if (
        normalized_done is not None
        and normalized_total is not None
        and normalized_done > normalized_total
    ):
        normalized_done = normalized_total
    return normalized_done, normalized_total

def _phase_primary_unit_for_phase(phase: str) -> str:
    primary_unit = _PHASE_PRIMARY_UNITS.get(phase)
    if primary_unit is not None:
        return primary_unit
    never("unknown phase for primary-unit lookup", phase=phase)

def _build_phase_progress_v2(
    *,
    phase: str,
    collection_progress: Mapping[str, JSONValue],
    semantic_progress: Mapping[str, JSONValue] | None = None,
    work_done: object | None,
    work_total: object | None,
    phase_progress_v2: Mapping[str, JSONValue] | None = None,
) -> tuple[JSONObject, int, int]:
    normalized_work_done, normalized_work_total = _normalize_progress_work(
        work_done=work_done,
        work_total=work_total,
    )
    if normalized_work_done is None or normalized_work_total is None:
        if phase == "collection":
            raw_completed = collection_progress.get("completed_files")
            raw_total = collection_progress.get("total_files")
            if (
                isinstance(raw_completed, int)
                and not isinstance(raw_completed, bool)
                and isinstance(raw_total, int)
                and not isinstance(raw_total, bool)
            ):
                normalized_work_done = max(int(raw_completed), 0)
                normalized_work_total = max(int(raw_total), 0)
    if normalized_work_done is None:
        normalized_work_done = 0
    if normalized_work_total is None:
        normalized_work_total = 0
    if normalized_work_total:
        normalized_work_done = min(normalized_work_done, normalized_work_total)

    primary_unit_for_phase = _phase_primary_unit_for_phase(phase)
    normalized: JSONObject = {
        "format_version": 1,
        "schema": "gabion/phase_progress_v2",
        "primary_unit": primary_unit_for_phase,
        "primary_done": normalized_work_done,
        "primary_total": normalized_work_total,
        "dimensions": {
            primary_unit_for_phase: {
                "done": normalized_work_done,
                "total": normalized_work_total,
            }
        },
        "inventory": {},
    }
    if isinstance(phase_progress_v2, Mapping):
        for key, value in phase_progress_v2.items():
            if isinstance(key, str):
                normalized[key] = value
    primary_unit = str(normalized.get("primary_unit", "") or "").strip()
    if not primary_unit:
        primary_unit = primary_unit_for_phase
    raw_primary_done = normalized.get("primary_done")
    raw_primary_total = normalized.get("primary_total")
    primary_done = (
        max(int(raw_primary_done), 0)
        if isinstance(raw_primary_done, int) and not isinstance(raw_primary_done, bool)
        else normalized_work_done
    )
    primary_total = (
        max(int(raw_primary_total), 0)
        if isinstance(raw_primary_total, int) and not isinstance(raw_primary_total, bool)
        else normalized_work_total
    )
    if primary_total:
        primary_done = min(primary_done, primary_total)
    normalized["primary_unit"] = primary_unit
    normalized["primary_done"] = primary_done
    normalized["primary_total"] = primary_total
    raw_dimensions = normalized.get("dimensions")
    dimensions: JSONObject = {}
    if isinstance(raw_dimensions, Mapping):
        for dim_name, dim_payload in raw_dimensions.items():
            if not isinstance(dim_name, str) or not isinstance(dim_payload, Mapping):
                continue
            raw_done = dim_payload.get("done")
            raw_total = dim_payload.get("total")
            if (
                isinstance(raw_done, int)
                and not isinstance(raw_done, bool)
                and isinstance(raw_total, int)
                and not isinstance(raw_total, bool)
            ):
                dim_done = max(int(raw_done), 0)
                dim_total = max(int(raw_total), 0)
                if dim_total:
                    dim_done = min(dim_done, dim_total)
                dimensions[dim_name] = {"done": dim_done, "total": dim_total}
    if primary_unit not in dimensions:
        dimensions[primary_unit] = {"done": primary_done, "total": primary_total}
    if phase == "collection" and isinstance(semantic_progress, Mapping):
        raw_cumulative_new = semantic_progress.get("cumulative_new_processed_functions")
        raw_cumulative_completed = semantic_progress.get("cumulative_completed_files_delta")
        raw_cumulative_hydrated = semantic_progress.get("cumulative_hydrated_paths_delta")
        raw_cumulative_regressed = semantic_progress.get("cumulative_regressed_functions")
        semantic_new = (
            max(int(raw_cumulative_new), 0)
            if isinstance(raw_cumulative_new, int) and not isinstance(raw_cumulative_new, bool)
            else 0
        )
        semantic_completed = (
            max(int(raw_cumulative_completed), 0)
            if isinstance(raw_cumulative_completed, int)
            and not isinstance(raw_cumulative_completed, bool)
            else 0
        )
        semantic_hydrated = (
            max(int(raw_cumulative_hydrated), 0)
            if isinstance(raw_cumulative_hydrated, int)
            and not isinstance(raw_cumulative_hydrated, bool)
            else 0
        )
        semantic_regressed = (
            max(int(raw_cumulative_regressed), 0)
            if isinstance(raw_cumulative_regressed, int)
            and not isinstance(raw_cumulative_regressed, bool)
            else 0
        )
        if semantic_hydrated > 0 or semantic_regressed > 0:
            dimensions["hydrated_paths_delta"] = {
                "done": semantic_hydrated,
                "total": semantic_hydrated + semantic_regressed,
            }
        semantic_done = semantic_new + semantic_completed + semantic_hydrated
        semantic_total = semantic_done + semantic_regressed
        if semantic_done > 0 or semantic_regressed > 0:
            dimensions["semantic_progress_points"] = {
                "done": semantic_done,
                "total": semantic_total,
            }
    normalized["dimensions"] = dimensions
    raw_inventory = normalized.get("inventory")
    inventory: JSONObject = {}
    if isinstance(raw_inventory, Mapping):
        for inv_key, inv_value in raw_inventory.items():
            if isinstance(inv_key, str):
                inventory[inv_key] = inv_value
    normalized["inventory"] = inventory
    return normalized, primary_done, primary_total

def _completed_path_set(
    collection_resume: Mapping[str, JSONValue] | None,
) -> set[str]:
    if not isinstance(collection_resume, Mapping):
        return set()
    raw_completed_paths = collection_resume.get("completed_paths")
    if not isinstance(raw_completed_paths, Sequence) or isinstance(
        raw_completed_paths, (str, bytes)
    ):
        return set()
    return {path for path in raw_completed_paths if isinstance(path, str)}

def _in_progress_scan_states(
    collection_resume: Mapping[str, JSONValue] | None,
) -> dict[str, Mapping[str, JSONValue]]:
    states: dict[str, Mapping[str, JSONValue]] = {}
    if not isinstance(collection_resume, Mapping):
        return states
    raw_in_progress = collection_resume.get("in_progress_scan_by_path")
    if not isinstance(raw_in_progress, Mapping):
        return states
    previous_path: str | None = None
    for raw_path, raw_state in raw_in_progress.items():
        check_deadline()
        if not isinstance(raw_path, str):
            continue
        if previous_path is not None and previous_path > raw_path:
            never(
                "in_progress_scan_by_path path order regression",
                previous_path=previous_path,
                current_path=raw_path,
            )
        previous_path = raw_path
        if not isinstance(raw_state, Mapping):
            continue
        states[raw_path] = cast(Mapping[str, JSONValue], raw_state)
    return states

def _state_processed_functions(state: Mapping[str, JSONValue]) -> set[str]:
    raw_processed = state.get("processed_functions")
    if not isinstance(raw_processed, Sequence) or isinstance(raw_processed, (str, bytes)):
        return set()
    return {entry for entry in raw_processed if isinstance(entry, str)}

def _state_processed_count(state: Mapping[str, JSONValue]) -> int:
    processed_functions = _state_processed_functions(state)
    if processed_functions:
        return len(processed_functions)
    raw_count = state.get("processed_functions_count")
    if isinstance(raw_count, int):
        return max(0, raw_count)
    return 0

def _state_processed_digest(state: Mapping[str, JSONValue]) -> str:
    processed_functions = _state_processed_functions(state)
    if processed_functions:
        return hashlib.sha1(
            _canonical_json_text(sort_once(processed_functions, source = 'src/gabion/server.py:1371')).encode("utf-8")
        ).hexdigest()
    raw_digest = state.get("processed_functions_digest")
    if isinstance(raw_digest, str) and raw_digest:
        return raw_digest
    return hashlib.sha1(
        _canonical_json_text({"count": _state_processed_count(state)}).encode("utf-8")
    ).hexdigest()

def _analysis_index_resume_hydrated_paths(
    collection_resume: Mapping[str, JSONValue] | None,
) -> set[str]:
    if not isinstance(collection_resume, Mapping):
        return set()
    raw_resume = collection_resume.get("analysis_index_resume")
    if not isinstance(raw_resume, Mapping):
        return set()
    raw_hydrated = raw_resume.get("hydrated_paths")
    if not isinstance(raw_hydrated, Sequence) or isinstance(raw_hydrated, (str, bytes)):
        return set()
    return {entry for entry in raw_hydrated if isinstance(entry, str)}

def _analysis_index_resume_hydrated_count(
    collection_resume: Mapping[str, JSONValue] | None,
) -> int:
    hydrated = _analysis_index_resume_hydrated_paths(collection_resume)
    if hydrated:
        return len(hydrated)
    if not isinstance(collection_resume, Mapping):
        return 0
    raw_resume = collection_resume.get("analysis_index_resume")
    if not isinstance(raw_resume, Mapping):
        return 0
    raw_count = raw_resume.get("hydrated_paths_count")
    if isinstance(raw_count, int):
        return max(0, raw_count)
    return 0

def _analysis_index_resume_hydrated_digest(
    collection_resume: Mapping[str, JSONValue] | None,
) -> str:
    hydrated = _analysis_index_resume_hydrated_paths(collection_resume)
    if hydrated:
        return hashlib.sha1(
            _canonical_json_text(sort_once(hydrated, source = 'src/gabion/server.py:1418')).encode("utf-8")
        ).hexdigest()
    if not isinstance(collection_resume, Mapping):
        return hashlib.sha1(b"[]").hexdigest()
    raw_resume = collection_resume.get("analysis_index_resume")
    if not isinstance(raw_resume, Mapping):
        return hashlib.sha1(b"[]").hexdigest()
    raw_digest = raw_resume.get("hydrated_paths_digest")
    if isinstance(raw_digest, str) and raw_digest:
        return raw_digest
    return hashlib.sha1(
        _canonical_json_text({"count": _analysis_index_resume_hydrated_count(collection_resume)}).encode("utf-8")
    ).hexdigest()

def _analysis_index_resume_signature(
    collection_resume: Mapping[str, JSONValue] | None,
) -> tuple[int, str, int, int, str, str]:
    hydrated_count = _analysis_index_resume_hydrated_count(collection_resume)
    hydrated_digest = _analysis_index_resume_hydrated_digest(collection_resume)
    if not isinstance(collection_resume, Mapping):
        return (hydrated_count, hydrated_digest, 0, 0, "", hydrated_digest)
    raw_resume = collection_resume.get("analysis_index_resume")
    if not isinstance(raw_resume, Mapping):
        return (hydrated_count, hydrated_digest, 0, 0, "", hydrated_digest)
    function_count = raw_resume.get("function_count")
    class_count = raw_resume.get("class_count")
    phase = raw_resume.get("phase")
    resume_digest = raw_resume.get("resume_digest")
    if not isinstance(function_count, int):
        function_count = 0
    if not isinstance(class_count, int):
        class_count = 0
    if not isinstance(phase, str):
        phase = ""
    if not isinstance(resume_digest, str) or not resume_digest:
        resume_digest = hydrated_digest
    return (
        hydrated_count,
        hydrated_digest,
        function_count,
        class_count,
        phase,
        resume_digest,
    )

def _collection_semantic_witness(
    *,
    collection_resume: Mapping[str, JSONValue] | None,
) -> JSONObject:
    states = _in_progress_scan_states(collection_resume)
    state_rows: list[JSONObject] = []
    processed_total = 0
    for path_key, state in states.items():
        check_deadline()
        phase = state.get("phase")
        phase_text = phase if isinstance(phase, str) and phase else "unknown"
        processed_count = _state_processed_count(state)
        processed_total += processed_count
        state_rows.append(
            {
                "path": path_key,
                "phase": phase_text,
                "processed_functions_count": processed_count,
                "processed_functions_digest": _state_processed_digest(state),
            }
        )
    index_signature = _analysis_index_resume_signature(collection_resume)
    digest = hashlib.sha1(
        _canonical_json_text(
            {
                "in_progress": state_rows,
                "index_hydrated_paths_count": _analysis_index_resume_hydrated_count(
                    collection_resume
                ),
                "index_hydrated_paths_digest": _analysis_index_resume_hydrated_digest(
                    collection_resume
                ),
                "index_resume_digest": index_signature[5],
                "index_function_count": index_signature[2],
                "index_class_count": index_signature[3],
            }
        ).encode("utf-8")
    ).hexdigest()
    return {
        "witness_digest": digest,
        "in_progress_paths": len(state_rows),
        "processed_functions_total": processed_total,
        "index_hydrated_paths_count": _analysis_index_resume_hydrated_count(
            collection_resume
        ),
        "index_hydrated_paths_digest": _analysis_index_resume_hydrated_digest(
            collection_resume
        ),
        "index_resume_digest": index_signature[5],
        "index_function_count": index_signature[2],
        "index_class_count": index_signature[3],
    }

def _collection_semantic_progress(
    *,
    previous_collection_resume: Mapping[str, JSONValue] | None,
    collection_resume: Mapping[str, JSONValue],
    total_files: int,
    cumulative: Mapping[str, JSONValue] | None = None,
) -> JSONObject:
    previous_states = _in_progress_scan_states(previous_collection_resume)
    current_states = _in_progress_scan_states(collection_resume)
    current_completed_paths = _completed_path_set(collection_resume)
    prev_progress = _analysis_resume_progress(
        collection_resume=previous_collection_resume,
        total_files=total_files,
    )
    current_progress = _analysis_resume_progress(
        collection_resume=collection_resume,
        total_files=total_files,
    )
    added_processed = 0
    regressed_processed = 0
    unchanged_in_progress_paths = 0
    changed_in_progress_paths = 0
    seen_paths: set[str] = set()

    def _accumulate_progress(path_key: str) -> None:
        nonlocal added_processed
        nonlocal regressed_processed
        nonlocal unchanged_in_progress_paths
        nonlocal changed_in_progress_paths
        previous_state = previous_states.get(path_key)
        current_state = current_states.get(path_key)
        if (
            previous_state is not None
            and current_state is None
            and path_key in current_completed_paths
        ):
            # Moving a path from in-progress to completed is monotonic progress,
            # not a semantic regression in processed functions.
            changed_in_progress_paths += 1
            return
        previous_keys = (
            _state_processed_functions(previous_state) if previous_state is not None else set()
        )
        current_keys = (
            _state_processed_functions(current_state) if current_state is not None else set()
        )
        if previous_keys or current_keys:
            added = current_keys - previous_keys
            regressed = previous_keys - current_keys
            added_count = len(added)
            regressed_count = len(regressed)
        else:
            previous_count = (
                _state_processed_count(previous_state) if previous_state is not None else 0
            )
            current_count = (
                _state_processed_count(current_state) if current_state is not None else 0
            )
            added_count = max(0, current_count - previous_count)
            regressed_count = max(0, previous_count - current_count)
        added_processed += added_count
        regressed_processed += regressed_count
        if added_count == 0 and regressed_count == 0:
            unchanged_in_progress_paths += 1
        else:
            changed_in_progress_paths += 1

    for path_key in previous_states:
        check_deadline()
        seen_paths.add(path_key)
        _accumulate_progress(path_key)
    for path_key in current_states:
        check_deadline()
        if path_key in seen_paths:
            continue
        seen_paths.add(path_key)
        _accumulate_progress(path_key)
    completed_delta = max(
        0, current_progress["completed_files"] - prev_progress["completed_files"]
    )
    completed_regressed = max(
        0, prev_progress["completed_files"] - current_progress["completed_files"]
    )
    previous_hydrated_paths = _analysis_index_resume_hydrated_paths(
        previous_collection_resume
    )
    current_hydrated_paths = _analysis_index_resume_hydrated_paths(collection_resume)
    if previous_hydrated_paths or current_hydrated_paths:
        hydrated_delta = len(current_hydrated_paths - previous_hydrated_paths)
        hydrated_regressed = len(previous_hydrated_paths - current_hydrated_paths)
    else:
        previous_hydrated_count = _analysis_index_resume_hydrated_count(
            previous_collection_resume
        )
        current_hydrated_count = _analysis_index_resume_hydrated_count(collection_resume)
        hydrated_delta = max(0, current_hydrated_count - previous_hydrated_count)
        hydrated_regressed = max(0, previous_hydrated_count - current_hydrated_count)
    cumulative_new = added_processed
    cumulative_completed_delta = completed_delta
    cumulative_hydrated_delta = hydrated_delta
    cumulative_regressed = regressed_processed + completed_regressed + hydrated_regressed
    if isinstance(cumulative, Mapping):
        raw_cumulative_new = cumulative.get("cumulative_new_processed_functions")
        raw_cumulative_completed = cumulative.get("cumulative_completed_files_delta")
        raw_cumulative_hydrated = cumulative.get("cumulative_hydrated_paths_delta")
        raw_cumulative_regressed = cumulative.get("cumulative_regressed_functions")
        if isinstance(raw_cumulative_new, int):
            cumulative_new += max(0, raw_cumulative_new)
        if isinstance(raw_cumulative_completed, int):
            cumulative_completed_delta += max(0, raw_cumulative_completed)
        if isinstance(raw_cumulative_hydrated, int):
            cumulative_hydrated_delta += max(0, raw_cumulative_hydrated)
        if isinstance(raw_cumulative_regressed, int):
            cumulative_regressed += max(0, raw_cumulative_regressed)
    current_witness = _collection_semantic_witness(collection_resume=collection_resume)
    previous_witness = (
        _collection_semantic_witness(collection_resume=previous_collection_resume)
        if isinstance(previous_collection_resume, Mapping)
        else {"witness_digest": None}
    )
    substantive_progress = (
        (
            cumulative_new > 0
            or cumulative_completed_delta > 0
            or cumulative_hydrated_delta > 0
        )
        and cumulative_regressed == 0
    )
    return {
        "current_witness_digest": current_witness.get("witness_digest"),
        "previous_witness_digest": previous_witness.get("witness_digest"),
        "new_processed_functions_count": added_processed,
        "regressed_processed_functions_count": regressed_processed,
        "completed_files_delta": completed_delta,
        "completed_files_regressed": completed_regressed,
        "hydrated_paths_delta": hydrated_delta,
        "hydrated_paths_regressed": hydrated_regressed,
        "changed_in_progress_paths": changed_in_progress_paths,
        "unchanged_in_progress_paths": unchanged_in_progress_paths,
        "cumulative_new_processed_functions": cumulative_new,
        "cumulative_completed_files_delta": cumulative_completed_delta,
        "cumulative_hydrated_paths_delta": cumulative_hydrated_delta,
        "cumulative_regressed_functions": cumulative_regressed,
        "monotonic_progress": cumulative_regressed == 0,
        "substantive_progress": substantive_progress,
    }

def _resolve_report_output_path(*, root: Path, report_path: str | None) -> Path | None:
    if report_path in (None, ""):
        return None
    if _is_stdout_target(report_path):
        return None
    candidate = Path(str(report_path))
    if candidate.is_absolute():
        return candidate
    return root / candidate

def _resolve_report_section_journal_path(
    *,
    root: Path,
    report_path: str | None,
) -> Path | None:
    resolved_report = _resolve_report_output_path(root=root, report_path=report_path)
    if resolved_report is None:
        return None
    default_journal = root / _DEFAULT_REPORT_SECTION_JOURNAL
    if resolved_report.name == "dataflow_report.md":
        return default_journal
    return resolved_report.with_name(f"{resolved_report.stem}_sections.json")

def _report_witness_digest(
    *,
    input_witness: Mapping[str, JSONValue] | None,
    manifest_digest: str | None,
) -> str | None:
    digest = input_witness.get("witness_digest") if input_witness is not None else None
    digest_text = digest if isinstance(digest, str) and digest else None
    manifest_text = (
        manifest_digest
        if isinstance(manifest_digest, str) and manifest_digest
        else None
    )
    return digest_text or manifest_text

def _coerce_section_lines(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    lines: list[str] = []
    for item in value:
        check_deadline()
        if isinstance(item, str):
            lines.append(item)
    return lines

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
    if not isinstance(payload, dict):
        return {}, "policy"
    if payload.get("format_version") != _REPORT_SECTION_JOURNAL_FORMAT_VERSION:
        return {}, "policy"
    expected_digest = payload.get("witness_digest")
    if isinstance(expected_digest, str):
        if not isinstance(witness_digest, str) or expected_digest != witness_digest:
            return {}, "stale_input"
    sections_payload = payload.get("sections")
    if not isinstance(sections_payload, Mapping):
        return {}, "policy"
    sections: dict[str, list[str]] = {}
    for key, entry in sections_payload.items():
        check_deadline()
        if not isinstance(key, str) or not isinstance(entry, Mapping):
            continue
        lines = _coerce_section_lines(entry.get("lines"))
        if not lines:
            continue
        sections[key] = lines
    return sections, None

def _write_report_section_journal(
    *,
    path: Path | None,
    witness_digest: str | None,
    projection_rows: Sequence[Mapping[str, JSONValue]],
    sections: Mapping[str, list[str]],
    pending_reasons: Mapping[str, str] | None = None,
) -> None:
    if path is None:
        return
    rows_payload: list[JSONObject] = []
    sections_payload: JSONObject = {}
    pending_reasons = pending_reasons or {}
    for row in projection_rows:
        check_deadline()
        section_id = str(row.get("section_id", "") or "")
        if not section_id:
            continue
        phase = str(row.get("phase", "") or "")
        deps_raw = row.get("deps")
        deps: list[str] = []
        if isinstance(deps_raw, list):
            deps = [str(dep) for dep in deps_raw if isinstance(dep, str)]
        status = "resolved" if section_id in sections else "pending"
        section_entry: JSONObject = {
            "phase": phase,
            "deps": deps,
            "status": status,
            "lines": sections.get(section_id, []),
        }
        section_entry["reason"] = pending_reasons.get(section_id)
        sections_payload[section_id] = section_entry
        rows_payload.append(
            {
                "section_id": section_id,
                "phase": phase,
                "deps": deps,
                "status": status,
            }
        )
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
    if _deadline_tick_budget_allows_check(get_deadline_clock()):
        check_deadline(allow_frame_fallback=False)
    if report_output_path is None or not projection_rows:
        return
    existing_reason: str | None = None
    if report_section_journal_path is not None and report_section_journal_path.exists():
        try:
            existing_payload = json.loads(
                _read_text_profiled(
                    report_section_journal_path,
                    io_name="report_section_journal.read",
                )
            )
        except (OSError, UnicodeError, json.JSONDecodeError):
            existing_reason = "policy"
        else:
            if not isinstance(existing_payload, dict):
                existing_reason = "policy"
            elif (
                existing_payload.get("format_version")
                != _REPORT_SECTION_JOURNAL_FORMAT_VERSION
            ):
                existing_reason = "policy"
            else:
                expected_digest = existing_payload.get("witness_digest")
                if isinstance(expected_digest, str) and expected_digest:
                    if not isinstance(witness_digest, str) or expected_digest != witness_digest:
                        existing_reason = "stale_input"
    intro_lines = [
        "Collection bootstrap checkpoint (provisional).",
        f"- `root`: `{root}`",
        f"- `paths_requested`: `{paths_requested}`",
    ]
    sections: dict[str, list[str]] = {"intro": intro_lines}
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
    ]
    rows_payload: list[JSONObject] = []
    sections_payload: JSONObject = {
        "intro": {
            "phase": "collection",
            "deps": [],
            "status": "resolved",
            "lines": intro_lines,
        }
    }
    rows_payload.append(
        {
            "section_id": "intro",
            "phase": "collection",
            "deps": [],
            "status": "resolved",
        }
    )
    for row in projection_rows:
        if _deadline_tick_budget_allows_check(get_deadline_clock()):
            check_deadline(allow_frame_fallback=False)
        else:
            get_deadline()
        section_id = str(row.get("section_id", "") or "")
        if not section_id or section_id == "intro":
            continue
        phase = str(row.get("phase", "") or "")
        deps_raw = row.get("deps")
        deps: list[str] = []
        if isinstance(deps_raw, list):
            for dep in deps_raw:
                if _deadline_tick_budget_allows_check(get_deadline_clock()):
                    check_deadline(allow_frame_fallback=False)
                else:
                    get_deadline()
                if isinstance(dep, str):
                    deps.append(str(dep))
        dep_text = ", ".join(deps) if deps else "none"
        reason = existing_reason or (
            "missing_dep" if any(dep not in sections for dep in deps) else "policy"
        )
        report_lines.append(f"## Section `{section_id}`")
        report_lines.append(f"PENDING (phase: {phase}; deps: {dep_text})")
        report_lines.append("")
        sections_payload[section_id] = {
            "phase": phase,
            "deps": deps,
            "status": "pending",
            "lines": [],
            "reason": reason,
        }
        rows_payload.append(
            {
                "section_id": section_id,
                "phase": phase,
                "deps": deps,
                "status": "pending",
            }
        )
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
    if isinstance(progress_payload, Mapping):
        phase = progress_payload.get("phase")
        if isinstance(phase, str) and phase:
            lines.append(f"- `phase`: `{phase}`")
        event_kind = progress_payload.get("event_kind")
        if isinstance(event_kind, str) and event_kind:
            lines.append(f"- `event_kind`: `{event_kind}`")
        work_done_raw = progress_payload.get("work_done")
        work_total_raw = progress_payload.get("work_total")
        if isinstance(work_done_raw, int) and isinstance(work_total_raw, int):
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
        phase_progress_v2 = progress_payload.get("phase_progress_v2")
        if isinstance(phase_progress_v2, Mapping):
            primary_unit = str(phase_progress_v2.get("primary_unit", "") or "")
            raw_primary_done = phase_progress_v2.get("primary_done")
            raw_primary_total = phase_progress_v2.get("primary_total")
            if (
                isinstance(raw_primary_done, int)
                and not isinstance(raw_primary_done, bool)
                and isinstance(raw_primary_total, int)
                and not isinstance(raw_primary_total, bool)
            ):
                primary_done = max(int(raw_primary_done), 0)
                primary_total = max(int(raw_primary_total), 0)
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
        stale_for_s = progress_payload.get("stale_for_s")
        if isinstance(stale_for_s, (int, float)):
            lines.append(f"- `stale_for_s`: `{float(stale_for_s):.1f}`")
        classification = progress_payload.get("classification")
        if isinstance(classification, str):
            lines.append(f"- `classification`: `{classification}`")
        retry_recommended = progress_payload.get("retry_recommended")
        if isinstance(retry_recommended, bool):
            lines.append(f"- `retry_recommended`: `{retry_recommended}`")
        resume_supported = progress_payload.get("resume_supported")
        if isinstance(resume_supported, bool):
            lines.append(f"- `resume_supported`: `{resume_supported}`")
    lines.append("")

    pending_reasons: dict[str, str] = {}
    for row in projection_rows:
        check_deadline()
        section_id = str(row.get("section_id", "") or "")
        if not section_id:
            continue
        phase = str(row.get("phase", "") or "")
        deps_raw = row.get("deps")
        deps: list[str] = []
        if isinstance(deps_raw, list):
            deps = [str(dep) for dep in deps_raw if isinstance(dep, str)]
        section_lines = sections.get(section_id)
        if section_lines:
            lines.append(f"## Section `{section_id}`")
            lines.extend(section_lines)
            lines.append("")
            continue
        dep_text = ", ".join(deps) if deps else "none"
        if any(dep not in sections for dep in deps):
            reason = "missing_dep"
        else:
            reason = "policy"
            section_phase = str(row.get("phase", "") or "")
            try:
                report_projection_phase_rank(
                    cast(Literal["collection", "forest", "edge", "post"], section_phase)
                )
            except KeyError:
                reason = "policy"
        pending_reasons[section_id] = reason
        lines.append(f"## Section `{section_id}`")
        lines.append(f"PENDING (phase: {phase}; deps: {dep_text})")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n", pending_reasons

def _phase_timeline_md_path(*, root: Path) -> Path:
    return root / _DEFAULT_PHASE_TIMELINE_MD

def _phase_timeline_jsonl_path(*, root: Path) -> Path:
    return root / _DEFAULT_PHASE_TIMELINE_JSONL

def _progress_heartbeat_seconds(payload: Mapping[str, JSONValue]) -> float:
    return runtime_contract.progress_heartbeat_seconds(payload)

def _markdown_table_cell(value: object) -> str:
    return ("" if value is None else str(value)).replace("\n", " ").replace("|", "\\|")

def _phase_progress_dimensions_summary(
    phase_progress_v2: Mapping[str, JSONValue] | None,
) -> str:
    if not isinstance(phase_progress_v2, Mapping):
        return ""
    raw_dimensions = phase_progress_v2.get("dimensions")
    if not isinstance(raw_dimensions, Mapping):
        return ""
    fragments: list[str] = []
    dim_names = sort_once(
        (name for name in raw_dimensions if isinstance(name, str)),
        source="src/gabion/server.py:2253",
    )
    for dim_name in dim_names:
        raw_payload = raw_dimensions.get(dim_name)
        if not isinstance(raw_payload, Mapping):
            continue
        raw_done = raw_payload.get("done")
        raw_total = raw_payload.get("total")
        if (
            isinstance(raw_done, int)
            and not isinstance(raw_done, bool)
            and isinstance(raw_total, int)
            and not isinstance(raw_total, bool)
        ):
            done = max(int(raw_done), 0)
            total = max(int(raw_total), 0)
            if total:
                done = min(done, total)
            fragments.append(f"{dim_name}={done}/{total}")
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
    resume_state = (
        cast(Mapping[str, JSONValue], resume_state_intro)
        if isinstance(resume_state_intro, Mapping)
        else {}
    )
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
    semantic_progress = collection_resume.get("semantic_progress")
    if isinstance(semantic_progress, Mapping):
        semantic_witness_digest = semantic_progress.get("current_witness_digest")
        if isinstance(semantic_witness_digest, str) and semantic_witness_digest:
            lines.append(f"- `semantic_witness_digest`: `{semantic_witness_digest}`")
        new_processed_functions = semantic_progress.get("new_processed_functions_count")
        if isinstance(new_processed_functions, int):
            lines.append(f"- `new_processed_functions`: `{new_processed_functions}`")
        regressed_processed_functions = semantic_progress.get(
            "regressed_processed_functions_count"
        )
        if isinstance(regressed_processed_functions, int):
            lines.append(
                f"- `regressed_processed_functions`: `{regressed_processed_functions}`"
            )
        completed_delta = semantic_progress.get("completed_files_delta")
        if isinstance(completed_delta, int):
            lines.append(f"- `completed_files_delta`: `{completed_delta}`")
        substantive_progress = semantic_progress.get("substantive_progress")
        if isinstance(substantive_progress, bool):
            lines.append(f"- `substantive_progress`: `{substantive_progress}`")
    in_progress_states = _in_progress_scan_states(collection_resume)
    if in_progress_states:
        in_progress_paths: list[str] = []
        for in_progress_path in in_progress_states:
            check_deadline()
            in_progress_paths.append(in_progress_path)
        sample = ", ".join(in_progress_paths[:3])
        lines.append(f"- `in_progress_path_sample`: `{sample}`")
        detail_entries: list[str] = []
        for raw_path, state_mapping in in_progress_states.items():
            check_deadline()
            phase = state_mapping.get("phase")
            phase_text = phase if isinstance(phase, str) and phase else "unknown"
            processed_count = state_mapping.get("processed_functions_count")
            if not isinstance(processed_count, int):
                raw_processed = state_mapping.get("processed_functions")
                if isinstance(raw_processed, Sequence):
                    processed_count = 0
                    for entry in raw_processed:
                        check_deadline()
                        if isinstance(entry, str):
                            processed_count += 1
                else:
                    processed_count = 0
            function_count = state_mapping.get("function_count")
            if not isinstance(function_count, int):
                raw_fn_names = state_mapping.get("fn_names")
                if isinstance(raw_fn_names, Mapping):
                    function_count = 0
                    for key in raw_fn_names:
                        check_deadline()
                        if isinstance(key, str):
                            function_count += 1
                else:
                    function_count = 0
            detail_entries.append(
                (
                    f"{raw_path} "
                    f"(phase={phase_text}, processed_functions={processed_count}, "
                    f"function_count={function_count})"
                )
            )
            if len(detail_entries) >= 3:
                break
        for detail in detail_entries:
            check_deadline()
            lines.append(f"- `in_progress_detail`: `{detail}`")
    raw_analysis_index_resume = collection_resume.get("analysis_index_resume")
    if isinstance(raw_analysis_index_resume, Mapping):
        hydrated_paths_count = raw_analysis_index_resume.get("hydrated_paths_count")
        if not isinstance(hydrated_paths_count, int):
            hydrated_paths_count = _analysis_index_resume_hydrated_count(collection_resume)
        lines.append(f"- `hydrated_paths_count`: `{hydrated_paths_count}`")
        function_count = raw_analysis_index_resume.get("function_count")
        if isinstance(function_count, int):
            lines.append(f"- `hydrated_function_count`: `{function_count}`")
        class_count = raw_analysis_index_resume.get("class_count")
        if isinstance(class_count, int):
            lines.append(f"- `hydrated_class_count`: `{class_count}`")
    return lines

def _collection_components_preview_lines(
    *,
    collection_resume: Mapping[str, JSONValue],
) -> list[str]:
    check_deadline()
    raw_groups = collection_resume.get("groups_by_path")
    if not isinstance(raw_groups, Mapping):
        return [
            "Component preview (provisional).",
            "- `paths_with_groups`: `0`",
            "- `functions_with_groups`: `0`",
            "- `bundle_alternatives`: `0`",
        ]
    path_count = 0
    function_count = 0
    bundle_alternatives = 0
    for raw_path, raw_path_groups in raw_groups.items():
        check_deadline()
        if not isinstance(raw_path, str) or not isinstance(raw_path_groups, Mapping):
            continue
        path_count += 1
        for raw_qual, raw_bundles in raw_path_groups.items():
            check_deadline()
            if not isinstance(raw_qual, str):
                continue
            if not isinstance(raw_bundles, Sequence) or isinstance(
                raw_bundles, (str, bytes)
            ):
                continue
            function_count += 1
            bundle_alternatives += sum(
                1
                for bundle in raw_bundles
                if isinstance(bundle, Sequence) and not isinstance(bundle, (str, bytes))
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
    groups_by_path: dict[Path, dict[str, list[set[str]]]] = {}
    raw_groups = collection_resume.get("groups_by_path")
    if not isinstance(raw_groups, Mapping):
        return groups_by_path
    for raw_path, raw_path_groups in raw_groups.items():
        check_deadline()
        if not isinstance(raw_path, str) or not isinstance(raw_path_groups, Mapping):
            continue
        path_groups: dict[str, list[set[str]]] = {}
        for raw_qual, raw_bundles in raw_path_groups.items():
            check_deadline()
            if not isinstance(raw_qual, str):
                continue
            if not isinstance(raw_bundles, Sequence) or isinstance(
                raw_bundles, (str, bytes)
            ):
                continue
            bundles: list[set[str]] = []
            for raw_bundle in raw_bundles:
                check_deadline()
                if not isinstance(raw_bundle, Sequence) or isinstance(
                    raw_bundle, (str, bytes)
                ):
                    continue
                bundles.append({entry for entry in raw_bundle if isinstance(entry, str)})
            path_groups[raw_qual] = bundles
        groups_by_path[Path(raw_path)] = path_groups
    return groups_by_path

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
    classification = (
        str(progress_payload.get("classification", "") or "")
        if isinstance(progress_payload, Mapping)
        else ""
    )
    resume_supported = (
        bool(progress_payload.get("resume_supported"))
        if isinstance(progress_payload, Mapping)
        else False
    )
    semantic_progress = (
        progress_payload.get("semantic_progress")
        if isinstance(progress_payload, Mapping)
        else None
    )
    semantic_monotonic_progress: bool | None = None
    semantic_substantive_progress: bool | None = None
    if isinstance(semantic_progress, Mapping):
        raw_monotonic = semantic_progress.get("monotonic_progress")
        raw_substantive = semantic_progress.get("substantive_progress")
        if isinstance(raw_monotonic, bool):
            semantic_monotonic_progress = raw_monotonic
        if isinstance(raw_substantive, bool):
            semantic_substantive_progress = raw_substantive
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
            1
            for row in projection_rows
            if isinstance(row, Mapping)
            and isinstance(row.get("section_id"), str)
            and str(row.get("section_id") or "")
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
    for row in projection_rows:
        check_deadline()
        section_id = str(row.get("section_id", "") or "")
        if not section_id:
            continue
        if section_id in sections:
            status = "SATISFIED"
            detail = "section reused from witness-matched journal"
        else:
            status = "OBLIGATION"
            detail = pending_reasons.get(section_id, "section pending")
            if stale_input_detected and detail == "policy":
                detail = "stale_input"
        obligations.append(
            {
                "status": status,
                "contract": "incremental_projection_contract",
                "kind": "section_projection_state",
                "section_id": section_id,
                "phase": str(row.get("phase", "") or ""),
                "detail": detail,
            }
        )
    return obligations

def _split_incremental_obligations(
    obligations: Sequence[Mapping[str, JSONValue]],
) -> tuple[list[JSONObject], list[JSONObject]]:
    check_deadline()
    resumability: list[JSONObject] = []
    incremental: list[JSONObject] = []
    for raw_entry in obligations:
        check_deadline()
        if not isinstance(raw_entry, Mapping):
            continue
        entry: JSONObject = {str(key): raw_entry[key] for key in raw_entry}
        contract = str(entry.get("contract", "") or "")
        if contract == "resume_contract":
            resumability.append(entry)
            continue
        if contract in {"progress_report_contract", "incremental_projection_contract"}:
            incremental.append(entry)
    return resumability, incremental

def _apply_journal_pending_reason(
    *,
    projection_rows: Sequence[Mapping[str, JSONValue]],
    sections: Mapping[str, object],
    pending_reasons: dict[str, str],
    journal_reason: str | None,
) -> None:
    if journal_reason not in {"stale_input", "policy"}:
        return
    for row in projection_rows:
        check_deadline()
        section_id = str(row.get("section_id", "") or "")
        if not section_id or section_id in sections:
            continue
        pending_reasons[section_id] = journal_reason

def _latest_report_phase(phases: Mapping[str, JSONValue] | None) -> str | None:
    check_deadline()
    if not isinstance(phases, Mapping):
        return None
    best_phase: str | None = None
    best_rank = -1
    for phase_name in phases:
        check_deadline()
        if not isinstance(phase_name, str):
            continue
        try:
            rank = report_projection_phase_rank(
                cast(Literal["collection", "forest", "edge", "post"], phase_name)
            )
        except KeyError:
            continue
        if rank > best_rank:
            best_rank = rank
            best_phase = phase_name
    return best_phase

def _parse_lint_line(line: str) -> LintEntryDTO | None:
    return parse_lint_line(line)

def _parse_lint_line_as_payload(line: str) -> dict[str, object] | None:
    entry = _parse_lint_line(line)
    return entry.model_dump() if entry is not None else None

def _normalize_dataflow_response(response: Mapping[str, object]) -> dict[str, object]:
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
    supported_analysis_surfaces = (
        sort_once(
            [str(item) for item in supported_analysis_surfaces_raw],
            source="server._normalize_dataflow_response.supported_analysis_surfaces",
        )
        if isinstance(supported_analysis_surfaces_raw, list)
        else []
    )
    disabled_surface_reasons = (
        {
            str(key): str(disabled_surface_reasons_raw[key])
            for key in sort_once(
                disabled_surface_reasons_raw,
                source="server._normalize_dataflow_response.disabled_surface_keys",
            )
        }
        if isinstance(disabled_surface_reasons_raw, Mapping)
        else {}
    )
    base = LegacyDataflowMonolithResponseDTO(
        exit_code=int(response.get("exit_code", 0) or 0),
        timeout=bool(response.get("timeout", False)),
        analysis_state=(str(response.get("analysis_state")) if response.get("analysis_state") is not None else None),
        errors=[str(err) for err in (response.get("errors") or [])] if isinstance(response.get("errors"), list) else [],
        lint_lines=lint_lines,
        lint_entries=[LintEntryDTO.model_validate(entry) for entry in lint_entries],
        selected_adapter=(
            str(response.get("selected_adapter"))
            if response.get("selected_adapter") is not None
            else None
        ),
        supported_analysis_surfaces=supported_analysis_surfaces,
        disabled_surface_reasons=disabled_surface_reasons,
        aspf_trace=aspf_trace_raw if isinstance(aspf_trace_raw, Mapping) else None,
        aspf_equivalence=(
            aspf_equivalence_raw if isinstance(aspf_equivalence_raw, Mapping) else None
        ),
        aspf_opportunities=(
            aspf_opportunities_raw
            if isinstance(aspf_opportunities_raw, Mapping)
            else None
        ),
        aspf_delta_ledger=(
            aspf_delta_ledger_raw if isinstance(aspf_delta_ledger_raw, Mapping) else None
        ),
        aspf_state=aspf_state_raw if isinstance(aspf_state_raw, Mapping) else None,
        payload={str(key): response[key] for key in response},
    )
    normalized = dict(base.payload)
    rewrite_plans = normalized.get("fingerprint_rewrite_plans")
    if isinstance(rewrite_plans, list):
        ordered_plans = normalize_rewrite_plan_order(
            [entry for entry in rewrite_plans if isinstance(entry, dict)]
        )
        normalized["fingerprint_rewrite_plans"] = ordered_plans
        rewrite_plan_schema_errors: list[dict[str, object]] = []
        for entry in ordered_plans:
            issues = validate_rewrite_plan_payload(entry)
            if issues:
                rewrite_plan_schema_errors.append(
                    {"plan_id": str(entry.get("plan_id", "")), "issues": issues}
                )
        if rewrite_plan_schema_errors:
            normalized["rewrite_plan_schema_errors"] = rewrite_plan_schema_errors

    normalized["exit_code"] = base.exit_code
    normalized["timeout"] = base.timeout
    normalized["analysis_state"] = base.analysis_state
    normalized["errors"] = base.errors
    normalized["lint_lines"] = base.lint_lines
    normalized["selected_adapter"] = base.selected_adapter
    normalized["supported_analysis_surfaces"] = list(
        base.supported_analysis_surfaces
    )
    normalized["disabled_surface_reasons"] = dict(base.disabled_surface_reasons)
    normalized["lint_entries"] = [entry.model_dump() for entry in base.lint_entries]
    payload = normalized.get("payload")
    if isinstance(payload, Mapping):
        payload_updates: dict[str, object] = {
            "selected_adapter": base.selected_adapter,
            "supported_analysis_surfaces": list(base.supported_analysis_surfaces),
            "disabled_surface_reasons": dict(base.disabled_surface_reasons),
        }
        normalized["payload"] = boundary_order.apply_boundary_updates_once(
            {str(key): payload[key] for key in payload},
            payload_updates,
            source="server._normalize_dataflow_response.payload_capabilities",
        )
    if base.aspf_trace is not None:
        normalized["aspf_trace"] = base.aspf_trace.model_dump()
    if base.aspf_equivalence is not None:
        normalized["aspf_equivalence"] = base.aspf_equivalence.model_dump()
    if base.aspf_opportunities is not None:
        normalized["aspf_opportunities"] = base.aspf_opportunities.model_dump()
    if base.aspf_delta_ledger is not None:
        normalized["aspf_delta_ledger"] = base.aspf_delta_ledger.model_dump()
    if base.aspf_state is not None:
        normalized["aspf_state"] = base.aspf_state.model_dump()
    return boundary_order.canonicalize_boundary_mapping(
        normalized,
        source="server._normalize_dataflow_response",
    )

def _truthy_flag(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return value != 0
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "on"}

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

def _analysis_timeout_total_ns(payload: dict[str, object]) -> int:
    return payload_codec.analysis_timeout_total_ns(
        payload,
        source="server._analysis_timeout_total_ns.payload_keys",
        reject_sub_millisecond_seconds=True,
    )

def _analysis_timeout_total_ticks(payload: dict[str, object]) -> int:
    return payload_codec.analysis_timeout_total_ticks(
        payload,
        source="server._analysis_timeout_total_ticks.payload_keys",
    )

def _analysis_timeout_grace_ns(
    payload: dict[str, object],
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
    payload: dict[str, object],
) -> tuple[int, int, int]:
    total_ns = _analysis_timeout_total_ns(payload)
    cleanup_grace_ns = _analysis_timeout_grace_ns(payload, total_ns=total_ns)
    analysis_ns = max(1, total_ns - cleanup_grace_ns)
    cleanup_ns = max(0, total_ns - analysis_ns)
    return total_ns, analysis_ns, cleanup_ns

def _deadline_profile_sample_interval(
    payload: dict[str, object],
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

def _normalize_csv_or_iterable_names(value: object, *, strict: bool) -> list[str]:
    check_deadline()
    if value is None:
        return []
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    if isinstance(value, (list, tuple, set)):
        items: list[str] = []
        for item in value:
            check_deadline()
            if isinstance(item, str):
                items.extend([part.strip() for part in item.split(",") if part.strip()])
            elif strict:
                never("name set contains non-string entry", value_type=type(item).__name__)
        return items
    if strict:
        never("invalid name set payload", value_type=type(value).__name__)
    return []

def _normalize_transparent_decorators(value: object) -> set[str] | None:
    items = _normalize_csv_or_iterable_names(value, strict=False)
    if not items:
        return None
    return set(items)

def _normalize_name_set(value: object) -> set[str] | None:
    if value is None:
        return None
    items = _normalize_csv_or_iterable_names(value, strict=True)
    return set(items)

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
        if isinstance(payload, Mapping):
            return {str(key): payload[key] for key in payload}
    return {
        "summary": "Analysis timed out.",
        "progress": {"classification": "timed_out_no_progress"},
    }

def _materialize_execution_plan(payload: Mapping[str, object]) -> ExecutionPlan:
    request_value = payload.get("execution_plan_request")
    if isinstance(request_value, Mapping):
        req_ops = request_value.get("requested_operations")
        requested_operations = [str(op) for op in req_ops] if isinstance(req_ops, list) else [DATAFLOW_COMMAND]
        inputs_value = request_value.get("inputs")
        if isinstance(inputs_value, Mapping):
            inputs = {str(key): inputs_value[key] for key in inputs_value}
        else:
            inputs = {str(key): payload[key] for key in payload if key != "execution_plan_request"}
        artifacts_value = request_value.get("derived_artifacts")
        derived_artifacts = (
            [str(path) for path in artifacts_value]
            if isinstance(artifacts_value, list)
            else ["artifacts/out/execution_plan.json"]
        )
        obligations_value = request_value.get("obligations")
        preconditions: list[str] = []
        postconditions: list[str] = []
        if isinstance(obligations_value, Mapping):
            pre_raw = obligations_value.get("preconditions")
            post_raw = obligations_value.get("postconditions")
            if isinstance(pre_raw, list):
                preconditions = [str(item) for item in pre_raw]
            if isinstance(post_raw, list):
                postconditions = [str(item) for item in post_raw]
        policy_value = request_value.get("policy_metadata")
        policy_deadline: dict[str, int] = {}
        policy_baseline_mode = "none"
        policy_docflow_mode = "disabled"
        if isinstance(policy_value, Mapping):
            deadline_value = policy_value.get("deadline")
            if isinstance(deadline_value, Mapping):
                for key, value in deadline_value.items():
                    if isinstance(value, bool):
                        continue
                    if isinstance(value, int):
                        policy_deadline[str(key)] = int(value)
            baseline_mode = policy_value.get("baseline_mode")
            if isinstance(baseline_mode, str):
                policy_baseline_mode = baseline_mode
            docflow_mode = policy_value.get("docflow_mode")
            if isinstance(docflow_mode, str):
                policy_docflow_mode = docflow_mode
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
    inputs = {str(key): payload[key] for key in payload}
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

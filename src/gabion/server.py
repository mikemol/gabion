from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import threading
import time
from datetime import datetime, timezone
from collections import defaultdict, deque
from contextlib import contextmanager
from dataclasses import dataclass, replace
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Callable, Literal, Mapping, Sequence, cast
from urllib.parse import unquote, urlparse

from pygls.lsp.server import LanguageServer
from pydantic import ValidationError
from lsprotocol.types import (
    TEXT_DOCUMENT_DID_OPEN,
    TEXT_DOCUMENT_DID_SAVE,
    TEXT_DOCUMENT_CODE_ACTION,
    CodeAction,
    CodeActionKind,
    CodeActionParams,
    Command,
    Diagnostic,
    DiagnosticSeverity,
    Position,
    Range,
    WorkspaceEdit,
)

from gabion.json_types import JSONObject, JSONValue
from gabion.plan import (
    ExecutionPlan,
    ExecutionPlanObligations,
    ExecutionPlanPolicyMetadata,
    write_execution_plan_artifact,
)

from gabion.analysis import (
    AnalysisResult,
    AuditConfig,
    ReportCarrier,
    analyze_paths,
    apply_baseline,
    build_analysis_collection_resume_seed,
    compute_structure_metrics,
    compute_structure_reuse,
    render_reuse_lemma_stubs,
    compute_violations,
    build_refactor_plan,
    build_synthesis_plan,
    diff_structure_snapshots,
    diff_decision_snapshots,
    load_structure_snapshot,
    load_decision_snapshot,
    load_baseline,
    extract_report_sections,
    project_report_sections,
    report_projection_phase_rank,
    report_projection_spec_rows,
    render_dot,
    render_structure_snapshot,
    render_decision_snapshot,
    DecisionSnapshotSurfaces,
    render_protocol_stubs,
    render_refactor_plan,
    render_report,
    render_synthesis_section,
    resolve_analysis_paths,
    resolve_baseline_path,
    write_baseline,
)
from gabion.analysis.aspf import Forest
from gabion.analysis import ambiguity_delta
from gabion.analysis import ambiguity_state
from gabion.analysis import call_cluster_consolidation
from gabion.analysis import call_clusters
from gabion.analysis import semantic_coverage_map
from gabion.analysis import test_annotation_drift
from gabion.analysis import test_annotation_drift_delta
from gabion.analysis import test_obsolescence
from gabion.analysis import test_obsolescence_delta
from gabion.analysis import test_obsolescence_state
from gabion.analysis import test_evidence_suggestions
from gabion.analysis.timeout_context import (
    Deadline,
    GasMeter,
    TimeoutExceeded,
    check_deadline,
    deadline_loop_iter,
    get_deadline,
    get_deadline_clock,
    record_deadline_io,
    reset_deadline_clock,
    forest_scope,
    reset_forest,
    set_forest,
    reset_deadline_profile,
    reset_deadline,
    set_deadline_profile,
    set_deadline,
    set_deadline_clock,
)
from gabion.invariants import never
from gabion.order_contract import OrderPolicy, ordered_or_sorted
from gabion.config import (
    dataflow_defaults,
    dataflow_deadline_roots,
    decision_defaults,
    decision_ignore_list,
    decision_require_tiers,
    decision_tier_map,
    exception_defaults,
    exception_never_list,
    fingerprint_defaults,
    merge_payload,
)
from gabion.analysis.type_fingerprints import (
    Fingerprint,
    PrimeRegistry,
    TypeConstructorRegistry,
    build_fingerprint_registry,
)
from gabion.refactor import (
    FieldSpec,
    RefactorEngine,
    RefactorCompatibilityShimConfig,
    RefactorRequest as RefactorRequestModel,
)
from gabion.refactor.rewrite_plan import normalize_rewrite_plan_order, validate_rewrite_plan_payload
from gabion.schema import (
    DataflowAuditResponseDTO,
    DecisionDiffResponseDTO,
    LintEntryDTO,
    RefactorProtocolResponseDTO,
    RefactorRequest,
    RefactorResponse,
    RewritePlanEntryDTO,
    StructureDiffResponseDTO,
    StructureReuseResponseDTO,
    SynthesisPlanResponseDTO,
    SynthesisResponse,
    SynthesisRequest,
    TextEditDTO,
)
from gabion.synthesis import NamingContext, SynthesisConfig, Synthesizer

server = LanguageServer("gabion", "0.1.0")
DATAFLOW_COMMAND = "gabion.dataflowAudit"
SYNTHESIS_COMMAND = "gabion.synthesisPlan"
REFACTOR_COMMAND = "gabion.refactorProtocol"
STRUCTURE_DIFF_COMMAND = "gabion.structureDiff"
STRUCTURE_REUSE_COMMAND = "gabion.structureReuse"
DECISION_DIFF_COMMAND = "gabion.decisionDiff"
IMPACT_COMMAND = "gabion.impactQuery"

_IMPACT_CHANGE_RE = re.compile(r"^(?P<path>.+?)(?::(?P<start>\d+)(?:-(?P<end>\d+))?)?$")
_IMPACT_DIFF_FILE_RE = re.compile(r"^\+\+\+\s+(?:a/|b/)?(?P<path>.+)$")
_IMPACT_DIFF_HUNK_RE = re.compile(
    r"^@@\s+-\d+(?:,\d+)?\s+\+(?P<start>\d+)(?:,(?P<count>\d+))?\s+@@"
)
_IMPACT_TEST_PATH_TOKENS = ("/tests/", "\\tests\\")

_SERVER_DEADLINE_OVERHEAD_MIN_NS = 10_000_000
_SERVER_DEADLINE_OVERHEAD_MAX_NS = 200_000_000
_SERVER_DEADLINE_OVERHEAD_DIVISOR = 20
_ANALYSIS_TIMEOUT_GRACE_RATIO_NUMERATOR = 1
_ANALYSIS_TIMEOUT_GRACE_RATIO_DENOMINATOR = 5
_ANALYSIS_RESUME_CHECKPOINT_FORMAT_VERSION = 1
_ANALYSIS_RESUME_STATE_REF_FORMAT_VERSION = 1
_ANALYSIS_RESUME_INLINE_STATE_MAX_BYTES = 64_000
_ANALYSIS_RESUME_CHUNK_DIR_SUFFIX = ".chunks"
_ANALYSIS_INPUT_MANIFEST_FORMAT_VERSION = 1
_ANALYSIS_INPUT_WITNESS_FORMAT_VERSION = 2
_DEFAULT_ANALYSIS_RESUME_CHECKPOINT = Path(
    "artifacts/audit_reports/dataflow_resume_checkpoint.json"
)
_DEFAULT_CHECKPOINT_INTRO_TIMELINE = Path(
    "artifacts/audit_reports/dataflow_checkpoint_intro_timeline.md"
)
_DEFAULT_PHASE_TIMELINE_MD = Path(
    "artifacts/audit_reports/dataflow_phase_timeline.md"
)
_DEFAULT_PHASE_TIMELINE_JSONL = Path(
    "artifacts/audit_reports/dataflow_phase_timeline.jsonl"
)
_REPORT_SECTION_JOURNAL_FORMAT_VERSION = 1
_DEFAULT_REPORT_SECTION_JOURNAL = Path(
    "artifacts/audit_reports/dataflow_report_sections.json"
)
_REPORT_PHASE_CHECKPOINT_FORMAT_VERSION = 1
_DEFAULT_REPORT_PHASE_CHECKPOINT = Path(
    "artifacts/audit_reports/dataflow_report_phase_checkpoint.json"
)
_COLLECTION_CHECKPOINT_FLUSH_INTERVAL_NS = 2_000_000_000
_COLLECTION_CHECKPOINT_MEANINGFUL_MIN_INTERVAL_NS = 1_000_000_000
_COLLECTION_REPORT_FLUSH_INTERVAL_NS = 10_000_000_000
_COLLECTION_REPORT_FLUSH_COMPLETED_STRIDE = 8
_DEFAULT_DEADLINE_PROFILE_SAMPLE_INTERVAL = 16
_DEFAULT_PROGRESS_HEARTBEAT_SECONDS = 55.0
_MIN_PROGRESS_HEARTBEAT_SECONDS = 5.0
_LINT_RE = re.compile(r"^(?P<path>.+?):(?P<line>\d+):(?P<col>\d+):\s*(?P<rest>.*)$")
_LSP_PROGRESS_NOTIFICATION_METHOD = "$/progress"
_LSP_PROGRESS_TOKEN = "gabion.dataflowAudit/progress-v1"
_STDOUT_ALIAS = "-"
_STDOUT_PATH = "/dev/stdout"


def _is_stdout_target(target: object) -> bool:
    if not isinstance(target, (str, Path)):
        return False
    text = str(target).strip()
    return text in {_STDOUT_ALIAS, _STDOUT_PATH}


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
    }
    if status == "checkpoint_loaded":
        if reused_files > 0:
            return "hit"
        return "miss"
    if status == "checkpoint_seeded":
        if compatibility_status in invalidation_statuses:
            return "invalidated"
        return "seeded"
    if compatibility_status in invalidation_statuses:
        return "invalidated"
    return "miss"


def _deadline_tick_budget_allows_check(clock: object) -> bool:
    limit = getattr(clock, "limit", None)
    current = getattr(clock, "current", None)
    if isinstance(limit, int) and isinstance(current, int):
        return (limit - current) > 1
    return True


@dataclass(frozen=True)
class ExecuteCommandDeps:
    analyze_paths_fn: Callable[..., AnalysisResult]
    load_analysis_resume_checkpoint_manifest_fn: Callable[..., tuple[JSONObject | None, JSONObject] | None]
    write_analysis_resume_checkpoint_fn: Callable[..., None]
    load_analysis_resume_checkpoint_fn: Callable[..., JSONObject | None]
    analysis_input_manifest_fn: Callable[..., JSONObject]
    analysis_input_manifest_digest_fn: Callable[[JSONObject], str]
    build_analysis_collection_resume_seed_fn: Callable[..., JSONObject]
    collection_semantic_progress_fn: Callable[..., JSONObject]
    load_report_phase_checkpoint_fn: Callable[..., JSONObject]
    project_report_sections_fn: Callable[..., dict[str, list[str]]]
    report_projection_spec_rows_fn: Callable[[], list[JSONObject]]
    collection_checkpoint_flush_due_fn: Callable[..., bool]
    write_bootstrap_incremental_artifacts_fn: Callable[..., None]
    load_report_section_journal_fn: Callable[..., tuple[dict[str, list[str]], str | None]]

    def with_overrides(self, **overrides: object) -> "ExecuteCommandDeps":
        return replace(self, **overrides)


def _collection_checkpoint_flush_due(
    *,
    intro_changed: bool,
    remaining_files: int,
    semantic_substantive_progress: bool = False,
    now_ns: int,
    last_flush_ns: int,
) -> bool:
    if intro_changed or remaining_files == 0:
        return True
    elapsed_ns = max(0, now_ns - last_flush_ns)
    if semantic_substantive_progress:
        return (
            elapsed_ns >= _COLLECTION_CHECKPOINT_MEANINGFUL_MIN_INTERVAL_NS
            or elapsed_ns >= _COLLECTION_CHECKPOINT_FLUSH_INTERVAL_NS
        )
    return elapsed_ns >= _COLLECTION_CHECKPOINT_FLUSH_INTERVAL_NS


def _collection_report_flush_due(
    *,
    completed_files: int,
    remaining_files: int,
    now_ns: int,
    last_flush_ns: int,
    last_flush_completed: int,
) -> bool:
    if last_flush_completed < 0:
        return True
    if completed_files - last_flush_completed >= _COLLECTION_REPORT_FLUSH_COMPLETED_STRIDE:
        return True
    if now_ns - last_flush_ns >= _COLLECTION_REPORT_FLUSH_INTERVAL_NS:
        return True
    return remaining_files == 0


def _projection_phase_flush_due(
    *,
    phase: Literal["collection", "forest", "edge", "post"],
    now_ns: int,
    last_flush_ns: int,
) -> bool:
    if phase == "post":
        return True
    return now_ns - last_flush_ns >= _COLLECTION_REPORT_FLUSH_INTERVAL_NS


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


def _resolve_analysis_resume_checkpoint_path(
    value: object,
    *,
    root: Path,
) -> Path | None:
    if value is False:
        return None
    if value is None or value is True:
        return root / _DEFAULT_ANALYSIS_RESUME_CHECKPOINT
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        path = Path(text)
        if path.is_absolute():
            return path
        return root / path
    never("invalid analysis resume checkpoint payload", value_type=type(value).__name__)


def _analysis_witness_config_payload(config: AuditConfig) -> JSONObject:
    return {
        "exclude_dirs": ordered_or_sorted(
            config.exclude_dirs,
            source="_analysis_witness_config_payload.exclude_dirs",
        ),
        "ignore_params": ordered_or_sorted(
            config.ignore_params,
            source="_analysis_witness_config_payload.ignore_params",
        ),
        "strictness": config.strictness,
        "external_filter": config.external_filter,
        "transparent_decorators": ordered_or_sorted(
            config.transparent_decorators or [],
            source="_analysis_witness_config_payload.transparent_decorators",
        ),
    }


def _analysis_resume_checkpoint_chunks_dir(path: Path) -> Path:
    return path.with_name(f"{path.name}{_ANALYSIS_RESUME_CHUNK_DIR_SUFFIX}")


def _analysis_resume_state_chunk_name(path_key: str) -> str:
    return f"{hashlib.sha1(path_key.encode('utf-8')).hexdigest()}.json"


def _analysis_resume_named_chunk_name(label: str) -> str:
    return f"{hashlib.sha1(label.encode('utf-8')).hexdigest()}.json"


def _externalize_collection_resume_states(
    *,
    path: Path,
    collection_resume: JSONObject,
) -> JSONObject:
    raw_in_progress = collection_resume.get("in_progress_scan_by_path")
    raw_analysis_index_resume = collection_resume.get("analysis_index_resume")
    if not isinstance(raw_in_progress, Mapping) and not isinstance(
        raw_analysis_index_resume,
        Mapping,
    ):
        return {str(key): collection_resume[key] for key in collection_resume}
    if not isinstance(raw_in_progress, Mapping):
        raw_in_progress = {}
    chunks_dir = _analysis_resume_checkpoint_chunks_dir(path)
    chunks_dir.mkdir(parents=True, exist_ok=True)
    used_chunk_names: set[str] = set()
    in_progress_payload: JSONObject = {}
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
            in_progress_payload[raw_path] = cast(JSONValue, raw_state)
            continue
        state_payload: JSONObject = {str(key): raw_state[key] for key in raw_state}
        state_text = _canonical_json_text(state_payload)
        if len(state_text.encode("utf-8")) <= _ANALYSIS_RESUME_INLINE_STATE_MAX_BYTES:
            in_progress_payload[raw_path] = state_payload
            continue
        chunk_name = _analysis_resume_state_chunk_name(raw_path)
        chunk_path = chunks_dir / chunk_name
        chunk_payload: JSONObject = {
            "format_version": _ANALYSIS_RESUME_STATE_REF_FORMAT_VERSION,
            "path": raw_path,
            "state": state_payload,
        }
        _write_text_profiled(
            chunk_path,
            _canonical_json_text(chunk_payload) + "\n",
            io_name="analysis_resume.chunk_write",
        )
        used_chunk_names.add(chunk_name)
        summary: JSONObject = {
            "phase": (
                state_payload.get("phase")
                if isinstance(state_payload.get("phase"), str)
                else "function_scan"
            ),
            "state_ref": chunk_name,
        }
        raw_processed = state_payload.get("processed_functions")
        if isinstance(raw_processed, Sequence):
            processed_functions = {
                entry for entry in raw_processed if isinstance(entry, str)
            }
            summary["processed_functions_count"] = len(processed_functions)
            summary["processed_functions_digest"] = hashlib.sha1(
                _canonical_json_text(sorted(processed_functions)).encode("utf-8")
            ).hexdigest()
        else:
            raw_processed_digest = state_payload.get("processed_functions_digest")
            if isinstance(raw_processed_digest, str) and raw_processed_digest:
                summary["processed_functions_digest"] = raw_processed_digest
        raw_fn_names = state_payload.get("fn_names")
        if isinstance(raw_fn_names, Mapping):
            summary["function_count"] = sum(
                1 for key in raw_fn_names if isinstance(key, str)
            )
        in_progress_payload[raw_path] = summary
    analysis_index_resume_payload: JSONValue | None = None
    if isinstance(raw_analysis_index_resume, Mapping):
        state_payload: JSONObject = {
            str(key): raw_analysis_index_resume[key] for key in raw_analysis_index_resume
        }
        state_text = _canonical_json_text(state_payload)
        if len(state_text.encode("utf-8")) <= _ANALYSIS_RESUME_INLINE_STATE_MAX_BYTES:
            analysis_index_resume_payload = state_payload
        else:
            chunk_name = _analysis_resume_named_chunk_name("analysis_index_resume")
            chunk_path = chunks_dir / chunk_name
            chunk_payload: JSONObject = {
                "format_version": _ANALYSIS_RESUME_STATE_REF_FORMAT_VERSION,
                "path": "analysis_index_resume",
                "state": state_payload,
            }
            _write_text_profiled(
                chunk_path,
                _canonical_json_text(chunk_payload) + "\n",
                io_name="analysis_resume.chunk_write",
            )
            used_chunk_names.add(chunk_name)
            summary: JSONObject = {
                "phase": (
                    state_payload.get("phase")
                    if isinstance(state_payload.get("phase"), str)
                    else "analysis_index_hydration"
                ),
                "state_ref": chunk_name,
            }
            hydrated_paths = state_payload.get("hydrated_paths")
            if isinstance(hydrated_paths, Sequence):
                summary["hydrated_paths_count"] = sum(
                    1 for entry in hydrated_paths if isinstance(entry, str)
                )
            function_count = state_payload.get("function_count")
            if isinstance(function_count, int):
                summary["function_count"] = function_count
            class_count = state_payload.get("class_count")
            if isinstance(class_count, int):
                summary["class_count"] = class_count
            analysis_index_resume_payload = summary
    for stale in chunks_dir.glob("*.json"):
        check_deadline()
        if stale.name in used_chunk_names:
            continue
        try:
            stale.unlink()
        except OSError:
            continue
    if not used_chunk_names:
        try:
            chunks_dir.rmdir()
        except OSError:
            pass
    payload: JSONObject = {str(key): collection_resume[key] for key in collection_resume}
    payload["in_progress_scan_by_path"] = in_progress_payload
    if analysis_index_resume_payload is not None:
        payload["analysis_index_resume"] = analysis_index_resume_payload
    return payload


def _inflate_collection_resume_states(
    *,
    path: Path,
    collection_resume: JSONObject,
) -> JSONObject:
    raw_in_progress = collection_resume.get("in_progress_scan_by_path")
    raw_analysis_index_resume = collection_resume.get("analysis_index_resume")
    if not isinstance(raw_in_progress, Mapping) and not isinstance(
        raw_analysis_index_resume,
        Mapping,
    ):
        return {str(key): collection_resume[key] for key in collection_resume}
    if not isinstance(raw_in_progress, Mapping):
        raw_in_progress = {}
    chunks_dir = _analysis_resume_checkpoint_chunks_dir(path)
    in_progress_payload: JSONObject = {}
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
            in_progress_payload[raw_path] = cast(JSONValue, raw_state)
            continue
        state_ref = raw_state.get("state_ref")
        if isinstance(state_ref, str) and state_ref:
            chunk_path = chunks_dir / state_ref
            try:
                chunk_data = json.loads(
                    _read_text_profiled(
                        chunk_path,
                        io_name="analysis_resume.chunk_read",
                    )
                )
            except (OSError, UnicodeError, json.JSONDecodeError):
                in_progress_payload[raw_path] = {str(key): raw_state[key] for key in raw_state}
                continue
            if (
                isinstance(chunk_data, Mapping)
                and chunk_data.get("format_version")
                == _ANALYSIS_RESUME_STATE_REF_FORMAT_VERSION
                and chunk_data.get("path") == raw_path
            ):
                state_payload = chunk_data.get("state")
                if isinstance(state_payload, Mapping):
                    in_progress_payload[raw_path] = {
                        str(key): state_payload[key] for key in state_payload
                    }
                    continue
        in_progress_payload[raw_path] = {str(key): raw_state[key] for key in raw_state}
    analysis_index_resume_payload: JSONValue | None = None
    if isinstance(raw_analysis_index_resume, Mapping):
        state_ref = raw_analysis_index_resume.get("state_ref")
        if isinstance(state_ref, str) and state_ref:
            chunk_path = chunks_dir / state_ref
            try:
                chunk_data = json.loads(
                    _read_text_profiled(
                        chunk_path,
                        io_name="analysis_resume.chunk_read",
                    )
                )
            except (OSError, UnicodeError, json.JSONDecodeError):
                analysis_index_resume_payload = {
                    str(key): raw_analysis_index_resume[key]
                    for key in raw_analysis_index_resume
                }
            else:
                if (
                    isinstance(chunk_data, Mapping)
                    and chunk_data.get("format_version")
                    == _ANALYSIS_RESUME_STATE_REF_FORMAT_VERSION
                    and chunk_data.get("path") == "analysis_index_resume"
                ):
                    state_payload = chunk_data.get("state")
                    if isinstance(state_payload, Mapping):
                        analysis_index_resume_payload = {
                            str(key): state_payload[key] for key in state_payload
                        }
                if analysis_index_resume_payload is None:
                    analysis_index_resume_payload = {
                        str(key): raw_analysis_index_resume[key]
                        for key in raw_analysis_index_resume
                    }
        else:
            analysis_index_resume_payload = {
                str(key): raw_analysis_index_resume[key]
                for key in raw_analysis_index_resume
            }
    payload: JSONObject = {str(key): collection_resume[key] for key in collection_resume}
    payload["in_progress_scan_by_path"] = in_progress_payload
    if analysis_index_resume_payload is not None:
        payload["analysis_index_resume"] = analysis_index_resume_payload
    return payload


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


def _analysis_manifest_digest_from_witness(input_witness: JSONObject) -> str | None:
    check_deadline()
    files = input_witness.get("files")
    if not isinstance(files, list):
        return None
    manifest_files: list[JSONObject] = []
    for raw_entry in files:
        check_deadline()
        if not isinstance(raw_entry, Mapping):
            return None
        path_value = raw_entry.get("path")
        if not isinstance(path_value, str):
            return None
        manifest_entry: JSONObject = {"path": path_value}
        missing_value = raw_entry.get("missing")
        if isinstance(missing_value, bool):
            manifest_entry["missing"] = missing_value
        size_value = raw_entry.get("size")
        if isinstance(size_value, int):
            manifest_entry["size"] = size_value
        content_sha1_value = raw_entry.get("content_sha1")
        if isinstance(content_sha1_value, str) and content_sha1_value:
            manifest_entry["content_sha1"] = content_sha1_value
        manifest_files.append(manifest_entry)
    config = input_witness.get("config")
    if not isinstance(config, Mapping):
        return None
    config_payload: JSONObject = {}
    for key in (
        "exclude_dirs",
        "ignore_params",
        "strictness",
        "external_filter",
        "transparent_decorators",
    ):
        check_deadline()
        value = config.get(key)
        if isinstance(value, list):
            config_payload[key] = [str(item) for item in value]
            continue
        if isinstance(value, bool | int | str):
            config_payload[key] = value
            continue
        return None
    root = input_witness.get("root")
    recursive = input_witness.get("recursive")
    include_invariant_propositions = input_witness.get("include_invariant_propositions")
    include_wl_refinement = input_witness.get("include_wl_refinement")
    if not isinstance(root, str):
        return None
    if not isinstance(recursive, bool):
        return None
    if not isinstance(include_invariant_propositions, bool):
        return None
    if not isinstance(include_wl_refinement, bool):
        return None
    manifest: JSONObject = {
        "format_version": _ANALYSIS_INPUT_MANIFEST_FORMAT_VERSION,
        "root": root,
        "recursive": recursive,
        "include_invariant_propositions": include_invariant_propositions,
        "include_wl_refinement": include_wl_refinement,
        "config": config_payload,
        "files": manifest_files,
    }
    return _analysis_input_manifest_digest(manifest)


def _analysis_input_witness(
    *,
    root: Path,
    file_paths: list[Path],
    recursive: bool,
    include_invariant_propositions: bool,
    include_wl_refinement: bool,
    config: AuditConfig,
    read_text_fn: Callable[..., str] | None = None,
    parse_source_fn: Callable[..., ast.AST] | None = None,
) -> JSONObject:
    if read_text_fn is None:
        read_text_fn = _read_text_profiled
    if parse_source_fn is None:
        parse_source_fn = ast.parse
    def _normalize_scalar(value: object) -> JSONValue:
        if value is None or isinstance(value, (bool, int, str)):
            return value
        if isinstance(value, float):
            return {"_py": "float", "value": repr(value)}
        if isinstance(value, bytes):
            return {"_py": "bytes", "hex": value.hex()}
        if isinstance(value, complex):
            return {
                "_py": "complex",
                "real": repr(value.real),
                "imag": repr(value.imag),
            }
        if value is Ellipsis:
            return {"_py": "ellipsis"}
        return {"_py": type(value).__name__, "repr": repr(value)}

    _ASTWitnessValue = (
        ast.AST
        | JSONValue
        | list[object]
        | tuple[object, ...]
        | dict[object, object]
        | set[object]
        | frozenset[object]
        | bytes
        | complex
        | type(Ellipsis)
    )

    def _normalize_ast_value(value: _ASTWitnessValue) -> JSONValue:
        check_deadline()
        if isinstance(value, ast.AST):
            fields: JSONObject = {}
            for name, raw_field in ast.iter_fields(value):
                check_deadline()
                fields[name] = _normalize_ast_value(raw_field)
            attrs: JSONObject = {}
            for name in getattr(value, "_attributes", ()):
                check_deadline()
                attrs[name] = _normalize_scalar(getattr(value, name, None))
            payload: JSONObject = {
                "_py": "ast",
                "node": type(value).__name__,
                "fields": fields,
            }
            if attrs:
                payload["attrs"] = attrs
            return payload
        if isinstance(value, list):
            return [_normalize_ast_value(item) for item in value]
        if isinstance(value, tuple):
            return {
                "_py": "tuple",
                "items": [_normalize_ast_value(item) for item in value],
            }
        if isinstance(value, dict):
            normalized: JSONObject = {}
            for key in ordered_or_sorted(
                value.keys(),
                source="server._normalize_ast_value.dict_keys",
                key=lambda item: str(item),
                policy=OrderPolicy.SORT,
            ):
                check_deadline()
                normalized[str(key)] = _normalize_ast_value(value[key])
            return normalized
        if isinstance(value, set):
            items = [_normalize_ast_value(item) for item in value]
            items.sort(key=_canonical_json_text)
            return {"_py": "set", "items": items}
        if isinstance(value, frozenset):
            items = [_normalize_ast_value(item) for item in value]
            items.sort(key=_canonical_json_text)
            return {"_py": "frozenset", "items": items}
        return _normalize_scalar(value)

    ast_intern_table: JSONObject = {}

    def _intern_ast(normalized_tree: JSONValue) -> str:
        witness_text = _canonical_json_text(normalized_tree)
        witness_key = hashlib.sha1(witness_text.encode("utf-8")).hexdigest()
        if witness_key not in ast_intern_table:
            ast_intern_table[witness_key] = cast(JSONValue, normalized_tree)
        return witness_key

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
            entry["mtime_ns"] = int(stat.st_mtime_ns)
            try:
                source = read_text_fn(
                    path,
                    io_name="analysis_input_witness.source_read",
                    encoding="utf-8",
                )
            except (OSError, UnicodeError) as exc:
                entry["parse_error"] = {
                    "kind": type(exc).__name__,
                    "message": str(exc),
                }
            else:
                try:
                    tree = parse_source_fn(source, filename=str(path))
                except SyntaxError as exc:
                    entry["parse_error"] = {
                        "kind": type(exc).__name__,
                        "message": str(exc),
                        "lineno": int(exc.lineno or 0),
                        "offset": int(exc.offset or 0),
                        "end_lineno": int(exc.end_lineno or 0),
                        "end_offset": int(exc.end_offset or 0),
                        "text": (exc.text or "").rstrip("\n"),
                    }
                else:
                    normalized_tree = _normalize_ast_value(tree)
                    entry["ast_ref"] = _intern_ast(normalized_tree)
        files.append(entry)
    witness: JSONObject = {
        "format_version": _ANALYSIS_INPUT_WITNESS_FORMAT_VERSION,
        "root": str(root),
        "recursive": recursive,
        "include_invariant_propositions": include_invariant_propositions,
        "include_wl_refinement": include_wl_refinement,
        "config": _analysis_witness_config_payload(config),
        "ast_intern_table": ast_intern_table,
        "files": files,
    }
    witness["witness_digest"] = hashlib.sha1(
        _canonical_json_text(witness).encode("utf-8")
    ).hexdigest()
    return witness


def _canonical_json_text(payload: object) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _load_analysis_resume_checkpoint(
    *,
    path: Path,
    input_witness: JSONObject,
) -> JSONObject | None:
    if not path.exists():
        return None
    try:
        raw_payload = json.loads(
            _read_text_profiled(path, io_name="analysis_resume.checkpoint_read")
        )
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    if not isinstance(raw_payload, dict):
        return None
    if raw_payload.get("format_version") != _ANALYSIS_RESUME_CHECKPOINT_FORMAT_VERSION:
        return None
    expected_digest = input_witness.get("witness_digest")
    if isinstance(expected_digest, str):
        observed_digest = raw_payload.get("input_witness_digest")
        if isinstance(observed_digest, str) and observed_digest != expected_digest:
            return None
    witness = raw_payload.get("input_witness")
    if not isinstance(witness, dict) or witness != input_witness:
        return None
    collection_resume = raw_payload.get("collection_resume")
    if not isinstance(collection_resume, dict):
        return None
    return _inflate_collection_resume_states(
        path=path,
        collection_resume=cast(JSONObject, collection_resume),
    )


def _load_analysis_resume_checkpoint_manifest(
    *,
    path: Path,
    manifest_digest: str,
) -> tuple[JSONObject | None, JSONObject] | None:
    if not path.exists():
        return None
    try:
        raw_payload = json.loads(
            _read_text_profiled(path, io_name="analysis_resume.checkpoint_read")
        )
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    if not isinstance(raw_payload, dict):
        return None
    if raw_payload.get("format_version") != _ANALYSIS_RESUME_CHECKPOINT_FORMAT_VERSION:
        return None
    observed_manifest_digest = raw_payload.get("input_manifest_digest")
    if not isinstance(observed_manifest_digest, str):
        witness = raw_payload.get("input_witness")
        if isinstance(witness, dict):
            observed_manifest_digest = _analysis_manifest_digest_from_witness(witness)
    if not isinstance(observed_manifest_digest, str):
        return None
    if observed_manifest_digest != manifest_digest:
        return None
    collection_resume = raw_payload.get("collection_resume")
    if not isinstance(collection_resume, dict):
        return None
    inflated_collection_resume = _inflate_collection_resume_states(
        path=path,
        collection_resume=cast(JSONObject, collection_resume),
    )
    witness = raw_payload.get("input_witness")
    if isinstance(witness, dict):
        return cast(JSONObject, witness), inflated_collection_resume
    return None, inflated_collection_resume


def _analysis_resume_checkpoint_compatibility(
    *,
    path: Path,
    manifest_digest: str,
) -> str:
    """Return a stable reason code describing checkpoint compatibility."""
    if not path.exists():
        return "checkpoint_missing"
    try:
        raw_payload = json.loads(
            _read_text_profiled(path, io_name="analysis_resume.checkpoint_read")
        )
    except (OSError, UnicodeError, json.JSONDecodeError):
        return "checkpoint_unreadable"
    if not isinstance(raw_payload, dict):
        return "checkpoint_invalid_payload"
    if raw_payload.get("format_version") != _ANALYSIS_RESUME_CHECKPOINT_FORMAT_VERSION:
        return "checkpoint_format_mismatch"
    observed_manifest_digest = raw_payload.get("input_manifest_digest")
    if not isinstance(observed_manifest_digest, str):
        witness = raw_payload.get("input_witness")
        if isinstance(witness, dict):
            observed_manifest_digest = _analysis_manifest_digest_from_witness(witness)
    if not isinstance(observed_manifest_digest, str):
        return "checkpoint_manifest_missing"
    if observed_manifest_digest != manifest_digest:
        return "checkpoint_manifest_mismatch"
    collection_resume = raw_payload.get("collection_resume")
    if not isinstance(collection_resume, dict):
        return "checkpoint_missing_collection_resume"
    return "checkpoint_compatible"


def _write_analysis_resume_checkpoint(
    *,
    path: Path,
    input_witness: JSONObject | None,
    input_manifest_digest: str | None,
    collection_resume: JSONObject,
) -> None:
    witness_digest = (
        input_witness.get("witness_digest") if input_witness is not None else None
    )
    manifest_digest = input_manifest_digest
    if manifest_digest is None and input_witness is not None:
        manifest_digest = _analysis_manifest_digest_from_witness(input_witness)
    if manifest_digest is None:
        never("analysis resume checkpoint missing input manifest digest")
    externalized_collection_resume = _externalize_collection_resume_states(
        path=path,
        collection_resume=collection_resume,
    )
    payload: JSONObject = {
        "format_version": _ANALYSIS_RESUME_CHECKPOINT_FORMAT_VERSION,
        "input_witness": input_witness,
        "input_witness_digest": (
            witness_digest if isinstance(witness_digest, str) else None
        ),
        "input_manifest_digest": manifest_digest,
        "collection_resume": externalized_collection_resume,
    }
    analysis_index_summary = _analysis_index_resume_summary_payload(collection_resume)
    if analysis_index_summary is not None:
        payload["analysis_index_hydration"] = analysis_index_summary
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_text_profiled(
        path,
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        io_name="analysis_resume.checkpoint_write",
    )


def _clear_analysis_resume_checkpoint(path: Path) -> None:
    try:
        path.unlink()
    except OSError:
        pass
    chunks_dir = _analysis_resume_checkpoint_chunks_dir(path)
    if chunks_dir.exists():
        for chunk_path in chunks_dir.glob("*.json"):
            check_deadline()
            try:
                chunk_path.unlink()
            except OSError:
                continue
        try:
            chunks_dir.rmdir()
        except OSError:
            pass


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
    if phase == "collection":
        return "collection_files"
    if phase == "forest":
        return "forest_mutable_steps"
    if phase == "edge":
        return "edge_tasks"
    if phase == "post":
        return "post_tasks"
    return "phase_work_units"


def _build_phase_progress_v2(
    *,
    phase: str,
    collection_progress: Mapping[str, JSONValue],
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

    normalized: JSONObject = {
        "format_version": 1,
        "schema": "gabion/phase_progress_v2",
        "primary_unit": _phase_primary_unit_for_phase(phase),
        "primary_done": normalized_work_done,
        "primary_total": normalized_work_total,
        "dimensions": {
            _phase_primary_unit_for_phase(phase): {
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
        primary_unit = _phase_primary_unit_for_phase(phase)
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
            _canonical_json_text(sorted(processed_functions)).encode("utf-8")
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
            _canonical_json_text(sorted(hydrated)).encode("utf-8")
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


def _analysis_index_resume_summary(
    collection_resume: Mapping[str, JSONValue] | None,
) -> JSONObject | None:
    if not isinstance(collection_resume, Mapping):
        return None
    normalized_resume: JSONObject = {
        str(key): collection_resume[key] for key in collection_resume
    }
    return _analysis_index_resume_summary_payload(normalized_resume)


def _analysis_index_resume_summary_payload(
    collection_resume: JSONObject,
) -> JSONObject | None:
    raw_resume = collection_resume.get("analysis_index_resume")
    if not isinstance(raw_resume, Mapping):
        return None
    (
        hydrated_count,
        hydrated_digest,
        function_count,
        class_count,
        phase,
        resume_digest,
    ) = (
        _analysis_index_resume_signature(collection_resume)
    )
    summary: JSONObject = {
        "phase": phase or "analysis_index_hydration",
        "hydrated_paths_count": hydrated_count,
        "hydrated_paths_digest": hydrated_digest,
        "resume_digest": resume_digest,
        "function_count": function_count,
        "class_count": class_count,
    }
    return summary


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


def _resolve_report_phase_checkpoint_path(
    *,
    root: Path,
    report_path: str | None,
) -> Path | None:
    resolved_report = _resolve_report_output_path(root=root, report_path=report_path)
    if resolved_report is None:
        return None
    default_checkpoint = root / _DEFAULT_REPORT_PHASE_CHECKPOINT
    if resolved_report.name == "dataflow_report.md":
        return default_checkpoint
    return resolved_report.with_name(f"{resolved_report.stem}_phase_checkpoint.json")


def _report_witness_digest(
    *,
    input_witness: Mapping[str, JSONValue] | None,
    manifest_digest: str | None,
) -> str | None:
    if input_witness is not None:
        digest = input_witness.get("witness_digest")
        if isinstance(digest, str) and digest:
            return digest
    if isinstance(manifest_digest, str) and manifest_digest:
        return manifest_digest
    return None


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
        if status != "resolved":
            reason = pending_reasons.get(section_id)
            if reason:
                section_entry["reason"] = reason
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
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        io_name="report_section_journal.write",
    )


def _load_report_phase_checkpoint(
    *,
    path: Path | None,
    witness_digest: str | None,
) -> JSONObject:
    check_deadline()
    if path is None or not path.exists():
        return {}
    try:
        payload = json.loads(
            _read_text_profiled(path, io_name="report_phase_checkpoint.read")
        )
    except (OSError, UnicodeError, json.JSONDecodeError):
        return {}
    if not isinstance(payload, dict):
        return {}
    if payload.get("format_version") != _REPORT_PHASE_CHECKPOINT_FORMAT_VERSION:
        return {}
    expected_digest = payload.get("witness_digest")
    if isinstance(expected_digest, str):
        if not isinstance(witness_digest, str) or witness_digest != expected_digest:
            return {}
    raw_phases = payload.get("phases")
    if not isinstance(raw_phases, Mapping):
        return {}
    phases: JSONObject = {}
    for phase_name, raw_entry in raw_phases.items():
        check_deadline()
        if not isinstance(phase_name, str) or not isinstance(raw_entry, Mapping):
            continue
        phases[phase_name] = {str(key): raw_entry[key] for key in raw_entry}
    return phases


def _write_report_phase_checkpoint(
    *,
    path: Path | None,
    witness_digest: str | None,
    phases: Mapping[str, JSONValue],
) -> None:
    get_deadline()
    if _deadline_tick_budget_allows_check(get_deadline_clock()):
        check_deadline(allow_frame_fallback=False)
    if path is None:
        return
    phases_payload: JSONObject = {}
    for phase_name, raw_entry in phases.items():
        if _deadline_tick_budget_allows_check(get_deadline_clock()):
            check_deadline(allow_frame_fallback=False)
        else:
            get_deadline()
        if not isinstance(phase_name, str) or not isinstance(raw_entry, Mapping):
            continue
        phases_payload[phase_name] = {str(key): raw_entry[key] for key in raw_entry}
    payload: JSONObject = {
        "format_version": _REPORT_PHASE_CHECKPOINT_FORMAT_VERSION,
        "witness_digest": witness_digest,
        "phases": phases_payload,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_text_profiled(
        path,
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        io_name="report_phase_checkpoint.write",
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
                sort_keys=True,
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
        "section_ids": sorted(sections),
    }
    _write_report_phase_checkpoint(
        path=report_phase_checkpoint_path,
        witness_digest=witness_digest,
        phases=phase_checkpoint_state,
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


def _checkpoint_intro_timeline_enabled(payload: Mapping[str, JSONValue]) -> bool:
    raw_enabled = payload.get("emit_checkpoint_intro_timeline")
    if raw_enabled is not None:
        return _truthy_flag(raw_enabled)
    github_actions = os.getenv("GITHUB_ACTIONS", "")
    ci = os.getenv("CI", "")
    return _truthy_flag(github_actions) or _truthy_flag(ci)


def _checkpoint_intro_timeline_path(*, root: Path) -> Path:
    return root / _DEFAULT_CHECKPOINT_INTRO_TIMELINE


def _phase_timeline_md_path(*, root: Path) -> Path:
    return root / _DEFAULT_PHASE_TIMELINE_MD


def _phase_timeline_jsonl_path(*, root: Path) -> Path:
    return root / _DEFAULT_PHASE_TIMELINE_JSONL


def _progress_heartbeat_seconds(payload: Mapping[str, JSONValue]) -> float:
    raw = payload.get("progress_heartbeat_seconds")
    if isinstance(raw, bool):
        return _DEFAULT_PROGRESS_HEARTBEAT_SECONDS
    if isinstance(raw, (int, float)):
        parsed = float(raw)
    elif isinstance(raw, str):
        text = raw.strip()
        if not text:
            return _DEFAULT_PROGRESS_HEARTBEAT_SECONDS
        try:
            parsed = float(text)
        except ValueError:
            return _DEFAULT_PROGRESS_HEARTBEAT_SECONDS
    elif raw is None:
        return _DEFAULT_PROGRESS_HEARTBEAT_SECONDS
    else:
        return _DEFAULT_PROGRESS_HEARTBEAT_SECONDS
    if parsed <= 0:
        return 0.0
    if parsed < _MIN_PROGRESS_HEARTBEAT_SECONDS:
        return _MIN_PROGRESS_HEARTBEAT_SECONDS
    return parsed


def _markdown_table_cell(value: object) -> str:
    if value is None:
        return ""
    return str(value).replace("\n", " ").replace("|", "\\|")


def _phase_progress_dimensions_summary(
    phase_progress_v2: Mapping[str, JSONValue] | None,
) -> str:
    if not isinstance(phase_progress_v2, Mapping):
        return ""
    raw_dimensions = phase_progress_v2.get("dimensions")
    if not isinstance(raw_dimensions, Mapping):
        return ""
    fragments: list[str] = []
    dim_names = sorted(name for name in raw_dimensions if isinstance(name, str))
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


def _phase_progress_primary_summary(
    phase_progress_v2: Mapping[str, JSONValue] | None,
) -> tuple[str, int | None, int | None]:
    if not isinstance(phase_progress_v2, Mapping):
        return "", None, None
    primary_unit = str(phase_progress_v2.get("primary_unit", "") or "")
    raw_done = phase_progress_v2.get("primary_done")
    raw_total = phase_progress_v2.get("primary_total")
    primary_done = (
        max(int(raw_done), 0)
        if isinstance(raw_done, int) and not isinstance(raw_done, bool)
        else None
    )
    primary_total = (
        max(int(raw_total), 0)
        if isinstance(raw_total, int) and not isinstance(raw_total, bool)
        else None
    )
    if (
        primary_done is not None
        and primary_total is not None
        and primary_total > 0
        and primary_done > primary_total
    ):
        primary_done = primary_total
    return primary_unit, primary_done, primary_total


def _resume_checkpoint_descriptor_from_progress_value(
    progress_value: Mapping[str, JSONValue],
) -> str:
    raw_resume = progress_value.get("resume_checkpoint")
    if not isinstance(raw_resume, Mapping):
        return ""
    checkpoint_path = str(raw_resume.get("checkpoint_path", "") or "")
    status = str(raw_resume.get("status", "") or "")
    raw_reused = raw_resume.get("reused_files")
    raw_total = raw_resume.get("total_files")
    reused = (
        max(int(raw_reused), 0)
        if isinstance(raw_reused, int) and not isinstance(raw_reused, bool)
        else None
    )
    total = (
        max(int(raw_total), 0)
        if isinstance(raw_total, int) and not isinstance(raw_total, bool)
        else None
    )
    if reused is None or total is None:
        return (
            f"path={checkpoint_path or '<none>'} "
            f"status={status or 'unknown'} reused_files=unknown"
        )
    return (
        f"path={checkpoint_path or '<none>'} "
        f"status={status or 'unknown'} reused_files={reused}/{total}"
    )


def _append_phase_timeline_event(
    *,
    markdown_path: Path,
    jsonl_path: Path,
    progress_value: Mapping[str, JSONValue],
) -> tuple[str | None, str]:
    header = _phase_timeline_header_columns()
    header_line = "| " + " | ".join(_markdown_table_cell(cell) for cell in header) + " |"
    separator_line = "| " + " | ".join(["---"] * len(header)) + " |"
    ts_utc = str(progress_value.get("ts_utc", "") or "")
    event_seq = progress_value.get("event_seq")
    event_kind = str(progress_value.get("event_kind", "") or "")
    phase = str(progress_value.get("phase", "") or "")
    analysis_state = str(progress_value.get("analysis_state", "") or "")
    classification = str(progress_value.get("classification", "") or "")
    progress_marker = str(progress_value.get("progress_marker", "") or "")
    primary_unit, primary_done, primary_total = _phase_progress_primary_summary(
        progress_value.get("phase_progress_v2")
        if isinstance(progress_value.get("phase_progress_v2"), Mapping)
        else None
    )
    if primary_done is not None and primary_total is not None:
        primary = f"{primary_done}/{primary_total}"
        if primary_unit:
            primary = f"{primary} {primary_unit}"
    elif primary_unit:
        primary = primary_unit
    else:
        primary = ""
    completed_files = progress_value.get("completed_files")
    remaining_files = progress_value.get("remaining_files")
    total_files = progress_value.get("total_files")
    files = ""
    if (
        isinstance(completed_files, int)
        and not isinstance(completed_files, bool)
        and isinstance(remaining_files, int)
        and not isinstance(remaining_files, bool)
        and isinstance(total_files, int)
        and not isinstance(total_files, bool)
    ):
        files = f"{completed_files}/{total_files} rem={remaining_files}"
    resume_checkpoint = _resume_checkpoint_descriptor_from_progress_value(progress_value)
    raw_stale_for_s = progress_value.get("stale_for_s")
    stale_for_s = (
        f"{float(raw_stale_for_s):.1f}"
        if isinstance(raw_stale_for_s, (int, float))
        else ""
    )
    dimensions_summary = _phase_progress_dimensions_summary(
        progress_value.get("phase_progress_v2")
        if isinstance(progress_value.get("phase_progress_v2"), Mapping)
        else None
    )
    row = [
        ts_utc,
        event_seq if isinstance(event_seq, int) else "",
        event_kind,
        phase,
        analysis_state,
        classification,
        progress_marker,
        primary,
        files,
        resume_checkpoint,
        stale_for_s,
        dimensions_summary,
    ]
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    row_line = "| " + " | ".join(_markdown_table_cell(cell) for cell in row) + " |"
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
        handle.write(json.dumps(progress_value, sort_keys=True) + "\n")
    return header_block, row_line


def _phase_timeline_header_columns() -> list[str]:
    return [
        "ts_utc",
        "event_seq",
        "event_kind",
        "phase",
        "analysis_state",
        "classification",
        "progress_marker",
        "primary",
        "files",
        "resume_checkpoint",
        "stale_for_s",
        "dimensions",
    ]


def _phase_timeline_header_block() -> str:
    header = _phase_timeline_header_columns()
    header_line = "| " + " | ".join(_markdown_table_cell(cell) for cell in header) + " |"
    separator_line = "| " + " | ".join(["---"] * len(header)) + " |"
    return header_line + "\n" + separator_line


def _append_checkpoint_intro_timeline_row(
    *,
    path: Path,
    collection_resume: Mapping[str, JSONValue],
    total_files: int,
    checkpoint_path: Path | None,
    checkpoint_status: str | None,
    checkpoint_reused_files: int,
    checkpoint_total_files: int | None = None,
    timestamp_utc: str | None = None,
) -> tuple[str | None, str]:
    progress = _analysis_resume_progress(
        collection_resume=collection_resume,
        total_files=total_files,
    )
    semantic_progress = collection_resume.get("semantic_progress")
    semantic_witness_digest: str | None = None
    new_processed_functions: int | None = None
    regressed_processed_functions: int | None = None
    completed_files_delta: int | None = None
    substantive_progress: bool | None = None
    if isinstance(semantic_progress, Mapping):
        raw_witness_digest = semantic_progress.get("current_witness_digest")
        if isinstance(raw_witness_digest, str) and raw_witness_digest:
            semantic_witness_digest = raw_witness_digest
        raw_new = semantic_progress.get("new_processed_functions_count")
        if not isinstance(raw_new, bool) and isinstance(raw_new, int):
            new_processed_functions = raw_new
        raw_regressed = semantic_progress.get("regressed_processed_functions_count")
        if not isinstance(raw_regressed, bool) and isinstance(raw_regressed, int):
            regressed_processed_functions = raw_regressed
        raw_delta = semantic_progress.get("completed_files_delta")
        if not isinstance(raw_delta, bool) and isinstance(raw_delta, int):
            completed_files_delta = raw_delta
        raw_substantive = semantic_progress.get("substantive_progress")
        if isinstance(raw_substantive, bool):
            substantive_progress = raw_substantive
    analysis_index_summary = _analysis_index_resume_summary(collection_resume)
    hydrated_paths_count: int | None = None
    hydrated_function_count: int | None = None
    hydrated_class_count: int | None = None
    if isinstance(analysis_index_summary, Mapping):
        raw_hydrated = analysis_index_summary.get("hydrated_paths_count")
        if not isinstance(raw_hydrated, bool) and isinstance(raw_hydrated, int):
            hydrated_paths_count = raw_hydrated
        raw_fn_count = analysis_index_summary.get("function_count")
        if not isinstance(raw_fn_count, bool) and isinstance(raw_fn_count, int):
            hydrated_function_count = raw_fn_count
        raw_class_count = analysis_index_summary.get("class_count")
        if not isinstance(raw_class_count, bool) and isinstance(raw_class_count, int):
            hydrated_class_count = raw_class_count
    checkpoint_total = checkpoint_total_files
    if checkpoint_total is None:
        checkpoint_total = progress["total_files"]
    if timestamp_utc is None:
        timestamp_utc = datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
            "+00:00", "Z"
        )
    checkpoint_descriptor = (
        f"path={checkpoint_path or '<none>'} "
        f"status={checkpoint_status or 'unknown'} "
        f"reused_files={checkpoint_reused_files}/{checkpoint_total}"
    )
    header = [
        "ts_utc",
        "completed_files",
        "in_progress_files",
        "remaining_files",
        "total_files",
        "resume_checkpoint",
        "semantic_witness_digest",
        "new_processed_functions",
        "regressed_processed_functions",
        "completed_files_delta",
        "substantive_progress",
        "hydrated_paths_count",
        "hydrated_function_count",
        "hydrated_class_count",
    ]
    row = [
        timestamp_utc,
        progress["completed_files"],
        progress["in_progress_files"],
        progress["remaining_files"],
        progress["total_files"],
        checkpoint_descriptor,
        semantic_witness_digest,
        new_processed_functions,
        regressed_processed_functions,
        completed_files_delta,
        substantive_progress,
        hydrated_paths_count,
        hydrated_function_count,
        hydrated_class_count,
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    header_line = "| " + " | ".join(_markdown_table_cell(cell) for cell in header) + " |"
    separator_line = "| " + " | ".join(["---"] * len(header)) + " |"
    row_line = "| " + " | ".join(_markdown_table_cell(cell) for cell in row) + " |"
    header_block: str | None = None
    if not path.exists():
        with path.open("w", encoding="utf-8") as handle:
            handle.write("# Checkpoint Intro Timeline\n\n")
            handle.write(header_line + "\n")
            handle.write(separator_line + "\n")
        header_block = header_line + "\n" + separator_line
    with path.open("a", encoding="utf-8") as handle:
        handle.write(row_line + "\n")
    return header_block, row_line


def _collection_progress_intro_lines(
    *,
    collection_resume: Mapping[str, JSONValue],
    total_files: int,
    resume_checkpoint_intro: Mapping[str, JSONValue] | None = None,
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
    if isinstance(resume_checkpoint_intro, Mapping):
        checkpoint_path = str(resume_checkpoint_intro.get("checkpoint_path", "") or "")
        status = str(resume_checkpoint_intro.get("status", "") or "")
        reused_files = int(resume_checkpoint_intro.get("reused_files", 0) or 0)
        total_resume_files = int(
            resume_checkpoint_intro.get("total_files", progress["total_files"]) or 0
        )
        lines.append(
            "- `resume_checkpoint`: "
            f"`path={checkpoint_path or '<none>'} "
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
    resume_checkpoint_path: Path | None,
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

    checkpoint_ok = True
    if resume_supported and is_timeout_state:
        checkpoint_ok = bool(resume_checkpoint_path and resume_checkpoint_path.exists())
    obligations.append(
        {
            "status": "SATISFIED" if checkpoint_ok else "VIOLATION",
            "contract": "resume_contract",
            "kind": "checkpoint_present_when_resumable",
            "detail": (
                str(resume_checkpoint_path)
                if resume_checkpoint_path is not None
                else "checkpoint path missing"
            ),
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


def _require_payload(
    payload: Mapping[str, object],
    *,
    command: str,
) -> dict[str, object]:
    if not isinstance(payload, Mapping):
        never(
            "invalid command payload type",
            command=command,
            payload_type=type(payload).__name__,
        )
    if isinstance(payload, dict):
        return payload
    return {str(key): payload[key] for key in payload}


def _require_optional_payload(
    payload: Mapping[str, object] | None,
    *,
    command: str,
) -> dict[str, object]:
    if payload is None:
        never(
            "invalid command payload type",
            command=command,
            payload_type="NoneType",
        )
    return _require_payload(payload, command=command)


def _parse_lint_line(line: str) -> LintEntryDTO | None:
    match = _LINT_RE.match(line.strip())
    if not match:
        return None
    rest = match.group("rest").strip()
    if not rest:
        return None
    code, _, message = rest.partition(" ")
    return LintEntryDTO(
        path=match.group("path"),
        line=int(match.group("line")),
        col=int(match.group("col")),
        code=code,
        message=message,
    )


def _lint_entries_from_lines(lines: Sequence[str]) -> list[dict[str, object]]:
    entries: list[dict[str, object]] = []
    for line in lines:
        check_deadline()
        entry = _parse_lint_line(line)
        if entry is not None:
            entries.append(entry.model_dump())
    return entries


def _normalize_dataflow_response(response: Mapping[str, object]) -> dict[str, object]:
    lint_lines_raw = response.get("lint_lines")
    lint_lines = [str(line) for line in lint_lines_raw] if isinstance(lint_lines_raw, list) else []
    lint_entries_raw = response.get("lint_entries")
    lint_entries = lint_entries_raw if isinstance(lint_entries_raw, list) else _lint_entries_from_lines(lint_lines)
    base = DataflowAuditResponseDTO(
        exit_code=int(response.get("exit_code", 0) or 0),
        timeout=bool(response.get("timeout", False)),
        analysis_state=(str(response.get("analysis_state")) if response.get("analysis_state") is not None else None),
        errors=[str(err) for err in (response.get("errors") or [])] if isinstance(response.get("errors"), list) else [],
        lint_lines=lint_lines,
        lint_entries=[LintEntryDTO.model_validate(entry) for entry in lint_entries],
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
    normalized["lint_entries"] = [entry.model_dump() for entry in base.lint_entries]
    return normalized


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
    timeout_ticks = payload.get("analysis_timeout_ticks")
    timeout_tick_ns = payload.get("analysis_timeout_tick_ns")
    timeout_ms = payload.get("analysis_timeout_ms")
    timeout_seconds = payload.get("analysis_timeout_seconds")
    if timeout_ticks not in (None, ""):
        try:
            ticks_value = int(timeout_ticks)
        except (TypeError, ValueError):
            never("invalid analysis timeout ticks", ticks=timeout_ticks)
        if ticks_value <= 0:
            never("invalid analysis timeout ticks", ticks=timeout_ticks)
        if timeout_tick_ns in (None, ""):
            never("missing analysis timeout tick_ns", ticks=ticks_value)
        try:
            tick_ns_value = int(timeout_tick_ns)
        except (TypeError, ValueError):
            never("invalid analysis timeout tick_ns", tick_ns=timeout_tick_ns)
        if tick_ns_value <= 0:
            never("invalid analysis timeout tick_ns", tick_ns=timeout_tick_ns)
        return ticks_value * tick_ns_value
    if timeout_ms not in (None, ""):
        try:
            ms_value = int(timeout_ms)
        except (TypeError, ValueError):
            never("invalid analysis timeout ms", ms=timeout_ms)
        if ms_value <= 0:
            never("invalid analysis timeout ms", ms=timeout_ms)
        return ms_value * 1_000_000
    if timeout_seconds not in (None, ""):
        # Deprecated: prefer analysis_timeout_ticks / analysis_timeout_tick_ns.
        try:
            seconds_value = Decimal(str(timeout_seconds))
        except (InvalidOperation, ValueError):
            never("invalid analysis timeout seconds", seconds=timeout_seconds)
        if seconds_value <= 0:
            never("invalid analysis timeout seconds", seconds=timeout_seconds)
        timeout_ns = int(seconds_value * Decimal(1_000_000_000))
        # Reject sub-millisecond timeout payloads; they truncate to unstable
        # sub-tick budgets and violate the integer tick contract.
        if timeout_ns < 1_000_000:
            never("invalid analysis timeout seconds", seconds=timeout_seconds)
        return timeout_ns
    never(
        "missing analysis timeout",
        payload_keys=ordered_or_sorted(
            payload.keys(),
            source="server._analysis_timeout_total_ns.payload_keys",
        ),
    )


def _analysis_timeout_total_ticks(payload: dict[str, object]) -> int:
    timeout_ticks = payload.get("analysis_timeout_ticks")
    timeout_ms = payload.get("analysis_timeout_ms")
    timeout_seconds = payload.get("analysis_timeout_seconds")
    if timeout_ticks not in (None, ""):
        try:
            ticks_value = int(timeout_ticks)
        except (TypeError, ValueError):
            never("invalid analysis timeout ticks", ticks=timeout_ticks)
        if ticks_value <= 0:
            never("invalid analysis timeout ticks", ticks=timeout_ticks)
        return ticks_value
    if timeout_ms not in (None, ""):
        try:
            ms_value = int(timeout_ms)
        except (TypeError, ValueError):
            never("invalid analysis timeout ms", ms=timeout_ms)
        if ms_value <= 0:
            never("invalid analysis timeout ms", ms=timeout_ms)
        return ms_value
    if timeout_seconds not in (None, ""):
        try:
            seconds_value = Decimal(str(timeout_seconds))
        except (InvalidOperation, ValueError):
            never("invalid analysis timeout seconds", seconds=timeout_seconds)
        if seconds_value <= 0:
            never("invalid analysis timeout seconds", seconds=timeout_seconds)
        ticks_value = int(seconds_value * Decimal(1000))
        if ticks_value <= 0:
            never("invalid analysis timeout seconds", seconds=timeout_seconds)
        return ticks_value
    never(
        "missing analysis timeout",
        payload_keys=ordered_or_sorted(
            payload.keys(),
            source="server._analysis_timeout_total_ticks.payload_keys",
        ),
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


def _deadline_from_payload(payload: dict[str, object]) -> Deadline:
    total_ns = _analysis_timeout_total_ns(payload)
    overhead_ns = _server_deadline_overhead_ns(total_ns)
    analysis_ns = max(1, total_ns - overhead_ns)
    return Deadline(deadline_ns=time.monotonic_ns() + analysis_ns)


@contextmanager
def _deadline_scope_from_payload(payload: Mapping[str, object]):
    normalized_payload = _require_payload(payload, command="deadline_scope")
    deadline = _deadline_from_payload(normalized_payload)
    base_ticks = _analysis_timeout_total_ticks(normalized_payload)
    tick_limit = base_ticks
    tick_limit_value = normalized_payload.get("analysis_tick_limit")
    if tick_limit_value not in (None, ""):
        try:
            explicit_tick_limit = int(tick_limit_value)
        except (TypeError, ValueError):
            never("invalid analysis tick limit", tick_limit=tick_limit_value)
        if explicit_tick_limit <= 0:
            never("invalid analysis tick limit", tick_limit=tick_limit_value)
        tick_limit = min(tick_limit, explicit_tick_limit)
    logical_clock = GasMeter(limit=tick_limit)
    profile_enabled = _truthy_flag(normalized_payload.get("deadline_profile"))
    root_value = normalized_payload.get("root")
    profile_root = Path(str(root_value)).resolve() if root_value not in (None, "") else None
    with forest_scope(Forest()):
        deadline_token = set_deadline(deadline)
        clock_token = set_deadline_clock(logical_clock)
        profile_token = set_deadline_profile(
            project_root=profile_root,
            enabled=profile_enabled,
            sample_interval=_deadline_profile_sample_interval(
                normalized_payload,
                default_interval=1,
            ),
        )
        try:
            yield
        finally:
            reset_deadline_profile(profile_token)
            reset_deadline_clock(clock_token)
            reset_deadline(deadline_token)


def _output_dirs(report_root: Path) -> tuple[Path, Path]:
    out_dir = report_root / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir = report_root / "artifacts" / "out"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    return out_dir, artifact_dir


def _uri_to_path(uri: str) -> Path:
    parsed = urlparse(uri)
    if parsed.scheme == "file":
        return Path(unquote(parsed.path))
    return Path(uri)


def _normalize_transparent_decorators(value: object) -> set[str] | None:
    check_deadline()
    if value is None:
        return None
    items: list[str] = []
    if isinstance(value, str):
        items = [part.strip() for part in value.split(",") if part.strip()]
    elif isinstance(value, (list, tuple, set)):
        for item in value:
            check_deadline()
            if isinstance(item, str):
                items.extend([part.strip() for part in item.split(",") if part.strip()])
    if not items:
        return None
    return set(items)


def _normalize_name_set(value: object) -> set[str] | None:
    check_deadline()
    if value is None:
        return None
    items: list[str] = []
    if isinstance(value, str):
        items = [part.strip() for part in value.split(",") if part.strip()]
    elif isinstance(value, (list, tuple, set)):
        for item in value:
            check_deadline()
            if isinstance(item, str):
                items.extend([part.strip() for part in item.split(",") if part.strip()])
            else:
                never("name set contains non-string entry", value_type=type(item).__name__)
    else:
        never("invalid name set payload", value_type=type(value).__name__)
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
        exclude_dirs = _normalize_name_set(payload.get("exclude"))
        if exclude_dirs is None:
            exclude_dirs = _normalize_name_set(defaults.get("exclude"))
        if exclude_dirs is None:
            exclude_dirs = set()

        ignore_params = _normalize_name_set(payload.get("ignore_params"))
        if ignore_params is None:
            ignore_params = _normalize_name_set(defaults.get("ignore_params"))
        if ignore_params is None:
            ignore_params = set()

        decision_ignore_params = set(ignore_params)
        decision_ignore_params.update(decision_ignore_list(decision_section))

        transparent_payload = payload.get("transparent_decorators")
        transparent_decorators = _normalize_transparent_decorators(transparent_payload)
        if transparent_decorators is None and transparent_payload is None:
            transparent_decorators = _normalize_transparent_decorators(
                defaults.get("transparent_decorators")
            )

        return cls(
            exclude_dirs=exclude_dirs,
            ignore_params=ignore_params,
            decision_ignore_params=decision_ignore_params,
            transparent_decorators=transparent_decorators,
        )


@dataclass(frozen=True)
class SnapshotDiffPayload:
    baseline: Path
    current: Path

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, object],
        *,
        error_message: str,
    ) -> "SnapshotDiffPayload":
        baseline = payload.get("baseline")
        current = payload.get("current")
        if not baseline or not current:
            raise ValueError(error_message)
        return cls(baseline=Path(str(baseline)), current=Path(str(current)))


def _diagnostics_for_path(
    path_str: str,
    project_root: Path | None,
    *,
    analyze_paths_fn: Callable[..., AnalysisResult] = analyze_paths,
) -> list[Diagnostic]:
    forest = Forest()
    with forest_scope(forest):
        check_deadline()
        result = analyze_paths_fn(
            [Path(path_str)],
            forest=forest,
            recursive=True,
            type_audit=False,
            type_audit_report=False,
            type_audit_max=0,
            include_constant_smells=False,
            include_unused_arg_smells=False,
            config=AuditConfig(project_root=project_root),
        )
    diagnostics: list[Diagnostic] = []
    for path, bundles in result.groups_by_path.items():
        check_deadline()
        span_map = result.param_spans_by_path.get(path, {})
        for fn_name, group_list in bundles.items():
            check_deadline()
            param_spans = span_map.get(fn_name, {})
            for bundle in group_list:
                check_deadline()
                ordered_bundle = ordered_or_sorted(
                    bundle,
                    source="server._lint_bundles.bundle",
                    key=str,
                )
                ordered_bundle = ordered_or_sorted(
                    ordered_bundle,
                    source="server._lint_bundles.bundle_enforce",
                    policy=OrderPolicy.ENFORCE,
                )
                message = f"Implicit bundle detected: {', '.join(ordered_bundle)}"
                for name in ordered_or_sorted(
                    ordered_bundle,
                    source="server._lint_bundles.bundle_loop",
                    policy=OrderPolicy.ENFORCE,
                ):
                    check_deadline()
                    span = param_spans.get(name)
                    if span is None:  # pragma: no cover - spans are derived from parsed params
                        start = Position(line=0, character=0)  # pragma: no cover
                        end = Position(line=0, character=1)  # pragma: no cover
                    else:
                        start_line, start_col, end_line, end_col = span
                        start = Position(line=start_line, character=start_col)
                        end = Position(line=end_line, character=end_col)
                    diagnostics.append(
                        Diagnostic(
                            range=Range(start=start, end=end),
                            message=message,
                            severity=DiagnosticSeverity.Information,
                            source="gabion",
                        )
                    )
    return diagnostics


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
        analyze_paths_fn=analyze_paths,
        load_analysis_resume_checkpoint_manifest_fn=_load_analysis_resume_checkpoint_manifest,
        write_analysis_resume_checkpoint_fn=_write_analysis_resume_checkpoint,
        load_analysis_resume_checkpoint_fn=_load_analysis_resume_checkpoint,
        analysis_input_manifest_fn=_analysis_input_manifest,
        analysis_input_manifest_digest_fn=_analysis_input_manifest_digest,
        build_analysis_collection_resume_seed_fn=build_analysis_collection_resume_seed,
        collection_semantic_progress_fn=_collection_semantic_progress,
        load_report_phase_checkpoint_fn=_load_report_phase_checkpoint,
        project_report_sections_fn=project_report_sections,
        report_projection_spec_rows_fn=report_projection_spec_rows,
        collection_checkpoint_flush_due_fn=_collection_checkpoint_flush_due,
        write_bootstrap_incremental_artifacts_fn=_write_bootstrap_incremental_artifacts,
        load_report_section_journal_fn=_load_report_section_journal,
    )


@server.command(DATAFLOW_COMMAND)
def execute_command(
    ls: LanguageServer,
    payload: dict | None = None,
) -> dict:
    normalized_payload = _require_optional_payload(payload, command=DATAFLOW_COMMAND)
    return _execute_command_total(ls, normalized_payload)


def execute_command_with_deps(
    ls: LanguageServer,
    payload: dict | None = None,
    *,
    deps: ExecuteCommandDeps | None = None,
) -> dict:
    normalized_payload = _require_optional_payload(payload, command=DATAFLOW_COMMAND)
    return _execute_command_total(ls, normalized_payload, deps=deps)


def _execute_command_total(
    ls: LanguageServer,
    payload: dict[str, object],
    *,
    deps: ExecuteCommandDeps | None = None,
) -> dict:
    execute_deps = deps or _default_execute_command_deps()
    execution_plan = _materialize_execution_plan(payload)
    payload = dict(execution_plan.inputs)
    write_execution_plan_artifact(
        execution_plan,
        root=Path(str(payload.get("root") or ls.workspace.root_path or ".")),
    )
    profile_enabled = _truthy_flag(payload.get("deadline_profile"))
    profile_root_value = payload.get("root") or ls.workspace.root_path or "."
    initial_root = Path(str(profile_root_value))
    initial_report_path = payload.get("report")
    initial_report_path_text = (
        str(initial_report_path) if isinstance(initial_report_path, str) else None
    )
    timeout_total_ns, analysis_window_ns, cleanup_grace_ns = _analysis_timeout_budget_ns(
        payload
    )
    timeout_total_ticks = _analysis_timeout_total_ticks(payload)
    explicit_tick_limit_value = payload.get("analysis_tick_limit")
    if explicit_tick_limit_value not in (None, ""):
        try:
            explicit_tick_limit = int(explicit_tick_limit_value)
        except (TypeError, ValueError):
            never("invalid analysis tick limit", tick_limit=explicit_tick_limit_value)
        if explicit_tick_limit <= 0:
            never("invalid analysis tick limit", tick_limit=explicit_tick_limit_value)
        timeout_total_ticks = min(timeout_total_ticks, explicit_tick_limit)
    timeout_start_ns = time.monotonic_ns()
    timeout_hard_deadline_ns = timeout_start_ns + timeout_total_ns
    analysis_deadline_ns = timeout_start_ns + analysis_window_ns
    deadline_token = set_deadline(Deadline(deadline_ns=analysis_deadline_ns))
    deadline_clock_token = set_deadline_clock(GasMeter(limit=timeout_total_ticks))
    profile_token = set_deadline_profile(
        project_root=initial_root,
        enabled=profile_enabled,
        sample_interval=_deadline_profile_sample_interval(payload),
    )
    forest = Forest()
    forest_token = set_forest(forest)
    explicit_resume_checkpoint = payload.get("resume_checkpoint") not in (None, False, "")
    analysis_resume_checkpoint_path: Path | None = _resolve_analysis_resume_checkpoint_path(
        payload.get("resume_checkpoint"),
        root=initial_root,
    )
    analysis_resume_input_witness: JSONObject | None = None
    analysis_resume_input_manifest_digest: str | None = None
    analysis_resume_total_files = 0
    analysis_resume_reused_files = 0
    analysis_resume_checkpoint_status: str | None = None
    analysis_resume_checkpoint_compatibility_status: str | None = None
    analysis_resume_intro_payload: JSONObject | None = None
    analysis_resume_intro_timeline_header: str | None = None
    analysis_resume_intro_timeline_row: str | None = None
    report_section_witness_digest: str | None = None
    report_output_path = _resolve_report_output_path(
        root=initial_root,
        report_path=initial_report_path_text,
    )
    report_section_journal_path = _resolve_report_section_journal_path(
        root=initial_root,
        report_path=initial_report_path_text,
    )
    report_phase_checkpoint_path = _resolve_report_phase_checkpoint_path(
        root=initial_root,
        report_path=initial_report_path_text,
    )
    projection_rows: list[JSONObject] = (
        execute_deps.report_projection_spec_rows_fn() if report_output_path else []
    )
    enable_phase_projection_checkpoints = False
    phase_checkpoint_state: JSONObject = {}
    last_collection_resume_payload: JSONObject | None = None
    report_sections_cache: dict[str, list[str]] = {}
    report_sections_cache_reason: str | None = None
    report_sections_cache_loaded = False
    semantic_progress_cumulative: JSONObject | None = None
    latest_collection_progress: JSONObject = {
        "completed_files": 0,
        "in_progress_files": 0,
        "remaining_files": 0,
        "total_files": analysis_resume_total_files,
    }

    def _emit_lsp_progress(**_kwargs: object) -> None:
        return

    def _ensure_report_sections_cache() -> tuple[dict[str, list[str]], str | None]:
        nonlocal report_sections_cache
        nonlocal report_sections_cache_reason
        nonlocal report_sections_cache_loaded
        if not report_sections_cache_loaded:
            report_sections_cache, report_sections_cache_reason = (
                execute_deps.load_report_section_journal_fn(
                path=report_section_journal_path,
                witness_digest=report_section_witness_digest,
                )
            )
            report_sections_cache_loaded = True
        return report_sections_cache, report_sections_cache_reason

    raw_initial_paths = payload.get("paths")
    initial_paths_count = len(raw_initial_paths) if isinstance(raw_initial_paths, list) else 1
    execute_deps.write_bootstrap_incremental_artifacts_fn(
        report_output_path=report_output_path,
        report_section_journal_path=report_section_journal_path,
        report_phase_checkpoint_path=report_phase_checkpoint_path,
        witness_digest=report_section_witness_digest,
        root=initial_root,
        paths_requested=initial_paths_count,
        projection_rows=projection_rows,
        phase_checkpoint_state=phase_checkpoint_state,
    )
    try:
        root = payload.get("root") or ls.workspace.root_path or "."
        config_path = payload.get("config")
        defaults = dataflow_defaults(
            Path(root), Path(config_path) if config_path else None
        )
        deadline_roots = set(dataflow_deadline_roots(defaults))
        decision_section = decision_defaults(
            Path(root), Path(config_path) if config_path else None
        )
        decision_tiers = decision_tier_map(decision_section)
        decision_require = decision_require_tiers(decision_section)
        exception_section = exception_defaults(
            Path(root), Path(config_path) if config_path else None
        )
        never_exceptions = set(exception_never_list(exception_section))
        fingerprint_section = fingerprint_defaults(
            Path(root), Path(config_path) if config_path else None
        )
        synth_min_occurrences = 0
        synth_version = "synth@1"
        try:
            synth_min_occurrences = int(
                fingerprint_section.get("synth_min_occurrences", 0) or 0
            )
        except (TypeError, ValueError):
            synth_min_occurrences = 0
        synth_version = str(
            fingerprint_section.get("synth_version", synth_version) or synth_version
        )
        fingerprint_registry: PrimeRegistry | None = None
        fingerprint_index: dict[Fingerprint, set[str]] = {}
        constructor_registry: TypeConstructorRegistry | None = None
        fingerprint_spec: dict[str, JSONValue] = {
            key: value
            for key, value in fingerprint_section.items()
            if not str(key).startswith("synth_")
        }
        if fingerprint_spec:
            registry, index = build_fingerprint_registry(fingerprint_spec)
            if index:
                fingerprint_registry = registry
                fingerprint_index = index
                constructor_registry = TypeConstructorRegistry(registry)
        payload = merge_payload(payload, defaults)
        deadline_roots = set(payload.get("deadline_roots", deadline_roots))

        raw_paths = payload.get("paths") or []
        if raw_paths:
            paths = [Path(p) for p in raw_paths]
        else:
            paths = [Path(root)]
        root = payload.get("root") or root
        report_path = payload.get("report")
        report_path_text = str(report_path) if isinstance(report_path, str) else None
        report_output_path = _resolve_report_output_path(
            root=Path(root),
            report_path=report_path_text,
        )
        report_section_journal_path = _resolve_report_section_journal_path(
            root=Path(root),
            report_path=report_path_text,
        )
        report_phase_checkpoint_path = _resolve_report_phase_checkpoint_path(
            root=Path(root),
            report_path=report_path_text,
        )
        projection_rows = (
            execute_deps.report_projection_spec_rows_fn() if report_output_path else []
        )
        emit_timeout_progress_report = _truthy_flag(
            payload.get("emit_timeout_progress_report")
        )
        resume_on_timeout_raw = payload.get("resume_on_timeout")
        try:
            resume_on_timeout_attempts = int(resume_on_timeout_raw or 0)
        except (TypeError, ValueError):
            resume_on_timeout_attempts = 0
        enable_phase_projection_checkpoints = bool(report_output_path) and (
            emit_timeout_progress_report or resume_on_timeout_attempts > 0
        )
        emit_checkpoint_intro_timeline = _checkpoint_intro_timeline_enabled(payload)
        checkpoint_intro_timeline_path = _checkpoint_intro_timeline_path(root=Path(root))
        phase_timeline_markdown_path = _phase_timeline_md_path(root=Path(root))
        phase_timeline_jsonl_path = _phase_timeline_jsonl_path(root=Path(root))
        progress_heartbeat_seconds = _progress_heartbeat_seconds(payload)
        dot_path = payload.get("dot")
        fail_on_violations = payload.get("fail_on_violations", False)
        no_recursive = payload.get("no_recursive", False)
        max_components = payload.get("max_components", 10)
        type_audit = payload.get("type_audit", False)
        type_audit_report = payload.get("type_audit_report", False)
        type_audit_max = payload.get("type_audit_max", 50)
        fail_on_type_ambiguities = payload.get("fail_on_type_ambiguities", False)
        lint = bool(payload.get("lint", False))
        name_filter_bundle = DataflowNameFilterBundle.from_payload(
            payload=payload,
            defaults=defaults,
            decision_section=decision_section,
        )
        allow_external = payload.get("allow_external", False)
        strictness = payload.get("strictness", "high")
        if strictness not in {"high", "low"}:
            never("invalid strictness", strictness=str(strictness))
        baseline_path = resolve_baseline_path(payload.get("baseline"), Path(root))
        baseline_write = bool(payload.get("baseline_write", False)) and baseline_path is not None
        synthesis_plan_path = payload.get("synthesis_plan")
        synthesis_report = payload.get("synthesis_report", False)
        structure_tree_path = payload.get("structure_tree")
        structure_metrics_path = payload.get("structure_metrics")
        decision_snapshot_path = payload.get("decision_snapshot")
        emit_test_obsolescence = bool(payload.get("emit_test_obsolescence", False))
        emit_test_obsolescence_state = bool(
            payload.get("emit_test_obsolescence_state", False)
        )
        test_obsolescence_state_path = payload.get("test_obsolescence_state")
        emit_test_obsolescence_delta = bool(
            payload.get("emit_test_obsolescence_delta", False)
        )
        write_test_obsolescence_baseline = bool(
            payload.get("write_test_obsolescence_baseline", False)
        )
        emit_test_evidence_suggestions = bool(
            payload.get("emit_test_evidence_suggestions", False)
        )
        emit_call_clusters = bool(payload.get("emit_call_clusters", False))
        emit_call_cluster_consolidation = bool(
            payload.get("emit_call_cluster_consolidation", False)
        )
        emit_test_annotation_drift = bool(
            payload.get("emit_test_annotation_drift", False)
        )
        emit_semantic_coverage_map = bool(
            payload.get("emit_semantic_coverage_map", False)
        )
        test_annotation_drift_state_path = payload.get("test_annotation_drift_state")
        semantic_coverage_mapping_path = payload.get("semantic_coverage_mapping")
        emit_test_annotation_drift_delta = bool(
            payload.get("emit_test_annotation_drift_delta", False)
        )
        write_test_annotation_drift_baseline = bool(
            payload.get("write_test_annotation_drift_baseline", False)
        )
        emit_ambiguity_delta = bool(payload.get("emit_ambiguity_delta", False))
        emit_ambiguity_state = bool(payload.get("emit_ambiguity_state", False))
        ambiguity_state_path = payload.get("ambiguity_state")
        write_ambiguity_baseline = bool(payload.get("write_ambiguity_baseline", False))
        synthesis_max_tier = payload.get("synthesis_max_tier", 2)
        synthesis_min_bundle_size = payload.get("synthesis_min_bundle_size", 2)
        synthesis_allow_singletons = payload.get("synthesis_allow_singletons", False)
        synthesis_protocols_path = payload.get("synthesis_protocols")
        synthesis_protocols_kind = payload.get("synthesis_protocols_kind", "dataclass")
        refactor_plan = payload.get("refactor_plan", False)
        refactor_plan_json = payload.get("refactor_plan_json")
        fingerprint_synth_json = payload.get("fingerprint_synth_json")
        fingerprint_provenance_json = payload.get("fingerprint_provenance_json")
        fingerprint_deadness_json = payload.get("fingerprint_deadness_json")
        fingerprint_coherence_json = payload.get("fingerprint_coherence_json")
        fingerprint_rewrite_plans_json = payload.get("fingerprint_rewrite_plans_json")
        fingerprint_exception_obligations_json = payload.get(
            "fingerprint_exception_obligations_json"
        )
        fingerprint_handledness_json = payload.get("fingerprint_handledness_json")

        config = AuditConfig(
            project_root=Path(root),
            exclude_dirs=name_filter_bundle.exclude_dirs,
            ignore_params=name_filter_bundle.ignore_params,
            decision_ignore_params=name_filter_bundle.decision_ignore_params,
            external_filter=not allow_external,
            strictness=strictness,
            transparent_decorators=name_filter_bundle.transparent_decorators,
            decision_tiers=decision_tiers,
            decision_require_tiers=decision_require,
            never_exceptions=never_exceptions,
            deadline_roots=deadline_roots,
            fingerprint_registry=fingerprint_registry,
            fingerprint_index=fingerprint_index,
            constructor_registry=constructor_registry,
            fingerprint_synth_min_occurrences=synth_min_occurrences,
            fingerprint_synth_version=synth_version,
        )
        if fail_on_type_ambiguities:
            type_audit = True
        include_decisions = bool(report_path) or bool(decision_snapshot_path) or bool(
            fail_on_violations
        )
        if decision_tiers:
            include_decisions = True
        include_rewrite_plans = bool(report_path) or bool(fingerprint_rewrite_plans_json)
        include_exception_obligations = bool(report_path) or bool(
            fingerprint_exception_obligations_json
        )
        include_handledness_witnesses = bool(report_path) or bool(
            fingerprint_handledness_json
        )
        include_never_invariants = bool(report_path)
        include_wl_refinement = _truthy_flag(payload.get("include_wl_refinement"))
        include_ambiguities = bool(report_path) or lint or emit_ambiguity_state
        if (emit_ambiguity_delta or write_ambiguity_baseline) and not ambiguity_state_path:
            include_ambiguities = True
        include_coherence = (
            bool(report_path) or bool(fingerprint_coherence_json) or include_rewrite_plans
        )
        needs_analysis = (
            bool(report_path)
            or bool(dot_path)
            or bool(structure_tree_path)
            or bool(structure_metrics_path)
            or bool(decision_snapshot_path)
            or bool(synthesis_plan_path)
            or bool(synthesis_report)
            or bool(synthesis_protocols_path)
            or bool(refactor_plan)
            or bool(refactor_plan_json)
            or bool(fingerprint_synth_json)
            or bool(fingerprint_provenance_json)
            or bool(fingerprint_deadness_json)
            or bool(fingerprint_coherence_json)
            or bool(fingerprint_rewrite_plans_json)
            or bool(fingerprint_exception_obligations_json)
            or bool(fingerprint_handledness_json)
            or bool(type_audit)
            or bool(type_audit_report)
            or bool(fail_on_type_ambiguities)
            or bool(fail_on_violations)
            or baseline_path is not None
            or bool(lint)
            or bool(emit_test_evidence_suggestions)
            or bool(include_ambiguities)
        )
        file_paths_for_run: list[Path] | None = None
        collection_resume_payload: JSONObject | None = None
        if needs_analysis:
            file_paths_for_run = resolve_analysis_paths(paths, config=config)
            analysis_resume_total_files = len(file_paths_for_run)
            analysis_resume_checkpoint_path = _resolve_analysis_resume_checkpoint_path(
                payload.get("resume_checkpoint"),
                root=Path(root),
            )
            if analysis_resume_checkpoint_path is not None:
                input_manifest = execute_deps.analysis_input_manifest_fn(
                    root=Path(root),
                    file_paths=file_paths_for_run,
                    recursive=not no_recursive,
                    include_invariant_propositions=bool(report_path),
                    include_wl_refinement=include_wl_refinement,
                    config=config,
                )
                analysis_resume_input_manifest_digest = (
                    execute_deps.analysis_input_manifest_digest_fn(
                    input_manifest
                    )
                )
                manifest_resume = execute_deps.load_analysis_resume_checkpoint_manifest_fn(
                    path=analysis_resume_checkpoint_path,
                    manifest_digest=analysis_resume_input_manifest_digest,
                )
                analysis_resume_checkpoint_status = (
                    _analysis_resume_checkpoint_compatibility(
                        path=analysis_resume_checkpoint_path,
                        manifest_digest=analysis_resume_input_manifest_digest,
                    )
                )
                analysis_resume_checkpoint_compatibility_status = (
                    analysis_resume_checkpoint_status
                )
                if manifest_resume is not None:
                    checkpoint_witness, checkpoint_resume = manifest_resume
                    if checkpoint_witness is not None:
                        analysis_resume_input_witness = checkpoint_witness
                    collection_resume_payload = checkpoint_resume
                    analysis_resume_reused_files = _analysis_resume_progress(
                            collection_resume=checkpoint_resume,
                            total_files=analysis_resume_total_files,
                        )["completed_files"]
                    analysis_resume_checkpoint_status = "checkpoint_loaded"
                if (
                    collection_resume_payload is None
                    and analysis_resume_input_manifest_digest is not None
                ):
                    seed_paths = file_paths_for_run[:1] if file_paths_for_run else []
                    collection_resume_payload = (
                        execute_deps.build_analysis_collection_resume_seed_fn(
                        in_progress_paths=seed_paths
                    )
                    )
                    execute_deps.write_analysis_resume_checkpoint_fn(
                        path=analysis_resume_checkpoint_path,
                        input_witness=analysis_resume_input_witness,
                        input_manifest_digest=analysis_resume_input_manifest_digest,
                        collection_resume=collection_resume_payload,
                    )
                    analysis_resume_checkpoint_status = "checkpoint_seeded"
                    if emit_checkpoint_intro_timeline:
                        (
                            analysis_resume_intro_timeline_header,
                            analysis_resume_intro_timeline_row,
                        ) = _append_checkpoint_intro_timeline_row(
                            path=checkpoint_intro_timeline_path,
                            collection_resume=collection_resume_payload,
                            total_files=analysis_resume_total_files,
                            checkpoint_path=analysis_resume_checkpoint_path,
                            checkpoint_status=analysis_resume_checkpoint_status,
                            checkpoint_reused_files=analysis_resume_reused_files,
                            checkpoint_total_files=analysis_resume_total_files,
                        )
                if explicit_resume_checkpoint:
                    analysis_resume_intro_payload = {
                        "checkpoint_path": str(analysis_resume_checkpoint_path),
                        "status": analysis_resume_checkpoint_status or "unknown",
                        "reused_files": analysis_resume_reused_files,
                        "total_files": analysis_resume_total_files,
                    }
            if file_paths_for_run is not None and analysis_resume_input_manifest_digest is None:
                input_manifest = execute_deps.analysis_input_manifest_fn(
                    root=Path(root),
                    file_paths=file_paths_for_run,
                    recursive=not no_recursive,
                    include_invariant_propositions=bool(report_path),
                    include_wl_refinement=include_wl_refinement,
                    config=config,
                )
                analysis_resume_input_manifest_digest = (
                    execute_deps.analysis_input_manifest_digest_fn(input_manifest)
                )
            report_section_witness_digest = _report_witness_digest(
                input_witness=analysis_resume_input_witness,
                manifest_digest=analysis_resume_input_manifest_digest,
            )
            if report_output_path is not None:
                phase_checkpoint_state = execute_deps.load_report_phase_checkpoint_fn(
                    path=report_phase_checkpoint_path,
                    witness_digest=report_section_witness_digest,
                )
        last_collection_resume_payload = collection_resume_payload
        if isinstance(collection_resume_payload, Mapping):
            raw_semantic_progress = collection_resume_payload.get("semantic_progress")
            if isinstance(raw_semantic_progress, Mapping):
                semantic_progress_cumulative = {
                    str(key): raw_semantic_progress[key]
                    for key in raw_semantic_progress
                }
        last_collection_intro_signature: tuple[int, int, int, int] | None = None
        last_collection_semantic_witness_digest: str | None = None
        last_collection_checkpoint_flush_ns = 0
        last_analysis_index_resume_signature: tuple[
            int, str, int, int, str, str
        ] | None = None
        last_collection_report_flush_ns = 0
        last_collection_report_flush_completed = -1
        phase_progress_signatures: dict[str, tuple[object, ...]] = {}
        phase_progress_last_flush_ns: dict[str, int] = {}
        profiling_stage_ns: dict[str, int] = {
            "server.analysis_call": 0,
            "server.projection_emit": 0,
        }
        profiling_counters: dict[str, int] = {
            "server.collection_resume_persist_calls": 0,
            "server.projection_emit_calls": 0,
        }
        progress_event_seq = 0
        progress_state_lock = threading.Lock()
        last_progress_template: dict[str, object] | None = None
        last_progress_change_ns = 0
        last_progress_notification_ns = 0
        phase_timeline_header_emitted = False
        heartbeat_stop_event = threading.Event()
        heartbeat_thread: threading.Thread | None = None
        send_notification = getattr(ls, "send_notification", None)

        def _emit_lsp_progress(
            *,
            phase: Literal["collection", "forest", "edge", "post"],
            collection_progress: Mapping[str, JSONValue],
            semantic_progress: Mapping[str, JSONValue] | None,
            work_done: int | None = None,
            work_total: int | None = None,
            include_timing: bool = False,
            done: bool = False,
            analysis_state: str | None = None,
            classification: str | None = None,
            resume_checkpoint: Mapping[str, JSONValue] | None = None,
            checkpoint_intro_timeline_header: str | None = None,
            checkpoint_intro_timeline_row: str | None = None,
            phase_progress_v2: Mapping[str, JSONValue] | None = None,
            progress_marker: str | None = None,
            event_kind: Literal["progress", "heartbeat", "terminal", "checkpoint"] = "progress",
            stale_for_s: float | None = None,
            record_for_heartbeat: bool = True,
        ) -> None:
            nonlocal progress_event_seq
            nonlocal last_progress_template
            nonlocal last_progress_change_ns
            nonlocal last_progress_notification_ns
            nonlocal phase_timeline_header_emitted
            semantic_payload: JSONObject = {}
            if isinstance(semantic_progress, Mapping):
                for raw_key, raw_value in semantic_progress.items():
                    if not isinstance(raw_key, str):
                        continue
                    if raw_key == "substantive_progress" or raw_key.startswith(
                        "cumulative_"
                    ):
                        semantic_payload[raw_key] = raw_value
            if "substantive_progress" not in semantic_payload:
                semantic_payload["substantive_progress"] = False
            progress_value: JSONObject = {
                "format_version": 1,
                "schema": "gabion/dataflow_progress_v1",
                "phase": phase,
                "completed_files": int(collection_progress.get("completed_files", 0)),
                "in_progress_files": int(
                    collection_progress.get("in_progress_files", 0)
                ),
                "remaining_files": int(collection_progress.get("remaining_files", 0)),
                "total_files": int(collection_progress.get("total_files", 0)),
                "semantic_deltas": semantic_payload,
            }
            normalized_phase_progress_v2, primary_done, primary_total = _build_phase_progress_v2(
                phase=phase,
                collection_progress=collection_progress,
                work_done=work_done,
                work_total=work_total,
                phase_progress_v2=phase_progress_v2,
            )
            progress_value["phase_progress_v2"] = normalized_phase_progress_v2
            progress_value["work_done"] = primary_done
            progress_value["work_total"] = primary_total
            progress_value["telemetry_semantics_version"] = 2
            progress_value["event_kind"] = event_kind
            progress_value["ts_utc"] = datetime.now(timezone.utc).isoformat(
                timespec="seconds"
            ).replace("+00:00", "Z")
            if isinstance(progress_marker, str) and progress_marker:
                progress_value["progress_marker"] = progress_marker
            if isinstance(stale_for_s, (int, float)):
                progress_value["stale_for_s"] = max(float(stale_for_s), 0.0)
            if include_timing:
                progress_value["profiling_v1"] = {
                    "format_version": 1,
                    "server": {
                        "stage_ns": dict(profiling_stage_ns),
                        "counters": dict(profiling_counters),
                    },
                }
            if done:
                progress_value["done"] = True
            if isinstance(analysis_state, str) and analysis_state:
                progress_value["analysis_state"] = analysis_state
            if isinstance(classification, str) and classification:
                progress_value["classification"] = classification
            if isinstance(resume_checkpoint, Mapping):
                progress_value["resume_checkpoint"] = {
                    str(key): resume_checkpoint[key] for key in resume_checkpoint
                }
            if (
                isinstance(checkpoint_intro_timeline_header, str)
                and checkpoint_intro_timeline_header
            ):
                progress_value["checkpoint_intro_timeline_header"] = (
                    checkpoint_intro_timeline_header
                )
            if (
                isinstance(checkpoint_intro_timeline_row, str)
                and checkpoint_intro_timeline_row
            ):
                progress_value["checkpoint_intro_timeline_row"] = (
                    checkpoint_intro_timeline_row
                )
            if not callable(send_notification):
                return
            now_ns = time.monotonic_ns()
            with progress_state_lock:
                progress_event_seq += 1
                progress_value["event_seq"] = progress_event_seq
                phase_timeline_header, phase_timeline_row = _append_phase_timeline_event(
                    markdown_path=phase_timeline_markdown_path,
                    jsonl_path=phase_timeline_jsonl_path,
                    progress_value=progress_value,
                )
                if not phase_timeline_header_emitted:
                    progress_value["phase_timeline_header"] = (
                        phase_timeline_header
                        if isinstance(phase_timeline_header, str) and phase_timeline_header
                        else _phase_timeline_header_block()
                    )
                    phase_timeline_header_emitted = True
                progress_value["phase_timeline_row"] = phase_timeline_row
                send_notification(
                    _LSP_PROGRESS_NOTIFICATION_METHOD,
                    {
                        "token": _LSP_PROGRESS_TOKEN,
                        "value": progress_value,
                    },
                )
                last_progress_notification_ns = now_ns
                if done:
                    last_progress_template = None
                elif record_for_heartbeat and event_kind != "heartbeat":
                    last_progress_template = {
                        "phase": phase,
                        "collection_progress": {
                            str(key): collection_progress[key] for key in collection_progress
                        },
                        "semantic_progress": (
                            {str(key): semantic_progress[key] for key in semantic_progress}
                            if isinstance(semantic_progress, Mapping)
                            else None
                        ),
                        "work_done": primary_done,
                        "work_total": primary_total,
                        "include_timing": include_timing,
                        "analysis_state": analysis_state,
                        "classification": classification,
                        "resume_checkpoint": (
                            {str(key): resume_checkpoint[key] for key in resume_checkpoint}
                            if isinstance(resume_checkpoint, Mapping)
                            else None
                        ),
                        "checkpoint_intro_timeline_header": checkpoint_intro_timeline_header,
                        "checkpoint_intro_timeline_row": checkpoint_intro_timeline_row,
                        "phase_progress_v2": {
                            str(key): normalized_phase_progress_v2[key]
                            for key in normalized_phase_progress_v2
                        },
                        "progress_marker": progress_marker,
                    }
                    last_progress_change_ns = now_ns

        def _progress_heartbeat_loop() -> None:
            heartbeat_interval_ns = int(progress_heartbeat_seconds * 1_000_000_000)
            while not heartbeat_stop_event.wait(1.0):
                with progress_state_lock:
                    template = (
                        dict(last_progress_template)
                        if isinstance(last_progress_template, dict)
                        else None
                    )
                    notification_ns = int(last_progress_notification_ns)
                    change_ns = int(last_progress_change_ns)
                if template and notification_ns > 0 and change_ns > 0:
                    now_ns = time.monotonic_ns()
                    if now_ns - notification_ns < heartbeat_interval_ns:
                        continue
                    stale_seconds = max(0.0, (now_ns - change_ns) / 1_000_000_000.0)
                    _emit_lsp_progress(
                        phase=cast(
                            Literal["collection", "forest", "edge", "post"],
                            str(template.get("phase", "collection")),
                        ),
                        collection_progress=cast(
                            Mapping[str, JSONValue],
                            template.get("collection_progress", {}),
                        ),
                        semantic_progress=cast(
                            Mapping[str, JSONValue] | None,
                            template.get("semantic_progress"),
                        ),
                        work_done=cast(int | None, template.get("work_done")),
                        work_total=cast(int | None, template.get("work_total")),
                        include_timing=bool(template.get("include_timing", False)),
                        done=False,
                        analysis_state=cast(str | None, template.get("analysis_state")),
                        classification=cast(str | None, template.get("classification")),
                        resume_checkpoint=cast(
                            Mapping[str, JSONValue] | None,
                            template.get("resume_checkpoint"),
                        ),
                        checkpoint_intro_timeline_header=cast(
                            str | None,
                            template.get("checkpoint_intro_timeline_header"),
                        ),
                        checkpoint_intro_timeline_row=cast(
                            str | None,
                            template.get("checkpoint_intro_timeline_row"),
                        ),
                        phase_progress_v2=cast(
                            Mapping[str, JSONValue] | None,
                            template.get("phase_progress_v2"),
                        ),
                        progress_marker=cast(str | None, template.get("progress_marker")),
                        event_kind="heartbeat",
                        stale_for_s=stale_seconds,
                        record_for_heartbeat=False,
                    )

        if callable(send_notification) and progress_heartbeat_seconds > 0:
            heartbeat_thread = threading.Thread(
                target=_progress_heartbeat_loop,
                name="gabion-progress-heartbeat",
                daemon=True,
            )
            heartbeat_thread.start()

        if analysis_resume_intro_payload is not None and callable(
            getattr(ls, "send_notification", None)
        ):
            _emit_lsp_progress(
                phase="collection",
                collection_progress={
                    "completed_files": analysis_resume_reused_files,
                    "in_progress_files": 0,
                    "remaining_files": max(
                        analysis_resume_total_files - analysis_resume_reused_files,
                        0,
                    ),
                    "total_files": analysis_resume_total_files,
                },
                semantic_progress=None,
                work_done=analysis_resume_reused_files,
                work_total=analysis_resume_total_files,
                include_timing=profile_enabled,
                analysis_state="analysis_collection_bootstrap",
                classification="resume_checkpoint_detected",
                resume_checkpoint=analysis_resume_intro_payload,
                checkpoint_intro_timeline_header=analysis_resume_intro_timeline_header,
                checkpoint_intro_timeline_row=analysis_resume_intro_timeline_row,
                event_kind="checkpoint",
            )

        def _persist_collection_resume(progress_payload: JSONObject) -> None:
            profiling_counters["server.collection_resume_persist_calls"] += 1
            nonlocal last_collection_resume_payload
            nonlocal semantic_progress_cumulative
            nonlocal last_collection_intro_signature
            nonlocal last_collection_semantic_witness_digest
            nonlocal last_collection_checkpoint_flush_ns
            nonlocal last_analysis_index_resume_signature
            nonlocal last_collection_report_flush_ns
            nonlocal last_collection_report_flush_completed
            nonlocal report_sections_cache_reason
            nonlocal latest_collection_progress
            semantic_progress = execute_deps.collection_semantic_progress_fn(
                previous_collection_resume=last_collection_resume_payload,
                collection_resume=progress_payload,
                total_files=analysis_resume_total_files,
                cumulative=semantic_progress_cumulative,
            )
            semantic_progress_cumulative = semantic_progress
            persisted_progress_payload: JSONObject = {
                str(key): progress_payload[key] for key in progress_payload
            }
            persisted_progress_payload["semantic_progress"] = semantic_progress
            last_collection_resume_payload = persisted_progress_payload
            collection_progress = _analysis_resume_progress(
                collection_resume=persisted_progress_payload,
                total_files=analysis_resume_total_files,
            )
            latest_collection_progress = dict(collection_progress)
            _emit_lsp_progress(
                phase="collection",
                collection_progress=collection_progress,
                semantic_progress=semantic_progress,
                work_done=collection_progress.get("completed_files"),
                work_total=collection_progress.get("total_files"),
                include_timing=profile_enabled,
            )
            collection_intro_signature = (
                collection_progress["completed_files"],
                collection_progress["in_progress_files"],
                collection_progress["remaining_files"],
                _analysis_index_resume_hydrated_count(persisted_progress_payload),
            )
            semantic_witness_digest = semantic_progress.get("current_witness_digest")
            if not isinstance(semantic_witness_digest, str):
                semantic_witness_digest = None
            analysis_index_signature = _analysis_index_resume_signature(
                persisted_progress_payload
            )
            intro_changed = collection_intro_signature != last_collection_intro_signature
            semantic_changed = (
                semantic_witness_digest != last_collection_semantic_witness_digest
            )
            analysis_index_changed = (
                analysis_index_signature != last_analysis_index_resume_signature
            )
            raw_substantive_progress = semantic_progress.get("substantive_progress")
            semantic_substantive_progress = (
                raw_substantive_progress
                if isinstance(raw_substantive_progress, bool)
                else False
            )
            now_ns = time.monotonic_ns()
            if (
                analysis_resume_checkpoint_path is not None
                and analysis_resume_input_manifest_digest is not None
                and (intro_changed or semantic_changed or analysis_index_changed)
            ):
                if execute_deps.collection_checkpoint_flush_due_fn(
                    intro_changed=intro_changed,
                    remaining_files=collection_progress["remaining_files"],
                    semantic_substantive_progress=semantic_substantive_progress,
                    now_ns=now_ns,
                    last_flush_ns=last_collection_checkpoint_flush_ns,
                ):
                    execute_deps.write_analysis_resume_checkpoint_fn(
                        path=analysis_resume_checkpoint_path,
                        input_witness=analysis_resume_input_witness,
                        input_manifest_digest=analysis_resume_input_manifest_digest,
                        collection_resume=persisted_progress_payload,
                    )
                    last_collection_checkpoint_flush_ns = now_ns
                    checkpoint_timeline_header: str | None = None
                    checkpoint_timeline_row: str | None = None
                    if emit_checkpoint_intro_timeline:
                        (
                            checkpoint_timeline_header,
                            checkpoint_timeline_row,
                        ) = _append_checkpoint_intro_timeline_row(
                            path=checkpoint_intro_timeline_path,
                            collection_resume=persisted_progress_payload,
                            total_files=analysis_resume_total_files,
                            checkpoint_path=analysis_resume_checkpoint_path,
                            checkpoint_status=analysis_resume_checkpoint_status,
                            checkpoint_reused_files=analysis_resume_reused_files,
                            checkpoint_total_files=analysis_resume_total_files,
                        )
                    if checkpoint_timeline_row is not None:
                        _emit_lsp_progress(
                            phase="collection",
                            collection_progress=collection_progress,
                            semantic_progress=semantic_progress,
                            work_done=collection_progress.get("completed_files"),
                            work_total=collection_progress.get("total_files"),
                            include_timing=profile_enabled,
                            analysis_state="analysis_collection_checkpoint_serialized",
                            classification="checkpoint_serialized",
                            resume_checkpoint=analysis_resume_intro_payload,
                            checkpoint_intro_timeline_header=checkpoint_timeline_header,
                            checkpoint_intro_timeline_row=checkpoint_timeline_row,
                            event_kind="checkpoint",
                        )
            last_collection_semantic_witness_digest = semantic_witness_digest
            last_analysis_index_resume_signature = analysis_index_signature
            if not intro_changed:
                return
            last_collection_intro_signature = collection_intro_signature
            if not report_output_path or not projection_rows:
                return
            completed_files = collection_progress["completed_files"]
            if not _collection_report_flush_due(
                completed_files=completed_files,
                remaining_files=collection_progress["remaining_files"],
                now_ns=now_ns,
                last_flush_ns=last_collection_report_flush_ns,
                last_flush_completed=last_collection_report_flush_completed,
            ):
                return
            last_collection_report_flush_ns = now_ns
            last_collection_report_flush_completed = completed_files
            sections, journal_reason = _ensure_report_sections_cache()
            sections["intro"] = _collection_progress_intro_lines(
                collection_resume=persisted_progress_payload,
                total_files=analysis_resume_total_files,
                resume_checkpoint_intro=analysis_resume_intro_payload,
            )
            if enable_phase_projection_checkpoints:
                preview_groups_by_path = _groups_by_path_from_collection_resume(
                    persisted_progress_payload
                )
                preview_report = ReportCarrier(
                    forest=forest,
                    parse_failure_witnesses=[],
                )
                preview_sections = execute_deps.project_report_sections_fn(
                    preview_groups_by_path,
                    preview_report,
                    max_phase="post",
                    include_previews=True,
                    preview_only=True,
                )
                sections.update(preview_sections)
            sections.setdefault(
                "components",
                _collection_components_preview_lines(
                    collection_resume=persisted_progress_payload,
                ),
            )
            partial_report, pending_reasons = _render_incremental_report(
                analysis_state="analysis_collection_in_progress",
                progress_payload=persisted_progress_payload,
                projection_rows=projection_rows,
                sections=sections,
            )
            pending_reasons.pop("intro", None)
            _apply_journal_pending_reason(
                projection_rows=projection_rows,
                sections=sections,
                pending_reasons=pending_reasons,
                journal_reason=journal_reason,
            )
            report_output_path.parent.mkdir(parents=True, exist_ok=True)
            _write_text_profiled(
                report_output_path,
                partial_report,
                io_name="report_markdown.write",
            )
            _write_report_section_journal(
                path=report_section_journal_path,
                witness_digest=report_section_witness_digest,
                projection_rows=projection_rows,
                sections=sections,
                pending_reasons=pending_reasons,
            )
            report_sections_cache_reason = None
            phase_checkpoint_state["collection"] = {
                "status": "checkpointed",
                "work_done": collection_progress["completed_files"],
                "work_total": collection_progress["total_files"],
                "completed_files": collection_progress["completed_files"],
                "in_progress_files": collection_progress["in_progress_files"],
                "remaining_files": collection_progress["remaining_files"],
                "total_files": collection_progress["total_files"],
                "section_ids": sorted(sections),
            }
            _write_report_phase_checkpoint(
                path=report_phase_checkpoint_path,
                witness_digest=report_section_witness_digest,
                phases=phase_checkpoint_state,
            )

        def _projection_phase_signature(
            phase: Literal["collection", "forest", "edge", "post"],
            groups_by_path: Mapping[Path, dict[str, list[set[str]]]],
            report_carrier: ReportCarrier,
        ) -> tuple[int, ...]:
            check_deadline()
            return (
                len(groups_by_path),
                len(report_carrier.forest.nodes),
                len(report_carrier.forest.alts),
                len(report_carrier.bundle_sites_by_path),
                len(report_carrier.type_suggestions),
                len(report_carrier.type_ambiguities),
                len(report_carrier.type_callsite_evidence),
                len(report_carrier.constant_smells),
                len(report_carrier.unused_arg_smells),
                len(report_carrier.deadness_witnesses),
                len(report_carrier.coherence_witnesses),
                len(report_carrier.rewrite_plans),
                len(report_carrier.exception_obligations),
                len(report_carrier.never_invariants),
                len(report_carrier.ambiguity_witnesses),
                len(report_carrier.handledness_witnesses),
                len(report_carrier.decision_surfaces),
                len(report_carrier.value_decision_surfaces),
                len(report_carrier.decision_warnings),
                len(report_carrier.fingerprint_warnings),
                len(report_carrier.fingerprint_matches),
                len(report_carrier.fingerprint_synth),
                len(report_carrier.fingerprint_provenance),
                len(report_carrier.context_suggestions),
                len(report_carrier.invariant_propositions),
                len(report_carrier.value_decision_rewrites),
                len(report_carrier.deadline_obligations),
                len(report_carrier.parse_failure_witnesses),
                report_projection_phase_rank(phase),
            )

        def _persist_projection_phase(
            phase: Literal["collection", "forest", "edge", "post"],
            groups_by_path: dict[Path, dict[str, list[set[str]]]],
            report_carrier: ReportCarrier,
            work_done: int,
            work_total: int,
        ) -> None:
            nonlocal report_sections_cache_reason
            progress_marker = str(getattr(report_carrier, "progress_marker", "") or "")
            progress_analysis_state = f"analysis_{phase}_in_progress"
            phase_progress_v2 = (
                report_carrier.phase_progress_v2
                if isinstance(report_carrier.phase_progress_v2, Mapping)
                else None
            )
            _emit_lsp_progress(
                phase=phase,
                collection_progress=latest_collection_progress,
                semantic_progress=semantic_progress_cumulative,
                work_done=work_done,
                work_total=work_total,
                include_timing=profile_enabled,
                analysis_state=progress_analysis_state,
                phase_progress_v2=phase_progress_v2,
                progress_marker=progress_marker,
            )
            if not report_output_path or not projection_rows:
                return
            projection_started_ns = time.monotonic_ns()
            phase_signature = _projection_phase_signature(
                phase,
                groups_by_path,
                report_carrier,
            ) + (int(work_done), int(work_total), progress_marker)
            if phase_progress_signatures.get(phase) == phase_signature:
                return
            phase_progress_signatures[phase] = phase_signature
            now_ns = time.monotonic_ns()
            last_flush_ns = phase_progress_last_flush_ns.get(phase, 0)
            if not _projection_phase_flush_due(
                phase=phase,
                now_ns=now_ns,
                last_flush_ns=last_flush_ns,
            ):
                return
            phase_progress_last_flush_ns[phase] = now_ns
            available_sections = execute_deps.project_report_sections_fn(
                groups_by_path,
                report_carrier,
                max_phase="post",
                include_previews=True,
                preview_only=True,
            )
            sections, journal_reason = _ensure_report_sections_cache()
            sections.update(available_sections)
            partial_report, pending_reasons = _render_incremental_report(
                analysis_state=progress_analysis_state,
                progress_payload={
                    "phase": phase,
                    "work_done": int(work_done),
                    "work_total": int(work_total),
                    "event_kind": "progress",
                    "progress_marker": progress_marker,
                    "phase_progress_v2": (
                        {
                            str(key): phase_progress_v2[key]
                            for key in phase_progress_v2
                        }
                        if isinstance(phase_progress_v2, Mapping)
                        else None
                    ),
                },
                projection_rows=projection_rows,
                sections=sections,
            )
            _apply_journal_pending_reason(
                projection_rows=projection_rows,
                sections=sections,
                pending_reasons=pending_reasons,
                journal_reason=journal_reason,
            )
            report_output_path.parent.mkdir(parents=True, exist_ok=True)
            _write_text_profiled(
                report_output_path,
                partial_report,
                io_name="report_markdown.write",
            )
            _write_report_section_journal(
                path=report_section_journal_path,
                witness_digest=report_section_witness_digest,
                projection_rows=projection_rows,
                sections=sections,
                pending_reasons=pending_reasons,
            )
            report_sections_cache_reason = None
            phase_checkpoint_state[phase] = {
                "status": "checkpointed",
                "work_done": int(work_done),
                "work_total": int(work_total),
                "section_ids": sorted(sections),
                "resolved_sections": len(sections),
            }
            _write_report_phase_checkpoint(
                path=report_phase_checkpoint_path,
                witness_digest=report_section_witness_digest,
                phases=phase_checkpoint_state,
            )
            profiling_stage_ns["server.projection_emit"] += (
                time.monotonic_ns() - projection_started_ns
            )
            profiling_counters["server.projection_emit_calls"] += 1

        if needs_analysis and file_paths_for_run is not None:
            bootstrap_collection_resume = collection_resume_payload
            if bootstrap_collection_resume is None:
                seed_paths = file_paths_for_run[:1] if file_paths_for_run else []
                bootstrap_collection_resume = (
                    execute_deps.build_analysis_collection_resume_seed_fn(
                    in_progress_paths=seed_paths
                )
                )
            _persist_collection_resume(bootstrap_collection_resume)

        if needs_analysis:
            analysis_started_ns = time.monotonic_ns()
            analysis = execute_deps.analyze_paths_fn(
                paths,
                forest=forest,
                recursive=not no_recursive,
                type_audit=type_audit or type_audit_report,
                type_audit_report=type_audit_report,
                type_audit_max=type_audit_max,
                include_constant_smells=bool(report_path),
                include_unused_arg_smells=bool(report_path),
                include_deadness_witnesses=bool(report_path)
                or bool(fingerprint_deadness_json),
                include_coherence_witnesses=include_coherence,
                include_rewrite_plans=include_rewrite_plans,
                include_exception_obligations=include_exception_obligations,
                include_handledness_witnesses=include_handledness_witnesses,
                include_never_invariants=include_never_invariants,
                include_wl_refinement=include_wl_refinement,
                include_deadline_obligations=bool(report_path) or lint,
                include_decision_surfaces=include_decisions,
                include_value_decision_surfaces=include_decisions,
                include_invariant_propositions=bool(report_path),
                include_lint_lines=lint,
                include_ambiguities=include_ambiguities,
                include_bundle_forest=True,
                config=config,
                file_paths_override=file_paths_for_run,
                collection_resume=collection_resume_payload,
                on_collection_progress=_persist_collection_resume,
                on_phase_progress=(
                    _persist_projection_phase
                    if enable_phase_projection_checkpoints
                    else None
                ),
            )
            profiling_stage_ns["server.analysis_call"] += (
                time.monotonic_ns() - analysis_started_ns
            )
        else:
            analysis = AnalysisResult(
                groups_by_path={},
                param_spans_by_path={},
                bundle_sites_by_path={},
                type_suggestions=[],
                type_ambiguities=[],
                type_callsite_evidence=[],
                constant_smells=[],
                unused_arg_smells=[],
                forest=forest,
            )
        response: dict = {
            "type_suggestions": analysis.type_suggestions,
            "type_ambiguities": analysis.type_ambiguities,
            "type_callsite_evidence": analysis.type_callsite_evidence,
            "unused_arg_smells": analysis.unused_arg_smells,
            "decision_surfaces": analysis.decision_surfaces,
            "value_decision_surfaces": analysis.value_decision_surfaces,
            "value_decision_rewrites": analysis.value_decision_rewrites,
            "decision_warnings": analysis.decision_warnings,
            "fingerprint_warnings": analysis.fingerprint_warnings,
            "fingerprint_matches": analysis.fingerprint_matches,
            "fingerprint_synth": analysis.fingerprint_synth,
            "fingerprint_synth_registry": analysis.fingerprint_synth_registry,
            "fingerprint_provenance": analysis.fingerprint_provenance,
            "fingerprint_deadness": analysis.deadness_witnesses,
            "fingerprint_coherence": analysis.coherence_witnesses,
            "fingerprint_rewrite_plans": analysis.rewrite_plans,
            "fingerprint_exception_obligations": analysis.exception_obligations,
            "fingerprint_handledness": analysis.handledness_witnesses,
            "never_invariants": analysis.never_invariants,
            "deadline_obligations": analysis.deadline_obligations,
            "ambiguity_witnesses": analysis.ambiguity_witnesses,
            "invariant_propositions": [
                prop.as_dict() for prop in analysis.invariant_propositions
            ],
            "context_suggestions": analysis.context_suggestions,
        }
        if analysis_resume_checkpoint_path is not None:
            cache_verdict = _analysis_resume_cache_verdict(
                status=analysis_resume_checkpoint_status,
                reused_files=analysis_resume_reused_files,
                compatibility_status=analysis_resume_checkpoint_compatibility_status,
            )
            response["analysis_resume"] = {
                "checkpoint_path": str(analysis_resume_checkpoint_path),
                "reused_files": analysis_resume_reused_files,
                "total_files": analysis_resume_total_files,
                "remaining_files": max(
                    analysis_resume_total_files - analysis_resume_reused_files, 0
                ),
                "status": analysis_resume_checkpoint_status,
                "compatibility_status": analysis_resume_checkpoint_compatibility_status,
                "cache_verdict": cache_verdict,
            }
        profiling_v1: JSONObject = {
            "format_version": 1,
            "server": {
                "stage_ns": profiling_stage_ns,
                "counters": profiling_counters,
            },
        }
        if isinstance(analysis.profiling_v1, Mapping):
            profiling_v1["analysis"] = {
                str(key): analysis.profiling_v1[key] for key in analysis.profiling_v1
            }
        response["profiling_v1"] = profiling_v1
        if lint:
            response["lint_lines"] = analysis.lint_lines

        synthesis_plan: JSONObject | None = None
        if synthesis_plan_path or synthesis_report or synthesis_protocols_path:
            try:
                synthesis_plan = build_synthesis_plan(
                    analysis.groups_by_path,
                    project_root=Path(root),
                    max_tier=synthesis_max_tier,
                    min_bundle_size=synthesis_min_bundle_size,
                    allow_singletons=synthesis_allow_singletons,
                    merge_overlap_threshold=payload.get("merge_overlap_threshold", None),
                    config=config,
                )
            except (TypeError, ValueError, OSError) as exc:
                response.setdefault("synthesis_errors", []).append(str(exc))
            if synthesis_plan is not None:
                if synthesis_plan_path:
                    output = json.dumps(synthesis_plan, indent=2, sort_keys=True)
                    if _is_stdout_target(synthesis_plan_path):
                        response["synthesis_plan"] = synthesis_plan
                    else:
                        Path(synthesis_plan_path).write_text(output)
                if synthesis_report:
                    response["synthesis_plan"] = synthesis_plan
        if synthesis_protocols_path and synthesis_plan is not None:
            output = render_protocol_stubs(
                synthesis_plan,
                kind=synthesis_protocols_kind,
            )
            if _is_stdout_target(synthesis_protocols_path):
                response["synthesis_protocols"] = output
            else:
                Path(synthesis_protocols_path).write_text(output)
        if refactor_plan or refactor_plan_json:
            plan_payload = build_refactor_plan(
                analysis.groups_by_path,
                paths,
                config=config,
            )
            if refactor_plan_json:
                if _is_stdout_target(refactor_plan_json):
                    response["refactor_plan"] = plan_payload
                else:
                    Path(refactor_plan_json).write_text(
                        json.dumps(plan_payload, indent=2, sort_keys=True)
                    )
            if refactor_plan:
                response["refactor_plan"] = plan_payload

        if decision_snapshot_path is not None:
            payload_value = render_decision_snapshot(
                surfaces=DecisionSnapshotSurfaces(
                    decision_surfaces=analysis.decision_surfaces,
                    value_decision_surfaces=analysis.value_decision_surfaces,
                ),
                forest=analysis.forest,
                project_root=Path(root),
                groups_by_path=analysis.groups_by_path,
            )
            if _is_stdout_target(decision_snapshot_path):
                response["decision_snapshot"] = payload_value
            else:
                Path(decision_snapshot_path).write_text(
                    json.dumps(payload_value, indent=2, sort_keys=True)
                )

        if structure_tree_path is not None:
            payload_value = render_structure_snapshot(
                analysis.groups_by_path,
                forest=analysis.forest,
                project_root=Path(root),
                invariant_propositions=analysis.invariant_propositions,
            )
            if _is_stdout_target(structure_tree_path):
                response["structure_tree"] = payload_value
            else:
                Path(structure_tree_path).write_text(
                    json.dumps(payload_value, indent=2, sort_keys=True)
                )

        if structure_metrics_path is not None:
            metrics = compute_structure_metrics(
                analysis.groups_by_path,
                forest=analysis.forest,
            )
            if _is_stdout_target(structure_metrics_path):
                response["structure_metrics"] = metrics
            else:
                Path(structure_metrics_path).write_text(
                    json.dumps(metrics, indent=2, sort_keys=True)
                )

        if emit_test_obsolescence_delta and write_test_obsolescence_baseline:
            never(
                "conflicting obsolescence flags",
                emit_test_obsolescence_delta=emit_test_obsolescence_delta,
                write_test_obsolescence_baseline=write_test_obsolescence_baseline,
            )
        if emit_test_annotation_drift_delta and write_test_annotation_drift_baseline:
            never(
                "conflicting annotation drift flags",
                emit_test_annotation_drift_delta=emit_test_annotation_drift_delta,
                write_test_annotation_drift_baseline=write_test_annotation_drift_baseline,
            )
        if emit_ambiguity_delta and write_ambiguity_baseline:
            never(
                "conflicting ambiguity flags",
                emit_ambiguity_delta=emit_ambiguity_delta,
                write_ambiguity_baseline=write_ambiguity_baseline,
            )
        if emit_test_obsolescence_state and test_obsolescence_state_path:
            never(
                "conflicting obsolescence state flags",
                emit_test_obsolescence_state=emit_test_obsolescence_state,
                test_obsolescence_state_path=test_obsolescence_state_path,
            )
        if emit_ambiguity_state and ambiguity_state_path:
            never(
                "conflicting ambiguity state flags",
                emit_ambiguity_state=emit_ambiguity_state,
                ambiguity_state_path=ambiguity_state_path,
            )

        if emit_test_evidence_suggestions:
            report_root = Path(root)
            evidence_path = report_root / "out" / "test_evidence.json"
            entries = test_evidence_suggestions.load_test_evidence(str(evidence_path))
            suggestions, summary = test_evidence_suggestions.suggest_evidence(
                entries,
                root=report_root,
                paths=paths,
                forest=analysis.forest,
                config=config,
            )
            suggestions_payload = test_evidence_suggestions.render_json_payload(
                suggestions, summary
            )
            report_md = test_evidence_suggestions.render_markdown(suggestions, summary)
            out_dir, artifact_dir = _output_dirs(report_root)
            report_json = json.dumps(suggestions_payload, indent=2, sort_keys=True) + "\n"
            (artifact_dir / "test_evidence_suggestions.json").write_text(report_json)
            (out_dir / "test_evidence_suggestions.md").write_text(report_md)
            response["test_evidence_suggestions_summary"] = (
                suggestions_payload.get("summary", {})
            )
        if emit_call_clusters:
            report_root = Path(root)
            evidence_path = report_root / "out" / "test_evidence.json"
            clusters_payload = call_clusters.build_call_clusters_payload(
                paths,
                root=report_root,
                evidence_path=evidence_path,
                config=config,
            )
            report_md = call_clusters.render_markdown(clusters_payload)
            out_dir, artifact_dir = _output_dirs(report_root)
            report_json = json.dumps(clusters_payload, indent=2, sort_keys=True) + "\n"
            (artifact_dir / "call_clusters.json").write_text(report_json)
            (out_dir / "call_clusters.md").write_text(report_md)
            response["call_clusters_summary"] = clusters_payload.get("summary", {})
        if emit_call_cluster_consolidation:
            report_root = Path(root)
            evidence_path = report_root / "out" / "test_evidence.json"
            consolidation_payload = (
                call_cluster_consolidation.build_call_cluster_consolidation_payload(
                    evidence_path=evidence_path,
                )
            )
            report_md = call_cluster_consolidation.render_markdown(consolidation_payload)
            out_dir, artifact_dir = _output_dirs(report_root)
            report_json = (
                json.dumps(consolidation_payload, indent=2, sort_keys=True) + "\n"
            )
            (artifact_dir / "call_cluster_consolidation.json").write_text(report_json)
            (out_dir / "call_cluster_consolidation.md").write_text(report_md)
            response["call_cluster_consolidation_summary"] = consolidation_payload.get(
                "summary", {}
            )
        drift_payload = None
        if test_annotation_drift_state_path:
            state_path = Path(test_annotation_drift_state_path)
            if not state_path.exists():
                never("annotation drift state not found", path=str(state_path))
            payload_value = json.loads(state_path.read_text(encoding="utf-8"))
            if not isinstance(payload_value, dict):
                never("annotation drift state must be a JSON object")
            drift_payload = payload_value
        elif (
            emit_test_annotation_drift
            or emit_test_annotation_drift_delta
            or write_test_annotation_drift_baseline
        ):
            report_root = Path(root)
            evidence_path = report_root / "out" / "test_evidence.json"
            drift_payload = test_annotation_drift.build_annotation_drift_payload(
                paths,
                root=report_root,
                evidence_path=evidence_path,
            )
        if drift_payload is not None and (
            emit_test_annotation_drift
            or emit_test_annotation_drift_delta
            or write_test_annotation_drift_baseline
        ):
            report_root = Path(root)
            out_dir, artifact_dir = _output_dirs(report_root)
            if emit_test_annotation_drift:
                report_json = json.dumps(drift_payload, indent=2, sort_keys=True) + "\n"
                report_md = test_annotation_drift.render_markdown(drift_payload)
                (artifact_dir / "test_annotation_drift.json").write_text(report_json)
                (out_dir / "test_annotation_drift.md").write_text(report_md)
                response["test_annotation_drift_summary"] = drift_payload.get(
                    "summary", {}
                )
            if emit_test_annotation_drift_delta or write_test_annotation_drift_baseline:
                summary = drift_payload.get("summary", {})
                baseline_payload = test_annotation_drift_delta.build_baseline_payload(
                    summary if isinstance(summary, dict) else {}
                )
                baseline_path = test_annotation_drift_delta.resolve_baseline_path(
                    report_root
                )
                response["test_annotation_drift_baseline_path"] = str(baseline_path)
                if write_test_annotation_drift_baseline:
                    baseline_path.parent.mkdir(parents=True, exist_ok=True)
                    test_annotation_drift_delta.write_baseline(
                        str(baseline_path), baseline_payload
                    )
                    response["test_annotation_drift_baseline_written"] = True
                else:
                    response["test_annotation_drift_baseline_written"] = False
                if emit_test_annotation_drift_delta:
                    if not baseline_path.exists():
                        never("annotation drift baseline not found", path=str(baseline_path))
                    baseline = test_annotation_drift_delta.load_baseline(
                        str(baseline_path)
                    )
                    current = test_annotation_drift_delta.parse_baseline_payload(
                        baseline_payload
                    )
                    delta_payload = test_annotation_drift_delta.build_delta_payload(
                        baseline, current, baseline_path=str(baseline_path)
                    )
                    report_json = json.dumps(delta_payload, indent=2, sort_keys=True) + "\n"
                    report_md = test_annotation_drift_delta.render_markdown(delta_payload)
                    (artifact_dir / "test_annotation_drift_delta.json").write_text(
                        report_json
                    )
                    (out_dir / "test_annotation_drift_delta.md").write_text(report_md)
                    response["test_annotation_drift_delta_summary"] = delta_payload.get(
                        "summary", {}
                    )
        obsolescence_candidates: list[dict[str, object]] | None = None
        obsolescence_summary: dict[str, int] | None = None
        obsolescence_baseline_payload: dict[str, object] | None = None
        obsolescence_baseline: test_obsolescence_delta.ObsolescenceBaseline | None = None
        if test_obsolescence_state_path:
            state_path = Path(test_obsolescence_state_path)
            if not state_path.exists():
                never("test obsolescence state not found", path=str(state_path))
            state = test_obsolescence_state.load_state(str(state_path))
            obsolescence_candidates = [
                {str(k): entry[k] for k in entry} for entry in state.candidates
            ]
            obsolescence_summary = state.baseline.summary
            obsolescence_baseline_payload = {
                str(k): state.baseline_payload[k] for k in state.baseline_payload
            }
            obsolescence_baseline = state.baseline
        elif (
            emit_test_obsolescence
            or emit_test_obsolescence_delta
            or write_test_obsolescence_baseline
            or emit_test_obsolescence_state
        ):
            report_root = Path(root)
            evidence_path = report_root / "out" / "test_evidence.json"
            risk_registry_path = report_root / "out" / "evidence_risk_registry.json"
            evidence_by_test, status_by_test = test_obsolescence.load_test_evidence(
                str(evidence_path)
            )
            risk_registry = test_obsolescence.load_risk_registry(str(risk_registry_path))
            candidates, summary_counts = test_obsolescence.classify_candidates(
                evidence_by_test, status_by_test, risk_registry
            )
            obsolescence_candidates = candidates
            obsolescence_summary = summary_counts
            obsolescence_baseline_payload = test_obsolescence_delta.build_baseline_payload(
                evidence_by_test, status_by_test, candidates, summary_counts
            )
            obsolescence_baseline = test_obsolescence_delta.parse_baseline_payload(
                obsolescence_baseline_payload
            )
            if emit_test_obsolescence_state:
                out_dir, artifact_dir = _output_dirs(report_root)
                state_payload = test_obsolescence_state.build_state_payload(
                    evidence_by_test,
                    status_by_test,
                    candidates,
                    summary_counts,
                )
                (artifact_dir / "test_obsolescence_state.json").write_text(
                    json.dumps(state_payload, indent=2, sort_keys=True) + "\n"
                )

        if emit_semantic_coverage_map:
            report_root = Path(root)
            _out_dir, artifact_dir = _output_dirs(report_root)
            mapping_path = (
                Path(str(semantic_coverage_mapping_path))
                if semantic_coverage_mapping_path
                else report_root / "out" / "semantic_coverage_mapping.json"
            )
            evidence_path = report_root / "out" / "test_evidence.json"
            semantic_payload = semantic_coverage_map.build_semantic_coverage_payload(
                paths=paths,
                root=Path(root),
                mapping_path=mapping_path,
                evidence_path=evidence_path,
                exclude=name_filter_bundle.exclude_dirs,
            )
            report_md = semantic_coverage_map.render_markdown(semantic_payload)
            semantic_coverage_map.write_semantic_coverage(
                semantic_payload,
                output_path=artifact_dir / "semantic_coverage_map.json",
            )
            (report_root / "artifacts" / "audit_reports").mkdir(parents=True, exist_ok=True)
            (report_root / "artifacts" / "audit_reports" / "semantic_coverage_map.md").write_text(report_md)
            response["semantic_coverage_map_summary"] = semantic_payload.get("summary", {})

        if emit_test_obsolescence and obsolescence_candidates is not None:
            report_root = Path(root)
            report_payload = test_obsolescence.render_json_payload(
                obsolescence_candidates, obsolescence_summary or {}
            )
            out_dir, artifact_dir = _output_dirs(report_root)
            report_json = json.dumps(report_payload, indent=2, sort_keys=True) + "\n"
            report_md = test_obsolescence.render_markdown(
                obsolescence_candidates, obsolescence_summary or {}
            )
            (artifact_dir / "test_obsolescence_report.json").write_text(report_json)
            (out_dir / "test_obsolescence_report.md").write_text(report_md)
            response["test_obsolescence_summary"] = obsolescence_summary or {}

        if (
            emit_test_obsolescence_delta or write_test_obsolescence_baseline
        ) and obsolescence_baseline_payload is not None:
            report_root = Path(root)
            baseline_path = test_obsolescence_delta.resolve_baseline_path(report_root)
            response["test_obsolescence_baseline_path"] = str(baseline_path)
            if write_test_obsolescence_baseline:
                baseline_path.parent.mkdir(parents=True, exist_ok=True)
                test_obsolescence_delta.write_baseline(
                    str(baseline_path), obsolescence_baseline_payload
                )
                response["test_obsolescence_baseline_written"] = True
            else:
                response["test_obsolescence_baseline_written"] = False
            if emit_test_obsolescence_delta:
                if not baseline_path.exists():
                    never("test obsolescence baseline not found", path=str(baseline_path))
                baseline = test_obsolescence_delta.load_baseline(str(baseline_path))
                current = (
                    obsolescence_baseline
                    if obsolescence_baseline is not None
                    else test_obsolescence_delta.parse_baseline_payload(
                        obsolescence_baseline_payload
                    )
                )
                delta_payload = test_obsolescence_delta.build_delta_payload(
                    baseline, current, baseline_path=str(baseline_path)
                )
                out_dir, artifact_dir = _output_dirs(report_root)
                report_json = json.dumps(delta_payload, indent=2, sort_keys=True) + "\n"
                report_md = test_obsolescence_delta.render_markdown(delta_payload)
                (artifact_dir / "test_obsolescence_delta.json").write_text(report_json)
                (out_dir / "test_obsolescence_delta.md").write_text(report_md)
                response["test_obsolescence_delta_summary"] = delta_payload.get(
                    "summary", {}
                )

        ambiguity_witnesses: list[dict[str, object]] | None = None
        ambiguity_baseline_payload: dict[str, object] | None = None
        ambiguity_baseline: ambiguity_delta.AmbiguityBaseline | None = None
        if ambiguity_state_path:
            state_path = Path(ambiguity_state_path)
            if not state_path.exists():
                never("ambiguity state not found", path=str(state_path))
            state = ambiguity_state.load_state(
                str(state_path),
            )
            ambiguity_witnesses = [
                {str(k): entry[k] for k in entry} for entry in state.witnesses
            ]
            ambiguity_baseline_payload = ambiguity_delta.build_baseline_payload(
                ambiguity_witnesses,
            )
            ambiguity_baseline = state.baseline
        elif emit_ambiguity_delta or write_ambiguity_baseline or emit_ambiguity_state:
            ambiguity_witnesses = [
                {str(k): entry[k] for k in entry} for entry in analysis.ambiguity_witnesses
            ]
            if emit_ambiguity_state:
                report_root = Path(root)
                out_dir, artifact_dir = _output_dirs(report_root)
                state_payload = ambiguity_state.build_state_payload(
                    ambiguity_witnesses,
                )
                (artifact_dir / "ambiguity_state.json").write_text(
                    json.dumps(state_payload, indent=2, sort_keys=True) + "\n"
                )
            ambiguity_baseline_payload = ambiguity_delta.build_baseline_payload(
                ambiguity_witnesses,
            )
            ambiguity_baseline = ambiguity_delta.parse_baseline_payload(
                ambiguity_baseline_payload,
            )
        if (
            emit_ambiguity_delta or write_ambiguity_baseline
        ) and ambiguity_baseline_payload is not None:
            report_root = Path(root)
            baseline_path = ambiguity_delta.resolve_baseline_path(report_root)
            response["ambiguity_baseline_path"] = str(baseline_path)
            if write_ambiguity_baseline:
                baseline_path.parent.mkdir(parents=True, exist_ok=True)
                ambiguity_delta.write_baseline(
                    str(baseline_path), ambiguity_baseline_payload
                )
                response["ambiguity_baseline_written"] = True
            else:
                response["ambiguity_baseline_written"] = False
            if emit_ambiguity_delta:
                if not baseline_path.exists():
                    never("ambiguity baseline not found", path=str(baseline_path))
                baseline = ambiguity_delta.load_baseline(str(baseline_path))
                current = (
                    ambiguity_baseline
                    if ambiguity_baseline is not None
                    else ambiguity_delta.parse_baseline_payload(
                        ambiguity_baseline_payload
                    )
                )
                delta_payload = ambiguity_delta.build_delta_payload(
                    baseline, current, baseline_path=str(baseline_path)
                )
                out_dir, artifact_dir = _output_dirs(report_root)
                report_json = json.dumps(delta_payload, indent=2, sort_keys=True) + "\n"
                report_md = ambiguity_delta.render_markdown(delta_payload)
                (artifact_dir / "ambiguity_delta.json").write_text(report_json)
                (out_dir / "ambiguity_delta.md").write_text(report_md)
                response["ambiguity_delta_summary"] = delta_payload.get("summary", {})

        report: str | None = None
        violations: list[str] = []
        effective_violations: list[str] | None = None
        baseline_entries: list[str] = []
        if report_path:
            report_carrier = ReportCarrier.from_analysis_result(analysis)
            report_markdown, _ = render_report(
                analysis.groups_by_path,
                max_components,
                report=report_carrier,
            )
            resolved_sections_for_obligations = extract_report_sections(report_markdown)
            pending_projection_reasons: dict[str, str] = {}
            for row in projection_rows:
                check_deadline()
                section_id = str(row.get("section_id", "") or "")
                if not section_id or section_id in resolved_sections_for_obligations:
                    continue
                pending_projection_reasons[section_id] = "missing_dep"
            success_progress_payload: JSONObject = {
                "classification": "succeeded",
                "resume_supported": analysis_resume_reused_files > 0,
            }
            runtime_obligations = _incremental_progress_obligations(
                analysis_state="succeeded",
                progress_payload=success_progress_payload,
                resume_checkpoint_path=analysis_resume_checkpoint_path,
                partial_report_written=False,
                report_requested=bool(report_path),
                projection_rows=projection_rows,
                sections=resolved_sections_for_obligations,
                pending_reasons=pending_projection_reasons,
            )
            (
                report_carrier.resumability_obligations,
                report_carrier.incremental_report_obligations,
            ) = _split_incremental_obligations(runtime_obligations)
            report_markdown, violations = render_report(
                analysis.groups_by_path,
                max_components,
                report=report_carrier,
            )
            report = report_markdown
            if baseline_path is not None:
                baseline_entries = load_baseline(baseline_path)
                if baseline_write:
                    write_baseline(baseline_path, violations)
                    effective_violations = []
                else:
                    effective_violations, _ = apply_baseline(violations, baseline_entries)
            if report_output_path and projection_rows:
                resolved_sections = extract_report_sections(report_markdown)
                _write_report_section_journal(
                    path=report_section_journal_path,
                    witness_digest=report_section_witness_digest,
                    projection_rows=projection_rows,
                    sections=resolved_sections,
                )
                phase_checkpoint_state["post"] = {
                    "status": "final",
                    "work_done": 1,
                    "work_total": 1,
                    "section_ids": sorted(resolved_sections),
                    "resolved_sections": len(resolved_sections),
                }
                _write_report_phase_checkpoint(
                    path=report_phase_checkpoint_path,
                    witness_digest=report_section_witness_digest,
                    phases=phase_checkpoint_state,
                )
            if decision_snapshot_path:
                decision_payload = render_decision_snapshot(
                    surfaces=DecisionSnapshotSurfaces(
                        decision_surfaces=analysis.decision_surfaces,
                        value_decision_surfaces=analysis.value_decision_surfaces,
                    ),
                    forest=analysis.forest,
                    project_root=Path(root),
                    groups_by_path=analysis.groups_by_path,
                )
                report = report + "\n" + json.dumps(
                    decision_payload, indent=2, sort_keys=True
                )
            if structure_tree_path:
                structure_payload = render_structure_snapshot(
                    analysis.groups_by_path,
                    forest=analysis.forest,
                    project_root=Path(root),
                    invariant_propositions=analysis.invariant_propositions,
                )
                report = report + "\n" + json.dumps(
                    structure_payload, indent=2, sort_keys=True
                )
            if structure_metrics_path:
                report = report + "\n" + json.dumps(metrics, indent=2, sort_keys=True)
            if synthesis_plan and (
                synthesis_report or synthesis_plan_path or synthesis_protocols_path
            ):
                report = report + render_synthesis_section(synthesis_plan)
            if refactor_plan and (refactor_plan or refactor_plan_json):
                report = report + render_refactor_plan(plan_payload)
            if report_output_path is not None:
                report_output_path.parent.mkdir(parents=True, exist_ok=True)
                _write_text_profiled(
                    report_output_path,
                    report,
                    io_name="report_markdown.write",
                )
        else:
            violation_carrier = ReportCarrier(
                forest=analysis.forest,
                type_suggestions=analysis.type_suggestions if type_audit_report else [],
                type_ambiguities=analysis.type_ambiguities if type_audit_report else [],
                decision_warnings=analysis.decision_warnings,
                fingerprint_warnings=analysis.fingerprint_warnings,
                parse_failure_witnesses=analysis.parse_failure_witnesses,
            )
            violations = compute_violations(
                analysis.groups_by_path,
                max_components,
                report=violation_carrier,
            )
            if baseline_path is not None:
                baseline_entries = load_baseline(baseline_path)
                if baseline_write:
                    write_baseline(baseline_path, violations)
                    effective_violations = []
                else:
                    effective_violations, _ = apply_baseline(violations, baseline_entries)

        if dot_path and analysis.forest is not None:
            dot_payload = render_dot(analysis.forest)
            if _is_stdout_target(dot_path):
                response["dot"] = dot_payload
                if report is not None:
                    report = report + "\n" + dot_payload
            else:
                Path(dot_path).write_text(dot_payload)

        if effective_violations is None:
            effective_violations = violations
        response["violations"] = len(effective_violations)
        if fingerprint_synth_json and analysis.fingerprint_synth_registry:
            payload_json = json.dumps(
                analysis.fingerprint_synth_registry, indent=2, sort_keys=True
            )
            if _is_stdout_target(fingerprint_synth_json):
                response["fingerprint_synth_registry"] = analysis.fingerprint_synth_registry
            else:
                Path(fingerprint_synth_json).write_text(payload_json)
        if fingerprint_provenance_json and analysis.fingerprint_provenance:
            payload_json = json.dumps(
                analysis.fingerprint_provenance, indent=2, sort_keys=True
            )
            if _is_stdout_target(fingerprint_provenance_json):
                response["fingerprint_provenance"] = analysis.fingerprint_provenance
            else:
                Path(fingerprint_provenance_json).write_text(payload_json)
        if fingerprint_deadness_json is not None:
            payload_json = json.dumps(
                analysis.deadness_witnesses, indent=2, sort_keys=True
            )
            if _is_stdout_target(fingerprint_deadness_json):
                response["fingerprint_deadness"] = analysis.deadness_witnesses
            else:
                Path(fingerprint_deadness_json).write_text(payload_json)
        if fingerprint_coherence_json is not None:
            payload_json = json.dumps(
                analysis.coherence_witnesses, indent=2, sort_keys=True
            )
            if _is_stdout_target(fingerprint_coherence_json):
                response["fingerprint_coherence"] = analysis.coherence_witnesses
            else:
                Path(fingerprint_coherence_json).write_text(payload_json)
        if fingerprint_rewrite_plans_json is not None:
            payload_json = json.dumps(analysis.rewrite_plans, indent=2, sort_keys=True)
            if _is_stdout_target(fingerprint_rewrite_plans_json):
                response["fingerprint_rewrite_plans"] = analysis.rewrite_plans
            else:
                Path(fingerprint_rewrite_plans_json).write_text(payload_json)
        if fingerprint_exception_obligations_json is not None:
            payload_json = json.dumps(
                analysis.exception_obligations, indent=2, sort_keys=True
            )
            if _is_stdout_target(fingerprint_exception_obligations_json):
                response["fingerprint_exception_obligations"] = (
                    analysis.exception_obligations
                )
            else:
                Path(fingerprint_exception_obligations_json).write_text(payload_json)
        if fingerprint_handledness_json is not None:
            payload_json = json.dumps(
                analysis.handledness_witnesses, indent=2, sort_keys=True
            )
            if _is_stdout_target(fingerprint_handledness_json):
                response["fingerprint_handledness"] = analysis.handledness_witnesses
            else:
                Path(fingerprint_handledness_json).write_text(payload_json)
        if baseline_path is not None:
            response["baseline_path"] = str(baseline_path)
            response["baseline_written"] = bool(baseline_write)
        if fail_on_type_ambiguities and analysis.type_ambiguities:
            response["exit_code"] = 1
        else:
            if baseline_write:
                response["exit_code"] = 0
            else:
                response["exit_code"] = 1 if (fail_on_violations and effective_violations) else 0
        response["analysis_state"] = "succeeded"
        response["execution_plan"] = execution_plan.as_json_dict()
        _emit_lsp_progress(
            phase="post",
            collection_progress=latest_collection_progress,
            semantic_progress=semantic_progress_cumulative,
            include_timing=True,
            done=True,
            analysis_state="succeeded",
            classification="succeeded",
            event_kind="terminal",
        )
        return _normalize_dataflow_response(response)
    except TimeoutExceeded as exc:
        cleanup_now_ns = time.monotonic_ns()
        cleanup_remaining_ns = max(0, timeout_hard_deadline_ns - cleanup_now_ns)
        cleanup_window_ns = min(cleanup_grace_ns, cleanup_remaining_ns)
        cleanup_deadline_token = set_deadline(
            Deadline(deadline_ns=cleanup_now_ns + max(1, cleanup_window_ns))
        )
        cleanup_timeout_steps: list[str] = []

        def _mark_cleanup_timeout(step: str) -> None:
            cleanup_timeout_steps.append(step)

        try:
            try:
                timeout_payload = _timeout_context_payload(exc)
            except TimeoutExceeded:
                _mark_cleanup_timeout("timeout_context_payload")
                timeout_payload = {
                    "summary": "Analysis timed out.",
                    "progress": {"classification": "timed_out_no_progress"},
                }
            progress_payload = timeout_payload.get("progress")
            if not isinstance(progress_payload, dict):
                progress_payload = {}
                timeout_payload["progress"] = progress_payload
            if not isinstance(progress_payload.get("classification"), str):
                progress_payload["classification"] = "timed_out_no_progress"
            progress_payload.setdefault(
                "timeout_budget",
                {
                    "total_timeout_ns": timeout_total_ns,
                    "analysis_window_ns": analysis_window_ns,
                    "cleanup_grace_ns": cleanup_grace_ns,
                    "hard_deadline_ns": timeout_hard_deadline_ns,
                },
            )
            if analysis_resume_checkpoint_path is not None:
                progress_payload["resume_supported"] = True
            timeout_collection_resume_payload: JSONObject | None = None
            if (
                analysis_resume_checkpoint_path is not None
                and analysis_resume_input_manifest_digest is not None
                and isinstance(last_collection_resume_payload, Mapping)
            ):
                try:
                    latest_collection_resume_payload: JSONObject = {
                        str(key): last_collection_resume_payload[key]
                        for key in last_collection_resume_payload
                    }
                    execute_deps.write_analysis_resume_checkpoint_fn(
                        path=analysis_resume_checkpoint_path,
                        input_witness=analysis_resume_input_witness,
                        input_manifest_digest=analysis_resume_input_manifest_digest,
                        collection_resume=latest_collection_resume_payload,
                    )
                    timeout_collection_resume_payload = latest_collection_resume_payload
                    checkpoint_timeline_header: str | None = None
                    checkpoint_timeline_row: str | None = None
                    if emit_checkpoint_intro_timeline:
                        (
                            checkpoint_timeline_header,
                            checkpoint_timeline_row,
                        ) = _append_checkpoint_intro_timeline_row(
                            path=checkpoint_intro_timeline_path,
                            collection_resume=latest_collection_resume_payload,
                            total_files=analysis_resume_total_files,
                            checkpoint_path=analysis_resume_checkpoint_path,
                            checkpoint_status=analysis_resume_checkpoint_status,
                            checkpoint_reused_files=analysis_resume_reused_files,
                            checkpoint_total_files=analysis_resume_total_files,
                        )
                    if checkpoint_timeline_row is not None:
                        timeout_resume_progress = _analysis_resume_progress(
                            collection_resume=latest_collection_resume_payload,
                            total_files=analysis_resume_total_files,
                        )
                        timeout_semantic_progress = latest_collection_resume_payload.get(
                            "semantic_progress"
                        )
                        timeout_resume_checkpoint_payload: JSONObject = {
                            "checkpoint_path": str(analysis_resume_checkpoint_path),
                            "status": analysis_resume_checkpoint_status or "unknown",
                            "reused_files": analysis_resume_reused_files,
                            "total_files": analysis_resume_total_files,
                        }
                        _emit_lsp_progress(
                            phase="collection",
                            collection_progress=timeout_resume_progress,
                            semantic_progress=(
                                timeout_semantic_progress
                                if isinstance(timeout_semantic_progress, Mapping)
                                else None
                            ),
                            work_done=timeout_resume_progress.get("completed_files"),
                            work_total=timeout_resume_progress.get("total_files"),
                            include_timing=profile_enabled,
                            analysis_state="analysis_collection_checkpoint_serialized",
                            classification="checkpoint_serialized",
                            resume_checkpoint=timeout_resume_checkpoint_payload,
                            checkpoint_intro_timeline_header=checkpoint_timeline_header,
                            checkpoint_intro_timeline_row=checkpoint_timeline_row,
                            event_kind="checkpoint",
                        )
                except TimeoutExceeded:
                    _mark_cleanup_timeout("write_analysis_resume_checkpoint")
            if analysis_resume_checkpoint_path is not None:
                try:
                    collection_resume: JSONObject | None = None
                    resume_input_witness: JSONObject | None = analysis_resume_input_witness
                    if timeout_collection_resume_payload is not None:
                        collection_resume = timeout_collection_resume_payload
                    elif analysis_resume_input_witness is not None:
                        collection_resume = execute_deps.load_analysis_resume_checkpoint_fn(
                            path=analysis_resume_checkpoint_path,
                            input_witness=analysis_resume_input_witness,
                        )
                    elif analysis_resume_input_manifest_digest is not None:
                        manifest_resume = (
                            execute_deps.load_analysis_resume_checkpoint_manifest_fn(
                            path=analysis_resume_checkpoint_path,
                            manifest_digest=analysis_resume_input_manifest_digest,
                        )
                        )
                        if manifest_resume is not None:
                            resume_input_witness, collection_resume = manifest_resume
                    if collection_resume is not None:
                        timeout_collection_resume_payload = collection_resume
                        resume_progress = _analysis_resume_progress(
                            collection_resume=collection_resume,
                            total_files=analysis_resume_total_files,
                        )
                        progress_payload["completed_files"] = resume_progress[
                            "completed_files"
                        ]
                        progress_payload["in_progress_files"] = resume_progress[
                            "in_progress_files"
                        ]
                        progress_payload["remaining_files"] = resume_progress[
                            "remaining_files"
                        ]
                        progress_payload["total_files"] = resume_progress["total_files"]
                        resume_supported = (
                            resume_progress["completed_files"] > 0
                            or resume_progress.get("in_progress_files", 0) > 0
                        )
                        progress_payload["resume_supported"] = resume_supported
                        semantic_substantive_progress: bool | None = None
                        semantic_progress = collection_resume.get("semantic_progress")
                        if isinstance(semantic_progress, Mapping):
                            progress_payload["semantic_progress"] = {
                                str(key): semantic_progress[key] for key in semantic_progress
                            }
                            raw_semantic_substantive = semantic_progress.get(
                                "substantive_progress"
                            )
                            semantic_substantive_progress = {
                                True: True,
                                False: False,
                            }.get(raw_semantic_substantive)
                        witness_digest = (
                            resume_input_witness.get("witness_digest")
                            if resume_input_witness is not None
                            else None
                        )
                        if not isinstance(witness_digest, str):
                            witness_digest = None
                        resume_token: JSONObject = {
                            "phase": "analysis_collection",
                            "checkpoint_path": str(analysis_resume_checkpoint_path),
                            "carrier_refs": {
                                "collection_resume": True,
                            },
                            **resume_progress,
                        }
                        if witness_digest is not None:
                            resume_token["witness_digest"] = witness_digest
                        resume_payload: JSONObject = {"resume_token": resume_token}
                        if resume_input_witness is not None:
                            resume_payload["input_witness"] = resume_input_witness
                        progress_payload["resume"] = resume_payload
                        classification = progress_payload.get("classification")
                        if (
                            resume_supported
                            and isinstance(classification, str)
                            and classification == "timed_out_no_progress"
                            and (
                                semantic_substantive_progress is None
                                or semantic_substantive_progress
                            )
                        ):
                            progress_payload["classification"] = "timed_out_progress_resume"
                        elif (
                            (
                                not resume_supported
                                or semantic_substantive_progress is False
                            )
                            and isinstance(classification, str)
                            and classification == "timed_out_progress_resume"
                        ):
                            progress_payload["classification"] = "timed_out_no_progress"
                except TimeoutExceeded:
                    _mark_cleanup_timeout("load_resume_progress")
            analysis_state = "timed_out_no_progress"
            classification = progress_payload.get("classification")
            if isinstance(classification, str) and classification:
                analysis_state = classification
            if analysis_state == "timed_out_progress_resume":
                progress_payload["resume_supported"] = True
            partial_report_written = False
            resolved_sections: dict[str, list[str]] = {}
            pending_reasons: dict[str, str] = {}
            if report_output_path is not None and projection_rows:
                try:
                    loaded_phase_checkpoint_state = (
                        execute_deps.load_report_phase_checkpoint_fn(
                        path=report_phase_checkpoint_path,
                        witness_digest=report_section_witness_digest,
                    )
                    )
                    phase_checkpoint_state = (
                        phase_checkpoint_state or loaded_phase_checkpoint_state or {}
                    )
                    resolved_sections, journal_reason = _ensure_report_sections_cache()
                    if (
                        timeout_collection_resume_payload is not None
                        and "components" not in resolved_sections
                    ):
                        resolved_sections["components"] = _collection_components_preview_lines(
                            collection_resume=timeout_collection_resume_payload,
                        )
                    if (
                        enable_phase_projection_checkpoints
                        and timeout_collection_resume_payload is not None
                    ):
                        preview_groups_by_path = _groups_by_path_from_collection_resume(
                            timeout_collection_resume_payload
                        )
                        preview_report = ReportCarrier(
                            forest=forest,
                            parse_failure_witnesses=[],
                        )
                        preview_sections = execute_deps.project_report_sections_fn(
                            preview_groups_by_path,
                            preview_report,
                            max_phase="post",
                            include_previews=True,
                            preview_only=True,
                        )
                        for section_id, section_lines in preview_sections.items():
                            check_deadline()
                            resolved_sections.setdefault(section_id, section_lines)
                    intro_lines = (
                        _collection_progress_intro_lines(
                            collection_resume=timeout_collection_resume_payload,
                            total_files=analysis_resume_total_files,
                            resume_checkpoint_intro=analysis_resume_intro_payload,
                        )
                        if timeout_collection_resume_payload is not None
                        else [
                            "Collection bootstrap checkpoint (provisional).",
                            f"- `root`: `{initial_root}`",
                            f"- `paths_requested`: `{initial_paths_count}`",
                        ]
                    )
                    resolved_sections.setdefault("intro", intro_lines)
                    partial_report, pending_reasons = _render_incremental_report(
                        analysis_state=analysis_state,
                        progress_payload=progress_payload,
                        projection_rows=projection_rows,
                        sections=resolved_sections,
                    )
                    _apply_journal_pending_reason(
                        projection_rows=projection_rows,
                        sections=resolved_sections,
                        pending_reasons=pending_reasons,
                        journal_reason=journal_reason,
                    )
                    report_output_path.parent.mkdir(parents=True, exist_ok=True)
                    _write_text_profiled(
                        report_output_path,
                        partial_report,
                        io_name="report_markdown.write",
                    )
                    _write_report_section_journal(
                        path=report_section_journal_path,
                        witness_digest=report_section_witness_digest,
                        projection_rows=projection_rows,
                        sections=resolved_sections,
                        pending_reasons=pending_reasons,
                    )
                    report_sections_cache_reason = None
                    phase_checkpoint_state["timeout"] = {
                        "status": "timed_out",
                        "analysis_state": analysis_state,
                        "section_ids": sorted(resolved_sections),
                        "resolved_sections": len(resolved_sections),
                        "completed_phase": _latest_report_phase(phase_checkpoint_state),
                    }
                    _write_report_phase_checkpoint(
                        path=report_phase_checkpoint_path,
                        witness_digest=report_section_witness_digest,
                        phases=phase_checkpoint_state,
                    )
                    partial_report_written = True
                except TimeoutExceeded:
                    _mark_cleanup_timeout("render_timeout_report")
            try:
                obligations = _incremental_progress_obligations(
                    analysis_state=analysis_state,
                    progress_payload=progress_payload,
                    resume_checkpoint_path=analysis_resume_checkpoint_path,
                    partial_report_written=partial_report_written,
                    report_requested=report_output_path is not None,
                    projection_rows=projection_rows,
                    sections=resolved_sections,
                    pending_reasons=pending_reasons,
                )
            except TimeoutExceeded:
                _mark_cleanup_timeout("incremental_obligations")
                obligations = []
            progress_payload["incremental_obligations"] = obligations
            if cleanup_timeout_steps:
                progress_payload["cleanup_truncated"] = True
                progress_payload["cleanup_timeout_steps"] = cleanup_timeout_steps
            timeout_classification = progress_payload.get("classification")
            _emit_lsp_progress(
                phase="post",
                collection_progress=latest_collection_progress,
                semantic_progress=semantic_progress_cumulative,
                include_timing=True,
                done=True,
                analysis_state=analysis_state,
                classification=(
                    timeout_classification
                    if isinstance(timeout_classification, str)
                    else "timed_out_no_progress"
                ),
                event_kind="terminal",
            )
            return _normalize_dataflow_response(
                {
                    "exit_code": 2,
                    "timeout": True,
                    "analysis_state": analysis_state,
                    "execution_plan": execution_plan.as_json_dict(),
                    "timeout_context": timeout_payload,
                }
            )
        finally:
            reset_deadline(cleanup_deadline_token)
    except Exception:
        emit_progress = locals().get("_emit_lsp_progress")
        if callable(emit_progress):
            emit_progress(
                phase="post",
                collection_progress=latest_collection_progress,
                semantic_progress=semantic_progress_cumulative,
                include_timing=True,
                done=True,
                analysis_state="failed",
                classification="failed",
                event_kind="terminal",
            )
        raise
    finally:
        stop_heartbeat = locals().get("heartbeat_stop_event")
        if isinstance(stop_heartbeat, threading.Event):
            stop_heartbeat.set()
        active_heartbeat_thread = locals().get("heartbeat_thread")
        if isinstance(active_heartbeat_thread, threading.Thread):
            active_heartbeat_thread.join(timeout=1.5)
        reset_forest(forest_token)
        reset_deadline_clock(deadline_clock_token)
        reset_deadline(deadline_token)
        reset_deadline_profile(profile_token)


@server.command(SYNTHESIS_COMMAND)
def execute_synthesis(
    ls: LanguageServer,
    payload: dict | None = None,
) -> dict:
    normalized_payload = _require_optional_payload(payload, command=SYNTHESIS_COMMAND)
    return _execute_synthesis_total(ls, normalized_payload)


def _execute_synthesis_total(
    ls: LanguageServer,
    payload: dict[str, object],
) -> dict:
    with _deadline_scope_from_payload(payload):
        check_deadline()
        try:
            request = SynthesisRequest.model_validate(payload)
        except ValidationError as exc:
            return SynthesisPlanResponseDTO(protocols=[], warnings=[], errors=[str(exc)]).model_dump()

        bundle_tiers: dict[frozenset[str], int] = {}
        for entry in request.bundles:
            check_deadline()
            bundle = entry.bundle
            if not bundle:
                continue
            bundle_tiers[frozenset(bundle)] = entry.tier

        field_types = request.field_types or {}
        config = SynthesisConfig(
            max_tier=request.max_tier,
            min_bundle_size=request.min_bundle_size,
            allow_singletons=request.allow_singletons,
            merge_overlap_threshold=request.merge_overlap_threshold,
        )
        naming_context = NamingContext(
            existing_names=set(request.existing_names),
            frequency=request.frequency or {},
            fallback_prefix=request.fallback_prefix,
        )
        plan = Synthesizer(config=config).plan(
            bundle_tiers=bundle_tiers,
            field_types=field_types,
            naming_context=naming_context,
        )
        response = SynthesisPlanResponseDTO(
            protocols=[
                {
                    "name": spec.name,
                    "fields": [
                        {
                            "name": field.name,
                            "type_hint": field.type_hint,
                            "source_params": sorted(field.source_params),
                        }
                        for field in spec.fields
                    ],
                    "bundle": sorted(spec.bundle),
                    "tier": spec.tier,
                    "rationale": spec.rationale,
                }
                for spec in plan.protocols
            ],
            warnings=plan.warnings,
            errors=plan.errors,
        )
        return response.model_dump()


@server.command(REFACTOR_COMMAND)
def execute_refactor(
    ls: LanguageServer,
    payload: dict | None = None,
) -> dict:
    normalized_payload = _require_optional_payload(payload, command=REFACTOR_COMMAND)
    return _execute_refactor_total(ls, normalized_payload)


def _execute_refactor_total(ls: LanguageServer, payload: dict[str, object]) -> dict:
    with _deadline_scope_from_payload(payload):
        try:
            request = RefactorRequest.model_validate(payload)
        except ValidationError as exc:
            return RefactorProtocolResponseDTO(errors=[str(exc)]).model_dump()

        project_root = None
        if ls.workspace.root_path:
            project_root = Path(ls.workspace.root_path)
        engine = RefactorEngine(project_root=project_root)
        normalized_bundle = request.bundle or [field.name for field in request.fields or []]
        compatibility_shim = request.compatibility_shim
        if isinstance(compatibility_shim, bool):
            normalized_shim: bool | RefactorCompatibilityShimConfig = compatibility_shim
        else:
            normalized_shim = RefactorCompatibilityShimConfig(
                enabled=compatibility_shim.enabled,
                emit_deprecation_warning=compatibility_shim.emit_deprecation_warning,
                emit_overload_stubs=compatibility_shim.emit_overload_stubs,
            )
        plan = engine.plan_protocol_extraction(
            RefactorRequestModel(
                protocol_name=request.protocol_name,
                bundle=normalized_bundle,
                fields=[
                    FieldSpec(name=field.name, type_hint=field.type_hint)
                    for field in request.fields or []
                ],
                target_path=request.target_path,
                target_functions=request.target_functions,
                compatibility_shim=normalized_shim,
                ambient_rewrite=request.ambient_rewrite,
                rationale=request.rationale,
            )
        )
        edits = [
            TextEditDTO(
                path=edit.path,
                start=edit.start,
                end=edit.end,
                replacement=edit.replacement,
            )
            for edit in plan.edits
        ]
        rewrite_plans = [
            RewritePlanEntryDTO(
                kind=entry.kind,
                status=entry.status,
                target=entry.target,
                summary=entry.summary,
                non_rewrite_reasons=entry.non_rewrite_reasons,
            )
            for entry in plan.rewrite_plans
        ]
        response = RefactorProtocolResponseDTO(
            edits=edits,
            rewrite_plans=rewrite_plans,
            warnings=plan.warnings,
            errors=plan.errors,
        )
        return response.model_dump()


@dataclass(frozen=True)
class SnapshotDiffPaths:
    baseline: Path
    current: Path


@dataclass(frozen=True)
class StructureReuseOptions:
    snapshot: Path
    lemma_stubs: str | None
    min_count: int | None


def _parse_snapshot_diff_paths(payload: Mapping[str, object]) -> SnapshotDiffPaths | None:
    baseline_path = payload.get("baseline")
    current_path = payload.get("current")
    if not baseline_path or not current_path:
        return None
    return SnapshotDiffPaths(baseline=Path(str(baseline_path)), current=Path(str(current_path)))


def _parse_structure_reuse_options(payload: Mapping[str, object]) -> StructureReuseOptions | None:
    snapshot_path = payload.get("snapshot")
    if not snapshot_path:
        return None
    min_count_raw = payload.get("min_count", 2)
    try:
        min_count_int: int | None = int(min_count_raw)
    except (TypeError, ValueError):
        min_count_int = None
    return StructureReuseOptions(
        snapshot=Path(str(snapshot_path)),
        lemma_stubs=cast(str | None, payload.get("lemma_stubs")),
        min_count=min_count_int,
    )

@server.command(STRUCTURE_DIFF_COMMAND)
def execute_structure_diff(
    ls: LanguageServer,
    payload: dict | None = None,
) -> dict:
    normalized_payload = _require_optional_payload(payload, command=STRUCTURE_DIFF_COMMAND)
    return _execute_structure_diff_total(ls, normalized_payload)


def _execute_structure_diff_total(
    ls: LanguageServer,
    payload: dict[str, object],
) -> dict:
    with _deadline_scope_from_payload(payload):
        try:
            snapshot_payload = SnapshotDiffPayload.from_payload(
                payload,
                error_message="baseline and current snapshot paths are required",
            )
            baseline = load_structure_snapshot(snapshot_payload.baseline)
            current = load_structure_snapshot(snapshot_payload.current)
        except ValueError as exc:
            return StructureDiffResponseDTO(exit_code=2, errors=[str(exc)]).model_dump()
        return StructureDiffResponseDTO(
            exit_code=0,
            diff=diff_structure_snapshots(baseline, current),
        ).model_dump()


@server.command(STRUCTURE_REUSE_COMMAND)
def execute_structure_reuse(
    ls: LanguageServer,
    payload: dict | None = None,
) -> dict:
    normalized_payload = _require_optional_payload(payload, command=STRUCTURE_REUSE_COMMAND)
    return _execute_structure_reuse_total(ls, normalized_payload)


def _execute_structure_reuse_total(
    ls: LanguageServer,
    payload: dict[str, object],
) -> dict:
    with _deadline_scope_from_payload(payload):
        options = _parse_structure_reuse_options(payload)
        if options is None:
            return StructureReuseResponseDTO(exit_code=2, errors=["snapshot path is required"]).model_dump()
        try:
            snapshot = load_structure_snapshot(options.snapshot)
        except ValueError as exc:
            return StructureReuseResponseDTO(exit_code=2, errors=[str(exc)]).model_dump()
        if options.min_count is None:
            return StructureReuseResponseDTO(exit_code=2, errors=["min_count must be an integer"]).model_dump()
        if options.min_count <= 0:
            return StructureReuseResponseDTO(exit_code=2, errors=["min_count must be positive"]).model_dump()
        reuse = compute_structure_reuse(snapshot, min_count=options.min_count)
        response: JSONObject = StructureReuseResponseDTO(exit_code=0, reuse=reuse).model_dump()
        if options.lemma_stubs:
            stubs = render_reuse_lemma_stubs(reuse)
            if _is_stdout_target(options.lemma_stubs):
                response["lemma_stubs"] = stubs
            else:
                Path(options.lemma_stubs).write_text(stubs)
        return response


@server.command(DECISION_DIFF_COMMAND)
def execute_decision_diff(
    ls: LanguageServer,
    payload: dict | None = None,
) -> dict:
    normalized_payload = _require_optional_payload(payload, command=DECISION_DIFF_COMMAND)
    return _execute_decision_diff_total(ls, normalized_payload)


def _execute_decision_diff_total(
    ls: LanguageServer,
    payload: dict[str, object],
) -> dict:
    with _deadline_scope_from_payload(payload):
        try:
            snapshot_payload = SnapshotDiffPayload.from_payload(
                payload,
                error_message="baseline and current decision snapshot paths are required",
            )
            baseline = load_decision_snapshot(snapshot_payload.baseline)
            current = load_decision_snapshot(snapshot_payload.current)
        except ValueError as exc:
            return DecisionDiffResponseDTO(exit_code=2, errors=[str(exc)]).model_dump()
        return DecisionDiffResponseDTO(
            exit_code=0,
            diff=diff_decision_snapshots(baseline, current),
        ).model_dump()


@dataclass(frozen=True)
class ImpactSpan:
    path: str
    start_line: int
    end_line: int


@dataclass(frozen=True)
class ImpactFunction:
    path: str
    qual: str
    name: str
    start_line: int
    end_line: int
    is_test: bool


@dataclass(frozen=True)
class ImpactEdge:
    caller: str
    callee: str
    confidence: float
    inferred: bool


def _normalize_impact_change_entry(entry: object) -> ImpactSpan | None:
    check_deadline()
    if isinstance(entry, Mapping):
        path = str(entry.get("path", "") or "").strip()
        if not path:
            return None
        try:
            start_line = int(entry.get("start_line", 1) or 1)
            end_line = int(entry.get("end_line", start_line) or start_line)
        except (TypeError, ValueError):
            return None
        if start_line <= 0 or end_line <= 0:
            return None
        if end_line < start_line:
            start_line, end_line = end_line, start_line
        return ImpactSpan(path=path, start_line=start_line, end_line=end_line)
    text = str(entry or "").strip()
    if not text:
        return None
    match = _IMPACT_CHANGE_RE.match(text)
    if match is None:
        return None
    path = str(match.group("path") or "").strip()
    if not path:  # pragma: no cover - regex requires a non-empty path token
        return None
    start_group = match.group("start")
    end_group = match.group("end")
    if start_group is None:
        return ImpactSpan(path=path, start_line=1, end_line=10**9)
    start_line = int(start_group)
    end_line = int(end_group) if end_group is not None else start_line
    if end_line < start_line:
        start_line, end_line = end_line, start_line
    return ImpactSpan(path=path, start_line=start_line, end_line=end_line)


def _parse_impact_diff_ranges(diff_text: str) -> list[ImpactSpan]:
    check_deadline()
    spans: list[ImpactSpan] = []
    current_path = ""
    for raw_line in diff_text.splitlines():
        check_deadline()
        line = raw_line.rstrip("\n")
        file_match = _IMPACT_DIFF_FILE_RE.match(line)
        if file_match is not None:
            current_path = str(file_match.group("path") or "").strip()
            if current_path == "/dev/null":
                current_path = ""
            continue
        if not current_path:
            continue
        hunk_match = _IMPACT_DIFF_HUNK_RE.match(line)
        if hunk_match is None:
            continue
        start_line = int(hunk_match.group("start"))
        count_text = hunk_match.group("count")
        count = int(count_text) if count_text else 1
        if count <= 0:
            count = 1
        spans.append(
            ImpactSpan(
                path=current_path,
                start_line=start_line,
                end_line=start_line + count - 1,
            )
        )
    return spans


def _impact_path_is_test(path: str) -> bool:
    normalized = path.replace("\\", "/")
    if normalized.startswith("tests/"):
        return True
    if any(token in path for token in _IMPACT_TEST_PATH_TOKENS):
        return True
    filename = Path(normalized).name
    return filename.startswith("test_") or filename.endswith("_test.py")


def _impact_functions_from_tree(path: str, tree: ast.AST) -> list[ImpactFunction]:
    check_deadline()
    out: list[ImpactFunction] = []

    def _walk(node: ast.AST, qual_parts: list[str]) -> None:
        for child in ast.iter_child_nodes(node):
            check_deadline()
            if isinstance(child, ast.ClassDef):
                _walk(child, [*qual_parts, child.name])
                continue
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                start_line = getattr(child, "lineno", None)
                end_line = getattr(child, "end_lineno", None)
                if isinstance(start_line, int) and isinstance(end_line, int):
                    qual = ".".join([*qual_parts, child.name])
                    out.append(
                        ImpactFunction(
                            path=path,
                            qual=qual,
                            name=child.name,
                            start_line=start_line,
                            end_line=end_line,
                            is_test=_impact_path_is_test(path) or child.name.startswith("test_"),
                        )
                    )
                _walk(child, [*qual_parts, child.name])

    _walk(tree, [])
    return out


def _impact_collect_edges(
    *,
    functions_by_qual: Mapping[str, ImpactFunction],
    trees_by_path: Mapping[str, ast.AST],
) -> list[ImpactEdge]:
    check_deadline()
    by_name: dict[str, list[ImpactFunction]] = defaultdict(list)
    by_path_qual: dict[tuple[str, str], ImpactFunction] = {
        (fn.path, fn.qual): fn for fn in functions_by_qual.values()
    }
    for fn in deadline_loop_iter(functions_by_qual.values()):
        check_deadline()
        by_name[fn.name].append(fn)
    for entries in deadline_loop_iter(by_name.values()):
        entries.sort(key=lambda fn: (fn.path, fn.qual))
    edges: list[ImpactEdge] = []
    for path, tree in deadline_loop_iter(trees_by_path.items()):
        check_deadline()
        for fn in deadline_loop_iter(_impact_functions_from_tree(path, tree)):
            check_deadline()
            caller = by_path_qual.get((fn.path, fn.qual))
            if caller is None:
                continue
            for node in deadline_loop_iter(ast.walk(tree)):
                check_deadline()
                if not isinstance(node, ast.Call):
                    continue
                if not isinstance(node.func, (ast.Name, ast.Attribute)):
                    continue
                line = getattr(node, "lineno", None)
                end_line = getattr(node, "end_lineno", line)
                if not isinstance(line, int) or not isinstance(end_line, int):
                    continue
                if line < caller.start_line or end_line > caller.end_line:
                    continue
                callee_name = node.func.id if isinstance(node.func, ast.Name) else node.func.attr
                candidates = by_name.get(callee_name, [])
                if not candidates:
                    continue
                inferred = len(candidates) > 1
                confidence = 1.0 if not inferred else max(0.1, 1.0 / float(len(candidates)))
                for callee in deadline_loop_iter(candidates):
                    check_deadline()
                    edges.append(
                        ImpactEdge(
                            caller=caller.qual,
                            callee=callee.qual,
                            confidence=confidence,
                            inferred=inferred,
                        )
                    )
    return edges


def _impact_parse_doc_sections(path: Path) -> list[tuple[str, str]]:
    check_deadline()
    sections: list[tuple[str, str]] = []
    current_heading = "(preamble)"
    bucket: list[str] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        check_deadline()
        if line.startswith("#"):
            if bucket:
                sections.append((current_heading, "\n".join(bucket)))
                bucket = []
            current_heading = line.strip("# ").strip() or "(unnamed)"
            continue
        bucket.append(line)
    if bucket:
        sections.append((current_heading, "\n".join(bucket)))
    return sections


def _impact_overlap(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    return not (a_end < b_start or b_end < a_start)


@server.command(IMPACT_COMMAND)
def execute_impact(
    ls: LanguageServer,
    payload: dict[str, object] | None = None,
) -> dict:
    normalized_payload = _require_optional_payload(payload, command=IMPACT_COMMAND)
    return _execute_impact_total(ls, normalized_payload)


def _execute_impact_total(ls: LanguageServer, payload: dict[str, object]) -> dict:
    with _deadline_scope_from_payload(payload):
        root = Path(str(payload.get("root") or ls.workspace.root_path or "."))
        max_call_depth_raw = payload.get("max_call_depth")
        try:
            max_call_depth = int(max_call_depth_raw) if max_call_depth_raw is not None else None
        except (TypeError, ValueError):
            return {"exit_code": 2, "errors": ["max_call_depth must be an integer"]}
        if isinstance(max_call_depth, int) and max_call_depth < 0:
            return {"exit_code": 2, "errors": ["max_call_depth must be non-negative"]}

        threshold_raw = payload.get("confidence_threshold", 0.5)
        try:
            confidence_threshold = float(threshold_raw)
        except (TypeError, ValueError):
            return {"exit_code": 2, "errors": ["confidence_threshold must be numeric"]}
        confidence_threshold = max(0.0, min(1.0, confidence_threshold))

        changes: list[ImpactSpan] = []
        raw_changes = payload.get("changes")
        if isinstance(raw_changes, Sequence) and not isinstance(raw_changes, (str, bytes)):
            for entry in deadline_loop_iter(raw_changes):
                parsed = _normalize_impact_change_entry(entry)
                if parsed is not None:
                    changes.append(parsed)
        raw_diff = payload.get("git_diff")
        if isinstance(raw_diff, str) and raw_diff.strip():
            changes.extend(_parse_impact_diff_ranges(raw_diff))
        if not changes:
            return {
                "exit_code": 2,
                "errors": ["Provide at least one change span or git diff"],
            }

        py_paths = ordered_or_sorted(
            root.rglob("*.py"),
            source="_execute_impact_total.py_paths",
            key=lambda path: str(path),
        )
        trees_by_path: dict[str, ast.AST] = {}
        functions: dict[str, ImpactFunction] = {}
        for py_path in deadline_loop_iter(py_paths):
            check_deadline()
            rel_path = str(py_path.relative_to(root)).replace("\\", "/")
            try:
                tree = ast.parse(py_path.read_text(encoding="utf-8"), filename=rel_path)
            except (OSError, SyntaxError):
                continue
            trees_by_path[rel_path] = tree
            for fn in deadline_loop_iter(_impact_functions_from_tree(rel_path, tree)):
                check_deadline()
                functions[fn.qual] = fn

        reverse_edges: dict[str, list[ImpactEdge]] = defaultdict(list)
        for edge in deadline_loop_iter(
            _impact_collect_edges(functions_by_qual=functions, trees_by_path=trees_by_path)
        ):
            check_deadline()
            reverse_edges[edge.callee].append(edge)

        seed_functions: set[str] = set()
        normalized_changes: list[dict[str, object]] = []
        for change in deadline_loop_iter(changes):
            check_deadline()
            normalized_path = change.path.replace("\\", "/")
            normalized_changes.append(
                {
                    "path": normalized_path,
                    "start_line": change.start_line,
                    "end_line": change.end_line,
                }
            )
            for fn in deadline_loop_iter(functions.values()):
                check_deadline()
                if fn.path != normalized_path:
                    continue
                if _impact_overlap(change.start_line, change.end_line, fn.start_line, fn.end_line):
                    seed_functions.add(fn.qual)

        queue: deque[tuple[str, int, bool, float]] = deque(
            (qual, 0, False, 1.0)
            for qual in ordered_or_sorted(
                seed_functions,
                source="_execute_impact_total.seed_functions",
            )
        )
        seen_state: set[tuple[str, bool]] = set()
        must_tests: dict[str, dict[str, object]] = {}
        likely_tests: dict[str, dict[str, object]] = {}
        max_queue_steps = max((len(functions) * 2) + len(seed_functions), 1)
        for _ in deadline_loop_iter(range(max_queue_steps)):
            if not queue:
                break
            check_deadline()
            current, depth, has_inferred, path_confidence = queue.popleft()
            state_key = (current, has_inferred)
            if state_key in seen_state:
                continue
            seen_state.add(state_key)
            if isinstance(max_call_depth, int) and depth >= max_call_depth:
                continue
            for edge in deadline_loop_iter(reverse_edges.get(current, [])):
                check_deadline()
                caller_fn = functions.get(edge.caller)
                if caller_fn is None:  # pragma: no cover - edges are derived from known functions
                    continue
                next_inferred = has_inferred or edge.inferred
                next_confidence = min(path_confidence, edge.confidence)
                if caller_fn.is_test:
                    target = likely_tests if next_inferred else must_tests
                    key = f"{caller_fn.path}::{caller_fn.qual}"
                    existing = target.get(key)
                    if existing is None or float(existing.get("confidence", 0.0)) < next_confidence:
                        target[key] = {
                            "id": key,
                            "path": caller_fn.path,
                            "qual": caller_fn.qual,
                            "confidence": round(next_confidence, 3),
                            "depth": depth + 1,
                        }
                queue.append((edge.caller, depth + 1, next_inferred, next_confidence))

        filtered_likely = [
            item
            for item in ordered_or_sorted(
                likely_tests.values(),
                source="_execute_impact_total.likely_tests",
                key=lambda item: str(item["id"]),
            )
            if float(item.get("confidence", 0.0)) >= confidence_threshold
        ]
        docs: list[dict[str, object]] = []
        impacted_names = {functions[qual].name for qual in seed_functions if qual in functions}
        for md_path in deadline_loop_iter(
            ordered_or_sorted(
                root.rglob("*.md"),
                source="_execute_impact_total.md_paths",
                key=lambda path: str(path),
            )
        ):
            check_deadline()
            rel_path = str(md_path.relative_to(root)).replace("\\", "/")
            for heading, text in deadline_loop_iter(_impact_parse_doc_sections(md_path)):
                check_deadline()
                lowered = text.lower()
                matches = ordered_or_sorted(
                    {name for name in impacted_names if name.lower() in lowered},
                    source="_execute_impact_total.doc_matches",
                )
                if not matches:
                    continue
                docs.append({"path": rel_path, "section": heading, "symbols": matches})

        return {
            "exit_code": 0,
            "changes": normalized_changes,
            "seed_functions": ordered_or_sorted(
                seed_functions,
                source="_execute_impact_total.seed_functions_out",
            ),
            "must_run_tests": [
                must_tests[key]
                for key in ordered_or_sorted(
                    must_tests,
                    source="_execute_impact_total.must_tests",
                )
            ],
            "likely_run_tests": filtered_likely,
            "impacted_docs": docs,
            "meta": {
                "max_call_depth": max_call_depth,
                "confidence_threshold": confidence_threshold,
            },
        }


@server.feature(TEXT_DOCUMENT_CODE_ACTION)
def code_action(ls: LanguageServer, params: CodeActionParams) -> list[CodeAction]:
    path = _uri_to_path(params.text_document.uri)
    payload = {
        "protocol_name": "TODO_Bundle",
        "bundle": [],
        "target_path": str(path),
        "target_functions": [],
        "rationale": "Stub code action; populate bundle details manually.",
    }
    title = "Gabion: Extract Protocol (stub)"
    return [
        CodeAction(
            title=title,
            kind=CodeActionKind.RefactorExtract,
            command=Command(title=title, command=REFACTOR_COMMAND, arguments=[payload]),
            edit=WorkspaceEdit(changes={}),
        )
    ]


@server.feature(TEXT_DOCUMENT_DID_OPEN)
def did_open(ls: LanguageServer, params) -> None:
    uri = params.text_document.uri
    doc = ls.workspace.get_document(uri)
    root = Path(ls.workspace.root_path) if ls.workspace.root_path else None
    diagnostics = _diagnostics_for_path(doc.path, root)
    ls.publish_diagnostics(uri, diagnostics)


@server.feature(TEXT_DOCUMENT_DID_SAVE)
def did_save(ls: LanguageServer, params) -> None:
    uri = params.text_document.uri
    doc = ls.workspace.get_document(uri)
    root = Path(ls.workspace.root_path) if ls.workspace.root_path else None
    diagnostics = _diagnostics_for_path(doc.path, root)
    ls.publish_diagnostics(uri, diagnostics)


def start(start_fn: Callable[[], None] | None = None) -> None:
    """Start the language server (stub)."""
    (start_fn or server.start_io)()


if __name__ == "__main__":  # pragma: no cover
    start()  # pragma: no cover

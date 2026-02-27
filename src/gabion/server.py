from __future__ import annotations
# gabion:decision_protocol_module
# gabion:boundary_normalization_module

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
from gabion.commands import boundary_order, command_ids, direct_dispatch, payload_codec
from gabion.commands.check_contract import LintEntriesDecision
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
from gabion.analysis import aspf_execution_fibration, aspf_resume_state
from gabion.analysis.aspf import Forest, NodeId, structural_key_atom
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
from gabion.exceptions import NeverThrown
from gabion.invariants import decision_protocol, never
from gabion.order_contract import OrderPolicy, sort_once
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
    LspParityGateResponseDTO,
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
from gabion.tooling.governance_rules import GovernanceRules, load_governance_rules

server = LanguageServer("gabion", "0.1.0")
CHECK_COMMAND = command_ids.CHECK_COMMAND
DATAFLOW_COMMAND = command_ids.DATAFLOW_COMMAND
SYNTHESIS_COMMAND = command_ids.SYNTHESIS_COMMAND
REFACTOR_COMMAND = command_ids.REFACTOR_COMMAND
STRUCTURE_DIFF_COMMAND = command_ids.STRUCTURE_DIFF_COMMAND
STRUCTURE_REUSE_COMMAND = command_ids.STRUCTURE_REUSE_COMMAND
DECISION_DIFF_COMMAND = command_ids.DECISION_DIFF_COMMAND
IMPACT_COMMAND = command_ids.IMPACT_COMMAND
LSP_PARITY_GATE_COMMAND = command_ids.LSP_PARITY_GATE_COMMAND

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
_ANALYSIS_INPUT_MANIFEST_FORMAT_VERSION = 1
_ANALYSIS_INPUT_WITNESS_FORMAT_VERSION = 2
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
_COLLECTION_CHECKPOINT_FLUSH_INTERVAL_NS = 2_000_000_000
_COLLECTION_CHECKPOINT_MEANINGFUL_MIN_INTERVAL_NS = 1_000_000_000
_COLLECTION_REPORT_FLUSH_INTERVAL_NS = 10_000_000_000
_COLLECTION_REPORT_FLUSH_COMPLETED_STRIDE = 8
_DEFAULT_DEADLINE_PROFILE_SAMPLE_INTERVAL = 16
_DEFAULT_PROGRESS_HEARTBEAT_SECONDS = 55.0
_MIN_PROGRESS_HEARTBEAT_SECONDS = 5.0
_PROGRESS_DEADLINE_FLUSH_SECONDS = 5.0
_PROGRESS_DEADLINE_WATCHDOG_SECONDS = 10.0
_PROGRESS_HEARTBEAT_POLL_SECONDS = 0.05
_PROGRESS_DEADLINE_FLUSH_MARGIN_SECONDS = 0.5
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
    limit = getattr(clock, "limit", None)
    current = getattr(clock, "current", None)
    if isinstance(limit, int) and isinstance(current, int):
        return (limit - current) > 1
    return True


@dataclass(frozen=True)
class ExecuteCommandDeps:
    analyze_paths_fn: Callable[..., AnalysisResult]
    load_aspf_resume_state_fn: Callable[..., JSONObject | None]
    analysis_input_manifest_fn: Callable[..., JSONObject]
    analysis_input_manifest_digest_fn: Callable[[JSONObject], str]
    build_analysis_collection_resume_seed_fn: Callable[..., JSONObject]
    collection_semantic_progress_fn: Callable[..., JSONObject]
    project_report_sections_fn: Callable[..., dict[str, list[str]]]
    report_projection_spec_rows_fn: Callable[[], list[JSONObject]]
    collection_checkpoint_flush_due_fn: Callable[..., bool]
    write_bootstrap_incremental_artifacts_fn: Callable[..., None]
    load_report_section_journal_fn: Callable[..., tuple[dict[str, list[str]], str | None]]
    start_trace_fn: Callable[..., object]
    record_1cell_fn: Callable[..., object]
    record_2cell_witness_fn: Callable[..., object]
    record_cofibration_fn: Callable[..., object]
    merge_imported_trace_fn: Callable[..., object]
    finalize_trace_fn: Callable[..., object]

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
            for key in sort_once(
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
            items = sort_once(
                items,
                source="server._normalize_ast_value.set_items",
                # Non-lexical comparator: canonical JSON text for mixed scalar/JSON-ish values.
                key=_canonical_json_text,
            )
            return {"_py": "set", "items": items}
        if isinstance(value, frozenset):
            items = [_normalize_ast_value(item) for item in value]
            items = sort_once(
                items,
                source="server._normalize_ast_value.frozenset_items",
                # Non-lexical comparator: canonical JSON text for mixed scalar/JSON-ish values.
                key=_canonical_json_text,
            )
            return {"_py": "frozenset", "items": items}
        return _normalize_scalar(value)

    ast_intern_table: JSONObject = {}
    ast_identity_forest = Forest()
    ast_ref_by_node_id: dict[NodeId, str] = {}
    ast_ref_counter = 0

    def _intern_ast(normalized_tree: JSONValue) -> str:
        nonlocal ast_ref_counter
        structural_identity = structural_key_atom(
            normalized_tree,
            source="server._analysis_input_witness.ast",
        )
        node_id = ast_identity_forest.add_node(
            "AstWitness",
            (structural_identity,),
        )
        cached_ref = ast_ref_by_node_id.get(node_id)
        if cached_ref is not None:
            return cached_ref
        ast_ref_counter += 1
        ast_ref = f"ast:{ast_ref_counter}"
        ast_ref_by_node_id[node_id] = ast_ref
        ast_intern_table[ast_ref] = cast(JSONValue, normalized_tree)
        return ast_ref

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
    return json.dumps(payload, sort_keys=False, separators=(",", ":"), ensure_ascii=True)




def _load_aspf_resume_state(
    *,
    import_state_paths: Sequence[Path],
) -> JSONObject | None:
    projection, records = aspf_resume_state.load_resume_projection_from_state_files(
        state_paths=import_state_paths
    )
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
    payload: JSONObject = {
        "resume_projection": projection if projection is not None else {},
        "delta_records": [dict(record) for record in records],
        "analysis_manifest_digest": latest_manifest_digest,
        "resume_source": latest_resume_source,
    }
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
        handle.write(json.dumps(progress_value, sort_keys=False) + "\n")
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
        "stale_for_s",
        "dimensions",
    ]


def _phase_timeline_header_block() -> str:
    header = _phase_timeline_header_columns()
    header_line = "| " + " | ".join(_markdown_table_cell(cell) for cell in header) + " |"
    separator_line = "| " + " | ".join(["---"] * len(header)) + " |"
    return header_line + "\n" + separator_line


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
    return boundary_order.normalize_boundary_mapping_once(
        payload,
        source=f"server._require_payload.{command}",
    )


def _require_optional_payload(
    payload: Mapping[str, object] | None,
    *,
    command: str,
) -> dict[str, object]:
    if payload is None:
        never(
            "invalid command payload type",
            payload_type="NoneType",
        )
    return _require_payload(payload, command=command)


def _ordered_command_response(
    response: Mapping[str, object],
    *,
    command: str,
) -> dict[str, object]:
    return boundary_order.normalize_boundary_mapping_once(
        response,
        source=f"server._ordered_command_response.{command}",
    )


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


def _parse_lint_line_as_payload(line: str) -> dict[str, object] | None:
    entry = _parse_lint_line(line)
    if entry is None:
        return None
    return entry.model_dump()


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
    base = DataflowAuditResponseDTO(
        exit_code=int(response.get("exit_code", 0) or 0),
        timeout=bool(response.get("timeout", False)),
        analysis_state=(str(response.get("analysis_state")) if response.get("analysis_state") is not None else None),
        errors=[str(err) for err in (response.get("errors") or [])] if isinstance(response.get("errors"), list) else [],
        lint_lines=lint_lines,
        lint_entries=[LintEntryDTO.model_validate(entry) for entry in lint_entries],
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
    normalized["lint_entries"] = [entry.model_dump() for entry in base.lint_entries]
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
                ordered_bundle = sort_once(
                    bundle,
                    source="server._lint_bundles.bundle",
                    key=str,
                )
                ordered_bundle = sort_once(
                    ordered_bundle,
                    source="server._lint_bundles.bundle_enforce",
                    policy=OrderPolicy.ENFORCE,
                )
                message = f"Implicit bundle detected: {', '.join(ordered_bundle)}"
                for name in sort_once(
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
        load_aspf_resume_state_fn=_load_aspf_resume_state,
        analysis_input_manifest_fn=_analysis_input_manifest,
        analysis_input_manifest_digest_fn=_analysis_input_manifest_digest,
        build_analysis_collection_resume_seed_fn=build_analysis_collection_resume_seed,
        collection_semantic_progress_fn=_collection_semantic_progress,
        project_report_sections_fn=project_report_sections,
        report_projection_spec_rows_fn=report_projection_spec_rows,
        collection_checkpoint_flush_due_fn=_collection_checkpoint_flush_due,
        write_bootstrap_incremental_artifacts_fn=_write_bootstrap_incremental_artifacts,
        load_report_section_journal_fn=_load_report_section_journal,
        start_trace_fn=aspf_execution_fibration.start_execution_trace,
        record_1cell_fn=aspf_execution_fibration.record_1cell,
        record_2cell_witness_fn=aspf_execution_fibration.record_2cell_witness,
        record_cofibration_fn=aspf_execution_fibration.record_cofibration,
        merge_imported_trace_fn=aspf_execution_fibration.merge_imported_trace,
        finalize_trace_fn=aspf_execution_fibration.finalize_execution_trace,
    )


def _invariant_error_message(error: NeverThrown) -> str:
    message = str(error).strip()
    if message:
        return message
    return "invariant violation"


@decision_protocol
def _invariant_failure_dataflow_response(
    *,
    command: str,
    error: NeverThrown,
) -> dict[str, object]:
    normalized = _normalize_dataflow_response(
        {
            "exit_code": 2,
            "timeout": False,
            "analysis_state": "failed",
            "classification": "failed",
            "error_kind": "invariant_violation",
            "errors": [_invariant_error_message(error)],
            "lint_lines": [],
            "lint_entries": [],
        }
    )
    return _ordered_command_response(normalized, command=command)


@decision_protocol
def _execute_dataflow_command_boundary(
    ls: LanguageServer,
    payload: dict | None,
    *,
    deps: ExecuteCommandDeps | None = None,
) -> dict:
    try:
        normalized_payload = _require_optional_payload(payload, command=DATAFLOW_COMMAND)
        normalized_result = _execute_command_total(ls, normalized_payload, deps=deps)
        return _ordered_command_response(
            normalized_result,
            command=DATAFLOW_COMMAND,
        )
    except NeverThrown as error:
        return _invariant_failure_dataflow_response(
            command=DATAFLOW_COMMAND,
            error=error,
        )


@server.command(CHECK_COMMAND)
@server.command(DATAFLOW_COMMAND)
def execute_command(
    ls: LanguageServer,
    payload: dict | None = None,
) -> dict:
    return _execute_dataflow_command_boundary(ls, payload)


def execute_command_with_deps(
    ls: LanguageServer,
    payload: dict | None = None,
    *,
    deps: ExecuteCommandDeps | None = None,
) -> dict:
    return _execute_dataflow_command_boundary(ls, payload, deps=deps)


def _execute_command_total(
    ls: LanguageServer,
    payload: dict[str, object],
    *,
    deps: ExecuteCommandDeps | None = None,
) -> dict:
    from gabion.server_core.command_orchestrator import execute_command_total

    return execute_command_total(ls, payload, deps=deps)


@server.command(SYNTHESIS_COMMAND)
def execute_synthesis(
    ls: LanguageServer,
    payload: dict | None = None,
) -> dict:
    normalized_payload = _require_optional_payload(payload, command=SYNTHESIS_COMMAND)
    return _ordered_command_response(
        _execute_synthesis_total(ls, normalized_payload),
        command=SYNTHESIS_COMMAND,
    )


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
                            "source_params": sort_once(field.source_params, source = 'src/gabion/server.py:5987'),
                        }
                        for field in spec.fields
                    ],
                    "bundle": sort_once(spec.bundle, source = 'src/gabion/server.py:5991'),
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
    return _ordered_command_response(
        _execute_refactor_total(ls, normalized_payload),
        command=REFACTOR_COMMAND,
    )


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
    return _ordered_command_response(
        _execute_structure_diff_total(ls, normalized_payload),
        command=STRUCTURE_DIFF_COMMAND,
    )


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
    return _ordered_command_response(
        _execute_structure_reuse_total(ls, normalized_payload),
        command=STRUCTURE_REUSE_COMMAND,
    )


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
    return _ordered_command_response(
        _execute_decision_diff_total(ls, normalized_payload),
        command=DECISION_DIFF_COMMAND,
    )


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
        entries[:] = sort_once(
            entries,
            source="server._impact_collect_edges.by_name_entries",
            # Lexical path/qual pair order for deterministic edge expansion.
            key=lambda fn: (fn.path, fn.qual),
        )
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



def _lsp_command_executor(command: str) -> Callable[[LanguageServer, dict[str, object] | None], dict] | None:
    mapping: dict[str, Callable[[LanguageServer, dict[str, object] | None], dict]] = {
        CHECK_COMMAND: execute_command,
        DATAFLOW_COMMAND: execute_command,
        STRUCTURE_DIFF_COMMAND: execute_structure_diff,
        STRUCTURE_REUSE_COMMAND: execute_structure_reuse,
        DECISION_DIFF_COMMAND: execute_decision_diff,
        SYNTHESIS_COMMAND: execute_synthesis,
        REFACTOR_COMMAND: execute_refactor,
        IMPACT_COMMAND: execute_impact,
    }
    return mapping.get(command)


def _strip_parity_ignored_keys(
    payload: Mapping[str, object],
    *,
    ignored_keys: tuple[str, ...],
) -> dict[str, object]:
    if not ignored_keys:
        return dict(payload)
    ignored = set(ignored_keys)
    return {key: value for key, value in payload.items() if key not in ignored}


def _normalize_probe_payload(
    probe_payload: Mapping[str, object],
    *,
    root: Path,
    command: str,
) -> dict[str, object]:
    payload = boundary_order.normalize_boundary_mapping_once(
        probe_payload,
        source=f"server.lsp_parity_gate.{command}.probe_payload",
    )
    if "root" not in payload:
        payload = boundary_order.apply_boundary_updates_once(
            payload,
            {"root": str(root)},
            source=f"server.lsp_parity_gate.{command}.probe_payload_root",
        )
    if (
        "analysis_timeout_ticks" not in payload
        and "analysis_timeout_ms" not in payload
        and "analysis_timeout_seconds" not in payload
    ):
        payload = boundary_order.apply_boundary_updates_once(
            payload,
            {"analysis_timeout_ticks": 100, "analysis_timeout_tick_ns": 1_000_000},
            source=f"server.lsp_parity_gate.{command}.probe_payload_timeout",
        )
    return payload


def _execute_lsp_parity_gate_total(
    ls: LanguageServer,
    payload: dict[str, object],
    *,
    load_rules: Callable[[], GovernanceRules] = load_governance_rules,
    lsp_executor_for_command: Callable[
        [str],
        Callable[[LanguageServer, dict[str, object] | None], dict] | None,
    ] | None = None,
    direct_executor_for_command: Callable[
        [str],
        Callable[[LanguageServer, dict[str, object] | None], dict] | None,
    ] | None = None,
) -> dict:
    rules = load_rules()
    resolved_lsp_executor_for_command = (
        _lsp_command_executor
        if lsp_executor_for_command is None
        else lsp_executor_for_command
    )
    resolved_direct_executor_for_command = (
        direct_dispatch.direct_executor
        if direct_executor_for_command is None
        else direct_executor_for_command
    )
    root = Path(str(payload.get("root") or ls.workspace.root_path or "."))
    selected_commands = list(payload_codec.normalized_command_id_list(payload, key="commands"))
    if not selected_commands:
        selected_commands = list(sort_once(rules.command_policies.keys(), source="server.lsp_parity_gate.selected_default"))
    checked_commands: list[dict[str, object]] = []
    errors: list[str] = []
    for command in selected_commands:
        policy = rules.command_policies.get(command)
        if policy is None:
            errors.append(f"missing command policy for {command}")
            continue
        lsp_validated = False
        parity_ok = True
        error: str | None = None
        probe_payload = policy.probe_payload
        if probe_payload is not None:
            normalized_probe = _normalize_probe_payload(probe_payload, root=root, command=command)
            lsp_executor = resolved_lsp_executor_for_command(command)
            direct_executor = resolved_direct_executor_for_command(command)
            if lsp_executor is None:
                error = f"no LSP executor registered for {command}"
            elif direct_executor is None:
                error = f"no direct executor registered for {command}"
            else:
                try:
                    lsp_result = boundary_order.normalize_boundary_mapping_once(
                        lsp_executor(ls, dict(normalized_probe)),
                        source=f"server.lsp_parity_gate.{command}.lsp_result",
                    )
                    lsp_validated = True
                    direct_result = boundary_order.normalize_boundary_mapping_once(
                        direct_executor(ls, dict(normalized_probe)),
                        source=f"server.lsp_parity_gate.{command}.direct_result",
                    )
                    lsp_comparable = boundary_order.normalize_boundary_mapping_once(
                        _strip_parity_ignored_keys(lsp_result, ignored_keys=policy.parity_ignore_keys),
                        source=f"server.lsp_parity_gate.{command}.lsp_comparable",
                    )
                    direct_comparable = boundary_order.normalize_boundary_mapping_once(
                        _strip_parity_ignored_keys(direct_result, ignored_keys=policy.parity_ignore_keys),
                        source=f"server.lsp_parity_gate.{command}.direct_comparable",
                    )
                    parity_ok = lsp_comparable == direct_comparable
                    if policy.parity_required and not parity_ok:
                        error = f"parity mismatch for {command}"
                except NeverThrown as exc:
                    error = str(exc) or f"invariant violation while probing {command}"
                except Exception as exc:  # pragma: no cover - defensive conversion
                    error = str(exc)
        if policy.require_lsp_carrier and not lsp_validated and error is None:
            error = f"beta/production command requires LSP validation: {command}"
        if error is not None:
            errors.append(error)
        checked_commands.append({
            "command_id": command,
            "maturity": policy.maturity,
            "require_lsp_carrier": policy.require_lsp_carrier,
            "parity_required": policy.parity_required,
            "lsp_validated": lsp_validated,
            "parity_ok": parity_ok,
            "error": error,
        })
    exit_code = 1 if errors else 0
    return LspParityGateResponseDTO(
        exit_code=exit_code,
        checked_commands=checked_commands,
        errors=errors,
    ).model_dump()


@server.command(LSP_PARITY_GATE_COMMAND)
def execute_lsp_parity_gate(
    ls: LanguageServer,
    payload: dict | None = None,
) -> dict:
    normalized_payload = _require_optional_payload(payload, command=LSP_PARITY_GATE_COMMAND)
    return _ordered_command_response(
        _execute_lsp_parity_gate_total(ls, normalized_payload),
        command=LSP_PARITY_GATE_COMMAND,
    )


@server.command(IMPACT_COMMAND)
def execute_impact(
    ls: LanguageServer,
    payload: dict[str, object] | None = None,
) -> dict:
    normalized_payload = _require_optional_payload(payload, command=IMPACT_COMMAND)
    return _ordered_command_response(
        _execute_impact_total(ls, normalized_payload),
        command=IMPACT_COMMAND,
    )


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

        py_paths = sort_once(
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
            for qual in sort_once(
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
            for item in sort_once(
                likely_tests.values(),
                source="_execute_impact_total.likely_tests",
                key=lambda item: str(item["id"]),
            )
            if float(item.get("confidence", 0.0)) >= confidence_threshold
        ]
        docs: list[dict[str, object]] = []
        impacted_names = {functions[qual].name for qual in seed_functions if qual in functions}
        for md_path in deadline_loop_iter(
            sort_once(
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
                matches = sort_once(
                    {name for name in impacted_names if name.lower() in lowered},
                    source="_execute_impact_total.doc_matches",
                )
                if not matches:
                    continue
                docs.append({"path": rel_path, "section": heading, "symbols": matches})

        return {
            "exit_code": 0,
            "changes": normalized_changes,
            "seed_functions": sort_once(
                seed_functions,
                source="_execute_impact_total.seed_functions_out",
            ),
            "must_run_tests": [
                must_tests[key]
                for key in sort_once(
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

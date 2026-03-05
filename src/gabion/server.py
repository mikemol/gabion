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
from typing import Callable, Literal, Mapping, Protocol, Sequence, cast
from urllib.parse import unquote, urlparse

from pygls.lsp.server import LanguageServer
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from lsprotocol.types import (
    TEXT_DOCUMENT_DID_OPEN, TEXT_DOCUMENT_DID_SAVE, TEXT_DOCUMENT_CODE_ACTION, CodeAction, CodeActionKind, CodeActionParams, Command, Diagnostic, DiagnosticSeverity, Position, Range, WorkspaceEdit)

from gabion.json_types import JSONObject, JSONValue
from gabion.commands import (
    boundary_order, command_ids, payload_codec, progress_contract as progress_timeline)
from gabion.commands.dispatch_registry import (
    CommandDispatchRegistration,
    CommandExecutorRefs,
    build_command_dispatch_registry,
    executor_for_transport,
)
from gabion.commands.lint_parser import parse_lint_line
from gabion.commands.check_contract import LintEntriesDecision
from gabion.plan import (
    ExecutionPlan, ExecutionPlanObligations, ExecutionPlanPolicyMetadata, write_execution_plan_artifact)

from gabion.analysis import (
    AnalysisResult, AuditConfig, ReportCarrier, analyze_paths, apply_baseline, build_analysis_collection_resume_seed, compute_structure_metrics, compute_structure_reuse, render_reuse_lemma_stubs, compute_violations, build_refactor_plan, build_synthesis_plan, diff_structure_snapshots, diff_decision_snapshots, load_structure_snapshot, load_decision_snapshot, load_baseline, extract_report_sections, project_report_sections, report_projection_phase_rank, report_projection_spec_rows, render_dot, render_structure_snapshot, render_decision_snapshot, DecisionSnapshotSurfaces, render_protocol_stubs, render_refactor_plan, render_report, render_synthesis_section, resolve_analysis_paths, resolve_baseline_path, write_baseline)
from gabion.analysis.aspf import aspf_execution_fibration, aspf_resume_state
from gabion.analysis.aspf.aspf import Forest, NodeId, structural_key_atom
from gabion.analysis.core import ambiguity_delta
from gabion.analysis.core import ambiguity_state
from gabion.analysis.call_cluster import call_cluster_consolidation
from gabion.analysis.call_cluster import call_clusters
from gabion.analysis.semantics import semantic_coverage_map
from gabion.analysis.taint import taint_delta
from gabion.analysis.taint import taint_lifecycle
from gabion.analysis.taint import taint_projection
from gabion.analysis.taint import taint_state
from gabion.analysis.surfaces import test_annotation_drift
from gabion.analysis.surfaces import test_annotation_drift_delta
from gabion.analysis.surfaces import test_obsolescence
from gabion.analysis.surfaces import test_obsolescence_delta
from gabion.analysis.surfaces import test_obsolescence_state
from gabion.analysis.surfaces import test_evidence_suggestions
from gabion.analysis.foundation.timeout_context import (
    Deadline, GasMeter, TimeoutExceeded, check_deadline, deadline_loop_iter, get_deadline, get_deadline_clock, record_deadline_io, reset_deadline_clock, forest_scope, reset_forest, set_forest, reset_deadline_profile, reset_deadline, set_deadline_profile, set_deadline, set_deadline_clock)
from gabion.exceptions import NeverThrown
from gabion.invariants import decision_protocol, never
from gabion.order_contract import OrderPolicy, sort_once
from gabion.config import (
    dataflow_defaults, dataflow_deadline_roots, decision_defaults, decision_ignore_list, decision_require_tiers, decision_tier_map, exception_defaults, exception_never_list, fingerprint_defaults, taint_boundary_registry, taint_defaults, taint_profile, merge_payload)
from gabion.analysis.core.type_fingerprints import (
    Fingerprint, PrimeRegistry, TypeConstructorRegistry, build_fingerprint_registry)
from gabion.refactor import (
    FieldSpec, LoopGeneratorRequest as LoopGeneratorRequestModel, RefactorEngine, RefactorCompatibilityShimConfig, RefactorRequest as RefactorRequestModel)
from gabion.refactor.rewrite_plan import normalize_rewrite_plan_order, validate_rewrite_plan_payload
from gabion.schema import (
    DataflowResponseEnvelopeDTO, LegacyDataflowMonolithResponseDTO, DecisionDiffResponseDTO, LspParityGateResponseDTO, LintEntryDTO, RefactorProtocolResponseDTO, RefactorRequest, RefactorResponse, RewritePlanEntryDTO, StructureDiffResponseDTO, StructureReuseResponseDTO, SynthesisPlanResponseDTO, SynthesisResponse, SynthesisRequest, TextEditDTO)
from gabion.server_core import command_orchestrator_primitives as orchestrator_primitives
from gabion.server_core import dataflow_runtime_contract as runtime_contract
from gabion.synthesis import NamingContext, SynthesisConfig, Synthesizer
from gabion.tooling.governance.governance_rules import GovernanceRules, load_governance_rules

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

_ANALYSIS_INPUT_MANIFEST_FORMAT_VERSION = 1
_ANALYSIS_INPUT_WITNESS_FORMAT_VERSION = 2
_DEFAULT_PHASE_TIMELINE_MD = runtime_contract.DEFAULT_PHASE_TIMELINE_MD
_DEFAULT_PHASE_TIMELINE_JSONL = runtime_contract.DEFAULT_PHASE_TIMELINE_JSONL
_REPORT_SECTION_JOURNAL_FORMAT_VERSION = 1
_DEFAULT_REPORT_SECTION_JOURNAL = runtime_contract.DEFAULT_REPORT_SECTION_JOURNAL
_COLLECTION_CHECKPOINT_FLUSH_INTERVAL_NS = runtime_contract.COLLECTION_CHECKPOINT_FLUSH_INTERVAL_NS
_COLLECTION_CHECKPOINT_MEANINGFUL_MIN_INTERVAL_NS = runtime_contract.COLLECTION_CHECKPOINT_MEANINGFUL_MIN_INTERVAL_NS
_COLLECTION_REPORT_FLUSH_INTERVAL_NS = runtime_contract.COLLECTION_REPORT_FLUSH_INTERVAL_NS
_COLLECTION_REPORT_FLUSH_COMPLETED_STRIDE = runtime_contract.COLLECTION_REPORT_FLUSH_COMPLETED_STRIDE
_DEFAULT_PROGRESS_HEARTBEAT_SECONDS = runtime_contract.DEFAULT_PROGRESS_HEARTBEAT_SECONDS
_MIN_PROGRESS_HEARTBEAT_SECONDS = runtime_contract.MIN_PROGRESS_HEARTBEAT_SECONDS
_PROGRESS_DEADLINE_FLUSH_SECONDS = runtime_contract.PROGRESS_DEADLINE_FLUSH_SECONDS
_PROGRESS_DEADLINE_WATCHDOG_SECONDS = runtime_contract.PROGRESS_DEADLINE_WATCHDOG_SECONDS
_PROGRESS_HEARTBEAT_POLL_SECONDS = runtime_contract.PROGRESS_HEARTBEAT_POLL_SECONDS
_PROGRESS_DEADLINE_FLUSH_MARGIN_SECONDS = runtime_contract.PROGRESS_DEADLINE_FLUSH_MARGIN_SECONDS
_LSP_PROGRESS_NOTIFICATION_METHOD = runtime_contract.LSP_PROGRESS_NOTIFICATION_METHOD
_LSP_PROGRESS_TOKEN_V2 = runtime_contract.LSP_PROGRESS_TOKEN_V2
_LSP_PROGRESS_TOKEN = _LSP_PROGRESS_TOKEN_V2
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
    return orchestrator_primitives._analysis_resume_cache_verdict(
        status=status,
        reused_files=reused_files,
        compatibility_status=compatibility_status,
    )


def _deadline_tick_budget_allows_check(clock: object) -> bool:
    return runtime_contract.deadline_tick_budget_allows_check(clock)


# Boundary aliases preserve server.py test/import surface while converging
# execution primitives in server_core.
ExecuteCommandDeps = orchestrator_primitives.ExecuteCommandDeps
_analysis_input_manifest = orchestrator_primitives._analysis_input_manifest
_analysis_input_manifest_digest = (
    orchestrator_primitives._analysis_input_manifest_digest
)
_collection_semantic_progress = orchestrator_primitives._collection_semantic_progress
_materialize_execution_plan = orchestrator_primitives._materialize_execution_plan
_default_execute_command_deps = orchestrator_primitives._default_execute_command_deps


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
    return orchestrator_primitives._analysis_resume_progress(
        collection_resume=collection_resume,
        total_files=total_files,
    )


def _normalize_progress_work(
    *,
    work_done: object | None,
    work_total: object | None,
) -> tuple[int | None, int | None]:
    return orchestrator_primitives._normalize_progress_work(
        work_done=work_done,
        work_total=work_total,
    )

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
    return orchestrator_primitives._build_phase_progress_v2(
        phase=phase,
        collection_progress=collection_progress,
        semantic_progress=semantic_progress,
        work_done=work_done,
        work_total=work_total,
        phase_progress_v2=phase_progress_v2,
    )


def _completed_path_set(
    collection_resume: CollectionResumeDTO | None,
) -> set[str]:
    if collection_resume is None:
        return set()
    return set(collection_resume.completed_paths)


def _in_progress_scan_states(
    collection_resume: CollectionResumeDTO | None,
) -> dict[str, InProgressScanStateDTO]:
    states: dict[str, InProgressScanStateDTO] = {}
    if collection_resume is None:
        return states
    previous_path: str | None = None
    for raw_path, raw_state in collection_resume.in_progress_scan_by_path.items():
        check_deadline()
        if previous_path is not None and previous_path > raw_path:
            never(
                "in_progress_scan_by_path path order regression",
                previous_path=previous_path,
                current_path=raw_path,
            )
        previous_path = raw_path
        states[raw_path] = raw_state
    return states


def _state_processed_functions(state: InProgressScanStateDTO) -> set[str]:
    return set(state.processed_functions)


def _state_processed_count(state: InProgressScanStateDTO) -> int:
    processed_functions = _state_processed_functions(state)
    if processed_functions:
        return len(processed_functions)
    if isinstance(state.processed_functions_count, int):
        return max(0, state.processed_functions_count)
    return 0


def _state_processed_digest(state: InProgressScanStateDTO) -> str:
    processed_functions = _state_processed_functions(state)
    if processed_functions:
        return hashlib.sha1(
            _canonical_json_text(sort_once(processed_functions, source = 'src/gabion/server.py:1371')).encode("utf-8")
        ).hexdigest()
    if isinstance(state.processed_functions_digest, str) and state.processed_functions_digest:
        return state.processed_functions_digest
    return hashlib.sha1(
        _canonical_json_text({"count": _state_processed_count(state)}).encode("utf-8")
    ).hexdigest()


def _analysis_index_resume_hydrated_paths(
    collection_resume: CollectionResumeDTO | None,
) -> set[str]:
    if collection_resume is None or collection_resume.analysis_index_resume is None:
        return set()
    return set(collection_resume.analysis_index_resume.hydrated_paths)


def _analysis_index_resume_hydrated_count(
    collection_resume: CollectionResumeDTO | None,
) -> int:
    hydrated = _analysis_index_resume_hydrated_paths(collection_resume)
    if hydrated:
        return len(hydrated)
    if collection_resume is None or collection_resume.analysis_index_resume is None:
        return 0
    raw_count = collection_resume.analysis_index_resume.hydrated_paths_count
    if isinstance(raw_count, int):
        return max(0, raw_count)
    return 0


def _analysis_index_resume_hydrated_digest(
    collection_resume: CollectionResumeDTO | None,
) -> str:
    hydrated = _analysis_index_resume_hydrated_paths(collection_resume)
    if hydrated:
        return hashlib.sha1(
            _canonical_json_text(sort_once(hydrated, source = 'src/gabion/server.py:1418')).encode("utf-8")
        ).hexdigest()
    if collection_resume is None or collection_resume.analysis_index_resume is None:
        return hashlib.sha1(b"[]").hexdigest()
    raw_digest = collection_resume.analysis_index_resume.hydrated_paths_digest
    if isinstance(raw_digest, str) and raw_digest:
        return raw_digest
    return hashlib.sha1(
        _canonical_json_text({"count": _analysis_index_resume_hydrated_count(collection_resume)}).encode("utf-8")
    ).hexdigest()


def _analysis_index_resume_signature(
    collection_resume: CollectionResumeDTO | None,
) -> tuple[int, str, int, int, str, str]:
    hydrated_count = _analysis_index_resume_hydrated_count(collection_resume)
    hydrated_digest = _analysis_index_resume_hydrated_digest(collection_resume)
    if collection_resume is None or collection_resume.analysis_index_resume is None:
        return (hydrated_count, hydrated_digest, 0, 0, "", hydrated_digest)
    analysis_index_resume = collection_resume.analysis_index_resume
    function_count = analysis_index_resume.function_count
    class_count = analysis_index_resume.class_count
    phase = analysis_index_resume.phase
    resume_digest = analysis_index_resume.resume_digest
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
    collection_resume: CollectionResumeDTO | None,
) -> JSONObject | None:
    if collection_resume is None:
        return None
    return _analysis_index_resume_summary_payload(collection_resume)


def _analysis_index_resume_summary_payload(
    collection_resume: CollectionResumeDTO,
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
    resume_dto = _collection_resume_dto(collection_resume)
    states = _in_progress_scan_states(resume_dto)
    state_rows: list[JSONObject] = []
    processed_total = 0
    for path_key, state in states.items():
        check_deadline()
        phase_text = state.phase if isinstance(state.phase, str) and state.phase else "unknown"
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
    index_signature = _analysis_index_resume_signature(resume_dto)
    digest = hashlib.sha1(
        _canonical_json_text(
            {
                "in_progress": state_rows,
                "index_hydrated_paths_count": _analysis_index_resume_hydrated_count(resume_dto),
                "index_hydrated_paths_digest": _analysis_index_resume_hydrated_digest(resume_dto),
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
        "index_hydrated_paths_count": _analysis_index_resume_hydrated_count(resume_dto),
        "index_hydrated_paths_digest": _analysis_index_resume_hydrated_digest(resume_dto),
        "index_resume_digest": index_signature[5],
        "index_function_count": index_signature[2],
        "index_class_count": index_signature[3],
    }


def _resolve_report_output_path(*, root: Path, report_path: str | None) -> Path | None:
    return orchestrator_primitives._resolve_report_output_path(
        root=root,
        report_path=report_path,
    )

def _resolve_report_section_journal_path(
    *,
    root: Path,
    report_path: str | None,
) -> Path | None:
    return orchestrator_primitives._resolve_report_section_journal_path(
        root=root,
        report_path=report_path,
    )

def _report_witness_digest(
    *,
    input_witness: Mapping[str, JSONValue] | None,
    manifest_digest: str | None,
) -> str | None:
    return orchestrator_primitives._report_witness_digest(
        input_witness=input_witness,
        manifest_digest=manifest_digest,
    )

def _coerce_section_lines(value: object) -> list[str]:
    return orchestrator_primitives._coerce_section_lines(value)

def _load_report_section_journal(
    *,
    path: Path | None,
    witness_digest: str | None,
) -> tuple[dict[str, list[str]], str | None]:
    return orchestrator_primitives._load_report_section_journal(
        path=path,
        witness_digest=witness_digest,
    )

def _write_report_section_journal(
    *,
    path: Path | None,
    witness_digest: str | None,
    projection_rows: Sequence[Mapping[str, JSONValue]],
    sections: Mapping[str, list[str]],
    pending_reasons: Mapping[str, str] | None = None,
) -> None:
    orchestrator_primitives._write_report_section_journal(
        path=path,
        witness_digest=witness_digest,
        projection_rows=projection_rows,
        sections=sections,
        pending_reasons=pending_reasons,
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
    resume_dto = _collection_resume_dto(collection_resume)
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
    in_progress_states = _in_progress_scan_states(resume_dto)
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
            phase_text = state_mapping.phase if isinstance(state_mapping.phase, str) and state_mapping.phase else "unknown"
            processed_count = _state_processed_count(state_mapping)
            function_count = state_mapping.function_count if isinstance(state_mapping.function_count, int) else 0
            if function_count == 0 and isinstance(state_mapping.fn_names, Mapping):
                function_count = 0
                for key in state_mapping.fn_names:
                    check_deadline()
                    if isinstance(key, str):
                        function_count += 1
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
    analysis_index_resume = resume_dto.analysis_index_resume if resume_dto is not None else None
    if analysis_index_resume is not None:
        hydrated_paths_count = analysis_index_resume.hydrated_paths_count
        if not isinstance(hydrated_paths_count, int):
            hydrated_paths_count = _analysis_index_resume_hydrated_count(resume_dto)
        lines.append(f"- `hydrated_paths_count`: `{hydrated_paths_count}`")
        if isinstance(analysis_index_resume.function_count, int):
            lines.append(f"- `hydrated_function_count`: `{analysis_index_resume.function_count}`")
        if isinstance(analysis_index_resume.class_count, int):
            lines.append(f"- `hydrated_class_count`: `{analysis_index_resume.class_count}`")
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
    return orchestrator_primitives._incremental_progress_obligations(
        analysis_state=analysis_state,
        progress_payload=progress_payload,
        resume_payload_available=resume_payload_available,
        partial_report_written=partial_report_written,
        report_requested=report_requested,
        projection_rows=projection_rows,
        sections=sections,
        pending_reasons=pending_reasons,
    )

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


def _parse_dataflow_command_payload(
    payload: Mapping[str, object] | None,
) -> DataflowCommandPayload:
    normalized_payload = _require_optional_payload(payload, command=DATAFLOW_COMMAND)
    normalized_payload = _normalize_dataflow_boundary_controls(normalized_payload)
    return DataflowCommandPayload(payload=normalized_payload)


def _parse_lsp_parity_gate_payload(
    payload: Mapping[str, object] | None,
) -> LspParityGatePayload:
    normalized_payload = _require_optional_payload(
        payload,
        command=LSP_PARITY_GATE_COMMAND,
    )
    return LspParityGatePayload(payload=normalized_payload)


def _parse_impact_command_payload(
    payload: Mapping[str, object] | None,
) -> ImpactCommandPayload:
    normalized_payload = _require_optional_payload(payload, command=IMPACT_COMMAND)
    return ImpactCommandPayload(payload=normalized_payload)


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
    return parse_lint_line(line)


def _parse_lint_line_as_payload(line: str) -> dict[str, object] | None:
    entry = _parse_lint_line(line)
    return entry.model_dump() if entry is not None else None


def _normalize_dataflow_boundary_controls(
    payload: dict[str, object],
) -> dict[str, object]:
    normalized_updates: dict[str, object] = {}
    for key in ("language", "ingest_profile"):
        raw_value = payload.get(key)
        if raw_value is None:
            continue
        if not isinstance(raw_value, str):
            never(  # pragma: no cover - invariant sink
                "invalid dataflow boundary control type",
                control=key,
                value_type=type(raw_value).__name__,
            )
        normalized_value = raw_value.strip().lower()
        if not normalized_value:
            never(  # pragma: no cover - invariant sink
                "empty dataflow boundary control",
                control=key,
            )
        normalized_updates[key] = normalized_value
    if not normalized_updates:
        return payload
    return boundary_order.apply_boundary_updates_once(
        payload,
        normalized_updates,
        source="server._normalize_dataflow_boundary_controls",
    )


def _normalize_dataflow_response_envelope(response: Mapping[str, object]) -> DataflowResponseEnvelopeDTO:
    return orchestrator_primitives._normalize_dataflow_response(response)


def _normalize_dataflow_response(response: Mapping[str, object]) -> dict[str, object]:
    return orchestrator_primitives._serialize_dataflow_response(
        _normalize_dataflow_response_envelope(response)
    )


def _truthy_flag(value: object) -> bool:
    return orchestrator_primitives._truthy_flag(value)


def _server_deadline_overhead_ns(
    total_ns: int,
    *,
    divisor: int | None = None,
) -> int:
    return orchestrator_primitives._server_deadline_overhead_ns(
        total_ns,
        divisor=divisor,
    )


def _analysis_timeout_total_ns(payload: Mapping[str, object] | MappingPayloadCarrier) -> int:
    return orchestrator_primitives._analysis_timeout_total_ns(dict(_payload_mapping(payload)))


def _analysis_timeout_total_ticks(payload: Mapping[str, object] | MappingPayloadCarrier) -> int:
    return orchestrator_primitives._analysis_timeout_total_ticks(dict(_payload_mapping(payload)))


def _analysis_timeout_grace_ns(
    payload: Mapping[str, object] | MappingPayloadCarrier,
    *,
    total_ns: int,
) -> int:
    return orchestrator_primitives._analysis_timeout_grace_ns(
        dict(_payload_mapping(payload)),
        total_ns=total_ns,
    )


def _analysis_timeout_budget_ns(
    payload: Mapping[str, object] | MappingPayloadCarrier,
) -> tuple[int, int, int]:
    return orchestrator_primitives._analysis_timeout_budget_ns(dict(_payload_mapping(payload)))


def _deadline_profile_sample_interval(
    payload: Mapping[str, object] | MappingPayloadCarrier,
    *,
    default_interval: int = 16,
) -> int:
    return orchestrator_primitives._deadline_profile_sample_interval(
        dict(_payload_mapping(payload)),
        default_interval=default_interval,
    )


def _deadline_from_payload(payload: Mapping[str, object] | MappingPayloadCarrier) -> Deadline:
    total_ns = _analysis_timeout_total_ns(payload)
    overhead_ns = _server_deadline_overhead_ns(total_ns)
    analysis_ns = max(1, total_ns - overhead_ns)
    return Deadline(deadline_ns=time.monotonic_ns() + analysis_ns)


@contextmanager
def _deadline_scope_from_payload(payload: Mapping[str, object] | MappingPayloadCarrier):
    normalized_payload = _require_payload(_payload_mapping(payload), command="deadline_scope")
    normalized_carrier = DataflowCommandPayload(payload=normalized_payload)
    deadline = _deadline_from_payload(normalized_carrier)
    base_ticks = _analysis_timeout_total_ticks(normalized_carrier)
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
                    if span is None:
                        start = Position(line=0, character=0)
                        end = Position(line=0, character=1)
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
    envelope = _normalize_dataflow_response_envelope(
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
    return _ordered_command_response(
        orchestrator_primitives._serialize_dataflow_response(envelope),
        command=command,
    )


@decision_protocol
def _execute_dataflow_command_boundary(
    ls: LanguageServer,
    payload: dict | None,
    *,
    deps: ExecuteCommandDeps | None = None,
) -> dict:
    try:
        command_payload = _parse_dataflow_command_payload(payload)
        normalized_result = _execute_command_total(ls, command_payload, deps=deps)
        return _ordered_command_response(
            orchestrator_primitives._serialize_dataflow_response(normalized_result),
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
    payload: DataflowCommandPayload,
    *,
    deps: ExecuteCommandDeps | None = None,
) -> DataflowResponseEnvelopeDTO:
    from gabion.server_core.command_orchestrator import execute_command_total

    return execute_command_total(ls, payload.payload, deps=deps)


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
            normalized_request_payload = dict(payload)
            normalized_request_payload.setdefault("kind", "protocol_extract")
            request = RefactorRequest.model_validate(normalized_request_payload)
        except ValidationError as exc:
            return RefactorProtocolResponseDTO(errors=[str(exc)]).model_dump()

        project_root = None
        if ls.workspace.root_path:
            project_root = Path(ls.workspace.root_path)
        engine = RefactorEngine(project_root=project_root)
        if request.kind == "loop_generator":
            plan = engine.plan_loop_generator_rewrite(
                LoopGeneratorRequestModel(
                    target_path=request.target_path,
                    target_functions=list(request.target_functions),
                    target_loop_lines=list(request.target_loop_lines),
                    rationale=request.rationale or "",
                )
            )
        else:
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
                    protocol_name=request.protocol_name or "",
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


@dataclass(frozen=True)
class ImpactPayloadDTO:
    root: Path
    max_call_depth: int | None
    confidence_threshold: float
    changes: tuple[ImpactSpan, ...]


@dataclass(frozen=True)
class ImpactEdgeBuckets:
    reverse_edges: dict[str, list[ImpactEdge]]
    unresolved_edges: tuple[dict[str, object], ...]


@dataclass(frozen=True)
class DataflowCommandPayload:
    payload: dict[str, object]


@dataclass(frozen=True)
class LspParityGatePayload:
    payload: dict[str, object]


@dataclass(frozen=True)
class ImpactCommandPayload:
    payload: dict[str, object]


@dataclass(frozen=True)
class ParityProbePayload:
    payload: dict[str, object]


class InProgressScanStateDTO(BaseModel):
    model_config = ConfigDict(extra="allow")

    phase: str | None = None
    processed_functions: tuple[str, ...] = ()
    processed_functions_count: object | None = None
    processed_functions_digest: object | None = None
    function_count: object | None = None
    fn_names: dict[str, object] | None = None


class AnalysisIndexResumeDTO(BaseModel):
    model_config = ConfigDict(extra="allow")

    hydrated_paths: tuple[str, ...] = ()
    hydrated_paths_count: object | None = None
    hydrated_paths_digest: object | None = None
    function_count: object | None = None
    class_count: object | None = None
    phase: object | None = None
    resume_digest: object | None = None


class CollectionResumeDTO(BaseModel):
    model_config = ConfigDict(extra="allow")

    completed_paths: tuple[str, ...] = ()
    in_progress_scan_by_path: dict[str, InProgressScanStateDTO] = Field(default_factory=dict)
    analysis_index_resume: AnalysisIndexResumeDTO | None = None


def _collection_resume_dto(
    collection_resume: Mapping[str, JSONValue] | None,
) -> CollectionResumeDTO | None:
    if not isinstance(collection_resume, Mapping):
        return None

    completed_paths: tuple[str, ...] = ()
    raw_completed_paths = collection_resume.get("completed_paths")
    if isinstance(raw_completed_paths, Sequence) and not isinstance(raw_completed_paths, (str, bytes)):
        completed_paths = tuple(path for path in raw_completed_paths if isinstance(path, str))

    in_progress_scan_by_path: dict[str, InProgressScanStateDTO] = {}
    raw_in_progress = collection_resume.get("in_progress_scan_by_path")
    if isinstance(raw_in_progress, Mapping):
        for raw_path, raw_state in raw_in_progress.items():
            if not isinstance(raw_path, str) or not isinstance(raw_state, Mapping):
                continue
            try:
                in_progress_scan_by_path[raw_path] = InProgressScanStateDTO.model_validate(raw_state)
            except ValidationError:
                continue

    analysis_index_resume: AnalysisIndexResumeDTO | None = None
    raw_analysis_index_resume = collection_resume.get("analysis_index_resume")
    if isinstance(raw_analysis_index_resume, Mapping):
        try:
            analysis_index_resume = AnalysisIndexResumeDTO.model_validate(raw_analysis_index_resume)
        except ValidationError:
            analysis_index_resume = None

    return CollectionResumeDTO(
        completed_paths=completed_paths,
        in_progress_scan_by_path=in_progress_scan_by_path,
        analysis_index_resume=analysis_index_resume,
    )


class MappingPayloadCarrier(Protocol):
    payload: Mapping[str, object]


def _payload_mapping(payload: Mapping[str, object] | MappingPayloadCarrier) -> Mapping[str, object]:
    if isinstance(payload, Mapping):
        return payload
    return payload.payload


class ParityProbeError(RuntimeError):
    pass


class LspProbeExecutionError(ParityProbeError):
    pass


class DirectProbeExecutionError(ParityProbeError):
    pass


def _normalize_impact_payload(
    payload: Mapping[str, object],
    *,
    workspace_root: str | None,
) -> ImpactPayloadDTO:
    root = Path(str(payload.get("root") or workspace_root or "."))

    max_call_depth_raw = payload.get("max_call_depth")
    try:
        max_call_depth = int(max_call_depth_raw) if max_call_depth_raw is not None else None
    except (TypeError, ValueError) as exc:
        raise ValueError("max_call_depth must be an integer") from exc
    if max_call_depth is not None and max_call_depth < 0:
        raise ValueError("max_call_depth must be non-negative")

    threshold_raw = payload.get("confidence_threshold", 0.5)
    try:
        confidence_threshold = float(threshold_raw)
    except (TypeError, ValueError) as exc:
        raise ValueError("confidence_threshold must be numeric") from exc
    normalized_threshold = max(0.0, min(1.0, confidence_threshold))

    changes: list[ImpactSpan] = []
    raw_changes = payload.get("changes")
    if isinstance(raw_changes, Sequence) and not isinstance(raw_changes, (str, bytes)):
        for entry in deadline_loop_iter(raw_changes):
            parsed = _normalize_impact_change_entry(entry)
            if parsed is not None:
                changes.append(
                    ImpactSpan(
                        path=parsed.path.replace("\\", "/"),
                        start_line=parsed.start_line,
                        end_line=parsed.end_line,
                    )
                )
    raw_diff = payload.get("git_diff")
    if isinstance(raw_diff, str) and raw_diff.strip():
        for diff_span in deadline_loop_iter(_parse_impact_diff_ranges(raw_diff)):
            changes.append(
                ImpactSpan(
                    path=diff_span.path.replace("\\", "/"),
                    start_line=diff_span.start_line,
                    end_line=diff_span.end_line,
                )
            )
    if not changes:
        raise ValueError("Provide at least one change span or git diff")

    return ImpactPayloadDTO(
        root=root,
        max_call_depth=max_call_depth,
        confidence_threshold=normalized_threshold,
        changes=tuple(changes),
    )


def _normalize_impact_edge_buckets(
    *,
    edges: Sequence[ImpactEdge],
    functions_by_qual: Mapping[str, ImpactFunction],
) -> ImpactEdgeBuckets:
    reverse_edges: dict[str, list[ImpactEdge]] = defaultdict(list)
    unresolved: list[dict[str, object]] = []
    for edge in deadline_loop_iter(edges):
        check_deadline()
        if edge.caller not in functions_by_qual or edge.callee not in functions_by_qual:
            unresolved.append(
                {
                    "caller": edge.caller,
                    "callee": edge.callee,
                    "reason": "unresolvable_function_id",
                }
            )
            continue
        reverse_edges[edge.callee].append(edge)
    return ImpactEdgeBuckets(reverse_edges=reverse_edges, unresolved_edges=tuple(unresolved))


def _probe_lsp_executor(
    executor: Callable[[LanguageServer, dict[str, object] | None], dict],
    *,
    ls: LanguageServer,
    command: str,
    probe_payload: ParityProbePayload,
) -> dict[str, object]:
    try:
        return boundary_order.normalize_boundary_mapping_once(
            executor(ls, dict(probe_payload.payload)),
            source=f"server.lsp_parity_gate.{command}.lsp_result",
        )
    except NeverThrown:
        raise
    except Exception as exc:
        raise LspProbeExecutionError(str(exc)) from exc


def _probe_direct_executor(
    executor: Callable[[LanguageServer, dict[str, object] | None], dict],
    *,
    ls: LanguageServer,
    command: str,
    probe_payload: ParityProbePayload,
) -> dict[str, object]:
    try:
        return boundary_order.normalize_boundary_mapping_once(
            executor(ls, dict(probe_payload.payload)),
            source=f"server.lsp_parity_gate.{command}.direct_result",
        )
    except NeverThrown:
        raise
    except Exception as exc:
        raise DirectProbeExecutionError(str(exc)) from exc


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


_COMMAND_DISPATCH_REGISTRY: dict[str, CommandDispatchRegistration] | None = None


def _command_dispatch_registry() -> dict[str, CommandDispatchRegistration]:
    global _COMMAND_DISPATCH_REGISTRY
    if _COMMAND_DISPATCH_REGISTRY is None:
        _COMMAND_DISPATCH_REGISTRY = build_command_dispatch_registry(
            CommandExecutorRefs(
                execute_command=cast(
                    Callable[[object, dict[str, object] | None], dict],
                    execute_command,
                ),
                execute_structure_diff=cast(
                    Callable[[object, dict[str, object] | None], dict],
                    execute_structure_diff,
                ),
                execute_structure_reuse=cast(
                    Callable[[object, dict[str, object] | None], dict],
                    execute_structure_reuse,
                ),
                execute_decision_diff=cast(
                    Callable[[object, dict[str, object] | None], dict],
                    execute_decision_diff,
                ),
                execute_synthesis=cast(
                    Callable[[object, dict[str, object] | None], dict],
                    execute_synthesis,
                ),
                execute_refactor=cast(
                    Callable[[object, dict[str, object] | None], dict],
                    execute_refactor,
                ),
                execute_impact=cast(
                    Callable[[object, dict[str, object] | None], dict],
                    execute_impact,
                ),
                execute_lsp_parity_gate=cast(
                    Callable[[object, dict[str, object] | None], dict],
                    execute_lsp_parity_gate,
                ),
            )
        )
    return _COMMAND_DISPATCH_REGISTRY


def _lsp_command_executor(command: str) -> Callable[[LanguageServer, dict[str, object] | None], dict] | None:
    executor = executor_for_transport(
        registry=_command_dispatch_registry(),
        command=command,
        transport="lsp",
    )
    return cast(Callable[[LanguageServer, dict[str, object] | None], dict] | None, executor)


def _direct_command_executor(
    command: str,
) -> Callable[[LanguageServer, dict[str, object] | None], dict] | None:
    executor = executor_for_transport(
        registry=_command_dispatch_registry(),
        command=command,
        transport="direct",
    )
    return cast(Callable[[LanguageServer, dict[str, object] | None], dict] | None, executor)


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
) -> ParityProbePayload:
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
    return ParityProbePayload(payload=payload)


def _execute_lsp_parity_gate_total(
    ls: LanguageServer,
    payload: LspParityGatePayload | Mapping[str, object],
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
        _direct_command_executor
        if direct_executor_for_command is None
        else direct_executor_for_command
    )
    command_payload = payload if isinstance(payload, LspParityGatePayload) else LspParityGatePayload(payload=_require_payload(payload, command=LSP_PARITY_GATE_COMMAND))
    root = Path(str(command_payload.payload.get("root") or ls.workspace.root_path or "."))
    selected_commands = list(payload_codec.normalized_command_id_list(command_payload.payload, key="commands"))
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
                    lsp_result = _probe_lsp_executor(
                        lsp_executor,
                        ls=ls,
                        command=command,
                        probe_payload=normalized_probe,
                    )
                    lsp_validated = True
                    direct_result = _probe_direct_executor(
                        direct_executor,
                        ls=ls,
                        command=command,
                        probe_payload=normalized_probe,
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
                except ParityProbeError as exc:
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
    command_payload = _parse_lsp_parity_gate_payload(payload)
    return _ordered_command_response(
        _execute_lsp_parity_gate_total(ls, command_payload),
        command=LSP_PARITY_GATE_COMMAND,
    )


@server.command(IMPACT_COMMAND)
def execute_impact(
    ls: LanguageServer,
    payload: dict[str, object] | None = None,
) -> dict:
    command_payload = _parse_impact_command_payload(payload)
    return _ordered_command_response(
        _execute_impact_total(ls, command_payload),
        command=IMPACT_COMMAND,
    )


def _execute_impact_total(ls: LanguageServer, payload: ImpactCommandPayload) -> dict:
    with _deadline_scope_from_payload(payload):
        try:
            options = _normalize_impact_payload(payload.payload, workspace_root=ls.workspace.root_path)
        except ValueError as exc:
            return {"exit_code": 2, "errors": [str(exc)]}
        root = options.root

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

        edge_buckets = _normalize_impact_edge_buckets(
            edges=_impact_collect_edges(functions_by_qual=functions, trees_by_path=trees_by_path),
            functions_by_qual=functions,
        )
        reverse_edges = edge_buckets.reverse_edges

        seed_functions: set[str] = set()
        normalized_changes: list[dict[str, object]] = []
        for change in deadline_loop_iter(options.changes):
            check_deadline()
            normalized_changes.append(
                {
                    "path": change.path,
                    "start_line": change.start_line,
                    "end_line": change.end_line,
                }
            )
            for fn in deadline_loop_iter(functions.values()):
                check_deadline()
                if fn.path != change.path:
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
            if options.max_call_depth is not None and depth >= options.max_call_depth:
                continue
            for edge in deadline_loop_iter(reverse_edges.get(current, [])):
                check_deadline()
                caller_fn = functions.get(edge.caller)
                if caller_fn is None:
                    never("reverse edge caller missing after normalization", caller=edge.caller)
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
            if float(item.get("confidence", 0.0)) >= options.confidence_threshold
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
                "max_call_depth": options.max_call_depth,
                "confidence_threshold": options.confidence_threshold,
                "unresolved_reverse_edges": list(edge_buckets.unresolved_edges),
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


if __name__ == "__main__":
    start()

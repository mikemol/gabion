from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Callable

from gabion.server_core import command_orchestrator_primitives as legacy


@dataclass(frozen=True)
class ContractConstantSpec:
    attr_name: str
    value: object


@dataclass(frozen=True)
class ContractCallableSpec:
    attr_name: str
    target: Callable[..., object]


@dataclass(frozen=True)
class PrimitiveContractSpec:
    contract_id: str
    constants: tuple[ContractConstantSpec, ...] = ()
    callables: tuple[ContractCallableSpec, ...] = ()


PRIMITIVE_CONTRACT_SPECS: dict[str, PrimitiveContractSpec] = {
    "AnalysisPrimitives": PrimitiveContractSpec(
        contract_id="AnalysisPrimitives",
        callables=(
            ContractCallableSpec("analysis_index_resume_hydrated_count", legacy._analysis_index_resume_hydrated_count),
            ContractCallableSpec("analysis_index_resume_signature", legacy._analysis_index_resume_signature),
            ContractCallableSpec("analysis_resume_cache_verdict", legacy._analysis_resume_cache_verdict),
            ContractCallableSpec("analysis_resume_progress", legacy._analysis_resume_progress),
            ContractCallableSpec("groups_by_path_from_collection_resume", legacy._groups_by_path_from_collection_resume),
            ContractCallableSpec("latest_report_phase", legacy._latest_report_phase),
            ContractCallableSpec("normalize_dataflow_response", legacy._normalize_dataflow_response),
            ContractCallableSpec("render_incremental_report", legacy._render_incremental_report),
            ContractCallableSpec("report_witness_digest", legacy._report_witness_digest),
        ),
    ),
    "TimeoutStageRuntime": PrimitiveContractSpec(
        contract_id="TimeoutStageRuntime",
        callables=(
            ContractCallableSpec("analysis_timeout_budget_ns", legacy._analysis_timeout_budget_ns),
            ContractCallableSpec("analysis_timeout_total_ticks", legacy._analysis_timeout_total_ticks),
            ContractCallableSpec("deadline_profile_sample_interval", legacy._deadline_profile_sample_interval),
            ContractCallableSpec("timeout_context_payload", legacy._timeout_context_payload),
        ),
    ),
    "ProgressStageContract": PrimitiveContractSpec(
        contract_id="ProgressStageContract",
        constants=(
            ContractConstantSpec("lsp_progress_notification_method", legacy._LSP_PROGRESS_NOTIFICATION_METHOD),
            ContractConstantSpec("lsp_progress_token_v2", legacy._LSP_PROGRESS_TOKEN_V2),
            ContractConstantSpec("canonical_progress_event_schema_v2", legacy._CANONICAL_PROGRESS_EVENT_SCHEMA_V2),
            ContractConstantSpec("progress_deadline_flush_margin_seconds", legacy._PROGRESS_DEADLINE_FLUSH_MARGIN_SECONDS),
            ContractConstantSpec("progress_deadline_flush_seconds", legacy._PROGRESS_DEADLINE_FLUSH_SECONDS),
            ContractConstantSpec("progress_deadline_watchdog_seconds", legacy._PROGRESS_DEADLINE_WATCHDOG_SECONDS),
            ContractConstantSpec("progress_heartbeat_poll_seconds", legacy._PROGRESS_HEARTBEAT_POLL_SECONDS),
        ),
        callables=(
            ContractCallableSpec("build_phase_progress_v2", legacy._build_phase_progress_v2),
            ContractCallableSpec("incremental_progress_obligations", legacy._incremental_progress_obligations),
            ContractCallableSpec("progress_heartbeat_seconds", legacy._progress_heartbeat_seconds),
        ),
    ),
    "ReportProjectionRuntime": PrimitiveContractSpec(
        contract_id="ReportProjectionRuntime",
        callables=(
            ContractCallableSpec("append_phase_timeline_event", legacy._append_phase_timeline_event),
            ContractCallableSpec("apply_journal_pending_reason", legacy._apply_journal_pending_reason),
            ContractCallableSpec("collection_components_preview_lines", legacy._collection_components_preview_lines),
            ContractCallableSpec("collection_progress_intro_lines", legacy._collection_progress_intro_lines),
            ContractCallableSpec("collection_report_flush_due", legacy._collection_report_flush_due),
            ContractCallableSpec("groups_by_path_from_collection_resume", legacy._groups_by_path_from_collection_resume),
            ContractCallableSpec("is_stdout_target", legacy._is_stdout_target),
            ContractCallableSpec("latest_report_phase", legacy._latest_report_phase),
            ContractCallableSpec("output_dirs", legacy._output_dirs),
            ContractCallableSpec("phase_timeline_header_block", legacy._phase_timeline_header_block),
            ContractCallableSpec("phase_timeline_jsonl_path", legacy._phase_timeline_jsonl_path),
            ContractCallableSpec("phase_timeline_md_path", legacy._phase_timeline_md_path),
            ContractCallableSpec("projection_phase_flush_due", legacy._projection_phase_flush_due),
            ContractCallableSpec("render_incremental_report", legacy._render_incremental_report),
            ContractCallableSpec("report_witness_digest", legacy._report_witness_digest),
            ContractCallableSpec("resolve_report_output_path", legacy._resolve_report_output_path),
            ContractCallableSpec("resolve_report_section_journal_path", legacy._resolve_report_section_journal_path),
            ContractCallableSpec("split_incremental_obligations", legacy._split_incremental_obligations),
            ContractCallableSpec("write_report_section_journal", legacy._write_report_section_journal),
            ContractCallableSpec("write_text_profiled", legacy._write_text_profiled),
        ),
    ),
    "OutputPrimitives": PrimitiveContractSpec(
        contract_id="OutputPrimitives",
        callables=(
            ContractCallableSpec("append_phase_timeline_event", legacy._append_phase_timeline_event),
            ContractCallableSpec("apply_journal_pending_reason", legacy._apply_journal_pending_reason),
            ContractCallableSpec("collection_components_preview_lines", legacy._collection_components_preview_lines),
            ContractCallableSpec("collection_progress_intro_lines", legacy._collection_progress_intro_lines),
            ContractCallableSpec("collection_report_flush_due", legacy._collection_report_flush_due),
            ContractCallableSpec("is_stdout_target", legacy._is_stdout_target),
            ContractCallableSpec("output_dirs", legacy._output_dirs),
            ContractCallableSpec("phase_timeline_header_block", legacy._phase_timeline_header_block),
            ContractCallableSpec("phase_timeline_jsonl_path", legacy._phase_timeline_jsonl_path),
            ContractCallableSpec("phase_timeline_md_path", legacy._phase_timeline_md_path),
            ContractCallableSpec("projection_phase_flush_due", legacy._projection_phase_flush_due),
            ContractCallableSpec("resolve_report_output_path", legacy._resolve_report_output_path),
            ContractCallableSpec("resolve_report_section_journal_path", legacy._resolve_report_section_journal_path),
            ContractCallableSpec("split_incremental_obligations", legacy._split_incremental_obligations),
            ContractCallableSpec("write_report_section_journal", legacy._write_report_section_journal),
            ContractCallableSpec("write_text_profiled", legacy._write_text_profiled),
        ),
    ),
    "ProgressPrimitives": PrimitiveContractSpec(
        contract_id="ProgressPrimitives",
        constants=(
            ContractConstantSpec("lsp_progress_notification_method", legacy._LSP_PROGRESS_NOTIFICATION_METHOD),
            ContractConstantSpec("lsp_progress_token_v2", legacy._LSP_PROGRESS_TOKEN_V2),
            ContractConstantSpec("canonical_progress_event_schema_v2", legacy._CANONICAL_PROGRESS_EVENT_SCHEMA_V2),
            ContractConstantSpec("progress_deadline_flush_margin_seconds", legacy._PROGRESS_DEADLINE_FLUSH_MARGIN_SECONDS),
            ContractConstantSpec("progress_deadline_flush_seconds", legacy._PROGRESS_DEADLINE_FLUSH_SECONDS),
            ContractConstantSpec("progress_deadline_watchdog_seconds", legacy._PROGRESS_DEADLINE_WATCHDOG_SECONDS),
            ContractConstantSpec("progress_heartbeat_poll_seconds", legacy._PROGRESS_HEARTBEAT_POLL_SECONDS),
        ),
        callables=(
            ContractCallableSpec("build_phase_progress_v2", legacy._build_phase_progress_v2),
            ContractCallableSpec("incremental_progress_obligations", legacy._incremental_progress_obligations),
            ContractCallableSpec("progress_heartbeat_seconds", legacy._progress_heartbeat_seconds),
        ),
    ),
    "TimeoutPrimitives": PrimitiveContractSpec(
        contract_id="TimeoutPrimitives",
        callables=(
            ContractCallableSpec("analysis_timeout_budget_ns", legacy._analysis_timeout_budget_ns),
            ContractCallableSpec("analysis_timeout_total_ticks", legacy._analysis_timeout_total_ticks),
            ContractCallableSpec("timeout_context_payload", legacy._timeout_context_payload),
            ContractCallableSpec("deadline_profile_sample_interval", legacy._deadline_profile_sample_interval),
        ),
    ),
}

INGRESS_STAGE_DEPENDENCY_DEFAULTS: dict[str, Callable[..., object]] = {
    "normalize_dataflow_response_fn": legacy._normalize_dataflow_response,
    "materialize_execution_plan_fn": legacy._materialize_execution_plan,
    "default_execute_command_deps_fn": legacy._default_execute_command_deps,
}


def primitive_contract_spec(contract_id: str) -> PrimitiveContractSpec:
    return PRIMITIVE_CONTRACT_SPECS[contract_id]


@lru_cache(maxsize=None)
def build_contract_class(contract_id: str, *, module_name: str) -> type:
    spec = primitive_contract_spec(contract_id)
    namespace: dict[str, object] = {"__module__": module_name, "__slots__": ()}
    for item in spec.constants:
        namespace[item.attr_name] = item.value
    for item in spec.callables:
        namespace[item.attr_name] = staticmethod(item.target)
    return type(contract_id, (), namespace)


def ingress_stage_dependency_defaults() -> dict[str, Callable[..., object]]:
    return dict(INGRESS_STAGE_DEPENDENCY_DEFAULTS)


__all__ = [
    "ContractCallableSpec",
    "ContractConstantSpec",
    "INGRESS_STAGE_DEPENDENCY_DEFAULTS",
    "PRIMITIVE_CONTRACT_SPECS",
    "PrimitiveContractSpec",
    "build_contract_class",
    "ingress_stage_dependency_defaults",
    "primitive_contract_spec",
]

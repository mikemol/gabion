from __future__ import annotations

from gabion.server_core import command_orchestrator_primitives as legacy
from gabion.server_core.analysis_primitives import AnalysisPrimitives, default_analysis_primitives
from gabion.server_core.primitive_contract_registry import (
    INGRESS_STAGE_DEPENDENCY_DEFAULTS,
    PRIMITIVE_CONTRACT_SPECS,
    primitive_contract_spec,
)
from gabion.server_core.progress_contracts import ProgressStageContract
from gabion.server_core.report_projection_runtime import ReportProjectionRuntime
from gabion.server_core.timeout_runtime import TimeoutStageRuntime


def test_first_layer_contract_registry_specs_cover_expected_bridge_surfaces() -> None:
    assert set(PRIMITIVE_CONTRACT_SPECS) >= {
        "AnalysisPrimitives",
        "TimeoutStageRuntime",
        "ProgressStageContract",
        "ReportProjectionRuntime",
    }
    assert primitive_contract_spec("AnalysisPrimitives").contract_id == "AnalysisPrimitives"
    assert INGRESS_STAGE_DEPENDENCY_DEFAULTS["normalize_dataflow_response_fn"] is legacy._normalize_dataflow_response


def test_generated_first_layer_contracts_preserve_legacy_members() -> None:
    assert AnalysisPrimitives.analysis_resume_progress is legacy._analysis_resume_progress
    assert AnalysisPrimitives.groups_by_path_from_collection_resume is legacy._groups_by_path_from_collection_resume
    assert AnalysisPrimitives.report_witness_digest is legacy._report_witness_digest
    assert TimeoutStageRuntime.analysis_timeout_budget_ns is legacy._analysis_timeout_budget_ns
    assert TimeoutStageRuntime.timeout_context_payload is legacy._timeout_context_payload
    assert ProgressStageContract.lsp_progress_notification_method == legacy._LSP_PROGRESS_NOTIFICATION_METHOD
    assert ProgressStageContract.progress_heartbeat_seconds is legacy._progress_heartbeat_seconds
    assert ReportProjectionRuntime.render_incremental_report is legacy._render_incremental_report
    assert ReportProjectionRuntime.resolve_report_output_path is legacy._resolve_report_output_path


def test_default_analysis_primitives_instantiates_generated_contract() -> None:
    primitives = default_analysis_primitives()
    assert primitives.analysis_index_resume_signature is legacy._analysis_index_resume_signature

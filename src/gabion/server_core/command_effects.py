from __future__ import annotations

from typing import Callable, Protocol, runtime_checkable


@runtime_checkable
class CommandEffects(Protocol):
    """Effect boundary used by server command orchestration."""

    analyze_paths_fn: Callable[..., object]
    load_aspf_resume_state_fn: Callable[..., object]
    append_aspf_delta_fn: Callable[..., None]
    finalize_aspf_resume_state_fn: Callable[..., object]
    analysis_input_manifest_fn: Callable[..., object]
    analysis_input_manifest_digest_fn: Callable[..., str]
    build_analysis_collection_resume_seed_fn: Callable[..., object]
    collection_semantic_progress_fn: Callable[..., object]
    project_report_sections_fn: Callable[..., object]
    report_projection_spec_rows_fn: Callable[[], object]
    collection_checkpoint_flush_due_fn: Callable[..., bool]
    write_bootstrap_incremental_artifacts_fn: Callable[..., None]
    load_report_section_journal_fn: Callable[..., object]
    start_trace_fn: Callable[..., object]
    record_1cell_fn: Callable[..., object]
    record_2cell_witness_fn: Callable[..., object]
    record_cofibration_fn: Callable[..., object]
    merge_imported_trace_fn: Callable[..., object]
    finalize_trace_fn: Callable[..., object]

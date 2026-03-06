from __future__ import annotations

import threading
from dataclasses import dataclass, replace
from typing import Callable

from gabion.analysis import AnalysisResult
from gabion.json_types import JSONObject
from gabion.server_core.command_contract import ProgressTraceStateContract


@dataclass(frozen=True)
class AnalysisDeps:
    analyze_paths_fn: Callable[..., AnalysisResult]
    load_aspf_resume_state_fn: Callable[..., JSONObject | None]
    analysis_input_manifest_fn: Callable[..., JSONObject]
    analysis_input_manifest_digest_fn: Callable[[JSONObject], str]
    build_analysis_collection_resume_seed_fn: Callable[..., JSONObject]
    collection_semantic_progress_fn: Callable[..., JSONObject]
    project_report_sections_fn: Callable[..., dict[str, list[str]]]
    report_projection_spec_rows_fn: Callable[[], list[JSONObject]]


@dataclass(frozen=True)
class OutputDeps:
    collection_checkpoint_flush_due_fn: Callable[..., bool]
    write_bootstrap_incremental_artifacts_fn: Callable[..., None]
    load_report_section_journal_fn: Callable[..., tuple[dict[str, list[str]], str | None]]


@dataclass(frozen=True)
class ProgressDeps:
    start_trace_fn: Callable[..., ProgressTraceStateContract]
    record_1cell_fn: Callable[..., ProgressTraceStateContract]
    record_2cell_witness_fn: Callable[..., ProgressTraceStateContract]
    record_cofibration_fn: Callable[..., ProgressTraceStateContract]
    merge_imported_trace_fn: Callable[..., ProgressTraceStateContract]
    finalize_trace_fn: Callable[..., ProgressTraceStateContract]


@dataclass(frozen=True)
class RuntimeDeps:
    monotonic_ns_fn: Callable[[], int]
    heartbeat_wait_fn: Callable[[threading.Event, float], bool]


@dataclass(frozen=True)
class ExecuteCommandDeps:
    analysis: AnalysisDeps
    output: OutputDeps
    progress: ProgressDeps
    runtime: RuntimeDeps


    @property
    def analyze_paths_fn(self):
        return self.analysis.analyze_paths_fn

    @property
    def load_aspf_resume_state_fn(self):
        return self.analysis.load_aspf_resume_state_fn

    @property
    def analysis_input_manifest_fn(self):
        return self.analysis.analysis_input_manifest_fn

    @property
    def analysis_input_manifest_digest_fn(self):
        return self.analysis.analysis_input_manifest_digest_fn

    @property
    def build_analysis_collection_resume_seed_fn(self):
        return self.analysis.build_analysis_collection_resume_seed_fn

    @property
    def collection_semantic_progress_fn(self):
        return self.analysis.collection_semantic_progress_fn

    @property
    def project_report_sections_fn(self):
        return self.analysis.project_report_sections_fn

    @property
    def report_projection_spec_rows_fn(self):
        return self.analysis.report_projection_spec_rows_fn

    @property
    def collection_checkpoint_flush_due_fn(self):
        return self.output.collection_checkpoint_flush_due_fn

    @property
    def write_bootstrap_incremental_artifacts_fn(self):
        return self.output.write_bootstrap_incremental_artifacts_fn

    @property
    def load_report_section_journal_fn(self):
        return self.output.load_report_section_journal_fn

    @property
    def start_trace_fn(self):
        return self.progress.start_trace_fn

    @property
    def record_1cell_fn(self):
        return self.progress.record_1cell_fn

    @property
    def record_2cell_witness_fn(self):
        return self.progress.record_2cell_witness_fn

    @property
    def record_cofibration_fn(self):
        return self.progress.record_cofibration_fn

    @property
    def merge_imported_trace_fn(self):
        return self.progress.merge_imported_trace_fn

    @property
    def finalize_trace_fn(self):
        return self.progress.finalize_trace_fn

    @property
    def monotonic_ns_fn(self):
        return self.runtime.monotonic_ns_fn

    @property
    def heartbeat_wait_fn(self):
        return self.runtime.heartbeat_wait_fn

    def with_overrides(self, **overrides: object) -> "ExecuteCommandDeps":
        if not overrides:
            return self
        analysis_fields = set(AnalysisDeps.__dataclass_fields__.keys())
        output_fields = set(OutputDeps.__dataclass_fields__.keys())
        progress_fields = set(ProgressDeps.__dataclass_fields__.keys())

        analysis_overrides: dict[str, object] = {}
        output_overrides: dict[str, object] = {}
        progress_overrides: dict[str, object] = {}
        root_overrides: dict[str, object] = {}

        for key, value in overrides.items():
            if key in {"analysis", "output", "progress", "runtime"}:
                root_overrides[key] = value
            elif key in analysis_fields:
                analysis_overrides[key] = value
            elif key in output_fields:
                output_overrides[key] = value
            elif key in progress_fields:
                progress_overrides[key] = value
            else:
                raise TypeError(f"Unknown ExecuteCommandDeps override: {key}")

        analysis = root_overrides.get("analysis", self.analysis)
        output = root_overrides.get("output", self.output)
        progress = root_overrides.get("progress", self.progress)
        runtime = root_overrides.get("runtime", self.runtime)

        if analysis_overrides:
            analysis = replace(self.analysis, **analysis_overrides)
        if output_overrides:
            output = replace(self.output, **output_overrides)
        if progress_overrides:
            progress = replace(self.progress, **progress_overrides)
        return replace(
            self,
            analysis=analysis,
            output=output,
            progress=progress,
            runtime=runtime,
        )


__all__ = [
    "AnalysisDeps",
    "OutputDeps",
    "ProgressDeps",
    "RuntimeDeps",
    "ExecuteCommandDeps",
]

from __future__ import annotations

from dataclasses import dataclass

from gabion.server_core.report_projection_runtime import ReportProjectionRuntime


@dataclass(frozen=True)
class OutputPrimitives:
    append_phase_timeline_event = staticmethod(ReportProjectionRuntime.append_phase_timeline_event)
    apply_journal_pending_reason = staticmethod(ReportProjectionRuntime.apply_journal_pending_reason)
    collection_components_preview_lines = staticmethod(ReportProjectionRuntime.collection_components_preview_lines)
    collection_progress_intro_lines = staticmethod(ReportProjectionRuntime.collection_progress_intro_lines)
    collection_report_flush_due = staticmethod(ReportProjectionRuntime.collection_report_flush_due)
    is_stdout_target = staticmethod(ReportProjectionRuntime.is_stdout_target)
    output_dirs = staticmethod(ReportProjectionRuntime.output_dirs)
    phase_timeline_header_block = staticmethod(ReportProjectionRuntime.phase_timeline_header_block)
    phase_timeline_jsonl_path = staticmethod(ReportProjectionRuntime.phase_timeline_jsonl_path)
    phase_timeline_md_path = staticmethod(ReportProjectionRuntime.phase_timeline_md_path)
    projection_phase_flush_due = staticmethod(ReportProjectionRuntime.projection_phase_flush_due)
    resolve_report_output_path = staticmethod(ReportProjectionRuntime.resolve_report_output_path)
    resolve_report_section_journal_path = staticmethod(ReportProjectionRuntime.resolve_report_section_journal_path)
    split_incremental_obligations = staticmethod(ReportProjectionRuntime.split_incremental_obligations)
    write_report_section_journal = staticmethod(ReportProjectionRuntime.write_report_section_journal)
    write_text_profiled = staticmethod(ReportProjectionRuntime.write_text_profiled)


def default_output_primitives() -> OutputPrimitives:
    return OutputPrimitives()

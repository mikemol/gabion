from __future__ import annotations

from dataclasses import dataclass

from gabion.server_core import command_orchestrator_primitives as legacy


@dataclass(frozen=True)
class OutputPrimitives:
    append_phase_timeline_event = staticmethod(legacy._append_phase_timeline_event)
    apply_journal_pending_reason = staticmethod(legacy._apply_journal_pending_reason)
    collection_components_preview_lines = staticmethod(legacy._collection_components_preview_lines)
    collection_progress_intro_lines = staticmethod(legacy._collection_progress_intro_lines)
    collection_report_flush_due = staticmethod(legacy._collection_report_flush_due)
    is_stdout_target = staticmethod(legacy._is_stdout_target)
    output_dirs = staticmethod(legacy._output_dirs)
    phase_timeline_header_block = staticmethod(legacy._phase_timeline_header_block)
    phase_timeline_jsonl_path = staticmethod(legacy._phase_timeline_jsonl_path)
    phase_timeline_md_path = staticmethod(legacy._phase_timeline_md_path)
    projection_phase_flush_due = staticmethod(legacy._projection_phase_flush_due)
    resolve_report_output_path = staticmethod(legacy._resolve_report_output_path)
    resolve_report_section_journal_path = staticmethod(legacy._resolve_report_section_journal_path)
    split_incremental_obligations = staticmethod(legacy._split_incremental_obligations)
    write_report_section_journal = staticmethod(legacy._write_report_section_journal)
    write_text_profiled = staticmethod(legacy._write_text_profiled)


def default_output_primitives() -> OutputPrimitives:
    return OutputPrimitives()

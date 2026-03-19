from __future__ import annotations

"""ASPF resume-state facade."""

from gabion.analysis.foundation.aspf_resume_state_impl import (
    append_delta_record,
    append_delta_stream_record,
    apply_resume_mutations,
    build_delta_ledger_payload,
    fold_resume_mutations,
    iter_delta_records,
    iter_delta_records_from_state_files,
    iter_delta_records_from_stream_paths,
    iter_resume_mutations,
    load_latest_resume_projection_from_state_files,
    load_resume_projection_from_state_files,
    replay_resume_projection,
    write_delta_stream,
)

__all__ = [
    "append_delta_record",
    "append_delta_stream_record",
    "apply_resume_mutations",
    "build_delta_ledger_payload",
    "fold_resume_mutations",
    "iter_delta_records",
    "iter_delta_records_from_state_files",
    "iter_delta_records_from_stream_paths",
    "iter_resume_mutations",
    "load_latest_resume_projection_from_state_files",
    "load_resume_projection_from_state_files",
    "replay_resume_projection",
    "write_delta_stream",
]

# gabion:decision_protocol_module
from __future__ import annotations

"""Canonical deadline-obligation summary runtime helper."""

from gabion.analysis.foundation.resume_codec import int_tuple4_or_none
from gabion.analysis.foundation.timeout_context import check_deadline
from gabion.analysis.indexed_scan.deadline.deadline_obligation_summary import (
    SummarizeDeadlineObligationsDeps,
    summarize_deadline_obligations,
)
from gabion.analysis.projection.projection_normalize import spec_hash as projection_spec_hash
from gabion.analysis.projection.projection_registry import (
    DEADLINE_OBLIGATIONS_SUMMARY_SPEC,
)
from gabion.analysis.dataflow.engine.dataflow_projection_materialization import (
    _format_span_fields,
)
from gabion.invariants import require_not_none


def _summarize_deadline_obligations(entries, *, max_entries=20, forest):
    return summarize_deadline_obligations(
        entries,
        max_entries=max_entries,
        forest=forest,
        deps=SummarizeDeadlineObligationsDeps(
            check_deadline_fn=check_deadline,
            projection_spec_hash_fn=projection_spec_hash,
            deadline_obligations_summary_spec=DEADLINE_OBLIGATIONS_SUMMARY_SPEC,
            require_not_none_fn=require_not_none,
            int_tuple4_or_none_fn=int_tuple4_or_none,
            format_span_fields_fn=_format_span_fields,
        ),
    )


__all__ = ["_summarize_deadline_obligations"]

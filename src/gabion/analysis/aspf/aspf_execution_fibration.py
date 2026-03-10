from __future__ import annotations

"""ASPF execution-fibration facade.

Implementation is hosted in foundation to keep ASPF neighborhood surfaces thin
under force-majeure strictification.
"""

from gabion.analysis.foundation.aspf_execution_fibration_impl import *  # noqa: F401,F403
from gabion.analysis.foundation.aspf_execution_fibration_impl import (
    _ImportedTraceMergeVisitor,
    _as_wire_value,
    _build_trace_replay_iterators,
    _iter_delta_records,
    _iter_trace_events,
    _iter_two_cell_witnesses,
    _merge_two_cell_payload,
    _normalize_stream_trace_payload,
    _publish_event,
    _semantic_surface_sequence,
)


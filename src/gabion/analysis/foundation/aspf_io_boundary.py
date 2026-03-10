from __future__ import annotations

"""ASPF artifact and control-key constants for wire I/O boundaries."""

TRACE_OUTPUT_OPTION_KEY = "aspf_trace_json"
EQUIVALENCE_INPUT_OPTION_KEY = "aspf_equivalence_against"
IMPORT_TRACE_OPTION_KEY = "aspf_import_trace"
OPPORTUNITIES_OUTPUT_OPTION_KEY = "aspf_opportunities_json"
STATE_OUTPUT_OPTION_KEY = "aspf_state_json"
IMPORT_STATE_OPTION_KEY = "aspf_import_state"
DELTA_OUTPUT_OPTION_KEY = "aspf_delta_jsonl"
SEMANTIC_SURFACE_OPTION_KEY = "aspf_semantic_surface"

TRACE_FILENAME = "aspf_trace.json"
EQUIVALENCE_FILENAME = "aspf_equivalence.json"
OPPORTUNITIES_FILENAME = "aspf_opportunities.json"
DELTA_STREAM_FILENAME = "aspf_delta.jsonl"
STATE_SNAPSHOT_FILENAME = "0001_aspf.snapshot.json"
MANIFEST_FILENAME = "manifest.json"

ONE_CELL_STREAM_FILENAME = "one_cells.jsonl"
TWO_CELL_STREAM_FILENAME = "two_cell_witnesses.jsonl"
COFIBRATION_STREAM_FILENAME = "cofibrations.jsonl"
DELTA_RECORD_STREAM_FILENAME = "delta_records.jsonl"

STATE_SNAPSHOT_SUFFIX = ".snapshot.json"
DELTA_STREAM_SUFFIX = ".delta.jsonl"
LINE_STREAM_SUFFIXES = (".jsonl", ".ndjson")


---
doc_revision: 1
doc_id: narrowing_inventory_server_cli
doc_role: inventory
doc_scope:
  - server
  - cli
---

# Narrowing inventory: `src/gabion/server.py` + `src/gabion/cli.py`

## `src/gabion/server.py`

- `_parse_dataflow_command_payload`: **boundary-valid** (payload shape normalization + command carrier construction).
- `_parse_impact_command_payload`: **boundary-valid** (payload shape normalization + command carrier construction).
- `_collection_resume_dto`: **boundary-valid** (ingress DTO validation for resume payload shape).
- `_execute_command_total`: **semantic-core** (typed `DataflowCommandPayload` only, no union narrowing).
- `_execute_impact_total`: **semantic-core** (typed `ImpactCommandPayload` only, no union narrowing).
- `_completed_path_set`: **semantic-core** (consumes `CollectionResumeDTO`; no runtime mapping checks).
- `_in_progress_scan_states`: **semantic-core** (consumes `CollectionResumeDTO`; order invariants only).
- `_analysis_index_resume_*` helpers: **semantic-core** (consume `CollectionResumeDTO` and `AnalysisIndexResumeDTO`).
- `_collection_progress_intro_lines`: **semantic-core** (uses DTO adapter output for in-progress/index projection fields).
- `_collection_semantic_witness`: **semantic-core** (uses DTO adapter output for deterministic witness digest inputs).

## `src/gabion/cli.py`

- `_phase_progress_dto_from_progress_notification`: **boundary-valid** (single notification-to-DTO adapter).
- `_phase_progress_from_progress_notification`: **boundary-valid** (compatibility wrapper; serializes validated DTO).
- `_emit_phase_progress_line`: **semantic-core** (formats typed progress payload).
- `_run_dataflow_raw_argv` notification handler: **semantic-core** (consumes typed progress DTO, reuses one mapping payload for signature/timeline).

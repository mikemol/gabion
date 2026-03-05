---
doc_revision: 1
doc_id: command_orchestrator_stage_boundaries
doc_role: architecture
doc_scope:
  - server_core
  - command_orchestrator
---

# Command orchestrator stage boundaries

This note defines stage ownership boundaries for `server_core` command orchestration.

## Stage ownership

- `ingress_contracts.py`: ingress-only normalization and command dependency wiring.
  - Owns default ingress dependency bundle creation.
  - Owns normalization + execution-plan materialization entry points used before runtime state is initialized.
- `progress_contracts.py`: progress token/schema constants and transition glue.
  - Owns heartbeat/progress projection helpers.
  - Must not own analysis/report rendering logic.
- `timeout_runtime.py`: timeout budget, timeout context payloads, and deadline-profile sampling helpers.
  - Owns timeout window calculations and timeout payload synthesis.
  - Must not own ingress or report projection concerns.
- `report_projection_runtime.py`: report/timeline/journal projection helpers.
  - Owns phase timeline emission, incremental report projection, and report section journal IO.
  - Must not own timeout budgeting or ingress validation.

## Allowed cross-stage dependencies

- `command_orchestrator.py` may compose all stage modules.
- Stage modules may depend on shared contracts (`ingress_primitives.py`, `command_contract.py`) and stable analysis/config primitives.
- Stage modules must not import each other for incidental convenience; new cross-stage behavior should be composed in `command_orchestrator.py`.
- Stage modules expose focused interfaces; broad import-and-forward barrels are not allowed for stage wiring.

---
doc_revision: 2
doc_id: policy_dsl_migration_notes
doc_role: note
---

# Policy DSL migration notes

Temporary boundary adapters kept:

- `src/gabion/tooling/governance_rules.py::gate_policy_to_dsl_sources` keeps legacy governance YAML shape and projects it into DSL rules.
  - removal_condition: `docs/governance_rules.yaml` is rewritten to native DSL schema.
- `src/gabion/tooling/delta_gate.py` preserves existing CLI messages/exit codes while delegating branching to DSL decisions.
  - removal_condition: downstream scripts consume typed `PolicyDecision` objects directly.

These adapters are boundary-only and should not be expanded into semantic-core compatibility layers.

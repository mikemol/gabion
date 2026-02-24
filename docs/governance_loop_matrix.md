---
doc_revision: 1
reader_reintern: "Reader-only: re-intern if doc_revision changed since you last read this doc."
doc_id: governance_loop_matrix
doc_role: policy
doc_scope:
  - repo
  - governance
  - ci
  - tooling
doc_authority: normative
doc_requires:
  - docs/governance_control_loops.md#governance_control_loops
  - docs/governance_rules.yaml
  - README.md#repo_contract
  - AGENTS.md#agent_obligations
  - CONTRIBUTING.md#contributing_contract
  - POLICY_SEED.md#policy_seed
  - glossary.md#contract
doc_reviewed_as_of:
  docs/governance_control_loops.md#governance_control_loops: 1
  README.md#repo_contract: 1
  AGENTS.md#agent_obligations: 1
  CONTRIBUTING.md#contributing_contract: 1
  POLICY_SEED.md#policy_seed: 1
  glossary.md#contract: 1
doc_review_notes:
  docs/governance_control_loops.md#governance_control_loops: "Control-loop domain registry reviewed; matrix rows align to declared first-order loops and correction semantics."
  README.md#repo_contract: "Repo contract and cross-reference structure reviewed; matrix linked from governance references."
  AGENTS.md#agent_obligations: "Agent obligations reviewed; matrix keeps mechanized-governance and override language aligned with agent constraints."
  CONTRIBUTING.md#contributing_contract: "Contributor contract reviewed; matrix commands align with repo-local tooling execution norms."
  POLICY_SEED.md#policy_seed: "Policy seed reviewed; bounded-step and override mechanics are preserved in per-gate rows."
  glossary.md#contract: "Glossary contract reviewed; gate identifiers and loop-domain labels are semantically stable."
doc_sections:
  governance_loop_matrix: 1
doc_section_requires:
  governance_loop_matrix:
    - docs/governance_control_loops.md#governance_control_loops
    - README.md#repo_contract
    - AGENTS.md#agent_obligations
    - CONTRIBUTING.md#contributing_contract
    - POLICY_SEED.md#policy_seed
    - glossary.md#contract
doc_section_reviews:
  governance_loop_matrix:
    docs/governance_control_loops.md#governance_control_loops:
      dep_version: 1
      self_version_at_review: 1
      outcome: no_change
      note: "Domain/correction model remains compatible with matrix columns."
    README.md#repo_contract:
      dep_version: 1
      self_version_at_review: 1
      outcome: no_change
      note: "Repo governance cross-reference expectations remain unchanged."
    AGENTS.md#agent_obligations:
      dep_version: 1
      self_version_at_review: 1
      outcome: no_change
      note: "Agent obligations remain aligned with matrix override and enforcement semantics."
    CONTRIBUTING.md#contributing_contract:
      dep_version: 1
      self_version_at_review: 1
      outcome: no_change
      note: "Contributor command discipline remains compatible with matrix sensors."
    POLICY_SEED.md#policy_seed:
      dep_version: 1
      self_version_at_review: 1
      outcome: no_change
      note: "Override and correction ratchet requirements unchanged."
    glossary.md#contract:
      dep_version: 1
      self_version_at_review: 1
      outcome: no_change
      note: "Loop and gate terms remain semantically consistent with glossary contract."
doc_change_protocol: "POLICY_SEED.md#change_protocol"
doc_invariants:
  - mechanized_governance_invariant
  - lsp_first_invariant
doc_erasure:
  - formatting
  - typos
doc_owner: maintainer
---

<a id="governance_loop_matrix"></a>
# Governance loop matrix

Semi-generated from:
- `docs/governance_control_loops.md`
- `docs/governance_rules.yaml`
- gate entrypoints in `src/gabion/tooling/*gate*.py`

Cross-reference anchors used by this matrix: `docs/governance_control_loops.md#governance_control_loops`, `README.md#repo_contract`, `CONTRIBUTING.md#contributing_contract`, `AGENTS.md#agent_obligations`, `POLICY_SEED.md#policy_seed`, and `glossary.md#contract`.

| loop domain | gate ID | sensor command | state artifact path | correction mode | warning/blocking thresholds | override mechanism |
| --- | --- | --- | --- | --- | --- | --- |
| baseline ratchets | `obsolescence_opaque` | `mise exec -- python -m gabion.tooling.obsolescence_delta_gate` | `artifacts/out/test_obsolescence_delta.json` | `hard-fail` | `warning=0, block=1` | Gate toggle: `GABION_GATE_OPAQUE_DELTA` (`default_true`); strictness reductions require `GABION_POLICY_OVERRIDE_TOKEN` + `GABION_POLICY_OVERRIDE_RATIONALE`. |
| baseline ratchets | `obsolescence_unmapped` | `mise exec -- python -m gabion.tooling.obsolescence_delta_unmapped_gate` | `artifacts/out/test_obsolescence_delta.json` | `ratchet` | `warning=0, block=1` | Gate toggle: `GABION_GATE_UNMAPPED_DELTA` (`default_true`); strictness reductions require `GABION_POLICY_OVERRIDE_TOKEN` + `GABION_POLICY_OVERRIDE_RATIONALE`. |
| baseline ratchets | `annotation_orphaned` | `mise exec -- python -m gabion.tooling.annotation_drift_orphaned_gate` | `artifacts/out/test_annotation_drift_delta.json` | `ratchet` | `warning=0, block=1` | Gate toggle: `GABION_GATE_ORPHANED_DELTA` (`default_true`); strictness reductions require `GABION_POLICY_OVERRIDE_TOKEN` + `GABION_POLICY_OVERRIDE_RATIONALE`. |
| baseline ratchets | `ambiguity` | `mise exec -- python -m gabion.tooling.ambiguity_delta_gate` | `artifacts/out/ambiguity_delta.json` | `hard-fail` | `warning=0, block=1` | Gate toggle: `GABION_GATE_AMBIGUITY_DELTA` (`default_true`); strictness reductions require `GABION_POLICY_OVERRIDE_TOKEN` + `GABION_POLICY_OVERRIDE_RATIONALE`. |
| docs/docflow | `docflow` | `mise exec -- python -m gabion.tooling.docflow_delta_gate` | `artifacts/out/docflow_compliance_delta.json` | `advisory` | `warning=0, block=1` | Gate toggle: `GABION_GATE_DOCFLOW_DELTA` (`truthy_only`); strictness reductions require `GABION_POLICY_OVERRIDE_TOKEN` + `GABION_POLICY_OVERRIDE_RATIONALE`. |

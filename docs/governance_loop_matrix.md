---
doc_revision: 3
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
  - docs/governance_control_loops.yaml
  - docs/governance_rules.yaml
  - README.md#repo_contract
  - AGENTS.md#agent_obligations
  - CONTRIBUTING.md#contributing_contract
  - POLICY_SEED.md#policy_seed
  - glossary.md#contract
doc_reviewed_as_of:
  docs/governance_control_loops.md#governance_control_loops: 2
  README.md#repo_contract: 2
  AGENTS.md#agent_obligations: 2
  CONTRIBUTING.md#contributing_contract: 2
  POLICY_SEED.md#policy_seed: 2
  glossary.md#contract: 1
doc_review_notes:
  docs/governance_control_loops.md#governance_control_loops: "Re-reviewed governance_control_loops section v2 under doc rev7; matrix rows now render from the shared loop catalog while remaining aligned to declared first-order loops including strict docflow packetization/enforcement."
  README.md#repo_contract: "Reviewed README.md rev2 (removed stale ASPF action-plan CLI/examples; continuation docs now state/delta only)."
  AGENTS.md#agent_obligations: "Reviewed AGENTS.md rev2 (required validation stack, forward-remediation preference, and ci_watch failure-bundle triage guidance)."
  CONTRIBUTING.md#contributing_contract: "Reviewed CONTRIBUTING.md rev2 (two-stage dual-sensor cadence, correction-unit validation stack, and strict-coverage trigger guidance)."
  POLICY_SEED.md#policy_seed: "Reviewed POLICY_SEED.md rev2 (forward-remediation order, ci_watch failure-bundle durability, and enforced execution-coverage policy wording)."
  glossary.md#contract: "Glossary contract reviewed; gate identifiers and loop-domain labels are semantically stable."
doc_sections:
  governance_loop_matrix: 2
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
      dep_version: 2
      self_version_at_review: 2
      outcome: no_change
      note: "Domain/correction model remains compatible with matrix columns including packetized strict docflow loop controls after re-reviewing section v2 under doc rev7."
    README.md#repo_contract:
      dep_version: 2
      self_version_at_review: 2
      outcome: no_change
      note: "Repo contract rev2 reviewed; command and artifact guidance remains aligned."
    AGENTS.md#agent_obligations:
      dep_version: 2
      self_version_at_review: 2
      outcome: no_change
      note: "Agent obligations rev2 reviewed; clause and cadence links remain aligned."
    CONTRIBUTING.md#contributing_contract:
      dep_version: 2
      self_version_at_review: 2
      outcome: no_change
      note: "Contributor contract rev2 reviewed; dual-sensor cadence and correction gates remain aligned."
    POLICY_SEED.md#policy_seed:
      dep_version: 2
      self_version_at_review: 2
      outcome: no_change
      note: "Policy seed rev2 reviewed; governance obligations remain aligned."
    glossary.md#contract:
      dep_version: 1
      self_version_at_review: 2
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

Cross-reference anchors used by this matrix: `docs/governance_control_loops.md#governance_control_loops`, `README.md#repo_contract`, `CONTRIBUTING.md#contributing_contract`, `AGENTS.md#agent_obligations`, `POLICY_SEED.md#policy_seed`, and `glossary.md#contract`.
<!-- BEGIN:generated_governance_loop_matrix -->
_This matrix is generated from `docs/governance_control_loops.yaml` and `docs/governance_rules.yaml` via `mise exec -- python -m scripts.policy.render_governance_loop_docs`._

| loop domain | gate ID | sensor command | state artifact path | correction mode | warning/blocking thresholds | override mechanism |
| --- | --- | --- | --- | --- | --- | --- |
| baseline ratchets | `obsolescence_opaque` | `mise exec -- python -m gabion.tooling.obsolescence_delta_gate` | artifacts/out/test_obsolescence_delta.json | `hard-fail` | warning=0, block=1 | Gate toggle: `GABION_GATE_OPAQUE_DELTA` (`default_true`); strictness reductions require `GABION_POLICY_OVERRIDE_TOKEN` + `GABION_POLICY_OVERRIDE_RATIONALE`. |
| baseline ratchets | `obsolescence_unmapped` | `mise exec -- python -m gabion.tooling.obsolescence_delta_unmapped_gate` | artifacts/out/test_obsolescence_delta.json | `ratchet` | warning=0, block=1 | Gate toggle: `GABION_GATE_UNMAPPED_DELTA` (`default_true`); strictness reductions require `GABION_POLICY_OVERRIDE_TOKEN` + `GABION_POLICY_OVERRIDE_RATIONALE`. |
| baseline ratchets | `annotation_orphaned` | `mise exec -- python -m gabion.tooling.annotation_drift_orphaned_gate` | artifacts/out/test_annotation_drift_delta.json | `ratchet` | warning=0, block=1 | Gate toggle: `GABION_GATE_ORPHANED_DELTA` (`default_true`); strictness reductions require `GABION_POLICY_OVERRIDE_TOKEN` + `GABION_POLICY_OVERRIDE_RATIONALE`. |
| baseline ratchets | `ambiguity` | `mise exec -- python -m gabion.tooling.ambiguity_delta_gate` | artifacts/out/ambiguity_delta.json | `hard-fail` | warning=0, block=1 | Gate toggle: `GABION_GATE_AMBIGUITY_DELTA` (`default_true`); strictness reductions require `GABION_POLICY_OVERRIDE_TOKEN` + `GABION_POLICY_OVERRIDE_RATIONALE`. |
| docs/docflow | `docflow` | `mise exec -- python -m gabion.tooling.docflow_delta_gate` | artifacts/out/docflow_compliance_delta.json | `advisory` | warning=0, block=1 | Gate toggle: `GABION_GATE_DOCFLOW_DELTA` (`truthy_only`); strictness reductions require `GABION_POLICY_OVERRIDE_TOKEN` + `GABION_POLICY_OVERRIDE_RATIONALE`. |
| docs/docflow | `docflow_packet_loop` | `mise exec -- python -m gabion policy docflow-packet-enforce --root . --packets artifacts/out/docflow_warning_doc_packets.json --baseline docs/baselines/docflow_packet_baseline.json --out artifacts/out/docflow_packet_enforcement.json --debt-out artifacts/out/docflow_packet_debt_ledger.json --check --run-proving-tests` | artifacts/out/docflow_packet_enforcement.json, artifacts/out/docflow_packet_debt_ledger.json | `hard-fail` | warning=0, block=1 | Gate toggle: `GABION_GATE_DOCFLOW_PACKET` (`default_true`); baseline movement remains explicit via `--write-baseline` in a dedicated correction unit with documented rationale. |
<!-- END:generated_governance_loop_matrix -->

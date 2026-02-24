---
doc_revision: 3
reader_reintern: "Reader-only: re-intern if doc_revision changed since you last read this doc."
doc_id: github_pull_request_template
doc_role: operational_template
doc_scope:
  - repo
  - governance
  - workflow
doc_authority: operational
doc_requires:
  - CONTRIBUTING.md#contributing_contract
  - POLICY_SEED.md#policy_seed
doc_relations:
  operationalizes:
    - CONTRIBUTING.md#contributing_contract
    - POLICY_SEED.md#policy_seed
doc_change_protocol: "POLICY_SEED.md#change_protocol"
doc_owner: maintainer
---
## Summary
- Describe the change.

## Validation
- List checks/tests run.

## Governance (required for governance/tooling changes)
- controller impact: <!-- required when touching governance/tooling; describe impacted controller anchors/sensors -->
- loop updated?: <!-- yes/no; if yes explain what enforcement/reporting loop changed -->

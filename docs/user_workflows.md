---
doc_revision: 2
reader_reintern: "Reader-only: re-intern if doc_revision changed since you last read this doc."
doc_id: user_workflows
doc_role: guide
doc_scope:
  - repo
  - usage
  - cli
  - editor
  - ci
doc_authority: informative
doc_requires:
  - README.md#repo_contract
  - CONTRIBUTING.md#contributing_contract
  - POLICY_SEED.md#policy_seed
doc_reviewed_as_of:
  README.md#repo_contract: 1
  CONTRIBUTING.md#contributing_contract: 1
  POLICY_SEED.md#policy_seed: 1
doc_review_notes:
  README.md#repo_contract: "Reviewed command surfaces and workflow sections for user-loop alignment."
  CONTRIBUTING.md#contributing_contract: "Reviewed contributor guardrails to keep this guide operational (not normative)."
  POLICY_SEED.md#policy_seed: "Reviewed execution-policy constraints; this guide links out instead of restating policy."
doc_change_protocol: "POLICY_SEED.md#change_protocol"
doc_sections:
  user_workflows: 1
doc_section_requires:
  user_workflows:
    - README.md#repo_contract
    - CONTRIBUTING.md#contributing_contract
    - POLICY_SEED.md#policy_seed
doc_section_reviews:
  user_workflows:
    README.md#repo_contract:
      dep_version: 1
      self_version_at_review: 1
      outcome: no_change
      note: "Workflow references match current README command entrypoints."
    CONTRIBUTING.md#contributing_contract:
      dep_version: 1
      self_version_at_review: 1
      outcome: no_change
      note: "No conflict with contributor contract; examples remain local-run focused."
    POLICY_SEED.md#policy_seed:
      dep_version: 1
      self_version_at_review: 1
      outcome: no_change
      note: "Policy is linked as advanced reading; this doc remains usage-centric."
doc_erasure:
  - formatting
  - typos
doc_owner: maintainer
---

<a id="user_workflows"></a>
# User Workflows

This page gives practical loops for running Gabion from the CLI, VS Code, and CI/PR reviews.
For governance and policy details, see:
- `POLICY_SEED.md#policy_seed`
- `CONTRIBUTING.md#contributing_contract`

## 1) Local CLI remediation loop

Use this loop when you are actively fixing findings and want fast reruns.

### Baseline setup (first run)
```bash
mise exec -- python -m gabion check \
  --baseline artifacts/audit_reports/dataflow_baseline.txt \
  --baseline-write
```

### Iterative loop with resume checkpoint reuse
```bash
mise exec -- python -m gabion check \
  --baseline artifacts/audit_reports/dataflow_baseline.txt \
  --resume-checkpoint artifacts/audit_reports/dataflow_resume_checkpoint_local.json \
  --resume-on-timeout 1
```

### Timeout handling
If a run times out, re-run the same command and keep the same checkpoint path.
Gabion can hydrate cached parse/index state when identity inputs are compatible.
If you intentionally changed core identity knobs and results look unexpectedly cold,
remove the local checkpoint once and rerun:
```bash
rm -f artifacts/audit_reports/dataflow_resume_checkpoint_local.json
mise exec -- python -m gabion check \
  --baseline artifacts/audit_reports/dataflow_baseline.txt \
  --resume-checkpoint artifacts/audit_reports/dataflow_resume_checkpoint_local.json \
  --resume-on-timeout 1
```

## 2) VS Code-assisted remediation loop

Use this loop when you want quick fix-and-verify cycles in the editor.

### Start from the CLI once (optional warm-up)
```bash
mise exec -- python -m gabion check \
  --resume-checkpoint artifacts/audit_reports/dataflow_resume_checkpoint_local.json \
  --resume-on-timeout 1
```

### In VS Code
1. Open the workspace and ensure the Gabion extension stub is active.
2. Use **Problems** view for diagnostics (location + rule messages).
3. Use **Quick Fix / Code Actions** where available to apply assisted edits.
4. Watch extension progress UI while analysis runs; avoid changing identity-related
   settings mid-loop unless needed.
5. Open the extension output channel to interpret run state:
   - normal: steady diagnostic refresh + no repeated cold-start messages,
   - fallback/cold path: explicit re-index/re-parse style logs after settings changes.

### Validate after editor changes
```bash
mise exec -- python -m gabion check \
  --resume-checkpoint artifacts/audit_reports/dataflow_resume_checkpoint_local.json \
  --resume-on-timeout 1
```

## 3) CI/PR loop

Use this loop to review whether a PR is healthy and whether cache reuse behaved as expected.

### Artifacts to inspect
- Dataflow grammar report artifact (`dataflow_report.md` output family).
- Job/step summaries emitted by CI helper scripts.
- PR comment output (same-repo PRs) from the dataflow grammar workflow.

### Cache-effectiveness counters
Look for the counters emitted by `gabion run-dataflow-stage` in logs/summaries:
- `completed_paths`: paths fully handled this run.
- `hydrated_paths`: paths restored from resume data.
- `paths_parsed_after_resume`: additional paths that still required parsing.

Interpretation quick guide:
- Higher `hydrated_paths` with lower `paths_parsed_after_resume` usually means good reuse.
- Very low hydration plus high post-resume parsing often indicates identity drift,
  changed file sets, or a cold-start path.

### Re-run with stable identity settings
If cache behavior seems unexpectedly poor, re-run while keeping identity-affecting
settings stable (strictness profile, external filtering mode, and checkpoint identity inputs).
For local reproduction before pushing another commit:
```bash
mise exec -- python -m gabion check \
  --resume-checkpoint artifacts/audit_reports/dataflow_resume_checkpoint_local.json \
  --resume-on-timeout 1
```

For deeper policy/governance expectations around CI execution and protections,
use `POLICY_SEED.md#policy_seed` as the canonical source.

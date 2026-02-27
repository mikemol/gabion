---
doc_revision: 8
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

### Iterative loop with ASPF snapshot + delta
```bash
mise exec -- python -m gabion check run \
  --baseline artifacts/audit_reports/dataflow_baseline.txt \
  --baseline-mode enforce \
  --aspf-state-json artifacts/out/aspf_state/session-local/0001_check-run.snapshot.json \
  --aspf-delta-jsonl artifacts/out/aspf_state/session-local/0001_check-run.delta.jsonl \
  --aspf-action-plan-json artifacts/out/aspf_state/session-local/0001_check-run.action_plan.json \
  --aspf-action-plan-md artifacts/out/aspf_state/session-local/0001_check-run.action_plan.md
```

### Continuation handling
To continue from previous work, import prior state snapshot(s):
```bash
mise exec -- python -m gabion check run \
  --aspf-state-json artifacts/out/aspf_state/session-local/0002_check-run.snapshot.json \
  --aspf-delta-jsonl artifacts/out/aspf_state/session-local/0002_check-run.delta.jsonl \
  --aspf-action-plan-json artifacts/out/aspf_state/session-local/0002_check-run.action_plan.json \
  --aspf-action-plan-md artifacts/out/aspf_state/session-local/0002_check-run.action_plan.md \
  --aspf-import-state artifacts/out/aspf_state/session-local/0001_check-run.snapshot.json
```

### Delta bundle/gates cutover
Use the single-pass delta emitter and gate-only follow-up to avoid duplicate
re-analysis:
```bash
mise exec -- python -m gabion check delta-bundle \
  --aspf-state-json artifacts/out/aspf_state/session-local/0003_delta-bundle.snapshot.json \
  --aspf-delta-jsonl artifacts/out/aspf_state/session-local/0003_delta-bundle.delta.jsonl \
  --aspf-action-plan-json artifacts/out/aspf_state/session-local/0003_delta-bundle.action_plan.json \
  --aspf-action-plan-md artifacts/out/aspf_state/session-local/0003_delta-bundle.action_plan.md \
  --aspf-import-state artifacts/out/aspf_state/session-local/0002_check-run.snapshot.json

mise exec -- python -m gabion check delta-gates
```

## 2) Dual-sensor remediation loop

Use this loop when you want local repro feedback and branch CI/status feedback
to drive correction together.

### Terminal A: local repro lane
```bash
mise exec -- python -m gabion check run \
  --aspf-state-json artifacts/out/aspf_state/session-local/0001_check-run.snapshot.json \
  --aspf-delta-jsonl artifacts/out/aspf_state/session-local/0001_check-run.delta.jsonl \
  --aspf-action-plan-json artifacts/out/aspf_action_plan.json \
  --aspf-action-plan-md artifacts/out/aspf_action_plan.md
```

### Terminal B: remote status-check lane
```bash
mise exec -- python scripts/ci_watch.py --branch stage --workflow ci
```

On watched-run failure, collect run metadata/logs/artifacts into:

`artifacts/out/ci_watch/run_<run_id>/`

Use explicit artifact filters when only selected bundles are needed:

```bash
mise exec -- python scripts/ci_watch.py \
  --branch stage \
  --workflow ci \
  --artifact-name test-runs \
  --artifact-name dataflow-report
```

If run watching fails and failure-bundle collection also fails, `ci_watch` exits
with code `2` so automation can distinguish collection errors from watched-run
status.

### Correction cadence
1. Start both lanes concurrently when available.
2. Take the first actionable failure signal from either lane as the next fix target.
3. Keep one bounded correction unit per push (one blocking signal or tightly coupled set).
4. Validate locally, then stage/commit/push immediately.
5. Resume both lanes; handle fallout in subsequent correction units.

### Degraded mode
If only one lane is available, continue with that lane and restore dual-sensor
operation when possible.

## 3) ASPF cross-script handoff loop

Use this loop when you want cumulative ASPF state reuse plus glued reasoning across
separate invocations.

### Script/lane A: emit first state object
```bash
mise exec -- python -m gabion check run \
  --aspf-state-json artifacts/out/aspf_state/session-a/0001_check-run.snapshot.json \
  --aspf-delta-jsonl artifacts/out/aspf_state/session-a/0001_check-run.delta.jsonl \
  --aspf-action-plan-json artifacts/out/aspf_state/session-a/0001_check-run.action_plan.json \
  --aspf-action-plan-md artifacts/out/aspf_state/session-a/0001_check-run.action_plan.md
```

### Script/lane B: import cumulative prior state
```bash
mise exec -- python -m gabion check run \
  --aspf-state-json artifacts/out/aspf_state/session-a/0002_check-run.snapshot.json \
  --aspf-delta-jsonl artifacts/out/aspf_state/session-a/0002_check-run.delta.jsonl \
  --aspf-action-plan-json artifacts/out/aspf_state/session-a/0002_check-run.action_plan.json \
  --aspf-action-plan-md artifacts/out/aspf_state/session-a/0002_check-run.action_plan.md \
  --aspf-import-state artifacts/out/aspf_state/session-a/0001_check-run.snapshot.json
```

### Use helper-driven cumulative planning
```bash
mise exec -- python scripts/aspf_handoff.py prepare \
  --session-id session-a \
  --step-id script-a.check.run \
  --command-profile check.run
```

The helper writes/updates `artifacts/out/aspf_handoff_manifest.json` and emits
`--aspf-state-json` plus cumulative `--aspf-import-state` args for the next
step.

### Read phase-1 artifacts
- `artifacts/out/aspf_trace.json`
- `artifacts/out/aspf_equivalence.json`
- `artifacts/out/aspf_opportunities.json`
- `artifacts/out/aspf_state/<session>/<seq>_<step>.snapshot.json`
- `artifacts/out/aspf_state/<session>/<seq>_<step>.delta.jsonl`
- `artifacts/out/aspf_state/<session>/<seq>_<step>.action_plan.json`
- `artifacts/out/aspf_state/<session>/<seq>_<step>.action_plan.md`
- `artifacts/out/aspf_handoff_manifest.json`

## 4) VS Code-assisted remediation loop

Use this loop when you want quick fix-and-verify cycles in the editor.

### Start from the CLI once (optional warm-up)
```bash
mise exec -- python -m gabion check run \
  --aspf-state-json artifacts/out/aspf_state/session-vscode/0001_check-run.snapshot.json \
  --aspf-delta-jsonl artifacts/out/aspf_state/session-vscode/0001_check-run.delta.jsonl
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
mise exec -- python -m gabion check run \
  --aspf-state-json artifacts/out/aspf_state/session-vscode/0002_check-run.snapshot.json \
  --aspf-delta-jsonl artifacts/out/aspf_state/session-vscode/0002_check-run.delta.jsonl \
  --aspf-import-state artifacts/out/aspf_state/session-vscode/0001_check-run.snapshot.json
```

## 5) CI/PR loop

Use this loop to review whether a PR is healthy and whether cache reuse behaved as expected.

### Artifacts to inspect
- Dataflow grammar report artifact (`dataflow_report.md` output family).
- Job/step summaries emitted by CI helper scripts.
- PR comment output (same-repo PRs) from the dataflow grammar workflow.

### ASPF continuity signals
Look at:
- `artifacts/out/aspf_handoff_manifest.json` entries (`sequence`, `import_state_paths`, `status`).
- per-step `*.snapshot.json` and `*.delta.jsonl` artifacts for deterministic continuation.
- per-step action plans (`*.action_plan.json`/`*.action_plan.md`) for ranked cleanup work.

### Re-run with stable identity settings
If continuity behavior seems unexpectedly poor, re-run while keeping identity-affecting
settings stable (strictness profile, external filtering mode, and semantic surfaces).
For local reproduction before pushing another commit:
```bash
mise exec -- python -m gabion check run \
  --aspf-state-json artifacts/out/aspf_state/session-ci/0002_check-run.snapshot.json \
  --aspf-delta-jsonl artifacts/out/aspf_state/session-ci/0002_check-run.delta.jsonl \
  --aspf-import-state artifacts/out/aspf_state/session-ci/0001_check-run.snapshot.json
```

For deeper policy/governance expectations around CI execution and protections,
use `POLICY_SEED.md#policy_seed` as the canonical source.

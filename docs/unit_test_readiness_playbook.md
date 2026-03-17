---
doc_revision: 1
reader_reintern: "Reader-only: re-intern if doc_revision changed since you last read this doc."
doc_id: unit_test_readiness_playbook
doc_role: guide
doc_scope:
  - repo
  - planning
  - testing
  - workflows
doc_authority: informative
doc_requires:
  - README.md#repo_contract
  - CONTRIBUTING.md#contributing_contract
  - AGENTS.md#agent_obligations
  - POLICY_SEED.md#policy_seed
  - glossary.md#contract
  - docs/planning_substrate.md#planning_substrate
  - docs/generated_artifact_manifest.md#generated_artifact_manifest
doc_reviewed_as_of:
  README.md#repo_contract: 2
  CONTRIBUTING.md#contributing_contract: 2
  AGENTS.md#agent_obligations: 2
  POLICY_SEED.md#policy_seed: 2
  glossary.md#contract: 1
  docs/planning_substrate.md#planning_substrate: 1
  docs/generated_artifact_manifest.md#generated_artifact_manifest: 1
doc_review_notes:
  README.md#repo_contract: "Reviewed README.md#repo_contract rev84/section v2 (repo-drain readiness remains a repo-local workflow/tooling concern rather than a product-facing feature)."
  CONTRIBUTING.md#contributing_contract: "Reviewed CONTRIBUTING.md#contributing_contract rev120/section v2 (correction-unit cadence and validation stack align with the shared UTR workflow)."
  AGENTS.md#agent_obligations: "Reviewed AGENTS.md#agent_obligations rev37/section v2 (agent correction-unit drainage and dual-sensor obligations align with the shared UTR workflow)."
  POLICY_SEED.md#policy_seed: "Reviewed POLICY_SEED.md#policy_seed rev57/section v2 (process-relative runtime and the distinction ladder justify documenting UTR as a real planning-runtime distinction)."
  glossary.md#contract: "Reviewed glossary.md#contract rev47/section v1 (queue/root/workstream semantics and runtime-distinction admissibility remain aligned with UTR)."
  docs/planning_substrate.md#planning_substrate: "Reviewed planning_substrate rev3 (planning roots, touchpoints, and artifact-ingress runtime remain the architectural basis for UTR)."
  docs/generated_artifact_manifest.md#generated_artifact_manifest: "Reviewed generated_artifact_manifest rev1 (junit/log and invariant-workstream artifacts remain the canonical UTR feed/projection surfaces)."
doc_sections:
  unit_test_readiness_playbook: 1
doc_change_protocol: "POLICY_SEED.md#change_protocol"
doc_owner: maintainer
---

<a id="unit_test_readiness_playbook"></a>
# Unit-Test Readiness Playbook

`UTR` is the planning-substrate root for full-suite unit-test readiness. Use it
when the full repo pytest suite is red and you need a machine-readable view of
the current failure clusters before repo-drain readiness can be considered
restored.

Cross-references:

- [README.md#repo_contract](../README.md#repo_contract)
- [CONTRIBUTING.md#contributing_contract](../CONTRIBUTING.md#contributing_contract)
- [AGENTS.md#agent_obligations](../AGENTS.md#agent_obligations)
- [POLICY_SEED.md#policy_seed](../POLICY_SEED.md#policy_seed)
- [glossary.md#contract](../glossary.md#contract)
- [docs/planning_substrate.md#planning_substrate](./planning_substrate.md#planning_substrate)
- [docs/generated_artifact_manifest.md#generated_artifact_manifest](./generated_artifact_manifest.md#generated_artifact_manifest)

`UTR` is not:

- a replacement for owner roots such as `SCC`, `RCI`, or `BIC`,
- a policy/docflow/ambiguity umbrella root,
- a prose-only incident note.

Its canonical feed artifacts are:

- `artifacts/test_runs/junit.xml`
- `artifacts/test_runs/pytest.log`

When the stable planning projection path is written, `UTR` also becomes visible
in:

- `artifacts/out/invariant_workstreams.json`

## Seed playbook

Use this when the full suite is red and you want the current `UTR` state.

1. Seed the canonical pytest feed:

```bash
mise exec -- python -m pytest --junitxml artifacts/test_runs/junit.xml --log-file artifacts/test_runs/pytest.log --log-file-level=INFO
```

2. Inspect the current root projection:

```bash
mise exec -- python -m gabion.tooling.runtime.invariant_graph workstream --object-id UTR
```

3. If you are using the stable artifact-writing projection path, confirm that
   `UTR` is present in `artifacts/out/invariant_workstreams.json`.

Interpret the seeded state as follows:

- the root failing-test count is the number of unique failing test cases
  currently matched into `UTR`,
- touchpoint counts may overlap when one failing file intentionally belongs to
  multiple readiness buckets,
- root and subqueue status stay `in_progress` until the full readiness surface
  is clear, even if one touchpoint has already landed.

## Correction-unit playbook

Use this when one `UTR` touchpoint is the active correction unit.

1. Choose one active touchpoint and use its declared selector set as the target
   failure slice.
2. Run targeted pytest for that slice.
3. After the local fix, refresh the current-indicator feed cheaply:

```bash
mise exec -- python -m pytest --lf --junitxml artifacts/test_runs/junit.xml --log-file artifacts/test_runs/pytest.log --log-file-level=INFO
```

4. Reinspect `UTR` and confirm that the relevant touchpoint count shrank or
   cleared.
5. Run the required correction-unit validation stack before drainage.

Important interpretation rules:

- one touchpoint can land while the root remains `in_progress`,
- a green targeted slice without a refreshed junit feed is not enough to claim
  `UTR` movement,
- once a refreshed feed exposes a different remaining cluster, that becomes the
  next correction unit rather than a reason to reopen the completed one.

## Closeout playbook

Use this when the remaining active `UTR` touchpoints appear clear.

1. Rerun the full suite with the canonical junit/log outputs:

```bash
mise exec -- python -m pytest --junitxml artifacts/test_runs/junit.xml --log-file artifacts/test_runs/pytest.log --log-file-level=INFO
```

2. Reinspect `UTR` and confirm there are no blocking current indicators.
3. Land any remaining active touchpoints, then the subqueues they were
   blocking, then the root.
4. Drain the final closeout correction unit only after the refreshed `UTR`
   projection shows no remaining unit-test readiness blockers.

## Practical command loop

Common repo-local commands for `UTR` work are:

```bash
mise exec -- python -m pytest --junitxml artifacts/test_runs/junit.xml --log-file artifacts/test_runs/pytest.log --log-file-level=INFO
mise exec -- python -m pytest --lf --junitxml artifacts/test_runs/junit.xml --log-file artifacts/test_runs/pytest.log --log-file-level=INFO
mise exec -- python -m gabion.tooling.runtime.invariant_graph workstream --object-id UTR
mise exec -- python -m scripts.policy.policy_check --workflows
```

Use the full-suite command at seed and closeout. Use the `--lf` refresh only
between correction units when you want a cheap current-indicator update against
the existing red field.

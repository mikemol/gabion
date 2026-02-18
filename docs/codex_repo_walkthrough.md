---
doc_revision: 2
reader_reintern: "Reader-only: re-intern if doc_revision changed since you last read this doc."
doc_id: codex_repo_walkthrough
doc_role: process
doc_scope:
  - repo
  - contributors
  - agents
  - tooling
doc_authority: informative
doc_requires:
  - README.md#repo_contract
  - CONTRIBUTING.md#contributing_contract
  - POLICY_SEED.md#policy_seed
  - glossary.md#contract
  - docs/coverage_semantics.md#coverage_semantics
  - docs/sppf_checklist.md#sppf_checklist
doc_reviewed_as_of:
  README.md#repo_contract: 1
  CONTRIBUTING.md#contributing_contract: 1
  POLICY_SEED.md#policy_seed: 1
  glossary.md#contract: 1
  docs/coverage_semantics.md#coverage_semantics: 1
  docs/sppf_checklist.md#sppf_checklist: 1
doc_review_notes:
  README.md#repo_contract: "Reviewed branch model, execution entry points, and quick-start check commands for this walkthrough."
  CONTRIBUTING.md#contributing_contract: "Reviewed contributor workflow constraints and quality gates used by Codex during repo study."
  POLICY_SEED.md#policy_seed: "Reviewed normative execution/security policy and change protocol expectations."
  glossary.md#contract: "Reviewed semantic contract and commutation obligations used as analysis truth source."
  docs/coverage_semantics.md#coverage_semantics: "Reviewed evidence-oriented coverage model and ratchet expectations."
  docs/sppf_checklist.md#sppf_checklist: "Reviewed convergence checklist usage for planning and debt burn-down tracking."
doc_sections:
  codex_walkthrough: 1
doc_change_protocol: "POLICY_SEED.md#change_protocol"
doc_erasure:
  - formatting
  - typos
doc_owner: maintainer
---

<a id="codex_walkthrough"></a>
# Codex Repository Walkthrough and Operating Model

This guide is for bringing a fresh Codex instance up to speed quickly and safely.
It explains what we are building, how we validate it, and how to produce useful
work packets for debt burn-down.

## 1. What Gabion is doing

Gabion is an architectural linter and refactor planner for Python. The core goal
is to detect recurring parameter bundles (implicit structure), project those
findings into evidence, and eventually reify them into explicit Protocol-style
structures.

In short:
- detect structure,
- prove evidence,
- synthesize/refactor conservatively,
- ratchet quality upward without regressions.

## 2. What to read first (and why)

For every new Codex session, read these in order:
1. `POLICY_SEED.md#policy_seed` (normative execution and governance contract)
2. `glossary.md#contract` (semantic meaning contract)
3. `README.md#repo_contract` (scope, branch model, entry points)
4. `CONTRIBUTING.md#contributing_contract` (workflow and validation rules)
5. `docs/coverage_semantics.md#coverage_semantics` (coverage as semantic evidence)

Why this order matters:
- policy + glossary define **validity**,
- readme + contributing define **execution workflow**,
- coverage semantics define **proof expectations**.

## 3. Architecture map (practical)

Primary runtime architecture:
- `src/gabion/server.py`: semantic core (LSP/server command execution)
- `src/gabion/cli.py`: thin interface over server semantics (LSP-first invariant)
- `src/gabion/analysis/dataflow_audit.py`: core analysis engine and projections
- `src/gabion/analysis/`: supporting analysis modules (impact index, visitors,
  pattern schema, evidence keys, type fingerprints, etc.)
- `src/gabion/synthesis/`: synthesis planning/protocol scaffolding
- `src/gabion/refactor/`: refactor planning and edits
- `tests/`: parity, edge, and semantic regression coverage

Design expectation for Codex:
- treat server output as canonical semantics,
- treat CLI as transport/projection,
- preserve deterministic ordering and deadline-aware execution behavior.

## 4. CI, linting, and auditing model

Gabion uses layered quality gates:
- tests (semantic and edge behavior)
- high-coverage enforcement (execution coverage in CI)
- evidence-surface checks (e.g., drift checks where configured)
- policy/docflow checks for governance and document integrity

Operational commands Codex should use (repo-local tooling):
- `mise exec -- python -m gabion check`
- `mise exec -- python -m pytest`
- targeted tests for changed modules first, then broader suites as needed

Key point: Gabion audits itself. The same analyzer concepts (evidence carriers,
determinism, policy constraints, dataflow grammar discipline) are applied to
Gabion's own codebase and docs.

## 5. How Gabion enforces its own cleanliness

Self-enforcement loops include:
- dataflow grammar checks (bundle detection and obligations)
- evidence-oriented test conventions and projections
- deterministic output expectations (ordering, normalized payloads)
- policy and glossary contracts as normative constraints on changes

Codex should treat "cleanliness" as multi-axis:
- code correctness,
- semantic correctness,
- governance correctness,
- artifact determinism.

## 6. Ratcheting and baseline policy

Ratcheting is "no regression, progressive improvement":
- existing debt can be baselined temporarily,
- net-new debt should be blocked,
- removals should be coupled to tests/evidence,
- baseline should trend toward zero.

Burndown priority should follow leverage:
1. deterministic identity / evidence quality debt,
2. false-positive/false-negative analysis debt,
3. documentation/checklist status drift,
4. ergonomics/performance debt that blocks throughput.

## 7. Burndown execution model (your workflow)

Current operating model:
1. Ask Codex to identify a debt class.
2. Ask Codex for concrete task stubs to attack that class.
3. Open one PR per task (or small coherent bundle of tasks).
4. Use another LLM agent to churn through integration, coverage, and merges.
5. Repeat until class is burned down; then move to next class.

Branch guardrail (compact): routine human integration is via `stage`; `next`
and `release` are automation-only. Treat `README.md#repo_contract` and
`CONTRIBUTING.md#contributing_contract` as the authoritative branch/workflow
references.

Task quality bar:
- bounded scope,
- explicit files/modules,
- acceptance tests,
- deterministic behavior requirements,
- no policy/glossary conflicts.

## 8. How a fresh Codex can reverify all of this

A fresh Codex should prove understanding by running and citing concrete checks.
Suggested sequence:

1. Repo contract and policy read:
   - `sed -n '1,220p' POLICY_SEED.md`
   - `sed -n '1,260p' glossary.md`
   - `sed -n '1,260p' README.md`
   - `sed -n '1,260p' CONTRIBUTING.md`

2. Architecture and command surface discovery:
   - `rg -n "def execute_command|@app\.command|gabion\." src/gabion/server.py src/gabion/cli.py`
   - `rg -n "class |def " src/gabion/analysis/dataflow_audit.py | head -n 120`

3. Test and evidence surface reconnaissance:
   - `rg --files tests | wc -l`
   - `rg -n "^def test_" tests | head -n 120`
   - `sed -n '1,260p' docs/coverage_semantics.md`

4. Baseline and ratchet reconnaissance:
   - `rg -n "baseline|ratchet|waiver|drift" src docs tests`
   - `sed -n '1,260p' docs/sppf_checklist.md`

5. Clean-state confirmation before proposing changes:
   - `git status --short`

The Codex response should include:
- concise architecture summary,
- one or more debt classes,
- prioritized task stubs,
- exact commands used.

## 9. Why this process works

This process separates concerns cleanly:
- discovery/planning (Codex agent focused on analysis quality),
- implementation/integration (separate LLM/human merge pipeline),
- governance guardrails (policy/glossary/coverage/docflow).

That separation reduces context collapse and keeps debt burn-down iterative,
auditable, and high-throughput.

---
doc_revision: 8
reader_reintern: "Reader-only: re-intern if doc_revision changed since you last read this doc."
doc_id: enforceable_rules_cheat_sheet
doc_role: reference
doc_scope:
  - repo
  - agents
  - policy
  - ci
  - tooling
  - testing
doc_authority: informative
doc_requires:
  - POLICY_SEED.md#policy_seed
  - AGENTS.md#agent_obligations
  - README.md#repo_contract
  - CONTRIBUTING.md#contributing_contract
  - glossary.md#contract
  - docs/coverage_semantics.md#coverage_semantics
  - docs/publishing_practices.md#publishing_practices
  - docs/allowed_actions.txt
  - in/in-23.md#in_in_23
  - in/in-24.md#in_in_24
doc_reviewed_as_of:
  POLICY_SEED.md#policy_seed: 2
  AGENTS.md#agent_obligations: 2
  README.md#repo_contract: 2
  CONTRIBUTING.md#contributing_contract: 2
  glossary.md#contract: 1
  docs/coverage_semantics.md#coverage_semantics: 1
  docs/publishing_practices.md#publishing_practices: 1
  in/in-23.md#in_in_23: 1
  in/in-24.md#in_in_24: 1
doc_review_notes:
  POLICY_SEED.md#policy_seed: "Reviewed POLICY_SEED.md rev2 (forward-remediation order, ci_watch failure-bundle durability, and enforced execution-coverage policy wording)."
  AGENTS.md#agent_obligations: "Reviewed AGENTS.md rev2 (required validation stack, forward-remediation preference, and ci_watch failure-bundle triage guidance)."
  README.md#repo_contract: "Reviewed README.md rev2 (removed stale ASPF action-plan CLI/examples; continuation docs now state/delta only)."
  CONTRIBUTING.md#contributing_contract: "Reviewed CONTRIBUTING.md rev2 (two-stage dual-sensor cadence, correction-unit validation stack, and strict-coverage trigger guidance)."
  glossary.md#contract: "Reviewed semantic typing rules for bundle/tier/protocol/decision/evidence terms referenced by this checklist."
  docs/coverage_semantics.md#coverage_semantics: "Reviewed required rule/evidence coverage obligations and ratchet rules mapped in this checklist."
  docs/publishing_practices.md#publishing_practices: "Reviewed release hardening and trusted publishing practices referenced by policy-linked security checks."
  in/in-23.md#in_in_23: "Reviewed ASPF/SPPF carrier discipline and deterministic artifact obligations relevant to ordering guarantees."
  in/in-24.md#in_in_24: "Reviewed deadness/evidence artifact contract and determinism requirements that require canonical sorted serialization surfaces."
doc_sections:
  enforceable_rules_cheat_sheet: 1
doc_section_requires:
  enforceable_rules_cheat_sheet:
    - POLICY_SEED.md#policy_seed
    - AGENTS.md#agent_obligations
    - README.md#repo_contract
    - CONTRIBUTING.md#contributing_contract
    - glossary.md#contract
    - docs/coverage_semantics.md#coverage_semantics
    - docs/publishing_practices.md#publishing_practices
    - docs/allowed_actions.txt
    - in/in-23.md#in_in_23
    - in/in-24.md#in_in_24
doc_section_reviews:
  enforceable_rules_cheat_sheet:
    POLICY_SEED.md#policy_seed:
      dep_version: 2
      self_version_at_review: 1
      outcome: no_change
      note: "Policy seed rev2 reviewed; governance obligations remain aligned."
    AGENTS.md#agent_obligations:
      dep_version: 2
      self_version_at_review: 1
      outcome: no_change
      note: "Agent obligations rev2 reviewed; clause and cadence links remain aligned."
    README.md#repo_contract:
      dep_version: 2
      self_version_at_review: 1
      outcome: no_change
      note: "Repo contract rev2 reviewed; command and artifact guidance remains aligned."
    CONTRIBUTING.md#contributing_contract:
      dep_version: 2
      self_version_at_review: 1
      outcome: no_change
      note: "Contributor contract rev2 reviewed; dual-sensor cadence and correction gates remain aligned."
    glossary.md#contract:
      dep_version: 1
      self_version_at_review: 1
      outcome: no_change
      note: "Glossary semantics remain the canonical meaning source for checklist rule IDs."
    docs/coverage_semantics.md#coverage_semantics:
      dep_version: 1
      self_version_at_review: 1
      outcome: no_change
      note: "Coverage policy remains consistent with checklist testing and evidence obligations."
    docs/publishing_practices.md#publishing_practices:
      dep_version: 1
      self_version_at_review: 1
      outcome: no_change
      note: "Publishing guidance remains advisory and policy-compatible for release hardening references."
    in/in-23.md#in_in_23:
      dep_version: 1
      self_version_at_review: 2
      outcome: no_change
      note: "ASPF/SPPF carrier constraints align with deterministic checklist interpretation."
    in/in-24.md#in_in_24:
      dep_version: 1
      self_version_at_review: 2
      outcome: no_change
      note: "Deadness/evidence determinism constraints align with canonical ordering expectations."
doc_change_protocol: "POLICY_SEED.md#change_protocol"
doc_erasure:
  - formatting
  - typos
doc_owner: maintainer
---

<a id="enforceable_rules_cheat_sheet"></a>
# Enforceable Rules Cheat Sheet

This document is **authoritative by proxy via cited canonical clauses**. It is
not a new normative root. Canonical authority remains:
- `POLICY_SEED.md#policy_seed` (execution/CI safety)
- `[glossary.md#contract](../glossary.md#contract)` (semantic correctness)

## Authority and Precedence

Precedence stack for interpretation and enforcement:
1. `POLICY_SEED.md#policy_seed` (execution and CI safety)
2. `[glossary.md#contract](../glossary.md#contract)` (semantic meaning and commutation)
3. `AGENTS.md#agent_obligations` (agent operating constraints)
4. `README.md#repo_contract` and `CONTRIBUTING.md#contributing_contract` (repo/workflow contract)

No rule in this cheat sheet is valid unless it is traceable to canonical
clauses in the source documents above.

Gate-level control-loop mapping lives in `docs/governance_loop_matrix.md#governance_loop_matrix`.

## Rule Matrix (Traceable)

| Rule ID | Enforceable Rule | Source Clause(s) | Operational Check | Failure Signal |
| --- | --- | --- | --- | --- |
| `SEC-001` | Self-hosted execution must stay on trusted triggers/actors/branches; no untrusted code on self-hosted runners. | [`POLICY_SEED.md#policy_seed` §1](../POLICY_SEED.md#policy_seed), [`POLICY_SEED.md#policy_seed` §4.1](../POLICY_SEED.md#policy_seed), [`POLICY_SEED.md#policy_seed` §4.3](../POLICY_SEED.md#policy_seed) | `mise exec -- python scripts/policy_check.py --workflows` | Policy check violation on trigger/actor guards, or manual review blocks merge. |
| `SEC-002` | Actions must be allow-listed and pinned to full commit SHAs. | [`POLICY_SEED.md#policy_seed` §4.5](../POLICY_SEED.md#policy_seed), [`docs/allowed_actions.txt`](./allowed_actions.txt) | `mise exec -- python scripts/policy_check.py --workflows` | Policy check reports unpinned or disallowed action usage. |
| `SEC-003` | Minimal token permissions only; self-hosted jobs must not request write scopes. | [`POLICY_SEED.md#policy_seed` §4.4](../POLICY_SEED.md#policy_seed) | `mise exec -- python scripts/policy_check.py --workflows` | Workflow policy failure on excessive permissions. |
| `SEC-004` | Self-hosted dependency installs are restricted to allow-listed registries and standard package managers. | [`POLICY_SEED.md#policy_seed` §4.6](../POLICY_SEED.md#policy_seed) | Workflow/script review + `scripts/policy_check.py --workflows` | Governance rejection for arbitrary download patterns or non-allow-listed sources. |
| `ARC-001` | LSP-first invariant: server is semantic core; CLI must remain a thin client (no reimplemented analysis logic). | [`README.md#repo_contract`](../README.md#repo_contract), [`CONTRIBUTING.md#contributing_contract`](../CONTRIBUTING.md#contributing_contract), [`AGENTS.md#agent_obligations`](../AGENTS.md#agent_obligations) | `mise exec -- python -m pytest` + code review of `src/gabion/cli.py` vs `src/gabion/server.py` | Behavioral divergence between CLI and server semantics or review rejection. |
| `ARC-002` | Repo-local tooling commands use `mise exec -- python` to bind pinned interpreter/deps. | [`AGENTS.md#agent_obligations`](../AGENTS.md#agent_obligations), [`CONTRIBUTING.md#contributing_contract`](../CONTRIBUTING.md#contributing_contract) | Command/scripts review; run checks through `mise exec -- python ...` | Execution drift or non-reproducible local/CI behavior. |
| `DFG-001` | Cross-boundary recurring bundles must be Protocol-reified or explicitly marked with `# dataflow-bundle:`; Tier-2 must be reified before merge. | [`AGENTS.md#agent_obligations`](../AGENTS.md#agent_obligations), [`glossary.md#bundle`](../glossary.md#bundle), [`glossary.md#tier`](../glossary.md#tier), [`CONTRIBUTING.md#contributing_contract`](../CONTRIBUTING.md#contributing_contract) | `mise exec -- python -m gabion check` | Dataflow grammar violations in report/lint outputs; merge blocked. |
| `DFG-002` | Ambiguity must be normalized at boundaries and reified structurally; no ad-hoc core alternation/sentinel shortcuts. | [`NCI-SHIFT-AMBIGUITY-LEFT`](./normative_clause_index.md#clause-shift-ambiguity-left), [`POLICY_SEED.md#policy_seed` §4.8](../POLICY_SEED.md#policy_seed) | `mise exec -- python -m gabion ambiguity-contract-gate --root .` | Ambiguity-contract gate fails when violations are present. |
| `DFG-003` | Repository JSON/mapping evidence surfaces must be canonically sorted to preserve deterministic replay and stable evidence artifacts. | [`glossary.md#contract`](../glossary.md#contract), [`in/in-23.md#in_in_23`](../in/in-23.md#in_in_23), [`in/in-24.md#in_in_24`](../in/in-24.md#in_in_24) | `mise exec -- python -m pytest tests/test_runtime_kernel_contracts.py` | Nondeterministic key ordering causes unstable artifacts/diffs and replay evidence drift. |
| `DFG-004` | Any enforced sortedness must disclose sort key/function (or comparator shape) and rationale; non-lexical sorting must declare comparator components. | [`POLICY_SEED.md#policy_seed` §4.9](../POLICY_SEED.md#policy_seed), [`CONTRIBUTING.md#contributing_contract`](../CONTRIBUTING.md#contributing_contract), [`in/in-24.md#in_in_24`](../in/in-24.md#in_in_24) | Code review + targeted ordering contract tests (`mise exec -- python -m pytest tests/test_command_boundary_order.py tests/test_runtime_kernel_contracts.py`) | Undocumented ordering semantics or silent non-lexical sorting leading to semantically ambiguous or unstable output. |
| `DFG-005` | Single-sort lifetime ratchet: each carrier may consume active sorting at most once; egress must enforce order without serializer `sort_keys=True` fallback for canonical carriers. | [`POLICY_SEED.md#policy_seed` §4.9](../POLICY_SEED.md#policy_seed), [`CONTRIBUTING.md#contributing_contract`](../CONTRIBUTING.md#contributing_contract) | `mise exec -- python scripts/order_lifetime_check.py --root .` | Second-sort attempts, boundary/runtime raw sorting shortcuts, or `json.dumps(..., sort_keys=True)` fallback in protected surfaces. |
| `DFG-006` | Derivation-cache identity surfaces must use structural interning keys (ASPF/forest-derived) and must not use digest/text canonicalization for key identity. | [`glossary.md#hash_consing`](../glossary.md#hash_consing), [`glossary.md#aspf`](../glossary.md#aspf), [`POLICY_SEED.md#policy_seed` §0.1](../POLICY_SEED.md#policy_seed) | `mise exec -- python scripts/structural_hash_policy_check.py --root .` | Structural hash policy check reports digest/hashlib or text-canonicalization keying in derivation identity paths. |
| `DEC-001` | Repeated decision logic must be centralized (Decision Bundle) or explicitly documented as Tier-3 (Decision Table). | [`glossary.md#decision_bundle`](../glossary.md#decision_bundle), [`glossary.md#decision_table`](../glossary.md#decision_table) | Decision-flow tests + code review | Duplicate branch logic remains scattered without declared tier treatment. |
| `DEC-002` | Critical cross-boundary decision logic must be represented as a Decision Protocol with explicit validation. | [`glossary.md#decision_protocol`](../glossary.md#decision_protocol), [`glossary.md#decision_surface`](../glossary.md#decision_surface) | Schema/edge tests (`mise exec -- python -m pytest`) | Invalid decision states pass, or ad-hoc branches bypass protocol constraints. |
| `TST-001` | Coverage is rule/evidence-oriented; new/changed invariants require positive, negative, and edge coverage. | [`docs/coverage_semantics.md#coverage_semantics` §1.2](./coverage_semantics.md#coverage_semantics), [`docs/coverage_semantics.md#coverage_semantics` §2](./coverage_semantics.md#coverage_semantics) | `mise exec -- python -m pytest --cov=src/gabion --cov-branch --cov-report=term-missing --cov-fail-under=100` + review | Invariant changes without rule-triad tests; coverage policy non-compliance. |
| `TST-002` | Evidence surface is explicit Evidence IDs, not percentages; test evidence carrier must stay deterministic. | [`glossary.md#evidence_id`](../glossary.md#evidence_id), [`glossary.md#evidence_surface`](../glossary.md#evidence_surface), [`docs/coverage_semantics.md#coverage_semantics` §1.6](./coverage_semantics.md#coverage_semantics) | `mise exec -- python scripts/extract_test_evidence.py --root . --tests tests --out out/test_evidence.json` then `git diff --exit-code out/test_evidence.json` | Evidence carrier drift or unmapped evidence surprises. |
| `POL-001` | Tests must not use monkeypatch/patch-style runtime mutation; seam control is DI-only. | [`AGENTS.md#agent_obligations`](../AGENTS.md#agent_obligations), [`CONTRIBUTING.md#contributing_contract`](../CONTRIBUTING.md#contributing_contract) | `mise exec -- python scripts/no_monkeypatch_policy_check.py --root .` | Monkeypatch/patched runtime mutation detected in tests. |
| `POL-002` | Branch constructs outside explicit Decision Protocol surfaces are violations (hard-zero). | [`AGENTS.md#agent_obligations`](../AGENTS.md#agent_obligations), [`glossary.md#decision_protocol`](../glossary.md#decision_protocol) | `mise exec -- python scripts/branchless_policy_check.py --root .` | Branchless policy violation reported. |
| `POL-003` | Defensive fallback/sentinel continuation outside approved boundaries are violations (hard-zero). | [`AGENTS.md#agent_obligations`](../AGENTS.md#agent_obligations), [`CONTRIBUTING.md#contributing_contract`](../CONTRIBUTING.md#contributing_contract) | `mise exec -- python scripts/defensive_fallback_policy_check.py --root .` | Defensive fallback policy violation reported. |
| `ACP-001` | Semantic core ambiguity signatures must be zero (`isinstance`, dynamic type alternation, sentinel control outcomes). | [`NCI-SHIFT-AMBIGUITY-LEFT`](./normative_clause_index.md#clause-shift-ambiguity-left), [`docs/architecture_zones.md#architecture_zones`](./architecture_zones.md#architecture_zones) | `mise exec -- python -m gabion ambiguity-contract-gate --root .` | Ambiguity-contract gate reports policy violations. |
| `DOC-001` | Documentation review stamping must be real (no mechanical `doc_reviewed_as_of` updates; explicit review notes required). | [`CONTRIBUTING.md#contributing_contract`](../CONTRIBUTING.md#contributing_contract), [`AGENTS.md#agent_obligations`](../AGENTS.md#agent_obligations), [`POLICY_SEED.md#policy_seed` §0.2](../POLICY_SEED.md#policy_seed) | `mise exec -- python -m gabion docflow --root . --fail-on-violations --sppf-gh-ref-mode required` | Docflow violations for review-note/metadata discipline. |
| `DOC-002` | Markdown docs require frontmatter discipline (`doc_revision`, dependency tracking, change protocol linkage). | [`AGENTS.md#agent_obligations`](../AGENTS.md#agent_obligations), [`CONTRIBUTING.md#contributing_contract`](../CONTRIBUTING.md#contributing_contract), [`POLICY_SEED.md#change_protocol`](../POLICY_SEED.md#change_protocol) | `mise exec -- python -m gabion docflow --root . --fail-on-violations --sppf-gh-ref-mode required` | Missing/invalid frontmatter or policy-linkage findings in docflow output. |

## Implementation Guardrails by Change Type

| Change Type | Mandatory Checks | Prohibited Shortcuts | Required Evidence Artifacts | Source Clause(s) |
| --- | --- | --- | --- | --- |
| Workflow/CI YAML (`.github/workflows/*`) | `scripts/policy_check.py --workflows`; run posture check when token is available. | Unpinned actions, disallowed actions, weak actor guards, broad token writes on self-hosted paths. | Policy-check pass output; workflow diff with explicit guard clauses. | [`POLICY_SEED.md#policy_seed` §4](../POLICY_SEED.md#policy_seed), [`CONTRIBUTING.md#contributing_contract`](../CONTRIBUTING.md#contributing_contract) |
| Server semantic core (`src/gabion/server.py`, analysis modules) | `gabion check`, targeted pytest for behavior and decision schemas. | Core-flow ambiguity patching via ad-hoc alternation/sentinels; CLI-side semantic duplication. | Deterministic semantic outputs and test evidence updates when obligations change. | [`POLICY_SEED.md#policy_seed` §4.8](../POLICY_SEED.md#policy_seed), [`AGENTS.md#agent_obligations`](../AGENTS.md#agent_obligations), [`glossary.md#decision_protocol`](../glossary.md#decision_protocol) |
| Tooling/CLI wrappers (`src/gabion/tooling/*`, `src/gabion/cli.py`) | Verify server-delegated semantics via pytest and CLI integration checks. | Reimplementing analysis logic in wrappers; bypassing `mise` execution norms for repo tooling. | Consistent telemetry/report behavior and stable evidence carrier outputs. | [`README.md#repo_contract`](../README.md#repo_contract), [`AGENTS.md#agent_obligations`](../AGENTS.md#agent_obligations) |
| Tests/evidence mapping (`tests/*`, `out/test_evidence.json`) | Rule-triad tests; coverage/evidence checks; evidence extraction drift check. | Counting line coverage as sole proof; uncited invariant claims without witness mapping. | `out/test_evidence.json` (canonical), coverage artifacts under `artifacts/test_runs/`. | [`docs/coverage_semantics.md#coverage_semantics`](./coverage_semantics.md#coverage_semantics), [`glossary.md#evidence_surface`](../glossary.md#evidence_surface) |
| Docs/governance (`*.md`) | `gabion docflow`; update `doc_revision` for conceptual changes; keep review notes explicit. | Mechanical review stamping, uncited new mandatory rules, policy weakening without protocol update. | Docflow pass output and citation-backed clause links in changed docs. | [`POLICY_SEED.md#change_protocol`](../POLICY_SEED.md#change_protocol), [`CONTRIBUTING.md#contributing_contract`](../CONTRIBUTING.md#contributing_contract), [`AGENTS.md#agent_obligations`](../AGENTS.md#agent_obligations) |

## Quick Validation Commands

Run these from repo root.

```bash
mise exec -- python scripts/policy_check.py --workflows
mise exec -- python -m gabion docflow --root . --fail-on-violations --sppf-gh-ref-mode required
mise exec -- python scripts/no_monkeypatch_policy_check.py --root .
mise exec -- python scripts/branchless_policy_check.py --root .
mise exec -- python scripts/defensive_fallback_policy_check.py --root .
mise exec -- python -m gabion ambiguity-contract-gate --root .
mise exec -- python scripts/order_lifetime_check.py --root .
mise exec -- python scripts/structural_hash_policy_check.py --root .
mise exec -- python -m gabion check
mise exec -- python -m pytest --cov=src/gabion --cov-branch --cov-report=term-missing --cov-fail-under=100
mise exec -- python scripts/extract_test_evidence.py --root . --tests tests --out out/test_evidence.json
git diff --exit-code out/test_evidence.json
```

Optional governance sanity:

```bash
mise exec -- python -m gabion status-consistency --fail-on-violations
```

Pass/fail interpretation:
- Nonzero exit on any required command means at least one enforceable clause is
  not currently satisfied.
- A clean run requires zero exits across required checks plus no unexpected
  evidence drift in `out/test_evidence.json`.

## Interpretation Discipline

- If this cheat sheet conflicts with canonical text, canonical text wins:
  `POLICY_SEED.md#policy_seed` and `[glossary.md#contract](../glossary.md#contract)` are authoritative.
- Ambiguity in interpretation is resolved by adding citation-backed
  clarifications to this file; do not introduce uncited mandatory rules.

## Change Protocol for This Cheat Sheet

Any new or modified rule entry in this file must include:
1. At least one canonical source anchor citation.
2. A concrete operational check command/script/test.
3. A concrete failure signal that can block acceptance.

This update discipline follows `POLICY_SEED.md#change_protocol` and preserves
"authoritative by proxy" status without creating a new normative root.

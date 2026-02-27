---
doc_revision: 21
reader_reintern: "Reader-only: re-intern if doc_revision changed since you last read this doc."
doc_id: coverage_semantics
doc_role: policy
doc_scope:
  - repo
  - testing
  - coverage
  - analysis
  - governance
doc_authority: normative
doc_dependency_projection: glossary_lifted
doc_requires:
  - POLICY_SEED.md#policy_seed
  - glossary.md#contract
  - glossary.md#rule_of_polysemy
  - glossary.md#bundle
  - glossary.md#tier
  - glossary.md#decision_table
  - glossary.md#decision_bundle
  - glossary.md#decision_protocol
  - glossary.md#evidence_surface
  - README.md#repo_contract
  - CONTRIBUTING.md#contributing_contract
  - AGENTS.md#agent_obligations
doc_reviewed_as_of:
  POLICY_SEED.md#policy_seed: 2
  glossary.md#contract: 1
  glossary.md#rule_of_polysemy: 1
  glossary.md#bundle: 1
  glossary.md#tier: 1
  glossary.md#decision_table: 1
  glossary.md#decision_bundle: 1
  glossary.md#decision_protocol: 1
  glossary.md#evidence_surface: 1
  README.md#repo_contract: 2
  CONTRIBUTING.md#contributing_contract: 2
  AGENTS.md#agent_obligations: 2
doc_review_notes:
  POLICY_SEED.md#policy_seed: "Reviewed POLICY_SEED.md rev2 (forward-remediation order, ci_watch failure-bundle durability, and enforced execution-coverage policy wording)."
  glossary.md#contract: "Reviewed glossary.md#contract rev1 (glossary contract + semantic typing discipline)."
  glossary.md#rule_of_polysemy: "Reviewed glossary.md#rule_of_polysemy rev1 (polysemy axes + commutation obligations)."
  glossary.md#bundle: "Reviewed glossary.md#bundle rev1 (bundle identity + alias invariance)."
  glossary.md#tier: "Reviewed glossary.md#tier rev1 (tier evidence thresholds + promotion requirements)."
  glossary.md#decision_table: "Reviewed glossary.md#decision_table rev1 (decision table tier definition)."
  glossary.md#decision_bundle: "Reviewed glossary.md#decision_bundle rev1 (decision bundle tier definition)."
  glossary.md#decision_protocol: "Reviewed glossary.md#decision_protocol rev1 (decision protocol tier definition)."
  glossary.md#evidence_surface: "Reviewed glossary.md#evidence_surface rev1 (evidence surfaces bind carriers to documented obligations); glossary-lifted dependencies."
  README.md#repo_contract: "Reviewed README.md rev2 (removed stale ASPF action-plan CLI/examples; continuation docs now state/delta only)."
  CONTRIBUTING.md#contributing_contract: "Reviewed CONTRIBUTING.md rev2 (two-stage dual-sensor cadence, correction-unit validation stack, and strict-coverage trigger guidance)."
  AGENTS.md#agent_obligations: "Reviewed AGENTS.md rev2 (required validation stack, forward-remediation preference, and ci_watch failure-bundle triage guidance)."
doc_sections:
  coverage_semantics: 1
doc_section_requires:
  coverage_semantics:
    - POLICY_SEED.md#policy_seed
    - glossary.md#contract
    - glossary.md#rule_of_polysemy
    - glossary.md#bundle
    - glossary.md#tier
    - glossary.md#decision_table
    - glossary.md#decision_bundle
    - glossary.md#decision_protocol
    - glossary.md#evidence_surface
    - README.md#repo_contract
    - CONTRIBUTING.md#contributing_contract
    - AGENTS.md#agent_obligations
doc_section_reviews:
  coverage_semantics:
    POLICY_SEED.md#policy_seed:
      dep_version: 2
      self_version_at_review: 1
      outcome: no_change
      note: "Policy seed rev2 reviewed; governance obligations remain aligned."
    glossary.md#contract:
      dep_version: 1
      self_version_at_review: 1
      outcome: no_change
      note: "Reviewed glossary.md#contract rev1 (glossary contract + semantic typing discipline)."
    glossary.md#rule_of_polysemy:
      dep_version: 1
      self_version_at_review: 1
      outcome: no_change
      note: "Reviewed glossary.md#rule_of_polysemy rev1 (polysemy axes + commutation obligations)."
    glossary.md#bundle:
      dep_version: 1
      self_version_at_review: 1
      outcome: no_change
      note: "Reviewed glossary.md#bundle rev1 (bundle identity + alias invariance)."
    glossary.md#tier:
      dep_version: 1
      self_version_at_review: 1
      outcome: no_change
      note: "Reviewed glossary.md#tier rev1 (tier evidence thresholds + promotion requirements)."
    glossary.md#decision_table:
      dep_version: 1
      self_version_at_review: 1
      outcome: no_change
      note: "Reviewed glossary.md#decision_table rev1 (decision table tier definition)."
    glossary.md#decision_bundle:
      dep_version: 1
      self_version_at_review: 1
      outcome: no_change
      note: "Reviewed glossary.md#decision_bundle rev1 (decision bundle tier definition)."
    glossary.md#decision_protocol:
      dep_version: 1
      self_version_at_review: 1
      outcome: no_change
      note: "Reviewed glossary.md#decision_protocol rev1 (decision protocol tier definition)."
    glossary.md#evidence_surface:
      dep_version: 1
      self_version_at_review: 1
      outcome: no_change
      note: "Reviewed glossary.md#evidence_surface rev1 (evidence surfaces bind carriers to documented obligations); glossary-lifted dependencies."
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
    AGENTS.md#agent_obligations:
      dep_version: 2
      self_version_at_review: 1
      outcome: no_change
      note: "Agent obligations rev2 reviewed; clause and cadence links remain aligned."
doc_change_protocol: "POLICY_SEED.md#change_protocol"
doc_invariants:
  - coverage_is_evidence
  - rule_coverage_required
  - ratchet_only
  - coverage_smell_tracking
  - name: docflow:cover:decision_surface
    kind: cover
    status: active
    cover_evidence_kind: decision_surface
  - name: docflow:cover:function_site
    kind: cover
    status: active
    cover_evidence_kind: function_site
  - name: docflow:cover:call_footprint
    kind: cover
    status: active
    cover_evidence_kind: call_footprint
  - name: docflow:cover:call_cluster
    kind: cover
    status: active
    cover_evidence_kind: call_cluster
doc_erasure:
  - formatting
  - typos
doc_owner: maintainer
---

<a id="coverage_semantics"></a>

# Coverage Semantics (Normative)

Coverage is not a vanity metric in this repository. It is **evidence** that the
semantic invariants described in [glossary.md#contract](glossary.md#contract) and [POLICY_SEED.md#policy_seed](POLICY_SEED.md#policy_seed) are actually
enforced by tests.

This policy is scoped by the repository contract ([README.md#repo_contract](README.md#repo_contract)), workflow
rules ([CONTRIBUTING.md#contributing_contract](CONTRIBUTING.md#contributing_contract)), and agent obligations ([AGENTS.md#agent_obligations](AGENTS.md#agent_obligations)).

Normative pointers (explicit): [POLICY_SEED.md#policy_seed](POLICY_SEED.md#policy_seed), [glossary.md#contract](glossary.md#contract), [glossary.md#rule_of_polysemy](glossary.md#rule_of_polysemy),
[glossary.md#bundle](glossary.md#bundle), [glossary.md#tier](glossary.md#tier), [glossary.md#decision_table](glossary.md#decision_table),
[glossary.md#decision_bundle](glossary.md#decision_bundle), [glossary.md#decision_protocol](glossary.md#decision_protocol), [glossary.md#evidence_surface](glossary.md#evidence_surface),
[README.md#repo_contract](README.md#repo_contract), [CONTRIBUTING.md#contributing_contract](CONTRIBUTING.md#contributing_contract), [AGENTS.md#agent_obligations](AGENTS.md#agent_obligations).

## 1. Coverage Axes (Evidence Types)

Coverage is tracked along multiple axes. Each axis is required for a different
kind of assurance.

### 1.1 Execution coverage (required)
Line/branch coverage shows what code executed, but **does not** guarantee
semantic correctness. This repository still requires full execution coverage as
an enforceable gate for regression control.

`# pragma: no cover` is permitted only when the covered branch is protected by
`never(...)` after ingress validation. Enum exhaustiveness fallbacks should use
this paired pattern so invariant drift faults at one obvious correction point.

### 1.2 Rule coverage (required)
Each normative rule or invariant must be exercised by tests that include:
- **Positive case:** detects a violation.
- **Negative case:** avoids a false positive.
- **Edge case:** aliasing, decorators, stars, or scope boundaries.

This is the minimum evidence threshold for changes to analysis logic.

### 1.3 Grammar/AST feature coverage (required for new features)
When a new AST feature or language construct is supported, add at least one
fixture where the feature **changes the analysis outcome**.

### 1.4 Convergence/commutation coverage (required for invariants)
Metamorphic tests must cover commutation laws in [glossary.md#contract](glossary.md#contract):
- rename/alias invariance of bundle identity
- fixed-point stability of analysis/refactor loops

### 1.5 Decision-flow tier coverage (required when refactoring control-flow)
When control-flow is refactored into decision structures, tests must include:
- **Tier-3 (Decision Table):** documentation aligned with code paths.
- **Tier-2 (Decision Bundle):** centralized guard map with positive/negative cases.
- **Tier-1 (Decision Protocol):** schema validation + edge cases derived from the protocol.

See [glossary.md#decision_table](glossary.md#decision_table), [glossary.md#decision_bundle](glossary.md#decision_bundle), and
[glossary.md#decision_protocol](glossary.md#decision_protocol) for decision-flow tier definitions.

### 1.6 Evidence surface coverage (required)
The canonical semantic coverage carrier is `out/test_evidence.json`. It records
the evidence surface discharged by tests and explicitly lists unmapped tests.

Projections derived from this carrier are advisory and must be deterministic.
JSON projections are written under `artifacts/out/*.json`, with Markdown
companions under `artifacts/audit_reports/*.md` (documentation artifacts).
CI enforces drift control for `out/test_evidence.json` by regenerating it and
failing if the output changes without being committed.

By default, only `out/test_evidence.json` is the **gated evidence carrier**.
Other `artifacts/audit_reports/*.md` files may be committed as
**documentation artifacts** or advisory projections (with docflow frontmatter),
provided they do **not** participate in gating. Non-gated JSON projections
belong in `artifacts/out/`.
This is a controlled polysemy of “artifact”:

- **Evidence artifact (gated):** canonical carrier used by CI (JSON).
- **Documentation artifact (ungated):** narrative or projection surface (Markdown).

These meanings live on orthogonal axes (evidence gating vs documentation) and
commute by erasure: documentation artifacts must not alter the gated evidence
surface or its checks.

The semantic-coverage mapping surface is emitted by
`gabion check --emit-semantic-coverage-map`, reading
`out/semantic_coverage_mapping.json` (or `--semantic-coverage-mapping`) and
writing `artifacts/out/semantic_coverage_map.json` plus
`artifacts/audit_reports/semantic_coverage_map.md`. The projection reports
mapped obligations, unmapped obligations, dead mapping entries, and duplicate
mapping entries.

## 2. Ratchet Policy (No Regression)

Coverage is ratcheted:
- Existing gaps may be baseline-accepted for execution coverage.
- **New or modified rules** MUST include rule/grammar/convergence coverage.
- New tests must be specific to the invariant they protect.

## 2.1 Docflow pattern-class mappings for semantic-core excess (normative)

`out/docflow_compliance.md` excess entries are mapped by reusable classes rather
than per-node enumeration. For semantic-core paths
(`src/gabion/server.py`, `src/gabion/server_core/`, `src/gabion/analysis/`),
the recurring classes are (from the pre-mapping excess snapshot):

- **Decision parameter surface class** (`decision_surface`, frequency **158**):
  direct branch-bearing parameter checks in semantic-core flows.
- **Test-to-core edge class** (`call_footprint`, frequency **8**): test
  callsites invoking semantic-core targets (including adapter-mediated calls).
- **Core adjacency cluster class** (`call_cluster`, frequency **3**): grouped
  helper/core call topology around semantic-core execution.

The class-to-coverage mapping is enforced through active `docflow:cover:*`
kind invariants. Any present/future evidence node of these kinds is covered by
class membership, not by explicit id listing.

## 3. Reporting (Current Practice)

Measurement command (gating):
```
mise exec -- python -m pytest --cov=src/gabion --cov-branch --cov-report=term-missing --cov-fail-under=100
```

CI enforces **100% line + 100% branch coverage** for execution coverage.

CI stores coverage artifacts under `artifacts/test_runs/`:
- `coverage.xml` (machine-readable)
- `htmlcov/` (human-readable)

Coverage reports may be stored under `artifacts/` for review, but enforcement
gates are policy decisions and should follow the ratchet rule above.

## 4. Interpretation Guidance

When assessing coverage:
- Prefer **rule coverage** over raw percentages.
- Treat low execution coverage in analysis cores as a risk signal.
- Require convergence tests whenever heuristics are added or adjusted.

Coverage is evidence, not proof. The goal is to make **semantic regressions
hard to hide**, not to chase a single numeric threshold.

## 5. Coverage Smells (Quality Signals)

### 5.1 Branch-irrelevance smell (required tracking)
**Smell:** A branch or path is exercised by tests, but the test does **not**
assert any invariant (or lemma). This indicates either:
- the invariant is undocumented or missing, or
- the implementation does not surface a violation when the invariant is broken.

This smell **does not count as evidence**. It must be tracked until resolved.

### 5.2 Tracking rule
When the smell is detected, record it as a tracked item with:
- the test(s) involved,
- the missing invariant/lemma, and
- a proposed remediation (document or refactor).

Default tracking vehicle: a GitHub issue labeled `coverage-smell`.
Alternative: an entry in `docs/sppf_checklist.md` if the smell maps to a node.

---
doc_revision: 6
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
doc_requires:
  - POLICY_SEED.md
  - glossary.md
  - README.md
  - CONTRIBUTING.md
  - AGENTS.md
doc_reviewed_as_of:
  POLICY_SEED.md: 28
  glossary.md: 20
  README.md: 58
  CONTRIBUTING.md: 71
  AGENTS.md: 12
doc_change_protocol: "POLICY_SEED.md §6"
doc_invariants:
  - coverage_is_evidence
  - rule_coverage_required
  - ratchet_only
  - coverage_smell_tracking
doc_erasure:
  - formatting
  - typos
doc_owner: maintainer
---

# Coverage Semantics (Normative)

Coverage is not a vanity metric in this repository. It is **evidence** that the
semantic invariants described in `glossary.md` and `POLICY_SEED.md` are actually
enforced by tests.

This policy is scoped by the repository contract (`README.md`), the workflow
rules (`CONTRIBUTING.md`), and agent obligations (`AGENTS.md`).

## 1. Coverage Axes (Evidence Types)

Coverage is tracked along multiple axes. Each axis is required for a different
kind of assurance.

### 1.1 Execution coverage (advisory)
Line/branch coverage shows what code executed, but **does not** guarantee
semantic correctness. It is used for trend monitoring and spot-checking gaps.

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
Metamorphic tests must cover commutation laws in `glossary.md`:
- rename/alias invariance of bundle identity
- fixed-point stability of analysis/refactor loops

### 1.5 Decision-flow tier coverage (required when refactoring control-flow)
When control-flow is refactored into decision structures, tests must include:
- **Tier-3 (Decision Table):** documentation aligned with code paths.
- **Tier-2 (Decision Bundle):** centralized guard map with positive/negative cases.
- **Tier-1 (Decision Protocol):** schema validation + edge cases derived from the protocol.

See `glossary.md` §§12–14 for decision-flow tier definitions.

## 2. Ratchet Policy (No Regression)

Coverage is ratcheted:
- Existing gaps may be baseline-accepted for execution coverage.
- **New or modified rules** MUST include rule/grammar/convergence coverage.
- New tests must be specific to the invariant they protect.

## 3. Reporting (Current Practice)

Measurement command (advisory, not gating by default):
```
mise exec -- python -m pytest --cov=src/gabion --cov-report=term-missing
```

CI enforces **100% line coverage** for execution coverage (fail-under=100).
Branch coverage remains **advisory** and may be measured locally:
```
mise exec -- python -m pytest --cov=src/gabion --cov-branch --cov-report=term-missing
```

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

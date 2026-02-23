---
doc_revision: 11
reader_reintern: Reader-only: re-intern if doc_revision changed since you last read this doc.
doc_id: matrix_acceptance
doc_role: reference
doc_scope:
  - repo
  - testing
  - semantics
  - governance
doc_authority: informative
doc_requires:
  - POLICY_SEED.md#policy_seed
  - glossary.md#contract
  - glossary.md#exception_obligation
  - CONTRIBUTING.md#contributing_contract
  - README.md#repo_contract
  - in/in-24.md#in_in_24
  - in/in-25.md#in_in_25
  - in/in-26.md#in_in_26
  - in/in-27.md#in_in_27
doc_reviewed_as_of:
  POLICY_SEED.md#policy_seed: 1
  glossary.md#contract: 1
  glossary.md#exception_obligation: 1
  CONTRIBUTING.md#contributing_contract: 1
  README.md#repo_contract: 1
  in/in-24.md#in_in_24: 1
  in/in-25.md#in_in_25: 1
  in/in-26.md#in_in_26: 1
  in/in-27.md#in_in_27: 1
doc_review_notes:
  POLICY_SEED.md#policy_seed: Reviewed POLICY_SEED.md rev1 (mechanized governance default; branch/tag CAS + check-before-use constraints); no conflicts with this document's scope.
  glossary.md#contract: Reviewed glossary.md#contract rev1 (glossary contract + semantic typing discipline).
  glossary.md#exception_obligation: Reviewed glossary.md#exception_obligation rev1 (exception obligation status + evidence linkage).
  CONTRIBUTING.md#contributing_contract: Reviewed CONTRIBUTING.md rev1 (docflow now fails on missing GH references for SPPF-relevant changes); no conflicts with this document's scope.
  README.md#repo_contract: Reviewed README.md rev1 (docflow audit now scans in/ by default); no conflicts with this document's scope.
  in/in-24.md#in_in_24: Reviewed in/in-24.md rev8 (deadness matrix acceptance checks align with artifact schema requirements).
  in/in-25.md#in_in_25: Reviewed in/in-25.md rev8 (coherence matrix acceptance mapping remains consistent with evidence artifacts).
  in/in-26.md#in_in_26: Reviewed in/in-26.md rev8 (rewrite-plan verification predicates match matrix obligations).
  in/in-27.md#in_in_27: Reviewed in/in-27.md rev6 (exception obligation mapping aligns with handledness/deadness requirements).
doc_sections:
  matrix_acceptance: 2
doc_section_requires:
  matrix_acceptance:
    - POLICY_SEED.md#policy_seed
    - glossary.md#contract
    - glossary.md#exception_obligation
    - CONTRIBUTING.md#contributing_contract
    - README.md#repo_contract
    - in/in-24.md#in_in_24
    - in/in-25.md#in_in_25
    - in/in-26.md#in_in_26
    - in/in-27.md#in_in_27
doc_section_reviews:
  matrix_acceptance:
    POLICY_SEED.md#policy_seed:
      dep_version: 1
      self_version_at_review: 2
      outcome: no_change
      note: Reviewed POLICY_SEED.md rev1 (mechanized governance default; branch/tag CAS + check-before-use constraints); no conflicts with this document's scope.
    glossary.md#contract:
      dep_version: 1
      self_version_at_review: 1
      outcome: no_change
      note: Reviewed glossary.md#contract rev1 (glossary contract + semantic typing discipline).
    glossary.md#exception_obligation:
      dep_version: 1
      self_version_at_review: 1
      outcome: no_change
      note: Reviewed glossary.md#exception_obligation rev1 (exception obligation status + evidence linkage).
    CONTRIBUTING.md#contributing_contract:
      dep_version: 1
      self_version_at_review: 1
      outcome: no_change
      note: Reviewed CONTRIBUTING.md rev1 (docflow now fails on missing GH references for SPPF-relevant changes); no conflicts with this document's scope.
    README.md#repo_contract:
      dep_version: 1
      self_version_at_review: 1
      outcome: no_change
      note: Reviewed README.md rev1 (docflow audit now scans in/ by default); no conflicts with this document's scope.
    in/in-24.md#in_in_24:
      dep_version: 1
      self_version_at_review: 1
      outcome: no_change
      note: Reviewed in/in-24.md rev8 (deadness matrix acceptance checks align with artifact schema requirements).
    in/in-25.md#in_in_25:
      dep_version: 1
      self_version_at_review: 1
      outcome: no_change
      note: Reviewed in/in-25.md rev8 (coherence matrix acceptance mapping remains consistent with evidence artifacts).
    in/in-26.md#in_in_26:
      dep_version: 1
      self_version_at_review: 1
      outcome: no_change
      note: Reviewed in/in-26.md rev8 (rewrite-plan verification predicates match matrix obligations).
    in/in-27.md#in_in_27:
      dep_version: 1
      self_version_at_review: 1
      outcome: no_change
      note: Reviewed in/in-27.md rev6 (exception obligation mapping aligns with handledness/deadness requirements).
doc_change_protocol: POLICY_SEED.md#change_protocol
doc_erasure:
  - formatting
  - typos
doc_owner: maintainer
---

<a id="matrix_acceptance"></a>

# Matrix-to-Tests Mapping (Acceptance Reference)

## 0. Purpose

The 3×3×3 coherence matrices in `in-24`, `in-25`, and `in-26` are **normative
intent**. Their presence implies a **test and schema contract**. This file
reifies that contract into minimal acceptance checks.

If a matrix is present without these checks, treat the matrix as **aspirational**
and the related feature as **incomplete**.

Current implementation: `tests/test_matrix_acceptance_artifacts.py` enforces
artifact schema, determinism (byte-identical JSON across runs), evidence
linkage, and rewrite-plan verification predicate executability (must fail on a
known counterexample and pass on a synthetic resolved post-state). The remaining
checks in this document are tracked as follow-on work (see the linked SPPF/GH
issues).

## Normative pointers (explicit)

- `POLICY_SEED.md#policy_seed`
- `[glossary.md#contract](glossary.md#contract)`
- `[glossary.md#exception_obligation](glossary.md#exception_obligation)`
- `CONTRIBUTING.md#contributing_contract`
- `README.md#repo_contract`
- `in/in-24.md#in_in_24`
- `in/in-25.md#in_in_25`
- `in/in-26.md#in_in_26`
- `in/in-27.md#in_in_27`

---

## 1. What a 3×3×3 Matrix Entails (Normative)

Each matrix implies:

1) **Schema obligations** (required fields and stable ordering).
2) **Determinism obligations** (byte-stable artifacts).
3) **Interface parity** (CLI/LSP/report agree on the same entries).
4) **UNKNOWN handling** (never coerce UNKNOWN to success).
5) **Traceability** (evidence links are explicit and stable).

These must be mapped to tests or audit checks.

---

## 2. in-24 Deadness Evidence: Minimum Acceptance Suite

Artifacts:
- `fingerprint_deadness.json`
- Report section: "Deadness evidence"
- LSP/CLI field: `fingerprint_deadness`

Minimum checks:

1) **Schema presence**
   - Each entry includes: deadness_id, site, environment, predicate, core, result.
   - Ordering is stable by `(path, function, bundle, predicate_key)`.

2) **Determinism**
   - Two runs on the same input produce byte-identical JSON.

3) **UNKNOWN handling**
   - UNKNOWN is preserved and not coerced to UNREACHABLE.

4) **Interface parity**
   - CLI/LSP/report reference the same entry count and identifiers.

5) **Evidence sufficiency**
   - Core is non-empty and smaller than a full trace (size or count metric).

Suggested tests (names are indicative):
- `test_deadness_artifact_schema()`
- `test_deadness_artifact_determinism()`
- `test_deadness_unknown_is_preserved()`
- `test_deadness_interface_parity()`
- `test_deadness_core_reduced()`

---

## 3. in-25 Coherence Evidence: Minimum Acceptance Suite

Artifacts:
- `fingerprint_coherence.json`
- Report section: "Coherence evidence"
- LSP/CLI field: `fingerprint_coherence`

Minimum checks:

1) **Schema presence**
   - Each entry includes: site, boundary, alternatives, fork_signature,
     frack_path, result, remainder (if any).

2) **Determinism**
   - Two runs on the same input produce byte-identical JSON.

3) **UNKNOWN handling**
   - UNKNOWN is preserved and not coerced to COHERENT.

4) **Interface parity**
   - CLI/LSP/report reference the same entry count and identifiers.

5) **Fork localization**
   - `fork_signature` and `frack_path` are non-empty and stable.

Suggested tests:
- `test_coherence_artifact_schema()`
- `test_coherence_artifact_determinism()`
- `test_coherence_unknown_is_preserved()`
- `test_coherence_interface_parity()`
- `test_coherence_fork_localization()`

---

## 4. in-26 Rewrite Plans: Minimum Acceptance Suite

Artifacts:
- `fingerprint_rewrite_plans.json`
- Report section: "Rewrite plans"
- LSP/CLI field: `fingerprint_rewrite_plans`

Minimum checks:

1) **Schema presence**
   - Each plan includes: plan_id, site, pre, rewrite, evidence, post_expectation,
     verification.

2) **Determinism**
   - Two runs on the same input produce byte-identical JSON.

3) **Evidence linkage**
   - Each plan references provenance/coherence/deadness entries by stable IDs.

4) **Verification predicates**
   - Acceptance predicates are executable and fail on known counterexamples.

5) **Interface parity**
   - CLI/LSP/report reference the same plan count and identifiers.

Suggested tests:
- `test_rewrite_plan_schema()`
- `test_rewrite_plan_determinism()`
- `test_rewrite_plan_evidence_links()`
- `test_rewrite_plan_verification_predicates()`
- `test_rewrite_plan_interface_parity()`

---

## 5. in-27 Exception Obligations: Minimum Acceptance Suite

Artifacts:
- `fingerprint_exception_obligations.json`
- Report section: "Exception obligations"
- LSP/CLI field: `fingerprint_exception_obligations`

Minimum checks:

1) **Schema presence**
   - Each entry includes: exception_path_id, site, source_kind, status,
     witness_ref, remainder (if UNKNOWN), environment_ref.

2) **Determinism**
   - Two runs on the same input produce byte-identical JSON.

3) **UNKNOWN handling**
   - UNKNOWN is preserved and not coerced to DEAD/HANDLED.

4) **Evidence requirement**
   - DEAD/HANDLED entries reference a corresponding witness artifact
     (deadness_id or handledness_id, respectively).

5) **Interface parity**
   - CLI/LSP/report reference the same entry count and identifiers.

Suggested tests:
- `test_exception_obligation_schema()`
- `test_exception_obligation_determinism()`
- `test_exception_obligation_unknown_is_preserved()`
- `test_exception_obligation_witness_links()`
- `test_exception_obligation_interface_parity()`

---

## 6. Status Signaling

If any of the above checks are missing:

* mark the feature **partial** in `docs/sppf_checklist.md`, and
* ensure artifacts emit `UNKNOWN` rather than asserted success.

This preserves honesty under the governance contract.


---

## 7. Governance telemetry acceptance (convergence-aware)

Point-in-time checks are necessary but insufficient. Acceptance must include
telemetry trend checks emitted by `scripts/governance_telemetry_emit.py`.

Artifacts:
- `artifacts/out/governance_telemetry.json`
- `artifacts/audit_reports/governance_telemetry.md`

Minimum checks:

1) **Stable schema contract**
   - JSON payload includes `schema_version`, run identity, per-loop metrics,
     trend deltas, recurrence rates, false-positive overrides,
     and `convergence_slos`.

2) **Per-loop recurrence visibility**
   - Every governance loop reports:
     - `violation_count`
     - `trend_delta`
     - `recurrence_rate`
     - `false_positive_overrides`
     - `time_to_correction_runs` (nullable until correction)

3) **Recent-run trend deltas**
   - Markdown summary presents per-loop trend deltas against recent history,
     not only current counts.

4) **Convergence SLO evaluation per domain**
   - Security domain objective example:
     - *No repeated SEC-* violation for N runs*.
   - Governance/ratchet domains must also expose explicit objective, window,
     and pass/fail status.

5) **Acceptance gating discipline**
   - A domain with failing convergence SLO status is treated as
     non-converged and must be tracked as partial in checklist/status docs
     until recurrence is eliminated.

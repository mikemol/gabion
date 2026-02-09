---
doc_revision: 4
reader_reintern: "Reader-only: re-intern if doc_revision changed since you last read this doc."
doc_id: matrix_acceptance
doc_role: reference
doc_scope:
  - repo
  - testing
  - semantics
  - governance
doc_authority: informative
doc_requires:
  - POLICY_SEED.md
  - glossary.md
  - CONTRIBUTING.md
  - README.md
  - in/in-24.md
  - in/in-25.md
  - in/in-26.md
  - in/in-27.md
doc_reviewed_as_of:
  POLICY_SEED.md: 32
  glossary.md: 29
  CONTRIBUTING.md: 78
  README.md: 59
  in/in-24.md: 3
  in/in-25.md: 3
  in/in-26.md: 3
  in/in-27.md: 1
doc_review_notes:
  POLICY_SEED.md: "Reviewed POLICY_SEED.md rev32 (branch/tag CAS + check-before-use constraints); no conflicts with this document's scope."
  glossary.md: "Reviewed glossary rev29 (obsolescence projection path + self-review/mirror definitions); matrix acceptance framing unchanged."
  CONTRIBUTING.md: "Reviewed CONTRIBUTING.md rev77 (docflow now fails on missing GH references for SPPF-relevant changes); no conflicts with this document's scope."
  README.md: "Reviewed README.md rev59 (docflow audit now scans in/ by default); no conflicts with this document's scope."
  in/in-24.md: "Deadness matrix acceptance checks align with artifact schema requirements."
  in/in-25.md: "Coherence matrix acceptance mapping remains consistent with evidence artifacts."
  in/in-26.md: "Rewrite-plan verification predicates match matrix obligations."
  in/in-27.md: "Exception obligation mapping aligns with handledness/deadness requirements."
doc_change_protocol: "POLICY_SEED.md §6"
doc_erasure:
  - formatting
  - typos
doc_owner: maintainer
---

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

- `POLICY_SEED.md`
- `glossary.md`
- `CONTRIBUTING.md`
- `README.md`
- `in/in-24.md`
- `in/in-25.md`
- `in/in-26.md`
- `in/in-27.md`

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

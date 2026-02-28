---
doc_revision: 1
reader_reintern: "Reader-only: re-intern if doc_revision changed since you last read this doc."
doc_id: aspf_fingerprint_contract
doc_role: design
doc_scope:
  - analysis
  - fingerprints
doc_authority: informative
doc_requires:
  - POLICY_SEED.md#policy_seed
  - glossary.md#contract
  - docs/normative_clause_index.md#normative_clause_index
doc_reviewed_as_of:
  POLICY_SEED.md#policy_seed: 2
  glossary.md#contract: 1
  docs/normative_clause_index.md#normative_clause_index: 2
doc_change_protocol: POLICY_SEED.md#change_protocol
---

# ASPF fingerprint contract

This document defines the formalized fingerprint contract used by analysis modules.

## Fibration vs cofibration

- **Fibration layer**: refinement paths inside the ASPF fingerprint object space (0-cells and 1-cells).
- **Cofibration layer**: explicit witness maps from domain-prime basis elements to ASPF-prime basis elements; these maps must validate injective/faithful constraints.
- Higher-path (2-cell) witnesses encode equivalence between alternate 1-cell representatives.

## Canonical identity strata

1. **Canonical semantic identity**: structural ASPF path payload (source, target, representative, basis path).
2. **Derived scalar projection**: prime-product projection, explicitly marked `canonical: false`.
3. **Derived digest alias**: hash alias derived from canonical payload, explicitly marked `canonical: false`.

Downstream consumers must treat only the canonical ASPF path as semantic source-of-truth.

## Representative selection and drift rules

- Representative choice is a Decision Protocol with explicit modes.
- Contradictions are validated at ingress via one validator surface.
- Drift is classified by homotopy class change:
  - representative change + valid higher-path witness => non-drift;
  - missing/non-equivalent witness class => drift.

## Policy anchors

Implementation is aligned with:

- `POLICY_SEED.md#policy_seed` for execution/control constraints.
- `glossary.md#contract` for semantic contract and commutation terms.
- `docs/normative_clause_index.md#normative_clause_index` (notably `NCI-LSP-FIRST` and `NCI-SHIFT-AMBIGUITY-LEFT`).

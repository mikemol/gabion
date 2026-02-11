---
doc_revision: 1
reader_reintern: Reader-only: re-intern if doc_revision changed since you last read this doc.
doc_id: docflow_canonicality
doc_role: report
doc_scope:
  - repo
  - docflow
  - report
doc_authority: informative
doc_change_protocol: POLICY_SEED.md#change_protocol
doc_requires: []
doc_reviewed_as_of: {}
doc_review_notes: {}
doc_sections:
  docflow_canonicality: 1
doc_section_requires:
  docflow_canonicality: []
doc_section_reviews:
  docflow_canonicality: {}
---

<a id="docflow_canonicality"></a>

Docflow canonicality report
- total_terms: 35
- candidates: 25
- ambiguous: 10
- no_induced_meaning: 5
- projection_spec_id: {"domain":"docflow_canonicality","name":"docflow_canonicality_ambiguity","params":{},"pipeline":[{"op":"select","params":{"predicates":["canonicality_is_ambiguous"]}},{"op":"project","params":{"fields":["term","signal","doc"]}},{"op":"count_by","params":{"fields":["term"]}},{"op":"sort","params":{"by":[{"field":"term","order":"asc"}]}}],"spec_version":1}

Canonicality candidates:
- ambiguity_set (Ambiguity Set) requires=2
- annotation_drift (Annotation Drift) requires=1
- aspf (ASPF (Algebraic Structural Prime Fingerprint)) requires=5
- bundle (Bundle) requires=1
- coherence_witness (Coherence Witness) requires=3
- deadness_witness (Deadness Witness) requires=6
- decision_bundle (Decision Bundle (Control-Flow Structural)) requires=4
- decision_protocol (Decision Protocol (Control-Flow Explicit)) requires=4
- decision_surface (Decision Surface (Control-Flow Boundary)) requires=4
- decision_table (Decision Table (Control-Flow Documentation)) requires=4
- evidence_dominance (Evidence Dominance (Strict)) requires=1
- evidence_id (Evidence ID) requires=1
- exception_path (Exception Path) requires=1
- forest (Forest (Interned Carrier Graph)) requires=3
- handledness_witness (Handledness Witness) requires=2
- hash_consing (Hash‑Consing (Internment)) requires=4
- partition_witness (Partition Witness) requires=1
- rewrite_plan (Rewrite Plan (Proof‑Carrying Refactor)) requires=1
- rule_of_polysemy (Rule of Polysemy) requires=2
- suite_site (SuiteSite) requires=3
- test_evidence_suggestions_projection (Test Evidence Suggestions Projection) requires=1
- test_obsolescence_projection (Test Obsolescence Projection) requires=1
- tier (Tier) requires=3
- value_encoded_decision (Value-Encoded Decision (Branchless Control)) requires=2
- witness (Witness) requires=1

Ambiguity signals:
- attribute_carrier: implicit_without_requires
- attribute_transport: implicit_without_requires
- contract: explicit_without_requires
- equivalent_witness: implicit_without_requires
- evidence_key: implicit_without_requires
- evidence_surface: implicit_without_requires
- exception_obligation: implicit_without_requires
- grothendieck_analysis: implicit_without_requires
- never_throw_exception_protocol: implicit_without_requires
- self_review: implicit_without_requires

No induced meaning (no doc_requires references):
- attribute_carrier (Attribute Carrier)
- attribute_transport (Attribute Transport (ContextVar))
- grothendieck_analysis (Grothendieck Analysis (Doc Review Cofibration))
- never_throw_exception_protocol (Never‑Throw Exception Protocol)
- self_review (Self‑Review (Docflow Exception))

Convergence (docflow vs projection spec):
- matched: True
- docflow_ambiguous_terms: 10
- projection_ambiguous_terms: 10

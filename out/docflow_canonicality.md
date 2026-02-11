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
- candidates: 5
- ambiguous: 30
- no_induced_meaning: 5
- projection_spec_id: {"domain":"docflow_canonicality","name":"docflow_canonicality_ambiguity","params":{},"pipeline":[{"op":"select","params":{"predicates":["canonicality_is_ambiguous"]}},{"op":"project","params":{"fields":["term","signal","doc"]}},{"op":"count_by","params":{"fields":["term"]}},{"op":"sort","params":{"by":[{"field":"term","order":"asc"}]}}],"spec_version":1}

Canonicality candidates:
- bundle (Bundle) requires=1
- exception_obligation (Exception Obligation) requires=4
- suite_site (SuiteSite) requires=3
- tier (Tier) requires=3
- witness (Witness) requires=1

Ambiguity signals:
- ambiguity_set: implicit_without_requires
- annotation_drift: implicit_without_requires
- aspf: implicit_without_requires
- attribute_carrier: implicit_without_requires
- attribute_transport: implicit_without_requires
- coherence_witness: implicit_without_requires
- contract: explicit_without_requires
- deadness_witness: implicit_without_requires
- decision_bundle: implicit_without_requires
- decision_protocol: implicit_without_requires
- decision_surface: implicit_without_requires
- decision_table: implicit_without_requires
- equivalent_witness: implicit_without_requires
- evidence_dominance: implicit_without_requires
- evidence_id: implicit_without_requires
- evidence_key: implicit_without_requires
- evidence_surface: implicit_without_requires
- exception_path: implicit_without_requires
- forest: implicit_without_requires
- grothendieck_analysis: implicit_without_requires
- handledness_witness: implicit_without_requires
- hash_consing: implicit_without_requires
- never_throw_exception_protocol: implicit_without_requires
- partition_witness: implicit_without_requires
- rewrite_plan: implicit_without_requires
- rule_of_polysemy: implicit_without_requires
- self_review: implicit_without_requires
- test_evidence_suggestions_projection: implicit_without_requires
- test_obsolescence_projection: implicit_without_requires
- value_encoded_decision: implicit_without_requires

No induced meaning (no doc_requires references):
- attribute_carrier (Attribute Carrier)
- attribute_transport (Attribute Transport (ContextVar))
- grothendieck_analysis (Grothendieck Analysis (Doc Review Cofibration))
- never_throw_exception_protocol (Never‑Throw Exception Protocol)
- self_review (Self‑Review (Docflow Exception))

Convergence (docflow vs projection spec):
- matched: True
- docflow_ambiguous_terms: 30
- projection_ambiguous_terms: 30

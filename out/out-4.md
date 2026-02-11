---
doc_revision: 7
reader_reintern: Reader-only: re-intern if doc_revision changed since you last read this doc.
doc_id: out_4
doc_role: report
doc_scope:
  - repo
  - governance
  - documentation
  - analysis
doc_authority: informative
doc_requires:
  - POLICY_SEED.md#policy_seed
  - glossary.md#aspf
  - glossary.md#forest
  - glossary.md#suite_site
  - glossary.md#hash_consing
  - glossary.md#ambiguity_set
  - glossary.md#annotation_drift
  - CONTRIBUTING.md#contributing_contract
  - README.md#repo_contract
  - AGENTS.md#agent_obligations
  - docs/coverage_semantics.md#coverage_semantics
  - in/in-23.md#in_in_23
  - in/in-24.md#in_in_24
  - in/in-30.md#in_in_30
doc_reviewed_as_of:
  POLICY_SEED.md#policy_seed: 1
  glossary.md#aspf: 1
  glossary.md#forest: 1
  glossary.md#suite_site: 1
  glossary.md#hash_consing: 1
  glossary.md#ambiguity_set: 1
  glossary.md#annotation_drift: 1
  CONTRIBUTING.md#contributing_contract: 1
  README.md#repo_contract: 1
  AGENTS.md#agent_obligations: 1
  docs/coverage_semantics.md#coverage_semantics: 1
  in/in-23.md#in_in_23: 1
  in/in-24.md#in_in_24: 1
  in/in-30.md#in_in_30: 1
doc_review_notes:
  POLICY_SEED.md#policy_seed: Reviewed POLICY_SEED.md rev1 (mechanized governance default; branch/tag CAS + check-before-use constraints); inventory report aligns with docflow audit scope.
  glossary.md#aspf: Reviewed glossary.md#aspf rev1 (ASPF carrier semantics + packed-forest labels).
  glossary.md#forest: Reviewed glossary.md#forest rev1 (Forest materialized carrier; interned identity; suite-locality facets).
  glossary.md#suite_site: Reviewed glossary.md#suite_site rev1 (SuiteSite locality carrier + containment semantics).
  glossary.md#hash_consing: Reviewed glossary.md#hash_consing rev1 (hash-consing/internment: hash==normalize; β-reduction via normalization rules).
  glossary.md#ambiguity_set: Reviewed glossary.md#ambiguity_set rev1 (ambiguity sets are canonical candidate carriers; ordering erased).
  glossary.md#annotation_drift: Reviewed glossary.md#annotation_drift rev1 (drift defined by evidence-key identity; advisory before ratchet).
  CONTRIBUTING.md#contributing_contract: Reviewed CONTRIBUTING.md rev1 (docflow now fails on missing GH references for SPPF-relevant changes); inventory report is advisory only.
  README.md#repo_contract: Reviewed README.md rev1 (docflow audit now scans in/ by default); scope aligned.
  AGENTS.md#agent_obligations: Reviewed AGENTS.md rev1 (agent obligations + evidence-first posture); inventory is non-normative.
  docs/coverage_semantics.md#coverage_semantics: Reviewed docs/coverage_semantics.md#coverage_semantics v1 (glossary-lifted projection + explicit core anchors); report treats excess as unbound evidence, not violations.
  in/in-23.md#in_in_23: Reviewed in/in-23.md rev7 (ASPF carrier formalization); inventory maps evidence kinds to carrier semantics.
  in/in-24.md#in_in_24: Reviewed in/in-24.md rev8 (deadness witness pinning); inventory uses those semantics for evidence interpretation.
  in/in-30.md#in_in_30: Reviewed in/in-30.md rev20 (SuiteSite locality carrier and loop-scoped obligations; ambiguity/partition witness alignment); inventory aligns with suite-level evidence expectations.
doc_change_protocol: POLICY_SEED.md#change_protocol
doc_erasure:
  - formatting
  - typos
doc_owner: maintainer
doc_sections:
  out_out_4: 1
doc_section_requires:
  out_out_4:
    - POLICY_SEED.md#policy_seed
    - glossary.md#aspf
    - glossary.md#forest
    - glossary.md#suite_site
    - glossary.md#hash_consing
    - glossary.md#ambiguity_set
    - glossary.md#annotation_drift
    - CONTRIBUTING.md#contributing_contract
    - README.md#repo_contract
    - AGENTS.md#agent_obligations
    - docs/coverage_semantics.md#coverage_semantics
    - in/in-23.md#in_in_23
    - in/in-24.md#in_in_24
    - in/in-30.md#in_in_30
doc_section_reviews:
  out_out_4:
    POLICY_SEED.md#policy_seed:
      dep_version: 1
      self_version_at_review: 1
      outcome: no_change
      note: Reviewed POLICY_SEED.md rev1 (mechanized governance default; branch/tag CAS + check-before-use constraints); inventory report aligns with docflow audit scope.
    glossary.md#aspf:
      dep_version: 1
      self_version_at_review: 1
      outcome: no_change
      note: Reviewed glossary.md#aspf rev1 (ASPF carrier semantics + packed-forest labels).
    glossary.md#forest:
      dep_version: 1
      self_version_at_review: 1
      outcome: no_change
      note: Reviewed glossary.md#forest rev1 (Forest materialized carrier; interned identity; suite-locality facets).
    glossary.md#suite_site:
      dep_version: 1
      self_version_at_review: 1
      outcome: no_change
      note: Reviewed glossary.md#suite_site rev1 (SuiteSite locality carrier + containment semantics).
    glossary.md#hash_consing:
      dep_version: 1
      self_version_at_review: 1
      outcome: no_change
      note: Reviewed glossary.md#hash_consing rev1 (hash-consing/internment: hash==normalize; β-reduction via normalization rules).
    glossary.md#ambiguity_set:
      dep_version: 1
      self_version_at_review: 1
      outcome: no_change
      note: Reviewed glossary.md#ambiguity_set rev1 (ambiguity sets are canonical candidate carriers; ordering erased).
    glossary.md#annotation_drift:
      dep_version: 1
      self_version_at_review: 1
      outcome: no_change
      note: Reviewed glossary.md#annotation_drift rev1 (drift defined by evidence-key identity; advisory before ratchet).
    CONTRIBUTING.md#contributing_contract:
      dep_version: 1
      self_version_at_review: 1
      outcome: no_change
      note: Reviewed CONTRIBUTING.md rev1 (docflow now fails on missing GH references for SPPF-relevant changes); inventory report is advisory only.
    README.md#repo_contract:
      dep_version: 1
      self_version_at_review: 1
      outcome: no_change
      note: Reviewed README.md rev1 (docflow audit now scans in/ by default); scope aligned.
    AGENTS.md#agent_obligations:
      dep_version: 1
      self_version_at_review: 1
      outcome: no_change
      note: Reviewed AGENTS.md rev1 (agent obligations + evidence-first posture); inventory is non-normative.
    docs/coverage_semantics.md#coverage_semantics:
      dep_version: 1
      self_version_at_review: 1
      outcome: no_change
      note: Reviewed docs/coverage_semantics.md#coverage_semantics v1 (glossary-lifted projection + explicit core anchors); report treats excess as unbound evidence, not violations.
    in/in-23.md#in_in_23:
      dep_version: 1
      self_version_at_review: 1
      outcome: no_change
      note: Reviewed in/in-23.md rev7 (ASPF carrier formalization); inventory maps evidence kinds to carrier semantics.
    in/in-24.md#in_in_24:
      dep_version: 1
      self_version_at_review: 1
      outcome: no_change
      note: Reviewed in/in-24.md rev8 (deadness witness pinning); inventory uses those semantics for evidence interpretation.
    in/in-30.md#in_in_30:
      dep_version: 1
      self_version_at_review: 1
      outcome: no_change
      note: Reviewed in/in-30.md rev20 (SuiteSite locality carrier and loop-scoped obligations; ambiguity/partition witness alignment); inventory aligns with suite-level evidence expectations.
---

<a id="out_out_4"></a>

# Outbox Inventory: Docflow Excess Evidence

This report inventories current **docflow excess evidence** (evidence that is not yet covered by active docflow cover invariants). It is intended to surface *documentable clusters* without creating a baseline that would mask technical meta-debt.

Snapshot source: `artifacts/out/docflow_compliance.json` (version 2).

Normative pointers (explicit): [POLICY_SEED.md#policy_seed](POLICY_SEED.md#policy_seed), [glossary.md#aspf](glossary.md#aspf), [glossary.md#forest](glossary.md#forest), [glossary.md#suite_site](glossary.md#suite_site), [glossary.md#hash_consing](glossary.md#hash_consing), [glossary.md#ambiguity_set](glossary.md#ambiguity_set), [glossary.md#annotation_drift](glossary.md#annotation_drift), [CONTRIBUTING.md#contributing_contract](CONTRIBUTING.md#contributing_contract), [README.md#repo_contract](README.md#repo_contract), [AGENTS.md#agent_obligations](AGENTS.md#agent_obligations), [docs/coverage_semantics.md#coverage_semantics](docs/coverage_semantics.md#coverage_semantics), [in/in-23.md#in_in_23](in/in-23.md#in_in_23), [in/in-24.md#in_in_24](in/in-24.md#in_in_24), [in/in-30.md#in_in_30](in/in-30.md#in_in_30).

**Summary**
Total excess evidence rows: 368.
Buckets: non-tests=295, tests=73, unknown=0.

**Counts By Evidence Kind**
| Evidence Kind | Count |
| --- | ---: |
| decision_surface | 192 |
| function_site | 150 |
| call_footprint | 22 |
| call_cluster | 4 |

**Top Sources (All Kinds)**
| Path | Count |
| --- | ---: |
| `dataflow_audit.py` | 143 |
| `type_fingerprints.py` | 22 |
| `cli.py` | 22 |
| `server.py` | 14 |
| `evidence_keys.py` | 10 |
| `test_evidence_suggestions.py` | 10 |
| `config.py` | 9 |
| `engine.py` | 8 |
| `timeout_context.py` | 7 |
| `tests/test_cli_commands.py` | 6 |
| `ambiguity_delta.py` | 6 |
| `forest_spec.py` | 6 |
| `test_obsolescence.py` | 6 |
| `tests/test_refactor_engine_more.py` | 5 |
| `schema_audit.py` | 5 |
| `lsp_client.py` | 5 |
| `projection_normalize.py` | 5 |
| `test_evidence.py` | 5 |
| `test_obsolescence_delta.py` | 5 |
| `tests/test_visitors_unit.py` | 4 |

**Top Sources By Kind**

Kind: `call_cluster`
| Path | Count |
| --- | ---: |
| `dataflow_audit.py` | 2 |
| `server.py` | 1 |
| `test_visitors_edges.py` | 1 |

Kind: `call_footprint`
| Path | Count |
| --- | ---: |
| `tests/test_cli_commands.py` | 6 |
| `tests/test_refactor_engine_more.py` | 5 |
| `tests/test_visitors_unit.py` | 4 |
| `tests/test_misc_coverage.py` | 3 |
| `tests/test_aspf.py` | 2 |
| `tests/test_dataflow_main.py` | 1 |
| `tests/test_test_evidence_suggestions_edges.py` | 1 |

Kind: `decision_surface`
| Path | Count |
| --- | ---: |
| `dataflow_audit.py` | 107 |
| `cli.py` | 16 |
| `type_fingerprints.py` | 14 |
| `server.py` | 9 |
| `engine.py` | 8 |
| `config.py` | 7 |
| `forest_spec.py` | 6 |
| `projection_normalize.py` | 5 |
| `evidence.py` | 4 |
| `schema_audit.py` | 4 |

Kind: `function_site`
| Path | Count |
| --- | ---: |
| `dataflow_audit.py` | 34 |
| `test_evidence_suggestions.py` | 10 |
| `evidence_keys.py` | 8 |
| `type_fingerprints.py` | 8 |
| `timeout_context.py` | 7 |
| `cli.py` | 6 |
| `test_obsolescence.py` | 6 |
| `test_evidence.py` | 5 |
| `test_obsolescence_delta.py` | 5 |
| `ambiguity_delta.py` | 4 |

**Documentable Buckets (Proposed Targets)**
These buckets group evidence into documentation targets that explain *why* the evidence exists, not just where it comes from.

- ASPF / Suite / Forest semantics: `in/in-23.md#in_in_23`, `in/in-24.md#in_in_24`, `in/in-30.md#in_in_30`, `out/out-3.md`.
- Docflow coverage semantics and invariants: `docs/coverage_semantics.md#coverage_semantics`, `[glossary.md#suite_site](glossary.md#suite_site)`, `[glossary.md#forest](glossary.md#forest)`.
- Governance pipeline (obsolescence/ambiguity/annotation drift): `docs/sppf_checklist.md`, `docs/publishing_practices.md#publishing_practices`.
- CLI + server integration surfaces: `README.md#repo_contract`, `CONTRIBUTING.md#contributing_contract`, `AGENTS.md#agent_obligations`.
- Test evidence surface (call footprints/clusters): either document explicitly or declare out-of-scope for docflow coverage in `docs/coverage_semantics.md#coverage_semantics`.

**Interpretation Notes**
Excess evidence currently reflects *unbound* evidence because cover invariants are `status: proposed`. This inventory should be used to decide which surfaces to document and which to explicitly exclude before any baseline is created.
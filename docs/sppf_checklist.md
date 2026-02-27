---
doc_revision: 160
reader_reintern: "Reader-only: re-intern if doc_revision changed since you last read this doc."
doc_id: sppf_checklist
doc_role: checklist
doc_scope:
  - repo
  - planning
  - tooling
doc_authority: informative
doc_requires:
  - README.md#repo_contract
  - CONTRIBUTING.md#contributing_contract
  - glossary.md#decision_table
  - glossary.md#decision_bundle
  - glossary.md#decision_protocol
  - glossary.md#decision_surface
  - glossary.md#value_encoded_decision
  - glossary.md#deadness_witness
  - glossary.md#exception_obligation
doc_reviewed_as_of:
  README.md#repo_contract: 2
  CONTRIBUTING.md#contributing_contract: 2
  glossary.md#decision_table: 1
  glossary.md#decision_bundle: 1
  glossary.md#decision_protocol: 1
  glossary.md#decision_surface: 1
  glossary.md#value_encoded_decision: 1
  glossary.md#deadness_witness: 1
  glossary.md#exception_obligation: 1
doc_review_notes:
  README.md#repo_contract: "Reviewed README.md rev2 (removed stale ASPF action-plan CLI/examples; continuation docs now state/delta only)."
  CONTRIBUTING.md#contributing_contract: "Reviewed CONTRIBUTING.md rev2 (two-stage dual-sensor cadence, correction-unit validation stack, and strict-coverage trigger guidance)."
  glossary.md#decision_table: "Reviewed glossary.md#decision_table rev1 (decision table tier definition)."
  glossary.md#decision_bundle: "Reviewed glossary.md#decision_bundle rev1 (decision bundle tier definition)."
  glossary.md#decision_protocol: "Reviewed glossary.md#decision_protocol rev1 (decision protocol tier definition)."
  glossary.md#decision_surface: "Reviewed glossary.md#decision_surface rev1 (decision surface tier boundary semantics)."
  glossary.md#value_encoded_decision: "Reviewed glossary.md#value_encoded_decision rev1 (value-encoded decision surface semantics)."
  glossary.md#deadness_witness: "Reviewed glossary.md#deadness_witness rev1 (deadness witness obligations for negative evidence)."
  glossary.md#exception_obligation: "Reviewed glossary.md#exception_obligation rev1 (exception obligation status + evidence linkage)."
doc_sections:
  sppf_checklist: 8
doc_section_requires:
  sppf_checklist:
    - README.md#repo_contract
    - CONTRIBUTING.md#contributing_contract
    - glossary.md#decision_table
    - glossary.md#decision_bundle
    - glossary.md#decision_protocol
    - glossary.md#decision_surface
    - glossary.md#value_encoded_decision
    - glossary.md#deadness_witness
    - glossary.md#exception_obligation
doc_section_reviews:
  sppf_checklist:
    README.md#repo_contract:
      dep_version: 2
      self_version_at_review: 8
      outcome: no_change
      note: "Repo contract rev2 reviewed; command and artifact guidance remains aligned."
    CONTRIBUTING.md#contributing_contract:
      dep_version: 2
      self_version_at_review: 8
      outcome: no_change
      note: "Contributor contract rev2 reviewed; dual-sensor cadence and correction gates remain aligned."
    glossary.md#decision_table:
      dep_version: 1
      self_version_at_review: 8
      outcome: no_change
      note: "Reviewed glossary.md#decision_table rev1 (decision table tier definition)."
    glossary.md#decision_bundle:
      dep_version: 1
      self_version_at_review: 8
      outcome: no_change
      note: "Reviewed glossary.md#decision_bundle rev1 (decision bundle tier definition)."
    glossary.md#decision_protocol:
      dep_version: 1
      self_version_at_review: 8
      outcome: no_change
      note: "Reviewed glossary.md#decision_protocol rev1 (decision protocol tier definition)."
    glossary.md#decision_surface:
      dep_version: 1
      self_version_at_review: 8
      outcome: no_change
      note: "Reviewed glossary.md#decision_surface rev1 (decision surface tier boundary semantics)."
    glossary.md#value_encoded_decision:
      dep_version: 1
      self_version_at_review: 8
      outcome: no_change
      note: "Reviewed glossary.md#value_encoded_decision rev1 (value-encoded decision surface semantics)."
    glossary.md#deadness_witness:
      dep_version: 1
      self_version_at_review: 8
      outcome: no_change
      note: "Reviewed glossary.md#deadness_witness rev1 (deadness witness obligations for negative evidence)."
    glossary.md#exception_obligation:
      dep_version: 1
      self_version_at_review: 8
      outcome: no_change
      note: "Reviewed glossary.md#exception_obligation rev1 (exception obligation status + evidence linkage)."
doc_change_protocol: "POLICY_SEED.md#change_protocol"
doc_erasure:
  - formatting
  - typos
doc_owner: maintainer
sppf_schema: v2
sppf_dimensions:
  - doc
  - impl
  - doc_ref
sppf_status_rule: "Checklist items are marked done only when doc=done and impl=done and doc_ref matches the in/ influence index."
---

<a id="sppf_checklist"></a>

# SPPF Convergence Checklist (Bottom-Up)

This checklist is the bottom-up complement to the governance layer. It captures
concept nodes derived from `in/` and tracks whether they are adopted, planned,
or explicitly out of scope. It is advisory only.

Tooling axis note: entries that touch scripts, Make targets, or execution
wrappers live on the `tooling` axis and should stay consistent with
`CONTRIBUTING.md#contributing_contract` and `README.md#repo_contract`.

Legend: [x] done · [ ] planned · [~] partial/heuristic

SPPF axis tags (doc-ref only for now): append `sppf{doc=...; impl=...; doc_ref=...}` to
lines that cite `in/in-XX.md` or other doc references. `[x]` is allowed only
when `doc=done` and `impl=done` and `doc_ref` matches the referenced doc
revision (and, for `in-XX`, the status in `docs/influence_index.md`).

Normative pointers (explicit): [README.md#repo_contract](README.md#repo_contract), [CONTRIBUTING.md#contributing_contract](CONTRIBUTING.md#contributing_contract),
[glossary.md#decision_table](glossary.md#decision_table), [glossary.md#decision_bundle](glossary.md#decision_bundle),
[glossary.md#decision_protocol](glossary.md#decision_protocol), [glossary.md#decision_surface](glossary.md#decision_surface),
[glossary.md#value_encoded_decision](glossary.md#value_encoded_decision), [glossary.md#deadness_witness](glossary.md#deadness_witness),
[glossary.md#exception_obligation](glossary.md#exception_obligation).

## GitHub tracking linkage
- Each `[ ]` or `[~]` node should have a corresponding GitHub issue created from the
  `SPPF node` issue form.
- Once the issue exists, append `(GH-####)` to the checklist line so planning and
  status remain bidirectionally linked.
- Use `scripts/sppf_sync.py` locally to sync commit trailers (e.g. `SPPF: GH-17`)
  with GitHub issue comments/labels without CI write permissions.
- Do not close issues until a release containing the fix ships; use the
  `status/pending-release` label once work lands on `stage`.
- Optional: enable `GABION_SPPF_SYNC=1` and re-run `scripts/install_hooks.sh` to
  auto-sync on `stage` pushes via the pre-push hook.

Docflow audit emits a violation when commits touching SPPF-relevant paths (`src/`,
`in/`, or this checklist) lack GH references in commit messages. Use `GH-####`
trailers or run `scripts/sppf_sync.py --comment` after adding references.

## Governance tooling nodes
- [~] Docflow audit outputs + frontmatter/anchorized report artifacts in `out/`. (in-28, GH-86) sppf{doc=done; impl=partial; doc_ref=in-28@8}

## Functional-core roadmap lane
- Lane audit tracker: `docs/audits/functional_core_audit.md#functional_core_audit`.
- Active slice state artifact: `artifacts/audit_reports/refactor_slice_state.json`.
- Step-level CI KPI artifact: `artifacts/audit_reports/ci_step_timings.json`.
- [~] Functional-core hard-cut lane instrumentation (per-slice brief/state capture + local CI step timing capture). (GH-63)
- [~] Dataflow-first hard-cut slice: dataclass call bundle iteration moved to operation-plan/effect-return helper module (`src/gabion/analysis/dataflow_bundle_iteration.py`), with side-effect dispatch retained only at boundary adapter (`_iter_dataclass_call_bundles`). (GH-63)
- [x] Dataflow-first hard-cut slice: `_resolve_callee` moved to pure operation/effect pipeline module (`src/gabion/analysis/dataflow_callee_resolution.py`), with ambiguity sink dispatch kept only at boundary adapters in `dataflow_audit`. (GH-63)
- [~] Dataflow-first hard-cut slice: run-output finalization extracted to operation-sequence module (`src/gabion/analysis/dataflow_run_outputs.py`) with typed context/outcome carriers; next hotspot remains `_run_impl` + `_analyze_file_internal` phase reduction. (GH-63)

## Analysis pipeline nodes
- [x] Import resolution / symbol table (deterministic callee resolution). (GH-6)
- [x] Import resolution: explicit imports + relative import handling.
- [x] Import resolution: star-import expansion.
- [x] Import resolution: re-exports / `__all__` handling.
- [x] Import resolution: class hierarchy dispatch resolution.
- [x] Alias-aware identity tracking (rename morphisms preserved). (GH-7)
- [x] Alias tracking: direct Name-to-Name assignments.
- [x] Alias tracking: tuple/sequence unpacking.
- [x] Alias tracking: attribute/collection aliasing (obj.attr, dict["key"]).
- [x] Alias tracking: alias propagation via call returns. (GH-8)
- [x] Noise controls: project root anchoring + exclude dirs + ignore params.
- [x] External-lib filter (ignore non-project imports by default).
- [x] Wildcard forwarding strictness (`*args/**kwargs`, high/low modes).
- [x] Wildcard forwarding: detect direct starred name in call site.
- [x] Wildcard forwarding: signature-aware mapping for `*args/**kwargs`.
- [x] Wildcard forwarding: pass-through via `*args/**kwargs` variables.
- [x] Type-flow tightening audit (downstream annotations).
- [~] Internal broad-type lint (Any/object + scalar types like str/int/float/bool/bytes, except NodeId) on non-boundary surfaces (heuristic allow/deny set remains intentionally narrow). Evidence anchors: `src/gabion/analysis/dataflow_audit.py::_internal_broad_type_lint_lines_indexed`, `tests/test_dataflow_audit_coverage_gaps.py::test_internal_broad_type_lint_lines_indexed_appends_multiple`. (GH-89) sppf{doc=partial; impl=partial; doc_ref=in-31@4}
- [x] Type-flow ambiguities fail `gabion check` in repo defaults.
- [x] Anonymous schema surface detection (dict[str, object]/Any payload annotations).
- [x] Constant-flow audit (dead knobs / always-constant params).
- [x] Constant-flow helper specificity (avoid false positives in internal helpers).
- [x] Unused-argument pass detection (non-test call sites).
- [x] Analysis: Decorator transparency/unwrapping. (GH-9)
- [x] Verification: Idempotency test (ensure Analysis(Refactor(Code)) == Stable). (GH-22)
- [~] Decision surface detection + boundary elevation (tier enforcement). (in-15, GH-60) sppf{doc=partial; impl=partial; doc_ref=in-15@2} — impl now emits explicit classification reasons + tier-pathway evidence; glossary-tier artifacts (Decision Table/Bundle docs) remain partial.
- [~] Decision surface hooks in grammar (`is_decision_surface`). (GH-60) — branch/guard coverage now includes `if`/`while`/`assert`/`ifexp`/`match`/comprehension guards; wider grammar harmonization still partial.
- [x] Decision surface boundary diagnostics (API surface vs internal depth). (GH-60)
- [x] Decision surface tier enforcement via glossary metadata. (GH-60)
- [~] Value-encoded decision surface detection (branchless / algebraic control). (in-18, GH-66) sppf{doc=partial; impl=partial; doc_ref=in-18@2}
- [~] Value-encoded decision heuristics (min/max, bitmask, boolean arithmetic). (GH-66)
- [~] Value-encoded decision surface reports in audit output. (GH-66)
- [x] Value-encoded decision glossary warnings (nonlinear contexts). (GH-66)
- [x] Value-encoded decision rewrite suggestions (rebranch). (GH-66)
- [x] Value-encoded decision diff tracking in audit snapshots. (GH-66)
- [~] Prime-labeled type fingerprints (algebraic bundle matching; rewrite synthesis closure still pending). Evidence anchors: `src/gabion/analysis/type_fingerprints.py::bundle_fingerprint`, `tests/test_type_fingerprints.py::test_bundle_fingerprint_multiplies_primes`. (in-20/in-21, GH-68) sppf{doc=partial; impl=partial; doc_ref=in-20@2,in-21@1}
- [x] Prime registry + canonical type key mapping. (GH-68)
- [x] Fingerprint arithmetic ops (gcd/lcm/subtyping checks). (GH-68)
- [x] Glossary fingerprint matching + CI warnings. (GH-68)
- [x] Hybrid fingerprint representation (prime products + bitmask existence checks). (GH-68)
- [~] Deterministic fingerprint registry policy (seeded-vs-learned assignment policy is serialized in the seed payload, deterministic rehydrate path accepts legacy cache identities, and registry/key stability is covered by suite-order perturbation regressions). Evidence anchors: `src/gabion/analysis/type_fingerprints.py::PrimeRegistry.seed_payload`, `src/gabion/analysis/type_fingerprints.py::build_synth_registry_from_payload`, `src/gabion/analysis/dataflow_audit.py::_canonical_cache_identity`, `tests/test_type_fingerprints.py::test_registry_assignment_policy_roundtrips_and_stays_deterministic`, `tests/test_dataflow_audit_helpers.py::test_cache_identity_supports_legacy_alias_matching`, `tests/test_aspf.py::test_suite_site_signature_stable_under_suite_order_perturbation`. (in-22, GH-68) sppf{doc=partial; impl=done; doc_ref=in-22@2}
- [x] Nested type constructor registry (dimensional prime mapping). (GH-68)
- [x] Fingerprint reverse mapping for synthesis (factorization → type keys). (GH-68)
- [~] ASPF dimensional fingerprints (base/ctor carriers + soundness invariants; entropy-controlled synthesis obligations still open). Evidence anchors: `src/gabion/analysis/type_fingerprints.py::bundle_fingerprint_dimensional`, `tests/test_type_fingerprints.py::test_dimensional_fingerprint_includes_constructors`. (in-22, GH-70) sppf{doc=partial; impl=partial; doc_ref=in-22@2}
- [~] ASPF provenance mapping to SPPF (packed-forest derivation reporting + invariants; base/ctor keys + JSON artifact + report summary with remaining matrix expansion work). Evidence anchors: `src/gabion/analysis/dataflow_audit.py::_compute_fingerprint_provenance`, `tests/test_fingerprint_warnings.py::test_fingerprint_provenance_emits_entries`. (in-22, GH-71) sppf{doc=partial; impl=partial; doc_ref=in-22@2}
- <a id="in-23-aspf-carrier-formalization"></a>[x] ASPF carrier obligations formalized (determinism, base conservation, ctor coherence, synth tail reversibility, provenance completeness, snapshot reproducibility). (in-23, GH-73; anchors: `src/gabion/analysis/dataflow_audit.py::_compute_fingerprint_provenance`, `src/gabion/analysis/dataflow_audit.py::_compute_fingerprint_synth`, `src/gabion/analysis/type_fingerprints.py::build_synth_registry_from_payload`, `tests/test_type_fingerprints.py::test_build_fingerprint_registry_deterministic_assignment`, `tests/test_type_fingerprints.py::test_synth_registry_payload_roundtrip`, `tests/test_fingerprint_warnings.py::test_fingerprint_provenance_emits_entries`, `scripts/audit_snapshot.sh`, `scripts/latest_snapshot.sh`). sppf{doc=done; impl=done; doc_ref=in-23@10}
- [~] SuiteSite carriers + loop-scoped deadline obligations (recursive loop attribution now outer-vs-inner precise; SuiteSite-native enforcement still incomplete). Evidence anchors: `src/gabion/analysis/dataflow_obligations.py::collect_deadline_obligations`, `tests/test_deadline_coverage.py::test_deadline_loop_unchecked_status_is_root_gated`. (in-30, GH-85) sppf{doc=partial; impl=partial; doc_ref=in-30@27}
- [~] Deadline propagation as gas (ticks-based carriers across LSP/CLI/server; carrier budget semantics are stable but broader lane acceptance remains partial). Evidence anchors: `scripts/deadline_runtime.py::deadline_scope_from_lsp_env`, `tests/test_deadline_runtime.py::test_deadline_scope_from_lsp_env_uses_default_and_explicit_gas_limit`. (in-30, GH-87) sppf{doc=partial; impl=partial; doc_ref=in-30@27}
- [~] Structural ambiguity as CallCandidate alts (SuiteSite) with virtual AmbiguitySet (materialization landed; phase-3/4 decision-surface migration deferred). Evidence anchors: `src/gabion/analysis/dataflow_audit.py::_materialize_call_candidates`, `tests/test_deadline_coverage.py::test_materialized_call_candidates_target_function_suites`. (in-30, GH-88) sppf{doc=partial; impl=partial; doc_ref=in-30@27}
- <a id="in-33-pattern-schema-unification"></a>[~] PatternSchema unification for dataflow bundles + execution patterns (shared cross-axis `schema:*` IDs, contract-versioned residue payloads, deterministic artifact ordering, Tier-2 residue ratchet/metafactory gate landed; execution rules still narrow). (in-33) sppf{doc=partial; impl=partial; doc_ref=in-33@3}
- <a id="in-34-lambda-callable-sites"></a>[~] Lambda/closure callable indexing as first-class function sites (stable synthetic identities + direct/bound/closure lambda resolution; conservative dynamic fallback retained for unresolved alias/dynamic paths). Evidence anchors: `src/gabion/analysis/dataflow_audit.py::_resolve_callee_outcome`, `tests/test_dataflow_resolve_callee.py::test_resolve_callee_bound_lambda_call`, `tests/test_dataflow_resolve_callee.py::test_resolve_callee_outcome_keeps_dynamic_fallback_for_attribute_calls`. (in-34) sppf{doc=partial; impl=partial; doc_ref=in-34@2}
- <a id="in-35-dict-key-carrier-tracking"></a>[~] Dict carrier tracking beyond literal subscript aliases (name-bound constant keys + unknown-key carrier evidence; key grammar remains conservative by design). Evidence anchors: `src/gabion/analysis/visitors.py::_normalize_key`, `tests/test_visitors_unit.py::test_subscript_forwarding_normalizes_const_keys`, `tests/test_visitors_unit.py::test_subscript_dynamic_key_marks_uncertainty`. (in-35) sppf{doc=partial; impl=partial; doc_ref=in-35@1}
- <a id="in-36-starred-dataclass-call-bundles"></a>[x] Conservative starred dataclass constructor argument handling (`*` list/tuple/set, `**` dict literal) with unresolved-starred witnesses for dynamic payloads. (in-36) sppf{doc=done; impl=done; doc_ref=in-36@1}
- <a id="in-37-dynamic-dispatch-uncertainty"></a>[x] Dynamic-dispatch uncertainty classification in call resolution (`unresolved_dynamic`) plus dedicated call-resolution obligation kind. (in-37) sppf{doc=done; impl=done; doc_ref=in-37@1}
- <a id="in-38-aspf-log-structured-archive-projection"></a>[ ] ASPF mutation log-structured archive projection (protobuf payload records + protobuf-defined filesystem envelope projection + tar-packaged transport container + snapshot/tail replay). (in-38, GH-196) sppf{doc=done; impl=planned; doc_ref=in-38@1}

## Reporting & visualization nodes
- [x] Component isolation (connected components in bundle graph).
- [x] Mermaid component diagrams embedded in Markdown report.
- [x] DOT/Graphviz output for bundle graphs.
- [x] Tiered bundle classification (declared vs. observed) + violation listing.
- [x] Anonymous schema surfaces section in Markdown report.
- [x] Bundle declaration sources (Config dataclasses, `dataflow-bundle` markers, dataclass calls). (GH-10)
- [x] Bundle declarations: `dataflow-bundle` markers.
- [x] Bundle declarations: local dataclass constructor calls (Name-only args).
- [x] Bundle declarations: general dataclass fields beyond `_fn` convention.
- [x] Bundle declarations: non-Name args/kwargs in dataclass calls.
- [x] Bundle declarations: external dataclass modules (cross-file).
- [x] FactorizationTree snapshot emission (canonical JSON). (in-16, GH-62) sppf{doc=done; impl=done; doc_ref=in-16@2}
- [x] Structural diff command + baseline comparison. (in-16, GH-63) sppf{doc=done; impl=done; doc_ref=in-16@2}
- [x] Structural metrics export (bundle/tier/violation stats). (in-16, GH-64) sppf{doc=done; impl=done; doc_ref=in-16@2}
- [x] Deadness evidence artifacts (constant-flow deadness witnesses + JSON/report/LSP + snapshot selectors + determinism/schema tests; see `docs/matrix_acceptance.md`). (in-24, GH-74) sppf{doc=done; impl=done; doc_ref=in-24@11}
- [x] Coherence evidence artifacts (glossary-ambiguity witnesses + JSON/report/LSP + snapshot selectors + determinism/schema tests; see `docs/matrix_acceptance.md`). (in-25, GH-75) sppf{doc=done; impl=done; doc_ref=in-25@9}
- [~] Exception obligation artifacts (E0 enumeration + JSON/report/LSP + snapshot selectors; handledness still broad-try/except biased, so reported as partial; see `docs/matrix_acceptance.md`). Evidence anchors: `src/gabion/analysis/dataflow_audit.py::_collect_exception_obligations`, `tests/test_exception_deadness_helpers.py::test_exception_obligation_deadness_parsing_skips_invalid_entries`. (in-27, GH-77) sppf{doc=partial; impl=partial; doc_ref=in-27@8}
- [~] Exception obligations: handledness refinement (typed except matching, explicit broad catch reasoning, conservative UNKNOWN for unresolved dynamic paths; refinement coverage incomplete). Evidence anchors: `src/gabion/analysis/dataflow_audit.py::_collect_handledness_witnesses`, `tests/test_evidence.py::test_exception_obligation_summary_for_site_skips_non_matching_and_normalizes_status`. (in-27, GH-80) sppf{doc=partial; impl=partial; doc_ref=in-27@8}

## Synthesis + refactoring nodes
- [x] Protocol/dataclass synthesis (tier thresholds, field typing) (prototype). (GH-11)
- [x] Synthesis output: dataclass stubs with field typing.
- [x] Synthesis output: typing.Protocol interface stubs.
- [x] Synthesis typing: resolve conflicts into `Union`/`Optional`.
- [x] Naming heuristics (frequency-based) (prototype).
- [x] Merge overlap threshold is configurable (no hardcoded axiom).
- [x] Topological refactoring schedule (callee-first order) (prototype). (GH-12)
- [x] Refactor schedule: basic topological order.
- [x] Refactor schedule: SCC-based cycle detection (explicit knots).
- [x] Partial-application merge heuristics (bundle fragmentation control) (prototype). (GH-13)
- [x] Bundle merge heuristic: Jaccard overlap merge function.
- [x] Bundle merge heuristic: integrated into synthesis/refactor pipeline.
- [x] LLM-ready naming stubs (TODO_Name_Me + context docstrings).
- [x] Type aggregation for synthesis (from type-audit + call-site evidence). (GH-14)
- [x] Type aggregation: single-type consensus assignment.
- [x] Type aggregation: conflict resolution into `Union`/`Any` guidance.
- [x] Refactor payload: Type hint preservation (pass FieldSpec from Analysis to Engine). (GH-15)
- [x] Const/default-aware partial-application detection (subset merge by knobs). (GH-16)
- [~] Contextvar/ambient context rewrite suggestions. (in-15, GH-61) sppf{doc=partial; impl=partial; doc_ref=in-15@2}
- [x] Contextvar suggestion heuristics (internal decision surfaces). (GH-61)
- [~] Contextvar rewrite: synthesis emits ContextVar definitions + accessors (conservative ambient scaffold + compatibility shim). (GH-61; tests: `tests/test_refactor_engine.py::test_refactor_engine_ambient_rewrite_threaded_parameter`)
- [~] Contextvar rewrite: callsite replacement for ambient access (safe direct-threading replacement; unsafe sites skipped with explicit reasons). (GH-61; tests: `tests/test_refactor_engine.py::test_refactor_engine_ambient_rewrite_partial_skip_unsafe`, `tests/test_refactor_engine.py::test_refactor_engine_ambient_rewrite_noop_when_no_pattern`)
- [~] Subtree reuse detection + lemma synthesis hooks. (in-17, GH-65) sppf{doc=partial; impl=partial; doc_ref=in-17@2}
- [x] Subtree hashing/fingerprinting for FactorizationTree reuse. (GH-65)
- [~] Lemma suggestion output + stable naming map. (GH-65)
- [x] Lemma suggestion CLI flag + output map (declare + replace). (GH-65)
- [~] Lemma emission target selection (inline vs stub module). (GH-65)
- [x] Glossary-backed lemma naming + missing-entry warnings. (GH-65)
- [~] Invariant extraction + dependent-type synthesis (Agda). (in-19, GH-67) sppf{doc=partial; impl=partial; doc_ref=in-19@2}
- [x] Proposition model + assert-based invariant extraction. (GH-67)
- [x] Invariant emitter hooks (pluggable callbacks). (GH-67)
- [ ] Dependent-type / Agda synthesis output from invariants. (GH-67)
- [~] SIGv2 bridge typing in the Agda research lane (trace/payload/command surface contracts tracked against the TC design bridge; runtime remains advisory-only at this stage). (research lane) sppf{doc=partial; impl=partial; doc_ref=in/universal-curve-lab-bundle/docs/tc-design-bridge.md@1}
- [~] CONSTRv2 constructor-shaping research lane (constructor bundle normalization and assembly contracts tracked in TC bridge mappings; production command handlers still own enforced payload shaping). (research lane) sppf{doc=partial; impl=partial; doc_ref=in/universal-curve-lab-bundle/docs/tc-design-bridge.md@1}
- [x] GLUEv2 bridge-plan traceability lane tracked as completed research scaffolding (explicit cross-surface mapping bundle + non-production-enforced marker documented in the TC bridge, with v2 extension to be referenced when published). (research lane) sppf{doc=done; impl=done; doc_ref=in/universal-curve-lab-bundle/docs/tc-design-bridge.md@1}
- [~] Admitted-image/path-NF gate tracked in research lane (normal-form gate semantics under evaluation via TC bridge notes and planned v2 extension; promotion blocked pending policy/contract approval). (research lane) sppf{doc=partial; impl=partial; doc_ref=in/universal-curve-lab-bundle/docs/tc-design-bridge.md@1}
- [x] Runtime-enforcement boundary note: dependent-type research artifacts remain non-authoritative for runtime behavior; enforcement continues in Python command handlers until a separately approved policy/contract promotion is ratified. (research lane) sppf{doc=done; impl=done; doc_ref=in/universal-curve-lab-bundle/docs/tc-design-bridge.md@1}
- [x] Invariant-enriched JSON output for bundles/trees. (GH-67)
- [x] Property-based test hook manifest generation from invariants (deterministic hook IDs, confidence gating, callable mapping, traceability keys; optional Hypothesis template snippets). (GH-67)
- [~] ASPF entropy-controlled synthesis (synth@k primes + tail mapping + versioned registry; report + JSON registry output + snapshots + loadable registry). (in-22, GH-72) sppf{doc=partial; impl=partial; doc_ref=in-22@2}
- [~] Proof-carrying rewrite plans (rewrite plan artifacts + evidence links + report/LSP/snapshots; verification predicates executable + tested, but lane remains partial until full matrix closure; see `docs/matrix_acceptance.md`). Evidence anchors: `src/gabion/analysis/dataflow_audit.py::_compute_fingerprint_rewrite_plans`, `tests/test_rewrite_plan_verification.py::test_verify_rewrite_plan_enforces_exception_obligation_non_regression_when_requested`. (in-26, GH-76) sppf{doc=partial; impl=partial; doc_ref=in-26@10}
- [~] Rewrite plan kinds beyond BUNDLE_ALIGN (CTOR_NORMALIZE, SURFACE_CANONICALIZE, AMBIENT_REWRITE) with per-kind payload schemas, gated emission + reasoned abstentions, and predicate wiring (implemented subset exists; adoption remains partial). Evidence anchors: `src/gabion/analysis/dataflow_audit.py::_make_rewrite_plan`, `tests/test_rewrite_plan_verification.py::test_verify_rewrite_plan_extended_kinds_have_deterministic_behavior`. (in-26, GH-78) sppf{doc=partial; impl=partial; doc_ref=in-26@10}
- [~] Rewrite-plan verification: exception obligation non-regression predicates (predicate executes and is tested; wider handledness model still partial). Evidence anchors: `src/gabion/analysis/dataflow_audit.py::verify_rewrite_plan`, `tests/test_rewrite_plan_verification.py::test_verify_rewrite_plan_detects_remainder_regression`. (in-27, GH-79) sppf{doc=partial; impl=partial; doc_ref=in-27@8}

## LSP operational semantics
- [x] CLI as pure LSP client (no engine import; server-only logic).
- [x] Server as single source of truth for diagnostics/code actions (basic).
- [x] Analysis→diagnostics mapping (param-span ranges).
- [x] LSP executeCommand smoke test (dataflow command).
- [x] Pytest wrapper for LSP smoke test (skips if pygls missing).
- [x] VS Code extension as thin wrapper (spawn server only).

## Governance/ops glue (optional)
- [x] Durable logs/artifacts guidance for audits.
- [x] Audit snapshot tooling (dataflow report + DOT + docflow summary).
- [x] Repo config defaults (`gabion.toml`) for dataflow settings.
- [x] Policy check script references.
- [x] Hook installer.
- [x] Doer/Judge/Witness framing (optional).
- [x] Ops: Baseline/Ratchet mechanism (allowlist existing violations, block new ones). (GH-23)
- [x] Redistributable GitHub Action wrapper (composite action for gabion check).
- [x] Locked dependency set for CI (`requirements.lock`).
- [~] Coverage smell tracking (map tests to invariants/lemmas; track unmapped tests; dead/duplicate mapping diagnostics). (GH-42) sppf{doc=partial; impl=partial; doc_ref=docs/coverage_semantics.md@20}

## Decision-flow tier nodes
- <a id="decision-flow-tier3"></a>[x] Decision Table documentation for branch-heavy modules (Tier-3 evidence; see `docs/decision_flow_tiers.md#decision-flow-tier3`). (GH-47)
- <a id="decision-flow-tier2"></a>[x] Decision Bundle centralization for repeated guard patterns (Tier-2 evidence; see `docs/decision_flow_tiers.md#decision-flow-tier2`). (GH-48)
- <a id="decision-flow-tier1"></a>[x] Decision Protocol schema enforcement for critical decision paths (Tier-1 evidence; see `docs/decision_flow_tiers.md#decision-flow-tier1`). (GH-49)

## Explicit non-goals
- [x] Agda proof kernel (deferred).
- [x] GPU/JAX/Prism VM operational guidance (out of scope).

## Phase 2: Integration (post-scaffold)
- [x] Synthesis plan available via LSP/CLI (`gabion.synthesisPlan`, `synthesis-plan`).
- [x] Synthesis payload schema doc (`docs/synthesis_payload.md`). sppf{doc=done; impl=done; doc_ref=docs/synthesis_payload.md@4}
- [x] Dataflow audit can emit synthesis plan outputs (report + JSON).
- [x] Protocol/dataclass stub emitter (writes to `artifacts/`).
- [x] Refactoring plan output (per-bundle schedule).
- [x] `gabion synth` command to run audit + synthesis in one step.

## Phase 3: Refactoring & UX
- [x] LibCST refactor engine scaffolding (preserve trivia/formatting).
- [x] LSP code action + workspace edit stub for Protocol extraction.
- [x] LSP code action: stub command wiring.
- [x] LSP workspace edit: real edits for Protocol extraction.
- [x] Precise diagnostic ranges (metadata-backed positions).
- [x] Refactor engine: Signature rewriting (def foo(a, b) -> def foo(bundle)). (GH-17)
- [x] Refactor engine: Call-site rewriting (foo(x, y) -> foo(Bundle(x, y))). (GH-18)
- [x] Refactor engine: Preamble injection (unpack bundle to preserve local logic). (GH-19)
- [x] Refactor engine: Import management (inject bundle import at call sites). (GH-20)
- [x] Refactor strategy: Compatibility shims (generate @overload + DeprecationWarning wrapper). (GH-24)
- [ ] Long-lived LSP server cache / incremental analysis (daemon mode). (GH-21)

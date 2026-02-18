---
doc_revision: 150
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
  README.md#repo_contract: 1
  CONTRIBUTING.md#contributing_contract: 1
  glossary.md#decision_table: 1
  glossary.md#decision_bundle: 1
  glossary.md#decision_protocol: 1
  glossary.md#decision_surface: 1
  glossary.md#value_encoded_decision: 1
  glossary.md#deadness_witness: 1
  glossary.md#exception_obligation: 1
doc_review_notes:
  README.md#repo_contract: "Reviewed README.md rev1 (docflow audit now scans in/ by default); no conflicts with this document's scope."
  CONTRIBUTING.md#contributing_contract: "Reviewed CONTRIBUTING.md rev1 (docflow now fails on missing GH references for SPPF-relevant changes); no conflicts with this document's scope."
  glossary.md#decision_table: "Reviewed glossary.md#decision_table rev1 (decision table tier definition)."
  glossary.md#decision_bundle: "Reviewed glossary.md#decision_bundle rev1 (decision bundle tier definition)."
  glossary.md#decision_protocol: "Reviewed glossary.md#decision_protocol rev1 (decision protocol tier definition)."
  glossary.md#decision_surface: "Reviewed glossary.md#decision_surface rev1 (decision surface tier boundary semantics)."
  glossary.md#value_encoded_decision: "Reviewed glossary.md#value_encoded_decision rev1 (value-encoded decision surface semantics)."
  glossary.md#deadness_witness: "Reviewed glossary.md#deadness_witness rev1 (deadness witness obligations for negative evidence)."
  glossary.md#exception_obligation: "Reviewed glossary.md#exception_obligation rev1 (exception obligation status + evidence linkage)."
doc_sections:
  sppf_checklist: 7
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
      dep_version: 1
      self_version_at_review: 7
      outcome: no_change
      note: "Reviewed README.md rev1 (docflow audit now scans in/ by default); no conflicts with this document's scope."
    CONTRIBUTING.md#contributing_contract:
      dep_version: 1
      self_version_at_review: 7
      outcome: no_change
      note: "Reviewed CONTRIBUTING.md rev1 (docflow now fails on missing GH references for SPPF-relevant changes); no conflicts with this document's scope."
    glossary.md#decision_table:
      dep_version: 1
      self_version_at_review: 7
      outcome: no_change
      note: "Reviewed glossary.md#decision_table rev1 (decision table tier definition)."
    glossary.md#decision_bundle:
      dep_version: 1
      self_version_at_review: 7
      outcome: no_change
      note: "Reviewed glossary.md#decision_bundle rev1 (decision bundle tier definition)."
    glossary.md#decision_protocol:
      dep_version: 1
      self_version_at_review: 7
      outcome: no_change
      note: "Reviewed glossary.md#decision_protocol rev1 (decision protocol tier definition)."
    glossary.md#decision_surface:
      dep_version: 1
      self_version_at_review: 7
      outcome: no_change
      note: "Reviewed glossary.md#decision_surface rev1 (decision surface tier boundary semantics)."
    glossary.md#value_encoded_decision:
      dep_version: 1
      self_version_at_review: 7
      outcome: no_change
      note: "Reviewed glossary.md#value_encoded_decision rev1 (value-encoded decision surface semantics)."
    glossary.md#deadness_witness:
      dep_version: 1
      self_version_at_review: 7
      outcome: no_change
      note: "Reviewed glossary.md#deadness_witness rev1 (deadness witness obligations for negative evidence)."
    glossary.md#exception_obligation:
      dep_version: 1
      self_version_at_review: 7
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
- [~] Internal broad-type lint (Any/object + scalar types like str/int/float/bool/bytes, except NodeId) on non-boundary surfaces. (GH-89) sppf{doc=partial; impl=done; doc_ref=in-31@3}
- [x] Type-flow ambiguities fail `gabion check` in repo defaults.
- [x] Anonymous schema surface detection (dict[str, object]/Any payload annotations).
- [x] Constant-flow audit (dead knobs / always-constant params).
- [x] Constant-flow helper specificity (avoid false positives in internal helpers).
- [x] Unused-argument pass detection (non-test call sites).
- [x] Analysis: Decorator transparency/unwrapping. (GH-9)
- [x] Verification: Idempotency test (ensure Analysis(Refactor(Code)) == Stable). (GH-22)
- [~] Decision surface detection + boundary elevation (tier enforcement). (in-15, GH-60) sppf{doc=partial; impl=partial; doc_ref=in-15@1}
- [~] Decision surface hooks in grammar (`is_decision_surface`). (GH-60)
- [x] Decision surface boundary diagnostics (API surface vs internal depth). (GH-60)
- [x] Decision surface tier enforcement via glossary metadata. (GH-60)
- [~] Value-encoded decision surface detection (branchless / algebraic control). (in-18, GH-66) sppf{doc=partial; impl=partial; doc_ref=in-18@1}
- [~] Value-encoded decision heuristics (min/max, bitmask, boolean arithmetic). (GH-66)
- [~] Value-encoded decision surface reports in audit output. (GH-66)
- [x] Value-encoded decision glossary warnings (nonlinear contexts). (GH-66)
- [x] Value-encoded decision rewrite suggestions (rebranch). (GH-66)
- [x] Value-encoded decision diff tracking in audit snapshots. (GH-66)
- [~] Prime-labeled type fingerprints (algebraic bundle matching). (in-20/in-21, GH-68) sppf{doc=partial; impl=done; doc_ref=in-20@1,in-21@1}
- [x] Prime registry + canonical type key mapping. (GH-68)
- [x] Fingerprint arithmetic ops (gcd/lcm/subtyping checks). (GH-68)
- [x] Glossary fingerprint matching + CI warnings. (GH-68)
- [x] Hybrid fingerprint representation (prime products + bitmask existence checks). (GH-68)
- [~] Deterministic fingerprint registry seeding (sorted key interning for primes/bits). (in-22, GH-68) sppf{doc=partial; impl=done; doc_ref=in-22@2}
- [x] Nested type constructor registry (dimensional prime mapping). (GH-68)
- [x] Fingerprint reverse mapping for synthesis (factorization → type keys). (GH-68)
- [~] ASPF dimensional fingerprints (base/ctor carriers + soundness invariants). (in-22, GH-70) sppf{doc=partial; impl=done; doc_ref=in-22@2}
- [~] ASPF provenance mapping to SPPF (packed-forest derivation reporting + invariants; base/ctor keys + JSON artifact + report summary). (in-22, GH-71) sppf{doc=partial; impl=done; doc_ref=in-22@2}
- <a id="in-23-aspf-carrier-formalization"></a>[x] ASPF carrier obligations formalized (determinism, base conservation, ctor coherence, synth tail reversibility, provenance completeness, snapshot reproducibility). (in-23, GH-73; anchors: `src/gabion/analysis/dataflow_audit.py::_compute_fingerprint_provenance`, `src/gabion/analysis/dataflow_audit.py::_compute_fingerprint_synth`, `src/gabion/analysis/type_fingerprints.py::build_synth_registry_from_payload`, `tests/test_type_fingerprints.py::test_build_fingerprint_registry_deterministic_assignment`, `tests/test_type_fingerprints.py::test_synth_registry_payload_roundtrip`, `tests/test_fingerprint_warnings.py::test_fingerprint_provenance_emits_entries`, `scripts/audit_snapshot.sh`, `scripts/latest_snapshot.sh`). sppf{doc=done; impl=done; doc_ref=in-23@9}
- [~] SuiteSite carriers + loop-scoped deadline obligations. (in-30, GH-85) sppf{doc=partial; impl=partial; doc_ref=in-30@24}
- [~] Deadline propagation as gas (ticks-based carriers across LSP/CLI/server). (in-30, GH-87) sppf{doc=partial; impl=done; doc_ref=in-30@24}
- [~] Structural ambiguity as CallCandidate alts (SuiteSite) with virtual AmbiguitySet. (in-30, GH-88) sppf{doc=partial; impl=done; doc_ref=in-30@24}
- <a id="in-33-pattern-schema-unification"></a>[~] PatternSchema unification for dataflow bundles + execution patterns (shared schema IDs + residue reporting; execution rules still narrow). (in-33) sppf{doc=partial; impl=partial; doc_ref=in-33@1}
- <a id="in-34-lambda-callable-sites"></a>[x] Lambda/closure callable indexing as first-class function sites (stable synthetic identities + direct/bound/closure lambda resolution; conservative dynamic fallback retained). Acceptance tests: `tests/test_dataflow_resolve_callee.py::test_resolve_callee_bound_lambda_call`, `tests/test_dataflow_resolve_callee.py::test_resolve_callee_closure_returned_and_invoked`, `tests/test_dataflow_resolve_callee.py::test_resolve_callee_bound_lambda_via_object_attribute`, `tests/test_dataflow_resolve_callee.py::test_resolve_callee_outcome_keeps_dynamic_fallback_for_attribute_calls`, `tests/test_callsite_evidence_helper.py::test_callsite_evidence_includes_callable_context`. (in-34) sppf{doc=implemented; impl=implemented; doc_ref=in-34@2}
- <a id="in-35-dict-key-carrier-tracking"></a>[~] Dict carrier tracking beyond literal subscript aliases (name-bound constant keys + unknown-key carrier evidence). (in-35) sppf{doc=partial; impl=done; doc_ref=in-35@1}
- <a id="in-36-starred-dataclass-call-bundles"></a>[x] Conservative starred dataclass constructor argument handling (`*` list/tuple/set, `**` dict literal) with unresolved-starred witnesses for dynamic payloads. (in-36) sppf{doc=done; impl=done; doc_ref=in-36@1}
- <a id="in-37-dynamic-dispatch-uncertainty"></a>[x] Dynamic-dispatch uncertainty classification in call resolution (`unresolved_dynamic`) plus dedicated call-resolution obligation kind. (in-37) sppf{doc=done; impl=done; doc_ref=in-37@1}

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
- [x] FactorizationTree snapshot emission (canonical JSON). (in-16, GH-62) sppf{doc=done; impl=done; doc_ref=in-16@1}
- [x] Structural diff command + baseline comparison. (in-16, GH-63) sppf{doc=done; impl=done; doc_ref=in-16@1}
- [x] Structural metrics export (bundle/tier/violation stats). (in-16, GH-64) sppf{doc=done; impl=done; doc_ref=in-16@1}
- [x] Deadness evidence artifacts (constant-flow deadness witnesses + JSON/report/LSP + snapshot selectors + determinism/schema tests; see `docs/matrix_acceptance.md`). (in-24, GH-74) sppf{doc=done; impl=done; doc_ref=in-24@9}
- [x] Coherence evidence artifacts (glossary-ambiguity witnesses + JSON/report/LSP + snapshot selectors + determinism/schema tests; see `docs/matrix_acceptance.md`). (in-25, GH-75) sppf{doc=done; impl=done; doc_ref=in-25@9}
- [~] Exception obligation artifacts (E0 enumeration + JSON/report/LSP + snapshot selectors; handledness via broad try/except; deadness discharge for constant-flow guarded branches; see `docs/matrix_acceptance.md`). (in-27, GH-77) sppf{doc=partial; impl=done; doc_ref=in-27@7}
- [ ] Exception obligations: handledness refinement (typed except + conservative UNKNOWN). (in-27, GH-80) sppf{doc=partial; impl=planned; doc_ref=in-27@7}

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
- [~] Contextvar/ambient context rewrite suggestions. (in-15, GH-61) sppf{doc=partial; impl=partial; doc_ref=in-15@1}
- [x] Contextvar suggestion heuristics (internal decision surfaces). (GH-61)
- [ ] Contextvar rewrite: synthesis emits ContextVar definitions + accessors. (GH-61)
- [ ] Contextvar rewrite: callsite replacement for ambient access. (GH-61)
- [~] Subtree reuse detection + lemma synthesis hooks. (in-17, GH-65) sppf{doc=partial; impl=partial; doc_ref=in-17@1}
- [x] Subtree hashing/fingerprinting for FactorizationTree reuse. (GH-65)
- [~] Lemma suggestion output + stable naming map. (GH-65)
- [x] Lemma suggestion CLI flag + output map (declare + replace). (GH-65)
- [~] Lemma emission target selection (inline vs stub module). (GH-65)
- [x] Glossary-backed lemma naming + missing-entry warnings. (GH-65)
- [~] Invariant extraction + dependent-type synthesis (Agda). (in-19, GH-67) sppf{doc=partial; impl=partial; doc_ref=in-19@1}
- [x] Proposition model + assert-based invariant extraction. (GH-67)
- [x] Invariant emitter hooks (pluggable callbacks). (GH-67)
- [ ] Dependent-type / Agda synthesis output from invariants. (GH-67)
- [x] Invariant-enriched JSON output for bundles/trees. (GH-67)
- [ ] Property-based test hooks from invariants. (GH-67)
- [~] ASPF entropy-controlled synthesis (synth@k primes + tail mapping + versioned registry; report + JSON registry output + snapshots + loadable registry). (in-22, GH-72) sppf{doc=partial; impl=partial; doc_ref=in-22@2}
- [~] Proof-carrying rewrite plans (rewrite plan artifacts + evidence links + report/LSP/snapshots; verification predicates executable + tested; see `docs/matrix_acceptance.md`). (in-26, GH-76) sppf{doc=partial; impl=done; doc_ref=in-26@9}
- [ ] Rewrite plan kinds beyond BUNDLE_ALIGN (CTOR_NORMALIZE, SURFACE_CANONICALIZE, AMBIENT_REWRITE). (in-26, GH-78) sppf{doc=partial; impl=planned; doc_ref=in-26@9}
- [~] Rewrite-plan verification: exception obligation non-regression predicates. (in-27, GH-79) sppf{doc=partial; impl=done; doc_ref=in-27@7}

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
- [ ] Coverage smell tracking (map tests to invariants/lemmas; track unmapped tests). (GH-42)

## Decision-flow tier nodes
- [ ] Decision Table documentation for branch-heavy modules (Tier-3 evidence). (GH-47)
- [ ] Decision Bundle centralization for repeated guard patterns (Tier-2 evidence). (GH-48)
- [ ] Decision Protocol schema enforcement for critical decision paths (Tier-1 evidence). (GH-49)

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

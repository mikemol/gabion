---
doc_revision: 39
reader_reintern: "Reader-only: re-intern if doc_revision changed since you last read this doc."
doc_id: projection_semantic_fragment_ledger
doc_role: audit
doc_scope:
  - repo
  - projection
  - policy
  - aspf
  - ttl
doc_authority: informative
doc_requires:
  - POLICY_SEED.md#policy_seed
  - glossary.md#contract
  - docs/projection_semantic_fragment_rfc.md#projection_semantic_fragment_rfc
  - docs/ttl_kernel_semantics.md#ttl_kernel_semantics
  - docs/aspf_execution_fibration.md#aspf_execution_fibration
  - docs/audits/projection_spec_history_ledger.md
doc_reviewed_as_of:
  POLICY_SEED.md#policy_seed: 55
  glossary.md#contract: 44
  docs/projection_semantic_fragment_rfc.md#projection_semantic_fragment_rfc: 47
  docs/ttl_kernel_semantics.md#ttl_kernel_semantics: 1
  docs/aspf_execution_fibration.md#aspf_execution_fibration: 7
  docs/audits/projection_spec_history_ledger.md: 1
doc_review_notes:
  POLICY_SEED.md#policy_seed: "Reviewed POLICY_SEED.md rev55 (fix-forward correction units and artifact-backed continuation remain aligned with this ledger)."
  glossary.md#contract: "Reviewed glossary.md rev44 (witness/evidence/projection terms remain aligned with the semantic-fragment queue language)."
  docs/projection_semantic_fragment_rfc.md#projection_semantic_fragment_rfc: "Reviewed the implementation RFC rev47 and kept the queue aligned with policy_check-owned continuation artifacts, direct semantic-summary sourcing, duplicate suite-side queue ownership removal, duplicate suite-payload semantic-summary embedding removal, carrier-owned lattice witness serialization, source-artifact queue naming, child-owned skip/fail-closed artifact ownership, wrapper reuse of preexisting child-owned canonical artifacts, payload-first hotspot queue generation, generic source-artifact hotspot queue ingress, generic hotspot queue source metadata naming, removal of the queue sidecar policy-check supplement contract, explicit hotspot source-artifact CLI ingress, removal of the wrapper-only top-level policy_check summary-decoder branch, removal of the stale top-level summary decoder fallback, explicit policy-scanner-suite artifact-path ownership, generic cached policy-results normalization, direct semantic-field hotspot queue reporting, canonical-only shared summary-helper ingestion, removal of the redundant policy-results alias entrypoint, removal of suite-cache hash projection from hotspot queue source metadata, removal of child-owned policy-result persistence from the wrapper cache artifact, replacement of nested `policy_results.policy_check` semantic ingress with direct `projection_fiber_semantics` on suite runtime payloads, removal of full child-policy payload retention from the runtime suite result carrier, removal of wrapper child-status retention from the runtime suite result carrier, removal of `cached` from the runtime suite result carrier itself, removal of `root` from the runtime suite result carrier itself, replacement of raw child-policy result mappings at the runtime load/scan boundary with a typed child-input bundle, removal of wrapper child-status traffic from the runtime child-input bundle itself, removal of the wrapper's temporary child-result mapping rendezvous, removal of raw child-result mapping traffic from wrapper orchestration after ingress validation, removal of the raw child-result parser from the runtime module itself, and removal of cache hash projection from the public suite result/payload surface."
  docs/ttl_kernel_semantics.md#ttl_kernel_semantics: "Reviewed the TTL explainer rev1 and kept SHACL/SPARQL realization language aligned with the ledger rows."
  docs/aspf_execution_fibration.md#aspf_execution_fibration: "Reviewed ASPF execution fibration rev7 and retained ASPF/global identity continuity as a non-negotiable bootstrap constraint."
  docs/audits/projection_spec_history_ledger.md: "Reviewed projection-spec history ledger rev1; legacy ProjectionSpec remains the compatibility surface referenced by queued cutover work."
doc_change_protocol: "POLICY_SEED.md#change_protocol"
doc_sections:
  projection_semantic_fragment_ledger: 1
doc_section_requires:
  projection_semantic_fragment_ledger:
    - POLICY_SEED.md#policy_seed
    - glossary.md#contract
    - docs/projection_semantic_fragment_rfc.md#projection_semantic_fragment_rfc
    - docs/ttl_kernel_semantics.md#ttl_kernel_semantics
    - docs/aspf_execution_fibration.md#aspf_execution_fibration
    - docs/audits/projection_spec_history_ledger.md
doc_section_reviews:
  projection_semantic_fragment_ledger:
    POLICY_SEED.md#policy_seed:
      dep_version: 55
      self_version_at_review: 1
      outcome: no_change
      note: "Policy seed rev55 reviewed; correction-unit and artifact-loop expectations remain aligned."
    glossary.md#contract:
      dep_version: 44
      self_version_at_review: 1
      outcome: no_change
      note: "Glossary rev44 reviewed; witness/evidence/projection language remains aligned."
    docs/projection_semantic_fragment_rfc.md#projection_semantic_fragment_rfc:
      dep_version: 33
      self_version_at_review: 25
      outcome: no_change
      note: "RFC rev33 reviewed; phase structure, fast-path continuation artifacts, duplicate-wrapper queue ownership removal, duplicate summary removal, carrier-owned lattice witness serialization, source-artifact queue naming, child-owned skip/fail-closed artifact ownership, generic preexisting child-artifact reuse, payload-first hotspot queue generation, generic source-artifact hotspot ingress, generic hotspot queue source metadata naming, queue sidecar policy-check supplement removal, explicit hotspot source-artifact CLI ingress, wrapper-only top-level policy_check summary-decoder removal, stale top-level summary decoder removal, explicit policy-scanner-suite artifact-path ownership, generic cached policy-results normalization, direct semantic-field hotspot queue reporting, canonical-only shared summary-helper ingestion, redundant policy-results alias entrypoint removal, and ratchet rules anchor the queue rows."
    docs/ttl_kernel_semantics.md#ttl_kernel_semantics:
      dep_version: 1
      self_version_at_review: 1
      outcome: no_change
      note: "TTL explainer rev1 reviewed; semantic/executable/friendly-layer stratification remains aligned."
    docs/aspf_execution_fibration.md#aspf_execution_fibration:
      dep_version: 7
      self_version_at_review: 1
      outcome: no_change
      note: "ASPF execution fibration rev7 reviewed; identity/fibration reuse remains aligned."
    docs/audits/projection_spec_history_ledger.md:
      dep_version: 1
      self_version_at_review: 1
      outcome: no_change
      note: "ProjectionSpec history ledger rev1 reviewed; compatibility-surface framing remains aligned."
doc_erasure:
  - formatting
  - typos
doc_owner: maintainer
---

<a id="projection_semantic_fragment_ledger"></a>
# Projection Semantic Fragment Ledger

This ledger is the human-readable continuation surface for the projection
semantic-fragment workstream. The generated queue artifact is the machine
source of truth for current status:

- `artifacts/out/policy_suite_results.json`
- `artifacts/out/policy_check_result.json`
- `artifacts/out/hotspot_neighborhood_queue.json`
- `artifacts/out/projection_semantic_fragment_queue.json`
- `artifacts/out/projection_semantic_fragment_queue.md`

Update this document per correction unit when queue IDs, cutover status, or
ratchet expectations change.

## Status Key

- `landed`: implemented on a real semantic path and validated.
- `in_progress`: active adapter-backed convergence work.
- `queued`: accepted next work, not yet landed.

## Current Semantic Boundary

As of the current correction unit, the stable landed slice is:

- canonical projection-fiber semantic rows reflected from ASPF/fibration
  witnesses
- deterministic reflect + projection-fiber quotient-face lowering for
  `projection_fiber.frontier` and
  `projection_fiber.reflective_boundary` into SHACL/SPARQL plan carriers
- policy and reporting artifacts that propagate semantic-plan summaries and
  semantic previews without moving semantic ownership back into presentation,
  with hotspot/report consumers preferring direct `policy_check` semantic
  artifacts when available

Still adapter-only:

- `projection_exec.py` remains the compatibility runtime for legacy row-shaped
  `ProjectionSpec` execution
- only declared quotient-face slices are promoted through typed lowering
- `semantic_carrier_adapter` boundaries remain temporary until RFC cutover
  criteria are satisfied

## Landed Slices

| date | slice | evidence_links |
| --- | --- | --- |
| `2026-03-11` | Carrier-first `projection_fiber` reflection emits canonical witnessed semantic rows with structural identity continuity. | `src/gabion/analysis/projection/semantic_fragment.py`; `src/gabion/tooling/policy_substrate/lattice_convergence_semantic.py`; `tests/gabion/tooling/runtime_policy/test_lattice_convergence_semantic.py` |
| `2026-03-11` | Reflect + `projection_fiber.frontier` quotient-face lowering compiles deterministically into SHACL/SPARQL plan carriers. | `src/gabion/analysis/projection/semantic_fragment_compile.py`; `src/gabion/analysis/projection/projection_semantic_lowering.py`; `src/gabion/analysis/projection/projection_semantic_lowering_compile.py`; `tests/gabion/analysis/projection/test_projection_semantic_lowering_compile.py` |
| `2026-03-11` | A second authoring-layer face, `projection_fiber.reflective_boundary`, now compiles over the same canonical carrier and shares the same typed lowering path. | `src/gabion/analysis/projection/projection_registry.py`; `src/gabion/analysis/projection/semantic_fragment_compile.py`; `src/gabion/analysis/projection/projection_semantic_lowering_compile.py`; `tests/gabion/analysis/projection/test_projection_semantic_lowering_compile.py` |
| `2026-03-11` | `policy_check` output, policy-suite summaries, and hotspot queue artifacts now carry semantic-plan context instead of opaque projection baggage. | `scripts/policy/policy_check.py`; `src/gabion/tooling/runtime/policy_scanner_suite.py`; `scripts/policy/hotspot_neighborhood_queue.py`; `tests/gabion/tooling/runtime_policy/test_policy_check_output.py`; `tests/gabion/tooling/policy/test_hotspot_neighborhood_queue.py` |
| `2026-03-11` | Hotspot queue sourcing now prefers direct `policy_check_result.json` semantic summaries when available, reducing dependence on suite-local semantic embedding while keeping the same preview surface. | `scripts/policy/hotspot_neighborhood_queue.py`; `scripts/policy/policy_scanner_suite.py`; `src/gabion/tooling/runtime/projection_fiber_semantics_summary.py`; `tests/gabion/tooling/policy/test_hotspot_neighborhood_queue.py`; `tests/gabion/tooling/policy/test_policy_scanner_suite_script.py` |
| `2026-03-11` | Policy-suite artifacts no longer embed their own duplicate `projection_fiber_semantics_summary`; consumers derive the same summary from `policy_results.policy_check` or the direct `policy_check_result.json` artifact. | `src/gabion/tooling/runtime/policy_scanner_suite.py`; `src/gabion/tooling/runtime/projection_fiber_semantics_summary.py`; `tests/gabion/tooling/runtime_policy/test_policy_scanner_suite.py`; `tests/gabion/tooling/policy/test_hotspot_neighborhood_queue.py` |
| `2026-03-11` | `policy_check` now emits the projection-semantic-fragment continuation queue directly from `policy_check_result.json`; the policy-suite wrapper preserves that fast-path artifact but no longer backfills or regenerates it. | `scripts/policy/policy_check.py`; `scripts/policy/policy_scanner_suite.py`; `tests/gabion/tooling/runtime_policy/test_policy_check_output.py`; `tests/gabion/tooling/policy/test_policy_scanner_suite_script.py` |
| `2026-03-11` | The policy-suite wrapper dropped its remaining semantic-queue ownership paths: no pre-scan duplicate emission, no post-scan backfill, and explicit tests for preserve-only behavior. | `scripts/policy/policy_scanner_suite.py`; `tests/gabion/tooling/policy/test_policy_scanner_suite_script.py`; `docs/projection_semantic_fragment_rfc.md#projection_semantic_fragment_rfc` |
| `2026-03-11` | Lattice witness serialization is now owned by `FrontierWitness.as_payload()` in the canonical ASPF algebra, and the lattice-convergence substrate no longer imports the runtime policy-suite wrapper just to validate a serializer contract. | `src/gabion/analysis/aspf/aspf_lattice_algebra.py`; `src/gabion/tooling/policy_substrate/lattice_convergence_semantic.py`; `src/gabion/tooling/runtime/policy_scanner_suite.py`; `src/gabion/tooling/policy_rules/branchless_rule.py`; `tests/gabion/tooling/runtime_policy/test_policy_substrate_runtime.py`; `tests/gabion/tooling/runtime_policy/test_lattice_convergence_semantic.py` |
| `2026-03-11` | Continuation queue artifacts are generated from a generic source artifact interface that defaults to `policy_check_result.json`, removing stale suite-era naming from the queue tool while keeping fast-path ownership explicit. | `scripts/policy/projection_semantic_fragment_queue.py`; `scripts/policy/policy_check.py`; `tests/gabion/tooling/policy/test_projection_semantic_fragment_queue.py`; `docs/projection_semantic_fragment_rfc.md#projection_semantic_fragment_rfc` |
| `2026-03-11` | The deprecated-nonerasability child check now emits its own canonical `skip` result when baseline/current inputs are absent, and the policy-suite wrapper no longer synthesizes any policy-result carriers; all child artifact gaps are fail-closed wrapper errors. | `scripts/policy/deprecated_nonerasability_policy_check.py`; `scripts/policy/policy_scanner_suite.py`; `tests/gabion/tooling/runtime_policy/test_deprecated_nonerasability_policy_check.py`; `tests/gabion/tooling/policy/test_policy_scanner_suite_script.py`; `docs/projection_semantic_fragment_rfc.md#projection_semantic_fragment_rfc` |
| `2026-03-11` | The policy-suite wrapper now consumes any valid preexisting child-owned canonical result (`policy_check`, `structural_hash`, `deprecated_nonerasability`) instead of rerunning the owning child check, so the wrapper preserves child artifact ownership rather than reasserting orchestration ownership. | `scripts/policy/policy_scanner_suite.py`; `tests/gabion/tooling/policy/test_policy_scanner_suite_script.py`; `docs/projection_semantic_fragment_rfc.md#projection_semantic_fragment_rfc` |
| `2026-03-11` | Hotspot-neighborhood queue generation no longer treats `policy_suite_results.json` as a rendezvous surface inside the wrapper; the suite runner now passes its in-memory payload directly to a payload-first queue entrypoint. | `scripts/policy/policy_scanner_suite.py`; `scripts/policy/hotspot_neighborhood_queue.py`; `tests/gabion/tooling/policy/test_policy_scanner_suite_script.py`; `tests/gabion/tooling/policy/test_hotspot_neighborhood_queue.py`; `docs/projection_semantic_fragment_rfc.md#projection_semantic_fragment_rfc` |
| `2026-03-11` | The hotspot-neighborhood queue tool now exposes a generic `source_artifact` ingress rather than a `policy-suite`-named one, removing another stale wrapper-era interface while preserving the same queue semantics. | `scripts/policy/hotspot_neighborhood_queue.py`; `tests/gabion/tooling/policy/test_hotspot_neighborhood_queue.py`; `docs/projection_semantic_fragment_rfc.md#projection_semantic_fragment_rfc` |
| `2026-03-11` | The hotspot-neighborhood queue artifact now serializes generic source metadata fields (`source_generated_at_utc`, `source_counts`) instead of suite-era `policy_suite_*` names, so emitted reporting payloads no longer project obsolete wrapper ownership language. | `scripts/policy/hotspot_neighborhood_queue.py`; `tests/gabion/tooling/policy/test_hotspot_neighborhood_queue.py`; `docs/projection_semantic_fragment_rfc.md#projection_semantic_fragment_rfc` |
| `2026-03-11` | The hotspot-neighborhood queue dropped its separate `policy_check` supplement contract and now reads projection-fiber semantics only from the normalized source payload, removing another dual-ingress wrapper-era surface. | `scripts/policy/hotspot_neighborhood_queue.py`; `scripts/policy/policy_scanner_suite.py`; `tests/gabion/tooling/policy/test_hotspot_neighborhood_queue.py`; `tests/gabion/tooling/policy/test_policy_scanner_suite_script.py`; `docs/projection_semantic_fragment_rfc.md#projection_semantic_fragment_rfc` |
| `2026-03-11` | The hotspot-neighborhood queue CLI no longer defaults to `policy_suite_results.json`; generic queue ingress is now explicit via required `--source-artifact`, and the queue tests use generic source-artifact fixture names instead of suite-era ones. | `scripts/policy/hotspot_neighborhood_queue.py`; `tests/gabion/tooling/policy/test_hotspot_neighborhood_queue.py`; `docs/projection_semantic_fragment_rfc.md#projection_semantic_fragment_rfc` |
| `2026-03-11` | The policy-scanner-suite runtime and CLI no longer bake in `policy_suite_results.json` as a default artifact path; callers must pass the compatibility wrapper’s output artifact explicitly, removing another implicit suite-era ownership contract. | `src/gabion/tooling/runtime/policy_scanner_suite.py`; `scripts/policy/policy_scanner_suite.py`; `tests/gabion/tooling/policy/test_policy_scanner_suite_script.py`; `docs/projection_semantic_fragment_rfc.md#projection_semantic_fragment_rfc` |
| `2026-03-11` | The runtime policy-scanner-suite cache loader now reuses the generic canonical `policy_results` mapping normalizer instead of a fixed three-child projection helper, removing another suite-specific assumption from cached artifact reads. | `src/gabion/tooling/runtime/policy_scanner_suite.py`; `tests/gabion/tooling/runtime_policy/test_policy_scanner_suite.py`; `docs/projection_semantic_fragment_rfc.md#projection_semantic_fragment_rfc` |
| `2026-03-11` | The hotspot-neighborhood queue no longer materializes a queue-owned `projection_fiber_semantics_summary` blob; it now carries only direct semantic decision/bundle-count/preview fields and renders markdown from those direct rows. | `scripts/policy/hotspot_neighborhood_queue.py`; `tests/gabion/tooling/policy/test_hotspot_neighborhood_queue.py`; `tests/gabion/tooling/policy/test_policy_scanner_suite_script.py`; `docs/projection_semantic_fragment_rfc.md#projection_semantic_fragment_rfc` |
| `2026-03-11` | The shared projection-fiber summary helper no longer exposes a dedicated parser for retired materialized-summary payloads; it now accepts only live canonical carriers and rejects dead summary-only wrapper shapes outright. | `src/gabion/tooling/runtime/projection_fiber_semantics_summary.py`; `tests/gabion/tooling/runtime_policy/test_policy_scanner_suite.py`; `docs/projection_semantic_fragment_rfc.md#projection_semantic_fragment_rfc` |
| `2026-03-11` | The shared projection-fiber summary helper no longer exposes a redundant `...from_policy_results(...)` alias entrypoint; canonical decoding now flows through the single live payload decoder. | `src/gabion/tooling/runtime/projection_fiber_semantics_summary.py`; `docs/projection_semantic_fragment_rfc.md#projection_semantic_fragment_rfc` |
| `2026-03-11` | The hotspot-neighborhood queue artifact no longer projects suite-cache hash fields (`inventory_hash`, `rule_set_hash`, `policy_results_hash`, `changed_scope_hash`) into its source metadata; those wrapper cache internals are now treated as non-contractual provenance and removed from the reporting surface. | `scripts/policy/hotspot_neighborhood_queue.py`; `tests/gabion/tooling/policy/test_hotspot_neighborhood_queue.py`; `docs/projection_semantic_fragment_rfc.md#projection_semantic_fragment_rfc` |
| `2026-03-11` | The shared projection-fiber summary decoder no longer accepts a top-level `policy_check` wrapper payload as a generic ingress shape; canonical decoding is restricted to direct `policy_check` payloads, `policy_results.policy_check`, or explicit summary payloads. | `src/gabion/tooling/runtime/projection_fiber_semantics_summary.py`; `tests/gabion/tooling/runtime_policy/test_policy_scanner_suite.py`; `docs/projection_semantic_fragment_rfc.md#projection_semantic_fragment_rfc` |
| `2026-03-11` | The shared projection-fiber summary decoder dropped its retired top-level `projection_fiber_semantics_summary` fallback; canonical payload decoding now uses real semantic carriers, while already-materialized summaries use an explicit summary-payload parser. | `src/gabion/tooling/runtime/projection_fiber_semantics_summary.py`; `scripts/policy/hotspot_neighborhood_queue.py`; `tests/gabion/tooling/runtime_policy/test_policy_scanner_suite.py`; `docs/projection_semantic_fragment_rfc.md#projection_semantic_fragment_rfc` |
| `2026-03-11` | The wrapper-owned `policy_suite_results.json` cache artifact no longer persists child-owned `policy_results`; cache hits now recover the in-memory child results from the explicit caller-supplied mapping instead of from the saved wrapper artifact. | `src/gabion/tooling/runtime/policy_scanner_suite.py`; `tests/gabion/tooling/runtime_policy/test_policy_scanner_suite.py`; `docs/projection_semantic_fragment_rfc.md#projection_semantic_fragment_rfc` |
| `2026-03-11` | Policy-scanner-suite runtime payloads no longer expose semantic context through nested `policy_results.policy_check`; they now surface direct top-level `projection_fiber_semantics`, and the shared summary helper no longer follows the wrapper-only nested path. | `src/gabion/tooling/runtime/policy_scanner_suite.py`; `src/gabion/tooling/runtime/projection_fiber_semantics_summary.py`; `tests/gabion/tooling/runtime_policy/test_policy_scanner_suite.py`; `tests/gabion/tooling/policy/test_hotspot_neighborhood_queue.py`; `tests/gabion/tooling/policy/test_policy_scanner_suite_script.py`; `docs/projection_semantic_fragment_rfc.md#projection_semantic_fragment_rfc` |
| `2026-03-11` | The runtime `PolicySuiteResult` carrier no longer retains full child-owned `policy_results` payloads; it now keeps only explicit child-status fields plus the direct `projection_fiber_semantics` carrier required by downstream reporting. | `src/gabion/tooling/runtime/policy_scanner_suite.py`; `scripts/policy/policy_scanner_suite.py`; `tests/gabion/tooling/runtime_policy/test_policy_scanner_suite.py`; `tests/test_policy_dsl.py`; `docs/projection_semantic_fragment_rfc.md#projection_semantic_fragment_rfc` |
| `2026-03-11` | The runtime `PolicySuiteResult` carrier no longer retains wrapper child-status metadata at all; wrapper console/status reporting now reads those statuses from the boundary-owned `PolicySuiteChildInputs` bundle instead of the runtime semantic carrier. | `src/gabion/tooling/runtime/policy_scanner_suite.py`; `scripts/policy/policy_scanner_suite.py`; `tests/gabion/tooling/runtime_policy/test_policy_scanner_suite.py`; `tests/test_policy_dsl.py`; `docs/projection_semantic_fragment_rfc.md#projection_semantic_fragment_rfc` |
| `2026-03-11` | The runtime `PolicySuiteResult` carrier no longer owns the `cached` bit at all; cache-hit state now lives only on the separate load outcome returned by `load_or_scan_policy_suite()`, and the outward payload remains free of that wrapper/runtime concern. | `src/gabion/tooling/runtime/policy_scanner_suite.py`; `scripts/policy/policy_scanner_suite.py`; `tests/gabion/tooling/runtime_policy/test_policy_scanner_suite.py`; `tests/test_policy_dsl.py`; `docs/projection_semantic_fragment_rfc.md#projection_semantic_fragment_rfc` |
| `2026-03-11` | The runtime `PolicySuiteResult` carrier no longer owns or projects `root`; repository-root provenance is now treated as wrapper/cache context rather than part of the outward suite semantic/reporting carrier. | `src/gabion/tooling/runtime/policy_scanner_suite.py`; `tests/gabion/tooling/runtime_policy/test_policy_scanner_suite.py`; `tests/test_policy_dsl.py`; `docs/projection_semantic_fragment_rfc.md#projection_semantic_fragment_rfc` |
| `2026-03-11` | The runtime policy-scanner-suite load/scan boundary no longer accepts raw child `policy_results` mappings; wrapper ingress now normalizes those payloads once into a typed `PolicySuiteChildInputs` bundle, and cache identity tracks that narrower boundary contract via `child_inputs_hash`. | `src/gabion/tooling/runtime/policy_scanner_suite.py`; `scripts/policy/policy_scanner_suite.py`; `tests/gabion/tooling/runtime_policy/test_policy_scanner_suite.py`; `tests/gabion/tooling/policy/test_policy_scanner_suite_script.py`; `docs/projection_semantic_fragment_rfc.md#projection_semantic_fragment_rfc` |
| `2026-03-11` | The runtime `PolicySuiteChildInputs` bundle no longer carries wrapper child-status metadata; runtime child inputs are now semantic-only (`projection_fiber_semantics`), while wrapper status reporting lives in a separate boundary-local `ExternalChildInputs` bundle. | `src/gabion/tooling/runtime/policy_scanner_suite.py`; `scripts/policy/policy_scanner_suite.py`; `tests/gabion/tooling/runtime_policy/test_policy_scanner_suite.py`; `tests/gabion/tooling/policy/test_policy_scanner_suite_script.py`; `docs/projection_semantic_fragment_rfc.md#projection_semantic_fragment_rfc` |
| `2026-03-11` | The policy-scanner-suite wrapper no longer materializes a temporary `dict[rule_id, payload]` child-result rendezvous during ingress resolution; preserved and newly emitted child artifacts are normalized directly into `PolicySuiteChildInputs` at the wrapper boundary. | `scripts/policy/policy_scanner_suite.py`; `tests/gabion/tooling/policy/test_policy_scanner_suite_script.py`; `docs/projection_semantic_fragment_rfc.md#projection_semantic_fragment_rfc` |
| `2026-03-11` | The policy-scanner-suite wrapper no longer traffics raw child-result mappings after ingress validation; external child checks now resolve directly to `PolicySuiteChildInputs`, so wrapper orchestration continues only over the typed child-input bundle. | `scripts/policy/policy_scanner_suite.py`; `tests/gabion/tooling/policy/test_policy_scanner_suite_script.py`; `docs/projection_semantic_fragment_rfc.md#projection_semantic_fragment_rfc` |
| `2026-03-11` | The runtime policy-scanner-suite module no longer exposes a raw child-result parser; `PolicySuiteChildInputs` is now a pure typed carrier, and raw child-payload normalization lives only in the wrapper boundary helper. | `src/gabion/tooling/runtime/policy_scanner_suite.py`; `scripts/policy/policy_scanner_suite.py`; `tests/gabion/tooling/runtime_policy/test_policy_scanner_suite.py`; `tests/gabion/tooling/policy/test_policy_scanner_suite_script.py`; `docs/projection_semantic_fragment_rfc.md#projection_semantic_fragment_rfc` |
| `2026-03-11` | The outward-facing `PolicySuiteResult` carrier and payload no longer project cache identity hashes (`inventory_hash`, `rule_set_hash`); those now remain artifact-only metadata used for cache validation rather than public reporting surface. | `src/gabion/tooling/runtime/policy_scanner_suite.py`; `tests/gabion/tooling/runtime_policy/test_policy_scanner_suite.py`; `tests/test_policy_dsl.py`; `docs/projection_semantic_fragment_rfc.md#projection_semantic_fragment_rfc` |
| `2026-03-11` | Projection history artifacts now classify registered specs by semantic/presentation/bridge lowering so the `ProjectionSpec` split is visible in ledger form. | `scripts/policy/build_projection_spec_history.py`; `artifacts/out/projection_spec_history_ledger.json`; `docs/audits/projection_spec_history_ledger.md`; `tests/gabion/tooling/policy/test_build_projection_spec_history.py` |

## Queue Rows

| queue_id | phase | status | surface | evidence_links | next_action |
| --- | --- | --- | --- | --- | --- |
| `PSF-001` | `Phase 2` | `landed` | Carrier-first reflection over ASPF/fibration witnesses | `src/gabion/analysis/projection/semantic_fragment.py`; `src/gabion/tooling/policy_substrate/lattice_convergence_semantic.py` | Preserve structural/site identity continuity as more semantic surfaces are promoted. |
| `PSF-002` | `Phase 3` | `landed` | Deterministic SHACL/SPARQL lowering for declared quotient faces | `src/gabion/analysis/projection/projection_semantic_lowering.py`; `src/gabion/analysis/projection/projection_semantic_lowering_compile.py`; `src/gabion/analysis/projection/semantic_fragment_compile.py` | Keep lowerings typed and deterministic; do not allow freehand query/shape authoring in the semantic core. |
| `PSF-003` | `Phase 4` | `landed` | Reporting-layer propagation of canonical semantic previews | `scripts/policy/policy_check.py`; `src/gabion/tooling/runtime/policy_scanner_suite.py`; `scripts/policy/hotspot_neighborhood_queue.py`; `scripts/policy/projection_semantic_fragment_queue.py` | Use preview propagation as the continuity surface while direct carrier consumers are added. |
| `PSF-004` | `Phase 4` | `in_progress` | Friendly-surface convergence via typed `ProjectionSpec` lowering | `src/gabion/analysis/projection/projection_registry.py`; `scripts/policy/build_projection_spec_history.py`; `artifacts/out/projection_spec_history_ledger.json`; `src/gabion/analysis/projection/projection_exec.py`; `docs/projection_semantic_fragment_rfc.md#projection_semantic_fragment_rfc` | Promote the next declared semantic op through lowering without adding semantic behavior directly to `projection_exec.py`. |
| `PSF-005` | `Phase 4` | `queued` | Expand semantic op coverage beyond declared quotient-face slices | `src/gabion/analysis/projection/semantic_fragment.py`; `docs/projection_semantic_fragment_rfc.md#projection_semantic_fragment_rfc`; `docs/ttl_kernel_semantics.md#ttl_kernel_semantics` | Add the next smallest lawful semantic op on top of the same carrier, not as a generic row transform. |
| `PSF-006` | `Phase 4` | `queued` | Move policy and authoring consumers toward direct canonical-carrier judgment | `scripts/policy/policy_check.py`; `src/gabion/tooling/runtime/policy_scanner_suite.py`; `docs/projection_semantic_fragment_rfc.md#projection_semantic_fragment_rfc` | Shift the next consumer from row-shape inference to direct carrier reads and preserve that path with policy tests. |
| `PSF-007` | `Phase 5` | `queued` | Cut over legacy adapters and retire `semantic_carrier_adapter` boundaries | `src/gabion/analysis/projection/projection_exec.py`; `src/gabion/analysis/projection/semantic_fragment.py`; `docs/projection_semantic_fragment_rfc.md#projection_semantic_fragment_rfc` | Remove temporary adapter status only after the RFC cutover criteria are met on at least one end-to-end path. |

## Ratchet Rules

- No new semantic behavior lands directly in `projection_exec.py`.
- New semantic features land as semantic ops first, presentation second.
- SHACL/SPARQL remain realizers of the semantic laws, not the source of those
  laws.
- Generated queue artifacts are the current-state source of truth; the queue is
  derived directly from `policy_check_result.json` when available, while this
  ledger records stable queue IDs, evidence links, and cutover intent.
- Slower aggregate artifacts should not carry duplicate semantic-summary
  projections once the same summary is derivable from `policy_results` or a
  faster semantic artifact; keep one canonical summary derivation path.
- Wrapper cache artifacts should not persist child-owned canonical policy
  results once those results are already explicit boundary inputs; cache only
  wrapper-owned scan output plus the hashes required to validate reuse.
- Runtime aggregate payloads should not preserve child-result nesting for
  semantic carriers once the same semantic data can be surfaced directly on
  the payload boundary.
- Slower wrappers must not regenerate continuation artifacts owned by the
  faster semantic path. Preserve or consume them, but do not re-emit them.
- Slower wrappers must consume valid preexisting child-owned canonical
  artifacts rather than rerunning the child checks that already own those
  artifacts.
- Wrappers that already hold canonical payloads in memory must pass those
  payloads directly to downstream reporting consumers rather than re-reading
  wrapper-owned artifacts from disk.
- Queue/report tools must not retain suite-era ingress names once their actual
  contract is a generic source artifact or canonical payload.
- Queue/report artifacts must not retain suite-era field names once the same
  metadata is sourced from a generic artifact or canonical payload.
- Queue/report consumers must not preserve sidecar semantic-supplement inputs
  once the canonical source payload already carries the same semantic carrier.
- Queue/report CLIs must not preserve suite-era default artifact paths once
  the input contract is intentionally generic and explicit.
- Shared semantic-summary decoders must not preserve wrapper-only ingress
  branches once the canonical artifact paths are explicit.
- Shared summary decoders must not retain retired suite-local embedding
  fallbacks once those embeddings are no longer emitted on real artifact paths.
- Wrappers must not invent surrogate policy-result carriers from child process
  return codes once those child checks own canonical artifact emission.
  Missing child artifacts are fail-closed wrapper errors, and any explicit
  skip protocol must be emitted by the owning child check rather than
  synthesized by the aggregate wrapper.
- Runtime/policy wrappers must not own canonical lattice witness serialization
  once the same payload contract exists on the semantic carrier. Consumers may
  delegate to carrier-owned payload methods, but the carrier remains the source
  of shape.

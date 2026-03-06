---
doc_revision: 18
reader_reintern: "Reader-only: re-intern if doc_revision changed since you last read this doc."
doc_id: docflow_strict_failure_analysis
doc_role: analysis
doc_scope:
  - repo
  - governance
  - docs
doc_authority: informative
doc_requires:
  - "CONTRIBUTING.md#contributing_contract"
  - "AGENTS.md#agent_obligations"
  - "docs/normative_clause_index.md#normative_clause_index"
  - src/gabion_governance/governance_audit_impl.py
doc_reviewed_as_of:
  "CONTRIBUTING.md#contributing_contract": 2
  "AGENTS.md#agent_obligations": 30
  "docs/normative_clause_index.md#normative_clause_index": 2
doc_review_notes:
  "CONTRIBUTING.md#contributing_contract": "Strict docflow review discipline and correction cadence reviewed."
  "AGENTS.md#agent_obligations": "Governance-relevant in/ drift treated as blocking signal; warning remediation executed as correction units."
  "docs/normative_clause_index.md#normative_clause_index": "Clause index reviewed for boundary-first ambiguity and lifecycle obligations."
doc_change_protocol: "POLICY_SEED.md#change_protocol"
doc_owner: maintainer
---

<a id="docflow_strict_failure_analysis"></a>
# Strict Docflow Failure Analysis (Observation-Only)

## Scope
This document captures a deep, recursive breakdown of current strict docflow failures without remediation changes.

Run basis:
- Command: `mise exec -- env PYTHONPATH=. python -m gabion docflow --root . --fail-on-violations --sppf-gh-ref-mode required`
- Date: 2026-03-06
- Source of truth for counts: `_docflow_audit_context(...)` + emitted `artifacts/out/docflow_*.json`.

## L0 Summary
- Violations: `106`
- Warnings: `29`
- Compliance summary (`artifacts/out/docflow_compliance.json`):
  - `contradicts=106`
  - `compliant=11`
  - `excess=5`

## L1 Taxonomy
### Violations
- `docflow:missing_explicit_reference`: `78`
- `docflow:review_pin_mismatch`: `21`
- `docflow:invalid_field_type`: `4`
- `docflow:missing_governance_ref`: `3`

### Warnings
- `section_review_stale_dep`: `17`
- `section_review_missing_review`: `3`
- `implicit_reference_tier2`: `9`

## L2 Recursive Breakdown
## 1) `missing_explicit_reference` (78)
Primary hotspot chain:
- `in/in-46.md` .. `in/in-58.md`
- strongest concentration in `in/in-56.md`, `in/in-57.md`, `in/in-58.md`, `in/in-53.md`

Pattern:
- Frontmatter `doc_requires` declares dependencies, but body text lacks required explicit link occurrences.
- Implicit references appear in warnings, especially for `in/in-56..58`, indicating inferred coupling instead of explicit citation.

Interpretation:
- These docs encode dependency intent in metadata but under-realize dependency explicitness in prose/anchors.
- Governance intent favors explicit traceability over inferred/implicit linkage.

## 2) `review_pin_mismatch` (21)
Hotspot docs:
- `in/in-32.md` (5 mismatches)
- `in/in-46.md` (3)
- `in/in-53.md` (3)
- several others in `in/in-47..52`

Pattern:
- `doc_reviewed_as_of[...]` versions lag current dependency revisions (for example `POLICY_SEED.md#policy_seed: 1 -> expected 2`).

Interpretation:
- Review attestation lifecycle is stale versus current governance document revisions.
- This is semantic confidence debt: declared dependency review state no longer certifies current dependency semantics.

## 3) `invalid_field_type` (4)
Files:
- `glossary.md`
- `docs/publishing_practices.md`

Pattern:
- `doc_reviewed_as_of` and `doc_review_notes` parse as null instead of map.

Interpretation:
- Root governance docs are carrying schema-shape drift in review metadata fields.
- This weakens docflow’s ability to verify review provenance at the governance root.

## 4) `missing_governance_ref` (3)
File:
- `in/in-33.md`

Pattern:
- Required governance roots (`AGENTS.md`, `CONTRIBUTING.md`, `README.md`) are absent from required-reference surface.

Interpretation:
- PatternSchema framing is linked to policy/glossary lineage, but governance-root coupling is incomplete.

## L3 Design-Intent vs Implementation Signals (Observation)
## A) Quotient/graded protocol doc chain references artifacts not present in repo outputs
Referenced by `in/in-54..58` but not found under `src/`, `scripts/`, or `tests` symbol/search surface:
- `out/quotient_axis_registry.json`
- `out/quotient_profile_catalog.json`
- `out/quotient_class_schema.json`
- `out/quotient_assignment.json`
- `out/quotient_basis_candidates.json`
- `out/quotient_diagnostic_candidates.json`
- `out/quotient_unresolved_ambiguities.json`
- `out/graded_execution_budget_profile.json`
- `out/graded_route_telemetry.json`
- `out/graded_trigger_map.json`
- `out/quotient_governance_report.json`
- `out/quotient_policy_violations.json`
- `out/quotient_ratchet_delta.json`

Interpretation:
- The in-step design intent is richer than currently discoverable implementation outputs for this chain.
- This is a candidate implementation-alignment gap, not yet a prescribed fix.

## B) A narrower quotient lifecycle implementation exists
Observed implementation/output surface in orchestrator/taint lifecycle:
- `quotient_protocol_readiness.json`
- `quotient_promotion_decision.json`
- `quotient_demotion_incidents.json`

Interpretation:
- There is partial implementation of quotient-lifecycle semantics, but not the full artifact family declared in `in/in-54..58`.
- This may indicate staged rollout or partially landed design intent.

## L4 Deep Slice Results (No Remediation)
Generated machine-readable artifacts:
- `artifacts/out/docflow_failure_dependency_matrix.json`
- `artifacts/out/docflow_declared_artifact_implementation_map.json`

### 1) Dependency-closure matrix for `in/in-46..58`
`doc_requires_ref` totals in this chain:
- total edges: `85`
- explicit references: `7`
- implicit-only references: `9`
- missing references: `69`

Per-document edge coverage:

| Path | Total refs | Explicit | Implicit-only | Missing |
| --- | ---:| ---:| ---:| ---:|
| `in/in-46.md` | 6 | 1 | 0 | 5 |
| `in/in-47.md` | 6 | 0 | 0 | 6 |
| `in/in-48.md` | 5 | 0 | 0 | 5 |
| `in/in-49.md` | 5 | 0 | 0 | 5 |
| `in/in-50.md` | 4 | 0 | 0 | 4 |
| `in/in-51.md` | 5 | 0 | 0 | 5 |
| `in/in-52.md` | 5 | 0 | 0 | 5 |
| `in/in-53.md` | 7 | 0 | 0 | 7 |
| `in/in-54.md` | 7 | 4 | 0 | 3 |
| `in/in-55.md` | 7 | 1 | 0 | 6 |
| `in/in-56.md` | 8 | 0 | 2 | 6 |
| `in/in-57.md` | 9 | 0 | 3 | 6 |
| `in/in-58.md` | 11 | 1 | 4 | 6 |

Dependency-type decomposition of the same edges:
- governance/glossary refs: `explicit=3`, `implicit-only=0`, `missing=48`
- in-chain refs (`in/in-*.md#...`): `explicit=4`, `implicit-only=9`, `missing=21`

Interpretation:
- Governance roots are mostly absent as explicit citations in this chain (`48` missing governance/glossary refs).
- In-chain coupling is often present only implicitly (`9` implicit-only), especially `in/in-56..58`.
- `in/in-54.md` is the only document in this chain with substantial explicit coverage.

### 2) Review-pin drift table (`doc_review_pin`) for `in/in-46..58`
Pin totals in this chain:
- total pins: `43`
- matches: `27`
- mismatches: `16`
- unresolved refs: `0`

Per-document pin status:

| Path | Total pins | Matches | Mismatches |
| --- | ---:| ---:| ---:|
| `in/in-46.md` | 6 | 3 | 3 |
| `in/in-47.md` | 6 | 5 | 1 |
| `in/in-48.md` | 5 | 3 | 2 |
| `in/in-49.md` | 5 | 3 | 2 |
| `in/in-50.md` | 4 | 3 | 1 |
| `in/in-51.md` | 5 | 3 | 2 |
| `in/in-52.md` | 5 | 3 | 2 |
| `in/in-53.md` | 7 | 4 | 3 |

Mismatch drift details (declared -> expected, delta):
- `in/in-46.md`:
  - `CONTRIBUTING.md#contributing_contract`: `1 -> 2` (`-1`)
  - `POLICY_SEED.md#policy_seed`: `1 -> 2` (`-1`)
  - `in/in-30.md#in_in_30`: `1 -> 4` (`-3`)
- `in/in-47.md`:
  - `POLICY_SEED.md#policy_seed`: `1 -> 2` (`-1`)
- `in/in-48.md`:
  - `CONTRIBUTING.md#contributing_contract`: `1 -> 2` (`-1`)
  - `POLICY_SEED.md#policy_seed`: `1 -> 2` (`-1`)
- `in/in-49.md`:
  - `CONTRIBUTING.md#contributing_contract`: `1 -> 2` (`-1`)
  - `POLICY_SEED.md#policy_seed`: `1 -> 2` (`-1`)
- `in/in-50.md`:
  - `POLICY_SEED.md#policy_seed`: `1 -> 2` (`-1`)
- `in/in-51.md`:
  - `CONTRIBUTING.md#contributing_contract`: `1 -> 2` (`-1`)
  - `POLICY_SEED.md#policy_seed`: `1 -> 2` (`-1`)
- `in/in-52.md`:
  - `CONTRIBUTING.md#contributing_contract`: `1 -> 2` (`-1`)
  - `POLICY_SEED.md#policy_seed`: `1 -> 2` (`-1`)
- `in/in-53.md`:
  - `CONTRIBUTING.md#contributing_contract`: `1 -> 2` (`-1`)
  - `POLICY_SEED.md#policy_seed`: `1 -> 2` (`-1`)
  - `README.md#repo_contract`: `1 -> 2` (`-1`)

Interpretation:
- Drift is concentrated on a small set of governance roots, usually lagging by one revision.
- One deeper lag (`in/in-30.md#in_in_30: 1 -> 4`) indicates an older stale anchor-level review claim, not just recent governance-version lag.

### 3) Declared-artifact vs implementation map (`in/in-54..58`)
From `artifacts/out/docflow_declared_artifact_implementation_map.json`:
- declared artifacts in chain: `16`
- implemented literal output surfaces found in code/tests: `3`
  - `out/quotient_protocol_readiness.json`
  - `out/quotient_promotion_decision.json`
  - `out/quotient_demotion_incidents.json`
- declared-only surfaces (no literal producer/consumer hit in `src/`, `scripts/`, `tests`): `13`

Interpretation:
- The chain describes a broader quotient/graded artifact contract than what is currently discoverable by literal output-surface scan.
- The current implementation appears to cover lifecycle decision artifacts, but not the full declared modeling/migration/governance artifacts in `in/in-54..58`.

## L5 Edge-by-Edge Semantic Intent Notes (No Remediation)
Generated machine-readable artifact:
- `artifacts/out/docflow_missing_explicit_ref_intent_notes.json`

Coverage:
- analyzed missing explicit-reference edges in `in/in-46.md` .. `in/in-58.md`: `69`
- semantic note rows emitted: `69` (one note per missing edge)

Dependency-class distribution:
- `governance_root`: `33`
- `glossary_anchor`: `15`
- `in_chain_dependency`: `17`
- `template_protocol_dependency`: `4`

Citation-absence hypothesis distribution:
- `frontmatter_only_governance_attestation`: `33`
- `term_level_reference_without_canonical_anchor`: `15`
- `shorthand_in_step_reference_without_path_anchor`: `17`
- `template_dependency_not_reified_in_prose`: `4`

Body-signal coverage across missing edges:
- missing edges with lexical body signals of dependency intent: `35`
- missing edges without lexical body signals: `34`

Meaning-level interpretation:
- Governance-root edges are mostly frontmatter-attested but not prose-cited. This is strong metadata intent with weak explicit narrative traceability.
- Glossary-anchor edges often have domain vocabulary in body text (for example contract/ambiguity/erasure terms), but lack canonical anchor references; this indicates semantic alignment without explicit citation discipline.
- In-chain dependencies are commonly referenced by step shorthand or sequence implication, not explicit `path#anchor` mentions; this preserves reader continuity but fails strict traceability invariants.
- Template protocol dependency (`in/in-28.md#in_in_28`) is consistently absent from body text in `in/in-55..58`, suggesting protocol inheritance is assumed rather than reified in prose.

Representative examples (observation only):
- `in/in-46.md -> glossary.md#contract`: body signals include `ambiguity`/`erasure`, but canonical glossary anchor is not explicit.
- `in/in-54.md -> AGENTS.md#agent_obligations`: policy/obligation language appears in body, but no explicit anchor citation.
- `in/in-58.md -> in/in-28.md#in_in_28`: no lexical signal found; dependency remains frontmatter-only for the template protocol anchor.

## L6 Review-Pin Chronology (No Remediation)
Generated machine-readable artifact:
- `artifacts/out/docflow_review_pin_chronology.json`

Chronology summary:
- total pin mismatches analyzed: `16`
- `born_stale`: `1`
- `became_stale_after_dependency_bump`: `15`
- `unknown_timing`: `0`

Chronology facts (derived from `git log` + mismatch rows):
- `in/in-46.md` .. `in/in-53.md` were introduced on `2026-02-23` (`310ffe3`) with `doc_reviewed_as_of` pins at `1` for governance refs.
- `CONTRIBUTING.md#contributing_contract` and `POLICY_SEED.md#policy_seed` moved to section version `2` on `2026-02-27` (`bd4aa27`).
- `README.md#repo_contract` moved to section version `2` on `2026-02-27` (`d03ba75`).
- `in/in-30.md#in_in_30` moved to section version `4` on `2026-02-11` (`80edbc2`), before `in/in-46.md` existed.

Mismatch episodes by dependency:

| Dependency | Mismatches | Seen -> Expected | Timing class |
| --- | ---:| ---:| --- |
| `POLICY_SEED.md#policy_seed` | 8 | `1 -> 2` | became stale after dependency bump |
| `CONTRIBUTING.md#contributing_contract` | 6 | `1 -> 2` | became stale after dependency bump |
| `README.md#repo_contract` | 1 | `1 -> 2` | became stale after dependency bump |
| `in/in-30.md#in_in_30` | 1 | `1 -> 4` | born stale |

Meaning-level interpretation:
- Most pin drift in this chain is **event drift**: documents started aligned to then-current section pins, then dependency section versions advanced while these docs remained pinned at `1`.
- One mismatch is **origin drift** (`in/in-46 -> in/in-30`): the pin was already stale at document birth (expected revision predated source introduction by 12 days).
- This split matters for remediation priority because born-stale edges indicate copy-forward/template debt, while event-drift edges indicate review cadence lag.

## L7 Declared-Only Artifact Non-Literal Map (No Remediation)
Generated machine-readable artifact:
- `artifacts/out/docflow_declared_artifact_non_literal_map.json`

Scope:
- target set: `13` declared-only artifacts from `in/in-54..58` with no literal producer/consumer hits.
- method: semantic-proxy mapping to existing runtime payloads/types/reports, with explicit confidence and mismatch notes.

Aggregate mapping shape:
- declared-only analyzed: `13`
- `partial_proxy`: `9`
- `weak_proxy`: `4`
- confidence: `high=3`, `medium=6`, `low=4`

Notable high-confidence partial proxies:
- `out/quotient_profile_catalog.json`:
  - proxy: taint profile lifecycle payloads (`profiles` rows in readiness carrier).
  - caveat: profile domain is taint lifecycle, not quotient-equivalence relation catalogs.
- `out/quotient_unresolved_ambiguities.json`:
  - proxy: `strict_unresolved` in taint state/delta payloads.
  - caveat: no dedicated quotient assignment unresolved ledger artifact.
- `out/quotient_ratchet_delta.json`:
  - proxy: `taint_delta` baseline/current/delta structure.
  - caveat: ratchet domain is taint projection namespace, not quotient artifact namespace.

Weak-proxy zone (highest intent-vs-implementation risk):
- `out/quotient_basis_candidates.json`
- `out/quotient_diagnostic_candidates.json`
- `out/graded_execution_budget_profile.json`
- `out/graded_route_telemetry.json`

Meaning-level interpretation:
- The repo has meaningful **semantic precursors** for many declared artifacts, but they are mostly emitted under taint/projection/governance payload surfaces rather than the declared quotient/graded artifact contract names.
- This indicates an implementation path that partially overlaps the declared intent while leaving a naming/contract boundary gap unresolved.
- No declared-only artifact in this slice had a full non-literal one-to-one contract replacement with matching domain and schema obligations.

## L8 Weak-Proxy Field Contract Deltas (No Remediation)
Generated machine-readable artifact:
- `artifacts/out/docflow_weak_proxy_field_deltas.json`

Scope:
- weak-proxy set analyzed: `4` artifacts
  - `out/quotient_basis_candidates.json`
  - `out/quotient_diagnostic_candidates.json`
  - `out/graded_execution_budget_profile.json`
  - `out/graded_route_telemetry.json`

Delta classes:
- `contract_identity_gap`: `1`
- `diagnostic_goal_gap`: `1`
- `execution_policy_shape_gap`: `1`
- `routing_observability_gap`: `1`

Required-field gap counts:
- `out/quotient_basis_candidates.json`: missing `2` required fields
- `out/quotient_diagnostic_candidates.json`: missing `2` required fields
- `out/graded_execution_budget_profile.json`: missing `3` required fields
- `out/graded_route_telemetry.json`: missing `1` required field

Representative field-level findings:
- `out/quotient_basis_candidates.json` vs `call_clusters.json` proxy:
  - partial correspondence on test lists (`representative_tests` vs `tests`)
  - missing explicit `class_id` and `minimal_cover_intent` contract carriers.
- `out/quotient_diagnostic_candidates.json` vs `test_evidence_suggestions.json` proxy:
  - proxy has suggestion/test evidence signals but no quotient `class_id` and no deterministic localization-ranking field.
- `out/graded_execution_budget_profile.json` vs `deadline_profile.json` proxy:
  - timing/budget telemetry exists, but no profile-indexed fan-out limits or escalation-rule fields.
- `out/graded_route_telemetry.json` vs phase-progress timeline proxy:
  - progress-event telemetry exists, but explicit `selected_routes` carrier is absent.

Meaning-level interpretation:
- Weak proxies are not merely naming mismatches; each misses at least one schema-bearing field that anchors the intended decision protocol.
- The largest gap sits in graded execution policy shape (`profile/fan-out/escalation`) where current telemetry is rich but policy-structural fields are absent.

## L9 Policy-Impact Matrix for Weak/Missing Surfaces (No Remediation)
Generated machine-readable artifact:
- `artifacts/out/docflow_policy_impact_matrix.json`

Coverage:
- artifacts analyzed: `13` (all current partial/weak missing-literal surfaces from `in/in-54..58`)

Matrix summary:
- layer distribution:
  - `model`: `3`
  - `migration`: `4`
  - `execution`: `3`
  - `governance`: `3`
- sensitivity distribution:
  - `high`: `10`
  - `medium`: `3`
- downstream decision-surface coverage:
  - `promotion_gate`: `10`
  - `promotion_decision`: `6`
  - `demotion_trigger`: `6`
  - `ratchet_enforcement`: `10`

Policy/decision interpretation:
- Model-surface gaps primarily weaken **promotion gate basis quality** (`F54-*`, `F58-1`), creating risk of false-green readiness semantics.
- Migration-surface gaps primarily weaken **assignment-debt ratchet interpretability** (`F55-*`, `F57-2`, `F58-1`), especially where unresolved/assignment artifacts are missing.
- Execution-surface gaps primarily weaken **demotion and quality-floor evidence** (`F56-*`, `F57-3`, `F58-1`), most notably in budget and route telemetry contracts.
- Governance-surface gaps directly weaken **machine-readable lifecycle auditability** (`F57-4`, `F58-2`, `F58-4`) because these artifacts are closest to explicit promote/demote rationale carriers.

## L10 High-Sensitivity Artifact Triage Packets (No Remediation)
Generated machine-readable artifact:
- `artifacts/out/docflow_high_sensitivity_triage_packets.json`

Coverage:
- high-sensitivity artifacts packetized: `10`
- layer split:
  - `model`: `3`
  - `migration`: `2`
  - `execution`: `2`
  - `governance`: `3`

Packet contents per artifact:
- declared-contract references (`in/in-54..58` artifact schemas + obligations),
- proxy-contract summary (semantic overlap + mismatch notes from non-literal map),
- downstream decision surfaces touched (promotion/demotion/ratchet),
- minimum viable remediation boundary (boundary scope only; no core semantic rewrite implied).

Meaning-level interpretation:
- High-sensitivity set is not limited to weak proxies; several partial proxies still touch promotion/demotion-critical surfaces.
- Boundary-first remediation slicing is feasible because packet scopes align cleanly with layer owners (`projection`, `migration/report emitters`, `execution routing`, `lifecycle governance`).

## L11 Staged Contradiction-Reduction Simulation (No Remediation)
Generated machine-readable artifact:
- `artifacts/out/docflow_remediation_ordering_simulation.json`

Simulation objective:
- minimize strict-docflow contradiction count while minimizing compatibility-thrash risk.

Baseline:
- strict contradictions: `106`

Recommended sequence (bounded-risk, 4 units):
1. `S1` chain governance/glossary explicit-reference normalization (`-48`) → `58`.
2. `S2` chain in-step explicit-reference normalization (`-21`) → `37`.
3. `S3` chain review-pin realignment (`-16`) → `21`.
4. `S4` residual repo-level contradiction cleanup (`-21`) → `0`.

Alternate (aggressive, 3 units):
- fewer units but concentrates residual cleanup into one high-thrash unit.

Meaning-level interpretation:
- The 4-step sequence is the smallest staged plan that keeps each correction unit semantically coherent and avoids a single high-blast-radius residual cleanup.
- Sequence naturally front-loads citation/pin contradictions before broader residue, which lowers rollback pressure and preserves evidence clarity.

## L12 Per-Step Evidence Prerequisites for S1..S4 (No Remediation)
Generated machine-readable artifact:
- `artifacts/out/docflow_step_evidence_prerequisites.json`

Scope:
- objective: pre-bind validation/evidence obligations to each simulated step (`S1..S4`) so remediation units remain bounded and auditable.
- basis artifacts:
  - `artifacts/out/docflow_remediation_ordering_simulation.json`
  - `artifacts/out/docflow_compliance.json`
  - `artifacts/out/docflow_missing_explicit_ref_intent_notes.json`
  - `artifacts/out/docflow_review_pin_chronology.json`

Common prerequisite stack (all steps):
- policy:
  - `scripts/policy/policy_check.py --workflows`
  - `scripts/policy/policy_check.py --ambiguity-contract`
  - strict docflow (`python -m gabion docflow --fail-on-violations --sppf-gh-ref-mode required`)
- tests catalog:
  - `T-DOCFLOW-CORE`: compliance/formatter/sppf-refs/governance-adapter suites
  - `T-DOCFLOW-CONTROL`: control-loop/warnings/implication-matrix/status-triplet suites
  - `T-NORMATIVE-SYMDIFF`: normative symdiff suites
- artifacts:
  - `artifacts/out/docflow_compliance.json`
  - `artifacts/out/docflow_section_reviews.json`
  - `artifacts/out/docflow_implication_matrices.json`

Step bindings:
- `S1` (governance/glossary explicit refs): target `48`; required tests `T-DOCFLOW-CORE`.
- `S2` (in-chain/template explicit refs): target `21`; required tests `T-DOCFLOW-CORE` + `T-DOCFLOW-CONTROL`.
- `S3` (chain review-pin realignment): target `16`; required tests `T-DOCFLOW-CORE` + `T-DOCFLOW-CONTROL`.
- `S4` (residual cleanup): target `21`; required tests `T-DOCFLOW-CORE` + `T-DOCFLOW-CONTROL` + `T-NORMATIVE-SYMDIFF`.

Critical guard discovered:
- Simulation assigns `9` residual `missing_explicit_reference` contradictions to `S4`.
- Current observed residual path scan in the prerequisites artifact reports no non-chain paths for that invariant.
- Therefore `S4` requires a mandatory pre-edit partition recomputation before any remediation unit is scoped.

Meaning-level interpretation:
- The prerequisites convert the staged simulation into correction-unit contracts that are test/policy/evidence complete by construction.
- The explicit `S4` partition guard prevents executing an invalid residual scope based on stale simulation assumptions.

## L13 Contradiction-to-Owner Routing for S1..S4 (No Remediation)
Generated machine-readable artifact:
- `artifacts/out/docflow_contradiction_owner_routing.json`

Scope:
- objective: pre-assign each contradiction slice to an owner triplet and concrete boundary (`artifact/module/doc`) so remediation units can be opened without ownership ambiguity.
- source coupling:
  - `artifacts/out/docflow_remediation_ordering_simulation.json`
  - `artifacts/out/docflow_step_evidence_prerequisites.json`

Routing model:
- owner triplet:
  - `doc_owner`: resolved from target-doc frontmatter.
  - `module_owner`: defaulted to `maintainer (assumed; CODEOWNERS absent)`.
  - `artifact_owner`: `governance tooling maintainer`.
- correction-unit boundaries are step-aligned (`S1..S4`) and carry contradiction partition + expected count.

Step routing summary:
- `S1`: `missing_explicit_reference` (`governance_root + glossary_anchor`, `48`) routed to `in/in-46..58`, governance audit/policy modules, and docflow artifacts.
- `S2`: `missing_explicit_reference` (`in_chain_dependency + template_protocol_dependency`, `21`) routed to same chain/module/artifact boundary.
- `S3`: `review_pin_mismatch` (`16`) routed to `in/in-46..53` metadata boundary with same module/artifact boundary.
- `S4`: residual mixed-class (`9+5+4+3`) routed to:
  - docs: `docs/publishing_practices.md`, `glossary.md`, `in/in-32.md`, `in/in-33.md`
  - modules: governance audit + policy check + governance adapter
  - artifacts: compliance/section-reviews/implication-matrices.

Observed ownership fact:
- All currently routed docs resolve to `doc_owner: maintainer`.

Meaning-level interpretation:
- Ownership ambiguity is effectively eliminated for the simulated sequence; each step now has a pre-declared correction-unit envelope.
- `S4` remains the only mixed-class boundary and keeps an explicit split-if-churn guard due to cross-document residual risk.

## L14 Recomputed Residual Partition Snapshot Post-L12/L13 (No Remediation)
Generated machine-readable artifact:
- `artifacts/out/docflow_residual_partition_snapshot.json`

Scope:
- objective: validate whether `S4` still legitimately includes `9` residual `missing_explicit_reference` contradictions after introducing L12/L13 constraints.
- source artifacts:
  - `artifacts/out/docflow_compliance.json` (fresh strict-docflow run)
  - `artifacts/out/docflow_missing_explicit_ref_intent_notes.json`
  - `artifacts/out/docflow_review_pin_chronology.json`
  - `artifacts/out/docflow_remediation_ordering_simulation.json`

Recomputed baseline (current):
- contradictions: `106`
- by invariant:
  - `docflow:missing_explicit_reference`: `78`
  - `docflow:review_pin_mismatch`: `21`
  - `docflow:invalid_field_type`: `4`
  - `docflow:missing_governance_ref`: `3`

Step-target reconstruction checks:
- `S1` target from intent notes (`governance_root + glossary_anchor`): `48` (matches simulation).
- `S2` target from intent notes (`in_chain_dependency + template_protocol_dependency`): `21` (matches simulation).
- `S3` target from current/chronology (`in/in-46..53` review pins): `16` (matches simulation).

Residual recompute after `S1+S2+S3` partitioning:
- `missing_explicit_reference` residual: `9` (count matches simulation), but all `9` are chain-local:
  - `in/in-56.md`: `2`
  - `in/in-57.md`: `3`
  - `in/in-58.md`: `4`
- non-chain residual `missing_explicit_reference`: `0`
- `review_pin_mismatch` residual: `5` (all in `in/in-32.md`; matches simulation count).
- `invalid_field_type` residual: `4` (matches simulation).
- `missing_governance_ref` residual: `3` (matches simulation).

`S4` legitimacy verdict:
- `count_legitimate`: `true`
- `scope_legitimate`: `false`
- verdict: `partially_valid_count_invalid_scope`

Meaning-level interpretation:
- The `S4` numeric assumption (`9`) is still correct, but its scope is not: those `9` rows are not repo-residual off-chain contradictions; they are in-chain implicit-only edges remaining after S1/S2 closure.
- Therefore `S4` should be re-scoped before remediation: treat these `9` as final in-chain explicit-reference closure, not as cross-document residual cleanup.

## L15 Per-Step Contradiction-to-Test Mapping (No Remediation)
Generated machine-readable artifact:
- `artifacts/out/docflow_step_contradiction_test_map.json`

Scope:
- objective: bind each step/contradiction class to the minimal proving tests needed to contain correction-unit blast radius before remediation starts.

Global proving sensor (all classes):
- `PC-STRICT-DOCFLOW`:
  - `mise exec -- env PYTHONPATH=. python -m gabion docflow --root . --fail-on-violations --sppf-gh-ref-mode required`

Per-step minimal proving map:
- `S1` (`missing_explicit_reference`, governance/glossary partition):
  - `PC-STRICT-DOCFLOW`
  - `UT-DOCFLOW-COMPLIANCE-DISPATCH`
    - `tests/gabion/tooling/docflow/test_docflow_compliance_rows.py::test_docflow_compliance_rows_dispatches_cover_never_require_active_and_proposed`
- `S2` (`missing_explicit_reference`, in-chain/template partition):
  - `PC-STRICT-DOCFLOW`
  - `UT-DOCFLOW-WARNING-FAILURE-GATE`
    - `tests/gabion/tooling/docflow/test_docflow_warning_failures.py::test_docflow_violations_fail_when_fail_on_violations_enabled`
- `S3` (`review_pin_mismatch`):
  - `PC-STRICT-DOCFLOW`
  - `UT-DOCFLOW-REVIEW-PIN-FORMAT`
    - `tests/gabion/tooling/docflow/test_docflow_violation_formatter.py::test_format_docflow_violation_doc_review_pin_branches`
- `S4` (residual mixed-class set):
  - `PC-STRICT-DOCFLOW` for all residual contradiction classes
  - `UT-DOCFLOW-REVIEW-PIN-FORMAT` for review-pin branch semantics
  - `UT-DOCFLOW-COMPLIANCE-DISPATCH` for require/never compliance dispatch stability

Coverage-gap observation (explicit):
- Existing unit tests do not currently provide class-specific, real-doc fixtures for `missing_explicit_reference`, `invalid_field_type`, or `missing_governance_ref` row generation.
- As of this slice, class-level proof is primarily from strict docflow sensor output plus generic dispatch/formatter unit guarantees.

Meaning-level interpretation:
- The mapping now provides a minimal, enforceable proof surface for each step and contradiction class.
- It also makes current class-specific unit-test blind spots explicit before remediation, reducing false confidence from broad-but-generic test greens.

## L16 Class-Specific Fixture-Gap Inventory (No Remediation)
Generated machine-readable artifact:
- `artifacts/out/docflow_fixture_gap_inventory.json`

Scope:
- objective: inventory fixture-level testing gaps for contradiction classes flagged in L15 as class-specific blind spots.
- target classes:
  - `docflow:missing_explicit_reference` (`source_row_kind=doc_requires_ref`)
  - `docflow:invalid_field_type` (`source_row_kind=doc_field_type`)
  - `docflow:missing_governance_ref` (`source_row_kind=doc_missing_governance_ref`)

Current contradiction footprint (from strict snapshot):
- `missing_explicit_reference`: `78`
- `invalid_field_type`: `4`
- `missing_governance_ref`: `3`

Coverage finding:
- class-specific fixture tests currently present for all three classes: `none`.
- available tests are generic support surfaces (compliance dispatch / warning-gate / formatter), not class-specific real-doc row generators.

Candidate fixture docs (concrete, no implementation applied):
- `missing_explicit_reference`:
  - `fx_missing_explicit_reference_minimal`
  - `fx_missing_explicit_reference_implicit_only`
- `invalid_field_type`:
  - `fx_invalid_field_type_reviewed_as_of_null`
  - `fx_invalid_field_type_review_notes_scalar`
- `missing_governance_ref`:
  - `fx_missing_governance_ref_normative_no_roots`
  - `fx_missing_governance_ref_partial_roots`

Each candidate includes in artifact:
- synthetic doc path,
- minimal frontmatter/body contract,
- expected contradiction row(s) (row kind + invariant + status),
- class-specific assertion target.

Meaning-level interpretation:
- Fixture gaps are now explicitly reified into concrete synthetic doc specs, reducing ambiguity for follow-up test hardening.
- The gap priority is evidence-based: explicit-reference first (largest surface), then field-type schema roots, then governance-ref closure.

## L17 Corrected S4 Split Simulation + Blast Radius Quantification (No Remediation)
Generated machine-readable artifact:
- `artifacts/out/docflow_s4_split_options_simulation.json`

Baseline (from L14 residual snapshot):
- residual total: `21`
  - `missing_explicit_reference`: `9` (in-chain implicit-only: `in/in-56..58`)
  - `review_pin_mismatch`: `5`
  - `invalid_field_type`: `4`
  - `missing_governance_ref`: `3`

Simulated options:
1. `S4A_then_S4B_recommended`
2. `S4B_then_S4A`
3. `S4_single_mixed`

Quantified blast-radius comparison:
- split (`S4A_then_S4B_recommended`) vs single mixed (`S4_single_mixed`):
  - `max_classes_per_unit`: `3` vs `4` (`delta=-1`)
  - `max_docs_per_unit`: `4` vs `7` (`delta=-3`)
  - `unit_count`: `2` vs `1` (`delta=+1`)
- heuristic risk score:
  - `S4A_then_S4B_recommended`: `23`
  - `S4B_then_S4A`: `23`
  - `S4_single_mixed`: `30`

Ordering interpretation:
- `S4A_then_S4B` and `S4B_then_S4A` are tied on static risk metrics.
- `S4A_then_S4B` is preferred in this analysis because it aligns with L14’s corrected scope fact: residual explicit-reference contradictions are chain-local and should be closed in-chain before cross-document schema/governance cleanup.

Meaning-level interpretation:
- Splitting S4 decreases per-unit coupling and document spread, at the cost of one additional correction unit.
- The simulation strengthens the case for bounded-risk sequencing over one-shot mixed cleanup.

## L18 Remediation-Ready Acceptance Packets for S4A/S4B (No Remediation)
Generated machine-readable artifact:
- `artifacts/out/docflow_s4_acceptance_packets.json`

Scope:
- objective: pre-bind exact contradiction rows, touch boundaries, proving tests, and fail-fast gates for corrected `S4A`/`S4B` units without applying doc edits.
- source method: row-level extraction from `_docflow_audit_context(...)` + invariant-spec matching (same predicate surface as strict docflow).

`S4A` packet (in-chain implicit-only explicit-reference closure):
- target invariant: `docflow:missing_explicit_reference`
- target row kind: `doc_requires_ref`
- exact row count: `9`
- exact doc touch set: `in/in-56.md`, `in/in-57.md`, `in/in-58.md`
- exact required-ref rows:
  - `in/in-56.md -> in/in-54.md#in_in_54`
  - `in/in-56.md -> in/in-55.md#in_in_55`
  - `in/in-57.md -> in/in-54.md#in_in_54`
  - `in/in-57.md -> in/in-55.md#in_in_55`
  - `in/in-57.md -> in/in-56.md#in_in_56`
  - `in/in-58.md -> in/in-54.md#in_in_54`
  - `in/in-58.md -> in/in-55.md#in_in_55`
  - `in/in-58.md -> in/in-56.md#in_in_56`
  - `in/in-58.md -> in/in-57.md#in_in_57`

`S4B` packet (off-chain schema + governance residual cleanup):
- target classes and exact row counts:
  - `docflow:review_pin_mismatch` (`doc_review_pin`): `5` (all in `in/in-32.md`)
  - `docflow:invalid_field_type` (`doc_field_type`): `4` (`glossary.md`, `docs/publishing_practices.md`)
  - `docflow:missing_governance_ref` (`doc_missing_governance_ref`): `3` (`in/in-33.md`)
- exact doc touch set: `in/in-32.md`, `in/in-33.md`, `glossary.md`, `docs/publishing_practices.md`

Acceptance packet contents (both packets):
- stable target row IDs for strict diff checks,
- proving-test command set,
- required policy checks,
- explicit fail-fast conditions,
- packet-level rollback condition.

Meaning-level interpretation:
- `S4A`/`S4B` are now correction-unit ready with row-identity-level acceptance contracts.
- This removes ambiguity about “what exactly must disappear” before any remediation edit starts.

## L19 Class-Specific Fixture Test Plan (Pre-Remediation CU) (No Remediation)
Generated machine-readable artifact:
- `artifacts/out/docflow_fixture_test_plan.json`

Scope:
- objective: define a dedicated test-only correction unit that lands class-specific fixture tests before contradiction remediation.
- planned correction unit: `CU-DFX-1`
- proposed module: `tests/gabion/tooling/docflow/test_docflow_class_fixture_rows.py`

Planned test inventory (from L16 fixtures):
- `DFX-MER-001` / `DFX-MER-002`: `docflow:missing_explicit_reference`
- `DFX-IFT-001` / `DFX-IFT-002`: `docflow:invalid_field_type`
- `DFX-MGR-001` / `DFX-MGR-002`: `docflow:missing_governance_ref`

Per-test plan includes:
- fixture reference (`fx_*`),
- expected row-kind/invariant assertions,
- deterministic harness strategy (`tmp_path` docs + invariant/compliance evaluation),
- explicit no-sleep policy.

Validation stack for this pre-remediation CU:
- policy:
  - `scripts/policy/policy_check.py --workflows`
  - `scripts/policy/policy_check.py --ambiguity-contract`
- tests:
  - new class-fixture module
  - existing docflow dispatch/formatter/warning-gate supports

Meaning-level interpretation:
- This plan closes the class-specific proof gap identified in L15/L16 before any contradiction-remediation edits.
- It reduces risk of “green but semantically underconstrained” remediation units.

## L20 Packet Dry-Run Readiness Verification (Read-Only, No Remediation)
Generated machine-readable artifact:
- `artifacts/out/docflow_packet_dry_run_readiness.json`

Scope:
- objective: execute the `S4A`/`S4B` proving stack in read-only mode and classify each packet as `ready`, `blocked`, or `drifted` without fixing contradictions.
- execution model:
  - run packet-required policy checks,
  - run packet proving-test commands,
  - recompute live contradiction row IDs,
  - compare live row IDs to packet target row IDs.

Dry-run summary:
- `ready`: `2`
- `blocked`: `0`
- `drifted`: `0`

Per-packet readiness:
- `S4A`: `ready` (target-row alignment: `aligned`)
- `S4B`: `ready` (target-row alignment: `aligned`)

Command-stack behavior:
- strict docflow command status: `pass_expected_nonzero`
  - interpretation: contradiction-bearing pre-remediation state is detected as expected under `--fail-on-violations`.
- required policy checks and packet proving tests executed and were classed executable in this dry-run.

Meaning-level interpretation:
- Both packets are contract-aligned and executable in pre-remediation mode.
- No packet definition drift is currently observed at row-identity level.

## L21 Execution-Sequencing Decision Memo (No Remediation)
Generated machine-readable artifact:
- `artifacts/out/docflow_execution_sequencing_decision_memo.json`

Scope:
- objective: compare `CU-DFX-1 -> S4A -> S4B` vs `S4A -> CU-DFX-1 -> S4B` with quantified risk/latency tradeoffs.
- residual baseline used: `21` contradictions (`S4A=9`, `S4B=12`).

Quantified sequence outcomes:
- `SEQ-A` (`CU-DFX-1 -> S4A -> S4B`):
  - first reduction step index: `2`
  - uninstrumented remediation steps: `0`
  - uninstrumented remediation classes: `0`
  - area under remaining-contradiction curve: `33`
  - composite risk score: `1`
- `SEQ-B` (`S4A -> CU-DFX-1 -> S4B`):
  - first reduction step index: `1`
  - uninstrumented remediation steps: `1`
  - uninstrumented remediation classes: `1`
  - area under remaining-contradiction curve: `24`
  - composite risk score: `8`

Tradeoff deltas (`SEQ-A - SEQ-B`):
- composite risk score: `-7` (A safer by this metric)
- first reduction step index: `+1` (A slower to first reduction)
- area under remaining curve: `+9` (A holds more outstanding contradictions longer)

Decision:
- default recommendation: `CU-DFX-1 -> S4A -> S4B`
- rationale: close class-specific fixture blind spots before first remediation edit; accept one-step latency to first contradiction reduction.

Meaning-level interpretation:
- The choice is explicit safety-versus-speed:
  - `SEQ-A`: lower instrumentation risk, slower first reduction.
  - `SEQ-B`: faster first reduction, but one uninstrumented remediation step.

## L22 Remediation Launch Checklist (No Remediation)
Generated machine-readable artifact:
- `artifacts/out/docflow_remediation_launch_checklist.json`

Scope:
- objective: emit an immutable launch checklist for the chosen execution sequence `CU-DFX-1 -> S4A -> S4B`.
- source coupling:
  - `artifacts/out/docflow_packet_dry_run_readiness.json`
  - `artifacts/out/docflow_execution_sequencing_decision_memo.json`
  - `artifacts/out/docflow_s4_acceptance_packets.json`
  - `artifacts/out/docflow_fixture_test_plan.json`

Checklist structure:
- selected sequence and step order pinned.
- immutable preflight gates (`4`):
  - `G1-PACKET-READINESS`
  - `G2-SEQUENCE-DECISION`
  - `G3-POLICY-STACK`
  - `G4-STRICT-DOCFLOW-SENSOR`
- per-step launch contracts (`CU-DFX-1`, `S4A`, `S4B`) with:
  - entry conditions,
  - required-before / required-after evidence carriers,
  - required command stack,
  - expected outcome.
- global fail-fast rules (drift, out-of-scope edits, new contradiction-class emergence).

Immutable gate semantics:
- packet readiness must remain `ready` + aligned before remediation starts.
- sequence changes require memo refresh (not ad-hoc reorder).
- policy stack and strict docflow sensor are mandatory proving gates for each remediation unit.

Meaning-level interpretation:
- Launch criteria are now machine-readable and step-scoped; this reduces coordination drift when moving from analysis to execution.
- The checklist hardens the boundary between “analysis ready” and “remediation authorized.”

## L23 Packet-to-Commit Templates (No Remediation)
Generated machine-readable artifact:
- `artifacts/out/docflow_packet_commit_templates.json`

Scope:
- objective: provide commit-ready templates for each step in the chosen sequence:
  - `CU-DFX-1`
  - `S4A`
  - `S4B`

Each template includes:
- commit title template,
- required artifacts,
- required changed-path boundary,
- pre-commit verification command stack,
- post-merge verification stack,
- commit-message footer evidence lines.

Template highlights:
- `CU-DFX-1`:
  - enforces test-only boundary + `out/test_evidence.json` refresh discipline.
- `S4A`:
  - enforces in-chain-only doc touch (`in/in-56..58`) and packet-row closure proof.
- `S4B`:
  - enforces off-chain doc touch (`in/in-32`, `in/in-33`, `glossary.md`, `docs/publishing_practices.md`) and residual mixed-class closure proof.

Meaning-level interpretation:
- Commit packaging is now standardized before remediation begins, reducing ad-hoc commit scope and evidence omissions.
- Templates make correction-unit integrity auditable at commit time, not only at post-hoc review.

## L24 Commit-Unit Failure Simulation + Deterministic Fallbacks (No Remediation)
Generated machine-readable artifact:
- `artifacts/out/docflow_commit_unit_failure_simulation.json`

Scope:
- objective: simulate commit-unit failure scenarios for `CU-DFX-1`, `S4A`, and `S4B` across:
  - `policy_fail`
  - `row_drift`
  - `out_of_scope_touch`
- source coupling:
  - `artifacts/out/docflow_remediation_launch_checklist.json`
  - `artifacts/out/docflow_packet_commit_templates.json`
  - `artifacts/out/docflow_s4_acceptance_packets.json`
  - `artifacts/out/docflow_packet_dry_run_readiness.json`

Simulation shape:
- commit units modeled: `3` (`CU-DFX-1`, `S4A`, `S4B`)
- scenarios per commit unit: `3`
- total scenario rows: `9`

Per-mode deterministic fallback pattern:
- `policy_fail`:
  - abort commit attempt,
  - capture failing command evidence,
  - open dedicated fix-forward correction unit,
  - rerun full pre-commit stack only after fix-forward.
- `row_drift`:
  - mark packet drifted and halt edits,
  - regenerate packet/readiness artifacts from current snapshot,
  - split fallout if new contradiction classes appear,
  - resume only when row alignment is restored.
- `out_of_scope_touch`:
  - abort commit and classify out-of-scope paths,
  - split coupled edits into separate unit (or remove noise),
  - rerun boundary and verification stack before retry.

Global fallback invariants encoded:
- fix-forward preferred over rollback-first mixed commits,
- packet/readiness regeneration is mandatory on row drift,
- one correction unit per blocking surface is maintained.

Meaning-level interpretation:
- Failure handling is now pre-compiled into deterministic action chains per step/mode.
- This reduces ambiguity and decision latency when remediation execution encounters real failures.

## L25 Remediation Execution Log Schema (No Remediation)
Generated machine-readable artifact:
- `artifacts/out/docflow_remediation_execution_log_schema.json`

Scope:
- objective: define a canonical logging schema for live remediation correction units.
- required semantic fields:
  - `step`
  - `gate_results`
  - `row_deltas`
  - `artifact_hashes`
  - `decision`

Schema highlights:
- JSON Schema draft 2020-12 with strict required keys and enum-constrained outcomes/statuses.
- top-level required keys:
  - `log_version`, `run_id`, `sequence_id`, `started_at`, `actor`, `steps`
- step-level required keys:
  - `step`, `started_at`, `gate_results`, `row_deltas`, `artifact_hashes`, `decision`
- decision outcome enum:
  - `proceed`, `halt`, `split`, `rollback`
- command status enum includes strict-docflow expected-nonzero pre-remediation semantics:
  - `pass`, `fail`, `pass_expected_nonzero`, `pass_unexpected_zero`

Included alongside schema:
- canonical record template with placeholder fields,
- field-intent map for each semantic field to reduce logging interpretation drift.

Meaning-level interpretation:
- Live remediation now has a reproducible, machine-validated evidence log contract.
- The schema ties gate outcomes, row-delta evidence, artifact integrity, and control decisions into one auditable carrier.

## L26 CU-DFX-1 Executed (Remediation)
Implemented class-specific fixture coverage in:
- `tests/gabion/tooling/docflow/test_docflow_class_fixture_rows.py`

Validation executed:
- `scripts/policy/policy_check.py --workflows` (pass)
- `scripts/policy/policy_check.py --ambiguity-contract` (pass)
- `pytest -q tests/gabion/tooling/docflow/test_docflow_class_fixture_rows.py` (pass; 6 tests)
- `pytest -q tests/gabion/tooling/docflow/test_docflow_compliance_rows.py tests/gabion/tooling/docflow/test_docflow_violation_formatter.py tests/gabion/tooling/docflow/test_docflow_warning_failures.py` (pass)
- `scripts/misc/extract_test_evidence.py --root . --tests tests --out out/test_evidence.json` (refreshed)

Outcome:
- Class-specific proving coverage for:
  - `docflow:missing_explicit_reference`
  - `docflow:invalid_field_type`
  - `docflow:missing_governance_ref`
- `out/test_evidence.json` drift reflects six new fixture tests (expected).

## L27 S4A + S4B Executed (Remediation)
Applied packet-scoped documentation remediation:
- `S4A` touch set:
  - `in/in-56.md`
  - `in/in-57.md`
  - `in/in-58.md`
- `S4B` touch set:
  - `in/in-32.md`
  - `in/in-33.md`
  - `glossary.md`
  - `docs/publishing_practices.md`

Verification:
- policy checks (`workflows`, `ambiguity-contract`) passed
- strict docflow rerun (`--fail-on-violations`) executed as canonical sensor
- proving tests passed:
  - warning failure gate
  - review-pin formatter branch
  - compliance dispatch branch

Emitted step-result artifacts:
- `artifacts/out/docflow_remediation_step_result_S4A.json`
- `artifacts/out/docflow_remediation_step_result_S4B.json`

Measured delta:
- strict contradictions: `106 -> 67` (`-39`)
- `S4A` target rows cleared: `9 / 9`
- `S4B` target rows cleared: `12 / 12`
- post-remediation contradiction classes:
  - `docflow:missing_explicit_reference`: `51`
  - `docflow:review_pin_mismatch`: `16`
  - `docflow:invalid_field_type`: `0`
  - `docflow:missing_governance_ref`: `0`

## L28 Residual Chain Closure Executed (Remediation)
Applied remaining contradiction-closure edits across:
- `in/in-46.md`
- `in/in-47.md`
- `in/in-48.md`
- `in/in-49.md`
- `in/in-50.md`
- `in/in-51.md`
- `in/in-52.md`
- `in/in-53.md`
- `in/in-54.md`
- `in/in-55.md`

Remediation actions:
- added canonical explicit reference anchors in document bodies for all previously-missing `doc_requires` links in the above chain,
- updated stale `doc_reviewed_as_of` pin values in `in/in-46.md` .. `in/in-53.md` to match current dependency section versions.

Verification:
- strict docflow canonical sensor re-run,
- policy checks (`workflows`, `ambiguity-contract`) re-run,
- docflow test suite slice (`compliance_rows`, `violation_formatter`, `warning_failures`, class fixtures) re-run.

Measured outcome:
- strict contradictions: `67 -> 0` (`-67`)
- post-remediation contradiction classes:
  - `docflow:missing_explicit_reference`: `0`
  - `docflow:review_pin_mismatch`: `0`
  - `docflow:invalid_field_type`: `0`
  - `docflow:missing_governance_ref`: `0`
- remaining strict docflow findings are warning-class only (no contradiction rows).

## L29 Warning Analysis Packets by Warned Doc (Analysis-Only)
Generated machine-readable artifacts:
- `artifacts/out/docflow_warning_doc_packets.json`
- `artifacts/out/docflow_warning_doc_packet_summary.json`

Scope:
- objective: classify each currently warned document into one of:
  - `materially_still_true`
  - `needs_semantic_update`
  - `metadata_only`
- source signals:
  - `artifacts/out/docflow_section_reviews.json` warning rows (`stale_dep`, `missing_review`)
  - strict docflow warning log (`artifacts/out/docflow_strict_after_chain_closure.log`)
  - implementation-anchor existence scan for referenced `src/*.py` paths.

Packetized warned docs (`6`):
- `in/in-23.md`
- `in/in-31.md`
- `in/in-32.md`
- `in/in-33.md`
- `in/in-46.md`
- `in/in-54.md`

Classification result:
- `needs_semantic_update`: `5`
- `metadata_only`: `1`
- `materially_still_true`: `0`

Interpretation:
- `in/in-46.md` is correctly classified as semantic-risk (`missing_review` on core carrier/locality dependencies).
- `in/in-54.md` is metadata-model mismatch (`dep_version` domain mismatch against expected section-version domain), not direct contradiction-class semantic drift.
- `in/in-23.md`, `in/in-31.md`, `in/in-32.md`, and `in/in-33.md` are warning-lagged docs with stale implementation anchors to moved/deleted module paths; they require content-level review before pin refresh.

Non-action note:
- This slice is analysis-only. No source/governance document remediation was applied.

## L30 Warning Remediation Slice A (Metadata-Only, Executed)
Scope:
- objective: restore section-review signal integrity for the single `metadata_only` warning packet.
- targeted doc: `in/in-54.md`.

Edits applied:
- normalized `doc_section_reviews.in_in_54.*.dep_version` to section-version domain:
  - `POLICY_SEED.md#policy_seed: 2`
  - `CONTRIBUTING.md#contributing_contract: 2`
  - `README.md#repo_contract: 2`
  - `AGENTS.md#agent_obligations: 2`
  - `glossary.md#contract: 1`
  - `in/in-28.md#in_in_28: 1`
- refreshed review-note wording to reflect section-version semantics.

Interpretation:
- This removed the warning-model mismatch (document-revision domain used in section-review pins) and restored warning signal meaning for this step.

## L31 Warning Remediation Slice B (Semantic Updates, Executed)
Scope:
- objective: resolve all `needs_semantic_update` warning packets and clear stale implementation-anchor drift in warned docs.
- targeted docs:
  - `in/in-46.md`
  - `in/in-23.md`
  - `in/in-31.md`
  - `in/in-32.md`
  - `in/in-33.md`
  - checklist coupling update: `docs/sppf_checklist.md`.

Edits applied:
- `in/in-46.md`:
  - added missing `doc_section_reviews` entries for `glossary.md#forest`, `glossary.md#suite_site`, and `in/in-30.md#in_in_30`;
  - refreshed `POLICY_SEED.md#policy_seed` section review pin to `dep_version: 2`.
- `in/in-23.md`:
  - refreshed stale section-review pins (`POLICY_SEED`, `CONTRIBUTING`, `README`, `AGENTS`) to section v2;
  - migrated stale implementation/test anchors from removed legacy paths to current module/test paths under `analysis/dataflow/engine`, `analysis/core`, and `tests/gabion/...`.
- `in/in-31.md`:
  - refreshed stale section-review pins (`POLICY_SEED`, `CONTRIBUTING`, `README`, `AGENTS`) to section v2;
  - migrated stale anchors from monolith/old projection paths to current `dataflow_projection_materialization.py`, `projection/projection_registry.py`, and `dataflow_lint_helpers.py`.
- `in/in-32.md`:
  - refreshed stale `POLICY_SEED` section-review pin to section v2;
  - updated milestone module/test anchors to current `dataflow_structure_reuse`, `core/structure_reuse_classes`, `dataflow_decision_surfaces`, `dataflow_fingerprint_helpers`, and `aspf/aspf_mutation_log` surfaces.
- `in/in-33.md`:
  - refreshed stale `POLICY_SEED` section-review pin to section v2;
  - migrated stale `PatternSchema` and execution-pattern anchors from monolith-era paths to `projection/pattern_schema.py` and `projection/pattern_schema_projection.py`.
- `docs/sppf_checklist.md`:
  - updated `doc_ref` pins impacted by doc revision bumps (`in-31@5`, `in-23@11`, `in-33@5`);
  - updated stale evidence anchors to current code/test locations.

Verification:
- `mise exec -- env PYTHONPATH=. python scripts/policy/policy_check.py --workflows` (pass)
- `mise exec -- env PYTHONPATH=. python scripts/policy/policy_check.py --ambiguity-contract` (pass)
- `mise exec -- env PYTHONPATH=. python -m gabion docflow --root . --fail-on-violations --sppf-gh-ref-mode required` (pass; `No issues detected.`)

Measured outcome:
- strict docflow warning backlog for packetized warned docs: `20 -> 0`.
- strict docflow global state at end of slice: `violations=0`, `warnings=0`.

## Execution State
Contradiction and warning remediation streams are complete for the current strict-docflow backlog.

## Remaining Remediation Work
1. No strict-docflow remediation work remains in this stream (`contradicts=0`, `warnings=0`).
2. Optional follow-on only: keep checklist/doc anchors current as module paths evolve under future WS-5 slices.

---
doc_revision: 7
reader_reintern: "Reader-only: re-intern if doc_revision changed since you last read this doc."
doc_id: meta_reify_structural_homology_plan
doc_role: playbook
doc_scope:
  - repo
  - analysis
  - agents
doc_authority: informative
doc_requires:
  - AGENTS.md#agent_obligations
  - POLICY_SEED.md#policy_seed
  - glossary.md#contract
  - docs/shift_ambiguity_left_protocol.md#shift_ambiguity_left_protocol
  - src/gabion/analysis/aspf_core.py
  - src/gabion/analysis/type_fingerprints.py
  - src/gabion/analysis/evidence_keys.py
doc_reviewed_as_of:
  AGENTS.md#agent_obligations: 2
  POLICY_SEED.md#policy_seed: 2
  glossary.md#contract: 1
  docs/shift_ambiguity_left_protocol.md#shift_ambiguity_left_protocol: 2
doc_review_notes:
  AGENTS.md#agent_obligations: "Aligned plan structure with boundary-normalization and protocol-reification obligations."
  POLICY_SEED.md#policy_seed: "Retained deterministic-core and policy-check execution expectations for correction units."
  glossary.md#contract: "Kept bundle/tier semantics explicit when introducing new identity carriers."
  docs/shift_ambiguity_left_protocol.md#shift_ambiguity_left_protocol: "Applied shift-left sequence to avoid introducing dual-shape ambiguity in semantic core."
---

<a id="meta_reify_structural_homology_plan"></a>
# META-REIFY Structural Homology Plan (Leverage-Ordered)

This plan enriches the prior baseline by making each step a **structural prerequisite** for the next, while also feeding constraints back into prior steps through explicit invariants and verification gates.

## Leverage principles (applies to all steps)

1. **Contract before transformation:** each step must publish an explicit interface contract before it mutates behavior.
2. **Single ingress alternation:** ambiguity is normalized once at boundaries, never propagated into semantic core.
3. **Monotone refinement:** each step can only reduce contradiction-ledger entropy; no step may widen accepted core shapes.
4. **Bidirectional coupling:** every step has a forward unlock and a backward correction channel.
5. **Artifact-first verification:** each step emits an inspectable artifact that becomes mandatory input to the next step.

## Step 1 — Canonical surface mapping and contradiction ledger

**Objective:** Build a single source-of-truth map from proposed concepts to existing implementation surfaces.

**Primary surfaces:**
- `src/gabion/analysis/aspf_core.py`
- `src/gabion/analysis/type_fingerprints.py`
- `src/gabion/analysis/evidence_keys.py`
- `src/gabion/analysis/canonical.py`

**Actions:**
1. Produce a map from conceptual names (identity projection, rope, basis path) to concrete classes/functions in the current code.
2. Record all schema/type contradictions (for example: integer-atom path proposal vs current `basis_path: tuple[str, ...]`).
3. Label each contradiction as boundary-level (adaptable) or core-level (requires semantic refactor).
4. Define contradiction severities (`blocking`, `non_blocking`) and attach required remediation owners per entry.
5. Define the **step-acceptance predicate**: no unidentified core-level contradiction may remain before Step 2.

**Output artifact:** `surface_map + contradiction_ledger`.

**Required inputs:** Existing ASPF identity surfaces, payload schemas, and downstream evidence consumers.

**Acceptance proof:** Every identity-relevant symbol in touched surfaces maps to exactly one ledger entry or canonical surface row.

**Leverage to Step 2:** Defines a finite, typed ingress boundary where normalization can be introduced once.

**Feedback to prior/future:** The contradiction ledger is updated by every later step; unresolved items block promotion to canonical contracts.

## Step 2 — Boundary normalization protocol for identity ingress

**Objective:** Introduce a single normalization protocol at ingress so semantic core paths operate on one deterministic representation.

**Actions:**
1. Define an explicit protocol/dataclass bundle for identity ingress options and legacy-shape acceptance.
2. Normalize incoming representations (legacy string basis paths, scalar aliases) into one canonical internal shape.
3. Enforce impossible-by-construction validation and discharge invalid states via `never(...)` in post-invariant dead branches.
4. Add observability fields (normalization mode, adapter provenance) so Step 5 evidence can attribute decisions to ingress contracts.
5. Define the **step-acceptance predicate**: semantic core receives only one normalized identity shape.

**Output artifact:** `identity_ingress_protocol + normalizer` with deterministic output schema.

**Required inputs:** `surface_map + contradiction_ledger` from Step 1.

**Acceptance proof:** Boundary tests demonstrate legacy input acceptance with canonical output equivalence.

**Leverage to Step 3:** Enables canonical identity contract migration without propagating dual-shape branches.

**Feedback loop:** If Step 3 introduces additional shape pressure, update only the boundary protocol rather than widening core control flow.

## Step 3 — Canonical identity contract migration (basis-path first)

**Objective:** Move canonical identity payloads to path-preserving representation while preserving temporary adapters at boundary only.

**Actions:**
1. Promote basis-path-preserving identity as canonical in payload builders.
2. Keep scalar/digest forms as explicit non-canonical derived aliases with lifecycle metadata:
   - `actor`
   - `rationale`
   - `scope`
   - `start`
   - `expiry`
   - `rollback_condition`
   - `evidence_links`
3. Ensure serialization/deserialization round-trips preserve canonical path semantics.
4. Define adapter expiry checkpoints tied to concrete evidence thresholds (consumer migration %, zero core reads, stable payload round-trip).
5. Define the **step-acceptance predicate**: canonical identity is path-preserving by contract, aliases are boundary-scoped only.

**Output artifact:** `canonical_identity_contract_v2` with explicit alias-lifecycle sidecar.

**Required inputs:** `identity_ingress_protocol + normalizer` from Step 2.

**Acceptance proof:** Round-trip and payload invariance tests pass for canonical identity while alias reads are explicitly tracked.

**Leverage to Step 4:** Provides stable canonical material for rope algebra and subset/intersection/union operations.

**Feedback loop:** If alias usage expands during Step 4, treat as regression and remediate by tightening boundary adapters.

## Step 4 — Rope carrier introduction and algebraic operation lift

**Objective:** Replace scalar-centric internal reasoning with multiset-rope carriers using canonical multiset normalization.

**Actions:**
1. Introduce rope carriers as normalized multisets compatible with existing canonical multiset semantics.
2. Re-express join/meet/subsumption as multiset operations (union/intersection/subset) at semantic decision points.
3. Remove arithmetic-dependent control decisions from core paths.
4. Add operation-level invariants (`idempotence`, `commutativity`, `associativity`, `subset monotonicity`) for rope algebra.
5. Define the **step-acceptance predicate**: no core decision surface depends on scalar prime-product arithmetic.

**Output artifact:** `rope_carrier + lifted_algebra_ops` integrated into identity consumers.

**Required inputs:** `canonical_identity_contract_v2` from Step 3.

**Acceptance proof:** Algebra property tests and callsite inventory confirm scalar-independent core reasoning.

**Leverage to Step 5:** Creates deterministic, inspectable invariants for targeted test expansion.

**Feedback loop:** Any operation that still depends on scalar arithmetic triggers a Step-1 contradiction ledger update and fix-forward correction unit.

## Step 5 — Evidence and decision-surface hardening

**Objective:** Lock in structural guarantees through explicit decision protocols and evidence carriers.

**Actions:**
1. Add/extend decision protocols where branch invariants are currently implicit.
2. Ensure evidence payloads declare canonical vs derived identity layers consistently.
3. Refresh targeted tests around:
   - canonical identity stability
   - rope normalization
   - subset/meet/join semantics
   - alias lifecycle boundaries
4. Introduce contradiction-ledger linkage in evidence output (which contradiction was remediated, deferred, or re-opened).
5. Define the **step-acceptance predicate**: every branch-affecting identity decision has an explicit protocol and evidence carrier.

**Output artifact:** `decision_surface_contracts + evidence_consistency_suite`.

**Required inputs:** `rope_carrier + lifted_algebra_ops` from Step 4.

**Acceptance proof:** Evidence snapshots prove canonical/derived separation and decision-protocol coverage without ambiguity regressions.

**Leverage to Step 6:** Provides measurable pass/fail gates for correction-unit completion and CI durability.

**Feedback loop:** Test or evidence drift feeds directly back to Step 2 (boundary normalization) or Step 4 (algebra lift), never to rollback-first behavior.

## Step 6 — Correction-unit validation and deprecation ratchet

**Objective:** Finalize a correction unit with required policy gates and a ratcheting deprecation path for temporary adapters.

**Actions:**
1. Run required validation stack for the correction unit:
   - `scripts/policy_check.py --workflows`
   - `scripts/policy_check.py --ambiguity-contract`
   - targeted pytest for touched semantic surfaces
   - evidence-carrier refresh/check when applicable
2. Stage/commit/push one blocking surface per correction unit.
3. Advance deprecation milestones for non-canonical adapters; fail closed when expiry criteria are met.
4. Publish a post-unit delta report with: contradictions closed/opened, adapter expiry movement, and newly enforced invariants.
5. Define the **step-acceptance predicate**: correction-unit report demonstrates net reduction in contradiction-ledger entropy.

**Output artifact:** `validated_correction_unit + adapter_deprecation_state`.

**Required inputs:** `decision_surface_contracts + evidence_consistency_suite` from Step 5.

**Acceptance proof:** Policy/ambiguity checks, targeted tests, and delta report all pass with no unresolved blocking contradictions.

**Leverage back to Step 1:** Post-validation findings become new entries in the contradiction ledger, creating a closed remediation loop.

---


## Structural-first execution path (recommended first correction unit)

To maximize leverage, execute the first correction unit as a **thin vertical slice** that touches every step at minimum breadth:

1. **Step-1 minimal closure:** map one end-to-end identity payload path (`fingerprint_identity_payload` producer → evidence carrier consumer).
2. **Step-2 boundary normalizer spike:** introduce one ingress protocol that canonicalizes only that path.
3. **Step-3 canonical contract lock:** publish `canonical_identity_contract_v2` for that path and mark scalar alias as temporary.
4. **Step-4 algebra pilot:** implement one rope-backed decision operation (prefer subset/subsumption check) on the same path.
5. **Step-5 evidence hardening:** emit contradiction-ledger linkage in the corresponding evidence payload.
6. **Step-6 validation closure:** run the policy/ambiguity/targeted-test stack and publish a delta report.

This thin-slice pattern creates immediate cross-step feedback with minimal blast radius, then scales breadth by repeating the same six-step packet over additional paths.

## Step-by-step remediation packet template

Each correction unit should carry the following packet so downstream steps inherit complete context:

- `packet_id`: stable identifier for the correction unit.
- `blocking_surface`: single blocking surface remediated by this packet.
- `inputs`: artifacts consumed from prior step.
- `transform`: exact contract/protocol/algebra change applied.
- `proof`: acceptance predicate and test/policy evidence.
- `delta`: contradictions closed/opened, adapters advanced/expired.
- `next_unlock`: which next-step contract is now unblocked.

Use packet review as the merge gate: if `next_unlock` or `proof` is absent, the packet is incomplete.


## Concrete first-slice blueprint (greatest-leverage path)

Use this as the default first packet because it crosses producer → contract → evidence in one tractable path.

| Slice layer | Concrete location | First change target | Unlocks |
|---|---|---|---|
| Producer | `src/gabion/analysis/type_fingerprints.py` | `fingerprint_identity_payload` output contract | Canonical identity material for downstream consumers |
| Contract | `src/gabion/analysis/evidence_keys.py` | `fingerprint_identity_layers` canonical/derived boundary | Consistent identity-layer semantics in evidence |
| Structure | `src/gabion/analysis/canonical.py` | multiset canonicalization bridge for rope carrier | Deterministic rope normalization semantics |
| Core path type | `src/gabion/analysis/aspf_core.py` | basis-path contract assumptions at ingress/parse boundaries | Boundary normalization without core dual-shape branching |
| Evidence/test | `tests/test_fingerprint_soundness.py`, `tests/test_type_fingerprints_sidecar.py` | assert canonical-vs-derived separation and alias lifecycle signals | Confidence gates for deprecation ratchet |

### First-slice definition of done (must all hold)

1. Exactly one producer path emits the new canonical/derived separation contract.
2. Exactly one consumer path reads canonical identity without scalar fallback in core logic.
3. Scalar alias remains boundary-only and explicitly marked temporary in emitted payload.
4. Rope normalization is deterministic and round-trip stable for the selected path.
5. Policy/ambiguity checks and targeted tests pass for this slice.

### First-slice execution cadence

1. Draft `surface_map + contradiction_ledger` row(s) only for the selected path.
2. Implement boundary protocol/normalizer only for selected ingress.
3. Migrate producer payload contract and add lifecycle metadata for adapters.
4. Add one rope-backed operation and its algebra invariant tests.
5. Wire evidence linkage to contradiction rows touched by this packet.
6. Run validation stack and publish packet delta before broadening scope.

Broaden to the next path only after this slice has a complete packet (`proof` + `next_unlock`).


## Packet backlog scaffold (ordered by leverage)

Use this backlog scaffold to schedule correction packets in strict leverage order after the first slice.

| Backlog order | Packet focus | Blocking surface | Required proof before next packet |
|---|---|---|---|
| B1 | Canonical identity producer parity | `fingerprint_identity_payload` and immediate consumers | Canonical/derived separation emitted and consumed without new core-shape branches |
| B2 | Boundary normalizer widening | Next ingress path that currently accepts mixed shapes | One normalized ingress shape with legacy adaptation isolated at boundary |
| B3 | Rope operation expansion | Next scalar-dependent decision surface | Algebra invariants + no scalar arithmetic in the migrated decision path |
| B4 | Evidence propagation | Evidence carriers for migrated path set | Contradiction linkage and protocol coverage in emitted evidence |
| B5 | Adapter deprecation ratchet | Temporary aliases from B1-B4 | Expiry threshold met + fail-closed behavior validated |

### Backlog gating rules

1. Do not start packet `B(n+1)` until packet `B(n)` publishes both `proof` and `next_unlock`.
2. If a packet opens a new `blocking` contradiction, it must close that contradiction before merge.
3. Coverage or ambiguity regressions convert the current packet into a dedicated remediation packet; do not continue breadth expansion.
4. Each packet must update the contradiction ledger delta (closed/opened/deferred) in the packet report.

### Minimal packet report schema (copy/paste)

```yaml
packet_id: "B1-identity-producer-parity"
blocking_surface: "src/gabion/analysis/type_fingerprints.py::fingerprint_identity_payload"
inputs:
  - surface_map_ref
  - contradiction_ledger_ref
transform:
  contract_change: "canonical_identity_contract_v2 emitted"
  boundary_change: "legacy scalar alias isolated to boundary payload"
proof:
  policy_checks:
    - scripts/policy_check.py --workflows
    - scripts/policy_check.py --ambiguity-contract
  targeted_tests:
    - tests/test_fingerprint_soundness.py
    - tests/test_type_fingerprints_sidecar.py
delta:
  contradictions_closed: []
  contradictions_opened: []
  contradictions_deferred: []
  adapters:
    advanced: []
    expired: []
next_unlock: "B2 boundary normalizer widening"
```


## Failure-routing decision table (maximize next-step leverage)

When a packet fails a gate, route remediation to the earliest step that can remove the ambiguity with the least downstream churn.

| Failure signal | Route to step | Why this is highest leverage | Required remediation artifact |
|---|---|---|---|
| Mixed-shape payload detected in core | Step 2 | Boundary normalization prevents repeated downstream branch growth | Updated `identity_ingress_protocol + normalizer` |
| Canonical vs derived identity confusion in payloads | Step 3 | Contract separation stabilizes all later consumers | Revised `canonical_identity_contract_v2` + alias lifecycle delta |
| Scalar arithmetic still used in decision path | Step 4 | Algebra lift removes entire class of non-structural decisions | Updated `rope_carrier + lifted_algebra_ops` |
| Evidence cannot attribute decision path | Step 5 | Protocol/evidence linkage restores auditability and merge confidence | `decision_surface_contracts + evidence_consistency_suite` delta |
| Policy/ambiguity checks regress | Step 6 (dedicated remediation packet) | Validation closure is the narrowest safe stop-the-line point | Remediation packet report with contradiction delta |

### Stop-the-line rules

1. Any `blocking` contradiction discovered post-Step-2 freezes breadth expansion and spawns a dedicated remediation packet.
2. Never patch around a Step-2/Step-3 failure in Step-4+ code; repair contracts at their source step.
3. If a temporary adapter survives past its expiry threshold, fail closed and route to Step 3 for lifecycle closure.

## Reviewer checklist (structural leverage enforcement)

Use this checklist to reject low-leverage or order-violating packets.

- [ ] Packet declares one `blocking_surface` and one `next_unlock`.
- [ ] Packet remediation occurs at the earliest leverage step indicated by the failure-routing table.
- [ ] No new semantic-core dual-shape branching was introduced.
- [ ] Canonical/derived boundary remains explicit; scalar alias remains non-canonical.
- [ ] Contradiction delta is included and monotone trend is preserved.
- [ ] Policy/ambiguity checks + targeted tests are attached as packet proof.


## Step-to-command execution map (enforced validation cadence)

Run these commands at the smallest scope that still proves the step contract. Escalate only when a gate fails.

| Step | Minimum required checks | Escalation trigger | Escalated checks |
|---|---|---|---|
| 1 (surface map) | targeted inventory/tests for touched path | unmapped identity symbol or unresolved `blocking` contradiction | full identity-path inventory + packet split |
| 2 (boundary normalizer) | `scripts/policy_check.py --ambiguity-contract` + targeted ingress tests | new dual-shape branch appears in core | add focused regression tests + contradiction remediation packet |
| 3 (contract migration) | targeted contract round-trip tests + canonical/derived assertions | canonical/derived confusion or adapter lifecycle drift | expand consumer matrix tests + lifecycle closure packet |
| 4 (rope algebra) | targeted algebra invariant tests (`idempotence`, `commutativity`, `associativity`, subset monotonicity) | any scalar-dependent decision remains | callsite audit + dedicated algebra remediation packet |
| 5 (evidence hardening) | evidence payload assertions + contradiction linkage checks | missing protocol attribution in evidence | add evidence snapshot delta tests |
| 6 (validation closure) | `scripts/policy_check.py --workflows`, `scripts/policy_check.py --ambiguity-contract`, targeted pytest | policy/ambiguity/coverage regression | stop-the-line remediation packet before breadth expansion |

## Prohibited low-leverage moves (reject on review)

1. Introducing new semantic-core `if/elif` shape branching to accommodate legacy payloads.
2. Expanding adapter scope instead of tightening boundary normalizers.
3. Treating scalar aliases as canonical identity in any decision surface.
4. Advancing backlog breadth when the current packet lacks `proof` or `next_unlock`.
5. Deferring newly introduced `blocking` contradictions to later packets.

## Contradiction-ledger minimum fields

Each contradiction entry should include enough structure to route and close it deterministically.

```yaml
contradiction_id: "C-0001"
status: "open"            # open|closed|deferred
severity: "blocking"      # blocking|non_blocking
step_owner: 2              # earliest leverage step expected to remediate
surface: "src/gabion/analysis/aspf_core.py::parse_2cell_witness"
signal: "mixed-shape payload detected in core"
opened_by_packet: "B2-boundary-normalizer"
closure_packet: null
closure_proof: []
next_action: "normalize at ingress protocol"
```

Use `step_owner` as the routing key when triaging failures.

## Cross-step leverage matrix

| Step | Produces | Most directly unlocks | Refines previous step by |
|---|---|---|---|
| 1 | Surface map + contradiction ledger | Step 2 boundary protocol | Making ambiguity concrete and typed |
| 2 | Deterministic ingress normalizer | Step 3 contract migration | Eliminating branch pressure discovered in Step 1 |
| 3 | Canonical identity contract v2 | Step 4 algebra lift | Constraining normalization to canonical payload invariants |
| 4 | Rope carrier + multiset algebra | Step 5 decision/evidence hardening | Exposing residual scalar dependencies missed in Step 3 |
| 5 | Decision protocols + test/evidence gates | Step 6 correction-unit closure | Stress-testing algebra/contract assumptions from Step 4 |
| 6 | Validated correction unit + deprecation ratchet | Next correction cycle (Step 1) | Converting test/policy findings into next contradiction ledger |

## Step interface contract table

| Step | Consumes | Publishes | Contract exported to next step |
|---|---|---|---|
| 1 | Existing surfaces + schemas | `surface_map + contradiction_ledger` | Boundary scope, contradiction classes, blocking gates |
| 2 | Step-1 artifacts | `identity_ingress_protocol + normalizer` | Single normalized ingress shape |
| 3 | Step-2 artifacts | `canonical_identity_contract_v2` | Canonical/derived identity separation |
| 4 | Step-3 artifacts | `rope_carrier + lifted_algebra_ops` | Structural algebra for meet/join/subsumption |
| 5 | Step-4 artifacts | `decision_surface_contracts + evidence_consistency_suite` | Observable decision correctness and evidence coverage |
| 6 | Step-5 artifacts | `validated_correction_unit + adapter_deprecation_state` | Next-cycle contradiction and deprecation deltas |

## Exit criteria (terminal coherence)

1. Canonical identity is path-preserving and deterministic across serialization boundaries.
2. Scalar prime product remains non-canonical and bounded to explicit temporary adapter scope.
3. Core semantic decisions operate on normalized structural carriers (rope/path), not scalar arithmetic shortcuts.
4. Policy/ambiguity checks and targeted tests pass for each correction unit.
5. Contradiction ledger trends monotonically downward across correction cycles.

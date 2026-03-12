---
doc_revision: 1
reader_reintern: "Reader-only: re-intern if doc_revision changed since you last read this doc."
doc_id: ambiguity_contract_policy_rules
doc_role: policy
doc_scope:
  - repo
  - governance
  - policy
  - ambiguity
doc_authority: normative
doc_requires:
  - POLICY_SEED.md#policy_seed
  - glossary.md#contract
  - docs/shift_ambiguity_left_protocol.md#shift_ambiguity_left_protocol
doc_reviewed_as_of:
  POLICY_SEED.md#policy_seed: 55
  glossary.md#contract: 44
  docs/shift_ambiguity_left_protocol.md#shift_ambiguity_left_protocol: 3
doc_review_notes:
  POLICY_SEED.md#policy_seed: "Reviewed POLICY_SEED rev55; ambiguity-gate remediation still prefers boundary normalization and contract reification over compatibility layering."
  glossary.md#contract: "Reviewed glossary rev44; carrier, witness, and boundary terminology remain aligned with these rule playbooks."
  docs/shift_ambiguity_left_protocol.md#shift_ambiguity_left_protocol: "Reviewed the protocol card rev3; the rule-local playbooks here specialize that generic refactor sequence."
doc_change_protocol: "POLICY_SEED.md#change_protocol"
doc_owner: maintainer
rules:
  - rule_id: ambiguity.new_violations
    domain: ambiguity_contract
    severity: blocking
    predicate:
      op: int_gte
      path: [new_violations]
      value: 1
    outcome:
      kind: block
      message: ambiguity contract policy violations
      guidance:
        why: new ambiguity-contract findings indicate semantic alternation or structure loss has moved too deep into deterministic core
        prefer: move the disputed carrier seam upstream, reify one strict internal contract, and re-run the ambiguity gate before widening any compatibility surface
        avoid:
          - do not silence the gate by marking more internal helpers as ambiguity boundaries
          - do not keep JSON-like or Mapping[str, object] carriers alive past true I/O seams
    evidence_contract: none
    playbook_anchor: ambiguity-new-violations

  - rule_id: ambiguity.ok
    domain: ambiguity_contract
    severity: info
    predicate:
      op: always
    outcome:
      kind: pass
      message: ambiguity contract policy check passed
    evidence_contract: none

  - rule_id: ACP-003
    domain: ambiguity_contract_ast
    severity: blocking
    predicate:
      op: str_eq
      path: [event]
      value: runtime_isinstance_call
    outcome:
      kind: block
      message: runtime type narrowing in deterministic core
      guidance:
        why: deterministic core is branching on runtime type alternatives instead of receiving a normalized carrier or explicit decision result
        prefer: normalize the input at ingress, or replace the branch surface with a tagged DTO / decision protocol returned from the boundary
        avoid:
          - do not spread new isinstance ladders deeper through core helpers
          - do not replace the dynamic branch with a loose Any/object carrier
    evidence_contract: none
    playbook_anchor: acp-003

  - rule_id: ACP-004
    domain: ambiguity_contract_ast
    severity: blocking
    predicate:
      op: str_eq
      path: [event]
      value: dynamic_annotation
    outcome:
      kind: block
      message: dynamic type alternation in deterministic core annotation
      guidance:
        why: the function contract itself admits unresolved runtime alternation, so every downstream caller inherits the ambiguity
        prefer: push the alternation to a true ingress seam and expose a strict internal carrier or explicit tagged result instead
        avoid:
          - do not normalize this by widening annotations to Any, object, or larger unions
          - do not add downstream branches that compensate for an already-loose signature
    evidence_contract: none
    playbook_anchor: acp-004

  - rule_id: ACP-002
    domain: ambiguity_contract_ast
    severity: blocking
    predicate:
      op: str_eq
      path: [event]
      value: sentinel_control
    outcome:
      kind: block
      message: sentinel control outcome in core
      guidance:
        why: core logic is using sentinel values as implicit control flow, which collapses richer carrier structure into scalar alternation
        prefer: reify the control choice as a structural outcome or normalize the carrier at ingress so the core receives one lawful shape
        avoid:
          - do not swap one sentinel for another magic scalar
          - do not stringify or otherwise scalarize a structured carrier just to drive control flow
    evidence_contract: none
    playbook_anchor: acp-002

  - rule_id: ACP-007
    domain: ambiguity_contract_ast
    severity: blocking
    predicate:
      op: str_eq
      path: [event]
      value: nullable_contract_control
    outcome:
      kind: block
      message: nullable contract leaked past ingress into deterministic core
      guidance:
        why: core logic is classifying a nullable carrier imperatively instead of receiving a strict contract from ingress
        prefer: normalize at ingress into a non-null DTO, tagged result, or explicit decision protocol so downstream code is only called on strict inputs
        avoid:
          - do not replace None with a custom sentinel or alternate magic value
          - do not add more is None or is not None branches deeper in deterministic core
    evidence_contract: none
    playbook_anchor: acp-007

  - rule_id: ACP-005
    domain: ambiguity_contract_ast
    severity: blocking
    predicate:
      op: str_eq
      path: [event]
      value: match_fallthrough_without_never
    outcome:
      kind: block
      message: fall-through match case must call never() in its body
      guidance:
        why: a wildcard fallthrough is preserving latent alternation instead of discharging an impossible state
        prefer: exhaust the lawful variants structurally, then use never() on the dead post-invariant path
        avoid:
          - do not leave pass/continue/return-None fallthrough branches in a supposedly exhausted match
          - do not add compatibility fallthroughs that merely postpone the impossible state
    evidence_contract: none
    playbook_anchor: acp-005

  - rule_id: ACP-006
    domain: ambiguity_contract_ast
    severity: blocking
    predicate:
      op: str_eq
      path: [event]
      value: probe_state_recovery
    outcome:
      kind: block
      message: downstream ambiguity recovery after structural probe
      guidance:
        why: stores unresolved shape or type ambiguity in local carrier state and resolves it later, recreating downstream control ambiguity
        prefer: move dispatch to the boundary, or use a reducer over already-accepted variants, or return an explicit tagged result
        avoid:
          - do not add matched_* locals, placeholder strings or ints, or post-probe if-not-matched recovery branches
          - do not silence the site by inserting never() downstream of the structural probe
    evidence_contract: none
    playbook_anchor: acp-006

  - rule_id: ambiguity.ast.ok
    domain: ambiguity_contract_ast
    severity: info
    predicate:
      op: always
    outcome:
      kind: pass
      message: ambiguity AST event ignored
    evidence_contract: none
---

<a id="ambiguity_contract_policy_rules"></a>
# Ambiguity Contract Policy Rules

These rules are authoritative policy for ambiguity-contract blocking and the
corresponding remediation playbooks.

<a id="ambiguity-new-violations"></a>
## `ambiguity.new_violations`

Meaning: the repo has introduced one or more new ambiguity-contract findings.

Preferred response:
- move the first real boundary upstream on the affected fiber
- reify a strict internal carrier or decision protocol once
- delete the downstream ambiguity branches that were compensating for the loose shape

Avoid:
- marking internal reshaping helpers as boundaries
- keeping JSON, generic mappings, or sentinels alive after true ingress

Reference: [Shift-Ambiguity-Left Protocol](../shift_ambiguity_left_protocol.md#shift_ambiguity_left_protocol).

<a id="acp-003"></a>
## `ACP-003`

Meaning: runtime type classification is happening inside deterministic core.

Preferred response:
- normalize the incoming carrier at a real I/O boundary
- or return a tagged result / decision protocol from that boundary

Avoid:
- cascading `isinstance(...)` branches through more helpers
- widening internal carriers to `Any` or `object`

<a id="acp-004"></a>
## `ACP-004`

Meaning: the function signature itself encodes unresolved dynamic alternation.

Preferred response:
- discharge the alternation at ingress and keep the internal signature strict

Avoid:
- treating wider annotations as harmless documentation
- compensating with more downstream narrowing branches

<a id="acp-002"></a>
## `ACP-002`

Meaning: a sentinel value is being used as control flow in core logic.

Preferred response:
- keep the richer carrier authoritative
- encode the control choice structurally, not as a magic scalar

Avoid:
- replacing one sentinel with another
- converting structured carriers to strings or scalar stand-ins

<a id="acp-007"></a>
## `ACP-007`

Meaning: nullable input is still being classified in deterministic core.

Preferred response:
- turn the ingress result into a strict non-null DTO or explicit tagged outcome

Avoid:
- `None`/sentinel swaps
- repeated `is None` checks further down the same fiber

<a id="acp-005"></a>
## `ACP-005`

Meaning: a wildcard/fallthrough match branch is not discharging the impossible state.

Preferred response:
- make the variant space exhaustive
- use `never(...)` on the dead post-invariant branch

Avoid:
- fallthrough `pass`, `continue`, or sentinel returns
- "compatibility" wildcard branches that just postpone the contradiction

<a id="acp-006"></a>
## `ACP-006`

Meaning: code probes a shape and then stores provisional state for later recovery.

Preferred response:
- move the probe to ingress
- or reduce already-accepted variants immediately
- or return an explicit tagged result from the probe

Avoid:
- `matched_*` locals
- placeholder strings/ints
- second-stage recovery branches after the first structural probe

Reference: [Shift-Ambiguity-Left Protocol](../shift_ambiguity_left_protocol.md#shift_ambiguity_left_protocol).

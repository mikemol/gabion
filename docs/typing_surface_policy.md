---
doc_revision: 1
doc_id: typing_surface_policy
doc_role: normative
doc_scope:
  - repo
  - typing
  - policy
doc_authority: normative
doc_requires:
  - POLICY_SEED.md#policy_seed
  - glossary.md#contract
doc_change_protocol: "POLICY_SEED.md#change_protocol"
doc_owner: maintainer
---

<a id="typing_surface_policy"></a>
# Typing Surface Policy (Normative)

## 1) Allowed coarse typing surfaces

`Any`, bare `object`, and `dict[str, object]` are allowed only at **explicit boundary ingress/egress** where the source shape is external and immediately normalized.

Allowed boundary examples:
- parsing raw external payloads before DTO/model validation,
- temporary adapters that convert legacy payloads into typed carriers,
- tooling-only probes that do not participate in semantic-core decisions.

These boundary uses must be followed by deterministic normalization into one of:
- DTOs,
- `Protocol` contracts,
- dataclass carriers,
- `TypedDict` or Pydantic models.

## 2) Forbidden surfaces

The following are forbidden policy surfaces:
- **semantic core** (`src/gabion/analysis/**`),
- **stage contracts** (modules under `*/stages/*`, `*stage_contract*`, or `*_stage.py`),
- **reducers** (modules under `*/reducers/*` or `*_reducer.py`).

In those surfaces, do not introduce `Any`, bare `object`, or `dict[str, object]` annotations.

## 3) Required alternatives

When a coarse annotation would otherwise be used, replace it with:
- a typed DTO (including Pydantic model DTOs),
- a `Protocol` decision/bundle contract,
- a dataclass carrier with explicit fields,
- a `TypedDict`/Pydantic model for dictionary-shaped payloads.

## 4) Waiver mechanism (required metadata)

Waivers are declared in `baselines/typing_surface_policy_waivers.json`.

Each waiver entry must include:
- `path`
- `qualname`
- `line`
- `kind`
- `rationale`
- `scope`
- `expiry`
- `owner`

Policy scanner behavior:
- waivers with complete metadata suppress matching findings,
- waivers with missing/invalid metadata produce `invalid_waiver` findings,
- ratchet policy is baseline-first: current exceptions are seeded in baseline, then reduced in future correction units.

## 5) Enforcement

`src/gabion/tooling/runtime/policy_scanner_suite.py` publishes machine-readable `typing_surface` findings with stable keys for baseline/waiver ratcheting.

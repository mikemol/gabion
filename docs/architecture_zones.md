---
doc_revision: 2
reader_reintern: "Reader-only: re-intern if doc_revision changed since you last read this doc."
doc_id: architecture_zones
doc_role: architecture
doc_scope:
  - repo
  - architecture
  - analysis
  - synthesis
  - cli
doc_authority: informative
doc_requires:
  - README.md#repo_contract
  - POLICY_SEED.md#policy_seed
  - glossary.md#contract
doc_reviewed_as_of:
  README.md#repo_contract: 1
  POLICY_SEED.md#policy_seed: 1
  glossary.md#contract: 1
doc_review_notes:
  README.md#repo_contract: "Reviewed LSP-first repo contract; this note scopes boundaries for enforcement."
  POLICY_SEED.md#policy_seed: "Reviewed dataflow grammar and execution invariants; boundary note must not weaken them."
  glossary.md#contract: "Reviewed tier semantics; handoff contract uses Tier-1 reification at core ingress."
doc_change_protocol: "POLICY_SEED.md#change_protocol"
doc_sections:
  architecture_zones: 1
doc_section_requires:
  architecture_zones:
    - README.md#repo_contract
    - POLICY_SEED.md#policy_seed
    - glossary.md#contract
doc_section_reviews:
  architecture_zones:
    README.md#repo_contract:
      dep_version: 1
      self_version_at_review: 1
      outcome: no_change
      note: "Architecture zoning stays aligned with LSP-first repository contract."
    POLICY_SEED.md#policy_seed:
      dep_version: 1
      self_version_at_review: 1
      outcome: no_change
      note: "No policy weakening; this note is a scoping aid."
    glossary.md#contract:
      dep_version: 1
      self_version_at_review: 1
      outcome: no_change
      note: "Tier language is glossary-aligned."
doc_erasure:
  - formatting
  - typos
doc_owner: maintainer
---

<a id="architecture_zones"></a>
# Architecture Zones

## Ambiguity admission zones
These zones are allowed to accept ambiguous, partial, or external shapes before normalization:

- **CLI parsing boundary:** `src/gabion/cli.py` (arg parsing, CSV/string option lifting, env/default merge).
- **External payload adapters:** `src/gabion/lsp_client.py` and `src/gabion/server.py` (JSON-RPC request/response transport and command payload intake).
- **Serialization boundaries:** `src/gabion/schema.py` and `src/gabion/json_types.py` (DTO validation, JSON-like carrier typing, artifact wire shapes).

## Deterministic core zones
These zones are expected to run deterministic semantics once data is reified:

- **Analysis semantics pipeline:** `src/gabion/analysis/` (dataflow graphing, evidence projection, decision/report surfaces).
- **Synthesis semantics pipeline:** `src/gabion/synthesis/` (bundle merge, naming, scheduling, protocol plan construction).
- **Refactor semantics engine:** `src/gabion/refactor/` (rewrite planning and edit synthesis driven by reified plans).

## Boundary handoff contract
Only **Tier-1 reified objects** cross from ambiguity zones into deterministic core zones.

- Inputs must be promoted to typed DTO/model/config carriers before they enter `src/gabion/analysis/`, `src/gabion/synthesis/`, or `src/gabion/refactor/`.
- Allowed cross-boundary forms are explicit objects (for example, schema DTOs/config dataclasses), not raw ad-hoc dictionaries or free-form tuples.
- Review and tooling should enforce by package scope:
  - outer adapters: `src/gabion/cli.py`, `src/gabion/lsp_client.py`, `src/gabion/server.py`, `src/gabion/schema.py`, `src/gabion/json_types.py`
  - deterministic core: `src/gabion/analysis/`, `src/gabion/synthesis/`, `src/gabion/refactor/`

## Test-evidence boundary semantics (semi-normative)
Docflow excess clustering should treat test evidence near semantic core by
architecture class, not by individual node listing:

- **Boundary probe edges (`call_footprint`)**: tests may originate outside core
  zones while legitimately targeting `src/gabion/server.py`,
  `src/gabion/server_core/`, or `src/gabion/analysis/`.
- **Adjacency probe clusters (`call_cluster`)**: helper fan-in/fan-out around a
  semantic-core target is a valid architectural probe shape.
- **Decision probe surfaces (`decision_surface`)**: parameterized decision points
  in semantic core are architecture-significant and should be tracked as a class
  of ingress-normalization/core-semantics checks.

These classes define reusable boundary semantics for future evidence nodes.

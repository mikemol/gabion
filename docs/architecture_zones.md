---
doc_revision: 4
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
  README.md#repo_contract: 2
  POLICY_SEED.md#policy_seed: 2
  glossary.md#contract: 1
doc_review_notes:
  README.md#repo_contract: "Reviewed README.md rev2 (removed stale ASPF action-plan CLI/examples; continuation docs now state/delta only)."
  POLICY_SEED.md#policy_seed: "Reviewed POLICY_SEED.md rev2 (forward-remediation order, ci_watch failure-bundle durability, and enforced execution-coverage policy wording)."
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
      dep_version: 2
      self_version_at_review: 1
      outcome: no_change
      note: "Repo contract rev2 reviewed; command and artifact guidance remains aligned."
    POLICY_SEED.md#policy_seed:
      dep_version: 2
      self_version_at_review: 1
      outcome: no_change
      note: "Policy seed rev2 reviewed; governance obligations remain aligned."
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

- **Server command-core orchestration zone:** `src/gabion/server_core/` (deterministic command orchestration, ingress-normalized command options, and stable command-core sequencing).
- **Analysis semantics pipeline:** `src/gabion/analysis/` (dataflow graphing, evidence projection, decision/report surfaces).
- **Synthesis semantics pipeline:** `src/gabion/synthesis/` (bundle merge, naming, scheduling, protocol plan construction).
- **Refactor semantics engine:** `src/gabion/refactor/` (rewrite planning and edit synthesis driven by reified plans).
- **Governance/control-loop engine zone:** `src/gabion_governance/` (docflow + governance control-loop checks and normalized governance diagnostics).


## Ambiguity control matrix

| Pattern | Ambiguity admission zones | Deterministic core zones |
| --- | --- | --- |
| `isinstance` runtime narrowing | Allowed only to normalize incoming shape once at ingress | Forbidden as recurring control strategy; use reified contracts instead |
| `Optional` / `Union` / `Any` / `|` alternation | Allowed in boundary DTO/adapter normalization | Forbidden as unresolved downstream alternation in core semantics |
| Sentinel outcomes (`None`, empty sentinels, `pass`/`continue` fallthrough) | Allowed only inside explicit boundary normalizers | Forbidden for core control flow; use structural decision outcomes |

The boundary/core distinction above operationalizes
[`NCI-SHIFT-AMBIGUITY-LEFT`](docs/normative_clause_index.md#clause-shift-ambiguity-left).

## Boundary handoff contract
Only **Tier-1 reified objects** cross from ambiguity zones into deterministic core zones.

- Inputs must be promoted to typed DTO/model/config carriers before they enter `src/gabion/server_core/`, `src/gabion/analysis/`, `src/gabion/synthesis/`, or `src/gabion/refactor/`.
- Acceptable carriers across `cli/lsp_client/server -> server_core` are JSON-RPC command payload DTOs, execution-plan dataclasses, and validated mapping carriers with normalized key order.
- `src/gabion/server_core/` may orchestrate command execution but must not ingest ad-hoc untyped transport payloads directly from shell/env ingress.
- Acceptable carriers across semantic core (`src/gabion/server_core/`, `src/gabion/analysis/`, `src/gabion/synthesis/`, `src/gabion/refactor/`) and governance/docflow core (`src/gabion_governance/`) are explicit audit/report artifacts and validated governance-rule DTOs; direct mutation coupling between these zones is out-of-contract.
- Allowed cross-boundary forms are explicit objects (for example, schema DTOs/config dataclasses), not raw ad-hoc dictionaries or free-form tuples.
- Review and tooling should enforce by package scope:
  - outer adapters: `src/gabion/cli.py`, `src/gabion/lsp_client.py`, `src/gabion/server.py`, `src/gabion/schema.py`, `src/gabion/json_types.py`
  - deterministic command/semantic core: `src/gabion/server_core/`, `src/gabion/analysis/`, `src/gabion/synthesis/`, `src/gabion/refactor/`
  - governance/docflow core: `src/gabion_governance/`

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

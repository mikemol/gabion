---
doc_revision: 19
reader_reintern: "Reader-only: re-intern if doc_revision changed since you last read this doc."
doc_id: agents
doc_role: agent
doc_scope:
  - repo
  - agents
  - tooling
doc_authority: normative
doc_requires:
  - README.md#repo_contract
  - CONTRIBUTING.md#contributing_contract
  - POLICY_SEED.md#policy_seed
  - glossary.md#contract
doc_reviewed_as_of:
  README.md#repo_contract: 1
  CONTRIBUTING.md#contributing_contract: 1
  POLICY_SEED.md#policy_seed: 1
  glossary.md#contract: 1
doc_review_notes:
  README.md#repo_contract: "Reviewed README.md rev1 (docflow audit now scans in/ by default); no conflicts with this document's scope."
  CONTRIBUTING.md#contributing_contract: "Reviewed CONTRIBUTING.md rev1 (docflow now fails on missing GH references for SPPF-relevant changes); no conflicts with this document's scope."
  POLICY_SEED.md#policy_seed: "Reviewed POLICY_SEED.md rev1 (mechanized governance default; branch/tag CAS + check-before-use constraints); no conflicts with this document's scope."
  glossary.md#contract: "Reviewed glossary.md#contract rev1 (glossary contract + semantic typing discipline)."
doc_sections:
  agent_obligations: 1
doc_section_requires:
  agent_obligations:
    - README.md#repo_contract
    - CONTRIBUTING.md#contributing_contract
    - POLICY_SEED.md#policy_seed
    - glossary.md#contract
doc_section_reviews:
  agent_obligations:
    README.md#repo_contract:
      dep_version: 1
      self_version_at_review: 1
      outcome: no_change
      note: "Repo contract reviewed; agent obligations unchanged."
    CONTRIBUTING.md#contributing_contract:
      dep_version: 1
      self_version_at_review: 1
      outcome: no_change
      note: "Contributor contract reviewed; agent obligations unchanged."
    POLICY_SEED.md#policy_seed:
      dep_version: 1
      self_version_at_review: 1
      outcome: no_change
      note: "Policy seed reviewed; agent obligations unchanged."
    glossary.md#contract:
      dep_version: 1
      self_version_at_review: 1
      outcome: no_change
      note: "Glossary contract reviewed; agent obligations unchanged."
doc_change_protocol: "POLICY_SEED.md#change_protocol"
doc_invariants:
  - read_policy_glossary_first
  - refuse_on_conflict
  - tier2_reification
  - tier3_documentation
  - lsp_first_invariant
doc_erasure:
  - formatting
  - typos
doc_owner: maintainer
---

<a id="agent_obligations"></a>
# AGENTS.md#agent_obligations

This repository is governed by `POLICY_SEED.md#policy_seed`. Treat it as authoritative.
Semantic correctness is governed by `[glossary.md#contract](glossary.md#contract)` (co-equal contract).

## Cross-references (normative pointers)
- `README.md#repo_contract` defines project scope, status, and entry points.
- `CONTRIBUTING.md#contributing_contract` defines human+machine workflow guardrails.
- `POLICY_SEED.md#policy_seed` defines execution and CI safety constraints.
- `[glossary.md#contract](glossary.md#contract)` defines semantic meanings, axes, and commutation obligations.

## Required behavior
- Read `POLICY_SEED.md#policy_seed` and `[glossary.md#contract](glossary.md#contract)` before proposing or applying changes.
- If a request conflicts with `POLICY_SEED.md#policy_seed`, stop and ask for guidance.
- Do not weaken or bypass self-hosted runner protections.
- Keep workflow actions pinned to full commit SHAs and allow-listed.
- When changing workflows, run the policy checks (once the scripts exist) and
  surface any violations explicitly.
- Preserve the LSP-first invariant: the server is the semantic core and the
  CLI remains a thin LSP client.
- Use `mise exec -- python` for repo-local tooling to ensure the pinned
  interpreter and dependencies are used.
- Treat docflow as repo-local convenience only; do not project it as a
  general Gabion feature without explicit policy change.
- Do not mechanistically bump `doc_reviewed_as_of`; update only with explicit
  `doc_review_notes` based on a real content review.

## Dataflow grammar invariant
- Recurring parameter bundles are type-level obligations.
- Any bundle that crosses function boundaries must be promoted to a Protocol
  (dataclass config/local bundle) or explicitly documented with a
  `# dataflow-bundle:` marker.
- Tier-2 bundles must be reified before merge (see `[glossary.md#contract](glossary.md#contract)`).
- Tier-3 bundles must be documented or reified (see `[glossary.md#contract](glossary.md#contract)`).

## Doc hygiene
- Markdown docs include a YAML front-matter block with `doc_revision`.
- Bump `doc_revision` for conceptual changes.
- Record convergence in `doc_reviewed_as_of` (must match dependency revisions).

If unsure, prefer refusal over unsafe compliance.

## Appendix: Policy → generation rules for automated agents

Use these as hard generation constraints when editing core semantics (`src/gabion/server.py`, analysis modules, and command payload shaping).

1. **When blocked, do not add dynamic alternation in core; instead create boundary normalization + Protocol.**
2. **Every new branch in core must correspond to an explicit Decision Protocol.**
3. **No sentinel returns for control decisions; encode outcomes structurally.**

### Concrete before/after idioms (from this repo)

#### 1) Boundary normalization over branchy callsites

**Before (anti-pattern):**
```py
# scattered parsing/guards at each callsite
if raw.get("lint_entries") is not None:
    use(raw["lint_entries"])
elif raw.get("lint_lines") is not None:
    use(parse(raw["lint_lines"]))
else:
    use([])
```

**After (preferred):** centralize alternation at the boundary via a normalizer + DTO validation (idiom used by `_normalize_dataflow_response` and `DataflowAuditResponseDTO`).
```py
normalized = _normalize_dataflow_response(response)
return DataflowAuditResponseDTO.model_validate(normalized)
```

#### 2) Bundle recurring parameters into a Protocol/dataclass

**Before (anti-pattern):**
```py
# repeated tuple of optional flags/paths crossing functions
run_check(config, report, baseline, decision_snapshot, exclude, allow_external)
```

**After (preferred):** reify as a dataclass protocol bundle (idiom used by `DataflowPayloadCommonOptions`, `CheckDeltaOptions`, `DataflowFilterBundle`).
```py
opts = DataflowPayloadCommonOptions(...)
payload = build_check_payload(opts)
```

#### 3) Branches become explicit Decision Protocols

**Before (anti-pattern):**
```py
if emit_state and state_path is not None:
    raise BadParameter(...)
if emit_delta and write_baseline:
    raise BadParameter(...)
# more ad-hoc branches in each command handler
```

**After (preferred):** collect branch invariants in one explicit validator protocol (idiom used by `CheckDeltaOptions.validate`).
```py
options = CheckDeltaOptions(...)
options.validate()  # single decision surface for contradictory flags
```

#### 4) Structural outcomes over sentinel control values

**Before (anti-pattern):**
```py
parsed = _parse_lint_line(line)
if parsed is None:
    # implicit control branch from sentinel
    continue
```

**After (preferred):** return a tagged/typed outcome so control is explicit.
```py
outcome = parse_lint_line_structured(line)
if outcome.kind is ParseKind.SKIP:
    continue
entries.append(outcome.entry)
```

When introducing new logic, prefer adding a small Protocol/dataclass at boundaries instead of widening dynamic `if/elif` trees inside core analysis paths.


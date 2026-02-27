---
doc_revision: 27
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
  - docs/normative_clause_index.md#normative_clause_index
doc_reviewed_as_of:
  README.md#repo_contract: 2
  CONTRIBUTING.md#contributing_contract: 2
  POLICY_SEED.md#policy_seed: 2
  glossary.md#contract: 1
  docs/normative_clause_index.md#normative_clause_index: 2
doc_review_notes:
  README.md#repo_contract: "Reviewed README.md rev2 (removed stale ASPF action-plan CLI/examples; continuation docs now state/delta only)."
  CONTRIBUTING.md#contributing_contract: "Reviewed CONTRIBUTING.md rev2 (two-stage dual-sensor cadence, correction-unit validation stack, and strict-coverage trigger guidance)."
  POLICY_SEED.md#policy_seed: "Reviewed POLICY_SEED.md rev2 (forward-remediation order, ci_watch failure-bundle durability, and enforced execution-coverage policy wording)."
  glossary.md#contract: "Reviewed glossary.md#contract rev1 (glossary contract + semantic typing discipline)."
  docs/normative_clause_index.md#normative_clause_index: "Reviewed normative_clause_index rev2 (extended existing dual-sensor/shift-ambiguity/deadline clauses without introducing new clause IDs)."
doc_sections:
  agent_obligations: 2
doc_section_requires:
  agent_obligations:
    - README.md#repo_contract
    - CONTRIBUTING.md#contributing_contract
    - POLICY_SEED.md#policy_seed
    - glossary.md#contract
    - docs/normative_clause_index.md#normative_clause_index
doc_section_reviews:
  agent_obligations:
    README.md#repo_contract:
      dep_version: 2
      self_version_at_review: 2
      outcome: no_change
      note: "Repo contract rev2 reviewed; command and artifact guidance remains aligned."
    CONTRIBUTING.md#contributing_contract:
      dep_version: 2
      self_version_at_review: 2
      outcome: no_change
      note: "Contributor contract rev2 reviewed; dual-sensor cadence and correction gates remain aligned."
    POLICY_SEED.md#policy_seed:
      dep_version: 2
      self_version_at_review: 2
      outcome: no_change
      note: "Policy seed rev2 reviewed; governance obligations remain aligned."
    glossary.md#contract:
      dep_version: 1
      self_version_at_review: 2
      outcome: no_change
      note: "Glossary contract reviewed; agent obligations unchanged."
    docs/normative_clause_index.md#normative_clause_index:
      dep_version: 2
      self_version_at_review: 2
      outcome: no_change
      note: "Clause index rev2 reviewed; canonical clause references remain aligned."
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
- `docs/normative_clause_index.md#normative_clause_index` defines stable clause IDs for repeated obligations.
- `docs/shift_ambiguity_left_protocol.md#shift_ambiguity_left_protocol` defines the compact refactor sequence used under ambiguity pressure.

## Required behavior
- Read `POLICY_SEED.md#policy_seed` and `[glossary.md#contract](glossary.md#contract)` before proposing or applying changes.
- If a request conflicts with `POLICY_SEED.md#policy_seed`, stop and ask for guidance.
- Do not weaken or bypass self-hosted runner protections.
- Action pinning: [`NCI-ACTIONS-PINNED`](docs/normative_clause_index.md#clause-actions-pinned).
- Action allow-list: [`NCI-ACTIONS-ALLOWLIST`](docs/normative_clause_index.md#clause-actions-allowlist).
- When changing workflows, run the policy checks (once the scripts exist) and
  surface any violations explicitly.
- Preserve [`NCI-LSP-FIRST`](docs/normative_clause_index.md#clause-lsp-first).
- Enforce [`NCI-SHIFT-AMBIGUITY-LEFT`](docs/normative_clause_index.md#clause-shift-ambiguity-left) in semantic core refactors.
- Enforce command maturity/carrier/parity policy: [`NCI-COMMAND-MATURITY-PARITY`](docs/normative_clause_index.md#clause-command-maturity-parity).
- Enforce controller-drift override lifecycle policy: [`NCI-CONTROLLER-DRIFT-LIFECYCLE`](docs/normative_clause_index.md#clause-controller-drift-lifecycle).
- Enforce temporal dual-sensor correction loop policy: [`NCI-DUAL-SENSOR-CORRECTION-LOOP`](docs/normative_clause_index.md#clause-dual-sensor-correction-loop).
- Keep semantic behavior in server command handlers exposed via `gabion` subcommands; treat `scripts/` as orchestration wrappers only.
- Per-correction-unit validation stack must include `scripts/policy_check.py --workflows`, `scripts/policy_check.py --ambiguity-contract`, targeted pytest, and evidence-carrier drift refresh/check (`out/test_evidence.json`) when tests or semantic surfaces changed.
- Ambiguity-policy regressions encountered during simplification are forward-remediation signals; prefer boundary normalization/protocol reification over rollback-first.
- Reject semantic-core compatibility-layer additions (wrappers, dual-shape bridges, legacy fallbacks) unless they are temporary boundary adapters with explicit Decision Protocol plus lifecycle metadata (`actor`, `rationale`, `scope`, `start`, `expiry`, `rollback_condition`, `evidence_links`).
- When using `scripts/ci_watch.py`, treat collected failure bundles under `artifacts/out/ci_watch/run_<run_id>/` as the triage source of truth for remote-first actionable failures.
- Use `mise exec -- python` for repo-local tooling to ensure the pinned
  interpreter and dependencies are used. In CI, `.venv/bin/python` is acceptable
  after workflow bootstrap has installed the pinned toolchain and locked
  dependencies.
- Prefer impossible-by-construction contracts over sentinel parse outcomes;
  after ingress validation, invalid states must be discharged via `never()`.
- Treat docflow as repo-local convenience only; do not project it as a
  general Gabion feature without explicit policy change.
- Do not mechanistically bump `doc_reviewed_as_of`; update only with explicit
  `doc_review_notes` based on a real content review.

## Agent actioning loop (normative)
Canonical rule: [`NCI-DUAL-SENSOR-CORRECTION-LOOP`](docs/normative_clause_index.md#clause-dual-sensor-correction-loop).

1. Start local repro tooling and GitHub status-check monitoring concurrently whenever both are available.
2. Act on the first actionable failure signal from either sensor; do not serialize waiting for the other sensor once one signal is actionable.
3. Stage A (pre-signal): bounded dependency-cluster publication is allowed before actionable failures exist.
4. Stage B (post-signal): once an actionable signal exists, form one correction unit per push (one blocking surface, or tightly coupled set for one blocking surface).
5. Validate the correction unit locally with the required policy/ambiguity/targeted-test/evidence-drift stack.
6. Stage, commit, and push the correction unit immediately after local validation.
7. Resume dual-sensor monitoring and continue the detection/correction/push loop; treat fallout as later correction units.

If only one sensor is available, proceed with that sensor and restore dual-sensor operation when available.

## Dataflow grammar invariant
- Canonical rule: [`NCI-DATAFLOW-BUNDLE-TIERS`](docs/normative_clause_index.md#clause-dataflow-bundle-tiers).
- Tier-3 documentation marker: `# dataflow-bundle:`.

## Doc hygiene
- Markdown docs include a YAML front-matter block with `doc_revision`.
- Use front-matter dependency anchors (`doc_requires`) and relation edges when
  a document contributes to the dependency/index tree.
- Bump `doc_revision` for conceptual changes.
- Record convergence in `doc_reviewed_as_of` (must match dependency revisions).

If unsure, prefer refusal over unsafe compliance.

## Appendix: Policy → generation rules for automated agents

Use these as hard generation constraints when editing core semantics (`src/gabion/server.py`, analysis modules, and command payload shaping).

1. **When blocked, do not add dynamic alternation in core; instead create boundary normalization + Protocol.**
2. **Every new branch in core must correspond to an explicit Decision Protocol.**
3. **No sentinel returns for control decisions; encode outcomes structurally.**
4. **Do not preserve compatibility layers in semantic core; collapse to one deterministic contract and keep any temporary compatibility adapter at boundary ingress only with lifecycle evidence.**

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

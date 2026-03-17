---
doc_revision: 37
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
  - docs/clause_obligation_decks.yaml
doc_reviewed_as_of:
  README.md#repo_contract: 2
  CONTRIBUTING.md#contributing_contract: 2
  POLICY_SEED.md#policy_seed: 2
  glossary.md#contract: 1
  docs/normative_clause_index.md#normative_clause_index: 3
doc_review_notes:
  README.md#repo_contract: "Reviewed README.md#repo_contract rev84/section v2 (removed stale ASPF action-plan CLI/examples; continuation docs now state/delta only)."
  CONTRIBUTING.md#contributing_contract: "Reviewed CONTRIBUTING.md#contributing_contract rev120/section v2; clause-backed architectural-invariant bullets now render from the shared clause-obligation catalog while contributor workflow prose remains hand-authored."
  POLICY_SEED.md#policy_seed: "Reviewed POLICY_SEED.md#policy_seed rev57/section v2 (runtime is now process-relative across program, analysis, formal, planning, and governance workflows); the generated clause deck remains policy-compatible."
  glossary.md#contract: "Reviewed glossary.md#contract rev47/section v1 (runtime scope is process-relative and the distinction ladder remains part of the semantic contract) while clause-backed obligation bullets move to generated rendering."
  docs/normative_clause_index.md#normative_clause_index: "Reviewed docs/normative_clause_index.md#normative_clause_index rev18/section v3; generated clause-backed bullets now resolve canonical clause links from the clause index."
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
      note: "Contributor contract section v2 reviewed; generated clause-backed invariant bullets stay aligned with the agent-facing obligation deck."
    POLICY_SEED.md#policy_seed:
      dep_version: 2
      self_version_at_review: 2
      outcome: no_change
      note: "Policy seed rev57 reviewed; process-relative runtime fits existing agent obligations while clause-backed bullets move to generated rendering."
    glossary.md#contract:
      dep_version: 1
      self_version_at_review: 2
      outcome: no_change
      note: "Glossary rev46 reviewed; process-relative runtime and the distinction ladder remain part of semantic typing discipline while clause-backed bullets render from the shared catalog."
    docs/normative_clause_index.md#normative_clause_index:
      dep_version: 3
      self_version_at_review: 2
      outcome: no_change
      note: "Clause index section v3 reviewed; the generated obligation deck now resolves canonical clause links from the shared clause catalog."
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
- When changing workflows, run the policy checks (once the scripts exist) and
  surface any violations explicitly.

## Project quick start
- Use the root `AGENTS.md` as the workspace instruction surface; do not add a parallel `.github/copilot-instructions.md` for this repo.
- Bootstrap the repo with `make bootstrap`.
- If the environment is already bootstrapped, prefer `mise exec -- python -m pip install -e .` for package installs and `mise trust --yes` if `mise` reports an untrusted config.
- Prefer `mise exec -- python ...` for repo-local tooling; bare `python`, `pytest`, and ad-hoc virtualenv selection can drift from the pinned toolchain.
- Common local commands: `make test`, `make dataflow`, `make docflow`, and `make check`.

## Architecture quick reference
- `src/gabion/server.py` and `src/gabion/server_core/` own semantic command behavior and command orchestration.
- `src/gabion/cli.py` and `src/gabion/cli_support/` are thin clients over the semantic core; do not reimplement analysis logic there.
- `src/gabion/analysis/`, `src/gabion/synthesis/`, and `src/gabion/refactor/` are core logic zones that should consume normalized, typed inputs.
- `scripts/` are orchestration wrappers for bootstrap, CI reproduction, policy checks, and audits; keep user-facing semantics in `gabion` subcommands.
- For clause-backed workflow and implementation checks, start with `docs/enforceable_rules_cheat_sheet.md#enforceable_rules_cheat_sheet` and `docs/normative_clause_index.md#normative_clause_index`.

## Working defaults
- Normalize ambiguity at ingress. Semantic-core paths should prefer DTO validation, Protocol/dataclass reification, and explicit decision surfaces over repeated `if`/`elif` shape checks.
- Treat Markdown edits as governed surfaces: maintain frontmatter, bump `doc_revision` for conceptual changes, and keep `doc_review_notes` tied to real dependency reviews.
- When a task touches validation or governance outputs, inspect `artifacts/out/`, `artifacts/audit_reports/`, and `artifacts/test_runs/` before assuming the failure mode.
- For command or workflow work, inspect `README.md#repo_contract`, `CONTRIBUTING.md#contributing_contract`, and `docs/governance_control_loops.md#governance_control_loops` before changing behavior.
<!-- BEGIN:generated_agent_clause_obligations -->
_The clause-backed bullets below are generated from `docs/clause_obligation_decks.yaml` and `docs/normative_clause_index.md` via `mise exec -- python -m scripts.policy.render_clause_obligation_decks`._

- Action pinning: [`NCI-ACTIONS-PINNED`](docs/normative_clause_index.md#clause-actions-pinned).
- Action allow-list: [`NCI-ACTIONS-ALLOWLIST`](docs/normative_clause_index.md#clause-actions-allowlist).
- Preserve [`NCI-LSP-FIRST`](docs/normative_clause_index.md#clause-lsp-first).
- Enforce process-relative runtime policy: [`NCI-RUNTIME-PROCESS-RELATIVE`](docs/normative_clause_index.md#clause-runtime-process-relative).
- Enforce runtime distinction admissibility: [`NCI-RUNTIME-DISTINCTION-LADDER`](docs/normative_clause_index.md#clause-runtime-distinction-ladder).
- Enforce [`NCI-SHIFT-AMBIGUITY-LEFT`](docs/normative_clause_index.md#clause-shift-ambiguity-left) in semantic core refactors.
- Enforce command maturity/carrier/parity policy: [`NCI-COMMAND-MATURITY-PARITY`](docs/normative_clause_index.md#clause-command-maturity-parity).
- Enforce controller-drift override lifecycle policy: [`NCI-CONTROLLER-DRIFT-LIFECYCLE`](docs/normative_clause_index.md#clause-controller-drift-lifecycle).
- Enforce temporal dual-sensor correction loop policy: [`NCI-DUAL-SENSOR-CORRECTION-LOOP`](docs/normative_clause_index.md#clause-dual-sensor-correction-loop).
- Enforce correction-unit git drainage and commit-boundary policy: [`NCI-CORRECTION-UNIT-COMMIT-BOUNDARY`](docs/normative_clause_index.md#clause-correction-unit-commit-boundary).
- Enforce packetized docflow control-loop policy: [`NCI-DOCFLOW-CLOSED-LOOP`](docs/normative_clause_index.md#clause-docflow-closed-loop).
<!-- END:generated_agent_clause_obligations -->
- Treat coverage-gate drops as dedicated fix-forward correction-unit signals; do not use rollback-first reasoning when coverage regresses.
- Treat any GitHub API error during monitoring/forensics as a process-remediation signal for API access; do not respond with backoff-only behavior.
- When a workstream sets an API polling cadence cap, obey the cap and maximize data per query.
- Keep semantic behavior in server command handlers exposed via `gabion` subcommands; treat `scripts/` as orchestration wrappers only.
- Per-correction-unit validation stack must include `scripts/policy/policy_check.py --workflows`, `scripts/policy/policy_check.py --ambiguity-contract`, strict docflow (`python -m gabion docflow --root . --fail-on-violations --sppf-gh-ref-mode required`), docflow packet loop (`scripts/policy/docflow_packetize.py` + `scripts/policy/docflow_packet_enforce.py --check`), targeted pytest, and evidence-carrier drift refresh/check (`out/test_evidence.json`) when tests or semantic surfaces changed.
- When full pytest is red, use `UTR` as the cross-cutting unit-test current-indicator root and drain one touchpoint-sized correction unit at a time; keep the detailed shared loop in [`docs/unit_test_readiness_playbook.md`](docs/unit_test_readiness_playbook.md) rather than duplicating an agent-only variant.
- Ambiguity-policy regressions encountered during simplification are forward-remediation signals; prefer boundary normalization/protocol reification over rollback-first.
- Reject semantic-core compatibility-layer additions (wrappers, dual-shape bridges, legacy fallbacks) unless they are temporary boundary adapters with explicit Decision Protocol plus lifecycle metadata (`actor`, `rationale`, `scope`, `start`, `expiry`, `rollback_condition`, `evidence_links`).
- When using `scripts/ci/ci_watch.py`, treat collected failure bundles under `artifacts/out/ci_watch/run_<run_id>/` as the triage source of truth for remote-first actionable failures.
- Use `mise exec -- python` for repo-local tooling to ensure the pinned
  interpreter and dependencies are used. In CI, `.venv/bin/python` is acceptable
  after workflow bootstrap has installed the pinned toolchain and locked
  dependencies.
- Prefer impossible-by-construction contracts over sentinel parse outcomes;
  after ingress validation, invalid states must be discharged via `never()`.
- `# pragma: no cover` is permitted only when the corresponding branch is
  discharged by `never(...)`.
- Enum exhaustiveness fallbacks should pair explicit `never(...)` with
  `# pragma: no cover` on the dead post-invariant path.
- Treat docflow as repo-local convenience only; do not project it as a
  general Gabion feature without explicit policy change.
- Do not mechanistically bump `doc_reviewed_as_of`; update only with explicit
  `doc_review_notes` based on a real content review.

## Operational toggle visibility (informational)
This index is visibility-only; it keeps repo-local operational toggles explicit
for agent instruction drift audits.

- CLI/git/workflow toggles: `--force-with-lease`, `--close`, `--output`, `--synthesis-plan`, `--synthesis-report`, `--synthesis-protocols`, `--refactor-plan`, `--refactor-plan-json`, `--pr-base-sha`, `--pr-head-sha`, `--skip-sppf-sync`, `--run-sppf-sync`
- Environment and signal toggles: `GABION_SPPF_SYNC`, `GH_TOKEN`, `GITHUB_TOKEN`, `SIGUSR1`, `POLICY_GITHUB_TOKEN`

## Agent actioning loop (normative)
Canonical rule: [`NCI-DUAL-SENSOR-CORRECTION-LOOP`](docs/normative_clause_index.md#clause-dual-sensor-correction-loop).

1. Start local repro tooling and GitHub status-check monitoring concurrently whenever both are available.
2. Act on the first actionable failure signal from either sensor; do not serialize waiting for the other sensor once one signal is actionable.
3. Stage A (pre-signal): bounded dependency-cluster publication is allowed before actionable failures exist.
4. Stage B (post-signal): once an actionable signal exists, form one correction unit per push (one blocking surface, or tightly coupled set for one blocking surface).
5. Coverage regressions and API-access failures are Stage-B actionable signals and must trigger dedicated remediation units.
6. Validate the correction unit locally with the required policy/ambiguity/targeted-test/evidence-drift stack.
7. Stage, commit, and push the correction unit immediately after local validation. Drain the workspace after each correction unit. Prefer hunk-level staging/commits when a file contains multiple separable correction units; otherwise commit the whole coherent correction unit.
8. Resume dual-sensor monitoring and continue the detection/correction/push loop; treat fallout as later correction units.

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

**After (preferred):** centralize alternation at the boundary via a normalizer + DTO validation (idiom used by `_normalize_dataflow_response` and `LegacyDataflowMonolithResponseDTO`).
```py
normalized = _normalize_dataflow_response(response)
return LegacyDataflowMonolithResponseDTO.model_validate(normalized)
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

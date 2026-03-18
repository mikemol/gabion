---
doc_revision: 3
reader_reintern: "Reader-only: re-intern if doc_revision changed since you last read this doc."
doc_id: wrd_dx_friction_log
doc_role: audit
doc_scope:
  - repo
  - tooling
  - dx
doc_authority: informative
doc_requires:
  - POLICY_SEED.md#policy_seed
  - AGENTS.md#agent_obligations
  - CONTRIBUTING.md#contributing_contract
  - docs/planning_substrate.md#planning_substrate
doc_reviewed_as_of:
  POLICY_SEED.md#policy_seed: 2
  AGENTS.md#agent_obligations: 2
  CONTRIBUTING.md#contributing_contract: 2
  docs/planning_substrate.md#planning_substrate: 1
doc_review_notes:
  POLICY_SEED.md#policy_seed: "Friction log is informative/audit only; no fix-forward entries, no remediation rows."
  AGENTS.md#agent_obligations: "Notes are observations during WRD execution, not correction units."
  CONTRIBUTING.md#contributing_contract: "Observations captured in situ; no doc-revision bumps required for new entries."
  docs/planning_substrate.md#planning_substrate: "Notes include observations about registry and workstream patterns."
doc_change_protocol: "POLICY_SEED.md#change_protocol"
---

# WRD DX Friction Log

This document records friction observations made during WRD (Wrapper Retirement Drain) execution. Each note was captured at the moment of friction — during planning or execution — before proceeding. Notes are raw observations, not fix proposals; they are intended for later DX analysis.

LLM agents are treated as first-class developers for impact assessment purposes.

---

## FN-001: No 1:1 canonical successor for run-dataflow-stage

**Trigger:** Investigating WRD-TP-001 — what does "retiring" `run-dataflow-stage` actually mean in practice?

**Friction:** `gabion check {run, delta-bundle, delta-gates}` cover dataflow analysis, but the multi-stage retry orchestration (stage-sequence: run → retry1-5), ASPF handoff coordination, deadline scope management, and signal handlers with debug dumps have no canonical home. The WRD registry says "converge on canonical gabion command ownership" but does not specify what those canonical surfaces must provide before retirement can proceed.

**Impact:** Both human and LLM agent. A human reading the touchpoint reason gets actionable-sounding text but must independently assess whether parity exists. An LLM agent will look for the canonical command and find an incomplete equivalent, then face an unguided choice between deferral and gap-fill.

**Hypothesis:** Touchpoint `reason` text should reference the specific canonical target (command name, planned workstream item, or explicit capability gap list), or the `blocking_dependencies` in the marker `reasoning` should name the prerequisite gaps explicitly.

---

## FN-002: Registry touchsite line numbers are unstated-confidence coordinates

**Trigger:** Touchsite WRD-TP-001-TS-001 declares `run_dataflow_stage.py:1079`; WRD-TP-002-TS-003 declares `ci_watch.py:559`.

**Friction:** These are static declarations. The files have changed since they were recorded. There is no automated check that declared line numbers still point at the intended symbols. Finding the actual current line requires independently reading the file.

**Impact:** LLM agents especially — they will trust the coordinates and read displaced context without noticing. Humans will notice the visual offset; agents may not.

**Hypothesis:** Either set `scan_touchsites=True` on WRD touchpoints (the schema supports it but WRD does not use it), or adopt symbol-name coordinates instead of line numbers for non-scanned touchpoints.

---

## FN-003: Landing language rules exist only in validation source

**Trigger:** Need to write closed `reason` + `summary` text for landing a touchpoint. What phrasing is allowed?

**Friction:** The allowed/forbidden token lists (`_LANDED_REASON_REQUIRED_TOKENS`, `_LANDED_REASON_FORBIDDEN_TOKENS`) are in `workstream_registry.py`. There is no doc-surface summary. A developer will only discover the rules after a validation failure.

**Impact:** Both, but especially LLM agents who may not proactively read the validation source before authoring closure text.

**Hypothesis:** Include a concise reference in the workstream closure guidance doc (or as a `## Closure language` section at the bottom of `workstream_registry.py` docstring) showing required tokens and examples of correct landed phrasing.

---

## FN-004: Landed hierarchy constraint is all-or-nothing within a branch

**Trigger:** Planning to land WRD-TP-001 while WRD-TP-002 and WRD-TP-003 remain `queued` in the same subqueue (WRD-SQ-001).

**Friction:** `validate_workstream_closure_consistency` forbids a `landed` subqueue from having non-landed touchpoints. WRD-SQ-001 cannot close until TP-001, TP-002, and TP-003 are all landed. This constraint is not stated anywhere near the subqueue definition — it is only discoverable by reading the validator source or receiving a validation error.

**Impact:** Both. Without knowing this rule, a developer may think landing TP-001 closes SQ-001 and plan work accordingly.

**Hypothesis:** The `touchpoint_ids` tuple in a subqueue definition implicitly encodes a completion dependency tree that is not mirrored in the marker reasoning chain. Making this explicit — either by surfacing it in `blockers` query output, or by noting it in registry authoring guidance — would prevent planning errors.

---

## FN-005: 551KB invariant_graph.py is a substantial orientation barrier

**Trigger:** Trying to understand how `declared_workstream_registries()` assembles registries, and where to find the WRD registry integration point.

**Friction:** The function of interest is at approximately line 9207 of a 551KB file. Finding it requires knowing the exact symbol name to search for. There is no module-level summary or navigation aid.

**Impact:** High for LLM agents — context window pressure, high token cost to read, and navigation overhead. High for new human contributors — requires jump-to-definition or grep to navigate.

**Hypothesis:** A module-level docstring at the top of `invariant_graph.py` listing the major public function families (with brief descriptions) would significantly reduce orientation cost.

---

## FN-006: No workstream closure protocol document exists

**Trigger:** Trying to understand the sequence of actions needed to close a WRD touchpoint — which tests to run, which policy checks apply, what artifact refresh is required.

**Friction:** The Unit Test Readiness playbook (`docs/unit_test_readiness_playbook.md`) covers UTR closure. Nothing covers WRD or wrapper retirement specifically. The general protocol must be assembled from AGENTS.md + CONTRIBUTING.md + validator source.

**Impact:** Both. LLM agents especially will attempt to author a closure without knowing the complete required validation stack.

**Hypothesis:** Either a general "workstream closure protocol" doc covering the per-touchpoint and per-subqueue closure sequence, or a brief `## Closure checklist` section at the bottom of each registry module, would close this gap.

---

## FN-007: ci.yml run-dataflow-stage invocation is not declared as a WRD touchsite

**Trigger:** Attempting to plan the ci.yml change needed before WRD-TP-001 can land (can't add hard-fail stub while CI still calls the passthrough).

**Friction:** `.github/workflows/ci.yml` line 51 invokes `gabion run-dataflow-stage` directly. This is not declared as a touchsite in any WRD touchpoint. WRD-TP-001's touchsites cover the wrapper main function, the CLI passthrough, and the user_workflows.md doc — but not the live CI invocation. WRD-TP-008-TS-004 references ci.yml at line 154 (the `checks` job), not line 51 (the `dataflow-grammar` job).

**Impact:** Both. An executor of WRD-TP-001 who follows only the declared touchsites will add a hard-fail stub, which will then immediately break CI because the undeclared invocation still calls the wrapper. Discovering this requires independently reading ci.yml.

**Hypothesis:** The ci.yml invocation should be declared as a touchsite on WRD-TP-001 (the most direct retirement touchpoint) or as an additional touchsite on WRD-TP-008 (the capstone). Either way, the gap between declared touchsites and actual live call sites should not require independent file reading to discover.

---

## FN-008: run-dataflow-stage GitHub Actions output contract is not documented

**Trigger:** Investigating what the canonical replacement for `run-dataflow-stage` in ci.yml would need to provide.

**Friction:** `run-dataflow-stage` emits step outputs to `$GITHUB_OUTPUT`: `exit_code`, `analysis_state`, `terminal_stage`, `terminal_status`, `attempts_run`. These are consumed by the finalization step (`ci_finalize_dataflow_outcome.py`) via environment variables. This output contract is not documented anywhere — not in the wrapper module, not in the CI workflow comments, not in any DX doc. Discovering it required reading the `main()` function in `run_dataflow_stage.py`, the ci.yml finalization step, and the finalization script.

**Impact:** High for both. Anyone replacing `run-dataflow-stage` must discover and replicate or update this interface independently. An LLM agent may not notice that the canonical replacement (`gabion check delta-bundle`) does not emit these outputs, leading to silent failure of the finalization step.

**Hypothesis:** A comment block in the ci.yml `dataflow-grammar` job, or a docstring in `run_dataflow_stage.py:main`, should document the step output contract and state whether it is a requirement of the canonical replacement or a wrapper-only artifact.

---

## FN-009: ci_finalize_dataflow_outcome.py has hidden coupling to run-dataflow-stage's output interface

**Trigger:** Checking whether `gabion check delta-bundle` can directly replace `run-dataflow-stage` in ci.yml.

**Friction:** `ci_finalize_dataflow_outcome.py` first tries to read `artifacts/audit_reports/dataflow_terminal_outcome.json`. If that file exists, the step outputs aren't needed. But `gabion check delta-bundle` (via `run_check_command` in `check_command_runtime.py`) does not write this file — only `run-dataflow-stage` does (via `_write_terminal_outcome_artifact`). So replacing `run-dataflow-stage` with `gabion check delta-bundle` requires either: (a) making `run_check_command` write the terminal outcome artifact, or (b) updating the finalization step to compute the outcome from the run's exit code directly. Neither path is obvious from the touchpoint definition or the registry reason text.

**Impact:** High for both. The finalization step will silently miscalculate (using empty `TERMINAL_EXIT` default) or fail outright unless the coupling is resolved. An LLM agent following the WRD-TP-001 touchpoint description would not know to investigate this dependency.

**Hypothesis:** Either `run_check_command` should write the terminal outcome artifact (closing the output parity gap), or the finalization script's dependency on the step output interface should be surfaced as an explicit touchsite in WRD-TP-001 or a blocking dependency in its marker reasoning.

---

## FN-010: No `--timeout` flag support in pytest

**Trigger:** Trying to run a targeted pytest with a timeout guard.

**Friction:** `mise exec -- python -m pytest ... --timeout=60` fails with "unrecognized arguments: --timeout=60". The `pytest-timeout` plugin is not installed.

**Impact:** Minor for humans (just omit the flag). Potentially confusing for LLM agents that try to add timeout protection.

**Hypothesis:** If pytest-timeout is not a desired dependency, a note in CONTRIBUTING.md or AGENTS.md about not using `--timeout` would prevent the confusion.

---

## FN-011: Touchsite TS-003 (user_workflows.md) does not reference the wrapper being retired

**Trigger:** Trying to determine what change is needed to user_workflows.md for WRD-TP-001 retirement.

**Friction:** The touchsite `WRD-TP-001-TS-003` declares `docs/user_workflows.md:user_workflows:56` as a `wrapper_runtime_entrypoint`. The actual file does not mention `run-dataflow-stage` at all — it already shows `gabion check run` and `gabion check delta-bundle` as the canonical CLI. The `scripts/misc/aspf_handoff.py` reference at line 182 is scoped to WRD-TP-003, not WRD-TP-001. Without independently reading the file, there's no way to know whether TS-003 represents work to be done or a boundary that has already converged.

**Impact:** Both. An executor of WRD-TP-001 checking TS-003 will read the doc looking for `run-dataflow-stage` references to remove, find none, and have no signal about whether the touchsite is already converged.

**Hypothesis:** Touchsite declarations should distinguish between "this surface still contains the pattern to be retired" vs "this is a surface that must be verified post-retirement to confirm it references the canonical successor". A `status_hint` of `"converged"` or `"verified"` distinct from `"landed"` might help, or the `seam_class` annotation could express this distinction.

---

## FN-012: Pre-existing test failures create validation ambiguity

**Trigger:** Running `python -m pytest tests/gabion/tooling/runtime_policy/ -x` and observing a failure in `test_perf_artifact.py::test_build_cprofile_perf_artifact_payload_enriches_structural_identity_from_graph`.

**Friction:** The failure is pre-existing (the test file is unchanged from HEAD and has status "unmapped" in `out/test_evidence.json`), but distinguishing it from a regression introduced by the current changes required: checking `git status` of the test file, inspecting the test evidence JSON, and confirming the test node ID is not in the modified list. An executor who doesn't know to do these checks would spend time investigating a false alarm.

**Impact:** Both. An LLM agent running the validation stack will see a failure, be unable to distinguish pre-existing from new, and may incorrectly attribute it to the current changes or spend investigation effort on a known state.

**Hypothesis:** The validation stack should either (a) run only tests that are known-passing against the current HEAD baseline (via `pytest --lf` or a test exclusion marker), or (b) the test evidence status for pre-existing failures should be surfaced in the validation output so the executor can immediately distinguish new failures from known states.

---

## FN-013: Capability token matching requires knowing which surfaces share capabilities

**Trigger:** Updating `workflow_checks_capabilities.docflow_packet_loop.source_alternatives` from `"scripts/policy/docflow_packetize.py"` to shorter tokens, and discovering 3–4 debug/search cycles later that the shared capability was still checking against ci.yml (which has `"gabion docflow-packetize"` contiguously) vs `checks_runtime.py` (which has `"gabion"` and `"docflow-packetize"` in separate tuple elements).

**Friction:** `_CiReproCapabilitySpec.source_alternatives` checks `all(token in source_text for token in group)` against raw file text. The same capability spec is reused across multiple surfaces with different source texts (ci.yml YAML, `checks_runtime.py`, `ci_local_repro.py`). The token format that matches ci.yml (`"gabion docflow-packetize"`) does NOT match Python source (where `"gabion"` and `"docflow-packetize"` are separate string literals). There is no documented guidance on how to write capability tokens that match across both source types. Discovery required trial-and-error with debug prints added to the test.

**Impact:** Both. An LLM agent updating the capability specs has no way to predict which token granularity will work for all source surfaces without reading both the YAML and the Python source. A human developer would hit the same problem without running the test.

**Hypothesis:** The `_CiReproCapabilitySpec` design should document whether tokens must be contiguous substrings of each source format. A single canonical token (the command name, e.g. `"docflow-packetize"`) that appears in both source formats would reduce this friction, or the spec should require a separate token group per source format to make the expectation explicit.

---

## Resolutions

*Resolution notes are captured here separately to preserve the raw-observation format of friction notes above.*

### Resolution: FN-009 (WRD-TP-001)

Added terminal outcome artifact writing to `run_check_command` in `check_command_runtime.py`. The artifact is written adjacent to `--report` when that flag is provided. The ci.yml finalization step was updated to use `${{ steps.dataflow_stage.outcome }}` as a fallback when the artifact is absent.

## FN-014: Flat-command migration drift hid the real namespace-family target

**Trigger:** Re-entering WRD to close it and discovering a local tree that already migrated many wrappers to flat commands like `gabion policy-check` and `gabion aspf-handoff`.

**Friction:** The repo had three overlapping command-shape stories at once: legacy script wrappers, partially published flat `gabion` commands, and a planning-substrate intent to converge on namespace families. Because the declared registry seam lived inside `invariant_graph.py`, the WRD registry was easy to update locally without re-reading the broader command topology. The result was premature `landed` markers that reflected an intermediate migration shape rather than the chosen end state.

**Impact:** Both. A human can eventually notice the mismatch by reading CLI help, docs, and workflows together. An LLM agent is especially vulnerable because the local planning substrate and some docs appeared to authorize the flat commands as complete.

**Hypothesis:** Command-surface migrations should not be marked landed from inside a monolithic registry assembly file. A dedicated declared-registry catalog plus explicit command-shape guidance in the WRD registry would make the intended end state harder to misread.

## FN-015: CLI publication drift let docs reference commands that were never actually exported

**Trigger:** Verifying the local WRD claim that `gabion aspf-handoff` already existed as the canonical surface.

**Friction:** Active docs and workflows referenced `gabion aspf-handoff`, but the CLI did not actually publish that top-level command. Some wrapper replacements were only registered in `_TOOLING_ARGV_RUNNERS`, some were real Typer commands, and some were just doc-level aspirations. Discovering the mismatch required checking CLI help and code together; reading only docs or only the runtime shim modules was insufficient.

**Impact:** Both. A human loses time reconciling docs against actual CLI help. An LLM agent can confidently “migrate” more surfaces onto a command that does not exist.

**Hypothesis:** Canonical command publication should have one source of truth, and WRD should treat “documented but unpublished” as an explicit migration failure mode rather than as converged state.

## FN-016: Workflow guardrails and WRD closure state drifted independently

**Trigger:** Re-cutting the flat-command migration to namespace families and finding that the operator docs and workflows had already moved, while `policy_check` workflow assertions and the WRD registry still encoded the older flat/script-path surface.

**Friction:** Three authority layers had to be reconciled by hand: the active workflow/doc invocation text, the workflow-policy validator in `scripts/policy/policy_check.py`, and the planning-substrate closure state in `wrapper_retirement_drain_registry.py`. Any one of the three could make the repo look converged in isolation while the others still described an obsolete command surface.

**Impact:** Both. Humans lose time triangulating which layer is stale. LLM agents are likely to stop after fixing whichever layer first produces a green local signal, leaving the others behind.

**Hypothesis:** Command-surface migrations should have one authoritative publication catalog that both workflow policy checks and WRD closure state derive from, rather than re-encoding the command family in each layer.

---
doc_revision: 22
reader_reintern: "Reader-only: re-intern if doc_revision changed since you last read this doc."
doc_id: sppf_checklist
doc_role: checklist
doc_scope:
  - repo
  - planning
  - tooling
doc_authority: informative
doc_requires: []
doc_change_protocol: "POLICY_SEED.md §6"
doc_erasure:
  - formatting
  - typos
doc_owner: maintainer
---

# SPPF Convergence Checklist (Bottom-Up)

This checklist is the bottom-up complement to the governance layer. It captures
concept nodes derived from `in/` and tracks whether they are adopted, planned,
or explicitly out of scope. It is advisory only.

Tooling axis note: entries that touch scripts, Make targets, or execution
wrappers live on the `tooling` axis and should stay consistent with
`CONTRIBUTING.md` and `README.md`.

## Analysis pipeline nodes
- [x] Import resolution / symbol table (deterministic callee resolution).
- [x] Alias-aware identity tracking (rename morphisms preserved).
- [x] Noise controls: project root anchoring + exclude dirs + ignore params.
- [x] External-lib filter (ignore non-project imports by default).
- [x] Wildcard forwarding strictness (`*args/**kwargs`, high/low modes).
- [x] Type-flow tightening audit (downstream annotations).
- [x] Constant-flow audit (dead knobs / always-constant params).
- [x] Unused-argument pass detection (non-test call sites).

## Synthesis + refactoring nodes
- [x] Protocol/dataclass synthesis (tier thresholds, field typing) (prototype).
- [x] Naming heuristics (frequency-based) (prototype).
- [x] Topological refactoring schedule (callee-first order) (prototype).
- [x] Partial-application merge heuristics (bundle fragmentation control) (prototype).
- [ ] LLM-ready naming stubs (TODO_Name_Me + context docstrings).
- [ ] Type aggregation for synthesis (from type-audit + call-site evidence).
- [ ] Const/default-aware partial-application detection (subset merge by knobs).

## LSP operational semantics
- [x] CLI as pure LSP client (no engine import; server-only logic).
- [x] Server as single source of truth for diagnostics/code actions (basic).
- [x] Analysis→diagnostics mapping (basic placeholder ranges).
- [x] LSP executeCommand smoke test (dataflow command).
- [x] Pytest wrapper for LSP smoke test (skips if pygls missing).
- [x] VS Code extension as thin wrapper (spawn server only).

## Governance/ops glue (optional)
- [x] Durable logs/artifacts guidance for audits.
- [x] Audit snapshot tooling (dataflow report + DOT + docflow summary).
- [x] Repo config defaults (`gabion.toml`) for dataflow settings.
- [x] Policy check script references.
- [x] Hook installer.
- [x] Doer/Judge/Witness framing (optional).

## Explicit non-goals
- [x] Agda proof kernel (deferred).
- [x] GPU/JAX/Prism VM operational guidance (out of scope).

## Phase 2: Integration (post-scaffold)
- [x] Synthesis plan available via LSP/CLI (`gabion.synthesisPlan`, `synthesis-plan`).
- [x] Synthesis payload schema doc (`docs/synthesis_payload.md`).
- [x] Dataflow audit can emit synthesis plan outputs (report + JSON).
- [x] Protocol/dataclass stub emitter (writes to `artifacts/`).
- [x] Refactoring plan output (per-bundle schedule).
- [x] `gabion synth` command to run audit + synthesis in one step.

## Phase 3: Refactoring & UX
- [ ] LibCST refactor engine scaffolding (preserve trivia/formatting).
- [ ] LSP code action + workspace edit stub for Protocol extraction.
- [ ] Precise diagnostic ranges (metadata-backed positions).
- [ ] Long-lived LSP server cache / incremental analysis (daemon mode).

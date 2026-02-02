---
doc_revision: 29
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

Legend: [x] done · [ ] planned · [~] partial/heuristic

## Analysis pipeline nodes
- [~] Import resolution / symbol table (deterministic callee resolution).
- [x] Import resolution: explicit imports + relative import handling.
- [ ] Import resolution: star-import expansion.
- [ ] Import resolution: re-exports / `__all__` handling.
- [ ] Import resolution: class hierarchy dispatch resolution.
- [~] Alias-aware identity tracking (rename morphisms preserved).
- [x] Alias tracking: direct Name-to-Name assignments.
- [ ] Alias tracking: tuple/sequence unpacking.
- [ ] Alias tracking: attribute/collection aliasing (obj.attr, dict["key"]).
- [ ] Alias tracking: alias propagation via call returns.
- [x] Noise controls: project root anchoring + exclude dirs + ignore params.
- [x] External-lib filter (ignore non-project imports by default).
- [~] Wildcard forwarding strictness (`*args/**kwargs`, high/low modes).
- [x] Wildcard forwarding: detect direct starred name in call site.
- [ ] Wildcard forwarding: signature-aware mapping for `*args/**kwargs`.
- [ ] Wildcard forwarding: pass-through via `*args/**kwargs` variables.
- [x] Type-flow tightening audit (downstream annotations).
- [x] Constant-flow audit (dead knobs / always-constant params).
- [x] Unused-argument pass detection (non-test call sites).

## Reporting & visualization nodes
- [x] Component isolation (connected components in bundle graph).
- [x] Mermaid component diagrams embedded in Markdown report.
- [x] DOT/Graphviz output for bundle graphs.
- [x] Tiered bundle classification (declared vs. observed) + violation listing.
- [~] Bundle declaration sources (Config dataclasses, `dataflow-bundle` markers, dataclass calls).
- [x] Bundle declarations: `dataflow-bundle` markers.
- [x] Bundle declarations: local dataclass constructor calls (Name-only args).
- [ ] Bundle declarations: general dataclass fields beyond `_fn` convention.
- [ ] Bundle declarations: non-Name args/kwargs in dataclass calls.
- [ ] Bundle declarations: external dataclass modules (cross-file).

## Synthesis + refactoring nodes
- [~] Protocol/dataclass synthesis (tier thresholds, field typing) (prototype).
- [x] Synthesis output: dataclass stubs with field typing.
- [ ] Synthesis output: typing.Protocol interface stubs.
- [ ] Synthesis typing: resolve conflicts into `Union`/`Optional`.
- [x] Naming heuristics (frequency-based) (prototype).
- [~] Topological refactoring schedule (callee-first order) (prototype).
- [x] Refactor schedule: basic topological order.
- [ ] Refactor schedule: SCC-based cycle detection (explicit knots).
- [~] Partial-application merge heuristics (bundle fragmentation control) (prototype).
- [x] Bundle merge heuristic: Jaccard overlap merge function.
- [ ] Bundle merge heuristic: integrated into synthesis/refactor pipeline.
- [x] LLM-ready naming stubs (TODO_Name_Me + context docstrings).
- [~] Type aggregation for synthesis (from type-audit + call-site evidence).
- [x] Type aggregation: single-type consensus assignment.
- [ ] Type aggregation: conflict resolution into `Union`/`Any` guidance.
- [ ] Const/default-aware partial-application detection (subset merge by knobs).

## LSP operational semantics
- [x] CLI as pure LSP client (no engine import; server-only logic).
- [x] Server as single source of truth for diagnostics/code actions (basic).
- [x] Analysis→diagnostics mapping (param-span ranges).
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
- [x] LibCST refactor engine scaffolding (preserve trivia/formatting).
- [~] LSP code action + workspace edit stub for Protocol extraction.
- [x] LSP code action: stub command wiring.
- [ ] LSP workspace edit: real edits for Protocol extraction.
- [x] Precise diagnostic ranges (metadata-backed positions).
- [ ] Long-lived LSP server cache / incremental analysis (daemon mode).

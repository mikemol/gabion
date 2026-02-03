---
doc_revision: 73
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

## GitHub tracking linkage
- Each `[ ]` or `[~]` node should have a corresponding GitHub issue created from the
  `SPPF node` issue form.
- Once the issue exists, append `(GH-####)` to the checklist line so planning and
  status remain bidirectionally linked.
- Use `scripts/sppf_sync.py` locally to sync commit trailers (e.g. `SPPF: GH-17`)
  with GitHub issue comments/labels without CI write permissions.
- Do not close issues until a release containing the fix ships; use the
  `status/pending-release` label once work lands on `stage`.
- Optional: enable `GABION_SPPF_SYNC=1` and re-run `scripts/install_hooks.sh` to
  auto-sync on `stage` pushes via the pre-push hook.

## Analysis pipeline nodes
- [x] Import resolution / symbol table (deterministic callee resolution). (GH-6)
- [x] Import resolution: explicit imports + relative import handling.
- [x] Import resolution: star-import expansion.
- [x] Import resolution: re-exports / `__all__` handling.
- [x] Import resolution: class hierarchy dispatch resolution.
- [x] Alias-aware identity tracking (rename morphisms preserved). (GH-7)
- [x] Alias tracking: direct Name-to-Name assignments.
- [x] Alias tracking: tuple/sequence unpacking.
- [x] Alias tracking: attribute/collection aliasing (obj.attr, dict["key"]).
- [x] Alias tracking: alias propagation via call returns. (GH-8)
- [x] Noise controls: project root anchoring + exclude dirs + ignore params.
- [x] External-lib filter (ignore non-project imports by default).
- [x] Wildcard forwarding strictness (`*args/**kwargs`, high/low modes).
- [x] Wildcard forwarding: detect direct starred name in call site.
- [x] Wildcard forwarding: signature-aware mapping for `*args/**kwargs`.
- [x] Wildcard forwarding: pass-through via `*args/**kwargs` variables.
- [x] Type-flow tightening audit (downstream annotations).
- [x] Type-flow ambiguities fail `gabion check` in repo defaults.
- [x] Constant-flow audit (dead knobs / always-constant params).
- [x] Constant-flow helper specificity (avoid false positives in internal helpers).
- [x] Unused-argument pass detection (non-test call sites).
- [x] Analysis: Decorator transparency/unwrapping. (GH-9)
- [x] Verification: Idempotency test (ensure Analysis(Refactor(Code)) == Stable). (GH-22)
- [ ] Decision surface detection + boundary elevation (tier enforcement). (in-15, GH-60)
- [ ] Value-encoded decision surface detection (branchless / algebraic control). (in-18, GH-66)
- [ ] Prime-labeled type fingerprints (algebraic bundle matching). (in-20/in-21, GH-68)

## Reporting & visualization nodes
- [x] Component isolation (connected components in bundle graph).
- [x] Mermaid component diagrams embedded in Markdown report.
- [x] DOT/Graphviz output for bundle graphs.
- [x] Tiered bundle classification (declared vs. observed) + violation listing.
- [x] Bundle declaration sources (Config dataclasses, `dataflow-bundle` markers, dataclass calls). (GH-10)
- [x] Bundle declarations: `dataflow-bundle` markers.
- [x] Bundle declarations: local dataclass constructor calls (Name-only args).
- [x] Bundle declarations: general dataclass fields beyond `_fn` convention.
- [x] Bundle declarations: non-Name args/kwargs in dataclass calls.
- [x] Bundle declarations: external dataclass modules (cross-file).
- [x] FactorizationTree snapshot emission (canonical JSON). (in-16, GH-62)
- [ ] Structural diff command + baseline comparison. (in-16, GH-63)
- [ ] Structural metrics export (bundle/tier/violation stats). (in-16, GH-64)

## Synthesis + refactoring nodes
- [x] Protocol/dataclass synthesis (tier thresholds, field typing) (prototype). (GH-11)
- [x] Synthesis output: dataclass stubs with field typing.
- [x] Synthesis output: typing.Protocol interface stubs.
- [x] Synthesis typing: resolve conflicts into `Union`/`Optional`.
- [x] Naming heuristics (frequency-based) (prototype).
- [x] Merge overlap threshold is configurable (no hardcoded axiom).
- [x] Topological refactoring schedule (callee-first order) (prototype). (GH-12)
- [x] Refactor schedule: basic topological order.
- [x] Refactor schedule: SCC-based cycle detection (explicit knots).
- [x] Partial-application merge heuristics (bundle fragmentation control) (prototype). (GH-13)
- [x] Bundle merge heuristic: Jaccard overlap merge function.
- [x] Bundle merge heuristic: integrated into synthesis/refactor pipeline.
- [x] LLM-ready naming stubs (TODO_Name_Me + context docstrings).
- [x] Type aggregation for synthesis (from type-audit + call-site evidence). (GH-14)
- [x] Type aggregation: single-type consensus assignment.
- [x] Type aggregation: conflict resolution into `Union`/`Any` guidance.
- [x] Refactor payload: Type hint preservation (pass FieldSpec from Analysis to Engine). (GH-15)
- [x] Const/default-aware partial-application detection (subset merge by knobs). (GH-16)
- [ ] Contextvar/ambient context rewrite suggestions. (in-15, GH-61)
- [ ] Subtree reuse detection + lemma synthesis hooks. (in-17, GH-65)
- [ ] Invariant extraction + dependent-type synthesis (Agda). (in-19, GH-67)

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
- [x] Ops: Baseline/Ratchet mechanism (allowlist existing violations, block new ones). (GH-23)
- [x] Redistributable GitHub Action wrapper (composite action for gabion check).
- [x] Locked dependency set for CI (`requirements.lock`).
- [ ] Coverage smell tracking (map tests to invariants/lemmas; track unmapped tests). (GH-42)

## Decision-flow tier nodes
- [ ] Decision Table documentation for branch-heavy modules (Tier-3 evidence). (GH-47)
- [ ] Decision Bundle centralization for repeated guard patterns (Tier-2 evidence). (GH-48)
- [ ] Decision Protocol schema enforcement for critical decision paths (Tier-1 evidence). (GH-49)

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
- [x] LSP code action + workspace edit stub for Protocol extraction.
- [x] LSP code action: stub command wiring.
- [x] LSP workspace edit: real edits for Protocol extraction.
- [x] Precise diagnostic ranges (metadata-backed positions).
- [x] Refactor engine: Signature rewriting (def foo(a, b) -> def foo(bundle)). (GH-17)
- [x] Refactor engine: Call-site rewriting (foo(x, y) -> foo(Bundle(x, y))). (GH-18)
- [x] Refactor engine: Preamble injection (unpack bundle to preserve local logic). (GH-19)
- [x] Refactor engine: Import management (inject bundle import at call sites). (GH-20)
- [x] Refactor strategy: Compatibility shims (generate @overload + DeprecationWarning wrapper). (GH-24)
- [ ] Long-lived LSP server cache / incremental analysis (daemon mode). (GH-21)

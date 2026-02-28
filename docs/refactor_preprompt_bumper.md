---
doc_revision: 1
reader_reintern: "Reader-only: re-intern if doc_revision changed since you last read this doc."
doc_id: refactor_preprompt_bumper
doc_role: guidance
doc_scope:
  - repo
  - docs
  - agents
doc_authority: informative
doc_requires:
  - AGENTS.md#agent_obligations
  - POLICY_SEED.md#policy_seed
  - glossary.md#contract
  - README.md#repo_contract
  - CONTRIBUTING.md#contributing_contract
doc_reviewed_as_of:
  AGENTS.md#agent_obligations: 2
  POLICY_SEED.md#policy_seed: 2
  glossary.md#contract: 1
  README.md#repo_contract: 2
  CONTRIBUTING.md#contributing_contract: 2
doc_review_notes:
  AGENTS.md#agent_obligations: "Reviewed AGENTS.md rev2 (required validation stack, forward-remediation preference, and ci_watch failure-bundle triage guidance)."
  POLICY_SEED.md#policy_seed: "Reviewed POLICY_SEED.md rev2 (forward-remediation order, ci_watch failure-bundle durability, and enforced execution-coverage policy wording)."
  glossary.md#contract: "Aligned to semantic typing + commutation obligations in refactor discovery workflow."
  README.md#repo_contract: "Reviewed README.md rev2 (removed stale ASPF action-plan CLI/examples; continuation docs now state/delta only)."
  CONTRIBUTING.md#contributing_contract: "Reviewed CONTRIBUTING.md rev2 (two-stage dual-sensor cadence, correction-unit validation stack, and strict-coverage trigger guidance)."
doc_change_protocol: "POLICY_SEED.md#change_protocol"
doc_owner: maintainer
---

# Refactor Pre-Prompt Bumper (Comprehensive Understanding Mode)

Use this bumper before giving an LLM refactor goals.

## Required operating posture

- Treat `POLICY_SEED.md#policy_seed` and `glossary.md#contract` as co-equal normative contracts.
- Respect instruction precedence: user instruction > governance contracts > local convenience docs.
- Assume the working tree may already contain unrelated dirty edits:
  - never discard or rewrite unrelated edits,
  - keep your changes minimal and scoped,
  - call out boundaries between pre-existing edits and your edits.
- Preserve the LSP-first invariant:
  - `src/gabion/server.py` = semantic core,
  - `src/gabion/cli.py` = thin client.
- Prefer impossible-by-construction flows over sentinel outcome chains:
  - validate at ingress,
  - use typed carriers in core,
  - discharge impossible post-ingress states with `never()`.

## Mandatory comprehension sequence

1. Governance contracts first: `AGENTS.md`, `POLICY_SEED.md`, `glossary.md`, `README.md`, `CONTRIBUTING.md`.
2. Operating docs: all `docs/*.md` with emphasis on architecture zones, decision-flow tiers, coverage semantics, and checklists.
3. Intent corpus: all `in/*.md` (including `in/in-*.md`) to recover design trajectory and constraints.
4. Code map: server core, analysis modules, synthesis modules, tooling surface, tests.
5. Before edits: produce a short “invariants + impacted modules + docs-to-update” brief.

## `in/in-*.md` detailed index (keywords + complete idea cues)

> Purpose: let a model decide quickly whether a file is required reading for the assigned refactor.

- `in/in-1.md`
  - **Keywords:** dataflow grammar, implicit bundles, data clumps, static analysis, refactoring signal.
  - **Idea cue:** introduces the foundational analysis objective: discover recurring argument co-occurrence and flow, then convert loose parameters into structured configs (typically dataclasses/protocol-like carriers).

- `in/in-2.md`
  - **Keywords:** novelty boundary, linter vs compiler, architectural inference.
  - **Idea cue:** distinguishes what is genuinely novel in the project from standard static-analysis patterns; frames Gabion as intent-recovery rather than mere syntax linting.

- `in/in-3.md`
  - **Keywords:** doer/judge/witness framing, validity criteria, crystal metaphor.
  - **Idea cue:** evaluates prior approach quality and formalizes acceptance criteria for whether discovered structures are semantically meaningful and actionable.

- `in/in-4.md`
  - **Keywords:** identity vs symbol, alias transitivity, structural lattice.
  - **Idea cue:** establishes that semantic identity must survive renaming/aliasing; refactor evidence cannot depend on superficial variable names.

- `in/in-5.md`
  - **Keywords:** wildcard forwarding, import resolution, protocol synthesis, topological planning, higher-order flow.
  - **Idea cue:** defines the next capability set required for completeness: support `*args/**kwargs`, cross-module symbol truth, automatic protocol generation, and safe ordering of edits.

- `in/in-6.md`
  - **Keywords:** ImportVisitor, SymbolTable, cross-file graph.
  - **Idea cue:** turns import resolution into a first-class prerequisite so structural reasoning becomes repository-wide rather than file-local.

- `in/in-7.md`
  - **Keywords:** integrated archeology tool, data structures, visitors, reporting.
  - **Idea cue:** provides an end-to-end prototype that combines alias-aware and import-aware analysis into a coherent pipeline.

- `in/in-8.md`
  - **Keywords:** protocol synthesizer, code generation, analysis-to-emission bridge.
  - **Idea cue:** moves from discovery/reporting to generated structured artifacts, reducing manual translation from findings to code.

- `in/in-9.md`
  - **Keywords:** component-local topo sort, refactoring scheduler, dependency-safe sequence.
  - **Idea cue:** explains why refactors require dependency-respecting order (callee/caller evolution) and proposes scheduling mechanics.

- `in/in-10.md`
  - **Keywords:** decouple file path/import path, parameterized noise filters, naming strategy, repo-agnostic linter.
  - **Idea cue:** generalization step that removes project-specific assumptions from the analysis tool.

- `in/in-11.md`
  - **Keywords:** architectural linter generalization (continuation), configurability.
  - **Idea cue:** companion to `in-10`; reinforces robust configuration and portability constraints.

- `in/in-12.md`
  - **Keywords:** LibCST necessity, LSP integration, IDE-native refactoring.
  - **Idea cue:** argues that concrete syntax preservation and protocol transport (LSP) are required for safe automated rewrites in editor workflows.

- `in/in-13.md`
  - **Keywords:** LSP-first architecture, headless IDE, thin CLI, server ABI.
  - **Idea cue:** codifies the architecture now reflected in repository invariants: semantic intelligence belongs in server; clients remain thin.

- `in/in-14.md`
  - **Keywords:** manifesto, naming identity, repo foundation.
  - **Idea cue:** mission statement linking conceptual framing (“gabion” as structure-from-rubble) to repository structure and early implementation posture.

- `in/in-15.md`
  - **Keywords:** contextual rewriting, decision-surface elevation, API boundary placement.
  - **Idea cue:** extends dataflow discovery into policy-guided rewrite decisions: where state/context should live and where control decisions should be surfaced.

- `in/in-16.md`
  - **Keywords:** persistent diffing, grammar audit trail, CI drift detection.
  - **Idea cue:** requires durable structural artifacts across runs/commits so architectural movement is measurable and enforceable.

- `in/in-17.md`
  - **Keywords:** subtree reuse, lemma hooks, structural deduplication.
  - **Idea cue:** identifies repeated structural subgraphs and connects them to reusable abstractions, reducing copy-paste architecture.

- `in/in-18.md`
  - **Keywords:** algebraic decision surfaces, branchless control, value-encoded decisions.
  - **Idea cue:** captures control semantics that appear as arithmetic/value transformations rather than explicit `if` branches.

- `in/in-19.md`
  - **Keywords:** dependent-type-aware synthesis, invariant extraction, structural indices.
  - **Idea cue:** pushes emitted structures toward proof-friendly forms where inter-field invariants are explicit, not implicit.

- `in/in-20.md`
  - **Keywords:** type factorization, prime-labeled compression, grammar deduplication.
  - **Idea cue:** explores algebraic compression of signature/bundle space to detect equivalence classes at scale.

- `in/in-21.md`
  - **Keywords:** semantic factorization, formal reconstruction, mathematical framing.
  - **Idea cue:** companion formalization of compression/factorization ideas with stronger mathematical expression.

- `in/in-22.md`
  - **Keywords:** ASPF carrier migration split, status accounting.
  - **Idea cue:** separates completed mechanics from remaining migration obligations so planning and implementation are accurately staged.

- `in/in-23.md`
  - **Keywords:** ASPF carrier formalization, attribute-grammar evidence.
  - **Idea cue:** recasts carrier mechanics as evidence-bearing grammar structures with clearer proof/audit interpretation.

- `in/in-24.md`
  - **Keywords:** deadness evidence, first-class witness.
  - **Idea cue:** makes dead code/dead paths explicit evidence objects rather than incidental byproducts of analysis.

- `in/in-25.md`
  - **Keywords:** branching evidence, coherence witness.
  - **Idea cue:** introduces explicit evidence constructs for branch coherence and consistency across decision structures.

- `in/in-26.md`
  - **Keywords:** proof-carrying refactor, transformation obligations.
  - **Idea cue:** positions refactoring output as accompanied by machine-checkable evidence that obligations were preserved.

- `in/in-27.md`
  - **Keywords:** exception safety by construction, exception obligations.
  - **Idea cue:** requires exception-path correctness to be encoded structurally rather than treated as after-the-fact lint.

- `in/in-28.md`
  - **Keywords:** in_step template, normative protocol artifact, auditability.
  - **Idea cue:** standardizes `in/` step document structure so future design steps are comparable, enforceable, and automation-friendly.

- `in/in-29.md`
  - **Keywords:** test obsolescence, evidence dominance, evidence surface preservation.
  - **Idea cue:** defines when tests are removable/mergeable using obligation evidence, avoiding naive line/branch-only deletion rules.

- `in/in-30.md`
  - **Keywords:** SuiteSite carrier, suite-locality, deadlines + ambiguity + decision surfaces.
  - **Idea cue:** shifts locality from function-only to suite-aware carriers (loop/branch bodies), fixing scope-mismatch errors and aligning analysis semantics with Python suites.

- `in/in-31.md`
  - **Keywords:** ProjectionSpec/internment/ASPF/reports unification, quotient + gauge-fix metaphor.
  - **Idea cue:** formal unification document connecting major abstractions into one higher-order structure with explicit projection/normalization interpretation.

- `in/in-32.md`
  - **Keywords:** Gödel-style structural navigation, migration phases, non-linear memory model.
  - **Idea cue:** long-horizon architecture direction replacing linear memory traversal assumptions with structural-addressed execution/analysis.

- `in/in-33.md`
  - **Keywords:** PatternSchema, unify dataflow and execution patterns.
  - **Idea cue:** proposes one schema abstraction to model both parameter bundles and execution motifs, reducing parallel representations.

- `in/in-34.md`
  - **Keywords:** lambda/closure as first-class function site, call resolution.
  - **Idea cue:** ensures anonymous/local callables participate as stable nodes in call candidate materialization and bundle propagation.

- `in/in-35.md`
  - **Keywords:** dict carrier tracking broadening, conservative precision.
  - **Idea cue:** improves dict-based bundle tracking beyond only literal subscript aliases while keeping soundness guardrails.

- `in/in-36.md`
  - **Keywords:** starred dataclass constructor args, conservative handling.
  - **Idea cue:** improves bundle detection around `*`/`**` constructor forwarding for dataclasses without unsound over-approximation.

- `in/in-37.md`
  - **Keywords:** dynamic-dispatch uncertainty class, unresolved-call taxonomy, ambiguity evidence.
  - **Idea cue:** separates unresolved calls due to runtime dispatch from unresolved naming/lookup issues and carries that distinction into reporting/evidence.

## Completion gate (what the model should output before coding)

- I have read governance contracts and extracted non-negotiable invariants.
- I have read `docs/` to map architectural and workflow constraints.
- I have scanned `in/in-*.md` and identified which steps are directly relevant to this refactor.
- I can name exactly which semantics change, which modules are impacted, and which docs need updates.
- I confirm the proposal preserves LSP-first architecture and policy/glossary contracts.

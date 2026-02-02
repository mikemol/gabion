---
doc_revision: 4
reader_reintern: "Reader-only: re-intern if doc_revision changed since you last read this doc."
doc_id: out_2
"doc_role": "hypothesis"
doc_scope:
  - repo
  - tooling
  - governance
  - research
doc_authority: informative
doc_requires:
  - POLICY_SEED.md
  - glossary.md
  - CONTRIBUTING.md
  - README.md
doc_reviewed_as_of:
  POLICY_SEED.md: 21
  glossary.md: 9
  CONTRIBUTING.md: 68
  README.md: 58
doc_change_protocol: "POLICY_SEED.md §6"
doc_erasure:
  - formatting
  - typos
doc_owner: maintainer
---

# Hypothesis: “Rsync for REST” + Gabion Semantics

## 0. Thesis
We want a reliable way to **sync repository configuration** (branch protections,
tag rulesets, Actions posture, environments) against a desired state, without
embedding GitHub‑specific semantics into the core engine.

Terraform can do this, but it is heavy and stateful. The alternative is a
**generic REST sync core** plus a **semantic overlay** that encodes the repo’s
meaning constraints. The core remains “REST‑agnostic.” Gabion supplies the
meaning and invariants.

## 1. Design Goals
- **Golden config without heavy state.** Desired state lives in a local file.
- **Drift detection via symmetric difference.** Canonicalize and diff live JSON.
- **Low semantic coupling.** Core does not encode GitHub semantics.
- **Semantic overlay for invariants.** Gabion enforces commutation and meaning.
- **Safe apply.** Apply is local and explicit (no CI writes).

## 2. Architecture Sketch

### 2.1 Generic REST Sync Core
Inputs:
- `base_url`, `auth`
- endpoint list
- “shape map” (fields to compare, ignore, sort)

Outputs:
- canonical snapshots
- diff / plan
- optional apply (PATCH/PUT/POST)

Core responsibilities:
- fetch
- normalize
- diff
- apply

Core **does not** understand GitHub policies, only JSON shapes.

### 2.2 Semantic Overlay (Gabion)
Overlay responsibilities:
- define “bundles” of settings that commute
- encode invariants (e.g., tag‑workflow constraints)
- refuse apply if invariants are violated
- output human‑readable explanations in `out/`

In other words, Gabion acts as the **semantic gate** on top of a generic sync.

## 3. Data Shapes (Sketch)

### 3.1 Desired State (YAML)
```yaml
repo: mikemol/gabion
base_url: https://api.github.com
endpoints:
  - name: actions_permissions
    method: GET
    url: /repos/{owner}/{repo}/actions/permissions
    compare:
      include: [allowed_actions, selected_actions, default_workflow_permissions]
      ignore: [url]
    apply:
      method: PUT
      url: /repos/{owner}/{repo}/actions/permissions
      body_from: .desired.actions_permissions
```

### 3.2 Canonical Snapshot (JSON)
```json
{
  "actions_permissions": {
    "allowed_actions": "selected",
    "default_workflow_permissions": "read",
    "can_approve_pull_request_reviews": false
  }
}
```

### 3.3 Diff Output (plan)
```json
{
  "actions_permissions": {
    "drift": {
      "allowed_actions": {"want": "selected", "have": "all"}
    }
  }
}
```

## 4. Adjunction (Repo Controls Axis)

This document stays on the **repo‑controls axis** only. It does **not** define
the `in/` ↔ `out/` relationship. The adjunction here is between:

- **Conceptual posture constraints** (repo controls) and
- **Machine‑enforceable repo configuration.**

Any `in/` ↔ `out/` Galois framing belongs in `out-1.md`.

### 4.1 Posets and Orderings
Let:
- **C** = conceptual repo‑controls constraints.
- **R** = repo configuration states.

Define partial orders:
- `c1 ≤_C c2` if **c2 refines c1** (adds constraints).
- `r1 ≤_R r2` if **r2 is at least as restrictive as r1**.

Define monotone maps:
- **F: C → R** (compile constraints into config).
- **G: R → C** (lift config into implied constraints).

### 4.2 Adjunction Condition
We want:
```
F(c) ≤_R r  ⇔  c ≤_C G(r)
```
Meaning: a config is sufficient for a constraint iff the constraint is implied
by that config. Drift is detected when the equivalence fails.

### 4.3 Worked Example (Sketch)
Constraint:
- “Only the release‑tag workflow may create `v*` and `test-v*` tags.”

Extraction:
- F(c) yields a tag ruleset plus workflow guard requirements.

Synthesis:
- G(r) yields a posture statement (“tag creation constrained”) and a policy
  assertion used by `policy_check.py`.

If the repo config drifts (tag ruleset missing), F(c) ≤_R r fails.

## 5. Risks / Constraints
- API drift or missing endpoints → requires adapter patches.
- Apply requires admin scopes → keep local and explicit.
- Some GitHub rulesets may not be exposed uniformly.

## 6. Next Steps
- Prototype `rest_sync.py` with `--dry-run` and `--apply`.
- Define a posture YAML for this repo.
- Add a Gabion semantic guard that refuses unsafe diffs.

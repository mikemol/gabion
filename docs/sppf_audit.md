---
doc_revision: 1
reader_reintern: "Reader-only: re-intern if doc_revision changed since you last read this doc."
doc_id: sppf_audit
doc_role: contract
doc_scope:
  - repo
  - governance
  - tooling
doc_authority: informative
doc_requires:
  - POLICY_SEED.md#policy_seed
  - docs/compliance_render.md#compliance_render
doc_reviewed_as_of:
  POLICY_SEED.md#policy_seed: 57
  docs/compliance_render.md#compliance_render: 1
doc_review_notes:
  POLICY_SEED.md#policy_seed: "Reviewed POLICY_SEED.md rev57; the module is a pure coordinator using dependency-injected callables with no direct execution side effects. Consistent with policy."
  docs/compliance_render.md#compliance_render: "Reviewed compliance_render.md rev1; sppf_audit produces SppfStatusConsistencyResult which compliance_render consumes — data flow contract is consistent."
doc_change_protocol: "POLICY_SEED.md#change_protocol"
doc_sections:
  sppf_audit: 1
doc_section_requires:
  sppf_audit:
    - POLICY_SEED.md#policy_seed
    - docs/compliance_render.md#compliance_render
doc_section_reviews:
  sppf_audit:
    POLICY_SEED.md#policy_seed:
      dep_version: 57
      self_version_at_review: 1
      outcome: no_change
      note: "Policy seed rev57 reviewed; dependency-injected coordinator pattern is policy-consistent."
    docs/compliance_render.md#compliance_render:
      dep_version: 1
      self_version_at_review: 1
      outcome: no_change
      note: "compliance_render rev1 reviewed; output type contract matches."
doc_erasure:
  - formatting
  - typos
doc_owner: maintainer
---

<a id="sppf_audit"></a>
# SPPF Audit

The `gabion_governance.sppf_audit` package is the audit coordination layer for
SPPF status consistency checks. It accepts all heavy operations as injected
callables and is itself a pure coordinator — it holds no validation logic and
carries no direct dependencies on specific validator implementations.

## Interface

### `build_sppf_graph`

```python
build_sppf_graph(
    *,
    root: Path,
    issues_json: Path | None,
    build_graph: Callable[..., dict[str, object]],
) -> SppfGraphResult
```

Delegates entirely to the injected `build_graph` callable. Returns a
`SppfGraphResult(graph=...)` wrapping the result.

### `run_status_consistency`

```python
run_status_consistency(
    *,
    root: Path,
    extra_paths: list[str] | None,
    load_docs: Callable[..., dict[str, Doc]],
    axis_audit: Callable[..., tuple[list[str], list[str]]],
    sync_check: Callable[..., tuple[list[str], list[str]]],
) -> SppfStatusConsistencyResult
```

Orchestrates a two-stage audit:

1. Calls `load_docs(root=root, extra_paths=extra_paths)` to obtain the document map.
2. Calls `axis_audit(root, docs)` → `(violations, warnings)`.
3. Calls `sync_check(root, mode="required")` → `(sync_violations, sync_warnings)`.
4. Concatenates results: axis violations first, then sync violations; axis
   warnings first, then sync warnings.

Returns `SppfStatusConsistencyResult(violations=[...], warnings=[...])`.

## Output types

**`SppfStatusConsistencyResult`** — frozen dataclass:

```
violations: list[str]   — ordered list of violation strings
warnings:   list[str]   — ordered list of warning strings
payload:    dict         — computed property; {"violations": N, "warnings": M}
```

**`SppfGraphResult`** — frozen dataclass:

```
graph: dict[str, object]  — raw graph dict from the injected build_graph callable
```

## Dependency injection contract

All four callables (`build_graph`, `load_docs`, `axis_audit`, `sync_check`)
are injected by `governance_audit_impl.py`, which provides the concrete
validator implementations. This keeps `sppf_audit` decoupled from specific
audit logic and testable with stub callables.

The callable signatures are:

| Parameter | Signature |
| --- | --- |
| `build_graph` | `(root: Path, *, issues_json: Path \| None) -> dict[str, object]` |
| `load_docs` | `(*, root: Path, extra_paths: list[str] \| None) -> dict[str, Doc]` |
| `axis_audit` | `(root: Path, docs: dict[str, Doc]) -> tuple[list[str], list[str]]` |
| `sync_check` | `(root: Path, *, mode: str) -> tuple[list[str], list[str]]` |

## Position in the governance audit pipeline

```
governance_audit_impl.py
    │  injects callables
    ▼
sppf_audit.run_status_consistency()
    │  produces SppfStatusConsistencyResult
    ▼
compliance_render.render_compliance()
    │  produces ComplianceRenderResult
    ▼
governance_audit_impl.py  (collects rendered artifact)
```

`sync_check` is always called with `mode="required"`, enforcing strict
synchronization validation regardless of caller configuration.

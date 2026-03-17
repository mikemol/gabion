---
doc_revision: 1
reader_reintern: "Reader-only: re-intern if doc_revision changed since you last read this doc."
doc_id: compliance_render
doc_role: contract
doc_scope:
  - repo
  - governance
  - tooling
doc_authority: informative
doc_requires:
  - POLICY_SEED.md#policy_seed
  - docs/sppf_audit.md#sppf_audit
doc_reviewed_as_of:
  POLICY_SEED.md#policy_seed: 57
  docs/sppf_audit.md#sppf_audit: 1
doc_review_notes:
  POLICY_SEED.md#policy_seed: "Reviewed POLICY_SEED.md rev57; the module is a pure rendering layer with no execution side effects, no external calls, and no ambiguous control flow. Consistent with policy."
  docs/sppf_audit.md#sppf_audit: "Reviewed sppf_audit.md rev1; SppfStatusConsistencyResult is the declared output of sppf_audit and the declared input to compliance_render — data flow contract is consistent."
doc_change_protocol: "POLICY_SEED.md#change_protocol"
doc_sections:
  compliance_render: 1
doc_section_requires:
  compliance_render:
    - POLICY_SEED.md#policy_seed
    - docs/sppf_audit.md#sppf_audit
doc_section_reviews:
  compliance_render:
    POLICY_SEED.md#policy_seed:
      dep_version: 57
      self_version_at_review: 1
      outcome: no_change
      note: "Policy seed rev57 reviewed; pure rendering layer is policy-consistent."
    docs/sppf_audit.md#sppf_audit:
      dep_version: 1
      self_version_at_review: 1
      outcome: no_change
      note: "sppf_audit rev1 reviewed; input type contract matches."
doc_erasure:
  - formatting
  - typos
doc_owner: maintainer
---

<a id="compliance_render"></a>
# Compliance Render

The `gabion_governance.compliance_render` package is the presentation layer for
SPPF governance audit results. It converts a `SppfStatusConsistencyResult`
(produced by `sppf_audit`) into a human-readable Markdown artifact.

## Interface

### `render_compliance`

```python
render_compliance(result: SppfStatusConsistencyResult) -> ComplianceRenderResult
```

Top-level entry point. Wraps `render_status_consistency_markdown` and packages
the result.

### `render_status_consistency_markdown`

```python
render_status_consistency_markdown(result: SppfStatusConsistencyResult) -> RenderedArtifact
```

Converts violations and warnings into a Markdown string.

## Output types

**`RenderedArtifact`** — frozen dataclass:

```
markdown: str   — the rendered Markdown string
```

**`ComplianceRenderResult`** — frozen dataclass:

```
status_consistency: RenderedArtifact
```

## Rendered Markdown format

The output follows this structure:

```markdown
# SPPF Status Consistency

- violations: N
- warnings: M

## Violations

- <violation string>
- ...

## Warnings

- <warning string>
- ...
```

If there are no violations and no warnings, the body is replaced with:

```markdown
No issues detected.
```

The `## Violations` and `## Warnings` sections are omitted when their
respective lists are empty.

## Position in the governance audit pipeline

```
sppf_audit.run_status_consistency()
    │  produces SppfStatusConsistencyResult
    ▼
compliance_render.render_compliance()
    │  produces ComplianceRenderResult(status_consistency=RenderedArtifact)
    ▼
governance_audit_impl.py  (collects rendered artifact for reporting)
```

`compliance_render` has no side effects and no external dependencies beyond
`sppf_audit.contracts`. It is a deterministic pure function of its input.

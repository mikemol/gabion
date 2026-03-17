---
doc_revision: 1
reader_reintern: "Reader-only: re-intern if doc_revision changed since you last read this doc."
doc_id: consolidation_audit
doc_role: contract
doc_scope:
  - repo
  - governance
  - tooling
doc_authority: informative
doc_requires:
  - POLICY_SEED.md#policy_seed
doc_reviewed_as_of:
  POLICY_SEED.md#policy_seed: 57
doc_review_notes:
  POLICY_SEED.md#policy_seed: "Reviewed POLICY_SEED.md rev57; the module is a pure parsing layer operating on plain text with no external calls, no execution side effects, and no ambiguous control flow. Consistent with policy."
doc_change_protocol: "POLICY_SEED.md#change_protocol"
doc_sections:
  consolidation_audit: 1
doc_section_requires:
  consolidation_audit:
    - POLICY_SEED.md#policy_seed
doc_section_reviews:
  consolidation_audit:
    POLICY_SEED.md#policy_seed:
      dep_version: 57
      self_version_at_review: 1
      outcome: no_change
      note: "Policy seed rev57 reviewed; pure text-parsing layer with no side effects is policy-consistent."
doc_erasure:
  - formatting
  - typos
doc_owner: maintainer
---

<a id="consolidation_audit"></a>
# Consolidation Audit

The `gabion_governance.consolidation_audit` package is the input ingestion layer
for the governance audit pipeline. It parses raw diagnostic text — produced by
linters and decision-surface detectors — into structured `LintEntry` and
`DecisionSurface` objects for downstream consumption.

## Interface

### `parse_lint_entry`

```python
parse_lint_entry(line: str) -> LintEntry | None
```

Parses a single line of linter output into a `LintEntry`. Returns `None` if the
line does not match the expected format.

### `parse_surface_line`

```python
parse_surface_line(line: str, *, value_encoded: bool) -> SurfaceParseResult | None
```

Parses a single decision-surface line. Selects between the standard
`DECISION_RE` pattern and the `VALUE_DECISION_RE` pattern based on
`value_encoded`. Returns `None` on no match.

### `parse_surfaces`

```python
parse_surfaces(lines: Iterable[str], *, value_encoded: bool) -> list[DecisionSurface]
```

Iterates `lines`, calling `parse_surface_line` on each. Non-matching lines are
silently skipped. Returns a list of `DecisionSurface` objects, in input order.

## Input formats

Three regex patterns govern parsing:

**`LINT_ENTRY_RE`** — standard linter diagnostic format:

```
<path>:<line>:<col>: <CODE> <message>
```

Example:

```
src/gabion/analysis/core/graph.py:42:8: B001 param 'ctx' (DataflowBundle candidate)
```

The `param` field is extracted from the message by `PARAM_RE` if present
(`param '<name>' (`); otherwise `param` is `None`.

**`DECISION_RE`** — decision surface declaration:

```
<path>:<qual> decision surface params: <p1>, <p2>, ... (<meta>)
```

**`VALUE_DECISION_RE`** — value-encoded decision surface declaration:

```
<path>:<qual> value-encoded decision params: <p1>, <p2>, ... (<meta>)
```

Both surface patterns capture `path`, `qual`, `params` (comma-separated tuple),
and `meta` (parenthesised metadata string).

## Output types

**`LintEntry`** — frozen dataclass (defined in `compliance_render.decision_contracts`):

```
path:    str        — source file path
line:    int        — line number
col:     int        — column number
code:    str        — diagnostic code (e.g. "B001")
message: str        — full message text
param:   str | None — extracted parameter name, if present
```

**`DecisionSurface`** — frozen dataclass (defined in `compliance_render.decision_contracts`):

```
path:   str           — source file path
qual:   str           — qualified name of the decision surface
params: tuple[str, …] — ordered parameter names
meta:   str           — raw metadata string
```

`DecisionSurface.is_boundary` — property; `True` if `"boundary"` appears in
`meta`.

**`SurfaceParseResult`** — frozen dataclass (internal intermediate):

```
path:   str
qual:   str
params: tuple[str, …]
meta:   str
```

Structurally identical to `DecisionSurface`; used as the return type of
`parse_surface_line` before the final `DecisionSurface` is constructed by
`parse_surfaces`.

## Position in the governance audit pipeline

```
linter / decision-surface detector output (plain text)
    │
    ▼
consolidation_audit.parse_lint_entry()
consolidation_audit.parse_surfaces()
    │  produce LintEntry and DecisionSurface lists
    ▼
governance_audit_impl.py  (consumes structured objects for audit orchestration)
```

`consolidation_audit` has no side effects and no external dependencies beyond
the standard library and `compliance_render.decision_contracts`. It is a
deterministic pure function of its input.

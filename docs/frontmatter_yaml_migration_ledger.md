---
doc_revision: 2
reader_reintern: "Reader-only: re-intern if doc_revision changed since you last read this doc."
doc_id: frontmatter_yaml_migration_ledger
doc_role: ledger
doc_scope:
  - repo
  - governance
  - tooling
doc_authority: informative
doc_requires:
  - CONTRIBUTING.md#contributing_contract
  - AGENTS.md#agent_obligations
  - src/gabion_governance/governance_audit_impl.py
  - scripts/policy/policy_check.py
doc_reviewed_as_of:
  CONTRIBUTING.md#contributing_contract: 2
  AGENTS.md#agent_obligations: 31
doc_review_notes:
  CONTRIBUTING.md#contributing_contract: "Front-matter policy and validation cadence reviewed."
  AGENTS.md#agent_obligations: "Governance obligations and doc hygiene reviewed."
doc_change_protocol: "POLICY_SEED.md#change_protocol"
doc_owner: maintainer
---

<a id="frontmatter_yaml_migration_ledger"></a>
# Frontmatter YAML Migration Ledger

## Purpose
Track the migration from custom yaml-like frontmatter parsing to strict YAML parsing, with explicit status and measurable completion criteria.

## Baseline
- Run date: 2026-03-06
- Tracked markdown with frontmatter: 130
- Strict YAML parse failures (tracked + generated scan): 63
- Known parser entrypoints:
  - `src/gabion_governance/governance_audit_impl.py`
  - `src/gabion/analysis/semantics/impact_index.py`
  - `scripts/audit/audit_in_step_structure.py`

## Current State
- Last update: 2026-03-06
- Tracked markdown with frontmatter (current): 117
- Strict YAML parse failures (tracked markdown): 0
- YAML-like fallback status:
  - `src/gabion_governance/governance_audit_impl.py`: removed
  - `src/gabion/analysis/semantics/impact_index.py`: removed
  - `scripts/audit/audit_in_step_structure.py`: removed
- Policy enforcement:
  - `scripts/policy/policy_check.py --workflows` now includes strict YAML frontmatter validation for tracked markdown.

## Validation Snapshot
- `mise exec -- env PYTHONPATH=. python scripts/policy/policy_check.py --workflows`: pass
- `mise exec -- env PYTHONPATH=. python scripts/policy/policy_check.py --ambiguity-contract`: pass
- `mise exec -- env PYTHONPATH=. python scripts/policy/private_symbol_import_guard.py --check ...`: pass (`new=0`)
- `mise exec -- env PYTHONPATH=. python -m pytest -q tests/gabion/tooling/impact/test_impact_index.py tests/gabion/tooling/governance/test_governance_audit_adapter.py tests/gabion/tooling/docflow/test_docflow_compliance_rows.py tests/gabion/tooling/docflow/test_docflow_violation_formatter.py`: pass (`20 passed`)
- `mise exec -- env PYTHONPATH=. python scripts/misc/extract_test_evidence.py --root . --tests tests --out out/test_evidence.json` + `git diff --exit-code out/test_evidence.json`: pass
- Strict coverage gate status:
  - `mise exec -- env PYTHONPATH=. python -m pytest -q --cov=src/gabion --cov-branch --cov-report=term-missing:skip-covered --cov-fail-under=100`
  - Result: fail (`TOTAL 95.25%`, unrelated repo-wide debt outside this migration scope)
- Docflow strict run status:
  - `mise exec -- env PYTHONPATH=. python -m gabion docflow --root . --fail-on-violations --sppf-gh-ref-mode required`
  - Result: fail due existing governance/reference drift in `in/` docs (not frontmatter parse syntax failures).

## Work Items
1. Add strict YAML parser path in governance/docflow and remove yaml-like fallback.
2. Normalize tracked markdown frontmatter to strict YAML emission.
3. Add guard in policy checks to fail on non-YAML frontmatter.
4. Keep migration ledger updated until full local repro/coverage stack is green.

## Progress
- [x] Baseline captured
- [x] Governance parser strict-YAML path (fallback removed)
- [x] Frontmatter normalization complete (tracked markdown)
- [x] Policy guard ratchet wired into workflow checks
- [ ] Local repro stack green with strict frontmatter checks (blocked by pre-existing repo-wide coverage/docflow debt)

## Acceptance
- `gabion docflow` reports no frontmatter parser fallback usage.
- Policy checks fail if newly introduced non-YAML frontmatter appears.
- Repository frontmatter remains parseable by `yaml.safe_load`.

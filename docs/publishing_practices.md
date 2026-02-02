---
doc_revision: 3
reader_reintern: "Reader-only: re-intern if doc_revision changed since you last read this doc."
doc_id: publishing_practices
doc_role: practices
doc_scope:
  - repo
  - release
  - packaging
  - ci
doc_authority: informative
doc_requires:
  - POLICY_SEED.md
  - CONTRIBUTING.md
doc_change_protocol: "POLICY_SEED.md §6"
doc_erasure:
  - formatting
  - typos
doc_owner: maintainer
---

# Publishing Practices (Best-Practice Register)

This document reifies the current best practices for publishing Gabion as a
Python package. It is **advisory**, but referenced from `POLICY_SEED.md` so the
practices remain visible and reviewable as policy evolves.

## 1. Metadata completeness (PEP 621)
Provide a complete `pyproject.toml` metadata block before first release:

- `readme` (long description source)
- `requires-python` (minimum supported version)
- `authors` / `maintainers`
- `license` (SPDX expression)
- `keywords` and `classifiers`
- `project.urls` (repo, docs, issues)

Rationale: ensures the published artifact is self-describing and discoverable.

## 2. License clarity
Use a single SPDX license expression in `project.license`.
Avoid redundant or conflicting license classifiers.

Rationale: modern tooling interprets the SPDX expression as the canonical source.

## 3. Build artifacts explicitly
Build **both** sdist and wheel artifacts before upload:

- `python -m build`

Rationale: catches packaging errors early and ensures reproducible artifacts.

## 4. TestPyPI dry run
Upload to TestPyPI first, then install from TestPyPI:

- Validate metadata rendering, installability, and entry points.

Rationale: avoids breaking the public index with a faulty first release.

## 5. Trusted Publishing (OIDC)
Use GitHub OIDC trusted publishing for the real release workflow.
Avoid long-lived API tokens.

Rationale: reduces secret leakage risk and matches current PyPI guidance.

Tag-only trigger constraint: release workflows should trigger only on tag pushes
(e.g. `v*` for PyPI, `test-v*` for TestPyPI).

## 6. Harden the release workflow
Release workflows should:

- Be dedicated (no PR triggers).
- Use pinned action SHAs.
- Request minimal permissions (`id-token: write`, `contents: read`).
- Run only from trusted branches/tags.

Rationale: publishing is a sensitive surface.

Current workflows:
- `.github/workflows/release-testpypi.yml` (tag `test-v*`)
- `.github/workflows/release-pypi.yml` (tag `v*`, excludes `test-v*`)

## 7. Versioning discipline
Follow semantic versioning for user-facing releases.
Document the meaning of pre-1.0 changes in `README.md`.

Rationale: avoids surprising users on initial adoption.

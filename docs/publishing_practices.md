---
doc_revision: 23
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
doc_reviewed_as_of:
  POLICY_SEED.md: 28
  CONTRIBUTING.md: 71
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
See `CONTRIBUTING.md` for workflow guardrails and execution constraints.

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
Release workflows must verify the tag commit is reachable from the mirror chain
(`main` → `next` → `release`).
Tags should be created by the `release-tag` workflow. The workflow enforces:

- `next` mirrors `main` before tagging.
- `release` mirrors `next` before tagging.
- `test-v*` tags are created only on `next`.
- `v*` tags are created only on `release`.

An optional automation can create `test-v*` tags after `mirror-next` succeeds.
This is allowed only when `next` mirrors `main` and the tag is derived from the
current `project.version` in `pyproject.toml`, with a `+YYYYMMDDTHHMMSSZ` suffix.
The TestPyPI workflow normalizes the tag into a PEP 440 dev release
(`X.Y.Z.devYYYYMMDDHHMMSS`) before rewriting `project.version` so uploads remain
unique and acceptable to PyPI.

A tag ruleset should limit `v*`/`test-v*` creation to the maintainer and GitHub Actions.
Note: personal repositories cannot enforce actor-restricted rulesets; rely on
workflow guardrails until the repo is moved to an organization.

Branch promotion is automated:

- `.github/workflows/mirror-next.yml` updates `next` after `main` merges (post-PR checks).
- `.github/workflows/promote-release.yml` updates `release` after `test-v*` succeeds.

## 6. Harden the release workflow
Release workflows should:

- Be dedicated (no PR triggers).
- Use pinned action SHAs.
- Request minimal permissions (`id-token: write`, `contents: read`).
- Run only from trusted branches/tags.
- Bind Trusted Publishing to a single GitHub environment.
- Keep workflow logic in `scripts/`; YAML should only orchestrate steps.

Rationale: publishing is a sensitive surface.

Metadata checks: `hatchling` emits core metadata version 2.4, which older
`pkginfo` versions cannot parse. The release workflows run `twine check dist/*`
with an up-to-date `twine/pkginfo` and disable the action’s built-in metadata
check (`verify-metadata: false`) to avoid false failures.

Current workflows:
- `.github/workflows/release-tag.yml` (creates tags from `next` and `release`)
- `.github/workflows/release-testpypi.yml` (tag `test-v*` or workflow_run from auto-test-tag)
- `.github/workflows/release-pypi.yml` (tag `v*`, excludes `test-v*`)

## 7. Versioning discipline
Follow semantic versioning for user-facing releases.
Document the meaning of pre-1.0 changes in `README.md`.

Rationale: avoids surprising users on initial adoption.

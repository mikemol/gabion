---
doc_revision: 2
reader_reintern: "Reader-only: re-intern if doc_revision changed since you last read this doc."
doc_id: pinning_actions
doc_role: guide
doc_scope:
  - repo
  - ci
  - tooling
doc_authority: informative
doc_requires:
  - POLICY_SEED.md
doc_reviewed_as_of:
  POLICY_SEED.md: 32
doc_review_notes:
  POLICY_SEED.md: "Reviewed POLICY_SEED.md rev32 (branch/tag CAS + check-before-use constraints); no conflicts with this document's scope."
doc_change_protocol: "POLICY_SEED.md §6"
doc_erasure:
  - formatting
  - typos
doc_owner: maintainer
---

# Pinning GitHub Actions (SHA)

Policy requires actions be pinned to full commit SHAs.

## Manual pinning
1. Find the release/tag you want (e.g. `v4`).
2. Resolve the tag to a commit SHA.
3. Replace `@v4` with `@<full_sha>` in workflows.

## Using GitHub CLI
```bash
# Example: actions/checkout@v4
gh api repos/actions/checkout/git/ref/tags/v4 --jq .object.sha

# Example: actions/setup-python@v5
gh api repos/actions/setup-python/git/ref/tags/v5 --jq .object.sha
```

If the tag points to an annotated tag, resolve again:
```bash
sha=$(gh api repos/actions/checkout/git/ref/tags/v4 --jq .object.sha)
gh api repos/actions/checkout/git/tags/$sha --jq .object.sha
```

## Repo helper (recommended)
Use `scripts/pin_actions.py` to pin `uses:` lines in place:
```bash
scripts/pin_actions.py docs/workflows/gabion_action_example.yml
```

## Notes
- Keep the allow-list in `docs/allowed_actions.txt` aligned with pinned usage.
- Prefer stable major tags (e.g. `v4`) and pin to their current commit.

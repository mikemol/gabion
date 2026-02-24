---
doc_revision: 3
reader_reintern: "Reader-only: re-intern if doc_revision changed since you last read this doc."
doc_id: gh_action_gabion_readme
doc_role: integration_guide
doc_scope:
  - repo
  - ci
  - tooling
doc_authority: informative
doc_requires:
  - README.md#repo_contract
  - CONTRIBUTING.md#contributing_contract
doc_relations:
  informs:
    - CONTRIBUTING.md#contributing_contract
    - README.md#repo_contract
doc_change_protocol: "POLICY_SEED.md#change_protocol"
doc_owner: maintainer
---
## Gabion Composite Action

This composite action installs and runs Gabion using the system Python.

### Inputs
- `version`: pip specifier (default: `gabion`)
- `command`: subcommand to run (default: `check`)
- `root`: project root (default: `.`)
- `config`: path to gabion config (optional)
- `report`: path to write a Markdown report (optional)
- `args`: extra CLI args (optional)

### Example
```yaml
name: gabion
on:
  pull_request:
jobs:
  gabion:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@<PINNED_SHA>
      - uses: actions/setup-python@<PINNED_SHA>
        with:
          python-version: "3.11"
      - uses: mikemol/gabion/.github/actions/gabion@<TAG_OR_SHA>
        with:
          command: "check"
          args: "--fail-on-violations"
```

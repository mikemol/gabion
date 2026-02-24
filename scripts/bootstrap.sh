#!/usr/bin/env bash
set -euo pipefail

if ! command -v mise >/dev/null 2>&1; then
  echo "mise is required. Install from https://mise.jdx.dev" >&2
  exit 1
fi

ensure_mise_trust() {
  if ! mise trust --yes >/dev/null 2>&1; then
    echo "Failed to trust this repository's mise config." >&2
    echo "Run: mise trust --yes \"$PWD/mise.toml\"" >&2
    echo "In CI, set MISE_TRUSTED_CONFIG_PATHS to include the workspace path." >&2
    exit 1
  fi
}

mise install
ensure_mise_trust
mise exec -- python -m pip install -e .
mise exec -- python -m pip install pytest

if ! mise exec -- python scripts/lsp_smoke_test.py --root .; then
  echo "LSP smoke test failed (pygls may be missing). Continuing." >&2
fi

#!/usr/bin/env bash
set -euo pipefail

if ! command -v mise >/dev/null 2>&1; then
  echo "mise is required. Install from https://mise.jdx.dev" >&2
  exit 1
fi

mise install
mise exec -- python -m pip install -e .
mise exec -- python -m pip install pytest

if ! mise exec -- python scripts/lsp_smoke_test.py --root .; then
  echo "LSP smoke test failed (pygls may be missing). Continuing." >&2
fi

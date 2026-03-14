#!/usr/bin/env bash
set -euo pipefail

repo_root="$(git rev-parse --show-toplevel)"
cd "$repo_root"

if ! command -v mise >/dev/null 2>&1; then
  echo "mise is required. Install from https://mise.jdx.dev" >&2
  exit 1
fi

VENV_DIR="$repo_root/.venv"
PYTHON_BIN="$VENV_DIR/bin/python"

bootstrap_ci_env() {
  echo "[ci-local] bootstrap: mise install"
  mise install

  echo "[ci-local] bootstrap: create .venv"
  mise exec -- python -m venv "$VENV_DIR"

  echo "[ci-local] bootstrap: install dependencies (locked)"
  "$PYTHON_BIN" -m pip install --upgrade pip uv
  "$PYTHON_BIN" -m uv pip sync requirements.lock
  "$PYTHON_BIN" -m uv pip install -e .
}

bootstrap_ci_env

exec "$PYTHON_BIN" -m gabion ci-local-repro "$@"

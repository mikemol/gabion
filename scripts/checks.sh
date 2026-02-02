#!/usr/bin/env bash
set -euo pipefail

run_docflow=true
run_dataflow=true
run_tests=true
list_only=false

for arg in "$@"; do
  case "$arg" in
    --no-docflow) run_docflow=false ;;
    --docflow) run_docflow=true ;;
    --docflow-only) run_docflow=true; run_dataflow=false; run_tests=false ;;
    --tests-only) run_tests=true; run_dataflow=false; run_docflow=false ;;
    --dataflow-only) run_dataflow=true; run_docflow=false; run_tests=false ;;
    --list) list_only=true ;;
    -h|--help)
      echo "Usage: scripts/checks.sh [--docflow|--no-docflow|--docflow-only|--dataflow-only|--tests-only|--list]" >&2
      exit 0
      ;;
  esac
done

if $list_only; then
  echo "Checks to run:" >&2
  $run_dataflow && echo "- dataflow (gabion check)" >&2
  $run_docflow && echo "- docflow (gabion docflow-audit)" >&2
  $run_tests && echo "- tests (pytest)" >&2
  exit 0
fi

if ! command -v mise >/dev/null 2>&1; then
  echo "mise is required. Install from https://mise.jdx.dev" >&2
  exit 1
fi

if $run_dataflow; then
  mise exec -- python -m gabion check
fi
if $run_docflow; then
  mise exec -- python -m gabion docflow-audit
fi
if $run_tests; then
  test_dir="${TEST_ARTIFACTS_DIR:-artifacts/test_runs}"
  mkdir -p "$test_dir"
  mise exec -- python -m pytest \
    --junitxml "$test_dir/junit.xml" \
    --log-file "$test_dir/pytest.log" \
    --log-file-level=INFO
fi

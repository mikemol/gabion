#!/usr/bin/env bash
set -euo pipefail

run_docflow=true
run_dataflow=true
run_tests=true
list_only=false
docflow_mode="required"

for arg in "$@"; do
  case "$arg" in
    --no-docflow) run_docflow=false ;;
    --docflow) run_docflow=true ;;
    --docflow-only) run_docflow=true; run_dataflow=false; run_tests=false ;;
    --tests-only) run_tests=true; run_dataflow=false; run_docflow=false ;;
    --dataflow-only) run_dataflow=true; run_docflow=false; run_tests=false ;;
    --list) list_only=true ;;
    --docflow-advisory) docflow_mode="advisory" ;;
    -h|--help)
      echo "Usage: scripts/checks.sh [--docflow|--no-docflow|--docflow-only|--dataflow-only|--tests-only|--docflow-advisory|--list]" >&2
      exit 0
      ;;
  esac
done

if $list_only; then
  echo "Checks to run:" >&2
  $run_dataflow && echo "- lsp parity gate (gabion lsp-parity-gate --command gabion.check)" >&2
  $run_dataflow && echo "- dataflow (gabion check)" >&2
  $run_docflow && echo "- docflow (gabion docflow --fail-on-violations --sppf-gh-ref-mode $docflow_mode)" >&2
  $run_tests && echo "- tests (pytest)" >&2
  exit 0
fi

if ! command -v mise >/dev/null 2>&1; then
  echo "mise is required. Install from https://mise.jdx.dev" >&2
  exit 1
fi

if $run_dataflow; then
  mise exec -- python -m gabion lsp-parity-gate --command gabion.check
  baseline_arg=()
  if [ -f baselines/dataflow_baseline.txt ]; then
    baseline_arg+=(--baseline baselines/dataflow_baseline.txt)
  fi
  mise exec -- python -m gabion check "${baseline_arg[@]}"
fi
if $run_docflow; then
  docflow_args=(--fail-on-violations --sppf-gh-ref-mode "$docflow_mode")
  if [ "$docflow_mode" = "advisory" ]; then
    echo "WARNING: running docflow in advisory GH-reference mode (local debugging only)." >&2
  fi
  mise exec -- python -m gabion docflow "${docflow_args[@]}"
  mise exec -- python scripts/sppf_status_audit.py --root .
fi
if $run_tests; then
  test_dir="${TEST_ARTIFACTS_DIR:-artifacts/test_runs}"
  mkdir -p "$test_dir"
  mise exec -- python -m pytest \
    --junitxml "$test_dir/junit.xml" \
    --log-file "$test_dir/pytest.log" \
    --log-file-level=INFO
fi

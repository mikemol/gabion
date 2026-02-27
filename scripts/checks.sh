#!/usr/bin/env bash
set -euo pipefail

run_docflow=true
run_dataflow=true
run_tests=true
list_only=false
docflow_mode="required"
aspf_handoff=true
aspf_handoff_manifest="${GABION_ASPF_HANDOFF_MANIFEST:-artifacts/out/aspf_handoff_manifest.json}"
aspf_handoff_session="${GABION_ASPF_HANDOFF_SESSION:-}"
aspf_state_root="${GABION_ASPF_STATE_ROOT:-artifacts/out/aspf_state}"

usage() {
  cat <<'USAGE' >&2
Usage: scripts/checks.sh [--docflow|--no-docflow|--docflow-only|--dataflow-only|--tests-only|--docflow-advisory|--list]
                         [--no-aspf-handoff|--aspf-handoff-manifest <path>|--aspf-handoff-session <id>|--aspf-state-root <path>]
USAGE
}

ensure_aspf_handoff_session() {
  if ! $aspf_handoff; then
    return 0
  fi
  if [ -n "$aspf_handoff_session" ]; then
    return 0
  fi
  aspf_handoff_session="$(mise exec -- python - <<'PY'
from datetime import datetime, timezone
import os
print(f"session-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{os.getpid()}")
PY
)"
}

prepare_aspf_handoff_step() {
  local step_id="$1"
  local command_profile="$2"
  shift 2
  if ! $aspf_handoff; then
    "$@"
    return $?
  fi
  ensure_aspf_handoff_session
  mise exec -- python scripts/aspf_handoff.py run \
    --root . \
    --session-id "$aspf_handoff_session" \
    --step-id "$step_id" \
    --command-profile "$command_profile" \
    --manifest "$aspf_handoff_manifest" \
    --state-root "$aspf_state_root" \
    -- "$@"
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --no-docflow) run_docflow=false ;;
    --docflow) run_docflow=true ;;
    --docflow-only) run_docflow=true; run_dataflow=false; run_tests=false ;;
    --tests-only) run_tests=true; run_dataflow=false; run_docflow=false ;;
    --dataflow-only) run_dataflow=true; run_docflow=false; run_tests=false ;;
    --list) list_only=true ;;
    --docflow-advisory) docflow_mode="advisory" ;;
    --no-aspf-handoff) aspf_handoff=false ;;
    --aspf-handoff-manifest)
      if [ "$#" -lt 2 ]; then
        echo "missing value for --aspf-handoff-manifest" >&2
        exit 2
      fi
      aspf_handoff_manifest="$2"
      shift
      ;;
    --aspf-handoff-session)
      if [ "$#" -lt 2 ]; then
        echo "missing value for --aspf-handoff-session" >&2
        exit 2
      fi
      aspf_handoff_session="$2"
      shift
      ;;
    --aspf-state-root)
      if [ "$#" -lt 2 ]; then
        echo "missing value for --aspf-state-root" >&2
        exit 2
      fi
      aspf_state_root="$2"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "unknown option: $1" >&2
      usage
      exit 2
      ;;
  esac
  shift
done

if $list_only; then
  echo "Checks to run:" >&2
  $run_dataflow && echo "- lsp parity gate (gabion lsp-parity-gate --command gabion.check)" >&2
  $run_dataflow && echo "- dataflow (gabion check run)" >&2
  $run_dataflow && $aspf_handoff && echo "- aspf handoff (state + cumulative imports)" >&2
  $run_docflow && echo "- docflow (gabion docflow --fail-on-violations --sppf-gh-ref-mode $docflow_mode)" >&2
  $run_tests && echo "- tests (pytest)" >&2
  exit 0
fi

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

ensure_mise_trust

if $run_dataflow; then
  mise exec -- python -m gabion lsp-parity-gate --command gabion.check
  baseline_arg=()
  if [ -f baselines/dataflow_baseline.txt ]; then
    baseline_arg+=(--baseline baselines/dataflow_baseline.txt --baseline-mode enforce)
  fi
  prepare_aspf_handoff_step \
    "checks.check.run" \
    "checks.check.run" \
    mise exec -- python -m gabion check run \
    "${baseline_arg[@]}"
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

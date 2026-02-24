#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
run_dir="$root_dir/artifacts/test_runs"
mkdir -p "$run_dir"

timestamp="$(date -u +"%Y%m%dT%H%M%SZ")"
log_file="$run_dir/pytest_${timestamp}.log"

if ! command -v mise >/dev/null 2>&1; then
  {
    echo "mise is required for deterministic test execution."
    echo "Install mise and run: mise install && mise exec -- pytest"
  } | tee "$log_file"
  exit 2
fi

mise exec -- pytest "$@" 2>&1 | tee "$log_file"

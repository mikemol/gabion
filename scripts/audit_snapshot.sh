#!/usr/bin/env bash
set -euo pipefail

if ! command -v mise >/dev/null 2>&1; then
  echo "mise is required. Install from https://mise.jdx.dev" >&2
  exit 1
fi

root="${1:-.}"
timestamp="$(date -u +"%Y%m%d_%H%M%S")"
artifacts_dir="artifacts/audit_snapshots/${timestamp}"
latest_marker="artifacts/audit_snapshots/LATEST.txt"

mkdir -p "$artifacts_dir"

dataflow_report="${artifacts_dir}/dataflow_report.md"
dataflow_dot="${artifacts_dir}/dataflow_graph.dot"
docflow_report="${artifacts_dir}/docflow_audit.txt"

mise exec -- python -m gabion dataflow-audit "$root" \
  --root "$root" \
  --report "$dataflow_report" \
  --dot "$dataflow_dot" \
  --type-audit-report

mise exec -- python scripts/docflow_audit.py --root "$root" > "$docflow_report"

echo "$timestamp" > "$latest_marker"

echo "Wrote:"
echo "- $dataflow_report"
echo "- $dataflow_dot"
echo "- $docflow_report"
echo "- $latest_marker"

#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
marker="$root_dir/artifacts/audit_snapshots/LATEST.txt"

if [[ ! -f "$marker" ]]; then
  echo "Latest snapshot marker not found: $marker" >&2
  exit 1
fi

stamp="$(tr -d '[:space:]' < "$marker")"
if [[ -z "$stamp" ]]; then
  echo "Latest snapshot marker is empty: $marker" >&2
  exit 1
fi

snapshot_dir="$root_dir/artifacts/audit_snapshots/$stamp"
if [[ ! -d "$snapshot_dir" ]]; then
  echo "Snapshot directory not found: $snapshot_dir" >&2
  exit 1
fi

report="$snapshot_dir/dataflow_report.md"
dot="$snapshot_dir/dataflow_graph.dot"
docflow="$snapshot_dir/docflow_audit.txt"

case "${1:-}" in
  --report) echo "$report" ;;
  --dot) echo "$dot" ;;
  --docflow) echo "$docflow" ;;
  --dir) echo "$snapshot_dir" ;;
  "")
    echo "Latest snapshot:"
    echo "- $report"
    echo "- $dot"
    echo "- $docflow"
    ;;
  *)
    echo "Usage: scripts/latest_snapshot.sh [--report|--dot|--docflow|--dir]" >&2
    exit 2
    ;;
esac

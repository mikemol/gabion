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
dataflow_plan="${artifacts_dir}/synthesis_plan.json"
dataflow_protocols="${artifacts_dir}/protocol_stubs.py"
dataflow_refactor="${artifacts_dir}/refactor_plan.json"
dataflow_fingerprint_synth="${artifacts_dir}/fingerprint_synth.json"
dataflow_fingerprint_provenance="${artifacts_dir}/fingerprint_provenance.json"
dataflow_fingerprint_deadness="${artifacts_dir}/fingerprint_deadness.json"
dataflow_fingerprint_coherence="${artifacts_dir}/fingerprint_coherence.json"
decision_snapshot="${artifacts_dir}/decision_snapshot.json"
docflow_report="${artifacts_dir}/docflow_audit.txt"

mise exec -- python -m gabion dataflow-audit "$root" \
  --root "$root" \
  --report "$dataflow_report" \
  --dot "$dataflow_dot" \
  --type-audit-report \
  --synthesis-plan "$dataflow_plan" \
  --synthesis-protocols "$dataflow_protocols" \
  --synthesis-report \
  --refactor-plan \
  --refactor-plan-json "$dataflow_refactor" \
  --fingerprint-synth-json "$dataflow_fingerprint_synth" \
  --fingerprint-provenance-json "$dataflow_fingerprint_provenance" \
  --fingerprint-deadness-json "$dataflow_fingerprint_deadness" \
  --fingerprint-coherence-json "$dataflow_fingerprint_coherence" \
  --emit-decision-snapshot "$decision_snapshot"

mise exec -- python scripts/docflow_audit.py --root "$root" > "$docflow_report"

echo "$timestamp" > "$latest_marker"

echo "Wrote:"
echo "- $dataflow_report"
echo "- $dataflow_dot"
echo "- $dataflow_plan"
echo "- $dataflow_protocols"
echo "- $dataflow_refactor"
echo "- $dataflow_fingerprint_synth"
echo "- $dataflow_fingerprint_provenance"
echo "- $dataflow_fingerprint_deadness"
echo "- $dataflow_fingerprint_coherence"
echo "- $decision_snapshot"
echo "- $docflow_report"
echo "- $latest_marker"

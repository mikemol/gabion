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
dataflow_fingerprint_rewrite_plans="${artifacts_dir}/fingerprint_rewrite_plans.json"
dataflow_fingerprint_exception_obligations="${artifacts_dir}/fingerprint_exception_obligations.json"
dataflow_fingerprint_handledness="${artifacts_dir}/fingerprint_handledness.json"
decision_snapshot="${artifacts_dir}/decision_snapshot.json"
decision_tier_candidates="${artifacts_dir}/decision_tier_candidates.toml"
consolidation_report="${artifacts_dir}/consolidation_report.md"
consolidation_suggestions="${artifacts_dir}/consolidation_suggestions.json"
docflow_report="${artifacts_dir}/docflow_audit.txt"
lint_report="${artifacts_dir}/lint.txt"
lint_jsonl="${artifacts_dir}/lint.jsonl"
lint_sarif="${artifacts_dir}/lint.sarif"

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
  --fingerprint-rewrite-plans-json "$dataflow_fingerprint_rewrite_plans" \
  --fingerprint-exception-obligations-json "$dataflow_fingerprint_exception_obligations" \
  --fingerprint-handledness-json "$dataflow_fingerprint_handledness" \
  --emit-decision-snapshot "$decision_snapshot" \
  --lint \
  --lint-jsonl "$lint_jsonl" \
  --lint-sarif "$lint_sarif" > "$lint_report"

mise exec -- python scripts/audit_tools.py docflow --root "$root" > "$docflow_report"
mise exec -- python scripts/audit_tools.py decision-tiers --root "$root" \
  --lint "$lint_report" > "$decision_tier_candidates"
mise exec -- python scripts/audit_tools.py consolidation --root "$root" \
  --decision "$decision_snapshot" \
  --lint "$lint_report" \
  --output "$consolidation_report" \
  --json-output "$consolidation_suggestions"

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
echo "- $dataflow_fingerprint_rewrite_plans"
echo "- $dataflow_fingerprint_exception_obligations"
echo "- $dataflow_fingerprint_handledness"
echo "- $decision_snapshot"
echo "- $decision_tier_candidates"
echo "- $consolidation_report"
echo "- $consolidation_suggestions"
echo "- $docflow_report"
echo "- $lint_report"
echo "- $lint_jsonl"
echo "- $lint_sarif"
echo "- $latest_marker"

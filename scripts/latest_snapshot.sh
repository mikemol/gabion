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
plan="$snapshot_dir/synthesis_plan.json"
protocols="$snapshot_dir/protocol_stubs.py"
refactor="$snapshot_dir/refactor_plan.json"
fingerprint_synth="$snapshot_dir/fingerprint_synth.json"
fingerprint_provenance="$snapshot_dir/fingerprint_provenance.json"
fingerprint_deadness="$snapshot_dir/fingerprint_deadness.json"
fingerprint_coherence="$snapshot_dir/fingerprint_coherence.json"
fingerprint_rewrite_plans="$snapshot_dir/fingerprint_rewrite_plans.json"
fingerprint_exception_obligations="$snapshot_dir/fingerprint_exception_obligations.json"
fingerprint_handledness="$snapshot_dir/fingerprint_handledness.json"
decision="$snapshot_dir/decision_snapshot.json"
decision_tier_candidates="$snapshot_dir/decision_tier_candidates.toml"
consolidation_report="$snapshot_dir/consolidation_report.md"
consolidation_suggestions="$snapshot_dir/consolidation_suggestions.json"
docflow="$snapshot_dir/docflow_audit.txt"
lint="$snapshot_dir/lint.txt"

case "${1:-}" in
  --report) echo "$report" ;;
  --dot) echo "$dot" ;;
  --plan) echo "$plan" ;;
  --protocols) echo "$protocols" ;;
  --refactor) echo "$refactor" ;;
  --fingerprint-synth) echo "$fingerprint_synth" ;;
  --fingerprint-provenance) echo "$fingerprint_provenance" ;;
  --fingerprint-deadness) echo "$fingerprint_deadness" ;;
  --fingerprint-coherence) echo "$fingerprint_coherence" ;;
  --fingerprint-rewrite-plans) echo "$fingerprint_rewrite_plans" ;;
  --fingerprint-exception-obligations) echo "$fingerprint_exception_obligations" ;;
  --fingerprint-handledness) echo "$fingerprint_handledness" ;;
  --decision) echo "$decision" ;;
  --decision-tier-candidates) echo "$decision_tier_candidates" ;;
  --consolidation-report) echo "$consolidation_report" ;;
  --consolidation-suggestions) echo "$consolidation_suggestions" ;;
  --docflow) echo "$docflow" ;;
  --lint) echo "$lint" ;;
  --dir) echo "$snapshot_dir" ;;
  "")
    echo "Latest snapshot:"
    echo "- $report"
    echo "- $dot"
    echo "- $plan"
    echo "- $protocols"
    echo "- $refactor"
    echo "- $fingerprint_synth"
    echo "- $fingerprint_provenance"
    echo "- $fingerprint_deadness"
    echo "- $fingerprint_coherence"
    echo "- $fingerprint_rewrite_plans"
    echo "- $fingerprint_exception_obligations"
    echo "- $fingerprint_handledness"
    echo "- $decision"
    echo "- $decision_tier_candidates"
    echo "- $consolidation_report"
    echo "- $consolidation_suggestions"
    echo "- $docflow"
    echo "- $lint"
    ;;
  *)
    echo "Usage: scripts/latest_snapshot.sh [--report|--dot|--plan|--protocols|--refactor|--fingerprint-synth|--fingerprint-provenance|--fingerprint-deadness|--fingerprint-coherence|--fingerprint-rewrite-plans|--fingerprint-exception-obligations|--fingerprint-handledness|--decision|--decision-tier-candidates|--consolidation-report|--consolidation-suggestions|--docflow|--lint|--dir]" >&2
    exit 2
    ;;
esac

#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE' >&2
Usage: scripts/audit_snapshot.sh [root]
                                 [--no-aspf-handoff|--aspf-handoff-manifest <path>|--aspf-handoff-session <id>|--aspf-state-root <path>]
USAGE
}

if ! command -v mise >/dev/null 2>&1; then
  echo "mise is required. Install from https://mise.jdx.dev" >&2
  exit 1
fi

root="."
root_seen=0
aspf_handoff=true
aspf_handoff_manifest="${GABION_ASPF_HANDOFF_MANIFEST:-artifacts/out/aspf_handoff_manifest.json}"
aspf_handoff_session="${GABION_ASPF_HANDOFF_SESSION:-}"
aspf_state_root="${GABION_ASPF_STATE_ROOT:-artifacts/out/aspf_state}"

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
      if [ "$root_seen" = "0" ]; then
        root="$1"
        root_seen=1
      else
        echo "unexpected argument: $1" >&2
        usage
        exit 2
      fi
      ;;
  esac
  shift
done

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

check_raw_command=(mise exec -- python -m gabion check raw -- "$root" \
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
  --lint-sarif "$lint_sarif")
set +e
if $aspf_handoff; then
  command_text="$(printf '%q ' "${check_raw_command[@]}")"
  command_text="${command_text}> $(printf '%q' "$lint_report")"
  prepare_aspf_handoff_step \
    "audit-snapshot.check.raw" \
    "audit-snapshot.check.raw" \
    bash -lc "$command_text" \
    >"${artifacts_dir}/aspf_handoff_run_check_raw.json"
  check_raw_rc=$?
else
  "${check_raw_command[@]}" >"$lint_report"
  check_raw_rc=$?
fi
set -e
[ "$check_raw_rc" -eq 0 ] || exit "$check_raw_rc"

mise exec -- python -m gabion docflow --root "$root" >"$docflow_report"
mise exec -- python -m gabion decision-tiers --root "$root" \
  --lint "$lint_report" >"$decision_tier_candidates"
mise exec -- python -m gabion consolidation --root "$root" \
  --decision "$decision_snapshot" \
  --lint "$lint_report" \
  --output "$consolidation_report" \
  --json-output "$consolidation_suggestions"

echo "$timestamp" >"$latest_marker"

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

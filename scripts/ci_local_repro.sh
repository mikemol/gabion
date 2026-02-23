#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/ci_local_repro.sh [--all|--checks-only|--dataflow-only] [--skip-sppf-sync|--run-sppf-sync] [--sppf-range <rev-range>] [--skip-gabion-check-step] [--allow-evidence-drift] [--skip-step-timing|--run-step-timing]

Reproduces the .github/workflows/ci.yml command set locally.

Options:
  --all             Run checks + dataflow reproduction (default).
  --checks-only     Run only the checks job commands.
  --dataflow-only   Run only the dataflow-grammar job commands.
  --skip-sppf-sync  Skip scripts/sppf_sync.py validation.
  --run-sppf-sync   Force scripts/sppf_sync.py validation (requires GH auth token).
  --sppf-range R    Override revision range passed to scripts/sppf_sync.py.
  --skip-gabion-check-step
                    Skip the dataflow run-dataflow-stage invocation (which wraps gabion check).
  --allow-evidence-drift
                    Keep strict extraction but do not fail on out/test_evidence.json drift.
                    Writes artifacts/audit_reports/test_evidence_drift.patch when drift exists.
  --skip-step-timing
                    Disable step timing capture (enabled by default).
  --run-step-timing
                    Force-enable step timing capture.
  -h, --help        Show this help text.
EOF
}

run_checks=true
run_dataflow=true
run_sppf_sync_mode="auto"
sppf_range="${GABION_LOCAL_SPPF_RANGE:-origin/stage..HEAD}"
skip_gabion_check_step="${GABION_LOCAL_SKIP_GABION_CHECK_STEP:-0}"
allow_evidence_drift="${GABION_LOCAL_ALLOW_EVIDENCE_DRIFT:-0}"
step_timing_enabled="${GABION_CI_STEP_TIMING_CAPTURE:-1}"
step_timing_artifact="${GABION_CI_STEP_TIMING_ARTIFACT:-artifacts/audit_reports/ci_step_timings.json}"
step_timing_run_id="${GABION_CI_STEP_TIMING_RUN_ID:-local-$(date -u +%Y%m%dT%H%M%SZ)-$$}"
step_timing_mode="all"

while [ $# -gt 0 ]; do
  case "$1" in
    --all)
      run_checks=true
      run_dataflow=true
      ;;
    --checks-only)
      run_checks=true
      run_dataflow=false
      ;;
    --dataflow-only)
      run_checks=false
      run_dataflow=true
      ;;
    --skip-sppf-sync)
      run_sppf_sync_mode="skip"
      ;;
    --run-sppf-sync)
      run_sppf_sync_mode="force"
      ;;
    --sppf-range)
      if [ $# -lt 2 ]; then
        echo "missing value for --sppf-range" >&2
        exit 2
      fi
      sppf_range="$2"
      shift
      ;;
    --skip-gabion-check-step)
      skip_gabion_check_step="1"
      ;;
    --allow-evidence-drift)
      allow_evidence_drift="1"
      ;;
    --skip-step-timing)
      step_timing_enabled="0"
      ;;
    --run-step-timing)
      step_timing_enabled="1"
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
  shift
done

if ! command -v mise >/dev/null 2>&1; then
  echo "mise is required. Install from https://mise.jdx.dev" >&2
  exit 1
fi

repo_root="$(git rev-parse --show-toplevel)"
cd "$repo_root"

log_dir="${CI_LOCAL_LOG_DIR:-artifacts/test_runs/local_ci}"
mkdir -p "$log_dir"

step() {
  echo
  echo "[ci-local] $*"
}

observability_enabled() {
  [ "${GABION_OBSERVABILITY_GUARD:-1}" != "0" ]
}

timing_enabled() {
  [ "$step_timing_enabled" != "0" ]
}

observed() {
  local label="$1"
  shift
  if observability_enabled; then
    local max_gap="${GABION_OBSERVABILITY_MAX_GAP_SECONDS:-5}"
    local max_wall="${GABION_OBSERVABILITY_MAX_WALL_SECONDS:-1200}"
    mise exec -- python scripts/ci_observability_guard.py \
      --label "$label" \
      --max-gap-seconds "$max_gap" \
      --max-wall-seconds "$max_wall" \
      --artifact-path "artifacts/audit_reports/observability_violations.json" \
      -- "$@"
  else
    "$@"
  fi
}

timed_observed() {
  local label="$1"
  shift
  if timing_enabled; then
    if observability_enabled; then
      local max_gap="${GABION_OBSERVABILITY_MAX_GAP_SECONDS:-5}"
      local max_wall="${GABION_OBSERVABILITY_MAX_WALL_SECONDS:-1200}"
      mise exec -- python scripts/ci_step_timing_capture.py \
        --label "$label" \
        --mode "$step_timing_mode" \
        --run-id "$step_timing_run_id" \
        --artifact-path "$step_timing_artifact" \
        -- \
        mise exec -- python scripts/ci_observability_guard.py \
          --label "$label" \
          --max-gap-seconds "$max_gap" \
          --max-wall-seconds "$max_wall" \
          --artifact-path "artifacts/audit_reports/observability_violations.json" \
          -- "$@"
    else
      mise exec -- python scripts/ci_step_timing_capture.py \
        --label "$label" \
        --mode "$step_timing_mode" \
        --run-id "$step_timing_run_id" \
        --artifact-path "$step_timing_artifact" \
        -- "$@"
    fi
  else
    observed "$label" "$@"
  fi
}

resolve_gh_token() {
  if [ -n "${GH_TOKEN:-}" ]; then
    printf '%s\n' "$GH_TOKEN"
    return 0
  fi
  if gh_token="$(gh auth token 2>/dev/null)"; then
    printf '%s\n' "$gh_token"
    return 0
  fi
  return 1
}

run_checks_job() {
  step_timing_mode="checks"

  step "checks: policy_check --workflows"
  timed_observed checks_policy_workflows mise exec -- python scripts/policy_check.py --workflows

  step "checks: policy_check --posture"
  if [ -z "${POLICY_GITHUB_TOKEN:-}" ]; then
    echo "POLICY_GITHUB_TOKEN not set; skipping posture check (matches CI skip path)."
  else
    observed checks_policy_posture env POLICY_GITHUB_TOKEN="$POLICY_GITHUB_TOKEN" mise exec -- python scripts/policy_check.py --posture
  fi

  step "checks: docflow"
  timed_observed checks_docflow mise exec -- python -m gabion docflow --root . --fail-on-violations --sppf-gh-ref-mode required

  step "checks: order_lifetime_check"
  observed checks_order_lifetime mise exec -- python scripts/order_lifetime_check.py --root .

  step "checks: structural_hash_policy_check"
  observed checks_structural_hash_policy mise exec -- python scripts/structural_hash_policy_check.py --root .

  step "checks: no_monkeypatch_policy_check"
  observed checks_no_monkeypatch_policy mise exec -- python scripts/no_monkeypatch_policy_check.py --root .

  step "checks: branchless_policy_check"
  observed checks_branchless_policy mise exec -- python scripts/branchless_policy_check.py --root .

  step "checks: defensive_fallback_policy_check"
  observed checks_defensive_fallback_policy mise exec -- python scripts/defensive_fallback_policy_check.py --root .

  step "checks: complexity_audit --fail-on-regression"
  timed_observed checks_complexity_audit mise exec -- python scripts/complexity_audit.py --root . --fail-on-regression

  step "checks: sppf_status_audit"
  observed checks_sppf_status_audit mise exec -- python scripts/sppf_status_audit.py --root .

  case "$run_sppf_sync_mode" in
    skip)
      step "checks: skipping sppf_sync validation (--skip-sppf-sync)"
      ;;
    auto|force)
      step "checks: sppf_sync --validate"
      if gh_token="$(resolve_gh_token)"; then
        observed checks_sppf_sync_validate env GH_TOKEN="$gh_token" mise exec -- python scripts/sppf_sync.py \
          --validate \
          --only-when-relevant \
          --range "$sppf_range" \
          --require-state open \
          --require-label done-on-stage \
          --require-label status/pending-release
      elif [ "$run_sppf_sync_mode" = "force" ]; then
        echo "GH auth token unavailable but --run-sppf-sync was requested." >&2
        exit 1
      else
        echo "GH auth token unavailable; skipping sppf_sync validation."
      fi
      ;;
  esac

  step "checks: extract_test_evidence"
  observed checks_extract_test_evidence env GABION_LSP_TIMEOUT_TICKS=300000 GABION_LSP_TIMEOUT_TICK_NS=1000000 mise exec -- python scripts/extract_test_evidence.py --root . --tests tests --out out/test_evidence.json
  if [ "$allow_evidence_drift" = "1" ]; then
    step "checks: evidence drift diff (allow-evidence-drift enabled)"
  else
    step "checks: evidence drift diff (strict)"
  fi
  observed checks_git_diff_test_evidence env GABION_LOCAL_ALLOW_EVIDENCE_DRIFT="$allow_evidence_drift" bash -lc '
    patch_path="artifacts/audit_reports/test_evidence_drift.patch"
    mkdir -p "$(dirname "$patch_path")"
    if git diff --quiet -- out/test_evidence.json; then
      : > "$patch_path"
      exit 0
    fi
    git diff -- out/test_evidence.json > "$patch_path"
    if [ "${GABION_LOCAL_ALLOW_EVIDENCE_DRIFT:-0}" = "1" ]; then
      echo "WARNING: out/test_evidence.json drift detected (explicit bypass enabled)."
      echo "WARNING: drift patch written to $patch_path"
      exit 0
    fi
    echo "out/test_evidence.json drift detected; see $patch_path" >&2
    exit 1
  '

  step "checks: pytest --cov"
  mkdir -p artifacts/test_runs
  timed_observed checks_pytest env PYTHONUNBUFFERED=1 mise exec -- python -m pytest \
    --cov=src/gabion \
    --cov-branch \
    --cov-report= \
    --junitxml artifacts/test_runs/junit.xml \
    --log-file artifacts/test_runs/pytest.log \
    --log-file-level=INFO

  step "checks: coverage xml"
  observed checks_coverage_xml mise exec -- python -m coverage xml -o artifacts/test_runs/coverage.xml

  step "checks: coverage html"
  observed checks_coverage_html mise exec -- python -m coverage html -d artifacts/test_runs/htmlcov

  step "checks: coverage term + threshold"
  timed_observed checks_coverage_term mise exec -- python -m coverage report --show-missing --fail-under=100

  step "checks: delta_state_emit"
  timed_observed checks_delta_state_emit env GABION_DIRECT_RUN=1 GABION_LSP_TIMEOUT_TICKS=65000000 GABION_LSP_TIMEOUT_TICK_NS=1000000 mise exec -- python -m gabion delta-state-emit

  step "checks: delta_triplets"
  timed_observed checks_delta_triplets env GABION_DIRECT_RUN=1 GABION_LSP_TIMEOUT_TICKS=65000000 GABION_LSP_TIMEOUT_TICK_NS=1000000 mise exec -- python -m gabion delta-triplets
}

seed_dataflow_checkpoint() {
  local target_dir="artifacts/audit_reports"
  local seed_checkpoint="baselines/dataflow_resume_checkpoint_ci.json"
  local seed_chunks="baselines/dataflow_resume_checkpoint_ci.json.chunks"
  mkdir -p "$target_dir"
  local restored=0
  if [ -f "$seed_checkpoint" ]; then
    cp "$seed_checkpoint" "$target_dir/dataflow_resume_checkpoint_ci.json"
    restored=1
  fi
  if [ -d "$seed_chunks" ]; then
    rm -rf "$target_dir/dataflow_resume_checkpoint_ci.json.chunks"
    mkdir -p "$target_dir/dataflow_resume_checkpoint_ci.json.chunks"
    cp -R "$seed_chunks"/. "$target_dir/dataflow_resume_checkpoint_ci.json.chunks/"
    restored=1
  fi
  if [ "$restored" = "1" ]; then
    echo "Seeded checkpoint from version-controlled baseline."
  else
    echo "No version-controlled checkpoint seed found; continuing."
  fi
}

restore_dataflow_checkpoint() {
  if gh_token="$(resolve_gh_token)"; then
    step "dataflow: restore-resume-checkpoint (best effort)"
    observed dataflow_restore_resume_checkpoint env GH_TOKEN="$gh_token" GH_REPO="$(gh repo view --json nameWithOwner --jq .nameWithOwner)" \
      GH_REF_NAME="${GABION_LOCAL_REF_NAME:-$(git rev-parse --abbrev-ref HEAD)}" \
      GH_RUN_ID="${GABION_LOCAL_RUN_ID:-0}" \
      mise exec -- python -m gabion restore-resume-checkpoint \
      --output-dir artifacts/audit_reports \
      --artifact-name dataflow-report \
      --checkpoint-name dataflow_resume_checkpoint_ci.json || true
  else
    step "dataflow: restore-resume-checkpoint skipped (GH token unavailable)"
  fi
}

run_dataflow_job() {
  step_timing_mode="dataflow"

  step "dataflow: seed version-controlled checkpoint (best effort)"
  seed_dataflow_checkpoint

  restore_dataflow_checkpoint

  if [ "$skip_gabion_check_step" = "1" ]; then
    step "dataflow: skipping run-dataflow-stage (--skip-gabion-check-step)"
    return 0
  fi

  step "dataflow: run-dataflow-stage (single invocation)"
  local outputs_file="$log_dir/dataflow_stage_outputs.env"
  local summary_file="$log_dir/dataflow_stage_summary.md"
  rm -f "$outputs_file"
  touch "$outputs_file"
  : > "$summary_file"
  observed dataflow_run_dataflow_stage env \
    GABION_DIRECT_RUN=1 \
    GABION_LSP_TIMEOUT_TICKS="${GABION_LSP_TIMEOUT_TICKS:-65000000}" \
    GABION_LSP_TIMEOUT_TICK_NS="${GABION_LSP_TIMEOUT_TICK_NS:-1000000}" \
    GABION_DATAFLOW_DEBUG_DUMP_INTERVAL_SECONDS="${GABION_DATAFLOW_DEBUG_DUMP_INTERVAL_SECONDS:-60}" \
    mise exec -- python -m gabion run-dataflow-stage \
    --stage-strictness-profile "run=high" \
    --github-output "$outputs_file" \
    --step-summary "$summary_file"

  step "dataflow: finalize outcome"
  local terminal_stage terminal_exit terminal_state terminal_status attempts_run
  terminal_stage="$(sed -n 's/^terminal_stage=//p' "$outputs_file" | tail -n1)"
  terminal_exit="$(sed -n 's/^exit_code=//p' "$outputs_file" | tail -n1)"
  terminal_state="$(sed -n 's/^analysis_state=//p' "$outputs_file" | tail -n1)"
  terminal_status="$(sed -n 's/^terminal_status=//p' "$outputs_file" | tail -n1)"
  attempts_run="$(sed -n 's/^attempts_run=//p' "$outputs_file" | tail -n1)"
  terminal_stage="${terminal_stage:-none}"
  terminal_state="${terminal_state:-none}"
  terminal_status="${terminal_status:-unknown}"
  attempts_run="${attempts_run:-0}"

  if [ -z "${terminal_exit:-}" ]; then
    echo "No dataflow audit stage produced an exit code." >&2
    return 1
  fi

  if [ "$terminal_status" = "unknown" ]; then
    if [ "$terminal_exit" = "0" ]; then
      terminal_status="success"
    elif [ "$terminal_state" = "timed_out_progress_resume" ]; then
      terminal_status="timeout_resume"
    else
      terminal_status="hard_failure"
    fi
  fi

  echo "terminal_stage=$terminal_stage attempts=$attempts_run exit_code=$terminal_exit analysis_state=$terminal_state status=$terminal_status"

  if [ "$terminal_status" != "success" ]; then
    if [ -f artifacts/audit_reports/dataflow_report.md ]; then
      echo "===== dataflow report ====="
      cat artifacts/audit_reports/dataflow_report.md
    fi
    if [ -f artifacts/audit_reports/timeout_progress.md ]; then
      echo "===== timeout progress ====="
      cat artifacts/audit_reports/timeout_progress.md
    fi
    if [ "$terminal_status" = "timeout_resume" ]; then
      echo "Dataflow audit invocation timed out with resumable progress." >&2
    else
      echo "Dataflow audit failed for a non-timeout reason." >&2
    fi
    return 1
  fi

  step "dataflow: deadline profile summary"
  if [ -f artifacts/out/deadline_profile.json ]; then
    mise exec -- python scripts/deadline_profile_ci_summary.py \
      --allow-missing-local \
      --step-summary "$log_dir/deadline_profile_summary.md"
  else
    echo "Skipping deadline profile summary (missing artifacts/out/deadline_profile.json)."
  fi
}

if $run_checks; then
  run_checks_job
fi

if $run_dataflow; then
  run_dataflow_job
fi

step "done"
echo "Local CI reproduction completed successfully."

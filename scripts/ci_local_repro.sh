#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: scripts/ci_local_repro.sh [--all|--checks-only|--dataflow-only|--pr-dataflow-only] [--extended-checks] [--skip-sppf-sync|--run-sppf-sync] [--sppf-range <rev-range>] [--skip-gabion-check-step] [--pr-base-sha <sha>] [--pr-head-sha <sha>] [--pr-body-file <path>] [--verify-pr-stage-ci] [--pr-stage-ci-timeout-minutes <minutes>] [--run-observability-guard] [--skip-step-timing|--run-step-timing]

Reproduces the .github/workflows/ci.yml command set locally.

Options:
  --all                   Run checks + dataflow reproduction (default).
  --checks-only           Run only the checks job commands.
  --dataflow-only         Run only the dataflow-grammar job commands.
  --pr-dataflow-only      Run PR dataflow-grammar reproduction
                          (.github/workflows/pr-dataflow-grammar.yml equivalent).
  --extended-checks       Run additional local hardening checks not present in ci.yml
                          (order_lifetime_check, structural_hash_policy_check, complexity_audit).
  --skip-sppf-sync        Skip scripts/sppf_sync.py validation.
  --run-sppf-sync         Force scripts/sppf_sync.py validation (requires GH auth token).
  --sppf-range R          Override revision range passed to scripts/sppf_sync.py.
  --skip-gabion-check-step
                          Skip the dataflow run-dataflow-stage invocation (which wraps gabion check).
  --pr-base-sha SHA       Override PR base SHA for --pr-dataflow-only mode.
  --pr-head-sha SHA       Override PR head SHA for --pr-dataflow-only mode.
  --pr-body-file PATH     Optional PR body file for governance template parity check.
  --verify-pr-stage-ci    Require successful stage ci.yml run for PR head SHA.
  --pr-stage-ci-timeout-minutes N
                          Timeout in minutes for --verify-pr-stage-ci polling (default: 70).
  --run-observability-guard
                          Enable ci_observability_guard wrappers (off by default for parity).
  --skip-step-timing      Disable step timing capture (off by default for parity).
  --run-step-timing       Enable step timing capture.
  -h, --help              Show this help text.
USAGE
}

run_checks=true
run_dataflow=true
run_pr_dataflow=false
run_extended_checks=false
run_sppf_sync_mode="auto"
sppf_range="${GABION_LOCAL_SPPF_RANGE:-}"
skip_gabion_check_step="${GABION_LOCAL_SKIP_GABION_CHECK_STEP:-0}"
pr_base_sha="${GABION_LOCAL_PR_BASE_SHA:-${GITHUB_BASE_SHA:-}}"
pr_head_sha="${GABION_LOCAL_PR_HEAD_SHA:-${GITHUB_HEAD_SHA:-}}"
pr_body_file="${GABION_LOCAL_PR_BODY_FILE:-}"
pr_verify_stage_ci="${GABION_LOCAL_PR_VERIFY_STAGE_CI:-0}"
pr_stage_ci_timeout_minutes="${GABION_LOCAL_PR_STAGE_CI_TIMEOUT_MINUTES:-70}"
step_timing_enabled="${GABION_CI_STEP_TIMING_CAPTURE:-0}"
observability_enabled_flag="${GABION_OBSERVABILITY_GUARD:-0}"
step_timing_artifact="${GABION_CI_STEP_TIMING_ARTIFACT:-artifacts/audit_reports/ci_step_timings.json}"
step_timing_run_id="${GABION_CI_STEP_TIMING_RUN_ID:-local-$(date -u +%Y%m%dT%H%M%SZ)-$$}"
step_timing_mode="all"
ci_event_name="${GABION_LOCAL_EVENT_NAME:-push}"

while [ $# -gt 0 ]; do
  case "$1" in
    --all)
      run_checks=true
      run_dataflow=true
      run_pr_dataflow=false
      ;;
    --checks-only)
      run_checks=true
      run_dataflow=false
      run_pr_dataflow=false
      ;;
    --dataflow-only)
      run_checks=false
      run_dataflow=true
      run_pr_dataflow=false
      ;;
    --pr-dataflow-only)
      run_checks=false
      run_dataflow=false
      run_pr_dataflow=true
      ;;
    --extended-checks)
      run_extended_checks=true
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
    --pr-base-sha)
      if [ $# -lt 2 ]; then
        echo "missing value for --pr-base-sha" >&2
        exit 2
      fi
      pr_base_sha="$2"
      shift
      ;;
    --pr-head-sha)
      if [ $# -lt 2 ]; then
        echo "missing value for --pr-head-sha" >&2
        exit 2
      fi
      pr_head_sha="$2"
      shift
      ;;
    --pr-body-file)
      if [ $# -lt 2 ]; then
        echo "missing value for --pr-body-file" >&2
        exit 2
      fi
      pr_body_file="$2"
      shift
      ;;
    --verify-pr-stage-ci)
      pr_verify_stage_ci="1"
      ;;
    --pr-stage-ci-timeout-minutes)
      if [ $# -lt 2 ]; then
        echo "missing value for --pr-stage-ci-timeout-minutes" >&2
        exit 2
      fi
      pr_stage_ci_timeout_minutes="$2"
      shift
      ;;
    --run-observability-guard)
      observability_enabled_flag="1"
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

VENV_DIR="$repo_root/.venv"
PYTHON_BIN="$VENV_DIR/bin/python"

step() {
  echo
  echo "[ci-local] $*"
}

observability_enabled() {
  [ "$observability_enabled_flag" != "0" ]
}

timing_enabled() {
  [ "$step_timing_enabled" != "0" ]
}

bootstrap_ci_env() {
  step "bootstrap: mise install"
  mise install

  step "bootstrap: create .venv"
  mise exec -- python -m venv "$VENV_DIR"

  step "bootstrap: install dependencies (locked)"
  "$PYTHON_BIN" -m pip install --upgrade pip uv
  "$PYTHON_BIN" -m uv pip sync requirements.lock
  "$PYTHON_BIN" -m uv pip install -e .
}

observed() {
  local label="$1"
  shift
  if observability_enabled; then
    local max_gap="${GABION_OBSERVABILITY_MAX_GAP_SECONDS:-5}"
    local max_wall="${GABION_OBSERVABILITY_MAX_WALL_SECONDS:-1200}"
    "$PYTHON_BIN" scripts/ci_observability_guard.py \
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
      "$PYTHON_BIN" scripts/ci_step_timing_capture.py \
        --label "$label" \
        --mode "$step_timing_mode" \
        --run-id "$step_timing_run_id" \
        --artifact-path "$step_timing_artifact" \
        -- \
        "$PYTHON_BIN" scripts/ci_observability_guard.py \
          --label "$label" \
          --max-gap-seconds "$max_gap" \
          --max-wall-seconds "$max_wall" \
          --artifact-path "artifacts/audit_reports/observability_violations.json" \
          -- "$@"
    else
      "$PYTHON_BIN" scripts/ci_step_timing_capture.py \
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
  if [ -n "${GITHUB_TOKEN:-}" ]; then
    printf '%s\n' "$GITHUB_TOKEN"
    return 0
  fi
  if command -v gh >/dev/null 2>&1; then
    local gh_token
    gh_token="$(gh auth token 2>/dev/null || true)"
    if [ -n "$gh_token" ]; then
      printf '%s\n' "$gh_token"
      return 0
    fi
  fi
  return 1
}

resolve_sppf_range() {
  if [ -n "$sppf_range" ]; then
    printf '%s\n' "$sppf_range"
    return 0
  fi

  local before_sha after_sha
  before_sha="${BEFORE_SHA:-${GABION_LOCAL_BEFORE_SHA:-}}"
  after_sha="${AFTER_SHA:-$(git rev-parse HEAD)}"

  if [ -z "$before_sha" ] || [ "$before_sha" = "0000000000000000000000000000000000000000" ]; then
    printf '%s\n' "HEAD~20..HEAD"
    return 0
  fi

  if ! git cat-file -e "${after_sha}^{commit}" 2>/dev/null || ! git cat-file -e "${before_sha}^{commit}" 2>/dev/null; then
    git fetch --no-tags origin "$before_sha" "$after_sha" || true
  fi

  if git cat-file -e "${after_sha}^{commit}" 2>/dev/null && git cat-file -e "${before_sha}^{commit}" 2>/dev/null; then
    printf '%s\n' "$before_sha..$after_sha"
    return 0
  fi

  echo "Push SHAs unavailable locally; falling back to safe local range."
  printf '%s\n' "HEAD~20..HEAD"
}

resolve_pr_head_sha() {
  if [ -n "$pr_head_sha" ]; then
    printf '%s\n' "$pr_head_sha"
    return 0
  fi
  git rev-parse HEAD
}

resolve_pr_base_sha() {
  if [ -n "$pr_base_sha" ]; then
    printf '%s\n' "$pr_base_sha"
    return 0
  fi

  local head_sha
  head_sha="$(resolve_pr_head_sha)"
  if git rev-parse --verify --quiet origin/main >/dev/null; then
    git merge-base origin/main "$head_sha"
    return 0
  fi
  if git rev-parse --verify --quiet main >/dev/null; then
    git merge-base main "$head_sha"
    return 0
  fi
  if git rev-parse --verify --quiet "${head_sha}~1" >/dev/null; then
    git rev-parse "${head_sha}~1"
    return 0
  fi
  printf '%s\n' "$head_sha"
}

resolve_pr_body_file() {
  if [ -n "$pr_body_file" ] && [ -f "$pr_body_file" ]; then
    printf '%s\n' "$pr_body_file"
    return 0
  fi

  if ! command -v gh >/dev/null 2>&1; then
    return 1
  fi

  local fetched_path="$log_dir/pr_body.md"
  if gh pr view --json body --jq .body >"$fetched_path" 2>/dev/null; then
    printf '%s\n' "$fetched_path"
    return 0
  fi
  return 1
}

verify_stage_ci_for_sha() {
  local sha="$1"
  local gh_token
  if ! gh_token="$(resolve_gh_token)"; then
    echo "GH auth token unavailable; cannot verify stage CI for SHA $sha." >&2
    return 1
  fi
  local repo_name
  repo_name="$(gh repo view --json nameWithOwner --jq .nameWithOwner 2>/dev/null || true)"
  if [ -z "$repo_name" ]; then
    echo "Unable to resolve repository name for stage CI verification." >&2
    return 1
  fi

  observed pr_dataflow_verify_stage_ci env \
    GITHUB_TOKEN="$gh_token" \
    REPO="$repo_name" \
    SHA="$sha" \
    TIMEOUT_MINUTES="$pr_stage_ci_timeout_minutes" \
    "$PYTHON_BIN" - <<'PY'
import json
import os
import time
import urllib.request

token = os.environ["GITHUB_TOKEN"]
repo = os.environ["REPO"]
sha = os.environ["SHA"]
timeout_minutes = int(os.environ.get("TIMEOUT_MINUTES", "70"))
deadline = time.time() + timeout_minutes * 60
url = (
    f"https://api.github.com/repos/{repo}/actions/workflows/ci.yml/runs"
    f"?branch=stage&per_page=50"
)

while True:
    req = urllib.request.Request(
        url,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
        },
    )
    with urllib.request.urlopen(req) as resp:
        payload = json.loads(resp.read().decode("utf-8"))

    runs = payload.get("workflow_runs", [])
    match = next((r for r in runs if r.get("head_sha") == sha), None)
    if match is None:
        if time.time() > deadline:
            raise SystemExit(f"Stage CI has not run for {sha}.")
        time.sleep(15)
        continue

    status = match.get("status")
    if status != "completed":
        if time.time() > deadline:
            raise SystemExit(f"Stage CI for {sha} not complete (status={status}).")
        time.sleep(15)
        continue

    conclusion = match.get("conclusion")
    if conclusion != "success":
        raise SystemExit(f"Stage CI for {sha} not successful (conclusion={conclusion}).")
    print(f"Stage CI OK for {sha}.")
    break
PY
}

run_checks_job() {
  step_timing_mode="checks"

  step "checks: policy_check --workflows"
  timed_observed checks_policy_workflows "$PYTHON_BIN" scripts/policy_check.py --workflows

  step "checks: policy_check --posture"
  if [ "$ci_event_name" != "push" ]; then
    echo "event '$ci_event_name' is not push; skipping posture check (matches CI skip path)."
  elif [ -z "${POLICY_GITHUB_TOKEN:-}" ]; then
    echo "POLICY_GITHUB_TOKEN not set; skipping posture check (matches CI skip path)."
  else
    observed checks_policy_posture env POLICY_GITHUB_TOKEN="$POLICY_GITHUB_TOKEN" "$PYTHON_BIN" scripts/policy_check.py --posture
  fi

  step "checks: docflow"
  timed_observed checks_docflow "$PYTHON_BIN" -m gabion docflow --root . --fail-on-violations --sppf-gh-ref-mode required

  step "checks: sppf_status_audit"
  observed checks_sppf_status_audit "$PYTHON_BIN" scripts/sppf_status_audit.py --root .

  case "$run_sppf_sync_mode" in
    skip)
      step "checks: skipping sppf_sync validation (--skip-sppf-sync)"
      ;;
    auto|force)
      step "checks: sppf_sync --validate"
      if [ "$ci_event_name" != "push" ]; then
        echo "event '$ci_event_name' is not push; skipping sppf_sync validation (matches CI skip path)."
      elif gh_token="$(resolve_gh_token)"; then
        local rev_range
        rev_range="$(resolve_sppf_range)"
        observed checks_sppf_sync_validate env GH_TOKEN="$gh_token" "$PYTHON_BIN" scripts/sppf_sync.py \
          --validate \
          --only-when-relevant \
          --range "$rev_range" \
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
  observed checks_extract_test_evidence env GABION_LSP_TIMEOUT_TICKS=300000 GABION_LSP_TIMEOUT_TICK_NS=1000000 "$PYTHON_BIN" scripts/extract_test_evidence.py --root . --tests tests --out out/test_evidence.json

  step "checks: evidence drift diff (strict)"
  observed checks_git_diff_test_evidence git diff --exit-code out/test_evidence.json

  step "checks: policy check (no monkeypatch)"
  observed checks_no_monkeypatch_policy "$PYTHON_BIN" scripts/no_monkeypatch_policy_check.py --root .

  step "checks: policy check (branchless)"
  observed checks_branchless_policy "$PYTHON_BIN" scripts/branchless_policy_check.py --root .

  step "checks: policy check (defensive fallback)"
  observed checks_defensive_fallback_policy "$PYTHON_BIN" scripts/defensive_fallback_policy_check.py --root .

  step "checks: controller drift audit (advisory, ratchet-ready)"
  local controller_args=(--out artifacts/out/controller_drift.json)
  if [ -n "${CONTROLLER_DRIFT_FAIL_ON:-}" ]; then
    controller_args+=(--fail-on-severity "$CONTROLLER_DRIFT_FAIL_ON")
  fi
  observed checks_controller_drift "$PYTHON_BIN" scripts/governance_controller_audit.py "${controller_args[@]}"

  step "checks: pytest --cov"
  mkdir -p artifacts/test_runs
  timed_observed checks_pytest env PYTHONUNBUFFERED=1 "$PYTHON_BIN" -m pytest \
    --cov=src/gabion \
    --cov-branch \
    --cov-report=term-missing \
    --cov-report=xml:artifacts/test_runs/coverage.xml \
    --cov-report=html:artifacts/test_runs/htmlcov \
    --cov-fail-under=100 \
    --junitxml artifacts/test_runs/junit.xml \
    --log-file artifacts/test_runs/pytest.log \
    --log-file-level=INFO

  step "checks: delta_state_emit"
  timed_observed checks_delta_state_emit env GABION_DIRECT_RUN=1 GABION_LSP_TIMEOUT_TICKS=65000000 GABION_LSP_TIMEOUT_TICK_NS=1000000 "$PYTHON_BIN" -m gabion delta-state-emit

  step "checks: delta_triplets"
  timed_observed checks_delta_triplets env GABION_DIRECT_RUN=1 GABION_LSP_TIMEOUT_TICKS=65000000 GABION_LSP_TIMEOUT_TICK_NS=1000000 "$PYTHON_BIN" -m gabion delta-triplets

  step "checks: governance telemetry emit"
  observed checks_governance_telemetry "$PYTHON_BIN" scripts/governance_telemetry_emit.py \
    --run-id "$step_timing_run_id" \
    --timings "$step_timing_artifact" \
    --history artifacts/out/governance_telemetry_history.json \
    --json-out artifacts/out/governance_telemetry.json \
    --md-out artifacts/audit_reports/governance_telemetry.md

  if $run_extended_checks; then
    step "checks(ext): order_lifetime_check"
    observed checks_order_lifetime "$PYTHON_BIN" scripts/order_lifetime_check.py --root .

    step "checks(ext): structural_hash_policy_check"
    observed checks_structural_hash_policy "$PYTHON_BIN" scripts/structural_hash_policy_check.py --root .

    step "checks(ext): complexity_audit --fail-on-regression"
    timed_observed checks_complexity_audit "$PYTHON_BIN" scripts/complexity_audit.py --root . --fail-on-regression
  fi
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
    local repo_name
    repo_name="$(gh repo view --json nameWithOwner --jq .nameWithOwner 2>/dev/null || true)"
    if [ -z "$repo_name" ]; then
      step "dataflow: restore-resume-checkpoint skipped (unable to resolve repo name)"
      return 0
    fi
    step "dataflow: restore-resume-checkpoint (best effort)"
    observed dataflow_restore_resume_checkpoint env GH_TOKEN="$gh_token" GH_REPO="$repo_name" \
      GH_REF_NAME="${GABION_LOCAL_REF_NAME:-$(git rev-parse --abbrev-ref HEAD)}" \
      GH_RUN_ID="${GABION_LOCAL_RUN_ID:-0}" \
      "$PYTHON_BIN" -m gabion restore-resume-checkpoint \
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
  local dataflow_stage_rc=0
  local dataflow_failed=0
  rm -f "$outputs_file"
  touch "$outputs_file"
  : > "$summary_file"

  set +e
  observed dataflow_run_dataflow_stage env \
    GITHUB_OUTPUT="$outputs_file" \
    GITHUB_STEP_SUMMARY="$summary_file" \
    GABION_DIRECT_RUN=1 \
    GABION_LSP_TIMEOUT_TICKS="${GABION_LSP_TIMEOUT_TICKS:-65000000}" \
    GABION_LSP_TIMEOUT_TICK_NS="${GABION_LSP_TIMEOUT_TICK_NS:-1000000}" \
    GABION_DATAFLOW_DEBUG_DUMP_INTERVAL_SECONDS="${GABION_DATAFLOW_DEBUG_DUMP_INTERVAL_SECONDS:-60}" \
    "$PYTHON_BIN" -m gabion run-dataflow-stage \
    --stage-strictness-profile "run=high"
  dataflow_stage_rc=$?
  set -e

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
    dataflow_failed=1
  fi

  if [ "$dataflow_failed" != "1" ] && [ "$terminal_status" = "unknown" ]; then
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
    dataflow_failed=1
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
  fi

  step "dataflow: deadline profile summary"
  if [ -f artifacts/out/deadline_profile.json ]; then
    "$PYTHON_BIN" scripts/deadline_profile_ci_summary.py \
      --allow-missing-local \
      --step-summary "$log_dir/deadline_profile_summary.md"
  else
    echo "Skipping deadline profile summary (missing artifacts/out/deadline_profile.json)."
  fi

  if [ "$dataflow_stage_rc" -ne 0 ]; then
    dataflow_failed=1
  fi

  if [ "$dataflow_failed" != "0" ]; then
    return 1
  fi
}

run_pr_dataflow_job() {
  step_timing_mode="pr-dataflow"

  local pr_base pr_head
  pr_base="$(resolve_pr_base_sha)"
  pr_head="$(resolve_pr_head_sha)"
  step "pr-dataflow: using diff range base=$pr_base head=$pr_head"

  if [ "$pr_verify_stage_ci" != "0" ]; then
    step "pr-dataflow: verify stage CI succeeded for this SHA"
    verify_stage_ci_for_sha "$pr_head"
  else
    step "pr-dataflow: stage CI verification skipped (enable with --verify-pr-stage-ci)"
  fi

  step "pr-dataflow: policy check (no monkeypatch)"
  observed pr_dataflow_no_monkeypatch_policy "$PYTHON_BIN" scripts/no_monkeypatch_policy_check.py --root .

  step "pr-dataflow: policy check (branchless)"
  observed pr_dataflow_branchless_policy "$PYTHON_BIN" scripts/branchless_policy_check.py --root .

  step "pr-dataflow: policy check (defensive fallback)"
  observed pr_dataflow_defensive_fallback_policy "$PYTHON_BIN" scripts/defensive_fallback_policy_check.py --root .

  step "pr-dataflow: governance PR template fields"
  local pr_template_body_file
  if pr_template_body_file="$(resolve_pr_body_file)"; then
    observed pr_dataflow_governance_template "$PYTHON_BIN" scripts/check_pr_governance_template.py \
      --base "$pr_base" \
      --head "$pr_head" \
      --body-file "$pr_template_body_file"
  else
    observed pr_dataflow_governance_template "$PYTHON_BIN" scripts/check_pr_governance_template.py \
      --base "$pr_base" \
      --head "$pr_head"
  fi

  step "pr-dataflow: controller drift audit (advisory)"
  local pr_controller_args=(--out artifacts/out/controller_drift.json)
  if [ -n "${CONTROLLER_DRIFT_FAIL_ON:-}" ]; then
    pr_controller_args+=(--fail-on-severity "$CONTROLLER_DRIFT_FAIL_ON")
  fi
  observed pr_dataflow_controller_drift "$PYTHON_BIN" scripts/governance_controller_audit.py "${pr_controller_args[@]}"

  step "pr-dataflow: select impacted tests"
  mkdir -p artifacts/audit_reports artifacts/test_runs
  observed pr_dataflow_impact_select "$PYTHON_BIN" -m gabion impact-select-tests \
    --root . \
    --diff-base "$pr_base" \
    --diff-head "$pr_head" \
    --out artifacts/audit_reports/impact_selection.json \
    --confidence-threshold 0.6

  step "pr-dataflow: run impacted tests first (fallback to full)"
  observed pr_dataflow_run_selected_pytest env IMPACT_GATE_MUST_RUN="${IMPACT_GATE_MUST_RUN:-false}" PYTHON_BIN="$PYTHON_BIN" "$PYTHON_BIN" - <<'PY'
import json
import os
import pathlib
import shlex
import subprocess

report = pathlib.Path("artifacts/audit_reports/impact_selection.json")
payload = json.loads(report.read_text(encoding="utf-8"))
mode = str(payload.get("mode", "full"))
selection = payload.get("selection") or {}
if not isinstance(selection, dict):
    selection = {}
impacted = [str(item) for item in selection.get("impacted_tests", []) if str(item).strip()]
must_run = [str(item) for item in selection.get("must_run_impacted_tests", []) if str(item).strip()]
gate_must_run = os.environ.get("IMPACT_GATE_MUST_RUN", "false").lower() in {"1", "true", "yes"}
pytest_base = [
    os.environ["PYTHON_BIN"],
    "-m",
    "pytest",
    "--cov=src/gabion",
    "--cov-branch",
    "--cov-report=term-missing",
    "--cov-report=xml:artifacts/test_runs/coverage.xml",
    "--cov-report=html:artifacts/test_runs/htmlcov",
    "--cov-fail-under=100",
    "--junitxml",
    "artifacts/test_runs/junit.xml",
    "--log-file",
    "artifacts/test_runs/pytest.log",
    "--log-file-level=INFO",
]

def run_pytest(extra):
    cmd = [*pytest_base, *extra]
    print("running:", " ".join(shlex.quote(part) for part in cmd))
    return subprocess.run(cmd, check=False).returncode

if mode == "targeted" and impacted:
    if must_run:
        rc = run_pytest(must_run)
        if rc != 0 and gate_must_run:
            raise SystemExit(rc)
    rc = run_pytest(impacted)
    if rc != 0:
        raise SystemExit(rc)
else:
    rc = run_pytest([])
    if rc != 0:
        raise SystemExit(rc)
PY

  step "pr-dataflow: render dataflow grammar report"
  mkdir -p artifacts/dataflow_grammar artifacts/audit_reports
  timed_observed pr_dataflow_render_check env \
    GABION_LSP_TIMEOUT_TICKS="${GABION_LSP_TIMEOUT_TICKS:-65000000}" \
    GABION_LSP_TIMEOUT_TICK_NS="${GABION_LSP_TIMEOUT_TICK_NS:-1000000}" \
    "$PYTHON_BIN" -m gabion check \
    --profile raw . \
    --root . \
    --report artifacts/dataflow_grammar/report.md \
    --dot artifacts/dataflow_grammar/dataflow_graph.dot \
    --resume-checkpoint artifacts/audit_reports/dataflow_resume_checkpoint_pr.json \
    --resume-on-timeout 1 \
    --emit-timeout-progress-report \
    --type-audit-report \
    --baseline baselines/dataflow_baseline.txt \
    --no-fail-on-violations \
    --no-fail-on-type-ambiguities
}

bootstrap_ci_env

if $run_checks; then
  run_checks_job
fi

if $run_dataflow; then
  run_dataflow_job
fi

if $run_pr_dataflow; then
  run_pr_dataflow_job
fi

step "done"
echo "Local CI reproduction completed successfully."

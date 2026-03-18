from __future__ import annotations

import argparse
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
import urllib.request

from gabion.analysis.foundation.timeout_context import check_deadline
from gabion.invariants import never
from gabion.tooling.runtime import checks_runtime

_DEFAULT_TIMEOUT_NS = "130000000000000ns"
_DEFAULT_PR_STAGE_CI_TIMEOUT_MINUTES = 70


LocalCiReproStep = checks_runtime.ToolingStep


@dataclass(frozen=True)
class LocalCiReproOptions:
    root: Path
    python_bin: Path
    log_dir: Path
    run_checks: bool
    run_dataflow: bool
    run_pr_dataflow: bool
    run_extended_checks: bool
    run_sppf_sync_mode: str
    sppf_range: str
    skip_gabion_check_step: bool
    pr_base_sha: str
    pr_head_sha: str
    pr_body_file: Path | None
    verify_pr_stage_ci: bool
    pr_stage_ci_timeout_minutes: int
    step_timing_enabled: bool
    observability_enabled: bool
    step_timing_artifact: Path
    step_timing_run_id: str
    ci_event_name: str
    aspf_handoff_enabled: bool
    aspf_handoff_manifest: Path
    aspf_handoff_session: str
    aspf_state_root: Path
    before_sha: str
    after_sha: str
    timeout_ns: str
    impact_gate_must_run: bool


def _step(label: str) -> None:
    print()
    print(f"[ci-local] {label}")


def _run_git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _repo_root() -> Path:
    return Path(_run_git("rev-parse", "--show-toplevel"))


def _ensure_aspf_handoff_session(session_id: str) -> str:
    if session_id:
        return session_id
    return f"session-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}-{os.getpid()}"


def _resolve_sppf_range(before_sha: str, after_sha: str, explicit: str) -> str:
    if explicit:
        return explicit
    if not before_sha or before_sha == "0000000000000000000000000000000000000000":
        return "HEAD~20..HEAD"
    try:
        subprocess.run(
            ["git", "cat-file", "-e", f"{after_sha}^{{commit}}"],
            check=True,
            capture_output=True,
            text=True,
        )
        subprocess.run(
            ["git", "cat-file", "-e", f"{before_sha}^{{commit}}"],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError:
        subprocess.run(
            ["git", "fetch", "--no-tags", "origin", before_sha, after_sha],
            check=False,
        )
    for sha in (after_sha, before_sha):
        probe = subprocess.run(
            ["git", "cat-file", "-e", f"{sha}^{{commit}}"],
            check=False,
            capture_output=True,
            text=True,
        )
        if probe.returncode != 0:
            print("Push SHAs unavailable locally; falling back to safe local range.")
            return "HEAD~20..HEAD"
    return f"{before_sha}..{after_sha}"


def _resolve_pr_head_sha(explicit: str) -> str:
    if explicit:
        return explicit
    return _run_git("rev-parse", "HEAD")


def _resolve_pr_base_sha(explicit: str, head_sha: str) -> str:
    if explicit:
        return explicit
    for candidate in ("origin/main", "main"):
        probe = subprocess.run(
            ["git", "rev-parse", "--verify", "--quiet", candidate],
            check=False,
            capture_output=True,
            text=True,
        )
        if probe.returncode == 0:
            return _run_git("merge-base", candidate, head_sha)
    probe = subprocess.run(
        ["git", "rev-parse", "--verify", "--quiet", f"{head_sha}~1"],
        check=False,
        capture_output=True,
        text=True,
    )
    if probe.returncode == 0:
        return _run_git("rev-parse", f"{head_sha}~1")
    return head_sha


def _resolve_pr_body_file(explicit: Path | None, *, log_dir: Path) -> Path | None:
    if explicit is not None and explicit.exists():
        return explicit
    if subprocess.run(
        ["which", "gh"],
        check=False,
        capture_output=True,
        text=True,
    ).returncode != 0:
        return None
    fetched_path = log_dir / "pr_body.md"
    with fetched_path.open("w", encoding="utf-8") as handle:
        proc = subprocess.run(
            ["gh", "pr", "view", "--json", "body", "--jq", ".body"],
            check=False,
            stdout=handle,
            stderr=subprocess.PIPE,
            text=True,
        )
    if proc.returncode == 0:
        return fetched_path
    fetched_path.unlink(missing_ok=True)
    return None


def _gh_auth_available() -> bool:
    if subprocess.run(
        ["which", "gh"],
        check=False,
        capture_output=True,
        text=True,
    ).returncode != 0:
        return False
    status = subprocess.run(
        ["gh", "auth", "status"],
        check=False,
        capture_output=True,
        text=True,
    )
    return status.returncode == 0


def _resolve_env_gh_token() -> str:
    if os.environ.get("GH_TOKEN", "").strip():
        return str(os.environ["GH_TOKEN"])
    if os.environ.get("GITHUB_TOKEN", "").strip():
        return str(os.environ["GITHUB_TOKEN"])
    return ""


def _resolve_repo_name() -> str:
    if os.environ.get("GH_REPO", "").strip():
        return str(os.environ["GH_REPO"])
    if _gh_auth_available():
        proc = subprocess.run(
            ["gh", "repo", "view", "--json", "nameWithOwner", "--jq", ".nameWithOwner"],
            check=False,
            capture_output=True,
            text=True,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            return proc.stdout.strip()
    remote_url = subprocess.run(
        ["git", "config", "--get", "remote.origin.url"],
        check=False,
        capture_output=True,
        text=True,
    ).stdout.strip()
    repo_name = ""
    if remote_url.startswith("git@github.com:"):
        repo_name = remote_url.removeprefix("git@github.com:")
    elif remote_url.startswith("ssh://git@github.com/"):
        repo_name = remote_url.removeprefix("ssh://git@github.com/")
    elif remote_url.startswith("https://github.com/"):
        repo_name = remote_url.removeprefix("https://github.com/")
    return repo_name.removesuffix(".git")


def _wrap_observability(command: Sequence[str], *, label: str, options: LocalCiReproOptions) -> list[str]:
    if not options.observability_enabled:
        return list(command)
    return [
        str(options.python_bin),
        "scripts/ci/ci_observability_guard.py",
        "--label",
        label,
        "--max-gap-seconds",
        os.environ.get("GABION_OBSERVABILITY_MAX_GAP_SECONDS", "5"),
        "--max-wall-seconds",
        os.environ.get("GABION_OBSERVABILITY_MAX_WALL_SECONDS", "1200"),
        "--artifact-path",
        "artifacts/audit_reports/observability_violations.json",
        "--",
        *command,
    ]


def _wrapped_command(
    step: LocalCiReproStep,
    *,
    options: LocalCiReproOptions,
) -> list[str]:
    wrapped = _wrap_observability(step.command, label=step.label, options=options)
    if not step.timed or not options.step_timing_enabled:
        return wrapped
    timing_args = [
        str(options.python_bin),
        "scripts/ci/ci_step_timing_capture.py",
        "--label",
        step.label,
        "--mode",
        step.mode,
        "--run-id",
        options.step_timing_run_id,
        "--artifact-path",
        str(options.step_timing_artifact),
    ]
    return [*timing_args, "--", *wrapped]


def _run_step(step: LocalCiReproStep, *, options: LocalCiReproOptions) -> None:
    _step(step.label)
    env = dict(os.environ)
    env.update({key: value for key, value in step.env})
    command = _wrapped_command(step, options=options)
    subprocess.run(command, check=True, cwd=options.root, env=env)


def _run_optional_posture(options: LocalCiReproOptions) -> None:
    _step("checks: policy_check --posture")
    if options.ci_event_name != "push":
        print(
            f"event '{options.ci_event_name}' is not push; skipping posture check (matches CI skip path)."
        )
        return
    token = os.environ.get("POLICY_GITHUB_TOKEN", "").strip()
    if not token:
        print("POLICY_GITHUB_TOKEN not set; skipping posture check (matches CI skip path).")
        return
    subprocess.run(
        [str(options.python_bin), "-m", "gabion", "policy", "check", "--posture"],
        check=True,
        cwd=options.root,
        env={**os.environ, "POLICY_GITHUB_TOKEN": token},
    )


def _run_optional_sppf_sync(options: LocalCiReproOptions) -> None:
    if options.run_sppf_sync_mode == "skip":
        _step("checks: skipping sppf_sync validation (--skip-sppf-sync)")
        return
    _step("checks: sppf_sync --validate")
    if options.ci_event_name != "push":
        print(
            f"event '{options.ci_event_name}' is not push; skipping sppf_sync validation (matches CI skip path)."
        )
        return
    command = [
        str(options.python_bin),
        "-m",
        "gabion",
        "sppf",
        "sync",
        "--validate",
        "--only-when-relevant",
        "--range",
        options.sppf_range,
        "--require-state",
        "open",
        "--require-label",
        "done-on-stage",
        "--require-label",
        "status/pending-release",
    ]
    if _gh_auth_available():
        subprocess.run(command, check=True, cwd=options.root)
        return
    token = _resolve_env_gh_token()
    if token:
        subprocess.run(
            command,
            check=True,
            cwd=options.root,
            env={**os.environ, "GH_TOKEN": token},
        )
        return
    if options.run_sppf_sync_mode == "force":
        raise SystemExit("GH auth token unavailable but --run-sppf-sync was requested.")
    print("GH auth token unavailable; skipping sppf_sync validation.")


def _render_pr_dataflow_command(options: LocalCiReproOptions) -> tuple[str, ...]:
    inner = [
        str(options.python_bin),
        "-m",
        "gabion",
        "--timeout",
        options.timeout_ns,
        "check",
        "raw",
        "--",
        ".",
        "--root",
        ".",
        "--report",
        "artifacts/dataflow_grammar/report.md",
        "--dot",
        "artifacts/dataflow_grammar/dataflow_graph.dot",
        "--type-audit-report",
        "--baseline",
        "baselines/dataflow_baseline.txt",
    ]
    if not options.aspf_handoff_enabled:
        return tuple(inner)
    return (
        str(options.python_bin),
        "scripts/misc/aspf_handoff.py",
        "run",
        "--root",
        ".",
        "--session-id",
        options.aspf_handoff_session,
        "--step-id",
        "ci-local.pr-dataflow.render-check.raw",
        "--command-profile",
        "ci-local.pr-dataflow.check.raw",
        "--manifest",
        str(options.aspf_handoff_manifest),
        "--state-root",
        str(options.aspf_state_root),
        "--",
        *inner,
    )


def _checks_steps(options: LocalCiReproOptions) -> tuple[LocalCiReproStep, ...]:
    return checks_runtime.build_ci_checks_steps(options)


def _dataflow_stage_command(options: LocalCiReproOptions, *, outputs_file: Path, summary_file: Path) -> tuple[str, ...]:
    command = [
        str(options.python_bin),
        "-m",
        "gabion",
        "--carrier",
        "direct",
        "--timeout",
        options.timeout_ns,
        "run-dataflow-stage",
        "--github-output",
        str(outputs_file),
        "--step-summary",
        str(summary_file),
        "--debug-dump-interval-seconds",
        os.environ.get("GABION_DATAFLOW_DEBUG_DUMP_INTERVAL_SECONDS", "60"),
        "--stage-strictness-profile",
        "run=high",
    ]
    if options.aspf_handoff_enabled:
        command.extend(
            [
                "--aspf-handoff-manifest",
                str(options.aspf_handoff_manifest),
                "--aspf-handoff-session",
                options.aspf_handoff_session,
                "--aspf-state-root",
                str(options.aspf_state_root),
            ]
        )
    else:
        command.append("--no-aspf-handoff")
    return tuple(command)


def _parse_env_file(path: Path) -> dict[str, str]:
    payload: dict[str, str] = {}
    if not path.exists():
        return payload
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        if "=" not in raw_line:
            continue
        key, value = raw_line.split("=", 1)
        payload[key.strip()] = value.strip()
    return payload


def _finalize_dataflow_outcome(options: LocalCiReproOptions, *, outputs_file: Path) -> None:
    values = _parse_env_file(outputs_file)
    terminal_exit = values.get("exit_code", "")
    terminal_state = values.get("analysis_state", "none")
    terminal_stage = values.get("terminal_stage", "none")
    terminal_status = values.get("terminal_status", "unknown")
    attempts_run = values.get("attempts_run", "0")
    if not terminal_exit:
        raise SystemExit("No dataflow audit stage produced an exit code.")
    _run_step(
        LocalCiReproStep(
            label="dataflow: finalize outcome",
            command=(
                str(options.python_bin),
                "scripts/ci/ci_finalize_dataflow_outcome.py",
                "--terminal-exit",
                terminal_exit,
                "--terminal-state",
                terminal_state,
                "--terminal-stage",
                terminal_stage,
                "--terminal-status",
                terminal_status,
                "--attempts-run",
                attempts_run,
            ),
            mode="dataflow",
        ),
        options=options,
    )


def _run_deadline_profile_summary(options: LocalCiReproOptions, *, step_summary: Path) -> None:
    _step("dataflow: deadline profile summary")
    if not (options.root / "artifacts" / "out" / "deadline_profile.json").exists():
        print("Skipping deadline profile summary (missing artifacts/out/deadline_profile.json).")
        return
    subprocess.run(
        [
            str(options.python_bin),
            "-m",
            "scripts.deadline_profile_ci_summary",
            "--allow-missing-local",
            "--step-summary",
            str(step_summary),
        ],
        check=True,
        cwd=options.root,
    )


def _verify_stage_ci_for_sha(options: LocalCiReproOptions, *, sha: str) -> None:
    repo_name = _resolve_repo_name()
    if not repo_name:
        raise SystemExit("Unable to resolve repository name for stage CI verification.")
    deadline = time.time() + options.pr_stage_ci_timeout_minutes * 60
    if _gh_auth_available():
        while True:
            check_deadline()
            proc = subprocess.run(
                ["gh", "api", f"repos/{repo_name}/actions/workflows/ci.yml/runs?branch=stage&per_page=50"],
                check=False,
                capture_output=True,
                text=True,
                cwd=options.root,
            )
            if proc.returncode != 0:
                raise SystemExit(proc.stderr.strip() or "gh api failed while verifying stage CI")
            payload = json.loads(proc.stdout)
            runs = payload.get("workflow_runs", [])
            match = next((r for r in runs if r.get("head_sha") == sha), None)
            if match is None:
                if time.time() > deadline:
                    raise SystemExit(f"Stage CI has not run for {sha}.")
                time.sleep(60)
                continue
            status = match.get("status")
            if status != "completed":
                if time.time() > deadline:
                    raise SystemExit(f"Stage CI for {sha} not complete (status={status}).")
                time.sleep(60)
                continue
            conclusion = match.get("conclusion")
            if conclusion != "success":
                raise SystemExit(f"Stage CI for {sha} not successful (conclusion={conclusion}).")
            print(f"Stage CI OK for {sha}.")
            return
    token = _resolve_env_gh_token()
    if not token:
        raise SystemExit("GitHub auth unavailable; use 'gh auth login' or set GH_TOKEN/GITHUB_TOKEN.")
    url = f"https://api.github.com/repos/{repo_name}/actions/workflows/ci.yml/runs?branch=stage&per_page=50"
    while True:
        check_deadline()
        req = urllib.request.Request(
            url,
            headers={
                "Authorization": f"Bearer {token}",
                "Accept": "application/vnd.github+json",
            },
        )
        with urllib.request.urlopen(req) as response:
            payload = json.loads(response.read().decode("utf-8"))
        runs = payload.get("workflow_runs", [])
        match = next((r for r in runs if r.get("head_sha") == sha), None)
        if match is None:
            if time.time() > deadline:
                raise SystemExit(f"Stage CI has not run for {sha}.")
            time.sleep(60)
            continue
        status = match.get("status")
        if status != "completed":
            if time.time() > deadline:
                raise SystemExit(f"Stage CI for {sha} not complete (status={status}).")
            time.sleep(60)
            continue
        conclusion = match.get("conclusion")
        if conclusion != "success":
            raise SystemExit(f"Stage CI for {sha} not successful (conclusion={conclusion}).")
        print(f"Stage CI OK for {sha}.")
        return


def _run_impacted_pytest(options: LocalCiReproOptions) -> None:
    report = options.root / "artifacts" / "audit_reports" / "impact_selection.json"
    payload = json.loads(report.read_text(encoding="utf-8"))
    mode = str(payload.get("mode", "full"))
    selection = payload.get("selection") or {}
    if not isinstance(selection, dict):
        selection = {}
    impacted = [str(item) for item in selection.get("impacted_tests", []) if str(item).strip()]
    must_run = [str(item) for item in selection.get("must_run_impacted_tests", []) if str(item).strip()]
    pytest_base = [
        str(options.python_bin),
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

    def _run(extra: Iterable[str]) -> int:
        command = [*pytest_base, *extra]
        print("running:", " ".join(shlex.quote(part) for part in command))
        return subprocess.run(command, check=False, cwd=options.root).returncode

    if mode == "targeted" and impacted:
        if must_run:
            rc = _run(must_run)
            if rc != 0 and options.impact_gate_must_run:
                raise SystemExit(rc)
        rc = _run(impacted)
        if rc != 0:
            raise SystemExit(rc)
        return
    rc = _run(())
    if rc != 0:
        raise SystemExit(rc)


def _pr_dataflow_steps(options: LocalCiReproOptions) -> tuple[LocalCiReproStep, ...]:
    body_file = options.pr_body_file
    governance_template_command = [
        str(options.python_bin),
        "scripts/audit/check_pr_governance_template.py",
        "--base",
        options.pr_base_sha,
        "--head",
        options.pr_head_sha,
    ]
    if body_file is not None:
        governance_template_command.extend(["--body-file", str(body_file)])
    return (
        LocalCiReproStep(
            label="pr-dataflow: policy_check --workflows --output artifacts/out/policy_check_result.json",
            command=(
                str(options.python_bin),
                "-m",
                "gabion",
                "policy",
                "check",
                "--workflows",
                "--output",
                "artifacts/out/policy_check_result.json",
            ),
            mode="pr-dataflow",
        ),
        LocalCiReproStep(
            label="pr-dataflow: policy scanner suite",
            command=(
                str(options.python_bin),
                "-m",
                "gabion",
                "policy",
                "scanner",
                "--root",
                ".",
                "--out-dir",
                "artifacts/out",
                "--base-sha",
                options.pr_base_sha,
                "--head-sha",
                options.pr_head_sha,
            ),
            mode="pr-dataflow",
        ),
        LocalCiReproStep(
            label="pr-dataflow: governance PR template fields",
            command=tuple(governance_template_command),
            mode="pr-dataflow",
        ),
        LocalCiReproStep(
            label="pr-dataflow: controller drift audit (advisory)",
            command=(str(options.python_bin), "-m", "gabion", "governance", "controller-audit", "--out", "artifacts/out/controller_drift.json"),
            mode="pr-dataflow",
        ),
        LocalCiReproStep(
            label="pr-dataflow: select impacted tests",
            command=(
                str(options.python_bin),
                "-m",
                "gabion",
                "impact-select-tests",
                "--root",
                ".",
                "--diff-base",
                options.pr_base_sha,
                "--diff-head",
                options.pr_head_sha,
                "--changed-lines-artifact",
                "artifacts/out/changed_lines.json",
                "--evidence-meta-artifact",
                "artifacts/out/evidence_index_meta.json",
                "--out",
                "artifacts/audit_reports/impact_selection.json",
                "--confidence-threshold",
                "0.6",
            ),
            mode="pr-dataflow",
        ),
        LocalCiReproStep(
            label="pr-dataflow: render dataflow grammar report",
            command=_render_pr_dataflow_command(options),
            timed=True,
            mode="pr-dataflow",
        ),
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--checks-only", action="store_true")
    parser.add_argument("--dataflow-only", action="store_true")
    parser.add_argument("--pr-dataflow-only", action="store_true")
    parser.add_argument("--extended-checks", action="store_true")
    parser.add_argument("--skip-sppf-sync", action="store_true")
    parser.add_argument("--run-sppf-sync", action="store_true")
    parser.add_argument("--sppf-range", default=os.environ.get("GABION_LOCAL_SPPF_RANGE", ""))
    parser.add_argument("--skip-gabion-check-step", action="store_true")
    parser.add_argument("--pr-base-sha", default=os.environ.get("GABION_LOCAL_PR_BASE_SHA", os.environ.get("GITHUB_BASE_SHA", "")))
    parser.add_argument("--pr-head-sha", default=os.environ.get("GABION_LOCAL_PR_HEAD_SHA", os.environ.get("GITHUB_HEAD_SHA", "")))
    parser.add_argument("--pr-body-file")
    parser.add_argument("--verify-pr-stage-ci", dest="verify_pr_stage_ci", action="store_true", default=None)
    parser.add_argument("--skip-verify-pr-stage-ci", dest="verify_pr_stage_ci", action="store_false")
    parser.add_argument("--pr-stage-ci-timeout-minutes", type=int, default=int(os.environ.get("GABION_LOCAL_PR_STAGE_CI_TIMEOUT_MINUTES", str(_DEFAULT_PR_STAGE_CI_TIMEOUT_MINUTES))))
    parser.add_argument("--run-observability-guard", action="store_true")
    parser.add_argument("--skip-step-timing", dest="step_timing_enabled", action="store_false", default=None)
    parser.add_argument("--run-step-timing", dest="step_timing_enabled", action="store_true")
    parser.add_argument("--no-aspf-handoff", action="store_true")
    parser.add_argument("--aspf-handoff-manifest", default=os.environ.get("GABION_ASPF_HANDOFF_MANIFEST", "artifacts/out/aspf_handoff_manifest.json"))
    parser.add_argument("--aspf-handoff-session", default=os.environ.get("GABION_ASPF_HANDOFF_SESSION", ""))
    parser.add_argument("--aspf-state-root", default=os.environ.get("GABION_ASPF_STATE_ROOT", "artifacts/out/aspf_state"))
    return parser


def _options_from_namespace(namespace: argparse.Namespace) -> LocalCiReproOptions:
    root = _repo_root()
    python_bin = Path(sys.executable)
    log_dir = root / os.environ.get("CI_LOCAL_LOG_DIR", "artifacts/test_runs/local_ci")
    log_dir.mkdir(parents=True, exist_ok=True)
    if namespace.checks_only:
        run_checks, run_dataflow, run_pr_dataflow = True, False, False
    elif namespace.dataflow_only:
        run_checks, run_dataflow, run_pr_dataflow = False, True, False
    elif namespace.pr_dataflow_only:
        run_checks, run_dataflow, run_pr_dataflow = False, False, True
    else:
        run_checks = True
        run_dataflow = True
        run_pr_dataflow = False
    run_sppf_sync_mode = "auto"
    if namespace.skip_sppf_sync:
        run_sppf_sync_mode = "skip"
    elif namespace.run_sppf_sync:
        run_sppf_sync_mode = "force"
    before_sha = os.environ.get("BEFORE_SHA", os.environ.get("GABION_LOCAL_BEFORE_SHA", ""))
    after_sha = os.environ.get("AFTER_SHA", _run_git("rev-parse", "HEAD"))
    pr_head_sha = _resolve_pr_head_sha(str(namespace.pr_head_sha or ""))
    pr_base_sha = _resolve_pr_base_sha(str(namespace.pr_base_sha or ""), pr_head_sha)
    pr_body_file = _resolve_pr_body_file(
        Path(namespace.pr_body_file) if namespace.pr_body_file else None,
        log_dir=log_dir,
    )
    verify_pr_stage_ci = (
        True if namespace.verify_pr_stage_ci is None else bool(namespace.verify_pr_stage_ci)
    )
    step_timing_enabled = (
        os.environ.get("GABION_CI_STEP_TIMING_CAPTURE", "0") != "0"
        if namespace.step_timing_enabled is None
        else bool(namespace.step_timing_enabled)
    )
    aspf_handoff_enabled = not namespace.no_aspf_handoff and os.environ.get(
        "GABION_ASPF_HANDOFF_ENABLED",
        "1",
    ) != "0"
    aspf_handoff_session = _ensure_aspf_handoff_session(str(namespace.aspf_handoff_session or ""))
    return LocalCiReproOptions(
        root=root,
        python_bin=python_bin,
        log_dir=log_dir,
        run_checks=run_checks,
        run_dataflow=run_dataflow,
        run_pr_dataflow=run_pr_dataflow,
        run_extended_checks=bool(namespace.extended_checks),
        run_sppf_sync_mode=run_sppf_sync_mode,
        sppf_range=_resolve_sppf_range(before_sha, after_sha, str(namespace.sppf_range or "")),
        skip_gabion_check_step=bool(namespace.skip_gabion_check_step),
        pr_base_sha=pr_base_sha,
        pr_head_sha=pr_head_sha,
        pr_body_file=pr_body_file,
        verify_pr_stage_ci=verify_pr_stage_ci,
        pr_stage_ci_timeout_minutes=int(namespace.pr_stage_ci_timeout_minutes),
        step_timing_enabled=step_timing_enabled,
        observability_enabled=bool(namespace.run_observability_guard),
        step_timing_artifact=root / os.environ.get("GABION_CI_STEP_TIMING_ARTIFACT", "artifacts/audit_reports/ci_step_timings.json"),
        step_timing_run_id=os.environ.get("GABION_CI_STEP_TIMING_RUN_ID", f"local-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}-{os.getpid()}"),
        ci_event_name=os.environ.get("GABION_LOCAL_EVENT_NAME", "push"),
        aspf_handoff_enabled=aspf_handoff_enabled,
        aspf_handoff_manifest=root / str(namespace.aspf_handoff_manifest),
        aspf_handoff_session=aspf_handoff_session,
        aspf_state_root=root / str(namespace.aspf_state_root),
        before_sha=before_sha,
        after_sha=after_sha,
        timeout_ns=os.environ.get("GABION_TIMEOUT_NS", _DEFAULT_TIMEOUT_NS),
        impact_gate_must_run=os.environ.get("IMPACT_GATE_MUST_RUN", "false").lower() in {"1", "true", "yes"},
    )


def _run_checks(options: LocalCiReproOptions) -> None:
    (options.root / "artifacts" / "test_runs").mkdir(parents=True, exist_ok=True)
    steps = _checks_steps(options)
    _run_step(steps[0], options=options)
    _run_step(steps[1], options=options)
    _run_step(steps[2], options=options)
    _run_optional_posture(options)
    for step in steps[3:7]:
        _run_step(step, options=options)
    _run_optional_sppf_sync(options)
    for step in steps[7:]:
        _run_step(step, options=options)


def _run_dataflow(options: LocalCiReproOptions) -> None:
    if options.skip_gabion_check_step:
        _step("dataflow: skipping run-dataflow-stage (--skip-gabion-check-step)")
        return
    outputs_file = options.log_dir / "dataflow_stage_outputs.env"
    summary_file = options.log_dir / "dataflow_stage_summary.md"
    outputs_file.parent.mkdir(parents=True, exist_ok=True)
    outputs_file.write_text("", encoding="utf-8")
    summary_file.write_text("", encoding="utf-8")
    _run_step(
        LocalCiReproStep(
            label="dataflow: run-dataflow-stage (single invocation)",
            command=_dataflow_stage_command(options, outputs_file=outputs_file, summary_file=summary_file),
            mode="dataflow",
        ),
        options=options,
    )
    _finalize_dataflow_outcome(options, outputs_file=outputs_file)
    _run_deadline_profile_summary(
        options,
        step_summary=options.log_dir / "deadline_profile_summary.md",
    )


def _run_pr_dataflow(options: LocalCiReproOptions) -> None:
    steps = _pr_dataflow_steps(options)
    _step(f"pr-dataflow: using diff range base={options.pr_base_sha} head={options.pr_head_sha}")
    if options.verify_pr_stage_ci:
        _step("pr-dataflow: verify stage CI succeeded for this SHA")
        _verify_stage_ci_for_sha(options, sha=options.pr_head_sha)
    else:
        _step("pr-dataflow: stage CI verification skipped")
    for step in steps[:-1]:
        _run_step(step, options=options)
    _step("pr-dataflow: run impacted tests first (fallback to full)")
    _run_impacted_pytest(options)
    _run_step(steps[-1], options=options)


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    namespace = parser.parse_args(argv)
    options = _options_from_namespace(namespace)
    os.chdir(options.root)
    if options.run_checks:
        _run_checks(options)
    if options.run_dataflow:
        _run_dataflow(options)
    if options.run_pr_dataflow:
        _run_pr_dataflow(options)
    return 0


__all__ = [
    "LocalCiReproOptions",
    "LocalCiReproStep",
    "_checks_steps",
    "_pr_dataflow_steps",
    "main",
]

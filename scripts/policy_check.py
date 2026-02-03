#!/usr/bin/env python3
import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

try:
    import yaml
except ImportError:  # pragma: no cover - handled as a hard error at runtime.
    yaml = None


REPO_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_DIR = REPO_ROOT / ".github" / "workflows"

ALLOWED_ACTIONS_FILE = REPO_ROOT / "docs" / "allowed_actions.txt"
REQUIRED_RUNNER_LABELS = {"self-hosted", "gpu", "local"}
TRUSTED_BRANCHES = {"main", "stage", "next", "release"}
CONTENT_WRITE_WORKFLOWS = {
    "auto-test-tag.yml",
    "release-tag.yml",
    "mirror-next.yml",
    "promote-release.yml",
}

_SHA_RE = re.compile(r"^[0-9a-fA-F]{40}$")


@dataclass(frozen=True)
class JobContext:
    job_name: str
    path: Path


def _fail(errors):
    for err in errors:
        print(f"policy-check: {err}", file=sys.stderr)
    raise SystemExit(2)


def _load_allowed_actions(path: Path) -> set[str]:
    if not path.exists():
        return set()
    try:
        raw = path.read_text()
    except OSError:
        return set()
    allowed: set[str] = set()
    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        allowed.add(line)
    return allowed


def _yaml_loader():
    if yaml is None:
        return None

    class Loader(yaml.SafeLoader):
        pass

    # Treat "on"/"off"/"yes"/"no" as strings, not booleans (YAML 1.1 quirk).
    for key, values in list(Loader.yaml_implicit_resolvers.items()):
        Loader.yaml_implicit_resolvers[key] = [
            (tag, regexp) for tag, regexp in values if tag != "tag:yaml.org,2002:bool"
        ]
    return Loader


def _load_yaml(path):
    if yaml is None:
        _fail(["PyYAML is required for policy checks (pip install pyyaml)."])
    with path.open("r", encoding="utf-8") as handle:
        loader = _yaml_loader()
        return yaml.load(handle, Loader=loader) or {}


def _normalize_if(expr):
    if not expr:
        return ""
    return re.sub(r"\s+", "", str(expr))


def _runs_on_labels(runs_on):
    if runs_on is None:
        return []
    if isinstance(runs_on, str):
        return [runs_on]
    if isinstance(runs_on, list):
        return [str(item) for item in runs_on]
    if isinstance(runs_on, dict):
        labels = runs_on.get("labels")
        if isinstance(labels, list):
            return [str(item) for item in labels]
        if isinstance(labels, str):
            return [labels]
    return []


def _is_self_hosted(runs_on):
    labels = _runs_on_labels(runs_on)
    return "self-hosted" in labels


def _event_names(on_block):
    if isinstance(on_block, str):
        return {on_block}
    if isinstance(on_block, list):
        return {str(item) for item in on_block}
    if isinstance(on_block, dict):
        return {str(key) for key in on_block.keys()}
    return set()


def _has_tag_push(on_block) -> bool:
    if not isinstance(on_block, dict):
        return False
    push_block = on_block.get("push")
    if push_block is None:
        return False
    if isinstance(push_block, dict):
        tags = push_block.get("tags")
        if isinstance(tags, list):
            return len(tags) > 0
        if isinstance(tags, str):
            return True
    return False


def _check_permissions(
    doc,
    path,
    errors,
    *,
    allow_pr_write=False,
    allow_id_token=False,
    allow_contents_write=False,
):
    # dataflow-bundle: doc, errors
    permissions = doc.get("permissions")
    if permissions is None:
        errors.append(f"{path}: missing top-level permissions")
        return
    if isinstance(permissions, str):
        errors.append(f"{path}: permissions must be a mapping, not {permissions!r}")
        return
    contents = permissions.get("contents")
    if allow_contents_write:
        if contents != "write":
            errors.append(f"{path}: permissions.contents must be 'write'")
    elif contents != "read":
        errors.append(f"{path}: permissions.contents must be 'read'")
    for key, value in permissions.items():
        if key == "contents":
            continue
        if value in ("read", "none"):
            continue
        if allow_pr_write and key == "pull-requests" and value == "write":
            continue
        if allow_id_token and key == "id-token" and value == "write":
            continue
        if allow_contents_write and key == "contents" and value == "write":
            continue
        errors.append(f"{path}: permissions.{key} must be 'read' or 'none'")


def _check_job_permissions(
    job,
    job_ctx: JobContext,
    errors,
    *,
    allow_pr_write=False,
    allow_id_token=False,
    allow_contents_write=False,
):
    # dataflow-bundle: errors, job, job_ctx
    permissions = job.get("permissions")
    if permissions is None:
        return
    if isinstance(permissions, str):
        errors.append(
            f"{job_ctx.path}:{job_ctx.job_name}: job permissions must be a mapping"
        )
        return
    contents = permissions.get("contents")
    if allow_contents_write:
        if contents != "write":
            errors.append(
                f"{job_ctx.path}:{job_ctx.job_name}: permissions.contents must be 'write'"
            )
    elif contents != "read":
        errors.append(
            f"{job_ctx.path}:{job_ctx.job_name}: permissions.contents must be 'read'"
        )
    is_self_hosted = _is_self_hosted(job.get("runs-on"))
    for key, value in permissions.items():
        if key == "contents":
            continue
        if value in ("read", "none"):
            continue
        if (
            allow_pr_write
            and not is_self_hosted
            and key == "pull-requests"
            and value == "write"
        ):
            continue
        if allow_id_token and not is_self_hosted and key == "id-token" and value == "write":
            continue
        if allow_contents_write and key == "contents" and value == "write":
            continue
        errors.append(
            f"{job_ctx.path}:{job_ctx.job_name}: permissions.{key} must be 'read' or 'none'"
        )


def _check_release_tag_workflow(doc, path, errors):
    events = _event_names(doc.get("on"))
    if events != {"workflow_dispatch"}:
        errors.append(f"{path}: release tag workflow must use workflow_dispatch only")
    jobs = doc.get("jobs", {})
    if not isinstance(jobs, dict):
        return
    for name, job in jobs.items():
        if _is_self_hosted(job.get("runs-on")):
            errors.append(f"{path}:{name}: release tag workflow must not use self-hosted")
        cond = _normalize_if(job.get("if"))
        if (
            "github.ref=='refs/heads/release'" not in cond
            and "github.ref=='refs/heads/next'" not in cond
        ):
            errors.append(
                f"{path}:{name}: release tag workflow must guard on refs/heads/release or refs/heads/next"
            )
        if "github.actor==github.repository_owner" not in cond:
            errors.append(
                f"{path}:{name}: release tag workflow must guard on repository owner"
            )


def _check_auto_test_tag_workflow(doc, path, errors):
    events = _event_names(doc.get("on"))
    if events != {"workflow_run"}:
        errors.append(f"{path}: auto test tag workflow must use workflow_run only")
    workflow_run = None
    if isinstance(doc.get("on"), dict):
        workflow_run = doc.get("on").get("workflow_run")
    workflows = None
    if isinstance(workflow_run, dict):
        workflows = workflow_run.get("workflows")
    if not workflows:
        errors.append(f"{path}: auto test tag workflow must specify workflows")
    jobs = doc.get("jobs", {})
    if not isinstance(jobs, dict):
        return
    for name, job in jobs.items():
        if _is_self_hosted(job.get("runs-on")):
            errors.append(f"{path}:{name}: auto test tag workflow must not use self-hosted")
        cond = _normalize_if(job.get("if"))
        if "github.event.workflow_run.conclusion=='success'" not in cond:
            errors.append(f"{path}:{name}: auto test tag workflow must guard on success")
        if "github.event.workflow_run.head_branch=='main'" not in cond:
            errors.append(f"{path}:{name}: auto test tag workflow must guard on main")
        if (
            "github.event.workflow_run.actor.login==github.repository_owner" not in cond
            and "github.event.workflow_run.actor.login=='github-actions[bot]'" not in cond
        ):
            errors.append(f"{path}:{name}: auto test tag workflow must guard on actor")


def _check_mirror_next_workflow(doc, path, errors):
    events = _event_names(doc.get("on"))
    if events != {"push"}:
        errors.append(f"{path}: mirror workflow must use push only")
    push_block = None
    if isinstance(doc.get("on"), dict):
        push_block = doc.get("on").get("push")
    branches = None
    if isinstance(push_block, dict):
        branches = push_block.get("branches")
    if not branches or ("main" not in branches):
        errors.append(f"{path}: mirror workflow must target main branch pushes")
    jobs = doc.get("jobs", {})
    if not isinstance(jobs, dict):
        return
    for name, job in jobs.items():
        if _is_self_hosted(job.get("runs-on")):
            errors.append(f"{path}:{name}: mirror workflow must not use self-hosted")
        cond = _normalize_if(job.get("if"))
        if "github.ref=='refs/heads/main'" not in cond:
            errors.append(f"{path}:{name}: mirror workflow must guard on main")
        if "github.actor==github.repository_owner" not in cond:
            errors.append(
                f"{path}:{name}: mirror workflow must guard on repository owner"
            )


def _check_promote_release_workflow(doc, path, errors):
    events = _event_names(doc.get("on"))
    if events != {"workflow_run"}:
        errors.append(f"{path}: promote workflow must use workflow_run only")
    workflow_run = None
    if isinstance(doc.get("on"), dict):
        workflow_run = doc.get("on").get("workflow_run")
    workflows = None
    if isinstance(workflow_run, dict):
        workflows = workflow_run.get("workflows")
    if not workflows:
        errors.append(f"{path}: promote workflow must specify workflows")
    jobs = doc.get("jobs", {})
    if not isinstance(jobs, dict):
        return
    for name, job in jobs.items():
        if _is_self_hosted(job.get("runs-on")):
            errors.append(f"{path}:{name}: promote workflow must not use self-hosted")
        cond = _normalize_if(job.get("if"))
        if "github.event.workflow_run.conclusion=='success'" not in cond:
            errors.append(f"{path}:{name}: promote workflow must guard on success")
        steps = job.get("steps", [])
        has_tag_artifact = False
        if isinstance(steps, list):
            for step in steps:
                uses = step.get("uses", "")
                if uses.startswith("actions/download-artifact@"):
                    with_block = step.get("with", {}) or {}
                    if with_block.get("name") == "release-testpypi-tag":
                        has_tag_artifact = True
                        break
                run = step.get("run", "")
                if "release-testpypi/tag.txt" in run:
                    has_tag_artifact = True
                    break
        if (
            "startswith(github.event.workflow_run.head_branch,'test-v')" not in cond
            and "startsWith(github.event.workflow_run.head_branch,'test-v')" not in cond
            and not has_tag_artifact
        ):
            errors.append(f"{path}:{name}: promote workflow must guard on test-v tags")
        if (
            "github.event.workflow_run.actor.login==github.repository_owner" not in cond
            and "github.event.workflow_run.actor.login=='github-actions[bot]'" not in cond
        ):
            errors.append(
                f"{path}:{name}: promote workflow must guard on repository owner or github-actions[bot]"
            )


def _check_actions(job, job_ctx: JobContext, errors, allowed_actions: set[str]):
    # dataflow-bundle: errors, job, job_ctx
    steps = job.get("steps", [])
    for idx, step in enumerate(steps):
        uses = step.get("uses")
        if not uses:
            continue
        if uses.startswith("./") or uses.startswith("docker://"):
            continue
        if "@" not in uses:
            errors.append(
                f"{job_ctx.path}:{job_ctx.job_name}: step {idx} uses unpinned action {uses!r}"
            )
            continue
        action_ref, ref = uses.split("@", 1)
        action_parts = action_ref.split("/")
        if len(action_parts) < 2:
            errors.append(
                f"{job_ctx.path}:{job_ctx.job_name}: step {idx} invalid action {uses!r}"
            )
            continue
        action_name = "/".join(action_parts[:2])
        if action_name not in allowed_actions:
            errors.append(
                f"{job_ctx.path}:{job_ctx.job_name}: step {idx} action not allow-listed ({action_name})"
            )
        if not _SHA_RE.match(ref):
            errors.append(
                f"{job_ctx.path}:{job_ctx.job_name}: step {idx} action not pinned to full SHA ({uses})"
            )


def _check_self_hosted_constraints(doc, path, errors):
    # dataflow-bundle: doc, errors
    jobs = doc.get("jobs", {})
    if not isinstance(jobs, dict):
        return
    self_hosted_jobs = []
    for name, job in jobs.items():
        if _is_self_hosted(job.get("runs-on")):
            self_hosted_jobs.append((name, job))

    if not self_hosted_jobs:
        return

    on_block = doc.get("on")
    events = _event_names(on_block)
    if events != {"push"}:
        errors.append(f"{path}: self-hosted workflow must trigger only on push")
    push_block = None
    if isinstance(on_block, dict):
        push_block = on_block.get("push")
    branches = None
    if isinstance(push_block, dict):
        branches = push_block.get("branches")
    if not branches:
        errors.append(f"{path}: push trigger must restrict branches")
    else:
        branch_set = {str(item) for item in branches}
        if not branch_set.issubset(TRUSTED_BRANCHES):
            errors.append(
                f"{path}: push branches must be subset of {sorted(TRUSTED_BRANCHES)}"
            )

    for name, job in self_hosted_jobs:
        runs_on = job.get("runs-on")
        labels = set(_runs_on_labels(runs_on))
        if not labels:
            errors.append(f"{path}:{name}: runs-on must be explicit labels")
        if not REQUIRED_RUNNER_LABELS.issubset(labels):
            errors.append(
                f"{path}:{name}: runs-on must include {sorted(REQUIRED_RUNNER_LABELS)}"
            )
        cond = _normalize_if(job.get("if"))
        if "github.actor==github.repository_owner" not in cond:
            errors.append(
                f"{path}:{name}: missing actor guard (github.actor == github.repository_owner)"
            )


def check_workflows():
    allowed_actions = _load_allowed_actions(ALLOWED_ACTIONS_FILE)
    if not allowed_actions:
        _fail([f"allowed actions list is empty or missing: {ALLOWED_ACTIONS_FILE}"])
    errors = []
    for path in sorted(WORKFLOW_DIR.glob("*.yml")):
        doc = _load_yaml(path)
        jobs = doc.get("jobs", {})
        has_self_hosted = False
        if isinstance(jobs, dict):
            for name, job in jobs.items():
                if _is_self_hosted(job.get("runs-on")):
                    has_self_hosted = True
        events = _event_names(doc.get("on"))
        allow_id_token = _has_tag_push(doc.get("on")) and (not has_self_hosted)
        allow_pr_write = (("pull_request" in events) or ("pull_request_target" in events)) and (
            not has_self_hosted
        )
        allow_contents_write = path.name in CONTENT_WRITE_WORKFLOWS
        if allow_contents_write:
            if path.name == "release-tag.yml":
                _check_release_tag_workflow(doc, path, errors)
            if path.name == "auto-test-tag.yml":
                _check_auto_test_tag_workflow(doc, path, errors)
            if path.name == "mirror-next.yml":
                _check_mirror_next_workflow(doc, path, errors)
            if path.name == "promote-release.yml":
                _check_promote_release_workflow(doc, path, errors)
        _check_permissions(
            doc,
            path,
            errors,
            allow_pr_write=allow_pr_write,
            allow_id_token=allow_id_token,
            allow_contents_write=allow_contents_write,
        )
        _check_self_hosted_constraints(doc, path, errors)
        if isinstance(jobs, dict):
            for name, job in jobs.items():
                job_ctx = JobContext(job_name=str(name), path=path)
                _check_job_permissions(
                    job,
                    job_ctx,
                    errors,
                    allow_pr_write=allow_pr_write,
                    allow_id_token=allow_id_token,
                    allow_contents_write=allow_contents_write,
                )
                _check_actions(job, job_ctx, errors, allowed_actions)
    if errors:
        _fail(errors)


def _api_json(url, token):
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "policy-check",
        "Authorization": f"Bearer {token}",
    }
    request = Request(url, headers=headers)
    with urlopen(request) as response:
        return json.loads(response.read().decode("utf-8"))


def check_posture():
    policy_token = os.environ.get("POLICY_GITHUB_TOKEN")
    token = policy_token or os.environ.get("GITHUB_TOKEN")
    using_policy_token = policy_token is not None
    repo = os.environ.get("GITHUB_REPOSITORY")
    if not token:
        _fail(["missing GITHUB_TOKEN or POLICY_GITHUB_TOKEN for posture check"])
    if policy_token is not None and len(policy_token) < 20:
        _fail(
            [
                "POLICY_GITHUB_TOKEN appears too short; reset the repo secret "
                "with scripts/reset_policy_secret.sh"
            ]
        )
    if not repo:
        _fail(["missing GITHUB_REPOSITORY for posture check"])

    base = f"https://api.github.com/repos/{repo}"
    errors = []
    try:
        perms = _api_json(f"{base}/actions/permissions", token)
    except (HTTPError, URLError) as exc:
        hint = ""
        if isinstance(exc, HTTPError):
            if exc.code == 401:
                hint = " (POLICY_GITHUB_TOKEN is invalid or missing required auth)"
            elif exc.code == 403 and not using_policy_token:
                hint = " (set POLICY_GITHUB_TOKEN with admin read access)"
        _fail([f"actions permissions query failed: {exc}{hint}"])
    allowed = perms.get("allowed_actions")
    if allowed != "selected":
        errors.append("allowed_actions must be 'selected'")

    try:
        workflow_perms = _api_json(f"{base}/actions/permissions/workflow", token)
    except (HTTPError, URLError) as exc:
        hint = ""
        if isinstance(exc, HTTPError):
            if exc.code == 401:
                hint = " (POLICY_GITHUB_TOKEN is invalid or missing required auth)"
            elif exc.code == 403 and not using_policy_token:
                hint = " (set POLICY_GITHUB_TOKEN with admin read access)"
        _fail([f"workflow permissions query failed: {exc}{hint}"])
    default_perms = workflow_perms.get("default_workflow_permissions")
    if default_perms != "read":
        errors.append("default_workflow_permissions must be 'read'")
    if workflow_perms.get("can_approve_pull_request_reviews"):
        errors.append("can_approve_pull_request_reviews must be false")

    try:
        selected = _api_json(
            f"{base}/actions/permissions/selected-actions", token
        )
    except (HTTPError, URLError) as exc:
        hint = ""
        if isinstance(exc, HTTPError):
            if exc.code == 401:
                hint = " (POLICY_GITHUB_TOKEN is invalid or missing required auth)"
            elif exc.code == 403 and not using_policy_token:
                hint = " (set POLICY_GITHUB_TOKEN with admin read access)"
        _fail([f"selected actions query failed: {exc}{hint}"])

    if selected.get("github_owned_allowed") or selected.get("verified_allowed"):
        errors.append("only allow-listed actions are permitted (disable broad allowances)")

    patterns = selected.get("patterns_allowed") or []
    normalized = []
    invalid_patterns = []
    for pattern in patterns:
        if "@" in pattern:
            name, ref = pattern.split("@", 1)
            if ref != "*":
                invalid_patterns.append(pattern)
                continue
            normalized.append(name)
        else:
            normalized.append(pattern)
    if invalid_patterns:
        errors.append(
            f"invalid action patterns (must end with @* or be bare): {sorted(invalid_patterns)}"
        )
    allowed_actions = _load_allowed_actions(ALLOWED_ACTIONS_FILE)
    if set(normalized) != set(allowed_actions):
        errors.append(
            f"allowed action patterns must match {sorted(allowed_actions)}"
        )

    if errors:
        _fail(errors)


def main():
    parser = argparse.ArgumentParser(description="POLICY_SEED guardrails")
    parser.add_argument("--workflows", action="store_true", help="lint workflows")
    parser.add_argument("--posture", action="store_true", help="check GitHub posture")
    args = parser.parse_args()

    if not args.workflows and not args.posture:
        args.workflows = True

    if args.workflows:
        check_workflows()
    if args.posture:
        check_posture()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

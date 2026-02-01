#!/usr/bin/env python3
import argparse
import json
import os
import re
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

try:
    import yaml
except ImportError:  # pragma: no cover - handled as a hard error at runtime.
    yaml = None


REPO_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_DIR = REPO_ROOT / ".github" / "workflows"

ALLOWED_ACTIONS = {
    "actions/checkout",
    "actions/setup-python",
    "actions/upload-artifact",
    "actions/download-artifact",
    "jdx/mise-action",
}
REQUIRED_RUNNER_LABELS = {"self-hosted", "gpu", "local"}
TRUSTED_BRANCHES = {"main", "stage"}

_SHA_RE = re.compile(r"^[0-9a-fA-F]{40}$")


def _fail(errors):
    for err in errors:
        print(f"policy-check: {err}", file=sys.stderr)
    raise SystemExit(2)


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


def _check_permissions(doc, path, errors, *, allow_pr_write=False):
    permissions = doc.get("permissions")
    if permissions is None:
        errors.append(f"{path}: missing top-level permissions")
        return
    if isinstance(permissions, str):
        errors.append(f"{path}: permissions must be a mapping, not {permissions!r}")
        return
    contents = permissions.get("contents")
    if contents != "read":
        errors.append(f"{path}: permissions.contents must be 'read'")
    for key, value in permissions.items():
        if value in ("read", "none"):
            continue
        if allow_pr_write and key == "pull-requests" and value == "write":
            continue
        errors.append(f"{path}: permissions.{key} must be 'read' or 'none'")


def _check_job_permissions(job, path, job_name, errors, *, allow_pr_write=False):
    permissions = job.get("permissions")
    if permissions is None:
        return
    if isinstance(permissions, str):
        errors.append(f"{path}:{job_name}: job permissions must be a mapping")
        return
    contents = permissions.get("contents")
    if contents != "read":
        errors.append(f"{path}:{job_name}: permissions.contents must be 'read'")
    is_self_hosted = _is_self_hosted(job.get("runs-on"))
    for key, value in permissions.items():
        if value in ("read", "none"):
            continue
        if (
            allow_pr_write
            and not is_self_hosted
            and key == "pull-requests"
            and value == "write"
        ):
            continue
        errors.append(
            f"{path}:{job_name}: permissions.{key} must be 'read' or 'none'"
        )


def _check_actions(job, path, job_name, errors):
    steps = job.get("steps", [])
    for idx, step in enumerate(steps):
        uses = step.get("uses")
        if not uses:
            continue
        if uses.startswith("./") or uses.startswith("docker://"):
            continue
        if "@" not in uses:
            errors.append(f"{path}:{job_name}: step {idx} uses unpinned action {uses!r}")
            continue
        action_ref, ref = uses.split("@", 1)
        action_parts = action_ref.split("/")
        if len(action_parts) < 2:
            errors.append(f"{path}:{job_name}: step {idx} invalid action {uses!r}")
            continue
        action_name = "/".join(action_parts[:2])
        if action_name not in ALLOWED_ACTIONS:
            errors.append(f"{path}:{job_name}: step {idx} action not allow-listed ({action_name})")
        if not _SHA_RE.match(ref):
            errors.append(
                f"{path}:{job_name}: step {idx} action not pinned to full SHA ({uses})"
            )


def _check_self_hosted_constraints(doc, path, errors):
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
        allow_pr_write = (("pull_request" in events) or ("pull_request_target" in events)) and (not has_self_hosted)
        _check_permissions(doc, path, errors, allow_pr_write=allow_pr_write)
        _check_self_hosted_constraints(doc, path, errors)
        if isinstance(jobs, dict):
            for name, job in jobs.items():
                _check_job_permissions(
                    job, path, name, errors, allow_pr_write=allow_pr_write
                )
                _check_actions(job, path, name, errors)
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
    allowed_patterns = {f"{name}@*" for name in ALLOWED_ACTIONS}
    if set(patterns) != allowed_patterns:
        errors.append(
            f"allowed action patterns must match {sorted(allowed_patterns)}"
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

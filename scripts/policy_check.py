#!/usr/bin/env python3
import argparse
import ast
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from scripts.deadline_runtime import DeadlineBudget, deadline_scope_from_ticks
from gabion.analysis.timeout_context import Deadline, check_deadline, deadline_clock_scope, deadline_scope
from gabion.invariants import never
from gabion.deadline_clock import MonotonicClock
from gabion.order_contract import ordered_or_sorted
from gabion.config import dataflow_adapter_payload, dataflow_defaults, dataflow_required_surfaces

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

_DEFAULT_POLICY_TIMEOUT_TICKS = 120_000
_DEFAULT_POLICY_TIMEOUT_TICK_NS = 1_000_000
_DEFAULT_POLICY_TIMEOUT_BUDGET = DeadlineBudget(
    ticks=_DEFAULT_POLICY_TIMEOUT_TICKS,
    tick_ns=_DEFAULT_POLICY_TIMEOUT_TICK_NS,
)


NORMATIVE_ENFORCEMENT_MAP = REPO_ROOT / "docs" / "normative_enforcement_map.yaml"
TIER2_RESIDUE_BASELINE = REPO_ROOT / "out" / "tier2_pattern_residue_baseline.json"
_REQUIRED_NORMATIVE_CLAUSES = {
    "NCI-LSP-FIRST",
    "NCI-ACTIONS-PINNED",
    "NCI-ACTIONS-ALLOWLIST",
    "NCI-DATAFLOW-BUNDLE-TIERS",
    "NCI-SHIFT-AMBIGUITY-LEFT",
    "NCI-BASELINE-RATCHET",
    "NCI-DEADLINE-TIMEOUT-PROPAGATION",
    "NCI-CONTROLLER-ADAPTATION-LAW",
    "NCI-OVERRIDE-LIFECYCLE",
    "NCI-CONTROLLER-DRIFT-LIFECYCLE",
    "NCI-COMMAND-MATURITY-PARITY",
    "NCI-DUAL-SENSOR-CORRECTION-LOOP",
}


def _workflow_doc(path: Path):
    if not path.exists():
        return {}
    return _load_yaml(path)


def check_normative_enforcement_map() -> None:
    errors: list[str] = []
    check_deadline()
    if not NORMATIVE_ENFORCEMENT_MAP.exists():
        _fail([f"missing normative enforcement map: {NORMATIVE_ENFORCEMENT_MAP}"])
    payload = _load_yaml(NORMATIVE_ENFORCEMENT_MAP)
    clauses = payload.get("clauses")
    if not isinstance(clauses, dict):
        _fail(["docs/normative_enforcement_map.yaml: clauses must be a mapping"])
    clause_ids = set(clauses.keys())
    missing = _REQUIRED_NORMATIVE_CLAUSES - clause_ids
    extra = clause_ids - _REQUIRED_NORMATIVE_CLAUSES
    if missing:
        errors.append(f"normative map missing required clauses: {_sorted(missing)}")
    if extra:
        errors.append(f"normative map has unknown clause keys: {_sorted(extra)}")

    for clause_id, entry in clauses.items():
        check_deadline()
        if not isinstance(entry, dict):
            errors.append(f"{clause_id}: entry must be mapping")
            continue
        status = entry.get("status")
        if status not in {"enforced", "partial", "document-only"}:
            errors.append(f"{clause_id}: invalid status {status!r}")
        for module_path in entry.get("enforcing_modules") or []:
            check_deadline()
            module_ref = REPO_ROOT / str(module_path)
            if not module_ref.exists():
                errors.append(f"{clause_id}: missing enforcing module path {module_path}")
        ci_anchors = entry.get("ci_anchors") or []
        if not isinstance(ci_anchors, list):
            errors.append(f"{clause_id}: ci_anchors must be a list")
            continue
        for anchor in ci_anchors:
            check_deadline()
            if not isinstance(anchor, dict):
                errors.append(f"{clause_id}: ci anchor must be mapping")
                continue
            workflow = REPO_ROOT / str(anchor.get("workflow", ""))
            job = str(anchor.get("job", ""))
            step = str(anchor.get("step", ""))
            if not workflow.exists():
                errors.append(f"{clause_id}: workflow does not exist: {workflow}")
                continue
            workflow_doc = _workflow_doc(workflow)
            jobs = workflow_doc.get("jobs")
            if not isinstance(jobs, dict) or job not in jobs:
                errors.append(f"{clause_id}: missing workflow job anchor {workflow}:{job}")
                continue
            steps = jobs[job].get("steps") if isinstance(jobs[job], dict) else []
            step_names = {
                str(item.get("name", ""))
                for item in steps
                if isinstance(item, dict)
            }
            if step and step not in step_names:
                errors.append(f"{clause_id}: missing workflow step anchor {workflow}:{job}:{step}")
        artifacts = entry.get("expected_artifacts") or []
        if not isinstance(artifacts, list):
            errors.append(f"{clause_id}: expected_artifacts must be a list")
            continue
        for artifact in artifacts:
            check_deadline()
            if not isinstance(artifact, str) or not artifact.strip():
                errors.append(f"{clause_id}: invalid expected artifact reference {artifact!r}")
    if errors:
        _fail(errors)


def _policy_timeout_budget() -> DeadlineBudget:
    raw_ticks = os.getenv("GABION_POLICY_TIMEOUT_TICKS", "").strip()
    raw_tick_ns = os.getenv("GABION_POLICY_TIMEOUT_TICK_NS", "").strip()
    if raw_ticks or raw_tick_ns:
        if not raw_ticks:
            never("missing policy timeout ticks", tick_ns=raw_tick_ns)
        if not raw_tick_ns:
            never("missing policy timeout tick_ns", ticks=raw_ticks)
        try:
            ticks_value = int(raw_ticks)
        except ValueError:
            ticks_value = -1
        try:
            tick_ns_value = int(raw_tick_ns)
        except ValueError:
            tick_ns_value = -1
        if ticks_value <= 0:
            never("invalid policy timeout ticks", ticks=raw_ticks)
        if tick_ns_value <= 0:
            never("invalid policy timeout tick_ns", tick_ns=raw_tick_ns)
        return DeadlineBudget(
            ticks=ticks_value,
            tick_ns=tick_ns_value,
        )
    return DeadlineBudget(
        ticks=_DEFAULT_POLICY_TIMEOUT_BUDGET.ticks,
        tick_ns=_DEFAULT_POLICY_TIMEOUT_BUDGET.tick_ns,
    )


def _policy_deadline_scope():
    return deadline_scope_from_ticks(
        budget=_policy_timeout_budget(),
    )


def _sorted(values, *, key=None, reverse: bool = False):
    return ordered_or_sorted(
        values,
        source="scripts.policy_check",
        key=key,
        reverse=reverse,
    )


@dataclass(frozen=True)
class JobContext:
    job_name: str
    path: Path


def _fail(errors):
    for err in errors:
        check_deadline()
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
        check_deadline()
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
        check_deadline()
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


def _workflow_run_names(on_block) -> set[str]:
    if not isinstance(on_block, dict):
        return set()
    workflow_run = on_block.get("workflow_run")
    if not isinstance(workflow_run, dict):
        return set()
    workflows = workflow_run.get("workflows")
    if isinstance(workflows, list):
        return {str(item) for item in workflows}
    if isinstance(workflows, str):
        return {workflows}
    return set()


def _push_branches(on_block) -> set[str]:
    if not isinstance(on_block, dict):
        return set()
    push_block = on_block.get("push")
    if not isinstance(push_block, dict):
        return set()
    branches = push_block.get("branches")
    if isinstance(branches, list):
        return {str(item) for item in branches}
    if isinstance(branches, str):
        return {branches}
    return set()


def _push_tags(on_block) -> set[str]:
    if not isinstance(on_block, dict):
        return set()
    push_block = on_block.get("push")
    if not isinstance(push_block, dict):
        return set()
    tags = push_block.get("tags")
    if isinstance(tags, list):
        return {str(item) for item in tags}
    if isinstance(tags, str):
        return {tags}
    return set()


def _is_tag_only_push(on_block) -> bool:
    if not isinstance(on_block, dict):
        return False
    if _event_names(on_block) != {"push"}:
        return False
    push_block = on_block.get("push")
    if not isinstance(push_block, dict):
        return False
    tags = push_block.get("tags")
    if not tags:
        return False
    branches = push_block.get("branches")
    if branches:
        return False
    return True


def _has_actor_guard(cond: str) -> bool:
    return (
        "github.actor==github.repository_owner" in cond
        or "github.actor=='" in cond
        or "github.actor==\"" in cond
    )


def _has_ref_guard(cond: str) -> bool:
    return (
        "github.ref=='refs/heads/" in cond
        or "github.ref==\"refs/heads/" in cond
        or "github.ref=='refs/tags/" in cond
        or "github.ref==\"refs/tags/" in cond
        or "startswith(github.ref,'refs/heads/')" in cond
        or "startsWith(github.ref,'refs/heads/')" in cond
        or "startswith(github.ref,'refs/tags/')" in cond
        or "startsWith(github.ref,'refs/tags/')" in cond
    )


def _check_workflow_dispatch_guards(doc, path, errors):
    events = _event_names(doc.get("on"))
    if "workflow_dispatch" not in events:
        return
    jobs = doc.get("jobs", {})
    if not isinstance(jobs, dict):
        return
    for name, job in jobs.items():
        check_deadline()
        cond = _normalize_if(job.get("if"))
        if not cond:
            errors.append(
                f"{path}:{name}: workflow_dispatch jobs must guard on actor and ref"
            )
            continue
        if not _has_actor_guard(cond):
            errors.append(
                f"{path}:{name}: workflow_dispatch jobs must guard on actor"
            )
        if not _has_ref_guard(cond):
            errors.append(
                f"{path}:{name}: workflow_dispatch jobs must guard on ref"
            )


def _check_permissions(
    doc,
    path,
    errors,
    *,
    allow_pr_write=False,
    allow_id_token=False,
    allow_contents_write=False,
    require_id_token=False,
    require_actions_read=False,
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
    if require_id_token and permissions.get("id-token") != "write":
        errors.append(f"{path}: permissions.id-token must be 'write'")
    if require_actions_read and permissions.get("actions") != "read":
        errors.append(f"{path}: permissions.actions must be 'read'")
    for key, value in permissions.items():
        check_deadline()
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
    require_id_token=False,
    require_actions_read=False,
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
    if require_id_token and permissions.get("id-token") != "write":
        errors.append(
            f"{job_ctx.path}:{job_ctx.job_name}: permissions.id-token must be 'write'"
        )
    if require_actions_read and permissions.get("actions") != "read":
        errors.append(
            f"{job_ctx.path}:{job_ctx.job_name}: permissions.actions must be 'read'"
        )
    is_self_hosted = _is_self_hosted(job.get("runs-on"))
    for key, value in permissions.items():
        check_deadline()
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
        check_deadline()
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
        steps = job.get("steps", [])
        if not _step_run_contains_any(steps, {"scripts/release_tag.py"}):
            errors.append(
                f"{path}:{name}: release tag workflow must use scripts/release_tag.py"
            )
    script_path = REPO_ROOT / "scripts" / "release_tag.py"
    if script_path.exists():
        script_text = script_path.read_text(encoding="utf-8")
        required_tokens = [
            "Next branch must mirror main",
            "Release branch must mirror next",
            "Tag already exists",
            "Test tags must be created from next",
            "Release tags must be created from release",
            "tag_target",
        ]
        missing = [token for token in required_tokens if token not in script_text]
        if missing:
            errors.append(
                f"{path}: release_tag.py missing required guard tokens: {missing}"
            )
    else:
        errors.append(f"{path}: release_tag.py missing (required for release tagging)")


def _check_auto_test_tag_workflow(doc, path, errors):
    events = _event_names(doc.get("on"))
    if events != {"workflow_run"}:
        errors.append(f"{path}: auto test tag workflow must use workflow_run only")
    workflows = _workflow_run_names(doc.get("on"))
    if not workflows:
        errors.append(f"{path}: auto test tag workflow must specify workflows")
    elif "mirror-next" not in workflows:
        errors.append(f"{path}: auto test tag workflow must target mirror-next")
    jobs = doc.get("jobs", {})
    if not isinstance(jobs, dict):
        return
    for name, job in jobs.items():
        check_deadline()
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
        steps = job.get("steps", [])
        if not _step_run_contains_tokens(
            steps, {"origin/main", "origin/next", "Next must mirror main"}
        ):
            errors.append(
                f"{path}:{name}: auto test tag workflow must verify next mirrors main"
            )
        if not _step_run_contains_any(steps, {"scripts/release_read_project_version.py"}):
            errors.append(
                f"{path}:{name}: auto test tag workflow must derive tag from pyproject.toml"
            )
        if not _step_run_contains_any(steps, {"steps.verify.outputs.target_sha"}):
            errors.append(
                f"{path}:{name}: auto test tag workflow must tag verified target_sha"
            )
        if not _step_run_contains_tokens(steps, {"rev-parse", "-q", "--verify", "refs/tags"}):
            errors.append(
                f"{path}:{name}: auto test tag workflow must guard against existing tags"
            )
        if not _step_run_contains_any(steps, {"test-v"}):
            errors.append(f"{path}:{name}: auto test tag workflow must create test-v tags")


def _check_mirror_next_workflow(doc, path, errors):
    events = _event_names(doc.get("on"))
    if events != {"push"}:
        errors.append(f"{path}: mirror workflow must use push only")
    branches = _push_branches(doc.get("on"))
    if branches != {"main"}:
        errors.append(f"{path}: mirror workflow must target main branch pushes")
    tags = _push_tags(doc.get("on"))
    if tags:
        errors.append(f"{path}: mirror workflow must not trigger on tags")
    jobs = doc.get("jobs", {})
    if not isinstance(jobs, dict):
        return
    for name, job in jobs.items():
        check_deadline()
        if _is_self_hosted(job.get("runs-on")):
            errors.append(f"{path}:{name}: mirror workflow must not use self-hosted")
        cond = _normalize_if(job.get("if"))
        if "github.ref=='refs/heads/main'" not in cond:
            errors.append(f"{path}:{name}: mirror workflow must guard on main")
        if "github.actor==github.repository_owner" not in cond:
            errors.append(
                f"{path}:{name}: mirror workflow must guard on repository owner"
            )
        steps = job.get("steps", [])
        if not _step_run_contains_tokens(
            steps, {"merge-base", "--is-ancestor", "origin/next", "origin/main"}
        ):
            errors.append(
                f"{path}:{name}: mirror workflow must verify next is ancestor of main"
            )
        if not _step_run_contains_tokens(
            steps, {"--force-with-lease=refs/heads/next:", "refs/heads/next"}
        ):
            errors.append(
                f"{path}:{name}: mirror workflow must use --force-with-lease for branch update"
            )
        if not _step_run_contains_any(steps, {"HEAD_SHA\":refs/heads/next", "HEAD_SHA:refs/heads/next"}):
            errors.append(
                f"{path}:{name}: mirror workflow must push explicit HEAD_SHA to next"
            )


def _check_promote_release_workflow(doc, path, errors):
    events = _event_names(doc.get("on"))
    if events != {"workflow_run"}:
        errors.append(f"{path}: promote workflow must use workflow_run only")
    workflows = _workflow_run_names(doc.get("on"))
    if not workflows:
        errors.append(f"{path}: promote workflow must specify workflows")
    elif "release-testpypi" not in workflows:
        errors.append(f"{path}: promote workflow must target release-testpypi")
    jobs = doc.get("jobs", {})
    if not isinstance(jobs, dict):
        return
    for name, job in jobs.items():
        check_deadline()
        if _is_self_hosted(job.get("runs-on")):
            errors.append(f"{path}:{name}: promote workflow must not use self-hosted")
        cond = _normalize_if(job.get("if"))
        if "github.event.workflow_run.conclusion=='success'" not in cond:
            errors.append(f"{path}:{name}: promote workflow must guard on success")
        steps = job.get("steps", [])
        has_tag_artifact = False
        if isinstance(steps, list):
            for step in steps:
                check_deadline()
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
        steps = job.get("steps", [])
        if not _step_run_contains_tokens(
            steps, {"merge-base", "--is-ancestor", "origin/main"}
        ):
            errors.append(
                f"{path}:{name}: promote workflow must verify tested commit is ancestor of main"
            )
        if not _step_run_contains_tokens(
            steps, {"NEXT_SHA", "HEAD_SHA", "Next must mirror tested commit"}
        ):
            errors.append(
                f"{path}:{name}: promote workflow must verify tested commit matches next"
            )
        if not _step_run_contains_tokens(
            steps, {"--force-with-lease=refs/heads/release:", "refs/heads/release"}
        ):
            errors.append(
                f"{path}:{name}: promote workflow must use --force-with-lease for branch update"
            )
        if not _step_run_contains_any(steps, {"HEAD_SHA\":refs/heads/release", "HEAD_SHA:refs/heads/release"}):
            errors.append(
                f"{path}:{name}: promote workflow must push explicit HEAD_SHA to release"
            )


def _step_run_contains_tokens(steps, tokens: set[str]) -> bool:
    for step in steps:
        check_deadline()
        if not isinstance(step, dict):
            continue
        run = step.get("run")
        if not isinstance(run, str):
            continue
        if all(token in run for token in tokens):
            return True
    return False


def _step_run_contains_any(steps, tokens: set[str]) -> bool:
    for step in steps:
        check_deadline()
        if not isinstance(step, dict):
            continue
        run = step.get("run")
        if not isinstance(run, str):
            continue
        if any(token in run for token in tokens):
            return True
    return False




def _check_ci_script_entrypoints(doc, path, errors):
    check_deadline()
    if path.name != "ci.yml":
        return
    jobs = doc.get("jobs", {})
    if not isinstance(jobs, dict):
        return
    required_tokens = {
        "scripts/ci_finalize_dataflow_outcome.py",
        "scripts/ci_controller_drift_gate.py",
        "scripts/ci_override_record_emit.py",
        "scripts/aspf_handoff.py run",
        "check delta-bundle",
        "check delta-gates",
    }
    steps = []
    for job in jobs.values():
        check_deadline()
        if not isinstance(job, dict):
            continue
        raw_steps = job.get("steps", [])
        if isinstance(raw_steps, list):
            steps.extend(raw_steps)
    for token in sorted(required_tokens):
        check_deadline()
        if not _step_run_contains_any(steps, {token}):
            errors.append(f"{path}: ci workflow must invoke {token}")

def _check_release_testpypi_workflow(doc, path, errors):
    on_block = doc.get("on")
    workflows = _workflow_run_names(on_block)
    if workflows and "auto-test-tag" not in workflows:
        errors.append(f"{path}: release-testpypi workflow_run must target auto-test-tag")
    tags = _push_tags(on_block)
    if "push" in _event_names(on_block) and not tags:
        errors.append(f"{path}: release-testpypi push trigger must include tag patterns")
    if tags and not any("test-v" in tag for tag in tags):
        errors.append(f"{path}: release-testpypi push tags must include test-v*")
    jobs = doc.get("jobs", {})
    if not isinstance(jobs, dict):
        return
    required_tokens = {
        "refs/tags/$TAG",
        "origin/main",
        "origin/next",
        "TAG_SHA",
    }
    required_scripts = {
        "scripts/release_verify_test_tag.py",
    }
    for name, job in jobs.items():
        check_deadline()
        cond = _normalize_if(job.get("if"))
        if "github.event.workflow_run.conclusion=='success'" not in cond:
            errors.append(
                f"{path}:{name}: release-testpypi workflow_run must guard on success"
            )
        if (
            "github.event.workflow_run.actor.login==github.repository_owner" not in cond
            and "github.event.workflow_run.actor.login=='github-actions[bot]'" not in cond
        ):
            errors.append(
                f"{path}:{name}: release-testpypi workflow_run must guard on actor"
            )
        steps = job.get("steps", [])
        if not (
            _step_run_contains_tokens(steps, required_tokens)
            or _step_run_contains_any(steps, required_scripts)
        ):
            errors.append(
                f"{path}:{name}: release-testpypi workflow must verify tag equals main/next"
            )


def _check_release_pypi_workflow(doc, path, errors):
    on_block = doc.get("on")
    workflows = _workflow_run_names(on_block)
    if workflows and "release-tag" not in workflows:
        errors.append(f"{path}: release-pypi workflow_run must target release-tag")
    tags = _push_tags(on_block)
    if "push" in _event_names(on_block) and not tags:
        errors.append(f"{path}: release-pypi push trigger must include tag patterns")
    if tags and not any(tag == "v*" or tag.startswith("v") for tag in tags):
        errors.append(f"{path}: release-pypi push tags must include v*")
    if tags and not any("!test-v" in tag for tag in tags):
        errors.append(f"{path}: release-pypi push tags must exclude test-v*")
    jobs = doc.get("jobs", {})
    if not isinstance(jobs, dict):
        return
    required_tokens = {
        "origin/main",
        "origin/next",
        "origin/release",
        "TAG_SHA",
    }
    required_scripts = {
        "scripts/release_verify_pypi_tag.py",
    }
    for name, job in jobs.items():
        check_deadline()
        cond = _normalize_if(job.get("if"))
        if "github.event.workflow_run.conclusion=='success'" not in cond:
            errors.append(
                f"{path}:{name}: release-pypi workflow_run must guard on success"
            )
        if (
            "github.event.workflow_run.actor.login==github.repository_owner" not in cond
            and "github.event.workflow_run.actor.login=='github-actions[bot]'" not in cond
        ):
            errors.append(
                f"{path}:{name}: release-pypi workflow_run must guard on actor"
            )
        steps = job.get("steps", [])
        if not (
            _step_run_contains_tokens(steps, required_tokens)
            or _step_run_contains_any(steps, required_scripts)
        ):
            errors.append(
                f"{path}:{name}: release-pypi workflow must verify tag equals main/next/release"
            )


def _job_uses_action(job, action_prefix: str) -> bool:
    steps = job.get("steps", [])
    for step in steps:
        check_deadline()
        if not isinstance(step, dict):
            continue
        uses = step.get("uses")
        if isinstance(uses, str) and uses.startswith(action_prefix):
            return True
    return False


def _job_has_same_repo_pr_guard(job) -> bool:
    guard_tokens = {
        "github.event.pull_request.head.repo.full_name==github.repository",
        "github.event.pull_request.head.repo.fork==false",
    }
    cond = _normalize_if(job.get("if"))
    if any(token in cond for token in guard_tokens):
        return True
    for step in job.get("steps", []):
        check_deadline()
        if not isinstance(step, dict):
            continue
        cond = _normalize_if(step.get("if"))
        if any(token in cond for token in guard_tokens):
            return True
    return False


def _check_pr_comment_guard(doc, path, errors):
    jobs = doc.get("jobs", {})
    if not isinstance(jobs, dict):
        return
    for name, job in jobs.items():
        check_deadline()
        permissions = job.get("permissions")
        if not isinstance(permissions, dict):
            continue
        if permissions.get("pull-requests") != "write":
            continue
        if not _job_has_same_repo_pr_guard(job):
            errors.append(
                f"{path}:{name}: pull-requests write requires same-repo PR guard"
            )


def _check_id_token_scoping(doc, path, errors):
    jobs = doc.get("jobs", {})
    if not isinstance(jobs, dict):
        return
    if len(jobs) <= 1:
        return
    top_perms = doc.get("permissions")
    if isinstance(top_perms, dict) and top_perms.get("id-token") == "write":
        errors.append(
            f"{path}: id-token write must be scoped to publishing jobs when multiple jobs are present"
        )
    for name, job in jobs.items():
        check_deadline()
        permissions = job.get("permissions")
        has_id_token = isinstance(permissions, dict) and permissions.get("id-token") == "write"
        uses_publish = _job_uses_action(job, "pypa/gh-action-pypi-publish@")
        if uses_publish and not has_id_token:
            errors.append(
                f"{path}:{name}: publishing job must request id-token: write"
            )
        if has_id_token and not uses_publish:
            errors.append(
                f"{path}:{name}: only publishing jobs may request id-token: write"
            )


def _check_actions(job, job_ctx: JobContext, errors, allowed_actions: set[str]):
    # dataflow-bundle: errors, job, job_ctx
    steps = job.get("steps", [])
    for idx, step in enumerate(steps):
        check_deadline()
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
        check_deadline()
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
        tags = push_block.get("tags")
        if tags:
            errors.append(f"{path}: self-hosted workflow must not trigger on tags")
    if not branches:
        errors.append(f"{path}: push trigger must restrict branches")
    else:
        branch_set = {str(item) for item in branches}
        if not branch_set.issubset(TRUSTED_BRANCHES):
            errors.append(
                f"{path}: push branches must be subset of {_sorted(TRUSTED_BRANCHES)}"
            )

    for name, job in self_hosted_jobs:
        check_deadline()
        runs_on = job.get("runs-on")
        labels = set(_runs_on_labels(runs_on))
        if not labels:
            errors.append(f"{path}:{name}: runs-on must be explicit labels")
        if not REQUIRED_RUNNER_LABELS.issubset(labels):
            errors.append(
                f"{path}:{name}: runs-on must include {_sorted(REQUIRED_RUNNER_LABELS)}"
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
    for path in _sorted(WORKFLOW_DIR.glob("*.yml")):
        check_deadline()
        doc = _load_yaml(path)
        jobs = doc.get("jobs", {})
        has_self_hosted = False
        if isinstance(jobs, dict):
            for name, job in jobs.items():
                check_deadline()
                if _is_self_hosted(job.get("runs-on")):
                    has_self_hosted = True
        events = _event_names(doc.get("on"))
        allow_id_token = _is_tag_only_push(doc.get("on")) and (not has_self_hosted)
        allow_pr_write = (("pull_request" in events) or ("pull_request_target" in events)) and (
            not has_self_hosted
        )
        allow_contents_write = path.name in CONTENT_WRITE_WORKFLOWS
        require_id_token = False
        require_actions_read = False
        if path.name in {"release-testpypi.yml", "release-pypi.yml"} and not has_self_hosted:
            allow_id_token = True
            require_id_token = True
            require_actions_read = True
        if allow_contents_write:
            if path.name == "release-tag.yml":
                _check_release_tag_workflow(doc, path, errors)
            if path.name == "auto-test-tag.yml":
                _check_auto_test_tag_workflow(doc, path, errors)
            if path.name == "mirror-next.yml":
                _check_mirror_next_workflow(doc, path, errors)
            if path.name == "promote-release.yml":
                _check_promote_release_workflow(doc, path, errors)
        if path.name == "release-testpypi.yml":
            _check_release_testpypi_workflow(doc, path, errors)
            _check_id_token_scoping(doc, path, errors)
        if path.name == "release-pypi.yml":
            _check_release_pypi_workflow(doc, path, errors)
            _check_id_token_scoping(doc, path, errors)
        _check_workflow_dispatch_guards(doc, path, errors)
        _check_ci_script_entrypoints(doc, path, errors)
        _check_permissions(
            doc,
            path,
            errors,
            allow_pr_write=allow_pr_write,
            allow_id_token=allow_id_token,
            allow_contents_write=allow_contents_write,
            require_id_token=require_id_token,
            require_actions_read=require_actions_read,
        )
        _check_self_hosted_constraints(doc, path, errors)
        if allow_pr_write:
            _check_pr_comment_guard(doc, path, errors)
        if isinstance(jobs, dict):
            for name, job in jobs.items():
                check_deadline()
                job_ctx = JobContext(job_name=str(name), path=path)
                _check_job_permissions(
                    job,
                    job_ctx,
                    errors,
                    allow_pr_write=allow_pr_write,
                    allow_id_token=allow_id_token,
                    allow_contents_write=allow_contents_write,
                    require_id_token=require_id_token,
                    require_actions_read=require_actions_read,
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


def _gh_auth_token() -> str | None:
    try:
        proc = subprocess.run(
            ["gh", "auth", "token"],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None
    if proc.returncode != 0:
        return None
    token = proc.stdout.strip()
    if not token:
        return None
    return token


def check_posture():
    policy_token = os.environ.get("POLICY_GITHUB_TOKEN")
    env_token = os.environ.get("GITHUB_TOKEN")
    gh_token = _gh_auth_token()
    token = policy_token or env_token or gh_token
    using_policy_token = policy_token is not None
    repo = os.environ.get("GITHUB_REPOSITORY")
    if not token:
        _fail(
            [
                "missing GITHUB_TOKEN or POLICY_GITHUB_TOKEN for posture check "
                "(or ensure `gh auth token` is available)"
            ]
        )
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
        check_deadline()
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
            f"invalid action patterns (must end with @* or be bare): {_sorted(invalid_patterns)}"
        )
    allowed_actions = _load_allowed_actions(ALLOWED_ACTIONS_FILE)
    if set(normalized) != set(allowed_actions):
        errors.append(
            f"allowed action patterns must match {_sorted(allowed_actions)}"
        )

    if errors:
        _fail(errors)


def _load_tier2_residue_baseline(path: Path) -> set[str]:
    if not path.exists():
        return set()
    payload = json.loads(path.read_text(encoding="utf-8"))
    residues = payload.get("residues", []) if isinstance(payload, dict) else []
    if not isinstance(residues, list):
        return set()
    return {str(item) for item in residues if isinstance(item, str)}


def check_tier2_residue_contract() -> None:
    from gabion.analysis import dataflow_audit

    source_path = REPO_ROOT / "src" / "gabion" / "analysis" / "dataflow_audit.py"
    try:
        source = source_path.read_text(encoding="utf-8")
    except OSError as exc:
        _fail([f"tier2 residue policy check failed to read source: {exc}"])
    with deadline_clock_scope(MonotonicClock()):
        with deadline_scope(Deadline.from_timeout_ms(10_000)):
            instances = dataflow_audit._pattern_schema_matches(
                groups_by_path={},
                source=source,
                source_path=source_path,
            )
            residues = dataflow_audit._tier2_unreified_residue_entries(
                dataflow_audit._pattern_schema_residue_entries(instances)
            )
    current = {
        f"{entry.reason}:{entry.payload.get('kind', '')}:{entry.schema_id}"
        for entry in residues
    }
    baseline = _load_tier2_residue_baseline(TIER2_RESIDUE_BASELINE)
    new_keys = current - baseline
    if new_keys:
        _fail([
            "tier2 residue policy check failed (new unreified Tier-2 residues)",
            *[f"new residue: {item}" for item in _sorted(new_keys)],
        ])

def check_ambiguity_contract() -> None:
    cmd = [
        sys.executable,
        "-m",
        "gabion",
        "ambiguity-contract-gate",
        "--root",
        str(REPO_ROOT),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        stdout = proc.stdout.strip()
        stderr = proc.stderr.strip()
        details = []
        if stdout:
            details.append(stdout)
        if stderr:
            details.append(stderr)
        _fail(["ambiguity contract policy check failed", *details])



_SEMANTIC_CORE_PAYLOAD_BRANCH_MODULES = (
    REPO_ROOT / "src" / "gabion" / "analysis" / "dataflow_audit.py",
    REPO_ROOT / "src" / "gabion" / "analysis" / "timeout_context.py",
)

_ALLOWED_PAYLOAD_BRANCH_FUNCTION_PREFIXES = (
    "_decode_",
)

_KNOWN_ADAPTER_SURFACES = {
    "bundle-inference": "bundle_inference",
    "decision-surfaces": "decision_surfaces",
    "type-flow": "type_flow",
    "exception-obligations": "exception_obligations",
    "rewrite-plan-support": "rewrite_plan_support",
}



def _raw_payload_branching_violations(path: Path) -> list[str]:
    check_deadline()
    try:
        source = path.read_text(encoding="utf-8")
    except OSError as exc:
        return [f"{path}: failed to read source ({exc})"]
    try:
        module = ast.parse(source, filename=str(path))
    except SyntaxError as exc:
        return [f"{path}: failed to parse source ({exc})"]

    function_ranges: list[tuple[int, int, str]] = []
    for node in ast.walk(module):
        check_deadline()
        if isinstance(node, ast.FunctionDef):
            start = int(getattr(node, "lineno", 0) or 0)
            end = int(getattr(node, "end_lineno", start) or start)
            function_ranges.append((start, end, node.name))

    def _function_name_for_line(line_no: int) -> str | None:
        check_deadline()
        best: tuple[int, str] | None = None
        for start, end, name in function_ranges:
            check_deadline()
            if start <= line_no <= end:
                width = end - start
                if best is None or width < best[0]:
                    best = (width, name)
        return best[1] if best is not None else None

    def _is_allowed(line_no: int) -> bool:
        name = _function_name_for_line(line_no)
        if name is None:
            return False
        return name.startswith(_ALLOWED_PAYLOAD_BRANCH_FUNCTION_PREFIXES)

    def _pattern_mentions_raw_payload(pattern: ast.pattern) -> bool:
        check_deadline()
        if isinstance(pattern, ast.MatchClass):
            cls = pattern.cls
            if isinstance(cls, ast.Name) and cls.id in {"Mapping", "list"}:
                return True
            if isinstance(cls, ast.Attribute) and cls.attr in {"Mapping", "list"}:
                return True
        for child in ast.iter_child_nodes(pattern):
            check_deadline()
            if isinstance(child, ast.pattern) and _pattern_mentions_raw_payload(child):
                return True
        return False

    def _isinstance_mentions_raw_payload(call: ast.Call) -> bool:
        check_deadline()
        if not (isinstance(call.func, ast.Name) and call.func.id == "isinstance"):
            return False
        if len(call.args) < 2:
            return False
        type_arg = call.args[1]
        targets: list[ast.expr] = []
        if isinstance(type_arg, ast.Tuple):
            targets.extend(type_arg.elts)
        else:
            targets.append(type_arg)
        for target in targets:
            check_deadline()
            if isinstance(target, ast.Name) and target.id in {"Mapping", "list"}:
                return True
            if isinstance(target, ast.Attribute) and target.attr in {"Mapping", "list"}:
                return True
        return False

    try:
        display_path = str(path.relative_to(REPO_ROOT))
    except ValueError:
        display_path = str(path)

    violations: list[str] = []
    for node in ast.walk(module):
        check_deadline()
        if isinstance(node, ast.match_case):
            lineno = int(getattr(node.pattern, "lineno", 0) or 0)
            if lineno > 0 and _pattern_mentions_raw_payload(node.pattern) and not _is_allowed(lineno):
                violations.append(f"{display_path}:{lineno}: raw Mapping/list match outside boundary decode")
        if isinstance(node, ast.Call):
            lineno = int(getattr(node, "lineno", 0) or 0)
            if lineno > 0 and _isinstance_mentions_raw_payload(node) and not _is_allowed(lineno):
                violations.append(f"{display_path}:{lineno}: isinstance Mapping/list branch outside boundary decode")
    return violations


def check_semantic_core_payload_branching() -> None:
    errors: list[str] = []
    for path in _SEMANTIC_CORE_PAYLOAD_BRANCH_MODULES:
        check_deadline()
        errors.extend(_raw_payload_branching_violations(path))
    if errors:
        _fail([
            "semantic-core payload branching policy check failed",
            *errors,
        ])

def check_adapter_surface_policy() -> None:
    errors: list[str] = []
    dataflow = dataflow_defaults(root=REPO_ROOT)
    required = set(dataflow_required_surfaces(dataflow))
    unknown = required - set(_KNOWN_ADAPTER_SURFACES)
    if unknown:
        errors.append(f"unknown required adapter surfaces: {_sorted(unknown)}")
    adapter_payload = dataflow_adapter_payload(dataflow)
    capabilities = adapter_payload.get("capabilities", {}) if isinstance(adapter_payload, dict) else {}
    if not isinstance(capabilities, dict):
        capabilities = {}
    for surface in required & set(_KNOWN_ADAPTER_SURFACES):
        check_deadline()
        capability_key = _KNOWN_ADAPTER_SURFACES[surface]
        if capabilities.get(capability_key) is False:
            errors.append(
                f"required adapter surface {surface!r} is disabled via capability {capability_key!r}"
            )
    if errors:
        _fail(errors)

def main():
    parser = argparse.ArgumentParser(description="POLICY_SEED guardrails")
    parser.add_argument("--workflows", action="store_true", help="lint workflows")
    parser.add_argument("--posture", action="store_true", help="check GitHub posture")
    parser.add_argument("--ambiguity-contract", action="store_true", help="run ambiguity contract policy checks")
    parser.add_argument("--normative-map", action="store_true", help="validate docs/normative_enforcement_map.yaml")
    parser.add_argument("--tier2-residue-contract", action="store_true", help="run tier-2 residue policy checks")
    parser.add_argument("--adapter-surfaces", action="store_true", help="validate configured adapter surface requirements")
    parser.add_argument("--semantic-core-payload-branching", action="store_true", help="forbid raw Mapping/list payload branching outside boundary decode functions")
    args = parser.parse_args()

    if not args.workflows and not args.posture and not args.ambiguity_contract and not args.normative_map and not args.tier2_residue_contract and not args.adapter_surfaces and not args.semantic_core_payload_branching:
        args.workflows = True

    with _policy_deadline_scope():
        if args.workflows:
            check_workflows()
        if args.posture:
            check_posture()
        if args.ambiguity_contract:
            check_ambiguity_contract()
        if args.normative_map:
            check_normative_enforcement_map()
        if args.tier2_residue_contract:
            check_tier2_residue_contract()
        if args.adapter_surfaces:
            check_adapter_surface_policy()
        if args.semantic_core_payload_branching:
            check_semantic_core_payload_branching()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

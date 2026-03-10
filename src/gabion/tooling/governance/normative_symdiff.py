#!/usr/bin/env python3
from __future__ import annotations

import argparse
import io
import json
import math
import re
from contextlib import contextmanager, redirect_stderr
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import singledispatch
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from typing import Callable, Iterable, Mapping

from gabion.order_contract import sort_once
from gabion.tooling.governance import governance_audit
from gabion.invariants import never

from gabion import server
from gabion.tooling.governance import ambiguity_contract_policy_check

from scripts.policy import branchless_policy_check
from scripts.policy import defensive_fallback_policy_check
from scripts.governance import governance_controller_audit
from scripts.policy import no_monkeypatch_policy_check
from scripts.misc import order_lifetime_check
from scripts.policy import policy_check
from scripts.policy import structural_hash_policy_check
from gabion.tooling.runtime.policy_scan_batch import build_policy_scan_batch


_CLAUSE_HEADING_RE = re.compile(r"^###\s+`(?P<id>NCI-[A-Z0-9-]+)`(?:\s|$)")
_SEVERITY_RANK = {"critical": 4, "high": 3, "medium": 2, "low": 1}
_SEVERITY_WEIGHT = {"critical": 8, "high": 5, "medium": 3, "low": 1}


@dataclass(frozen=True)
class GapItem:
    gap_id: str
    layer: str
    direction: str
    model: str
    severity: str
    count: int
    message: str
    evidence: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "gap_id": self.gap_id,
            "layer": self.layer,
            "direction": self.direction,
            "model": self.model,
            "severity": self.severity,
            "count": int(self.count),
            "message": self.message,
            "evidence": list(self.evidence),
        }


@dataclass(frozen=True)
class ScopeInventory:
    normative_docs: tuple[str, ...]
    core_layer_docs: tuple[str, ...]
    extended_layer_docs: tuple[str, ...]
    outside_default_strict_docs: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "normative_docs": list(self.normative_docs),
            "core_layer_docs": list(self.core_layer_docs),
            "extended_layer_docs": list(self.extended_layer_docs),
            "outside_default_strict_docs": list(self.outside_default_strict_docs),
            "counts": {
                "normative_docs": len(self.normative_docs),
                "core_layer_docs": len(self.core_layer_docs),
                "extended_layer_docs": len(self.extended_layer_docs),
                "outside_default_strict_docs": len(self.outside_default_strict_docs),
            },
        }


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


@singledispatch
def _coerce_int(value: object) -> int:
    never("unregistered runtime type", value_type=type(value).__name__)


@_coerce_int.register(bool)
def _sd_reg_1(value: bool) -> int:
    return int(value)


@_coerce_int.register(int)
def _sd_reg_2(value: int) -> int:
    return value


@_coerce_int.register(float)
def _sd_reg_3(value: float) -> int:
    return int(value)


@_coerce_int.register(str)
def _sd_reg_4(value: str) -> int:
    try:
        return int(value.strip())
    except ValueError:
        return 0


@_coerce_int.register(type(None))
def _sd_reg_5(value: None) -> int:
    _ = value
    return 0


@_coerce_int.register(object)
def _sd_reg_6(value: object) -> int:
    _ = value
    return 0


def _message_path_prefix(message: str) -> str | None:
    if ":" not in message:
        return None
    prefix = message.split(":", 1)[0].strip()
    if not prefix:
        return None
    return prefix


@singledispatch
def _mapping_optional(value: object):
    never("unregistered runtime type", value_type=type(value).__name__)


@_mapping_optional.register(dict)
def _sd_reg_7(value: dict[object, object]):
    return value


def _none_mapping(value: object):
    _ = value
    return None


for _mapping_none_type in (
    list,
    tuple,
    set,
    str,
    int,
    float,
    bool,
    type(None),
):
    _mapping_optional.register(_mapping_none_type)(_none_mapping)


@singledispatch
def _list_optional(value: object):
    never("unregistered runtime type", value_type=type(value).__name__)


@_list_optional.register(list)
def _sd_reg_8(value: list[object]):
    return value


@_list_optional.register(tuple)
def _sd_reg_9(value: tuple[object, ...]):
    return list(value)


def _none_list(value: object):
    _ = value
    return None


for _list_none_type in (
    dict,
    set,
    str,
    int,
    float,
    bool,
    type(None),
):
    _list_optional.register(_list_none_type)(_none_list)


@singledispatch
def _str_optional(value: object):
    never("unregistered runtime type", value_type=type(value).__name__)


@_str_optional.register(str)
def _sd_reg_10(value: str):
    return value


def _none_str(value: object):
    _ = value
    return None


for _str_none_type in (
    dict,
    list,
    tuple,
    set,
    int,
    float,
    bool,
    type(None),
):
    _str_optional.register(_str_none_type)(_none_str)


def _mapping_entries(values: object) -> list[Mapping[object, object]]:
    entries: list[Mapping[object, object]] = []
    items = _list_optional(values)
    if items is None:
        return entries
    for item in items:
        mapping = _mapping_optional(item)
        if mapping is not None:
            entries.append(mapping)
    return entries


def _string_entries(values: object) -> list[str]:
    entries: list[str] = []
    items = _list_optional(values)
    if items is None:
        return entries
    for item in items:
        text = _str_optional(item)
        if text is not None:
            entries.append(text)
    return entries


def _ordered_strings(values: Iterable[object], *, source: str) -> list[str]:
    seen: dict[str, None] = {}
    for value in values:
        text = str(value).strip()
        if not text:
            continue
        if text not in seen:
            seen[text] = None
    return list(sort_once(seen.keys(), source=source))


def _ordered_gap_items(items: Iterable[GapItem], *, source: str) -> list[GapItem]:
    return list(
        sort_once(
            list(items),
            source=source,
            key=lambda item: (
                -int(_SEVERITY_RANK.get(item.severity, 0)),
                item.layer,
                item.direction,
                item.model,
                item.gap_id,
                item.message,
            ),
        )
    )


def _parse_frontmatter(path: Path) -> tuple[dict[str, object], str]:
    text = path.read_text(encoding="utf-8")
    frontmatter, body = governance_audit._parse_frontmatter(text)
    mapping = _mapping_optional(frontmatter)
    normalized = dict(mapping) if mapping is not None else {}
    return normalized, body


def _iter_markdown_paths(root: Path) -> list[Path]:
    return list(
        sort_once(
            root.rglob("*.md"),
            source="normative_symdiff.iter_markdown_paths",
            key=lambda item: item.as_posix(),
        )
    )


def collect_scope_inventory(root: Path) -> ScopeInventory:
    normative_docs: list[str] = []
    for path in _iter_markdown_paths(root):
        if not path.is_file():
            continue
        rel = path.relative_to(root).as_posix()
        frontmatter, _ = _parse_frontmatter(path)
        authority = frontmatter.get("doc_authority")
        if authority == "normative":
            normative_docs.append(rel)
    normative_ordered = _ordered_strings(
        normative_docs,
        source="normative_symdiff.collect_scope_inventory.normative",
    )
    core_layer_docs = _ordered_strings(
        governance_audit.GOVERNANCE_DOCS,
        source="normative_symdiff.collect_scope_inventory.core_layer",
    )
    core_set = set(core_layer_docs)
    extended_layer_docs = _ordered_strings(
        [path for path in normative_ordered if path not in core_set],
        source="normative_symdiff.collect_scope_inventory.extended_layer",
    )
    outside_default = _ordered_strings(
        extended_layer_docs,
        source="normative_symdiff.collect_scope_inventory.outside_default",
    )
    return ScopeInventory(
        normative_docs=tuple(normative_ordered),
        core_layer_docs=tuple(core_layer_docs),
        extended_layer_docs=tuple(extended_layer_docs),
        outside_default_strict_docs=tuple(outside_default),
    )


def _parse_clause_ids(clause_index_path: Path) -> list[str]:
    ids: list[str] = []
    for raw in clause_index_path.read_text(encoding="utf-8").splitlines():
        match = _CLAUSE_HEADING_RE.match(raw.strip())
        if match is not None:
            ids.append(match.group("id"))
    return _ordered_strings(ids, source="normative_symdiff.parse_clause_ids")


def _workflow_anchor_errors(
    *,
    root: Path,
    clauses_payload: Mapping[str, object],
) -> list[str]:
    errors: list[str] = []
    workflow_cache: dict[str, Mapping[str, object]] = {}
    for clause_id, raw in clauses_payload.items():
        clause_key = _str_optional(clause_id)
        clause_payload = _mapping_optional(raw)
        if clause_key is None or clause_payload is None:
            continue
        ci_anchor_entries = _list_optional(clause_payload.get("ci_anchors"))
        if ci_anchor_entries is None:
            errors.append(f"{clause_key}: ci_anchors must be a list")
            continue
        for anchor in ci_anchor_entries:
            anchor_mapping = _mapping_optional(anchor)
            if anchor_mapping is None:
                errors.append(f"{clause_key}: ci anchor must be mapping")
                continue
            workflow_ref = str(anchor_mapping.get("workflow", "")).strip()
            job = str(anchor_mapping.get("job", "")).strip()
            step = str(anchor_mapping.get("step", "")).strip()
            workflow_path = root / workflow_ref
            if not workflow_path.exists():
                errors.append(f"{clause_key}: missing workflow {workflow_ref}")
                continue
            cache_key = workflow_path.as_posix()
            workflow_doc = workflow_cache.get(cache_key)
            if workflow_doc is None:
                loaded = policy_check._load_yaml(workflow_path)
                loaded_mapping = _mapping_optional(loaded)
                workflow_doc = dict(loaded_mapping) if loaded_mapping is not None else {}
                workflow_cache[cache_key] = workflow_doc
            jobs = workflow_doc.get("jobs")
            jobs_mapping = _mapping_optional(jobs)
            if jobs_mapping is None or job not in jobs_mapping:
                errors.append(f"{clause_key}: missing workflow job anchor {workflow_ref}:{job}")
                continue
            if not step:
                continue
            job_payload = jobs_mapping.get(job)
            job_mapping = _mapping_optional(job_payload)
            steps = _list_optional(job_mapping.get("steps")) if job_mapping is not None else None
            step_names: list[str] = []
            if steps is not None:
                for item in steps:
                    step_mapping = _mapping_optional(item)
                    if step_mapping is not None:
                        step_names.append(str(step_mapping.get("name", "")))
            if step not in step_names:
                errors.append(
                    f"{clause_key}: missing workflow step anchor {workflow_ref}:{job}:{step}"
                )
    return _ordered_strings(errors, source="normative_symdiff.workflow_anchor_errors")


def analyze_clause_enforcement(
    *,
    root: Path,
    clause_ids: list[str],
    enforcement_map_path: Path,
) -> dict[str, object]:
    payload = policy_check._load_yaml(enforcement_map_path)
    payload_mapping = _mapping_optional(payload)
    clauses_payload = payload_mapping.get("clauses") if payload_mapping is not None else None
    normalized_clauses = (
        dict(clauses_payload)
        if _mapping_optional(clauses_payload) is not None
        else {}
    )
    clause_set = set(clause_ids)
    map_ids = _ordered_strings(
        normalized_clauses.keys(),
        source="normative_symdiff.analyze_clause_enforcement.map_ids",
    )
    map_set = set(map_ids)
    missing_from_map = _ordered_strings(
        [clause for clause in clause_ids if clause not in map_set],
        source="normative_symdiff.analyze_clause_enforcement.missing_from_map",
    )
    unknown_in_map = _ordered_strings(
        [clause for clause in map_ids if clause not in clause_set],
        source="normative_symdiff.analyze_clause_enforcement.unknown_in_map",
    )
    status_by_clause: dict[str, str] = {}
    partial_clauses: list[str] = []
    missing_modules: list[str] = []
    for clause_id in map_ids:
        raw = normalized_clauses.get(clause_id)
        raw_mapping = _mapping_optional(raw)
        if raw_mapping is None:
            continue
        status = str(raw_mapping.get("status", "")).strip()
        status_by_clause[clause_id] = status
        if status == "partial":
            partial_clauses.append(clause_id)
        enforcing_modules = _list_optional(raw_mapping.get("enforcing_modules"))
        if enforcing_modules is None:
            continue
        for module_path in enforcing_modules:
            module_ref = root / str(module_path)
            if not module_ref.exists():
                missing_modules.append(f"{clause_id}: missing enforcing module {module_path}")
    ci_anchor_errors = _workflow_anchor_errors(root=root, clauses_payload=normalized_clauses)
    return {
        "clause_ids": list(clause_ids),
        "map_ids": list(map_ids),
        "missing_from_map": list(missing_from_map),
        "unknown_in_map": list(unknown_in_map),
        "partial_clauses": _ordered_strings(
            partial_clauses,
            source="normative_symdiff.analyze_clause_enforcement.partial_clauses",
        ),
        "status_by_clause": status_by_clause,
        "missing_modules": _ordered_strings(
            missing_modules,
            source="normative_symdiff.analyze_clause_enforcement.missing_modules",
        ),
        "ci_anchor_errors": list(ci_anchor_errors),
    }


@contextmanager
def _policy_check_repo_scope(root: Path):
    saved = {
        "REPO_ROOT": policy_check.REPO_ROOT,
        "WORKFLOW_DIR": policy_check.WORKFLOW_DIR,
        "ALLOWED_ACTIONS_FILE": policy_check.ALLOWED_ACTIONS_FILE,
        "NORMATIVE_ENFORCEMENT_MAP": policy_check.NORMATIVE_ENFORCEMENT_MAP,
    }
    policy_check.REPO_ROOT = root
    policy_check.WORKFLOW_DIR = root / ".github" / "workflows"
    policy_check.ALLOWED_ACTIONS_FILE = root / "docs" / "allowed_actions.txt"
    policy_check.NORMATIVE_ENFORCEMENT_MAP = root / "docs" / "normative_enforcement_map.yaml"
    try:
        yield
    finally:
        policy_check.REPO_ROOT = saved["REPO_ROOT"]
        policy_check.WORKFLOW_DIR = saved["WORKFLOW_DIR"]
        policy_check.ALLOWED_ACTIONS_FILE = saved["ALLOWED_ACTIONS_FILE"]
        policy_check.NORMATIVE_ENFORCEMENT_MAP = saved["NORMATIVE_ENFORCEMENT_MAP"]


@contextmanager
def _controller_audit_repo_scope(root: Path):
    saved = {
        "REPO_ROOT": governance_controller_audit.REPO_ROOT,
        "POLICY_PATH": governance_controller_audit.POLICY_PATH,
        "DEFAULT_OUT": governance_controller_audit.DEFAULT_OUT,
    }
    governance_controller_audit.REPO_ROOT = root
    governance_controller_audit.POLICY_PATH = root / "POLICY_SEED.md"
    governance_controller_audit.DEFAULT_OUT = root / "artifacts" / "out" / "controller_drift.json"
    try:
        yield
    finally:
        governance_controller_audit.REPO_ROOT = saved["REPO_ROOT"]
        governance_controller_audit.POLICY_PATH = saved["POLICY_PATH"]
        governance_controller_audit.DEFAULT_OUT = saved["DEFAULT_OUT"]


def _capture_policy_check(name: str, fn: Callable[[], object]) -> dict[str, object]:
    stream = io.StringIO()
    with redirect_stderr(stream):
        try:
            fn()
            exit_code = 0
        except SystemExit as exc:
            code = exc.code
            code_int = _coerce_int(code)
            exit_code = code_int if code_int != 0 else 1
    stderr_text = stream.getvalue().strip()
    return {
        "name": name,
        "ok": exit_code == 0,
        "exit_code": exit_code,
        "stderr": stderr_text,
    }


def _collect_controller_drift(root: Path) -> dict[str, object]:
    with TemporaryDirectory(prefix="normative_symdiff_controller_") as temp_dir:
        out_path = Path(temp_dir) / "controller_drift.json"
        with _controller_audit_repo_scope(root):
            exit_code = governance_controller_audit.run(
                policy_path=root / "POLICY_SEED.md",
                out_path=out_path,
                fail_on_severity=None,
            )
        if out_path.exists():
            payload = json.loads(out_path.read_text(encoding="utf-8"))
        else:
            payload = {}
    findings = payload.get("findings", [])
    findings_list = _mapping_entries(findings)
    findings_by_sensor: dict[str, int] = {}
    for item in findings_list:
        sensor = str(item.get("sensor", "")).strip()
        if not sensor:
            continue
        findings_by_sensor[sensor] = findings_by_sensor.get(sensor, 0) + 1
    return {
        "ok": int(exit_code) == 0,
        "exit_code": int(exit_code),
        "summary": payload.get("summary", {}),
        "findings_by_sensor": findings_by_sensor,
        "findings": [dict(item) for item in findings_list],
    }


def _collect_lsp_parity(root: Path) -> dict[str, object]:
    ls = SimpleNamespace(workspace=SimpleNamespace(root_path=str(root)))
    payload: dict[str, object] = {"root": str(root)}
    result = server._execute_lsp_parity_gate_total(ls, payload)
    checked_commands = result.get("checked_commands", [])
    checked_list = _list_optional(checked_commands)
    checked_count = len(checked_list) if checked_list is not None else 0
    errors = result.get("errors", [])
    error_list = _list_optional(errors)
    error_count = len(error_list) if error_list is not None else 0
    exit_code = _coerce_int(result.get("exit_code"))
    return {
        "ok": exit_code == 0,
        "exit_code": exit_code,
        "checked_command_count": checked_count,
        "error_count": error_count,
        "errors": list(error_list) if error_list is not None else [],
    }


def _collect_ambiguity_probe(root: Path) -> dict[str, object]:
    batch = build_policy_scan_batch(
        root=root.resolve(),
        target_globs=ambiguity_contract_policy_check.TARGETS,
    )
    violations = ambiguity_contract_policy_check.collect_violations(batch=batch)
    by_rule: dict[str, int] = {}
    by_path: dict[str, int] = {}
    for item in violations:
        by_rule[item.rule_id] = by_rule.get(item.rule_id, 0) + 1
        by_path[item.path] = by_path.get(item.path, 0) + 1
    return {
        "total": len(violations),
        "by_rule": by_rule,
        "by_path": by_path,
    }


def _collect_branchless_probe(root: Path) -> dict[str, object]:
    baseline_path = root / "baselines" / "branchless_policy_baseline.json"
    batch = build_policy_scan_batch(
        root=root.resolve(),
        target_globs=(branchless_policy_check.TARGET_GLOB,),
    )
    violations = branchless_policy_check.collect_violations(batch=batch)
    baseline_keys = branchless_policy_check._load_baseline(baseline_path)
    new_violations = [item for item in violations if item.key not in baseline_keys]
    return {
        "baseline_path": baseline_path.as_posix(),
        "total": len(violations),
        "baseline_keys": len(baseline_keys),
        "new": len(new_violations),
    }


def _collect_defensive_probe(root: Path) -> dict[str, object]:
    baseline_path = root / "baselines" / "defensive_fallback_policy_baseline.json"
    batch = build_policy_scan_batch(
        root=root.resolve(),
        target_globs=(defensive_fallback_policy_check.TARGET_GLOB,),
    )
    violations = defensive_fallback_policy_check.collect_violations(batch=batch)
    baseline_keys = defensive_fallback_policy_check._load_baseline(baseline_path)
    new_violations = [item for item in violations if item.key not in baseline_keys]
    return {
        "baseline_path": baseline_path.as_posix(),
        "total": len(violations),
        "baseline_keys": len(baseline_keys),
        "new": len(new_violations),
    }


def _collect_no_monkeypatch_probe(root: Path) -> dict[str, object]:
    batch = build_policy_scan_batch(
        root=root.resolve(),
        target_globs=no_monkeypatch_policy_check.TARGET_GLOBS,
    )
    violations = no_monkeypatch_policy_check.collect_violations(batch=batch)
    by_path: dict[str, int] = {}
    for item in violations:
        by_path[item.path] = by_path.get(item.path, 0) + 1
    return {
        "total": len(violations),
        "by_path": by_path,
    }


def _collect_order_lifetime_probe(root: Path) -> dict[str, object]:
    batch = build_policy_scan_batch(
        root=root.resolve(),
        target_globs=(order_lifetime_check.TARGET_GLOB,),
    )
    violations = order_lifetime_check.collect_violations(batch=batch)
    by_path: dict[str, int] = {}
    for item in violations:
        by_path[item.path] = by_path.get(item.path, 0) + 1
    return {
        "total": len(violations),
        "by_path": by_path,
    }


def _collect_structural_hash_probe(root: Path) -> dict[str, object]:
    batch = build_policy_scan_batch(
        root=root.resolve(),
        target_globs=(),
        files=structural_hash_policy_check._target_files(root.resolve()),
    )
    violations = structural_hash_policy_check.collect_violations(batch=batch)
    by_path: dict[str, int] = {}
    for item in violations:
        by_path[item.path] = by_path.get(item.path, 0) + 1
    return {
        "total": len(violations),
        "by_path": by_path,
    }


def _collect_docflow_context(
    *,
    root: Path,
    extra_paths: list[str],
    extra_strict: bool,
) -> dict[str, object]:
    with governance_audit._audit_deadline_scope():
        context = governance_audit._docflow_audit_context(
            root=root,
            extra_paths=extra_paths,
            extra_strict=extra_strict,
            sppf_gh_ref_mode="required",
        )
    return {
        "violations": list(context.violations),
        "warnings": list(context.warnings),
        "violation_count": len(context.violations),
        "warning_count": len(context.warnings),
    }


def _collect_agent_instruction_probe(root: Path) -> dict[str, object]:
    with TemporaryDirectory(prefix="normative_symdiff_agent_instruction_") as temp_dir:
        json_path = Path(temp_dir) / "agent_instruction_drift.json"
        md_path = Path(temp_dir) / "agent_instruction_drift.md"
        with governance_audit._audit_deadline_scope():
            docs = governance_audit._load_docflow_docs(root=root, extra_paths=["in", "out"])
            warnings, violations = governance_audit._agent_instruction_graph(
                root=root,
                docs=docs,
                json_output=json_path,
                md_output=md_path,
            )
        payload = (
            json.loads(json_path.read_text(encoding="utf-8"))
            if json_path.exists()
            else {}
        )
    payload_mapping = _mapping_optional(payload)
    summary = payload_mapping.get("summary", {}) if payload_mapping is not None else {}
    hidden = payload.get("hidden_operational_toggles", [])
    summary_mapping = _mapping_optional(summary)
    hidden_entries = _list_optional(hidden)
    return {
        "warnings": list(warnings),
        "violations": list(violations),
        "summary": dict(summary_mapping) if summary_mapping is not None else {},
        "hidden_operational_toggles": list(hidden_entries) if hidden_entries is not None else [],
    }


def _collect_default_probes(
    *,
    root: Path,
    extended_layer_docs: list[str],
) -> dict[str, object]:
    with governance_audit._audit_deadline_scope():
        with _policy_check_repo_scope(root):
            workflow_policy = _capture_policy_check(
                "workflow_policy_check",
                policy_check.check_workflows,
            )
            normative_map_policy = _capture_policy_check(
                "normative_map_policy_check",
                policy_check.check_normative_enforcement_map,
            )
        controller_drift = _collect_controller_drift(root)
        lsp_parity = _collect_lsp_parity(root)
        ambiguity = _collect_ambiguity_probe(root)
        branchless = _collect_branchless_probe(root)
        defensive_fallback = _collect_defensive_probe(root)
        no_monkeypatch = _collect_no_monkeypatch_probe(root)
        order_lifetime = _collect_order_lifetime_probe(root)
        structural_hash = _collect_structural_hash_probe(root)
        core_docflow = _collect_docflow_context(
            root=root,
            extra_paths=["in", "out"],
            extra_strict=False,
        )
        extended_docflow = _collect_docflow_context(
            root=root,
            extra_paths=extended_layer_docs,
            extra_strict=True,
        )
    extended_doc_set = set(extended_layer_docs)
    extended_violations = [
        item
        for item in extended_docflow["violations"]
        if _message_path_prefix(str(item)) in extended_doc_set
    ]
    extended_warnings = [
        item
        for item in extended_docflow["warnings"]
        if _message_path_prefix(str(item)) in extended_doc_set
    ]
    agent_instruction = _collect_agent_instruction_probe(root)
    return {
        "workflow_policy": workflow_policy,
        "normative_map_policy": normative_map_policy,
        "controller_drift": controller_drift,
        "lsp_parity_gate": lsp_parity,
        "ambiguity_contract": ambiguity,
        "branchless_policy": branchless,
        "defensive_fallback_policy": defensive_fallback,
        "no_monkeypatch_policy": no_monkeypatch,
        "order_lifetime_policy": order_lifetime,
        "structural_hash_policy": structural_hash,
        "docflow_core": core_docflow,
        "docflow_extended_strict": {
            **extended_docflow,
            "extended_violations": extended_violations,
            "extended_warnings": extended_warnings,
            "extended_violation_count": len(extended_violations),
            "extended_warning_count": len(extended_warnings),
        },
        "agent_instruction_graph": agent_instruction,
    }


def _add_gap(
    items: list[GapItem],
    *,
    gap_id: str,
    layer: str,
    direction: str,
    model: str,
    severity: str,
    count: int,
    message: str,
    evidence: list[str],
) -> None:
    if count <= 0:
        return
    items.append(
        GapItem(
            gap_id=gap_id,
            layer=layer,
            direction=direction,
            model=model,
            severity=severity,
            count=count,
            message=message,
            evidence=tuple(_ordered_strings(evidence, source=f"normative_symdiff.gap.{gap_id}.evidence")),
        )
    )


def synthesize_gaps(
    *,
    scope_inventory: ScopeInventory,
    clause_analysis: Mapping[str, object],
    probes: Mapping[str, object],
) -> dict[str, object]:
    doc_to_code: list[GapItem] = []
    code_to_doc: list[GapItem] = []

    missing_from_map = list(clause_analysis.get("missing_from_map", []))
    unknown_in_map = list(clause_analysis.get("unknown_in_map", []))
    partial_clauses = list(clause_analysis.get("partial_clauses", []))
    missing_modules = list(clause_analysis.get("missing_modules", []))
    ci_anchor_errors = list(clause_analysis.get("ci_anchor_errors", []))

    _add_gap(
        doc_to_code,
        gap_id="DOC-CODE-CLAUSE-MISSING",
        layer="core",
        direction="doc_to_code",
        model="both",
        severity="high",
        count=len(missing_from_map),
        message="Canonical clauses missing from normative enforcement map.",
        evidence=[str(item) for item in missing_from_map],
    )
    _add_gap(
        doc_to_code,
        gap_id="DOC-CODE-CLAUSE-UNKNOWN",
        layer="core",
        direction="doc_to_code",
        model="both",
        severity="medium",
        count=len(unknown_in_map),
        message="Normative enforcement map contains non-canonical clause keys.",
        evidence=[str(item) for item in unknown_in_map],
    )
    _add_gap(
        doc_to_code,
        gap_id="DOC-CODE-CLAUSE-PARTIAL",
        layer="core",
        direction="doc_to_code",
        model="absolute",
        severity="medium",
        count=len(partial_clauses),
        message="Canonical clauses are mapped but only partially enforced.",
        evidence=[str(item) for item in partial_clauses],
    )
    _add_gap(
        doc_to_code,
        gap_id="DOC-CODE-ENFORCEMENT-PATHS",
        layer="core",
        direction="doc_to_code",
        model="both",
        severity="high",
        count=len(missing_modules) + len(ci_anchor_errors),
        message="Normative enforcement map references missing module/CI anchors.",
        evidence=[*map(str, missing_modules), *map(str, ci_anchor_errors)],
    )

    workflow_policy = _mapping_optional(probes.get("workflow_policy", {}))
    if workflow_policy is not None:
        _add_gap(
            doc_to_code,
            gap_id="DOC-CODE-WORKFLOW-POLICY",
            layer="core",
            direction="doc_to_code",
            model="both",
            severity="high",
            count=0 if bool(workflow_policy.get("ok")) else 1,
            message="Workflow policy checks report policy-seed contradictions.",
            evidence=[str(workflow_policy.get("stderr", ""))],
        )

    lsp_parity = _mapping_optional(probes.get("lsp_parity_gate", {}))
    if lsp_parity is not None:
        error_count = _coerce_int(lsp_parity.get("error_count"))
        _add_gap(
            doc_to_code,
            gap_id="DOC-CODE-LSP-PARITY",
            layer="core",
            direction="doc_to_code",
            model="both",
            severity="high",
            count=error_count,
            message="LSP parity gate reports command maturity/carrier/parity drift.",
            evidence=_string_entries(lsp_parity.get("errors", [])),
        )

    controller_drift = _mapping_optional(probes.get("controller_drift", {}))
    if controller_drift is not None:
        summary = _mapping_optional(controller_drift.get("summary", {}))
        high_findings = _coerce_int(summary.get("high_severity_findings")) if summary is not None else 0
        _add_gap(
            doc_to_code,
            gap_id="DOC-CODE-CONTROLLER-DRIFT",
            layer="core",
            direction="doc_to_code",
            model="both",
            severity="high",
            count=high_findings,
            message="Controller drift audit reports unresolved high-severity findings.",
            evidence=[
                f"{item.get('sensor')}::{item.get('detail')}"
                for item in _mapping_entries(controller_drift.get("findings", []))
            ],
        )

    ambiguity_probe = _mapping_optional(probes.get("ambiguity_contract", {}))
    if ambiguity_probe is not None:
        _add_gap(
            doc_to_code,
            gap_id="DOC-CODE-AMBIGUITY-TOTAL",
            layer="core",
            direction="doc_to_code",
            model="absolute",
            severity="medium",
            count=_coerce_int(ambiguity_probe.get("total")),
            message="Ambiguity-contract unshielded debt remains non-zero.",
            evidence=[f"by_rule={ambiguity_probe.get('by_rule', {})}"],
        )

    branchless_probe = _mapping_optional(probes.get("branchless_policy", {}))
    if branchless_probe is not None:
        _add_gap(
            doc_to_code,
            gap_id="DOC-CODE-BRANCHLESS-NEW",
            layer="core",
            direction="doc_to_code",
            model="ratchet",
            severity="high",
            count=_coerce_int(branchless_probe.get("new")),
            message="Branchless policy baseline ratchet regressed with net-new violations.",
            evidence=[f"baseline_keys={branchless_probe.get('baseline_keys', 0)}"],
        )
        _add_gap(
            doc_to_code,
            gap_id="DOC-CODE-BRANCHLESS-TOTAL",
            layer="core",
            direction="doc_to_code",
            model="absolute",
            severity="medium",
            count=_coerce_int(branchless_probe.get("total")),
            message="Branchless policy absolute debt remains non-zero.",
            evidence=[f"baseline_keys={branchless_probe.get('baseline_keys', 0)}"],
        )

    defensive_probe = _mapping_optional(probes.get("defensive_fallback_policy", {}))
    if defensive_probe is not None:
        _add_gap(
            doc_to_code,
            gap_id="DOC-CODE-DEFENSIVE-NEW",
            layer="core",
            direction="doc_to_code",
            model="ratchet",
            severity="high",
            count=_coerce_int(defensive_probe.get("new")),
            message="Defensive-fallback baseline ratchet regressed with net-new violations.",
            evidence=[f"baseline_keys={defensive_probe.get('baseline_keys', 0)}"],
        )
        _add_gap(
            doc_to_code,
            gap_id="DOC-CODE-DEFENSIVE-TOTAL",
            layer="core",
            direction="doc_to_code",
            model="absolute",
            severity="medium",
            count=_coerce_int(defensive_probe.get("total")),
            message="Defensive-fallback absolute debt remains non-zero.",
            evidence=[f"baseline_keys={defensive_probe.get('baseline_keys', 0)}"],
        )

    no_monkeypatch_probe = _mapping_optional(probes.get("no_monkeypatch_policy", {}))
    if no_monkeypatch_probe is not None:
        _add_gap(
            doc_to_code,
            gap_id="DOC-CODE-NO-MONKEYPATCH",
            layer="core",
            direction="doc_to_code",
            model="absolute",
            severity="medium",
            count=_coerce_int(no_monkeypatch_probe.get("total")),
            message="No-monkeypatch policy reports unresolved test/runtime mutation seams.",
            evidence=[
                f"{path}={count}"
                for path, count in (no_monkeypatch_probe.get("by_path") or {}).items()
            ],
        )

    order_lifetime_probe = _mapping_optional(probes.get("order_lifetime_policy", {}))
    if order_lifetime_probe is not None:
        _add_gap(
            doc_to_code,
            gap_id="DOC-CODE-ORDER-LIFETIME",
            layer="core",
            direction="doc_to_code",
            model="absolute",
            severity="high",
            count=_coerce_int(order_lifetime_probe.get("total")),
            message="Single-sort lifetime policy reports forbidden active sorting surfaces.",
            evidence=[
                f"{path}={count}"
                for path, count in (order_lifetime_probe.get("by_path") or {}).items()
            ],
        )

    structural_hash_probe = _mapping_optional(probes.get("structural_hash_policy", {}))
    if structural_hash_probe is not None:
        _add_gap(
            doc_to_code,
            gap_id="DOC-CODE-STRUCTURAL-HASH",
            layer="core",
            direction="doc_to_code",
            model="absolute",
            severity="high",
            count=_coerce_int(structural_hash_probe.get("total")),
            message="Structural-hash policy reports digest/text-key identity drift.",
            evidence=[
                f"{path}={count}"
                for path, count in (structural_hash_probe.get("by_path") or {}).items()
            ],
        )

    core_docflow = _mapping_optional(probes.get("docflow_core", {}))
    if core_docflow is not None:
        _add_gap(
            doc_to_code,
            gap_id="DOC-CODE-DOCFLOW-CORE-VIOLATIONS",
            layer="core",
            direction="doc_to_code",
            model="both",
            severity="high",
            count=_coerce_int(core_docflow.get("violation_count")),
            message="Core docflow invariants report violations.",
            evidence=_string_entries(core_docflow.get("violations", [])),
        )
        _add_gap(
            doc_to_code,
            gap_id="DOC-CODE-DOCFLOW-CORE-WARNINGS",
            layer="core",
            direction="doc_to_code",
            model="absolute",
            severity="low",
            count=_coerce_int(core_docflow.get("warning_count")),
            message="Core docflow invariants report warnings.",
            evidence=_string_entries(core_docflow.get("warnings", [])),
        )

    extended_docflow = _mapping_optional(probes.get("docflow_extended_strict", {}))
    if extended_docflow is not None:
        _add_gap(
            doc_to_code,
            gap_id="DOC-CODE-DOCFLOW-EXTENDED-VIOLATIONS",
            layer="extended",
            direction="doc_to_code",
            model="absolute",
            severity="medium",
            count=_coerce_int(extended_docflow.get("extended_violation_count")),
            message="Extended normative docs fail strict docflow invariants.",
            evidence=_string_entries(extended_docflow.get("extended_violations", [])),
        )
        _add_gap(
            doc_to_code,
            gap_id="DOC-CODE-DOCFLOW-EXTENDED-WARNINGS",
            layer="extended",
            direction="doc_to_code",
            model="absolute",
            severity="low",
            count=_coerce_int(extended_docflow.get("extended_warning_count")),
            message="Extended normative docs report strict docflow warnings.",
            evidence=_string_entries(extended_docflow.get("extended_warnings", [])),
        )

    controller_findings = []
    if controller_drift is not None:
        controller_findings = [
            item for item in _mapping_entries(controller_drift.get("findings", []))
        ]
    unanchored = [
        item for item in controller_findings
        if str(item.get("sensor", "")) == "checks_without_normative_anchor"
    ]
    unindexed = [
        item for item in controller_findings
        if str(item.get("sensor", "")) == "unindexed_enforcement_surfaces"
    ]
    _add_gap(
        code_to_doc,
        gap_id="CODE-DOC-UNANCHORED-CHECKS",
        layer="core",
        direction="code_to_doc",
        model="both",
        severity="high",
        count=len(unanchored),
        message="Enforcement checks exist without normative controller anchors.",
        evidence=[str(item.get("detail", "")) for item in unanchored],
    )
    _add_gap(
        code_to_doc,
        gap_id="CODE-DOC-UNINDEXED-SURFACES",
        layer="core",
        direction="code_to_doc",
        model="both",
        severity="high",
        count=len(unindexed),
        message="Policy/enforcement surfaces are not canonically clause-indexed.",
        evidence=[str(item.get("detail", "")) for item in unindexed],
    )

    agent_instruction = _mapping_optional(probes.get("agent_instruction_graph", {}))
    if agent_instruction is not None:
        hidden_toggles = agent_instruction.get("hidden_operational_toggles", [])
        hidden_toggle_entries = _mapping_entries(hidden_toggles)
        _add_gap(
            code_to_doc,
            gap_id="CODE-DOC-HIDDEN-TOGGLES",
            layer="cross",
            direction="code_to_doc",
            model="absolute",
            severity="medium",
            count=len(_list_optional(hidden_toggles) or []),
            message="Operational toggles are present in governance docs but hidden from AGENTS surfaces.",
            evidence=[
                f"{item.get('source')}::{item.get('token')}"
                for item in hidden_toggle_entries
            ],
        )

    _add_gap(
        code_to_doc,
        gap_id="CODE-DOC-OUTSIDE-DEFAULT-STRICT",
        layer="extended",
        direction="code_to_doc",
        model="absolute",
        severity="medium",
        count=len(scope_inventory.outside_default_strict_docs),
        message="Normative docs are outside default strict docflow coverage set.",
        evidence=list(scope_inventory.outside_default_strict_docs),
    )

    ordered_doc_to_code = _ordered_gap_items(
        doc_to_code,
        source="normative_symdiff.synthesize_gaps.doc_to_code",
    )
    ordered_code_to_doc = _ordered_gap_items(
        code_to_doc,
        source="normative_symdiff.synthesize_gaps.code_to_doc",
    )
    return {
        "doc_to_code_gaps": [item.as_dict() for item in ordered_doc_to_code],
        "code_to_doc_gaps": [item.as_dict() for item in ordered_code_to_doc],
    }


def _gap_penalty_points(gaps: Iterable[Mapping[str, object]]) -> int:
    points = 0
    for gap in gaps:
        severity = str(gap.get("severity", "")).strip().lower()
        weight = _SEVERITY_WEIGHT.get(severity, 1)
        count = max(0, _coerce_int(gap.get("count")))
        if count <= 0:
            continue
        magnitude = 1 + int(math.log10(count))
        points += weight * magnitude
    return points


def _distance_band(score: int) -> str:
    if score >= 95:
        return "very_close"
    if score >= 80:
        return "close"
    if score >= 60:
        return "moderate_gap"
    return "far"


def _score_for_layer(
    *,
    model: str,
    layer: str,
    gaps: list[Mapping[str, object]],
) -> dict[str, object]:
    selected: list[Mapping[str, object]] = []
    for gap in gaps:
        gap_model = str(gap.get("model", ""))
        if gap_model not in {model, "both"}:
            continue
        gap_layer = str(gap.get("layer", ""))
        if layer == "overall":
            selected.append(gap)
            continue
        if layer == "core" and gap_layer in {"core", "cross"}:
            selected.append(gap)
            continue
        if layer == "extended" and gap_layer in {"extended", "cross"}:
            selected.append(gap)
            continue
    penalty_points = _gap_penalty_points(selected)
    score = max(0, 100 - penalty_points)
    return {
        "model": model,
        "layer": layer,
        "score": score,
        "distance_band": _distance_band(score),
        "penalty_points": penalty_points,
        "gap_count": len(selected),
        "gaps": [dict(item) for item in selected],
    }


def score_gaps(gap_payload: Mapping[str, object]) -> dict[str, object]:
    all_gaps: list[Mapping[str, object]] = []
    for key in ("doc_to_code_gaps", "code_to_doc_gaps"):
        values = gap_payload.get(key, [])
        all_gaps.extend(_mapping_entries(values))
    models = ("ratchet", "absolute")
    layers = ("core", "extended", "overall")
    matrix: dict[str, dict[str, object]] = {}
    for model in models:
        model_scores: dict[str, object] = {}
        for layer in layers:
            model_scores[layer] = _score_for_layer(
                model=model,
                layer=layer,
                gaps=all_gaps,
            )
        matrix[model] = model_scores
    return matrix


def _render_gap_lines(gaps: list[Mapping[str, object]]) -> list[str]:
    if not gaps:
        return ["- (none)"]
    lines: list[str] = []
    ordered = sort_once(
        gaps,
        source="normative_symdiff.render_gap_lines",
        key=lambda item: (
            -int(_SEVERITY_RANK.get(str(item.get("severity", "")), 0)),
            str(item.get("gap_id", "")),
        ),
    )
    for gap in ordered:
        line = (
            f"- `{gap.get('gap_id')}` [{gap.get('severity')}] "
            f"(layer={gap.get('layer')}, model={gap.get('model')}, count={gap.get('count')}): "
            f"{gap.get('message')}"
        )
        lines.append(line)
    return lines


def render_markdown(report: Mapping[str, object]) -> str:
    summary = _mapping_optional(report.get("summary", {})) or {}
    inventory = _mapping_optional(report.get("inventory", {})) or {}
    clauses = _mapping_optional(report.get("clauses", {})) or {}
    gaps = _mapping_optional(report.get("gaps", {})) or {}
    scoring = _mapping_optional(report.get("scoring", {}))
    inventory_counts = _mapping_optional(inventory.get("counts", {})) or {}
    clause_ids = _list_optional(clauses.get("clause_ids", [])) or []
    partial_clauses = _list_optional(clauses.get("partial_clauses", [])) or []

    doc_to_code_gaps = _list_optional(gaps.get("doc_to_code_gaps", [])) or []
    code_to_doc_gaps = _list_optional(gaps.get("code_to_doc_gaps", [])) or []

    lines = [
        "# Normative Symmetric Diff",
        "",
        f"- generated_at_utc: `{report.get('generated_at_utc', '')}`",
        f"- scope_model: `{report.get('scope_model', '')}`",
        f"- debt_model: `{report.get('debt_model', '')}`",
        f"- code_state: `{report.get('code_state', '')}`",
        "",
        "## Executive Summary",
        "",
        f"- doc_to_code_gap_count: `{summary.get('doc_to_code_gap_count', 0)}`",
        f"- code_to_doc_gap_count: `{summary.get('code_to_doc_gap_count', 0)}`",
        f"- ratchet_overall_score: `{summary.get('ratchet_overall_score', 0)}` (`{summary.get('ratchet_overall_band', '')}`)",
        f"- absolute_overall_score: `{summary.get('absolute_overall_score', 0)}` (`{summary.get('absolute_overall_band', '')}`)",
        "",
        "## Core Layer Matrix",
        "",
        f"- canonical_core_docs: `{inventory_counts.get('core_layer_docs', 0)}`",
        f"- canonical_clause_count: `{len(clause_ids)}`",
        f"- partial_clause_count: `{len(partial_clauses)}`",
        "",
        "## Extended Layer Matrix",
        "",
        f"- extended_normative_docs: `{inventory_counts.get('extended_layer_docs', 0)}`",
        f"- outside_default_strict_docs: `{inventory_counts.get('outside_default_strict_docs', 0)}`",
        "",
        "## Doc to Code Gaps",
        "",
    ]
    lines.extend(_render_gap_lines(doc_to_code_gaps))
    lines.extend(["", "## Code to Doc Gaps", ""])
    lines.extend(_render_gap_lines(code_to_doc_gaps))

    if scoring is not None:
        lines.extend(["", "## How Close/Far", ""])
        for model in ("ratchet", "absolute"):
            model_block = _mapping_optional(scoring.get(model, {}))
            if model_block is None:
                continue
            lines.append(f"### {model.title()} View")
            lines.append("")
            for layer in ("core", "extended", "overall"):
                layer_block = _mapping_optional(model_block.get(layer, {}))
                if layer_block is None:
                    continue
                lines.append(
                    "- "
                    f"{layer}: score={layer_block.get('score', 0)} "
                    f"band={layer_block.get('distance_band', '')} "
                    f"penalty_points={layer_block.get('penalty_points', 0)} "
                    f"gaps={layer_block.get('gap_count', 0)}"
                )
            lines.append("")
    return "\n".join(lines)


def build_report(
    *,
    root: Path,
    scope_model: str,
    debt_model: str,
    code_state: str,
    probe_collector: Callable[[Path, list[str]], Mapping[str, object]] | None = None,
) -> dict[str, object]:
    with governance_audit._audit_deadline_scope():
        inventory = collect_scope_inventory(root)
        clause_index_path = root / "docs" / "normative_clause_index.md"
        enforcement_map_path = root / "docs" / "normative_enforcement_map.yaml"
        clause_ids = _parse_clause_ids(clause_index_path)
        clause_analysis = analyze_clause_enforcement(
            root=root,
            clause_ids=clause_ids,
            enforcement_map_path=enforcement_map_path,
        )
    probe_fn = probe_collector or (
        lambda root_path, extended_docs: _collect_default_probes(
            root=root_path,
            extended_layer_docs=extended_docs,
        )
    )
    probes = dict(probe_fn(root, list(inventory.extended_layer_docs)))
    gap_payload = synthesize_gaps(
        scope_inventory=inventory,
        clause_analysis=clause_analysis,
        probes=probes,
    )
    scoring = score_gaps(gap_payload)
    summary = {
        "doc_to_code_gap_count": len(gap_payload.get("doc_to_code_gaps", [])),
        "code_to_doc_gap_count": len(gap_payload.get("code_to_doc_gaps", [])),
        "ratchet_overall_score": scoring.get("ratchet", {}).get("overall", {}).get("score", 0),
        "ratchet_overall_band": scoring.get("ratchet", {}).get("overall", {}).get("distance_band", ""),
        "absolute_overall_score": scoring.get("absolute", {}).get("overall", {}).get("score", 0),
        "absolute_overall_band": scoring.get("absolute", {}).get("overall", {}).get("distance_band", ""),
    }
    return {
        "schema_version": 1,
        "generated_at_utc": _now_utc(),
        "scope_model": scope_model,
        "debt_model": debt_model,
        "code_state": code_state,
        "summary": summary,
        "inventory": inventory.as_dict(),
        "clauses": clause_analysis,
        "probes": probes,
        "gaps": gap_payload,
        "scoring": scoring,
    }


def run(
    *,
    root: Path,
    json_out: Path,
    md_out: Path,
    scope_model: str = "two-layer",
    debt_model: str = "dual",
    code_state: str = "worktree",
    probe_mode: str = "full",
    probe_collector: Callable[[Path, list[str]], Mapping[str, object]] | None = None,
) -> int:
    resolved_probe_collector = probe_collector
    if resolved_probe_collector is None and probe_mode == "skip":
        resolved_probe_collector = lambda _root, _extended: {}
    report = build_report(
        root=root,
        scope_model=scope_model,
        debt_model=debt_model,
        code_state=code_state,
        probe_collector=resolved_probe_collector,
    )
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    md_out.parent.mkdir(parents=True, exist_ok=True)
    md_out.write_text(render_markdown(report), encoding="utf-8")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compute normative ↔ code/tooling symmetric-difference report."
    )
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--scope-model",
        choices=("two-layer",),
        default="two-layer",
    )
    parser.add_argument(
        "--debt-model",
        choices=("dual",),
        default="dual",
    )
    parser.add_argument(
        "--code-state",
        choices=("worktree",),
        default="worktree",
    )
    parser.add_argument(
        "--probe-mode",
        choices=("full", "skip"),
        default="full",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=Path("artifacts/out/normative_symdiff.json"),
    )
    parser.add_argument(
        "--md-out",
        type=Path,
        default=Path("artifacts/audit_reports/normative_symdiff.md"),
    )
    args = parser.parse_args(argv)
    root = args.root.resolve()
    return run(
        root=root,
        json_out=args.json_out,
        md_out=args.md_out,
        scope_model=str(args.scope_model),
        debt_model=str(args.debt_model),
        code_state=str(args.code_state),
        probe_mode=str(args.probe_mode),
    )


if __name__ == "__main__":
    raise SystemExit(main())

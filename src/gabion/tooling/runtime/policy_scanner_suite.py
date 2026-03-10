from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import itertools
import json
from pathlib import Path
import subprocess
from typing import Any, Iterable, Mapping
from gabion.tooling.policy_rules import (
    aspf_normalization_idempotence_rule,
    boundary_core_contract_rule,
    branchless_rule,
    defensive_fallback_rule,
    fiber_filter_processor_contract_rule,
    fiber_loop_structure_contract_rule,
    fiber_normalization_contract_rule,
    fiber_return_shape_contract_rule,
    fiber_scalar_sentinel_contract_rule,
    fiber_type_dispatch_contract_rule,
    no_anonymous_tuple_rule,
    no_legacy_monolith_import_rule,
    no_mutable_dict_rule,
    no_monkeypatch_rule,
    no_scalar_conversion_boundary_rule,
    orchestrator_primitive_barrel_rule,
    runtime_narrowing_boundary_rule,
    test_sleep_hygiene_rule,
    test_subprocess_hygiene_rule,
    typing_surface_rule,
)
from gabion.tooling.policy_rules.fiber_diagnostics import (
    to_payload_bounds as _fiber_bounds_payload,
    to_payload_counterfactual as _fiber_counterfactual_payload,
    to_payload_trace as _fiber_trace_payload,
)
from gabion.tooling.runtime import policy_result_schema
from gabion.tooling.runtime.policy_scan_batch import build_policy_scan_batch

_POLICY_ARTIFACT = Path("artifacts/out/policy_suite_results.json")
_FORMAT_VERSION = 1
_BRANCHLESS_BASELINE = Path("baselines/branchless_policy_baseline.json")
_DEFENSIVE_BASELINE = Path("baselines/defensive_fallback_policy_baseline.json")
_TYPING_SURFACE_BASELINE = Path("baselines/typing_surface_policy_baseline.json")
_TYPING_SURFACE_WAIVERS = Path("baselines/typing_surface_policy_waivers.json")
_RUNTIME_NARROWING_BOUNDARY_BASELINE = Path("baselines/runtime_narrowing_boundary_policy_baseline.json")
_RUNTIME_NARROWING_BOUNDARY_WAIVERS = Path("baselines/runtime_narrowing_boundary_policy_waivers.json")
_ASPF_NORMALIZATION_IDEMPOTENCE_BASELINE = Path("baselines/aspf_normalization_idempotence_policy_baseline.json")
_TEST_SUBPROCESS_HYGIENE_ALLOWLIST = Path("docs/policy/test_subprocess_hygiene_allowlist.txt")
_TEST_SLEEP_HYGIENE_ALLOWLIST = Path("docs/policy/test_sleep_hygiene_allowlist.txt")

_POLICY_RULE_IDS = (
    "no_monkeypatch",
    "branchless",
    "defensive_fallback",
    "fiber_loop_structure_contract",
    "fiber_filter_processor_contract",
    "fiber_return_shape_contract",
    "fiber_scalar_sentinel_contract",
    "fiber_type_dispatch_contract",
    "no_anonymous_tuple",
    "no_mutable_dict",
    "no_scalar_conversion_boundary",
    "no_legacy_monolith_import",
    "orchestrator_primitive_barrel",
    "typing_surface",
    "runtime_narrowing_boundary",
    "aspf_normalization_idempotence",
    "boundary_core_contract",
    "fiber_normalization_contract",
    "test_subprocess_hygiene",
    "test_sleep_hygiene",
)
_BOUNDARY_MARKER = "gabion:boundary_normalization_module"


def _normalize_policy_results(raw: object) -> dict[str, dict[str, Any]]:
    match raw:
        case dict() as payload:
            return _normalized_external_policy_results(payload)
        case _:
            return {}


def _normalized_external_policy_results(
    payload: Mapping[str, object],
) -> dict[str, dict[str, Any]]:
    candidates = (
        _policy_result_candidate(("policy_check", payload.get("policy_check"))),
        _policy_result_candidate(("structural_hash", payload.get("structural_hash"))),
        _policy_result_candidate(
            ("deprecated_nonerasability", payload.get("deprecated_nonerasability"))
        ),
    )
    return _policy_result_candidates_to_mapping(candidates)


def _changed_paths_from_git(
    *,
    root: Path,
    base_sha: str | None,
    head_sha: str | None,
) -> set[str] | None:
    if base_sha and head_sha:
        command = ["git", "diff", "--name-only", f"{base_sha}..{head_sha}"]
    else:
        command = ["git", "diff", "--name-only", "HEAD"]
    try:
        completed = subprocess.run(
            command,
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (subprocess.CalledProcessError, OSError):
        return None
    changed = set(_iter_nonempty_stripped_lines(completed.stdout))
    if base_sha and head_sha:
        return changed

    # Include untracked files for local touched+new checks.
    try:
        untracked = subprocess.run(
            ["git", "ls-files", "--others", "--exclude-standard"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (subprocess.CalledProcessError, OSError):
        return changed
    changed.update(_iter_nonempty_stripped_lines(untracked.stdout))
    return changed


def _iter_nonempty_stripped_lines(payload: str) -> Iterable[str]:
    for line in payload.splitlines():
        yield from _strip_default_empty(line)


def _strip_default_empty(line: str) -> tuple[str, ...]:
    stripped = line.strip()
    if stripped:
        return (stripped,)
    return ()


def _boundary_scoped_files(
    *,
    root: Path,
    inventory: tuple[Path, ...],
    changed_paths: set[str] | None,
) -> tuple[Path, ...]:
    scoped = tuple(
        _iter_boundary_scoped_candidates(
            root=root,
            inventory=inventory,
            changed_paths=changed_paths,
        )
    )
    return tuple(sorted(set(scoped), key=lambda item: str(item)))


def _iter_boundary_scoped_candidates(
    *,
    root: Path,
    inventory: tuple[Path, ...],
    changed_paths: set[str] | None,
) -> Iterable[Path]:
    for path in inventory:
        yield from _boundary_scoped_candidate(
            path=path,
            root=root,
            changed_paths=changed_paths,
        )


def _boundary_scoped_candidate(
    *,
    path: Path,
    root: Path,
    changed_paths: set[str] | None,
) -> tuple[Path, ...]:
    rel = path.relative_to(root).as_posix()
    path_in_scope = changed_paths is None or rel in changed_paths
    path_under_src = rel.startswith("src/gabion/")
    if not (path_in_scope and path_under_src):
        return ()
    try:
        source = path.read_text(encoding="utf-8")
    except OSError:
        return ()
    if _BOUNDARY_MARKER in source:
        return (path,)
    return ()


@dataclass(frozen=True)
class PolicySuiteResult:
    root: Path
    inventory_hash: str
    rule_set_hash: str
    violations_by_rule: dict[str, list[dict[str, Any]]]
    policy_results: dict[str, dict[str, Any]]
    cached: bool

    def total_violations(self) -> int:
        return sum(_iter_rule_violation_counts(self.violations_by_rule.values()))

    def to_payload(self) -> dict[str, object]:
        counts = dict(map(_rule_count_pair, self.violations_by_rule.items()))
        return {
            "format_version": _FORMAT_VERSION,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "root": str(self.root),
            "inventory_hash": self.inventory_hash,
            "rule_set_hash": self.rule_set_hash,
            "cached": self.cached,
            "counts": counts,
            "violations": self.violations_by_rule,
            "policy_results": self.policy_results,
        }


def _iter_rule_violation_counts(values: Iterable[list[dict[str, Any]]]) -> Iterable[int]:
    for items in values:
        yield len(items)


def _rule_count_pair(item: tuple[str, list[dict[str, Any]]]) -> tuple[str, int]:
    rule, items = item
    return (rule, len(items))


# gabion:decision_protocol
def load_or_scan_policy_suite(
    *,
    root: Path,
    artifact_path: Path = _POLICY_ARTIFACT,
    policy_results: Mapping[str, Mapping[str, Any]] | None = None,
    base_sha: str | None = None,
    head_sha: str | None = None,
) -> PolicySuiteResult:
    resolved_root = root.resolve()
    files = _inventory_files(resolved_root)
    inventory_hash = _inventory_hash(files, resolved_root)
    changed_paths = _changed_paths_from_git(
        root=resolved_root,
        base_sha=base_sha,
        head_sha=head_sha,
    )
    changed_scope_hash = hashlib.sha256(
        json.dumps(sorted(changed_paths) if changed_paths is not None else ["<all>"]).encode(
            "utf-8"
        )
    ).hexdigest()
    rule_set_hash = _rule_set_hash()
    normalized_policy_results = _normalized_policy_result_mapping(
        policy_results or {},
    )
    policy_results_hash = hashlib.sha256(json.dumps(normalized_policy_results, sort_keys=True).encode("utf-8")).hexdigest()
    cached_payload = _load_cached_payload(artifact_path)
    if cached_payload is not None:
        if (
            str(cached_payload.get("inventory_hash", "")) == inventory_hash
            and str(cached_payload.get("rule_set_hash", "")) == rule_set_hash
            and str(cached_payload.get("policy_results_hash", "")) == policy_results_hash
            and str(cached_payload.get("changed_scope_hash", "")) == changed_scope_hash
        ):
            violations = _violations_from_payload(cached_payload)
            return PolicySuiteResult(
                root=resolved_root,
                inventory_hash=inventory_hash,
                rule_set_hash=rule_set_hash,
                violations_by_rule=violations,
                policy_results=_normalize_policy_results(cached_payload.get("policy_results")),
                cached=True,
            )

    result = scan_policy_suite(
        root=resolved_root,
        files=files,
        policy_results=normalized_policy_results,
        base_sha=base_sha,
        head_sha=head_sha,
        changed_paths=changed_paths,
    )
    payload = result.to_payload()
    payload["policy_results_hash"] = policy_results_hash
    payload["changed_scope_hash"] = changed_scope_hash
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    return result


# gabion:decision_protocol
def scan_policy_suite(
    *,
    root: Path,
    files: tuple[Path, ...] | None = None,
    policy_results: Mapping[str, Mapping[str, Any]] | None = None,
    base_sha: str | None = None,
    head_sha: str | None = None,
    changed_paths: set[str] | None = None,
) -> PolicySuiteResult:
    resolved_root = root.resolve()
    inventory = files if files is not None else _inventory_files(resolved_root)
    resolved_changed_paths = (
        changed_paths
        if changed_paths is not None
        else _changed_paths_from_git(
            root=resolved_root,
            base_sha=base_sha,
            head_sha=head_sha,
        )
    )
    boundary_scope_files = _boundary_scoped_files(
        root=resolved_root,
        inventory=inventory,
        changed_paths=resolved_changed_paths,
    )
    src_inventory = tuple(
        path for path in inventory if path.relative_to(resolved_root).as_posix().startswith("src/gabion/")
    )
    test_inventory = tuple(
        path for path in inventory if path.relative_to(resolved_root).as_posix().startswith("tests/")
    )
    inventory_batch = build_policy_scan_batch(
        root=resolved_root,
        target_globs=(),
        files=inventory,
    )
    src_batch = build_policy_scan_batch(
        root=resolved_root,
        target_globs=(),
        files=src_inventory,
    )
    test_batch = build_policy_scan_batch(
        root=resolved_root,
        target_globs=(),
        files=test_inventory,
    )
    boundary_batch = build_policy_scan_batch(
        root=resolved_root,
        target_globs=(),
        files=boundary_scope_files,
    )
    inventory_hash = _inventory_hash(inventory, resolved_root)
    rule_set_hash = _rule_set_hash()
    branchless_allowed = _load_rule_baseline_keys(
        module=branchless_rule,
        baseline_path=resolved_root / _BRANCHLESS_BASELINE,
    )
    defensive_allowed = _load_rule_baseline_keys(
        module=defensive_fallback_rule,
        baseline_path=resolved_root / _DEFENSIVE_BASELINE,
    )
    typing_surface_allowed = _load_rule_baseline_keys(
        module=typing_surface_rule,
        baseline_path=resolved_root / _TYPING_SURFACE_BASELINE,
    )
    runtime_narrowing_boundary_allowed = _load_rule_baseline_keys(
        module=runtime_narrowing_boundary_rule,
        baseline_path=resolved_root / _RUNTIME_NARROWING_BOUNDARY_BASELINE,
    )
    aspf_normalization_idempotence_allowed = _load_rule_baseline_keys(
        module=aspf_normalization_idempotence_rule,
        baseline_path=resolved_root / _ASPF_NORMALIZATION_IDEMPOTENCE_BASELINE,
    )
    typing_surface_waiver_result = typing_surface_rule.load_waivers(
        resolved_root / _TYPING_SURFACE_WAIVERS,
    )
    runtime_narrowing_boundary_waiver_result = runtime_narrowing_boundary_rule.load_waivers(
        resolved_root / _RUNTIME_NARROWING_BOUNDARY_WAIVERS,
    )

    violations_by_rule: dict[str, list[dict[str, Any]]] = {
        "no_monkeypatch": [],
        "branchless": [],
        "defensive_fallback": [],
        "fiber_loop_structure_contract": [],
        "fiber_filter_processor_contract": [],
        "fiber_return_shape_contract": [],
        "fiber_scalar_sentinel_contract": [],
        "fiber_type_dispatch_contract": [],
        "no_anonymous_tuple": [],
        "no_mutable_dict": [],
        "no_scalar_conversion_boundary": [],
        "no_legacy_monolith_import": [],
        "orchestrator_primitive_barrel": [],
        "typing_surface": [],
        "runtime_narrowing_boundary": [],
        "aspf_normalization_idempotence": [],
        "boundary_core_contract": [],
        "fiber_normalization_contract": [],
        "test_subprocess_hygiene": [],
        "test_sleep_hygiene": [],
    }

    for invalid in typing_surface_waiver_result.invalid_waivers:
        violations_by_rule["typing_surface"].append(
            {
                "path": _TYPING_SURFACE_WAIVERS.as_posix(),
                "line": int(invalid.index),
                "column": 1,
                "qualname": "<waiver>",
                "kind": "invalid_waiver",
                "scope": "waiver",
                "annotation": "<none>",
                "message": f"invalid typing-surface waiver metadata: {invalid.reason}",
                "key": f"{_TYPING_SURFACE_WAIVERS.as_posix()}:<waiver>:{int(invalid.index)}:invalid_waiver",
                "render": f"{_TYPING_SURFACE_WAIVERS.as_posix()}:{int(invalid.index)}:1: invalid_waiver: {invalid.reason}",
            }
        )

    for invalid in runtime_narrowing_boundary_waiver_result.invalid_waivers:
        violations_by_rule["runtime_narrowing_boundary"].append(
            {
                "path": _RUNTIME_NARROWING_BOUNDARY_WAIVERS.as_posix(),
                "line": int(invalid.index),
                "column": 1,
                "qualname": "<waiver>",
                "kind": "invalid_waiver",
                "call": "<none>",
                "message": f"invalid runtime-narrowing-boundary waiver metadata: {invalid.reason}",
                "key": f"{_RUNTIME_NARROWING_BOUNDARY_WAIVERS.as_posix()}:<waiver>:{int(invalid.index)}:invalid_waiver",
                "render": f"{_RUNTIME_NARROWING_BOUNDARY_WAIVERS.as_posix()}:{int(invalid.index)}:1: invalid_waiver: {invalid.reason}",
            }
        )

    legacy_monolith_violations = no_legacy_monolith_import_rule.collect_violations(
        batch=inventory_batch,
    )
    violations_by_rule["no_legacy_monolith_import"].extend(
        _serialize_legacy_monolith(item) for item in legacy_monolith_violations
    )
    orchestrator_barrel_violations = orchestrator_primitive_barrel_rule.collect_violations(
        batch=inventory_batch,
    )
    violations_by_rule["orchestrator_primitive_barrel"].extend(
        _serialize_orchestrator_primitive_barrel(item)
        for item in orchestrator_barrel_violations
    )
    no_mp_violations = no_monkeypatch_rule.collect_violations(batch=inventory_batch)
    violations_by_rule["no_monkeypatch"].extend(
        _serialize_no_monkeypatch(item) for item in no_mp_violations
    )
    branchless_violations = _filter_baseline_violations(
        branchless_rule.collect_violations(batch=src_batch),
        allowed_keys=branchless_allowed,
    )
    violations_by_rule["branchless"].extend(
        _serialize_branchless(item) for item in branchless_violations
    )
    defensive_violations = _filter_baseline_violations(
        defensive_fallback_rule.collect_violations(batch=src_batch),
        allowed_keys=defensive_allowed,
    )
    violations_by_rule["defensive_fallback"].extend(
        _serialize_defensive(item) for item in defensive_violations
    )
    loop_structure_violations = fiber_loop_structure_contract_rule.collect_violations(
        batch=src_batch,
    )
    violations_by_rule["fiber_loop_structure_contract"].extend(
        _serialize_fiber_loop_structure_contract(item) for item in loop_structure_violations
    )
    filter_processor_violations = fiber_filter_processor_contract_rule.collect_violations(
        batch=src_batch,
    )
    violations_by_rule["fiber_filter_processor_contract"].extend(
        _serialize_fiber_filter_processor_contract(item) for item in filter_processor_violations
    )
    return_shape_violations = fiber_return_shape_contract_rule.collect_violations(
        batch=src_batch,
    )
    violations_by_rule["fiber_return_shape_contract"].extend(
        _serialize_fiber_return_shape_contract(item) for item in return_shape_violations
    )
    scalar_sentinel_violations = fiber_scalar_sentinel_contract_rule.collect_violations(
        batch=src_batch,
    )
    violations_by_rule["fiber_scalar_sentinel_contract"].extend(
        _serialize_fiber_scalar_sentinel_contract(item) for item in scalar_sentinel_violations
    )
    type_dispatch_violations = fiber_type_dispatch_contract_rule.collect_violations(
        batch=src_batch,
    )
    violations_by_rule["fiber_type_dispatch_contract"].extend(
        _serialize_fiber_type_dispatch_contract(item) for item in type_dispatch_violations
    )
    no_anonymous_tuple_violations = no_anonymous_tuple_rule.collect_violations(
        batch=src_batch,
    )
    violations_by_rule["no_anonymous_tuple"].extend(
        _serialize_no_anonymous_tuple(item) for item in no_anonymous_tuple_violations
    )
    no_mutable_dict_violations = no_mutable_dict_rule.collect_violations(
        batch=src_batch,
    )
    violations_by_rule["no_mutable_dict"].extend(
        _serialize_no_mutable_dict(item) for item in no_mutable_dict_violations
    )
    no_scalar_conversion_boundary_violations = (
        no_scalar_conversion_boundary_rule.collect_violations(
            batch=src_batch,
        )
    )
    violations_by_rule["no_scalar_conversion_boundary"].extend(
        _serialize_no_scalar_conversion_boundary(item)
        for item in no_scalar_conversion_boundary_violations
    )
    typing_surface_violations = _filter_baseline_violations(
        typing_surface_rule.collect_violations(batch=src_batch),
        allowed_keys=(typing_surface_allowed | typing_surface_waiver_result.allowed_keys),
    )
    violations_by_rule["typing_surface"].extend(
        _serialize_typing_surface(item) for item in typing_surface_violations
    )
    runtime_narrowing_boundary_violations = _filter_baseline_violations(
        runtime_narrowing_boundary_rule.collect_violations(batch=src_batch),
        allowed_keys=(
            runtime_narrowing_boundary_allowed
            | runtime_narrowing_boundary_waiver_result.allowed_keys
        ),
    )
    violations_by_rule["runtime_narrowing_boundary"].extend(
        _serialize_runtime_narrowing_boundary(item)
        for item in runtime_narrowing_boundary_violations
    )
    aspf_normalization_idempotence_violations = _filter_baseline_violations(
        aspf_normalization_idempotence_rule.collect_violations(batch=src_batch),
        allowed_keys=aspf_normalization_idempotence_allowed,
    )
    aspf_ingress_violations = aspf_normalization_idempotence_rule.collect_ingress_violations(
        root=resolved_root,
        baseline_path=(resolved_root / _ASPF_NORMALIZATION_IDEMPOTENCE_BASELINE),
    )
    violations_by_rule["aspf_normalization_idempotence"].extend(
        _serialize_aspf_normalization_idempotence(item)
        for item in aspf_ingress_violations
    )
    violations_by_rule["aspf_normalization_idempotence"].extend(
        _serialize_aspf_normalization_idempotence(item)
        for item in aspf_normalization_idempotence_violations
    )
    boundary_core_contract_violations = boundary_core_contract_rule.collect_violations(
        batch=boundary_batch,
    )
    violations_by_rule["boundary_core_contract"].extend(
        _serialize_boundary_core_contract(item)
        for item in boundary_core_contract_violations
    )
    fiber_contract_violations = fiber_normalization_contract_rule.collect_violations(
        batch=boundary_batch,
    )
    violations_by_rule["fiber_normalization_contract"].extend(
        _serialize_fiber_normalization_contract(item)
        for item in fiber_contract_violations
    )
    test_subprocess_hygiene_violations = test_subprocess_hygiene_rule.collect_violations(
        batch=test_batch,
        allowlist_path=resolved_root / _TEST_SUBPROCESS_HYGIENE_ALLOWLIST,
    )
    violations_by_rule["test_subprocess_hygiene"].extend(
        _serialize_test_subprocess_hygiene(item)
        for item in test_subprocess_hygiene_violations
    )
    test_sleep_hygiene_violations = test_sleep_hygiene_rule.collect_violations(
        batch=test_batch,
        allowlist_path=resolved_root / _TEST_SLEEP_HYGIENE_ALLOWLIST,
    )
    violations_by_rule["test_sleep_hygiene"].extend(
        _serialize_test_sleep_hygiene(item)
        for item in test_sleep_hygiene_violations
    )

    _drain(_iter_sort_violations_by_rule(violations_by_rule))
    return PolicySuiteResult(
        root=resolved_root,
        inventory_hash=inventory_hash,
        rule_set_hash=rule_set_hash,
        violations_by_rule=violations_by_rule,
        policy_results=_normalized_policy_result_mapping(policy_results or {}),
        cached=False,
    )


def _drain(items: Iterable[object]) -> None:
    deque(items, maxlen=0)

def _iter_sort_violations_by_rule(
    violations_by_rule: dict[str, list[dict[str, Any]]],
) -> Iterable[None]:
    for rule, items in list(violations_by_rule.items()):
        violations_by_rule[rule] = sorted(items, key=_violation_sort_key)
        yield None


def _violation_sort_key(item: dict[str, Any]) -> tuple[str, str, int, str]:
    return (
        str(item.get("path", "")),
        str(item.get("qualname", "")),
        int(item.get("line", 0) or 0),
        str(item.get("kind", "")),
    )


def violations_for_rule(result: PolicySuiteResult, *, rule: str) -> list[dict[str, Any]]:
    return list(result.violations_by_rule.get(rule, []))


def _load_cached_payload(path: Path) -> dict[str, object] | None:
    payload: dict[str, object] | None = None
    if path.exists():
        try:
            raw_payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            raw_payload = None
        match raw_payload:
            case dict() as payload_dict:
                format_version = int(payload_dict.get("format_version", 0) or 0)
                if format_version == _FORMAT_VERSION:
                    payload = payload_dict
            case _:
                pass
    return payload


def _violations_from_payload(payload: Mapping[str, Any]) -> dict[str, list[dict[str, Any]]]:
    violations_raw = payload.get("violations")
    match violations_raw:
        case dict() as raw_mapping:
            pairs = map(
                lambda rule: (
                    rule,
                    _normalized_violation_items(raw_mapping.get(rule)),
                ),
                _POLICY_RULE_IDS,
            )
            return dict(pairs)
        case _:
            return _empty_violations_payload()


def _empty_violations_payload() -> dict[str, list[dict[str, Any]]]:
    return {
        "no_monkeypatch": [],
        "branchless": [],
        "defensive_fallback": [],
        "fiber_loop_structure_contract": [],
        "fiber_filter_processor_contract": [],
        "fiber_return_shape_contract": [],
        "fiber_scalar_sentinel_contract": [],
        "fiber_type_dispatch_contract": [],
        "no_anonymous_tuple": [],
        "no_mutable_dict": [],
        "no_scalar_conversion_boundary": [],
        "no_legacy_monolith_import": [],
        "orchestrator_primitive_barrel": [],
        "typing_surface": [],
        "runtime_narrowing_boundary": [],
        "aspf_normalization_idempotence": [],
        "boundary_core_contract": [],
        "fiber_normalization_contract": [],
        "test_subprocess_hygiene": [],
        "test_sleep_hygiene": [],
    }


def _normalized_violation_items(raw_items: object) -> list[dict[str, Any]]:
    match raw_items:
        case list() as items:
            return list(_iter_normalized_violation_items(items))
        case _:
            return []


def _iter_normalized_violation_items(items: list[object]) -> Iterable[dict[str, Any]]:
    for item in items:
        yield from _normalized_violation_item(item)


def _normalized_violation_item(item: object) -> tuple[dict[str, Any], ...]:
    match item:
        case dict() as mapping:
            return (dict(mapping),)
        case _:
            return ()


def _inventory_files(root: Path) -> tuple[Path, ...]:
    patterns = ("src/gabion/**/*.py", "tests/**/*.py")
    per_pattern_paths = map(lambda pattern: sorted(root.glob(pattern)), patterns)
    flattened_paths = itertools.chain.from_iterable(per_pattern_paths)
    files = tuple(map(Path.resolve, filter(_is_inventory_file, flattened_paths)))
    deduped = sorted(set(files), key=lambda item: str(item))
    return tuple(deduped)


def _is_inventory_file(path: Path) -> bool:
    if not path.is_file():
        return False
    return "__pycache__" not in path.parts


def _inventory_hash(files: Iterable[Path], root: Path) -> str:
    digest = hashlib.sha256()
    digest.update(b"".join(map(lambda path: _inventory_hash_chunk(path, root), files)))
    return digest.hexdigest()


def _inventory_hash_chunk(path: Path, root: Path) -> bytes:
    stat = path.stat()
    rel = path.relative_to(root).as_posix()
    return (
        rel.encode("utf-8")
        + str(int(stat.st_mtime_ns)).encode("utf-8")
        + str(int(stat.st_size)).encode("utf-8")
    )


def _rule_set_hash() -> str:
    material = "|".join(
        [
            "no_monkeypatch:v3",
            "branchless:v4",
            "defensive_fallback:v4",
            "fiber_loop_structure_contract:v2",
            "fiber_filter_processor_contract:v2",
            "fiber_return_shape_contract:v1",
            "fiber_scalar_sentinel_contract:v2",
            "fiber_type_dispatch_contract:v2",
            "no_anonymous_tuple:v1",
            "no_mutable_dict:v1",
            "no_scalar_conversion_boundary:v1",
            "no_legacy_monolith_import:v2",
            "orchestrator_primitive_barrel:v2",
            "typing_surface:v4",
            "runtime_narrowing_boundary:v4",
            "aspf_normalization_idempotence:v4",
            "boundary_core_contract:v3",
            "fiber_normalization_contract:v3",
            "test_subprocess_hygiene:v4",
            "test_sleep_hygiene:v3",
        ]
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def _load_rule_baseline_keys(*, module: object, baseline_path: Path) -> set[str]:
    loader = getattr(module, "_load_baseline", None)
    if loader is None:
        return set()
    try:
        loaded = loader(baseline_path)
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return set()
    match loaded:
        case set() as keys:
            maybe_keys = map(_str_optional, keys)
            return set(filter(_is_not_none, maybe_keys))
        case _:
            return set()


def _str_optional(item: object) -> str | None:
    match item:
        case str() as text:
            return text
        case _:
            return None


def _is_not_none(value: object) -> bool:
    return value is not None

def _normalized_policy_result_mapping(
    payload: Mapping[str, Mapping[str, Any]] | object,
) -> dict[str, dict[str, Any]]:
    items_outcome = _mapping_items_outcome(payload)
    if not items_outcome.available:
        return {}
    candidates = map(_policy_result_candidate, items_outcome.items)
    return _policy_result_candidates_to_mapping(candidates)


@dataclass(frozen=True)
class _MappingItemsOutcome:
    available: bool
    items: tuple[tuple[object, object], ...]


def _mapping_items_outcome(
    payload: Mapping[str, Mapping[str, Any]] | object,
) -> _MappingItemsOutcome:
    try:
        items = payload.items()  # type: ignore[attr-defined]
    except AttributeError:
        return _MappingItemsOutcome(available=False, items=())
    return _MappingItemsOutcome(available=True, items=tuple(items))


@dataclass(frozen=True)
class _PolicyResultCandidate:
    key: str
    mapping: dict[str, Any]
    include: bool


def _policy_result_candidate(item: tuple[object, object]) -> _PolicyResultCandidate:
    key, value = item
    outcome = _mapping_copy_outcome(value)
    return _PolicyResultCandidate(
        key=str(key),
        mapping=outcome.mapping,
        include=outcome.accepted,
    )


@dataclass(frozen=True)
class _MappingCopyOutcome:
    accepted: bool
    mapping: dict[str, Any]


def _mapping_copy_outcome(value: object) -> _MappingCopyOutcome:
    try:
        mapping = dict(value.items())  # type: ignore[attr-defined]
    except (AttributeError, TypeError, ValueError):
        return _MappingCopyOutcome(accepted=False, mapping={})
    return _MappingCopyOutcome(accepted=True, mapping=mapping)


def _policy_result_candidates_to_mapping(
    candidates: Iterable[_PolicyResultCandidate],
) -> dict[str, dict[str, Any]]:
    included_candidates = filter(_policy_result_candidate_included, candidates)
    pairs = map(_policy_result_candidate_pair, included_candidates)
    return dict(pairs)


def _policy_result_candidate_included(candidate: _PolicyResultCandidate) -> bool:
    return candidate.include


def _policy_result_candidate_pair(
    candidate: _PolicyResultCandidate,
) -> tuple[str, dict[str, Any]]:
    return (candidate.key, candidate.mapping)


def _filter_baseline_violations(
    violations: Iterable[object],
    *,
    allowed_keys: set[str],
) -> list[object]:
    if not allowed_keys:
        return list(violations)
    return list(
        filter(
            lambda violation: not _violation_matches_baseline(
                violation,
                allowed_keys=allowed_keys,
            ),
            violations,
        )
    )


def _violation_matches_baseline(violation: object, *, allowed_keys: set[str]) -> bool:
    key = str(getattr(violation, "key", "") or "")
    legacy_key = str(getattr(violation, "legacy_key", "") or "")
    return (key and key in allowed_keys) or (
        legacy_key and legacy_key in allowed_keys
    )


def _applicability_bounds_payload(violation: object) -> dict[str, object] | None:
    bounds = getattr(violation, "applicability_bounds", None)
    match bounds:
        case None:
            return None
        case _:
            return _fiber_bounds_payload(bounds)


def _recombination_frontier_payload(violation: object) -> dict[str, object]:
    frontier = getattr(violation, "recombination_frontier", None)
    if frontier is None:
        return {}
    return {
        "branch_site_id": getattr(frontier, "branch_site_id", ""),
        "branch_site_identity": getattr(frontier, "branch_site_identity", ""),
        "branch_line": int(getattr(frontier, "branch_line", 0)),
        "branch_column": int(getattr(frontier, "branch_column", 0)),
        "branch_node_kind": getattr(frontier, "branch_node_kind", ""),
        "required_symbols": list(getattr(frontier, "required_symbols", ())),
        "unresolved_symbols": list(getattr(frontier, "unresolved_symbols", ())),
        "anchor_site_id": getattr(frontier, "anchor_site_id", ""),
        "anchor_site_identity": getattr(frontier, "anchor_site_identity", ""),
        "anchor_line": int(getattr(frontier, "anchor_line", 0)),
        "anchor_column": int(getattr(frontier, "anchor_column", 0)),
        "anchor_ordinal": int(getattr(frontier, "anchor_ordinal", 0)),
        "upstream_site_ids": list(getattr(frontier, "upstream_site_ids", ())),
        "upstream_site_identities": list(
            getattr(frontier, "upstream_site_identities", ())
        ),
        "upstream_edge_ids": list(getattr(frontier, "upstream_edge_ids", ())),
        "execution_frontier_site_id": getattr(
            frontier, "execution_frontier_site_id", ""
        ),
        "execution_frontier_site_identity": getattr(
            frontier, "execution_frontier_site_identity", ""
        ),
        "execution_frontier_line": int(
            getattr(frontier, "execution_frontier_line", 0)
        ),
        "execution_frontier_column": int(
            getattr(frontier, "execution_frontier_column", 0)
        ),
        "execution_frontier_ordinal": int(
            getattr(frontier, "execution_frontier_ordinal", 0)
        ),
        "execution_upstream_site_ids": list(
            getattr(frontier, "execution_upstream_site_ids", ())
        ),
        "execution_upstream_site_identities": list(
            getattr(frontier, "execution_upstream_site_identities", ())
        ),
        "execution_upstream_edge_ids": list(
            getattr(frontier, "execution_upstream_edge_ids", ())
        ),
        "bundle_event_count": int(getattr(frontier, "bundle_event_count", 0)),
        "bundle_edge_count": int(getattr(frontier, "bundle_edge_count", 0)),
        "execution_event_count": int(
            getattr(frontier, "execution_event_count", 0)
        ),
        "execution_edge_count": int(getattr(frontier, "execution_edge_count", 0)),
    }


def _serialize_no_monkeypatch(violation: object) -> dict[str, object]:
    return {
        "path": getattr(violation, "path"),
        "line": getattr(violation, "line"),
        "column": getattr(violation, "column"),
        "qualname": getattr(violation, "qualname", "<module>"),
        "kind": getattr(violation, "kind", "violation"),
        "message": getattr(violation, "message"),
        "input_slot": getattr(violation, "input_slot", ""),
        "flow_identity": getattr(violation, "flow_identity", ""),
        "fiber_trace": _fiber_trace_payload(
            getattr(violation, "fiber_trace", ())
        ),
        "applicability_bounds": _applicability_bounds_payload(violation),
        "counterfactual_boundary": _fiber_counterfactual_payload(
            getattr(violation, "counterfactual_boundary", None)
        ),
        "fiber_id": getattr(violation, "fiber_id", ""),
        "taint_interval_id": getattr(violation, "taint_interval_id", ""),
        "condition_overlap_id": getattr(violation, "condition_overlap_id", ""),
        "structured_hash": getattr(violation, "structured_hash", ""),
        "key": getattr(violation, "key", ""),
        "render": getattr(violation, "render")(),
    }


def _serialize_branchless(violation: object) -> dict[str, object]:
    return {
        "path": getattr(violation, "path"),
        "line": getattr(violation, "line"),
        "column": getattr(violation, "column"),
        "qualname": getattr(violation, "qualname"),
        "kind": getattr(violation, "kind"),
        "message": getattr(violation, "message"),
        "input_slot": getattr(violation, "input_slot", ""),
        "flow_identity": getattr(violation, "flow_identity", ""),
        "fiber_trace": _fiber_trace_payload(
            getattr(violation, "fiber_trace", ())
        ),
        "applicability_bounds": _applicability_bounds_payload(violation),
        "counterfactual_boundary": _fiber_counterfactual_payload(
            getattr(violation, "counterfactual_boundary", None)
        ),
        "fiber_id": getattr(violation, "fiber_id", ""),
        "taint_interval_id": getattr(violation, "taint_interval_id", ""),
        "condition_overlap_id": getattr(violation, "condition_overlap_id", ""),
        "recombination_frontier": _recombination_frontier_payload(violation),
        "structured_hash": getattr(violation, "structured_hash", ""),
        "legacy_key": getattr(violation, "legacy_key", ""),
        "key": getattr(violation, "key"),
        "render": getattr(violation, "render")(),
    }


def _serialize_defensive(violation: object) -> dict[str, object]:
    return {
        "path": getattr(violation, "path"),
        "line": getattr(violation, "line"),
        "column": getattr(violation, "column"),
        "qualname": getattr(violation, "qualname"),
        "kind": getattr(violation, "kind"),
        "guard_form": getattr(violation, "guard_form", ""),
        "input_slot": getattr(violation, "input_slot", ""),
        "flow_identity": getattr(violation, "flow_identity", ""),
        "fiber_trace": _fiber_trace_payload(
            getattr(violation, "fiber_trace", ())
        ),
        "applicability_bounds": _applicability_bounds_payload(violation),
        "counterfactual_boundary": _fiber_counterfactual_payload(
            getattr(violation, "counterfactual_boundary", None)
        ),
        "fiber_id": getattr(violation, "fiber_id", ""),
        "taint_interval_id": getattr(violation, "taint_interval_id", ""),
        "condition_overlap_id": getattr(violation, "condition_overlap_id", ""),
        "message": getattr(violation, "message"),
        "structured_hash": getattr(violation, "structured_hash", ""),
        "legacy_key": getattr(violation, "legacy_key", ""),
        "key": getattr(violation, "key"),
        "render": getattr(violation, "render")(),
    }


def _serialize_fiber_scalar_sentinel_contract(violation: object) -> dict[str, object]:
    return {
        "path": getattr(violation, "path"),
        "line": getattr(violation, "line"),
        "column": getattr(violation, "column"),
        "qualname": getattr(violation, "qualname"),
        "kind": getattr(violation, "kind"),
        "message": getattr(violation, "message"),
        "scalar_literal": getattr(violation, "scalar_literal"),
        "comparison_operator": getattr(violation, "comparison_operator"),
        "input_slot": getattr(violation, "input_slot"),
        "flow_identity": getattr(violation, "flow_identity"),
        "fiber_trace": _fiber_trace_payload(
            getattr(violation, "fiber_trace", ())
        ),
        "applicability_bounds": _applicability_bounds_payload(violation),
        "counterfactual_boundary": _fiber_counterfactual_payload(
            getattr(violation, "counterfactual_boundary", None)
        ),
        "structured_hash": getattr(violation, "structured_hash", ""),
        "key": getattr(violation, "key"),
        "render": getattr(violation, "render")(),
    }


def _serialize_fiber_loop_structure_contract(violation: object) -> dict[str, object]:
    return {
        "path": getattr(violation, "path"),
        "line": getattr(violation, "line"),
        "column": getattr(violation, "column"),
        "qualname": getattr(violation, "qualname"),
        "kind": getattr(violation, "kind"),
        "message": getattr(violation, "message"),
        "loop_form": getattr(violation, "loop_form"),
        "input_slot": getattr(violation, "input_slot"),
        "flow_identity": getattr(violation, "flow_identity"),
        "fiber_trace": _fiber_trace_payload(
            getattr(violation, "fiber_trace", ())
        ),
        "applicability_bounds": _applicability_bounds_payload(violation),
        "counterfactual_boundary": _fiber_counterfactual_payload(
            getattr(violation, "counterfactual_boundary", None)
        ),
        "structured_hash": getattr(violation, "structured_hash", ""),
        "key": getattr(violation, "key"),
        "render": getattr(violation, "render")(),
    }


def _serialize_fiber_filter_processor_contract(violation: object) -> dict[str, object]:
    return {
        "path": getattr(violation, "path"),
        "line": getattr(violation, "line"),
        "column": getattr(violation, "column"),
        "qualname": getattr(violation, "qualname"),
        "kind": getattr(violation, "kind"),
        "message": getattr(violation, "message"),
        "branch_form": getattr(violation, "branch_form"),
        "input_slot": getattr(violation, "input_slot"),
        "flow_identity": getattr(violation, "flow_identity"),
        "fiber_trace": _fiber_trace_payload(
            getattr(violation, "fiber_trace", ())
        ),
        "applicability_bounds": _applicability_bounds_payload(violation),
        "counterfactual_boundary": _fiber_counterfactual_payload(
            getattr(violation, "counterfactual_boundary", None)
        ),
        "structured_hash": getattr(violation, "structured_hash", ""),
        "key": getattr(violation, "key"),
        "render": getattr(violation, "render")(),
    }


def _serialize_fiber_return_shape_contract(violation: object) -> dict[str, object]:
    return {
        "path": getattr(violation, "path"),
        "line": getattr(violation, "line"),
        "column": getattr(violation, "column"),
        "qualname": getattr(violation, "qualname"),
        "kind": getattr(violation, "kind"),
        "message": getattr(violation, "message"),
        "return_form": getattr(violation, "return_form"),
        "input_slot": getattr(violation, "input_slot"),
        "flow_identity": getattr(violation, "flow_identity"),
        "fiber_trace": _fiber_trace_payload(
            getattr(violation, "fiber_trace", ())
        ),
        "applicability_bounds": _applicability_bounds_payload(violation),
        "counterfactual_boundary": _fiber_counterfactual_payload(
            getattr(violation, "counterfactual_boundary", None)
        ),
        "structured_hash": getattr(violation, "structured_hash", ""),
        "key": getattr(violation, "key"),
        "render": getattr(violation, "render")(),
    }


def _serialize_fiber_type_dispatch_contract(violation: object) -> dict[str, object]:
    return {
        "path": getattr(violation, "path"),
        "line": getattr(violation, "line"),
        "column": getattr(violation, "column"),
        "qualname": getattr(violation, "qualname"),
        "kind": getattr(violation, "kind"),
        "message": getattr(violation, "message"),
        "guard_form": getattr(violation, "guard_form"),
        "input_slot": getattr(violation, "input_slot"),
        "flow_identity": getattr(violation, "flow_identity"),
        "fiber_trace": _fiber_trace_payload(
            getattr(violation, "fiber_trace", ())
        ),
        "applicability_bounds": _applicability_bounds_payload(violation),
        "counterfactual_boundary": _fiber_counterfactual_payload(
            getattr(violation, "counterfactual_boundary", None)
        ),
        "structured_hash": getattr(violation, "structured_hash", ""),
        "key": getattr(violation, "key"),
        "render": getattr(violation, "render")(),
    }


def _serialize_no_anonymous_tuple(violation: object) -> dict[str, object]:
    return {
        "path": getattr(violation, "path"),
        "line": getattr(violation, "line"),
        "column": getattr(violation, "column"),
        "qualname": getattr(violation, "qualname"),
        "kind": getattr(violation, "kind"),
        "message": getattr(violation, "message"),
        "input_slot": getattr(violation, "input_slot", ""),
        "flow_identity": getattr(violation, "flow_identity", ""),
        "fiber_trace": _fiber_trace_payload(
            getattr(violation, "fiber_trace", ())
        ),
        "applicability_bounds": _applicability_bounds_payload(violation),
        "counterfactual_boundary": _fiber_counterfactual_payload(
            getattr(violation, "counterfactual_boundary", None)
        ),
        "fiber_id": getattr(violation, "fiber_id", ""),
        "taint_interval_id": getattr(violation, "taint_interval_id", ""),
        "condition_overlap_id": getattr(violation, "condition_overlap_id", ""),
        "structured_hash": getattr(violation, "structured_hash", ""),
        "key": getattr(violation, "key", ""),
        "render": getattr(violation, "render")(),
    }


def _serialize_no_mutable_dict(violation: object) -> dict[str, object]:
    return {
        "path": getattr(violation, "path"),
        "line": getattr(violation, "line"),
        "column": getattr(violation, "column"),
        "qualname": getattr(violation, "qualname"),
        "kind": getattr(violation, "kind"),
        "message": getattr(violation, "message"),
        "input_slot": getattr(violation, "input_slot", ""),
        "flow_identity": getattr(violation, "flow_identity", ""),
        "fiber_trace": _fiber_trace_payload(
            getattr(violation, "fiber_trace", ())
        ),
        "applicability_bounds": _applicability_bounds_payload(violation),
        "counterfactual_boundary": _fiber_counterfactual_payload(
            getattr(violation, "counterfactual_boundary", None)
        ),
        "fiber_id": getattr(violation, "fiber_id", ""),
        "taint_interval_id": getattr(violation, "taint_interval_id", ""),
        "condition_overlap_id": getattr(violation, "condition_overlap_id", ""),
        "structured_hash": getattr(violation, "structured_hash", ""),
        "key": getattr(violation, "key", ""),
        "render": getattr(violation, "render")(),
    }


def _serialize_no_scalar_conversion_boundary(violation: object) -> dict[str, object]:
    return {
        "path": getattr(violation, "path"),
        "line": getattr(violation, "line"),
        "column": getattr(violation, "column"),
        "qualname": getattr(violation, "qualname"),
        "kind": getattr(violation, "kind"),
        "conversion": getattr(violation, "conversion", ""),
        "message": getattr(violation, "message"),
        "input_slot": getattr(violation, "input_slot", ""),
        "flow_identity": getattr(violation, "flow_identity", ""),
        "fiber_trace": _fiber_trace_payload(
            getattr(violation, "fiber_trace", ())
        ),
        "applicability_bounds": _applicability_bounds_payload(violation),
        "counterfactual_boundary": _fiber_counterfactual_payload(
            getattr(violation, "counterfactual_boundary", None)
        ),
        "fiber_id": getattr(violation, "fiber_id", ""),
        "taint_interval_id": getattr(violation, "taint_interval_id", ""),
        "condition_overlap_id": getattr(violation, "condition_overlap_id", ""),
        "structured_hash": getattr(violation, "structured_hash", ""),
        "key": getattr(violation, "key", ""),
        "render": getattr(violation, "render")(),
    }


def _serialize_legacy_monolith(violation: object) -> dict[str, object]:
    return {
        "path": getattr(violation, "path"),
        "line": getattr(violation, "line"),
        "column": getattr(violation, "column"),
        "kind": getattr(violation, "kind"),
        "input_slot": getattr(violation, "input_slot", ""),
        "flow_identity": getattr(violation, "flow_identity", ""),
        "fiber_trace": _fiber_trace_payload(
            getattr(violation, "fiber_trace", ())
        ),
        "applicability_bounds": _applicability_bounds_payload(violation),
        "counterfactual_boundary": _fiber_counterfactual_payload(
            getattr(violation, "counterfactual_boundary", None)
        ),
        "fiber_id": getattr(violation, "fiber_id", ""),
        "taint_interval_id": getattr(violation, "taint_interval_id", ""),
        "condition_overlap_id": getattr(violation, "condition_overlap_id", ""),
        "message": getattr(violation, "message"),
        "key": getattr(violation, "key"),
        "structured_hash": getattr(violation, "structured_hash", ""),
        "render": getattr(violation, "render")(),
    }


def _serialize_orchestrator_primitive_barrel(violation: object) -> dict[str, object]:
    return {
        "path": getattr(violation, "path"),
        "line": getattr(violation, "line"),
        "column": getattr(violation, "column"),
        "kind": getattr(violation, "kind"),
        "input_slot": getattr(violation, "input_slot", ""),
        "flow_identity": getattr(violation, "flow_identity", ""),
        "fiber_trace": _fiber_trace_payload(
            getattr(violation, "fiber_trace", ())
        ),
        "applicability_bounds": _applicability_bounds_payload(violation),
        "counterfactual_boundary": _fiber_counterfactual_payload(
            getattr(violation, "counterfactual_boundary", None)
        ),
        "fiber_id": getattr(violation, "fiber_id", ""),
        "taint_interval_id": getattr(violation, "taint_interval_id", ""),
        "condition_overlap_id": getattr(violation, "condition_overlap_id", ""),
        "message": getattr(violation, "message"),
        "key": getattr(violation, "key"),
        "structured_hash": getattr(violation, "structured_hash", ""),
        "render": getattr(violation, "render")(),
    }


def _serialize_typing_surface(violation: object) -> dict[str, object]:
    return {
        "path": getattr(violation, "path"),
        "line": getattr(violation, "line"),
        "column": getattr(violation, "column"),
        "qualname": getattr(violation, "qualname"),
        "kind": getattr(violation, "kind"),
        "scope": getattr(violation, "scope"),
        "annotation": getattr(violation, "annotation"),
        "input_slot": getattr(violation, "input_slot", ""),
        "flow_identity": getattr(violation, "flow_identity", ""),
        "fiber_trace": _fiber_trace_payload(
            getattr(violation, "fiber_trace", ())
        ),
        "applicability_bounds": _applicability_bounds_payload(violation),
        "counterfactual_boundary": _fiber_counterfactual_payload(
            getattr(violation, "counterfactual_boundary", None)
        ),
        "fiber_id": getattr(violation, "fiber_id", ""),
        "taint_interval_id": getattr(violation, "taint_interval_id", ""),
        "condition_overlap_id": getattr(violation, "condition_overlap_id", ""),
        "message": getattr(violation, "message"),
        "structured_hash": getattr(violation, "structured_hash", ""),
        "legacy_key": getattr(violation, "legacy_key", ""),
        "key": getattr(violation, "key"),
        "render": getattr(violation, "render")(),
    }


def _serialize_runtime_narrowing_boundary(violation: object) -> dict[str, object]:
    return {
        "path": getattr(violation, "path"),
        "line": getattr(violation, "line"),
        "column": getattr(violation, "column"),
        "qualname": getattr(violation, "qualname"),
        "kind": getattr(violation, "kind"),
        "call": getattr(violation, "call"),
        "input_slot": getattr(violation, "input_slot", ""),
        "flow_identity": getattr(violation, "flow_identity", ""),
        "fiber_trace": _fiber_trace_payload(
            getattr(violation, "fiber_trace", ())
        ),
        "applicability_bounds": _applicability_bounds_payload(violation),
        "counterfactual_boundary": _fiber_counterfactual_payload(
            getattr(violation, "counterfactual_boundary", None)
        ),
        "fiber_id": getattr(violation, "fiber_id", ""),
        "taint_interval_id": getattr(violation, "taint_interval_id", ""),
        "condition_overlap_id": getattr(violation, "condition_overlap_id", ""),
        "message": getattr(violation, "message"),
        "structured_hash": getattr(violation, "structured_hash", ""),
        "legacy_key": getattr(violation, "legacy_key", ""),
        "key": getattr(violation, "key"),
        "render": getattr(violation, "render")(),
    }


def _serialize_aspf_normalization_idempotence(
    violation: object,
) -> dict[str, object]:
    return {
        "path": getattr(violation, "path"),
        "line": getattr(violation, "line"),
        "column": getattr(violation, "column"),
        "qualname": getattr(violation, "qualname"),
        "kind": getattr(violation, "kind"),
        "normalization_class": getattr(violation, "normalization_class"),
        "flow_identity": getattr(violation, "flow_identity"),
        "event_kind": getattr(violation, "event_kind"),
        "fiber_trace": _fiber_trace_payload(
            getattr(violation, "fiber_trace", ())
        ),
        "applicability_bounds": _applicability_bounds_payload(violation),
        "counterfactual_boundary": _fiber_counterfactual_payload(
            getattr(violation, "counterfactual_boundary", None)
        ),
        "message": getattr(violation, "message"),
        "structured_hash": getattr(violation, "structured_hash", ""),
        "legacy_key": getattr(violation, "legacy_key", ""),
        "key": getattr(violation, "key"),
        "render": getattr(violation, "render")(),
    }


def _serialize_boundary_core_contract(violation: object) -> dict[str, object]:
    return {
        "path": getattr(violation, "path"),
        "line": getattr(violation, "line"),
        "column": getattr(violation, "column"),
        "qualname": getattr(violation, "qualname"),
        "kind": getattr(violation, "kind"),
        "input_slot": getattr(violation, "input_slot", ""),
        "flow_identity": getattr(violation, "flow_identity", ""),
        "fiber_trace": _fiber_trace_payload(
            getattr(violation, "fiber_trace", ())
        ),
        "applicability_bounds": _applicability_bounds_payload(violation),
        "counterfactual_boundary": _fiber_counterfactual_payload(
            getattr(violation, "counterfactual_boundary", None)
        ),
        "fiber_id": getattr(violation, "fiber_id", ""),
        "taint_interval_id": getattr(violation, "taint_interval_id", ""),
        "condition_overlap_id": getattr(violation, "condition_overlap_id", ""),
        "message": getattr(violation, "message"),
        "structured_hash": getattr(violation, "structured_hash", ""),
        "legacy_key": getattr(violation, "legacy_key", ""),
        "key": getattr(violation, "key"),
        "render": getattr(violation, "render")(),
    }


def _serialize_fiber_normalization_contract(violation: object) -> dict[str, object]:
    return {
        "path": getattr(violation, "path"),
        "line": getattr(violation, "line"),
        "column": getattr(violation, "column"),
        "qualname": getattr(violation, "qualname"),
        "kind": getattr(violation, "kind"),
        "message": getattr(violation, "message"),
        "normalization_class": getattr(violation, "normalization_class"),
        "input_slot": getattr(violation, "input_slot"),
        "flow_identity": getattr(violation, "flow_identity"),
        "fiber_trace": _fiber_trace_payload(
            getattr(violation, "fiber_trace", ())
        ),
        "applicability_bounds": _applicability_bounds_payload(violation),
        "counterfactual_boundary": _fiber_counterfactual_payload(
            getattr(violation, "counterfactual_boundary", None)
        ),
        "structured_hash": getattr(violation, "structured_hash", ""),
        "legacy_key": getattr(violation, "legacy_key", ""),
        "key": getattr(violation, "key"),
        "render": getattr(violation, "render")(),
    }


def _serialize_test_subprocess_hygiene(violation: object) -> dict[str, object]:
    return {
        "path": getattr(violation, "path"),
        "line": getattr(violation, "line"),
        "column": getattr(violation, "column"),
        "kind": getattr(violation, "kind"),
        "call": getattr(violation, "call"),
        "input_slot": getattr(violation, "input_slot", ""),
        "flow_identity": getattr(violation, "flow_identity", ""),
        "fiber_trace": _fiber_trace_payload(
            getattr(violation, "fiber_trace", ())
        ),
        "applicability_bounds": _applicability_bounds_payload(violation),
        "counterfactual_boundary": _fiber_counterfactual_payload(
            getattr(violation, "counterfactual_boundary", None)
        ),
        "fiber_id": getattr(violation, "fiber_id", ""),
        "taint_interval_id": getattr(violation, "taint_interval_id", ""),
        "condition_overlap_id": getattr(violation, "condition_overlap_id", ""),
        "message": getattr(violation, "message"),
        "key": getattr(violation, "key"),
        "structured_hash": getattr(violation, "structured_hash", ""),
        "render": getattr(violation, "render")(),
    }


def _serialize_test_sleep_hygiene(violation: object) -> dict[str, object]:
    return {
        "path": getattr(violation, "path"),
        "line": getattr(violation, "line"),
        "column": getattr(violation, "column"),
        "kind": getattr(violation, "kind"),
        "call": getattr(violation, "call"),
        "input_slot": getattr(violation, "input_slot", ""),
        "flow_identity": getattr(violation, "flow_identity", ""),
        "fiber_trace": _fiber_trace_payload(
            getattr(violation, "fiber_trace", ())
        ),
        "applicability_bounds": _applicability_bounds_payload(violation),
        "counterfactual_boundary": _fiber_counterfactual_payload(
            getattr(violation, "counterfactual_boundary", None)
        ),
        "fiber_id": getattr(violation, "fiber_id", ""),
        "taint_interval_id": getattr(violation, "taint_interval_id", ""),
        "condition_overlap_id": getattr(violation, "condition_overlap_id", ""),
        "message": getattr(violation, "message"),
        "key": getattr(violation, "key"),
        "structured_hash": getattr(violation, "structured_hash", ""),
        "render": getattr(violation, "render")(),
    }


__all__ = [
    "PolicySuiteResult",
    "load_or_scan_policy_suite",
    "scan_policy_suite",
    "violations_for_rule",
]

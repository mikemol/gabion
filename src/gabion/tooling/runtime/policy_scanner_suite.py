from __future__ import annotations

from collections import deque
import itertools
import json
from pathlib import Path
from typing import Any, Iterable
from gabion.tooling.policy_substrate.policy_scanner_identity import (
    PolicyScannerIdentitySpace,
    canonical_policy_scanner_site_identity,
    canonical_policy_scanner_structural_identity,
)
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
from gabion.tooling.runtime.policy_scan_batch import build_policy_scan_batch
_BRANCHLESS_BASELINE = Path("baselines/branchless_policy_baseline.json")
_DEFENSIVE_BASELINE = Path("baselines/defensive_fallback_policy_baseline.json")
_TYPING_SURFACE_BASELINE = Path("baselines/typing_surface_policy_baseline.json")
_TYPING_SURFACE_WAIVERS = Path("baselines/typing_surface_policy_waivers.json")
_RUNTIME_NARROWING_BOUNDARY_BASELINE = Path("baselines/runtime_narrowing_boundary_policy_baseline.json")
_RUNTIME_NARROWING_BOUNDARY_WAIVERS = Path("baselines/runtime_narrowing_boundary_policy_waivers.json")
_ASPF_NORMALIZATION_IDEMPOTENCE_BASELINE = Path("baselines/aspf_normalization_idempotence_policy_baseline.json")
_TEST_SUBPROCESS_HYGIENE_ALLOWLIST = Path("docs/policy/test_subprocess_hygiene_allowlist.txt")
_TEST_SLEEP_HYGIENE_ALLOWLIST = Path("docs/policy/test_sleep_hygiene_allowlist.txt")

_BOUNDARY_MARKER = "gabion:boundary_normalization_module"


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


# gabion:decision_protocol
def scan_policy_suite(
    *,
    root: Path,
    changed_paths: set[str] | None = None,
    identities: PolicyScannerIdentitySpace | None = None,
) -> dict[str, list[dict[str, Any]]]:
    resolved_root = root.resolve()
    identity_space = identities if identities is not None else PolicyScannerIdentitySpace()
    inventory = _inventory_files(resolved_root)
    boundary_scope_files = _boundary_scoped_files(
        root=resolved_root,
        inventory=inventory,
        changed_paths=changed_paths,
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
        identities=identity_space,
    )
    src_batch = build_policy_scan_batch(
        root=resolved_root,
        target_globs=(),
        files=src_inventory,
        identities=identity_space,
    )
    test_batch = build_policy_scan_batch(
        root=resolved_root,
        target_globs=(),
        files=test_inventory,
        identities=identity_space,
    )
    boundary_batch = build_policy_scan_batch(
        root=resolved_root,
        target_globs=(),
        files=boundary_scope_files,
        identities=identity_space,
    )
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
            _with_scanner_identity(
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
                },
                rule_id="typing_surface",
                identities=identity_space,
            )
        )

    for invalid in runtime_narrowing_boundary_waiver_result.invalid_waivers:
        violations_by_rule["runtime_narrowing_boundary"].append(
            _with_scanner_identity(
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
                },
                rule_id="runtime_narrowing_boundary",
                identities=identity_space,
            )
        )

    legacy_monolith_violations = no_legacy_monolith_import_rule.collect_violations(
        batch=inventory_batch,
    )
    violations_by_rule["no_legacy_monolith_import"].extend(
        _serialize_rule_violations(
            rule_id="no_legacy_monolith_import",
            violations=legacy_monolith_violations,
            serializer=_serialize_legacy_monolith,
            identities=identity_space,
        )
    )
    orchestrator_barrel_violations = orchestrator_primitive_barrel_rule.collect_violations(
        batch=inventory_batch,
    )
    violations_by_rule["orchestrator_primitive_barrel"].extend(
        _serialize_rule_violations(
            rule_id="orchestrator_primitive_barrel",
            violations=orchestrator_barrel_violations,
            serializer=_serialize_orchestrator_primitive_barrel,
            identities=identity_space,
        )
    )
    no_mp_violations = no_monkeypatch_rule.collect_violations(batch=inventory_batch)
    violations_by_rule["no_monkeypatch"].extend(
        _serialize_rule_violations(
            rule_id="no_monkeypatch",
            violations=no_mp_violations,
            serializer=_serialize_no_monkeypatch,
            identities=identity_space,
        )
    )
    branchless_violations = _filter_baseline_violations(
        branchless_rule.collect_violations(batch=src_batch),
        allowed_keys=branchless_allowed,
    )
    violations_by_rule["branchless"].extend(
        _serialize_rule_violations(
            rule_id="branchless",
            violations=branchless_violations,
            serializer=_serialize_branchless,
            identities=identity_space,
        )
    )
    defensive_violations = _filter_baseline_violations(
        defensive_fallback_rule.collect_violations(batch=src_batch),
        allowed_keys=defensive_allowed,
    )
    violations_by_rule["defensive_fallback"].extend(
        _serialize_rule_violations(
            rule_id="defensive_fallback",
            violations=defensive_violations,
            serializer=_serialize_defensive,
            identities=identity_space,
        )
    )
    loop_structure_violations = fiber_loop_structure_contract_rule.collect_violations(
        batch=src_batch,
    )
    violations_by_rule["fiber_loop_structure_contract"].extend(
        _serialize_rule_violations(
            rule_id="fiber_loop_structure_contract",
            violations=loop_structure_violations,
            serializer=_serialize_fiber_loop_structure_contract,
            identities=identity_space,
        )
    )
    filter_processor_violations = fiber_filter_processor_contract_rule.collect_violations(
        batch=src_batch,
    )
    violations_by_rule["fiber_filter_processor_contract"].extend(
        _serialize_rule_violations(
            rule_id="fiber_filter_processor_contract",
            violations=filter_processor_violations,
            serializer=_serialize_fiber_filter_processor_contract,
            identities=identity_space,
        )
    )
    return_shape_violations = fiber_return_shape_contract_rule.collect_violations(
        batch=src_batch,
    )
    violations_by_rule["fiber_return_shape_contract"].extend(
        _serialize_rule_violations(
            rule_id="fiber_return_shape_contract",
            violations=return_shape_violations,
            serializer=_serialize_fiber_return_shape_contract,
            identities=identity_space,
        )
    )
    scalar_sentinel_violations = fiber_scalar_sentinel_contract_rule.collect_violations(
        batch=src_batch,
    )
    violations_by_rule["fiber_scalar_sentinel_contract"].extend(
        _serialize_rule_violations(
            rule_id="fiber_scalar_sentinel_contract",
            violations=scalar_sentinel_violations,
            serializer=_serialize_fiber_scalar_sentinel_contract,
            identities=identity_space,
        )
    )
    type_dispatch_violations = fiber_type_dispatch_contract_rule.collect_violations(
        batch=src_batch,
    )
    violations_by_rule["fiber_type_dispatch_contract"].extend(
        _serialize_rule_violations(
            rule_id="fiber_type_dispatch_contract",
            violations=type_dispatch_violations,
            serializer=_serialize_fiber_type_dispatch_contract,
            identities=identity_space,
        )
    )
    no_anonymous_tuple_violations = no_anonymous_tuple_rule.collect_violations(
        batch=src_batch,
    )
    violations_by_rule["no_anonymous_tuple"].extend(
        _serialize_rule_violations(
            rule_id="no_anonymous_tuple",
            violations=no_anonymous_tuple_violations,
            serializer=_serialize_no_anonymous_tuple,
            identities=identity_space,
        )
    )
    no_mutable_dict_violations = no_mutable_dict_rule.collect_violations(
        batch=src_batch,
    )
    violations_by_rule["no_mutable_dict"].extend(
        _serialize_rule_violations(
            rule_id="no_mutable_dict",
            violations=no_mutable_dict_violations,
            serializer=_serialize_no_mutable_dict,
            identities=identity_space,
        )
    )
    no_scalar_conversion_boundary_violations = (
        no_scalar_conversion_boundary_rule.collect_violations(
            batch=src_batch,
        )
    )
    violations_by_rule["no_scalar_conversion_boundary"].extend(
        _serialize_rule_violations(
            rule_id="no_scalar_conversion_boundary",
            violations=no_scalar_conversion_boundary_violations,
            serializer=_serialize_no_scalar_conversion_boundary,
            identities=identity_space,
        )
    )
    typing_surface_violations = _filter_baseline_violations(
        typing_surface_rule.collect_violations(batch=src_batch),
        allowed_keys=(typing_surface_allowed | typing_surface_waiver_result.allowed_keys),
    )
    violations_by_rule["typing_surface"].extend(
        _serialize_rule_violations(
            rule_id="typing_surface",
            violations=typing_surface_violations,
            serializer=_serialize_typing_surface,
            identities=identity_space,
        )
    )
    runtime_narrowing_boundary_violations = _filter_baseline_violations(
        runtime_narrowing_boundary_rule.collect_violations(batch=src_batch),
        allowed_keys=(
            runtime_narrowing_boundary_allowed
            | runtime_narrowing_boundary_waiver_result.allowed_keys
        ),
    )
    violations_by_rule["runtime_narrowing_boundary"].extend(
        _serialize_rule_violations(
            rule_id="runtime_narrowing_boundary",
            violations=runtime_narrowing_boundary_violations,
            serializer=_serialize_runtime_narrowing_boundary,
            identities=identity_space,
        )
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
        _serialize_rule_violations(
            rule_id="aspf_normalization_idempotence",
            violations=aspf_ingress_violations,
            serializer=_serialize_aspf_normalization_idempotence,
            identities=identity_space,
        )
    )
    violations_by_rule["aspf_normalization_idempotence"].extend(
        _serialize_rule_violations(
            rule_id="aspf_normalization_idempotence",
            violations=aspf_normalization_idempotence_violations,
            serializer=_serialize_aspf_normalization_idempotence,
            identities=identity_space,
        )
    )
    boundary_core_contract_violations = boundary_core_contract_rule.collect_violations(
        batch=boundary_batch,
    )
    violations_by_rule["boundary_core_contract"].extend(
        _serialize_rule_violations(
            rule_id="boundary_core_contract",
            violations=boundary_core_contract_violations,
            serializer=_serialize_boundary_core_contract,
            identities=identity_space,
        )
    )
    fiber_contract_violations = fiber_normalization_contract_rule.collect_violations(
        batch=boundary_batch,
    )
    violations_by_rule["fiber_normalization_contract"].extend(
        _serialize_rule_violations(
            rule_id="fiber_normalization_contract",
            violations=fiber_contract_violations,
            serializer=_serialize_fiber_normalization_contract,
            identities=identity_space,
        )
    )
    test_subprocess_hygiene_violations = test_subprocess_hygiene_rule.collect_violations(
        batch=test_batch,
        allowlist_path=resolved_root / _TEST_SUBPROCESS_HYGIENE_ALLOWLIST,
    )
    violations_by_rule["test_subprocess_hygiene"].extend(
        _serialize_rule_violations(
            rule_id="test_subprocess_hygiene",
            violations=test_subprocess_hygiene_violations,
            serializer=_serialize_test_subprocess_hygiene,
            identities=identity_space,
        )
    )
    test_sleep_hygiene_violations = test_sleep_hygiene_rule.collect_violations(
        batch=test_batch,
        allowlist_path=resolved_root / _TEST_SLEEP_HYGIENE_ALLOWLIST,
    )
    violations_by_rule["test_sleep_hygiene"].extend(
        _serialize_rule_violations(
            rule_id="test_sleep_hygiene",
            violations=test_sleep_hygiene_violations,
            serializer=_serialize_test_sleep_hygiene,
            identities=identity_space,
        )
    )

    _drain(_iter_sort_violations_by_rule(violations_by_rule))
    return violations_by_rule


def _drain(items: Iterable[object]) -> None:
    deque(items, maxlen=0)


def _serialize_rule_violations(
    *,
    rule_id: str,
    violations: Iterable[object],
    serializer,
    identities: PolicyScannerIdentitySpace,
) -> list[dict[str, object]]:
    return [
        _with_scanner_identity(
            serializer(violation),
            rule_id=rule_id,
            identities=identities,
        )
        for violation in violations
    ]


def _with_scanner_identity(
    payload: dict[str, object],
    *,
    rule_id: str,
    identities: PolicyScannerIdentitySpace,
) -> dict[str, object]:
    item = dict(payload)
    path = _payload_text(item.get("path"))
    qualname = _payload_text(item.get("qualname")) or "<module>"
    line = _payload_int(item.get("line"))
    column = _payload_int(item.get("column"))
    kind = _payload_text(item.get("kind")) or "violation"
    site_identity = _payload_text(item.get("site_identity")) or canonical_policy_scanner_site_identity(
        rel_path=path,
        qualname=qualname,
        line=line,
        column=column,
        scanner_kind="violation",
        surface=rule_id,
    )
    structural_identity = _payload_text(item.get("structural_identity")) or canonical_policy_scanner_structural_identity(
        rel_path=path,
        qualname=qualname,
        structural_path=_scanner_structural_basis(item),
        scanner_kind="violation",
        surface=rule_id,
    )
    identity = identities.item_id(
        scanner_kind="violation",
        rule_id=rule_id,
        rel_path=path,
        qualname=qualname,
        line=line,
        column=column,
        kind=kind,
        site_identity=site_identity,
        structural_identity=structural_identity,
        label=_payload_text(item.get("render")) or f"{rule_id}:{path}:{kind}",
    )
    item["site_identity"] = site_identity
    item["structural_identity"] = structural_identity
    item["identity"] = identity.as_payload()
    return item


def _scanner_structural_basis(item: dict[str, object]) -> str:
    for key in (
        "structured_hash",
        "flow_identity",
        "fiber_id",
        "taint_interval_id",
        "condition_overlap_id",
        "call",
        "annotation",
        "normalization_class",
        "guard_form",
        "branch_form",
        "return_form",
        "scalar_literal",
        "comparison_operator",
        "input_slot",
        "legacy_key",
        "key",
        "message",
    ):
        value = _payload_text(item.get(key))
        if value:
            return value
    kind = _payload_text(item.get("kind")) or "violation"
    line = _payload_int(item.get("line"))
    column = _payload_int(item.get("column"))
    return f"{kind}:{line}:{column}"


def _payload_text(value: object) -> str:
    match value:
        case str() as text:
            return text.strip()
        case _:
            return ""


def _payload_int(value: object) -> int:
    match value:
        case bool():
            return 0
        case int() as integer:
            return integer
        case _:
            return 0

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


def _lattice_witness_payload(violation: object) -> dict[str, object]:
    witness = getattr(violation, "lattice_witness", None)
    if witness is None:
        return {}
    serializer = getattr(witness, "as_payload", None)
    if callable(serializer):
        payload = serializer()
        if isinstance(payload, dict):
            return payload
    return {}


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
        "lattice_witness": _lattice_witness_payload(violation),
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
    "scan_policy_suite",
]

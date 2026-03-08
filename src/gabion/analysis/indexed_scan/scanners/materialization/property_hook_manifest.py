from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Mapping, Sequence, cast

from gabion.analysis.dataflow.engine.dataflow_contracts import InvariantProposition
from gabion.analysis.foundation.json_types import JSONObject, JSONValue


@dataclass(frozen=True)
class PropertyHookManifestDeps:
    check_deadline_fn: Callable[[], None]
    sort_once_fn: Callable[..., list[object]]
    invariant_confidence_fn: Callable[[object], float]
    normalize_invariant_proposition_fn: Callable[..., InvariantProposition]
    invariant_digest_fn: Callable[..., str]


@dataclass(frozen=True)
class PropertyHookCallableIndexDeps:
    check_deadline_fn: Callable[[], None]
    sort_once_fn: Callable[..., list[object]]


def generate_property_hook_manifest(
    invariants: Sequence[InvariantProposition],
    *,
    min_confidence: float = 0.7,
    emit_hypothesis_templates: bool = False,
    deps: PropertyHookManifestDeps,
) -> JSONObject:
    threshold = max(0.0, min(1.0, min_confidence))
    hooks: list[JSONObject] = []
    for proposition in deps.sort_once_fn(
        invariants,
        key=lambda prop: (
            prop.scope or "",
            prop.form,
            prop.terms,
            prop.invariant_id or "",
        ),
        source="gabion.analysis.dataflow_indexed_file_scan.generate_property_hook_manifest.site_1",
    ):
        deps.check_deadline_fn()
        scope = proposition.scope or ""
        if not scope or ":" not in scope:
            continue
        confidence = deps.invariant_confidence_fn(proposition.confidence)
        if confidence < threshold:
            continue
        normalized = deps.normalize_invariant_proposition_fn(
            proposition,
            default_scope=scope,
            default_source=proposition.source or "inferred",
        )
        hook_id = deps.invariant_digest_fn(
            {
                "invariant_id": normalized.invariant_id,
                "scope": normalized.scope,
            },
            prefix="hook",
        )
        path, callable_name = scope.rsplit(":", 1)
        hook_payload: JSONObject = {
            "hook_id": hook_id,
            "invariant_id": normalized.invariant_id or "",
            "callable": {
                "path": path,
                "qual": callable_name,
            },
            "form": normalized.form,
            "terms": list(normalized.terms),
            "confidence": confidence,
            "source": normalized.source or "",
            "source_invariant_evidence_keys": list(normalized.evidence_keys),
        }
        if emit_hypothesis_templates:
            params = ", ".join(normalized.terms)
            hypothesis_name = (
                f"test_{callable_name}_{(normalized.invariant_id or '').replace(':', '_')}"
                .replace("-", "_")
            )
            hook_payload["hypothesis_template"] = "\n".join(
                [
                    "from hypothesis import given",
                    "",
                    f"def {hypothesis_name}():",
                    f"    # invariant: {normalized.form}({params})",
                    "    # TODO: provide strategies and callable invocation.",
                    "    pass",
                ]
            )
        hooks.append(hook_payload)
    hooks = [
        hooks[idx]
        for idx in deps.sort_once_fn(
            range(len(hooks)),
            key=lambda idx: (
                str(hooks[idx].get("hook_id", "")),
                str(hooks[idx].get("invariant_id", "")),
            ),
            source="gabion.analysis.dataflow_indexed_file_scan.generate_property_hook_manifest.site_2",
        )
    ]
    callable_index = build_property_hook_callable_index(
        hooks,
        deps=PropertyHookCallableIndexDeps(
            check_deadline_fn=deps.check_deadline_fn,
            sort_once_fn=deps.sort_once_fn,
        ),
    )
    return {
        "format_version": 1,
        "kind": "property_hook_manifest",
        "min_confidence": threshold,
        "emit_hypothesis_templates": emit_hypothesis_templates,
        "hooks": hooks,
        "callable_index": callable_index,
    }


def build_property_hook_callable_index(
    hooks: Sequence[JSONValue],
    *,
    deps: PropertyHookCallableIndexDeps,
) -> list[JSONObject]:
    callables: dict[str, list[str]] = defaultdict(list)
    for hook in hooks:
        deps.check_deadline_fn()
        match hook:
            case dict() as hook_payload:
                pass
            case _:
                continue
        callable_payload = hook_payload.get("callable")
        match callable_payload:
            case dict() as callable_mapping:
                pass
            case _:
                continue
        path = str(callable_mapping.get("path", "") or "")
        qual = str(callable_mapping.get("qual", "") or "")
        if not path or not qual:
            continue
        callables[f"{path}:{qual}"].append(str(hook_payload.get("hook_id", "") or ""))
    return [
        {
            "scope": scope,
            "hook_ids": deps.sort_once_fn(
                hook_ids,
                source="gabion.analysis.dataflow_indexed_file_scan._build_property_hook_callable_index.site_1",
            ),
        }
        for scope, hook_ids in deps.sort_once_fn(
            callables.items(),
            source="gabion.analysis.dataflow_indexed_file_scan._build_property_hook_callable_index.site_2",
        )
    ]

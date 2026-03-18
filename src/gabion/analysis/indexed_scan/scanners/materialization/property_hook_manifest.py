from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from collections.abc import Iterator
from typing import Callable, Mapping, Sequence, cast

from gabion.analysis.dataflow.engine.dataflow_contracts import InvariantProposition
from gabion.analysis.foundation.json_types import JSONObject, JSONValue
from gabion.analysis.foundation.resume_codec import mapping_optional
from gabion.invariants import never


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


def _hook_payloads(
    hooks: Sequence[JSONValue], *, check_deadline_fn: Callable[[], None]
) -> Iterator[JSONObject]:
    for hook in filter(mapping_optional, hooks):
        check_deadline_fn()
        yield hook


# gabion:ambiguity_boundary
# gabion:boundary_normalization
# gabion:grade_boundary kind=semantic_carrier_adapter name=property_hook_manifest._hook_callable_entries
def _hook_callable_entries(
    hooks: Sequence[JSONValue], *, check_deadline_fn: Callable[[], None]
) -> Iterator[tuple[str, str]]:
    for hook_payload in _hook_payloads(hooks, check_deadline_fn=check_deadline_fn):
        check_deadline_fn()
        for callable_payload in filter(mapping_optional, (hook_payload.get("callable"),)):
            path = str(callable_payload.get("path", "") or "")
            qual = str(callable_payload.get("qual", "") or "")
            if path and qual:
                yield (f"{path}:{qual}", str(hook_payload.get("hook_id", "") or ""))


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
    for scope, hook_id in _hook_callable_entries(
        hooks,
        check_deadline_fn=deps.check_deadline_fn,
    ):
        deps.check_deadline_fn()
        callables[scope].append(hook_id)
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

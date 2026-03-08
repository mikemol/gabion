from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from gabion.analysis.foundation.json_types import JSONObject


@dataclass(frozen=True)
class CallsiteEvidenceDeps:
    check_deadline_fn: Callable[[], None]
    sort_once_fn: Callable[..., list[object]]
    require_not_none_fn: Callable[..., object]
    span_identity_from_tuple_fn: Callable[..., object]


def callsite_evidence_for_bundle(
    calls: list[object],
    bundle: set[str],
    *,
    limit: int = 12,
    deps: CallsiteEvidenceDeps,
) -> list[JSONObject]:
    """Collect callsite evidence for where bundle params are forwarded."""
    deps.check_deadline_fn()
    out: list[JSONObject] = []
    seen: set[tuple[tuple[int, int, int, int], str, tuple[str, ...], tuple[str, ...]]] = set()
    for call in calls:
        deps.check_deadline_fn()
        if call.span is not None:
            params_in_call: list[str] = []
            slots: list[str] = []
            mapping = call.argument_mapping()
            for idx, param in mapping.positional.items():
                deps.check_deadline_fn()
                if param.value in bundle:
                    params_in_call.append(param.value)
                    slots.append(f"arg[{idx}]")
            for name, param in mapping.keywords.items():
                deps.check_deadline_fn()
                if param.value in bundle:
                    params_in_call.append(param.value)
                    slots.append(f"kw[{name}]")
            for idx, param in mapping.star_positional:
                deps.check_deadline_fn()
                if param.value in bundle:
                    params_in_call.append(param.value)
                    slots.append(f"arg[{idx}]*")
            for param in mapping.star_keywords:
                deps.check_deadline_fn()
                if param.value in bundle:
                    params_in_call.append(param.value)
                    slots.append("kw[**]")
            distinct = tuple(
                deps.sort_once_fn(
                    set(params_in_call),
                    source="gabion.analysis.dataflow_indexed_file_scan._callsite_evidence_for_bundle.site_1",
                )
            )
            if distinct:
                slot_list = tuple(
                    deps.sort_once_fn(
                        set(slots),
                        source="gabion.analysis.dataflow_indexed_file_scan._callsite_evidence_for_bundle.site_2",
                    )
                )
                span_identity = deps.span_identity_from_tuple_fn(
                    deps.require_not_none_fn(
                        call.span,
                        reason="callsite evidence requires span",
                        strict=True,
                    )
                )
                span_tuple = (
                    span_identity.start_line,
                    span_identity.start_col,
                    span_identity.end_line,
                    span_identity.end_col,
                )
                callable_id = call.callable_id()
                key = (span_tuple, callable_id.value, distinct, slot_list)
                if key not in seen:
                    seen.add(key)
                    out.append(
                        {
                            "callee": callable_id.value,
                            "span": list(span_tuple),
                            "params": list(distinct),
                            "slots": list(slot_list),
                            "callable_kind": call.callable_kind,
                            "callable_source": call.callable_source,
                        }
                    )
    out = deps.sort_once_fn(
        out,
        source="_ranked_callargs_evidence.out",
        key=lambda entry: (
            -len(entry.get("params") or []),
            tuple(entry.get("span") or []),
            str(entry.get("callee") or ""),
            tuple(entry.get("params") or []),
        ),
    )
    return out[:limit]

# gabion:boundary_normalization_module
# gabion:decision_protocol_module
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import cast


@dataclass(frozen=True)
class SummarizeDeadlineObligationsDeps:
    check_deadline_fn: Callable[[], None]
    projection_spec_hash_fn: Callable[..., str]
    deadline_obligations_summary_spec: object
    require_not_none_fn: Callable[..., object]
    int_tuple4_or_none_fn: Callable[..., object]
    format_span_fields_fn: Callable[..., str]


def summarize_deadline_obligations(
    entries,
    *,
    max_entries: int = 20,
    forest,
    deps: SummarizeDeadlineObligationsDeps,
) -> list[str]:
    deps.check_deadline_fn()
    if not entries:
        return []
    spec_hash = deps.projection_spec_hash_fn(deps.deadline_obligations_summary_spec)
    spec = deps.deadline_obligations_summary_spec
    spec_site = forest.add_spec_site(
        spec_hash=spec_hash,
        spec_name=str(spec.name),
        spec_domain=str(spec.domain),
        spec_version=int(spec.spec_version),
    )
    lines: list[str] = []
    for entry in entries[:max_entries]:
        deps.check_deadline_fn()
        site_payload = entry.get("site", {}) if type(entry) is dict else {}
        site = cast(Mapping[str, object], site_payload if type(site_payload) is dict else {})
        path = str(site.get("path", "?") or "?")
        function = str(site.get("function", "?") or "?")
        parsed_span = deps.require_not_none_fn(
            deps.int_tuple4_or_none_fn(entry.get("span") if type(entry) is dict else None),
            reason="deadline summary requires valid span",
            strict=True,
        )
        suite_kind = str(site.get("suite_kind", "function") or "function")
        status = entry.get("status", "UNKNOWN") if type(entry) is dict else "UNKNOWN"
        kind = entry.get("kind", "?") if type(entry) is dict else "?"
        detail = entry.get("detail", "") if type(entry) is dict else ""
        suite_site = forest.add_suite_site(path, function, "spec", span=parsed_span)
        forest.add_alt(
            "SpecFacet",
            (spec_site, suite_site),
            evidence={
                "spec_hash": spec_hash,
                "spec_name": str(spec.name),
                "status": status,
                "kind": kind,
            },
        )
        span_text = deps.format_span_fields_fn(*parsed_span)
        suffix = f"@{span_text}" if span_text else ""
        line = f"{path}:{function}{suffix} status={status} kind={kind} {detail}".strip()
        lines.append(line)
    if len(entries) > max_entries:
        lines.append(f"... {len(entries) - max_entries} more")
    return lines

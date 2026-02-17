"""Decision-surface helpers extracted from ``dataflow_audit``.

These helpers are intentionally dependency-light and receive runtime hooks
(deadline checks, sorting, and payload conversion) from the caller so
``dataflow_audit`` can keep stable public façades while internal call sites
migrate incrementally.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
import re

from gabion.analysis.json_types import JSONObject


def summarize_deadness_witnesses(
    entries: list[JSONObject],
    *,
    max_entries: int = 10,
    check_deadline: Callable[[], None],
) -> list[str]:
    check_deadline()
    if not entries:
        return []
    lines: list[str] = []
    for entry in entries[:max_entries]:
        check_deadline()
        path = entry.get("path", "?")
        function = entry.get("function", "?")
        bundle = entry.get("bundle", [])
        predicate = entry.get("predicate", "")
        environment = entry.get("environment", {})
        result = entry.get("result", "UNKNOWN")
        core = entry.get("core", [])
        core_count = len(core) if isinstance(core, list) else 0
        lines.append(
            f"{path}:{function} bundle {bundle} result={result} "
            f"predicate={predicate} env={environment} core={core_count}"
        )
    if len(entries) > max_entries:
        lines.append(f"... {len(entries) - max_entries} more")
    return lines


def compute_fingerprint_coherence(
    entries: list[JSONObject],
    *,
    synth_version: str,
    check_deadline: Callable[[], None],
    ordered_or_sorted: Callable[..., list],
) -> list[JSONObject]:
    check_deadline()
    witnesses: list[JSONObject] = []
    for entry in entries:
        check_deadline()
        matches = entry.get("glossary_matches") or []
        if not isinstance(matches, list) or len(matches) < 2:
            continue
        path = entry.get("path")
        function = entry.get("function")
        bundle = entry.get("bundle")
        provenance_id = entry.get("provenance_id")
        base_keys = entry.get("base_keys") or []
        ctor_keys = entry.get("ctor_keys") or []
        bundle_key = ",".join(bundle or [])
        witnesses.append(
            {
                "coherence_id": f"{path}:{function}:{bundle_key}:glossary-ambiguity",
                "site": {
                    "path": path,
                    "function": function,
                    "bundle": bundle,
                },
                "boundary": {
                    "base_keys": base_keys,
                    "ctor_keys": ctor_keys,
                    "synth_version": synth_version,
                },
                "alternatives": ordered_or_sorted(
                    set(str(m) for m in matches),
                    source="_compute_fingerprint_coherence.alternatives",
                ),
                "fork_signature": "glossary-ambiguity",
                "frack_path": ["provenance", "glossary"],
                "result": "UNKNOWN",
                "remainder": {"glossary_matches": matches},
                "provenance_id": provenance_id,
            }
        )
    return ordered_or_sorted(
        witnesses,
        source="_compute_fingerprint_coherence.witnesses",
        key=lambda entry: (
            str(entry.get("site", {}).get("path", "")),
            str(entry.get("site", {}).get("function", "")),
            ",".join(entry.get("site", {}).get("bundle", []) or []),
            str(entry.get("fork_signature", "")),
        ),
    )


def summarize_coherence_witnesses(
    entries: list[JSONObject],
    *,
    max_entries: int = 10,
    check_deadline: Callable[[], None],
) -> list[str]:
    check_deadline()
    if not entries:
        return []
    lines: list[str] = []
    for entry in entries[:max_entries]:
        check_deadline()
        site = entry.get("site", {})
        path = site.get("path", "?")
        function = site.get("function", "?")
        bundle = site.get("bundle", [])
        result = entry.get("result", "UNKNOWN")
        fork_signature = entry.get("fork_signature", "")
        alternatives = entry.get("alternatives", [])
        lines.append(
            f"{path}:{function} bundle {bundle} result={result} "
            f"fork={fork_signature} alternatives={alternatives}"
        )
    if len(entries) > max_entries:
        lines.append(f"... {len(entries) - max_entries} more")
    return lines


def compute_fingerprint_rewrite_plans(
    provenance: list[JSONObject],
    coherence: list[JSONObject],
    *,
    synth_version: str,
    exception_obligations: list[JSONObject] | None,
    check_deadline: Callable[[], None],
    ordered_or_sorted: Callable[..., list],
    site_from_payload: Callable[[JSONObject], object | None],
) -> list[JSONObject]:
    check_deadline()
    coherence_map: dict[tuple[str, str, str], JSONObject] = {}
    for entry in coherence:
        check_deadline()
        raw_site = entry.get("site", {}) or {}
        site = site_from_payload(raw_site)
        if site is None:
            continue
        coherence_map[site.key()] = entry

    include_exception_predicates = exception_obligations is not None
    exception_summary_map: dict[tuple[str, str, str], dict[str, int]] = {}
    if exception_obligations is not None:
        for entry in exception_obligations:
            check_deadline()
            raw_site = entry.get("site", {}) or {}
            site = site_from_payload(raw_site)
            if site is None:
                continue
            if not site.path or not site.function:
                continue
            summary = exception_summary_map.setdefault(
                site.key(),
                {"UNKNOWN": 0, "DEAD": 0, "HANDLED": 0, "total": 0},
            )
            status = str(entry.get("status", "UNKNOWN") or "UNKNOWN")
            if status not in {"UNKNOWN", "DEAD", "HANDLED"}:
                status = "UNKNOWN"
            summary[status] += 1
            summary["total"] += 1

    plans: list[JSONObject] = []
    for entry in provenance:
        check_deadline()
        matches = entry.get("glossary_matches") or []
        if not isinstance(matches, list) or len(matches) < 2:
            continue
        site = site_from_payload(entry)
        if site is None or not site.path or not site.function:
            continue
        bundle_key = site.bundle_key()
        coherence_entry = coherence_map.get(site.key())
        coherence_id = coherence_entry.get("coherence_id") if coherence_entry else None
        candidates = ordered_or_sorted(
            set(str(m) for m in matches),
            source="_compute_fingerprint_rewrite_plans.candidates",
        )
        pre_exception_summary: dict[str, int] | None = None
        if include_exception_predicates:
            pre_exception_summary = exception_summary_map.get(
                site.key(),
                {"UNKNOWN": 0, "DEAD": 0, "HANDLED": 0, "total": 0},
            )
        pre_payload: JSONObject = {
            "base_keys": entry.get("base_keys") or [],
            "ctor_keys": entry.get("ctor_keys") or [],
            "glossary_matches": matches,
            "remainder": entry.get("remainder") or {},
            "synth_version": synth_version,
            **(
                {"exception_obligations_summary": pre_exception_summary}
                if pre_exception_summary is not None
                else {}
            ),
        }
        verification_predicates: list[JSONObject] = [
            {"kind": "base_conservation", "expect": True},
            {"kind": "ctor_coherence", "expect": True},
            {
                "kind": "match_strata",
                "expect": "exact",
                "candidates": candidates,
            },
            {
                "kind": "remainder_non_regression",
                "expect": "no-new-remainder",
            },
            *(
                [
                    {
                        "kind": "exception_obligation_non_regression",
                        "expect": "XV1",
                    }
                ]
                if include_exception_predicates
                else []
            ),
        ]

        def _make_plan(
            *,
            kind: str,
            suffix: str,
            selector: JSONObject,
            parameters: JSONObject,
            post_expectation: JSONObject,
            predicates: list[JSONObject],
        ) -> JSONObject:
            return {
                "plan_id": (
                    f"rewrite:{site.path}:{site.function}:{bundle_key}:"
                    f"glossary-ambiguity:{suffix}"
                ),
                "status": "UNVERIFIED",
                "site": {
                    "path": site.path,
                    "function": site.function,
                    "bundle": list(site.bundle),
                },
                "pre": dict(pre_payload),
                "rewrite": {
                    "kind": kind,
                    "selector": selector,
                    "parameters": parameters,
                },
                "evidence": {
                    "provenance_id": entry.get("provenance_id"),
                    "coherence_id": coherence_id,
                },
                "post_expectation": post_expectation,
                "verification": {
                    "mode": "re-audit",
                    "status": "UNVERIFIED",
                    "predicates": predicates,
                },
            }

        plans.append(
            _make_plan(
                kind="BUNDLE_ALIGN",
                suffix="bundle-align",
                selector={"bundle": list(site.bundle)},
                parameters={"candidates": candidates},
                post_expectation={
                    "match_strata": "exact",
                    "base_conservation": True,
                    "ctor_coherence": True,
                },
                predicates=list(verification_predicates),
            )
        )

        plans.append(
            _make_plan(
                kind="CTOR_NORMALIZE",
                suffix="ctor-normalize",
                selector={"bundle": list(site.bundle)},
                parameters={
                    "target_ctor_keys": list(entry.get("ctor_keys") or []),
                    "candidates": candidates,
                },
                post_expectation={
                    "ctor_normalized": True,
                    "match_strata": "exact",
                    "base_conservation": True,
                },
                predicates=[
                    {"kind": "base_conservation", "expect": True},
                    {"kind": "ctor_coherence", "expect": True},
                    {
                        "kind": "match_strata",
                        "expect": "exact",
                        "candidates": candidates,
                    },
                    {
                        "kind": "remainder_non_regression",
                        "expect": "no-new-remainder",
                    },
                    *(
                        [
                            {
                                "kind": "exception_obligation_non_regression",
                                "expect": "XV1",
                            }
                        ]
                        if include_exception_predicates
                        else []
                    ),
                ],
            )
        )

        plans.append(
            _make_plan(
                kind="SURFACE_CANONICALIZE",
                suffix="surface-canonicalize",
                selector={"bundle": list(site.bundle), "glossary_matches": matches},
                parameters={
                    "canonical_candidate": candidates[0] if candidates else "",
                    "candidates": candidates,
                },
                post_expectation={
                    "match_strata": "exact",
                    "surface_canonicalized": True,
                    "base_conservation": True,
                },
                predicates=[
                    {"kind": "base_conservation", "expect": True},
                    {
                        "kind": "match_strata",
                        "expect": "exact",
                        "candidates": candidates,
                    },
                    {
                        "kind": "remainder_non_regression",
                        "expect": "no-new-remainder",
                    },
                    *(
                        [
                            {
                                "kind": "exception_obligation_non_regression",
                                "expect": "XV1",
                            }
                        ]
                        if include_exception_predicates
                        else []
                    ),
                ],
            )
        )

        plans.append(
            _make_plan(
                kind="AMBIENT_REWRITE",
                suffix="ambient-rewrite",
                selector={"bundle": list(site.bundle)},
                parameters={
                    "strategy": "context-explicit",
                    "candidates": candidates,
                },
                post_expectation={
                    "match_strata": "exact",
                    "ambient_normalized": True,
                    "base_conservation": True,
                },
                predicates=[
                    {"kind": "base_conservation", "expect": True},
                    {
                        "kind": "match_strata",
                        "expect": "exact",
                        "candidates": candidates,
                    },
                    {
                        "kind": "remainder_non_regression",
                        "expect": "no-new-remainder",
                    },
                    *(
                        [
                            {
                                "kind": "exception_obligation_non_regression",
                                "expect": "XV1",
                            }
                        ]
                        if include_exception_predicates
                        else []
                    ),
                ],
            )
        )
    return ordered_or_sorted(
        plans,
        source="_compute_fingerprint_rewrite_plans.plans",
        key=lambda entry: str(entry.get("plan_id", "")),
    )


def summarize_rewrite_plans(
    entries: list[JSONObject],
    *,
    max_entries: int = 10,
    check_deadline: Callable[[], None],
) -> list[str]:
    check_deadline()
    if not entries:
        return []
    lines: list[str] = []
    for entry in entries[:max_entries]:
        check_deadline()
        plan_id = entry.get("plan_id", "?")
        site = entry.get("site", {})
        path = site.get("path", "?")
        function = site.get("function", "?")
        bundle = site.get("bundle", [])
        kind = entry.get("rewrite", {}).get("kind", "?")
        status = entry.get("status", "UNVERIFIED")
        lines.append(
            f"{plan_id} {path}:{function} bundle={bundle} kind={kind} status={status}"
        )
    if len(entries) > max_entries:
        lines.append(f"... {len(entries) - max_entries} more")
    return lines


def parse_lint_location(line: str) -> tuple[str, int, int, str] | None:
    match = re.match(r"^(?P<path>[^:]+):(?P<line>\d+):(?P<col>\d+)", line)
    if not match:
        return None
    path = match.group("path")
    lineno = int(match.group("line"))
    col = int(match.group("col"))
    remainder = line[match.end() :].lstrip(": ").strip()
    if remainder.startswith("-"):
        trimmed = remainder[1:]
        range_match = re.match(r"^(\d+):(\d+)(:)?\s*", trimmed)
        if range_match:
            remainder = trimmed[range_match.end() :].strip()
    return path, lineno, col, remainder


def lint_lines_from_bundle_evidence(
    evidence: Iterable[str],
    *,
    check_deadline: Callable[[], None],
    lint_line: Callable[[str, int, int, str, str], str],
) -> list[str]:
    check_deadline()
    lines: list[str] = []
    for entry in evidence:
        check_deadline()
        parsed = parse_lint_location(entry)
        if not parsed:
            continue
        path, lineno, col, remainder = parsed
        message = remainder or "undocumented bundle"
        lines.append(lint_line(path, lineno, col, "GABION_BUNDLE_UNDOC", message))
    return lines


def lint_lines_from_type_evidence(
    evidence: Iterable[str],
    *,
    check_deadline: Callable[[], None],
    lint_line: Callable[[str, int, int, str, str], str],
) -> list[str]:
    check_deadline()
    lines: list[str] = []
    for entry in evidence:
        check_deadline()
        parsed = parse_lint_location(entry)
        if not parsed:
            continue
        path, lineno, col, remainder = parsed
        message = remainder or "type-flow evidence"
        lines.append(lint_line(path, lineno, col, "GABION_TYPE_FLOW", message))
    return lines


def lint_lines_from_unused_arg_smells(
    smells: Iterable[str],
    *,
    check_deadline: Callable[[], None],
    lint_line: Callable[[str, int, int, str, str], str],
) -> list[str]:
    check_deadline()
    lines: list[str] = []
    for entry in smells:
        check_deadline()
        parsed = parse_lint_location(entry)
        if not parsed:
            continue
        path, lineno, col, remainder = parsed
        message = remainder or "unused argument flow"
        lines.append(lint_line(path, lineno, col, "GABION_UNUSED_ARG", message))
    return lines


def extract_smell_sample(entry: str) -> str | None:
    match = re.search(r"\(e\.g\.\s*([^)]+)\)", entry)
    if not match:
        return None
    return match.group(1).strip()


def lint_lines_from_constant_smells(
    smells: Iterable[str],
    *,
    check_deadline: Callable[[], None],
    lint_line: Callable[[str, int, int, str, str], str],
) -> list[str]:
    check_deadline()
    lines: list[str] = []
    for entry in smells:
        check_deadline()
        parsed = parse_lint_location(entry)
        if not parsed:
            sample = extract_smell_sample(entry)
            if sample:
                parsed = parse_lint_location(sample)
        if not parsed:
            continue
        path, lineno, col, _ = parsed
        lines.append(lint_line(path, lineno, col, "GABION_CONST_FLOW", entry))
    return lines

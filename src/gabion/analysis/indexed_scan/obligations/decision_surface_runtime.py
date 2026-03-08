from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class DecisionSurfaceAnalyzeDeps:
    build_call_graph_fn: Callable[..., tuple[object, dict[str, object], dict[str, set[str]]]]
    check_deadline_fn: Callable[[], None]
    is_test_path_fn: Callable[..., bool]
    sort_once_fn: Callable[..., list[str]]
    decision_reason_summary_fn: Callable[..., str]
    decision_surface_alt_evidence_fn: Callable[..., object]
    suite_site_label_fn: Callable[..., str]
    decision_tier_for_fn: Callable[..., object]
    decision_param_lint_line_fn: Callable[..., object]


def analyze_decision_surface_indexed(
    context: object,
    *,
    spec: object,
    decision_tiers,
    require_tiers: bool,
    forest: object,
    deps: DecisionSurfaceAnalyzeDeps,
) -> tuple[list[str], list[str], list[str], list[str]]:
    _, by_qual, transitive_callers = deps.build_call_graph_fn(
        context.paths,
        project_root=context.project_root,
        ignore_params=context.ignore_params,
        strictness=context.strictness,
        external_filter=context.external_filter,
        transparent_decorators=context.transparent_decorators,
        parse_failure_witnesses=context.parse_failure_witnesses,
        analysis_index=context.analysis_index,
    )
    surfaces: list[str] = []
    warnings: list[str] = []
    rewrites: list[str] = []
    lint_lines: list[str] = []
    tier_map = decision_tiers or {}
    for info in by_qual.values():
        deps.check_deadline_fn()
        if deps.is_test_path_fn(info.path):
            continue
        params = deps.sort_once_fn(
            spec.params(info),
            source=f"_analyze_decision_surface_indexed.{spec.pass_id}.params",
        )
        if not params:
            continue
        caller_count = len(transitive_callers.get(info.qual, set()))
        boundary = (
            "boundary"
            if caller_count == 0
            else f"internal callers (transitive): {caller_count}"
        )
        descriptor = spec.descriptor(info, boundary)
        suite_id = forest.add_suite_site(
            info.path.name,
            info.qual,
            "function_body",
            span=info.function_span,
        )
        paramset_id = forest.add_paramset(params)
        reason_summary = (
            deps.decision_reason_summary_fn(info, params)
            if spec.pass_id == "decision_surfaces"
            else descriptor
        )
        forest.add_alt(
            spec.alt_kind,
            (suite_id, paramset_id),
            evidence=deps.decision_surface_alt_evidence_fn(
                spec=spec,
                boundary=boundary,
                descriptor=descriptor,
                params=params,
                caller_count=caller_count,
                reason_summary=reason_summary,
            ),
        )
        surfaces.append(
            f"{deps.suite_site_label_fn(forest=forest, suite_id=suite_id)} {spec.surface_label}: "
            + ", ".join(params)
            + f" ({descriptor})"
        )
        if spec.rewrite_line is not None:
            rewrites.append(spec.rewrite_line(info, params, descriptor))
        for param in params:
            deps.check_deadline_fn()
            tier = deps.decision_tier_for_fn(
                info,
                param,
                tier_map=tier_map,
                project_root=context.project_root,
            )
            if spec.emit_surface_lint(caller_count, tier):
                lint = deps.decision_param_lint_line_fn(
                    info,
                    param,
                    project_root=context.project_root,
                    code=spec.surface_lint_code,
                    message=spec.surface_lint_message(param, boundary, descriptor),
                )
                if lint is not None:
                    lint_lines.append(lint)
            if not tier_map:
                continue
            if tier is None:
                if require_tiers:
                    message = spec.tier_missing_message(param, descriptor)
                    warnings.append(f"{info.path.name}:{info.qual} {message}")
                    lint = deps.decision_param_lint_line_fn(
                        info,
                        param,
                        project_root=context.project_root,
                        code=spec.tier_lint_code,
                        message=message,
                    )
                    if lint is not None:
                        lint_lines.append(lint)
                continue
            if tier in {2, 3} and caller_count > 0:
                message = spec.tier_internal_message(param, tier, boundary, descriptor)
                warnings.append(f"{info.path.name}:{info.qual} {message}")
                lint = deps.decision_param_lint_line_fn(
                    info,
                    param,
                    project_root=context.project_root,
                    code=spec.tier_lint_code,
                    message=message,
                )
                if lint is not None:
                    lint_lines.append(lint)
    return (
        deps.sort_once_fn(
            surfaces,
            source="_analyze_decision_surface_indexed.surfaces",
        ),
        deps.sort_once_fn(
            set(warnings),
            source="_analyze_decision_surface_indexed.warnings",
        ),
        deps.sort_once_fn(
            rewrites,
            source="_analyze_decision_surface_indexed.rewrites",
        ),
        deps.sort_once_fn(
            set(lint_lines),
            source="_analyze_decision_surface_indexed.lint_lines",
        ),
    )

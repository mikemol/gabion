from __future__ import annotations

from pathlib import Path
from typing import Callable

from .contracts import Doc, DocflowAuditContext, DocflowDomainResult, DocflowObligationResult


def run_docflow_domain(
    *,
    root: Path,
    extra_paths: list[str] | None,
    extra_strict: list[str] | None,
    sppf_gh_ref_mode: str,
    baseline_write_emitted: bool,
    delta_guard_checked: bool,
    build_context: Callable[..., DocflowAuditContext],
    load_docs: Callable[..., dict[str, Doc]],
    evaluate_obligations: Callable[..., DocflowObligationResult],
) -> tuple[DocflowDomainResult, dict[str, Doc]]:
    context = build_context(
        root,
        extra_paths=extra_paths,
        extra_strict=extra_strict,
        sppf_gh_ref_mode=sppf_gh_ref_mode,
    )
    docs = load_docs(root=root, extra_paths=extra_paths)
    obligations = evaluate_obligations(
        root=root,
        violations=context.violations,
        baseline_write_emitted=baseline_write_emitted,
        delta_guard_checked=delta_guard_checked,
    )
    warnings = [*context.warnings, *obligations.warnings]
    violations = [*context.violations, *obligations.violations]
    return (
        DocflowDomainResult(
            context=context,
            obligations=obligations,
            warnings=warnings,
            violations=violations,
        ),
        docs,
    )

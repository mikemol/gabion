from __future__ import annotations

from dataclasses import dataclass

from gabion.analysis.dataflow.engine.scan_kernel import (
    ScanKernelDeps,
    ScanKernelPass,
    ScanKernelRequest,
    run_canonical_scan_ingress,
)


@dataclass(frozen=True)
class DecisionSurfaceAnalyzerInput:
    kernel_request: ScanKernelRequest
    decision_tiers: object
    require_tiers: bool
    forest: object


@dataclass(frozen=True)
class DecisionSurfaceAnalyzerOutput:
    surfaces: list[str]
    warnings: list[str]
    lint_lines: list[str]


@dataclass(frozen=True)
class ValueDecisionAnalyzerOutput:
    surfaces: list[str]
    warnings: list[str]
    rewrites: list[str]
    lint_lines: list[str]


def analyze_decision_surfaces(
    *,
    data: DecisionSurfaceAnalyzerInput,
    deps: ScanKernelDeps[tuple[list[str], list[str], list[str]]],
    runner,
) -> DecisionSurfaceAnalyzerOutput:
    result = run_canonical_scan_ingress(
        request=data.kernel_request,
        spec=ScanKernelPass(pass_id="decision_surfaces", run=runner),
        deps=deps,
    )
    surfaces, warnings, lint_lines = result
    return DecisionSurfaceAnalyzerOutput(
        surfaces=surfaces,
        warnings=warnings,
        lint_lines=lint_lines,
    )


def analyze_value_encoded_decisions(
    *,
    data: DecisionSurfaceAnalyzerInput,
    deps: ScanKernelDeps[tuple[list[str], list[str], list[str], list[str]]],
    runner,
) -> ValueDecisionAnalyzerOutput:
    result = run_canonical_scan_ingress(
        request=data.kernel_request,
        spec=ScanKernelPass(pass_id="value_encoded_decisions", run=runner),
        deps=deps,
    )
    surfaces, warnings, rewrites, lint_lines = result
    return ValueDecisionAnalyzerOutput(
        surfaces=surfaces,
        warnings=warnings,
        rewrites=rewrites,
        lint_lines=lint_lines,
    )

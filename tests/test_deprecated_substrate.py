from __future__ import annotations

from gabion.analysis.deprecated_substrate import (
    DeprecatedBlocker,
    DeprecatedLifecycleState,
    build_deprecated_extraction_artifacts,
    deprecated,
    detect_report_section_extinction,
    ingest_perf_samples,
)


def test_deprecated_requires_canonical_path_and_blocker_payload() -> None:
    blocker = DeprecatedBlocker(blocker_id="B1", kind="owner", summary="needs owner")
    fiber = deprecated(canonical_aspf_path=("pkg", "fn"), blockers=(blocker,))
    assert fiber.fiber_id == "aspf:pkg/fn"


def test_extraction_pipeline_is_deterministic() -> None:
    samples = ingest_perf_samples(
        [
            {"stack": ["a", "b"], "weight": 2},
            {"stack": ["a", "b"], "weight": 3},
            {"stack": ["a", "c"], "weight": 1},
        ]
    )
    blocker = DeprecatedBlocker(
        blocker_id="B2",
        kind="dependency",
        summary="upstream pending",
        lifecycle=DeprecatedLifecycleState.BLOCKED,
        depends_on=("B1",),
    )
    fibers = (
        deprecated(canonical_aspf_path=("a", "b"), blockers=(blocker,)),
    )
    artifacts = build_deprecated_extraction_artifacts(
        perf_samples=samples,
        deprecated_fibers=fibers,
        branch_coverage_previous={"a::b": 0.8},
        branch_coverage_current={"a::b": 0.6},
    )
    assert list(artifacts.perf_fiber_groups) == [
        {"fiber_group": "a::b", "canonical_aspf_path": ["a", "b"], "weight": 5},
        {"fiber_group": "a::c", "canonical_aspf_path": ["a", "c"], "weight": 1},
    ]
    assert list(artifacts.fiber_group_rankings)[0]["fiber_group"] == "a::b"
    assert artifacts.blocker_dag["edges"] == ({"from": "B2", "to": "B1"},)
    assert artifacts.informational_signals == (
        "informational: branch coverage loss for a::b (0.800 -> 0.600)",
    )


def test_report_section_extinction_detection() -> None:
    extinctions = detect_report_section_extinction(
        previous_sections=("intro", "violations", "deprecated_substrate"),
        current_sections=("intro",),
    )
    assert extinctions == ("deprecated_substrate", "violations")

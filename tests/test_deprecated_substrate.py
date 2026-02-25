from __future__ import annotations

import pytest

from gabion.analysis.deprecated_substrate import (
    DeprecatedBlocker,
    DeprecatedFiber,
    DeprecatedLifecycleState,
    build_deprecated_extraction_artifacts,
    check_semantic_fiber_continuity,
    classify_branch_coverage_loss,
    deprecated,
    detect_report_section_extinction,
    enforce_non_erasability_policy,
    ingest_perf_samples,
    rank_fiber_groups,
)


# gabion:evidence E:function_site::tests/test_deprecated_substrate.py::tests.test_deprecated_substrate.test_deprecated_requires_canonical_path_and_blocker_payload
def test_deprecated_requires_canonical_path_and_blocker_payload() -> None:
    blocker = DeprecatedBlocker(blocker_id="B1", kind="owner", summary="needs owner")
    fiber = deprecated(canonical_aspf_path=("pkg", "fn"), blockers=(blocker,))
    assert fiber.fiber_id == "aspf:pkg/fn"


# gabion:evidence E:function_site::tests/test_deprecated_substrate.py::tests.test_deprecated_substrate.test_extraction_pipeline_is_deterministic
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


# gabion:evidence E:function_site::tests/test_deprecated_substrate.py::tests.test_deprecated_substrate.test_report_section_extinction_detection
def test_report_section_extinction_detection() -> None:
    extinctions = detect_report_section_extinction(
        previous_sections=("intro", "violations", "deprecated_substrate"),
        current_sections=("intro",),
    )
    assert extinctions == ("deprecated_substrate", "violations")


# gabion:evidence E:function_site::tests/test_deprecated_substrate.py::tests.test_deprecated_substrate.test_blocker_and_fiber_payload_edges
def test_blocker_and_fiber_payload_edges() -> None:
    blocker = DeprecatedBlocker.from_payload(
        {
            "blocker_id": "B3",
            "kind": "policy",
            "summary": "needs review",
            "lifecycle": "resolved",
            "depends_on": "not-a-sequence",
        }
    )
    assert blocker.depends_on == ()
    assert blocker.lifecycle is DeprecatedLifecycleState.RESOLVED

    with pytest.raises(ValueError):
        DeprecatedBlocker.from_payload({"blocker_id": "", "kind": "x", "summary": "s"})

    with pytest.raises(ValueError):
        DeprecatedFiber.from_payload({"canonical_aspf_path": "bad"})

    fiber = DeprecatedFiber.from_payload(
        {
            "canonical_aspf_path": ["pkg", "fn"],
            "blocker_payload": [
                {"blocker_id": "B4", "kind": "owner", "summary": "missing owner"},
                "skip-me",
            ],
            "resolution_metadata": "not-a-mapping",
        }
    )
    assert fiber.blocker_payload[0].blocker_id == "B4"
    assert fiber.resolution_metadata is None

    resolved_without_sequence_blockers = DeprecatedFiber.from_payload(
        {
            "canonical_aspf_path": ["pkg", "resolved"],
            "lifecycle": "resolved",
            "blocker_payload": "bad-shape",
            "resolution_metadata": {"ticket": "GH-2"},
        }
    )
    assert resolved_without_sequence_blockers.lifecycle is DeprecatedLifecycleState.RESOLVED


# gabion:evidence E:function_site::tests/test_deprecated_substrate.py::tests.test_deprecated_substrate.test_deprecated_constructor_and_gating_edges
def test_deprecated_constructor_and_gating_edges() -> None:
    blocker = DeprecatedBlocker(blocker_id="B5", kind="owner", summary="owner needed")

    with pytest.raises(ValueError):
        deprecated(canonical_aspf_path=(), blockers=(blocker,))
    with pytest.raises(ValueError):
        deprecated(canonical_aspf_path=("pkg",), blockers=())
    with pytest.raises(ValueError):
        deprecated(
            canonical_aspf_path=("pkg",),
            blockers=(),
            lifecycle=DeprecatedLifecycleState.RESOLVED,
        )

    resolved = deprecated(
        canonical_aspf_path=("pkg", "resolved"),
        blockers=(),
        lifecycle=DeprecatedLifecycleState.RESOLVED,
        resolution_metadata={"ticket": "GH-1"},
    )
    assert resolved.fiber_id == "aspf:pkg/resolved"

    explicit = deprecated(
        canonical_aspf_path=("pkg", "explicit"),
        blockers=(blocker,),
        fiber_id="custom:fiber",
    )
    assert explicit.fiber_id == "custom:fiber"

    previous = (
        deprecated(canonical_aspf_path=("pkg", "missing"), blockers=(blocker,)),
        resolved,
    )
    continuity = check_semantic_fiber_continuity(
        previous_fibers=previous,
        current_fibers=(resolved,),
    )
    assert continuity == ("aspf:pkg/missing",)

    result = enforce_non_erasability_policy(
        previous_fibers=previous,
        current_fibers=(resolved,),
    )
    assert result.errors == (
        "deprecated fiber erased without explicit resolution metadata: aspf:pkg/missing",
    )

    resolved_only = enforce_non_erasability_policy(
        previous_fibers=(resolved,),
        current_fibers=(),
    )
    assert resolved_only.ok is True


# gabion:evidence E:function_site::tests/test_deprecated_substrate.py::tests.test_deprecated_substrate.test_ingest_rank_and_branch_loss_edges
def test_ingest_rank_and_branch_loss_edges() -> None:
    samples = ingest_perf_samples(
        [
            {"stack": "bad"},
            {"stack": []},
            {"stack": ["a", ""], "weight": 0},
            {"stack": ["a", "b"], "weight": 2},
        ]
    )
    assert [sample.stack for sample in samples] == [("a",), ("a", "b")]
    assert [sample.weight for sample in samples] == [1, 2]

    rankings = rank_fiber_groups(
        [
            {"fiber_group": "", "weight": 9},
            {"fiber_group": "a::b", "weight": 3},
        ]
    )
    assert rankings == ({"rank": 1, "fiber_group": "a::b", "weight": 3},)

    assert classify_branch_coverage_loss(
        previous={"pkg::fn": 0.4},
        current={"pkg::fn": 0.9},
    ) == []

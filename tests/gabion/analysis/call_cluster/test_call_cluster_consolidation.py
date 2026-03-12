from __future__ import annotations

import json
from pathlib import Path

from gabion.analysis.call_cluster import call_cluster_consolidation, call_cluster_shared
from gabion.analysis.foundation.baseline_io import ParsedSpecMetadata
from gabion.analysis.semantics import evidence_keys


def _call_footprint_display(*, test_id: str, file: str, targets: list[tuple[str, str]]) -> str:
    key = evidence_keys.make_call_footprint_key(
        path=file,
        qual=test_id.split("::", 1)[1],
        targets=[{"path": path, "qual": qual} for path, qual in targets],
    )
    return evidence_keys.render_display(evidence_keys.normalize_key(key))


def _generated_by(
    *,
    spec_id: str = "call_cluster_consolidation",
    spec_payload: dict[str, object] | None = None,
) -> ParsedSpecMetadata:
    return ParsedSpecMetadata(
        spec_id=spec_id,
        spec={
            str(key): value
            for key, value in (spec_payload or {
                "name": "call_cluster_consolidation",
                "spec_version": 1,
            }).items()
        },
    )


def _payload(
    *,
    clusters: int = 0,
    tests: int = 0,
    replacements: int = 0,
    min_cluster_size: int = 2,
    cluster_entries: tuple[call_cluster_consolidation.ClusterSummary, ...] = (),
    plan_entries: tuple[call_cluster_consolidation.ConsolidationPlanEntry, ...] = (),
    generated_by_spec_id: str = "call_cluster_consolidation",
    generated_by_spec: dict[str, object] | None = None,
) -> call_cluster_consolidation.CallClusterConsolidationPayload:
    return call_cluster_consolidation.CallClusterConsolidationPayload(
        version=call_cluster_consolidation.CONSOLIDATION_VERSION,
        summary=call_cluster_consolidation.ConsolidationSummary(
            clusters=clusters,
            tests=tests,
            replacements=replacements,
            min_cluster_size=min_cluster_size,
        ),
        clusters=cluster_entries,
        plan=plan_entries,
        generated_by=_generated_by(
            spec_id=generated_by_spec_id,
            spec_payload=generated_by_spec,
        ),
    )


def _emitted_payload(
    payload: call_cluster_consolidation.CallClusterConsolidationPayload,
) -> dict[str, object]:
    return call_cluster_consolidation.render_json_payload(payload)


# gabion:evidence E:function_site::call_cluster_consolidation.py::gabion.analysis.call_cluster_consolidation.build_call_cluster_consolidation_payload
# gabion:evidence E:function_site::call_cluster_consolidation.py::gabion.analysis.call_cluster_consolidation.render_markdown E:decision_surface/direct::call_cluster_consolidation.py::gabion.analysis.call_cluster_consolidation.render_markdown::stale_53de2d5d4377_a503856f
# gabion:behavior primary=desired
def test_call_cluster_consolidation_payload_and_render(
    tmp_path: Path,
    write_test_evidence_payload,
    test_evidence_path: Path,
) -> None:
    evidence_path = test_evidence_path
    shared_targets = [("sample.py", "pkg.fn")]
    display_one = _call_footprint_display(
        test_id="tests/test_sample.py::test_one",
        file="tests/test_sample.py",
        targets=shared_targets,
    )
    display_two = _call_footprint_display(
        test_id="tests/test_sample.py::test_two",
        file="tests/test_sample.py",
        targets=shared_targets,
    )
    display_other = _call_footprint_display(
        test_id="tests/test_sample.py::test_three",
        file="tests/test_sample.py",
        targets=[("other.py", "pkg.other")],
    )
    cluster_key = evidence_keys.make_call_cluster_key(targets=[{"path": "sample.py", "qual": "pkg.fn"}])
    cluster_display = evidence_keys.render_display(evidence_keys.normalize_key(cluster_key))
    entries = [
        {
            "test_id": "tests/test_sample.py::test_one",
            "file": "tests/test_sample.py",
            "line": 10,
            "evidence": [display_one],
            "status": "mapped",
        },
        {
            "test_id": "tests/test_sample.py::test_two",
            "file": "tests/test_sample.py",
            "line": 20,
            "evidence": [display_two],
            "status": "mapped",
        },
        {
            "test_id": "tests/test_sample.py::test_three",
            "file": "tests/test_sample.py",
            "line": 30,
            "evidence": [display_other],
            "status": "mapped",
        },
        {
            "test_id": "tests/test_sample.py::test_four",
            "file": "tests/test_sample.py",
            "line": 40,
            "evidence": [display_one, cluster_display],
            "status": "mapped",
        },
    ]
    write_test_evidence_payload(evidence_path, entries=entries)

    payload = call_cluster_consolidation.build_call_cluster_consolidation_payload(
        evidence_path=evidence_path,
        min_cluster_size=2,
    )
    assert payload.summary.clusters == 1
    assert payload.summary.tests == 2
    assert payload.clusters
    assert payload.plan

    markdown = call_cluster_consolidation.render_markdown(payload)
    assert "generated_by_spec_id" in markdown
    assert "Consolidation plan" in markdown
    assert "E:call_cluster" in markdown


# gabion:evidence E:function_site::call_cluster_consolidation.py::gabion.analysis.call_cluster_consolidation.render_markdown E:decision_surface/direct::call_cluster_consolidation.py::gabion.analysis.call_cluster_consolidation.render_markdown::stale_d3f9e19479c9
# gabion:behavior primary=verboten facets=empty
def test_call_cluster_consolidation_render_empty() -> None:
    payload = _payload()
    markdown = call_cluster_consolidation.render_markdown(payload)
    assert "No consolidation candidates" in markdown


# gabion:evidence E:function_site::call_cluster_consolidation.py::gabion.analysis.call_cluster_consolidation.build_call_cluster_consolidation_payload E:decision_surface/direct::call_cluster_consolidation.py::gabion.analysis.call_cluster_consolidation.build_call_cluster_consolidation_payload::stale_f457d4364b38
# gabion:behavior primary=verboten facets=empty
def test_call_cluster_consolidation_skips_unparseable_and_empty_targets(
    tmp_path: Path,
    write_test_evidence_payload,
    test_evidence_path: Path,
) -> None:
    evidence_path = test_evidence_path
    display_empty = _call_footprint_display(
        test_id="tests/test_sample.py::test_empty",
        file="tests/test_sample.py",
        targets=[],
    )
    entries = [
        {
            "test_id": "tests/test_sample.py::test_invalid",
            "file": "tests/test_sample.py",
            "line": 10,
            "evidence": ["not-a-key"],
            "status": "mapped",
        },
        {
            "test_id": "tests/test_sample.py::test_empty",
            "file": "tests/test_sample.py",
            "line": 20,
            "evidence": [display_empty],
            "status": "mapped",
        },
    ]
    write_test_evidence_payload(evidence_path, entries=entries)

    payload = call_cluster_consolidation.build_call_cluster_consolidation_payload(
        evidence_path=evidence_path,
        min_cluster_size=2,
    )
    assert payload.summary.clusters == 0
    assert payload.summary.tests == 0
    assert payload.plan == ()


# gabion:evidence E:function_site::call_cluster_consolidation.py::gabion.analysis.call_cluster_consolidation.build_call_cluster_consolidation_payload E:decision_surface/direct::call_cluster_consolidation.py::gabion.analysis.call_cluster_consolidation.build_call_cluster_consolidation_payload::stale_0ac9f5d3d039_0e8f5b26
# gabion:behavior primary=desired
def test_call_cluster_consolidation_skips_multiple_target_sets(
    tmp_path: Path,
    write_test_evidence_payload,
    test_evidence_path: Path,
) -> None:
    evidence_path = test_evidence_path
    display_one = _call_footprint_display(
        test_id="tests/test_sample.py::test_multi",
        file="tests/test_sample.py",
        targets=[("sample.py", "pkg.fn")],
    )
    display_two = _call_footprint_display(
        test_id="tests/test_sample.py::test_multi",
        file="tests/test_sample.py",
        targets=[("other.py", "pkg.other")],
    )
    entries = [
        {
            "test_id": "tests/test_sample.py::test_multi",
            "file": "tests/test_sample.py",
            "line": 10,
            "evidence": [display_one, display_two],
            "status": "mapped",
        }
    ]
    write_test_evidence_payload(evidence_path, entries=entries)

    payload = call_cluster_consolidation.build_call_cluster_consolidation_payload(
        evidence_path=evidence_path,
        min_cluster_size=2,
    )
    assert payload.summary.clusters == 0
    assert payload.summary.tests == 0


# gabion:evidence E:call_footprint::tests/test_call_cluster_consolidation.py::test_call_cluster_consolidation_accepts_call_cluster_tokens_in_evidence::call_cluster_consolidation.py::gabion.analysis.call_cluster_consolidation.build_call_cluster_consolidation_payload::evidence_keys.py::gabion.analysis.evidence_keys.make_call_cluster_key::evidence_keys.py::gabion.analysis.evidence_keys.normalize_key::evidence_keys.py::gabion.analysis.evidence_keys.render_display
# gabion:behavior primary=desired
def test_call_cluster_consolidation_accepts_call_cluster_tokens_in_evidence(
    tmp_path: Path,
    write_test_evidence_payload,
    test_evidence_path: Path,
) -> None:
    evidence_path = test_evidence_path
    cluster_key = evidence_keys.make_call_cluster_key(
        targets=[{"path": "sample.py", "qual": "pkg.fn"}]
    )
    cluster_display = evidence_keys.render_display(evidence_keys.normalize_key(cluster_key))
    entries = [
        {
            "test_id": "tests/test_sample.py::test_cluster_only",
            "file": "tests/test_sample.py",
            "line": 10,
            "evidence": [cluster_display],
            "status": "mapped",
        }
    ]
    write_test_evidence_payload(evidence_path, entries=entries)

    payload = call_cluster_consolidation.build_call_cluster_consolidation_payload(
        evidence_path=evidence_path,
        min_cluster_size=2,
    )
    assert payload.summary.clusters == 0
    assert payload.summary.tests == 0


# gabion:evidence E:call_footprint::tests/test_call_cluster_consolidation.py::test_call_cluster_consolidation_ignores_non_cluster_kinds_with_targets::call_cluster_consolidation.py::gabion.analysis.call_cluster_consolidation.build_call_cluster_consolidation_payload::evidence_keys.py::gabion.analysis.evidence_keys.render_display
# gabion:behavior primary=desired
def test_call_cluster_consolidation_ignores_non_cluster_kinds_with_targets(
    tmp_path: Path,
    write_test_evidence_payload,
    test_evidence_path: Path,
) -> None:
    evidence_path = test_evidence_path
    unknown_display = evidence_keys.render_display(
        {
            "k": "custom_kind",
            "targets": [{"path": "sample.py", "qual": "pkg.fn"}],
        }
    )
    entries = [
        {
            "test_id": "tests/test_sample.py::test_custom",
            "file": "tests/test_sample.py",
            "line": 10,
            "evidence": [unknown_display],
            "status": "mapped",
        }
    ]
    write_test_evidence_payload(evidence_path, entries=entries)

    payload = call_cluster_consolidation.build_call_cluster_consolidation_payload(
        evidence_path=evidence_path,
        min_cluster_size=2,
    )
    assert payload.summary.clusters == 0
    assert payload.summary.tests == 0


# gabion:evidence E:call_footprint::tests/test_call_cluster_consolidation.py::test_call_cluster_consolidation_takes_call_cluster_branch_via_payload::call_cluster_consolidation.py::gabion.analysis.call_cluster_consolidation.build_call_cluster_consolidation_payload::evidence_keys.py::gabion.analysis.evidence_keys.make_call_cluster_key::evidence_keys.py::gabion.analysis.evidence_keys.render_display
# gabion:behavior primary=desired
def test_call_cluster_consolidation_takes_call_cluster_branch_via_payload(
    write_test_evidence_payload,
    test_evidence_path: Path,
) -> None:
    key = evidence_keys.make_call_cluster_key(
        targets=[{"path": "sample.py", "qual": "pkg.fn"}]
    )
    write_test_evidence_payload(
        test_evidence_path,
        entries=[
            {
                "test_id": "tests/test_sample.py::test_call_cluster",
                "file": "tests/test_sample.py",
                "line": 10,
                "evidence": [evidence_keys.render_display(key)],
                "status": "mapped",
            }
        ],
    )
    payload = call_cluster_consolidation.build_call_cluster_consolidation_payload(
        evidence_path=test_evidence_path,
        min_cluster_size=2,
    )
    assert payload.summary.clusters == 0
    assert payload.summary.tests == 0


# gabion:evidence E:function_site::call_cluster_consolidation.py::gabion.analysis.call_cluster_consolidation.build_call_cluster_consolidation_payload
# gabion:behavior primary=desired
def test_call_cluster_consolidation_orders_plan_via_typed_execution_ops(
    write_test_evidence_payload,
    test_evidence_path: Path,
) -> None:
    cluster_one = [("alpha.py", "pkg.alpha")]
    cluster_two = [("beta.py", "pkg.beta")]
    write_test_evidence_payload(
        test_evidence_path,
        entries=[
            {
                "test_id": "tests/test_sample.py::test_b",
                "file": "tests/test_sample.py",
                "line": 20,
                "evidence": [
                    _call_footprint_display(
                        test_id="tests/test_sample.py::test_b",
                        file="tests/test_sample.py",
                        targets=cluster_one,
                    )
                ],
                "status": "mapped",
            },
            {
                "test_id": "tests/test_sample.py::test_a",
                "file": "tests/test_sample.py",
                "line": 10,
                "evidence": [
                    _call_footprint_display(
                        test_id="tests/test_sample.py::test_a",
                        file="tests/test_sample.py",
                        targets=cluster_one,
                    )
                ],
                "status": "mapped",
            },
            {
                "test_id": "tests/test_sample.py::test_c",
                "file": "tests/test_sample.py",
                "line": 30,
                "evidence": [
                    _call_footprint_display(
                        test_id="tests/test_sample.py::test_c",
                        file="tests/test_sample.py",
                        targets=cluster_one,
                    )
                ],
                "status": "mapped",
            },
            {
                "test_id": "tests/test_sample.py::test_y",
                "file": "tests/test_sample.py",
                "line": 50,
                "evidence": [
                    _call_footprint_display(
                        test_id="tests/test_sample.py::test_y",
                        file="tests/test_sample.py",
                        targets=cluster_two,
                    )
                ],
                "status": "mapped",
            },
            {
                "test_id": "tests/test_sample.py::test_x",
                "file": "tests/test_sample.py",
                "line": 40,
                "evidence": [
                    _call_footprint_display(
                        test_id="tests/test_sample.py::test_x",
                        file="tests/test_sample.py",
                        targets=cluster_two,
                    )
                ],
                "status": "mapped",
            },
        ],
    )

    payload = call_cluster_consolidation.build_call_cluster_consolidation_payload(
        evidence_path=test_evidence_path,
        min_cluster_size=2,
    )

    plan = payload.plan
    cluster_counts = {
        cluster.cluster.identity: len(cluster.tests)
        for cluster in payload.clusters
    }
    assert [cluster_counts[entry.cluster.identity] for entry in plan] == [3, 3, 3, 2, 2]
    assert [entry.test_id for entry in plan[:3]] == [
        "tests/test_sample.py::test_a",
        "tests/test_sample.py::test_b",
        "tests/test_sample.py::test_c",
    ]
    assert [entry.test_id for entry in plan[3:]] == [
        "tests/test_sample.py::test_x",
        "tests/test_sample.py::test_y",
    ]


# gabion:evidence E:function_site::server_core/command_orchestrator.py::gabion.server_core.command_orchestrator._emit_analysis_reports
# gabion:behavior primary=desired
def test_call_cluster_consolidation_emitted_payload_shape() -> None:
    payload = _payload()
    assert _emitted_payload(payload) == {
        "version": call_cluster_consolidation.CONSOLIDATION_VERSION,
        "summary": {
            "clusters": 0,
            "tests": 0,
            "replacements": 0,
            "min_cluster_size": 2,
        },
        "clusters": [],
        "plan": [],
        "generated_by_spec_id": "call_cluster_consolidation",
        "generated_by_spec": {
            "name": "call_cluster_consolidation",
            "spec_version": 1,
        },
    }


# gabion:evidence E:function_site::call_cluster_consolidation.py::gabion.analysis.call_cluster_consolidation.render_markdown
# gabion:behavior primary=desired
def test_call_cluster_consolidation_render_uses_payload_spec_metadata() -> None:
    payload = _payload(
        generated_by_spec_id="custom-spec-id",
        generated_by_spec={"name": "custom", "spec_version": 99},
    )
    markdown = call_cluster_consolidation.render_markdown(payload)
    assert "generated_by_spec_id: custom-spec-id" in markdown
    assert 'generated_by_spec: {"name":"custom","spec_version":99}' in markdown


# gabion:evidence E:function_site::call_cluster_consolidation.py::gabion.analysis.call_cluster_consolidation.write_call_cluster_consolidation
# gabion:behavior primary=desired
def test_call_cluster_consolidation_write_creates_file(tmp_path: Path) -> None:
    payload = _payload()
    output_path = tmp_path / "nested" / "call_cluster_consolidation.json"
    call_cluster_consolidation.write_call_cluster_consolidation(
        payload, output_path=output_path
    )
    assert output_path.exists()
    assert json.loads(output_path.read_text(encoding="utf-8")) == _emitted_payload(payload)


# gabion:evidence E:function_site::call_cluster_consolidation.py::gabion.analysis.call_cluster_consolidation.render_json_payload
# gabion:behavior primary=desired
def test_call_cluster_consolidation_emitted_payload_preserves_identity() -> None:
    payload = _payload(
        clusters=1,
        tests=2,
        replacements=2,
        cluster_entries=(
            call_cluster_consolidation.ClusterSummary(
                cluster=call_cluster_shared.ClusterIdentity(
                    identity="call-cluster-1",
                    key={"k": "call_cluster", "targets": ["pkg.mod:helper"]},
                    display="pkg.mod:helper",
                ),
                tests=("tests/test_mod.py::test_one", "tests/test_mod.py::test_two"),
            ),
        ),
        plan_entries=(
            call_cluster_consolidation.ConsolidationPlanEntry(
                cluster=call_cluster_shared.ClusterIdentity(
                    identity="call-cluster-1",
                    key={"k": "call_cluster", "targets": ["pkg.mod:helper"]},
                    display="pkg.mod:helper",
                ),
                test_id="tests/test_mod.py::test_one",
                file="tests/test_mod.py",
                line=10,
                replace=("E:call_footprint::pkg.mod::helper",),
            ),
        ),
    )
    wire_payload = call_cluster_consolidation.render_json_payload(payload)
    assert wire_payload["clusters"] == [
        {
            "identity": "call-cluster-1",
            "key": {"k": "call_cluster", "targets": ["pkg.mod:helper"]},
            "display": "pkg.mod:helper",
            "tests": [
                "tests/test_mod.py::test_one",
                "tests/test_mod.py::test_two",
            ],
            "count": 2,
        }
    ]

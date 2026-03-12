from __future__ import annotations

import json
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from collections.abc import Iterable, Mapping

from gabion.analysis.semantics import evidence_keys
from gabion.analysis.foundation.baseline_io import parse_spec_metadata
from gabion.analysis.surfaces import test_evidence_suggestions
from gabion.analysis.call_cluster.call_cluster_shared import (
    cluster_identity_from_key,
    render_cluster_heading,
    render_string_codeblock,
    sorted_unique_strings,
)
from gabion.analysis.projection.projection_exec import apply_execution_ops
from gabion.analysis.projection.projection_exec_plan import execution_ops_from_spec
from gabion.analysis.projection.projection_registry import (
    CALL_CLUSTER_SUMMARY_SPEC,
    spec_metadata_lines_from_payload,
    spec_metadata_payload,
)
from gabion.analysis.semantics.report_doc import ReportDoc
from gabion.analysis.foundation.timeout_context import check_deadline
from gabion.json_types import JSONValue
from gabion.invariants import decision_protocol, grade_boundary, never


CALL_CLUSTER_VERSION = 1


@decision_protocol
@cache
def _call_cluster_summary_execution_ops():
    return execution_ops_from_spec(CALL_CLUSTER_SUMMARY_SPEC)


@dataclass(frozen=True)
class CallClusterEntry:
    identity: str
    key: dict[str, JSONValue]
    display: str
    tests: tuple[str, ...]
    count: int


@dataclass(frozen=True)
class CallClustersSummary:
    clusters: int
    tests: int


@dataclass(frozen=True)
class CallClustersPayload:
    version: int
    summary: CallClustersSummary
    clusters: tuple[CallClusterEntry, ...]
    generated_by_spec_id: str
    generated_by_spec: dict[str, JSONValue]


@dataclass
class _CallClusterAccumulator:
    identity: str
    key: dict[str, JSONValue]
    display: str
    tests: list[str]


@grade_boundary(
    kind="semantic_carrier_adapter",
    name="call_clusters.build_call_clusters_payload",
)
def build_call_clusters_payload(
    paths: Iterable[Path],
    *,
    root: Path,
    evidence_path: Path,
    config: object = None,
) -> CallClustersPayload:
    # dataflow-bundle: evidence_path, paths, root, config
    check_deadline(allow_frame_fallback=True)
    entries = test_evidence_suggestions.load_test_evidence(str(evidence_path))
    footprints = test_evidence_suggestions.collect_call_footprints(
        entries,
        root=root,
        paths=paths,
        config=config,
    )
    clusters: dict[str, _CallClusterAccumulator] = {}
    for entry in entries:
        check_deadline()
        targets = footprints.get(entry.test_id)
        if not targets:
            continue
        metadata = cluster_identity_from_key(
            evidence_keys.make_call_cluster_key(targets=targets)
        )
        identity = metadata.identity
        cluster = clusters.get(identity)
        if cluster is None:
            cluster = _CallClusterAccumulator(
                identity=metadata.identity,
                key=dict(metadata.key),
                display=metadata.display,
                tests=[],
            )
            clusters[identity] = cluster
        cluster.tests.append(entry.test_id)

    cluster_rows: list[dict[str, JSONValue]] = []
    for cluster in clusters.values():
        check_deadline()
        tests = sorted_unique_strings(
            cluster.tests,
            source="build_call_clusters_payload.cluster.tests",
        )
        count = len(tests)
        cluster_rows.append(
            {
                "identity": cluster.identity,
                "display": cluster.display,
                "count": count,
            }
        )
        clusters[cluster.identity] = _CallClusterAccumulator(
            identity=cluster.identity,
            key=cluster.key,
            display=cluster.display,
            tests=list(tests),
        )

    projected = apply_execution_ops(
        _call_cluster_summary_execution_ops(),
        cluster_rows,
    )
    ordered: list[CallClusterEntry] = []
    for row in projected:
        check_deadline()
        match row["identity"]:
            case str() as identity if identity:
                cluster_identity = identity
            case impossible_identity:
                _ = impossible_identity
                never("projected call cluster row must carry string identity")
        cluster = clusters[cluster_identity]
        ordered.append(
            CallClusterEntry(
                identity=cluster_identity,
                key=cluster.key,
                display=cluster.display,
                tests=tuple(cluster.tests),
                count=len(cluster.tests),
            )
        )

    summary = CallClustersSummary(
        clusters=len(ordered),
        tests=len({test for entry in ordered for test in entry.tests}),
    )
    metadata = parse_spec_metadata(spec_metadata_payload(CALL_CLUSTER_SUMMARY_SPEC))
    return CallClustersPayload(
        version=CALL_CLUSTER_VERSION,
        summary=summary,
        clusters=tuple(ordered),
        generated_by_spec_id=metadata.spec_id,
        generated_by_spec=metadata.spec,
    )


@grade_boundary(
    kind="semantic_carrier_adapter",
    name="call_clusters.render_json_payload",
)
def render_json_payload(payload: CallClustersPayload) -> dict[str, JSONValue]:
    return {
        "version": payload.version,
        "summary": {
            "clusters": payload.summary.clusters,
            "tests": payload.summary.tests,
        },
        "clusters": [
            {
                "identity": entry.identity,
                "key": entry.key,
                "display": entry.display,
                "tests": list(entry.tests),
                "count": entry.count,
            }
            for entry in payload.clusters
        ],
        "generated_by_spec_id": payload.generated_by_spec_id,
        "generated_by_spec": payload.generated_by_spec,
    }


def render_markdown(
    payload: CallClustersPayload,
) -> str:
    with grade_boundary(
        kind="semantic_carrier_adapter",
        name="call_clusters.render_markdown",
    ):
        check_deadline(allow_frame_fallback=True)
        doc = ReportDoc("out_call_clusters")
        doc.lines(
            [
                f"generated_by_spec_id: {payload.generated_by_spec_id}",
                "generated_by_spec: "
                + json.dumps(
                    payload.generated_by_spec,
                    sort_keys=False,
                    separators=(",", ":"),
                ),
            ]
        )
        doc.section("Summary")
        doc.codeblock(
            json.dumps(
                {
                    "clusters": payload.summary.clusters,
                    "tests": payload.summary.tests,
                },
                indent=2,
                sort_keys=False,
            )
        )
        doc.line()
        if not payload.clusters:
            doc.line("No call clusters found.")
            return doc.emit()
        doc.section("Call clusters")
        for entry in payload.clusters:
            check_deadline()
            render_cluster_heading(doc, display=entry.display, count=entry.count)
            if entry.tests:
                render_string_codeblock(doc, entry.tests)
        return doc.emit()

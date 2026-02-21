from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

from gabion.analysis import evidence_keys, test_evidence_suggestions
from gabion.analysis.baseline_io import write_json
from gabion.analysis.call_cluster_shared import (
    cluster_identity_from_key,
    render_cluster_heading,
    render_string_codeblock,
    sorted_unique_strings,
)
from gabion.analysis.dataflow_audit import AuditConfig
from gabion.analysis.projection_exec import apply_spec
from gabion.analysis.projection_registry import (
    CALL_CLUSTER_SUMMARY_SPEC,
    spec_metadata_lines_from_payload,
    spec_metadata_payload,
)
from gabion.analysis.projection_spec import ProjectionSpec
from gabion.analysis.report_doc import ReportDoc
from gabion.analysis.timeout_context import check_deadline
from gabion.json_types import JSONValue

CALL_CLUSTER_VERSION = 1


@dataclass(frozen=True)
class CallClusterEntry:
    identity: str
    key: dict[str, object]
    display: str
    tests: tuple[str, ...]
    count: int


def build_call_clusters_payload(
    paths: Iterable[Path],
    *,
    root: Path,
    evidence_path: Path,
    config: AuditConfig | None = None,
    summary_spec: ProjectionSpec | None = None,
) -> dict[str, JSONValue]:
    # dataflow-bundle: evidence_path, paths, root, config
    check_deadline(allow_frame_fallback=True)
    entries = test_evidence_suggestions.load_test_evidence(str(evidence_path))
    footprints = test_evidence_suggestions.collect_call_footprints(
        entries,
        root=root,
        paths=paths,
        config=config,
    )
    clusters: dict[str, dict[str, object]] = {}
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
            cluster = {
                "identity": metadata.identity,
                "key": metadata.key,
                "display": metadata.display,
                "tests": [],
            }
            clusters[identity] = cluster
        cluster["tests"].append(entry.test_id)

    cluster_rows: list[dict[str, JSONValue]] = []
    for cluster in clusters.values():
        check_deadline()
        tests = sorted_unique_strings(
            cluster["tests"],
            source="build_call_clusters_payload.cluster.tests",
        )
        cluster["tests"] = tests
        count = len(tests)
        cluster["count"] = count
        cluster_rows.append(
            {
                "identity": cluster["identity"],
                "display": cluster["display"],
                "count": count,
            }
        )

    spec = summary_spec or CALL_CLUSTER_SUMMARY_SPEC
    projected = apply_spec(spec, cluster_rows)
    ordered: list[CallClusterEntry] = []
    for row in projected:
        check_deadline()
        identity = str(row.get("identity", "") or "")
        cluster = clusters.get(identity)
        if cluster is None:
            continue
        ordered.append(
            CallClusterEntry(
                identity=identity,
                key=cluster["key"],
                display=str(cluster["display"]),
                tests=tuple(cluster["tests"]),
                count=int(cluster["count"]),
            )
        )

    summary = {
        "clusters": len(ordered),
        "tests": len({test for entry in ordered for test in entry.tests}),
    }
    payload: dict[str, JSONValue] = {
        "version": CALL_CLUSTER_VERSION,
        "summary": summary,
        "clusters": [
            {
                "key": entry.key,
                "display": entry.display,
                "tests": list(entry.tests),
                "count": entry.count,
            }
            for entry in ordered
        ],
    }
    payload.update(spec_metadata_payload(spec))
    return payload


def render_markdown(
    payload: Mapping[str, JSONValue],
) -> str:
    check_deadline(allow_frame_fallback=True)
    summary = payload.get("summary", {})
    clusters = payload.get("clusters", [])
    doc = ReportDoc("out_call_clusters")
    doc.lines(spec_metadata_lines_from_payload(payload))
    doc.section("Summary")
    doc.codeblock(summary)
    doc.line()
    if not isinstance(clusters, list) or not clusters:
        doc.line("No call clusters found.")
        return doc.emit()
    doc.section("Call clusters")
    for entry in clusters:
        check_deadline()
        if not isinstance(entry, Mapping):
            continue
        display = str(entry.get("display", "") or "")
        count = entry.get("count", 0)
        render_cluster_heading(doc, display=display, count=count)
        tests = entry.get("tests", [])
        if isinstance(tests, list) and tests:
            render_string_codeblock(doc, tests)
    return doc.emit()


def write_call_clusters(
    payload: Mapping[str, JSONValue],
    *,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(output_path, payload)

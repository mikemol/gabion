from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

from gabion.analysis import evidence_keys, test_evidence_suggestions
from gabion.analysis.dataflow_audit import AuditConfig
from gabion.analysis.projection_exec import apply_spec
from gabion.analysis.projection_registry import (
    CALL_CLUSTER_SUMMARY_SPEC,
    spec_metadata_lines,
    spec_metadata_payload,
)
from gabion.analysis.projection_spec import ProjectionSpec
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
    entries = test_evidence_suggestions.load_test_evidence(str(evidence_path))
    footprints = test_evidence_suggestions.collect_call_footprints(
        entries,
        root=root,
        paths=paths,
        config=config,
    )
    clusters: dict[str, dict[str, object]] = {}
    for entry in entries:
        targets = footprints.get(entry.test_id)
        if not targets:
            continue
        key = evidence_keys.make_call_cluster_key(targets=targets)
        identity = evidence_keys.key_identity(key)
        cluster = clusters.get(identity)
        if cluster is None:
            normalized = evidence_keys.normalize_key(key)
            display = evidence_keys.render_display(normalized)
            cluster = {
                "identity": identity,
                "key": normalized,
                "display": display,
                "tests": [],
            }
            clusters[identity] = cluster
        cluster["tests"].append(entry.test_id)

    cluster_rows: list[dict[str, JSONValue]] = []
    for cluster in clusters.values():
        tests = sorted({str(test_id) for test_id in cluster["tests"]})
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


def render_markdown(payload: Mapping[str, JSONValue]) -> str:
    summary = payload.get("summary", {})
    clusters = payload.get("clusters", [])
    lines: list[str] = []
    lines.extend(spec_metadata_lines(CALL_CLUSTER_SUMMARY_SPEC))
    lines.append("Summary:")
    lines.append("```")
    lines.append(json.dumps(summary, sort_keys=True))
    lines.append("```")
    lines.append("")
    if not isinstance(clusters, list) or not clusters:
        lines.append("No call clusters found.")
        return "\n".join(lines)
    lines.append("Call clusters:")
    for entry in clusters:
        if not isinstance(entry, Mapping):
            continue
        display = str(entry.get("display", "") or "")
        count = entry.get("count", 0)
        lines.append("")
        lines.append(f"Cluster: {display} (count: {count})")
        tests = entry.get("tests", [])
        if isinstance(tests, list) and tests:
            lines.append("```")
            for test_id in tests:
                lines.append(str(test_id))
            lines.append("```")
    return "\n".join(lines)


def write_call_clusters(
    payload: Mapping[str, JSONValue],
    *,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

from gabion.analysis import evidence_keys, test_evidence_suggestions
from gabion.analysis.projection_exec import apply_spec
from gabion.analysis.projection_registry import (
    CALL_CLUSTER_CONSOLIDATION_SPEC,
    spec_metadata_lines,
    spec_metadata_payload,
)
from gabion.analysis.report_markdown import render_report_markdown
from gabion.analysis.timeout_context import check_deadline
from gabion.json_types import JSONValue

CONSOLIDATION_VERSION = 1


@dataclass(frozen=True)
class ConsolidationEntry:
    test_id: str
    file: str
    line: int
    replace: tuple[str, ...]
    cluster_identity: str
    cluster_display: str
    cluster_key: dict[str, object]


@dataclass(frozen=True)
class ClusterSummary:
    identity: str
    key: dict[str, object]
    display: str
    tests: tuple[str, ...]

    @property
    def count(self) -> int:
        return len(self.tests)


def build_call_cluster_consolidation_payload(
    *,
    evidence_path: Path,
    min_cluster_size: int = 2,
) -> dict[str, JSONValue]:
    # dataflow-bundle: evidence_path, min_cluster_size
    check_deadline(allow_frame_fallback=True)
    entries = test_evidence_suggestions.load_test_evidence(str(evidence_path))
    clusters: dict[str, dict[str, object]] = {}
    plan: list[ConsolidationEntry] = []

    for entry in entries:
        check_deadline()
        call_footprints: list[tuple[tuple[tuple[str, str], ...], str]] = []
        call_clusters: set[tuple[tuple[str, str], ...]] = set()
        for token in entry.evidence:
            check_deadline()
            key = evidence_keys.parse_display(token)
            if key is None:
                continue
            normalized = evidence_keys.normalize_key(key)
            kind = str(normalized.get("k", ""))
            targets = _targets_signature(normalized.get("targets"))
            if not targets:
                continue
            if kind == "call_footprint":
                call_footprints.append((targets, token))
            elif kind == "call_cluster":
                call_clusters.add(targets)
        if not call_footprints:
            continue
        target_sets = {targets for targets, _ in call_footprints}
        if len(target_sets) != 1:
            continue
        target_signature = next(iter(target_sets))
        if target_signature in call_clusters:
            continue
        cluster_key = evidence_keys.make_call_cluster_key(
            targets=_targets_payload(target_signature)
        )
        normalized_cluster_key = evidence_keys.normalize_key(cluster_key)
        cluster_identity = evidence_keys.key_identity(normalized_cluster_key)
        cluster_display = evidence_keys.render_display(normalized_cluster_key)
        cluster = clusters.get(cluster_identity)
        if cluster is None:
            cluster = {
                "identity": cluster_identity,
                "key": normalized_cluster_key,
                "display": cluster_display,
                "tests": set(),
            }
            clusters[cluster_identity] = cluster
        cluster["tests"].add(entry.test_id)
        replace_tokens = sorted({token for _, token in call_footprints})
        plan.append(
            ConsolidationEntry(
                test_id=entry.test_id,
                file=entry.file,
                line=entry.line,
                replace=tuple(replace_tokens),
                cluster_identity=cluster_identity,
                cluster_display=cluster_display,
                cluster_key=normalized_cluster_key,
            )
        )

    cluster_summaries: list[ClusterSummary] = []
    for cluster in clusters.values():
        check_deadline()
        tests = tuple(sorted({str(test_id) for test_id in cluster["tests"]}))
        cluster_summaries.append(
            ClusterSummary(
                identity=str(cluster["identity"]),
                key=cluster["key"],
                display=str(cluster["display"]),
                tests=tests,
            )
        )

    eligible = {
        summary.identity: summary
        for summary in cluster_summaries
        if summary.count >= min_cluster_size
    }

    relation: list[dict[str, JSONValue]] = []
    for entry in plan:
        check_deadline()
        summary = eligible.get(entry.cluster_identity)
        if summary is None:
            continue
        relation.append(
            {
                "cluster_identity": entry.cluster_identity,
                "cluster_display": entry.cluster_display,
                "cluster_count": summary.count,
                "test_id": entry.test_id,
                "file": entry.file,
                "line": entry.line,
                "replace": list(entry.replace),
                "with": {"key": entry.cluster_key, "display": entry.cluster_display},
            }
        )

    ordered_plan = apply_spec(CALL_CLUSTER_CONSOLIDATION_SPEC, relation)
    ordered_clusters = sorted(
        eligible.values(),
        key=lambda item: (-item.count, item.display, item.identity),
    )

    summary = {
        "clusters": len(ordered_clusters),
        "tests": len(ordered_plan),
        "replacements": sum(len(entry.get("replace", [])) for entry in ordered_plan),
        "min_cluster_size": min_cluster_size,
    }
    payload: dict[str, JSONValue] = {
        "version": CONSOLIDATION_VERSION,
        "summary": summary,
        "clusters": [
            {
                "identity": cluster.identity,
                "key": cluster.key,
                "display": cluster.display,
                "tests": list(cluster.tests),
                "count": cluster.count,
            }
            for cluster in ordered_clusters
        ],
        "plan": ordered_plan,
    }
    payload.update(spec_metadata_payload(CALL_CLUSTER_CONSOLIDATION_SPEC))
    return payload


def render_markdown(
    payload: Mapping[str, JSONValue],
) -> str:
    check_deadline(allow_frame_fallback=True)
    summary = payload.get("summary", {})
    clusters = payload.get("clusters", [])
    plan = payload.get("plan", [])
    lines: list[str] = []
    lines.extend(spec_metadata_lines(CALL_CLUSTER_CONSOLIDATION_SPEC))
    lines.append("Summary:")
    lines.append("```")
    lines.append(json.dumps(summary, sort_keys=True))
    lines.append("```")
    lines.append("")
    if not isinstance(plan, list) or not plan:
        lines.append("No consolidation candidates.")
        return render_report_markdown("out_call_cluster_consolidation", lines)
    cluster_index: dict[str, Mapping[str, JSONValue]] = {}
    if isinstance(clusters, list):
        for cluster in clusters:
            check_deadline()
            if not isinstance(cluster, Mapping):
                continue
            identity = str(cluster.get("identity", "") or "")
            if identity:
                cluster_index[identity] = cluster
    lines.append("Consolidation plan:")
    current_cluster = None
    for entry in plan:
        check_deadline()
        if not isinstance(entry, Mapping):
            continue
        identity = str(entry.get("cluster_identity", "") or "")
        if identity != current_cluster:
            current_cluster = identity
            cluster = cluster_index.get(identity, {})
            display = str(cluster.get("display", "") or entry.get("cluster_display", ""))
            count = cluster.get("count", entry.get("cluster_count", 0))
            lines.append("")
            lines.append(f"Cluster: {display} (count: {count})")
            lines.append("```")
        test_id = str(entry.get("test_id", "") or "")
        file_path = str(entry.get("file", "") or "")
        line = entry.get("line", 0)
        replace = entry.get("replace", [])
        if isinstance(replace, list):
            replace_tokens = ", ".join(str(item) for item in replace)
        else:
            replace_tokens = str(replace)
        with_entry = entry.get("with", {})
        with_display = ""
        if isinstance(with_entry, Mapping):
            with_display = str(with_entry.get("display", "") or "")
        lines.append(
            f"{test_id} ({file_path}:{line}) replace [{replace_tokens}] -> {with_display}"
        )
    if lines and lines[-1] != "```":
        lines.append("```")
    return render_report_markdown("out_call_cluster_consolidation", lines)


def write_call_cluster_consolidation(
    payload: Mapping[str, JSONValue],
    *,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _targets_signature(value: object) -> tuple[tuple[str, str], ...]:
    check_deadline()
    if not isinstance(value, Iterable):
        return ()
    pairs: list[tuple[str, str]] = []
    for item in value:
        check_deadline()
        if not isinstance(item, Mapping):
            continue
        path = str(item.get("path", "") or "").strip()
        qual = str(item.get("qual", "") or "").strip()
        if not path or not qual:
            continue
        pairs.append((path, qual))
    return tuple(pairs)


def _targets_payload(targets: Iterable[tuple[str, str]]) -> list[dict[str, str]]:
    return [{"path": path, "qual": qual} for path, qual in targets]

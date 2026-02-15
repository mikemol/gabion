from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

from gabion.analysis import evidence_keys, test_evidence_suggestions
from gabion.analysis.baseline_io import write_json
from gabion.analysis.projection_exec import apply_spec
from gabion.analysis.projection_registry import (
    CALL_CLUSTER_CONSOLIDATION_SPEC,
    spec_metadata_lines_from_payload,
    spec_metadata_payload,
)
from gabion.analysis.report_doc import ReportDoc
from gabion.analysis.timeout_context import check_deadline
from gabion.json_types import JSONValue
from gabion.order_contract import ordered_or_sorted

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
        replace_tokens = ordered_or_sorted(
            {token for _, token in call_footprints},
            source="build_call_cluster_consolidation_payload.replace_tokens",
        )
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
        tests = tuple(
            ordered_or_sorted(
                {str(test_id) for test_id in cluster["tests"]},
                source="build_call_cluster_consolidation_payload.cluster.tests",
            )
        )
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
    ordered_clusters = ordered_or_sorted(
        eligible.values(),
        source="build_call_cluster_consolidation_payload.ordered_clusters",
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
    doc = ReportDoc("out_call_cluster_consolidation")
    doc.lines(spec_metadata_lines_from_payload(payload))
    doc.section("Summary")
    doc.codeblock(summary)
    doc.line()
    if not isinstance(plan, list) or not plan:
        doc.line("No consolidation candidates.")
        return doc.emit()
    cluster_index: dict[str, Mapping[str, JSONValue]] = {}
    if isinstance(clusters, list):
        for cluster in clusters:
            check_deadline()
            if not isinstance(cluster, Mapping):
                continue
            identity = str(cluster.get("identity", "") or "")
            if identity:
                cluster_index[identity] = cluster
    doc.line("Consolidation plan:")
    current_cluster = None
    current_lines: list[str] = []

    def _flush_current_lines() -> None:
        nonlocal current_lines
        if not current_lines:
            return
        doc.codeblock("\n".join(current_lines))
        current_lines = []

    for entry in plan:
        check_deadline()
        if not isinstance(entry, Mapping):
            continue
        identity = str(entry.get("cluster_identity", "") or "")
        if identity != current_cluster:
            _flush_current_lines()
            current_cluster = identity
            cluster = cluster_index.get(identity, {})
            display = str(cluster.get("display", "") or entry.get("cluster_display", ""))
            count = cluster.get("count", entry.get("cluster_count", 0))
            doc.line()
            doc.line(f"Cluster: {display} (count: {count})")
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
        current_lines.append(
            f"{test_id} ({file_path}:{line}) replace [{replace_tokens}] -> {with_display}"
        )
    _flush_current_lines()
    return doc.emit()


def write_call_cluster_consolidation(
    payload: Mapping[str, JSONValue],
    *,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(output_path, payload)


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

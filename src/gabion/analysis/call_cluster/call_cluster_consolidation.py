from __future__ import annotations

import json
from dataclasses import dataclass
from functools import cache
from pathlib import Path

from gabion.analysis.semantics import evidence_keys
from gabion.analysis.surfaces import test_evidence_suggestions
from gabion.analysis.foundation.baseline_io import parse_spec_metadata, write_json
from gabion.analysis.foundation.resume_codec import sequence_optional
from gabion.analysis.call_cluster.call_cluster_shared import (
    cluster_identity_from_key,
    render_cluster_heading,
    sorted_unique_strings,
)
from gabion.analysis.projection.projection_exec import apply_execution_ops
from gabion.analysis.projection.projection_exec_plan import execution_ops_from_spec
from gabion.analysis.projection.projection_registry import (
    CALL_CLUSTER_CONSOLIDATION_SPEC,
    spec_metadata_payload,
)
from gabion.analysis.semantics.report_doc import ReportDoc
from gabion.analysis.foundation.timeout_context import check_deadline
from gabion.json_types import JSONValue
from gabion.order_contract import sort_once
from gabion.invariants import decision_protocol, grade_boundary, never

CONSOLIDATION_VERSION = 1


@decision_protocol
@cache
def _call_cluster_consolidation_execution_ops():
    return execution_ops_from_spec(CALL_CLUSTER_CONSOLIDATION_SPEC)


@dataclass(frozen=True)
class ConsolidationEntry:
    test_id: str
    file: str
    line: int
    replace: tuple[str, ...]
    cluster_identity: str
    cluster_display: str
    cluster_key: dict[str, JSONValue]


@dataclass(frozen=True)
class ClusterSummary:
    identity: str
    key: dict[str, JSONValue]
    display: str
    tests: tuple[str, ...]

    @property
    def count(self) -> int:
        return len(self.tests)


@dataclass(frozen=True)
class ConsolidationSummary:
    clusters: int
    tests: int
    replacements: int
    min_cluster_size: int


@dataclass(frozen=True)
class ConsolidationPlanEntry:
    cluster_identity: str
    cluster_display: str
    cluster_count: int
    test_id: str
    file: str
    line: int
    replace: tuple[str, ...]
    replacement_key: dict[str, JSONValue]
    replacement_display: str


@dataclass(frozen=True)
class CallClusterConsolidationPayload:
    version: int
    summary: ConsolidationSummary
    clusters: tuple[ClusterSummary, ...]
    plan: tuple[ConsolidationPlanEntry, ...]
    generated_by_spec_id: str
    generated_by_spec: dict[str, JSONValue]


@dataclass
class _ClusterAccumulator:
    identity: str
    key: dict[str, JSONValue]
    display: str
    tests: list[str]


@grade_boundary(
    kind="semantic_carrier_adapter",
    name="call_cluster_consolidation.build_call_cluster_consolidation_payload",
)
def build_call_cluster_consolidation_payload(
    *,
    evidence_path: Path,
    min_cluster_size: int = 2,
) -> CallClusterConsolidationPayload:
    # dataflow-bundle: evidence_path, min_cluster_size
    check_deadline(allow_frame_fallback=True)
    entries = test_evidence_suggestions.load_test_evidence(str(evidence_path))
    clusters: dict[str, _ClusterAccumulator] = {}
    plan: list[ConsolidationEntry] = []

    for entry in entries:
        check_deadline()
        call_footprints: list[tuple[tuple[tuple[str, str], ...], str]] = []
        call_clusters: set[tuple[tuple[str, str], ...]] = set()
        for token in entry.evidence:
            check_deadline()
            key = evidence_keys.parse_display(token)
            if key is not None:
                normalized = evidence_keys.normalize_key(key)
                kind = str(normalized.get("k", ""))
                raw_targets = sequence_optional(normalized.get("targets"))
                targets = ()
                if raw_targets is not None:
                    targets = tuple(
                        (target["path"], target["qual"])
                        for target in evidence_keys.normalize_targets(raw_targets)
                    )
                if targets:
                    if kind == "call_footprint":
                        call_footprints.append((targets, token))
                    elif kind == "call_cluster":  # pragma: no branch
                        call_clusters.add(targets)
        if not call_footprints:
            continue
        target_sets = {targets for targets, _ in call_footprints}
        if len(target_sets) != 1:
            continue
        target_signature = next(iter(target_sets))
        if target_signature in call_clusters:
            continue
        metadata = cluster_identity_from_key(
            evidence_keys.make_call_cluster_key(
                targets=target_signature
            )
        )
        cluster_identity = metadata.identity
        cluster_display = metadata.display
        cluster = clusters.get(cluster_identity)
        if cluster is None:
            cluster = _ClusterAccumulator(
                identity=cluster_identity,
                key=dict(metadata.key),
                display=cluster_display,
                tests=[],
            )
            clusters[cluster_identity] = cluster
        cluster.tests.append(entry.test_id)
        replace_tokens = sort_once(
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
                cluster_key=dict(metadata.key),
            )
        )

    cluster_summaries: list[ClusterSummary] = []
    for cluster in clusters.values():
        check_deadline()
        tests = sorted_unique_strings(
            cluster.tests,
            source="build_call_cluster_consolidation_payload.cluster.tests",
        )
        cluster_summaries.append(
            ClusterSummary(
                identity=cluster.identity,
                key=cluster.key,
                display=cluster.display,
                tests=tests,
            )
        )

    eligible = {
        summary.identity: summary
        for summary in cluster_summaries
        if summary.count >= min_cluster_size
    }

    relation: list[dict[str, JSONValue]] = []
    plan_by_identity: dict[tuple[str, str], ConsolidationEntry] = {}
    for entry in plan:
        check_deadline()
        summary = eligible.get(entry.cluster_identity)
        if summary is not None:
            plan_by_identity[(entry.cluster_identity, entry.test_id)] = entry
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

    ordered_plan_rows = apply_execution_ops(
        _call_cluster_consolidation_execution_ops(),
        relation,
    )
    ordered_clusters = sort_once(
        eligible.values(),
        source="build_call_cluster_consolidation_payload.ordered_clusters",
        key=lambda item: (-item.count, item.display, item.identity),
    )

    ordered_plan: list[ConsolidationPlanEntry] = []
    for row in ordered_plan_rows:
        check_deadline()
        match row.get("cluster_identity"):
            case str() as cluster_identity if cluster_identity:
                pass
            case impossible_cluster_identity:
                _ = impossible_cluster_identity
                never(
                    "projected consolidation row must carry string cluster identity"
                )
        match row.get("test_id"):
            case str() as test_id if test_id:
                pass
            case impossible_test_id:
                _ = impossible_test_id
                never("projected consolidation row must carry string test id")
        match row.get("cluster_count"):
            case int() as cluster_count:
                pass
            case count if count is not None:
                try:
                    cluster_count = int(count)
                except (TypeError, ValueError):
                    _ = count
                    never("projected consolidation row must carry numeric cluster count")
            case impossible_cluster_count:
                _ = impossible_cluster_count
                never("projected consolidation row must carry cluster count")
        source_entry = plan_by_identity[(cluster_identity, test_id)]
        ordered_plan.append(
            ConsolidationPlanEntry(
                cluster_identity=cluster_identity,
                cluster_display=source_entry.cluster_display,
                cluster_count=cluster_count,
                test_id=test_id,
                file=source_entry.file,
                line=source_entry.line,
                replace=source_entry.replace,
                replacement_key=source_entry.cluster_key,
                replacement_display=source_entry.cluster_display,
            )
        )

    metadata = parse_spec_metadata(
        spec_metadata_payload(CALL_CLUSTER_CONSOLIDATION_SPEC)
    )
    return CallClusterConsolidationPayload(
        version=CONSOLIDATION_VERSION,
        summary=ConsolidationSummary(
            clusters=len(ordered_clusters),
            tests=len(ordered_plan),
            replacements=sum(len(entry.replace) for entry in ordered_plan),
            min_cluster_size=min_cluster_size,
        ),
        clusters=tuple(ordered_clusters),
        plan=tuple(ordered_plan),
        generated_by_spec_id=metadata.spec_id,
        generated_by_spec=metadata.spec,
    )


@grade_boundary(
    kind="semantic_carrier_adapter",
    name="call_cluster_consolidation.render_json_payload",
)
def render_json_payload(
    payload: CallClusterConsolidationPayload,
) -> dict[str, JSONValue]:
    return {
        "version": payload.version,
        "summary": {
            "clusters": payload.summary.clusters,
            "tests": payload.summary.tests,
            "replacements": payload.summary.replacements,
            "min_cluster_size": payload.summary.min_cluster_size,
        },
        "clusters": [
            {
                "identity": cluster.identity,
                "key": cluster.key,
                "display": cluster.display,
                "tests": list(cluster.tests),
                "count": cluster.count,
            }
            for cluster in payload.clusters
        ],
        "plan": [
            {
                "cluster_identity": entry.cluster_identity,
                "cluster_display": entry.cluster_display,
                "cluster_count": entry.cluster_count,
                "test_id": entry.test_id,
                "file": entry.file,
                "line": entry.line,
                "replace": list(entry.replace),
                "with": {
                    "key": entry.replacement_key,
                    "display": entry.replacement_display,
                },
            }
            for entry in payload.plan
        ],
        "generated_by_spec_id": payload.generated_by_spec_id,
        "generated_by_spec": payload.generated_by_spec,
    }


def render_markdown(
    payload: CallClusterConsolidationPayload,
) -> str:
    with grade_boundary(
        kind="semantic_carrier_adapter",
        name="call_cluster_consolidation.render_markdown",
    ):
        check_deadline(allow_frame_fallback=True)
        doc = ReportDoc("out_call_cluster_consolidation")
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
                    "replacements": payload.summary.replacements,
                    "min_cluster_size": payload.summary.min_cluster_size,
                },
                indent=2,
                sort_keys=False,
            )
        )
        doc.line()
        if not payload.plan:
            doc.line("No consolidation candidates.")
            return doc.emit()

        cluster_index = {cluster.identity: cluster for cluster in payload.clusters}
        doc.line("Consolidation plan:")
        current_cluster = ""
        current_lines: list[str] = []

        for entry in payload.plan:
            check_deadline()
            if entry.cluster_identity != current_cluster:
                if current_lines:
                    doc.codeblock("\n".join(current_lines))
                    current_lines = []
                current_cluster = entry.cluster_identity
                if entry.cluster_identity not in cluster_index:
                    never("consolidation plan identity must exist in cluster summary")
                cluster = cluster_index[entry.cluster_identity]
                render_cluster_heading(doc, display=cluster.display, count=cluster.count)
            replace_tokens = ", ".join(entry.replace)
            current_lines.append(
                f"{entry.test_id} ({entry.file}:{entry.line}) replace "
                f"[{replace_tokens}] -> {entry.replacement_display}"
            )
        if current_lines:
            doc.codeblock("\n".join(current_lines))
        return doc.emit()


def write_call_cluster_consolidation(
    payload: CallClusterConsolidationPayload,
    *,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(output_path, render_json_payload(payload))

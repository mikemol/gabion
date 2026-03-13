from __future__ import annotations

import argparse
import json
from pathlib import Path

from gabion.order_contract import ordered_or_sorted
from gabion.tooling.policy_substrate.invariant_graph import (
    InvariantGraph,
    blocker_chains,
    build_invariant_graph,
    build_invariant_workstreams,
    compare_invariant_workstreams,
    load_invariant_graph,
    load_invariant_workstreams,
    trace_nodes,
    write_invariant_graph,
    write_invariant_workstreams,
)

_DEFAULT_ARTIFACT = Path("artifacts/out/invariant_graph.json")
_DEFAULT_WORKSTREAMS_ARTIFACT = Path("artifacts/out/invariant_workstreams.json")


def _sorted[T](values: list[T], *, key=None) -> list[T]:
    return ordered_or_sorted(
        values,
        source="tooling.runtime.invariant_graph",
        key=key,
    )


def _load_or_build_graph(*, root: Path, artifact: Path) -> InvariantGraph:
    if artifact.exists():
        return load_invariant_graph(artifact)
    return build_invariant_graph(root)


def _descendant_ids(graph: InvariantGraph, node_id: str) -> tuple[str, ...]:
    node_by_id = graph.node_by_id()
    edges_from = graph.edges_from()
    pending = [node_id]
    seen: set[str] = set()
    while pending:
        current = pending.pop()
        if current in seen:
            continue
        seen.add(current)
        for edge in edges_from.get(current, ()):
            if edge.edge_kind == "contains" and edge.target_id in node_by_id:
                pending.append(edge.target_id)
    return tuple(_sorted(list(seen)))


def _coverage_and_signal_ids(
    graph: InvariantGraph,
    *,
    node_ids: tuple[str, ...],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    node_by_id = graph.node_by_id()
    edges_from = graph.edges_from()
    edges_to = graph.edges_to()
    test_case_ids = {
        edge.target_id
        for node_id in node_ids
        for edge in edges_from.get(node_id, ())
        if edge.edge_kind == "covered_by"
        and edge.target_id in node_by_id
        and node_by_id[edge.target_id].node_kind == "test_case"
    }
    signal_ids = {
        edge.source_id
        for node_id in node_ids
        for edge in edges_to.get(node_id, ())
        if edge.edge_kind == "blocks"
        and edge.source_id in node_by_id
        and node_by_id[edge.source_id].node_kind == "policy_signal"
    }
    return (tuple(_sorted(list(test_case_ids))), tuple(_sorted(list(signal_ids))))


def _load_impacted_tests(path: Path | None) -> tuple[str, ...]:
    if path is None or not path.exists():
        return ()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return ()
    selection = payload.get("selection")
    if not isinstance(selection, dict):
        return ()
    impacted_tests = selection.get("impacted_tests", [])
    if not isinstance(impacted_tests, list):
        return ()
    return tuple(_sorted([str(item) for item in impacted_tests if isinstance(item, str)]))


def _workstream_by_object_id(
    *,
    graph: InvariantGraph,
    object_id: str,
):
    projection = build_invariant_workstreams(graph)
    for item in projection.iter_workstreams():
        if item.object_id.wire() == object_id:
            return item
    return None


def _print_summary(*, graph: InvariantGraph) -> None:
    payload = graph.as_payload()
    counts = payload.get("counts", {})
    print(f"root: {graph.root}")
    print(f"workstreams: {len(graph.workstream_root_ids)}")
    print(f"nodes: {counts.get('node_count', 0)}")
    print(f"edges: {counts.get('edge_count', 0)}")
    print(f"diagnostics: {counts.get('diagnostic_count', 0)}")
    node_kind_counts = counts.get("node_kind_counts", {})
    if isinstance(node_kind_counts, dict):
        print("node kinds:")
        for key, value in _sorted(
            [(str(key), int(value)) for key, value in node_kind_counts.items()],
            key=lambda item: item[0],
        ):
            print(f"- {key}: {value}")
    edge_kind_counts = counts.get("edge_kind_counts", {})
    if isinstance(edge_kind_counts, dict):
        print("edge kinds:")
        for key, value in _sorted(
            [(str(key), int(value)) for key, value in edge_kind_counts.items()],
            key=lambda item: item[0],
        ):
            print(f"- {key}: {value}")


def _ownership_chain(graph: InvariantGraph, node_id: str) -> tuple[str, ...]:
    node_by_id = graph.node_by_id()
    edges_to = graph.edges_to()
    chain: list[str] = []
    current = node_id
    seen: set[str] = set()
    while current not in seen:
        seen.add(current)
        chain.append(current)
        parents = [
            edge.source_id
            for edge in edges_to.get(current, ())
            if edge.edge_kind == "contains" and edge.source_id in node_by_id
        ]
        if not parents:
            break
        current = _sorted(parents)[0]
    return tuple(reversed(chain))


def _print_trace(*, graph: InvariantGraph, raw_id: str) -> int:
    nodes = trace_nodes(graph, raw_id)
    if not nodes:
        print(f"no invariant-graph nodes matched: {raw_id}")
        return 1
    node_by_id = graph.node_by_id()
    edges_from = graph.edges_from()
    edges_to = graph.edges_to()
    for index, node in enumerate(nodes, start=1):
        if index > 1:
            print("")
        print(f"node: {node.node_id}")
        print(f"kind: {node.node_kind}")
        print(f"title: {node.title}")
        if node.marker_id:
            print(f"marker_id: {node.marker_id}")
        if node.object_ids:
            print(f"object_ids: {', '.join(node.object_ids)}")
        if node.doc_ids:
            print(f"doc_ids: {', '.join(node.doc_ids)}")
        if node.policy_ids:
            print(f"policy_ids: {', '.join(node.policy_ids)}")
        if node.invariant_ids:
            print(f"invariant_ids: {', '.join(node.invariant_ids)}")
        print(f"site_identity: {node.site_identity}")
        print(f"structural_identity: {node.structural_identity}")
        if node.rel_path:
            print(
                f"location: {node.rel_path}:{node.line}:{node.column} "
                f"({node.qualname or '<none>'})"
            )
        if node.reasoning_control:
            print(f"reasoning.control: {node.reasoning_control}")
        if node.blocking_dependencies:
            print(
                "blocking_dependencies: "
                + ", ".join(node.blocking_dependencies)
            )
        chain = _ownership_chain(graph, node.node_id)
        if chain:
            print("ownership_chain:")
            for value in chain:
                owner = node_by_id.get(value)
                print(f"- {value} :: {owner.title if owner is not None else '<missing>'}")
        outbound = [
            edge for edge in edges_from.get(node.node_id, ()) if edge.target_id in node_by_id
        ]
        inbound = [
            edge for edge in edges_to.get(node.node_id, ()) if edge.source_id in node_by_id
        ]
        print("outbound_edges:")
        for edge in outbound:
            target = node_by_id[edge.target_id]
            print(f"- {edge.edge_kind}: {target.node_id} :: {target.title}")
        if not outbound:
            print("- <none>")
        print("inbound_edges:")
        for edge in inbound:
            source = node_by_id[edge.source_id]
            print(f"- {edge.edge_kind}: {source.node_id} :: {source.title}")
        if not inbound:
            print("- <none>")
        descendant_ids = _descendant_ids(graph, node.node_id)
        coverage_ids, signal_ids = _coverage_and_signal_ids(graph, node_ids=descendant_ids)
        print(f"coverage_summary: tests={len(coverage_ids)}")
        if coverage_ids:
            print("covering_tests:")
            for test_case_id in coverage_ids[:20]:
                print(f"- {node_by_id[test_case_id].title}")
        print(f"policy_signal_summary: signals={len(signal_ids)}")
        if signal_ids:
            print("policy_signals:")
            for signal_id in signal_ids[:20]:
                signal_node = node_by_id[signal_id]
                print(f"- {signal_node.title} :: {signal_node.reasoning_summary}")
    return 0


def _print_blockers(*, graph: InvariantGraph, object_id: str) -> int:
    grouped = blocker_chains(graph, object_id=object_id)
    if not grouped:
        print(f"no blocker chains for object_id: {object_id}")
        return 1
    node_by_id = graph.node_by_id()
    for seam_class, chains in _sorted(list(grouped.items()), key=lambda item: item[0]):
        print(f"{seam_class}:")
        for chain in chains:
            rendered = " -> ".join(
                f"{node_id} ({node_by_id[node_id].title})"
                for node_id in chain
                if node_id in node_by_id
            )
            print(f"- {rendered}")
    return 0


def _print_workstream(*, graph: InvariantGraph, object_id: str) -> int:
    workstream = _workstream_by_object_id(graph=graph, object_id=object_id)
    if workstream is None:
        print(f"no workstream projection for object_id: {object_id}")
        return 1
    health_summary = workstream.health_summary()
    print(f"object_id: {workstream.object_id.wire()}")
    print(f"title: {workstream.title}")
    print(f"status: {workstream.status}")
    print(f"touchsites: {workstream.touchsite_count}")
    print(f"collapsible_touchsites: {workstream.collapsible_touchsite_count}")
    print(f"surviving_touchsites: {workstream.surviving_touchsite_count}")
    print(f"policy_signals: {workstream.policy_signal_count}")
    print(f"coverage_count: {workstream.coverage_count}")
    print(f"diagnostics: {workstream.diagnostic_count}")
    print(
        "health_summary: covered={covered} :: uncovered={uncovered} :: governed={governed} :: diagnosed={diagnosed}".format(
            covered=health_summary.covered_touchsite_count,
            uncovered=health_summary.uncovered_touchsite_count,
            governed=health_summary.governed_touchsite_count,
            diagnosed=health_summary.diagnosed_touchsite_count,
        )
    )
    print(
        "touchsite_blockers: ready={ready} :: coverage_gap={coverage_gap} :: policy={policy} :: diagnostic={diagnostic}".format(
            ready=health_summary.ready_touchsite_count,
            coverage_gap=health_summary.coverage_gap_touchsite_count,
            policy=health_summary.policy_blocked_touchsite_count,
            diagnostic=health_summary.diagnostic_blocked_touchsite_count,
        )
    )
    print(
        "health_cuts: touchpoints(ready={tp_ready}, coverage_gap={tp_gap}, policy={tp_policy}, diagnostic={tp_diag}) :: "
        "subqueues(ready={sq_ready}, coverage_gap={sq_gap}, policy={sq_policy}, diagnostic={sq_diag})".format(
            tp_ready=health_summary.ready_touchpoint_cut_count,
            tp_gap=health_summary.coverage_gap_touchpoint_cut_count,
            tp_policy=health_summary.policy_blocked_touchpoint_cut_count,
            tp_diag=health_summary.diagnostic_blocked_touchpoint_cut_count,
            sq_ready=health_summary.ready_subqueue_cut_count,
            sq_gap=health_summary.coverage_gap_subqueue_cut_count,
            sq_policy=health_summary.policy_blocked_subqueue_cut_count,
            sq_diag=health_summary.diagnostic_blocked_subqueue_cut_count,
        )
    )
    print(f"dominant_blocker_class: {workstream.dominant_blocker_class()}")
    print(f"recommended_remediation_family: {workstream.recommended_remediation_family()}")
    recommended_cut = workstream.recommended_cut()
    recommended_ready_cut = workstream.recommended_ready_cut()
    recommended_coverage_gap_cut = workstream.recommended_coverage_gap_cut()
    recommended_policy_blocked_cut = workstream.recommended_policy_blocked_cut()
    recommended_diagnostic_blocked_cut = workstream.recommended_diagnostic_blocked_cut()
    if recommended_cut is None:
        print("recommended_cut: <none>")
    else:
        print(
            "recommended_cut: {cut_kind} :: {object_id} :: touchsites={touchsites} :: surviving={surviving}".format(
                cut_kind=recommended_cut.cut_kind,
                object_id=recommended_cut.object_id.wire(),
                touchsites=recommended_cut.touchsite_count,
                surviving=recommended_cut.surviving_touchsite_count,
            )
        )
    if recommended_ready_cut is None:
        print("recommended_ready_cut: <none>")
    else:
        print(
            "recommended_ready_cut: {cut_kind} :: {object_id} :: touchsites={touchsites} :: uncovered={uncovered}".format(
                cut_kind=recommended_ready_cut.cut_kind,
                object_id=recommended_ready_cut.object_id.wire(),
                touchsites=recommended_ready_cut.touchsite_count,
                uncovered=recommended_ready_cut.uncovered_touchsite_count,
            )
        )
    if recommended_coverage_gap_cut is None:
        print("recommended_coverage_gap_cut: <none>")
    else:
        print(
            "recommended_coverage_gap_cut: {cut_kind} :: {object_id} :: touchsites={touchsites} :: uncovered={uncovered}".format(
                cut_kind=recommended_coverage_gap_cut.cut_kind,
                object_id=recommended_coverage_gap_cut.object_id.wire(),
                touchsites=recommended_coverage_gap_cut.touchsite_count,
                uncovered=recommended_coverage_gap_cut.uncovered_touchsite_count,
            )
        )
    if recommended_policy_blocked_cut is None:
        print("recommended_policy_blocked_cut: <none>")
    else:
        print(
            "recommended_policy_blocked_cut: {cut_kind} :: {object_id} :: touchsites={touchsites} :: signals={signals}".format(
                cut_kind=recommended_policy_blocked_cut.cut_kind,
                object_id=recommended_policy_blocked_cut.object_id.wire(),
                touchsites=recommended_policy_blocked_cut.touchsite_count,
                signals=recommended_policy_blocked_cut.policy_signal_count,
            )
        )
    if recommended_diagnostic_blocked_cut is None:
        print("recommended_diagnostic_blocked_cut: <none>")
    else:
        print(
            "recommended_diagnostic_blocked_cut: {cut_kind} :: {object_id} :: touchsites={touchsites} :: diagnostics={diagnostics}".format(
                cut_kind=recommended_diagnostic_blocked_cut.cut_kind,
                object_id=recommended_diagnostic_blocked_cut.object_id.wire(),
                touchsites=recommended_diagnostic_blocked_cut.touchsite_count,
                diagnostics=recommended_diagnostic_blocked_cut.diagnostic_count,
            )
        )
    print("remediation_lanes:")
    remediation_lanes = workstream.remediation_lanes()
    if not remediation_lanes:
        print("- <none>")
    else:
        for lane in remediation_lanes:
            best_cut = lane.best_cut
            best_cut_rendered = (
                "<none>"
                if best_cut is None
                else "{cut_kind}::{object_id}".format(
                    cut_kind=best_cut.cut_kind,
                    object_id=best_cut.object_id.wire(),
                )
            )
            print(
                "- {family} :: blocker={blocker} :: touchsites={touchsites} :: touchpoints={touchpoints} :: subqueues={subqueues} :: best={best}".format(
                    family=lane.remediation_family,
                    blocker=lane.blocker_class,
                    touchsites=lane.touchsite_count,
                    touchpoints=lane.touchpoint_cut_count,
                    subqueues=lane.subqueue_cut_count,
                    best=best_cut_rendered,
                )
            )
    print("ranked_touchpoint_cuts:")
    touchpoint_cuts = workstream.ranked_touchpoint_cuts()
    if not touchpoint_cuts:
        print("- <none>")
    else:
        for item in touchpoint_cuts:
            print(
                "- {object_id} :: readiness={readiness} :: touchsites={touchsites} :: collapsible={collapsible} :: surviving={surviving} :: uncovered={uncovered}".format(
                    object_id=item.object_id.wire(),
                    readiness=item.readiness_class,
                    touchsites=item.touchsite_count,
                    collapsible=item.collapsible_touchsite_count,
                    surviving=item.surviving_touchsite_count,
                    uncovered=item.uncovered_touchsite_count,
                )
            )
    print("ranked_subqueue_cuts:")
    subqueue_cuts = workstream.ranked_subqueue_cuts()
    if not subqueue_cuts:
        print("- <none>")
    else:
        for item in subqueue_cuts:
            print(
                "- {object_id} :: readiness={readiness} :: touchsites={touchsites} :: collapsible={collapsible} :: surviving={surviving} :: uncovered={uncovered}".format(
                    object_id=item.object_id.wire(),
                    readiness=item.readiness_class,
                    touchsites=item.touchsite_count,
                    collapsible=item.collapsible_touchsite_count,
                    surviving=item.surviving_touchsite_count,
                    uncovered=item.uncovered_touchsite_count,
                )
            )
    print("subqueues:")
    subqueues = tuple(workstream.iter_subqueues())
    if not subqueues:
        print("- <none>")
    else:
        for item in subqueues:
            print(
                "- {object_id} :: {status} :: touchsites={touchsites} :: signals={signals} :: coverage={coverage}".format(
                    object_id=item.object_id.wire(),
                    status=item.status,
                    touchsites=item.touchsite_count,
                    signals=item.policy_signal_count,
                    coverage=item.coverage_count,
                )
            )
    return 0


def _print_blast_radius(
    *,
    graph: InvariantGraph,
    raw_id: str,
    impact_artifact: Path | None,
) -> int:
    node_by_id = graph.node_by_id()
    nodes = trace_nodes(graph, raw_id)
    if not nodes:
        print(f"no invariant-graph nodes matched: {raw_id}")
        return 1
    descendant_ids = tuple(
        _sorted(
            list(
                {
                    descendant_id
                    for node in nodes
                    for descendant_id in _descendant_ids(graph, node.node_id)
                }
            )
        )
    )
    coverage_ids, _signal_ids = _coverage_and_signal_ids(graph, node_ids=descendant_ids)
    impacted_tests = set(_load_impacted_tests(impact_artifact))
    print(f"node_matches: {len(nodes)}")
    print(f"covering_tests: {len(coverage_ids)}")
    if impact_artifact is not None:
        print(f"impact_artifact: {impact_artifact}")
    for test_case_id in coverage_ids:
        test_id = node_by_id[test_case_id].title
        suffix = " [impacted]" if test_id in impacted_tests else ""
        print(f"- {test_id}{suffix}")
    return 0


def _print_compare(
    *,
    before_workstreams_artifact: Path,
    after_workstreams_artifact: Path,
    object_id: str | None,
) -> int:
    before_payload = load_invariant_workstreams(before_workstreams_artifact)
    after_payload = load_invariant_workstreams(after_workstreams_artifact)
    drifts = compare_invariant_workstreams(before_payload, after_payload)
    if object_id is not None:
        drifts = tuple(item for item in drifts if item.object_id == object_id)
        if not drifts:
            print(f"no workstream drift for object_id: {object_id}")
            return 1
    print(f"before: {before_workstreams_artifact}")
    print(f"after: {after_workstreams_artifact}")
    classification_counts: dict[str, int] = {}
    for item in drifts:
        classification_counts[item.classification] = (
            classification_counts.get(item.classification, 0) + 1
        )
    print("classification_counts:")
    for key, value in _sorted(
        list(classification_counts.items()),
        key=lambda item: item[0],
    ):
        print(f"- {key}: {value}")
    print("workstream_drifts:")
    for item in drifts:
        print(
            "- {object_id} :: {classification} :: touchsites={before}->{after} ({delta:+d}) :: surviving={before_surviving}->{after_surviving} ({surviving_delta:+d}) :: dominant={before_blocker}->{after_blocker} :: recommended={before_cut}->{after_cut}".format(
                object_id=item.object_id,
                classification=item.classification,
                before=item.before_touchsite_count,
                after=item.after_touchsite_count,
                delta=item.touchsite_delta,
                before_surviving=item.before_surviving_touchsite_count,
                after_surviving=item.after_surviving_touchsite_count,
                surviving_delta=item.surviving_touchsite_delta,
                before_blocker=item.before_dominant_blocker_class,
                after_blocker=item.after_dominant_blocker_class,
                before_cut=item.before_recommended_cut_object_id or "<none>",
                after_cut=item.after_recommended_cut_object_id or "<none>",
            )
        )
        print(
            "  blocker_deltas: ready={ready:+d} :: coverage_gap={coverage:+d} :: policy={policy:+d} :: diagnostic={diagnostic:+d}".format(
                ready=item.blocker_deltas.get("ready_touchsite_count", 0),
                coverage=item.blocker_deltas.get("coverage_gap_touchsite_count", 0),
                policy=item.blocker_deltas.get("policy_blocked_touchsite_count", 0),
                diagnostic=item.blocker_deltas.get("diagnostic_blocked_touchsite_count", 0),
            )
        )
        print(
            "  touchsite_identity_delta: added={added} :: removed={removed}".format(
                added=len(item.added_touchsite_ids),
                removed=len(item.removed_touchsite_ids),
            )
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--artifact", default=str(_DEFAULT_ARTIFACT))
    parser.add_argument(
        "--workstreams-artifact",
        default=str(_DEFAULT_WORKSTREAMS_ARTIFACT),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("build")
    subparsers.add_parser("summary")

    trace_parser = subparsers.add_parser("trace")
    trace_parser.add_argument("--id", required=True)

    blockers_parser = subparsers.add_parser("blockers")
    blockers_parser.add_argument("--object-id", required=True)

    workstream_parser = subparsers.add_parser("workstream")
    workstream_parser.add_argument("--object-id", required=True)

    blast_radius_parser = subparsers.add_parser("blast-radius")
    blast_radius_parser.add_argument("--id", required=True)
    blast_radius_parser.add_argument("--impact-artifact", default=None)

    compare_parser = subparsers.add_parser("compare")
    compare_parser.add_argument("--before-workstreams-artifact", required=True)
    compare_parser.add_argument("--after-workstreams-artifact", required=True)
    compare_parser.add_argument("--object-id", default=None)

    args = parser.parse_args(argv)
    root = Path(args.root).resolve()
    artifact = Path(args.artifact).resolve()
    workstreams_artifact = Path(args.workstreams_artifact).resolve()

    if args.command == "build":
        graph = build_invariant_graph(root)
        write_invariant_graph(artifact, graph)
        write_invariant_workstreams(workstreams_artifact, build_invariant_workstreams(graph))
        print(str(artifact))
        return 0
    if args.command == "summary":
        _print_summary(graph=_load_or_build_graph(root=root, artifact=artifact))
        return 0
    if args.command == "trace":
        return _print_trace(
            graph=_load_or_build_graph(root=root, artifact=artifact),
            raw_id=str(args.id),
        )
    if args.command == "blockers":
        return _print_blockers(
            graph=_load_or_build_graph(root=root, artifact=artifact),
            object_id=str(args.object_id),
        )
    if args.command == "workstream":
        return _print_workstream(
            graph=_load_or_build_graph(root=root, artifact=artifact),
            object_id=str(args.object_id),
        )
    if args.command == "blast-radius":
        impact_artifact = (
            Path(args.impact_artifact).resolve()
            if args.impact_artifact is not None
            else None
        )
        return _print_blast_radius(
            graph=_load_or_build_graph(root=root, artifact=artifact),
            raw_id=str(args.id),
            impact_artifact=impact_artifact,
        )
    if args.command == "compare":
        return _print_compare(
            before_workstreams_artifact=Path(
                str(args.before_workstreams_artifact)
            ).resolve(),
            after_workstreams_artifact=Path(
                str(args.after_workstreams_artifact)
            ).resolve(),
            object_id=None if args.object_id is None else str(args.object_id),
        )
    return 1


__all__ = ["main"]
